// Rcpp glue for magmaan's fitting + post-fit-inference layer. Composition-first:
// one thin wrapper per C++ entry point, no convenience bundling. fit_fit()
// mirrors fit() and returns a "fit object" (a transparent base-R list carrying
// the partable / sample stats / theta needed downstream); the other functions
// each take a fit object (and, where the C++ signature does, the upstream
// results — an se result, a baseline result, implied moments) and mirror one
// C++ function. Errors -> Rcpp::stop with magmaan's error kind + detail. Eigen
// <-> R via RcppEigen. Shared plumbing lives in internal.hpp.

#include "internal.hpp"

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/shrinkage.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/measures/effects.hpp"
#include "magmaan/measures/factor_scores.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/measures/residuals.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/inference/score.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;

namespace {

// The estimate::fit* core takes an explicit start vector. These helpers
// compute the starting values from a parsed model and abort to R on failure.
// The R-facing layer owns the choice of starting algorithm; the default is
// FABIN3 (Hägglund 1982), matching lavaan's default.
Eigen::VectorXd start_values_or_stop(const Ctx& ctx,
                                     const magmaan::spec::Starts& starts) {
  auto x = magmaan::estimate::fabin_start_values(ctx.pt, ctx.rep, ctx.samp,
                                                 starts);
  if (!x.has_value()) stop_fit(x.error());
  return std::move(*x);
}

Eigen::VectorXd ordinal_starts_or_stop(const Ctx& ctx,
                                       const magmaan::data::OrdinalStats& stats,
                                       const magmaan::spec::Starts& starts) {
  auto x = magmaan::estimate::ordinal_start_values(ctx.pt, ctx.rep, stats,
                                                   starts);
  if (!x.has_value()) stop_fit(x.error());
  return std::move(*x);
}

Eigen::VectorXd mixed_ordinal_starts_or_stop(
    const Ctx& ctx, const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::spec::Starts& starts) {
  auto x = magmaan::estimate::mixed_ordinal_start_values(ctx.pt, ctx.rep, stats,
                                                         starts);
  if (!x.has_value()) stop_fit(x.error());
  return std::move(*x);
}

Rcpp::List names_list(const std::vector<std::vector<std::string>>& nn) {
  Rcpp::List out(static_cast<R_xlen_t>(nn.size()));
  for (std::size_t b = 0; b < nn.size(); ++b) out[static_cast<R_xlen_t>(b)] = Rcpp::wrap(nn[b]);
  return out;
}
Rcpp::List dims_list(const std::vector<lvm::BlockDims>& dd) {
  Rcpp::List out(static_cast<R_xlen_t>(dd.size()));
  for (std::size_t b = 0; b < dd.size(); ++b)
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(Rcpp::_["n_observed"] = static_cast<int>(dd[b].n_observed),
                                                       Rcpp::_["n_latent"]   = static_cast<int>(dd[b].n_latent));
  return out;
}

Rcpp::List standardized_to_list(
    const magmaan::measures::standardize::StandardizedSolution& r) {
  return Rcpp::List::create(Rcpp::_["theta"] = Rcpp::wrap(r.theta),
                            Rcpp::_["se"] = Rcpp::wrap(r.se));
}

Rcpp::List matrix_blocks_to_r(const std::vector<Eigen::MatrixXd>& blocks,
                              const std::vector<std::vector<std::string>>& names,
                              bool square_names) {
  Rcpp::List out(static_cast<R_xlen_t>(blocks.size()));
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    Rcpp::NumericMatrix M = Rcpp::wrap(blocks[b]);
    if (b < names.size()) {
      Rcpp::CharacterVector nm = Rcpp::wrap(names[b]);
      M.attr("dimnames") = square_names
          ? Rcpp::List::create(nm, nm)
          : Rcpp::List::create(R_NilValue, nm);
    }
    out[static_cast<R_xlen_t>(b)] = M;
  }
  return out;
}

Rcpp::List vector_blocks_to_r(const std::vector<Eigen::VectorXd>& blocks,
                              const std::vector<std::vector<std::string>>& names) {
  Rcpp::List out(static_cast<R_xlen_t>(blocks.size()));
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    Rcpp::NumericVector v = Rcpp::wrap(blocks[b]);
    if (b < names.size() && v.size() == static_cast<R_xlen_t>(names[b].size())) {
      v.attr("names") = Rcpp::wrap(names[b]);
    }
    out[static_cast<R_xlen_t>(b)] = v;
  }
  return out;
}

magmaan::measures::FactorScoreMethod factor_score_method_from(
    const std::string& method) {
  if (method == "regression" || method == "Regression" ||
      method == "thurstone" || method == "Thurstone") {
    return magmaan::measures::FactorScoreMethod::Regression;
  }
  if (method == "bartlett" || method == "Bartlett") {
    return magmaan::measures::FactorScoreMethod::Bartlett;
  }
  Rcpp::stop("magmaan: factor score method must be 'regression' or 'bartlett' "
             "(got '%s')", method);
}

const char* score_candidate_kind_str(
    magmaan::inference::ScoreCandidateKind kind) {
  using K = magmaan::inference::ScoreCandidateKind;
  switch (kind) {
  case K::FixedParam: return "fixed";
  case K::EqualityRelease: return "equality_release";
  }
  return "unknown";
}

magmaan::inference::ScoreInformation score_information_from(
    const std::string& information) {
  if (information == "observed" || information == "Observed" ||
      information == "OBSERVED") {
    return magmaan::inference::ScoreInformation::Observed;
  }
  if (information == "expected" || information == "Expected" ||
      information == "EXPECTED" || information.empty()) {
    return magmaan::inference::ScoreInformation::Expected;
  }
  Rcpp::stop("magmaan: score information must be 'expected' or 'observed' "
             "(got '%s')", information);
}

magmaan::inference::ScoreCandidateSet score_candidates_from(
    const std::string& candidates) {
  if (candidates == "all" || candidates == "absent" ||
      candidates == "with_absent") {
    return magmaan::inference::ScoreCandidateSet::WithAbsentRows;
  }
  if (candidates == "fixed" || candidates == "fixed_rows" ||
      candidates == "fixed_rows_only" || candidates.empty()) {
    return magmaan::inference::ScoreCandidateSet::FixedRowsOnly;
  }
  Rcpp::stop("magmaan: score candidates must be 'fixed' or 'all' (got '%s')",
             candidates);
}

magmaan::inference::ModificationIndexOptions modification_options_from(
    const std::string& information, const std::string& candidates,
    bool include_loadings, bool include_covariances) {
  magmaan::inference::ModificationIndexOptions opts;
  opts.information = score_information_from(information);
  opts.candidates = score_candidates_from(candidates);
  opts.include_loadings = include_loadings;
  opts.include_covariances = include_covariances;
  return opts;
}

Rcpp::DataFrame score_table_df(
    const magmaan::inference::ScoreTestTable& tab,
    const magmaan::spec::LatentNames& names) {
  const R_xlen_t n = static_cast<R_xlen_t>(tab.rows.size());
  Rcpp::CharacterVector kind(n), op(n), lhs(n), rhs(n);
  Rcpp::IntegerVector row(n), group(n), df(n);
  Rcpp::NumericVector score(n), information(n), mi(n), pvalue(n), epc(n),
      epc_lv(n), epc_all(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& r = tab.rows[static_cast<std::size_t>(i)];
    const auto& c = r.candidate;
    kind[i] = score_candidate_kind_str(c.kind);
    row[i] = static_cast<int>(c.row) + 1;
    op[i] = std::string(magmaan::parse::to_string(c.op));
    group[i] = c.group;
    if (c.lhs_var >= 0 &&
        static_cast<std::size_t>(c.lhs_var) < names.var_name.size()) {
      lhs[i] = names.var_name[static_cast<std::size_t>(c.lhs_var)];
    } else if (c.row < names.row_lhs.size()) {
      lhs[i] = names.row_lhs[c.row];
    } else {
      lhs[i] = "";
    }
    if (c.rhs_var >= 0 &&
        static_cast<std::size_t>(c.rhs_var) < names.var_name.size()) {
      rhs[i] = names.var_name[static_cast<std::size_t>(c.rhs_var)];
    } else if (c.row < names.row_rhs.size()) {
      rhs[i] = names.row_rhs[c.row];
    } else {
      rhs[i] = "";
    }
    score[i] = r.score;
    information[i] = r.information;
    mi[i] = r.mi;
    df[i] = r.df;
    pvalue[i] = r.p_value;
    epc[i] = r.epc;
    epc_lv[i] = r.epc_lv;
    epc_all[i] = r.epc_all;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["kind"] = kind,
      Rcpp::_["row"] = row,
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["group"] = group,
      Rcpp::_["score"] = score,
      Rcpp::_["information"] = information,
      Rcpp::_["mi"] = mi,
      Rcpp::_["df"] = df,
      Rcpp::_["pvalue"] = pvalue,
      Rcpp::_["epc"] = epc,
      Rcpp::_["epc.lv"] = epc_lv,
      Rcpp::_["epc.all"] = epc_all,
      Rcpp::_["stringsAsFactors"] = false);
}
Rcpp::DataFrame cells_df(const std::vector<lvm::Cell>& cells) {
  const R_xlen_t m = static_cast<R_xlen_t>(cells.size());
  Rcpp::CharacterVector mat(m);
  Rcpp::IntegerVector row(m), col(m), block(m);
  Rcpp::LogicalVector used(m);
  for (R_xlen_t i = 0; i < m; ++i) {
    const lvm::Cell& c = cells[static_cast<std::size_t>(i)];
    mat[i]   = std::string(lvm::to_string(c.mat));
    row[i]   = c.row;
    col[i]   = c.col;
    block[i] = c.block;
    used[i]  = c.used;
  }
  Rcpp::List l = Rcpp::List::create(Rcpp::_["mat"] = mat, Rcpp::_["row"] = row, Rcpp::_["col"] = col,
                                    Rcpp::_["block"] = block, Rcpp::_["used"] = used);
  l.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -static_cast<int>(m));
  l.attr("class") = "data.frame";
  return Rcpp::DataFrame(l);
}
Rcpp::DataFrame structural_cells_df(const std::vector<lvm::StructuralCell>& sc) {
  const R_xlen_t m = static_cast<R_xlen_t>(sc.size());
  Rcpp::CharacterVector mat(m);
  Rcpp::IntegerVector row(m), col(m), block(m);
  Rcpp::NumericVector value(m);
  for (R_xlen_t i = 0; i < m; ++i) {
    const lvm::StructuralCell& c = sc[static_cast<std::size_t>(i)];
    mat[i]   = std::string(lvm::to_string(c.mat));
    row[i]   = c.row;
    col[i]   = c.col;
    block[i] = c.block;
    value[i] = c.value;
  }
  Rcpp::List l = Rcpp::List::create(Rcpp::_["mat"] = mat, Rcpp::_["row"] = row, Rcpp::_["col"] = col,
                                    Rcpp::_["block"] = block, Rcpp::_["value"] = value);
  l.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -static_cast<int>(m));
  l.attr("class") = "data.frame";
  return Rcpp::DataFrame(l);
}

Rcpp::List fit_result(Ctx& ctx,
                      const magmaan::estimate::Estimates& est,
                      const magmaan::spec::Starts* starts,
                      const char* estimator) {
  const std::size_t nb = ctx.samp.S.size();
  Rcpp::List S_out(static_cast<R_xlen_t>(nb));
  for (std::size_t b = 0; b < nb; ++b) {
    Rcpp::NumericMatrix Sb = Rcpp::wrap(ctx.samp.S[b]);
    Rcpp::CharacterVector nm = Rcpp::wrap(ctx.rep.ov_names[b]);
    Sb.attr("dimnames") = Rcpp::List::create(nm, nm);
    S_out[static_cast<R_xlen_t>(b)] = Sb;
  }
  SEXP mean_out = R_NilValue;
  if (!ctx.samp.mean.empty()) {
    Rcpp::List Ml(static_cast<R_xlen_t>(nb));
    for (std::size_t b = 0; b < nb; ++b)
      Ml[static_cast<R_xlen_t>(b)] = Rcpp::wrap(ctx.samp.mean[b]);
    mean_out = Ml;
  }
  Rcpp::IntegerVector nobs_out(static_cast<R_xlen_t>(nb));
  std::int64_t ntotal = 0;
  for (std::size_t b = 0; b < nb; ++b) {
    nobs_out[static_cast<R_xlen_t>(b)] = static_cast<int>(ctx.samp.n_obs[b]);
    ntotal += ctx.samp.n_obs[b];
  }

  return Rcpp::List::create(
      Rcpp::_["converged"]     = true,
      Rcpp::_["estimator"]     = estimator,
      Rcpp::_["fmin"]          = est.fmin,
      Rcpp::_["iterations"]    = est.iterations,
      Rcpp::_["npar"]          = static_cast<int>(ctx.pt.n_free()),
      Rcpp::_["ngroups"]       = static_cast<int>(nb),
      Rcpp::_["ntotal"]        = static_cast<int>(ntotal),
      Rcpp::_["group_var"]     = ctx.names.group_var,
      Rcpp::_["group_labels"]  = Rcpp::wrap(ctx.names.group_labels),
      Rcpp::_["theta"]         = Rcpp::wrap(est.theta),
      Rcpp::_["ov_names"]      = Rcpp::wrap(ctx.ov_names),
      Rcpp::_["partable"]      = partable_df(ctx.pt, ctx.names, est, starts),
      Rcpp::_["S"]             = S_out,
      Rcpp::_["nobs"]          = nobs_out,
      Rcpp::_["sample_mean"]   = mean_out,
      Rcpp::_["meanstructure"] = ctx.meanstructure);
}

Rcpp::List snlls_fit_result(Ctx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const magmaan::spec::Starts* starts,
                            const char* estimator,
                            const char* backend) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  out["backend"] = backend;
  out["snlls_compatible"] = true;
  return out;
}

magmaan::estimate::Bounds bounds_from_nullable(Rcpp::Nullable<Rcpp::List> bounds) {
  magmaan::estimate::Bounds out;
  if (bounds.isNull()) return out;
  Rcpp::List b(bounds.get());
  if (!b.containsElementNamed("lower") || !b.containsElementNamed("upper"))
    Rcpp::stop("magmaan: bounds must be a list with $lower and $upper");
  out.lower = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(b["lower"]));
  out.upper = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(b["upper"]));
  return out;
}

magmaan::estimate::gmm::Weight wls_from_arg(SEXP W, std::size_t n_blocks) {
  std::vector<Eigen::MatrixXd> weights;
  weights.reserve(n_blocks);
  if (Rf_isMatrix(W)) {
    if (n_blocks != 1)
      Rcpp::stop("magmaan: WLS weights must be a list of %d matrices for a %d-group model",
                 static_cast<int>(n_blocks), static_cast<int>(n_blocks));
    weights.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(W)));
  } else if (TYPEOF(W) == VECSXP) {
    Rcpp::List Wl(W);
    if (static_cast<std::size_t>(Wl.size()) != n_blocks)
      Rcpp::stop("magmaan: WLS weights list has length %d but the model has %d group(s)",
                 static_cast<int>(Wl.size()), static_cast<int>(n_blocks));
    for (R_xlen_t b = 0; b < Wl.size(); ++b)
      weights.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wl[b])));
  } else {
    Rcpp::stop("magmaan: WLS weights must be a matrix or a list of matrices");
  }
  return weights;
}

Rcpp::List ordinal_stats_to_r(const magmaan::data::OrdinalStats& s) {
  const R_xlen_t nb = static_cast<R_xlen_t>(s.R.size());
  Rcpp::List R(nb), thresholds(nb), threshold_ov(nb), threshold_level(nb),
      moments(nb), NACOV(nb), W_dwls(nb), W_wls(nb), n_levels(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    R[b] = Rcpp::wrap(s.R[bi]);
    thresholds[b] = Rcpp::wrap(s.thresholds[bi]);
    const Eigen::Index p = s.R[bi].rows();
    const Eigen::Index nth = s.thresholds[bi].size();
    Eigen::VectorXd mb(nth + p * (p - 1) / 2);
    mb.head(nth) = s.thresholds[bi];
    Eigen::Index pos = nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        mb(pos++) = s.R[bi](i, j);
      }
    }
    moments[b] = Rcpp::wrap(mb);
    Rcpp::IntegerVector ov(static_cast<R_xlen_t>(s.threshold_ov[bi].size()));
    Rcpp::IntegerVector lev(static_cast<R_xlen_t>(s.threshold_level[bi].size()));
    for (R_xlen_t k = 0; k < ov.size(); ++k) {
      ov[k] = s.threshold_ov[bi][static_cast<std::size_t>(k)] + 1;
      lev[k] = s.threshold_level[bi][static_cast<std::size_t>(k)];
    }
    threshold_ov[b] = ov;
    threshold_level[b] = lev;
    NACOV[b] = Rcpp::wrap(s.NACOV[bi]);
    W_dwls[b] = Rcpp::wrap(s.W_dwls[bi]);
    W_wls[b] = Rcpp::wrap(s.W_wls[bi]);
    nobs[b] = static_cast<int>(s.n_obs[bi]);
    n_levels[b] = Rcpp::wrap(s.n_levels[bi]);
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["R"] = R,
      Rcpp::_["thresholds"] = thresholds,
      Rcpp::_["threshold_ov"] = threshold_ov,
      Rcpp::_["threshold_level"] = threshold_level,
      Rcpp::_["moments"] = moments,
      Rcpp::_["NACOV"] = NACOV,
      Rcpp::_["W_dwls"] = W_dwls,
      Rcpp::_["W_wls"] = W_wls,
      Rcpp::_["nobs"] = nobs,
      Rcpp::_["n_levels"] = n_levels);
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_ordinal_data", "list");
  return out;
}

Rcpp::List mixed_ordinal_stats_to_r(const magmaan::data::MixedOrdinalStats& s) {
  const R_xlen_t nb = static_cast<R_xlen_t>(s.R.size());
  Rcpp::List R(nb), mean(nb), ordered_mask(nb), thresholds(nb), threshold_ov(nb),
      threshold_level(nb), moments(nb), NACOV(nb), W_dwls(nb), W_wls(nb),
      n_levels(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    R[b] = Rcpp::wrap(s.R[bi]);
    mean[b] = Rcpp::wrap(s.mean[bi]);
    ordered_mask[b] = Rcpp::wrap(s.ordered[bi]);
    thresholds[b] = Rcpp::wrap(s.thresholds[bi]);
    moments[b] = Rcpp::wrap(s.moments[bi]);
    Rcpp::IntegerVector ov(static_cast<R_xlen_t>(s.threshold_ov[bi].size()));
    Rcpp::IntegerVector lev(static_cast<R_xlen_t>(s.threshold_level[bi].size()));
    for (R_xlen_t k = 0; k < ov.size(); ++k) {
      ov[k] = s.threshold_ov[bi][static_cast<std::size_t>(k)] + 1;
      lev[k] = s.threshold_level[bi][static_cast<std::size_t>(k)];
    }
    threshold_ov[b] = ov;
    threshold_level[b] = lev;
    NACOV[b] = Rcpp::wrap(s.NACOV[bi]);
    W_dwls[b] = Rcpp::wrap(s.W_dwls[bi]);
    W_wls[b] = Rcpp::wrap(s.W_wls[bi]);
    nobs[b] = static_cast<int>(s.n_obs[bi]);
    n_levels[b] = Rcpp::wrap(s.n_levels[bi]);
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["R"] = R,
      Rcpp::_["mean"] = mean,
      Rcpp::_["ordered_mask"] = ordered_mask,
      Rcpp::_["thresholds"] = thresholds,
      Rcpp::_["threshold_ov"] = threshold_ov,
      Rcpp::_["threshold_level"] = threshold_level,
      Rcpp::_["moments"] = moments,
      Rcpp::_["NACOV"] = NACOV,
      Rcpp::_["W_dwls"] = W_dwls,
      Rcpp::_["W_wls"] = W_wls,
      Rcpp::_["nobs"] = nobs,
      Rcpp::_["n_levels"] = n_levels);
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_mixed_ordinal_data", "list");
  return out;
}

magmaan::data::PolychoricHScoreKind h_score_kind_from(std::string kind) {
  if (kind == "ml") return magmaan::data::PolychoricHScoreKind::ML;
  if (kind == "wma_hard_cap") return magmaan::data::PolychoricHScoreKind::WmaHardCap;
  if (kind == "smooth_cap") return magmaan::data::PolychoricHScoreKind::SmoothCap;
  if (kind == "exp_cap") return magmaan::data::PolychoricHScoreKind::ExpCap;
  Rcpp::stop("magmaan: h_kind must be one of 'ml', 'wma_hard_cap', 'smooth_cap', or 'exp_cap'");
}

magmaan::data::PairwiseOrdinalHWeightedStatsOptions
h_weighted_options_from(std::string h_kind, double k, double a, double b,
                        double lambda) {
  magmaan::data::PairwiseOrdinalHWeightedStatsOptions options;
  options.rho.h_score.kind = h_score_kind_from(h_kind);
  options.rho.h_score.k = k;
  options.rho.h_score.a = a;
  options.rho.h_score.b = b;
  options.rho.h_score.lambda = lambda;
  return options;
}

magmaan::data::PairwiseOrdinalDpdStatsOptions
ordinal_dpd_options_from(double alpha) {
  magmaan::data::PairwiseOrdinalDpdStatsOptions options;
  options.alpha = alpha;
  return options;
}

magmaan::data::PolyserialPairDpdOptions
polyserial_dpd_options_from(double alpha) {
  magmaan::data::PolyserialPairDpdOptions options;
  options.alpha = alpha;
  return options;
}

magmaan::data::MixedOrdinalHuberResidualOptions
huber_residual_options_from(std::string clip, double k) {
  magmaan::data::MixedOrdinalHuberResidualOptions options;
  if (clip == "none") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::None;
  } else if (clip == "hard_huber") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::HardHuber;
  } else if (clip == "pseudo_huber") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::PseudoHuber;
  } else if (clip == "tukey_biweight") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::TukeyBiweight;
  } else {
    Rcpp::stop("magmaan: unknown Huber residual clip kind '%s'", clip);
  }
  options.clip.k = k;
  return options;
}

Rcpp::List pairwise_ordinal_diagnostics_to_r(
    const std::vector<magmaan::data::PairwiseOrdinalBlockDiagnostics>& d) {
  Rcpp::List out(static_cast<R_xlen_t>(d.size()));
  for (std::size_t b = 0; b < d.size(); ++b) {
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["moment_influence"] = Rcpp::wrap(d[b].moment_influence),
        Rcpp::_["gamma"] = Rcpp::wrap(d[b].gamma),
        Rcpp::_["min_eigen_r"] = d[b].min_eigen_r,
        Rcpp::_["raw_min_eigen_r"] = d[b].raw_min_eigen_r,
        Rcpp::_["r_repair_applied"] = d[b].r_repair_applied);
  }
  return out;
}

Rcpp::List mixed_polyserial_dpd_diagnostics_to_r(
    const std::vector<magmaan::data::MixedOrdinalPolyserialDpdBlockDiagnostics>& d) {
  Rcpp::List out(static_cast<R_xlen_t>(d.size()));
  for (std::size_t b = 0; b < d.size(); ++b) {
    const auto& block = d[b];
    Rcpp::DataFrame pairs;
    {
      const R_xlen_t n = static_cast<R_xlen_t>(block.dpd_pairs.size());
      Rcpp::IntegerVector i(n), j(n), moment_index(n);
      for (R_xlen_t r = 0; r < n; ++r) {
        const auto& pair = block.dpd_pairs[static_cast<std::size_t>(r)];
        i[r] = pair.i + 1;
        j[r] = pair.j + 1;
        moment_index[r] = pair.moment_index + 1;
      }
      pairs = Rcpp::DataFrame::create(Rcpp::_["i"] = i,
                                      Rcpp::_["j"] = j,
                                      Rcpp::_["moment_index"] = moment_index,
                                      Rcpp::_["stringsAsFactors"] = false);
    }
    Rcpp::NumericVector rho(static_cast<R_xlen_t>(block.dpd_fits.size()));
    Rcpp::NumericVector objective(static_cast<R_xlen_t>(block.dpd_fits.size()));
    for (R_xlen_t r = 0; r < rho.size(); ++r) {
      const auto& fit = block.dpd_fits[static_cast<std::size_t>(r)];
      rho[r] = fit.rho;
      objective[r] = fit.objective;
    }
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["pairs"] = pairs,
        Rcpp::_["rho"] = rho,
        Rcpp::_["objective"] = objective,
        Rcpp::_["moment_influence"] = Rcpp::wrap(block.moment_influence),
        Rcpp::_["gamma"] = Rcpp::wrap(block.gamma));
  }
  return out;
}

Rcpp::List mixed_huber_residual_diagnostics_to_r(
    const std::vector<magmaan::data::MixedOrdinalHuberResidualBlockDiagnostics>& d) {
  Rcpp::List out(static_cast<R_xlen_t>(d.size()));
  for (std::size_t b = 0; b < d.size(); ++b) {
    const auto& block = d[b];
    const R_xlen_t n = static_cast<R_xlen_t>(block.robust_pairs.size());
    Rcpp::IntegerVector i(n), j(n), moment_index(n);
    Rcpp::CharacterVector kind(n);
    for (R_xlen_t r = 0; r < n; ++r) {
      const auto& pair = block.robust_pairs[static_cast<std::size_t>(r)];
      i[r] = pair.i + 1;
      j[r] = pair.j + 1;
      moment_index[r] = pair.moment_index + 1;
      kind[r] = pair.kind == magmaan::data::MixedPairKind::ordinal_ordinal
          ? "ordinal_ordinal"
          : (pair.kind == magmaan::data::MixedPairKind::continuous_ordinal
                 ? "continuous_ordinal"
                 : "continuous_continuous");
    }
    Rcpp::DataFrame pairs = Rcpp::DataFrame::create(
        Rcpp::_["i"] = i,
        Rcpp::_["j"] = j,
        Rcpp::_["moment_index"] = moment_index,
        Rcpp::_["kind"] = kind,
        Rcpp::_["stringsAsFactors"] = false);
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["pairs"] = pairs,
        Rcpp::_["rho"] = Rcpp::wrap(block.rho),
        Rcpp::_["objective"] = Rcpp::wrap(block.objective),
        Rcpp::_["moment_influence"] = Rcpp::wrap(block.moment_influence),
        Rcpp::_["gamma"] = Rcpp::wrap(block.gamma),
        Rcpp::_["min_eigen_r"] = block.min_eigen_r,
        Rcpp::_["raw_min_eigen_r"] = block.raw_min_eigen_r,
        Rcpp::_["r_repair_applied"] = block.r_repair_applied);
  }
  return out;
}

Rcpp::LogicalMatrix block_mask_matrix(SEXP M, std::size_t b, std::size_t n_blocks,
                                      const char* what) {
  if (Rf_isMatrix(M)) {
    if (n_blocks != 1)
      Rcpp::stop("magmaan: the model has %d groups; pass a list of %d per-group %s matrices",
                 static_cast<int>(n_blocks), static_cast<int>(n_blocks), what);
    return Rcpp::LogicalMatrix(M);
  }
  if (TYPEOF(M) != VECSXP)
    Rcpp::stop("magmaan: %s must be a logical matrix (single-group) or a list of "
               "per-group logical matrices", what);
  Rcpp::List Ml(M);
  if (static_cast<std::size_t>(Ml.size()) != n_blocks)
    Rcpp::stop("magmaan: %s is a list of %d matrices but the model has %d groups",
               what, static_cast<int>(Ml.size()), static_cast<int>(n_blocks));
  return Rcpp::LogicalMatrix(Ml[static_cast<R_xlen_t>(b)]);
}

magmaan::data::RawData fiml_raw_from_arg(const lvm::MatrixRep& rep, SEXP raw_data) {
  SEXP X_arg = raw_data;
  SEXP mask_arg = R_NilValue;
  if (TYPEOF(raw_data) == VECSXP) {
    Rcpp::List rd(raw_data);
    if (rd.containsElementNamed("X")) {
      X_arg = rd["X"];
      if (rd.containsElementNamed("mask")) mask_arg = rd["mask"];
    }
  }

  const std::size_t n_blocks = rep.dims.size();
  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  raw.mask.reserve(n_blocks);

  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "raw_data$X");
    const std::vector<int> perm = perm_for_cols(Xb, rep.ov_names[b], "raw_data$X");
    const int n = Xb.nrow();
    const int p = static_cast<int>(perm.size());

    Rcpp::LogicalMatrix Mb;
    const bool has_mask = !Rf_isNull(mask_arg);
    if (has_mask) {
      Mb = block_mask_matrix(mask_arg, b, n_blocks, "raw_data$mask");
      if (Mb.nrow() != n || Mb.ncol() != Xb.ncol())
        Rcpp::stop("magmaan: raw_data$mask block %d has shape %dx%d but raw_data$X has %dx%d",
                   static_cast<int>(b + 1), Mb.nrow(), Mb.ncol(), Xb.nrow(), Xb.ncol());
    }

    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
    for (int r = 0; r < n; ++r) {
      for (int k = 0; k < p; ++k) {
        const int src = perm[static_cast<std::size_t>(k)];
        const double x = Xb(r, src);
        bool observed = has_mask
            ? (Mb(r, src) != NA_LOGICAL && Mb(r, src) != 0)
            : std::isfinite(x);
        if (observed && !std::isfinite(x)) {
          Rcpp::stop("magmaan: raw_data$mask marks a non-finite value as observed "
                     "in block %d, row %d", static_cast<int>(b + 1), r + 1);
        }
        M(r, k) = static_cast<std::uint8_t>(observed ? 1 : 0);
        X(r, k) = observed ? x : std::numeric_limits<double>::quiet_NaN();
      }
    }

    raw.X.push_back(std::move(X));
    raw.mask.push_back(std::move(M));
  }
  return raw;
}

magmaan::data::RawData complete_raw_from_arg(const lvm::MatrixRep& rep,
                                             SEXP raw_data) {
  SEXP X_arg = raw_data;
  if (TYPEOF(raw_data) == VECSXP) {
    Rcpp::List rd(raw_data);
    if (rd.containsElementNamed("X")) X_arg = rd["X"];
  }

  const std::size_t n_blocks = rep.dims.size();
  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "raw_data$X");
    const std::vector<int> perm = perm_for_cols(Xb, rep.ov_names[b], "raw_data$X");
    Eigen::MatrixXd X = reorder_data_cols(Xb, perm);
    validate_finite_matrix(X, "raw data", b);
    raw.X.push_back(std::move(X));
  }
  return raw;
}

Rcpp::List fiml_raw_to_r(const magmaan::data::RawData& raw,
                         const std::vector<std::vector<std::string>>& ov_names,
                         const std::vector<std::string>& group_labels) {
  const R_xlen_t nb = static_cast<R_xlen_t>(raw.X.size());
  Rcpp::List X_out(nb), M_out(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    Rcpp::NumericMatrix Xb = Rcpp::wrap(raw.X[bi]);
    Rcpp::LogicalMatrix Mb(raw.mask[bi].rows(), raw.mask[bi].cols());
    for (R_xlen_t r = 0; r < Mb.nrow(); ++r)
      for (R_xlen_t c = 0; c < Mb.ncol(); ++c)
        Mb(r, c) = raw.mask[bi](r, c) != 0;
    Rcpp::CharacterVector nm = Rcpp::wrap(ov_names[bi]);
    Xb.attr("dimnames") = Rcpp::List::create(R_NilValue, nm);
    Mb.attr("dimnames") = Rcpp::List::create(R_NilValue, nm);
    X_out[b] = Xb;
    M_out[b] = Mb;
    nobs[b] = static_cast<int>(raw.X[bi].rows());
  }
  if (!group_labels.empty() && static_cast<R_xlen_t>(group_labels.size()) == nb) {
    Rcpp::CharacterVector gl = Rcpp::wrap(group_labels);
    X_out.attr("names") = gl;
    M_out.attr("names") = gl;
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["X"] = X_out,
      Rcpp::_["mask"] = M_out,
      Rcpp::_["ov_names"] = names_list(ov_names),
      Rcpp::_["group_labels"] = Rcpp::wrap(group_labels),
      Rcpp::_["nobs"] = nobs);
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_fiml_data", "list");
  return out;
}

Rcpp::List fiml_fit_result(Ctx& ctx,
                           const magmaan::data::RawData& raw,
                           const magmaan::estimate::Estimates& est,
                           const magmaan::spec::Starts* starts) {
  Rcpp::List out = fit_result(ctx, est, starts, "FIML");
  out["fiml"] = true;
  out["raw_data"] = fiml_raw_to_r(raw, ctx.rep.ov_names, ctx.names.group_labels);
  return out;
}

magmaan::data::OrdinalStats ordinal_stats_from_arg(Rcpp::List x) {
  const char* what = "ordinal_stats";
  for (const char* nm : {"R", "thresholds", "threshold_ov", "threshold_level",
                         "NACOV", "W_dwls", "W_wls", "nobs", "n_levels"}) {
    if (!x.containsElementNamed(nm)) Rcpp::stop("magmaan: %s is missing $%s", what, nm);
  }
  Rcpp::List Rl(x["R"]), thl(x["thresholds"]), ovl(x["threshold_ov"]),
      levl(x["threshold_level"]), NAl(x["NACOV"]),
      Wdl(x["W_dwls"]), Wfl(x["W_wls"]),
      nlevl(x["n_levels"]);
  Rcpp::IntegerVector nobs(x["nobs"]);
  const R_xlen_t nb = Rl.size();
  magmaan::data::OrdinalStats out;
  out.R.reserve(static_cast<std::size_t>(nb));
  out.thresholds.reserve(static_cast<std::size_t>(nb));
  out.threshold_ov.reserve(static_cast<std::size_t>(nb));
  out.threshold_level.reserve(static_cast<std::size_t>(nb));
  out.NACOV.reserve(static_cast<std::size_t>(nb));
  out.W_dwls.reserve(static_cast<std::size_t>(nb));
  out.W_wls.reserve(static_cast<std::size_t>(nb));
  out.n_obs.reserve(static_cast<std::size_t>(nb));
  out.n_levels.reserve(static_cast<std::size_t>(nb));
  for (R_xlen_t b = 0; b < nb; ++b) {
    out.R.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Rl[b])));
    out.thresholds.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(thl[b])));
    Rcpp::IntegerVector ov(ovl[b]), lev(levl[b]);
    std::vector<std::int32_t> ov0(static_cast<std::size_t>(ov.size()));
    std::vector<std::int32_t> lev0(static_cast<std::size_t>(lev.size()));
    for (R_xlen_t k = 0; k < ov.size(); ++k) {
      ov0[static_cast<std::size_t>(k)] = ov[k] - 1;
      lev0[static_cast<std::size_t>(k)] = lev[k];
    }
    out.threshold_ov.push_back(std::move(ov0));
    out.threshold_level.push_back(std::move(lev0));
    out.NACOV.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(NAl[b])));
    out.W_dwls.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wdl[b])));
    out.W_wls.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wfl[b])));
    out.n_obs.push_back(static_cast<std::int64_t>(nobs[b]));
    out.n_levels.push_back(Rcpp::as<std::vector<std::int32_t>>(Rcpp::IntegerVector(nlevl[b])));
  }
  return out;
}

std::vector<Eigen::MatrixXd> matrix_blocks_from_arg(SEXP X) {
  std::vector<Eigen::MatrixXd> blocks;
  if (Rf_isMatrix(X)) {
    blocks.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X)));
    return blocks;
  }
  if (TYPEOF(X) != VECSXP) {
    Rcpp::stop("magmaan: X must be a matrix or list of matrices");
  }
  Rcpp::List xl(X);
  blocks.reserve(static_cast<std::size_t>(xl.size()));
  for (R_xlen_t b = 0; b < xl.size(); ++b) {
    blocks.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(xl[b])));
  }
  return blocks;
}

Rcpp::List ordinal_fit_result(Ctx& ctx,
                              const magmaan::data::OrdinalStats& stats,
                              const magmaan::estimate::Estimates& est,
                              const magmaan::spec::Starts* starts,
                              const char* estimator,
                              const char* parameterization = "delta") {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  out["ordinal"] = true;
  out["parameterization"] = parameterization;
  out["thresholds"] = ordinal_stats_to_r(stats)["thresholds"];
  out["polychoric"] = ordinal_stats_to_r(stats)["R"];
  return out;
}

magmaan::data::CovarianceShrinkageKind shrinkage_kind_from_string(const std::string& kind) {
  using K = magmaan::data::CovarianceShrinkageKind;
  if (kind == "none") return K::None;
  if (kind == "ridge") return K::Ridge;
  if (kind == "identity") return K::IdentityTarget;
  if (kind == "diagonal") return K::DiagonalTarget;
  if (kind == "constant_correlation") return K::ConstantCorrelation;
  Rcpp::stop("magmaan: shrinkage kind must be none, ridge, identity, diagonal, or constant_correlation");
}

#ifdef MAGMAAN_WITH_CERES
magmaan::optim::CeresOptions ceres_opts_from(Rcpp::Nullable<Rcpp::List> ceres) {
  magmaan::optim::CeresOptions o;
  if (ceres.isNotNull()) {
    Rcpp::List l(ceres.get());
    if (l.containsElementNamed("max_iter")) o.max_iter = Rcpp::as<int>(l["max_iter"]);
    if (l.containsElementNamed("ftol"))     o.ftol     = Rcpp::as<double>(l["ftol"]);
    if (l.containsElementNamed("gtol"))     o.gtol     = Rcpp::as<double>(l["gtol"]);
    if (l.containsElementNamed("ptol"))     o.ptol     = Rcpp::as<double>(l["ptol"]);
    if (l.containsElementNamed("verbose"))  o.verbose  = Rcpp::as<bool>(l["verbose"]);
  }
  return o;
}

// The composers take the optimizer-agnostic LbfgsOptions; the Ceres backend
// reads max_iter / ftol / gtol from it. (ptol / verbose were already dropped
// by the old fit<D,O> shim, so this is behavior-preserving.)
magmaan::optim::LbfgsOptions ceres_opts_as_lbfgs(Rcpp::Nullable<Rcpp::List> ceres) {
  const magmaan::optim::CeresOptions c = ceres_opts_from(ceres);
  magmaan::optim::LbfgsOptions o;
  o.max_iter = c.max_iter;
  o.ftol     = c.ftol;
  o.gtol     = c.gtol;
  return o;
}
#endif

}  // namespace

// =============================================================================

// model_matrix_rep() — mirrors build_matrix_rep(pt). A function of the partable
// alone (no fit / data / θ̂): the LISREL layout that downstream evaluation uses.
// `partable` a partable data.frame. Returns list(form = "PureCFA"|"Reduced",
// dims = list(list(n_observed, n_latent), ...) per block, ov_names = list(<chr>,
// ...) per block — the observed-variable order everything else (Σ̂ columns, the
// u-factor's B rows, casewise vech) is in — lv_names = list(<chr>, ...) per block
// (extended: includes phantom latents in Reduced form), cells = data.frame(mat,
// row, col, block, used) one row per partable row, structural_cells = data.frame(
// mat, row, col, block, value)).
//
// [[Rcpp::export]]
Rcpp::List model_matrix_rep(SEXP partable) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "model_matrix_rep");
  auto rep_or = lvm::build_matrix_rep(parsed.structure, &parsed.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  const lvm::MatrixRep& rep = *rep_or;
  return Rcpp::List::create(
      Rcpp::_["form"]             = std::string(rep.form == lvm::RepForm::PureCFA ? "PureCFA" : "Reduced"),
      Rcpp::_["dims"]             = dims_list(rep.dims),
      Rcpp::_["ov_names"]         = names_list(rep.ov_names),
      Rcpp::_["lv_names"]         = names_list(rep.lv_names),
      Rcpp::_["cells"]            = cells_df(rep.cell_for_row),
      Rcpp::_["structural_cells"] = structural_cells_df(rep.structural_cells));
}

// =============================================================================

// fit_fit() — mirrors fit(pt, rep, samp, ML{}, LbfgsOptimizer{lbfgs}). `partable`
// is a partable data.frame (e.g. from lavaan_lavaanify(), possibly hand-edited).
// `sample_stats` is the {S, nobs, mean} bundle — exactly what
// data_sample_stats_from_raw() returns, or a hand-built list(S = , nobs = ):
//   $S    a covariance matrix (single group) or a list of per-group ones;
//   $nobs the matching sample size (scalar or per-group vector);
//   $mean (optional) NULL, a vector, or a list of vectors — used only when the
//         model has mean structure. Values used verbatim (no rescaling).
// `lbfgs` is a named list read for max_iter/ftol/gtol/history. Returns a fit
// object: a transparent list carrying everything the infer_information_* /
// infer_chi2_stat / infer_df_stat / infer_*_test / infer_baseline / measures_fit /
// model_implied / robust functions need to recompose, plus convenient views
// (theta, partable+est). Always uniform:
// $S / $sample_mean are per-group lists (length ngroups ≥ 1); $nobs a vector.
//
// [[Rcpp::export]]
Rcpp::List fit_fit(SEXP partable, Rcpp::List sample_stats,
                     Rcpp::Nullable<Rcpp::List> lbfgs = R_NilValue) {
  // `==` / shared-label equality rows are enforced by fit() (reparam θ = K·α);
  // `<` / `>` rows and arbitrary-expression `==` rows make fit() error with a
  // clear message; `:=` rows are ignored during the fit (post-fit quantities).
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_fit");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_ml(ctx.pt, ctx.rep, ctx.samp, x0,
      magmaan::estimate::Bounds{}, magmaan::estimate::Backend::Lbfgs,
      lbfgs_opts_from(lbfgs));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);

  return fit_result(ctx, est, &starts, "ML");
}

// fit_ml() — public ML spelling. Mirrors fit_fit(); kept separate so R users can
// see estimator choice without a dispatcher hiding the C++ path.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_impl(SEXP partable, Rcpp::List sample_stats,
                       Rcpp::Nullable<Rcpp::List> lbfgs = R_NilValue) {
  return fit_fit(partable, sample_stats, lbfgs);
}

// fit_fiml() — mirrors estimate::fit_fiml(pt, rep, raw, FIML{}, LbfgsOptimizer).
// `raw_data` is a magmaan_fiml_data object from df_to_fiml_data(), or a list
// with $X and optional $mask. Missing values are retained in $X and represented
// by $mask; columns are reordered to the model's observed-variable order.
//
// [[Rcpp::export]]
Rcpp::List fit_fiml_impl(SEXP partable, SEXP raw_data,
                         Rcpp::Nullable<Rcpp::List> lbfgs = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_fiml");
  magmaan::spec::Starts starts = std::move(parsed.starts);

  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  if (ctx.rep.ov_names.empty() || ctx.rep.ov_names[0].empty())
    Rcpp::stop("magmaan: model has no observed variables");

  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, raw_data);
  auto start_samp_or = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) stop_fit(start_samp_or.error());
  ctx.samp = std::move(*start_samp_or);
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_fiml(
      ctx.pt, ctx.rep, raw, x0, magmaan::estimate::fiml::FIML{},
      lbfgs_opts_from(lbfgs));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fiml_fit_result(ctx, raw, est, &starts);
}

// fit_uls() — composes fit_gmm(pt, rep, samp, x0, {}, bounds) (LBFGS-B).
//
// [[Rcpp::export]]
Rcpp::List fit_uls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                        Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, bounds_from_nullable(bounds), magmaan::estimate::Backend::Lbfgs,
      lbfgs_opts_from(lbfgsb));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ULS");
}

// fit_gls() — composes fit_gls(pt, rep, samp, x0, bounds) (LBFGS-B).
//
// [[Rcpp::export]]
Rcpp::List fit_gls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                        Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), magmaan::estimate::Backend::Lbfgs,
      lbfgs_opts_from(lbfgsb));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "GLS");
}

// fit_wls() — composes fit_gmm(pt, rep, samp, x0, W, bounds) (LBFGS-B).
//
// [[Rcpp::export]]
Rcpp::List fit_wls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                        Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                        Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls), bounds_from_nullable(bounds),
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "WLS");
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_from_raw_impl(SEXP X) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::ordinal_stats_from_integer_data(blocks);
  if (!out_or.has_value()) stop_post(out_or.error());
  return ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_h_weighted_from_raw_impl(
    SEXP X, std::string h_kind = "wma_hard_cap",
    double k = 1.5, double a = 1.6, double b = 2.2, double lambda = 0.2) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      blocks, h_weighted_options_from(h_kind, k, a, b, lambda));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "h_weighted";
  out["diagnostics"] = pairwise_ordinal_diagnostics_to_r(out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_dpd_from_raw_impl(SEXP X, double alpha = 0.3) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      blocks, ordinal_dpd_options_from(alpha));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "dpd";
  out["alpha"] = alpha;
  out["diagnostics"] = pairwise_ordinal_diagnostics_to_r(out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_from_raw_impl(SEXP X, SEXP ordered_mask) {
  auto blocks = matrix_blocks_from_arg(X);
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(blocks.size());
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != blocks.size())
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  auto out_or = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_shrink_mixed_ordinal_stats_impl(
    Rcpp::List mixed_stats, std::string kind = "diagonal",
    double intensity = 0.0, bool estimate_intensity = false) {
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  magmaan::data::CovarianceShrinkageOptions opts;
  opts.kind = shrinkage_kind_from_string(kind);
  opts.intensity = intensity;
  opts.estimate_intensity = estimate_intensity;
  auto out_or = magmaan::data::shrink_mixed_ordinal_stats(stats, opts);
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = mixed_ordinal_stats_to_r(out_or->stats);
  Rcpp::List diag(static_cast<R_xlen_t>(out_or->block_diagnostics.size()));
  for (std::size_t b = 0; b < out_or->block_diagnostics.size(); ++b) {
    const auto& d = out_or->block_diagnostics[b];
    diag[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["raw_min_eigen"] = d.raw_min_eigen,
        Rcpp::_["min_eigen"] = d.min_eigen,
        Rcpp::_["intensity"] = d.intensity,
        Rcpp::_["shrunk"] = d.shrunk);
  }
  out["shrinkage"] = diag;
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl(
    SEXP X, SEXP ordered_mask, double alpha = 0.3) {
  auto blocks = matrix_blocks_from_arg(X);
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(blocks.size());
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != blocks.size())
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  auto out_or = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
      blocks, ordered, polyserial_dpd_options_from(alpha));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = mixed_ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "polyserial_dpd";
  out["alpha"] = alpha;
  out["diagnostics"] = mixed_polyserial_dpd_diagnostics_to_r(
      out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_huber_residual_from_raw_impl(
    SEXP X, SEXP ordered_mask, std::string clip = "hard_huber",
    double k = 1.345) {
  auto blocks = matrix_blocks_from_arg(X);
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(blocks.size());
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != blocks.size())
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  auto out_or = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      blocks, ordered, huber_residual_options_from(clip, k));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = mixed_ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "huber_residual";
  out["clip"] = clip;
  out["k"] = k;
  out["diagnostics"] = mixed_huber_residual_diagnostics_to_r(
      out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_dwls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                 Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_dwls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::DWLS, x0,
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "DWLS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_wls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                                Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS, x0,
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "WLS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_dwls_mixed_ordinal_impl(SEXP partable, Rcpp::List mixed_stats,
                                       Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                                       Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_dwls_mixed_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  auto prep_or = magmaan::estimate::prepare_mixed_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.mean = stats.mean;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = true;
  const Eigen::VectorXd x0 = mixed_ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_mixed_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::DWLS, x0,
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fit_result(ctx, est, &starts, "DWLS");
  out["mixed_ordinal"] = true;
  out["parameterization"] = parameterization_name;
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_wls_mixed_ordinal_impl(SEXP partable, Rcpp::List mixed_stats,
                                      Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                                      Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_mixed_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  auto prep_or = magmaan::estimate::prepare_mixed_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.mean = stats.mean;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = true;
  const Eigen::VectorXd x0 = mixed_ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_mixed_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS, x0,
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fit_result(ctx, est, &starts, "WLS");
  out["mixed_ordinal"] = true;
  out["parameterization"] = parameterization_name;
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_uls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "ULS-SNLLS", "lbfgsb");
}

// [[Rcpp::export]]
Rcpp::List fit_gls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  auto e_or = magmaan::estimate::fit_snlls_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "GLS-SNLLS", "lbfgsb");
}

// [[Rcpp::export]]
Rcpp::List fit_wls_snlls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                              Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls),
      magmaan::estimate::Backend::Lbfgs, lbfgs_opts_from(lbfgsb));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "WLS-SNLLS", "lbfgsb");
}

// [[Rcpp::export]]
Rcpp::List fit_uls_ceres_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, bounds_from_nullable(bounds), magmaan::estimate::Backend::Ceres,
      ceres_opts_as_lbfgs(ceres));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ULS");
#else
  (void)partable; (void)sample_stats; (void)ceres; (void)bounds;
  Rcpp::stop("magmaan: Ceres backend is not available in this build");
#endif
}

// [[Rcpp::export]]
Rcpp::List fit_uls_snlls_ceres_impl(SEXP partable, Rcpp::List sample_stats,
                                    Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                                    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_snlls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, magmaan::estimate::Backend::Ceres, ceres_opts_as_lbfgs(ceres));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "ULS-SNLLS", "ceres");
#else
  (void)partable; (void)sample_stats; (void)ceres; (void)bounds;
  Rcpp::stop("magmaan: Ceres backend is not available in this build");
#endif
}

// [[Rcpp::export]]
Rcpp::List fit_gls_snlls_ceres_impl(SEXP partable, Rcpp::List sample_stats,
                                    Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                                    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_snlls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  auto e_or = magmaan::estimate::fit_snlls_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      magmaan::estimate::Backend::Ceres, ceres_opts_as_lbfgs(ceres));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "GLS-SNLLS", "ceres");
#else
  (void)partable; (void)sample_stats; (void)ceres; (void)bounds;
  Rcpp::stop("magmaan: Ceres backend is not available in this build");
#endif
}

// [[Rcpp::export]]
Rcpp::List fit_wls_snlls_ceres_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                                    Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                                    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_snlls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls),
      magmaan::estimate::Backend::Ceres, ceres_opts_as_lbfgs(ceres));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "WLS-SNLLS", "ceres");
#else
  (void)partable; (void)sample_stats; (void)W; (void)ceres; (void)bounds;
  Rcpp::stop("magmaan: Ceres backend is not available in this build");
#endif
}

// [[Rcpp::export]]
Rcpp::List fit_gls_ceres_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), magmaan::estimate::Backend::Ceres,
      ceres_opts_as_lbfgs(ceres));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "GLS");
#else
  (void)partable; (void)sample_stats; (void)ceres; (void)bounds;
  Rcpp::stop("magmaan: Ceres backend is not available in this build");
#endif
}

// [[Rcpp::export]]
Rcpp::List fit_wls_ceres_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                              Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls), bounds_from_nullable(bounds),
      magmaan::estimate::Backend::Ceres, ceres_opts_as_lbfgs(ceres));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "WLS");
#else
  (void)partable; (void)sample_stats; (void)W; (void)ceres; (void)bounds;
  Rcpp::stop("magmaan: Ceres backend is not available in this build");
#endif
}

// fit_start_values() — mirrors simple_start_values(pt, rep, samp). Returns the
// theta-ordered start vector (length npar). `sample_stats` as in fit_fit();
// values used verbatim.
//
// [[Rcpp::export]]
Rcpp::NumericVector fit_start_values(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_start_values");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto sv_or = magmaan::estimate::simple_start_values(ctx.pt, ctx.rep, ctx.samp);
  if (!sv_or.has_value()) stop_fit(sv_or.error());
  return Rcpp::wrap(*sv_or);
}

// model_implied() — mirrors ModelEvaluator::build(pt, rep).sigma(est.theta).
// Returns list(sigma = list of per-block p x p matrices, mu = list of per-block
// vectors — empty unless the model has mean structure).
//
// [[Rcpp::export]]
Rcpp::List model_implied(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto ev_or = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev_or.has_value()) stop_model(ev_or.error());
  const lvm::ModelEvaluator ev = std::move(*ev_or);
  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) stop_model(im_or.error());
  const lvm::ImpliedMoments& im = *im_or;

  Rcpp::List sigma(static_cast<R_xlen_t>(im.sigma.size()));
  for (std::size_t b = 0; b < im.sigma.size(); ++b)
    sigma[static_cast<R_xlen_t>(b)] = Rcpp::wrap(im.sigma[b]);
  Rcpp::List mu(static_cast<R_xlen_t>(im.mu.size()));
  for (std::size_t b = 0; b < im.mu.size(); ++b)
    mu[static_cast<R_xlen_t>(b)] = Rcpp::wrap(im.mu[b]);
  return Rcpp::List::create(Rcpp::_["sigma"] = sigma, Rcpp::_["mu"] = mu);
}

// infer_information_expected() — mirrors information_expected(pt, rep, samp, est).
// Returns the (n_free × n_free) expected Fisher information matrix at θ̂.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_expected(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::inference::information_expected(ctx.pt, ctx.rep, ctx.samp, est);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_information_observed_fd() — mirrors information_observed_fd(...).
// Observed-information matrix via central-difference Hessian of the analytic
// ML gradient.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_observed_fd(Rcpp::List fit,
                                                  double h_step = 1e-4) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::inference::information_observed_fd(ctx.pt, ctx.rep, ctx.samp, est, h_step);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_information_observed_analytic() — mirrors information_observed_analytic(...).
// Closed-form observed-information matrix; rejects mean-structure models (use
// the FD variant there).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_observed_analytic(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::inference::information_observed_analytic(ctx.pt, ctx.rep, ctx.samp, est);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_vcov() — mirrors vcov(info, pt). Inverts the information matrix,
// applying the constraint projection K·(KᵀIK)⁻¹·Kᵀ when shared labels /
// invariance / general-linear equalities are active. Takes `fit` so the
// partable comes along (constraints live on pt). The `info` matrix is the
// only numerical input — caller chooses which information variant.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_vcov(Rcpp::NumericMatrix info, Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const Eigen::MatrixXd info_m = Rcpp::as<Eigen::MatrixXd>(info);
  // θ̂ comes along so the constraint Jacobian H(θ̂) can be evaluated when the
  // model carries nonlinear equality constraints; ignored otherwise.
  const Eigen::VectorXd theta = Rcpp::as<Eigen::VectorXd>(fit["theta"]);
  auto r = magmaan::inference::vcov(info_m, ctx.pt, theta);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_vcov_partable() — primitive form of infer_vcov(): the information
// matrix plus the model partable, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_vcov_partable(Rcpp::NumericMatrix info, SEXP partable) {
  auto parsed = partable_from_arg(partable, "infer_vcov_partable");
  const Eigen::MatrixXd info_m = Rcpp::as<Eigen::MatrixXd>(info);
  auto r = magmaan::inference::vcov(info_m, parsed.structure);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_se() — mirrors se(vcov). Returns √diag(vcov); NaN for negative
// diagonal entries (Heywood case). Never errors.
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_se(Rcpp::NumericMatrix vcov) {
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  return Rcpp::wrap(magmaan::inference::se(vcov_m));
}

// compute_defined_impl() — mirrors measures::effects::compute_defined(flat, pt,
// names, est, vcov). `syntax` must be the original model syntax because the
// lavaan-shaped partable only carries the projected `:=` rows, not their parsed
// expression trees.
//
// [[Rcpp::export]]
Rcpp::DataFrame compute_defined_impl(std::string syntax,
                                     Rcpp::List fit,
                                     Rcpp::NumericMatrix vcov) {
  auto flat_or = magmaan::parse::Parser::parse(syntax);
  if (!flat_or.has_value()) {
    const auto& e = flat_or.error();
    Rcpp::stop("magmaan parse error at %u:%u (bytes %u..%u): %s",
               e.span.line, e.span.col, e.span.begin, e.span.end, e.detail);
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto defs_or = magmaan::measures::effects::compute_defined(
      *flat_or, ctx.pt, ctx.names, est, vcov_m);
  if (!defs_or.has_value()) stop_post(defs_or.error());

  const R_xlen_t n = static_cast<R_xlen_t>(defs_or->entries.size());
  Rcpp::CharacterVector lhs(n), op(n), rhs(n);
  Rcpp::NumericVector est_out(n), se(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& entry = defs_or->entries[static_cast<std::size_t>(i)];
    lhs[i] = entry.name;
    op[i] = ":=";
    rhs[i] = "";
    est_out[i] = entry.value;
    se[i] = entry.se;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["est"] = est_out,
      Rcpp::_["se"] = se,
      Rcpp::_["stringsAsFactors"] = false);
}

// infer_chi2_stat() — mirrors chi2_stat(samp, est). Returns N_total · F_ML(θ̂).
// This primitive does not need a fit object: sample_stats supplies nobs and
// fmin is the optimizer's final discrepancy value (fit$fmin when called after
// fit_fit()).
//
// [[Rcpp::export]]
double infer_chi2_stat(Rcpp::List sample_stats, double fmin) {
  if (!sample_stats.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: sample_stats must contain $nobs");
  magmaan::data::SampleStats samp;
  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  for (R_xlen_t i = 0; i < nv.size(); ++i)
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[i]));
  magmaan::estimate::Estimates est;
  est.fmin = fmin;
  return magmaan::inference::chi2_stat(samp, est);
}

// infer_df_stat() — mirrors df_stat(pt, samp). Returns Σ_b p_b(p_b+1)/2 (+ means)
// − fixed_x − n_free + constraint.rank. Pure function of the model and the
// data dimensions; doesn't depend on θ̂. Errors on unenforced constraints.
//
// [[Rcpp::export]]
int infer_df_stat(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "infer_df_stat");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto r = magmaan::inference::df_stat(ctx.pt, ctx.samp);
  if (!r.has_value()) stop_post(r.error());
  return *r;
}

// infer_baseline() — mirrors baseline_chi2(samp). Takes sample stats directly.
//
// [[Rcpp::export]]
Rcpp::List infer_baseline(Rcpp::List sample_stats) {
  if (!sample_stats.containsElementNamed("S") || !sample_stats.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: sample_stats must contain $S and $nobs");
  magmaan::data::SampleStats samp;
  Rcpp::List Sl(sample_stats["S"]);
  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  if (Sl.size() != nv.size())
    Rcpp::stop("magmaan: sample_stats$S and sample_stats$nobs must have the same length");
  for (R_xlen_t b = 0; b < Sl.size(); ++b) {
    samp.S.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Sl[b])));
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[b]));
  }
  const magmaan::measures::BaselineFit bl = magmaan::measures::baseline_chi2(samp);
  return Rcpp::List::create(Rcpp::_["chi2"] = bl.chi2, Rcpp::_["df"] = bl.df);
}

// measures_fit() — mirrors fit_measures(chi2, df, baseline, samp) plus
// fit_extras(pt, rep, samp, est) (the logl-based information criteria + SRMR).
// `chi2` and `df` are scalars from infer_chi2_stat() / infer_df_stat() (or any
// equivalent statistic / dof); `baseline` is a infer_baseline() result.
//
// [[Rcpp::export]]
Rcpp::List measures_fit(Rcpp::List fit, double chi2, int df,
                              Rcpp::List baseline) {
  Ctx ctx = ctx_from_fit(fit);
  magmaan::measures::BaselineFit bl;
  bl.chi2 = Rcpp::as<double>(baseline["chi2"]);
  bl.df   = Rcpp::as<int>(baseline["df"]);
  const magmaan::measures::FitMeasures fm = magmaan::measures::fit_measures(chi2, df, bl, ctx.samp);
  const magmaan::estimate::Estimates   est = est_from_fit(fit);
  auto fx = magmaan::measures::fit_extras(ctx.pt, ctx.rep, ctx.samp, est);
  const bool have = fx.has_value();
  return Rcpp::List::create(
      Rcpp::_["cfi"]               = fm.cfi,
      Rcpp::_["tli"]               = fm.tli,
      Rcpp::_["rmsea"]             = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"]    = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"]    = fm.rmsea_ci_upper,
      Rcpp::_["rmsea.pvalue"]      = fm.rmsea_pvalue,
      Rcpp::_["rmsea.close.h0"]    = fm.rmsea_close_h0,
      Rcpp::_["rmsea.notclose.pvalue"] = fm.rmsea_notclose_pvalue,
      Rcpp::_["rmsea.notclose.h0"] = fm.rmsea_notclose_h0,
      Rcpp::_["srmr"]              = have ? fx->srmr              : NA_REAL,
      Rcpp::_["logl"]              = have ? fx->logl              : NA_REAL,
      Rcpp::_["unrestricted.logl"] = have ? fx->unrestricted_logl : NA_REAL,
      Rcpp::_["aic"]               = have ? fx->aic               : NA_REAL,
      Rcpp::_["bic"]               = have ? fx->bic               : NA_REAL,
      Rcpp::_["bic2"]              = have ? fx->bic2              : NA_REAL,
      Rcpp::_["npar"]              = have ? fx->npar              : NA_INTEGER,
      Rcpp::_["ntotal"]            = have ? static_cast<double>(fx->ntotal) : NA_REAL);
}

// estimate_fiml_robust_mlr() — mirrors estimate::fiml::fiml_robust_mlr().
// Computes observed-pattern sandwich SEs plus Yuan-Bentler/Mplus scaled-test
// traces for a FIML fit carrying $raw_data.
//
// [[Rcpp::export]]
Rcpp::List estimate_fiml_robust_mlr(Rcpp::List fit, double h_step = 1e-4) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: estimate_fiml_robust_mlr() requires a FIML fit with $raw_data");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(ctx.pt, ctx.rep, raw, est);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto r_or = magmaan::estimate::fiml::fiml_robust_mlr(
      ctx.pt, ctx.rep, raw, est, *df_or, extras_or->chi2,
      magmaan::estimate::fiml::FIML{}, h_step);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
      Rcpp::_["se"] = Rcpp::wrap(r_or->se),
      Rcpp::_["eigvals"] = Rcpp::wrap(r_or->eigvals),
      Rcpp::_["chisq_scaled"] = r_or->chisq_scaled,
      Rcpp::_["scaling_factor"] = r_or->scaling_factor,
      Rcpp::_["trace_ugamma"] = r_or->trace_ugamma,
      Rcpp::_["trace_ugamma_h1"] = r_or->trace_ugamma_h1,
      Rcpp::_["trace_ugamma_h0"] = r_or->trace_ugamma_h0,
      Rcpp::_["df"] = r_or->df,
      Rcpp::_["ntotal"] = static_cast<double>(r_or->ntotal),
      Rcpp::_["chisq"] = extras_or->chi2);
}

// measures_standardize_lv() — mirrors measures::standardize::standardize_lv().
//
// [[Rcpp::export]]
Rcpp::List measures_standardize_lv(Rcpp::List fit, Rcpp::NumericMatrix vcov) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto r_or = magmaan::measures::standardize::standardize_lv(
      ctx.pt, ctx.rep, est, vcov_m);
  if (!r_or.has_value()) stop_post(r_or.error());
  return standardized_to_list(*r_or);
}

// measures_standardize_all() — mirrors measures::standardize::standardize_all().
//
// [[Rcpp::export]]
Rcpp::List measures_standardize_all(Rcpp::List fit, Rcpp::NumericMatrix vcov) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto r_or = magmaan::measures::standardize::standardize_all(
      ctx.pt, ctx.rep, est, vcov_m);
  if (!r_or.has_value()) stop_post(r_or.error());
  return standardized_to_list(*r_or);
}

// measures_residuals() — mirrors measures::residuals(); raw S - Sigma-hat
// covariance residuals plus mean residuals when the model has means.
//
// [[Rcpp::export]]
Rcpp::List measures_residuals(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r_or = magmaan::measures::residuals(ctx.pt, ctx.rep, ctx.samp, est);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["cov"] = matrix_blocks_to_r(r_or->cov, ctx.rep.ov_names,
                                          /*square_names=*/true),
      Rcpp::_["mean"] = vector_blocks_to_r(r_or->mean, ctx.rep.ov_names));
}

// measures_standardized_residuals() — mirrors
// measures::standardized_residuals(); deterministic lavResiduals-style raw and
// correlation-metric residuals plus SRMR. Asymptotic residual z-statistics are
// intentionally not part of this primitive yet.
//
// [[Rcpp::export]]
Rcpp::List measures_standardized_residuals(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r_or = magmaan::measures::standardized_residuals(
      ctx.pt, ctx.rep, ctx.samp, est);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["cov_raw"] = matrix_blocks_to_r(r_or->cov_raw, ctx.rep.ov_names,
                                              /*square_names=*/true),
      Rcpp::_["cov_cor"] = matrix_blocks_to_r(r_or->cov_cor, ctx.rep.ov_names,
                                              /*square_names=*/true),
      Rcpp::_["mean_raw"] = vector_blocks_to_r(r_or->mean_raw,
                                               ctx.rep.ov_names),
      Rcpp::_["mean_cor"] = vector_blocks_to_r(r_or->mean_cor,
                                               ctx.rep.ov_names),
      Rcpp::_["srmr"] = r_or->srmr);
}

// measures_factor_scores() — mirrors measures::factor_scores(); `raw_data`
// must contain complete observed data in the model's observed-variable order,
// or carry column names so it can be reordered.
//
// [[Rcpp::export]]
Rcpp::List measures_factor_scores(Rcpp::List fit, SEXP raw_data,
                                  std::string method = "regression") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  auto r_or = magmaan::measures::factor_scores(
      ctx.pt, ctx.rep, raw, est, factor_score_method_from(method));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["scores"] = matrix_blocks_to_r(r_or->scores, ctx.rep.lv_names,
                                             /*square_names=*/false),
      Rcpp::_["method"] = method);
}

// inference_modification_indices() — mirrors inference::modification_indices()
// for ML/FIML/continuous LS fit objects. WLS requires an explicit weight matrix
// because current R fit lists do not retain W.
//
// [[Rcpp::export]]
Rcpp::DataFrame inference_modification_indices(
    Rcpp::List fit, SEXP weight = R_NilValue,
    std::string information = "expected", std::string candidates = "fixed",
    bool include_loadings = true, bool include_covariances = true,
    double h_step = 1e-4) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const auto opts = modification_options_from(
      information, candidates, include_loadings, include_covariances);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;
  if (estimator == "FIML") {
    if (!fit.containsElementNamed("raw_data")) {
      Rcpp::stop("magmaan: FIML modification indices require fit$raw_data");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
    out = magmaan::inference::modification_indices_fiml(
        ctx.pt, ctx.rep, raw, est, opts, magmaan::estimate::fiml::FIML{},
        h_step);
  } else if (estimator == "ULS") {
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, magmaan::estimate::gmm::Weight{}, opts);
  } else if (estimator == "GLS") {
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) stop_model(ev_or.error());
    auto w_or = magmaan::estimate::gmm::normal_theory_weight(
        *ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) stop_fit(w_or.error());
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, *w_or, opts);
  } else if (estimator == "WLS") {
    if (Rf_isNull(weight)) {
      Rcpp::stop("magmaan: WLS modification indices require explicit `weight`");
    }
    auto w = wls_from_arg(weight, ctx.samp.S.size());
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, w, opts);
  } else if (estimator == "ML" || estimator.empty()) {
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, opts);
  } else {
    Rcpp::stop("magmaan: modification indices are not yet exposed for estimator '%s'",
               estimator);
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// inference_score_tests() — mirrors inference::score_tests() for ML/FIML and
// continuous LS fit objects. WLS requires an explicit weight matrix.
//
// [[Rcpp::export]]
Rcpp::DataFrame inference_score_tests(Rcpp::List fit, SEXP weight = R_NilValue,
                                      double h_step = 1e-4) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;
  if (estimator == "FIML") {
    if (!fit.containsElementNamed("raw_data")) {
      Rcpp::stop("magmaan: FIML score tests require fit$raw_data");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
    out = magmaan::inference::score_tests_fiml(
        ctx.pt, ctx.rep, raw, est, magmaan::estimate::fiml::FIML{}, h_step);
  } else if (estimator == "ULS") {
    out = magmaan::inference::score_tests(
        ctx.pt, ctx.rep, ctx.samp, est, magmaan::estimate::gmm::Weight{});
  } else if (estimator == "GLS") {
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) stop_model(ev_or.error());
    auto w_or = magmaan::estimate::gmm::normal_theory_weight(
        *ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) stop_fit(w_or.error());
    out = magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est,
                                          *w_or);
  } else if (estimator == "WLS") {
    if (Rf_isNull(weight)) {
      Rcpp::stop("magmaan: WLS score tests require explicit `weight`");
    }
    auto w = wls_from_arg(weight, ctx.samp.S.size());
    out = magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est, w);
  } else if (estimator == "ML" || estimator.empty()) {
    out = magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est);
  } else {
    Rcpp::stop("magmaan: score tests are not yet exposed for estimator '%s'",
               estimator);
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// infer_z_test() — mirrors z_test(est, se). `se` is the SE vector from
// infer_se(infer_vcov(info, fit)).
//
// [[Rcpp::export]]
Rcpp::List infer_z_test(Rcpp::List fit, Rcpp::NumericVector se) {
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::VectorXd se_v = Rcpp::as<Eigen::VectorXd>(se);
  const magmaan::inference::ZTestResult zt = magmaan::inference::z_test(est, se_v);
  return Rcpp::List::create(Rcpp::_["z"] = Rcpp::wrap(zt.z),
                            Rcpp::_["pvalue"] = Rcpp::wrap(zt.p_value));
}

// infer_z_test_theta() — primitive form of infer_z_test(): estimate vector plus
// SE vector, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::List infer_z_test_theta(Rcpp::NumericVector theta, Rcpp::NumericVector se) {
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  const Eigen::VectorXd se_v = Rcpp::as<Eigen::VectorXd>(se);
  const magmaan::inference::ZTestResult zt = magmaan::inference::z_test(est, se_v);
  return Rcpp::List::create(Rcpp::_["z"] = Rcpp::wrap(zt.z),
                            Rcpp::_["pvalue"] = Rcpp::wrap(zt.p_value));
}

// infer_chi2_pvalue() — mirrors chi2_pvalue(chi2, df), vectorized over `chi2`
// (recycles `df` if length 1); NA/NaN in -> NA out. Also the LR-test primitive
// (subtract test statistics / dfs of two nested fits, call this).
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_chi2_pvalue(Rcpp::NumericVector chi2, Rcpp::IntegerVector df) {
  if (df.size() == 0) Rcpp::stop("magmaan: df must have length >= 1");
  const R_xlen_t n = chi2.size();
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const double c = chi2[i];
    const int d = df[df.size() == 1 ? 0 : (i % df.size())];
    out[i] = (ISNA(c) || d == NA_INTEGER) ? NA_REAL : magmaan::inference::chi2_pvalue(c, d);
  }
  return out;
}

// infer_wald_test() — mirrors wald_test(R, q, est, vcov). `R` is k x npar
// (columns indexed by partable$free); `q` defaults to zeros; `vcov` is the
// parameter covariance matrix (from infer_vcov()).
//
// [[Rcpp::export]]
Rcpp::List infer_wald_test(Rcpp::List fit, Rcpp::NumericMatrix R,
                           Rcpp::NumericMatrix vcov,
                           Rcpp::Nullable<Rcpp::NumericVector> q = R_NilValue) {
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Eigen::MatrixXd Rmat = Rcpp::as<Eigen::MatrixXd>(R);
  {
    Rcpp::List pt_df(fit["partable"]);
    Rcpp::IntegerVector freev(pt_df["free"]);
    int npar = 0;
    for (R_xlen_t i = 0; i < freev.size(); ++i) if (freev[i] > npar) npar = freev[i];
    if (static_cast<int>(Rmat.cols()) != npar)
      Rcpp::stop("magmaan: R must have %d columns (one per free parameter)", npar);
  }
  Eigen::VectorXd qv = q.isNotNull() ? Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(q.get()))
                                     : Eigen::VectorXd::Zero(Rmat.rows());
  if (qv.size() != Rmat.rows())
    Rcpp::stop("magmaan: q must have length %d (= nrow(R))", static_cast<int>(Rmat.rows()));
  Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto w_or = magmaan::inference::wald_test(Rmat, qv, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w_or->chi2, Rcpp::_["df"] = w_or->df,
                            Rcpp::_["pvalue"] = magmaan::inference::chi2_pvalue(w_or->chi2, w_or->df));
}

// infer_wald_test_theta() — primitive form of infer_wald_test(): R/q, theta,
// and parameter vcov. R must have one column per element of theta.
//
// [[Rcpp::export]]
Rcpp::List infer_wald_test_theta(Rcpp::NumericVector theta,
                                 Rcpp::NumericMatrix R,
                                 Rcpp::NumericMatrix vcov,
                                 Rcpp::Nullable<Rcpp::NumericVector> q = R_NilValue) {
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  Eigen::MatrixXd Rmat = Rcpp::as<Eigen::MatrixXd>(R);
  if (Rmat.cols() != est.theta.size())
    Rcpp::stop("magmaan: R must have %d columns (one per free parameter)",
               static_cast<int>(est.theta.size()));
  Eigen::VectorXd qv = q.isNotNull() ? Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(q.get()))
                                     : Eigen::VectorXd::Zero(Rmat.rows());
  if (qv.size() != Rmat.rows())
    Rcpp::stop("magmaan: q must have length %d (= nrow(R))", static_cast<int>(Rmat.rows()));
  Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto w_or = magmaan::inference::wald_test(Rmat, qv, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w_or->chi2, Rcpp::_["df"] = w_or->df,
                            Rcpp::_["pvalue"] = magmaan::inference::chi2_pvalue(w_or->chi2, w_or->df));
}

// infer_browne_residual_nt() — mirrors browne_residual_nt(pt, rep, samp, est).
// Returns just the statistic (the model df is infer_df_stat(); the p-value
// is infer_chi2_pvalue(statistic, df)).
//
// [[Rcpp::export]]
Rcpp::List infer_browne_residual_nt(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto s_or = magmaan::inference::browne_residual_nt(ctx.pt, ctx.rep, ctx.samp, est);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}

// infer_rls_chi2() — mirrors rls_chi2(samp, implied). `implied` is a
// model_implied() result. Returns just the statistic.
//
// [[Rcpp::export]]
Rcpp::List infer_rls_chi2(Rcpp::List fit, Rcpp::List implied) {
  Ctx ctx = ctx_from_fit(fit);
  lvm::ImpliedMoments im;
  Rcpp::List sig(implied["sigma"]);
  for (R_xlen_t b = 0; b < sig.size(); ++b)
    im.sigma.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(sig[b])));
  if (implied.containsElementNamed("mu") && !Rf_isNull(implied["mu"])) {
    Rcpp::List m(implied["mu"]);
    for (R_xlen_t b = 0; b < m.size(); ++b)
      im.mu.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(m[b])));
  }
  auto s_or = magmaan::inference::rls_chi2(ctx.samp, im);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}

// infer_rls_chi2_sample() — primitive form of infer_rls_chi2(): sample moments
// plus model-implied moments, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::List infer_rls_chi2_sample(Rcpp::List sample_stats, Rcpp::List implied) {
  if (!sample_stats.containsElementNamed("S") || !sample_stats.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: `sample_stats` must be a list with $S and $nobs");
  magmaan::data::SampleStats samp;
  Rcpp::List Sl = TYPEOF(sample_stats["S"]) == VECSXP
      ? Rcpp::List(sample_stats["S"])
      : Rcpp::List::create(Rcpp::NumericMatrix(sample_stats["S"]));
  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  if (Sl.size() != nv.size())
    Rcpp::stop("magmaan: sample_stats$S and sample_stats$nobs must have the same length");
  for (R_xlen_t b = 0; b < Sl.size(); ++b) {
    samp.S.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Sl[b])));
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[b]));
  }
  if (sample_stats.containsElementNamed("mean") && !Rf_isNull(sample_stats["mean"])) {
    Rcpp::List Ml = TYPEOF(sample_stats["mean"]) == VECSXP
        ? Rcpp::List(sample_stats["mean"])
        : Rcpp::List::create(Rcpp::NumericVector(sample_stats["mean"]));
    if (Ml.size() != Sl.size())
      Rcpp::stop("magmaan: sample_stats$mean and sample_stats$S must have the same length");
    for (R_xlen_t b = 0; b < Ml.size(); ++b)
      samp.mean.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(Ml[b])));
  }

  lvm::ImpliedMoments im;
  Rcpp::List sig(implied["sigma"]);
  for (R_xlen_t b = 0; b < sig.size(); ++b)
    im.sigma.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(sig[b])));
  if (implied.containsElementNamed("mu") && !Rf_isNull(implied["mu"])) {
    Rcpp::List m(implied["mu"]);
    for (R_xlen_t b = 0; b < m.size(); ++b)
      im.mu.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(m[b])));
  }
  auto s_or = magmaan::inference::rls_chi2(samp, im);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}
