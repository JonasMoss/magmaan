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
#include "magmaan/estimate/snlls.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/gls/gls.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/nt/measures.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;

namespace {

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
                            const magmaan::estimate::SnllsEstimates& est,
                            const magmaan::spec::Starts* starts,
                            const char* estimator,
                            const char* backend) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  out["backend"] = backend;
  out["snlls_compatible"] = true;
  out["snlls_nonlinear_npar"] = est.snlls.n_nonlinear;
  out["snlls_linear_npar"] = est.snlls.n_linear;
  out["snlls_profile_evaluations"] = est.snlls.profile_evaluations;
  out["snlls_profile_cache_hits"] = est.snlls.profile_cache_hits;
  out["snlls_gradient_evaluations"] = est.snlls.gradient_evaluations;
  out["snlls_jacobian_evaluations"] = est.snlls.jacobian_evaluations;
  out["snlls_admissible"] = est.snlls.admissible;
  out["snlls_min_variance"] = std::isfinite(est.snlls.min_variance)
      ? est.snlls.min_variance
      : NA_REAL;
  return out;
}

magmaan::fit::Bounds bounds_from_nullable(Rcpp::Nullable<Rcpp::List> bounds) {
  magmaan::fit::Bounds out;
  if (bounds.isNull()) return out;
  Rcpp::List b(bounds.get());
  if (!b.containsElementNamed("lower") || !b.containsElementNamed("upper"))
    Rcpp::stop("magmaan: bounds must be a list with $lower and $upper");
  out.lower = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(b["lower"]));
  out.upper = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(b["upper"]));
  return out;
}

magmaan::gls::WLS wls_from_arg(SEXP W, std::size_t n_blocks) {
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
  return magmaan::gls::WLS(std::move(weights));
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
                              const char* estimator) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  out["ordinal"] = true;
  out["thresholds"] = ordinal_stats_to_r(stats)["thresholds"];
  out["polychoric"] = ordinal_stats_to_r(stats)["R"];
  return out;
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "model_matrix_rep");
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_fit");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);

  auto e_or = magmaan::estimate::fit<magmaan::nt::ml::ML, magmaan::optim::LbfgsOptimizer>(
      ctx.pt, ctx.rep, ctx.samp, magmaan::nt::ml::ML{},
      magmaan::optim::LbfgsOptimizer{lbfgs_opts_from(lbfgs)}, starts);
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

// fit_uls() — mirrors fit_bounded<ULS, LbfgsBOptimizer>(pt, rep, samp, bounds).
//
// [[Rcpp::export]]
Rcpp::List fit_uls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                        Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::ULS{},
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ULS");
}

// fit_gls() — mirrors fit_bounded<GLS, LbfgsBOptimizer>(pt, rep, samp, bounds).
//
// [[Rcpp::export]]
Rcpp::List fit_gls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                        Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::GLS{},
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "GLS");
}

// fit_wls() — mirrors fit_bounded<WLS, LbfgsBOptimizer>(pt, rep, samp, bounds).
//
// [[Rcpp::export]]
Rcpp::List fit_wls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                        Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                        Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::gls::WLS wls = wls_from_arg(W, ctx.samp.S.size());
  auto e_or = magmaan::estimate::fit_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), std::move(wls),
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
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
Rcpp::List fit_dwls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                 Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_dwls_ordinal");
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
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "DWLS");
}

// [[Rcpp::export]]
Rcpp::List fit_wls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                                Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_ordinal");
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
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS,
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "WLS");
}

// [[Rcpp::export]]
Rcpp::List fit_uls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_snlls_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::ULS{},
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::SnllsEstimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "ULS-SNLLS", "lbfgsb");
}

// [[Rcpp::export]]
Rcpp::List fit_gls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_snlls_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::GLS{},
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::SnllsEstimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "GLS-SNLLS", "lbfgsb");
}

// [[Rcpp::export]]
Rcpp::List fit_wls_snlls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                              Rcpp::Nullable<Rcpp::List> lbfgsb = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::gls::WLS wls = wls_from_arg(W, ctx.samp.S.size());
  auto e_or = magmaan::estimate::fit_snlls_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), std::move(wls),
      magmaan::optim::LbfgsBOptimizer{lbfgs_opts_from(lbfgsb)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::SnllsEstimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "WLS-SNLLS", "lbfgsb");
}

// [[Rcpp::export]]
Rcpp::List fit_uls_ceres_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> ceres = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
#ifdef MAGMAAN_WITH_CERES
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::ULS{},
      magmaan::optim::CeresBoundedOptimizer{ceres_opts_from(ceres)}, starts);
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_snlls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_snlls_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::ULS{},
      magmaan::optim::CeresBoundedOptimizer{ceres_opts_from(ceres)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::SnllsEstimates est = std::move(*e_or);
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_snlls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_snlls_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::GLS{},
      magmaan::optim::CeresBoundedOptimizer{ceres_opts_from(ceres)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::SnllsEstimates est = std::move(*e_or);
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_snlls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::gls::WLS wls = wls_from_arg(W, ctx.samp.S.size());
  auto e_or = magmaan::estimate::fit_snlls_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), std::move(wls),
      magmaan::optim::CeresBoundedOptimizer{ceres_opts_from(ceres)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::SnllsEstimates est = std::move(*e_or);
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto e_or = magmaan::estimate::fit_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), magmaan::gls::GLS{},
      magmaan::optim::CeresBoundedOptimizer{ceres_opts_from(ceres)}, starts);
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_ceres");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::gls::WLS wls = wls_from_arg(W, ctx.samp.S.size());
  auto e_or = magmaan::estimate::fit_bounded(ctx.pt, ctx.rep, ctx.samp,
      bounds_from_nullable(bounds), std::move(wls),
      magmaan::optim::CeresBoundedOptimizer{ceres_opts_from(ceres)}, starts);
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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_start_values");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto sv_or = magmaan::estimate::simple_start_values(ctx.pt, ctx.rep, ctx.samp, starts);
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
  auto r = magmaan::nt::infer::information_expected(ctx.pt, ctx.rep, ctx.samp, est);
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
  auto r = magmaan::nt::infer::information_observed_fd(ctx.pt, ctx.rep, ctx.samp, est, h_step);
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
  auto r = magmaan::nt::infer::information_observed_analytic(ctx.pt, ctx.rep, ctx.samp, est);
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
  auto r = magmaan::nt::infer::vcov(info_m, ctx.pt);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_se() — mirrors se(vcov). Returns √diag(vcov); NaN for negative
// diagonal entries (Heywood case). Never errors.
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_se(Rcpp::NumericMatrix vcov) {
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  return Rcpp::wrap(magmaan::nt::infer::se(vcov_m));
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
  return magmaan::nt::infer::chi2_stat(samp, est);
}

// infer_df_stat() — mirrors df_stat(pt, samp). Returns Σ_b p_b(p_b+1)/2 (+ means)
// − fixed_x − n_free + constraint.rank. Pure function of the model and the
// data dimensions; doesn't depend on θ̂. Errors on unenforced constraints.
//
// [[Rcpp::export]]
int infer_df_stat(SEXP partable, Rcpp::List sample_stats) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "infer_df_stat");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto r = magmaan::nt::infer::df_stat(ctx.pt, ctx.samp);
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
  const magmaan::nt::measures::BaselineFit bl = magmaan::nt::measures::baseline_chi2(samp);
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
  magmaan::nt::measures::BaselineFit bl;
  bl.chi2 = Rcpp::as<double>(baseline["chi2"]);
  bl.df   = Rcpp::as<int>(baseline["df"]);
  const magmaan::nt::measures::FitMeasures fm = magmaan::nt::measures::fit_measures(chi2, df, bl, ctx.samp);
  const magmaan::estimate::Estimates   est = est_from_fit(fit);
  auto fx = magmaan::nt::measures::fit_extras(ctx.pt, ctx.rep, ctx.samp, est);
  const bool have = fx.has_value();
  return Rcpp::List::create(
      Rcpp::_["cfi"]               = fm.cfi,
      Rcpp::_["tli"]               = fm.tli,
      Rcpp::_["rmsea"]             = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"]    = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"]    = fm.rmsea_ci_upper,
      Rcpp::_["srmr"]              = have ? fx->srmr              : NA_REAL,
      Rcpp::_["logl"]              = have ? fx->logl              : NA_REAL,
      Rcpp::_["unrestricted.logl"] = have ? fx->unrestricted_logl : NA_REAL,
      Rcpp::_["aic"]               = have ? fx->aic               : NA_REAL,
      Rcpp::_["bic"]               = have ? fx->bic               : NA_REAL,
      Rcpp::_["bic2"]              = have ? fx->bic2              : NA_REAL,
      Rcpp::_["npar"]              = have ? fx->npar              : NA_INTEGER,
      Rcpp::_["ntotal"]            = have ? static_cast<double>(fx->ntotal) : NA_REAL);
}

// infer_z_test() — mirrors z_test(est, se). `se` is the SE vector from
// infer_se(infer_vcov(info, fit)).
//
// [[Rcpp::export]]
Rcpp::List infer_z_test(Rcpp::List fit, Rcpp::NumericVector se) {
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::VectorXd se_v = Rcpp::as<Eigen::VectorXd>(se);
  const magmaan::nt::infer::ZTestResult zt = magmaan::nt::infer::z_test(est, se_v);
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
    out[i] = (ISNA(c) || d == NA_INTEGER) ? NA_REAL : magmaan::nt::infer::chi2_pvalue(c, d);
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
  auto w_or = magmaan::nt::infer::wald_test(Rmat, qv, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w_or->chi2, Rcpp::_["df"] = w_or->df,
                            Rcpp::_["pvalue"] = magmaan::nt::infer::chi2_pvalue(w_or->chi2, w_or->df));
}

// infer_browne_residual_nt() — mirrors browne_residual_nt(pt, rep, samp, est).
// Returns just the statistic (the model df is infer_df_stat(); the p-value
// is infer_chi2_pvalue(statistic, df)).
//
// [[Rcpp::export]]
Rcpp::List infer_browne_residual_nt(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto s_or = magmaan::nt::infer::browne_residual_nt(ctx.pt, ctx.rep, ctx.samp, est);
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
  auto s_or = magmaan::nt::infer::rls_chi2(ctx.samp, im);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}
