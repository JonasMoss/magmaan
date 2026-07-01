// Rcpp glue for magmaan's fitting + post-fit-inference layer. Composition-first:
// one thin wrapper per C++ entry point, no convenience bundling. fit_fit()
// mirrors fit() and returns a "fit object" (a transparent base-R list carrying
// the partable / sample stats / theta needed downstream); the other functions
// each take a fit object (and, where the C++ signature does, the upstream
// results — an se result, a baseline result, implied moments) and mirror one
// C++ function. Errors -> Rcpp::stop with magmaan's error kind + detail. Eigen
// <-> R via RcppEigen. Shared plumbing lives in internal.hpp.

#include "internal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/frontier/shrinkage.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/measures/effects.hpp"
#include "magmaan/measures/factor_scores.hpp"
#include "magmaan/measures/composite_weights.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/measures/residuals.hpp"
#include "magmaan/measures/reliability.hpp"
#include "magmaan/model/auto_identification.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/twolevel.hpp"
#include "magmaan/estimate/frontier/rbm.hpp"
#include "magmaan/estimate/frontier/pairwise.hpp"
#include "magmaan/estimate/ml_continuation.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/gmm/structured_gamma_weight.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/inference/score.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;

namespace {

// Forward declarations for the two terminal-audit converters. The full
// definitions live next to `fit_result` further down; they are declared
// up here so `fcsem_fit_result` (defined before `fit_result` in this TU)
// can call them.
Rcpp::List audit_to_r(const magmaan::optim::TerminalAudit& a);
Rcpp::List diagnostics_to_r(const magmaan::estimate::FitDiagnostics& d);
magmaan::estimate::gmm::Weight continuous_ls_weight(
    const Ctx& ctx, const magmaan::estimate::Estimates& est,
    const std::string& estimator, SEXP weight, const char* call);
magmaan::estimate::ContinuousLsIJWeightMode continuous_ij_mode(
    const std::string& estimator);

std::string start_name_from_arg(Rcpp::Nullable<Rcpp::String> start,
                                const char* caller,
                                const char* default_name) {
  if (start.isNull()) return default_name;
  std::string name = Rcpp::as<std::string>(start.get());
  for (char& ch : name) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (name.empty() || name == "default" || name == "lavaan" ||
      name == "fabin") {
    return "fabin3";
  }
  if (name == "simple" || name == "fabin2" || name == "fabin3" ||
      name == "guttman" || name == "guttman1952" ||
      name == "bentler" || name == "bentler1982" ||
      name == "jamesstein" || name == "james-stein" || name == "js") {
    return name;
  }
  Rcpp::stop("magmaan: %s(): unsupported start '%s' "
             "(accepted: simple, fabin2, fabin3, guttman, bentler1982, jamesstein)",
             caller, name.c_str());
  return default_name;
}

// The estimate::fit* core takes an explicit start vector. These helpers
// compute the starting values from a parsed model and abort to R on failure.
// The R-facing layer owns the choice of starting algorithm; ordinary fits
// default to FABIN3 (Hägglund 1982), matching lavaan's default.
Eigen::VectorXd start_values_or_stop(const Ctx& ctx,
                                     const magmaan::spec::Starts& starts,
                                     const std::string& start_name = "fabin3") {
  magmaan::fit_expected<Eigen::VectorXd> x;
  if (start_name == "simple") {
    x = magmaan::estimate::simple_start_values(ctx.pt, ctx.rep, ctx.samp,
                                               starts);
  } else if (start_name == "fabin2") {
    x = magmaan::estimate::fabin_start_values(
        ctx.pt, ctx.rep, ctx.samp, starts,
        magmaan::estimate::FabinVariant::Fabin2);
  } else if (start_name == "guttman" || start_name == "guttman1952") {
    x = magmaan::estimate::guttman_start_values(ctx.pt, ctx.rep, ctx.samp,
                                                starts);
  } else if (start_name == "bentler" || start_name == "bentler1982") {
    x = magmaan::estimate::bentler1982_start_values(ctx.pt, ctx.rep,
                                                    ctx.samp, starts);
  } else if (start_name == "jamesstein" || start_name == "james-stein" ||
             start_name == "js") {
    x = magmaan::estimate::jamesstein_start_values(ctx.pt, ctx.rep, ctx.samp,
                                                   starts);
  } else {
    x = magmaan::estimate::fabin_start_values(ctx.pt, ctx.rep, ctx.samp,
                                              starts);
  }
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

Rcpp::List reliability_results_to_r(
    const Eigen::MatrixXd& S,
    Rcpp::Nullable<Rcpp::NumericMatrix> gamma,
    int n) {
  namespace rel = magmaan::measures::frontier::reliability;

  const rel::Coefficient coefs[] = {
      rel::Coefficient::Alpha,
      rel::Coefficient::Lambda6,
      rel::Coefficient::SpearmanGuttmanOmega};
  const char* names[] = {"alpha", "lambda6", "spearman_guttman_omega"};
  const char* grad_method[] = {"analytic", "analytic", "finite_difference"};
  constexpr R_xlen_t n_coef = 3;

  const bool has_gamma = gamma.isNotNull();
  Eigen::MatrixXd G;
  if (has_gamma) {
    if (n <= 0) Rcpp::stop("magmaan: measures_reliability_cov() needs n > 0 when gamma is supplied");
    G = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(gamma.get()));
  }

  Rcpp::CharacterVector coef_col(n_coef), grad_method_col(n_coef);
  Rcpp::NumericVector value_col(n_coef), avar_col(n_coef), se_col(n_coef);
  Rcpp::IntegerVector n_col(n_coef);
  Rcpp::List gradients(n_coef);

  for (R_xlen_t i = 0; i < n_coef; ++i) {
    coef_col[i] = names[i];
    grad_method_col[i] = grad_method[i];
    if (has_gamma) {
      auto out_or = rel::delta_method(coefs[i], S, G, n);
      if (!out_or.has_value()) stop_post(out_or.error());
      value_col[i] = out_or->value;
      avar_col[i] = out_or->avar;
      se_col[i] = out_or->se;
      gradients[i] = Rcpp::wrap(out_or->gradient);
      n_col[i] = n;
    } else {
      auto value_or = rel::value(coefs[i], S);
      if (!value_or.has_value()) stop_post(value_or.error());
      auto grad_or = rel::gradient(coefs[i], S);
      if (!grad_or.has_value()) stop_post(grad_or.error());
      value_col[i] = *value_or;
      avar_col[i] = NA_REAL;
      se_col[i] = NA_REAL;
      gradients[i] = Rcpp::wrap(*grad_or);
      n_col[i] = NA_INTEGER;
    }
  }

  gradients.attr("names") = coef_col;
  Rcpp::DataFrame table = Rcpp::DataFrame::create(
      Rcpp::_["coefficient"] = coef_col,
      Rcpp::_["value"] = value_col,
      Rcpp::_["se"] = se_col,
      Rcpp::_["avar"] = avar_col,
      Rcpp::_["n"] = n_col,
      Rcpp::_["gradient_method"] = grad_method_col,
      Rcpp::_["stringsAsFactors"] = false);
  return Rcpp::List::create(Rcpp::_["table"] = table,
                            Rcpp::_["gradient"] = gradients);
}

Rcpp::DataFrame composite_weights_df(
    const std::vector<magmaan::measures::composite::CompositeWeights>& rows) {
  R_xlen_t n = 0;
  for (const auto& cw : rows) n += static_cast<R_xlen_t>(cw.indicators.size());

  Rcpp::CharacterVector composite(n), indicator(n);
  Rcpp::IntegerVector group(n);
  Rcpp::NumericVector weight(n), se(n);
  R_xlen_t pos = 0;
  for (const auto& cw : rows) {
    for (std::size_t j = 0; j < cw.indicators.size(); ++j) {
      composite[pos] = cw.composite;
      group[pos] = cw.group;
      indicator[pos] = cw.indicators[j];
      weight[pos] = cw.weight(static_cast<Eigen::Index>(j));
      se[pos] = cw.se(static_cast<Eigen::Index>(j));
      ++pos;
    }
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["composite"] = composite,
      Rcpp::_["group"] = group,
      Rcpp::_["indicator"] = indicator,
      Rcpp::_["weight"] = weight,
      Rcpp::_["se"] = se,
      Rcpp::_["stringsAsFactors"] = false);
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

// One RMS residual section -> the 11-row column of lavaan's $summary table.
// Non-finite inferential fields map to R's NA (lavaan reports NA there).
Rcpp::NumericVector residual_rms_to_r(const magmaan::measures::ResidualRms& r) {
  auto na = [](double x) { return std::isfinite(x) ? x : NA_REAL; };
  return Rcpp::NumericVector::create(
      na(r.srmr), na(r.srmr_se), na(r.srmr_exactfit_z),
      na(r.srmr_exactfit_pvalue), na(r.usrmr), na(r.usrmr_se),
      na(r.usrmr_ci_lower), na(r.usrmr_ci_upper), na(r.usrmr_closefit_h0),
      na(r.usrmr_closefit_z), na(r.usrmr_closefit_pvalue));
}

// Per-block residual summary -> list of data frames shaped like
// lavResiduals(fit)$summary: rows are the SRMR-family statistics, columns are
// cov (and mean/total when the block carries a mean structure).
Rcpp::List residual_summary_to_r(
    const std::vector<magmaan::measures::ResidualSummary>& summ) {
  const Rcpp::CharacterVector row_names = Rcpp::CharacterVector::create(
      "srmr", "srmr.se", "srmr.exactfit.z", "srmr.exactfit.pvalue", "usrmr",
      "usrmr.se", "usrmr.ci.lower", "usrmr.ci.upper", "usrmr.closefit.h0.value",
      "usrmr.closefit.z", "usrmr.closefit.pvalue");
  Rcpp::List out(static_cast<R_xlen_t>(summ.size()));
  for (std::size_t b = 0; b < summ.size(); ++b) {
    const auto& s = summ[b];
    Rcpp::List df =
        s.has_mean
            ? Rcpp::List::create(Rcpp::_["cov"] = residual_rms_to_r(s.cov),
                                 Rcpp::_["mean"] = residual_rms_to_r(s.mean),
                                 Rcpp::_["total"] = residual_rms_to_r(s.total))
            : Rcpp::List::create(Rcpp::_["cov"] = residual_rms_to_r(s.cov));
    df.attr("class") = "data.frame";
    df.attr("row.names") = row_names;
    out[static_cast<R_xlen_t>(b)] = df;
  }
  return out;
}

// A measures::StandardizedResiduals -> the R list shared by the NT and the
// estimated-weight residual bindings (raw/cor/se/z matrices, SRMR, $summary).
Rcpp::List standardized_residuals_to_r(
    const magmaan::measures::StandardizedResiduals& r,
    const std::vector<std::vector<std::string>>& ov_names) {
  return Rcpp::List::create(
      Rcpp::_["cov_raw"] = matrix_blocks_to_r(r.cov_raw, ov_names, true),
      Rcpp::_["cov_cor"] = matrix_blocks_to_r(r.cov_cor, ov_names, true),
      Rcpp::_["cov_se"] = matrix_blocks_to_r(r.cov_se, ov_names, true),
      Rcpp::_["cov_z"] = matrix_blocks_to_r(r.cov_z, ov_names, true),
      Rcpp::_["mean_raw"] = vector_blocks_to_r(r.mean_raw, ov_names),
      Rcpp::_["mean_cor"] = vector_blocks_to_r(r.mean_cor, ov_names),
      Rcpp::_["mean_se"] = vector_blocks_to_r(r.mean_se, ov_names),
      Rcpp::_["mean_z"] = vector_blocks_to_r(r.mean_z, ov_names),
      Rcpp::_["srmr"] = r.srmr,
      Rcpp::_["summary"] = residual_summary_to_r(r.summary));
}

const char* optim_status_to_r(magmaan::optim::OptimStatus status) {
  using magmaan::optim::OptimStatus;
  return status == OptimStatus::Converged           ? "converged"
       : status == OptimStatus::LineSearchSalvaged ? "line_search_salvaged"
       : status == OptimStatus::SingularConvergence ? "singular_convergence"
       : status == OptimStatus::NoisyObjective     ? "noisy_objective"
       : status == OptimStatus::FalseConvergence   ? "false_convergence"
       : status == OptimStatus::BudgetExhausted    ? "budget_exhausted"
                                                    : "unknown";
}

const char* continuation_target_to_r(
    magmaan::estimate::frontier::MlContinuationTarget target) {
  using Target = magmaan::estimate::frontier::MlContinuationTarget;
  switch (target) {
    case Target::Diagonal:
      return "diagonal";
    case Target::ScaledIdentity:
      return "scaled_identity";
    case Target::Identity:
      return "identity";
  }
  return "unknown";
}

magmaan::estimate::frontier::MlContinuationTarget
continuation_target_from_string(const std::string& target) {
  std::string key = target;
  for (char& ch : key) {
    if (ch == '-') ch = '_';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  using Target = magmaan::estimate::frontier::MlContinuationTarget;
  if (key.empty() || key == "diagonal" || key == "diag") {
    return Target::Diagonal;
  }
  if (key == "scaled_identity" || key == "scale_identity") {
    return Target::ScaledIdentity;
  }
  if (key == "identity" || key == "raw_identity") return Target::Identity;
  Rcpp::stop("magmaan: continuation target must be 'diagonal' or "
             "'scaled_identity' or 'identity' (got '%s')", target.c_str());
}

Rcpp::List continuation_to_r(
    const magmaan::estimate::frontier::MlRidgeContinuationResult& result,
    magmaan::estimate::frontier::MlContinuationTarget target) {
  const R_xlen_t n = static_cast<R_xlen_t>(result.steps.size());
  Rcpp::IntegerVector step(n), iterations(n), f_evals(n), g_evals(n);
  Rcpp::NumericVector alpha(n), min_ev(n), max_ev(n), condition(n), fmin(n),
      grad_norm(n);
  Rcpp::LogicalVector converged(n), sigma_pd_all(n);
  Rcpp::CharacterVector optimizer_status(n);

  Eigen::Index npar = 0;
  if (!result.steps.empty()) npar = result.steps.front().estimates.theta.size();
  Eigen::MatrixXd theta_path(n, npar);

  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& s = result.steps[static_cast<std::size_t>(i)];
    step[i] = static_cast<int>(i + 1);
    alpha[i] = s.alpha;
    min_ev[i] = s.min_sample_eigen;
    max_ev[i] = s.max_sample_eigen;
    condition[i] = s.sample_condition;
    fmin[i] = s.estimates.fmin;
    iterations[i] = s.estimates.iterations;
    f_evals[i] = s.estimates.f_evals;
    g_evals[i] = s.estimates.g_evals;
    grad_norm[i] = s.estimates.grad_inf_norm;
    converged[i] =
        s.estimates.optimizer_status == magmaan::optim::OptimStatus::Converged;
    sigma_pd_all[i] = s.estimates.diagnostics.sigma_pd_all;
    optimizer_status[i] = optim_status_to_r(s.estimates.optimizer_status);
    if (s.estimates.theta.size() == npar) {
      theta_path.row(i) = s.estimates.theta.transpose();
    }
  }

  Rcpp::DataFrame path = Rcpp::DataFrame::create(
      Rcpp::_["step"] = step,
      Rcpp::_["alpha"] = alpha,
      Rcpp::_["min_sample_eigen"] = min_ev,
      Rcpp::_["max_sample_eigen"] = max_ev,
      Rcpp::_["sample_condition"] = condition,
      Rcpp::_["converged"] = converged,
      Rcpp::_["optimizer_status"] = optimizer_status,
      Rcpp::_["fmin"] = fmin,
      Rcpp::_["grad_norm"] = grad_norm,
      Rcpp::_["iterations"] = iterations,
      Rcpp::_["f_evals"] = f_evals,
      Rcpp::_["g_evals"] = g_evals,
      Rcpp::_["sigma_pd_all"] = sigma_pd_all,
      Rcpp::_["stringsAsFactors"] = false);

  return Rcpp::List::create(
      Rcpp::_["target"] = continuation_target_to_r(target),
      Rcpp::_["path"] = path,
      Rcpp::_["theta"] = Rcpp::wrap(theta_path),
      Rcpp::_["total_iterations"] = result.total_iterations,
      Rcpp::_["total_f_evals"] = result.total_f_evals,
      Rcpp::_["total_g_evals"] = result.total_g_evals);
}

std::vector<std::string>
fcsem_observed_names(const magmaan::spec::LatentStructure& pt,
                     const magmaan::spec::LatentNames& names) {
  std::vector<std::string> out;
  out.reserve(pt.ov_order.size());
  for (std::int32_t id : pt.ov_order) {
    if (id < 0 || static_cast<std::size_t>(id) >= names.var_name.size()) {
      Rcpp::stop("magmaan: native FC-SEM observed-variable inventory is invalid");
    }
    out.push_back(names.var_name[static_cast<std::size_t>(id)]);
  }
  return out;
}

struct FcSemCtx {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames names;
  magmaan::spec::Starts starts;
  magmaan::data::SampleStats samp;
  std::vector<std::string> ov_names;
};

FcSemCtx fcsem_model_from_syntax(const std::string& syntax) {
  auto flat = magmaan::parse::Parser::parse(syntax);
  if (!flat.has_value()) {
    const auto& e = flat.error();
    Rcpp::stop("magmaan parse error at %u:%u (bytes %u..%u): %s",
               e.span.line, e.span.col, e.span.begin, e.span.end, e.detail);
  }
  magmaan::spec::BuildOptions opts;
  opts.composite_mode = magmaan::spec::CompositeMode::FcSem;
  magmaan::spec::Starts starts;
  magmaan::spec::LatentNames names;
  auto pt_or = magmaan::spec::build(*flat, opts, &starts, &names);
  if (!pt_or.has_value()) {
    Rcpp::stop("magmaan lavaanify error: %s", pt_or.error().detail);
  }
  if (pt_or->composite_mode != magmaan::spec::CompositeMode::FcSem ||
      pt_or->composite_blocks.empty()) {
    Rcpp::stop("magmaan: native FC-SEM requires at least one `<~` composite");
  }
  if (pt_or->n_groups() != 1) {
    Rcpp::stop("magmaan: native FC-SEM R frontier currently supports one group");
  }

  FcSemCtx ctx;
  ctx.pt = std::move(*pt_or);
  ctx.names = std::move(names);
  ctx.starts = std::move(starts);
  ctx.ov_names = fcsem_observed_names(ctx.pt, ctx.names);
  return ctx;
}

magmaan::data::SampleStats fcsem_sample_stats_from_arg(
    Rcpp::List sample_stats, const std::vector<std::string>& ov_names) {
  if (!sample_stats.containsElementNamed("S") ||
      !sample_stats.containsElementNamed("nobs")) {
    Rcpp::stop("magmaan: native FC-SEM sample_stats must contain $S and $nobs");
  }
  Rcpp::List Sl = TYPEOF(sample_stats["S"]) == VECSXP
      ? Rcpp::List(sample_stats["S"])
      : Rcpp::List::create(Rcpp::NumericMatrix(sample_stats["S"]));
  Rcpp::IntegerVector nobs = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  if (Sl.size() != 1 || nobs.size() != 1) {
    Rcpp::stop("magmaan: native FC-SEM R frontier currently supports one group");
  }
  const int nb = nobs[0];
  if (nb == NA_INTEGER || nb <= 0) {
    Rcpp::stop("magmaan: native FC-SEM nobs must be a positive integer");
  }

  Rcpp::NumericMatrix S0(Sl[0]);
  const std::vector<int> perm = perm_for_cols(S0, ov_names, "S");
  magmaan::data::SampleStats out;
  out.S.push_back(reorder_cov(S0, perm));
  validate_finite_matrix(out.S.back(), "sample covariance", 0);
  out.n_obs.push_back(static_cast<std::int64_t>(nb));
  return out;
}

FcSemCtx fcsem_ctx_from_syntax_sample_stats(const std::string& syntax,
                                            Rcpp::List sample_stats) {
  FcSemCtx ctx = fcsem_model_from_syntax(syntax);
  ctx.samp = fcsem_sample_stats_from_arg(sample_stats, ctx.ov_names);
  return ctx;
}

FcSemCtx fcsem_ctx_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("fcsem") || !Rcpp::as<bool>(fit["fcsem"]) ||
      !fit.containsElementNamed("syntax")) {
    Rcpp::stop("magmaan: expected a native FC-SEM fit object");
  }
  Rcpp::List ss = Rcpp::List::create(
      Rcpp::_["S"] = fit["S"],
      Rcpp::_["nobs"] = fit["nobs"]);
  return fcsem_ctx_from_syntax_sample_stats(Rcpp::as<std::string>(fit["syntax"]),
                                            ss);
}

Rcpp::DataFrame fcsem_partable_df(
    const magmaan::spec::LatentStructure& pt,
    const magmaan::spec::LatentNames& names,
    const magmaan::spec::Starts& starts,
    const magmaan::estimate::Estimates* est = nullptr) {
  const magmaan::compat::lavaan::LavaanParTable native =
      magmaan::compat::lavaan::to_lavaan_partable(pt, names, starts);
  Rcpp::DataFrame out = partable_df_from_lavaan(native, est);
  Rf_setAttrib(out, Rf_install("magmaan.fcsem"), Rf_ScalarLogical(1));
  return out;
}

Rcpp::List fcsem_fit_result(FcSemCtx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const std::string& syntax) {
  Rcpp::NumericMatrix S0 = Rcpp::wrap(ctx.samp.S[0]);
  Rcpp::CharacterVector nm = Rcpp::wrap(ctx.ov_names);
  S0.attr("dimnames") = Rcpp::List::create(nm, nm);
  Rcpp::List S_out = Rcpp::List::create(S0);
  Rcpp::IntegerVector nobs_out =
      Rcpp::IntegerVector::create(static_cast<int>(ctx.samp.n_obs[0]));

  using magmaan::optim::OptimStatus;
  const char* opt_status = optim_status_to_r(est.optimizer_status);

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["converged"]     = (est.optimizer_status == OptimStatus::Converged),
      Rcpp::_["estimator"]     = "FCSEM-ML",
      Rcpp::_["fmin"]          = est.fmin,
      Rcpp::_["iterations"]    = est.iterations,
      Rcpp::_["f_evals"]       = est.f_evals,
      Rcpp::_["g_evals"]       = est.g_evals,
      Rcpp::_["npar"]          = static_cast<int>(ctx.pt.n_free()),
      Rcpp::_["ngroups"]       = 1,
      Rcpp::_["ntotal"]        = static_cast<int>(ctx.samp.n_obs[0]),
      Rcpp::_["group_var"]     = "",
      Rcpp::_["group_labels"]  = Rcpp::CharacterVector::create(),
      Rcpp::_["theta"]         = Rcpp::wrap(est.theta),
      Rcpp::_["ov_names"]      = Rcpp::wrap(ctx.ov_names),
      Rcpp::_["partable"]      = fcsem_partable_df(ctx.pt, ctx.names,
                                                   ctx.starts, &est),
      Rcpp::_["S"]             = S_out,
      Rcpp::_["nobs"]          = nobs_out,
      Rcpp::_["sample_mean"]   = R_NilValue,
      Rcpp::_["meanstructure"] = false,
      Rcpp::_["syntax"]        = syntax,
      Rcpp::_["fcsem"]         = true);
  out["optimizer_status"] = opt_status;
  out["grad_norm"] = est.grad_inf_norm;
  out["audit"] = audit_to_r(est.audit);
  // FCSEM fits do not run the L2 finalization audit (different evaluator
  // type); diagnostics is default-constructed and surfaces as the "audit
  // did not run" schema slot.
  out["diagnostics"] = diagnostics_to_r(est.diagnostics);
  return out;
}

Rcpp::DataFrame fcsem_standardized_rows_df(
    const std::vector<magmaan::measures::standardize::FcSemStandardizedRow>& rows) {
  const R_xlen_t n = static_cast<R_xlen_t>(rows.size());
  Rcpp::IntegerVector row(n), group(n), freev(n);
  Rcpp::CharacterVector lhs(n), op(n), rhs(n);
  Rcpp::NumericVector est(n), se(n), std_lv(n), std_lv_se(n), std_all(n),
      std_all_se(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& r = rows[static_cast<std::size_t>(i)];
    row[i] = static_cast<int>(r.row) + 1;
    lhs[i] = r.lhs;
    op[i] = std::string(magmaan::parse::to_string(r.op));
    rhs[i] = r.rhs;
    group[i] = r.group;
    freev[i] = r.free;
    est[i] = r.est;
    se[i] = r.se;
    std_lv[i] = r.std_lv;
    std_lv_se[i] = r.std_lv_se;
    std_all[i] = r.std_all;
    std_all_se[i] = r.std_all_se;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["row"] = row,
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["group"] = group,
      Rcpp::_["free"] = freev,
      Rcpp::_["est"] = est,
      Rcpp::_["se"] = se,
      Rcpp::_["std.lv"] = std_lv,
      Rcpp::_["std.lv.se"] = std_lv_se,
      Rcpp::_["std.all"] = std_all,
      Rcpp::_["std.all.se"] = std_all_se,
      Rcpp::_["stringsAsFactors"] = false);
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
  if (method == "ebm" || method == "EBM" || method == "Ebm") {
    return magmaan::measures::FactorScoreMethod::Ebm;
  }
  if (method == "ml" || method == "ML" || method == "Ml") {
    return magmaan::measures::FactorScoreMethod::Ml;
  }
  if (method == "eap" || method == "EAP" || method == "Eap") {
    return magmaan::measures::FactorScoreMethod::Eap;
  }
  Rcpp::stop("magmaan: factor score method must be 'regression', 'bartlett', "
             "'EBM', 'ML', or 'EAP' (got '%s')", method);
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

magmaan::estimate::OrdinalWeightKind ordinal_weight_from_estimator(
    const std::string& estimator, const char* call) {
  if (estimator == "ULS") return magmaan::estimate::OrdinalWeightKind::ULS;
  if (estimator == "DWLS") return magmaan::estimate::OrdinalWeightKind::DWLS;
  if (estimator == "WLS") return magmaan::estimate::OrdinalWeightKind::WLS;
  Rcpp::stop("magmaan: %s requires an ordinal ULS/DWLS/WLS fit", call);
}

magmaan::estimate::frontier::OrdinalStage2Weight
ordinal_stage2_weight_from_string(const std::string& s) {
  std::string key = s;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (key == "uls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Uls;
  }
  if (key == "dwls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Dwls;
  }
  if (key == "wls" || key == "adf") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Wls;
  }
  if (key == "nt" || key == "gls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Nt;
  }
  if (key == "dls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Dls;
  }
  Rcpp::stop("magmaan: ordinal stage2_weight must be one of "
             "'uls', 'dwls', 'wls'/'adf', 'nt'/'gls', or 'dls' (got '%s')",
             s);
}

std::string ordinal_stage2_label(const std::string& s) {
  std::string key = s;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (key == "gls") return "GLS";
  if (key == "adf") return "ADF";
  auto kind = ordinal_stage2_weight_from_string(s);
  switch (kind) {
    case magmaan::estimate::frontier::OrdinalStage2Weight::Uls:
      return "ULS";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Dwls:
      return "DWLS";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Wls:
      return "WLS";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Nt:
      return "NT";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Dls:
      return "DLS";
  }
  return "UNKNOWN";
}

std::string ordinal_weight_for_postfit(Rcpp::List fit,
                                       const std::string& estimator) {
  if (fit.containsElementNamed("ordinal_computational_weight")) {
    return Rcpp::as<std::string>(fit["ordinal_computational_weight"]);
  }
  return estimator;
}

Rcpp::List stats_from_fit_or_arg(Rcpp::List fit, SEXP arg,
                                 const char* field, const char* call) {
  if (!Rf_isNull(arg)) {
    if (TYPEOF(arg) != VECSXP) {
      Rcpp::stop("magmaan: %s requires `%s` as a categorical stats list",
                 call, field);
    }
    return Rcpp::List(arg);
  }
  if (!fit.containsElementNamed(field)) {
    Rcpp::stop("magmaan: %s requires `%s` (pass the data object used for fitting)",
               call, field);
  }
  return Rcpp::List(SEXP(fit[field]));
}

Rcpp::DataFrame score_table_df(
    const magmaan::inference::ScoreTestTable& tab,
    const magmaan::spec::LatentNames& names) {
  const R_xlen_t n = static_cast<R_xlen_t>(tab.rows.size());
  Rcpp::CharacterVector kind(n), op(n), lhs(n), rhs(n);
  Rcpp::IntegerVector row(n), group(n), df(n);
  Rcpp::NumericVector score(n), information(n), mi(n), pvalue(n), epc(n),
      epc_lv(n), epc_all(n), v_eff(n), mi_scaled(n), scaling_factor(n);
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
    v_eff[i] = r.v_eff;
    mi_scaled[i] = (r.mi_scaled == 0.0 && r.scaling_factor == 1.0 &&
                    r.v_eff == 0.0)
        ? r.mi
        : r.mi_scaled;
    scaling_factor[i] = r.scaling_factor;
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
      Rcpp::_["v.eff"] = v_eff,
      Rcpp::_["mi.scaled"] = mi_scaled,
      Rcpp::_["scaling.factor"] = scaling_factor,
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

// Convert a TerminalAudit (L1, driven coordinates) to an R sub-list. Surfaced
// as `fit$audit` — same mapping for OptimStatus as the existing `optimizer_status`
// string. `active_set` carries {-1, 0, +1} per driven coordinate, distinct
// from L2's `active_bounds_full` (which indexes the expanded θ).
Rcpp::List audit_to_r(const magmaan::optim::TerminalAudit& a) {
  const char* advisory = optim_status_to_r(a.advisory_status);
  Rcpp::IntegerVector active(static_cast<R_xlen_t>(a.active_set.size()));
  for (std::size_t i = 0; i < a.active_set.size(); ++i)
    active[static_cast<R_xlen_t>(i)] = static_cast<int>(a.active_set[i]);
  return Rcpp::List::create(
      Rcpp::_["stationary"]       = a.stationary,
      Rcpp::_["grad_inf_norm"]    = a.grad_inf_norm,
      Rcpp::_["grad_scaled_inf"]  = a.grad_scaled_inf,
      Rcpp::_["stationarity_rhs"] = a.stationarity_rhs,
      Rcpp::_["f_recomputed"]     = a.f_recomputed,
      Rcpp::_["f_consistent"]     = a.f_consistent,
      Rcpp::_["f_finite"]         = a.f_finite,
      Rcpp::_["active_set"]       = active,
      Rcpp::_["advisory_status"]  = advisory);
}

// Convert FitDiagnostics (L2, expanded θ) to an R sub-list. Surfaced as
// `fit$diagnostics`. Active-bound indices are converted 0-based → 1-based at
// the R boundary so they index `theta`/`partable` rows directly in R.
Rcpp::List diagnostics_to_r(const magmaan::estimate::FitDiagnostics& d) {
  Rcpp::LogicalVector sigma_pd(static_cast<R_xlen_t>(d.sigma_pd_per_block.size()));
  for (std::size_t b = 0; b < d.sigma_pd_per_block.size(); ++b)
    sigma_pd[static_cast<R_xlen_t>(b)] = d.sigma_pd_per_block[b];

  Rcpp::IntegerVector at_lo(d.active_bounds_full.at_lower.begin(),
                            d.active_bounds_full.at_lower.end());
  Rcpp::IntegerVector at_up(d.active_bounds_full.at_upper.begin(),
                            d.active_bounds_full.at_upper.end());
  for (R_xlen_t i = 0; i < at_lo.size(); ++i) at_lo[i] += 1;
  for (R_xlen_t i = 0; i < at_up.size(); ++i) at_up[i] += 1;

  return Rcpp::List::create(
      Rcpp::_["sigma_pd_per_block"]     = sigma_pd,
      Rcpp::_["sigma_pd_all"]           = d.sigma_pd_all,
      Rcpp::_["lin_eq_residual_inf"]    = d.lin_eq_residual_inf,
      Rcpp::_["lin_eq_satisfied"]       = d.lin_eq_satisfied,
      Rcpp::_["nl_eq_residual"]         = Rcpp::wrap(d.nl_eq_residual),
      Rcpp::_["nl_eq_residual_inf"]     = d.nl_eq_residual_inf,
      Rcpp::_["nl_eq_satisfied"]        = d.nl_eq_satisfied,
      Rcpp::_["active_bounds_lower"]    = at_lo,
      Rcpp::_["active_bounds_upper"]    = at_up,
      Rcpp::_["snlls_profile_fallback"] = d.snlls_profile_fallback);
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

  // Refined optimizer status, surfaced to R alongside the boolean `converged`
  // (which now means "clean stationary stop", not merely "did not error").
  using magmaan::optim::OptimStatus;
  const char* opt_status = optim_status_to_r(est.optimizer_status);

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["converged"]     = (est.optimizer_status == OptimStatus::Converged),
      Rcpp::_["estimator"]     = estimator,
      Rcpp::_["fmin"]          = est.fmin,
      Rcpp::_["iterations"]    = est.iterations,
      Rcpp::_["f_evals"]       = est.f_evals,
      Rcpp::_["g_evals"]       = est.g_evals,
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
  out["optimizer_status"] = opt_status;
  out["grad_norm"]        = est.grad_inf_norm;
  out["audit"]            = audit_to_r(est.audit);
  out["diagnostics"]      = diagnostics_to_r(est.diagnostics);
  return out;
}

Rcpp::List snlls_fit_result(Ctx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const magmaan::spec::Starts* starts,
                            const char* estimator,
                            const char* backend) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  out["backend"] = backend;
  out["snlls_compatible"] = true;
  // SNLLS profile shape: the outer optimizer drives n_nonlinear (β block);
  // n_linear (α block) is profiled out in closed form. The sentinel `-1` in
  // Estimates would mean "no separable split applied" — it should never reach
  // this path, so map negatives to NA so callers can guard cleanly anyway.
  out["n_nonlinear"] = est.n_nonlinear >= 0
      ? Rcpp::IntegerVector::create(est.n_nonlinear)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  out["n_linear"] = est.n_linear >= 0
      ? Rcpp::IntegerVector::create(est.n_linear)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  // SNLLS inner-solve telemetry: count of `profile_at()` cache misses that
  // took the fast Cholesky-on-normal-equations path vs the rank-revealing
  // QR fallback. Same sentinel semantics as `n_nonlinear` / `n_linear`.
  out["n_alpha_solve_fast"] = est.n_alpha_solve_fast >= 0
      ? Rcpp::IntegerVector::create(est.n_alpha_solve_fast)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  out["n_alpha_solve_fallback"] = est.n_alpha_solve_fallback >= 0
      ? Rcpp::IntegerVector::create(est.n_alpha_solve_fallback)
      : Rcpp::IntegerVector::create(NA_INTEGER);
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

Rcpp::List bounds_to_r(const magmaan::estimate::Bounds& bounds) {
  return Rcpp::List::create(
      Rcpp::_["lower"] = Rcpp::wrap(bounds.lower),
      Rcpp::_["upper"] = Rcpp::wrap(bounds.upper));
}

std::string bounds_preset_key(Rcpp::Nullable<Rcpp::String> preset) {
  if (preset.isNull()) return "none";
  std::string key = Rcpp::as<std::string>(preset.get());
  const auto first = key.find_first_not_of(" \t\r\n");
  const auto last = key.find_last_not_of(" \t\r\n");
  key = (first == std::string::npos) ? std::string{} :
      key.substr(first, last - first + 1);
  for (char& ch : key) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return key;
}

magmaan::estimate::Bounds twolevel_bounds_from_nullable(
    Rcpp::Nullable<Rcpp::List> bounds,
    Rcpp::Nullable<Rcpp::String> preset,
    const magmaan::spec::LatentStructure& pt,
    const magmaan::model::MatrixRep& rep,
    const magmaan::data::ClusterSampleStats& cs) {
  magmaan::estimate::Bounds explicit_bounds = bounds_from_nullable(bounds);
  const std::string key = bounds_preset_key(preset);
  if (key.empty() || key == "none" || key == "no" || key == "false" ||
      key == "unbounded") {
    return explicit_bounds;
  }
  if (!explicit_bounds.empty()) {
    Rcpp::stop("magmaan: fit_twolevel(): pass either explicit `bounds` or a "
               "named bounds preset, not both");
  }

  if (key == "pos.var" || key == "pos-var" || key == "variance" ||
      key == "variance-bounds") {
    auto b_or = magmaan::estimate::variance_bounds(pt);
    if (!b_or.has_value()) stop_post(b_or.error());
    return *b_or;
  }
  if (key != "standard" && key != "wide" && key != "default" &&
      key != "loading" && key != "loadings") {
    Rcpp::stop("magmaan: fit_twolevel(): unsupported bounds preset '%s'",
               key.c_str());
  }

  auto samp_or = magmaan::estimate::twolevel::twolevel_h1_sample_stats(cs, rep);
  if (!samp_or.has_value()) stop_fit(samp_or.error());

  magmaan::post_expected<magmaan::estimate::Bounds> b_or;
  if (key == "standard") {
    b_or = magmaan::estimate::standard_bounds(pt, *samp_or);
  } else if (key == "wide" || key == "default") {
    b_or = magmaan::estimate::wide_bounds(pt, *samp_or);
  } else if (key == "loading" || key == "loadings") {
    b_or = magmaan::estimate::loading_bounds(pt, *samp_or);
  }
  if (!b_or.has_value()) stop_post(b_or.error());
  return *b_or;
}

magmaan::estimate::frontier::RBMOptions
rbm_options_from(Rcpp::Nullable<Rcpp::String> optimizer,
                 Rcpp::Nullable<Rcpp::List> control) {
  magmaan::estimate::frontier::RBMOptions opts;
  if (optimizer.isNotNull()) {
    opts.backend = backend_from_optimizer_arg(optimizer);
  }
  opts.optim = optim_opts_from(control);
  if (control.isNotNull()) {
    Rcpp::List l(control.get());
    if (l.containsElementNamed("fd_rel_step"))
      opts.fd_rel_step = Rcpp::as<double>(l["fd_rel_step"]);
    if (l.containsElementNamed("fd_abs_step"))
      opts.fd_abs_step = Rcpp::as<double>(l["fd_abs_step"]);
    if (l.containsElementNamed("check_admissibility"))
      opts.check_admissibility = Rcpp::as<bool>(l["check_admissibility"]);
    if (l.containsElementNamed("admissibility_tol"))
      opts.admissibility_tol = Rcpp::as<double>(l["admissibility_tol"]);
  }
  return opts;
}

std::string rbm_method_key(std::string method) {
  for (char& ch : method) {
    if (ch == '-') ch = '_';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (method == "explicit" || method == "erb" || method == "erbm") {
    return "explicit";
  }
  if (method == "implicit" || method == "irb" || method == "irbm") {
    return "implicit";
  }
  Rcpp::stop("magmaan: RBM method must be 'explicit' or 'implicit'");
}

Rcpp::List rbm_metadata_to_r(
    const magmaan::estimate::frontier::RBMResult& r,
    const std::string& method,
    const char* base_estimator) {
  return Rcpp::List::create(
      Rcpp::_["method"] = method,
      Rcpp::_["base_estimator"] = base_estimator,
      Rcpp::_["correction"] = Rcpp::wrap(r.correction),
      Rcpp::_["adjustment"] = Rcpp::wrap(r.adjustment),
      Rcpp::_["information"] = Rcpp::wrap(r.information),
      Rcpp::_["meat"] = Rcpp::wrap(r.meat),
      Rcpp::_["information_reduced"] = Rcpp::wrap(r.information_reduced),
      Rcpp::_["meat_reduced"] = Rcpp::wrap(r.meat_reduced),
      Rcpp::_["trace"] = r.trace_term,
      Rcpp::_["penalty"] = r.penalty,
      Rcpp::_["penalty_per_observation"] = r.penalty_per_observation,
      Rcpp::_["penalized_fmin"] = r.penalized_fmin,
      Rcpp::_["admissible"] = r.admissible,
      Rcpp::_["bounds_satisfied"] = r.bounds_satisfied,
      Rcpp::_["sigma_pd"] = r.sigma_pd,
      Rcpp::_["warnings"] = Rcpp::wrap(r.warnings));
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

SEXP weight_to_r(const magmaan::estimate::gmm::Weight& W) {
  if (W.size() == 1) return Rcpp::wrap(W[0]);
  Rcpp::List out(static_cast<R_xlen_t>(W.size()));
  for (std::size_t b = 0; b < W.size(); ++b) {
    out[static_cast<R_xlen_t>(b)] = Rcpp::wrap(W[b]);
  }
  return out;
}

Rcpp::List ordinal_stats_to_r(const magmaan::data::OrdinalStats& s) {
  const R_xlen_t nb = static_cast<R_xlen_t>(s.R.size());
  Rcpp::List R(nb), thresholds(nb), threshold_ov(nb), threshold_level(nb),
      moments(nb), NACOV(nb), W_dwls(nb), W_wls(nb), moment_influence(nb),
      int_data(nb), moment_bread(nb), n_levels(nb);
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
    NACOV[b] = bi < s.NACOV.size()
        ? Rcpp::wrap(s.NACOV[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    W_dwls[b] = bi < s.W_dwls.size()
        ? Rcpp::wrap(s.W_dwls[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    W_wls[b] = bi < s.W_wls.size()
        ? Rcpp::wrap(s.W_wls[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    moment_influence[b] = bi < s.moment_influence.size()
        ? Rcpp::wrap(s.moment_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    int_data[b] = bi < s.int_data.size()
        ? Rcpp::wrap(s.int_data[bi])
        : Rcpp::wrap(Eigen::MatrixXi(0, 0));
    moment_bread[b] = bi < s.moment_bread.size()
        ? Rcpp::wrap(s.moment_bread[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
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
      Rcpp::_["moment_influence"] = moment_influence,
      Rcpp::_["int_data"] = int_data,
      Rcpp::_["moment_bread"] = moment_bread,
      Rcpp::_["nobs"] = nobs,
      Rcpp::_["n_levels"] = n_levels);
  if (!s.pairwise_gamma.empty()) {
    Rcpp::List support_i(static_cast<R_xlen_t>(s.moment_support_i.size()));
    Rcpp::List support_j(static_cast<R_xlen_t>(s.moment_support_j.size()));
    Rcpp::List moment_nobs(static_cast<R_xlen_t>(s.moment_n_obs.size()));
    Rcpp::List overlap(static_cast<R_xlen_t>(s.moment_overlap_n_obs.size()));
    for (R_xlen_t b = 0; b < support_i.size(); ++b) {
      const auto bi = static_cast<std::size_t>(b);
      Rcpp::IntegerVector si(static_cast<R_xlen_t>(s.moment_support_i[bi].size()));
      Rcpp::IntegerVector sj(static_cast<R_xlen_t>(s.moment_support_j[bi].size()));
      Rcpp::NumericVector mn(static_cast<R_xlen_t>(s.moment_n_obs[bi].size()));
      for (R_xlen_t k = 0; k < si.size(); ++k) {
        si[k] = s.moment_support_i[bi][static_cast<std::size_t>(k)] + 1;
        sj[k] = s.moment_support_j[bi][static_cast<std::size_t>(k)] >= 0
                    ? s.moment_support_j[bi][static_cast<std::size_t>(k)] + 1
                    : NA_INTEGER;
        mn[k] = static_cast<double>(s.moment_n_obs[bi][static_cast<std::size_t>(k)]);
      }
      support_i[b] = si;
      support_j[b] = sj;
      moment_nobs[b] = mn;
      overlap[b] = Rcpp::wrap(s.moment_overlap_n_obs[bi].cast<double>());
    }
    out.attr("pd_gamma") = s.pairwise_gamma;
    out.attr("moment_support_i") = support_i;
    out.attr("moment_support_j") = support_j;
    out.attr("moment_nobs") = moment_nobs;
    out.attr("moment_overlap_nobs") = overlap;
  }
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_ordinal_data", "list");
  return out;
}

Rcpp::List mixed_ordinal_stats_to_r(const magmaan::data::MixedOrdinalStats& s) {
  const R_xlen_t nb = static_cast<R_xlen_t>(s.R.size());
  Rcpp::List R(nb), mean(nb), ordered_mask(nb), thresholds(nb), threshold_ov(nb),
      threshold_level(nb), moments(nb), NACOV(nb), W_dwls(nb), W_wls(nb),
      moment_influence(nb), gamma_diag_influence(nb),
      gamma_full_influence(nb), raw_data(nb), n_levels(nb);
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
    moment_influence[b] = bi < s.moment_influence.size()
        ? Rcpp::wrap(s.moment_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    gamma_diag_influence[b] = bi < s.gamma_diag_influence.size()
        ? Rcpp::wrap(s.gamma_diag_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    gamma_full_influence[b] = bi < s.gamma_full_influence.size()
        ? Rcpp::wrap(s.gamma_full_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    raw_data[b] = bi < s.raw_data.size()
        ? Rcpp::wrap(s.raw_data[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
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
      Rcpp::_["moment_influence"] = moment_influence,
      Rcpp::_["gamma_diag_influence"] = gamma_diag_influence,
      Rcpp::_["gamma_full_influence"] = gamma_full_influence,
      Rcpp::_["raw_data"] = raw_data,
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

magmaan::data::PairwiseOrdinalHuberResidualStatsOptions
ordinal_huber_residual_options_from(std::string clip, double k) {
  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions options;
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

using FimlPack = magmaan::estimate::fiml::FIMLPack;
using FimlH1 = magmaan::estimate::fiml::FIMLH1;

Rcpp::XPtr<FimlPack> fiml_pack_xptr(FimlPack pack) {
  Rcpp::XPtr<FimlPack> xp(new FimlPack(std::move(pack)), true);
  xp.attr("class") =
      Rcpp::CharacterVector::create("magmaan_fiml_pack", "externalptr");
  return xp;
}

Rcpp::XPtr<FimlH1> fiml_h1_xptr(FimlH1 h1) {
  Rcpp::XPtr<FimlH1> xp(new FimlH1(std::move(h1)), true);
  xp.attr("class") =
      Rcpp::CharacterVector::create("magmaan_fiml_h1", "externalptr");
  return xp;
}

const FimlPack* fiml_pack_ptr_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("fiml_pack")) return nullptr;
  SEXP xp = fit["fiml_pack"];
  if (Rf_isNull(xp)) return nullptr;
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("magmaan: fit$fiml_pack is not an external pointer");
  }
  void* addr = R_ExternalPtrAddr(xp);
  if (addr == nullptr) Rcpp::stop("magmaan: fit$fiml_pack is null");
  return static_cast<const FimlPack*>(addr);
}

const FimlH1* fiml_h1_ptr_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("fiml_h1")) return nullptr;
  SEXP xp = fit["fiml_h1"];
  if (Rf_isNull(xp)) return nullptr;
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("magmaan: fit$fiml_h1 is not an external pointer");
  }
  void* addr = R_ExternalPtrAddr(xp);
  if (addr == nullptr) Rcpp::stop("magmaan: fit$fiml_h1 is null");
  return static_cast<const FimlH1*>(addr);
}

const FimlPack& fiml_pack_for_fit(Rcpp::List fit,
                                  const magmaan::data::RawData& raw,
                                  std::unique_ptr<FimlPack>& owned) {
  if (const FimlPack* pack = fiml_pack_ptr_from_fit(fit)) return *pack;
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  owned = std::make_unique<FimlPack>(std::move(*pack_or));
  return *owned;
}

const FimlH1& fiml_h1_for_fit(Rcpp::List fit,
                              const magmaan::data::RawData& raw,
                              const FimlPack& pack,
                              std::unique_ptr<FimlH1>& owned) {
  if (const FimlH1* h1 = fiml_h1_ptr_from_fit(fit)) return *h1;
  auto h1_or = magmaan::estimate::fiml::fiml_h1_moments(raw, pack);
  if (!h1_or.has_value()) stop_fit(h1_or.error());
  owned = std::make_unique<FimlH1>(std::move(*h1_or));
  return *owned;
}

using SaturatedMoments = magmaan::estimate::fiml::SaturatedMoments;

// Shared Stage-1 saturated FIML moments (the EM moments + H/J/acov = Γ_mis):
// reuse the fit$stage1 reconstruction when present (ML2S, via
// magmaanr::saturated_from_stage1), otherwise compute once. Mirrors
// fiml_pack_for_fit / fiml_h1_for_fit so FMG, two-stage SB, and the nested LRT
// all consume one saturated build instead of three.
const SaturatedMoments& fiml_saturated_for_fit(
    Rcpp::List fit, const magmaan::data::RawData& raw, const FimlPack& pack,
    const FimlH1& h1, std::unique_ptr<SaturatedMoments>& owned) {
  owned = std::make_unique<SaturatedMoments>();
  if (magmaanr::saturated_from_stage1(fit, *owned)) return *owned;
  auto sm_or = magmaan::estimate::fiml::saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) stop_post(sm_or.error());
  *owned = std::move(*sm_or);
  return *owned;
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
  // The estimated-weight (Hall-Inoue) score-test path needs the per-case
  // influence + integer data; read them through when present (the high-level
  // DWLS path stores them).
  const bool has_mi = x.containsElementNamed("moment_influence");
  Rcpp::List mil = has_mi ? Rcpp::List(x["moment_influence"]) : Rcpp::List();
  const bool has_id = x.containsElementNamed("int_data");
  Rcpp::List idl = has_id ? Rcpp::List(x["int_data"]) : Rcpp::List();
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
    if (has_mi)
      out.moment_influence.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(mil[b])));
    if (has_id)
      out.int_data.push_back(
          Rcpp::as<Eigen::MatrixXi>(Rcpp::IntegerMatrix(idl[b])));
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
  // Carry the group.equal families on the partable so the nested ordinal LR
  // test (which rebuilds each structure via ctx_from_fit) re-applies the
  // Wu-Estabrook release; from_lavaan_partable would otherwise drop them.
  if (!ctx.pt.group_equal.empty()) {
    Rcpp::DataFrame pt_out = out["partable"];
    stamp_group_equal_attr(pt_out, ctx.pt.group_equal);
    out["partable"] = pt_out;
  }
  Rcpp::List stats_r = ordinal_stats_to_r(stats);
  out["ordinal_stats"] = stats_r;
  out["thresholds"] = stats_r["thresholds"];
  out["polychoric"] = stats_r["R"];
  return out;
}

std::vector<std::vector<std::int32_t>>
n_levels_from_arg(Rcpp::List n_levels) {
  std::vector<std::vector<std::int32_t>> out;
  out.reserve(static_cast<std::size_t>(n_levels.size()));
  for (R_xlen_t b = 0; b < n_levels.size(); ++b) {
    Rcpp::IntegerVector lev(n_levels[b]);
    out.emplace_back(Rcpp::as<std::vector<std::int32_t>>(lev));
  }
  return out;
}

Ctx pairwise_ordinal_ctx_from_partable(SEXP partable,
                                       const char* caller,
                                       const magmaan::data::OrdinalStats& stats,
                                       magmaan::spec::Starts& starts) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, caller);
  starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty()
      ? std::vector<std::string>{}
      : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  return ctx;
}

Rcpp::List pairwise_objective_to_r(
    const magmaan::estimate::frontier::PairwiseOrdinalCompositeResult& obj) {
  int n_pairs = 0;
  std::int64_t n_obs = 0;
  for (const auto& block : obj.blocks) {
    n_pairs += static_cast<int>(block.pairs.size());
    n_obs += block.n_obs;
  }
  return Rcpp::List::create(
      Rcpp::_["negloglik"] = obj.negloglik,
      Rcpp::_["weighted_negloglik"] = obj.weighted_negloglik,
      Rcpp::_["n_pairs"] = n_pairs,
      Rcpp::_["nobs_total"] = static_cast<double>(n_obs),
      Rcpp::_["df"] = obj.df);
}

Rcpp::List pairwise_godambe_to_r(
    const magmaan::estimate::frontier::PairwiseOrdinalCompositeGodambe& g) {
  return Rcpp::List::create(
      Rcpp::_["bread"] = Rcpp::wrap(g.bread),
      Rcpp::_["meat"] = Rcpp::wrap(g.meat),
      Rcpp::_["vcov"] = Rcpp::wrap(g.vcov),
      Rcpp::_["vcov_naive"] = Rcpp::wrap(g.vcov_naive),
      Rcpp::_["se"] = Rcpp::wrap(g.se),
      Rcpp::_["se_naive"] = Rcpp::wrap(g.se_naive),
      Rcpp::_["casewise_scores"] = Rcpp::wrap(g.casewise_scores),
      Rcpp::_["condition_bread"] = g.condition_bread);
}

Rcpp::List pairwise_lr_to_r(const magmaan::robust::LRSatorra2000Result& r) {
  Rcpp::CharacterVector warns(static_cast<R_xlen_t>(r.warnings.size()));
  for (R_xlen_t i = 0; i < warns.size(); ++i) {
    warns[i] = r.warnings[static_cast<std::size_t>(i)];
  }
  return Rcpp::List::create(
      Rcpp::_["T_diff"] = r.T_diff,
      Rcpp::_["df_diff"] = r.df_diff,
      Rcpp::_["p_unscaled"] = r.p_unscaled,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(r.eigenvalues),
      Rcpp::_["scale_c"] = r.scale_c,
      Rcpp::_["T_scaled"] = r.T_scaled,
      Rcpp::_["p_scaled"] = r.p_scaled,
      Rcpp::_["adjust_d0"] = r.adjust_d0,
      Rcpp::_["T_adjusted"] = r.T_adjusted,
      Rcpp::_["p_adjusted"] = r.p_adjusted,
      Rcpp::_["p_mixture"] = r.p_mixture,
      Rcpp::_["warnings"] = warns);
}

magmaan::data::frontier::CovarianceShrinkageKind
shrinkage_kind_from_string(const std::string& kind) {
  using K = magmaan::data::frontier::CovarianceShrinkageKind;
  if (kind == "none") return K::None;
  if (kind == "ridge") return K::Ridge;
  if (kind == "identity") return K::IdentityTarget;
  if (kind == "diagonal") return K::DiagonalTarget;
  if (kind == "constant_correlation") return K::ConstantCorrelation;
  Rcpp::stop("magmaan: shrinkage kind must be none, ridge, identity, diagonal, or constant_correlation");
}

// The Phase 3-era Ceres-specific option helpers were
// retired in Phase 4 alongside the per-Ceres Rcpp shim explosion. All
// optimizer-control fields now route through `optim_opts_from(control)` in
// internal.hpp — the OptimOptions struct (max_iter / ftol / gtol / history)
// is the shared option vocabulary across magmaan's optimizer roster, and
// Ceres-specific extras (ptol / verbose) were already dropped on the old
// path, so the unified control list is behaviour-preserving.

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

// [[Rcpp::export]]
Rcpp::List bounds_variance_impl(SEXP partable) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_variance");
  auto b_or = magmaan::estimate::variance_bounds(parsed.structure);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List bounds_standard_impl(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_standard");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  auto b_or = magmaan::estimate::standard_bounds(ctx.pt, ctx.samp);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List bounds_wide_impl(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_wide");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  auto b_or = magmaan::estimate::wide_bounds(ctx.pt, ctx.samp);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List bounds_loading_impl(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_loading");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  auto b_or = magmaan::estimate::loading_bounds(ctx.pt, ctx.samp);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// =============================================================================

// fit_fit() declaration — the historical entry point lives as a thin alias of
// fit_ml_impl() for the `estimate_fit` magmaan_core slot. Definition is below
// fit_ml_impl so the alias can call it.
Rcpp::List fit_ml_impl(SEXP partable, Rcpp::List sample_stats,
                       Rcpp::Nullable<Rcpp::String> optimizer,
                       Rcpp::Nullable<Rcpp::List>   control,
                       Rcpp::Nullable<Rcpp::List>   bounds);

// [[Rcpp::export]]
Rcpp::List fit_fit(SEXP partable, Rcpp::List sample_stats,
                   Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                   Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                   Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  return fit_ml_impl(partable, sample_stats, optimizer, control, bounds);
}

// fit_ml() — public ML spelling. Threads through an `optimizer` string and
// optional `control` list. Backends:
//   "nlopt-lbfgs" (default), "ipopt", "port", "nlopt-slsqp",
//   "nlopt-tnewton", "nlopt-var2"  (any scalar-shape backend)
// "ceres" / "ceres-bfgs" are rejected — Ceres applies to the LS path only.
// "nlopt-bobyqa" requires finite bounds, supplied via `bounds`.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_impl(SEXP partable, Rcpp::List sample_stats,
                       Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                       Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                       Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_ml");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_ml(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML");
}

// fit_twolevel() — two-level (multilevel) normal-theory ML over clustered raw
// data. `data` holds the observed columns (named); `cluster_id` is a (per-row)
// cluster index; `group_id` (optional) is a per-row 0-based, contiguous group
// index for multi-group models (NULL ⇒ single group). Returns theta-hat,
// observed-info SEs, the LRT chi-square (vs the saturated H1), df, a fitted
// partable, and basic diagnostics.
//
// DIRECT-CORE BYPASS (v1 contract): this builds matrix_rep + cluster sample
// stats + start values and drives estimate::twolevel directly; it does *not*
// route through api::fit / sem.cpp.
//
// [[Rcpp::export]]
Rcpp::List fit_twolevel_impl(SEXP partable, Rcpp::NumericMatrix data,
                             Rcpp::IntegerVector cluster_id,
                             Rcpp::Nullable<Rcpp::IntegerVector> group_id = R_NilValue,
                             Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                             Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                             Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
                             Rcpp::Nullable<Rcpp::String> bounds_preset = R_NilValue) {
  namespace tl = magmaan::estimate::twolevel;
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_twolevel");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  magmaan::spec::LatentStructure pt = std::move(parsed.structure);
  magmaan::spec::LatentNames names = std::move(parsed.names);

  auto rep_or = magmaan::model::build_matrix_rep(pt, &names);
  if (!rep_or.has_value()) {
    Rcpp::stop("magmaan: fit_twolevel(): matrix_rep failed: %s",
               rep_or.error().detail.c_str());
  }
  const magmaan::model::MatrixRep rep = std::move(*rep_or);
  if (rep.ov_names.empty()) Rcpp::stop("magmaan: fit_twolevel(): empty model");

  // v1 (shared observed set): every block — within/between, all groups — carries
  // the same p observed variables in the same order, so a single column
  // permutation to the block-0 within order serves every group.
  for (std::size_t b = 1; b < rep.ov_names.size(); ++b) {
    if (rep.ov_names[b] != rep.ov_names[0]) {
      Rcpp::stop("magmaan: fit_twolevel(): v1 requires a shared observed set "
                 "with identical ordering across level/group blocks");
    }
  }

  // Order data columns to the within-block observed order (shared variable set).
  const std::vector<int> perm = perm_for_cols(data, rep.ov_names[0], "data");
  const int n = data.nrow();
  const int p = static_cast<int>(perm.size());
  Eigen::MatrixXd X(n, p);
  for (int r = 0; r < n; ++r)
    for (int k = 0; k < p; ++k)
      X(r, k) = data(r, perm[static_cast<std::size_t>(k)]);

  if (cluster_id.size() != n) {
    Rcpp::stop("magmaan: fit_twolevel(): cluster_id length (%d) != nrow(data) (%d)",
               static_cast<int>(cluster_id.size()), n);
  }
  std::vector<std::int32_t> cid(cluster_id.begin(), cluster_id.end());

  // Per-row group index. NULL ⇒ a single group (all rows). When supplied, the
  // values are taken as 0-based, contiguous group indices that line up with the
  // model's per-group level blocks (no first-appearance remap) so the index is
  // unambiguous and matches lavaan's fixed group ordering.
  std::vector<int> grow(static_cast<std::size_t>(n), 0);
  int ng = 1;
  if (group_id.isNotNull()) {
    Rcpp::IntegerVector gv(group_id.get());
    if (gv.size() != n) {
      Rcpp::stop("magmaan: fit_twolevel(): group_id length (%d) != nrow(data) (%d)",
                 static_cast<int>(gv.size()), n);
    }
    int gmax = -1;
    for (int r = 0; r < n; ++r) {
      const int g = gv[r];
      if (g < 0) Rcpp::stop("magmaan: fit_twolevel(): group_id must be 0-based");
      grow[static_cast<std::size_t>(r)] = g;
      if (g > gmax) gmax = g;
    }
    ng = gmax + 1;
    std::vector<char> seen(static_cast<std::size_t>(ng), 0);
    for (int r = 0; r < n; ++r)
      seen[static_cast<std::size_t>(grow[static_cast<std::size_t>(r)])] = 1;
    for (int g = 0; g < ng; ++g)
      if (!seen[static_cast<std::size_t>(g)])
        Rcpp::stop("magmaan: fit_twolevel(): group_id has an empty group index %d", g);
  }

  std::vector<std::int32_t> cols(static_cast<std::size_t>(p));
  for (int k = 0; k < p; ++k) cols[static_cast<std::size_t>(k)] = k;

  // Build per-group two-level sufficient statistics and stitch them into one
  // ClusterSampleStats. The per-group sufficient statistics are independent (no
  // cross-group coupling), so calling the single-group cluster_sample_stats once
  // per group subset is numerically identical to one multi-group pass.
  //
  // TODO(stream-A multi-group core): once data::cluster_sample_stats gains its
  // group selector (and data_from_cluster accepts multiple groups), replace this
  // per-group stitch with that single canonical multi-group call. The downstream
  // estimation (objective/H1/information) already loops over cs.groups, so only
  // this construction step changes.
  magmaan::data::ClusterSampleStats cs;
  cs.within_ov_index = cols;
  cs.between_ov_index = cols;
  cs.groups.reserve(static_cast<std::size_t>(ng));
  for (int g = 0; g < ng; ++g) {
    std::vector<int> rows;
    rows.reserve(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r)
      if (grow[static_cast<std::size_t>(r)] == g) rows.push_back(r);
    const int ng_rows = static_cast<int>(rows.size());
    Eigen::MatrixXd Xg(ng_rows, p);
    std::vector<std::int32_t> cidg(static_cast<std::size_t>(ng_rows));
    for (int i = 0; i < ng_rows; ++i) {
      Xg.row(i) = X.row(rows[static_cast<std::size_t>(i)]);
      cidg[static_cast<std::size_t>(i)] =
          cid[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])];
    }
    auto cg_or = magmaan::data::cluster_sample_stats(Xg, cidg, cols, cols);
    if (!cg_or.has_value()) {
      Rcpp::stop("magmaan: fit_twolevel(): cluster statistics failed: %s",
                 cg_or.error().detail.c_str());
    }
    cs.groups.push_back(std::move(cg_or->groups.front()));
  }

  // Multi-group two-level needs per-group within/between level blocks in the rep.
  const auto pairs = magmaan::model::level_block_pairs(rep);
  if (static_cast<int>(pairs.size()) < ng) {
    Rcpp::stop("magmaan: fit_twolevel(): data has %d group(s) but the model has "
               "%d level-block group(s); multi-group two-level requires a "
               "grouped two-level model", ng, static_cast<int>(pairs.size()));
  }

  auto x0_or = tl::twolevel_start_values(pt, rep, cs, starts);
  if (!x0_or.has_value()) stop_fit(x0_or.error());
  const Eigen::VectorXd x0 = std::move(*x0_or);

  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  const magmaan::estimate::Bounds fit_bounds =
      twolevel_bounds_from_nullable(bounds, bounds_preset, pt, rep, cs);
  auto est_or = tl::fit_ml_twolevel(pt, rep, cs, x0, fit_bounds, backend,
                                    optim_opts_from(control));
  if (!est_or.has_value()) stop_fit(est_or.error());
  const magmaan::estimate::Estimates est = std::move(*est_or);

  // Observed-information SEs.
  Rcpp::NumericVector se(est.theta.size(), NA_REAL);
  auto ev_or = magmaan::model::ModelEvaluator::build(pt, rep);
  if (ev_or.has_value()) {
    const magmaan::model::ModelEvaluator ev = std::move(*ev_or);
    auto info_or = tl::twolevel_information(ev, cs, est.theta, /*expected=*/false);
    if (info_or.has_value()) {
      const Eigen::MatrixXd vcov = info_or->inverse();
      for (Eigen::Index k = 0; k < est.theta.size(); ++k) {
        const double v = vcov(k, k);
        se[k] = (v > 0.0) ? std::sqrt(v) : NA_REAL;
      }
    }
  }

  // LRT chi-square vs the saturated H1 (chi2 = F_model - F_H1 = 2*fmin - F_H1).
  double chisq = NA_REAL;
  int df = NA_INTEGER;
  auto h1_or = tl::twolevel_h1_moments(cs);
  if (h1_or.has_value()) {
    chisq = 2.0 * est.fmin - h1_or->value;
    // Saturated free parameters, summed over groups: per group Σ_W and Σ_B are
    // each unstructured p_g(p_g+1)/2 and μ_B adds p_g, i.e. p_g(p_g+1) + p_g.
    // df = Σ_g [ p_g(p_g+1) + p_g ] − q.  Single group reduces to p(p+1)+p.
    long psat = 0;
    for (const auto& gst : cs.groups) {
      const long pg = static_cast<long>(gst.p_within);
      psat += pg * (pg + 1L) + pg;
    }
    df = static_cast<int>(psat - static_cast<long>(pt.n_free()));
  }

  Rcpp::NumericVector theta(est.theta.size());
  for (Eigen::Index k = 0; k < est.theta.size(); ++k) theta[k] = est.theta[k];

  // Per-group level-1 sizes (within) and cluster counts (between), for printing.
  Rcpp::IntegerVector nobs_out(ng), nclusters_out(ng);
  long ntotal = 0, ncl_total = 0;
  for (int g = 0; g < ng; ++g) {
    const auto& gst = cs.groups[static_cast<std::size_t>(g)];
    nobs_out[g]      = static_cast<int>(gst.n_within);
    nclusters_out[g] = static_cast<int>(gst.n_clusters);
    ntotal    += gst.n_within;
    ncl_total += gst.n_clusters;
  }

  using magmaan::optim::OptimStatus;
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["converged"]    = (est.optimizer_status == OptimStatus::Converged),
      Rcpp::_["estimator"]    = "ML",
      Rcpp::_["fmin"]         = est.fmin,
      Rcpp::_["iterations"]   = est.iterations,
      Rcpp::_["f_evals"]      = est.f_evals,
      Rcpp::_["g_evals"]      = est.g_evals,
      Rcpp::_["npar"]         = static_cast<int>(pt.n_free()),
      Rcpp::_["ngroups"]      = ng,
      Rcpp::_["ntotal"]       = static_cast<int>(ntotal),
      Rcpp::_["nclusters"]    = static_cast<int>(ncl_total),
      Rcpp::_["group_var"]    = names.group_var,
      Rcpp::_["group_labels"] = Rcpp::wrap(names.group_labels),
      Rcpp::_["theta"]        = theta,
      Rcpp::_["se"]           = se,
      Rcpp::_["ov_names"]     = Rcpp::wrap(rep.ov_names[0]),
      Rcpp::_["partable"]     = partable_df(pt, names, est, &starts),
      Rcpp::_["nobs"]         = nobs_out,
      Rcpp::_["nclusters_by_group"] = nclusters_out,
      Rcpp::_["meanstructure"] = true);
  out["chisq"] = chisq;
  out["df"]    = df;
  out["level"] = 2;
  out["optimizer_status"] = optim_status_to_r(est.optimizer_status);
  out["grad_norm"]        = est.grad_inf_norm;
  out["audit"]            = audit_to_r(est.audit);
  out["diagnostics"]      = diagnostics_to_r(est.diagnostics);
  return out;
}

// fit_ml_fisher() — normal-theory ML via local Fisher scoring. This path uses
// `control = list(max_iter, ftol, gtol)` for the scoring loop; there is no
// optimizer backend because the Fisher step plus Armijo globalization is the
// optimizer.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_fisher_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> control = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds  = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_fisher");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_ml_fisher(
      ctx.pt, ctx.rep, ctx.samp, x0, bounds_from_nullable(bounds),
      optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML-Fisher");
}

// fit_ml_fisher_snlls() — local Fisher scoring with a Schur-complement solve
// over the same beta/alpha split used by the SNLLS research paths.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_fisher_snlls_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds  = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_fisher_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  auto e_or = magmaan::estimate::fit_ml_fisher_snlls(
      ctx.pt, ctx.rep, ctx.samp, x0, bounds_from_nullable(bounds),
      optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML-Fisher-SNLLS");
}

// fit_ml_irls() — normal-theory ML via iteratively reweighted GLS (Fisher
// scoring). Same ML objective as fit_ml(), different algorithm: at iterate θ_k
// build the expected Fisher information weight W(θ_k) = ½D'(Σ(θ_k)⁻¹⊗Σ(θ_k)⁻¹)D,
// solve the inner GLS subproblem, accept a damped step via Armijo on F_ML.
// Mean structures adjust the frozen inner covariance target by the current
// mean residual d_k d_k' so the inner score matches the ML score up to scale.
//
// `optimizer` names the *inner* LS solver. NULL defaults to "port-nls" (the
// natural LS-shape Gauss-Newton trust region for the inner subproblem);
// "nlopt-lbfgs" and "ceres" also work but throw away the residual structure.
// `control` accepts both the inner solver's max_iter / ftol / gtol and the
// outer-loop irls_max_outer / irls_ftol / irls_gtol / irls_armijo_c knobs.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_irls_impl(SEXP partable, Rcpp::List sample_stats,
                            Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                            Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                            Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_irls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::PortNls
                         : backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_ml_irls(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), backend, optim_opts_from(control),
      irls_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML");
}

// fit_ml_irls_snlls() — same as fit_ml_irls but each outer iterate's inner
// GLS subproblem is solved by Golub–Pereyra variable projection (β = Λ, B
// optimized; α = Θ, Ψ, ν closed-form). Rejects box bounds, nonlinear
// constraints, and the non-separable models gmm::gp_compatible rejects.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_irls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                                  Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                  Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                  Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_irls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::PortNls
                         : backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_ml_irls_snlls(ctx.pt, ctx.rep, ctx.samp,
      x0, bounds_from_nullable(bounds), backend, optim_opts_from(control),
      irls_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  // Use snlls_fit_result so the n_nonlinear / n_linear columns surface on
  // the SNLLS speed-survey path the same way fit_snlls_gls's do.
  return snlls_fit_result(ctx, est, &starts, "ML-IRLS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// frontier_fit_ml_ridge_continuation() — complete-data ML through a covariance
// continuation path, warm-starting each stage from the previous fit.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_ml_ridge_continuation_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> alphas = R_NilValue,
    std::string target = "diagonal",
    bool include_endpoint = true,
    double diagonal_floor = 1e-8) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_ml_ridge_continuation");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);

  magmaan::estimate::frontier::MlRidgeContinuationOptions cont;
  if (alphas.isNotNull()) {
    Rcpp::NumericVector av(alphas.get());
    cont.alphas.clear();
    cont.alphas.reserve(static_cast<std::size_t>(av.size()));
    for (R_xlen_t i = 0; i < av.size(); ++i) cont.alphas.push_back(av[i]);
  }
  cont.target = continuation_target_from_string(target);
  cont.include_endpoint = include_endpoint;
  cont.diagonal_floor = diagonal_floor;

  auto fit_or = magmaan::estimate::frontier::fit_ml_ridge_continuation(
      ctx.pt, ctx.rep, ctx.samp, x0, bounds_from_nullable(bounds), backend,
      optim_opts_from(control), cont);
  if (!fit_or.has_value()) stop_fit(fit_or.error());

  magmaan::estimate::frontier::MlRidgeContinuationResult fit =
      std::move(*fit_or);
  Ctx final_ctx = ctx;
  final_ctx.samp = fit.final_sample_stats;
  Rcpp::List out = fit_result(final_ctx, fit.final, &starts, "ML");
  out["frontier_method"] = "ml_ridge_continuation";
  out["continuation"] = continuation_to_r(fit, cont.target);
  return out;
}

// [[Rcpp::export]]
Rcpp::List frontier_rbm_impl(
    Rcpp::List fit,
    SEXP raw_data = R_NilValue,
    SEXP weight = R_NilValue,
    std::string stage2_weight = "nt",
    double dls_a = 0.5,
    std::string method = "explicit",
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const std::string method_key = rbm_method_key(std::move(method));
  const magmaan::estimate::frontier::RBMOptions opts =
      rbm_options_from(optimizer, control);
  const magmaan::estimate::Bounds b = bounds_from_nullable(bounds);

  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats", "frontier_rbm"));
    const std::string parameterization_name =
        fit.containsElementNamed("parameterization")
            ? Rcpp::as<std::string>(fit["parameterization"])
            : ordinal_parameterization_attr(fit["partable"]);
    const auto parameterization =
        ordinal_parameterization_from_string(parameterization_name);
    const auto ow = ordinal_weight_from_estimator(
        ordinal_weight_for_postfit(fit, estimator), "frontier_rbm");
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = ordinal_fit_result(
        ctx, stats, rbm->estimates, nullptr, "RBM-ORDINAL",
        parameterization_name.c_str());
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, estimator.c_str());
    return out;
  }

  if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "mixed_ordinal_stats", "frontier_rbm"));
    const std::string parameterization_name =
        fit.containsElementNamed("parameterization")
            ? Rcpp::as<std::string>(fit["parameterization"])
            : ordinal_parameterization_attr(fit["partable"]);
    const auto parameterization =
        ordinal_parameterization_from_string(parameterization_name);
    const auto ow = ordinal_weight_from_estimator(
        ordinal_weight_for_postfit(fit, estimator), "frontier_rbm");
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_mixed_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_mixed_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr,
                                "RBM-MIXED-ORDINAL");
    out["mixed_ordinal"] = true;
    out["parameterization"] = parameterization_name;
    out["mixed_ordinal_stats"] = mixed_ordinal_stats_to_r(stats);
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, estimator.c_str());
    return out;
  }

  if (fit.containsElementNamed("stage1")) {
    SEXP rd = raw_data;
    if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) {
      rd = fit["raw_data"];
    }
    if (Rf_isNull(rd)) {
      Rcpp::stop("magmaan: frontier_rbm() needs raw_data for ML2S fits");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
    std::unique_ptr<FimlH1> owned_h1;
    const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
    const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
    magmaan::estimate::fiml::TwoStageDlsOptions dls;
    dls.a = dls_a;
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_two_stage(
                  ctx.pt, ctx.rep, raw, pack, h1, est, kind, dls, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_two_stage(
                  ctx.pt, ctx.rep, raw, pack, h1, est, kind, dls, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr, "RBM-ML2S");
    out["stage1"] = fit["stage1"];
    out["stage2_weight"] = stage2_weight;
    out["stage2_dls_a"] = dls_a;
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, "ML2S");
    return out;
  }

  const bool is_fiml = fit.containsElementNamed("fiml") &&
                       Rcpp::as<bool>(fit["fiml"]);
  if (is_fiml) {
    SEXP rd = raw_data;
    if (Rf_isNull(rd)) {
      if (!fit.containsElementNamed("raw_data")) {
        Rcpp::stop("magmaan: frontier_rbm() needs raw_data or a FIML fit with $raw_data");
      }
      rd = fit["raw_data"];
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
    auto pack = magmaan::estimate::fiml::fiml_pack(raw);
    if (!pack.has_value()) stop_fit(pack.error());

    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_fiml(
                  ctx.pt, ctx.rep, raw, *pack, est, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_fiml(
                  ctx.pt, ctx.rep, raw, *pack, est, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());

    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr, "RBM-FIML");
    out["fiml"] = true;
    out["raw_data"] =
        fiml_raw_to_r(raw, ctx.rep.ov_names, ctx.names.group_labels);
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, "FIML");
    return out;
  }

  if (estimator == "ULS" || estimator == "GLS" || estimator == "WLS") {
    if (Rf_isNull(raw_data)) {
      Rcpp::stop("magmaan: frontier_rbm() needs raw_data for continuous LS fits");
    }
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
    auto w = continuous_ls_weight(ctx, est, estimator, weight, "RBM");
    const auto mode = continuous_ij_mode(estimator);
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_continuous_ls(
                  ctx.pt, ctx.rep, ctx.samp, est, w, raw, mode, {}, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_continuous_ls(
                  ctx.pt, ctx.rep, ctx.samp, est, w, raw, mode, {}, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr,
                                ("RBM-" + estimator).c_str());
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, estimator.c_str());
    return out;
  }

  if (Rf_isNull(raw_data)) {
    Rcpp::stop("magmaan: frontier_rbm() needs raw_data for complete-data ML fits");
  }
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
      method_key == "explicit"
          ? magmaan::estimate::frontier::rbm_explicit_ml(
                ctx.pt, ctx.rep, ctx.samp, raw, est, b, opts)
          : magmaan::estimate::frontier::rbm_implicit_ml(
                ctx.pt, ctx.rep, ctx.samp, raw, est, b, opts);
  if (!rbm.has_value()) stop_fit(rbm.error());

  Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr, "RBM-ML");
  out["rbm"] = rbm_metadata_to_r(*rbm, method_key, "ML");
  return out;
}

// fcsem_model_spec() — build the native, non-folded FC-SEM partable directly
// from syntax. This is the R frontier analogue of api::frontier::model_spec().
//
// [[Rcpp::export]]
Rcpp::List fcsem_model_spec_impl(std::string syntax) {
  FcSemCtx ctx = fcsem_model_from_syntax(syntax);
  return Rcpp::List::create(
      Rcpp::_["syntax"]   = syntax,
      Rcpp::_["partable"] = fcsem_partable_df(ctx.pt, ctx.names, ctx.starts),
      Rcpp::_["ov_names"] = Rcpp::wrap(ctx.ov_names));
}

// fit_ml_fcsem() — native FC-SEM ML, using covariance-only sample statistics.
// Starts come from simple_fcsem_start_values(); optimization currently uses
// the same optimizer control list as the ordinary R ML bridge.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_fcsem_impl(std::string syntax, Rcpp::List sample_stats,
                             Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
  FcSemCtx ctx = fcsem_ctx_from_syntax_sample_stats(syntax, sample_stats);
  auto x0_or = magmaan::estimate::simple_fcsem_start_values(ctx.pt, ctx.samp);
  if (!x0_or.has_value()) stop_fit(x0_or.error());
  auto e_or = magmaan::estimate::fit_ml_fcsem(
      ctx.pt, ctx.samp, *x0_or, {}, magmaan::estimate::Backend::NloptLbfgs,
      optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fcsem_fit_result(ctx, est, syntax);
}

// [[Rcpp::export]]
Rcpp::List fcsem_standard_errors_impl(Rcpp::List fit) {
  FcSemCtx ctx = fcsem_ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto info_or =
      magmaan::inference::information_expected_fcsem(ctx.pt, ctx.samp, est);
  if (!info_or.has_value()) stop_post(info_or.error());
  auto vcov_or = magmaan::inference::vcov(*info_or, ctx.pt, est.theta);
  if (!vcov_or.has_value()) stop_post(vcov_or.error());
  return Rcpp::List::create(
      Rcpp::_["information"] = Rcpp::wrap(*info_or),
      Rcpp::_["vcov"]        = Rcpp::wrap(*vcov_or),
      Rcpp::_["se"]          = Rcpp::wrap(magmaan::inference::se(*vcov_or)));
}

// [[Rcpp::export]]
Rcpp::List fcsem_fit_measures_impl(Rcpp::List fit) {
  FcSemCtx ctx = fcsem_ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const double chi2 = magmaan::inference::chi2_stat(ctx.samp, est);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  const magmaan::measures::BaselineFit bl =
      magmaan::measures::baseline_chi2(ctx.samp);
  const magmaan::measures::FitMeasures fm =
      magmaan::measures::fit_measures(chi2, *df_or, bl, ctx.samp);
  auto fx_or = magmaan::measures::fit_extras_fcsem(ctx.pt, ctx.samp, est);
  if (!fx_or.has_value()) stop_post(fx_or.error());
  return Rcpp::List::create(
      Rcpp::_["chisq"]                  = chi2,
      Rcpp::_["df"]                     = *df_or,
      Rcpp::_["baseline.chisq"]         = bl.chi2,
      Rcpp::_["baseline.df"]            = bl.df,
      Rcpp::_["cfi"]                    = fm.cfi,
      Rcpp::_["tli"]                    = fm.tli,
      Rcpp::_["rmsea"]                  = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"]         = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"]         = fm.rmsea_ci_upper,
      Rcpp::_["rmsea.pvalue"]           = fm.rmsea_pvalue,
      Rcpp::_["rmsea.close.h0"]         = fm.rmsea_close_h0,
      Rcpp::_["rmsea.notclose.pvalue"]  = fm.rmsea_notclose_pvalue,
      Rcpp::_["rmsea.notclose.h0"]      = fm.rmsea_notclose_h0,
      Rcpp::_["srmr"]                   = fx_or->srmr,
      Rcpp::_["logl"]                   = fx_or->logl,
      Rcpp::_["unrestricted.logl"]      = fx_or->unrestricted_logl,
      Rcpp::_["aic"]                    = fx_or->aic,
      Rcpp::_["bic"]                    = fx_or->bic,
      Rcpp::_["bic2"]                   = fx_or->bic2,
      Rcpp::_["npar"]                   = fx_or->npar,
      Rcpp::_["ntotal"]                 = static_cast<double>(fx_or->ntotal));
}

// [[Rcpp::export]]
Rcpp::DataFrame fcsem_standardized_rows_impl(Rcpp::List fit,
                                             Rcpp::NumericMatrix vcov) {
  FcSemCtx ctx = fcsem_ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto rows_or = magmaan::measures::standardize::standardized_rows_fcsem(
      ctx.pt, ctx.names, ctx.samp, est, vcov_m);
  if (!rows_or.has_value()) stop_post(rows_or.error());
  return fcsem_standardized_rows_df(*rows_or);
}

// fit_fiml() — mirrors estimate::fit_fiml(pt, rep, raw, FIML{}).
// `raw_data` is a magmaan_fiml_data object from df_to_fiml_data(), or a list
// with $X and optional $mask. Missing values are retained in $X and represented
// by $mask; columns are reordered to the model's observed-variable order.
//
// [[Rcpp::export]]
Rcpp::List fit_fiml_impl(SEXP partable, SEXP raw_data,
                         Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                         Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
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
  if (auto e = magmaan::estimate::fiml::validate_fiml_fixed_x_missing_policy(
          ctx.pt, raw); !e.has_value()) {
    stop_fit(e.error());
  }
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  ctx.samp = pack_or->start_stats;
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_fiml(
      ctx.pt, ctx.rep, raw, x0, *pack_or, backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fiml_fit_result(ctx, raw, est, &starts);
  auto h1_or = magmaan::estimate::fiml::fiml_h1_moments(
      raw, *pack_or, fiml_h1_opts_from(control));
  if (!h1_or.has_value()) stop_fit(h1_or.error());
  out["fiml_h1"] = fiml_h1_xptr(std::move(*h1_or));
  out["fiml_pack"] = fiml_pack_xptr(std::move(*pack_or));
  return out;
}

// saturated_em_moments_impl() — Stage-1 of the Savalei-Bentler (2009) two-stage
// missing-data path. Takes raw data only (no `partable`/spec needed because the
// saturated model has no structural restrictions) and returns the per-block EM
// mean and covariance plus the block-diagonal saturated information `H`,
// score-covariance `J`, and sandwich `ACOV = H^{-1} J H^{-1}`. See the C++
// `SaturatedMoments` doc comment for the η = (μ, vech(Σ)) layout convention.
//
// [[Rcpp::export]]
Rcpp::List saturated_em_moments_impl(SEXP raw_data, double h_step = 1e-4) {
  SEXP X_arg = raw_data;
  SEXP mask_arg = R_NilValue;
  if (TYPEOF(raw_data) == VECSXP) {
    Rcpp::List rd(raw_data);
    if (rd.containsElementNamed("X")) {
      X_arg = rd["X"];
      if (rd.containsElementNamed("mask")) mask_arg = rd["mask"];
    }
  }

  const std::size_t n_blocks = TYPEOF(X_arg) == VECSXP
      ? static_cast<std::size_t>(Rcpp::List(X_arg).size())
      : 1u;
  if (n_blocks == 0) Rcpp::stop("magmaan: saturated_em_moments needs at least one data block");

  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  bool any_missing = false;
  std::vector<Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>> masks;
  masks.reserve(n_blocks);

  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "data$X");
    const int n = Xb.nrow();
    const int p = Xb.ncol();
    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);

    Rcpp::LogicalMatrix Mb;
    const bool has_mask = !Rf_isNull(mask_arg);
    if (has_mask) {
      Mb = block_mask_matrix(mask_arg, b, n_blocks, "data$mask");
      if (Mb.nrow() != n || Mb.ncol() != p)
        Rcpp::stop("magmaan: data$mask block %d has shape %dx%d but data$X has %dx%d",
                   static_cast<int>(b + 1), Mb.nrow(), Mb.ncol(), n, p);
    }

    for (int r = 0; r < n; ++r) {
      for (int k = 0; k < p; ++k) {
        const double x = Xb(r, k);
        const bool observed = has_mask
            ? (Mb(r, k) != NA_LOGICAL && Mb(r, k) != 0)
            : std::isfinite(x);
        if (observed && !std::isfinite(x)) {
          Rcpp::stop("magmaan: data$mask marks a non-finite value as observed "
                     "in block %d, row %d", static_cast<int>(b + 1), r + 1);
        }
        M(r, k) = static_cast<std::uint8_t>(observed ? 1 : 0);
        X(r, k) = observed ? x : std::numeric_limits<double>::quiet_NaN();
        if (!observed) any_missing = true;
      }
    }
    raw.X.push_back(std::move(X));
    masks.push_back(std::move(M));
  }
  if (any_missing) raw.mask = std::move(masks);

  auto out_or = magmaan::estimate::fiml::saturated_em_moments(raw, h_step);
  if (!out_or.has_value()) stop_post(out_or.error());
  const auto& out = *out_or;

  const R_xlen_t nb = static_cast<R_xlen_t>(out.mean.size());
  Rcpp::List mean_out(nb), cov_out(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    mean_out[b] = Rcpp::wrap(out.mean[bi]);
    cov_out[b]  = Rcpp::wrap(out.cov[bi]);
    nobs[b]     = static_cast<int>(out.n_obs[bi]);
  }

  return Rcpp::List::create(
      Rcpp::Named("mean") = mean_out,
      Rcpp::Named("cov")  = cov_out,
      Rcpp::Named("n_obs") = nobs,
      Rcpp::Named("warnings") = Rcpp::wrap(out.warnings),
      Rcpp::Named("H")    = Rcpp::wrap(out.H),
      Rcpp::Named("J")    = Rcpp::wrap(out.J),
      Rcpp::Named("acov") = Rcpp::wrap(out.acov));
}

// fit_uls() — composes fit_gmm(pt, rep, samp, x0, {}, bounds, backend).
// `optimizer` selects the backend (default "nlopt-lbfgs"); "ceres" / "ceres-bfgs"
// dispatch to the Ceres LS path, "port-nls" to PORT NL2SOL, etc.
//
// [[Rcpp::export]]
Rcpp::List fit_uls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, bounds_from_nullable(bounds), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ULS");
}

// fit_gls_pairwise() — composes fit_gls_pairwise(pt, rep, raw, pw, x0,
// bounds, backend). Pairwise covariance from raw incomplete data with the
// Γ_NT^pw weight (asymptotically efficient under MAR; breaks the trace
// identity); see fit.hpp for the design context. `X` is a numeric matrix
// (single group) or a list of per-group matrices; `mask` is an optional
// logical matrix / list of logical matrices.
//
// [[Rcpp::export]]
Rcpp::List fit_gls_pairwise_impl(SEXP partable, SEXP X,
                                 SEXP mask = R_NilValue,
                                 Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_pairwise");
  magmaan::spec::Starts starts = std::move(parsed.starts);

  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  if (ctx.rep.ov_names.empty() || ctx.rep.ov_names[0].empty())
    Rcpp::stop("magmaan: model has no observed variables");

  // Reorder raw columns to the model's observed-variable order using the
  // existing FIML helper, then re-pack into a plain RawData via the shared
  // raw_from_data_args path (handles mask layout and NA detection uniformly).
  magmaan::data::RawData raw_reordered = fiml_raw_from_arg(ctx.rep, X);
  // Pack the reordered data back into the X/mask format expected by
  // raw_from_data_args — easier than threading both reorder and the
  // multi-block parsing through a single helper.
  const std::size_t nb = raw_reordered.X.size();
  Rcpp::List X_list(static_cast<R_xlen_t>(nb));
  Rcpp::List mask_list(static_cast<R_xlen_t>(nb));
  for (std::size_t b = 0; b < nb; ++b) {
    X_list[static_cast<R_xlen_t>(b)] = Rcpp::wrap(raw_reordered.X[b]);
    if (b < raw_reordered.mask.size()) {
      const auto& Mb = raw_reordered.mask[b];
      Rcpp::LogicalMatrix Mr(Mb.rows(), Mb.cols());
      for (Eigen::Index i = 0; i < Mb.rows(); ++i)
        for (Eigen::Index j = 0; j < Mb.cols(); ++j)
          Mr(i, j) = Mb(i, j) != 0;
      mask_list[static_cast<R_xlen_t>(b)] = Mr;
    }
  }
  SEXP X_arg = X_list;
  SEXP mask_arg = raw_reordered.mask.empty() ? R_NilValue : SEXP(mask_list);
  magmaan::data::RawData raw = raw_from_data_args(X_arg, mask_arg);

  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  if (!pw_or.has_value()) stop_post(pw_or.error());

  // Layout / sample-stats packing for the standard ctx_from_* path.
  ctx.samp.S = pw_or->S;
  ctx.samp.mean = pw_or->mean;
  ctx.samp.n_obs = pw_or->n_obs;
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gls_pairwise(
      ctx.pt, ctx.rep, raw, *pw_or, x0, bounds_from_nullable(bounds),
      backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "GLSpw");
}

// fit_gls() — composes fit_gls(pt, rep, samp, x0, bounds, backend).
//
// [[Rcpp::export]]
Rcpp::List fit_gls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "GLS");
}

// fit_wls() — composes fit_gmm(pt, rep, samp, x0, W, bounds, backend).
//
// [[Rcpp::export]]
Rcpp::List fit_wls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                        Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls), bounds_from_nullable(bounds),
      backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "WLS");
}

// evaluate_at() — no-optimizer companion to fit_uls/gls/wls/ml. Runs the
// terminal audit at an externally-supplied θ (e.g. lavaan's `theta_hat`),
// reusing the same objective builders the fit composers use so the audit
// asks the same KKT-projected-gradient question as it would at the end of
// a normal fit. The return shape is identical to the fit_* impls'
// (`fit_result` is the same helper), with `audit`/`diagnostics` carrying
// the verdict and `iterations = 0`, `f_evals = g_evals = 1` reflecting
// that no optimizer was invoked. `audit_options` lets a caller flip
// stationarity_mode / absolute_tol from the lavaan-compatible defaults.
namespace {
inline magmaan::optim::TerminalAuditOptions
audit_opts_from(Rcpp::Nullable<Rcpp::List> audit_options) {
  magmaan::optim::TerminalAuditOptions o;  // struct defaults: Absolute, 1e-3
  if (audit_options.isNotNull()) {
    Rcpp::List l(audit_options.get());
    if (l.containsElementNamed("stationarity_mode")) {
      const std::string m = Rcpp::as<std::string>(l["stationarity_mode"]);
      if      (m == "absolute") o.stationarity_mode =
          magmaan::optim::TerminalAuditOptions::StationarityMode::Absolute;
      else if (m == "relative") o.stationarity_mode =
          magmaan::optim::TerminalAuditOptions::StationarityMode::Relative;
      else Rcpp::stop("magmaan: audit_options$stationarity_mode must be "
                      "\"absolute\" or \"relative\" (got \"%s\")", m);
    }
    if (l.containsElementNamed("absolute_tol"))
      o.absolute_tol = Rcpp::as<double>(l["absolute_tol"]);
    if (l.containsElementNamed("stationarity_tol"))
      o.stationarity_tol = Rcpp::as<double>(l["stationarity_tol"]);
    if (l.containsElementNamed("active_bound_tol"))
      o.active_bound_tol = Rcpp::as<double>(l["active_bound_tol"]);
    if (l.containsElementNamed("f_consistency_rel"))
      o.f_consistency_rel = Rcpp::as<double>(l["f_consistency_rel"]);
  }
  return o;
}
}  // namespace

// [[Rcpp::export]]
Rcpp::List evaluate_at_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::NumericVector theta, std::string estimator,
    Rcpp::Nullable<Rcpp::RObject> W = R_NilValue,
    Rcpp::Nullable<Rcpp::List>    bounds = R_NilValue,
    Rcpp::Nullable<Rcpp::List>    audit_options = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "evaluate_at");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);

  magmaan::estimate::Estimator est_enum;
  if      (estimator == "ULS") est_enum = magmaan::estimate::Estimator::ULS;
  else if (estimator == "GLS") est_enum = magmaan::estimate::Estimator::GLS;
  else if (estimator == "WLS") est_enum = magmaan::estimate::Estimator::WLS;
  else if (estimator == "ML")  est_enum = magmaan::estimate::Estimator::ML;
  else Rcpp::stop("magmaan: evaluate_at: estimator must be one of "
                  "\"ULS\", \"GLS\", \"WLS\", \"ML\" (got \"%s\")", estimator);

  magmaan::estimate::gmm::Weight wls;
  if (est_enum == magmaan::estimate::Estimator::WLS) {
    if (W.isNull())
      Rcpp::stop("magmaan: evaluate_at: estimator = \"WLS\" requires a "
                 "non-NULL W weight matrix (or list of matrices for "
                 "multi-group)");
    wls = wls_from_arg(Rcpp::RObject(W.get()), ctx.samp.S.size());
  }

  const Eigen::VectorXd theta_vec = Rcpp::as<Eigen::VectorXd>(theta);
  auto e_or = magmaan::estimate::evaluate_at(
      ctx.pt, ctx.rep, ctx.samp, theta_vec, est_enum, wls,
      bounds_from_nullable(bounds), audit_opts_from(audit_options));
  if (!e_or.has_value()) stop_fit(e_or.error());
  return fit_result(ctx, *e_or, &starts, estimator.c_str());
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_from_raw_impl(SEXP X, bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::ordinal_stats_from_integer_data(blocks,
                                                               full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return ordinal_stats_to_r(*out_or);
}

magmaan::data::OrdinalPairwiseGammaKind ordinal_pairwise_gamma_from_string(
    const std::string& gamma) {
  if (gamma == "overlap") return magmaan::data::OrdinalPairwiseGammaKind::Overlap;
  if (gamma == "nominal") return magmaan::data::OrdinalPairwiseGammaKind::Nominal;
  Rcpp::stop("magmaan: pd_gamma must be \"overlap\" or \"nominal\" (got \"%s\")",
             gamma);
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_observed_from_raw_impl(
    SEXP X, std::string pd_gamma = "overlap", bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::ordinal_stats_from_observed_integer_data(
      blocks, ordinal_pairwise_gamma_from_string(pd_gamma), full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List ordinal_stage2_weight_blocks_impl(Rcpp::List ordinal_stats,
                                             std::string stage2_weight = "dwls",
                                             double dls_a = 0.5) {
  auto stats = ordinal_stats_from_arg(ordinal_stats);
  magmaan::estimate::frontier::OrdinalStage2DlsOptions dls;
  dls.a = dls_a;
  auto W_or = magmaan::estimate::frontier::ordinal_stage2_weight_blocks(
      stats, ordinal_stage2_weight_from_string(stage2_weight), dls);
  if (!W_or.has_value()) stop_post(W_or.error());
  Rcpp::List out(static_cast<R_xlen_t>(W_or->size()));
  for (R_xlen_t b = 0; b < out.size(); ++b) {
    out[b] = Rcpp::wrap((*W_or)[static_cast<std::size_t>(b)]);
  }
  out.attr("stage2_weight") = ordinal_stage2_label(stage2_weight);
  out.attr("dls_a") = dls_a;
  return out;
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
Rcpp::List data_ordinal_stats_huber_residual_from_raw_impl(
    SEXP X, std::string clip = "hard_huber", double k = 1.345) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      blocks, ordinal_huber_residual_options_from(clip, k));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "huber_residual";
  out["clip"] = clip;
  out["k"] = k;
  out["diagnostics"] = pairwise_ordinal_diagnostics_to_r(out_or->block_diagnostics);
  return out;
}

std::vector<std::vector<std::int32_t>>
mixed_ordered_mask_from_arg(SEXP ordered_mask, std::size_t n_blocks);

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_from_raw_impl(SEXP X, SEXP ordered_mask,
                                                  bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto ordered = mixed_ordered_mask_from_arg(ordered_mask, blocks.size());
  auto out_or = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered,
                                                            full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

std::vector<std::vector<std::int32_t>>
mixed_ordered_mask_from_arg(SEXP ordered_mask, std::size_t n_blocks) {
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(n_blocks);
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (n_blocks != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != n_blocks)
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (n_blocks != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  return ordered;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_observed_from_raw_impl(
    SEXP X, SEXP ordered_mask, bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto ordered = mixed_ordered_mask_from_arg(ordered_mask, blocks.size());
  auto out_or = magmaan::data::mixed_ordinal_stats_from_observed_data(
      blocks, ordered, full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_hybrid_fiml_from_raw_impl(
    SEXP X, SEXP ordered_mask, bool full_wls_weight = true,
    double h_step = 1e-4) {
  auto blocks = matrix_blocks_from_arg(X);
  auto ordered = mixed_ordered_mask_from_arg(ordered_mask, blocks.size());
  auto out_or =
      magmaan::estimate::fiml::mixed_ordinal_stats_hybrid_fiml_from_observed_data(
          blocks, ordered, full_wls_weight, h_step);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_shrink_mixed_ordinal_stats_impl(
    Rcpp::List mixed_stats, std::string kind = "diagonal",
    double intensity = 0.0, bool estimate_intensity = false) {
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  magmaan::data::frontier::CovarianceShrinkageOptions opts;
  opts.kind = shrinkage_kind_from_string(kind);
  opts.intensity = intensity;
  opts.estimate_intensity = estimate_intensity;
  auto out_or = magmaan::data::frontier::shrink_mixed_ordinal_stats(stats, opts);
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
                                 Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_dwls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  // Re-attach the group.equal families stamped by lavaan_lavaanify so the
  // ordinal prep applies the fit-time Wu-Estabrook release (lost on round-trip).
  ctx.pt.group_equal = group_equal_attr(partable);
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
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "DWLS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_uls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  // Re-attach the group.equal families stamped by lavaan_lavaanify so the
  // ordinal prep applies the fit-time Wu-Estabrook release (lost on round-trip).
  ctx.pt.group_equal = group_equal_attr(partable);
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
      magmaan::estimate::OrdinalWeightKind::ULS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "ULS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_wls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  // Re-attach the group.equal families stamped by lavaan_lavaanify so the
  // ordinal prep applies the fit-time Wu-Estabrook release (lost on round-trip).
  ctx.pt.group_equal = group_equal_attr(partable);
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
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "WLS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_ordinal_stage2_impl(SEXP partable, Rcpp::List ordinal_stats,
                                   std::string stage2_weight = "dwls",
                                   double dls_a = 0.5,
                                   Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                   Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                   Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ordinal_stage2");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  magmaan::estimate::frontier::OrdinalStage2DlsOptions dls;
  dls.a = dls_a;
  auto weighted_or = magmaan::estimate::frontier::ordinal_stats_with_stage2_weight(
      stats, ordinal_stage2_weight_from_string(stage2_weight), dls);
  if (!weighted_or.has_value()) stop_post(weighted_or.error());
  magmaan::data::OrdinalStats weighted = std::move(*weighted_or);

  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(
      ctx.pt, weighted, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = weighted.R;
  ctx.samp.n_obs = weighted.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{}
                                          : ctx.rep.ov_names[0];
  ctx.meanstructure = false;

  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, weighted, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, weighted, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);

  const std::string label = ordinal_stage2_label(stage2_weight);
  Rcpp::List out = ordinal_fit_result(ctx, weighted, est, &starts,
                                      label.c_str(),
                                      parameterization_name.c_str());
  out["stage2_weight"] = label;
  out["stage2_dls_a"] = dls_a;
  out["ordinal_computational_weight"] = "WLS";
  return out;
}

// [[Rcpp::export]]
Rcpp::List frontier_pairwise_ordinal_composite_nested_impl(
    SEXP partable_H1,
    SEXP partable_H0,
    SEXP X,
    Rcpp::List n_levels,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double fd_step = 1e-5) {
  auto blocks = matrix_blocks_from_arg(X);
  auto levels = n_levels_from_arg(n_levels);
  auto data_or =
      magmaan::estimate::frontier::pairwise_ordinal_observed_data(blocks,
                                                                  levels);
  if (!data_or.has_value()) stop_post(data_or.error());

  magmaan::spec::Starts starts1;
  magmaan::spec::Starts starts0;
  Ctx ctx1 = pairwise_ordinal_ctx_from_partable(
      partable_H1, "frontier_pairwise_ordinal_composite_nested",
      data_or->stats, starts1);
  Ctx ctx0 = pairwise_ordinal_ctx_from_partable(
      partable_H0, "frontier_pairwise_ordinal_composite_nested",
      data_or->stats, starts0);

  const magmaan::estimate::Backend backend =
      backend_from_optimizer_arg(optimizer);
  const magmaan::optim::OptimOptions opts = optim_opts_from(control);

  auto fit1_or = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      ctx1.pt, ctx1.rep, *data_or, {}, {}, backend, opts, starts1);
  if (!fit1_or.has_value()) stop_fit(fit1_or.error());
  auto fit0_or = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      ctx0.pt, ctx0.rep, *data_or, {}, {}, backend, opts, starts0);
  if (!fit0_or.has_value()) stop_fit(fit0_or.error());

  auto god1_or = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      ctx1.pt, ctx1.rep, *data_or, fit1_or->estimates, fd_step);
  if (!god1_or.has_value()) stop_post(god1_or.error());
  auto god0_or = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      ctx0.pt, ctx0.rep, *data_or, fit0_or->estimates, fd_step);
  if (!god0_or.has_value()) stop_post(god0_or.error());
  auto lr_or = magmaan::estimate::frontier::lr_test_pairwise_ordinal_composite(
      ctx1.pt, ctx1.rep, *data_or, *fit1_or, ctx0.pt, ctx0.rep, *fit0_or,
      magmaan::robust::SatorraAMethod::Exact, fd_step);
  if (!lr_or.has_value()) stop_post(lr_or.error());

  Rcpp::List h1 = ordinal_fit_result(ctx1, data_or->stats, fit1_or->estimates,
                                     &starts1, "PAIRWISE-ORDINAL-COMPOSITE",
                                     "delta");
  h1["pairwise_composite"] = true;
  h1["objective"] = pairwise_objective_to_r(fit1_or->objective);
  h1["godambe"] = pairwise_godambe_to_r(*god1_or);

  Rcpp::List h0 = ordinal_fit_result(ctx0, data_or->stats, fit0_or->estimates,
                                     &starts0, "PAIRWISE-ORDINAL-COMPOSITE",
                                     "delta");
  h0["pairwise_composite"] = true;
  h0["objective"] = pairwise_objective_to_r(fit0_or->objective);
  h0["godambe"] = pairwise_godambe_to_r(*god0_or);

  return Rcpp::List::create(
      Rcpp::_["h1"] = h1,
      Rcpp::_["h0"] = h0,
      Rcpp::_["lr"] = pairwise_lr_to_r(*lr_or),
      Rcpp::_["stats"] = ordinal_stats_to_r(data_or->stats),
      Rcpp::_["saturated"] = pairwise_objective_to_r(data_or->saturated));
}

// [[Rcpp::export]]
Rcpp::List fit_dwls_mixed_ordinal_impl(SEXP partable, Rcpp::List mixed_stats,
                                       Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                       Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                       Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
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
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fit_result(ctx, est, &starts, "DWLS");
  out["mixed_ordinal"] = true;
  out["parameterization"] = parameterization_name;
  out["mixed_ordinal_stats"] = mixed_ordinal_stats_to_r(stats);
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_wls_mixed_ordinal_impl(SEXP partable, Rcpp::List mixed_stats,
                                      Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                      Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                      Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
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
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fit_result(ctx, est, &starts, "WLS");
  out["mixed_ordinal"] = true;
  out["parameterization"] = parameterization_name;
  out["mixed_ordinal_stats"] = mixed_ordinal_stats_to_r(stats);
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_uls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "ULS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_gls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_snlls_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "GLS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_wls_snlls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                              Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  (void)bounds;  // SNLLS optimizes the unbounded nonlinear (Λ, Β) block
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "WLS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// The flat per-Backend shim explosion (fit_uls_ceres_impl, fit_uls_snlls_ceres_impl,
// fit_gls_ceres_impl, fit_gls_snlls_ceres_impl, fit_gls_snlls_ceres_bfgs_impl,
// fit_wls_ceres_impl, fit_wls_snlls_ceres_impl) lived here through Phase 3 and
// was retired in Phase 4. Callers now pass `optimizer = "ceres"` / `"ceres-bfgs"`
// to the unified per-family entries above; the C++ Backend enum is the
// single dispatch surface.

// fit_start_values() — returns a theta-ordered start vector (length npar).
// `sample_stats` is as in fit_fit(); values are used verbatim. The default
// remains "simple" for backward compatibility with the old helper.
//
// [[Rcpp::export]]
Rcpp::NumericVector fit_start_values(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> start = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_start_values");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const std::string start_name =
      start_name_from_arg(start, "fit_start_values", "simple");
  return Rcpp::wrap(start_values_or_stop(ctx, starts, start_name));
}

// estimate_structured_gamma() — explicit MI4 / structured-ADF Gamma builder.
// Returns Gamma itself so paper-local R code can inspect, regularize, or invert
// it before passing a weight to the existing WLS path.
//
// [[Rcpp::export]]
SEXP estimate_structured_gamma(Rcpp::List fit, SEXP raw_data) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);

  auto ev_or = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev_or.has_value()) stop_model(ev_or.error());
  auto G_or = magmaan::estimate::frontier::structured_gamma_matrix(
      *ev_or, ctx.rep, ctx.samp, raw, est.theta);
  if (!G_or.has_value()) stop_fit(G_or.error());
  return weight_to_r(*G_or);
}

// estimate_structured_gamma_weight() — explicit MI4 / structured-ADF working
// weight builder. Returns only W, which is passed to the existing WLS path.
//
// [[Rcpp::export]]
SEXP estimate_structured_gamma_weight(Rcpp::List fit, SEXP raw_data) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);

  auto ev_or = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev_or.has_value()) stop_model(ev_or.error());
  auto W_or = magmaan::estimate::frontier::structured_gamma_weight(
      *ev_or, ctx.rep, ctx.samp, raw, est.theta);
  if (!W_or.has_value()) stop_fit(W_or.error());
  return weight_to_r(*W_or);
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

// infer_information_cross_products() — mirrors information_cross_products(...).
// Parameter-level outer-product-of-scores Σᵢ sᵢsᵢᵀ at θ̂. Needs raw data
// (per-case moment contributions). Mplus calls the SE built from this method
// "MLF".
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_cross_products(Rcpp::List fit,
                                                     SEXP raw_data) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  auto r = magmaan::inference::information_cross_products(
      ctx.pt, ctx.rep, ctx.samp, raw, est);
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
  // Exposed for ordinal/mixed fits too: ctx_from_fit() rebuilds the prepared
  // partable, which lines up with the reduced est/vcov, and the delta-method
  // value/SE are parameterization-agnostic (mirrors api::compute_defined).
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

// infer_chi2_stat() — mirrors chi2_stat(samp, est). Returns 2·N_total·fmin =
// N·F (the GOF χ²). This primitive does not need a fit object: sample_stats
// supplies nobs and fmin is the optimizer's minimised objective ½·F (fit$fmin
// when called after fit_fit()); see docs/design/numerical-conventions.md.
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

// infer_baseline_fit() — partable-aware baseline_chi2(pt, samp). Applies the
// fixed.x exogenous correction: lavaan's independence/baseline model frees the
// exogenous (co)variances, so baseline.df drops by px(px-1)/2 and baseline.chisq
// loses the exo-block fit. A no-op when there are < 2 exogenous variables, so it
// is safe for every fit; only fixed.x models change (matching lavaan).
//
// [[Rcpp::export]]
Rcpp::List infer_baseline_fit(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::measures::BaselineFit bl =
      magmaan::measures::baseline_chi2(ctx.pt, ctx.samp);
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

// ordinal_catml_dwls_rmsea_impl() — lavaan-compatible categorical robust RMSEA
// ingredients for an all-ordinal DWLS/WLSMV fit. This evaluates the CATML
// normal-theory correlation discrepancy at the existing ordinal estimates; it
// does not re-optimize.
//
// [[Rcpp::export]]
Rcpp::List ordinal_catml_dwls_rmsea_impl(Rcpp::List fit,
                                         SEXP ordinal_stats = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Rcpp::List stats_r = stats_from_fit_or_arg(
      fit, ordinal_stats, "ordinal_stats", "ordinal_catml_dwls_rmsea_impl");
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(stats_r);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto out_or = magmaan::estimate::catml_dwls_rmsea_ordinal(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name));
  if (!out_or.has_value()) stop_post(out_or.error());
  const auto& out = *out_or;
  return Rcpp::List::create(
      Rcpp::_["XX3"] = out.xx3,
      Rcpp::_["df3"] = out.df3,
      Rcpp::_["c.hat3"] = out.c_hat3,
      Rcpp::_["XX3.scaled"] = out.xx3_scaled,
      Rcpp::_["rmsea.robust"] = out.rmsea_robust);
}

// infer_fiml_observed_vcov() inverts the analytic observed FIML information
// for a fit carrying its retained raw-data/pack state.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_observed_vcov(Rcpp::List fit) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: infer_fiml_observed_vcov() requires a FIML fit with $raw_data");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  auto info_or = magmaan::estimate::fiml::fiml_observed_information(
      ctx.pt, ctx.rep, raw, est, pack);
  if (!info_or.has_value()) stop_post(info_or.error());
  auto vcov_or = magmaan::inference::vcov(*info_or, ctx.pt, est.theta);
  if (!vcov_or.has_value()) stop_post(vcov_or.error());
  return Rcpp::List::create(
      Rcpp::_["information"] = Rcpp::wrap(*info_or),
      Rcpp::_["vcov"] = Rcpp::wrap(*vcov_or),
      Rcpp::_["se"] = Rcpp::wrap(magmaan::inference::se(*vcov_or)));
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
  if (!(h_step > 0.0)) {
    Rcpp::stop("magmaan: estimate_fiml_robust_mlr() requires h_step > 0");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto r_or = magmaan::estimate::fiml::fiml_robust_mlr(
      ctx.pt, ctx.rep, raw, est, *df_or, extras_or->chi2, pack, h1);
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

// fiml_fit_measures_impl() — FIML-specific standard and optional robust/scaled
// global fit measures. Standard measures use the FIML model-vs-saturated LRT
// and FIML independence baseline; robust = TRUE adds the corrected XX3/c.hat3
// user and baseline reductions used by lavaan-style robust CFI/RMSEA.
//
// [[Rcpp::export]]
Rcpp::List fiml_fit_measures_impl(Rcpp::List fit, bool robust = false) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: fiml_fit_measures_impl() requires a FIML fit with $raw_data");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);

  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto baseline_or = magmaan::estimate::fiml::fiml_baseline_chi2(
      ctx.pt, raw, pack, h1);
  if (!baseline_or.has_value()) stop_post(baseline_or.error());

  const auto& fx = *extras_or;
  const auto& bl = *baseline_or;
  const magmaan::measures::FitMeasures fm =
      magmaan::measures::fit_measures(
          fx.chi2, *df_or, bl, pack.cache.n_total, pack.cache.block_p.size());

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["chisq"] = fx.chi2,
      Rcpp::_["df"] = *df_or,
      Rcpp::_["pvalue"] = magmaan::inference::chi2_pvalue(fx.chi2, *df_or),
      Rcpp::_["baseline.chisq"] = bl.chi2,
      Rcpp::_["baseline.df"] = bl.df,
      Rcpp::_["cfi"] = fm.cfi,
      Rcpp::_["tli"] = fm.tli,
      Rcpp::_["rmsea"] = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"] = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"] = fm.rmsea_ci_upper,
      Rcpp::_["rmsea.pvalue"] = fm.rmsea_pvalue,
      Rcpp::_["rmsea.close.h0"] = fm.rmsea_close_h0,
      Rcpp::_["rmsea.notclose.pvalue"] = fm.rmsea_notclose_pvalue,
      Rcpp::_["rmsea.notclose.h0"] = fm.rmsea_notclose_h0,
      Rcpp::_["srmr"] = fx.srmr,
      Rcpp::_["logl"] = fx.logl,
      Rcpp::_["unrestricted.logl"] = fx.unrestricted_logl,
      Rcpp::_["aic"] = fx.aic,
      Rcpp::_["bic"] = fx.bic,
      Rcpp::_["bic2"] = fx.bic2,
      Rcpp::_["npar"] = fx.npar,
      Rcpp::_["ntotal"] = static_cast<double>(fx.ntotal));

  if (robust && *df_or > 0) {
    std::unique_ptr<SaturatedMoments> owned_sm;
    const SaturatedMoments& sm =
        fiml_saturated_for_fit(fit, raw, pack, h1, owned_sm);
    auto r_or = magmaan::estimate::fiml::fiml_corrected_fit_measures(
        ctx.pt, ctx.rep, raw, est, *df_or, pack, h1, sm);
    if (!r_or.has_value()) stop_post(r_or.error());
    const auto& r = *r_or;
    const auto& rf = r.indices;
    out["XX3"] = r.xx3;
    out["df3"] = r.df3;
    out["c.hat3"] = r.c_hat3;
    out["XX3.scaled"] = r.xx3_scaled;
    out["baseline.XX3"] = r.xx3_null;
    out["baseline.df3"] = r.df3_null;
    out["baseline.c.hat3"] = r.c_hat3_null;
    out["baseline.XX3.scaled"] = r.xx3_null_scaled;
    out["chisq.scaled"] = rf.chisq_scaled;
    out["df.scaled"] = rf.df_scaled;
    out["pvalue.scaled"] = rf.pvalue_scaled;
    out["chisq.scaling.factor"] = rf.chisq_scaling_factor;
    out["baseline.chisq.scaled"] = rf.baseline_chisq_scaled;
    out["baseline.df.scaled"] = rf.baseline_df_scaled;
    out["baseline.pvalue.scaled"] = rf.baseline_pvalue_scaled;
    out["baseline.chisq.scaling.factor"] =
        rf.baseline_chisq_scaling_factor;
    out["cfi.scaled"] = rf.cfi_scaled;
    out["tli.scaled"] = rf.tli_scaled;
    out["cfi.robust"] = rf.cfi_robust;
    out["tli.robust"] = rf.tli_robust;
    out["rmsea.scaled"] = rf.rmsea_scaled;
    out["rmsea.ci.lower.scaled"] = rf.rmsea_ci_lower_scaled;
    out["rmsea.ci.upper.scaled"] = rf.rmsea_ci_upper_scaled;
    out["rmsea.pvalue.scaled"] = rf.rmsea_pvalue_scaled;
    out["rmsea.notclose.pvalue.scaled"] =
        rf.rmsea_notclose_pvalue_scaled;
    out["rmsea.robust"] = rf.rmsea_robust;
    out["rmsea.ci.lower.robust"] = rf.rmsea_ci_lower_robust;
    out["rmsea.ci.upper.robust"] = rf.rmsea_ci_upper_robust;
    out["rmsea.pvalue.robust"] = rf.rmsea_pvalue_robust;
    out["rmsea.notclose.pvalue.robust"] =
        rf.rmsea_notclose_pvalue_robust;
  }

  return out;
}

// infer_ml2s_casewise_influence_ij_fit() — per-case one-step misspecification-
// robust ("complete-sandwich") parameter influences for a two-stage (ML2S) fit:
// the missing-data member of the estimated-weight case-influence family, the
// casewise dual of estimate_two_stage_em_ml_inference()'s observed-bread vcov.
// Returns the N_total x n_free `influence` matrix (column-Gram = the ML2S IJ
// vcov) and its fixed-weight `naive` counterpart. Reuses the Stage-1 EM
// pack/H1 the fit carries. The NT Stage-2 weight (lavaan robust.two.stage)
// treats the weight as fixed, so its correction is zero (complete == naive);
// the non-NT weights (DWLS/ADF/DLS) carry the live data-dependent-weight term.
// Beyond lavaan/semfindr.
//
// [[Rcpp::export]]
Rcpp::List infer_ml2s_casewise_influence_ij_fit(
    Rcpp::List fit, SEXP raw_data, std::string stage2_weight = "nt",
    double dls_a = 0.5) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, raw_data);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  dls.a = dls_a;
  auto r_or = magmaan::estimate::fiml::two_stage_casewise_influence_ij(
      ctx.pt, ctx.rep, raw, est, pack, h1, kind, dls);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::Named("influence") = Rcpp::wrap(r_or->influence),
      Rcpp::Named("influence_naive") = Rcpp::wrap(r_or->influence_naive),
      Rcpp::Named("n_total") = static_cast<double>(r_or->n_total));
}

// estimate_two_stage_em_ml_inference() — mirrors
// estimate::fiml::two_stage_em_ml_inference(). Takes the Stage-2 ML fit on EM
// moments plus the original raw data used for Stage 1.
//
// [[Rcpp::export]]
Rcpp::List estimate_two_stage_em_ml_inference(Rcpp::List fit, SEXP raw_data,
                                              double h_step = 1e-4,
                                              std::string stage2_weight = "nt",
                                              double dls_a = 0.5) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, raw_data);
  // Reuse the Stage-1 saturated moments the fit already carries; only fall back
  // to a from-scratch EM (and its pack) when no usable $stage1 is present.
  SaturatedMoments sm;
  std::unique_ptr<SaturatedMoments> owned_sm;
  const SaturatedMoments* sm_ptr = &sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    auto sm_or = magmaan::estimate::fiml::saturated_em_moments(raw, h_step);
    if (!sm_or.has_value()) stop_post(sm_or.error());
    owned_sm = std::make_unique<SaturatedMoments>(std::move(*sm_or));
    sm_ptr = owned_sm.get();
  }
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  dls.a = dls_a;
  auto r_or = magmaan::estimate::fiml::two_stage_em_ml_inference(
      ctx.pt, ctx.rep, est, *sm_ptr, kind, dls,
      magmaan::estimate::fiml::TwoStageBread::Expected);
  if (!r_or.has_value()) stop_post(r_or.error());
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
      Rcpp::_["se"] = Rcpp::wrap(r_or->se),
      Rcpp::_["eigvals"] = Rcpp::wrap(r_or->eigvals),
      Rcpp::_["chisq"] = r_or->chisq,
      Rcpp::_["chisq_scaled"] = r_or->chisq_scaled,
      Rcpp::_["scaling_factor"] = r_or->scaling_factor,
      Rcpp::_["trace_ugamma"] = r_or->trace_ugamma,
      Rcpp::_["df"] = r_or->df,
      Rcpp::_["ntotal"] = static_cast<double>(r_or->ntotal));
  out["chisq.scaled"] = r_or->chisq_scaled;
  out["chisq.scaling.factor"] = r_or->scaling_factor;

  if (r_or->df > 0) {
    auto fm_or = magmaan::estimate::fiml::two_stage_fit_measures(
        ctx.pt, ctx.rep, est, *sm_ptr, kind, dls);
    if (!fm_or.has_value()) stop_post(fm_or.error());
    const auto& bl = fm_or->baseline;
    const auto& fm = fm_or->indices;
    out["baseline_chisq"] = bl.chi2;
    out["baseline.chisq"] = bl.chi2;
    out["baseline_df"] = bl.df;
    out["baseline.df"] = bl.df;
    out["baseline_chisq_scaled"] = fm.baseline_chisq_scaled;
    out["baseline.chisq.scaled"] = fm.baseline_chisq_scaled;
    out["baseline_scaling_factor"] = fm.baseline_chisq_scaling_factor;
    out["baseline.chisq.scaling.factor"] = fm.baseline_chisq_scaling_factor;
    out["baseline_pvalue_scaled"] = fm.baseline_pvalue_scaled;
    out["baseline.pvalue.scaled"] = fm.baseline_pvalue_scaled;
    out["cfi_scaled"] = fm.cfi_scaled;
    out["cfi.scaled"] = fm.cfi_scaled;
    out["tli_scaled"] = fm.tli_scaled;
    out["tli.scaled"] = fm.tli_scaled;
    out["cfi_robust"] = fm.cfi_robust;
    out["cfi.robust"] = fm.cfi_robust;
    out["tli_robust"] = fm.tli_robust;
    out["tli.robust"] = fm.tli_robust;
    out["rmsea_scaled"] = fm.rmsea_scaled;
    out["rmsea.scaled"] = fm.rmsea_scaled;
    out["rmsea_ci_lower_scaled"] = fm.rmsea_ci_lower_scaled;
    out["rmsea.ci.lower.scaled"] = fm.rmsea_ci_lower_scaled;
    out["rmsea_ci_upper_scaled"] = fm.rmsea_ci_upper_scaled;
    out["rmsea.ci.upper.scaled"] = fm.rmsea_ci_upper_scaled;
    out["rmsea_pvalue_scaled"] = fm.rmsea_pvalue_scaled;
    out["rmsea.pvalue.scaled"] = fm.rmsea_pvalue_scaled;
    out["rmsea_notclose_pvalue_scaled"] = fm.rmsea_notclose_pvalue_scaled;
    out["rmsea.notclose.pvalue.scaled"] = fm.rmsea_notclose_pvalue_scaled;
    out["rmsea_robust"] = fm.rmsea_robust;
    out["rmsea.robust"] = fm.rmsea_robust;
    out["rmsea_ci_lower_robust"] = fm.rmsea_ci_lower_robust;
    out["rmsea.ci.lower.robust"] = fm.rmsea_ci_lower_robust;
    out["rmsea_ci_upper_robust"] = fm.rmsea_ci_upper_robust;
    out["rmsea.ci.upper.robust"] = fm.rmsea_ci_upper_robust;
    out["rmsea_pvalue_robust"] = fm.rmsea_pvalue_robust;
    out["rmsea.pvalue.robust"] = fm.rmsea_pvalue_robust;
    out["rmsea_notclose_pvalue_robust"] = fm.rmsea_notclose_pvalue_robust;
    out["rmsea.notclose.pvalue.robust"] = fm.rmsea_notclose_pvalue_robust;
  }
  return out;
}

// two_stage_stage2_weight_blocks_impl() — mirrors
// estimate::fiml::two_stage_stage2_weight_blocks(). Builds the per-block Stage-2
// weight (Nt / Dwls / Adf / Dls) from a Stage-1 saturated-moments list, returned
// as a list of per-block matrices suitable for `fit_wls(..., W = .)`.
//
// [[Rcpp::export]]
Rcpp::List two_stage_stage2_weight_blocks_impl(Rcpp::List stage1,
                                               std::string stage2_weight = "nt",
                                               double dls_a = 0.5) {
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_list(stage1, sm)) {
    Rcpp::stop("magmaan: two_stage_stage2_weight_blocks needs a Stage-1 list "
               "with mean/cov/n_obs/acov");
  }
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  dls.a = dls_a;
  auto w_or = magmaan::estimate::fiml::two_stage_stage2_weight_blocks(
      sm, kind, dls);
  if (!w_or.has_value()) stop_post(w_or.error());
  Rcpp::List out(static_cast<R_xlen_t>(w_or->size()));
  for (std::size_t b = 0; b < w_or->size(); ++b) {
    out[static_cast<R_xlen_t>(b)] = Rcpp::wrap((*w_or)[b]);
  }
  return out;
}

// infer_fiml_fmg_spectrum() — mirrors estimate::fiml::fiml_ugamma_spectrum().
// First-principles missing-data UΓ spectrum for FMG goodness-of-fit tests: the
// df nonzero eigenvalues of U·Γ_mis built from the saturated H1 information and
// the saturated-moment ACOV. Biased gamma only (the Du-Bentler unbiased gamma
// is undefined under FIML — no $unbiased key), ML/LRT base statistic.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_fmg_spectrum(Rcpp::List fit, double h_step = 1e-4) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: infer_fiml_fmg_spectrum() requires a FIML fit with $raw_data");
  }
  if (!(h_step > 0.0)) {
    Rcpp::stop("magmaan: infer_fiml_fmg_spectrum() requires h_step > 0");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  std::unique_ptr<SaturatedMoments> owned_sm;
  const SaturatedMoments& sm = fiml_saturated_for_fit(fit, raw, pack, h1, owned_sm);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto sp_or = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      ctx.pt, ctx.rep, raw, est, *df_or, extras_or->chi2, pack, h1, sm);
  if (!sp_or.has_value()) stop_post(sp_or.error());
  return Rcpp::List::create(
      Rcpp::_["biased"] = Rcpp::wrap(sp_or->eigvals),
      Rcpp::_["chi2_lrt"] = sp_or->chi2_lrt,
      Rcpp::_["df"] = sp_or->df,
      Rcpp::_["trace_xcheck"] = sp_or->trace_xcheck);
}

// measures_standardize_lv() — mirrors measures::standardize::standardize_lv().
//
// [[Rcpp::export]]
Rcpp::List measures_standardize_lv(Rcpp::List fit, Rcpp::NumericMatrix vcov) {
  // Standardization is parameterization-agnostic: it divides by the
  // model-implied indicator variance and uses the assembled latent covariance,
  // so it is well-defined for ordinal/mixed-ordinal (delta or theta) fits.
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
  // See measures_standardize_lv(): standardization is well-defined for
  // ordinal/mixed-ordinal fits, so no ordinal guard here. For ordinal/mixed
  // fits under the delta parameterization, the categorical indicators' latent
  // responses are unit-variance, so std.all standardizes their loadings by the
  // latent SD only (no √σ_rr division) — pass that through to the core.
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  const bool is_ordinal =
      (fit.containsElementNamed("ordinal") &&
       Rcpp::as<bool>(fit["ordinal"])) ||
      (fit.containsElementNamed("mixed_ordinal") &&
       Rcpp::as<bool>(fit["mixed_ordinal"]));
  const bool ordinal_delta_unit =
      is_ordinal && fit.containsElementNamed("partable") &&
      ordinal_parameterization_attr(fit["partable"]) == "delta";
  auto r_or = magmaan::measures::standardize::standardize_all(
      ctx.pt, ctx.rep, est, vcov_m, ordinal_delta_unit);
  if (!r_or.has_value()) stop_post(r_or.error());
  return standardized_to_list(*r_or);
}

// measures_composite_weights() — recovered `<~` weights and delta-method SEs.
//
// [[Rcpp::export]]
Rcpp::DataFrame measures_composite_weights(Rcpp::List fit,
                                           Rcpp::NumericMatrix vcov) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto w_or = magmaan::measures::composite::composite_weights(
      ctx.pt, ctx.names, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return composite_weights_df(*w_or);
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
// correlation-metric residuals, residual SE/z-statistics, the SRMR, and the
// per-block $summary table (cor.bentler SRMR family: SRMR/USRMR with SE,
// exact-fit and close-fit z-tests and a close-fit CI).
//
// [[Rcpp::export]]
Rcpp::List measures_standardized_residuals(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r_or = magmaan::measures::standardized_residuals(
      ctx.pt, ctx.rep, ctx.samp, est);
  if (!r_or.has_value()) stop_post(r_or.error());
  return standardized_residuals_to_r(*r_or, ctx.rep.ov_names);
}

// measures_reliability_cov() — covariance-only reliability coefficients and
// optional delta-method SEs from an asymptotic covariance of vech(S).
//
// [[Rcpp::export]]
Rcpp::List measures_reliability_cov(
    Rcpp::NumericMatrix S,
    Rcpp::Nullable<Rcpp::NumericMatrix> gamma = R_NilValue,
    int n = 0) {
  const Eigen::MatrixXd S_m = Rcpp::as<Eigen::MatrixXd>(S);
  return reliability_results_to_r(S_m, gamma, n);
}

// measures_reliability_omega_multidim() — closed-form multidimensional omega
// (omega-total or omega-hierarchical) with an optional full-Gamma delta-method
// SE. `block` gives each item's factor id (any integer labels; remapped to dense
// 0-based codes). `target` is "total" or "hierarchical". `weights` (length p)
// applies to the total only. With `gamma` (asymptotic cov of vech(S)) and n > 0
// the value, SE, avar, and gradient are returned; otherwise value + FD gradient.
//
// [[Rcpp::export]]
Rcpp::List measures_reliability_omega_multidim(
    Rcpp::NumericMatrix S,
    Rcpp::IntegerVector block,
    std::string target = "total",
    Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> gamma = R_NilValue,
    int n = 0) {
  namespace rel = magmaan::measures::frontier::reliability;
  const Eigen::MatrixXd S_m = Rcpp::as<Eigen::MatrixXd>(S);
  const R_xlen_t p = S_m.rows();
  if (static_cast<R_xlen_t>(block.size()) != p) {
    Rcpp::stop("magmaan: block length must equal nrow(S)");
  }

  rel::OmegaTarget tgt;
  if (target == "total") {
    tgt = rel::OmegaTarget::Total;
  } else if (target == "hierarchical") {
    tgt = rel::OmegaTarget::Hierarchical;
  } else {
    Rcpp::stop("magmaan: target must be 'total' or 'hierarchical'");
  }

  // Remap arbitrary integer block labels to dense 0-based codes (sorted unique).
  std::vector<int> labels(block.begin(), block.end());
  std::vector<int> uniq = labels;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  rel::OmegaSpec spec;
  spec.block.resize(p);
  for (R_xlen_t i = 0; i < p; ++i) {
    const auto it = std::lower_bound(uniq.begin(), uniq.end(), labels[i]);
    spec.block(i) = static_cast<int>(it - uniq.begin());
  }
  if (weights.isNotNull()) {
    spec.weights = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(weights.get()));
    if (static_cast<R_xlen_t>(spec.weights.size()) != p) {
      Rcpp::stop("magmaan: weights length must equal nrow(S)");
    }
  }
  const int k = static_cast<int>(uniq.size());

  if (gamma.isNotNull()) {
    if (n <= 0) {
      Rcpp::stop("magmaan: measures_reliability_omega_multidim() needs n > 0 when gamma is supplied");
    }
    const Eigen::MatrixXd G =
        Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(gamma.get()));
    auto r_or = rel::omega_multidim_delta(tgt, S_m, spec, G, n);
    if (!r_or.has_value()) stop_post(r_or.error());
    return Rcpp::List::create(
        Rcpp::_["value"] = r_or->value,
        Rcpp::_["se"] = r_or->se,
        Rcpp::_["avar"] = r_or->avar,
        Rcpp::_["gradient"] = Rcpp::wrap(r_or->gradient),
        Rcpp::_["n"] = n,
        Rcpp::_["target"] = target,
        Rcpp::_["k"] = k);
  }
  auto v_or = rel::omega_multidim(tgt, S_m, spec);
  if (!v_or.has_value()) stop_post(v_or.error());
  auto g_or = rel::omega_multidim_gradient(tgt, S_m, spec);
  if (!g_or.has_value()) stop_post(g_or.error());
  return Rcpp::List::create(
      Rcpp::_["value"] = *v_or,
      Rcpp::_["se"] = NA_REAL,
      Rcpp::_["avar"] = NA_REAL,
      Rcpp::_["gradient"] = Rcpp::wrap(*g_or),
      Rcpp::_["n"] = NA_INTEGER,
      Rcpp::_["target"] = target,
      Rcpp::_["k"] = k);
}

// measures_reliability_omega_from_fit() — model-based coefficient omega
// (omega-total or omega-hierarchical) from a FITTED CFA, with an analytic-
// Jacobian delta-method robust SE that reuses a caller-supplied empirical Gamma.
// Mirrors measures::frontier::reliability::omega_from_fit. `target` is "total"
// or "hierarchical"; `weight` (the fitting estimator) is "ML", "GLS", or "ULS"
// and selects the robust-vcov path. `gamma` is the p* x p* empirical ACOV of
// vech(S) in the evaluator's lower-tri column-major vech metric. Returns the
// value, robust SE, asymptotic variance, and the per-free-parameter gradient.
//
// [[Rcpp::export]]
Rcpp::List measures_reliability_omega_from_fit(Rcpp::List fit,
                                               std::string target,
                                               std::string weight,
                                               Rcpp::NumericMatrix gamma,
                                               double n) {
  namespace rel = magmaan::measures::frontier::reliability;
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);

  rel::OmegaTarget tgt;
  if (target == "total") {
    tgt = rel::OmegaTarget::Total;
  } else if (target == "hierarchical") {
    tgt = rel::OmegaTarget::Hierarchical;
  } else {
    Rcpp::stop("magmaan: target must be 'total' or 'hierarchical'");
  }

  rel::FitWeight w;
  if (weight == "ML") {
    w = rel::FitWeight::ML;
  } else if (weight == "GLS") {
    w = rel::FitWeight::GLS;
  } else if (weight == "ULS") {
    w = rel::FitWeight::ULS;
  } else {
    Rcpp::stop("magmaan: weight must be 'ML', 'GLS', or 'ULS'");
  }

  const Eigen::MatrixXd G = Rcpp::as<Eigen::MatrixXd>(gamma);
  auto r_or = rel::omega_from_fit(tgt, w, ctx.pt, ctx.rep, ctx.samp, est, G,
                                  static_cast<std::int64_t>(n));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["value"] = r_or->value,
      Rcpp::_["se"] = r_or->se,
      Rcpp::_["avar"] = r_or->avar,
      Rcpp::_["gradient"] = Rcpp::wrap(r_or->gradient));
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
  const auto score_method = factor_score_method_from(method);
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  magmaan::post_expected<magmaan::measures::FactorScores> r_or;
  if (is_ordinal_fit) {
    if (!fit.containsElementNamed("ordinal_stats")) {
      Rcpp::stop("magmaan: ordinal factor scores require fit$ordinal_stats");
    }
    auto stats = ordinal_stats_from_arg(Rcpp::List(fit["ordinal_stats"]));
    r_or = magmaan::measures::factor_scores_ordinal(
        ctx.pt, ctx.rep, raw, stats, est, score_method,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else if (is_mixed_ordinal_fit) {
    if (!fit.containsElementNamed("mixed_ordinal_stats")) {
      Rcpp::stop("magmaan: mixed ordinal factor scores require fit$mixed_ordinal_stats");
    }
    auto stats =
        mixed_ordinal_stats_from_arg(Rcpp::List(fit["mixed_ordinal_stats"]));
    r_or = magmaan::measures::factor_scores_mixed_ordinal(
        ctx.pt, ctx.rep, raw, stats, est, score_method,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else {
    r_or = magmaan::measures::factor_scores(
        ctx.pt, ctx.rep, raw, est, score_method);
  }
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["scores"] = matrix_blocks_to_r(r_or->scores, ctx.rep.lv_names,
                                             /*square_names=*/false),
      Rcpp::_["method"] = method);
}

// measures_factor_score_precision() — EAP posterior variance and sample PRMSE
// for ordinal/mixed-ordinal one-factor scores.
//
// [[Rcpp::export]]
Rcpp::List measures_factor_score_precision(Rcpp::List fit, SEXP raw_data) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  magmaan::post_expected<magmaan::measures::FactorScorePrecision> r_or;
  if (is_ordinal_fit) {
    if (!fit.containsElementNamed("ordinal_stats")) {
      Rcpp::stop("magmaan: ordinal factor score precision requires fit$ordinal_stats");
    }
    auto stats = ordinal_stats_from_arg(Rcpp::List(fit["ordinal_stats"]));
    r_or = magmaan::measures::factor_score_precision_ordinal(
        ctx.pt, ctx.rep, raw, stats, est,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else if (is_mixed_ordinal_fit) {
    if (!fit.containsElementNamed("mixed_ordinal_stats")) {
      Rcpp::stop("magmaan: mixed ordinal factor score precision requires fit$mixed_ordinal_stats");
    }
    auto stats =
        mixed_ordinal_stats_from_arg(Rcpp::List(fit["mixed_ordinal_stats"]));
    r_or = magmaan::measures::factor_score_precision_mixed_ordinal(
        ctx.pt, ctx.rep, raw, stats, est,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else {
    Rcpp::stop("magmaan: factor score precision requires an ordinal or mixed-ordinal fit");
  }
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["scores"] = matrix_blocks_to_r(r_or->scores.scores,
                                             ctx.rep.lv_names,
                                             /*square_names=*/false),
      Rcpp::_["posterior_variance"] =
          matrix_blocks_to_r(r_or->posterior_variance, ctx.rep.lv_names,
                             /*square_names=*/false),
      Rcpp::_["posterior_se"] =
          matrix_blocks_to_r(r_or->posterior_se, ctx.rep.lv_names,
                             /*square_names=*/false),
      Rcpp::_["prmse_by_group"] = Rcpp::wrap(r_or->prmse_by_group),
      Rcpp::_["pooled_prmse"] = r_or->pooled_prmse,
      Rcpp::_["concrete_ordinal_reliability_by_group"] =
          Rcpp::wrap(r_or->concrete_ordinal_reliability_by_group),
      Rcpp::_["pooled_concrete_ordinal_reliability"] =
          r_or->pooled_concrete_ordinal_reliability,
      Rcpp::_["method"] = "EAP",
      Rcpp::_["targets"] = Rcpp::CharacterVector::create(
          "sample_prmse", "concrete_ordinal_reliability"),
      Rcpp::_["population"] = "sample");
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
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  const auto opts = modification_options_from(
      information, candidates, include_loadings, include_covariances);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;
  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "ordinal_stats", "ordinal modification indices"));
    out = magmaan::estimate::modification_indices_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal modification indices"),
        opts, ordinal_parameterization_from_string(
                  fit.containsElementNamed("parameterization")
                      ? Rcpp::as<std::string>(fit["parameterization"])
                      : ordinal_parameterization_attr(fit["partable"])));
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "mixed_ordinal_stats",
        "mixed ordinal modification indices"));
    out = magmaan::estimate::modification_indices_mixed_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal modification indices"),
        opts, ordinal_parameterization_from_string(
                  fit.containsElementNamed("parameterization")
                      ? Rcpp::as<std::string>(fit["parameterization"])
                      : ordinal_parameterization_attr(fit["partable"])));
  } else if (estimator == "FIML") {
    if (!fit.containsElementNamed("raw_data")) {
      Rcpp::stop("magmaan: FIML modification indices require fit$raw_data");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
    out = magmaan::inference::modification_indices_fiml(
        ctx.pt, ctx.rep, raw, est, opts, pack, magmaan::estimate::fiml::FIML{},
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
  } else if (estimator == "ML" || estimator.empty() || estimator == "ML2S") {
    // ML2S: fit$S/sample_mean are the Stage-1 EM-completed moments (the fit is
    // a Stage-2 ML on them), so ctx.samp already carries them and the one-step
    // sweep is the normal-theory ML score test on the EM moments (the naive
    // comparator; Stage-1 uncertainty enters only the robust lrt_p_obs column).
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
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;
  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "ordinal_stats", "ordinal score tests"));
    out = magmaan::estimate::score_tests_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "mixed_ordinal_stats", "mixed ordinal score tests"));
    out = magmaan::estimate::score_tests_mixed_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else if (estimator == "FIML") {
    if (!fit.containsElementNamed("raw_data")) {
      Rcpp::stop("magmaan: FIML score tests require fit$raw_data");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
    out = magmaan::inference::score_tests_fiml(
        ctx.pt, ctx.rep, raw, est, pack, magmaan::estimate::fiml::FIML{},
        h_step);
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

// ── Robust (generalized / Satorra-Bentler-scaled) MI & score tests ──────────
// Frontier mirror of inference_modification_indices/_score_tests, routed to the
// *_robust entry points: inference::frontier for continuous ML/ULS/GLS/WLS,
// estimate::frontier for ordinal/mixed. Each row keeps the ordinary `mi` and
// adds `mi_scaled = mi / scaling_factor` (the same DataFrame columns the
// non-robust sweep already emits).
//
// Meat: ordinal/mixed use the polychoric NACOV carried by the fit, so the
// scaling is intrinsic to W != NACOV^-1 (DWLS/ULS scale even on normal data)
// and `bread`/`moments`/`cov` do not apply. Continuous fits build the bread from
// the estimation weight (ULS identity / GLS normal-theory / WLS the supplied
// `weight`) and the meat from `cov`: 'empirical'/'browne_unbiased' need the raw
// fitting data (`raw`); 'model_implied' uses Gamma_NT(S) and collapses to the
// ordinary statistic. Full-WLS collapses regardless (W = Gamma^-1).

namespace {

// Estimation weight for a continuous LS fit (empty for ULS; ML carries none and
// is handled by the caller via the weight-free overloads).
magmaan::estimate::gmm::Weight continuous_ls_weight(
    const Ctx& ctx, const magmaan::estimate::Estimates& est,
    const std::string& estimator, SEXP weight, const char* call) {
  if (estimator == "ULS") return magmaan::estimate::gmm::Weight{};
  if (estimator == "GLS") {
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) stop_model(ev_or.error());
    auto w_or =
        magmaan::estimate::gmm::normal_theory_weight(*ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) stop_fit(w_or.error());
    return *w_or;
  }
  if (estimator == "WLS") {
    if (Rf_isNull(weight))
      Rcpp::stop("magmaan: WLS robust %s require an explicit `weight`", call);
    return wls_from_arg(weight, ctx.samp.S.size());
  }
  Rcpp::stop("magmaan: robust %s are not exposed for estimator '%s'", call,
             estimator.c_str());
}

// Which second-stage weight's data influence the estimated-weight continuous-LS
// IJ meat should carry, from the fit's estimator label (ULS has a fixed weight).
magmaan::estimate::ContinuousLsIJWeightMode continuous_ij_mode(
    const std::string& estimator) {
  if (estimator == "GLS")
    return magmaan::estimate::ContinuousLsIJWeightMode::SampleNormalTheory;
  if (estimator == "WLS")
    return magmaan::estimate::ContinuousLsIJWeightMode::SampleEmpiricalWls;
  return magmaan::estimate::ContinuousLsIJWeightMode::Fixed;  // ULS
}

}  // namespace

// infer_continuous_ls_profile_lrt() — misspecification-robust ("observed-Hessian
// profile bread") nested difference test for two continuous moment-quadratic
// (ULS/GLS/WLS) fits sharing the same observed data and one caller-fixed weight.
// The LRT counterpart of infer_ml_profile_lrt for continuous LS. `X_per_group`
// is the per-group raw data in the model's ov order (the empirical Gamma is built
// from it). The shared weight is built at the H0 (anchor) theta, mirroring
// inference_modification_indices: ULS=identity, GLS=normal-theory, WLS=explicit
// `weight` (not retained on the fit, so it must be passed for WLS).
//
// [[Rcpp::export]]
Rcpp::List infer_continuous_ls_profile_lrt(Rcpp::List fit_H1,
                                           Rcpp::List fit_H0,
                                           Rcpp::List X_per_group,
                                           SEXP weight = R_NilValue,
                                           double eig_tol = 1e-10) {
  Ctx ctx1 = ctx_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates est0 = est_from_fit(fit_H0);

  const std::string est_H0 = fit_H0.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H0["estimator"]) : "";
  const std::string est_H1 = fit_H1.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H1["estimator"]) : "";
  if (est_H0 != est_H1) {
    Rcpp::stop("infer_continuous_ls_profile_lrt: H1/H0 estimators differ "
               "('%s' vs '%s')", est_H1.c_str(), est_H0.c_str());
  }
  // One weight shared by H1 and H0, built at the anchor (H0) theta.
  const magmaan::estimate::gmm::Weight w =
      continuous_ls_weight(ctx0, est0, est_H0, weight, "profile LRT");

  const std::size_t G = ctx1.samp.S.size();
  if (static_cast<std::size_t>(X_per_group.size()) != G) {
    Rcpp::stop("infer_continuous_ls_profile_lrt: X_per_group has length %d but "
               "the model has %d group(s)",
               static_cast<int>(X_per_group.size()), static_cast<int>(G));
  }
  magmaan::data::RawData raw;
  raw.X.reserve(G);
  for (std::size_t g = 0; g < G; ++g) {
    raw.X.emplace_back(
        Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X_per_group[g])));
  }

  auto r_or = magmaan::estimate::continuous_ls_profile_lrt(
      std::move(ctx1.pt), ctx1.rep, ctx1.samp, est1,
      std::move(ctx0.pt), ctx0.rep, est0,
      w, raw, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_lrt_to_list(*r_or);
}

// infer_fiml_profile_lrt() — misspecification-robust ("observed-Hessian profile
// bread") nested difference test for two raw-data FIML fits sharing the same
// incomplete data. The LRT counterpart of infer_ml_profile_lrt for FIML: H1/H0
// share one missingness/saturated stage (pack + h1), and the model-vs-saturated
// chi-squares come from fiml_extras() per model. No `data` arg — reads
// fit$raw_data (the magmaan_fiml_data object) directly.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_profile_lrt(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                  double eig_tol = 1e-10) {
  if (!fit_H1.containsElementNamed("raw_data") ||
      !fit_H0.containsElementNamed("raw_data")) {
    Rcpp::stop("infer_fiml_profile_lrt: both H1 and H0 require a FIML fit "
               "with $raw_data");
  }
  Ctx ctx1 = ctx_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates est0 = est_from_fit(fit_H0);

  // One missingness/saturated stage shared by H1 and H0 (same observed data).
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx1.rep, fit_H1["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit_H1, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit_H1, raw, pack, owned_h1);

  // Model-vs-saturated FIML chi-square for each model (fiml_extras copies pt).
  auto x1 = magmaan::estimate::fiml::fiml_extras(ctx1.pt, ctx1.rep, raw, est1,
                                                 pack, h1);
  if (!x1.has_value()) stop_post(x1.error());
  auto x0 = magmaan::estimate::fiml::fiml_extras(ctx0.pt, ctx0.rep, raw, est0,
                                                 pack, h1);
  if (!x0.has_value()) stop_post(x0.error());

  auto r_or = magmaan::estimate::fiml::fiml_profile_lrt(
      std::move(ctx1.pt), ctx1.rep, raw, est1, x1->chi2,
      std::move(ctx0.pt), ctx0.rep, est0, x0->chi2,
      pack, h1, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_lrt_to_list(*r_or);
}

// infer_two_stage_nt_profile_lrt() — misspecification-robust profile-LRT for two
// nested two-stage ML (ML2S, NT Stage-2) fits. The Stage-1 EM moments are the
// complete-data ML sample statistics and Stage-1 uncertainty is the block stacked
// [mean; vech(cov)] Gamma; reconstructed from the anchor's $stage1 (no recompute).
// T_diff / p_unscaled are the plain difference, p_mixture the robust tail.
//
// [[Rcpp::export]]
Rcpp::List infer_two_stage_nt_profile_lrt(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                          double eig_tol = 1e-10) {
  Ctx ctx1 = ctx_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates est0 = est_from_fit(fit_H0);

  // Shared Stage-1 EM moments (reconstructed from the anchor's $stage1).
  magmaan::estimate::fiml::SaturatedMoments sm;
  if (!saturated_from_stage1(fit_H1, sm)) {
    Rcpp::stop("infer_two_stage_nt_profile_lrt: H1 fit is missing $stage1 "
               "(not an ML2S fit?)");
  }
  auto r_or = magmaan::estimate::fiml::two_stage_nt_profile_lrt(
      std::move(ctx1.pt), ctx1.rep, est1,
      std::move(ctx0.pt), ctx0.rep, est0, sm, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_lrt_to_list(*r_or);
}

// measures_standardized_residuals_estimated_weight() — the estimated-weight
// ("complete-sandwich") residual SE/z and $summary for a continuous LS fit. The
// residual ACOV uses the Hall-Inoue infinitesimal-jackknife rows (with the
// data-dependent-weight influence) instead of the NT projection. Continuous
// GLS/WLS/ULS only; needs the fitting `data` (raw observations). Beyond lavaan.
//
// [[Rcpp::export]]
Rcpp::List measures_standardized_residuals_estimated_weight(
    Rcpp::List fit, SEXP raw_data, SEXP weight = R_NilValue,
    double conf_level = 0.90) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if ((fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"])) ||
      (fit.containsElementNamed("mixed_ordinal") &&
       Rcpp::as<bool>(fit["mixed_ordinal"]))) {
    Rcpp::stop("magmaan: estimated_weight residuals are continuous-LS only "
               "(GLS/WLS/ULS); not available for ordinal fits");
  }
  if (estimator == "ML" || estimator.empty() || estimator == "FIML") {
    Rcpp::stop("magmaan: estimated_weight residuals need an estimated second-"
               "stage weight (GLS/WLS); estimator '%s' carries none",
               estimator.c_str());
  }
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  auto wls = continuous_ls_weight(ctx, est, estimator, weight,
                                  "estimated_weight residuals");
  auto r_or =
      magmaan::measures::frontier::standardized_residuals_estimated_weight(
          ctx.pt, ctx.rep, ctx.samp, est, wls, raw,
          continuous_ij_mode(estimator), {}, conf_level);
  if (!r_or.has_value()) stop_post(r_or.error());
  return standardized_residuals_to_r(*r_or, ctx.rep.ov_names);
}

// infer_casewise_influence_ij_fit() — per-case one-step misspecification-robust
// ("complete-sandwich") parameter influences for a continuous-LS fit: the
// casewise dual of semfindr's est_change_raw_approx. Returns the N_total x
// n_free `influence` matrix (the moment-quadratic analogue of scores·V, whose
// column-Gram is the estimated-weight IJ vcov) and its fixed-weight `naive`
// counterpart; the R layer forms DFTHETAS / gCD and the data-dependent-weight
// diagnostic (influence − naive) from these. Continuous GLS/WLS/ULS only; needs
// the fitting raw data. Beyond lavaan/semfindr.
//
// [[Rcpp::export]]
Rcpp::List infer_casewise_influence_ij_fit(
    Rcpp::List fit, SEXP raw_data, SEXP weight = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if ((fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"])) ||
      (fit.containsElementNamed("mixed_ordinal") &&
       Rcpp::as<bool>(fit["mixed_ordinal"]))) {
    Rcpp::stop("magmaan: estimated_weight case influence is continuous-LS only "
               "(GLS/WLS/ULS); not available for ordinal fits");
  }
  if (estimator == "ML" || estimator.empty() || estimator == "FIML") {
    Rcpp::stop("magmaan: estimated_weight case influence needs an estimated "
               "second-stage weight (GLS/WLS); estimator '%s' carries none",
               estimator.c_str());
  }
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  auto wls = continuous_ls_weight(ctx, est, estimator, weight,
                                  "estimated_weight case influence");
  auto r_or = magmaan::estimate::continuous_ls_casewise_influence_ij(
      ctx.pt, ctx.rep, ctx.samp, est, wls, raw, continuous_ij_mode(estimator));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::Named("influence") = Rcpp::wrap(r_or->influence),
      Rcpp::Named("influence_naive") = Rcpp::wrap(r_or->influence_naive),
      Rcpp::Named("n_total") = static_cast<double>(r_or->n_total));
}

// [[Rcpp::export]]
Rcpp::DataFrame inference_modification_indices_robust(
    Rcpp::List fit, SEXP raw = R_NilValue, SEXP weight = R_NilValue,
    std::string bread = "expected", std::string moments = "structured",
    std::string cov = "empirical", std::string information = "expected",
    std::string candidates = "fixed", bool include_loadings = true,
    bool include_covariances = true, bool estimated_weight = false) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  const auto base = modification_options_from(
      information, candidates, include_loadings, include_covariances);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;

  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats",
        "ordinal robust modification indices"));
    out = magmaan::estimate::frontier::modification_indices_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal robust modification indices"),
        base,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "mixed_ordinal_stats",
        "mixed ordinal robust modification indices"));
    out = magmaan::estimate::frontier::modification_indices_mixed_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal robust modification indices"),
        base,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else {
    const bool is_ml = (estimator == "ML" || estimator.empty());
    magmaan::inference::frontier::RobustScoreOptions opts;
    opts.base = base;
    if (estimated_weight) {
      // The complete (estimated-weight) sandwich needs the fitting data and an
      // estimated second-stage weight; ML has none.
      if (is_ml) {
        Rcpp::stop("magmaan: estimated_weight robust modification indices need "
                   "an estimated second-stage weight (GLS/WLS, or an ordinal "
                   "fit), not ML");
      }
      if (Rf_isNull(raw)) {
        Rcpp::stop("magmaan: estimated_weight robust modification indices "
                   "require the fitting data; pass data=");
      }
      opts.spec = spec_from(bread, moments,
                            cov == "model_implied" ? "empirical" : cov);
      opts.estimated_weight = true;
      opts.ij_weight_mode = continuous_ij_mode(estimator);
      magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
      auto w = continuous_ls_weight(ctx, est, estimator, weight,
                                    "modification indices");
      out = magmaan::inference::frontier::modification_indices_robust(
          ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
    } else {
      const bool model_implied = (cov == "model_implied") || (estimator == "WLS");
      opts.spec = spec_from(bread, moments, model_implied ? "model_implied" : cov);
      if (model_implied) {
        if (is_ml) {
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, est, opts);
        } else {
          auto w = continuous_ls_weight(ctx, est, estimator, weight,
                                        "modification indices");
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, est, w, opts);
        }
      } else {
        if (Rf_isNull(raw)) {
          Rcpp::stop("magmaan: robust modification indices with cov='%s' "
                     "require the fitting data; pass data= (or use "
                     "cov='model_implied')", cov.c_str());
        }
        magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
        if (is_ml) {
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, opts);
        } else {
          auto w = continuous_ls_weight(ctx, est, estimator, weight,
                                        "modification indices");
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
        }
      }
    }
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// [[Rcpp::export]]
Rcpp::DataFrame inference_score_tests_robust(
    Rcpp::List fit, SEXP raw = R_NilValue, SEXP weight = R_NilValue,
    std::string bread = "expected", std::string moments = "structured",
    std::string cov = "empirical", bool estimated_weight = false) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;

  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats", "ordinal robust score tests"));
    out = magmaan::estimate::frontier::score_tests_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal robust score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "mixed_ordinal_stats",
        "mixed ordinal robust score tests"));
    out = magmaan::estimate::frontier::score_tests_mixed_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal robust score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else {
    const bool is_ml = (estimator == "ML" || estimator.empty());
    magmaan::inference::frontier::RobustScoreOptions opts;
    if (estimated_weight) {
      if (is_ml) {
        Rcpp::stop("magmaan: estimated_weight robust score tests need an "
                   "estimated second-stage weight (GLS/WLS, or an ordinal "
                   "fit), not ML");
      }
      if (Rf_isNull(raw)) {
        Rcpp::stop("magmaan: estimated_weight robust score tests require the "
                   "fitting data; pass data=");
      }
      opts.spec = spec_from(bread, moments,
                            cov == "model_implied" ? "empirical" : cov);
      opts.estimated_weight = true;
      opts.ij_weight_mode = continuous_ij_mode(estimator);
      magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
      auto w = continuous_ls_weight(ctx, est, estimator, weight, "score tests");
      out = magmaan::inference::frontier::score_tests_robust(
          ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
    } else {
      const bool model_implied = (cov == "model_implied") || (estimator == "WLS");
      opts.spec = spec_from(bread, moments, model_implied ? "model_implied" : cov);
      if (model_implied) {
        if (is_ml) {
          Rcpp::stop("magmaan: ML robust score tests require the fitting data "
                     "(cov='empirical'); pass data=");
        }
        auto w = continuous_ls_weight(ctx, est, estimator, weight, "score tests");
        out = magmaan::inference::frontier::score_tests_robust(
            ctx.pt, ctx.rep, ctx.samp, est, w, opts);
      } else {
        if (Rf_isNull(raw)) {
          Rcpp::stop("magmaan: robust score tests with cov='%s' require the "
                     "fitting data; pass data= (or use cov='model_implied')",
                     cov.c_str());
        }
        magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
        if (is_ml) {
          out = magmaan::inference::frontier::score_tests_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, opts);
        } else {
          auto w =
              continuous_ls_weight(ctx, est, estimator, weight, "score tests");
          out = magmaan::inference::frontier::score_tests_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
        }
      }
    }
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

// ---- frontier: marker <-> std_lv identification swap --------------------
// Exposed to R only via the `frontier_*` aliases in r-package/R/zzz_core.R.
// Implementation lives in src/model/auto_identification.cpp.

namespace {

magmaan::compat::lavaan::LavaanParTable lavaan_pt_from_df(
    Rcpp::DataFrame df) {
  auto col = [&](const char* nm) -> SEXP {
    if (!df.containsElementNamed(nm))
      Rcpp::stop("magmaan: partable data.frame is missing required column '%s'", nm);
    return df[nm];
  };
  Rcpp::IntegerVector id(col("id")), user(col("user")), block(col("block")),
      group(col("group")), freev(col("free")), exo(col("exo"));
  Rcpp::NumericVector ustart(col("ustart"));
  std::vector<std::string> lhs   = Rcpp::as<std::vector<std::string>>(col("lhs"));
  std::vector<std::string> opstr = Rcpp::as<std::vector<std::string>>(col("op"));
  std::vector<std::string> rhs   = Rcpp::as<std::vector<std::string>>(col("rhs"));
  std::vector<std::string> label = Rcpp::as<std::vector<std::string>>(col("label"));
  std::vector<std::string> plab  = Rcpp::as<std::vector<std::string>>(col("plabel"));

  const std::size_t m = static_cast<std::size_t>(id.size());
  magmaan::compat::lavaan::LavaanParTable lvpt;
  lvpt.id.resize(m); lvpt.user.resize(m); lvpt.lhs.resize(m); lvpt.op.resize(m);
  lvpt.rhs.resize(m); lvpt.block.resize(m); lvpt.group.resize(m);
  lvpt.free.resize(m); lvpt.exo.resize(m); lvpt.ustart.resize(m);
  lvpt.label.resize(m); lvpt.plabel.resize(m);
  for (std::size_t i = 0; i < m; ++i) {
    const R_xlen_t ri = static_cast<R_xlen_t>(i);
    lvpt.id[i]     = id[ri];
    lvpt.user[i]   = static_cast<std::int8_t>(user[ri]);
    lvpt.lhs[i]    = lhs[i];
    lvpt.op[i]     = op_from_string(opstr[i]);
    lvpt.rhs[i]    = rhs[i];
    lvpt.block[i]  = block[ri];
    lvpt.group[i]  = group[ri];
    lvpt.free[i]   = freev[ri];
    lvpt.exo[i]    = static_cast<std::int8_t>(exo[ri]);
    lvpt.ustart[i] = ustart[ri];
    lvpt.label[i]  = label[i];
    lvpt.plabel[i] = plab[i];
  }
  return lvpt;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List frontier_is_std_lv_admissible_impl(
    Rcpp::DataFrame marker_partable,
    Rcpp::Nullable<Rcpp::DataFrame> std_lv_partable) {
  const auto marker_pt = lavaan_pt_from_df(marker_partable);
  magmaan::model::AdmissibilityVerdict v;
  if (std_lv_partable.isNotNull()) {
    const auto std_lv_pt = lavaan_pt_from_df(std_lv_partable.get());
    v = magmaan::model::is_std_lv_admissible(marker_pt, std_lv_pt);
  } else {
    v = magmaan::model::is_std_lv_admissible(marker_pt);
  }
  return Rcpp::List::create(Rcpp::_["admissible"] = v.admissible,
                            Rcpp::_["reason"]    = v.reason);
}

// [[Rcpp::export]]
Rcpp::DataFrame frontier_partable_marker_to_std_lv_impl(
    Rcpp::DataFrame marker_partable) {
  const auto marker_pt = lavaan_pt_from_df(marker_partable);
  const auto std_lv_pt = magmaan::model::partable_marker_to_std_lv(marker_pt);
  return magmaanr::partable_df_from_lavaan(std_lv_pt);
}

// [[Rcpp::export]]
Rcpp::NumericVector frontier_backconvert_std_lv_to_marker_impl(
    Rcpp::DataFrame marker_partable,
    Rcpp::NumericVector std_lv_est) {
  const auto marker_pt = lavaan_pt_from_df(marker_partable);
  const std::size_t n = marker_pt.size();
  if (static_cast<std::size_t>(std_lv_est.size()) != n) {
    Rcpp::stop("frontier_backconvert: length(std_lv_est) (%d) != partable rows (%d)",
               static_cast<int>(std_lv_est.size()), static_cast<int>(n));
  }
  Eigen::Map<const Eigen::VectorXd> est_map(
      &std_lv_est[0], static_cast<Eigen::Index>(n));
  const Eigen::VectorXd out =
      magmaan::model::backconvert_std_lv_to_marker(marker_pt, est_map);
  Rcpp::NumericVector r_out(static_cast<R_xlen_t>(n));
  for (std::size_t i = 0; i < n; ++i)
    r_out[static_cast<R_xlen_t>(i)] = out[static_cast<Eigen::Index>(i)];
  return r_out;
}
