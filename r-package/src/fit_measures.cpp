#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

Rcpp::List standardized_to_list(
    const magmaan::measures::standardize::StandardizedSolution& r);
Rcpp::List reliability_results_to_r(
    const Eigen::MatrixXd& S,
    Rcpp::Nullable<Rcpp::NumericMatrix> gamma,
    int n);
Rcpp::DataFrame composite_weights_df(
    const std::vector<magmaan::measures::composite::CompositeWeights>& rows);
magmaan::measures::FactorScoreMethod factor_score_method_from(
    const std::string& method);

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

}  // namespace

// model_implied() — mirrors ModelEvaluator::build(pt, rep).sigma(est.theta).
// Returns list(sigma = list of per-block p x p matrices, mu = list of per-block
// vectors — empty unless the model has mean structure).
//
// [[Rcpp::export]]
Rcpp::List model_implied(Rcpp::List fit) {
  if (auto cached=magmaanr::ntml_snapshot(fit)) {
    auto g=magmaan::robust::frontier::ntml_geometry(*cached);
    if (!g) stop_post(g.error());
    Rcpp::List sigma((*g)->base.blocks.size()),mu((*g)->mean_hat.size());
    for (std::size_t b=0;b<(*g)->base.blocks.size();++b) sigma[b]=Rcpp::wrap((*g)->base.blocks[b].Sigma_hat);
    for (std::size_t b=0;b<(*g)->mean_hat.size();++b) mu[b]=Rcpp::wrap((*g)->mean_hat[b]);
    return Rcpp::List::create(Rcpp::_["sigma"]=sigma,Rcpp::_["mu"]=mu);
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit, true);
  auto ev_or = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev_or.has_value()) stop_model(ev_or.error());
  const lvm::ModelEvaluator ev = std::move(*ev_or);
  auto eval = ev.evaluate(est.theta, false, false);
  if (!eval) stop_model(eval.error());
  if (est.association) {
    eval = lvm::correlation_evaluation(std::move(*eval));
    if (!eval) stop_model(eval.error());
  }
  const lvm::ImpliedMoments& im = eval->moments;

  Rcpp::List sigma(static_cast<R_xlen_t>(im.sigma.size()));
  for (std::size_t b = 0; b < im.sigma.size(); ++b)
    sigma[static_cast<R_xlen_t>(b)] = Rcpp::wrap(im.sigma[b]);
  Rcpp::List mu(static_cast<R_xlen_t>(im.mu.size()));
  for (std::size_t b = 0; b < im.mu.size(); ++b)
    mu[static_cast<R_xlen_t>(b)] = Rcpp::wrap(im.mu[b]);
  return Rcpp::List::create(Rcpp::_["sigma"] = sigma, Rcpp::_["mu"] = mu);
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
  // response SDs are 1/delta; the core includes their live/fixed scales in
  // std.all instead of using the residual matrix placeholder.
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
    int n = 0,
    Rcpp::Nullable<Rcpp::NumericMatrix> raw_data = R_NilValue) {
  if (gamma.isNull()) {
    if (raw_data.isNull())
      Rcpp::stop("reliability: gamma = NULL requires raw_data; supply raw data or an explicit Gamma");
    const Eigen::MatrixXd X = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(raw_data.get()));
    auto g = magmaan::data::empirical_gamma(X);
    if (!g) stop_post(g.error());
    gamma = Rcpp::wrap(*g);
    if (n == 0) n = static_cast<int>(X.rows());
  }
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
    int n = 0,
    Rcpp::Nullable<Rcpp::NumericMatrix> raw_data = R_NilValue) {
  namespace rel = magmaan::measures::frontier::reliability;
  if (gamma.isNull()) {
    if (raw_data.isNull())
      Rcpp::stop("reliability: gamma = NULL requires raw_data; supply raw data or an explicit Gamma");
    const Eigen::MatrixXd X = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(raw_data.get()));
    auto g = magmaan::data::empirical_gamma(X);
    if (!g) stop_post(g.error());
    gamma = Rcpp::wrap(*g);
    if (n == 0) n = static_cast<int>(X.rows());
  }
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

// measures_reliability_ordinal_observed_omega() — observed-category-score
// omega from a fitted single-group all-ordinal LS model. The fitted thresholds
// and latent-response correlations induce the covariance of integer category
// scores (0,1,...,K-1); the SE is the complete ordinal IJ sandwich delta SE.
//
// [[Rcpp::export]]
Rcpp::List measures_reliability_ordinal_observed_omega(
    Rcpp::List fit,
    Rcpp::IntegerVector block,
    std::string target = "total",
    std::string weight = "fit",
    SEXP ordinal_stats = R_NilValue) {
  namespace rel = magmaan::measures::frontier::reliability;
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  if (!fit.containsElementNamed("ordinal") ||
      !Rcpp::as<bool>(fit["ordinal"])) {
    Rcpp::stop("magmaan: measures_reliability_ordinal_observed_omega() "
               "requires an all-ordinal fit");
  }

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(
      stats_from_fit_or_arg(fit, ordinal_stats, "ordinal_stats",
                            "measures_reliability_ordinal_observed_omega"));
  if (stats.R.empty()) {
    Rcpp::stop("magmaan: ordinal_stats has no blocks");
  }
  const R_xlen_t p = stats.R[0].rows();
  if (static_cast<R_xlen_t>(block.size()) != p) {
    Rcpp::stop("magmaan: block length must equal the number of ordinal indicators");
  }

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
  const int k = static_cast<int>(uniq.size());

  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "measures_reliability_ordinal_observed_omega");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const auto r_or = magmaan::estimate::frontier::ordinal_observed_omega(
      ctx.pt, ctx.rep, stats, est, spec,
      omega_target_from_string(target,
                               "measures_reliability_ordinal_observed_omega"),
      ow, ordinal_parameterization_from_string(parameterization_name));
  if (!r_or.has_value()) stop_post(r_or.error());

  std::int64_t n_total = 0;
  for (std::int64_t n : stats.n_obs) n_total += n;
  return Rcpp::List::create(
      Rcpp::_["value"] = r_or->value,
      Rcpp::_["se"] = r_or->se,
      Rcpp::_["avar"] = r_or->avar,
      Rcpp::_["gradient"] = Rcpp::wrap(r_or->gradient),
      Rcpp::_["n"] = static_cast<double>(n_total),
      Rcpp::_["target"] = target,
      Rcpp::_["k"] = k,
      Rcpp::_["weight"] = weight_key,
      Rcpp::_["parameterization"] = parameterization_name);
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
