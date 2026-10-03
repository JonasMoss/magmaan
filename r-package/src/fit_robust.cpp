#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

// infer_continuous_ls_robust() — the R binding for the existing
// robust_continuous_ls() C++ post-fit path. ULS uses the identity weight, GLS
// rebuilds its fitted normal-theory weight, and WLS requires the explicit
// fitting weight because fit lists do not retain caller-supplied W. The meat
// can be empirical (from raw rows) or normal-theory (from the fitted sample
// covariance); both are existing C++ overloads.
//
// [[Rcpp::export]]
Rcpp::List infer_continuous_ls_robust(
    Rcpp::List fit, SEXP raw_data, SEXP weight = R_NilValue,
    std::string bread = "expected", std::string gamma = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  if (estimator != "ULS" && estimator != "GLS" && estimator != "WLS") {
    Rcpp::stop("infer_continuous_ls_robust() requires a continuous ULS/GLS/WLS "
               "fit, got estimator '%s'", estimator.c_str());
  }

  magmaan::estimate::gmm::Weight w =
      continuous_ls_weight(fit, ctx, est, estimator, weight,
                           "continuous-LS inference");
  for (char& ch : gamma) {
    if (ch == '-' || ch == '.') ch = '_';
    else ch = static_cast<char>(
        std::tolower(static_cast<unsigned char>(ch)));
  }
  magmaan::post_expected<magmaan::estimate::WeightedRobustResult> r_or;
  if (gamma == "empirical" || gamma == "adf") {
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
    r_or = magmaan::estimate::robust_continuous_ls(
        std::move(ctx.pt), ctx.rep, ctx.samp, est, w, raw,
        info_from_string(bread));
  } else if (gamma == "normal" || gamma == "normal_theory" ||
             gamma == "nt") {
    std::vector<Eigen::MatrixXd> gamma_nt;
    gamma_nt.reserve(ctx.samp.S.size());
    for (const auto& S : ctx.samp.S) {
      auto g_or = ctx.meanstructure
          ? magmaan::data::gamma_nt_with_means(S)
          : magmaan::data::gamma_nt(S);
      if (!g_or.has_value()) stop_post(g_or.error());
      gamma_nt.push_back(std::move(*g_or));
    }
    r_or = magmaan::estimate::robust_continuous_ls(
        std::move(ctx.pt), ctx.rep, ctx.samp, est, w, gamma_nt,
        info_from_string(bread));
  } else {
    Rcpp::stop("infer_continuous_ls_robust(): `gamma` must be 'empirical' "
               "or 'normal'");
  }
  if (!r_or.has_value()) stop_post(r_or.error());
  const magmaan::estimate::WeightedRobustResult& r = *r_or;
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r.vcov),
      Rcpp::_["se"] = Rcpp::wrap(r.se),
      Rcpp::_["df"] = r.df,
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["chisq_standard"] = r.chisq_standard,
      Rcpp::_["satorra_bentler"] = Rcpp::List::create(
          Rcpp::_["chi2_scaled"] = r.satorra_bentler.chi2_scaled,
          Rcpp::_["scale_c"] = r.satorra_bentler.scale_c,
          Rcpp::_["df"] = r.satorra_bentler.df),
      Rcpp::_["mean_var_adjusted"] = Rcpp::List::create(
          Rcpp::_["chi2_adj"] = r.mean_var_adjusted.chi2_adj,
          Rcpp::_["df_adj"] = r.mean_var_adjusted.df_adj),
      Rcpp::_["scaled_shifted"] =
          scaled_shifted_to_list(r.scaled_shifted));
}

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
      continuous_ls_weight(fit_H0, ctx0, est0, est_H0, weight, "profile LRT");

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
  auto wls = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                  "estimated_weight residuals");
  magmaan::estimate::gmm::FixedWeightOptions dls_opts;
  const auto mode = continuous_ij_mode_for_fit(fit, estimator, &dls_opts);
  auto r_or =
      magmaan::measures::frontier::standardized_residuals_estimated_weight(
          ctx.pt, ctx.rep, ctx.samp, est, wls, raw, mode, dls_opts,
          conf_level);
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
  auto wls = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                  "estimated_weight case influence");
  magmaan::estimate::gmm::FixedWeightOptions dls_opts;
  const auto mode = continuous_ij_mode_for_fit(fit, estimator, &dls_opts);
  auto r_or = magmaan::estimate::continuous_ls_casewise_influence_ij(
      ctx.pt, ctx.rep, ctx.samp, est, wls, raw, mode, dls_opts);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::Named("influence") = Rcpp::wrap(r_or->influence),
      Rcpp::Named("influence_naive") = Rcpp::wrap(r_or->influence_naive),
      Rcpp::Named("n_total") = static_cast<double>(r_or->n_total));
}
