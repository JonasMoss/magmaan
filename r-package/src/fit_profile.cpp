#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

struct OmegaBlockSpec {
  magmaan::measures::frontier::reliability::OmegaSpec spec;
  int k = 0;
};

Rcpp::List fiml_profile_fit_result(
    Ctx& ctx,
    const magmaan::data::RawData& raw,
    const magmaan::estimate::Estimates& est,
    Rcpp::List source_fit);
Rcpp::List ml2s_profile_fit_result(
    Ctx& ctx,
    const magmaan::estimate::Estimates& est,
    Rcpp::List source_fit);
double nan_if_not_finite(double x);
magmaan::estimate::frontier::ScalarProfileCiOptions profile_ci_options_from_args(
    double level, double lower, double upper, double initial_step,
    double root_tol, double statistic_tol);
magmaan::estimate::frontier::ScalarProfileReference
scalar_reference_from_nullable(Rcpp::Nullable<Rcpp::String> reference,
                               bool robust,
                               const char* call);
bool scalar_reference_needs_sandwich(
    magmaan::estimate::frontier::ScalarProfileReference reference);
OmegaBlockSpec omega_spec_from_block(Rcpp::IntegerVector block,
                                     Eigen::Index p,
                                     const char* call);
Rcpp::List scalar_profile_lrt_base_to_list(
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter,
    Rcpp::List constrained);
Rcpp::List scalar_profile_lrt_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats = nullptr,
    const char* ordinal_parameterization = "delta");
Rcpp::List scalar_profile_ci_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats = nullptr,
    const char* ordinal_parameterization = "delta");
Rcpp::List scalar_profile_lrt_to_list_fiml(
    Ctx& ctx,
    const magmaan::data::RawData& raw,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter);
Rcpp::List scalar_profile_lrt_to_list_ml2s(
    Ctx& ctx,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter);
Rcpp::List scalar_profile_lrt_to_list_mixed_ordinal(
    Ctx& ctx,
    const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter,
    const char* estimator,
    const char* parameterization);
Rcpp::List scalar_profile_ci_to_list_fiml(
    Ctx& ctx,
    const magmaan::data::RawData& raw,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter);
Rcpp::List scalar_profile_ci_to_list_ml2s(
    Ctx& ctx,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter);
Rcpp::List scalar_profile_ci_to_list_mixed_ordinal(
    Ctx& ctx,
    const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter,
    const char* estimator,
    const char* parameterization);
Rcpp::List scalar_functional_profile_lrt_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    const char* coefficient,
    const char* coefficient_target,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats = nullptr,
    const char* ordinal_parameterization = "delta");
Rcpp::List scalar_functional_profile_ci_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    const char* coefficient,
    const char* coefficient_target,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats = nullptr,
    const char* ordinal_parameterization = "delta");

Rcpp::List fiml_profile_fit_result(
    Ctx& ctx,
    const magmaan::data::RawData& raw,
    const magmaan::estimate::Estimates& est,
    Rcpp::List source_fit) {
  Rcpp::List out = fiml_fit_result(ctx, raw, est, nullptr);
  if (source_fit.containsElementNamed("fiml_pack")) {
    out["fiml_pack"] = source_fit["fiml_pack"];
  }
  if (source_fit.containsElementNamed("fiml_h1")) {
    out["fiml_h1"] = source_fit["fiml_h1"];
  }
  return out;
}

Rcpp::List ml2s_profile_fit_result(
    Ctx& ctx,
    const magmaan::estimate::Estimates& est,
    Rcpp::List source_fit) {
  Rcpp::List out = fit_result(ctx, est, nullptr, "ML2S");
  if (source_fit.containsElementNamed("stage1")) {
    out["stage1"] = source_fit["stage1"];
  }
  if (source_fit.containsElementNamed("stage1_raw")) {
    out["stage1_raw"] = source_fit["stage1_raw"];
  }
  if (source_fit.containsElementNamed("stage1_regularization")) {
    out["stage1_regularization"] = source_fit["stage1_regularization"];
  }
  if (source_fit.containsElementNamed("raw_data")) {
    out["raw_data"] = source_fit["raw_data"];
  }
  if (source_fit.containsElementNamed("stage2_weight")) {
    out["stage2_weight"] = source_fit["stage2_weight"];
  } else {
    out["stage2_weight"] = "nt";
  }
  if (source_fit.containsElementNamed("stage2_dls_a")) {
    out["stage2_dls_a"] = source_fit["stage2_dls_a"];
  }
  return out;
}

double nan_if_not_finite(double x) {
  return std::isfinite(x) ? x : std::numeric_limits<double>::quiet_NaN();
}

magmaan::estimate::frontier::ScalarProfileCiOptions profile_ci_options_from_args(
    double level, double lower, double upper, double initial_step,
    double root_tol, double statistic_tol) {
  magmaan::estimate::frontier::ScalarProfileCiOptions out;
  out.confidence_level = level;
  out.lower_bound = nan_if_not_finite(lower);
  out.upper_bound = nan_if_not_finite(upper);
  out.initial_step = nan_if_not_finite(initial_step);
  out.target_tol = root_tol;
  out.statistic_tol = statistic_tol;
  return out;
}

magmaan::estimate::frontier::ScalarProfileReference
scalar_reference_from_nullable(Rcpp::Nullable<Rcpp::String> reference,
                               bool robust,
                               const char* call) {
  using magmaan::estimate::frontier::ScalarProfileReference;
  if (reference.isNull()) {
    return robust ? ScalarProfileReference::RobustScaled
                  : ScalarProfileReference::MisspecScaled;
  }
  std::string key = Rcpp::as<std::string>(reference.get());
  for (char& ch : key) {
    if (ch == '-') ch = '_';
    else ch = static_cast<char>(
        std::tolower(static_cast<unsigned char>(ch)));
  }
  if (key == "ordinary" || key == "chisq" || key == "chi_square") {
    return ScalarProfileReference::Ordinary;
  }
  if (key == "robust_scaled" || key == "satorra_scaled" ||
      key == "scaled") {
    return ScalarProfileReference::RobustScaled;
  }
  if (key == "misspec_scaled" || key == "misspecification_scaled") {
    return ScalarProfileReference::MisspecScaled;
  }
  if (key == "misspec_mixture" || key == "misspecification_mixture" ||
      key == "mixture") {
    return ScalarProfileReference::MisspecMixture;
  }
  Rcpp::stop("magmaan: %s reference must be one of ordinary, robust_scaled, "
             "misspec_scaled, or misspec_mixture", call);
}

bool scalar_reference_needs_sandwich(
    magmaan::estimate::frontier::ScalarProfileReference reference) {
  using magmaan::estimate::frontier::ScalarProfileReference;
  return reference == ScalarProfileReference::RobustScaled ||
         reference == ScalarProfileReference::MisspecScaled ||
         reference == ScalarProfileReference::MisspecMixture;
}

OmegaBlockSpec omega_spec_from_block(Rcpp::IntegerVector block,
                                     Eigen::Index p,
                                     const char* call) {
  if (p <= 0) {
    Rcpp::stop("magmaan: %s requires at least one ordinal indicator", call);
  }
  if (block.size() != p) {
    Rcpp::stop("magmaan: %s block length must equal the number of ordinal "
               "indicators", call);
  }
  std::vector<int> labels(block.begin(), block.end());
  std::vector<int> uniq = labels;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

  OmegaBlockSpec out;
  out.spec.block.resize(p);
  for (Eigen::Index i = 0; i < p; ++i) {
    const auto it = std::lower_bound(
        uniq.begin(), uniq.end(), labels[static_cast<std::size_t>(i)]);
    out.spec.block(i) = static_cast<int>(it - uniq.begin());
  }
  out.k = static_cast<int>(uniq.size());
  return out;
}

Rcpp::List scalar_profile_lrt_base_to_list(
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter,
    Rcpp::List constrained) {
  return Rcpp::List::create(
      Rcpp::_["parameter"] = parameter,
      Rcpp::_["target"] = r.target,
      Rcpp::_["unrestricted_value"] = r.unrestricted_value,
      Rcpp::_["constrained_value"] = r.constrained_value,
      Rcpp::_["constraint_residual"] = r.constraint_residual,
      Rcpp::_["fmin_unrestricted"] = r.fmin_unrestricted,
      Rcpp::_["fmin_constrained"] = r.fmin_constrained,
      Rcpp::_["T"] = r.T,
      Rcpp::_["p_value"] = r.p_value,
      Rcpp::_["scaling_factor"] = r.scaling_factor,
      Rcpp::_["T_scaled"] = r.T_scaled,
      Rcpp::_["p_value_scaled"] = r.p_value_scaled,
      Rcpp::_["misspec_scaling_factor"] = r.misspec_scaling_factor,
      Rcpp::_["T_misspec_scaled"] = r.T_misspec_scaled,
      Rcpp::_["p_value_misspec_scaled"] = r.p_value_misspec_scaled,
      Rcpp::_["misspec_eigvals"] = Rcpp::wrap(r.misspec_eigvals),
      Rcpp::_["p_value_misspec_mixture"] = r.p_value_misspec_mixture,
      Rcpp::_["misspec_mixture_cutoff"] = r.misspec_mixture_cutoff,
      Rcpp::_["df"] = r.df,
      Rcpp::_["nobs"] = r.n_obs,
      Rcpp::_["constrained"] = constrained);
}

Rcpp::List scalar_profile_lrt_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats ,
    const char* ordinal_parameterization ) {
  Rcpp::List constrained =
      ordinal_stats == nullptr
          ? fit_result(ctx, r.constrained, nullptr, estimator)
          : ordinal_fit_result(ctx, *ordinal_stats, r.constrained, nullptr,
                               estimator, ordinal_parameterization);
  return scalar_profile_lrt_base_to_list(r, parameter, constrained);
}

Rcpp::List scalar_profile_ci_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats ,
    const char* ordinal_parameterization ) {
  return Rcpp::List::create(
      Rcpp::_["parameter"] = parameter,
      Rcpp::_["estimate"] = r.estimate,
      Rcpp::_["lower"] = r.lower,
      Rcpp::_["upper"] = r.upper,
      Rcpp::_["confidence_level"] = r.confidence_level,
      Rcpp::_["cutoff"] = r.cutoff,
      Rcpp::_["lower_cutoff"] = r.lower_cutoff,
      Rcpp::_["upper_cutoff"] = r.upper_cutoff,
      Rcpp::_["lower_evals"] = r.lower_evals,
      Rcpp::_["upper_evals"] = r.upper_evals,
      Rcpp::_["lower_at_bound"] = r.lower_at_bound,
      Rcpp::_["upper_at_bound"] = r.upper_at_bound,
      Rcpp::_["lower_profile"] = scalar_profile_lrt_to_list(
          ctx, r.lower_profile, parameter, estimator, ordinal_stats,
          ordinal_parameterization),
      Rcpp::_["upper_profile"] = scalar_profile_lrt_to_list(
          ctx, r.upper_profile, parameter, estimator, ordinal_stats,
          ordinal_parameterization));
}

Rcpp::List scalar_profile_lrt_to_list_fiml(
    Ctx& ctx,
    const magmaan::data::RawData& raw,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter) {
  Rcpp::List constrained =
      fiml_profile_fit_result(ctx, raw, r.constrained, source_fit);
  return scalar_profile_lrt_base_to_list(r, parameter, constrained);
}

Rcpp::List scalar_profile_lrt_to_list_ml2s(
    Ctx& ctx,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter) {
  Rcpp::List constrained =
      ml2s_profile_fit_result(ctx, r.constrained, source_fit);
  return scalar_profile_lrt_base_to_list(r, parameter, constrained);
}

Rcpp::List scalar_profile_lrt_to_list_mixed_ordinal(
    Ctx& ctx,
    const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    int parameter,
    const char* estimator,
    const char* parameterization) {
  Rcpp::List constrained = mixed_ordinal_fit_result(
      ctx, stats, r.constrained, nullptr, estimator, parameterization);
  return scalar_profile_lrt_base_to_list(r, parameter, constrained);
}

Rcpp::List scalar_profile_ci_to_list_fiml(
    Ctx& ctx,
    const magmaan::data::RawData& raw,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter) {
  return Rcpp::List::create(
      Rcpp::_["parameter"] = parameter,
      Rcpp::_["estimate"] = r.estimate,
      Rcpp::_["lower"] = r.lower,
      Rcpp::_["upper"] = r.upper,
      Rcpp::_["confidence_level"] = r.confidence_level,
      Rcpp::_["cutoff"] = r.cutoff,
      Rcpp::_["lower_cutoff"] = r.lower_cutoff,
      Rcpp::_["upper_cutoff"] = r.upper_cutoff,
      Rcpp::_["lower_evals"] = r.lower_evals,
      Rcpp::_["upper_evals"] = r.upper_evals,
      Rcpp::_["lower_at_bound"] = r.lower_at_bound,
      Rcpp::_["upper_at_bound"] = r.upper_at_bound,
      Rcpp::_["lower_profile"] = scalar_profile_lrt_to_list_fiml(
          ctx, raw, source_fit, r.lower_profile, parameter),
      Rcpp::_["upper_profile"] = scalar_profile_lrt_to_list_fiml(
          ctx, raw, source_fit, r.upper_profile, parameter));
}

Rcpp::List scalar_profile_ci_to_list_ml2s(
    Ctx& ctx,
    Rcpp::List source_fit,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter) {
  return Rcpp::List::create(
      Rcpp::_["parameter"] = parameter,
      Rcpp::_["estimate"] = r.estimate,
      Rcpp::_["lower"] = r.lower,
      Rcpp::_["upper"] = r.upper,
      Rcpp::_["confidence_level"] = r.confidence_level,
      Rcpp::_["cutoff"] = r.cutoff,
      Rcpp::_["lower_cutoff"] = r.lower_cutoff,
      Rcpp::_["upper_cutoff"] = r.upper_cutoff,
      Rcpp::_["lower_evals"] = r.lower_evals,
      Rcpp::_["upper_evals"] = r.upper_evals,
      Rcpp::_["lower_at_bound"] = r.lower_at_bound,
      Rcpp::_["upper_at_bound"] = r.upper_at_bound,
      Rcpp::_["lower_profile"] = scalar_profile_lrt_to_list_ml2s(
          ctx, source_fit, r.lower_profile, parameter),
      Rcpp::_["upper_profile"] = scalar_profile_lrt_to_list_ml2s(
          ctx, source_fit, r.upper_profile, parameter));
}

Rcpp::List scalar_profile_ci_to_list_mixed_ordinal(
    Ctx& ctx,
    const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    int parameter,
    const char* estimator,
    const char* parameterization) {
  return Rcpp::List::create(
      Rcpp::_["parameter"] = parameter,
      Rcpp::_["estimate"] = r.estimate,
      Rcpp::_["lower"] = r.lower,
      Rcpp::_["upper"] = r.upper,
      Rcpp::_["confidence_level"] = r.confidence_level,
      Rcpp::_["cutoff"] = r.cutoff,
      Rcpp::_["lower_cutoff"] = r.lower_cutoff,
      Rcpp::_["upper_cutoff"] = r.upper_cutoff,
      Rcpp::_["lower_evals"] = r.lower_evals,
      Rcpp::_["upper_evals"] = r.upper_evals,
      Rcpp::_["lower_at_bound"] = r.lower_at_bound,
      Rcpp::_["upper_at_bound"] = r.upper_at_bound,
      Rcpp::_["lower_profile"] = scalar_profile_lrt_to_list_mixed_ordinal(
          ctx, stats, r.lower_profile, parameter, estimator, parameterization),
      Rcpp::_["upper_profile"] = scalar_profile_lrt_to_list_mixed_ordinal(
          ctx, stats, r.upper_profile, parameter, estimator, parameterization));
}

Rcpp::List scalar_functional_profile_lrt_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileLrtResult& r,
    const char* coefficient,
    const char* coefficient_target,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats ,
    const char* ordinal_parameterization ) {
  Rcpp::List constrained =
      ordinal_stats == nullptr
          ? fit_result(ctx, r.constrained, nullptr, estimator)
          : ordinal_fit_result(ctx, *ordinal_stats, r.constrained, nullptr,
                               estimator, ordinal_parameterization);
  return Rcpp::List::create(
      Rcpp::_["coefficient"] = coefficient,
      Rcpp::_["coefficient_target"] = coefficient_target,
      Rcpp::_["omega_target"] = coefficient_target,
      Rcpp::_["target"] = r.target,
      Rcpp::_["unrestricted_value"] = r.unrestricted_value,
      Rcpp::_["constrained_value"] = r.constrained_value,
      Rcpp::_["constraint_residual"] = r.constraint_residual,
      Rcpp::_["fmin_unrestricted"] = r.fmin_unrestricted,
      Rcpp::_["fmin_constrained"] = r.fmin_constrained,
      Rcpp::_["T"] = r.T,
      Rcpp::_["p_value"] = r.p_value,
      Rcpp::_["scaling_factor"] = r.scaling_factor,
      Rcpp::_["T_scaled"] = r.T_scaled,
      Rcpp::_["p_value_scaled"] = r.p_value_scaled,
      Rcpp::_["misspec_scaling_factor"] = r.misspec_scaling_factor,
      Rcpp::_["T_misspec_scaled"] = r.T_misspec_scaled,
      Rcpp::_["p_value_misspec_scaled"] = r.p_value_misspec_scaled,
      Rcpp::_["misspec_eigvals"] = Rcpp::wrap(r.misspec_eigvals),
      Rcpp::_["p_value_misspec_mixture"] = r.p_value_misspec_mixture,
      Rcpp::_["misspec_mixture_cutoff"] = r.misspec_mixture_cutoff,
      Rcpp::_["df"] = r.df,
      Rcpp::_["nobs"] = r.n_obs,
      Rcpp::_["constrained"] = constrained);
}

Rcpp::List scalar_functional_profile_ci_to_list(
    Ctx& ctx,
    const magmaan::estimate::frontier::ScalarProfileCiResult& r,
    const char* coefficient,
    const char* coefficient_target,
    const char* estimator,
    const magmaan::data::OrdinalStats* ordinal_stats ,
    const char* ordinal_parameterization ) {
  return Rcpp::List::create(
      Rcpp::_["coefficient"] = coefficient,
      Rcpp::_["coefficient_target"] = coefficient_target,
      Rcpp::_["omega_target"] = coefficient_target,
      Rcpp::_["estimate"] = r.estimate,
      Rcpp::_["lower"] = r.lower,
      Rcpp::_["upper"] = r.upper,
      Rcpp::_["confidence_level"] = r.confidence_level,
      Rcpp::_["cutoff"] = r.cutoff,
      Rcpp::_["lower_cutoff"] = r.lower_cutoff,
      Rcpp::_["upper_cutoff"] = r.upper_cutoff,
      Rcpp::_["lower_evals"] = r.lower_evals,
      Rcpp::_["upper_evals"] = r.upper_evals,
      Rcpp::_["lower_at_bound"] = r.lower_at_bound,
      Rcpp::_["upper_at_bound"] = r.upper_at_bound,
      Rcpp::_["lower_profile"] = scalar_functional_profile_lrt_to_list(
          ctx, r.lower_profile, coefficient, coefficient_target, estimator,
          ordinal_stats, ordinal_parameterization),
      Rcpp::_["upper_profile"] = scalar_functional_profile_lrt_to_list(
          ctx, r.upper_profile, coefficient, coefficient_target, estimator,
          ordinal_stats, ordinal_parameterization));
}

}  // namespace

// frontier_profile_lrt_parameter_ml() - ordinary df-1 profile-LR test for one
// complete-data ML free parameter. `parameter` is the 1-based free-parameter
// ordinal from `fit$partable$free`; the C++ core receives the 0-based theta index.
//
// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_ml_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  if (fit.containsElementNamed("estimator")) {
    const std::string estimator = Rcpp::as<std::string>(fit["estimator"]);
    if (estimator != "ML" && estimator != "ML-Fisher" &&
        estimator != "ML-Fisher-SNLLS" && estimator != "ML-IRLS" &&
        estimator != "ML-IRLS-SNLLS") {
      Rcpp::stop("frontier_profile_lrt_parameter_ml() requires a complete-data "
                 "ML fit, got estimator '%s'", estimator.c_str());
    }
  }
  if (parameter <= 0 ||
      parameter > static_cast<int>(ctx.pt.n_free())) {
    Rcpp::stop("frontier_profile_lrt_parameter_ml(): parameter index %d is "
               "outside 1..%d", parameter, static_cast<int>(ctx.pt.n_free()));
  }
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_ml()");
  std::unique_ptr<magmaan::data::RawData> raw_holder;
  if (scalar_reference_needs_sandwich(reference_mode)) {
    if (Rf_isNull(raw_data)) {
      Rcpp::stop("frontier_profile_lrt_parameter_ml() robust/misspec "
                 "reference requires raw_data; supply raw data or choose reference = 'ordinary' explicitly");
    }
    raw_holder = std::make_unique<magmaan::data::RawData>(
        complete_raw_from_arg(ctx.rep, raw_data));
  }
  auto r_or = magmaan::estimate::frontier::profile_lrt_parameter_ml(
      ctx.pt, ctx.rep, ctx.samp, est,
      static_cast<Eigen::Index>(parameter - 1), target,
      bounds_from_nullable(bounds), backend, optim_opts_from(control),
      constraint_tol, raw_holder.get(), reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list(ctx, *r_or, parameter, "ML");
}

// frontier_profile_lrt_parameter_gmm() - ordinary df-1 profile statistic for one
// continuous moment-quadratic free parameter. `parameter` is the 1-based
// free-parameter ordinal from `fit$partable$free`; the C++ core receives the
// 0-based theta index. ULS uses the identity weight, GLS rebuilds the
// normal-theory weight, and WLS-computed fits use the recorded fit$W (or the
// caller's `weight` for fits without one). With `estimated_weight`, the weight
// influence follows the fit's recorded recipe; a supplied W is refused.
//
// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_gmm_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    SEXP weight = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    bool estimated_weight = true,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (estimator != "ULS" && estimator != "GLS" && estimator != "WLS") {
    Rcpp::stop("frontier_profile_lrt_parameter_gmm() requires a continuous "
               "ULS/GLS/WLS fit, got estimator '%s'", estimator.c_str());
  }
  if (parameter <= 0 ||
      parameter > static_cast<int>(ctx.pt.n_free())) {
    Rcpp::stop("frontier_profile_lrt_parameter_gmm(): parameter index %d is "
               "outside 1..%d", parameter, static_cast<int>(ctx.pt.n_free()));
  }
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::gmm::Weight w =
      continuous_ls_weight(fit, ctx, est, estimator, weight,
                           "frontier_profile_lrt_parameter_gmm");
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_gmm()");
  std::unique_ptr<magmaan::data::RawData> raw_holder;
  magmaan::estimate::frontier::GmmProfileRobustOptions robust_opts;
  if (estimated_weight) {
    if (!scalar_reference_needs_sandwich(reference_mode)) {
      Rcpp::stop("frontier_profile_lrt_parameter_gmm() estimated_weight=TRUE "
                 "requires a robust/misspec reference");
    }
    robust_opts.estimated_weight = true;
    robust_opts.ij_weight_mode =
        continuous_ij_mode_for_fit(fit, estimator, &robust_opts.dls_opts);
  }
  if (scalar_reference_needs_sandwich(reference_mode)) {
    if (Rf_isNull(raw_data)) {
      Rcpp::stop("frontier_profile_lrt_parameter_gmm() robust/misspec "
                 "reference requires raw_data; supply raw data or choose reference = 'ordinary' explicitly");
    }
    raw_holder = std::make_unique<magmaan::data::RawData>(
        complete_raw_from_arg(ctx.rep, raw_data));
  }
  auto r_or = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      ctx.pt, ctx.rep, ctx.samp, est, std::move(w),
      static_cast<Eigen::Index>(parameter - 1), target,
      bounds_from_nullable(bounds), backend, optim_opts_from(control),
      constraint_tol, raw_holder.get(), robust_opts, reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list(ctx, *r_or, parameter, estimator.c_str());
}

// frontier_profile_lrt_parameter_ordinal() - ordinary df-1 profile statistic
// for one all-ordinal ULS/DWLS/WLS free parameter. `parameter` is the 1-based
// free-parameter ordinal from the prepared ordinal fit partable.
//
// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_ordinal_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    std::string weight = "fit",
    SEXP ordinal_stats = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!fit.containsElementNamed("ordinal") ||
      !Rcpp::as<bool>(fit["ordinal"])) {
    Rcpp::stop("frontier_profile_lrt_parameter_ordinal() requires an "
               "all-ordinal ULS/DWLS/WLS fit");
  }
  if (parameter <= 0 ||
      parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_parameter_ordinal(): parameter index %d "
               "is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(
      stats_from_fit_or_arg(fit, ordinal_stats, "ordinal_stats",
                            "frontier_profile_lrt_parameter_ordinal"));
  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "frontier_profile_lrt_parameter_ordinal");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_ordinal()");
  const bool need_sandwich = scalar_reference_needs_sandwich(reference_mode);

  auto r_or = magmaan::estimate::frontier::profile_lrt_parameter_ordinal(
      ctx.pt, ctx.rep, stats, est,
      static_cast<Eigen::Index>(parameter - 1), target,
      bounds_from_nullable(bounds), ow, backend, optim_opts_from(control),
      ordinal_parameterization_from_string(parameterization_name),
      constraint_tol, need_sandwich, reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list(
      ctx, *r_or, parameter, weight_key.c_str(), &stats,
      parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_ml_impl(
    Rcpp::List fit,
    int parameter,
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  if (parameter <= 0 ||
      parameter > static_cast<int>(ctx.pt.n_free())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml(): parameter index %d is "
               "outside 1..%d", parameter, static_cast<int>(ctx.pt.n_free()));
  }
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_ci_parameter_ml()");
  std::unique_ptr<magmaan::data::RawData> raw_holder;
  if (scalar_reference_needs_sandwich(ci_opts.reference)) {
    if (Rf_isNull(raw_data)) {
      Rcpp::stop("frontier_profile_lrt_ci_parameter_ml() robust/misspec "
                 "reference requires raw_data; supply raw data or choose reference = 'ordinary' explicitly");
    }
    raw_holder = std::make_unique<magmaan::data::RawData>(
        complete_raw_from_arg(ctx.rep, raw_data));
  }
  auto r_or = magmaan::estimate::frontier::profile_lrt_ci_parameter_ml(
      ctx.pt, ctx.rep, ctx.samp, est,
      static_cast<Eigen::Index>(parameter - 1), ci_opts,
      bounds_from_nullable(bounds), backend, optim_opts_from(control),
      constraint_tol, raw_holder.get());
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list(ctx, *r_or, parameter, "ML");
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_gmm_impl(
    Rcpp::List fit,
    int parameter,
    SEXP weight = R_NilValue,
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    bool estimated_weight = true,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (estimator != "ULS" && estimator != "GLS" && estimator != "WLS") {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_gmm() requires a continuous "
               "ULS/GLS/WLS fit, got estimator '%s'", estimator.c_str());
  }
  if (parameter <= 0 ||
      parameter > static_cast<int>(ctx.pt.n_free())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_gmm(): parameter index %d is "
               "outside 1..%d", parameter, static_cast<int>(ctx.pt.n_free()));
  }
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::gmm::Weight w =
      continuous_ls_weight(fit, ctx, est, estimator, weight,
                           "frontier_profile_lrt_ci_parameter_gmm");
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_ci_parameter_gmm()");
  std::unique_ptr<magmaan::data::RawData> raw_holder;
  magmaan::estimate::frontier::GmmProfileRobustOptions robust_opts;
  if (estimated_weight) {
    if (!scalar_reference_needs_sandwich(ci_opts.reference)) {
      Rcpp::stop("frontier_profile_lrt_ci_parameter_gmm() "
                 "estimated_weight=TRUE requires a robust/misspec reference");
    }
    robust_opts.estimated_weight = true;
    robust_opts.ij_weight_mode =
        continuous_ij_mode_for_fit(fit, estimator, &robust_opts.dls_opts);
  }
  if (scalar_reference_needs_sandwich(ci_opts.reference)) {
    if (Rf_isNull(raw_data)) {
      Rcpp::stop("frontier_profile_lrt_ci_parameter_gmm() robust/misspec "
                 "reference requires raw_data; supply raw data or choose reference = 'ordinary' explicitly");
    }
    raw_holder = std::make_unique<magmaan::data::RawData>(
        complete_raw_from_arg(ctx.rep, raw_data));
  }
  auto r_or = magmaan::estimate::frontier::profile_lrt_ci_parameter_gmm(
      ctx.pt, ctx.rep, ctx.samp, est, std::move(w),
      static_cast<Eigen::Index>(parameter - 1), ci_opts,
      bounds_from_nullable(bounds), backend, optim_opts_from(control),
      constraint_tol, raw_holder.get(), robust_opts);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list(ctx, *r_or, parameter, estimator.c_str());
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_ordinal_impl(
    Rcpp::List fit,
    int parameter,
    std::string weight = "fit",
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    SEXP ordinal_stats = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!fit.containsElementNamed("ordinal") ||
      !Rcpp::as<bool>(fit["ordinal"])) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ordinal() requires an "
               "all-ordinal ULS/DWLS/WLS fit");
  }
  if (parameter <= 0 ||
      parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ordinal(): parameter index "
               "%d is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(
      stats_from_fit_or_arg(fit, ordinal_stats, "ordinal_stats",
                            "frontier_profile_lrt_ci_parameter_ordinal"));
  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "frontier_profile_lrt_ci_parameter_ordinal");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_ci_parameter_ordinal()");
  const bool need_sandwich =
      scalar_reference_needs_sandwich(ci_opts.reference);

  auto r_or = magmaan::estimate::frontier::profile_lrt_ci_parameter_ordinal(
      ctx.pt, ctx.rep, stats, est,
      static_cast<Eigen::Index>(parameter - 1), ci_opts,
      bounds_from_nullable(bounds), ow, backend, optim_opts_from(control),
      ordinal_parameterization_from_string(parameterization_name),
      constraint_tol, need_sandwich);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list(
      ctx, *r_or, parameter, weight_key.c_str(), &stats,
      parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_fiml_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    double constraint_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  const bool is_fiml = fit.containsElementNamed("fiml") &&
                       Rcpp::as<bool>(fit["fiml"]);
  if (!is_fiml && estimator != "FIML") {
    Rcpp::stop("frontier_profile_lrt_parameter_fiml() requires a FIML fit");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_parameter_fiml(): parameter index %d is "
               "outside 1..%d", parameter, static_cast<int>(est.theta.size()));
  }
  SEXP rd = raw_data;
  if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) rd = fit["raw_data"];
  if (Rf_isNull(rd)) {
    Rcpp::stop("frontier_profile_lrt_parameter_fiml() requires raw_data or a "
               "FIML fit carrying $raw_data");
  }
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : fiml_backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_fiml()");
  auto r_or = magmaan::estimate::fiml::frontier::profile_lrt_parameter_fiml(
      ctx.pt, ctx.rep, raw, est, pack,
      static_cast<Eigen::Index>(parameter - 1), target, backend,
      optim_opts_from(control), constraint_tol, reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list_fiml(ctx, raw, fit, *r_or, parameter);
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_ml2s_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    double constraint_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue,
    bool estimated_weight = true) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!is_ml2s_estimator_label(estimator)) {
    Rcpp::stop("frontier_profile_lrt_parameter_ml2s() requires an ML2S fit");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_parameter_ml2s(): parameter index %d is "
               "outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    Rcpp::stop("frontier_profile_lrt_parameter_ml2s() requires fit$stage1");
  }
  const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
      ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  const auto dls = ml2s_dls_options_from_fit(fit);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_ml2s()");

  std::unique_ptr<magmaan::data::RawData> raw_holder;
  std::unique_ptr<FimlPack> owned_pack;
  std::unique_ptr<FimlH1> owned_h1;
  const FimlPack* pack_ptr = nullptr;
  const FimlH1* h1_ptr = nullptr;
  if (estimated_weight && scalar_reference_needs_sandwich(reference_mode) &&
      ml2s_weight_needs_raw_ij(kind)) {
    SEXP rd = raw_data;
    if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) {
      rd = fit["raw_data"];
    }
    if (Rf_isNull(rd)) {
      Rcpp::stop("frontier_profile_lrt_parameter_ml2s() estimated_weight=TRUE "
                 "requires raw_data or an ML2S fit carrying $raw_data");
    }
    raw_holder = std::make_unique<magmaan::data::RawData>(
        fiml_raw_from_arg(ctx.rep, rd));
    pack_ptr = &fiml_pack_for_fit(fit, *raw_holder, owned_pack);
    h1_ptr = &fiml_h1_for_fit(fit, *raw_holder, *pack_ptr, owned_h1);
  }

  magmaan::estimate::fiml::frontier::Ml2sProfileRobustOptions robust_opts;
  robust_opts.estimated_weight = estimated_weight;
  robust_opts.raw = raw_holder.get();
  robust_opts.pack = pack_ptr;
  robust_opts.h1 = h1_ptr;
  auto r_or = magmaan::estimate::fiml::frontier::profile_lrt_parameter_ml2s(
      ctx.pt, ctx.rep, est, sm, static_cast<Eigen::Index>(parameter - 1),
      target, kind, dls, backend, optim_opts_from(control), constraint_tol,
      reference_mode, robust_opts);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list_ml2s(ctx, fit, *r_or, parameter);
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_ml2s_nt_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    double constraint_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
      ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  if (estimator != "ML2S" || stage2_weight != "nt") {
    Rcpp::stop("frontier_profile_lrt_parameter_ml2s_nt() requires an ML2S fit "
               "with stage2_weight = 'nt'");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_parameter_ml2s_nt(): parameter index %d "
               "is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    Rcpp::stop("frontier_profile_lrt_parameter_ml2s_nt() requires fit$stage1");
  }
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_ml2s_nt()");
  auto r_or =
      magmaan::estimate::fiml::frontier::profile_lrt_parameter_ml2s_nt(
          ctx.pt, ctx.rep, est, sm,
          static_cast<Eigen::Index>(parameter - 1), target, backend,
          optim_opts_from(control), constraint_tol, reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list_ml2s(ctx, fit, *r_or, parameter);
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_parameter_mixed_ordinal_impl(
    Rcpp::List fit,
    int parameter,
    double target,
    std::string weight = "fit",
    SEXP mixed_ordinal_stats = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!fit.containsElementNamed("mixed_ordinal") ||
      !Rcpp::as<bool>(fit["mixed_ordinal"])) {
    Rcpp::stop("frontier_profile_lrt_parameter_mixed_ordinal() requires a "
               "mixed-ordinal ULS/DWLS/WLS fit");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_parameter_mixed_ordinal(): parameter "
               "index %d is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(
      stats_from_fit_or_arg(fit, mixed_ordinal_stats, "mixed_ordinal_stats",
                            "frontier_profile_lrt_parameter_mixed_ordinal"));
  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "frontier_profile_lrt_parameter_mixed_ordinal");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_parameter_mixed_ordinal()");
  const bool need_sandwich = scalar_reference_needs_sandwich(reference_mode);
  auto r_or = magmaan::estimate::frontier::profile_lrt_parameter_mixed_ordinal(
      ctx.pt, ctx.rep, stats, est,
      static_cast<Eigen::Index>(parameter - 1), target,
      bounds_from_nullable(bounds), ow, backend, optim_opts_from(control),
      ordinal_parameterization_from_string(parameterization_name),
      constraint_tol, need_sandwich, reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_lrt_to_list_mixed_ordinal(
      ctx, stats, *r_or, parameter, weight_key.c_str(),
      parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_fiml_impl(
    Rcpp::List fit,
    int parameter,
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const bool is_fiml = fit.containsElementNamed("fiml") &&
                       Rcpp::as<bool>(fit["fiml"]);
  if (!is_fiml) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_fiml() requires a FIML fit");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_fiml(): parameter index %d "
               "is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  SEXP rd = raw_data;
  if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) rd = fit["raw_data"];
  if (Rf_isNull(rd)) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_fiml() requires raw_data or "
               "a FIML fit carrying $raw_data");
  }
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : fiml_backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_ci_parameter_fiml()");
  auto r_or =
      magmaan::estimate::fiml::frontier::profile_lrt_ci_parameter_fiml(
          ctx.pt, ctx.rep, raw, est, pack,
          static_cast<Eigen::Index>(parameter - 1), ci_opts, backend,
          optim_opts_from(control), constraint_tol);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list_fiml(ctx, raw, fit, *r_or, parameter);
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_ml2s_impl(
    Rcpp::List fit,
    int parameter,
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    SEXP raw_data = R_NilValue,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue,
    bool estimated_weight = true) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!is_ml2s_estimator_label(estimator)) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s() requires an ML2S fit");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s(): parameter index %d "
               "is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s() requires fit$stage1");
  }
  const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
      ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  const auto dls = ml2s_dls_options_from_fit(fit);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_ci_parameter_ml2s()");

  std::unique_ptr<magmaan::data::RawData> raw_holder;
  std::unique_ptr<FimlPack> owned_pack;
  std::unique_ptr<FimlH1> owned_h1;
  const FimlPack* pack_ptr = nullptr;
  const FimlH1* h1_ptr = nullptr;
  if (estimated_weight && scalar_reference_needs_sandwich(ci_opts.reference) &&
      ml2s_weight_needs_raw_ij(kind)) {
    SEXP rd = raw_data;
    if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) {
      rd = fit["raw_data"];
    }
    if (Rf_isNull(rd)) {
      Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s() "
                 "estimated_weight=TRUE requires raw_data or an ML2S fit "
                 "carrying $raw_data");
    }
    raw_holder = std::make_unique<magmaan::data::RawData>(
        fiml_raw_from_arg(ctx.rep, rd));
    pack_ptr = &fiml_pack_for_fit(fit, *raw_holder, owned_pack);
    h1_ptr = &fiml_h1_for_fit(fit, *raw_holder, *pack_ptr, owned_h1);
  }

  magmaan::estimate::fiml::frontier::Ml2sProfileRobustOptions robust_opts;
  robust_opts.estimated_weight = estimated_weight;
  robust_opts.raw = raw_holder.get();
  robust_opts.pack = pack_ptr;
  robust_opts.h1 = h1_ptr;
  auto r_or =
      magmaan::estimate::fiml::frontier::profile_lrt_ci_parameter_ml2s(
          ctx.pt, ctx.rep, est, sm, static_cast<Eigen::Index>(parameter - 1),
          ci_opts, kind, dls, backend, optim_opts_from(control),
          constraint_tol, robust_opts);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list_ml2s(ctx, fit, *r_or, parameter);
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_ml2s_nt_impl(
    Rcpp::List fit,
    int parameter,
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
      ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  if (estimator != "ML2S" || stage2_weight != "nt") {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s_nt() requires an ML2S "
               "fit with stage2_weight = 'nt'");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s_nt(): parameter index "
               "%d is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_ml2s_nt() requires fit$stage1");
  }
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust, "frontier_profile_lrt_ci_parameter_ml2s_nt()");
  auto r_or =
      magmaan::estimate::fiml::frontier::profile_lrt_ci_parameter_ml2s_nt(
          ctx.pt, ctx.rep, est, sm,
          static_cast<Eigen::Index>(parameter - 1), ci_opts, backend,
          optim_opts_from(control), constraint_tol);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list_ml2s(ctx, fit, *r_or, parameter);
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_parameter_mixed_ordinal_impl(
    Rcpp::List fit,
    int parameter,
    std::string weight = "fit",
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    SEXP mixed_ordinal_stats = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!fit.containsElementNamed("mixed_ordinal") ||
      !Rcpp::as<bool>(fit["mixed_ordinal"])) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_mixed_ordinal() requires a "
               "mixed-ordinal ULS/DWLS/WLS fit");
  }
  if (parameter <= 0 || parameter > static_cast<int>(est.theta.size())) {
    Rcpp::stop("frontier_profile_lrt_ci_parameter_mixed_ordinal(): parameter "
               "index %d is outside 1..%d", parameter,
               static_cast<int>(est.theta.size()));
  }
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(
      stats_from_fit_or_arg(fit, mixed_ordinal_stats, "mixed_ordinal_stats",
                            "frontier_profile_lrt_ci_parameter_mixed_ordinal"));
  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "frontier_profile_lrt_ci_parameter_mixed_ordinal");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust,
      "frontier_profile_lrt_ci_parameter_mixed_ordinal()");
  const bool need_sandwich =
      scalar_reference_needs_sandwich(ci_opts.reference);
  auto r_or =
      magmaan::estimate::frontier::profile_lrt_ci_parameter_mixed_ordinal(
          ctx.pt, ctx.rep, stats, est,
          static_cast<Eigen::Index>(parameter - 1), ci_opts,
          bounds_from_nullable(bounds), ow, backend, optim_opts_from(control),
          ordinal_parameterization_from_string(parameterization_name),
          constraint_tol, need_sandwich);
  if (!r_or.has_value()) stop_fit(r_or.error());
  return scalar_profile_ci_to_list_mixed_ordinal(
      ctx, stats, *r_or, parameter, weight_key.c_str(),
      parameterization_name.c_str());
}

// frontier_profile_lrt_ordinal_polychoric_omega() - ordinary df-1 profile
// statistic for the model-implied latent-response/polychoric omega functional.
//
// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ordinal_polychoric_omega_impl(
    Rcpp::List fit,
    Rcpp::IntegerVector block,
    double omega0,
    std::string target = "total",
    std::string weight = "fit",
    SEXP ordinal_stats = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!fit.containsElementNamed("ordinal") ||
      !Rcpp::as<bool>(fit["ordinal"])) {
    Rcpp::stop("frontier_profile_lrt_ordinal_polychoric_omega() requires an "
               "all-ordinal ULS/DWLS/WLS fit");
  }

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(
      stats_from_fit_or_arg(fit, ordinal_stats, "ordinal_stats",
                            "frontier_profile_lrt_ordinal_polychoric_omega"));
  if (stats.R.size() != 1) {
    Rcpp::stop("frontier_profile_lrt_ordinal_polychoric_omega() currently "
               "supports single-group all-ordinal fits only");
  }
  const OmegaBlockSpec omega_spec = omega_spec_from_block(
      block, stats.R[0].rows(),
      "frontier_profile_lrt_ordinal_polychoric_omega()");
  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "frontier_profile_lrt_ordinal_polychoric_omega");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  const auto reference_mode = scalar_reference_from_nullable(
      reference, robust,
      "frontier_profile_lrt_ordinal_polychoric_omega()");
  const bool need_sandwich = scalar_reference_needs_sandwich(reference_mode);

  auto r_or =
      magmaan::estimate::frontier::profile_lrt_ordinal_polychoric_omega(
          ctx.pt, ctx.rep, stats, est, omega_spec.spec,
          omega_target_from_string(
              target, "frontier_profile_lrt_ordinal_polychoric_omega"),
          omega0, bounds_from_nullable(bounds), ow, backend,
          optim_opts_from(control),
          ordinal_parameterization_from_string(parameterization_name),
          constraint_tol, need_sandwich, reference_mode);
  if (!r_or.has_value()) stop_fit(r_or.error());

  Rcpp::List out = scalar_functional_profile_lrt_to_list(
      ctx, *r_or, "ordinal_polychoric_omega", target.c_str(),
      weight_key.c_str(), &stats, parameterization_name.c_str());
  out["block"] = Rcpp::clone(block);
  out["k"] = omega_spec.k;
  out["weight"] = weight_key;
  out["parameterization"] = parameterization_name;
  return out;
}

// [[Rcpp::export]]
Rcpp::List frontier_profile_lrt_ci_ordinal_polychoric_omega_impl(
    Rcpp::List fit,
    Rcpp::IntegerVector block,
    std::string target = "total",
    std::string weight = "fit",
    double level = 0.95,
    double lower = NA_REAL,
    double upper = NA_REAL,
    double initial_step = NA_REAL,
    SEXP ordinal_stats = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
    Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
    double constraint_tol = 1e-6,
    double root_tol = 1e-5,
    double statistic_tol = 1e-6,
    bool robust = false,
    Rcpp::Nullable<Rcpp::String> reference = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "";
  if (!fit.containsElementNamed("ordinal") ||
      !Rcpp::as<bool>(fit["ordinal"])) {
    Rcpp::stop("frontier_profile_lrt_ci_ordinal_polychoric_omega() requires "
               "an all-ordinal ULS/DWLS/WLS fit");
  }

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(
      stats_from_fit_or_arg(
          fit, ordinal_stats, "ordinal_stats",
          "frontier_profile_lrt_ci_ordinal_polychoric_omega"));
  if (stats.R.size() != 1) {
    Rcpp::stop("frontier_profile_lrt_ci_ordinal_polychoric_omega() currently "
               "supports single-group all-ordinal fits only");
  }
  const OmegaBlockSpec omega_spec = omega_spec_from_block(
      block, stats.R[0].rows(),
      "frontier_profile_lrt_ci_ordinal_polychoric_omega()");
  const std::string weight_key =
      ordinal_weight_key_from_arg(weight, fit, estimator);
  const auto ow = ordinal_weight_from_estimator(
      weight_key, "frontier_profile_lrt_ci_ordinal_polychoric_omega");
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : ordinal_parameterization_attr(fit["partable"]);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                         : backend_from_optimizer_arg(optimizer);
  auto ci_opts = profile_ci_options_from_args(
      level, lower, upper, initial_step, root_tol, statistic_tol);
  ci_opts.reference = scalar_reference_from_nullable(
      reference, robust,
      "frontier_profile_lrt_ci_ordinal_polychoric_omega()");
  const bool need_sandwich =
      scalar_reference_needs_sandwich(ci_opts.reference);

  auto r_or =
      magmaan::estimate::frontier::profile_lrt_ci_ordinal_polychoric_omega(
          ctx.pt, ctx.rep, stats, est, omega_spec.spec,
          omega_target_from_string(
              target, "frontier_profile_lrt_ci_ordinal_polychoric_omega"),
          ci_opts, bounds_from_nullable(bounds), ow, backend,
          optim_opts_from(control),
          ordinal_parameterization_from_string(parameterization_name),
          constraint_tol, need_sandwich);
  if (!r_or.has_value()) stop_fit(r_or.error());

  Rcpp::List out = scalar_functional_profile_ci_to_list(
      ctx, *r_or, "ordinal_polychoric_omega", target.c_str(),
      weight_key.c_str(), &stats, parameterization_name.c_str());
  out["block"] = Rcpp::clone(block);
  out["k"] = omega_spec.k;
  out["weight"] = weight_key;
  out["parameterization"] = parameterization_name;
  return out;
}
