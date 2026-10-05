#include "magmaan/api/conventions.hpp"

#include <cmath>
#include <algorithm>
#include "magmaan/robust/restriction.hpp"

#include "magmaan/inference/inference.hpp"
#include "magmaan/estimate/nl_constraints.hpp"

namespace magmaan::api {

std::string_view convention_name(LavaanConvention c) noexcept {
  switch (c) {
    case LavaanConvention::ML: return "ML";
    case LavaanConvention::MLM: return "MLM";
    case LavaanConvention::MLR: return "MLR";
    case LavaanConvention::DWLS: return "DWLS";
    case LavaanConvention::WLSMV: return "WLSMV";
    case LavaanConvention::WLSM: return "WLSM";
    case LavaanConvention::ULS: return "ULS";
    case LavaanConvention::ULSMV: return "ULSMV";
    case LavaanConvention::WLS: return "WLS";
  }
  return "unknown";
}

ConventionInference convention_unavailable(LavaanConvention c, InferenceReason reason,
    std::string detail, const PolicyFitState& state) {
  ConventionInference out;
  out.convention = convention_name(c);
  out.covariance_reason = out.test.reason = reason;
  out.covariance_detail = out.test.detail = std::move(detail);
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  return out;
}

namespace {
bool ml_convention(LavaanConvention c) {
  return c == LavaanConvention::ML || c == LavaanConvention::MLM || c == LavaanConvention::MLR;
}
ConventionTest unavailable(InferenceReason reason, std::string detail) {
  ConventionTest out;
  out.reason = reason;
  out.detail = std::move(detail);
  return out;
}
void finish(ConventionTest& t) {
  if (!std::isfinite(t.statistic) || !std::isfinite(t.scale) || t.scale <= 0) {
    t = unavailable(InferenceReason::NumericFailure, "the compatibility test has a non-finite statistic or non-positive scaling factor");
    return;
  }
  t.p_value = inference::chi2_pvalue(t.statistic, t.df);
}
}  // namespace

ConventionInference lavaan_inference_ml(robust::frontier::NTMLFit& fit,
    LavaanConvention c, const PolicyFitState& state) {
  if (!ml_convention(c))
    return convention_unavailable(c, InferenceReason::UnsupportedModel,
        "this convention requires a different fitted estimator", state);
  if (state.penalized)
    return convention_unavailable(c, InferenceReason::Penalized, std::string(penalized_detail), state);
  if (!state.converged)
    return convention_unavailable(c, InferenceReason::NotConverged, "the fit did not pass its convergence verdict", state);
  ConventionInference out;
  out.convention = convention_name(c);
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  if (fit.estimates.theta.size() == 0) {
    out.covariance.resize(0, 0);
  } else {
    auto v = c == LavaanConvention::MLR
        ? robust::frontier::ntml_score_sandwich(fit, robust::Information::Observed)
        : robust::frontier::ntml_covariance(fit, c == LavaanConvention::MLM);
    if (v) out.covariance = **v;
    else {
      out.covariance_reason = InferenceReason::NumericFailure;
      out.covariance_detail = v.error().detail;
    }
  }
  auto sample = fit.data->sample;
  if (!fit.data->has_means) sample.mean.clear();
  auto df = inference::df_stat(fit.pt, sample, fit.estimates.theta);
  if (!df) {
    out.test = unavailable(InferenceReason::NumericFailure, df.error().detail);
    return out;
  }
  auto& t = out.test;
  t.df = *df;
  t.unscaled_statistic = t.statistic = inference::chi2_stat(sample, fit.estimates);
  t.method = "standard";
  if (c != LavaanConvention::ML && t.df <= 0) {
    t.reason = InferenceReason::Saturated;
    t.detail = "the model has no positive degrees of freedom for a scaled test";
    t.statistic = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  // Unlike the policy, lavaan reports the zero-df statistic as well.
  if (c != LavaanConvention::ML && t.df > 0) {
    t.method = c == LavaanConvention::MLM ? "satorra.bentler" : "yuan.bentler.mplus";
    if (c == LavaanConvention::MLM) {
      auto q = robust::frontier::ntml_quadratic(fit, false);
      if (!q) { t = unavailable(InferenceReason::NumericFailure, q.error().detail); return out; }
      auto spectrum = robust::frontier::ntml_spectrum(**q);
      if (!spectrum) { t = unavailable(InferenceReason::NumericFailure, spectrum.error().detail); return out; }
      t.scale = (*spectrum)->sum() / t.df;
    } else {
      // MLR uses tr(A1^-1 B1) - tr(A0^-1 B0), with saturated H1 at
      // sample moments and exact H0 likelihood scores. The existing
      // pattern-likelihood primitive reduces to that formula on complete data.
      // H1 is supplied directly: no EM or model refit runs here.
      estimate::fiml::FIMLH1 h1;
      h1.mu = fit.data->sample.mean;
      h1.sigma = fit.data->sample.S;
      auto result = estimate::fiml::fiml_robust_mlr(fit.pt, fit.rep, fit.data->raw,
          fit.estimates, t.df, t.unscaled_statistic, fit.data->pack, h1);
      if (!result) { t = unavailable(InferenceReason::NumericFailure, result.error().detail); return out; }
      t.scale = result->scaling_factor;
    }
    t.statistic /= t.scale;
  }
  finish(t);
  return out;
}

// lavaan 0.7.2 missing="ml", fixed.x=FALSE: ML uses standard SEs,
// observed information / Hessian (h1.information="structured"). MLR uses
// robust.huber.white SEs and yuan.bentler.mplus. That test overrides H1 to
// unstructured EM moments: c = [tr(A1^-1 B1) - tr(A0^-1 B0)] / df.
ConventionInference lavaan_inference_fiml(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, const estimate::Estimates& estimates,
    LavaanConvention c, const PolicyFitState& state) {
  if (c != LavaanConvention::ML && c != LavaanConvention::MLR)
    return convention_unavailable(c, InferenceReason::Inapplicable,
        "FIML compatibility covers ML and MLR", state);
  if (state.penalized)
    return convention_unavailable(c, InferenceReason::Penalized, std::string(penalized_detail), state);
  if (!state.converged)
    return convention_unavailable(c, InferenceReason::NotConverged,
        "the fit did not pass its convergence verdict", state);
  if (std::any_of(pt.exo.begin(), pt.exo.end(), [](auto x) { return x != 0; }) ||
      pt.has_inequality_constraints || !pt.nonlinear_eq_rows.empty())
    return convention_unavailable(c, InferenceReason::UnsupportedModel,
        "FIML compatibility requires random x and affine equality constraints", state);
  ConventionInference out;
  out.convention = convention_name(c);
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  auto fail = [&](const std::string& detail) {
    return convention_unavailable(c, InferenceReason::NumericFailure, detail, state);
  };
  auto df = inference::df_stat(pt, pack.start_stats, estimates.theta);
  if (!df) return fail(df.error().detail);
  auto h1 = estimate::fiml::fiml_h1_moments(raw, pack);
  if (!h1) return fail(h1.error().detail);
  auto extras = estimate::fiml::fiml_extras(pt, rep, raw, estimates, pack, *h1);
  if (!extras) return fail(extras.error().detail);
  auto& t = out.test;
  t.df = *df;
  t.method = c == LavaanConvention::ML ? "standard" : "yuan.bentler.mplus";
  t.unscaled_statistic = t.statistic = extras->chi2;
  if (c == LavaanConvention::ML) {
    auto info = estimate::fiml::fiml_observed_information(pt, rep, raw, estimates, pack);
    if (!info) return fail(info.error().detail);
    auto covariance = inference::vcov(*info, pt, estimates.theta);
    if (!covariance) return fail(covariance.error().detail);
    out.covariance = *covariance;
  } else {
    auto robust = estimate::fiml::fiml_robust_mlr(pt, rep, raw, estimates,
        *df, extras->chi2, pack, *h1);
    if (!robust) return fail(robust.error().detail);
    out.covariance = robust->vcov;
    if (*df <= 0) {
      t.reason = InferenceReason::Saturated;
      t.detail = "the model has no positive degrees of freedom for a scaled test";
      t.statistic = std::numeric_limits<double>::quiet_NaN();
      return out;
    }
    t.scale = robust->scaling_factor;
    t.statistic = robust->chisq_scaled;
  }
  finish(t);
  return out;
}

ConventionTest lavaan_nested_fiml(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep, const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, LavaanConvention c) {
  if (null_state.penalized || alternative_state.penalized)
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!null_state.converged || !alternative_state.converged)
    return unavailable(InferenceReason::NotConverged, "a fit did not pass its convergence verdict");
  auto con0 = estimate::build_eq_constraints(null_pt);
  auto con1 = estimate::build_eq_constraints(alternative_pt);
  if (!con0 || !con1) return unavailable(InferenceReason::UnsupportedModel,
      !con0 ? con0.error().detail : con1.error().detail);
  auto embedding = robust::embed_nested_null(alternative_pt, alternative_rep, null_pt,
      null_rep, null_estimates.theta, *con1, *con0, true, &alternative_estimates.theta);
  if (!embedding) return unavailable(
      embedding.error().kind == PostError::Kind::NotNested ? InferenceReason::NotNested :
      embedding.error().kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting :
      embedding.error().kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting :
      InferenceReason::NumericFailure, embedding.error().detail);
  auto a = lavaan_inference_fiml(null_pt, null_rep, raw, pack, null_estimates, c, null_state);
  auto b = lavaan_inference_fiml(alternative_pt, alternative_rep, raw, pack, alternative_estimates, c, alternative_state);
  if (a.test.reason != InferenceReason::Available) return a.test;
  if (b.test.reason != InferenceReason::Available &&
      !(b.test.reason == InferenceReason::Saturated && b.test.df == 0)) return b.test;
  ConventionTest t;
  t.df = a.test.df - b.test.df;
  if (t.df <= 0 || embedding->restriction.A.rows() != t.df)
    return unavailable(InferenceReason::NotNested, "the comparison needs a positive difference in degrees of freedom");
  t.statistic = t.unscaled_statistic = a.test.unscaled_statistic - b.test.unscaled_statistic;
  t.method = "standard";
  if (t.statistic < -1e-8 * std::max(1.0, a.test.unscaled_statistic))
    return unavailable(InferenceReason::NotConverged, "the alternative fits worse than the null");
  if (c == LavaanConvention::MLR) {
    // lavTestLRT defaults to SB2001, using the YB-Mplus single-model scales.
    // Its saturated alternative contributes zero to df1*c1.
    auto result = robust::lr_test_satorra_bentler2001(a.test.unscaled_statistic,
        b.test.unscaled_statistic, a.test.df, b.test.df, a.test.scale, b.test.scale);
    if (!result) return unavailable(InferenceReason::NumericFailure, result.error().detail);
    t.method = "satorra.bentler.2001";
    t.statistic = result->T_scaled;
    t.scale = result->scale_c;
  }
  finish(t);
  return t;
}

ConventionInference lavaan_inference_ordinal(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const estimate::Estimates& estimates, estimate::OrdinalWeightKind weight,
    estimate::OrdinalParameterization parameterization,
    LavaanConvention c, const PolicyFitState& state) {
  using estimate::OrdinalWeightKind;
  const bool compatible =
      (weight == OrdinalWeightKind::DWLS && (c == LavaanConvention::DWLS || c == LavaanConvention::WLSMV)) ||
      (weight == OrdinalWeightKind::ULS && (c == LavaanConvention::ULS || c == LavaanConvention::ULSMV)) ||
      (weight == OrdinalWeightKind::WLS && c == LavaanConvention::WLS);
  if (!compatible)
    return convention_unavailable(c, InferenceReason::UnsupportedModel, "this convention requires a different fitted estimator", state);
  if (state.penalized)
    return convention_unavailable(c, InferenceReason::Penalized, std::string(penalized_detail), state);
  if (!state.converged)
    return convention_unavailable(c, InferenceReason::NotConverged, "the fit did not pass its convergence verdict", state);
  // lavaan reports ordinal inference with n_g - 1 per group. Keep the fit,
  // moments and weights intact and evaluate their quadratic criterion at the
  // retained theta under these reporting counts (including unequal groups).
  auto reporting_stats = stats;
  for (auto& n : reporting_stats.n_obs) {
    if (n <= 1) return convention_unavailable(c, InferenceReason::UnsupportedModel,
        "ordinal compatibility inference requires at least two observations per group", state);
    --n;
  }
  auto reporting_estimates = estimates;
  auto objective = estimate::frontier::ordinal_ls_objective(pt, rep, reporting_stats,
      estimates, weight, parameterization);
  if (!objective) return convention_unavailable(c, InferenceReason::NumericFailure, objective.error().detail, state);
  auto residual = objective->problem.r(estimates.theta);
  if (!residual) return convention_unavailable(c, InferenceReason::NumericFailure, residual.error().detail, state);
  reporting_estimates.fmin = 0.5 * residual->squaredNorm();
  auto result = estimate::robust_ordinal(std::move(pt), rep, reporting_stats,
      reporting_estimates, weight, parameterization);
  if (!result) return convention_unavailable(c, InferenceReason::NumericFailure, result.error().detail, state);
  ConventionInference out;
  out.convention = convention_name(c);
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  out.covariance = result->vcov;
  auto& t = out.test;
  t.df = result->df;
  t.unscaled_statistic = t.statistic = result->chisq_standard;
  t.method = "standard";
  if ((c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) && t.df <= 0) {
    t.reason = InferenceReason::Saturated;
    t.detail = "the model has no positive degrees of freedom for a scaled test";
    t.statistic = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  if ((c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) && t.df > 0) {
    t.method = "scaled.shifted";
    t.statistic = result->scaled_shifted.chi2_adj;
    t.scale = 1.0 / result->scaled_shifted.scale_a;
    t.shift = result->scaled_shifted.shift_b;
  }
  finish(t);
  // An unscaled DWLS/ULS objective has no standard chi-square reference;
  // lavaan retains its statistic and leaves the p-value unavailable.
  if (c == LavaanConvention::DWLS || c == LavaanConvention::ULS)
    t.p_value = std::numeric_limits<double>::quiet_NaN();
  return out;
}

ConventionTest lavaan_nested_ordinal(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep, const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::OrdinalStats& stats,
    estimate::OrdinalWeightKind weight, estimate::OrdinalParameterization parameterization,
    LavaanConvention c, const std::vector<std::int8_t>* null_row_user,
    const std::vector<std::int8_t>* alternative_row_user) {
  if (null_state.penalized || alternative_state.penalized)
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!null_state.converged || !alternative_state.converged)
    return unavailable(InferenceReason::NotConverged, "a fit did not pass its convergence verdict");
  auto reason_from = [](const PostError& error) {
    return error.kind == PostError::Kind::NotNested ? InferenceReason::NotNested :
        error.kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting :
        error.kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting :
        InferenceReason::NumericFailure;
  };
  // Establish actual nesting before computing the delta restriction map:
  // a Jacobian rank difference alone does not establish nested models.
  if (auto p = estimate::prepare_ordinal_partable(alternative_pt, stats, parameterization,
          nullptr, alternative_row_user); !p)
    return unavailable(InferenceReason::NumericFailure, p.error().detail);
  if (auto p = estimate::prepare_ordinal_partable(null_pt, stats, parameterization,
          nullptr, null_row_user); !p)
    return unavailable(InferenceReason::NumericFailure, p.error().detail);
  auto c1 = estimate::build_eq_constraints(alternative_pt, true);
  auto c0 = estimate::build_eq_tangent(null_pt, null_estimates.theta);
  if (!c1 || !c0) return unavailable(InferenceReason::NumericFailure,
      !c1 ? c1.error().detail : c0.error().detail);
  auto embedding = robust::embed_nested_null(alternative_pt, alternative_rep,
      null_pt, null_rep, null_estimates.theta, *c1, *c0, false,
      &alternative_estimates.theta);
  if (!embedding) return unavailable(reason_from(embedding.error()), embedding.error().detail);
  if (!alternative_pt.nonlinear_eq_rows.empty()) {
    // Curved tangents at two distinct estimates need not be nested. Check
    // containment and the shared nonlinear restrictions at the embedded null.
    const auto nl=estimate::build_nl_constraints(alternative_pt);
    const Eigen::VectorXd h=nl.h(embedding->theta);
    if (!h.allFinite() || h.lpNorm<Eigen::Infinity>()>1e-6)
      return unavailable(InferenceReason::NotNested, "the null violates an alternative nonlinear equality");
    c1=estimate::build_eq_tangent(alternative_pt,embedding->theta);
    if (!c1) return unavailable(InferenceReason::NumericFailure,c1.error().detail);
    embedding=robust::embed_nested_null(alternative_pt,alternative_rep,
        null_pt,null_rep,null_estimates.theta,*c1,*c0,false,&alternative_estimates.theta);
    if (!embedding) return unavailable(reason_from(embedding.error()),embedding.error().detail);
  }
  if (embedding->restriction.A.rows() == 0)
    return unavailable(InferenceReason::NotNested, "the models impose the same restrictions");
  auto a = lavaan_inference_ordinal(null_pt, null_rep, stats, null_estimates,
      weight, parameterization, c, null_state);
  auto b = lavaan_inference_ordinal(alternative_pt, alternative_rep, stats,
      alternative_estimates, weight, parameterization, c, alternative_state);
  if (a.test.reason != InferenceReason::Available) return a.test;
  if (b.test.reason != InferenceReason::Available &&
      !(b.test.reason == InferenceReason::Saturated && b.test.df == 0)) return b.test;
  ConventionTest t;
  t.df = a.test.df - b.test.df;
  t.statistic = t.unscaled_statistic = a.test.unscaled_statistic - b.test.unscaled_statistic;
  t.method = "standard";
  if (t.df <= 0) return unavailable(InferenceReason::NotNested,
      "the comparison needs a positive difference in degrees of freedom");
  if (t.statistic < -1e-8 * std::max(1.0, a.test.unscaled_statistic))
    return unavailable(InferenceReason::NotConverged, "the alternative fits worse than the null");
  if (c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) {
    // lavaan 0.7.2 lavTestLRT defaults: satorra.2000, A.method=delta,
    // scaled.shifted=TRUE, H1 information/Jacobian. The standard objective
    // above uses n_g-1; the sandwich uses original n_g/N group fractions.
    auto result = estimate::lr_test_satorra2000_ordinal(alternative_pt,
        alternative_rep, stats, alternative_estimates, null_pt, null_rep,
        null_estimates, weight, a.test.unscaled_statistic, b.test.unscaled_statistic,
        a.test.df, b.test.df, robust::SatorraAMethod::Delta, parameterization,
        alternative_row_user, null_row_user);
    if (!result) return unavailable(reason_from(result.error()), result.error().detail);
    t.method = "satorra.2000";
    t.statistic = result->scaled_shifted.chi2_adj;
    t.scale = 1.0 / result->scaled_shifted.scale_a;
    t.shift = result->scaled_shifted.shift_b;
  }
  finish(t);
  if (c == LavaanConvention::DWLS || c == LavaanConvention::ULS)
    t.p_value = std::numeric_limits<double>::quiet_NaN();
  return t;
}

ConventionInference lavaan_inference_mixed_ordinal(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::MixedOrdinalStats& stats,
    const estimate::Estimates& estimates, estimate::OrdinalWeightKind weight,
    estimate::OrdinalParameterization parameterization,
    LavaanConvention c, const PolicyFitState& state) {
  using estimate::OrdinalWeightKind;
  const bool compatible = weight == OrdinalWeightKind::DWLS && c == LavaanConvention::WLSMV;
  for (const auto& raw : stats.raw_data)
    if (!raw.allFinite()) return convention_unavailable(c, InferenceReason::UnsupportedModel,
        "mixed WLSMV compatibility requires complete observations", state);
  if (!compatible)
    return convention_unavailable(c, InferenceReason::UnsupportedModel, "this convention requires a different fitted estimator", state);
  if (state.penalized)
    return convention_unavailable(c, InferenceReason::Penalized, std::string(penalized_detail), state);
  if (!state.converged)
    return convention_unavailable(c, InferenceReason::NotConverged, "the fit did not pass its convergence verdict", state);
  // lavaan reports ordinal inference with n_g - 1 per group. Keep the fit,
  // moments and weights intact and evaluate their quadratic criterion at the
  // retained theta under these reporting counts (including unequal groups).
  auto reporting_stats = stats;
  for (auto& n : reporting_stats.n_obs) {
    if (n <= 1) return convention_unavailable(c, InferenceReason::UnsupportedModel,
        "ordinal compatibility inference requires at least two observations per group", state);
    --n;
  }
  auto reporting_estimates = estimates;
  auto objective = estimate::frontier::mixed_ordinal_ls_objective(pt, rep, reporting_stats,
      estimates, weight, parameterization);
  if (!objective) return convention_unavailable(c, InferenceReason::NumericFailure, objective.error().detail, state);
  auto residual = objective->problem.r(estimates.theta);
  if (!residual) return convention_unavailable(c, InferenceReason::NumericFailure, residual.error().detail, state);
  double n = 0, reporting_n = 0;
  for (const auto count : stats.n_obs) n += static_cast<double>(count);
  for (const auto count : reporting_stats.n_obs) reporting_n += static_cast<double>(count);
  reporting_estimates.fmin = 0.5 * residual->squaredNorm() * reporting_n / n;
  auto result = estimate::robust_mixed_ordinal(std::move(pt), rep, stats,
      reporting_estimates, weight, parameterization);
  if (!result) return convention_unavailable(c, InferenceReason::NumericFailure, result.error().detail, state);
  ConventionInference out;
  out.convention = convention_name(c);
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  // lavaan retains n_g/N geometry, with total reporting denominator N-G.
  out.covariance = result->vcov * n / reporting_n;
  auto& t = out.test;
  t.df = result->df;
  t.unscaled_statistic = t.statistic = result->chisq_standard;
  t.method = "standard";
  if ((c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) && t.df <= 0) {
    t.reason = InferenceReason::Saturated;
    t.detail = "the model has no positive degrees of freedom for a scaled test";
    t.statistic = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  if ((c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) && t.df > 0) {
    t.method = "scaled.shifted";
    t.statistic = result->scaled_shifted.chi2_adj;
    t.scale = 1.0 / result->scaled_shifted.scale_a;
    t.shift = result->scaled_shifted.shift_b;
  }
  finish(t);
  // An unscaled DWLS/ULS objective has no standard chi-square reference;
  // lavaan retains its statistic and leaves the p-value unavailable.
  if (c == LavaanConvention::DWLS || c == LavaanConvention::ULS)
    t.p_value = std::numeric_limits<double>::quiet_NaN();
  return out;
}

ConventionTest lavaan_nested_mixed_ordinal(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep, const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::MixedOrdinalStats& stats,
    estimate::OrdinalWeightKind weight, estimate::OrdinalParameterization parameterization,
    LavaanConvention c, const std::vector<std::int8_t>* null_row_user,
    const std::vector<std::int8_t>* alternative_row_user) {
  if (null_state.penalized || alternative_state.penalized)
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!null_state.converged || !alternative_state.converged)
    return unavailable(InferenceReason::NotConverged, "a fit did not pass its convergence verdict");
  auto reason_from = [](const PostError& error) {
    return error.kind == PostError::Kind::NotNested ? InferenceReason::NotNested :
        error.kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting :
        error.kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting :
        InferenceReason::NumericFailure;
  };
  // Establish actual nesting before computing the delta restriction map:
  // a Jacobian rank difference alone does not establish nested models.
  if (auto p = estimate::prepare_mixed_ordinal_partable(alternative_pt, stats, parameterization,
          nullptr, alternative_row_user); !p)
    return unavailable(InferenceReason::NumericFailure, p.error().detail);
  if (auto p = estimate::prepare_mixed_ordinal_partable(null_pt, stats, parameterization,
          nullptr, null_row_user); !p)
    return unavailable(InferenceReason::NumericFailure, p.error().detail);
  auto c1 = estimate::build_eq_constraints(alternative_pt, true);
  auto c0 = estimate::build_eq_tangent(null_pt, null_estimates.theta);
  if (!c1 || !c0) return unavailable(InferenceReason::NumericFailure,
      !c1 ? c1.error().detail : c0.error().detail);
  auto embedding = robust::embed_nested_null(alternative_pt, alternative_rep,
      null_pt, null_rep, null_estimates.theta, *c1, *c0, false,
      &alternative_estimates.theta);
  if (!embedding) return unavailable(reason_from(embedding.error()), embedding.error().detail);
  if (!alternative_pt.nonlinear_eq_rows.empty()) {
    // Curved tangents at two distinct estimates need not be nested. Check
    // containment and the shared nonlinear restrictions at the embedded null.
    const auto nl=estimate::build_nl_constraints(alternative_pt);
    const Eigen::VectorXd h=nl.h(embedding->theta);
    if (!h.allFinite() || h.lpNorm<Eigen::Infinity>()>1e-6)
      return unavailable(InferenceReason::NotNested, "the null violates an alternative nonlinear equality");
    c1=estimate::build_eq_tangent(alternative_pt,embedding->theta);
    if (!c1) return unavailable(InferenceReason::NumericFailure,c1.error().detail);
    embedding=robust::embed_nested_null(alternative_pt,alternative_rep,
        null_pt,null_rep,null_estimates.theta,*c1,*c0,false,&alternative_estimates.theta);
    if (!embedding) return unavailable(reason_from(embedding.error()),embedding.error().detail);
  }
  if (embedding->restriction.A.rows() == 0)
    return unavailable(InferenceReason::NotNested, "the models impose the same restrictions");
  auto a = lavaan_inference_mixed_ordinal(null_pt, null_rep, stats, null_estimates,
      weight, parameterization, c, null_state);
  auto b = lavaan_inference_mixed_ordinal(alternative_pt, alternative_rep, stats,
      alternative_estimates, weight, parameterization, c, alternative_state);
  if (a.test.reason != InferenceReason::Available) return a.test;
  if (b.test.reason != InferenceReason::Available &&
      !(b.test.reason == InferenceReason::Saturated && b.test.df == 0)) return b.test;
  ConventionTest t;
  t.df = a.test.df - b.test.df;
  t.statistic = t.unscaled_statistic = a.test.unscaled_statistic - b.test.unscaled_statistic;
  t.method = "standard";
  if (t.df <= 0) return unavailable(InferenceReason::NotNested,
      "the comparison needs a positive difference in degrees of freedom");
  if (t.statistic < -1e-8 * std::max(1.0, a.test.unscaled_statistic))
    return unavailable(InferenceReason::NotConverged, "the alternative fits worse than the null");
  if (c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) {
    // lavaan 0.7.2 lavTestLRT defaults: satorra.2000, A.method=delta,
    // scaled.shifted=TRUE, H1 information/Jacobian. The standard objective
    // above uses n_g-1; the sandwich uses original n_g/N group fractions.
    auto result = estimate::lr_test_satorra2000_mixed_ordinal(alternative_pt,
        alternative_rep, stats, alternative_estimates, null_pt, null_rep,
        null_estimates, weight, a.test.unscaled_statistic, b.test.unscaled_statistic,
        a.test.df, b.test.df, robust::SatorraAMethod::Delta, parameterization,
        alternative_row_user, null_row_user);
    if (!result) return unavailable(reason_from(result.error()), result.error().detail);
    t.method = "satorra.2000";
    t.statistic = result->scaled_shifted.chi2_adj;
    t.scale = 1.0 / result->scaled_shifted.scale_a;
    t.shift = result->scaled_shifted.shift_b;
  }
  finish(t);
  if (c == LavaanConvention::DWLS || c == LavaanConvention::ULS)
    t.p_value = std::numeric_limits<double>::quiet_NaN();
  return t;
}

ConventionTest lavaan_nested_ml(std::shared_ptr<robust::frontier::NTMLFit> null,
    const PolicyFitState& null_state,
    std::shared_ptr<robust::frontier::NTMLFit> alternative,
    const PolicyFitState& alternative_state, LavaanConvention c) {
  if (!ml_convention(c)) return unavailable(InferenceReason::UnsupportedModel, "nested conventions cover complete-data ML so far");
  if (null_state.penalized || alternative_state.penalized)
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!null_state.converged || !alternative_state.converged)
    return unavailable(InferenceReason::NotConverged, "a fit did not pass its convergence verdict");
  auto h = robust::frontier::prepare_ntml_hypothesis(null, alternative);
  if (!h) {
    const auto kind = h.error().kind;
    return unavailable(kind == PostError::Kind::NotNested ? InferenceReason::NotNested :
        kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting :
        kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting :
        InferenceReason::NumericFailure, h.error().detail);
  }
  auto a = lavaan_inference_ml(*null, c, null_state);
  auto b = lavaan_inference_ml(*alternative, c, alternative_state);
  if (a.test.reason != InferenceReason::Available) return a.test;
  // The alternative's scale drops out of SB2001 when df_H1 = 0.
  // Its unscaled statistic is retained even though no scaled global test exists.
  if (b.test.reason != InferenceReason::Available &&
      !(b.test.reason == InferenceReason::Saturated && b.test.df == 0)) return b.test;
  ConventionTest t;
  t.df = a.test.df - b.test.df;
  t.statistic = t.unscaled_statistic = a.test.unscaled_statistic - b.test.unscaled_statistic;
  t.method = "standard";
  if (t.df <= 0) return unavailable(InferenceReason::NotNested, "the comparison needs a positive difference in degrees of freedom");
  if (t.statistic < -1e-8 * std::max(1.0, a.test.unscaled_statistic))
    return unavailable(InferenceReason::NotConverged, "the alternative fits worse than the null");
  if (c != LavaanConvention::ML) {
    t.method = "satorra.bentler.2001";
    auto result = robust::lr_test_satorra_bentler2001(a.test.unscaled_statistic,
        b.test.unscaled_statistic, a.test.df, b.test.df, a.test.scale, b.test.scale);
    if (!result) return unavailable(InferenceReason::NumericFailure, result.error().detail);
    t.statistic = result->T_scaled;
    t.scale = result->scale_c;
  }
  finish(t);
  return t;
}

}  // namespace magmaan::api
