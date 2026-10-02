#include "magmaan/api/policy.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/robust/frontier/fmg.hpp"

namespace magmaan::api {

std::string_view reason_name(InferenceReason reason) noexcept {
  switch (reason) {
    case InferenceReason::Available:        return "available";
    case InferenceReason::NotConverged:     return "not_converged";
    case InferenceReason::Saturated:        return "saturated";
    case InferenceReason::UnsupportedModel: return "unsupported_model";
    case InferenceReason::NumericFailure:   return "numeric_failure";
    case InferenceReason::NotNested:        return "not_nested";
    case InferenceReason::UnsupportedNesting: return "unsupported_nesting";
    case InferenceReason::BoundaryNesting:  return "boundary_nesting";
    case InferenceReason::Penalized:        return "penalized";
    case InferenceReason::Inapplicable:     return "inapplicable";
  }
  return "unknown";
}

PolicyFitState policy_fit_state(const estimate::Estimates& estimates) {
  const auto& diagnostics = estimates.diagnostics;
  const auto verdict = estimate::fit_verdict(estimates);
  PolicyFitState state;
  state.converged = verdict.status != estimate::FitCheck::Failed;
  state.psd_boundary = verdict.domain == estimate::StationarityDomain::Psd &&
                       diagnostics.newton_accuracy.checked &&
                       !diagnostics.newton_accuracy.covariance_interior;
  if (estimates.selected_verdict) {
    const auto native = estimate::common_fit_verdict(diagnostics).status;
    if (native != estimate::FitCheck::Unchecked)
      state.native_converged = native == estimate::FitCheck::Passed;
  }
  return state;
}

PolicyFitState policy_fit_state(const estimate::frontier::PenalizedFit& fit) {
  auto state = policy_fit_state(fit.estimates);
  state.penalized = fit.weight > 0.0;
  return state;
}

bool verdict_disagreement(const PolicyFitState& state) noexcept {
  return state.native_converged && *state.native_converged != state.converged;
}

PolicyInference policy_unavailable(InferenceReason reason, std::string detail) {
  PolicyInference out;
  out.covariance_reason = reason;
  out.covariance_detail = detail;
  out.score.reason = out.lr.reason = reason;
  out.score.detail = out.lr.detail = std::move(detail);
  return out;
}

namespace {

// SB and PEBA4 for a statistic with its df-length ascending spectrum.
void calibrate_spectrum(PolicyTest& out) {
  using robust::frontier::FmgMethod;
  const auto sb = robust::frontier::fmg_test(out.statistic, out.df, out.eigenvalues,
                                             {FmgMethod::SatorraBentler, 0.0, true});
  out.p_sb = sb.p_value;
  out.sb_scale = sb.lambdas.sum() / static_cast<double>(out.df);
  out.p_peba4 = robust::frontier::fmg_test(out.statistic, out.df, out.eigenvalues,
                                           {FmgMethod::Peba, 4.0, true}).p_value;
}

InferenceReason reason_from(const PostError& error) {
  return error.kind == PostError::Kind::UnsupportedInference
      ? InferenceReason::UnsupportedModel : InferenceReason::NumericFailure;
}

void calibrate(const post_expected<std::shared_ptr<robust::frontier::NTMLQuadratic>>& q,
               PolicyTest& out) {
  if (!q) {
    out.reason = InferenceReason::NumericFailure;
    out.detail = q.error().detail;
    return;
  }
  auto spectrum = robust::frontier::ntml_spectrum(**q);
  if (!spectrum) {
    out.reason = InferenceReason::NumericFailure;
    out.detail = spectrum.error().detail;
    return;
  }
  out.statistic = (*q)->statistic;
  out.df = (*q)->df;
  out.eigenvalues = **spectrum;
  calibrate_spectrum(out);
}

void set_unavailable(PolicyTest& test, InferenceReason reason, const std::string& detail) {
  test = PolicyTest{};
  test.reason = reason;
  test.detail = detail;
}

}  // namespace

PolicyInference policy_inference_ml(robust::frontier::NTMLFit& fit,
                                    const PolicyFitState& state) {
  if (state.penalized) {
    auto out = policy_unavailable(InferenceReason::Penalized, std::string(penalized_detail));
    out.verdict_disagreement = verdict_disagreement(state);
    return out;
  }
  if (!state.converged) {
    auto out = policy_unavailable(InferenceReason::NotConverged,
                                  "the fit did not pass its convergence verdict");
    out.verdict_disagreement = verdict_disagreement(state);
    return out;
  }
  PolicyInference out;
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  if (fit.estimates.theta.size() == 0) {
    // Fixed parameters have no uncertainty; all saturated-moment directions
    // still contribute to the global tests below.
    out.covariance.resize(0, 0);
  } else {
    auto covariance = robust::frontier::ntml_score_sandwich(fit, robust::Information::Observed);
    if (covariance) {
      out.covariance = **covariance;
    } else {
      out.covariance_reason = InferenceReason::NumericFailure;
      out.covariance_detail = covariance.error().detail;
    }
  }

  // df_stat counts mean moments whenever the sample carries means; the NTML
  // sample always does, so drop them for a covariance-only fit.
  data::SampleStats sample = fit.data->sample;
  if (!fit.data->has_means) sample.mean.clear();
  auto df = inference::df_stat(fit.pt, sample, fit.estimates.theta);
  if (!df) {
    out.score.reason = out.lr.reason = InferenceReason::NumericFailure;
    out.score.detail = out.lr.detail = df.error().detail;
    return out;
  }
  if (*df <= 0) {
    out.score.reason = out.lr.reason = InferenceReason::Saturated;
    out.score.detail = out.lr.detail =
        "the model has zero degrees of freedom, so there is no global test";
    return out;
  }
  calibrate(robust::frontier::ntml_quadratic(fit, true), out.score);
  calibrate(robust::frontier::ntml_quadratic(fit, false), out.lr);
  return out;
}

PolicyInference policy_inference_dwls(spec::LatentStructure pt,
                                      const model::MatrixRep& rep,
                                      const data::OrdinalStats& stats,
                                      const estimate::Estimates& estimates,
                                      estimate::OrdinalParameterization parameterization,
                                      const PolicyFitState& state,
                                      const std::vector<std::int8_t>* row_user) {
  PolicyInference out;
  if (state.penalized) {
    out = policy_unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  } else if (!state.converged) {
    out = policy_unavailable(InferenceReason::NotConverged,
                             "the fit did not pass its convergence verdict");
  } else if (estimates.association) {
    out = policy_unavailable(InferenceReason::UnsupportedModel,
        "ordinal association ML is not a DWLS fit");
  }
  if (state.penalized || !state.converged || estimates.association) {
    out.verdict_disagreement = verdict_disagreement(state);
    return out;
  }
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  using estimate::OrdinalWeightKind;
  auto ij = estimate::robust_ordinal_ij(pt, rep, stats, estimates, OrdinalWeightKind::DWLS,
                                        parameterization, row_user);
  if (ij) {
    out.covariance = ij->vcov;
  } else {
    out.covariance_reason = reason_from(ij.error());
    out.covariance_detail = ij.error().detail;
  }
  out.lr.reason = InferenceReason::Inapplicable;
  out.lr.detail = "DWLS has no likelihood; its global test is the fit-function "
                  "statistic, reported as the score test, which it equals";
  auto fixed = estimate::robust_ordinal(std::move(pt), rep, stats, estimates,
                                        OrdinalWeightKind::DWLS, parameterization,
                                        robust::Information::Expected, row_user);
  if (!fixed) {
    set_unavailable(out.score, reason_from(fixed.error()), fixed.error().detail);
    return out;
  }
  if (fixed->df <= 0) {
    set_unavailable(out.score, InferenceReason::Saturated,
                    "the model has zero degrees of freedom, so there is no global test");
    return out;
  }
  if (fixed->eigvals.size() > fixed->df || !fixed->eigvals.allFinite()) {
    set_unavailable(out.score, InferenceReason::NumericFailure,
                    "DWLS global test: invalid UGamma spectrum");
    return out;
  }
  out.score.statistic = fixed->chisq_standard;
  out.score.df = fixed->df;
  out.score.eigenvalues = Eigen::VectorXd::Zero(fixed->df);
  out.score.eigenvalues.tail(fixed->eigvals.size()) = fixed->eigvals;
  std::sort(out.score.eigenvalues.data(), out.score.eigenvalues.data() + out.score.df);
  out.score.label = "fit_function";
  calibrate_spectrum(out.score);
  return out;
}

PolicyNested policy_nested_ml(std::shared_ptr<robust::frontier::NTMLFit> null,
                              const PolicyFitState& null_state,
                              std::shared_ptr<robust::frontier::NTMLFit> alternative,
                              const PolicyFitState& alternative_state) {
  PolicyNested out;
  out.psd_boundary = null_state.psd_boundary || alternative_state.psd_boundary;
  out.verdict_disagreement =
      verdict_disagreement(null_state) || verdict_disagreement(alternative_state);
  auto unavailable = [&](InferenceReason reason, const std::string& detail) {
    set_unavailable(out.score, reason, detail);
    set_unavailable(out.lr, reason, detail);
    return out;
  };
  if (null_state.penalized || alternative_state.penalized) {
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  }
  if (!null_state.converged || !alternative_state.converged) {
    return unavailable(InferenceReason::NotConverged,
                       "a fit did not pass its convergence verdict");
  }
  auto hypothesis = robust::frontier::prepare_ntml_hypothesis(std::move(null),
                                                              std::move(alternative));
  if (!hypothesis) {
    const auto kind=hypothesis.error().kind;
    const auto reason=kind==PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting
        : kind==PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting
        : kind==PostError::Kind::NotNested ? InferenceReason::NotNested
        : InferenceReason::NumericFailure;
    return unavailable(reason,hypothesis.error().detail);
  }
  // Observed geometry: the nested comparison usually has a misspecified
  // larger model, where expected information gives an inconsistent reference
  // law for both statistics.
  using robust::Information;
  calibrate(robust::frontier::ntml_quadratic(**hypothesis, true, Information::Observed),
            out.score);
  calibrate(robust::frontier::ntml_quadratic(**hypothesis, false, Information::Observed),
            out.lr);
  // A negative difference means the alternative stopped above the null's
  // optimum, so at least one fit is not at its minimum. The score statistic
  // needs only the null fit and stays.
  const auto& h = **hypothesis;
  const double scale = inference::chi2_stat(h.null_fit->data->sample, h.null_fit->estimates);
  if (out.lr.reason == InferenceReason::Available &&
      out.lr.statistic < -1e-8 * std::max(1.0, scale)) {
    set_unavailable(out.lr, InferenceReason::NotConverged,
                    "the alternative fits worse than the null (likelihood-ratio "
                    "statistic " + std::to_string(out.lr.statistic) + ")");
  }
  return out;
}

}  // namespace magmaan::api
