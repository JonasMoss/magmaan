#include "magmaan/api/policy.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Cholesky>
#include "magmaan/inference/score.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"

#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/weighted_inference.hpp"

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
  const auto peba = robust::frontier::fmg_test(out.statistic, out.df, out.eigenvalues,
                                              {FmgMethod::Peba, 4.0, true});
  out.p_peba4 = peba.p_value;
  out.peba_blocks = peba.blocks_effective;
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

PolicyNested policy_nested_dwls(spec::LatentStructure null_pt,
                                const model::MatrixRep& null_rep,
                                const estimate::Estimates& null_estimates,
                                const PolicyFitState& null_state,
                                spec::LatentStructure alternative_pt,
                                const model::MatrixRep& alternative_rep,
                                const estimate::Estimates& alternative_estimates,
                                const PolicyFitState& alternative_state,
                                const data::OrdinalStats& stats,
                                estimate::OrdinalParameterization parameterization,
                                const std::vector<std::int8_t>* null_row_user,
                                const std::vector<std::int8_t>* alternative_row_user) {
  PolicyNested out;
  out.psd_boundary = null_state.psd_boundary || alternative_state.psd_boundary;
  out.verdict_disagreement =
      verdict_disagreement(null_state) || verdict_disagreement(alternative_state);
  auto unavailable = [&](InferenceReason reason, const std::string& detail) {
    set_unavailable(out.score, reason, detail);
    set_unavailable(out.lr, reason, detail);
    return out;
  };
  if (null_state.penalized || alternative_state.penalized)
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!null_state.converged || !alternative_state.converged)
    return unavailable(InferenceReason::NotConverged, "a fit did not pass its convergence verdict");
  if (null_estimates.association || alternative_estimates.association)
    return unavailable(InferenceReason::UnsupportedModel, "ordinal association ML is not a DWLS fit");

  // Nesting: lift the null into the alternative's parameter space on the
  // prepared (threshold- and scale-augmented) structures.
  {
    spec::LatentStructure p1 = alternative_pt, p0 = null_pt;
    auto prepared1 = estimate::prepare_ordinal_delta_partable(p1, stats, nullptr, alternative_row_user);
    auto prepared0 = estimate::prepare_ordinal_delta_partable(p0, stats, nullptr, null_row_user);
    if (!prepared1 || !prepared0)
      return unavailable(InferenceReason::NumericFailure,
                         !prepared1 ? prepared1.error().detail : prepared0.error().detail);
    auto c1 = estimate::build_eq_constraints(p1);
    auto c0 = estimate::build_eq_constraints(p0);
    if (!c1 || !c0)
      return unavailable(InferenceReason::NumericFailure,
                         !c1 ? c1.error().detail : c0.error().detail);
    auto embedding = robust::embed_nested_null(p1, alternative_rep, p0, null_rep,
        null_estimates.theta, *c1, *c0, false, &alternative_estimates.theta);
    if (!embedding) {
      const auto kind = embedding.error().kind;
      return unavailable(kind == PostError::Kind::NotNested ? InferenceReason::NotNested
          : kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting
          : kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting
          : InferenceReason::NumericFailure, embedding.error().detail);
    }
    if (embedding->restriction.A.rows() == 0)
      return unavailable(InferenceReason::NotNested, "the models impose the same restrictions");
    alternative_pt = std::move(p1);
    null_pt = std::move(p0);
  }

  set_unavailable(out.score, InferenceReason::UnsupportedModel,
      "no nested DWLS score test is derived; the fit-function difference test is reported");
  auto profile = estimate::ordinal_dwls_profile_lrt(std::move(alternative_pt), alternative_rep,
      stats, alternative_estimates, std::move(null_pt), null_rep, null_estimates,
      parameterization);
  if (!profile) {
    set_unavailable(out.lr, reason_from(profile.error()), profile.error().detail);
    return out;
  }
  const auto& p = *profile;
  if (p.df_diff <= 0) {
    set_unavailable(out.lr, InferenceReason::NotNested,
                    "the null does not restrict the alternative");
    return out;
  }
  if (!std::isfinite(p.T_diff) || !p.eigvals.allFinite()) {
    set_unavailable(out.lr, InferenceReason::NumericFailure,
                    "DWLS nested test: non-finite statistic or spectrum");
    return out;
  }
  // Eigenvalues that are zero to working precision (they appear, for
  // instance, from the extra scale directions of the theta parameterization)
  // are zeros of the reference law, not terms of it; keeping them would make
  // PEBA4 depend on the parameterization.
  const double largest = p.eigvals.size() ? p.eigvals.maxCoeff() : 0.0;
  std::vector<double> kept;
  for (Eigen::Index i = 0; i < p.eigvals.size(); ++i)
    if (p.eigvals(i) > 1e-8 * largest) kept.push_back(p.eigvals(i));
  const Eigen::Index k = std::max<Eigen::Index>(p.df_diff, static_cast<Eigen::Index>(kept.size()));
  Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(k);
  for (std::size_t i = 0; i < kept.size(); ++i)
    eigenvalues(k - static_cast<Eigen::Index>(kept.size()) + static_cast<Eigen::Index>(i)) = kept[i];
  std::sort(eigenvalues.data(), eigenvalues.data() + k);
  const double trace = eigenvalues.sum();
  if (!(trace > 0.0)) {
    set_unavailable(out.lr, InferenceReason::NumericFailure,
                    "DWLS nested test: the reference spectrum has no positive mass");
    return out;
  }
  // A clearly negative difference means the alternative stopped above the
  // null's optimum.
  if (p.T_diff < -1e-8 * std::max(1.0, trace)) {
    set_unavailable(out.lr, InferenceReason::NotConverged,
                    "the alternative fits worse than the null (fit-function difference " +
                    std::to_string(p.T_diff) + ")");
    return out;
  }
  PolicyTest& t = out.lr;
  t.statistic = std::max(0.0, p.T_diff);
  t.df = p.df_diff;
  t.eigenvalues = std::move(eigenvalues);
  t.sb_scale = trace / static_cast<double>(t.df);
  t.p_sb = inference::chi2_pvalue(t.statistic / t.sb_scale, t.df);
  const auto peba = robust::frontier::fmg_test(t.statistic, static_cast<int>(k), t.eigenvalues,
      {robust::frontier::FmgMethod::Peba, 4.0, true});
  t.p_peba4 = peba.p_value;
  t.peba_blocks = peba.blocks_effective;
  t.label = "fit_function_difference";
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

namespace {
InferenceReason nested_reason(const PostError& error) {
  switch (error.kind) {
    case PostError::Kind::NotNested: return InferenceReason::NotNested;
    case PostError::Kind::UnsupportedNesting: return InferenceReason::UnsupportedNesting;
    case PostError::Kind::BoundaryNesting: return InferenceReason::BoundaryNesting;
    default: return reason_from(error);
  }
}

void fiml_score(const post_expected<inference::frontier::ScoreComponents>& components,
                PolicyTest& out) {
  if (!components) {
    set_unavailable(out, nested_reason(components.error()), components.error().detail);
    return;
  }
  // Joint observational-unit sampling: retain raw score cross-products.
  // Missingness patterns are not fixed sampling strata.
  auto projected = inference::frontier::project_scores(*components);
  if (!projected) {
    set_unavailable(out, reason_from(projected.error()), projected.error().detail);
    return;
  }
  auto spectrum = inference::frontier::score_spectrum(*projected);
  if (!spectrum) {
    set_unavailable(out, reason_from(spectrum.error()), spectrum.error().detail);
    return;
  }
  out.statistic = projected->statistic;
  out.df = static_cast<int>(spectrum->size());
  out.eigenvalues = *spectrum;
  calibrate_spectrum(out);
}
}

PolicyInference policy_inference_fiml(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, const estimate::Estimates& estimates,
    const PolicyFitState& state) {
  using namespace estimate::fiml;
  if (state.penalized || !state.converged) {
    auto out = policy_unavailable(state.penalized ? InferenceReason::Penalized
        : InferenceReason::NotConverged, state.penalized ? std::string(penalized_detail)
        : "the fit did not pass its convergence verdict");
    out.verdict_disagreement = verdict_disagreement(state);
    return out;
  }
  PolicyInference out;
  out.psd_boundary = state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(state);
  if (auto resolved = estimate::resolve_fixed_x_from_sample(pt, rep, pack.start_stats);
      !resolved) return policy_unavailable(InferenceReason::UnsupportedModel, resolved.error().detail);
  if (std::any_of(pt.exo.begin(), pt.exo.end(), [](auto x) { return x != 0; }) ||
      pt.has_inequality_constraints || !pt.nonlinear_eq_rows.empty()) {
    return policy_unavailable(InferenceReason::UnsupportedModel,
        "FIML policy requires random x and affine equality constraints");
  }
  auto con = estimate::build_eq_constraints(pt);
  if (!con) return policy_unavailable(reason_from(con.error()), con.error().detail);
  const auto& K = con->K();
  if (K.cols() == 0) out.covariance = Eigen::MatrixXd::Zero(pt.n_free(), pt.n_free());
  else {
    auto mb = fiml_score_meat_bread(pt, rep, raw, pack, estimates);
    if (!mb) {
      out.covariance_reason = reason_from(mb.error());
      out.covariance_detail = mb.error().detail;
    } else {
      Eigen::MatrixXd A = K.transpose() * mb->hessian * K;
      Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (A + A.transpose()));
      if (llt.info() != Eigen::Success) {
        out.covariance_reason = InferenceReason::NumericFailure;
        out.covariance_detail = "FIML observed bread is not positive definite";
      } else {
        const Eigen::MatrixXd inverse = llt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
        const Eigen::MatrixXd rows = mb->scores * K;
        // H is averaged deviance, scores are casewise deviance gradients.
        const double N = static_cast<double>(pack.cache.n_total);
        out.covariance = K * inverse * (rows.transpose() * rows) * inverse * K.transpose() / (N * N);
      }
    }
  }
  auto df = inference::df_stat(pt, pack.start_stats, estimates.theta);
  if (!df) {
    set_unavailable(out.score, reason_from(df.error()), df.error().detail);
    out.lr = out.score;
    return out;
  }
  if (*df <= 0) {
    set_unavailable(out.score, InferenceReason::Saturated,
        "the model has zero degrees of freedom, so there is no global test");
    out.lr = out.score;
    return out;
  }
  using namespace inference::frontier;
  fiml_score(global_score_components(pt, rep, raw, pack, estimates,
      {ScoreSensitivity::ObservedInformation, ScoreMetric::ExpectedInformation}), out.score);
  auto h1 = fiml_h1_moments(raw, pack);
  if (!h1) {
    set_unavailable(out.lr, InferenceReason::NumericFailure, h1.error().detail);
    return out;
  }
  auto extras = fiml_extras(pt, rep, raw, estimates, pack, *h1);
  if (!extras) {
    set_unavailable(out.lr, reason_from(extras.error()), extras.error().detail);
    return out;
  }
  auto spectrum = fiml_ugamma_spectrum(pt, rep, raw, estimates, *df, extras->chi2, pack, *h1);
  if (!spectrum) set_unavailable(out.lr, reason_from(spectrum.error()), spectrum.error().detail);
  else {
    out.lr.statistic = extras->chi2;
    out.lr.df = *df;
    out.lr.eigenvalues = spectrum->eigvals;
    calibrate_spectrum(out.lr);
  }
  return out;
}

PolicyNested policy_nested_fiml(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep,
    const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack) {
  PolicyNested out;
  out.psd_boundary = null_state.psd_boundary || alternative_state.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(null_state) || verdict_disagreement(alternative_state);
  auto unavailable = [&](InferenceReason reason, const std::string& detail) {
    set_unavailable(out.score, reason, detail);
    out.lr = out.score;
    return out;
  };
  if (null_state.penalized || alternative_state.penalized)
    return unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!null_state.converged || !alternative_state.converged)
    return unavailable(InferenceReason::NotConverged, "a fit did not pass its convergence verdict");
  for (auto* pt : {&null_pt, &alternative_pt}) {
    if (std::any_of(pt->exo.begin(), pt->exo.end(), [](auto x) { return x != 0; }) ||
        pt->has_inequality_constraints || !pt->nonlinear_eq_rows.empty())
      return unavailable(InferenceReason::UnsupportedModel,
          "FIML policy requires random x and affine equality constraints");
  }
  auto resolved0 = estimate::resolve_fixed_x_from_sample(null_pt, null_rep, pack.start_stats);
  auto resolved1 = estimate::resolve_fixed_x_from_sample(alternative_pt, alternative_rep, pack.start_stats);
  if (!resolved0 || !resolved1) return unavailable(InferenceReason::UnsupportedModel,
      !resolved0 ? resolved0.error().detail : resolved1.error().detail);
  auto con0 = estimate::build_eq_constraints(null_pt);
  auto con1 = estimate::build_eq_constraints(alternative_pt);
  if (!con0 || !con1) return unavailable(InferenceReason::NumericFailure,
      !con0 ? con0.error().detail : con1.error().detail);
  auto embedding = robust::embed_nested_null(alternative_pt, alternative_rep, null_pt,
      null_rep, null_estimates.theta, *con1, *con0, true, &alternative_estimates.theta);
  if (!embedding) return unavailable(nested_reason(embedding.error()), embedding.error().detail);
  if (embedding->restriction.A.rows() == 0)
    return unavailable(InferenceReason::NotNested, "model pair has no released restrictions");
  using namespace inference::frontier;
  fiml_score(nested_score_components(alternative_pt, alternative_rep, null_pt, null_rep,
      nullptr, raw, &pack, null_estimates, ScoreSensitivity::ObservedInformation), out.score);
  auto mb = estimate::fiml::fiml_score_meat_bread(alternative_pt, alternative_rep,
      raw, pack, alternative_estimates);
  if (!mb) {
    set_unavailable(out.lr, reason_from(mb.error()), mb.error().detail);
    return out;
  }
  const auto& K = con1->K();
  const double N = static_cast<double>(pack.cache.n_total);
  const Eigen::MatrixXd A = (N / 2.0) * K.transpose() * mb->hessian * K;
  const Eigen::MatrixXd rows = -0.5 * mb->scores * K;
  const Eigen::MatrixXd B = rows.transpose() * rows;
  auto spectrum = robust::compute_satorra2000_from_sandwich(A, B, embedding->restriction.A);
  if (!spectrum) {
    set_unavailable(out.lr, reason_from(spectrum.error()), spectrum.error().detail);
    return out;
  }
  out.lr.statistic = 2.0 * N * (null_estimates.fmin - alternative_estimates.fmin);
  out.lr.df = static_cast<int>(embedding->restriction.A.rows());
  out.lr.eigenvalues = spectrum->eigenvalues;
  if (out.lr.statistic < -1e-8 * std::max(1.0, std::abs(2.0 * N * null_estimates.fmin)))
    set_unavailable(out.lr, InferenceReason::NotConverged, "the alternative fits worse than the null");
  else calibrate_spectrum(out.lr);
  return out;
}

}  // namespace magmaan::api
