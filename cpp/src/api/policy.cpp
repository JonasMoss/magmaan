#include "magmaan/api/policy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <type_traits>

#include <Eigen/Cholesky>
#include "magmaan/inference/score.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"

#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/weighted_inference.hpp"

namespace magmaan::api {

struct FimlPolicyFit::Impl {
  const spec::LatentStructure pt;
  const model::MatrixRep rep;
  const data::RawData raw;
  const estimate::fiml::FIMLPack pack;
  const estimate::Estimates estimates;
  std::optional<post_expected<estimate::fiml::FIMLScoreMeatBread>> bread;
  std::optional<PolicyInference> inference;
  std::size_t builds = 0;
};
struct DwlsPolicyFit::Impl {
  const spec::LatentStructure pt;
  const model::MatrixRep rep;
  const data::OrdinalStats stats;
  const estimate::Estimates estimates;
  const estimate::OrdinalParameterization parameterization;
  const std::vector<std::int8_t> row_user;
  std::optional<post_expected<estimate::OrdinalRobustResult>> ij;
  std::optional<fit_expected<estimate::frontier::OrdinalNewtonParts>> parts;
  std::optional<PolicyInference> inference;
  std::size_t builds = 0;
};
struct MixedDwlsPolicyFit::Impl {
  const spec::LatentStructure pt;
  const model::MatrixRep rep;
  const data::MixedOrdinalStats stats;
  std::optional<std::vector<Eigen::MatrixXd>> sampling;
  const estimate::Estimates estimates;
  const estimate::OrdinalParameterization parameterization;
  const std::vector<std::int8_t> row_user;
  std::optional<post_expected<estimate::OrdinalRobustResult>> ij;
  std::optional<fit_expected<estimate::frontier::OrdinalNewtonParts>> parts;
  std::optional<PolicyInference> inference;
  std::size_t builds = 0;
};
FimlPolicyFit::FimlPolicyFit(spec::LatentStructure pt, model::MatrixRep rep,
    data::RawData raw, estimate::fiml::FIMLPack pack, estimate::Estimates estimates)
    : impl(std::make_shared<Impl>(Impl{std::move(pt), std::move(rep),
        std::move(raw), std::move(pack), std::move(estimates), {}, {}, 0})) {}
DwlsPolicyFit::DwlsPolicyFit(spec::LatentStructure pt, model::MatrixRep rep,
    data::OrdinalStats stats, estimate::Estimates estimates,
    estimate::OrdinalParameterization parameterization, std::vector<std::int8_t> row_user)
    : impl(std::make_shared<Impl>(Impl{std::move(pt), std::move(rep),
        std::move(stats), std::move(estimates), parameterization,
        std::move(row_user), {}, {}, {}, 0})) {}
MixedDwlsPolicyFit::MixedDwlsPolicyFit(spec::LatentStructure pt, model::MatrixRep rep,
    data::MixedOrdinalStats stats, estimate::Estimates estimates,
    estimate::OrdinalParameterization parameterization, std::vector<std::int8_t> row_user)
    : impl(std::make_shared<Impl>(Impl{std::move(pt), std::move(rep),
        std::move(stats), {}, std::move(estimates), parameterization,
        std::move(row_user), {}, {}, {}, 0})) {}
std::size_t policy_ingredient_builds(const FimlPolicyFit& fit) { return fit.impl->builds; }
std::size_t policy_ingredient_builds(const DwlsPolicyFit& fit) { return fit.impl->builds; }
std::size_t policy_ingredient_builds(const MixedDwlsPolicyFit& fit) { return fit.impl->builds; }



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
    case InferenceReason::EquivalentModels: return "equivalent_models";
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

static post_expected<data::MixedOrdinalStats> mixed_policy_stats(
    const data::MixedOrdinalStats& input, MixedDwlsPolicyFit::Impl* cache) {
  auto stats = input;
  const auto blocks = stats.R.size();
  if (stats.moments.size() != blocks || stats.mean.size() != blocks ||
      stats.ordered.size() != blocks || stats.n_levels.size() != blocks ||
      stats.thresholds.size() != blocks || stats.n_obs.size() != blocks ||
      stats.NACOV.size() != blocks)
    return std::unexpected(PostError{PostError::Kind::NumericIssue,
        "mixed DWLS policy: inconsistent first-stage block layout"});
  if (stats.W_dwls.size() != blocks)
    return std::unexpected(PostError{PostError::Kind::NumericIssue,
        "mixed DWLS policy: fitting weights unavailable"});
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto m = stats.moments[b].size();
    const auto& gamma = stats.NACOV[b];
    const auto& weight = stats.W_dwls[b];
    if (gamma.rows() != m || gamma.cols() != m || weight.rows() != m ||
        weight.cols() != m || !gamma.allFinite() || !weight.allFinite() ||
        (gamma.diagonal().array() <= 0.0).any())
      return std::unexpected(PostError{PostError::Kind::NumericIssue,
          "mixed DWLS policy: invalid NACOV or fitting weight"});
    const Eigen::MatrixXd expected = gamma.diagonal().cwiseInverse().asDiagonal();
    if ((weight - expected).norm() > 1e-6 * expected.norm())
      return std::unexpected(PostError{PostError::Kind::UnsupportedInference,
          "mixed DWLS policy requires the DWLS weight diag(NACOV)^-1"});
  }
  if (cache && cache->sampling) stats.sampling_moment_influence = *cache->sampling;
  if (stats.sampling_moment_influence.size() != stats.R.size()) {
    if (stats.raw_data.size() != stats.R.size())
      return std::unexpected(PostError{PostError::Kind::UnsupportedInference,
          "mixed DWLS policy requires complete raw data or exact sampling influence rows"});
    stats.sampling_moment_influence.clear();
    for (std::size_t b = 0; b < stats.R.size(); ++b) {
      if (!stats.raw_data[b].allFinite())
        return std::unexpected(PostError{PostError::Kind::UnsupportedInference,
            "mixed DWLS policy requires complete observations for exact first-stage influence"});
      auto rows = data::mixed_moment_sampling_influence(stats.raw_data[b],
          stats.ordered[b], stats.n_levels[b], stats.thresholds[b], stats.mean[b], stats.R[b]);
      if (!rows) return std::unexpected(rows.error());
      stats.sampling_moment_influence.push_back(std::move(*rows));
    }
    if (cache) cache->sampling = stats.sampling_moment_influence;
  }
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const auto& rows = stats.sampling_moment_influence[b];
    if (stats.n_obs[b] <= 0 || rows.rows() != stats.n_obs[b] || rows.cols() != stats.moments[b].size() || !rows.allFinite())
      return std::unexpected(PostError{PostError::Kind::NumericIssue,
          "mixed DWLS policy: invalid exact sampling influence rows"});
  }
  return stats;
}

template<class Stats, class Cache>
static post_expected<Stats> dwls_policy_stats(const Stats& input, Cache* cache) {
  if constexpr (std::is_same_v<Stats, data::MixedOrdinalStats>)
    return mixed_policy_stats(input, cache);
  else return input;
}

template<class Stats>
static auto dwls_policy_ij(spec::LatentStructure pt, const model::MatrixRep& rep,
    const Stats& stats, const estimate::Estimates& estimates,
    estimate::OrdinalWeightKind weights, estimate::OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* row_user) {
  if constexpr (std::is_same_v<Stats, data::MixedOrdinalStats>)
    return estimate::robust_mixed_ordinal_ij(std::move(pt), rep, stats, estimates, weights, parameterization, row_user);
  else return estimate::robust_ordinal_ij(std::move(pt), rep, stats, estimates, weights, parameterization, row_user);
}

template<class Stats>
static auto dwls_policy_global(spec::LatentStructure pt, const model::MatrixRep& rep,
    const Stats& stats, const estimate::Estimates& estimates,
    estimate::OrdinalWeightKind weights, estimate::OrdinalParameterization parameterization,
    robust::Information information, const std::vector<std::int8_t>* row_user) {
  if constexpr (std::is_same_v<Stats, data::MixedOrdinalStats>) {
    // Preserve OPG fitting weights; only the global sampling law changes.
    auto exact = stats;
    for (std::size_t b = 0; b < stats.R.size(); ++b) {
      const auto& rows = stats.sampling_moment_influence[b];
      exact.NACOV[b] = rows.transpose() * rows / static_cast<double>(stats.n_obs[b]);
    }
    return estimate::robust_mixed_ordinal(std::move(pt), rep, exact, estimates, weights, parameterization, information, row_user);
  } else return estimate::robust_ordinal(std::move(pt), rep, stats, estimates, weights, parameterization, information, row_user);
}

template<class Stats>
static auto dwls_policy_prepare(spec::LatentStructure& pt, const Stats& stats,
    estimate::OrdinalParameterization parameterization, spec::Starts* starts,
    const std::vector<std::int8_t>* row_user) {
  if constexpr (std::is_same_v<Stats, data::MixedOrdinalStats>)
    return estimate::prepare_mixed_ordinal_partable(pt, stats, parameterization, starts, row_user);
  else return estimate::prepare_ordinal_partable(pt, stats, parameterization, starts, row_user);
}

template<class Stats>
static auto dwls_policy_parts(const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const Stats& stats, const Eigen::VectorXd& theta, estimate::OrdinalWeightKind weights,
    estimate::OrdinalParameterization parameterization) {
  if constexpr (std::is_same_v<Stats, data::MixedOrdinalStats>)
    return estimate::frontier::mixed_ordinal_ls_newton_parts_prepared(pt, rep, stats, theta, weights, parameterization);
  else return estimate::frontier::ordinal_ls_newton_parts_prepared(pt, rep, stats, theta, weights, parameterization);
}

template<class Stats, class Cache>
static PolicyInference policy_inference_dwls_cached(spec::LatentStructure pt,
                                      const model::MatrixRep& rep,
                                      const Stats& input_stats,
                                      const estimate::Estimates& estimates,
                                      estimate::OrdinalParameterization parameterization,
                                      const PolicyFitState& state,
                                      const std::vector<std::int8_t>* row_user, Cache* cache) {
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
  if (auto ok = estimate::require_linear_sensitivity(pt); !ok) {
    auto unavailable = policy_unavailable(InferenceReason::UnsupportedModel, ok.error().detail);
    unavailable.verdict_disagreement = out.verdict_disagreement;
    unavailable.psd_boundary = out.psd_boundary;
    return unavailable;
  }
  auto exact = dwls_policy_stats(input_stats, cache);
  if (!exact) return policy_unavailable(reason_from(exact.error()), exact.error().detail);
  const auto& stats = *exact;
  using estimate::OrdinalWeightKind;
  if (cache && !cache->ij) {
    cache->ij = dwls_policy_ij(pt, rep, stats, estimates,
        OrdinalWeightKind::DWLS, parameterization, row_user);
    ++cache->builds;
  }
  auto ij = cache ? *cache->ij : dwls_policy_ij(pt, rep, stats,
      estimates, OrdinalWeightKind::DWLS, parameterization, row_user);
  if (ij) {
    out.covariance = ij->vcov;
  } else {
    out.covariance_reason = reason_from(ij.error());
    out.covariance_detail = ij.error().detail;
  }
  out.lr.reason = InferenceReason::Inapplicable;
  out.lr.detail = "DWLS has no likelihood; its global test is the fit-function "
                  "statistic, reported as the score test, which it equals";
  auto fixed = dwls_policy_global(std::move(pt), rep, stats, estimates,
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
  out.score.reference = "all";
  out.score.p_all = robust::frontier::fmg_test(out.score.statistic, out.score.df,
      out.score.eigenvalues, {robust::frontier::FmgMethod::All, 0.0, true}).p_value;
  if (!std::isfinite(out.score.p_all))
    set_unavailable(out.score, InferenceReason::NumericFailure,
                    "DWLS global test: All reference tail evaluation failed");
  return out;
}

template<class Stats, class Cache>
static PolicyNested policy_nested_dwls_cached(spec::LatentStructure null_pt,
                                const model::MatrixRep& null_rep,
                                const estimate::Estimates& null_estimates,
                                const PolicyFitState& null_state,
                                spec::LatentStructure alternative_pt,
                                const model::MatrixRep& alternative_rep,
                                const estimate::Estimates& alternative_estimates,
                                const PolicyFitState& alternative_state,
                                const Stats& input_stats,
                                estimate::OrdinalParameterization parameterization,
                                const std::vector<std::int8_t>* null_row_user,
                                const std::vector<std::int8_t>* alternative_row_user, Cache* cache) {
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

  for (const auto* pt : {&null_pt, &alternative_pt}) {
    if (auto ok = estimate::require_linear_sensitivity(*pt); !ok)
      return unavailable(InferenceReason::UnsupportedModel, ok.error().detail);
  }

  auto exact = dwls_policy_stats(input_stats, cache);
  if (!exact) return unavailable(reason_from(exact.error()), exact.error().detail);
  const auto& stats = *exact;
  Eigen::MatrixXd K, restriction;
  // Nesting: lift the null into the alternative's parameter space on the
  // prepared (threshold- and scale-augmented) structures.
  {
    spec::LatentStructure p1 = alternative_pt, p0 = null_pt;
    auto prepared1 = dwls_policy_prepare(p1, stats, parameterization, nullptr, alternative_row_user);
    auto prepared0 = dwls_policy_prepare(p0, stats, parameterization, nullptr, null_row_user);
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
    if (!embedding && embedding.error().kind == PostError::Kind::NotNested) {
      auto moment = frontier::moment_nested_tangent(null_pt, null_rep, null_estimates,
          alternative_pt, alternative_rep, alternative_estimates, stats, parameterization,
          null_row_user, alternative_row_user);
      if (!moment) return unavailable(
          moment.error().kind == PostError::Kind::NotNested ? InferenceReason::NotNested
          : moment.error().kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting
          : moment.error().kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting
          : InferenceReason::NumericFailure, moment.error().detail);
      restriction = moment->A;
    } else if (!embedding) {
      const auto kind = embedding.error().kind;
      return unavailable(kind == PostError::Kind::NotNested ? InferenceReason::NotNested
          : kind == PostError::Kind::UnsupportedNesting ? InferenceReason::UnsupportedNesting
          : kind == PostError::Kind::BoundaryNesting ? InferenceReason::BoundaryNesting
          : InferenceReason::NumericFailure, embedding.error().detail);
    } else {
      restriction = embedding->restriction.A;
    }
    if (restriction.rows() == 0)
      return unavailable(InferenceReason::EquivalentModels, "equivalent models in moment space");
    K = c1->K();
    alternative_pt = std::move(p1);
    null_pt = std::move(p0);
  }

  set_unavailable(out.score, InferenceReason::UnsupportedModel,
      "no nested DWLS score test is derived; the fit-function difference test is reported");
  if (cache && !cache->ij) {
    cache->ij = dwls_policy_ij(alternative_pt, alternative_rep, stats,
        alternative_estimates, estimate::OrdinalWeightKind::DWLS, parameterization,
        alternative_row_user);
    ++cache->builds;
  }
  auto ij = cache ? *cache->ij : dwls_policy_ij(alternative_pt,
      alternative_rep, stats, alternative_estimates, estimate::OrdinalWeightKind::DWLS,
      parameterization, alternative_row_user);
  if (!ij) {
    set_unavailable(out.lr, reason_from(ij.error()), ij.error().detail);
    return out;
  }
  if (cache && !cache->parts) {
    cache->parts = dwls_policy_parts(alternative_pt,
        alternative_rep, stats, alternative_estimates.theta,
        estimate::OrdinalWeightKind::DWLS, parameterization);
    ++cache->builds;
  }
  auto parts = cache ? *cache->parts :
      dwls_policy_parts(alternative_pt,
          alternative_rep, stats, alternative_estimates.theta,
          estimate::OrdinalWeightKind::DWLS, parameterization);
  if (!parts) {
    set_unavailable(out.lr, InferenceReason::NumericFailure, parts.error().detail);
    return out;
  }
  const double N = std::accumulate(stats.n_obs.begin(), stats.n_obs.end(), 0.0);
  const Eigen::MatrixXd H = K.transpose() * parts->hessian * K / N;
  // Pure-merge constraint coordinates are not orthonormal. Recover their
  // covariance with K's left inverse, then undo the IJ bread to obtain B.
  Eigen::LDLT<Eigen::MatrixXd> gram(K.transpose() * K);
  if (gram.info() != Eigen::Success || !gram.isPositive()) {
    set_unavailable(out.lr, InferenceReason::NumericFailure,
                    "DWLS nested test: singular constraint coordinates");
    return out;
  }
  const Eigen::MatrixXd L = gram.solve(K.transpose());
  const Eigen::MatrixXd V = N * L * ij->vcov * L.transpose();
  const Eigen::MatrixXd B = H * V * H.transpose();
  auto spectrum = robust::compute_satorra2000_from_sandwich(H, B, restriction);
  if (!spectrum) {
    set_unavailable(out.lr, reason_from(spectrum.error()), spectrum.error().detail);
    return out;
  }
  // est.fmin is half the discrepancy, including the existing group weights.
  const double statistic = 2.0 * N * (null_estimates.fmin - alternative_estimates.fmin);
  const double trace = spectrum->eigenvalues.sum();
  if (!std::isfinite(statistic) || !spectrum->eigenvalues.allFinite() || !(trace > 0.0)) {
    set_unavailable(out.lr, InferenceReason::NumericFailure,
                    "DWLS nested test: invalid statistic or reference spectrum");
    return out;
  }
  if (statistic < -1e-8 * std::max(1.0, trace)) {
    set_unavailable(out.lr, InferenceReason::NotConverged,
                    "the alternative fits worse than the null (fit-function difference " +
                    std::to_string(statistic) + ")");
    return out;
  }
  out.lr.statistic = std::max(0.0, statistic);
  out.lr.df = static_cast<int>(restriction.rows());
  out.lr.eigenvalues = spectrum->eigenvalues;
  out.lr.label = "fit_function_difference";
  calibrate_spectrum(out.lr);
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
  // Detect the negative raw LR before calibration: the quadratic constructor
  // rejects negative statistics, so checking a calibrated result hides this
  // recoverable fit-order failure behind NumericFailure.
  const auto& h = **hypothesis;
  const double scale = inference::chi2_stat(h.null_fit->data->sample, h.null_fit->estimates);
  const double difference = scale - inference::chi2_stat(
      h.alternative->data->sample, h.alternative->estimates);
  if (difference < -1e-8 * std::max(1.0, scale)) {
    set_unavailable(out.lr, InferenceReason::NotConverged,
                    "the alternative fits worse than the null (likelihood-ratio "
                    "statistic " + std::to_string(difference) + ")");
  } else {
    calibrate(robust::frontier::ntml_quadratic(**hypothesis, false, Information::Observed),
              out.lr);
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

static PolicyInference policy_inference_fiml_cached(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, const estimate::Estimates& estimates,
    const PolicyFitState& state, FimlPolicyFit::Impl* cache) {
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
    if (cache && !cache->bread) {
      cache->bread = fiml_score_meat_bread(pt, rep, raw, pack, estimates);
      ++cache->builds;
    }
    auto mb = cache ? *cache->bread : fiml_score_meat_bread(pt, rep, raw, pack, estimates);
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

static PolicyNested policy_nested_fiml_cached(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep,
    const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, FimlPolicyFit::Impl* cache) {
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
  if (cache && !cache->bread) {
    cache->bread = estimate::fiml::fiml_score_meat_bread(alternative_pt,
        alternative_rep, raw, pack, alternative_estimates);
    ++cache->builds;
  }
  auto mb = cache ? *cache->bread : estimate::fiml::fiml_score_meat_bread(
      alternative_pt, alternative_rep, raw, pack, alternative_estimates);
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


PolicyInference policy_inference_fiml(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, const estimate::Estimates& estimates,
    const PolicyFitState& state) {
  return policy_inference_fiml_cached(std::move(pt), rep, raw, pack, estimates, state, nullptr);
}
PolicyNested policy_nested_fiml(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep, const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack) {
  return policy_nested_fiml_cached(std::move(null_pt), null_rep, null_estimates,
      null_state, std::move(alternative_pt), alternative_rep, alternative_estimates,
      alternative_state, raw, pack, nullptr);
}
PolicyInference policy_inference_dwls(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const estimate::Estimates& estimates, estimate::OrdinalParameterization parameterization,
    const PolicyFitState& state, const std::vector<std::int8_t>* row_user) {
  return policy_inference_dwls_cached(std::move(pt), rep, stats, estimates,
      parameterization, state, row_user, static_cast<DwlsPolicyFit::Impl*>(nullptr));
}
PolicyNested policy_nested_dwls(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep, const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::OrdinalStats& stats,
    estimate::OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* null_row_user,
    const std::vector<std::int8_t>* alternative_row_user) {
  return policy_nested_dwls_cached(std::move(null_pt), null_rep, null_estimates,
      null_state, std::move(alternative_pt), alternative_rep, alternative_estimates,
      alternative_state, stats, parameterization, null_row_user, alternative_row_user, static_cast<DwlsPolicyFit::Impl*>(nullptr));
}
PolicyInference policy_inference_dwls(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::MixedOrdinalStats& stats,
    const estimate::Estimates& estimates, estimate::OrdinalParameterization parameterization,
    const PolicyFitState& state, const std::vector<std::int8_t>* row_user) {
  return policy_inference_dwls_cached(std::move(pt), rep, stats, estimates,
      parameterization, state, row_user, static_cast<MixedDwlsPolicyFit::Impl*>(nullptr));
}
PolicyNested policy_nested_dwls(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep, const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::MixedOrdinalStats& stats,
    estimate::OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* null_row_user,
    const std::vector<std::int8_t>* alternative_row_user) {
  return policy_nested_dwls_cached(std::move(null_pt), null_rep, null_estimates,
      null_state, std::move(alternative_pt), alternative_rep, alternative_estimates,
      alternative_state, stats, parameterization, null_row_user, alternative_row_user, static_cast<MixedDwlsPolicyFit::Impl*>(nullptr));
}
namespace {
template<class Matrix> bool same_blocks(const std::vector<Matrix>& a,
                                        const std::vector<Matrix>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].rows() != b[i].rows() || a[i].cols() != b[i].cols()) return false;
    const auto x = a[i].template cast<double>().array();
    const auto y = b[i].template cast<double>().array();
    if (!((x == y) || (x.isNaN() && y.isNaN())).all()) return false;
  }
  return true;
}
PolicyNested different_policy_data(const PolicyFitState& a, const PolicyFitState& b) {
  PolicyNested out;
  out.psd_boundary = a.psd_boundary || b.psd_boundary;
  out.verdict_disagreement = verdict_disagreement(a) || verdict_disagreement(b);
  set_unavailable(out.score, InferenceReason::NotNested,
      "policy snapshots must use the same observations, statistics and weight recipe");
  out.lr = out.score;
  return out;
}
}
PolicyInference policy_inference_fiml(FimlPolicyFit& fit, const PolicyFitState& state) {
  auto& f = *fit.impl;
  if (state.penalized || !state.converged)
    return policy_inference_fiml_cached(f.pt, f.rep, f.raw, f.pack, f.estimates, state, &f);
  // Retain which state flags the original path propagates, including its
  // early unsupported-model returns. State never changes the ingredients.
  if (!f.inference) f.inference = policy_inference_fiml_cached(
      f.pt, f.rep, f.raw, f.pack, f.estimates, {true, true, false, false}, &f);
  auto out = *f.inference;
  out.psd_boundary = out.psd_boundary && state.psd_boundary;
  out.verdict_disagreement = out.verdict_disagreement && verdict_disagreement(state);
  return out;
}
PolicyNested policy_nested_fiml(FimlPolicyFit& null, const PolicyFitState& null_state,
    FimlPolicyFit& alternative, const PolicyFitState& alternative_state) {
  auto& a = *null.impl; auto& b = *alternative.impl;
  if (!null_state.penalized && !alternative_state.penalized &&
      null_state.converged && alternative_state.converged &&
      (!same_blocks(a.raw.X, b.raw.X) || !same_blocks(a.raw.mask, b.raw.mask)))
    return different_policy_data(null_state, alternative_state);
  return policy_nested_fiml_cached(a.pt, a.rep, a.estimates, null_state,
      b.pt, b.rep, b.estimates, alternative_state, b.raw, b.pack, &b);
}
PolicyInference policy_inference_dwls(DwlsPolicyFit& fit, const PolicyFitState& state) {
  auto& f = *fit.impl;
  const auto* row_user = f.row_user.empty() ? nullptr : &f.row_user;
  if (state.penalized || !state.converged)
    return policy_inference_dwls_cached(f.pt, f.rep, f.stats, f.estimates,
        f.parameterization, state, row_user, &f);
  if (!f.inference) f.inference = policy_inference_dwls_cached(f.pt, f.rep,
      f.stats, f.estimates, f.parameterization, {true, true, false, false}, row_user, &f);
  auto out = *f.inference;
  out.psd_boundary = out.psd_boundary && state.psd_boundary;
  out.verdict_disagreement = out.verdict_disagreement && verdict_disagreement(state);
  return out;
}
PolicyNested policy_nested_dwls(DwlsPolicyFit& null, const PolicyFitState& null_state,
    DwlsPolicyFit& alternative, const PolicyFitState& alternative_state) {
  auto& a = *null.impl; auto& b = *alternative.impl;
  if (!null_state.penalized && !alternative_state.penalized &&
      null_state.converged && alternative_state.converged &&
      (a.parameterization != b.parameterization || a.stats.n_obs != b.stats.n_obs ||
      !same_blocks(a.stats.R, b.stats.R) || !same_blocks(a.stats.thresholds, b.stats.thresholds) ||
      !same_blocks(a.stats.int_data, b.stats.int_data) ||
      !same_blocks(a.stats.NACOV, b.stats.NACOV) || !same_blocks(a.stats.W_dwls, b.stats.W_dwls) ||
      !same_blocks(a.stats.moment_influence, b.stats.moment_influence) ||
      !same_blocks(a.stats.moment_bread, b.stats.moment_bread)))
    return different_policy_data(null_state, alternative_state);
  return policy_nested_dwls_cached(a.pt, a.rep, a.estimates, null_state,
      b.pt, b.rep, b.estimates, alternative_state, b.stats, b.parameterization,
      a.row_user.empty() ? nullptr : &a.row_user,
      b.row_user.empty() ? nullptr : &b.row_user, &b);
}

PolicyInference policy_inference_dwls(MixedDwlsPolicyFit& fit, const PolicyFitState& state) {
  auto& f = *fit.impl;
  const auto* row_user = f.row_user.empty() ? nullptr : &f.row_user;
  if (state.penalized || !state.converged)
    return policy_inference_dwls_cached(f.pt, f.rep, f.stats, f.estimates,
        f.parameterization, state, row_user, &f);
  if (!f.inference) f.inference = policy_inference_dwls_cached(f.pt, f.rep,
      f.stats, f.estimates, f.parameterization, {true, true, false, false}, row_user, &f);
  auto out = *f.inference;
  out.psd_boundary = out.psd_boundary && state.psd_boundary;
  out.verdict_disagreement = out.verdict_disagreement && verdict_disagreement(state);
  return out;
}
PolicyNested policy_nested_dwls(MixedDwlsPolicyFit& null, const PolicyFitState& null_state,
    MixedDwlsPolicyFit& alternative, const PolicyFitState& alternative_state) {
  auto& a = *null.impl; auto& b = *alternative.impl;
  if (!null_state.penalized && !alternative_state.penalized &&
      null_state.converged && alternative_state.converged &&
      (a.parameterization != b.parameterization || a.stats.n_obs != b.stats.n_obs ||
      !same_blocks(a.stats.R, b.stats.R) || !same_blocks(a.stats.thresholds, b.stats.thresholds) ||
      !same_blocks(a.stats.raw_data, b.stats.raw_data) ||
      !same_blocks(a.stats.NACOV, b.stats.NACOV) || !same_blocks(a.stats.W_dwls, b.stats.W_dwls) ||
      !same_blocks(a.stats.moment_influence, b.stats.moment_influence) ||
      !same_blocks(a.stats.mean, b.stats.mean) || !same_blocks(a.stats.moments, b.stats.moments) ||
      !same_blocks(a.stats.sampling_moment_influence, b.stats.sampling_moment_influence) ||
      a.stats.ordered != b.stats.ordered || a.stats.n_levels != b.stats.n_levels ||
      !same_blocks(a.stats.gamma_diag_influence, b.stats.gamma_diag_influence)))
    return different_policy_data(null_state, alternative_state);
  return policy_nested_dwls_cached(a.pt, a.rep, a.estimates, null_state,
      b.pt, b.rep, b.estimates, alternative_state, b.stats, b.parameterization,
      a.row_user.empty() ? nullptr : &a.row_user,
      b.row_user.empty() ? nullptr : &b.row_user, &b);
}

}  // namespace magmaan::api
