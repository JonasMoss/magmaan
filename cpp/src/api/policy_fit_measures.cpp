#include "magmaan/api/policy.hpp"
#include "magmaan/api/frontier/fit_index_intervals.hpp"
#include "magmaan/api/sem.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <numeric>

namespace magmaan::api {
namespace {
using estimate::OrdinalFirstStage;
using estimate::OrdinalParameterization;
PostError invalid(std::string detail) {
  return {PostError::Kind::NumericIssue, std::move(detail)};
}
void add(PolicyFitMeasures& out, std::string index, double value) {
  out.indices.push_back({std::move(index), value, InferenceReason::Available, {}});
}
void unavailable(PolicyFitMeasures& out, std::string index, InferenceReason reason,
                 std::string detail) {
  out.indices.push_back({std::move(index), std::numeric_limits<double>::quiet_NaN(),
                        reason, std::move(detail)});
}
void finish(PolicyFitMeasures& out) {
  out.user.corrected = std::max(out.user.discrepancy - out.user.trace / static_cast<double>(out.ntotal), 0.0);
  out.baseline.corrected = std::max(out.baseline.discrepancy - out.baseline.trace / static_cast<double>(out.ntotal), 0.0);
  const auto u = out.user.corrected, b = out.baseline.corrected;
  if (out.user.df > 0)
    add(out, "rmsea", std::sqrt(static_cast<double>(out.n_groups) * u / out.user.df));
  else unavailable(out, "rmsea", InferenceReason::Saturated, "RMSEA requires positive model df");
  if (out.user.df == 0 && b > 0.0) add(out, "cfi", 1.0);
  else if (std::max(u, b) > 0.0) add(out, "cfi", 1.0 - u / std::max(u, b));
  else unavailable(out, "cfi", InferenceReason::Inapplicable, "neither model has positive corrected misfit");
  if (out.user.df <= 0) unavailable(out, "tli", InferenceReason::Saturated, "TLI requires positive model df");
  else if (b > 0.0 && out.baseline.df > 0)
    add(out, "tli", 1.0 - (u / out.user.df) / (b / out.baseline.df));
  else unavailable(out, "tli", InferenceReason::Inapplicable, "TLI requires positive corrected baseline misfit and df");
}
post_expected<Eigen::MatrixXd> inverse(const Eigen::MatrixXd& h) {
  if (!h.rows()) return Eigen::MatrixXd(0, 0);
  Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (h + h.transpose()).eval());
  if (llt.info() != Eigen::Success)
    return std::unexpected(invalid("fit-measure observed information is not positive definite"));
  return Eigen::MatrixXd(llt.solve(Eigen::MatrixXd::Identity(h.rows(), h.cols())));
}
struct ContinuousParts {
  PolicyFitDiscrepancy discrepancy;
  model::Evaluation evaluation;
  // Total-sample casewise influence, so crossprod is covariance of theta_hat.
  Eigen::MatrixXd theta_rows;
  Eigen::MatrixXd observed_covariance;
};
post_expected<ContinuousParts> continuous_parts(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, const estimate::Estimates& est,
    const model::ImpliedMoments& saturated, double saturated_trace, bool means,
    bool fiml, bool uls, const Eigen::MatrixXd& saturated_rows) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return std::unexpected(invalid(ev.error().detail));
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval) return std::unexpected(invalid(eval.error().detail));
  auto constraints = estimate::build_eq_constraints(pt);
  if (!constraints) return std::unexpected(constraints.error());
  const auto& k = constraints->K();
  Eigen::MatrixXd rows, observed_covariance;
  double model_trace = 0.0;
  if (uls) {
    data::SampleStats sample{ saturated.sigma, means ? saturated.mu : std::vector<Eigen::VectorXd>{}, pack.start_stats.n_obs };
    auto profile = estimate::continuous_ls_profile_rmsea(pt, rep, sample, est, {}, raw);
    if (!profile) return std::unexpected(profile.error());
    auto h = estimate::gmm::moment_quadratic_hessian(*ev, sample, est.theta);
    if (!h) return std::unexpected(invalid(h.error().detail));
    auto v = inverse(k.transpose() * *h * k);
    if (!v) return std::unexpected(v.error());
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(saturated_rows.cols(), est.theta.size());
    d.topRows(eval->J_sigma.rows()) = eval->J_sigma;
    if (means) d.bottomRows(eval->J_mu.rows()) = eval->J_mu;
    const auto cov_dim = eval->J_sigma.rows();
    double moment_bias = 0.0;
    for (std::size_t b = 0; b < saturated.sigma.size(); ++b) {
      const auto p = saturated.sigma[b].rows(), nc = p * (p + 1) / 2;
      const auto co = pack.cache.sigma_offsets[b], mo = pack.cache.mu_offsets[b];
      Eigen::Index row = 0;
      for (Eigen::Index j = 0; j < p; ++j) for (Eigen::Index i = j; i < p; ++i, ++row)
        moment_bias += 2.0 * (eval->moments.sigma[b](i,j) - saturated.sigma[b](i,j)) * saturated.sigma[b](i,j);
      const auto n = static_cast<double>(pack.start_stats.n_obs[b]);
      d.middleRows(co, nc) *= n;
      if (means) d.middleRows(cov_dim + mo, p) *= n;
    }
    rows = saturated_rows * d * k * *v * k.transpose();
    model_trace = saturated_trace - profile->trace_signed - moment_bias;
  } else if (fiml) {
    auto parts = estimate::fiml::fiml_score_meat_bread(pt, rep, raw, pack, est);
    if (!parts) return std::unexpected(parts.error());
    Eigen::MatrixXd h = (static_cast<double>(pack.cache.n_total) / 2.0) * k.transpose() * parts->hessian * k;
    auto v = inverse(h);
    if (!v) return std::unexpected(v.error());
    Eigen::MatrixXd scores = -0.5 * parts->scores * k;
    model_trace = (scores.transpose() * scores * *v).trace();
    rows = scores * *v * k.transpose();
    observed_covariance = k * *v * k.transpose();
  } else {
    auto prepared = robust::frontier::prepare_ntml_data(raw, means);
    if (!prepared) return std::unexpected(prepared.error());
    auto fit = robust::frontier::prepare_ntml_fit(*prepared, pt, rep, est);
    if (!fit) return std::unexpected(fit.error());
    auto h = robust::frontier::ntml_observed_information(**fit);
    auto v = robust::frontier::ntml_observed_covariance(**fit);
    if (!h) return std::unexpected(h.error());
    if (!v) return std::unexpected(v.error());
    // Covariance-only ML profiles group means; its score residuals must use
    // those same means rather than treating the raw observations as mean zero.
    auto score_raw = raw;
    if (!means) for (std::size_t b = 0; b < score_raw.X.size(); ++b)
      score_raw.X[b].rowwise() -= saturated.mu[b].transpose();
    auto scores = estimate::fiml::fiml_casewise_deviance_scores(pt, rep, score_raw, pack, est);
    if (!scores) return std::unexpected(scores.error());
    Eigen::MatrixXd likelihood_scores = -0.5 * *scores;
    model_trace = (likelihood_scores.transpose() * likelihood_scores * **v).trace();
    rows = likelihood_scores * **v;
    observed_covariance = **v;
    // A covariance-only ML fit profiles out each group's unconstrained mean.
    // Those nuisance means contribute to Takeuchi even under covariance misfit.
    if (!means) {
      for (std::size_t b = 0; b < saturated.sigma.size(); ++b) {
        auto inv_sigma = inverse(eval->moments.sigma[b]);
        if (!inv_sigma) return std::unexpected(inv_sigma.error());
        model_trace += (*inv_sigma * saturated.sigma[b]).trace();
      }
    }
  }
  data::SampleStats sample{ saturated.sigma, means ? saturated.mu : std::vector<Eigen::VectorXd>{}, pack.start_stats.n_obs };
  auto df = inference::df_stat(pt, sample, est.theta);
  if (!df) return std::unexpected(df.error());
  ContinuousParts out;
  out.discrepancy.df = *df;
  out.discrepancy.trace = saturated_trace - model_trace;
  out.evaluation = std::move(*eval);
  out.theta_rows = std::move(rows);
  out.observed_covariance = std::move(observed_covariance);
  return out;
}

// The residual vector is standardized by the observed moments. Differentiate
// both numerator and its observed SD denominator; under misfit the latter
// contributes at first order. Independent groups pool squared residual roots.
post_expected<void> continuous_residual(PolicyFitMeasures& out,
    const ContinuousParts& parts, const model::ImpliedMoments& sat,
    const estimate::fiml::FIMLPack& pack, const Eigen::MatrixXd& sat_rows,
    bool means) {
  const Eigen::Index m = sat_rows.cols();
  const Eigen::Index q = parts.theta_rows.cols();
  Eigen::Index cov_dim = 0;
  for (const auto& s : sat.sigma) cov_dim += s.rows() * (s.rows() + 1) / 2;
  double squared = 0.0, trace = 0.0;
  for (std::size_t b = 0; b < sat.sigma.size(); ++b) {
    const auto& s = sat.sigma[b];
    const auto& fitted = parts.evaluation.moments.sigma[b];
    const auto p = s.rows();
    const auto nc = p * (p + 1) / 2;
    const auto nr = nc + (means ? p : 0);
    Eigen::VectorXd z(nr);
    Eigen::MatrixXd ds = Eigen::MatrixXd::Zero(nr, m);
    Eigen::MatrixXd dt = Eigen::MatrixXd::Zero(nr, q);
    Eigen::VectorX<Eigen::Index> diag(p);
    Eigen::Index row = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      diag[j] = row;
      row += p - j;
    }
    row = 0;
    const auto co = pack.cache.sigma_offsets[b], mo = pack.cache.mu_offsets[b];
    for (Eigen::Index j = 0; j < p; ++j) for (Eigen::Index i = j; i < p; ++i, ++row) {
      if (!(s(i,i) > 0 && s(j,j) > 0)) return std::unexpected(invalid("SRMR requires positive observed variances"));
      const double scale = 1.0 / std::sqrt(s(i,i) * s(j,j));
      z(row) = (s(i,j) - fitted(i,j)) * scale;
      ds(row, co + row) = scale;
      ds(row, co + diag[i]) -= 0.5 * z(row) / s(i,i);
      ds(row, co + diag[j]) -= 0.5 * z(row) / s(j,j);
      dt.row(row) = -scale * parts.evaluation.J_sigma.row(co + row);
    }
    if (means) for (Eigen::Index j = 0; j < p; ++j) {
      const double scale = 1.0 / std::sqrt(s(j,j));
      z(nc+j) = (sat.mu[b](j) - parts.evaluation.moments.mu[b](j)) * scale;
      ds(nc+j, cov_dim + mo + j) = scale;
      ds(nc+j, co + diag[j]) = -0.5 * z(nc+j) / s(j,j);
      dt.row(nc+j) = -scale * parts.evaluation.J_mu.row(mo + j);
    }
    Eigen::MatrixXd residual_rows = sat_rows * ds.transpose() + parts.theta_rows * dt.transpose();
    const double weight = static_cast<double>(pack.start_stats.n_obs[b]) / static_cast<double>(out.ntotal);
    squared += weight * z.squaredNorm() / static_cast<double>(nr);
    trace += static_cast<double>(out.ntotal) * weight * residual_rows.squaredNorm() / static_cast<double>(nr);
  }
  out.residual_uncorrected = std::sqrt(std::max(squared, 0.0));
  out.residual_trace = trace;
  add(out, "srmr", std::sqrt(std::max(squared - trace / static_cast<double>(out.ntotal), 0.0)));
  return {};
}

// Synthetic observed-only baseline: variable ids need no user names. Free
// variances and (when present) means; no across-group equality restrictions.
Result<Model> independence_model(Eigen::Index p, int groups, bool means) {
  std::string syntax;
  for (Eigen::Index j = 0; j < p; ++j) syntax += "v" + std::to_string(j) + " ~~ v" + std::to_string(j) + "\n";
  ModelOptions opts;
  opts.build.n_groups = groups;
  opts.build.meanstructure = means;
  opts.build.fixed_x = false;
  return model_from_lavaan(syntax, opts);
}
} // namespace

PolicyFitMeasures policy_fit_measures_unavailable(InferenceReason reason,
    std::string detail, bool ordinal, bool likelihood) {
  PolicyFitMeasures out;
  for (const auto* index : {"rmsea", "cfi", "tli", "srmr"}) unavailable(out, index, reason, detail);
  if (ordinal) unavailable(out, "crmr", reason, detail);
  if (likelihood) for (const auto* index : {"logl", "unrestricted.logl", "aic", "bic"}) unavailable(out, index, reason, detail);
  return out;
}

PolicyFitMeasures policy_fit_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::Estimates& est, const PolicyFitState& state, bool fiml, bool uls) {
  if (state.penalized) return policy_fit_measures_unavailable(InferenceReason::Penalized, std::string(penalized_detail));
  if (!state.converged) return policy_fit_measures_unavailable(InferenceReason::NotConverged, "the fit did not pass its convergence verdict");
  if (std::any_of(pt.exo.begin(), pt.exo.end(), [](auto x) { return x != 0; }) ||
      pt.has_inequality_constraints || !pt.nonlinear_eq_rows.empty())
    return policy_fit_measures_unavailable(InferenceReason::UnsupportedModel,
        "policy fit measures require random x and affine equality constraints", false, !uls);
  auto fail = [uls](const std::string& detail) { return policy_fit_measures_unavailable(InferenceReason::NumericFailure, detail, false, !uls); };
  auto evaluator = model::ModelEvaluator::build(pt, rep);
  if (!evaluator) return fail(evaluator.error().detail);
  auto eval = evaluator->evaluate(est.theta, true, true);
  if (!eval) return fail(eval.error().detail);
  const bool means = eval->J_mu.rows() > 0;
  if (raw.X.empty()) return fail("fit measures require raw observations");
  auto pack = estimate::fiml::fiml_pack(raw);
  if (!pack) return fail(pack.error().detail);
  // Trace corrections need tighter H1 accuracy than fitting starts: an
  // absolute EM stop at 1e-5 otherwise changes them when units are rescaled.
  estimate::fiml::FIMLH1Options h1_options;
  h1_options.parameter_tol = 1e-10;
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack, h1_options);
  if (!h1) return fail(h1.error().detail);
  model::ImpliedMoments sat{h1->sigma, h1->mu};
  // Include all unrestricted means in the saturated Takeuchi trace. The
  // covariance-only model has its profiled mean nuisance contribution below.
  auto h_sat = estimate::fiml::fiml_saturated_observed_information(raw, *pack, sat, true);
  auto s_sat = estimate::fiml::fiml_saturated_casewise_deviance_scores(raw, *pack, sat, true);
  if (!h_sat) return fail(h_sat.error().detail);
  if (!s_sat) return fail(s_sat.error().detail);
  auto v_sat = inverse(*h_sat);
  if (!v_sat) return fail(v_sat.error().detail);
  const Eigen::MatrixXd scores_sat = -0.5 * *s_sat;
  const double t_sat = (scores_sat.transpose() * scores_sat * *v_sat).trace();
  Eigen::MatrixXd saturated_rows = scores_sat * *v_sat;
  if (!means) saturated_rows = saturated_rows.leftCols(eval->J_sigma.rows()).eval();
  auto user = continuous_parts(pt, rep, raw, *pack, est, sat, t_sat, means, fiml, uls, saturated_rows);
  if (!user) return fail(user.error().detail);
  auto base_model = independence_model(raw.X[0].cols(), static_cast<int>(raw.X.size()), means);
  if (!base_model) return fail(base_model.error().detail);
  data::SampleStats baseline_sample{sat.sigma,
      means ? sat.mu : std::vector<Eigen::VectorXd>{}, pack->start_stats.n_obs};
  auto base_data = fiml ? data_from_raw(*base_model, raw) :
      data_from_sample_stats(*base_model, std::move(baseline_sample));
  if (!base_data) return fail(base_data.error().detail);
  auto base_fit = fit(*base_model, *base_data, uls ? api::uls() : fiml ? api::fiml() : ml());
  if (!base_fit) return fail(base_fit.error().detail);
  if (!policy_fit_state(base_fit->estimates()).converged) return fail("independence baseline did not converge");
  auto baseline = continuous_parts(base_model->structure(), base_model->matrix_rep(), raw,
      *pack, base_fit->estimates(), sat, t_sat, means, fiml, uls, saturated_rows);
  if (!baseline) return fail(baseline.error().detail);
  PolicyFitMeasures out;
  out.ntotal = pack->cache.n_total;
  out.n_groups = raw.X.size();
  out.user = user->discrepancy;
  out.baseline = baseline->discrepancy;
  if (uls) {
    out.user.discrepancy = 2.0 * est.fmin;
    out.baseline.discrepancy = 2.0 * base_fit->estimates().fmin;
    finish(out);
  } else if (fiml) {
    auto extras = estimate::fiml::fiml_extras(pt, rep, raw, est, *pack, *h1);
    auto base_extras = estimate::fiml::fiml_extras(base_model->structure(), base_model->matrix_rep(), raw, base_fit->estimates(), *pack, *h1);
    if (!extras) return fail(extras.error().detail);
    if (!base_extras) return fail(base_extras.error().detail);
    out.user.discrepancy = extras->chi2 / static_cast<double>(out.ntotal);
    out.baseline.discrepancy = base_extras->chi2 / static_cast<double>(out.ntotal);
    finish(out);
    add(out, "logl", extras->logl); add(out, "unrestricted.logl", extras->unrestricted_logl);
    add(out, "aic", extras->aic); add(out, "bic", extras->bic);
  } else {
    data::SampleStats sample{sat.sigma, means ? sat.mu : std::vector<Eigen::VectorXd>{}, pack->start_stats.n_obs};
    auto extras = measures::fit_extras(pt, rep, sample, est);
    auto base_extras = measures::fit_extras(base_model->structure(), base_model->matrix_rep(), sample, base_fit->estimates());
    if (!extras) return fail(extras.error().detail);
    if (!base_extras) return fail(base_extras.error().detail);
    out.user.discrepancy = 2.0 * (extras->unrestricted_logl - extras->logl) / static_cast<double>(out.ntotal);
    out.baseline.discrepancy = 2.0 * (base_extras->unrestricted_logl - base_extras->logl) / static_cast<double>(out.ntotal);
    finish(out);
    add(out, "logl", extras->logl); add(out, "unrestricted.logl", extras->unrestricted_logl);
    add(out, "aic", extras->aic); add(out, "bic", extras->bic);
  }
  auto residual = continuous_residual(out, *user, sat, *pack, saturated_rows, means);
  if (!residual) unavailable(out, "srmr", InferenceReason::NumericFailure, residual.error().detail);
  return out;
}

namespace {
template<class Stats>
PolicyFitMeasures ordinal_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const Stats& stats, const estimate::Estimates& est,
    const PolicyFitState& state, OrdinalParameterization parameterization) {
  if (state.penalized) return policy_fit_measures_unavailable(InferenceReason::Penalized, std::string(penalized_detail), true, false);
  if (!state.converged) return policy_fit_measures_unavailable(InferenceReason::NotConverged, "the fit did not pass its convergence verdict", true, false);
  auto incremental = [&]() {
    if constexpr (std::is_same_v<Stats, data::OrdinalStats>)
      return estimate::ordinal_cfi_tli_misspec_inference(pt, rep, stats, est, parameterization, true, 0.9, 1e-10, OrdinalFirstStage::Exact);
    else return estimate::mixed_ordinal_cfi_tli_misspec_inference(pt, rep, stats, est, parameterization, true, 0.9, 1e-10, OrdinalFirstStage::Exact);
  }();
  PolicyFitMeasures out;
  out.ntotal = std::accumulate(stats.n_obs.begin(), stats.n_obs.end(), std::int64_t{0});
  out.n_groups = stats.R.size();
  if (incremental) {
    out.user = {incremental->stat_user / static_cast<double>(out.ntotal), incremental->gendf_user, 0, incremental->df_user};
    out.baseline = {incremental->stat_baseline / static_cast<double>(out.ntotal), incremental->gendf_baseline, 0, incremental->df_baseline};
    finish(out);
    // Preserve the primitive's operation ordering for its shared CFI point.
    // Its clipping is algebraically the policy's max(user, baseline) divisor.
    if (out.user.df > 0 && incremental->delta_baseline > 0.0)
      for (auto& i : out.indices) if (i.index == "cfi") i.estimate = incremental->cfi;
  } else for (const auto* index : {"rmsea", "cfi", "tli"}) unavailable(out, index,
      incremental.error().kind == PostError::Kind::UnsupportedInference ? InferenceReason::UnsupportedModel : InferenceReason::NumericFailure, incremental.error().detail);
  for (bool srmr : {false, true}) {
    auto residual = [&]() {
      if constexpr (std::is_same_v<Stats, data::OrdinalStats>)
        return estimate::ordinal_crmr_misspec_inference(pt, rep, stats, est, parameterization, true, srmr, 0.9, 1e-10, OrdinalFirstStage::Exact);
      else return estimate::mixed_ordinal_crmr_misspec_inference(pt, rep, stats, est, parameterization, true, srmr, 0.9, 1e-10, OrdinalFirstStage::Exact);
    }();
    if (residual) {
      add(out, srmr ? "srmr" : "crmr", residual->point_bias_corrected);
      if (srmr) { out.residual_uncorrected = residual->point; out.residual_trace = residual->bias_trace / residual->k; }
    } else unavailable(out, srmr ? "srmr" : "crmr", residual.error().kind == PostError::Kind::UnsupportedInference ? InferenceReason::UnsupportedModel : InferenceReason::NumericFailure, residual.error().detail);
  }
  return out;
}
} // namespace
PolicyFitMeasures policy_fit_measures(spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::OrdinalStats& stats, const estimate::Estimates& est, const PolicyFitState& state,
    OrdinalParameterization parameterization) {
  return ordinal_measures(std::move(pt), rep, stats, est, state, parameterization);
}
PolicyFitMeasures policy_fit_measures(spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats, const estimate::Estimates& est, const PolicyFitState& state,
    OrdinalParameterization parameterization) {
  return ordinal_measures(std::move(pt), rep, stats, est, state, parameterization);
}
PolicyFitMeasures policy_fit_measures_two_stage(spec::LatentStructure pt,
    const model::MatrixRep& rep, const estimate::Estimates& est,
    const estimate::SaturatedMoments& stage1, const PolicyFitState& state) {
  if (state.penalized || !state.converged)
    return policy_fit_measures_unavailable(state.penalized ? InferenceReason::Penalized : InferenceReason::NotConverged,
        state.penalized ? std::string(penalized_detail) : "the fit did not pass its convergence verdict", false, false);
  auto fail = [](std::string detail) {
    return policy_fit_measures_unavailable(InferenceReason::NumericFailure, std::move(detail), false, false);
  };
  auto profile = estimate::fiml::two_stage_nt_profile_rmsea(pt, rep, est, stage1);
  if (!profile) return fail(profile.error().detail);
  auto model = independence_model(stage1.cov[0].rows(), static_cast<int>(stage1.cov.size()), true);
  if (!model) return fail(model.error().detail);
  data::SampleStats sample{stage1.cov, stage1.mean, stage1.n_obs};
  auto data = data_from_sample_stats(*model, sample);
  if (!data) return fail(data.error().detail);
  auto baseline = fit(*model, *data, ml());
  if (!baseline) return fail(baseline.error().detail);
  if (!policy_fit_state(baseline->estimates()).converged) return fail("ML2S independence baseline did not converge");
  auto base_profile = estimate::fiml::two_stage_nt_profile_rmsea(model->structure(), model->matrix_rep(), baseline->estimates(), stage1);
  if (!base_profile) return fail(base_profile.error().detail);
  PolicyFitMeasures out;
  out.ntotal = profile->ntotal;
  out.n_groups = profile->n_groups;
  out.user = {profile->fmin, profile->trace_signed, 0, profile->df};
  out.baseline = {base_profile->fmin, base_profile->trace_signed, 0, base_profile->df};
  finish(out);
  unavailable(out, "srmr", InferenceReason::UnsupportedModel,
      "missing ML2S Stage-2 residual influence composition over EM-moment rows");
  return out;
}
PolicyFitMeasures policy_fit_measures(const Fit& fit) {
  const auto state = policy_fit_state(fit.estimates());
  const auto& pt = fit.model().structure();
  const auto& rep = fit.model().matrix_rep();
  const auto& est = fit.estimates();
  const auto ordinal = fit.data().ordinal();
  const auto mixed = fit.data().mixed_ordinal();
  if (fit.estimator() == EstimatorKind::DWLS) {
    if (ordinal) return policy_fit_measures(pt, rep, *ordinal, est, state, fit.estimator_spec().ordinal_parameterization);
    if (mixed) return policy_fit_measures(pt, rep, *mixed, est, state, fit.estimator_spec().ordinal_parameterization);
  }
  if ((fit.estimator() == EstimatorKind::ML || fit.estimator() == EstimatorKind::FIML || fit.estimator() == EstimatorKind::ULS) && fit.data().raw())
    return policy_fit_measures(pt, rep, *fit.data().raw(), est, state, fit.estimator() == EstimatorKind::FIML, fit.estimator() == EstimatorKind::ULS);
  return policy_fit_measures_unavailable(state.penalized ? InferenceReason::Penalized :
      !state.converged ? InferenceReason::NotConverged : InferenceReason::UnsupportedModel,
      fit.estimator() == EstimatorKind::GLS || fit.estimator() == EstimatorKind::WLS ? "missing natural-discrepancy estimated-weight profile trace for this estimator" : "missing natural-discrepancy profile correction or raw sampling rows for this estimator", ordinal || mixed,
      fit.estimator() == EstimatorKind::ML || fit.estimator() == EstimatorKind::FIML);
}

namespace frontier {
namespace {
// Constants cancel in this contrast, including with observed-pattern FIML.
post_expected<Eigen::VectorXd> likelihood_contrast(const data::RawData& raw,
    const model::ImpliedMoments& sat, const model::ImpliedMoments& fitted) {
  Eigen::Index n = 0;
  for (const auto& x : raw.X) n += x.rows();
  Eigen::VectorXd out(n);
  Eigen::Index row = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& x = raw.X[b];
    for (Eigen::Index r = 0; r < x.rows(); ++r, ++row) {
      std::vector<Eigen::Index> obs;
      for (Eigen::Index c = 0; c < x.cols(); ++c)
        if (raw.mask.empty() || raw.mask[b](r,c)) obs.push_back(c);
      const auto p = static_cast<Eigen::Index>(obs.size());
      Eigen::MatrixXd s(p,p), t(p,p);
      Eigen::VectorXd ds(p), dt(p);
      for (Eigen::Index j = 0; j < p; ++j) {
        const auto c = obs[static_cast<std::size_t>(j)];
        ds(j) = x(r,c) - sat.mu[b](c);
        dt(j) = x(r,c) - fitted.mu[b](c);
        for (Eigen::Index i = 0; i < p; ++i) {
          s(i,j) = sat.sigma[b](obs[static_cast<std::size_t>(i)],c);
          t(i,j) = fitted.sigma[b](obs[static_cast<std::size_t>(i)],c);
        }
      }
      Eigen::LLT<Eigen::MatrixXd> ls(s), lt(t);
      if (!p || ls.info() != Eigen::Success || lt.info() != Eigen::Success)
        return std::unexpected(invalid("casewise likelihood requires positive definite observed-pattern covariances"));
      const double log_s = 2.0 * ls.matrixL().toDenseMatrix().diagonal().array().log().sum();
      const double log_t = 2.0 * lt.matrixL().toDenseMatrix().diagonal().array().log().sum();
      out(row) = 0.5 * (log_t - log_s + dt.dot(lt.solve(dt)) - ds.dot(ls.solve(ds)));
    }
  }
  return out;
}
double normal_critical(double level) {
  // Monotone inversion avoids another normal-quantile dependency in the API.
  double lo = 0, hi = 40;
  for (int i = 0; i < 80; ++i) {
    const double mid = 0.5*(lo+hi);
    if (std::erfc(mid/std::sqrt(2.0)) > 1.0-level) lo = mid;
    else hi = mid;
  }
  return 0.5*(lo+hi);
}
} // namespace
FitIndexIntervals fit_index_intervals_misspec_unavailable(InferenceReason reason,
    std::string detail) {
  FitIndexIntervals out;
  for (const auto* name : {"rmsea", "cfi", "tli"})
    for (const auto* variant : {"wald_f", "wald_sqrt"}) {
      FitIndexInterval row;
      row.index = name; row.variant = variant;
      row.reason = reason; row.detail = detail;
      out.indices.push_back(std::move(row));
    }
  return out;
}
FitIndexIntervals fit_index_intervals_misspec(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::Estimates& est, const PolicyFitState& state, bool fiml,
    double level) {
  auto fail = [](std::string detail) {
    return fit_index_intervals_misspec_unavailable(InferenceReason::NumericFailure, std::move(detail));
  };
  if (!std::isfinite(level) || !(level > 0 && level < 1))
    return fit_index_intervals_misspec_unavailable(InferenceReason::Inapplicable, "level must be strictly between zero and one");
  if (state.penalized || !state.converged)
    return fit_index_intervals_misspec_unavailable(state.penalized ? InferenceReason::Penalized : InferenceReason::NotConverged,
        state.penalized ? std::string(penalized_detail) : "the fit did not pass its convergence verdict");
  if (raw.X.empty() || std::any_of(pt.exo.begin(), pt.exo.end(), [](auto x) { return x != 0; }) ||
      pt.has_inequality_constraints || !pt.nonlinear_eq_rows.empty() ||
      (!fiml && !raw.mask.empty()))
    return fit_index_intervals_misspec_unavailable(InferenceReason::UnsupportedModel,
        "continuous ML/FIML intervals require raw sampling rows, random x and affine equality constraints");
  auto points = policy_fit_measures(pt, rep, raw, est, state, fiml);
  for (const auto& i : points.indices)
    if (i.index != "srmr" && (i.reason == InferenceReason::NumericFailure || i.reason == InferenceReason::UnsupportedModel))
      return fit_index_intervals_misspec_unavailable(i.reason, i.detail);
  auto pack = estimate::fiml::fiml_pack(raw);
  if (!pack) return fail(pack.error().detail);
  estimate::fiml::FIMLH1Options h1_options;
  h1_options.parameter_tol = 1e-10;
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack, h1_options);
  if (!h1) return fail(h1.error().detail);
  const model::ImpliedMoments sat{h1->sigma, h1->mu};
  auto evaluator = model::ModelEvaluator::build(pt, rep);
  if (!evaluator) return fail(evaluator.error().detail);
  auto eval = evaluator->evaluate(est.theta, true, true);
  if (!eval) return fail(eval.error().detail);
  const bool means = eval->J_mu.rows() > 0;
  // A meanless FIML model has fixed zero means, unlike profiled complete ML.
  if (fiml && !means)
    return fit_index_intervals_misspec_unavailable(InferenceReason::UnsupportedModel,
        "FIML intervals require an explicit mean structure");
  auto hs = estimate::fiml::fiml_saturated_observed_information(raw, *pack, sat, true);
  auto ss = estimate::fiml::fiml_saturated_casewise_deviance_scores(raw, *pack, sat, true);
  if (!hs) return fail(hs.error().detail);
  if (!ss) return fail(ss.error().detail);
  auto vs = inverse(*hs);
  if (!vs) return fail(vs.error().detail);
  const Eigen::MatrixXd scores = -0.5 * *ss;
  const Eigen::MatrixXd meat = scores.transpose()*scores;
  const double t_sat = (meat * *vs).trace();
  Eigen::MatrixXd sat_rows = scores * *vs;
  if (!means) sat_rows = sat_rows.leftCols(eval->J_sigma.rows()).eval();
  auto user = continuous_parts(pt, rep, raw, *pack, est, sat, t_sat, means, fiml, false, sat_rows);
  if (!user) return fail(user.error().detail);
  Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(hs->rows(),est.theta.size());
  const auto nc = eval->J_sigma.rows();
  delta.topRows(nc) = eval->J_sigma;
  if (means) delta.bottomRows(eval->J_mu.rows()) = eval->J_mu;
  Eigen::MatrixXd projection = delta*user->observed_covariance*delta.transpose();
  if (!means) {
    // J_0 augmented with free group means has zero mean/covariance cross
    // blocks at their profiled optimum and inverse mean block Sigma_0/n_g.
    for (std::size_t b = 0; b < raw.X.size(); ++b) {
      const auto p = sat.sigma[b].rows(), offset = nc+pack->cache.mu_offsets[b];
      projection.block(offset,offset,p,p) += eval->moments.sigma[b]/static_cast<double>(pack->start_stats.n_obs[b]);
    }
    eval->moments.mu = sat.mu;
  }
  const Eigen::MatrixXd m = (*vs-projection)*meat;
  auto d = likelihood_contrast(raw,sat,eval->moments);
  if (!d) return fail(d.error().detail);
  auto baseline = independence_model(raw.X[0].cols(),static_cast<int>(raw.X.size()),means);
  if (!baseline) return fail(baseline.error().detail);
  data::SampleStats sample{sat.sigma, means ? sat.mu : std::vector<Eigen::VectorXd>{},pack->start_stats.n_obs};
  auto bd = fiml ? data_from_raw(*baseline,raw) : data_from_sample_stats(*baseline,sample);
  if (!bd) return fail(bd.error().detail);
  auto bf = fit(*baseline,*bd,fiml ? api::fiml() : ml());
  if (!bf) return fail(bf.error().detail);
  auto be = model::ModelEvaluator::build(baseline->structure(),baseline->matrix_rep());
  if (!be) return fail(be.error().detail);
  auto bm = be->evaluate(bf->estimates().theta,false,false);
  if (!bm) return fail(bm.error().detail);
  if (!means) bm->moments.mu = sat.mu;
  auto db = likelihood_contrast(raw,sat,bm->moments);
  if (!db) return fail(db.error().detail);
  const double n = static_cast<double>(pack->cache.n_total);
  if (n <= 1) return fail("casewise variance requires at least two observations");
  const Eigen::VectorXd centered = d->array()-d->mean();
  const Eigen::VectorXd centered_b = db->array()-db->mean();
  const double v1 = 4.0*centered.squaredNorm()/(n*(n-1));
  const double v1b = 4.0*centered_b.squaredNorm()/(n*(n-1));
  const double cross = 4.0*centered.dot(centered_b)/(n*(n-1));
  const double tr2 = (m*m).trace();
  if (!std::isfinite(tr2) || tr2 < 0) return fail("second-order variance is nonfinite or negative");
  const double v2 = 2.0*tr2/(n*n), z = normal_critical(level);
  FitIndexIntervals out;
  out.points = std::move(points);
  out.casewise_discrepancy = 2.0*d->mean();
  out.quadratic_trace = m.trace();
  out.quadratic_squared_trace = tr2;
  const double u = out.points.user.corrected, b = out.points.baseline.corrected;
  for (const auto* name : {"rmsea", "cfi", "tli"}) for (const auto* variant : {"wald_f", "wald_sqrt"}) {
    FitIndexInterval row;
    row.index = name; row.variant = variant; row.v1 = v1; row.v2 = v2;
    for (const auto& point : out.points.indices) if (point.index == name) {
      row.estimate = point.estimate; row.reason = point.reason; row.detail = point.detail;
    }
    const bool root = row.variant == "wald_sqrt";
    if (row.reason == InferenceReason::Available && root && !(u > 0)) {
      row.reason = InferenceReason::Inapplicable;
      row.detail = "wald_sqrt requires positive corrected user discrepancy";
    }
    if (row.reason == InferenceReason::Available) {
      if (row.index == "rmsea") {
        const double se = std::sqrt(v1+v2);
        double lower = std::max(u-z*se,0.0), upper = u+z*se;
        if (root) {
          const double center = std::sqrt(u), width = z*se/(2.0*center);
          lower = std::pow(std::max(center-width,0.0),2);
          upper = std::pow(center+width,2);
        }
        const double scale = static_cast<double>(out.points.n_groups)/out.points.user.df;
        row.lower = std::sqrt(scale*lower); row.upper = std::sqrt(scale*upper);
      } else if (!(b > 0) || out.points.baseline.df <= 0) {
        row.reason = InferenceReason::Inapplicable;
        row.detail = "incremental intervals require positive corrected baseline discrepancy and df";
      } else {
        // The sqrt coordinate's chain rule yields the same incremental delta
        // variance; only RMSEA's mapped finite endpoints differ by variant.
        const double factor = row.index == "tli" ? static_cast<double>(out.points.baseline.df)/out.points.user.df : 1.0;
        double gu = -factor/b, gb = factor*u/(b*b);
        if (row.index == "cfi" && u >= b) gu = gb = 0.0;
        const double variance = gu*gu*(v1+v2)+gb*gb*v1b+2.0*gu*gb*cross;
        const double width = z*std::sqrt(std::max(variance,0.0));
        row.lower = row.estimate-width; row.upper = row.estimate+width;
        if (row.index == "cfi") {
          row.lower = std::clamp(row.lower,0.0,1.0); row.upper = std::clamp(row.upper,0.0,1.0);
        }
        // TLI Wald endpoints remain unbounded, as in the lab's convention.
      }
    }
    out.indices.push_back(std::move(row));
  }
  return out;
}
FitIndexIntervals fit_index_intervals_misspec(const Fit& fit, double level) {
  const auto state = policy_fit_state(fit.estimates());
  if ((fit.estimator() == EstimatorKind::ML || fit.estimator() == EstimatorKind::FIML) &&
      fit.data().raw() && !fit.data().ordinal() && !fit.data().mixed_ordinal())
    return fit_index_intervals_misspec(fit.model().structure(),fit.model().matrix_rep(),
        *fit.data().raw(),fit.estimates(),state,fit.estimator() == EstimatorKind::FIML,level);
  return fit_index_intervals_misspec_unavailable(state.penalized ? InferenceReason::Penalized :
      !state.converged ? InferenceReason::NotConverged : InferenceReason::UnsupportedModel,
      "candidate intervals support only continuous ML/FIML with raw observations");
}
} // namespace frontier

} // namespace magmaan::api
