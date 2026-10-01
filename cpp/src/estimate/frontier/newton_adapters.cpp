#include "magmaan/estimate/frontier/newton_adapters.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include "magmaan/estimate/nt.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/optim/optimizers.hpp"

namespace magmaan::estimate::frontier {
namespace {
FitError error(std::string detail) {
  return {FitError::Kind::NumericIssue, "Newton adapter: " + std::move(detail)};
}
double total_n(const std::vector<std::int64_t>& counts) {
  return std::accumulate(counts.begin(), counts.end(), 0.0);
}
fit_expected<void> validate(const spec::LatentStructure& pt,
                            const Eigen::VectorXd& theta,
                            const NewtonAdapterOptions& opts, bool ls) {
  if (theta.size() != pt.n_free() || !theta.allFinite())
    return std::unexpected(error("invalid full-theta vector"));
  if (opts.gauss_newton && !ls)
    return std::unexpected(error("Gauss-Newton is only defined for LS adapters"));
  if (!std::isfinite(opts.active_bound_tol) || opts.active_bound_tol < 0)
    return std::unexpected(error("invalid active-bound tolerance"));
  if (!opts.bounds.empty() &&
      (opts.bounds.lower.size() != theta.size() || opts.bounds.upper.size() != theta.size() ||
       opts.bounds.lower.array().isNaN().any() || opts.bounds.upper.array().isNaN().any() ||
       (opts.bounds.lower.array() > opts.bounds.upper.array()).any()))
    return std::unexpected(error("invalid bounds"));
  return {};
}
fit_expected<model::ModelEvaluator> evaluator(const spec::LatentStructure& pt,
                                             const model::MatrixRep& rep) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return std::unexpected(error(ev.error().detail));
  return std::move(*ev);
}
NewtonDerivatives point(const optim::ScalarProblem& problem,
                        const Eigen::VectorXd& theta, double n, double multiplier,
                        NewtonObjectiveKind kind) {
  NewtonDerivatives d;
  d.theta = theta;
  d.n_obs = n;
  d.native_to_total = multiplier;
  d.objective_kind = kind;
  if (!problem.f || theta.size() != problem.n_param || !theta.allFinite() ||
      !(n > 0) || !std::isfinite(n) || !(multiplier > 0) || !std::isfinite(multiplier)) {
    d.detail = "invalid objective dimensions or normalization";
    return d;
  }
  d.gradient = Eigen::VectorXd::Zero(theta.size());
  d.objective = problem.f(theta, d.gradient) * multiplier / n;
  d.gradient *= multiplier;
  if (!std::isfinite(d.objective) || d.gradient.size() != theta.size() ||
      !d.gradient.allFinite()) {
    d.detail = "objective or gradient unavailable at theta";
    return d;
  }
  d.status = NewtonAccuracyStatus::Available;
  return d;
}
NewtonAudit finish(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                   NewtonDerivatives d, const NewtonAdapterOptions& opts) {
  return audit_newton_derivatives(pt, rep, std::move(d), opts.domain,
                                   opts.accuracy, opts.bounds, opts.active_bound_tol);
}
NewtonDerivatives ls_derivatives(const optim::GmmProblem& problem,
                                 const Eigen::VectorXd& theta, double n,
                                 NewtonObjectiveKind kind,
                                 const NewtonAdapterOptions& opts) {
  auto scalar = optim::scalarize(problem);
  auto d = opts.gauss_newton ? point(scalar, theta, n, n, kind)
      : evaluate_newton_objective(scalar, theta, n, n, kind, opts.differences);
  if (opts.gauss_newton) d.curvature_kind = NewtonCurvatureKind::GaussNewton;
  if (d.status != NewtonAccuracyStatus::Available) return d;
  auto r = problem.r(theta);
  auto J = problem.J(theta);
  if (!r || !J || J->cols() != theta.size() || J->rows() != r->size() ||
      !r->allFinite() || !J->allFinite()) {
    d.status = NewtonAccuracyStatus::Unavailable;
    d.detail = "LS residual or Jacobian unavailable";
    return d;
  }
  d.whitened_residual = std::move(*r);
  d.whitened_jacobian = std::move(*J);
  if (opts.gauss_newton)
    d.hessian = n * d.whitened_jacobian.transpose() * d.whitened_jacobian;
  return d;
}

optim::ScalarProblem fiml_scalar(const model::ModelEvaluator& ev,
                                 const data::RawData& raw, const fiml::FIMLPack& pack) {
  optim::ScalarProblem prob;
  prob.n_param = static_cast<Eigen::Index>(ev.param_locations().size());
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.f = [&ev, &raw, &pack](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    auto e = ev.evaluate(x, true, true);
    if (!e) return std::numeric_limits<double>::infinity();
    auto vg = fiml::FIML{}.value_gradient(raw, pack.cache, e->moments, e->J_sigma, e->J_mu);
    if (!vg) return std::numeric_limits<double>::infinity();
    g = 0.5 * vg->gradient;
    return 0.5 * vg->value;
  };
  return prob;
}
fit_expected<void> prepare_fiml(spec::LatentStructure& pt, const model::MatrixRep& rep,
                                const data::RawData& raw, const fiml::FIMLPack& pack) {
  if (auto ok = fiml::validate_fiml_fixed_x_missing_policy(pt, raw); !ok)
    return std::unexpected(ok.error());
  return resolve_fixed_x_from_sample(pt, rep, pack.start_stats);
}
} // namespace

NewtonDerivatives evaluate_newton_moment_quadratic(
    const model::ModelEvaluator& ev, const SampleStats& sample,
    const Eigen::VectorXd& theta, const gmm::Weight& weight) {
  NewtonDerivatives d;
  d.theta = theta;
  d.objective_kind = NewtonObjectiveKind::LeastSquares;
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  d.metric_kind = NewtonMetricKind::Sandwich;
  auto problem = gmm::residuals(ev, sample, theta, weight);
  if (!problem) {
    d.detail = problem.error().detail;
    return d;
  }
  const double n = total_n(sample.n_obs);
  d = point(optim::scalarize(*problem), theta, n, n, NewtonObjectiveKind::LeastSquares);
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  d.metric_kind = NewtonMetricKind::Sandwich;
  if (d.status != NewtonAccuracyStatus::Available) return d;
  for (const auto& l : ev.param_locations()) {
    if (l.row < 0 || l.col < 0) {
      // Not a model-matrix cell: the closed-form second derivatives do not
      // cover it, so no analytic Hessian exists for this objective.
      d.status = NewtonAccuracyStatus::Unsupported;
      d.detail = "a free parameter is not a model-matrix cell";
      return d;
    }
  }
  d.status = NewtonAccuracyStatus::Unavailable;
  auto H = gmm::moment_quadratic_hessian(ev, sample, theta, weight);
  if (!H) {
    d.detail = H.error().detail;
    return d;
  }
  auto Omega = gmm::moment_quadratic_nt_gradient_variance(ev, sample, theta, weight);
  if (!Omega) {
    d.detail = Omega.error().detail;
    return d;
  }
  auto r = problem->r(theta);
  auto J = problem->J(theta);
  if (!r || !J || J->cols() != theta.size() || J->rows() != r->size()) {
    d.detail = "LS residual or Jacobian unavailable";
    return d;
  }
  d.whitened_residual = std::move(*r);
  d.whitened_jacobian = std::move(*J);
  d.hessian = std::move(*H);
  d.metric = std::move(*Omega);
  d.status = NewtonAccuracyStatus::Available;
  return d;
}

NewtonDerivatives evaluate_newton_objective(
    const optim::ScalarProblem& problem, const Eigen::VectorXd& theta,
    double n, double multiplier, NewtonObjectiveKind kind, NewtonDifferenceOptions opts) {
  auto d = point(problem, theta, n, multiplier, kind);
  d.curvature_kind = NewtonCurvatureKind::GradientDifference;
  d.difference_relative_step = opts.relative_step;
  d.difference_relative_tolerance = opts.max_relative_error;
  d.difference_max_shrink = opts.max_shrink;
  if (d.status != NewtonAccuracyStatus::Available) return d;
  d.status = NewtonAccuracyStatus::Unavailable;
  if (!(opts.relative_step > 0) || !std::isfinite(opts.relative_step) ||
      !(opts.max_relative_error >= 0) || !std::isfinite(opts.max_relative_error) ||
      opts.max_shrink < 0 || opts.max_shrink > 60 ||
      (opts.parameter_scales.size() &&
       (opts.parameter_scales.size() != theta.size() ||
        !opts.parameter_scales.allFinite() || (opts.parameter_scales.array() <= 0).any()))) {
    d.detail = "invalid Hessian difference controls";
    return d;
  }
  const Eigen::Index p = theta.size();
  d.hessian = Eigen::MatrixXd::Zero(p, p);
  d.difference_steps = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd coarse = d.hessian;
  auto gradient = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = Eigen::VectorXd::Zero(p);
    const double value = problem.f(x, g);
    g *= multiplier;
    return std::isfinite(value) && g.size() == p && g.allFinite();
  };
  for (Eigen::Index j = 0; j < p; ++j) {
    const double scale = opts.parameter_scales.size() ? opts.parameter_scales(j)
                                                     : std::max(1.0, std::abs(theta(j)));
    double h = opts.relative_step * scale;
    bool evaluated = false;
    for (int attempt = 0; attempt <= opts.max_shrink; ++attempt, h *= 0.5) {
      Eigen::VectorXd xp = theta, xm = theta, fp = theta, fm = theta;
      xp(j) += h; xm(j) -= h; fp(j) += 0.5 * h; fm(j) -= 0.5 * h;
      if (fp(j) == theta(j) || fm(j) == theta(j)) break;
      Eigen::VectorXd gp, gm, gfp, gfm;
      if (!gradient(xp, gp) || !gradient(xm, gm) ||
          !gradient(fp, gfp) || !gradient(fm, gfm)) continue;
      coarse.col(j) = (gp - gm) / (2 * h);
      d.hessian.col(j) = (gfp - gfm) / h;
      d.difference_steps(j) = 0.5 * h;
      evaluated = true;
      break;
    }
    if (!evaluated) {
      d.detail = "central Hessian probes unavailable in coordinate " + std::to_string(j);
      return d;
    }
  }
  // Check before symmetrization so inconsistent mixed derivatives are visible.
  d.hessian_relative_error = std::max((d.hessian - coarse).norm(),
      (d.hessian - d.hessian.transpose()).norm()) / std::max(1.0, d.hessian.norm());
  d.hessian = (0.5 * (d.hessian + d.hessian.transpose())).eval();
  if (!d.hessian.allFinite() || !std::isfinite(d.hessian_relative_error) ||
      d.hessian_relative_error > opts.max_relative_error) {
    d.detail = "Hessian step comparison or symmetry check failed";
    return d;
  }
  d.status = NewtonAccuracyStatus::Available;
  return d;
}

fit_expected<NewtonAudit> audit_newton_objective(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const optim::ScalarProblem& problem, const Eigen::VectorXd& theta,
    double n, double multiplier, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, false); !ok) return std::unexpected(ok.error());
  return finish(pt, rep, evaluate_newton_objective(problem, theta, n, multiplier,
      NewtonObjectiveKind::Supplied, opts.differences), opts);
}

fit_expected<NewtonAudit> audit_newton_gmm(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, const gmm::Weight& weight, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, true); !ok) return std::unexpected(ok.error());
  if (auto ok = resolve_fixed_x_from_sample(pt, rep, sample); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  if (!opts.gauss_newton)
    return finish(pt, rep, evaluate_newton_moment_quadratic(*ev, sample, theta, weight), opts);
  auto problem = gmm::residuals(*ev, sample, theta, weight);
  if (!problem) return std::unexpected(problem.error());
  return finish(pt, rep, ls_derivatives(*problem, theta, total_n(sample.n_obs),
      NewtonObjectiveKind::LeastSquares, opts), opts);
}

fit_expected<NewtonAudit> audit_newton_ml2s(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const fiml::SaturatedMoments& stage1, const Eigen::VectorXd& theta,
    fiml::TwoStageWeight kind, fiml::TwoStageDlsOptions dls, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, kind != fiml::TwoStageWeight::Nt); !ok)
    return std::unexpected(ok.error());
  switch (kind) {
    case fiml::TwoStageWeight::Nt:
    case fiml::TwoStageWeight::Uls:
    case fiml::TwoStageWeight::Dwls:
    case fiml::TwoStageWeight::Adf:
    case fiml::TwoStageWeight::Dls: break;
    default: return std::unexpected(error("unknown Stage-2 weight kind"));
  }
  const auto blocks = stage1.cov.size();
  if (blocks == 0 || stage1.mean.size() != blocks || stage1.n_obs.size() != blocks)
    return std::unexpected(error("inconsistent Stage-1 block counts"));
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto p = stage1.cov[b].rows();
    if (p == 0 || stage1.cov[b].cols() != p || stage1.mean[b].size() != p ||
        stage1.n_obs[b] <= 0 || !stage1.cov[b].allFinite() || !stage1.mean[b].allFinite())
      return std::unexpected(error("invalid Stage-1 moments"));
  }
  SampleStats sample;
  sample.S = stage1.cov;
  sample.mean = stage1.mean;
  sample.n_obs = stage1.n_obs;
  if (auto ok = resolve_fixed_x_from_sample(pt, rep, sample); !ok) return std::unexpected(ok.error());
  if (kind == fiml::TwoStageWeight::Nt)
    return finish(pt, rep, evaluate_newton_ml(pt, rep, sample, theta), opts);
  auto weight = fiml::two_stage_stage2_weight_structured(stage1, kind, dls);
  if (!weight) return std::unexpected(error(weight.error().detail));
  return audit_newton_gmm(std::move(pt), rep, sample, theta, *weight, std::move(opts));
}

fit_expected<NewtonAudit> audit_newton_uls(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, NewtonAdapterOptions opts) {
  return audit_newton_gmm(std::move(pt), rep, sample, theta, {}, std::move(opts));
}
fit_expected<NewtonAudit> audit_newton_wls(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, const gmm::Weight& weight, NewtonAdapterOptions opts) {
  if (weight.empty()) return std::unexpected(error("WLS requires an explicit weight"));
  return audit_newton_gmm(std::move(pt), rep, sample, theta, weight, std::move(opts));
}
fit_expected<NewtonAudit> audit_newton_snlls(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, const gmm::Weight& weight, NewtonAdapterOptions opts) {
  return audit_newton_gmm(std::move(pt), rep, sample, theta, weight, std::move(opts));
}
fit_expected<NewtonAudit> audit_newton_gls(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, true); !ok) return std::unexpected(ok.error());
  if (auto ok = resolve_fixed_x_from_sample(pt, rep, sample); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  auto weight = gmm::normal_theory_weight(*ev, sample, theta);
  if (!weight) return std::unexpected(weight.error());
  return audit_newton_gmm(std::move(pt), rep, sample, theta, *weight, std::move(opts));
}

fit_expected<NewtonAudit> audit_newton_fiml(
    spec::LatentStructure pt, const model::MatrixRep& rep, const data::RawData& raw,
    const fiml::FIMLPack& pack, const Eigen::VectorXd& theta, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, false); !ok) return std::unexpected(ok.error());
  if (auto ok = prepare_fiml(pt, rep, raw, pack); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  const double n = static_cast<double>(pack.cache.n_total);
  auto d = point(fiml_scalar(*ev, raw, pack), theta, n, n, NewtonObjectiveKind::Fiml);
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  if (d.status == NewtonAccuracyStatus::Available) {
    Estimates at; at.theta = theta;
    auto info = fiml::fiml_observed_information(pt, rep, raw, at, pack);
    if (info && info->rows() == theta.size() && info->cols() == theta.size() && info->allFinite()) {
      d.hessian = std::move(*info);
    } else {
      d.status = NewtonAccuracyStatus::Unavailable;
      d.detail = info ? "invalid FIML observed information" : info.error().detail;
    }
  }
  return finish(pt, rep, std::move(d), opts);
}
namespace {
NewtonDerivatives ordinal_derivatives(const OrdinalLsObjective& original,
                                      const fit_expected<OrdinalNewtonParts>& parts,
                                      const Eigen::VectorXd& theta, double n,
                                      NewtonObjectiveKind kind) {
  auto d = point(optim::scalarize(original.problem), theta, n, n, kind);
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  d.metric_kind = NewtonMetricKind::Sandwich;
  if (d.status != NewtonAccuracyStatus::Available) return d;
  d.status = NewtonAccuracyStatus::Unavailable;
  if (!parts) {
    d.detail = parts.error().detail;
    return d;
  }
  auto r = original.problem.r(theta);
  auto J = original.problem.J(theta);
  if (!r || !J || J->cols() != theta.size() || J->rows() != r->size() ||
      parts->hessian.rows() != theta.size() || !parts->hessian.allFinite() ||
      parts->gradient_variance.rows() != theta.size() ||
      !parts->gradient_variance.allFinite()) {
    d.detail = "ordinal Hessian, gradient variance, residual or Jacobian unavailable";
    return d;
  }
  d.whitened_residual = std::move(*r);
  d.whitened_jacobian = std::move(*J);
  d.hessian = parts->hessian;
  d.metric = parts->gradient_variance;
  d.status = NewtonAccuracyStatus::Available;
  return d;
}
}  // namespace

fit_expected<NewtonAudit> audit_newton_ordinal(
    spec::LatentStructure pt, const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const Eigen::VectorXd& theta, OrdinalWeightKind weights,
    OrdinalParameterization parameterization, NewtonAdapterOptions opts) {
  Estimates at; at.theta = theta;
  auto original = ordinal_ls_objective(pt, rep, stats, at, weights, parameterization);
  if (!original) return std::unexpected(original.error());
  if (auto ok = validate(original->pt, theta, opts, true); !ok) return std::unexpected(ok.error());
  if (!opts.gauss_newton) {
    auto parts = ordinal_ls_newton_parts(std::move(pt), rep, stats, theta, weights, parameterization);
    return finish(original->pt, rep, ordinal_derivatives(*original, parts, theta,
        total_n(stats.n_obs), NewtonObjectiveKind::OrdinalLeastSquares), opts);
  }
  return finish(original->pt, rep, ls_derivatives(original->problem, theta, total_n(stats.n_obs),
      NewtonObjectiveKind::OrdinalLeastSquares, opts), opts);
}
fit_expected<NewtonAudit> audit_newton_mixed_ordinal(
    spec::LatentStructure pt, const model::MatrixRep& rep, const data::MixedOrdinalStats& stats,
    const Eigen::VectorXd& theta, OrdinalWeightKind weights,
    OrdinalParameterization parameterization, NewtonAdapterOptions opts) {
  Estimates at; at.theta = theta;
  auto original = mixed_ordinal_ls_objective(pt, rep, stats, at, weights, parameterization);
  if (!original) return std::unexpected(original.error());
  if (auto ok = validate(original->pt, theta, opts, true); !ok) return std::unexpected(ok.error());
  if (!opts.gauss_newton) {
    auto parts = mixed_ordinal_ls_newton_parts(std::move(pt), rep, stats, theta, weights, parameterization);
    return finish(original->pt, rep, ordinal_derivatives(*original, parts, theta,
        total_n(stats.n_obs), NewtonObjectiveKind::MixedOrdinalLeastSquares), opts);
  }
  return finish(original->pt, rep, ls_derivatives(original->problem, theta, total_n(stats.n_obs),
      NewtonObjectiveKind::MixedOrdinalLeastSquares, opts), opts);
}
fit_expected<NewtonAudit> audit_newton_catml(
    spec::LatentStructure pt, const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const Eigen::VectorXd& theta, NewtonAdapterOptions opts) {
  if (auto ok = prepare_ordinal_delta_partable(pt, stats, nullptr); !ok) return std::unexpected(ok.error());
  if (auto ok = validate(pt, theta, opts, false); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  data::SampleStats sample;
  sample.S = stats.R;
  sample.n_obs = stats.n_obs;
  auto problem = ml_objective(*ev, sample, model::MomentTarget::Correlation);
  if (!problem) return std::unexpected(problem.error());
  const double n = total_n(stats.n_obs);
  auto d = evaluate_newton_objective(*problem, theta, n, n, NewtonObjectiveKind::CatMl, opts.differences);
  for (std::size_t r = 0; r < pt.size(); ++r)
    if (pt.op[r] == parse::Op::Threshold && pt.free[r] > 0)
      d.fixed_coordinates.push_back(pt.free[r] - 1);
  return finish(pt, rep, std::move(d), opts);
}
fit_expected<NewtonAudit> audit_newton_twolevel(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::ClusterSampleStats& stats, const Eigen::VectorXd& theta, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, false); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  auto problem = twolevel::twolevel_ml_objective(*ev, stats);
  if (!problem) return std::unexpected(problem.error());
  double n = 0;
  for (const auto& group : stats.groups) n += static_cast<double>(group.n_within);
  // twolevel_ml_objective already returns TOTAL negative log likelihood.
  return finish(pt, rep, evaluate_newton_objective(*problem, theta, n, 1.0,
      NewtonObjectiveKind::TwoLevelMl, opts.differences), opts);
}
fit_expected<NewtonAudit> audit_newton_penalized_ml(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, MultiInfoPenaltyOptions penalty, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, false); !ok) return std::unexpected(ok.error());
  if (auto ok = resolve_fixed_x_from_sample(pt, rep, sample); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  auto base = ml_objective(*ev, sample);
  if (!base) return std::unexpected(base.error());
  auto layout = multiinfo_penalty_layout(*ev, theta, penalty.target);
  if (!layout) return std::unexpected(layout.error());
  auto weight = multiinfo_penalty_weight(penalty);
  if (!weight) return std::unexpected(weight.error());
  const double n = total_n(sample.n_obs);
  auto problem = multiinfo_penalized_problem(*base, *layout, *ev, *weight, n);
  auto d = point(problem, theta, n, n, NewtonObjectiveKind::PenalizedMl);
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  d.penalty_weight = *weight;
  if (d.status == NewtonAccuracyStatus::Available) {
    Estimates at; at.theta = theta;
    auto info = inference::information_observed_analytic(pt, rep, sample, at);
    auto curvature = multiinfo_penalty_hessian(*layout, *ev, theta);
    if (info && curvature && info->rows() == theta.size()) {
      d.hessian = *info - *weight * (*curvature);
    } else {
      d.status = NewtonAccuracyStatus::Unavailable;
      d.detail = !info ? info.error().detail : curvature ? "Hessian dimension mismatch"
                                                         : curvature.error().detail;
    }
  }
  return finish(pt, rep, std::move(d), opts);
}
fit_expected<NewtonAudit> audit_newton_penalized_fiml(
    spec::LatentStructure pt, const model::MatrixRep& rep, const data::RawData& raw,
    const fiml::FIMLPack& pack, const Eigen::VectorXd& theta,
    MultiInfoPenaltyOptions penalty, NewtonAdapterOptions opts) {
  if (auto ok = validate(pt, theta, opts, false); !ok) return std::unexpected(ok.error());
  if (auto ok = prepare_fiml(pt, rep, raw, pack); !ok) return std::unexpected(ok.error());
  auto ev = evaluator(pt, rep);
  if (!ev) return std::unexpected(ev.error());
  auto base = fiml_scalar(*ev, raw, pack);
  auto layout = multiinfo_penalty_layout(*ev, theta, penalty.target);
  if (!layout) return std::unexpected(layout.error());
  auto weight = multiinfo_penalty_weight(penalty);
  if (!weight) return std::unexpected(weight.error());
  const double n = static_cast<double>(pack.cache.n_total);
  auto problem = multiinfo_penalized_problem(base, *layout, *ev, *weight, n);
  auto d = point(problem, theta, n, n, NewtonObjectiveKind::PenalizedFiml);
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  d.penalty_weight = *weight;
  if (d.status == NewtonAccuracyStatus::Available) {
    Estimates at; at.theta = theta;
    auto info = fiml::fiml_observed_information(pt, rep, raw, at, pack);
    auto curvature = multiinfo_penalty_hessian(*layout, *ev, theta);
    if (info && curvature && info->rows() == theta.size()) {
      d.hessian = *info - *weight * (*curvature);
    } else {
      d.status = NewtonAccuracyStatus::Unavailable;
      d.detail = !info ? info.error().detail : curvature ? "Hessian dimension mismatch"
                                                         : curvature.error().detail;
    }
  }
  return finish(pt, rep, std::move(d), opts);
}

} // namespace magmaan::estimate::frontier
