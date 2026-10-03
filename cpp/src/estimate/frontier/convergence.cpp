#include "magmaan/estimate/frontier/convergence.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/evaluate.hpp"

namespace magmaan::estimate::frontier {
namespace {
FitError error(const char* detail) { return {FitError::Kind::NumericIssue, detail}; }
bool nonnegative(double x) { return std::isfinite(x) && x >= 0; }
fit_expected<void> validate(const spec::LatentStructure& pt, const Eigen::VectorXd& theta,
                            const ConvergenceRequest& r) {
  if (theta.size() != pt.n_free()) return std::unexpected(error("convergence: full-theta size mismatch"));
  if (r.bounds.lower.size() != r.bounds.upper.size() ||
      (!r.bounds.empty() && (r.bounds.lower.size() != theta.size() ||
       r.bounds.lower.array().isNaN().any() || r.bounds.upper.array().isNaN().any() ||
       (r.bounds.lower.array() > r.bounds.upper.array()).any())))
    return std::unexpected(error("convergence: invalid bounds"));
  const auto& g = r.geometry;
  const auto& d = r.diagnostics;
  if (!nonnegative(g.stationarity_tol) || !nonnegative(g.covariance_eigen_tol) ||
      !nonnegative(g.equality_tol) || !nonnegative(g.active_bound_tol) ||
      !nonnegative(g.projection_tol) || g.projection_max_iter < 0 ||
      !nonnegative(d.active_bound_tol) || !nonnegative(d.lin_eq_residual_tol) ||
      !nonnegative(d.nl_eq_residual_tol) || !nonnegative(d.covariance_eigen_tol) ||
      !nonnegative(d.variance_tol) || !nonnegative(d.correlation_tol))
    return std::unexpected(error("convergence: invalid collection tolerances"));
  return {};
}
fit_expected<ConvergenceReport> collect(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    NewtonAudit audit, ConvergenceRequest request, std::optional<double> reported) {
  const auto& d = audit.derivatives;
  if (auto ok = validate(pt, d.theta, request); !ok) return std::unexpected(ok.error());
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return std::unexpected(error("convergence: model evaluator construction failed"));
  auto con = build_eq_constraints(pt, true);
  if (!con) return std::unexpected(error(con.error().detail.c_str()));
  const auto nl = build_nl_constraints(pt);
  ConvergenceReport report;
  report.request = std::move(request);
  report.evidence = finalize_fit_diagnostics(d.theta, pt, *ev, *con, nl,
      report.request.bounds, false, report.request.diagnostics);
  Eigen::VectorXd gradient = d.gradient;
  if (d.n_obs > 0 && std::isfinite(d.n_obs)) gradient /= d.n_obs;
  else gradient = Eigen::VectorXd::Constant(d.theta.size(), std::numeric_limits<double>::quiet_NaN());
  Bounds first_order_bounds = report.request.bounds;
  if (!d.fixed_coordinates.empty()) {
    if (first_order_bounds.empty()) {
      first_order_bounds.lower = Eigen::VectorXd::Constant(d.theta.size(), -std::numeric_limits<double>::infinity());
      first_order_bounds.upper = -first_order_bounds.lower;
    }
    for (auto index : d.fixed_coordinates) {
      if (index < 0 || index >= d.theta.size())
        return std::unexpected(error("convergence: invalid held coordinate"));
      // Preserve any original box violation rather than masking it by fixing
      // the coordinate to an out-of-domain value.
      first_order_bounds.lower[index] = std::max(first_order_bounds.lower[index], d.theta[index]);
      first_order_bounds.upper[index] = std::min(first_order_bounds.upper[index], d.theta[index]);
    }
  }
  audit_full_model_fit(report.evidence, d.theta, gradient,
      reported.value_or(d.objective), d.objective, pt, *ev, *con, nl,
      first_order_bounds, report.request.domain, report.request.geometry);
  report.evidence.objective.reported_available = reported.has_value();
  if (!reported) {
    report.evidence.objective.reported = std::numeric_limits<double>::quiet_NaN();
    report.evidence.objective.consistent = false;
  }
  if (report.request.newton) report.evidence.newton_accuracy = audit.diagnostics;
  report.computations = std::move(audit);
  return report;
}
NewtonDerivatives point(const optim::ScalarProblem& p, const Eigen::VectorXd& theta,
                        double n, double multiplier) {
  NewtonDerivatives d;
  d.theta = theta; d.n_obs = n; d.native_to_total = multiplier;
  d.gradient = Eigen::VectorXd::Zero(theta.size());
  d.objective = p.f(theta, d.gradient) * multiplier / n;
  d.gradient *= multiplier;
  d.status = std::isfinite(d.objective) && d.gradient.size() == theta.size() && d.gradient.allFinite()
      ? NewtonAccuracyStatus::Available : NewtonAccuracyStatus::Unavailable;
  return d;
}
} // namespace

fit_expected<ConvergenceReport> audit_convergence(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const optim::ScalarProblem& problem, const Eigen::VectorXd& theta,
    double n, double multiplier, ConvergenceRequest request, std::optional<double> reported) {
  if (auto ok = validate(pt, theta, request); !ok) return std::unexpected(ok.error());
  if (!problem.f || problem.n_param != theta.size() || !std::isfinite(n) || n <= 0 ||
      !std::isfinite(multiplier) || multiplier <= 0)
    return std::unexpected(error("convergence: invalid objective or normalization"));
  NewtonAudit a;
  auto d = request.newton ? evaluate_newton_objective(problem, theta, n, multiplier,
      NewtonObjectiveKind::Supplied, request.differences) : point(problem, theta, n, multiplier);
  if (request.newton)
    a = audit_newton_derivatives(pt, rep, std::move(d), request.domain,
        request.curvature, request.bounds, request.diagnostics.active_bound_tol);
  else a.derivatives = std::move(d);
  if (reported) *reported *= multiplier / n;
  return collect(pt, rep, std::move(a), std::move(request), reported);
}

fit_expected<ConvergenceReport> audit_convergence_ml(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, ConvergenceRequest request, std::optional<double> reported) {
  if (auto ok = validate(pt, theta, request); !ok) return std::unexpected(ok.error());
  if (auto ok = resolve_fixed_x_from_sample(pt, rep, sample); !ok) return std::unexpected(ok.error());
  if (request.newton) {
    auto a = audit_newton_derivatives(pt, rep, evaluate_newton_ml(pt, rep, sample, theta),
        request.domain, request.curvature, request.bounds, request.diagnostics.active_bound_tol);
    return collect(pt, rep, std::move(a), std::move(request), reported);
  }
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return std::unexpected(error("convergence: model evaluator construction failed"));
  auto p = ml_objective(*ev, sample);
  if (!p) return std::unexpected(p.error());
  const double n = std::accumulate(sample.n_obs.begin(), sample.n_obs.end(), 0.0);
  auto report = audit_convergence(pt, rep, *p, theta, n, n, std::move(request), reported);
  if (report) report->computations.derivatives.objective_kind = NewtonObjectiveKind::CompleteDataMl;
  return report;
}

fit_expected<ConvergenceReport> audit_convergence(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    NewtonAudit audit, DiagnosticsOptions diagnostics, GeometricStationarityOptions geometry,
    std::optional<double> reported) {
  ConvergenceRequest r;
  r.newton = true; r.domain = audit.geometry.domain; r.bounds = audit.bounds;
  r.curvature = audit.options; r.diagnostics = diagnostics; r.geometry = geometry;
  r.diagnostics.active_bound_tol = audit.active_bound_tol;
  r.differences.relative_step = audit.derivatives.difference_relative_step;
  r.differences.max_relative_error = audit.derivatives.difference_relative_tolerance;
  r.differences.max_shrink = audit.derivatives.difference_max_shrink;
  return collect(pt, rep, std::move(audit), std::move(r), reported);
}
fit_expected<ConvergenceReport> audit_convergence_covariance(
    spec::LatentStructure pt,const model::MatrixRep& rep,const SampleStats& sample,
    const Eigen::VectorXd& theta,Estimator estimator,ConvergenceRequest request,
    std::optional<double> reported) {
  request.newton=true;
  if(auto ok=validate(pt,theta,request); !ok) return std::unexpected(ok.error());
  if(auto ok=resolve_fixed_x_from_sample(pt,rep,sample); !ok) return std::unexpected(ok.error());
  NewtonAudit audit;
  if(estimator==Estimator::ML) {
    audit=audit_newton_derivatives(pt,rep,evaluate_newton_ml(pt,rep,sample,theta),
        request.domain,request.curvature,request.bounds,request.diagnostics.active_bound_tol);
  } else if(estimator==Estimator::ULS) {
    NewtonAdapterOptions options;
    options.domain=request.domain; options.accuracy=request.curvature;
    options.bounds=request.bounds; options.active_bound_tol=request.diagnostics.active_bound_tol;
    auto a=audit_newton_uls(pt,rep,sample,theta,options);
    if(!a) return std::unexpected(a.error());
    audit=std::move(*a);
  } else return std::unexpected(error("construction-aware covariance report supports ULS/ML"));
  audit.input_errors=newton_input_error_bounds(pt,rep,sample,theta,audit,estimator);
  return collect(pt,rep,std::move(audit),std::move(request),reported);
}

ConvergenceAssessment assess_convergence(const ConvergenceReport& r, ConvergencePolicy p) {
  auto evidence = r.evidence;
  if (r.request.newton && p.kind != ConvergencePolicyKind::Compatibility)
    evidence.newton_accuracy = assess_newton_accuracy(r.computations, p.newton);
  if(p.require_verified_inputs && r.computations.input_errors) {
    const auto interval=newton_input_distance_interval(r.computations,*r.computations.input_errors,p.newton.budget);
    return assess_convergence(evidence,p,&interval);
  }
  return assess_convergence(evidence, p);
}
} // namespace magmaan::estimate::frontier
