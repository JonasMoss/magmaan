#include <doctest/doctest.h>
#include <cmath>

#include "magmaan/estimate/frontier/convergence.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using namespace magmaan;
namespace cf = magmaan::estimate::frontier;
namespace {
struct Model { spec::LatentStructure pt; model::MatrixRep rep; };
Model scalar_model() {
  auto parsed = parse::Parser::parse("x ~~ x");
  REQUIRE(parsed.has_value());
  spec::BuildOptions options; options.fixed_x = false;
  auto pt = spec::build(*parsed, options);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}
optim::ScalarProblem quadratic(int& calls) {
  optim::ScalarProblem p; p.n_param = 1;
  p.f = [&calls](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    ++calls; g = x.array() - 1.; return .5 * g.squaredNorm();
  };
  return p;
}
} // namespace

TEST_CASE("Convergence policy: requested Newton never falls back to first order") {
  auto m = scalar_model(); int calls = 0;
  auto p = quadratic(calls);
  auto report = cf::audit_convergence(m.pt, m.rep, p, Eigen::VectorXd::Ones(1), 100, 100);
  REQUIRE(report.has_value());
  CHECK(cf::assess_convergence(*report).status == estimate::FitCheck::Passed);
  auto required = cf::newton_convergence_policy();
  auto absent = cf::assess_convergence(*report, required);
  CHECK(absent.status == estimate::FitCheck::Unchecked);
  CHECK(absent.newton.required);
  CHECK(absent.newton.status == estimate::FitCheck::Unchecked);
  CHECK(absent.first_order.status == estimate::FitCheck::Passed);
  CHECK_FALSE(absent.first_order.required);
  required.require_objective_consistency = true;
  auto missing_value = cf::assess_convergence(*report, required);
  CHECK(missing_value.objective_consistency.status == estimate::FitCheck::Unchecked);
  CHECK_FALSE(report->evidence.objective.reported_available);
  CHECK(calls == 1);
}

TEST_CASE("Convergence report: reassessment retains computations and separates checks") {
  auto m = scalar_model(); int calls = 0;
  auto p = quadratic(calls);
  cf::ConvergenceRequest request; request.newton = true;
  auto report = cf::audit_convergence(m.pt, m.rep, p,
      Eigen::VectorXd::Constant(1, 1.0005), 100, 100, request, .5 * .0005 * .0005);
  REQUIRE(report.has_value());
  REQUIRE(report->computations.solution.status == estimate::NewtonAccuracyStatus::Available);
  const auto H = report->computations.derivatives.hessian;
  const int collected_calls = calls;
  auto policy = cf::newton_convergence_policy();
  policy.require_objective_consistency = true;
  CHECK(cf::assess_convergence(*report, policy).status == estimate::FitCheck::Passed);
  policy.newton.budget = .001;
  CHECK(cf::assess_convergence(*report, policy).status == estimate::FitCheck::Failed);
  policy.newton.budget = .01;
  policy.stationarity = cf::RequiredStationarity::Both;
  policy.stationarity_tol = 1e-5;
  auto both = cf::assess_convergence(*report, policy);
  CHECK(both.newton.status == estimate::FitCheck::Passed);
  CHECK(both.first_order.status == estimate::FitCheck::Failed);
  CHECK(both.status == estimate::FitCheck::Failed);
  policy.stationarity = cf::RequiredStationarity::Newton;
  CHECK(cf::assess_convergence(*report, policy).status == estimate::FitCheck::Passed);
  CHECK(calls == collected_calls);
  CHECK(report->computations.derivatives.hessian.isApprox(H));
  CHECK(H(0, 0) == doctest::Approx(100));
  CHECK(report->computations.derivatives.gradient[0] == doctest::Approx(.05));
  auto bad_value = cf::audit_convergence(m.pt, m.rep, p,
      Eigen::VectorXd::Ones(1), 100, 100, request, 2.0);
  REQUIRE(bad_value.has_value());
  auto inconsistent = cf::assess_convergence(*bad_value, policy);
  CHECK(inconsistent.objective_consistency.status == estimate::FitCheck::Failed);
  CHECK(inconsistent.status == estimate::FitCheck::Failed);
}

TEST_CASE("Convergence report: box feasibility and constrained curvature remain distinct") {
  auto m = scalar_model(); int calls = 0; auto p = quadratic(calls);
  cf::ConvergenceRequest request; request.newton = true;
  request.bounds.lower = Eigen::VectorXd::Ones(1);
  request.bounds.upper = Eigen::VectorXd::Constant(1, 2);
  auto boundary = cf::audit_convergence(m.pt, m.rep, p,
      Eigen::VectorXd::Ones(1), 10, 10, request);
  REQUIRE(boundary.has_value());
  auto second = cf::assess_convergence(*boundary, cf::newton_convergence_policy());
  CHECK(second.feasibility.status == estimate::FitCheck::Passed);
  CHECK(second.newton.status == estimate::FitCheck::Passed);
  CHECK(second.status == estimate::FitCheck::Passed);
  CHECK(cf::assess_convergence(*boundary).status == estimate::FitCheck::Passed);
  auto outside = cf::audit_convergence(m.pt, m.rep, p,
      Eigen::VectorXd::Constant(1, .9), 10, 10, request);
  REQUIRE(outside.has_value());
  auto rejected = cf::assess_convergence(*outside, cf::newton_convergence_policy());
  CHECK(rejected.feasibility.status == estimate::FitCheck::Failed);
  CHECK(rejected.status == estimate::FitCheck::Failed);
  CHECK(outside->computations.derivatives.hessian.rows() == 1);
}

TEST_CASE("Convergence report: failed curvature probes remain unresolved with valid first order") {
  auto m = scalar_model(); int calls = 0; auto p = quadratic(calls);
  cf::ConvergenceRequest request; request.newton = true;
  request.differences.relative_step = -1;
  auto report = cf::audit_convergence(m.pt, m.rep, p, Eigen::VectorXd::Ones(1), 1, 1, request);
  REQUIRE(report.has_value());
  CHECK(cf::assess_convergence(*report, cf::newton_convergence_policy()).status == estimate::FitCheck::Unchecked);
  CHECK(cf::assess_convergence(*report).status == estimate::FitCheck::Passed);
  // Invalid numerical controls never trigger a weaker acceptance policy.
  CHECK_FALSE(cf::assess_convergence(*report, cf::newton_convergence_policy()).first_order.required);
}

TEST_CASE("Convergence: fit and post-fit ML share evidence and compatibility acceptance") {
  auto m = scalar_model();
  data::SampleStats s; s.S = {Eigen::MatrixXd::Constant(1, 1, 1.5)}; s.n_obs = {200};
  auto start = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(start.has_value());
  auto fit = estimate::fit_ml(m.pt, m.rep, s, *start);
  REQUIRE(fit.has_value());
  auto bounds = estimate::variance_bounds(m.pt);
  REQUIRE(bounds.has_value());
  auto evaluated = estimate::evaluate_at(m.pt, m.rep, s, fit->theta, estimate::Estimator::ML, {}, *bounds);
  REQUIRE(evaluated.has_value());
  REQUIRE(evaluated->diagnostics.newton_accuracy.checked);
  CHECK(estimate::fit_verdict(*evaluated).criterion == estimate::fit_verdict(*fit).criterion);
  CHECK(estimate::fit_verdict(*evaluated).status == estimate::fit_verdict(*fit).status);
  cf::ConvergenceRequest request; request.newton = true; request.bounds = *bounds;
  auto report = cf::audit_convergence_ml(m.pt, m.rep, s, fit->theta, request, fit->fmin);
  REQUIRE(report.has_value());
  for (auto policy : {cf::compatibility_convergence_policy(), cf::first_order_convergence_policy(), cf::newton_convergence_policy()}) {
    auto a = cf::assess_convergence(fit->diagnostics, policy);
    auto b = cf::assess_convergence(*report, policy);
    auto c = cf::assess_convergence(evaluated->diagnostics, policy);
    CHECK(a.status == b.status);
    CHECK(a.status == c.status);
  }
  CHECK(report->computations.derivatives.curvature_kind == cf::NewtonCurvatureKind::AnalyticObserved);
  // A retained PSD-domain computation must not certify another declared domain.
  report->evidence.stationarity_domain = estimate::StationarityDomain::Psd;
  CHECK(cf::assess_convergence(*report, cf::newton_convergence_policy()).newton.status == estimate::FitCheck::Unchecked);
}

TEST_CASE("Convergence: LS adapter artifacts compose without recomputing the Hessian") {
  auto m = scalar_model();
  data::SampleStats s; s.S = {Eigen::MatrixXd::Constant(1, 1, 1.5)}; s.n_obs = {200};
  auto a = cf::audit_newton_uls(m.pt, m.rep, s, Eigen::VectorXd::Constant(1, 1.5));
  REQUIRE(a.has_value());
  const Eigen::MatrixXd H = a->derivatives.hessian;
  auto report = cf::audit_convergence(m.pt, m.rep, std::move(*a));
  REQUIRE(report.has_value());
  CHECK(report->computations.derivatives.hessian.isApprox(H));
  CHECK(cf::assess_convergence(*report, cf::newton_convergence_policy()).status == estimate::FitCheck::Passed);
}

TEST_CASE("Convergence policy: nonpositive curvature is a failure rather than missing evidence") {
  auto m = scalar_model();
  optim::ScalarProblem p; p.n_param = 1;
  p.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 1. - x.array(); return -.5 * g.squaredNorm();
  };
  cf::ConvergenceRequest request; request.newton = true;
  auto report = cf::audit_convergence(m.pt, m.rep, p, Eigen::VectorXd::Ones(1), 10, 10, request);
  REQUIRE(report.has_value());
  CHECK(cf::assess_convergence(*report).status == estimate::FitCheck::Passed);
  CHECK(cf::assess_convergence(*report, cf::newton_convergence_policy()).status == estimate::FitCheck::Failed);
}

TEST_CASE("Convergence policy: guard thresholds can be relaxed from retained evidence") {
  auto parsed = parse::Parser::parse("x ~~ x\ny ~~ y");
  REQUIRE(parsed.has_value());
  spec::BuildOptions options; options.fixed_x = false;
  auto pt = spec::build(*parsed, options); REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  optim::ScalarProblem p; p.n_param = pt->n_free(); REQUIRE(p.n_param == 2);
  Eigen::Matrix2d H; H << 1, .9, .9, 1;
  int calls = 0;
  p.f = [&calls, H](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    ++calls; const Eigen::VectorXd delta = x.array() - 1.; g = H * delta;
    return .5 * delta.dot(g);
  };
  cf::ConvergenceRequest request; request.newton = true; request.curvature.max_condition = 2;
  auto report = cf::audit_convergence(*pt, *rep, p, Eigen::VectorXd::Ones(2), 10, 10, request);
  REQUIRE(report.has_value());
  REQUIRE(report->evidence.newton_accuracy.status == estimate::NewtonAccuracyStatus::IllConditioned);
  CHECK(std::isfinite(report->evidence.newton_accuracy.distance));
  const auto count = calls;
  auto policy = cf::newton_convergence_policy(); policy.newton.max_condition = 2;
  CHECK(cf::assess_convergence(*report, policy).status == estimate::FitCheck::Failed);
  policy.newton.max_condition = 100;
  CHECK(cf::assess_convergence(*report, policy).status == estimate::FitCheck::Passed);
  CHECK(cf::assess_convergence(report->evidence, policy).status == estimate::FitCheck::Passed);
  CHECK(calls == count);
}

TEST_CASE("Convergence policy: covariance admissibility is domain dependent") {
  auto m = scalar_model();
  optim::ScalarProblem p; p.n_param = 1;
  p.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = x.array() + 1.; return .5 * g.squaredNorm();
  };
  const auto theta = Eigen::VectorXd::Constant(1, -1);
  cf::ConvergenceRequest request;
  auto ambient = cf::audit_convergence(m.pt, m.rep, p, theta, 10, 10, request);
  REQUIRE(ambient.has_value());
  CHECK_FALSE(ambient->evidence.admissibility.admissible);
  CHECK(cf::assess_convergence(*ambient).status == estimate::FitCheck::Passed);
  request.domain = estimate::StationarityDomain::Psd;
  auto psd = cf::audit_convergence(m.pt, m.rep, p, theta, 10, 10, request);
  REQUIRE(psd.has_value());
  CHECK(cf::assess_convergence(*psd).feasibility.status == estimate::FitCheck::Failed);
  CHECK(cf::assess_convergence(*psd).status == estimate::FitCheck::Failed);
}
