#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/optim/nlopt_optimizer.hpp"

using magmaan::FitError;
using magmaan::optim::ConstrainedScalarProblem;
using magmaan::optim::NloptOptimizer;
using magmaan::optim::NloptAlgorithm;
using magmaan::optim::ScalarProblem;

// ============================================================================
// Behavioural tests for the NLopt SLSQP adapter — a gradient-based sequential
// quadratic programming optimizer with native box-bound support.
// ============================================================================

TEST_CASE("NloptOptimizer — minimizes a well-behaved quadratic") {
  Eigen::VectorXd c(3);  c << 1.0, -2.0, 0.5;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  NloptOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-8);
  CHECK((out->theta_hat - c).norm() < 1e-4);
}

TEST_CASE("NloptOptimizer — solves the Rosenbrock function") {
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g.resize(2);
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
  };
  NloptOptimizer opt;
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(f, x0);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-6);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 1e-3);
}

TEST_CASE("NloptOptimizer — lower bound is enforced (Heywood story)") {
  // Optimum at (-1,-1); both coordinates bounded below at 0 — SLSQP must park
  // the solution at the corner (0,0).
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(2);  target << -1.0, -1.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  NloptOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(2);  lb << 0.0, 0.0;
  Eigen::VectorXd ub(2);  ub << inf, inf;
  Eigen::VectorXd x0(2);  x0 << 0.5, 0.5;        // start feasible
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) >= -1e-9);             // bound holds
  CHECK(out->theta_hat(1) >= -1e-9);
  CHECK(out->theta_hat(0) < 1e-4);               // sat at the boundary
  CHECK(out->theta_hat(1) < 1e-4);
}

TEST_CASE("NloptOptimizer — bound size mismatch is an error value") {
  auto f = [](const Eigen::VectorXd&, Eigen::VectorXd&) { return 0.0; };
  NloptOptimizer opt;
  Eigen::VectorXd x0(3);  x0.setZero();
  Eigen::VectorXd lb(2);  lb.setZero();
  Eigen::VectorXd ub(2);  ub.setZero();
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

TEST_CASE("NloptOptimizer/SLSQP — enforces a nonlinear equality constraint") {
  ScalarProblem obj;
  obj.n_param = 2;
  obj.expand = [](const Eigen::VectorXd& x) { return x; };
  obj.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    Eigen::Vector2d target;
    target << 1.0, 1.0;
    grad = x - target;
    return 0.5 * (x - target).squaredNorm();
  };

  ConstrainedScalarProblem prob;
  prob.objective = obj;
  prob.n_constraint = 1;
  prob.constraint_lower = Eigen::VectorXd::Zero(1);
  prob.constraint_upper = Eigen::VectorXd::Zero(1);
  prob.h = [](const Eigen::VectorXd& x) {
    Eigen::VectorXd h(1);
    h(0) = x(0) - x(1) * x(1);
    return h;
  };
  prob.J_h = [](const Eigen::VectorXd& x) {
    Eigen::MatrixXd J(1, 2);
    J << 1.0, -2.0 * x(1);
    return J;
  };

  const double inf = std::numeric_limits<double>::infinity();
  Eigen::Vector2d x0;
  x0 << 0.25, 0.5;
  auto out = NloptOptimizer{}.minimize_constrained(
      prob, x0, Eigen::VectorXd::Constant(2, -inf),
      Eigen::VectorXd::Constant(2,  inf));
  REQUIRE(out.has_value());
  CHECK(std::abs(out->theta_hat(0) -
                 out->theta_hat(1) * out->theta_hat(1)) < 1e-7);
  CHECK(out->fmin < 0.1);
}

TEST_CASE("NloptOptimizer/SLSQP — constrained path accepts equal nonzero targets") {
  ScalarProblem obj;
  obj.n_param = 1;
  obj.expand = [](const Eigen::VectorXd& x) { return x; };
  obj.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    grad = Eigen::VectorXd::Constant(1, x(0));
    return 0.5 * x(0) * x(0);
  };

  ConstrainedScalarProblem prob;
  prob.objective = obj;
  prob.n_constraint = 1;
  prob.constraint_lower = Eigen::VectorXd::Constant(1, 2.0);
  prob.constraint_upper = Eigen::VectorXd::Constant(1, 2.0);
  prob.h = [](const Eigen::VectorXd& x) { return x; };
  prob.J_h = [](const Eigen::VectorXd&) {
    Eigen::MatrixXd J(1, 1);
    J(0, 0) = 1.0;
    return J;
  };

  const double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 0.0);
  auto out = NloptOptimizer{}.minimize_constrained(
      prob, x0, Eigen::VectorXd::Constant(1, -inf),
      Eigen::VectorXd::Constant(1,  inf));
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) == doctest::Approx(2.0).epsilon(1e-8));
}

TEST_CASE("NloptOptimizer/SLSQP — constrained path rejects inequalities") {
  ScalarProblem obj;
  obj.n_param = 1;
  obj.expand = [](const Eigen::VectorXd& x) { return x; };
  obj.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    grad = Eigen::VectorXd::Constant(1, x(0));
    return 0.5 * x(0) * x(0);
  };

  ConstrainedScalarProblem prob;
  prob.objective = obj;
  prob.n_constraint = 1;
  prob.constraint_lower = Eigen::VectorXd::Zero(1);
  prob.constraint_upper = Eigen::VectorXd::Constant(1, 1.0);
  prob.h = [](const Eigen::VectorXd& x) { return x; };
  prob.J_h = [](const Eigen::VectorXd&) {
    Eigen::MatrixXd J(1, 1);
    J(0, 0) = 1.0;
    return J;
  };

  const double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 0.0);
  auto out = NloptOptimizer{}.minimize_constrained(
      prob, x0, Eigen::VectorXd::Constant(1, -inf),
      Eigen::VectorXd::Constant(1,  inf));
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
  CHECK(out.error().detail.find("equality") != std::string::npos);
}

// ============================================================================
// Per-algorithm smoke tests for the four new NLopt backends. Each runs the
// same quadratic + Rosenbrock pair the SLSQP cases above use, plus the
// algorithm-specific kicker (BOBYQA bounds, etc.). The shared NloptOptimizer
// adapter is parameterised over `nlopt_algorithm`, so passing the algorithm
// enum to the constructor is the only difference per test.
// ============================================================================

namespace {

auto quadratic_objective(const Eigen::Vector3d& c) {
  return [c](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
}

auto rosenbrock_objective() {
  return [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g.resize(2);
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
  };
}

}  // namespace

// --- Powell BOBYQA (derivative-free quadratic-model TR) --------------------

TEST_CASE("NloptOptimizer/BOBYQA — minimizes a bounded quadratic without gradient") {
  // BOBYQA ignores any gradient the callback writes; the trampoline already
  // calls the objective with grad=nullptr for derivative-free algorithms.
  // We supply a callback that writes a gradient anyway — it should be
  // discarded, and BOBYQA should still find the optimum.
  Eigen::Vector3d c(1.0, -2.0, 0.5);
  NloptOptimizer opt({}, NloptAlgorithm::Bobyqa);
  // BOBYQA requires finite bounds.
  Eigen::VectorXd lb(3);  lb << -5.0, -5.0, -5.0;
  Eigen::VectorXd ub(3);  ub <<  5.0,  5.0,  5.0;
  auto out = opt.minimize(quadratic_objective(c), Eigen::VectorXd::Zero(3),
                          lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-6);
  CHECK((out->theta_hat - c).norm() < 1e-3);
}

TEST_CASE("NloptOptimizer/BOBYQA — solves Rosenbrock in a generous box") {
  NloptOptimizer opt({1000, 1e-12, 1e-8, 10}, NloptAlgorithm::Bobyqa);
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  Eigen::VectorXd lb(2);  lb << -10.0, -10.0;
  Eigen::VectorXd ub(2);  ub <<  10.0,  10.0;
  auto out = opt.minimize(rosenbrock_objective(), x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-3);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 5e-2);
}

// BOBYQA's empty-bounds rejection lives at the dispatcher (`optim::nlopt_bobyqa`)
// rather than the low-level adapter, because NLopt's BOBYQA tolerates
// ±HUGE_VAL bounds opportunistically — we don't want to rely on that
// undocumented tolerance. That rejection is tested in the optimizers_test
// file alongside the other dispatcher-level contracts.

// --- Nash truncated Newton (LD_TNEWTON_PRECOND_RESTART) --------------------

TEST_CASE("NloptOptimizer/TNEWTON — minimizes a well-behaved quadratic") {
  Eigen::Vector3d c(1.0, -2.0, 0.5);
  NloptOptimizer opt({}, NloptAlgorithm::Tnewton);
  auto out = opt.minimize(quadratic_objective(c), Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-8);
  CHECK((out->theta_hat - c).norm() < 1e-4);
}

TEST_CASE("NloptOptimizer/TNEWTON — solves the Rosenbrock function") {
  NloptOptimizer opt({1000, 1e-12, 1e-8, 10}, NloptAlgorithm::Tnewton);
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(rosenbrock_objective(), x0);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-4);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 5e-2);
}

// --- Shanno-Phua full BFGS (LD_VAR2) ---------------------------------------

TEST_CASE("NloptOptimizer/VAR2 — minimizes a well-behaved quadratic") {
  Eigen::Vector3d c(1.0, -2.0, 0.5);
  NloptOptimizer opt({}, NloptAlgorithm::Var2);
  auto out = opt.minimize(quadratic_objective(c), Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-8);
  CHECK((out->theta_hat - c).norm() < 1e-4);
}

TEST_CASE("NloptOptimizer/VAR2 — solves the Rosenbrock function") {
  NloptOptimizer opt({1000, 1e-12, 1e-8, 10}, NloptAlgorithm::Var2);
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(rosenbrock_objective(), x0);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-4);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 5e-2);
}

// --- NLopt's own LBFGS (LD_LBFGS) ------------------------------------------

TEST_CASE("NloptOptimizer/LBFGS — minimizes a well-behaved quadratic") {
  Eigen::Vector3d c(1.0, -2.0, 0.5);
  NloptOptimizer opt({}, NloptAlgorithm::Lbfgs);
  auto out = opt.minimize(quadratic_objective(c), Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-8);
  CHECK((out->theta_hat - c).norm() < 1e-4);
}

TEST_CASE("NloptOptimizer/LBFGS — solves the Rosenbrock function") {
  NloptOptimizer opt({1000, 1e-12, 1e-8, 10}, NloptAlgorithm::Lbfgs);
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(rosenbrock_objective(), x0);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-4);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 5e-2);
}
