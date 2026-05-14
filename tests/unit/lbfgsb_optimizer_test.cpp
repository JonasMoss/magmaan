#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/fit/concepts.hpp"
#include "magmaan/fit/lbfgsb_optimizer.hpp"

using magmaan::FitError;
using magmaan::fit::LbfgsBOptimizer;

// ============================================================================
// Concept conformance — LbfgsBOptimizer models both Optimizer (via the
// unbounded overload) and BoundedOptimizer (via the bounded overload).
// These static_asserts are the compile-time gate.
// ============================================================================

TEST_CASE("LbfgsBOptimizer satisfies Optimizer and BoundedOptimizer concepts") {
  static_assert(magmaan::fit::Optimizer<LbfgsBOptimizer>,
                "LbfgsBOptimizer must model Optimizer");
  static_assert(magmaan::fit::BoundedOptimizer<LbfgsBOptimizer>,
                "LbfgsBOptimizer must model BoundedOptimizer");
  CHECK(true);
}

// ============================================================================
// Behavioural — unbounded path is a regular L-BFGS-equivalent solve.
// ============================================================================

TEST_CASE("LbfgsBOptimizer — unbounded path minimizes a quadratic") {
  Eigen::VectorXd c(3);
  c << 1.0, -2.0, 0.5;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  LbfgsBOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-10);
  CHECK((out->theta_hat - c).norm() < 1e-5);
}

// ============================================================================
// Bounds — the actual reason this optimizer exists.
// ============================================================================

TEST_CASE("LbfgsBOptimizer — lower bound is enforced (Heywood story)") {
  // Optimum at (-1, -1), but both axes bounded below at 0 — solver must
  // park at (0, 0). Same shape as the Ceres test, different backend.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(2);  target << -1.0, -1.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  LbfgsBOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(2);  lb << 0.0, 0.0;
  Eigen::VectorXd ub(2);  ub << inf, inf;
  Eigen::VectorXd x0(2);  x0 << 0.5, 0.5;
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) >= -1e-12);
  CHECK(out->theta_hat(1) >= -1e-12);
  CHECK(out->theta_hat(0) < 1e-6);
  CHECK(out->theta_hat(1) < 1e-6);
}

TEST_CASE("LbfgsBOptimizer — upper bound is enforced too") {
  // Optimum at 10, upper bound at 3. Solver must park at 3.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(1);  target << 10.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  LbfgsBOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(1);  lb << -inf;
  Eigen::VectorXd ub(1);  ub << 3.0;
  Eigen::VectorXd x0(1);  x0 << 0.0;
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) <= 3.0 + 1e-12);
  CHECK(out->theta_hat(0) > 3.0 - 1e-6);
}

TEST_CASE("LbfgsBOptimizer — infinite bounds match the unbounded path") {
  Eigen::VectorXd c(2);  c << 0.7, -1.3;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  LbfgsBOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, -inf);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(2,  inf);
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(2), lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-10);
  CHECK((out->theta_hat - c).norm() < 1e-5);
}

// ============================================================================
// Boundary contracts — size mismatch and infeasible x0.
// ============================================================================

TEST_CASE("LbfgsBOptimizer — bound size mismatch is an error value") {
  auto f = [](const Eigen::VectorXd&, Eigen::VectorXd&) { return 0.0; };
  LbfgsBOptimizer opt;
  Eigen::VectorXd x0(3);  x0.setZero();
  Eigen::VectorXd lb(2);  lb.setZero();
  Eigen::VectorXd ub(2);  ub.setZero();
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

TEST_CASE("LbfgsBOptimizer — infeasible x0 is projected, not rejected") {
  // x0 starts below the lower bound; the adapter projects it back into
  // the box before LBFGSpp ever sees it. This is the documented
  // initial-point contract — important because LBFGSpp's own
  // proj_grad_norm goes haywire on out-of-box starts.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(2);  target << 5.0, 5.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  LbfgsBOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(2);  lb << 0.0, 0.0;
  Eigen::VectorXd ub(2);  ub << inf, inf;
  Eigen::VectorXd x0(2);  x0 << -2.0, -3.0;       // both infeasible
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) > 5.0 - 1e-5);
  CHECK(out->theta_hat(1) > 5.0 - 1e-5);
}

// ============================================================================
// Line-search exhaustion path — same exception-handling story as the
// unconstrained LbfgsOptimizer test. A "lying" objective (flat, but
// non-zero reported gradient) forces backtracking past min_step.
// ============================================================================

TEST_CASE("LbfgsBOptimizer — line-search failure becomes a value, not a crash") {
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = Eigen::VectorXd::Constant(x.size(), 1.0);
    return 0.0;
  };
  LbfgsBOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x0(4);  x0.setZero();
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(4, -inf);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(4,  inf);
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::LineSearchFailed);
}
