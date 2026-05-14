#include <doctest/doctest.h>

#ifdef MAGMAAN_WITH_CERES

#include <cmath>
#include <limits>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/concepts.hpp"

using magmaan::FitError;
using magmaan::optim::CeresBoundedOptimizer;
using magmaan::optim::CeresOptimizer;
using magmaan::optim::CeresOptions;

// ============================================================================
// Concept guards — compile-time checks that both adapters model the right
// concepts. CeresBoundedOptimizer satisfies both Optimizer (via the
// (f, x0) overload) and BoundedOptimizer (via the (f, x0, lb, ub) overload).
// ============================================================================

TEST_CASE("Optimizer concept is satisfied by both Ceres backends") {
  static_assert(magmaan::optim::Optimizer<CeresOptimizer>,
                "CeresOptimizer must model Optimizer");
  static_assert(magmaan::optim::Optimizer<CeresBoundedOptimizer>,
                "CeresBoundedOptimizer must model Optimizer too "
                "(via its unbounded minimize overload)");
  static_assert(magmaan::optim::BoundedOptimizer<CeresBoundedOptimizer>,
                "CeresBoundedOptimizer must model BoundedOptimizer");
  CHECK(true);
}

// ============================================================================
// Behavioural tests — quadratic minimization with each backend.
// ============================================================================

TEST_CASE("CeresOptimizer — minimizes a well-behaved quadratic") {
  Eigen::VectorXd c(3);  c << 1.0, -2.0, 0.5;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  CeresOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-10);
  CHECK((out->theta_hat - c).norm() < 1e-5);
}

TEST_CASE("CeresBoundedOptimizer — unbounded path matches CeresOptimizer") {
  Eigen::VectorXd c(2);  c << 0.7, -1.3;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  CeresBoundedOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(2));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-10);
  CHECK((out->theta_hat - c).norm() < 1e-5);
}

TEST_CASE("CeresBoundedOptimizer — lower bound is enforced (Heywood story)") {
  // Optimum at x = (-1, -1); both bounded below at 0. The optimizer must
  // park x at the boundary at (0, 0). This is the load-bearing
  // demonstration that the Problem-API path actually clamps Heywood drift.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(2);  target << -1.0, -1.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  CeresBoundedOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(2);  lb << 0.0, 0.0;
  Eigen::VectorXd ub(2);  ub << inf, inf;
  Eigen::VectorXd x0(2);  x0 << 0.5, 0.5;        // start in feasible region
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) >= -1e-9);             // bound holds
  CHECK(out->theta_hat(1) >= -1e-9);
  CHECK(out->theta_hat(0) < 1e-5);               // sat at the boundary
  CHECK(out->theta_hat(1) < 1e-5);
}

TEST_CASE("CeresBoundedOptimizer — upper bound is enforced too") {
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(1);  target << 10.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  CeresBoundedOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(1);  lb << -inf;
  Eigen::VectorXd ub(1);  ub << 3.0;
  Eigen::VectorXd x0(1);  x0 << 0.0;
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) <= 3.0 + 1e-9);
  CHECK(out->theta_hat(0) > 3.0 - 1e-5);          // sat at upper boundary
}

TEST_CASE("CeresBoundedOptimizer — bound size mismatch is an error value") {
  auto f = [](const Eigen::VectorXd&, Eigen::VectorXd&) { return 0.0; };
  CeresBoundedOptimizer opt;
  Eigen::VectorXd x0(3);  x0.setZero();
  Eigen::VectorXd lb(2);  lb.setZero();
  Eigen::VectorXd ub(2);  ub.setZero();
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

#endif  // MAGMAAN_WITH_CERES
