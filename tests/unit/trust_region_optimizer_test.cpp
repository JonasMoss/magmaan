#include <doctest/doctest.h>

#include <cmath>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/optim/trust_region_optimizer.hpp"

using magmaan::FitError;
using magmaan::optim::TrustRegionOptimizer;

// ============================================================================
// Behavioural tests for the CppNumericalSolvers Newton trust-region adapter.
// The objective contract is `double(const VectorXd& x, VectorXd& grad)`; the
// adapter forms the Hessian by finite-differencing this analytic gradient.
// ============================================================================

TEST_CASE("TrustRegionOptimizer — minimizes a well-behaved quadratic") {
  // A pure quadratic: Newton's method is exact in one step, so the optimum is
  // recovered to roundoff regardless of the solver's stopping tolerances.
  Eigen::VectorXd c(3);  c << 1.0, -2.0, 0.5;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  TrustRegionOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-10);
  CHECK((out->theta_hat - c).norm() < 1e-6);
}

TEST_CASE("TrustRegionOptimizer — solves the Rosenbrock function") {
  // Rosenbrock f = (1-x)² + 100(y-x²)², min at (1,1), f = 0. The curved valley
  // is the canonical case where a trust region beats plain line search.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g.resize(2);
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
  };
  TrustRegionOptimizer opt;
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(f, x0);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-4);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 1e-2);
}

TEST_CASE("TrustRegionOptimizer — empty parameter vector is an error value") {
  auto f = [](const Eigen::VectorXd&, Eigen::VectorXd&) { return 0.0; };
  TrustRegionOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd(0));
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}
