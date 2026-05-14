#include <doctest/doctest.h>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/fit/lbfgs_optimizer.hpp"

using magmaan::FitError;
using magmaan::fit::LbfgsOptimizer;
using magmaan::fit::LbfgsOptions;

TEST_CASE("LbfgsOptimizer — minimizes a well-behaved quadratic") {
  // f(x) = ||x - c||^2, minimum at x = c.
  Eigen::VectorXd c(3);
  c << 1.0, -2.0, 0.5;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  LbfgsOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-12);
  CHECK((out->theta_hat - c).norm() < 1e-6);
}

TEST_CASE("LbfgsOptimizer — line-search failure becomes a value, not a crash") {
  // A "lying" objective: f is flat (==0 everywhere) but the gradient is
  // reported as a nonzero constant. LBFGS picks direction d = -g, the
  // Armijo sufficient-decrease test can never be satisfied (f never drops
  // below f0 = 0 while the predicted decrease is strictly negative), so
  // the line search shrinks the step past min_step and LBFGSpp throws.
  // Before the fix that throw escaped the -fexceptions TU into
  // -fno-exceptions code and std::terminate'd; now minimize() must catch
  // it and return FitError::Kind::LineSearchFailed.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = Eigen::VectorXd::Constant(x.size(), 1.0);
    return 0.0;
  };
  LbfgsOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(4));
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::LineSearchFailed);
}
