#include <doctest/doctest.h>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"

using magmaan::estimate::Bounds;
using magmaan::optim::ConstrainedScalarProblem;
using magmaan::optim::OptimOptions;
using magmaan::optim::ScalarProblem;

#ifdef MAGMAAN_WITH_IPOPT

TEST_CASE("IPOPT optimizer minimizes an unconstrained quadratic") {
  ScalarProblem prob;
  prob.n_param = 2;
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    Eigen::Vector2d target;
    target << 1.0, -2.0;
    grad = x - target;
    return 0.5 * (x - target).squaredNorm();
  };

  Eigen::Vector2d x0;
  x0 << 10.0, 10.0;
  auto out = magmaan::optim::ipopt(prob, x0, {}, OptimOptions{});
  if (!out.has_value()) {
    FAIL(out.error().detail);
    return;
  }
  CHECK(out->x(0) == doctest::Approx(1.0).epsilon(1e-7));
  CHECK(out->x(1) == doctest::Approx(-2.0).epsilon(1e-7));
}

TEST_CASE("IPOPT optimizer honors box bounds") {
  ScalarProblem prob;
  prob.n_param = 1;
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    grad.resize(1);
    grad(0) = x(0) + 2.0;
    return 0.5 * (x(0) + 2.0) * (x(0) + 2.0);
  };

  Bounds b;
  b.lower = Eigen::VectorXd::Constant(1, 0.0);
  b.upper = Eigen::VectorXd::Constant(1, 10.0);
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 4.0);
  auto out = magmaan::optim::ipopt(prob, x0, b, OptimOptions{});
  if (!out.has_value()) {
    FAIL(out.error().detail);
    return;
  }
  CHECK(out->x(0) == doctest::Approx(0.0).epsilon(1e-7));
}

TEST_CASE("IPOPT optimizer enforces a nonlinear equality constraint") {
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

  Eigen::Vector2d x0;
  x0 << 0.25, 0.5;
  auto out = magmaan::optim::ipopt_constrained(prob, x0, {}, OptimOptions{});
  if (!out.has_value()) {
    FAIL(out.error().detail);
    return;
  }
  CHECK(std::abs(out->x(0) - out->x(1) * out->x(1)) < 1e-7);
  CHECK(out->fmin < 0.1);
}

#endif  // MAGMAAN_WITH_IPOPT
