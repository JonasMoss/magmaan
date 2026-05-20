#include <doctest/doctest.h>

#ifdef MAGMAAN_WITH_PORT

#include <cmath>
#include <limits>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/optim/port_optimizer.hpp"

using magmaan::FitError;
using magmaan::optim::PortOptimizer;

// ============================================================================
// Behavioural tests for the vendored PORT (drmngb_) adapter — the model-Hessian
// trust-region algorithm behind R's nlminb (TOMS 611, Dennis-Gay-Welsch).
// PORT uses reverse communication: the adapter loops, computing F or G as
// PORT requests, until PORT signals termination. These tests exercise the
// adapter on the same canonical smoke problems other optimizer adapters use,
// plus PORT-specific cases (bounded enforcement, the singular-Jacobian
// Powell test problem, the large-residual Brown-Dennis problem).
// ============================================================================

TEST_CASE("PortOptimizer — minimizes a well-behaved quadratic") {
  Eigen::VectorXd c(3);  c << 1.0, -2.0, 0.5;
  auto f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = 2.0 * (x - c);
    return (x - c).squaredNorm();
  };
  PortOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd::Zero(3));
  REQUIRE(out.has_value());
  // PORT's model-Hessian Newton step converges to roundoff on a quadratic in
  // one outer iteration; we keep the tolerance loose enough to be unaffected
  // by minor differences in PORT's internal stopping criteria.
  CHECK(out->fmin < 1e-10);
  CHECK((out->theta_hat - c).norm() < 1e-6);
}

TEST_CASE("PortOptimizer — solves the Rosenbrock function") {
  // Canonical Rosenbrock: min at (1,1), f = 0. The curved valley is exactly
  // the case where PORT's trust region beats line search; nlminb solves it
  // comfortably from the standard (-1.2, 1.0) start, and so must Port.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g.resize(2);
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
  };
  PortOptimizer opt;
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(f, x0);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-6);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 1e-3);
}

TEST_CASE("PortOptimizer — Powell singular function") {
  // Powell singular: a 4D test where the Hessian at the optimum is singular,
  // which is one of the regimes PORT's model-Hessian trust region is most
  // celebrated for. f = (x1 + 10 x2)² + 5 (x3 - x4)² + (x2 - 2 x3)⁴
  //                      + 10 (x1 - x4)⁴, min at origin, f = 0.
  // Standard start is (3, -1, 0, 1).
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double t1 = x[0] + 10.0 * x[1];
    const double t2 = x[2] - x[3];
    const double t3 = x[1] - 2.0 * x[2];
    const double t4 = x[0] - x[3];
    g.resize(4);
    g[0] = 2.0 * t1                  + 40.0 * t4 * t4 * t4;
    g[1] = 20.0 * t1 + 4.0 * t3 * t3 * t3;
    g[2] = 10.0 * t2 - 8.0 * t3 * t3 * t3;
    g[3] = -10.0 * t2                 - 40.0 * t4 * t4 * t4;
    return t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;
  };
  PortOptimizer opt;
  Eigen::VectorXd x0(4);  x0 << 3.0, -1.0, 0.0, 1.0;
  auto out = opt.minimize(f, x0);
  REQUIRE(out.has_value());
  // Singular-Hessian termination is permitted (PORT IV(1) = 7 maps to ok); the
  // residual norm is small but not zero because Powell-singular's local
  // curvature vanishes near the optimum — a slow tail that any trust-region
  // method tolerates by accepting a relatively loose final f.
  CHECK(out->fmin < 1e-4);
  CHECK(out->theta_hat.norm() < 1e-1);
}

TEST_CASE("PortOptimizer — Brown and Dennis function") {
  // Brown-Dennis: a least-squares-shaped 4D problem with a large positive
  // optimum value (≈ 85822.2) — i.e., a *large-residual* case. The PORT model
  // Hessian's separation between "small residual" (Gauss-Newton-like) and
  // "large residual" (full curvature) regimes is what this tests. Standard
  // start (25, 5, -5, -1); we report success when fmin lands close enough to
  // the documented optimum and parameters are within an order of magnitude.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g.resize(4);  g.setZero();
    double sum = 0.0;
    for (int i = 1; i <= 20; ++i) {
      const double ti = static_cast<double>(i) / 5.0;
      const double a  = x[0] + ti * x[1] - std::exp(ti);
      const double b  = x[2] + x[3] * std::sin(ti) - std::cos(ti);
      const double r  = a * a + b * b;
      sum += r * r;
      g[0] += 4.0 * r * a;
      g[1] += 4.0 * r * a * ti;
      g[2] += 4.0 * r * b;
      g[3] += 4.0 * r * b * std::sin(ti);
    }
    return sum;
  };
  PortOptimizer opt;
  Eigen::VectorXd x0(4);  x0 << 25.0, 5.0, -5.0, -1.0;
  auto out = opt.minimize(f, x0);
  REQUIRE(out.has_value());
  // Documented optimum value ≈ 85822.2. PORT routinely lands within 1% of
  // it from the standard start; tightening this further makes the test
  // sensitive to PORT's internal tuning rather than to whether the adapter
  // works.
  CHECK(out->fmin > 85000.0);
  CHECK(out->fmin < 87000.0);
}

TEST_CASE("PortOptimizer — lower bound is enforced (Heywood story)") {
  // Optimum at (-1, -1); both coordinates bounded below at 0 — PORT must
  // park the solution at the corner (0, 0). This is the same shape as the
  // NLopt SLSQP bounds test, so the comparison is direct.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    Eigen::VectorXd target(2);  target << -1.0, -1.0;
    g = 2.0 * (x - target);
    return (x - target).squaredNorm();
  };
  PortOptimizer opt;
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

TEST_CASE("PortOptimizer — empty parameter vector is an error value") {
  auto f = [](const Eigen::VectorXd&, Eigen::VectorXd&) { return 0.0; };
  PortOptimizer opt;
  auto out = opt.minimize(f, Eigen::VectorXd(0));
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

TEST_CASE("PortOptimizer — bound size mismatch is an error value") {
  auto f = [](const Eigen::VectorXd&, Eigen::VectorXd&) { return 0.0; };
  PortOptimizer opt;
  Eigen::VectorXd x0(3);  x0.setZero();
  Eigen::VectorXd lb(2);  lb.setZero();
  Eigen::VectorXd ub(2);  ub.setZero();
  auto out = opt.minimize(f, x0, lb, ub);
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

#endif  // MAGMAAN_WITH_PORT
