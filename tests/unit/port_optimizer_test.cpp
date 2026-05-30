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

TEST_CASE("PortOptimizer — budget exhaustion at non-stationary iterate "
          "returns BudgetExhausted instead of erroring") {
  // §E regression: PORT used to throw on IV(1)=10 (budget exhausted) even
  // when the iterate was usable. Now the wrapper returns the iterate
  // tagged so R-level callers can read theta / fmin / audit and apply
  // their own policy. On Rosenbrock from the canonical (-1.2, 1.0) start
  // with a tiny budget, PORT cannot reach the optimum — the audit will
  // disagree with PORT's "I'm stuck here" verdict and *not* salvage to
  // LineSearchSalvaged, so the BudgetExhausted tag should surface.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g.resize(2);
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
  };
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3;
  PortOptimizer opt(opts);
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize(f, x0);
  REQUIRE(out.has_value());
  CHECK(out->status == magmaan::optim::OptimStatus::BudgetExhausted);
  CHECK(out->iterations >= opts.max_iter);
  // Audit recorded the gradient at the truncated iterate, not the -1
  // "not computed" sentinel — i.e. the wrapper ran the audit before
  // returning, as §E requires for the rescued iterates.
  CHECK(out->grad_inf_norm >= 0.0);
  CHECK(out->fmin > 1e-3);   // still far from the optimum
}

TEST_CASE("PortOptimizer — ftol is forwarded (looser tolerance does no more work)") {
  // opts.ftol (> 0) reaches PORT's relative-function-convergence tolerance
  // v[kV_RfcTol]. A loose tolerance must stop earlier and at a less-optimal
  // iterate than a tight one on the same problem — proving the option is wired
  // through, not that PORT's trust region works. Exercises the ftol > 0
  // forwarding branch the existing budget-exhaustion case (ftol == 0,
  // uniform-stop) does not.
  auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g.resize(2);
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
  };
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto loose = PortOptimizer({1000, /*ftol=*/1e-1,  1e-7, 10}).minimize(f, x0);
  auto tight = PortOptimizer({1000, /*ftol=*/1e-12, 1e-7, 10}).minimize(f, x0);
  REQUIRE(loose.has_value());
  REQUIRE(tight.has_value());
  CHECK(loose->iterations <= tight->iterations);
  CHECK(loose->fmin >= tight->fmin);
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

// ============================================================================
// PortNlsOptimizer — NL2SOL (drn2gb_, TOMS 573, the algorithm behind R `nls`).
// LS-shape: takes residual + Jacobian closures, sees the multi-residual
// structure directly. Tests parallel the scalar PortOptimizer cases plus a
// linear-regression closed-form anchor.
// ============================================================================

using magmaan::optim::PortNlsOptimizer;
using magmaan::optim::ResidualFn;
using magmaan::optim::JacobianFn;
using magmaan::optim::LsEvaluationFn;
using magmaan::optim::LsEvaluation;

TEST_CASE("PortNlsOptimizer — linear LS recovers OLS estimate") {
  // r_i(x) = y_i − X_i · x;  optimum x* = (XᵀX)⁻¹ Xᵀy. Closed-form check —
  // NL2SOL must hit OLS to machine precision since the problem is exactly
  // quadratic in x and one Gauss-Newton step suffices.
  Eigen::MatrixXd X(8, 3);
  X << 1.0,  0.1,  0.4,
       1.0, -0.2,  0.1,
       1.0,  0.5, -0.3,
       1.0,  0.0,  0.7,
       1.0,  0.3,  0.2,
       1.0, -0.4,  0.0,
       1.0,  0.6, -0.1,
       1.0,  0.2,  0.5;
  Eigen::VectorXd beta_true(3);  beta_true << 1.5, -0.7, 2.0;
  Eigen::VectorXd y = X * beta_true;
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();

  ResidualFn r_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    return Eigen::VectorXd{y - X * x};
  };
  JacobianFn J_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    (void)x;
    return Eigen::MatrixXd{-X};
  };
  LsEvaluationFn eval_fn;  // empty — adapter falls back to r_fn + J_fn

  PortNlsOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(p, -inf);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(p,  inf);
  auto out = opt.minimize_ls(r_fn, J_fn, eval_fn, n,
                             Eigen::VectorXd::Zero(p), lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-18);
  CHECK((out->theta_hat - beta_true).norm() < 1e-8);
}

TEST_CASE("PortNlsOptimizer — Rosenbrock as LS recovers (1,1)") {
  // Standard Rosenbrock as a 2-residual LS: r_1 = 10(x_2 - x_1²),
  // r_2 = 1 - x_1. Half-sum-of-squares is exactly Rosenbrock; the LS-shape
  // adapter sees the (n=2, p=2) Jacobian directly, exactly as Ceres LM does.
  const Eigen::Index n = 2;
  const Eigen::Index p = 2;

  ResidualFn r_fn = [](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    Eigen::VectorXd r(2);
    r[0] = 10.0 * (x[1] - x[0] * x[0]);
    r[1] = 1.0 - x[0];
    return r;
  };
  JacobianFn J_fn = [](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    Eigen::MatrixXd J(2, 2);
    J(0, 0) = -20.0 * x[0];
    J(0, 1) =  10.0;
    J(1, 0) =  -1.0;
    J(1, 1) =   0.0;
    return J;
  };
  LsEvaluationFn eval_fn;

  PortNlsOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(p, -inf);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(p,  inf);
  Eigen::VectorXd x0(2);  x0 << -1.2, 1.0;
  auto out = opt.minimize_ls(r_fn, J_fn, eval_fn, n, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->fmin < 1e-12);
  CHECK((out->theta_hat - Eigen::Vector2d(1.0, 1.0)).norm() < 1e-5);
}

TEST_CASE("PortNlsOptimizer — combined eval fast path agrees with split") {
  // When the GmmProblem supplies an `eval` closure that returns both
  // residual + Jacobian at one shot, the adapter prefers it for the
  // "compute both" sub-states. Validate that the path agrees with the
  // split r + J path on the same problem.
  Eigen::MatrixXd X(5, 2);
  X << 1.0,  0.3,
       1.0, -0.2,
       1.0,  0.5,
       1.0,  0.0,
       1.0,  0.7;
  Eigen::VectorXd beta_true(2);  beta_true << 0.4, -1.1;
  Eigen::VectorXd y = X * beta_true;

  ResidualFn r_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    return Eigen::VectorXd{y - X * x};
  };
  JacobianFn J_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    (void)x;
    return Eigen::MatrixXd{-X};
  };
  LsEvaluationFn eval_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<LsEvaluation> {
    return LsEvaluation{Eigen::VectorXd{y - X * x}, Eigen::MatrixXd{-X}};
  };

  PortNlsOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, -inf);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(2,  inf);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

  auto out_split    = opt.minimize_ls(r_fn, J_fn, /*eval*/{},     5, x0, lb, ub);
  auto out_combined = opt.minimize_ls(r_fn, J_fn,  eval_fn,        5, x0, lb, ub);
  REQUIRE(out_split.has_value());
  REQUIRE(out_combined.has_value());
  CHECK((out_split->theta_hat - out_combined->theta_hat).norm() < 1e-10);
  CHECK(out_split->fmin == doctest::Approx(out_combined->fmin).epsilon(1e-10));
}

TEST_CASE("PortNlsOptimizer — lower bound is enforced") {
  // Linear LS optimum at β = (−1, −1); bound both coordinates ≥ 0 so PORT
  // must park at the corner. Same structural test as the scalar PortOptimizer
  // bound case; here exercised through the LS-shape entry.
  Eigen::MatrixXd X = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd target(2);  target << -1.0, -1.0;

  ResidualFn r_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    return Eigen::VectorXd{target - X * x};
  };
  JacobianFn J_fn = [=](const Eigen::VectorXd& x)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    (void)x;
    return Eigen::MatrixXd{-X};
  };

  PortNlsOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lb(2);  lb << 0.0, 0.0;
  Eigen::VectorXd ub(2);  ub << inf, inf;
  Eigen::VectorXd x0(2);  x0 << 0.5, 0.5;        // start feasible
  auto out = opt.minimize_ls(r_fn, J_fn, /*eval*/{}, 2, x0, lb, ub);
  REQUIRE(out.has_value());
  CHECK(out->theta_hat(0) >= -1e-9);
  CHECK(out->theta_hat(1) >= -1e-9);
  CHECK(out->theta_hat(0) < 1e-4);
  CHECK(out->theta_hat(1) < 1e-4);
}

TEST_CASE("PortNlsOptimizer — empty parameter vector is an error value") {
  ResidualFn r_fn = [](const Eigen::VectorXd&)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    return Eigen::VectorXd::Zero(1);
  };
  JacobianFn J_fn = [](const Eigen::VectorXd&)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    return Eigen::MatrixXd::Zero(1, 0);
  };
  PortNlsOptimizer opt;
  auto out = opt.minimize_ls(r_fn, J_fn, /*eval*/{}, 1,
                             Eigen::VectorXd(0),
                             Eigen::VectorXd(0), Eigen::VectorXd(0));
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

TEST_CASE("PortNlsOptimizer — non-positive residual count is an error value") {
  ResidualFn r_fn = [](const Eigen::VectorXd&)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    return Eigen::VectorXd{};
  };
  JacobianFn J_fn = [](const Eigen::VectorXd&)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    return Eigen::MatrixXd{};
  };
  PortNlsOptimizer opt;
  const auto inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd x0(2);  x0.setZero();
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, -inf);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(2,  inf);
  auto out = opt.minimize_ls(r_fn, J_fn, /*eval*/{}, 0, x0, lb, ub);
  REQUIRE_FALSE(out.has_value());
  CHECK(out.error().kind == FitError::Kind::NumericIssue);
}

#endif  // MAGMAAN_WITH_PORT
