#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include <Eigen/Core>

#include "magmaan/optim/problem.hpp"        // OptimStatus, ObjectiveFn
#include "magmaan/optim/terminal_audit.hpp"

using magmaan::optim::audit_terminal_iterate;
using magmaan::optim::ObjectiveFn;
using magmaan::optim::OptimStatus;
using magmaan::optim::TerminalAudit;
using magmaan::optim::TerminalAuditOptions;

namespace {

// f(x) = ½‖x − c‖², ∇f = x − c. Used by several cases below; bare lambdas
// inline so each test can capture its own `c`.
ObjectiveFn make_quadratic(const Eigen::VectorXd& c) {
  return [c](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = x - c;
    return 0.5 * (x - c).squaredNorm();
  };
}

constexpr double inf = std::numeric_limits<double>::infinity();

Eigen::VectorXd unbounded_lower(Eigen::Index n) {
  return Eigen::VectorXd::Constant(n, -inf);
}
Eigen::VectorXd unbounded_upper(Eigen::Index n) {
  return Eigen::VectorXd::Constant(n,  inf);
}

}  // namespace

TEST_CASE("terminal_audit — quadratic at optimum is stationary") {
  Eigen::VectorXd c(3);
  c << 1.0, -2.0, 0.5;
  const auto f = make_quadratic(c);
  TerminalAudit a = audit_terminal_iterate(
      f, c, /*reported_f=*/0.0, unbounded_lower(3), unbounded_upper(3));

  CHECK(a.stationary);
  CHECK(a.f_finite);
  CHECK(a.f_recomputed == doctest::Approx(0.0));
  CHECK(a.grad_inf_norm == doctest::Approx(0.0));
  CHECK(a.advisory_status == OptimStatus::Converged);
  CHECK(a.f_consistent);  // 0 vs 0
  CHECK(a.active_set.size() == 3u);
  for (auto bit : a.active_set) CHECK(bit == 0);
}

TEST_CASE("terminal_audit — displaced quadratic is non-stationary, advisory Unknown") {
  Eigen::VectorXd c(2);
  c << 0.0, 0.0;
  Eigen::VectorXd x(2);
  x << 1.0, 0.0;  // gradient is x - c = (1, 0); norm 1
  const auto f = make_quadratic(c);
  TerminalAudit a = audit_terminal_iterate(
      f, x, /*reported_f=*/0.5, unbounded_lower(2), unbounded_upper(2));

  CHECK_FALSE(a.stationary);
  CHECK(a.grad_inf_norm == doctest::Approx(1.0));
  CHECK(a.advisory_status == OptimStatus::Unknown);
  // RHS is tol * (1 + |f|) = 2e-6 * (1 + 0.5) = 3e-6, definitely < 1.
  CHECK(a.stationarity_rhs == doctest::Approx(3e-6));
}

TEST_CASE("terminal_audit — at lower bound with outward gradient is stationary") {
  // f(x) = ½(x − (−1))² = ½(x + 1)². Optimum is at x = -1, but the box
  // [0, +inf] cuts that off; the constrained optimum is x = 0, where the
  // gradient is +1 (pushing OUTWARD against the active lower bound). The
  // projected gradient is then 0, so the audit must declare stationary.
  const auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = x + Eigen::VectorXd::Constant(x.size(), 1.0);
    return 0.5 * (x + Eigen::VectorXd::Constant(x.size(), 1.0)).squaredNorm();
  };
  Eigen::VectorXd x(1); x << 0.0;
  Eigen::VectorXd lo(1); lo << 0.0;
  Eigen::VectorXd up(1); up << inf;

  TerminalAudit a = audit_terminal_iterate(f, x, /*reported_f=*/0.5, lo, up);
  CHECK(a.stationary);
  CHECK(a.grad_inf_norm == doctest::Approx(0.0));  // outward component zeroed
  REQUIRE(a.active_set.size() == 1u);
  CHECK(a.active_set[0] == -1);
  CHECK(a.advisory_status == OptimStatus::Converged);
}

TEST_CASE("terminal_audit — at lower bound with inward gradient is non-stationary") {
  // f(x) = ½(x − 1)² with bounds [0, +inf]. At x = 0, gradient is x − 1 = −1
  // (pushing INWARD). The projected gradient retains the full magnitude
  // because the descent direction is feasible. Non-stationary.
  const auto f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = x - Eigen::VectorXd::Constant(x.size(), 1.0);
    return 0.5 * (x - Eigen::VectorXd::Constant(x.size(), 1.0)).squaredNorm();
  };
  Eigen::VectorXd x(1); x << 0.0;
  Eigen::VectorXd lo(1); lo << 0.0;
  Eigen::VectorXd up(1); up << inf;

  TerminalAudit a = audit_terminal_iterate(f, x, /*reported_f=*/0.5, lo, up);
  CHECK_FALSE(a.stationary);
  CHECK(a.grad_inf_norm == doctest::Approx(1.0));
  REQUIRE(a.active_set.size() == 1u);
  CHECK(a.active_set[0] == -1);  // still on the bound
  CHECK(a.advisory_status == OptimStatus::Unknown);
}

TEST_CASE("terminal_audit — non-finite objective produces Unknown") {
  const auto f = [](const Eigen::VectorXd&, Eigen::VectorXd& g) {
    g.setZero();
    return std::numeric_limits<double>::infinity();
  };
  Eigen::VectorXd x(2); x << 1.0, 2.0;
  TerminalAudit a = audit_terminal_iterate(
      f, x, /*reported_f=*/1.0, unbounded_lower(2), unbounded_upper(2));
  CHECK_FALSE(a.f_finite);
  CHECK_FALSE(a.stationary);
  CHECK(a.advisory_status == OptimStatus::Unknown);
  // The relative consistency check requires both sides finite — should fail.
  CHECK_FALSE(a.f_consistent);
}

TEST_CASE("terminal_audit — inconsistent reported_f flagged") {
  Eigen::VectorXd c(1); c << 0.0;
  const auto f = make_quadratic(c);
  Eigen::VectorXd x(1); x << 0.0;  // f(x) = 0
  // Backend "reports" 100 — wildly off. Audit should still declare stationary
  // (geometry is independent of reported_f) but flag the inconsistency.
  TerminalAudit a = audit_terminal_iterate(
      f, x, /*reported_f=*/100.0, unbounded_lower(1), unbounded_upper(1));
  CHECK(a.stationary);
  CHECK_FALSE(a.f_consistent);
}

TEST_CASE("terminal_audit — RELATIVE stationarity tolerance scales with |f|") {
  // f at optimum with a fake huge scale: construct a (correct) stationary
  // point where |f| = 1e6 and the gradient is exactly 0. The audit's RHS is
  // tol * (1 + |f|) ≈ 2.0; a non-zero gradient of magnitude < 2 should still
  // be accepted as stationary. (This documents the scale-invariance of the
  // test — a fixed-absolute threshold would mis-classify here.)
  const auto f = [](const Eigen::VectorXd&, Eigen::VectorXd& g) {
    g = Eigen::VectorXd::Constant(g.size(), 0.5);   // |g|_inf = 0.5
    return 1e6;
  };
  Eigen::VectorXd x(2); x << 0.0, 0.0;
  TerminalAudit a = audit_terminal_iterate(
      f, x, /*reported_f=*/1e6, unbounded_lower(2), unbounded_upper(2));
  CHECK(a.stationary);
  CHECK(a.stationarity_rhs == doctest::Approx(2.0 + 2e-6).epsilon(1e-6));
  CHECK(a.grad_inf_norm == doctest::Approx(0.5));
}

TEST_CASE("terminal_audit — one-ppm ML line-search remainder is salvageable") {
  const auto f = [](const Eigen::VectorXd&, Eigen::VectorXd& g) {
    g = Eigen::VectorXd::Constant(g.size(), 1.2664784776461602e-6);
    return 0.19217060692071364;
  };
  Eigen::VectorXd x(8);
  x.setZero();
  TerminalAudit a = audit_terminal_iterate(
      f, x, /*reported_f=*/0.19217060692071364, unbounded_lower(8),
      unbounded_upper(8));

  CHECK(a.stationary);
  CHECK(a.f_consistent);
  CHECK(a.grad_inf_norm == doctest::Approx(1.2664784776461602e-6));
  CHECK(a.stationarity_rhs == doctest::Approx(2.3843412138414273e-6));
}

TEST_CASE("terminal_audit — bounds-size mismatch returns null audit, does not crash") {
  Eigen::VectorXd c(2); c << 0.0, 0.0;
  const auto f = make_quadratic(c);
  Eigen::VectorXd x(2); x << 1.0, 2.0;
  Eigen::VectorXd lo(1); lo << 0.0;     // intentionally wrong size
  Eigen::VectorXd up(2); up << inf, inf;

  TerminalAudit a = audit_terminal_iterate(f, x, /*reported_f=*/0.0, lo, up);
  // We short-circuit before invoking f; the audit is the default-constructed
  // null result. Verify a few hallmark fields rather than checking the entire
  // default state (which is what TerminalAudit{} gives us).
  CHECK_FALSE(a.f_finite);
  CHECK_FALSE(a.stationary);
  CHECK(a.grad_inf_norm == -1.0);
  CHECK(a.advisory_status == OptimStatus::Unknown);
}
