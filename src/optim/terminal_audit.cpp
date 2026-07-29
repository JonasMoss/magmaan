#include "magmaan/optim/terminal_audit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

namespace magmaan::optim {

TerminalAudit audit_terminal_iterate(
    const ObjectiveFn&     f,
    const Eigen::VectorXd& x,
    double                 reported_f,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    TerminalAuditOptions   opts) {
  TerminalAudit a;
  const Eigen::Index n = x.size();

  // Degenerate inputs short-circuit to a non-stationary, non-finite audit
  // without touching f. The caller may be invoking us on a state that the
  // wrapper itself rejected (bounds mismatch, empty x); we don't compound the
  // problem by calling an objective on garbage.
  if (n <= 0 || lower.size() != n || upper.size() != n) {
    return a;
  }

  // Recompute f and ∇f at x. The Objective contract is "+inf on invalid x";
  // we treat +inf the same way as NaN — neither is something we can certify
  // stationarity on.
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
  const double    f_rec = f(x, grad);
  a.f_recomputed = f_rec;
  a.f_finite     = std::isfinite(f_rec);

  if (a.f_finite && std::isfinite(reported_f)) {
    const double rhs = opts.f_consistency_rel * (1.0 + std::abs(reported_f));
    a.f_consistent = std::abs(f_rec - reported_f) <= rhs;
  }

  // Active-set readout and projected gradient — built in the same pass.
  // A coordinate is "active" only if its bound is finite *and* x is within
  // `active_bound_tol` of it. ±infinity bounds are always interior.
  a.active_set.assign(static_cast<std::size_t>(n), 0);
  double gnorm = 0.0;
  double gnorm_scaled_num = 0.0;  // max_i |Pg_i| · max(|x_i|, 1)
  for (Eigen::Index i = 0; i < n; ++i) {
    double      gi   = grad[i];
    const bool  lo_finite = std::isfinite(lower[i]);
    const bool  up_finite = std::isfinite(upper[i]);
    const bool  at_lo = lo_finite && (x[i] - lower[i] <= opts.active_bound_tol);
    const bool  at_up = up_finite && (upper[i] - x[i] <= opts.active_bound_tol);
    // KKT for box bounds: a coordinate pushing OUTWARD against an active
    // bound is feasible (we cannot decrease f along that direction without
    // leaving the box), so its gradient contribution is zeroed. A coordinate
    // pushing INWARD is still a degree of freedom and counts toward gnorm.
    if (at_lo && gi > 0.0) gi = 0.0;
    if (at_up && gi < 0.0) gi = 0.0;
    if (at_lo) a.active_set[static_cast<std::size_t>(i)] = -1;
    if (at_up) a.active_set[static_cast<std::size_t>(i)] = +1;
    // If both bounds are active (x is pinned, lo == up to tol), the +1
    // overwrites the -1 — degenerate but consistent: gnorm contribution is
    // zero either way and the consumer can detect it via lo_finite && up_finite
    // && |upper - lower| small.
    const double agi = std::abs(gi);
    if (agi > gnorm) gnorm = agi;
    const double scale_x = std::max(std::abs(x[i]), 1.0);
    const double scaled_i = agi * scale_x;
    if (scaled_i > gnorm_scaled_num) gnorm_scaled_num = scaled_i;
  }
  a.grad_inf_norm = gnorm;
  a.raw_grad_inf_norm =
      grad.size() > 0 && grad.allFinite()
          ? grad.cwiseAbs().maxCoeff()
          : -1.0;
  if (a.f_finite) {
    a.grad_scaled_inf = gnorm_scaled_num / std::max(std::abs(f_rec), 1.0);
  }

  if (a.f_finite) {
    // v1: Absolute mode is the default, matching lavaan's `optim.dx.tol`.
    // Relative mode remains available pending an empirical calibration of
    // the SEM-side stationarity test (see docs/design/terminal-audit.md
    // and the backlog "Tolerance calibration" entry).
    if (opts.stationarity_mode == TerminalAuditOptions::StationarityMode::Absolute) {
      a.stationarity_rhs = opts.absolute_tol;
    } else {
      a.stationarity_rhs = opts.stationarity_tol * (1.0 + std::abs(f_rec));
    }
    a.stationary = gnorm <= a.stationarity_rhs;
  }

  // v1 advisory policy: report Converged ONLY when geometrically stationary.
  // Otherwise the audit refuses to volunteer a refinement — the wrapper still
  // owns the reported code, and the wrapper is the one that knows whether the
  // backend's verdict was "clean stop", "soft failure", or "singular". A
  // non-stationary advisory `Unknown` is the correct null signal.
  if (a.stationary) a.advisory_status = OptimStatus::Converged;
  return a;
}

TerminalAudit audit_equality_constrained_terminal_iterate(
    const ConstrainedScalarProblem& prob,
    const Eigen::VectorXd&          x,
    double                          reported_f,
    const Eigen::VectorXd&          lower,
    const Eigen::VectorXd&          upper,
    TerminalAuditOptions            opts) {
  TerminalAudit a;
  a.constrained = true;
  const Eigen::Index n = x.size();
  const Eigen::Index m = prob.n_constraint;
  if (n <= 0 || m <= 0 || prob.objective.n_param != n ||
      lower.size() != n || upper.size() != n ||
      prob.constraint_lower.size() != m ||
      prob.constraint_upper.size() != m ||
      !prob.objective.f || !prob.h || !prob.J_h) {
    return a;
  }
  if ((prob.constraint_lower - prob.constraint_upper)
          .cwiseAbs().maxCoeff() > 0.0) {
    return a;
  }

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
  const double f_rec = prob.objective.f(x, grad);
  a.f_recomputed = f_rec;
  a.f_finite = std::isfinite(f_rec);
  if (a.f_finite && std::isfinite(reported_f)) {
    const double rhs =
        opts.f_consistency_rel * (1.0 + std::abs(reported_f));
    a.f_consistent = std::abs(f_rec - reported_f) <= rhs;
  }
  if (!a.f_finite || grad.size() != n || !grad.allFinite()) return a;
  a.raw_grad_inf_norm = grad.cwiseAbs().maxCoeff();

  const Eigen::VectorXd h = prob.h(x);
  const Eigen::MatrixXd J = prob.J_h(x);
  if (h.size() != m || !h.allFinite() ||
      J.rows() != m || J.cols() != n || !J.allFinite()) {
    return a;
  }
  const Eigen::VectorXd residual = h - prob.constraint_lower;
  a.constraint_violation_inf = residual.cwiseAbs().maxCoeff();
  const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> rank_cod(J);
  a.constraint_jacobian_rank =
      static_cast<std::int32_t>(rank_cod.rank());

  a.active_set.assign(static_cast<std::size_t>(n), 0);
  std::vector<Eigen::Index> interior;
  interior.reserve(static_cast<std::size_t>(n));
  for (Eigen::Index i = 0; i < n; ++i) {
    const bool lo_finite = std::isfinite(lower[i]);
    const bool up_finite = std::isfinite(upper[i]);
    const bool at_lo =
        lo_finite && (x[i] - lower[i] <= opts.active_bound_tol);
    const bool at_up =
        up_finite && (upper[i] - x[i] <= opts.active_bound_tol);
    if (at_lo) a.active_set[static_cast<std::size_t>(i)] = -1;
    if (at_up) a.active_set[static_cast<std::size_t>(i)] = +1;
    if (!at_lo && !at_up) interior.push_back(i);
  }

  Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m);
  if (!interior.empty()) {
    Eigen::MatrixXd A(
        static_cast<Eigen::Index>(interior.size()), m);
    Eigen::VectorXd rhs(static_cast<Eigen::Index>(interior.size()));
    for (Eigen::Index r = 0;
         r < static_cast<Eigen::Index>(interior.size()); ++r) {
      const Eigen::Index i = interior[static_cast<std::size_t>(r)];
      A.row(r) = J.col(i).transpose();
      rhs(r) = -grad(i);
    }
    const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A);
    lambda = cod.solve(rhs);
    if (lambda.size() != m || !lambda.allFinite()) return a;
  }

  Eigen::VectorXd lag_grad = grad + J.transpose() * lambda;
  double kkt_inf = 0.0;
  double kkt_scaled_num = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    double gi = lag_grad(i);
    const bool at_lo =
        a.active_set[static_cast<std::size_t>(i)] == -1;
    const bool at_up =
        a.active_set[static_cast<std::size_t>(i)] == +1;
    const bool pinned =
        std::isfinite(lower(i)) && std::isfinite(upper(i)) &&
        upper(i) - lower(i) <= 2.0 * opts.active_bound_tol;
    if (pinned || (at_lo && gi > 0.0) || (at_up && gi < 0.0)) gi = 0.0;
    const double agi = std::abs(gi);
    kkt_inf = std::max(kkt_inf, agi);
    kkt_scaled_num =
        std::max(kkt_scaled_num, agi * std::max(std::abs(x(i)), 1.0));
  }
  a.grad_inf_norm = kkt_inf;
  a.grad_scaled_inf =
      kkt_scaled_num / std::max(std::abs(f_rec), 1.0);
  if (opts.stationarity_mode ==
      TerminalAuditOptions::StationarityMode::Absolute) {
    a.stationarity_rhs = opts.absolute_tol;
  } else {
    a.stationarity_rhs =
        opts.stationarity_tol * (1.0 + std::abs(f_rec));
  }
  a.stationary =
      a.constraint_violation_inf <= opts.constraint_tol &&
      kkt_inf <= a.stationarity_rhs;
  if (a.stationary) a.advisory_status = OptimStatus::Converged;
  return a;
}

}  // namespace magmaan::optim
