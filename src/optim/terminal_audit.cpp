#include "magmaan/optim/terminal_audit.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

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
  }
  a.grad_inf_norm = gnorm;

  if (a.f_finite) {
    a.stationarity_rhs = opts.stationarity_tol * (1.0 + std::abs(f_rec));
    a.stationary       = gnorm <= a.stationarity_rhs;
  }

  // v1 advisory policy: report Converged ONLY when geometrically stationary.
  // Otherwise the audit refuses to volunteer a refinement — the wrapper still
  // owns the reported code, and the wrapper is the one that knows whether the
  // backend's verdict was "clean stop", "soft failure", or "singular". A
  // non-stationary advisory `Unknown` is the correct null signal.
  if (a.stationary) a.advisory_status = OptimStatus::Converged;
  return a;
}

}  // namespace magmaan::optim
