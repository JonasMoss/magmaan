#pragma once

#include <Eigen/Core>

#include "magmaan/optim/problem.hpp"  // OptimStatus, ObjectiveFn, TerminalAudit, TerminalAuditOptions

// Terminal audit for optimizer iterates.
//
// "Optimizers propose, the audit disposes." NLopt and PORT both
// confuse two distinct things: (a) the search machinery's procedural verdict
// ("line search could not find a measurable decrease", "trust-region predicted
// reduction disagreed with actual") and (b) the geometric fact at the returned
// point (is the projected gradient small enough that we are sitting on a
// first-order stationary point?). For near-perfect-fit GLS models — the
// floating-point noise floor of F = ½‖s − σ‖²_W is around 1e-8 relative when
// s ≈ σ — every gradient-based line search trips its sufficient-decrease test
// at the optimum because there is no further measurable decrease to make. The
// iterate is correct; the certifier is unhappy. The fix is not "trust the
// optimizer more"; it is to recompute f, ∇f at the returned x and classify by
// geometry.
//
// The audit operates in the DRIVEN coordinates the optimizer minimized over —
// the same `x`, `f`, bounds the backend just saw. It never calls `expand` and
// never knows anything about the user's full θ. Constraint reduction and
// SNLLS profiling are the caller's coordinate system; the audit honours
// whatever the caller passes it.
//
// The audit *observes* — it never retries, falls back, or warm-restarts. That
// is a separate concern (Layer 3, `allFit`-style cross-checks). The wrapper
// owns the final OptimStatus it returns; the audit only suggests (via
// `advisory_status`) and provides the raw geometric facts.
//
// `TerminalAudit` and `TerminalAuditOptions` are defined in `problem.hpp`
// because they are part of the optimizer's output contract; this header
// only declares the free function.

namespace magmaan::optim {

// Re-run the user objective at `x` to fill `f_recomputed` and the gradient,
// build the projected-gradient infinity norm against `lower`/`upper`, apply
// the relative stationarity test, and record an active-set snapshot.
//
// `f`, `x`, `lower`, `upper` are exactly the ones the backend just used.
// `lower`/`upper` may carry ±infinity sentinels (treated as never-active).
// `reported_f` is the `fmin` the backend was about to return.
//
// The audit NEVER calls `expand`; it operates in driven coordinates.
//
// If the sizes of `lower` or `upper` disagree with `x.size()`, the audit
// silently produces a non-finite, non-stationary result rather than asserting
// (the caller may be invoking on a degenerate state; we don't make it worse).
// Likewise non-finite `f(x)` produces `f_finite = false`, `stationary = false`
// and the advisory `Unknown`.
TerminalAudit audit_terminal_iterate(
    const ObjectiveFn&     f,
    const Eigen::VectorXd& x,
    double                 reported_f,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    TerminalAuditOptions   opts = {});

}  // namespace magmaan::optim
