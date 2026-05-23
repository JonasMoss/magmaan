#pragma once

#include <vector>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"  // Bounds, ActiveBoundDiagnostics

// Fit finalization audit (Layer 2).
//
// L1 (`optim::audit_terminal_iterate`) certifies first-order stationarity in
// the *driven* coordinate system the optimizer minimized over. L2 lives on the
// other side of `prob.expand(out->x)`: it operates on full θ and answers a
// different question — "is this fit usable for downstream inference?" — by
// recording diagnostics that SE/χ²/robust-correction code needs to decide
// whether its formulae apply. L2 never blocks a fit; it records.
//
// Concretely:
//   - Implied-Σ positive-definiteness per group. ML/GLS need Σ⁻¹; ULS does
//     not. Consumers decide what they need.
//   - Linear equality constraint residual. The K-reparameterization is exact
//     in the pure-merge case (residual to roundoff) and tiny under
//     general-linear constraints. A non-tiny residual means the expansion
//     itself is broken — a hard correctness signal worth surfacing.
//   - Nonlinear equality constraint residual at θ̂. IPOPT drives
//     `h(θ̂) → 0`; recording the achieved infinity-norm tells the user
//     whether the nonlinear-constraint solve actually converged.
//   - Active-bound set on the FULL expanded θ — this is the Heywood-case
//     detector: a variance parameter at its 0 bound has a one-sided
//     derivative, and the standard info-matrix SE for it is not valid.
//
// Forward-declarations keep the header light; the cpp pulls in the model and
// constraint headers it actually needs.

namespace magmaan::model {
class ModelEvaluator;
}
namespace magmaan::estimate {
struct EqConstraints;
struct NonlinearEqConstraints;
}

namespace magmaan::estimate {

struct FitDiagnostics {
  // Implied Σ Cholesky per group; `sigma_pd_all` is the && over the vector.
  // ML/GLS need PD; ULS only needs finite. Empty when the evaluator could
  // not compute Σ at all (e.g. non-finite θ propagated through model
  // evaluation) — `sigma_pd_all` then defaults to false.
  std::vector<bool>      sigma_pd_per_block;
  bool                   sigma_pd_all = false;

  // Linear equality constraint residual at θ̂ = max |(A_eq · θ - b_eq)_k|.
  // 0 when the model has no linear equality constraints (`EqConstraints::active() == false`).
  double                 lin_eq_residual_inf = 0.0;
  bool                   lin_eq_satisfied    = true;

  // Nonlinear equality constraint residual at θ̂ = h(θ̂). Empty when there
  // are no nonlinear constraints. `_inf` is the infinity-norm; `_satisfied`
  // is `_inf <= nl_eq_residual_tol`.
  Eigen::VectorXd        nl_eq_residual;
  double                 nl_eq_residual_inf  = 0.0;
  bool                   nl_eq_satisfied     = true;

  // Which expanded-θ coordinates sit on a finite bound (the Heywood-case
  // diagnostic). Empty when `bounds.empty()` or no parameter is active.
  ActiveBoundDiagnostics active_bounds_full;

  // SNLLS-only: did the gp expand take the affine fallback `θ₀ + K_β·β`
  // because `profiled(β)` returned an error? v1 leaves this `false` —
  // wiring it requires a flag on `GpProblem`. Recorded here so consumers
  // (and a follow-up PR) have a stable schema slot.
  bool                   snlls_profile_fallback = false;
};

struct DiagnosticsOptions {
  // Active-bound coordinate tolerance — matches the existing
  // `estimate::active_bounds` default (a parameter is "active" if it sits
  // within `1e-6` of a finite bound).
  double active_bound_tol    = 1e-6;
  // The K reparameterization is exact under pure-merge constraints and at
  // worst orthonormal-roundoff under general-linear ones. A residual
  // exceeding 1e-8 is a correctness signal that the expansion machinery,
  // not the optimizer, is misbehaving.
  double lin_eq_residual_tol = 1e-8;
  // Nonlinear equality feasibility target.
  double nl_eq_residual_tol  = 1e-6;
};

// Build a FitDiagnostics from an expanded θ and the prelude bits the fit
// path already carried. Never errors — every check that cannot be performed
// (no bounds, no constraints, no nonlinear block, evaluator error) records a
// benign default that downstream consumers can read uniformly.
//
// `snlls_profile_fallback_flag` is wired only from the SNLLS expand site
// (v1: always `false` — see `FitDiagnostics::snlls_profile_fallback`).
FitDiagnostics
finalize_fit_diagnostics(const Eigen::VectorXd&        theta_full,
                         const model::ModelEvaluator&  ev,
                         const EqConstraints&          con,
                         const NonlinearEqConstraints& nl,
                         const Bounds&                 bounds,
                         bool                          snlls_profile_fallback_flag = false,
                         DiagnosticsOptions            opts = {});

}  // namespace magmaan::estimate
