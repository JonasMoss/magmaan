#include "magmaan/estimate/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"          // active_bounds
#include "magmaan/estimate/constraints.hpp"     // EqConstraints
#include "magmaan/estimate/nl_constraints.hpp"  // NonlinearEqConstraints
#include "magmaan/model/model_evaluator.hpp"    // sigma, ImpliedMoments

namespace magmaan::estimate {

FitDiagnostics
finalize_fit_diagnostics(const Eigen::VectorXd&        theta_full,
                         const model::ModelEvaluator&  ev,
                         const EqConstraints&          con,
                         const NonlinearEqConstraints& nl,
                         const Bounds&                 bounds,
                         bool                          snlls_profile_fallback_flag,
                         DiagnosticsOptions            opts) {
  FitDiagnostics d;
  d.snlls_profile_fallback = snlls_profile_fallback_flag;

  // --- Implied-Σ positive-definiteness per group ------------------------
  // Reuses the pattern from src/measures/fit_measures.cpp:130-135 — LLT on
  // the symmetrized matrix; the symmetrize step is conservative because Σ(θ)
  // is mathematically symmetric but its numerical realisation can pick up
  // roundoff asymmetry that nudges LLT to spurious failure.
  auto moments = ev.sigma(theta_full);
  if (!moments.has_value()) {
    // Evaluator could not produce Σ at all (non-finite θ, etc.). Leave the
    // per-block vector empty and the `_all` flag false — consumers reading
    // either field see the absence.
    d.sigma_pd_all = false;
  } else {
    const auto& sig = moments->sigma;
    d.sigma_pd_per_block.reserve(sig.size());
    bool all_pd = !sig.empty();
    for (const auto& Sigma : sig) {
      const Eigen::MatrixXd Sym = 0.5 * (Sigma + Sigma.transpose());
      Eigen::LLT<Eigen::MatrixXd> llt(Sym);
      const bool pd = (llt.info() == Eigen::Success);
      d.sigma_pd_per_block.push_back(pd);
      if (!pd) all_pd = false;
    }
    d.sigma_pd_all = all_pd;
  }

  // --- Linear equality constraint residual ------------------------------
  // The K-reparameterization enforces `A_eq · θ = b_eq` by construction
  // (`θ = θ₀ + K·α` with `K` in ker(A_eq)); the residual at θ̂ should be
  // either 0 (pure-merge K) or orthonormal roundoff (general-linear). A
  // residual exceeding `lin_eq_residual_tol` indicates the expansion itself
  // is misbehaving — not the optimizer.
  if (con.active() && con.A_eq.rows() > 0 && con.A_eq.cols() == theta_full.size()) {
    const Eigen::VectorXd r = con.A_eq * theta_full - con.b_eq;
    d.lin_eq_residual_inf = r.cwiseAbs().maxCoeff();
    d.lin_eq_satisfied    = d.lin_eq_residual_inf <= opts.lin_eq_residual_tol;
  }
  // No active linear constraints ⇒ residual is 0, satisfied is true (defaults).

  // --- Nonlinear equality constraint residual ---------------------------
  // The augmented-Lagrangian outer loop drives `h(θ̂) → 0`; recording the
  // achieved infinity-norm tells the user whether AL actually converged the
  // constraints (separate from whether the augmented objective stopped).
  // `nl.h()` is sized to `nl.m()`, which `active() == false` reports as 0 —
  // we still evaluate so `nl_eq_residual` is well-defined as the empty vector.
  if (nl.active() && nl.npar == theta_full.size()) {
    d.nl_eq_residual = nl.h(theta_full);
    if (d.nl_eq_residual.size() > 0) {
      d.nl_eq_residual_inf = d.nl_eq_residual.cwiseAbs().maxCoeff();
    }
    d.nl_eq_satisfied = d.nl_eq_residual_inf <= opts.nl_eq_residual_tol;
  }

  // --- Active-bound diagnostic on full θ --------------------------------
  // Reuses estimate::active_bounds(). An empty `bounds` reports nothing
  // active; that's the desired default. The function returns a
  // `post_expected` to be uniform with the rest of the post-fit machinery,
  // but the only error path here would be a size mismatch (caught by the
  // fit composer long before we get here) — we treat any error as "no
  // active bounds recorded" rather than failing the audit.
  auto ab = active_bounds(theta_full, bounds, opts.active_bound_tol);
  if (ab.has_value()) {
    d.active_bounds_full = std::move(*ab);
  }

  return d;
}

}  // namespace magmaan::estimate
