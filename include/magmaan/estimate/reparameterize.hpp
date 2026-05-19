#pragma once

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/problem.hpp"

// Equality constraints as a closure transformer. A θ-space problem becomes a
// problem over the constraint-reduced α (θ = θ₀ + K·α) — the optimizer then
// sees a plain unconstrained problem in α, and the result's `expand` recovers
// full θ. Constraints stay out of the objective builders entirely; this is
// just another composable transform, like `gmm::gp`'s β→θ profiling.

namespace magmaan::estimate {

// Reduce a least-squares problem to the α parameterization.
//   r_α(α) = r_θ(θ₀ + K·α),   J_α(α) = J_θ(θ₀ + K·α)·K
optim::GmmProblem
reparameterize(const optim::GmmProblem& prob, const EqConstraints& con);

// Reduce a scalar problem to the α parameterization.
//   F_α(α) = F_θ(θ₀ + K·α),   ∇_α F = Kᵀ·∇_θ F
optim::ScalarProblem
reparameterize(const optim::ScalarProblem& prob, const EqConstraints& con);

// Fold per-θ box bounds onto the constraint-reduced α for a *pure-merge*
// reparameterization (each θ_k is a copy of α_{group[k]}, so `con.group` is
// non-empty). A merged α gets the intersection of the boxes of every θ it
// stands in for. Callers pass this only when `con.active()` and the merge is
// pure; the general-linear case has no box-preserving α and is optimized
// unbounded with a post-hoc bound check.
Bounds fold_alpha_bounds(const EqConstraints& con, const Bounds& b);

}  // namespace magmaan::estimate
