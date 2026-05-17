#pragma once

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

}  // namespace magmaan::estimate
