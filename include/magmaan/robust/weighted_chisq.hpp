#pragma once

#include <Eigen/Core>

namespace magmaan::nt::robust {


// Tail probability of a weighted sum of independent central χ²₁ variates:
//
//     Q = Σⱼ λⱼ · Xⱼ ,    Xⱼ  ~ᵢᵢᵈ  χ²₁ ,
//     imhof_upper(λ, x) = Pr(Q > x).
//
// Imhof (Biometrika 48, 1961): single 1-D quadrature
//
//     Pr(Q > x) = ½ − (1/π) ∫₀^∞ sin θ(u) / (u · ρ(u))  du
//     θ(u)      = ½ Σⱼ arctan(λⱼ u) − ½ x u
//     ρ(u)      = Πⱼ (1 + λⱼ² u²)^{1/4}
//
// The integrand has a removable 0/0 at u = 0 (lim = (Σλⱼ − x)/2); it then
// oscillates while damping as u → ∞.  We integrate via adaptive Simpson on
// dyadic intervals [0, 2^n] until the next interval's contribution falls
// below `rel_tol`, hard-capped at 2^30 to keep pathological inputs bounded.
//
// Preconditions:
//   • `lambda` non-empty, all entries non-negative (caller clips tiny
//     negatives from generalised-eigenvalue noise to 0 before passing).
//   • `x` finite.
//
// The returned probability is clamped to [0, 1]; the unclamped value can
// drift by a small ε for very tight tolerances and Q in the deep tails.
// `max_subdivisions` caps the Simpson recursion depth per interval (so the
// total function evaluation count is at most O(max_subdivisions · log U)).
double imhof_upper(const Eigen::Ref<const Eigen::VectorXd>& lambda,
                   double x,
                   double rel_tol         = 1e-7,
                   int    max_doublings   = 8);

}  // namespace magmaan::nt::robust
