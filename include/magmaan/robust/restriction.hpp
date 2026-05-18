#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;

// The restriction Jacobian of a *nested* model pair, expressed in the
// less-restricted model's reduced parameter space.
//
// Setup.  Let H1 be a model with `r1` free reduced parameters α₁ such that the
// admissible θ-vectors are `{ θ₀_H1 + K_H1·α₁ }`.  Let H0 be a strictly more
// restricted model, with `r0 < r1` reduced parameters α₀ and admissible set
// `{ θ₀_H0 + K_H0·α₀ } ⊆ { θ₀_H1 + K_H1·α₁ }`.
//
// Then there exist
//   M  (r1 × r0)            with K_H0 = K_H1 · M  (range inclusion),
//   b₀ (r1)                 with θ₀_H0 = θ₀_H1 + K_H1·b₀ (shift),
// and the H0 manifold inside H1's α-space is the affine subspace
//
//   {α₁ ∈ R^{r1} : A·α₁ = b}    with  A = (basis of null(Mᵀ))ᵀ  (m × r1),
//                                     b = A·b₀                  (m-vector),
//                                     m = r1 − r0.
//
// This is exactly the restriction Jacobian Satorra-2000 needs at the level of
// the H1 reduced parameter; the Satorra trick then forms an `m × m` reduced
// problem instead of the ambient `p* × p*` UΓ matrix.
//
// The orthonormalisation is canonical only up to right-multiplication by an
// `m × m` orthogonal: any (A, b) with the same row-space and corresponding
// shift gives the same eigenvalues and traces downstream.
struct RestrictionAlpha {
  Eigen::MatrixXd A;     // m × r1; orthonormal rows
  Eigen::VectorXd b;     // m-vector; the inhomogeneous shift
};

// Derive `RestrictionAlpha` from the two fits' affine reparameterisations.
//
// Preconditions:
//   • Both K_H1 and K_H0 act on the same npar-vector θ_full
//     (same lavaanified partable shape — same number of θ slots).
//   • r1 ≥ r0 (H0 cannot be less restricted than H1).
//
// Returns `PostError::Kind::NumericIssue` when
//   • the npar dimensions disagree;
//   • the range-inclusion residual ‖K_H1·M − K_H0‖_F exceeds the threshold
//     `tol_range_F`, i.e. H0 is *not* a sub-model of H1;
//   • the resulting null-space dimension fails to match `r1 − r0`.
//
// Tolerance default 1e-8 is conservative for typical SVD numerical noise on
// orthonormal K matrices (~1e-14 per entry) scaled up to r·npar entries.
post_expected<RestrictionAlpha>
restriction_alpha_from_K(const EqConstraints& K_H1,
                         const EqConstraints& K_H0,
                         double               tol_range_F = 1e-8,
                         double               tol_singular = 1e-9);

// lavaan-style `A.method = "delta"` restriction: derive A from the column
// spaces of the H1/H0 moment Jacobians, both already projected into their
// reduced parameter spaces.
post_expected<Eigen::MatrixXd>
restriction_alpha_delta_from_jacobians(
    const Eigen::Ref<const Eigen::MatrixXd>& Pi_H1_alpha,
    const Eigen::Ref<const Eigen::MatrixXd>& Pi_H0_alpha,
    int                                      expected_m,
    double                                   tol_singular = 1e-10);

}  // namespace magmaan::robust
