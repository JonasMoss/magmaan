#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/fit.hpp"          // Estimates
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/partable/partable.hpp"

namespace magmaan::fit {

// Standardized parameter values (same indexing as `Estimates::theta`) and
// delta-method SEs of the standardized values.
struct StandardizedSolution {
  Eigen::VectorXd theta;
  Eigen::VectorXd se;
};

// `std.lv` standardization: latent variables are scaled to unit total
// model-implied variance. Λ loadings are rescaled by √diag(AΨAᵀ),
// structural Β coefficients by the predictor/outcome latent SD ratio, and
// Ψ rows by the corresponding total latent SDs. Indicator-side parameter
// kinds (Θ, Nu) pass through as identity transforms.
//
// SEs come from the delta method: SE_std = √diag(J · vcov · Jᵀ) where
// J is the (n_free × n_free) Jacobian of the transformation.
post_expected<StandardizedSolution>
standardize_lv(const partable::LatentStructure& pt,
               const model::MatrixRep&   rep,
               const Estimates&          est,
               const Eigen::MatrixXd&    vcov);

// `std.all` standardization: on top of `std.lv` (latent variables → unit
// total variance), observed indicators are also standardized
// to unit variance — every parameter that touches indicator i is divided
// by an additional √σ_ii (where σ_ii is the model-implied variance of
// indicator i, evaluated at θ̂).
//
//   Λ_{r,c} → λ · √Var(η_c) / √σ_rr    (latent + indicator scale)
//   Θ_{r,r} → θ_rr / σ_rr              (residual variance ratio)
//   ν_r     → ν_r / √σ_rr              (intercept in SD units)
//   α_j     → α_j / √Var(η_j)          (latent mean in latent-SD units)
//   β_ij    → β_ij · √Var(η_j)/√Var(η_i)
//   ψ_jk    → ψ_jk / √(Var(η_j)·Var(η_k))
//   Θ off-diagonals pass through as identity in v0.
//
// SEs via delta method on the Jacobian of the transformation, which
// includes terms from ∂σ_ii/∂θ (from `dsigma_dtheta`) for the indicator
// scale, plus the latent-scale terms shared with `std.lv`.
post_expected<StandardizedSolution>
standardize_all(const partable::LatentStructure& pt,
                const model::MatrixRep&   rep,
                const Estimates&          est,
                const Eigen::MatrixXd&    vcov);

}  // namespace magmaan::fit
