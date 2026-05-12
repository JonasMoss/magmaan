#pragma once

#include <Eigen/Core>

#include "latva/expected.hpp"
#include "latva/fit/fit.hpp"          // Estimates
#include "latva/model/matrix_rep.hpp"
#include "latva/partable/partable.hpp"

namespace latva::fit {

// Standardized parameter values (same indexing as `Estimates::theta`) and
// delta-method SEs of the standardized values.
struct StandardizedSolution {
  Eigen::VectorXd theta;
  Eigen::VectorXd se;
};

// `std.lv` standardization: factor variances are scaled to 1 and Λ
// loadings are rescaled by √ψ_jj. Other parameter kinds are left
// alone in v0 (Θ, Β, Nu, Alpha, Ψ off-diagonals all pass through as
// identity transforms).
//
// SEs come from the delta method: SE_std = √diag(J · vcov · Jᵀ) where
// J is the (n_free × n_free) Jacobian of the transformation.
post_expected<StandardizedSolution>
standardize_lv(const partable::ParTable& pt,
               const model::MatrixRep&   rep,
               const Estimates&          est,
               const Eigen::MatrixXd&    vcov);

// `std.all` standardization: on top of `std.lv` (latent variances → 1,
// loadings rescaled by √ψ_jj), observed indicators are also standardized
// to unit variance — every parameter that touches indicator i is divided
// by an additional √σ_ii (where σ_ii is the model-implied variance of
// indicator i, evaluated at θ̂).
//
//   Λ_{r,c} → λ · √ψ_cc / √σ_rr        (factor + indicator scale)
//   Θ_{r,r} → θ_rr / σ_rr              (residual variance ratio)
//   ν_r     → ν_r / √σ_rr              (intercept in SD units)
//   α_j     → α_j / √ψ_jj              (latent mean in latent-SD units)
//   ψ_jj    → 1, ψ_jk (j≠k) → ψ_jk / √(ψ_jj·ψ_kk)
//   Θ off-diagonals and B coefficients pass through as identity in v0.
//
// SEs via delta method on the Jacobian of the transformation, which
// includes terms from ∂σ_ii/∂θ (from `dsigma_dtheta`) for the indicator
// scale, plus the ψ-only terms shared with `std.lv`.
post_expected<StandardizedSolution>
standardize_all(const partable::ParTable& pt,
                const model::MatrixRep&   rep,
                const Estimates&          est,
                const Eigen::MatrixXd&    vcov);

}  // namespace latva::fit
