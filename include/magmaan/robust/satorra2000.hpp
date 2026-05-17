#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;


// ============================================================================
// Satorra (2000) — scaled difference test for nested SEM models.
//
// For a nested pair (H1 ⊃ H0) with restriction Jacobian `A` (m × k) on the
// unconstrained Jacobian Π = ∂σ/∂θ (p* × k) of the less-restricted H1, the
// non-zero eigenvalues of the (huge) UΓ matrix Satorra-Bentler's scaled
// difference test depends on are exactly the *m* generalised eigenvalues of
//
//     S · v = λ · C · v ,
//     C = A · P⁻¹ · Aᵀ                            (m × m)
//     S = A · P⁻¹ · Πᵀ · V · Γ · V · Π · P⁻¹ · Aᵀ (m × m)
//     P = Πᵀ · V · Π                              (k × k, expected info at H1)
//
// with V the moment-structure weight and Γ the 4th-moment ACOV of the
// sample moments.  The algebra reduces a p* × p* eigenproblem to an m × m
// one (m = df_H0 − df_H1, typically 1–10).
//
// In α-space (after the affine reparameterisation θ = θ₀ + K_H1·α):
//   Π_α = Π · K_H1     (p* × r1)
//   P_α = Π_αᵀ · V · Π_α     (r1 × r1)
//   A_α is the m × r1 restriction matrix from `restriction_alpha_from_K`.
//   Plug those into the formulas above with k ← r1.  This file works in
//   α-space everywhere — callers project Π → Π_α before invoking the core.
//
// Streaming Γ: when Γ is the empirical 4th-moment ACOV
//
//     Γ̂_g = (1 / n_g) · Σᵢ (d_gi − s_g)(d_gi − s_g)ᵀ ,
//     d_gi = vech((x_gi − m̄_g)(x_gi − m̄_g)ᵀ) ,  s_g = vech(S_g) ,
//
// (n divisor — matches magmaan's `empirical_gamma`, the lavaan ML convention).
// The formula for S admits a casewise reduction that *never forms Γ̂*:
//
//     D_g = V_g · Π_α_g · Y          (p*_g × m,  Y = P⁻¹·A_αᵀ)
//     u_gi = D_gᵀ · (d_gi − s_g)     (m-vector)
//     S = Σ_g (weight_g / n_g) · Σᵢ u_gi · u_giᵀ .
//
// One semi-large object per group — `D_g` (p*_g × m) — gets touched; Γ̂_g
// stays implicit.  This is the main performance win of Satorra-2000.
//
// Multi-group: pooled P = Σ_g weight_g · Π_α_gᵀ · V_g · Π_α_g, with
//             weight_g = n_g / N_total to match lavaan's ML F = Σ_g f_g · F_g.
// ============================================================================

// Per-group inputs at θ̂_H1.  Caller is responsible for ensuring Σ is the
// model-implied moments at H1 (used to build V = Γ_NT(Σ)⁻¹), and `mean` is
// the *sample* mean m̄_g (the casewise centring for the d_gi residuals).
struct SatorraGroup {
  Eigen::MatrixXd Pi_alpha;   // p*_g × r1   (already projected through K_H1)
  Eigen::MatrixXd Sigma;      // p_g × p_g   (model-implied Σ̂_g at θ̂_H1)
  Eigen::MatrixXd X;          // n_g × p_g   (raw data, uncentred)
  Eigen::VectorXd mean;       // p_g         (sample mean m̄_g)
  double          weight;     // n_g / N_total
  std::int32_t    n_g;        // sample size (for the 1/(n_g − 1) divisor)
};

enum class GammaSource {
  Empirical,  // Γ̂ from raw data — Satorra-Bentler / MLR
  NT          // Γ_NT(Σ̂)         — sanity check: all eigvals → 1
};

struct SatorraDiffResult {
  Eigen::MatrixXd          C;                 // m × m
  Eigen::MatrixXd          S;                 // m × m
  Eigen::VectorXd          eigenvalues;       // length m, ascending
  double                   trace_CinvS;       // = Σ λⱼ
  double                   trace_CinvS_sq;    // = Σ λⱼ²
  std::vector<std::string> warnings;
};

// Compute (C, S, eigenvalues, trace_CinvS, trace_CinvS_sq) for a nested pair.
// Returns `PostError::Kind::NumericIssue` on:
//   • shape mismatch between A_alpha and per-group Π_α (column counts);
//   • non-PD P (pathological H1 — model not identified at θ̂);
//   • non-PD or rank-deficient C (the restriction degenerates against H1);
//   • non-PD Σ_g for any group (V undefined).
//
// `m == 0` (no actual restriction) short-circuits with all-zero outputs.
post_expected<SatorraDiffResult>
compute_satorra2000(const std::vector<SatorraGroup>& groups,
                    const Eigen::MatrixXd&           A_alpha,
                    GammaSource                      gamma = GammaSource::Empirical);

}  // namespace magmaan::robust
