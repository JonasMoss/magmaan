#pragma once

#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::fit {

// Unweighted least squares (ULS) discrepancy.
//
//   F_ULS(θ) = Σ_b (n_b/N) · ½·[ (m̄_b − μ_b(θ))ᵀ(m̄_b − μ_b(θ))
//                              + vech(S_b − Σ_b(θ))ᵀ vech(S_b − Σ_b(θ)) ]
//
// Each off-diagonal residual is counted ONCE via vech (lower triangle,
// column-major) — there is no duplication-matrix factor of 2 like in the
// ML gradient. F_ULS = 0 at the saturated model (Σ = S, μ = m̄).
//
// Per-block (n_b/N) weighting mirrors the ML convention so multi-group
// behavior is consistent and N·F_ULS still gives the standard reference
// test statistic for ULS.
//
// Gradient (per block, before stacking):
//   ∂F_b/∂θ = (μ_b − m̄_b)ᵀ · Jmu_b  +  vech(Σ_b − S_b)ᵀ · J_b
// stacked into:
//   ∇_θ F = Jᵀ · w  +  Jmuᵀ · u
// with w_b = (n_b/N) · vech(Σ_b − S_b) and u_b = (n_b/N) · (μ_b − m̄_b).
//
// LS-shape interface — `LsDiscrepancy<ULS>` is satisfied via `residuals`
// and `residual_jacobian`. Per block, stacked over blocks in the same order
// the existing scalar `gradient` uses (mean rows first when present, then
// vech rows):
//
//     r_b = √(n_b/N) · [ μ_b(θ) − m̄_b ; vech(Σ_b(θ) − S_b) ]
//     J_b = √(n_b/N) · [ J_μ,b         ; J_σ,b              ]
//
// Sign convention r = model − sample gives `½||r||² = F_ULS` and
// `J_rᵀ·r = ∇_θ F_ULS` (matches the scalar `value`/`gradient` to within
// floating-point noise — see the residual/Jacobian FD-parity tests).
struct ULS {
  static constexpr std::string_view name = "ULS";
  bool is_ml() const noexcept { return false; }

  // F_ULS(θ). Cannot fail on PD grounds — F is finite for any (S, Σ).
  // Errors only on shape mismatch / non-finite result.
  fit_expected<double>
  value(const SampleStats& s, const model::ImpliedMoments& m) const;

  // ∇_θ F_ULS. `J` = ∂vech(Σ)/∂θ (Σ_b vech_len(p_b) rows × n_free cols);
  // `Jmu` = ∂μ/∂θ (Σ_b p_b rows × n_free cols); pass empty for cov-only.
  fit_expected<Eigen::VectorXd>
  gradient(const SampleStats& s, const model::ImpliedMoments& m,
           const Eigen::MatrixXd& J,
           const Eigen::MatrixXd& Jmu = Eigen::MatrixXd()) const;

  // r(θ): stacked residual vector with `½||r||² = F_ULS`. Length is
  // `Σ_b (vech_len(p_b) + (has_means ? p_b : 0))`. The mean/vech mix
  // per block is decided by checking whether both `s.mean[b]` and
  // `m.mu[b]` are populated — same gating as `value`/`gradient`. The
  // `has_means` choice is driven by `m.mu[b]` being non-empty for any
  // block; if `m.mu` is consistently empty the layout is vech-only.
  fit_expected<Eigen::VectorXd>
  residuals(const SampleStats& s, const model::ImpliedMoments& m) const;

  // J_r(θ) = ∂r/∂θ. Shape: `r.size() × n_free`. The rows are sliced from
  // (the `√(n_b/N)`-scaled versions of) `J_mu` (mean rows) and `J_sigma`
  // (vech rows) per block; `J_mu` empty ⇒ vech-only layout.
  fit_expected<Eigen::MatrixXd>
  residual_jacobian(const SampleStats& s, const model::ImpliedMoments& m,
                    const Eigen::MatrixXd& J_sigma,
                    const Eigen::MatrixXd& J_mu = Eigen::MatrixXd()) const;
};

}  // namespace magmaan::fit
