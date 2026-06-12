#pragma once

#include <Eigen/Core>

namespace magmaan::robust {

struct WeightedChiSquareMoments {
  int    df       = 0;
  double trace    = 0.0;  // Σλ
  double trace_sq = 0.0;  // Σλ²
};

WeightedChiSquareMoments
weighted_chisq_moments(int df,
                       const Eigen::Ref<const Eigen::VectorXd>& eigvals) noexcept;

// Same moments computed directly from the reduced df × df symmetric matrix
// `M = BᵀΓ̂B`. Skips the O(k³) eigendecomposition entirely:
//   trace    = tr(UΓ̂) = tr(M)         = M.diagonal().sum()      — O(k)
//   trace_sq = tr((UΓ̂)²) = tr(M²)      = ‖M‖_F² (M symmetric)    — O(k²)
// Use this when only mean / mean+variance adjustments are needed (SB, MV-adj,
// scaled-shifted). Keep `weighted_chisq_moments(df, eigvals)` for callers
// that need the spectrum itself (Satorra 2000 mixture p-values, Imhof tail).
WeightedChiSquareMoments
weighted_chisq_moments_from_M(int df,
                              const Eigen::Ref<const Eigen::MatrixXd>& M) noexcept;

// Satorra-Bentler scaled chi²: c = Σλ / df, T_SB = T / c.
struct SatorraBentlerResult {
  double chi2_scaled = 0.0;
  double scale_c     = 0.0;
  int    df          = 0;
};
SatorraBentlerResult
satorra_bentler(double t_ml, const WeightedChiSquareMoments& moments) noexcept;
SatorraBentlerResult
satorra_bentler(double                                    t_ml,
                int                                       df,
                const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept;

// Satterthwaite mean-and-variance-adjusted chi².
struct MeanVarAdjustedResult {
  double chi2_adj = 0.0;
  double df_adj   = 0.0;
};
MeanVarAdjustedResult
mean_var_adjusted(double t_ml, const WeightedChiSquareMoments& moments) noexcept;
MeanVarAdjustedResult
mean_var_adjusted(double                                    t_ml,
                  int                                       df,
                  const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept;

// Scaled-and-shifted chi²: T_adj = T * a + b, df unchanged.
struct ScaledShiftedResult {
  double chi2_adj = 0.0;
  int    df       = 0;
  double scale_a  = 0.0;
  double shift_b  = 0.0;
};
ScaledShiftedResult
scaled_shifted(double t_ml, const WeightedChiSquareMoments& moments) noexcept;
ScaledShiftedResult
scaled_shifted(double                                    t_ml,
               int                                       df,
               const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept;

// Tail probability of a weighted sum of independent central χ²₁ variates:
//
//     Q = Σⱼ λⱼ · Xⱼ ,    Xⱼ  ~ᵢᵢᵈ  χ²₁ ,
//     imhof_upper(λ, x) = Pr(Q > x).
//
// Positive spectra are evaluated first by Ruben's non-negative central-χ²
// series, which avoids oscillatory cancellation in the deep tail. If that
// series does not converge, the implementation falls back to Imhof
// (Biometrika 48, 1961): single 1-D quadrature
//
//     Pr(Q > x) = ½ − (1/π) ∫₀^∞ sin θ(u) / (u · ρ(u))  du
//     θ(u)      = ½ Σⱼ arctan(λⱼ u) − ½ x u
//     ρ(u)      = Πⱼ (1 + λⱼ² u²)^{1/4}
//
// The integrand has a removable 0/0 at u = 0 (lim = (Σλⱼ − x)/2); it then
// oscillates while damping as u → ∞.  We integrate via QUADPACK qagi, with a
// dense Simpson fallback for weakly damped small-df tails.
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

}  // namespace magmaan::robust
