#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/frontier/noniterative_cfa.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

// Residual-based inference for the non-iterative CFA estimators of
// estimate/frontier/noniterative_cfa.hpp. For a closed-form estimator θ̂ = τ(s)
// with map Jacobian J = ∂τ/∂s and model Jacobian Δ = ∂σ/∂θ', the residual
// projector is M = I − ΔJ and every output is one contraction of M, a
// discrepancy weight V, and the fourth-moment matrix Γ:
//   • standard errors  Ω = J Γ Jᵀ / N,
//   • goodness of fit  T = N rᵀ V r  with reference Σλⱼ χ²₁ = eig(M'VM · Γ),
//   • nested tests      Wald (exact χ²) and a difference / pseudo-LRT.
// Theory: docs/research/notes/noniterative_cfa_tests.tex. The reducer, Γ, and
// Wald machinery are reused; only M (built from the estimator's own J, not the
// minimizer's Γ-orthogonal projector) is new. Scope matches the estimator map:
// continuous, marker-identified, simple-structure CFA, with grouped and
// mean-structure paths below.

namespace magmaan::robust::frontier {

// Which discrepancy defines the fit statistic and its weight V.
//   ULS  → V = D'D (unit on variances, 2 on covariances). Model-free; robust.
//   NTML → V = ½D'(Σ(θ̂)⁻¹⊗Σ(θ̂)⁻¹)D, the model-implied normal-theory (RLS)
//          weight; NOT the sample GLS weight.
enum class Discrepancy : std::uint8_t { ULS, NTML };

struct NonIterativeInference {
  // Delta-method standard errors.
  Eigen::MatrixXd Omega;  // q × q = J Γ Jᵀ / N
  Eigen::VectorXd se;     // √diag(Ω), length q

  // Goodness of fit: T = N rᵀ V r, referred to the weighted-χ² spectrum.
  double          T_gof = 0.0;
  int             df    = 0;                 // p* − q
  Eigen::VectorXd gof_eigenvalues;           // df λⱼ of (M'VM)·Γ
  double          scale_c          = 0.0;    // Σλ / df (Satorra-Bentler)
  double          p_scaled         = 0.0;    // χ²(df) tail of T/c
  double          p_meanvar        = 0.0;    // Satterthwaite mean-and-variance
  double          p_scaled_shifted = 0.0;
  double          p_mixture        = 0.0;    // exact Σλⱼχ²₁ tail
  double          rls_check        = 0.0;    // NTML cross-check via rls_chi2 (NaN for ULS)

  // Retained pieces the nested tests need.
  Eigen::MatrixXd J, Delta, M, V, Gamma, U;  // U = M'VM

  std::vector<std::string> warnings;
};

// Core: inference at θ̂ with a caller-supplied fourth-moment matrix Γ (p* × p*,
// lower-triangle column-major, matching Δ). This is the Γ-agnostic seam — a
// future polychoric-ADF Γ drops in here unchanged.
post_expected<NonIterativeInference>
noniterative_inference(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                       const data::SampleStats& samp, const Eigen::VectorXd& theta,
                       estimate::frontier::NonIterativeEstimator which,
                       Discrepancy disc, const Eigen::MatrixXd& gamma);

// Convenience: normal-theory Γ = Γ_NT(Σ(θ̂)).
post_expected<NonIterativeInference>
noniterative_inference_nt(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                          const data::SampleStats& samp, const Eigen::VectorXd& theta,
                          estimate::frontier::NonIterativeEstimator which, Discrepancy disc);

// Convenience: empirical (distribution-free) Γ̂ from complete raw data.
post_expected<NonIterativeInference>
noniterative_inference_empirical(const spec::LatentStructure& pt,
                                 const model::MatrixRep& rep, const data::SampleStats& samp,
                                 const data::RawData& raw, const Eigen::VectorXd& theta,
                                 estimate::frontier::NonIterativeEstimator which,
                                 Discrepancy disc);

// Same inferential algebra as `noniterative_inference*`, but the finite-
// difference Jacobian is taken over `fit_noniterative_cfa_restricted()`.
post_expected<NonIterativeInference>
noniterative_inference_restricted(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    const Eigen::MatrixXd& gamma);

post_expected<NonIterativeInference>
noniterative_inference_restricted_nt(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc);

post_expected<NonIterativeInference>
noniterative_inference_restricted_empirical(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc);

// Wald test of the linear restriction R·θ = q, using the delta-method Ω.
// Exact χ²(R.rows()); the primary nested test.
post_expected<inference::WaldTestResult>
noniterative_wald(const Eigen::VectorXd& theta, const NonIterativeInference& inf,
                  const Eigen::MatrixXd& R, const Eigen::VectorXd& q);

// Difference / pseudo-LRT: T_d = T_0 − T_1 with reference = spectrum of
// (U_0 − U_1)·Γ. Anchored at H1 (inf1's V and Γ). Secondary: T_d can be
// negative and the spectrum indefinite for a non-minimizing estimator.
struct NonIterativeDiffTest {
  double          T_d  = 0.0;
  int             df_d = 0;
  Eigen::VectorXd eigenvalues;
  double          p_scaled         = 0.0;
  double          p_adjusted       = 0.0;
  double          p_scaled_shifted = 0.0;
  double          p_mixture        = 0.0;
  std::vector<std::string> warnings;
};

post_expected<NonIterativeDiffTest>
noniterative_difference_test(const NonIterativeInference& inf0 /*restricted H0*/,
                             const NonIterativeInference& inf1 /*full H1*/, int df_d);

// ---------------------------------------------------------------------------
// Multi-group / grouped inference and linearly-constrained fits.
// ---------------------------------------------------------------------------

// Per-block (group/level) Guttman fits with a block-diagonal Ω (blocks are
// independent samples, so Ω = Σ_b J_b Γ_b J_bᵀ / N_b is block-diagonal) and a
// joint GOF (T = Σ_b N_b r_bᵀ V_b r_b, df = Σ_b (p*_b − q_b), reference spectrum
// = the union of the per-block spectra). Reduces to `noniterative_inference`
// numerically when there is a single block. Configural: no cross-block
// constraints are imposed here; `noniterative_constrained_fit` projects onto
// them afterwards.
struct GroupedNonIterativeInference {
  Eigen::VectorXd theta_hat;                 // stacked configural θ̂ (q)
  Eigen::MatrixXd Omega;                     // q × q, block-diagonal
  Eigen::VectorXd se;                        // √diag(Ω)
  double          T_gof = 0.0;
  int             df    = 0;
  Eigen::VectorXd gof_eigenvalues;
  double          scale_c          = 0.0;
  double          p_scaled         = 0.0;
  double          p_meanvar        = 0.0;
  double          p_scaled_shifted = 0.0;
  double          p_mixture        = 0.0;
  double          rls_check        = 0.0;    // NTML cross-check (Σ_b), NaN for ULS
  std::vector<std::int32_t> block_of_param;  // q; from param_locations
  // Covariance-moment pieces retained for grouped pseudo-LRTs.
  Eigen::MatrixXd M, V, U, Gamma;            // Σ_b p*_b square
  std::vector<std::string> warnings;
};

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                               const data::SampleStats& samp, const Eigen::VectorXd& theta,
                               estimate::frontier::NonIterativeEstimator which,
                               Discrepancy disc,
                               const std::vector<Eigen::MatrixXd>& gamma_per_block);

// Convenience: per-block normal-theory Γ = Γ_NT(Σ_b(θ̂)).
post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_nt(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                                  const data::SampleStats& samp, const Eigen::VectorXd& theta,
                                  estimate::frontier::NonIterativeEstimator which,
                                  Discrepancy disc);

// Convenience: per-block empirical (distribution-free) Γ̂_b from raw.X[b].
post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_empirical(const spec::LatentStructure& pt,
                                         const model::MatrixRep& rep,
                                         const data::SampleStats& samp,
                                         const data::RawData& raw, const Eigen::VectorXd& theta,
                                         estimate::frontier::NonIterativeEstimator which,
                                         Discrepancy disc);

// Grouped inference for the restricted estimator map. Equality constraints can
// couple blocks, so the residual projector is assembled on the full stacked
// covariance-moment space rather than through the configural block-diagonal
// shortcut.
post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_restricted(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    const std::vector<Eigen::MatrixXd>& gamma_per_block);

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_restricted_nt(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc);

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_restricted_empirical(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc);

post_expected<NonIterativeDiffTest>
noniterative_difference_test(
    const GroupedNonIterativeInference& inf0 /*restricted H0*/,
    const GroupedNonIterativeInference& inf1 /*full H1*/,
    int df_d);

// One-step minimum-distance projection of the configural fit onto the linear
// equality constraint surface R θ = c, where (R, c, k) = (A_eq, b_eq, rank) from
// build_eq_constraints(pt). Closed-form:
//   θ̃ = θ̂ − Ω R'(R Ω R')⁻¹(R θ̂ − c),  Ω̃ = Ω − Ω R'(R Ω R')⁻¹ R Ω  (rank q−k),
// and the constraint test W = (R θ̂ − c)'(R Ω R')⁻¹(R θ̂ − c) is exact χ²_k
// (Wald = min-distance duality; Ω is the true covariance, so W is not a mixture).
// This is the closed-form measurement-invariance test (metric/strict) and the
// general linear-constraint fit. Errors on inequality / nonlinear constraints
// (via build_eq_constraints). k == 0 (no constraints) returns the configural fit.
struct ConstrainedNonIterativeFit {
  Eigen::VectorXd theta_hat, theta_tilde;
  Eigen::MatrixXd Omega, Omega_tilde;        // Ω̃ has rank q − k
  Eigen::VectorXd se_constrained;            // √diag(Ω̃)
  double          W      = 0.0;              // χ²_k constraint statistic
  int             k      = 0;                // # independent constraints
  double          p_wald = 0.0;
  std::vector<std::string> warnings;
};

post_expected<ConstrainedNonIterativeFit>
noniterative_constrained_fit(const spec::LatentStructure& pt,
                             const GroupedNonIterativeInference& inf);

// ---------------------------------------------------------------------------
// True (lavaan-compatible) scalar invariance via the reference-group mean map.
// ---------------------------------------------------------------------------
//
// Scalar invariance frees the latent means (α_g) instead of fixing them at 0.
// Given the configural mean-structure fit `inf` (Phase B: free intercepts
// ν_g = m_g with a mean-augmented Ω) and a reference group r, the closed-form
// map reads the common metric off the reference group:
//
//   ν      = m_r                         (reference intercept, α_r = 0)
//   α_g    = (Λ_r'Λ_r)⁻¹ Λ_r'(m_g − m_r)  (latent means, LS through col(Λ_r))
//   d_g    = (I − P_{Λ_r})(m_g − m_r)      (mean residual ⟂ col(Λ_r))
//
// where P_{Λ_r} projects onto the reference loadings. Scalar invariance holds
// iff every d_g = 0. Unlike the linear-constraint projection, this map is
// Λ-dependent (bilinear), so the test is a *linearized* Wald: to leading order
// ∂d_g/∂Λ_r ≈ −(I − P) (dΛ_r) α_g and ∂d_g/∂m = ±(I − P), and
//   W = d' Cov(d)⁺ d,   Cov(d) = G Ω Gᵀ,   df = (G − 1)(p − #factors),
// with G the stacked ∂d/∂θ and Cov(d) singular of rank df (pseudo-inverse Wald,
// asymptotically exact χ²). Cov(d) couples the non-reference groups through the
// shared reference mean / loadings; Ω supplies that automatically. The latent
// means α_g are shipped as free coordinates with delta-method SEs.
//
// The reference loadings Λ_r are the reference group's configural loadings (the
// natural reference-group metric); a pooled-loading refinement (test given
// metric invariance) is future work. Errors unless the model has mean structure
// and at least two groups with p > #factors.
struct ScalarInvarianceFit {
  std::size_t                  ref_group = 0;
  std::vector<std::size_t>     groups;     // non-reference group indices, in order
  std::vector<Eigen::VectorXd> alpha;      // latent means α_g (m), per non-ref group
  std::vector<Eigen::MatrixXd> alpha_cov;  // Cov(α_g) (m × m)
  std::vector<Eigen::VectorXd> alpha_se;   // √diag(Cov(α_g))
  Eigen::VectorXd              nu;         // common intercept ν = m_ref (p)
  Eigen::VectorXd              d_stacked;  // stacked mean residuals d_g ((G−1)·p)
  double                       W       = 0.0;  // scalar-invariance statistic
  int                          df      = 0;    // (G−1)(p−#factors)
  int                          rank    = 0;    // numeric rank of Cov(d)
  double                       p_value = 0.0;
  std::vector<std::string>     warnings;
};

post_expected<ScalarInvarianceFit>
noniterative_scalar_invariance(const spec::LatentStructure& pt,
                               const model::MatrixRep& rep,
                               const GroupedNonIterativeInference& inf,
                               std::size_t ref_group = 0);

}  // namespace magmaan::robust::frontier
