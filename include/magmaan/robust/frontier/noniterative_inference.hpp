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
// minimizer's Γ-orthogonal projector) is new. v1 scope matches the estimator
// map: single group, covariance-only, continuous, marker-identified CFA.

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

}  // namespace magmaan::robust::frontier
