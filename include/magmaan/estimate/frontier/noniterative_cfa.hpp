#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"

// Non-iterative CFA estimators as covariance maps τ: vech(S) ↦ θ, plus the
// finite-difference Jacobian J = ∂τ/∂vech(S) that turns any such map into a
// delta-method-inferable estimator. Unlike the start-value producers in
// start_values.hpp (which only fill free loadings and clamp), these return a
// COMPLETE parameter vector (Λ, Φ, ψ) so σ(θ̂) and the residual r = vech(S) −
// σ(θ̂) are well-defined; the theory is in docs/research/notes/
// noniterative_cfa_tests.tex and guttman_cfa_asymptotics.tex.
//
// Scope: multi-group (one block per group/level, fit independently and stacked),
// pure CFA with marker (fixed unit loading) identification, continuous data, and
// simple-structure indicators. Mean structure is supported for free intercepts
// with latent means fixed at 0 (ν_g = m_g saturated); free latent means (true
// scalar invariance) are handled by the reference-group mean map downstream, not
// here. Cross-group / general linear equality constraints (measurement
// invariance, tau-equivalence, ...) are imposed downstream by the
// minimum-distance projection in robust::frontier, not here: the map always
// produces the configural (per-block unconstrained) θ̂. The
// `NonIterativeEstimator` enum is the generality seam: FABIN2/Bentler-1982/
// James-Stein/MIIV-2SLS slot in as further maps without any change to the
// residual-based inference bundle that consumes (τ, J).

namespace magmaan::estimate::frontier {

enum class NonIterativeEstimator : std::uint8_t {
  Guttman,            // legacy lavaan-like Spearman / incidence Guttman map
  GuttmanGlsAligned,  // block-GLS H diagonal plus aligned score reconstruction
};

// The pure map τ: vech(samp.S) ↦ full θ̂ (size ev.n_free()). Deterministic given
// (pt, rep, ev); reads only `samp.S`. `ev` supplies the free-parameter layout
// (structural, S-independent) and is reused verbatim across the FD perturbations
// of `estimator_map_jacobian`. Returns an error (never clamps) when the
// interior-point assumptions fail (a factor with < 3 indicators, a singular
// score covariance, a zero marker loading) or the model is out of scope (a
// structural part, a cross-loading, a markerless factor, a residual covariance,
// free latent means, or std.lv identification).
fit_expected<Eigen::VectorXd>
noniterative_cfa_theta(const spec::LatentStructure& pt,
                       const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev,
                       const data::SampleStats& samp,
                       NonIterativeEstimator which = NonIterativeEstimator::Guttman);

// Diagnostic wrapper: builds the evaluator internally, returns θ̂ plus the clean
// per-block matrices (Φ is the latent covariance = magmaan's Ψ; ψ is the
// residual-variance diagonal = magmaan's Θ diagonal).
struct NonIterativeFit {
  Eigen::VectorXd              theta;
  std::vector<Eigen::MatrixXd> Lambda;
  std::vector<Eigen::MatrixXd> Phi;
  std::vector<Eigen::VectorXd> psi;
};

fit_expected<NonIterativeFit>
fit_noniterative_cfa(const spec::LatentStructure& pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     NonIterativeEstimator which = NonIterativeEstimator::Guttman);

// J_block = ∂θ / ∂vech(S_block), shape q × p*_block (q = ev.n_free(),
// p*_block = p_block(p_block+1)/2), by central finite differences of the
// multi-block map w.r.t. block `block`'s covariance only. Column k is the
// derivative w.r.t. the k-th lower-triangle column-major vech(S_block)
// coordinate. Rows for parameters in other blocks are exactly zero (independent
// blocks), so the inference layer slices out the rows with
// param_location.block == block to form the per-block Jacobian. This ordering
// matches ModelEvaluator::dsigma_dtheta and data::gamma_nt so J aligns with Δ
// and Γ. `rel_step` scales the per-coordinate step by max(|S_rc|, 1).
fit_expected<Eigen::MatrixXd>
estimator_map_jacobian_block(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             const model::ModelEvaluator& ev,
                             const data::SampleStats& samp,
                             NonIterativeEstimator which,
                             std::size_t block, double rel_step = 1e-6);

// Single-block convenience wrapper (block 0), preserving the original signature.
fit_expected<Eigen::MatrixXd>
estimator_map_jacobian(const spec::LatentStructure& pt,
                       const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev,
                       const data::SampleStats& samp,
                       NonIterativeEstimator which = NonIterativeEstimator::Guttman,
                       double rel_step = 1e-6);

}  // namespace magmaan::estimate::frontier
