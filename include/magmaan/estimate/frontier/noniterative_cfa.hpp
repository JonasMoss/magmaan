#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"

// Non-iterative CFA estimators as clean covariance maps τ: vech(S) ↦ θ, plus the
// finite-difference Jacobian J = ∂τ/∂vech(S) that turns any such map into a
// delta-method-inferable estimator. Unlike the start-value producers in
// start_values.hpp (which only fill free loadings and clamp), these return a
// COMPLETE parameter vector (Λ, Φ, ψ) so σ(θ̂) and the residual r = vech(S) −
// σ(θ̂) are well-defined; the theory is in docs/research/notes/
// noniterative_cfa_tests.tex and guttman_cfa_asymptotics.tex.
//
// v1 scope: single group, covariance-only (no mean structure), pure CFA with
// marker (fixed unit loading) identification, continuous data, Guttman's (1952)
// multiple-group map. The `NonIterativeEstimator` enum is the generality seam:
// FABIN2/Bentler-1982/James-Stein/MIIV-2SLS slot in as further maps without any
// change to the residual-based inference bundle that consumes (τ, J).

namespace magmaan::estimate::frontier {

enum class NonIterativeEstimator : std::uint8_t {
  Guttman,  // v1 — Guttman (1952) multiple-group method
};

// The pure map τ: vech(samp.S) ↦ full θ̂ (size ev.n_free()). Deterministic given
// (pt, rep, ev); reads only `samp.S`. `ev` supplies the free-parameter layout
// (structural, S-independent) and is reused verbatim across the FD perturbations
// of `estimator_map_jacobian`. Returns an error (never clamps) when the
// interior-point assumptions fail (a factor with < 3 indicators, a non-positive
// latent variance, a singular factor correlation, a zero marker loading) or the
// model is out of v1 scope (a structural part, mean structure, > 1 block, a
// cross-loading, a markerless factor, a residual covariance, or std.lv
// identification).
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

// J = ∂θ / ∂vech(S), shape q × p* (q = ev.n_free(), p* = p(p+1)/2), by central
// finite differences of `noniterative_cfa_theta`. Column k is the derivative
// w.r.t. the k-th lower-triangle column-major vech(S) coordinate, perturbed as a
// symmetric dS (both (r,c) and (c,r)); this ordering matches
// ModelEvaluator::dsigma_dtheta and data::gamma_nt so J aligns with Δ and Γ.
// `rel_step` scales the per-coordinate step by max(|S_rc|, 1).
fit_expected<Eigen::MatrixXd>
estimator_map_jacobian(const spec::LatentStructure& pt,
                       const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev,
                       const data::SampleStats& samp,
                       NonIterativeEstimator which = NonIterativeEstimator::Guttman,
                       double rel_step = 1e-6);

}  // namespace magmaan::estimate::frontier
