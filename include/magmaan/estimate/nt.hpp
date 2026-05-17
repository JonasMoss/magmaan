#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"

// Normal-theory maximum-likelihood discrepancy, as free functions.
//
//   F_ML(θ) = Σ_b w_b · [ log|Σ_b(θ)| + tr(S_b Σ_b(θ)⁻¹) − log|S_b| − p_b ]
//             + (optional) (m̄_b − μ_b)ᵀ Σ_b⁻¹ (m̄_b − μ_b)
//
// where w_b = n_b/N is the block weight. The −log|S_b| − p_b constants make
// F_ML = 0 at the saturated model. Not a sum of squares ⇒ scalar form only;
// `ml_objective` packages it as an optimizer-ready `optim::ScalarProblem`.

namespace magmaan::estimate {

using data::SampleStats;

// Per-sample ML cache — block weights (n_b/N) and log|S_b|, built once by
// `ml_prepare` and reused across optimizer iterations.
struct MlCache {
  std::vector<double> weight;
  std::vector<double> log_det_S;
  std::int64_t        n_total = 0;
};

struct MlValueGradient {
  double          value = 0.0;
  Eigen::VectorXd gradient;
};

// Pre-compute the per-sample cache. Fails NonPositiveDefiniteSample if any
// sample S_b is not PD.
fit_expected<MlCache>
ml_prepare(const SampleStats& s);

// F_ML(θ). The cache-free overload calls `ml_prepare` internally; pass a
// cache on the hot path. Fails NonPositiveDefiniteSigma if any Σ_b is not PD.
fit_expected<double>
ml_value(const SampleStats& s, const model::ImpliedMoments& m);
fit_expected<double>
ml_value(const SampleStats& s, const MlCache& cache,
         const model::ImpliedMoments& m);

// ∇_θ F_ML. `J` = ∂vech(Σ)/∂θ; `Jmu` = ∂μ/∂θ (empty ⇒ covariance-only).
fit_expected<Eigen::VectorXd>
ml_gradient(const SampleStats& s, const model::ImpliedMoments& m,
            const Eigen::MatrixXd& J,
            const Eigen::MatrixXd& Jmu = Eigen::MatrixXd());

// F_ML and ∇_θ F_ML together — one Cholesky per block.
fit_expected<MlValueGradient>
ml_value_gradient(const SampleStats& s, const MlCache& cache,
                  const model::ImpliedMoments& m,
                  const Eigen::MatrixXd& J,
                  const Eigen::MatrixXd& Jmu = Eigen::MatrixXd());

// Un-weighted per-block gradient ∂F_b/∂θ — used by `information_observed_fd`
// to FD-Hessian each block separately for the multi-group info weighting.
fit_expected<Eigen::VectorXd>
ml_gradient_block(const SampleStats& s, const model::ImpliedMoments& m,
                  const Eigen::MatrixXd& J, std::size_t block_idx,
                  const Eigen::MatrixXd& Jmu = Eigen::MatrixXd());

// Package the ML objective as an optimizer-ready scalar problem. `ev` must
// outlive the returned `ScalarProblem` (its closure borrows it by reference).
fit_expected<optim::ScalarProblem>
ml_objective(const model::ModelEvaluator& ev, const SampleStats& s);

}  // namespace magmaan::estimate
