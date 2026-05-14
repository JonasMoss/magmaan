#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::fit {

struct MLSampleCache {
  std::vector<double> weight;
  std::vector<double> log_det_S;
  std::int64_t        n_total = 0;
};

struct MLValueGradient {
  double          value = 0.0;
  Eigen::VectorXd gradient;
};

// ML discrepancy.
//
//   F_ML(θ) = sum_b w_b · [ log|Σ_b(θ)| + tr(S_b Σ_b(θ)⁻¹) − log|S_b| − p_b ]
//
// where w_b is the block weight (single block in v0 → w=1). The
// −log|S_b| − p_b constants make F_ML = 0 at the saturated model so
// values are interpretable; they don't affect the optimizer.
//
// Gradient (per block):
//   G_b = Σ_b⁻¹ − Σ_b⁻¹ S_b Σ_b⁻¹
//   ∂F_b/∂θ_k = sum_{i,j} G_b[i,j] · ∂Σ_b[i,j]/∂θ_k
//             = J_b^T · w(G_b)        // see vech-doubled trick in ml.cpp
struct ML {
  static constexpr std::string_view name = "ML";
  bool is_ml() const noexcept { return true; }

  // Returns NumericIssue if any Σ_b is not positive-definite (its log|Σ|
  // is undefined). Cheap to detect via LLT.
  fit_expected<double>
  value(const SampleStats& s, const model::ImpliedMoments& m) const;

  fit_expected<MLSampleCache>
  prepare(const SampleStats& s) const;

  fit_expected<double>
  value(const SampleStats& s, const MLSampleCache& cache,
        const model::ImpliedMoments& m) const;

  // Caller must have called ev.dsigma_dtheta(theta) and passed the result
  // as `J`. Same shape as the J returned by ModelEvaluator::dsigma_dtheta:
  // (sum_b vech_len(p_b)) rows × n_free cols.
  //
  // `Jmu` is the dμ/dθ Jacobian from ModelEvaluator::dmu_dtheta(theta) —
  // shape (sum_b p_b) × n_free. Pass an empty matrix (the default) for
  // covariance-only models; the mean term is skipped automatically. When
  // the model has mean structure, both `ImpliedMoments::mu[b]` and
  // `SampleStats::mean[b]` must be populated for each mean-bearing block.
  fit_expected<Eigen::VectorXd>
  gradient(const SampleStats& s, const model::ImpliedMoments& m,
           const Eigen::MatrixXd& J,
           const Eigen::MatrixXd& Jmu = Eigen::MatrixXd()) const;

  fit_expected<MLValueGradient>
  value_gradient(const SampleStats& s, const MLSampleCache& cache,
                 const model::ImpliedMoments& m,
                 const Eigen::MatrixXd& J,
                 const Eigen::MatrixXd& Jmu = Eigen::MatrixXd()) const;

  // Per-block gradient: just block `block_idx`'s ∂F_b/∂θ contribution
  // (un-weighted — the (n_b/N) scaling lives in the full `gradient()`).
  // Used by `information_observed_fd` to FD-Hessian each block separately so the
  // multi-group info weighting (n_b/2 per block) is correct.
  fit_expected<Eigen::VectorXd>
  gradient_block(const SampleStats& s, const model::ImpliedMoments& m,
                 const Eigen::MatrixXd& J, std::size_t block_idx,
                 const Eigen::MatrixXd& Jmu = Eigen::MatrixXd()) const;
};

}  // namespace magmaan::fit
