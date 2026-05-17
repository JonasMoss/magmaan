#pragma once

#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"

// The moment-quadratic objective family — F(θ) = ½·r(θ)ᵀ W r(θ) — as free
// functions. ULS/DWLS/WLS/GLS are all this objective; they differ only in the
// supplied weight:
//
//   ULS  → empty weight (identity; no whitening matmul)
//   GLS  → normal_theory_weight(...)        (W from S⁻¹)
//   WLS  → caller-supplied full weight
//   DWLS → caller-supplied diagonal weight, as a (dense) Weight
//
// `residuals` packages the objective as an optim::GmmProblem; the weight is
// Cholesky-factored once at build time.

namespace magmaan::estimate::gmm {

// Per-block weight, aligned to the stacked [mean ; vech(cov)] moment vector
// of each block. Empty ⇒ identity weight (ULS). Each entry is symmetric PD.
using Weight = std::vector<Eigen::MatrixXd>;

// Build the moment-quadratic least-squares problem F(θ) = ½‖r̃(θ)‖².
// `theta0` is evaluated once to fix the moment layout (block dimensions +
// whether the model carries mean structure); `weight` (empty ⇒ identity) is
// validated and Cholesky-factored once here. `ev` and `samp` must outlive the
// returned problem — its closures borrow them.
fit_expected<optim::GmmProblem>
residuals(const model::ModelEvaluator& ev, const data::SampleStats& samp,
          const Eigen::VectorXd& theta0, const Weight& weight = {});

// Normal-theory (GLS) weight: per block, the normal-theory asymptotic-
// covariance inverse built from the sample covariance S, over the
// [mean ; vech(cov)] layout. `theta0` is evaluated once to detect mean
// structure. Pass the result straight to `residuals` as its `weight`.
fit_expected<Weight>
normal_theory_weight(const model::ModelEvaluator& ev,
                     const data::SampleStats& samp,
                     const Eigen::VectorXd& theta0);

}  // namespace magmaan::estimate::gmm
