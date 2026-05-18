#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::measures {

using data::SampleStats;
using estimate::Estimates;

// Raw residual moment matrices — the discrepancy between the sample moments
// and the model-implied moments at θ̂ (lavaan's `lavResiduals(fit, type =
// "raw")` / `resid(fit)`):
//
//   cov[b]  = S_b − Σ̂_b(θ̂)      (p_b × p_b, symmetric)
//   mean[b] = m̄_b − μ̂_b(θ̂)      (p_b);  an empty vector for a block
//                                         without mean structure
//
// One entry per block; `mean[b]` is empty (size 0) when block `b` carries no
// mean structure, so `cov` and `mean` always index in lock-step. Σ̂ depends
// only on θ̂, so the residual is estimator-agnostic — well-defined for ML and
// least-squares fits alike.
struct ResidualMoments {
  std::vector<Eigen::MatrixXd> cov;
  std::vector<Eigen::VectorXd> mean;
};

// `pt` is taken by value because `residuals` resolves fixed.x `fixed_value`s
// from `samp` internally — symmetric with `fit_extras` / `information_expected`.
// Returns `PostError::Kind::NumericIssue` on an evaluator-build or θ-size
// mismatch.
post_expected<ResidualMoments>
residuals(spec::LatentStructure pt, const model::MatrixRep& rep,
          const SampleStats& samp, const Estimates& est);

// Standardized / correlation-metric residual table — magmaan's analogue of
// lavaan's `lavResiduals(fit)`. It reports the deterministic residual metrics:
//
//   cov_raw[b]  = S_b − Σ̂_b                        (== residuals().cov)
//   cov_cor[b]  = correlation-metric residual, Bentler standardization:
//                   off-diagonal  (s_ij − σ̂_ij)/√(s_ii·s_jj)
//                   diagonal      (s_ii − σ̂_ii)/s_ii
//                 — the same metric the SRMR sums over (`fit_extras`).
//   mean_raw[b] = m̄_b − μ̂_b           (empty for a block without means)
//   mean_cor[b] = (m̄_i − μ̂_i)/√(s_ii)  (empty for a block without means)
//   srmr        = Bentler-type SRMR (== fit_extras().srmr)
//
// `cov_cor[b]` is returned as a full symmetric matrix. `cov_se` and `cov_z`
// follow lavaan's default continuous `lavResiduals(type = "cor.bentler")`
// standardized residual convention: project the H1 normal-theory moment ACOV
// through the fitted-model tangent space, then rescale by the observed sample
// variances. Near-zero residuals use lavaan's z-stat safeguard and divide by
// 1, so exact fitted diagonal residuals remain numerically zero.
struct StandardizedResiduals {
  std::vector<Eigen::MatrixXd> cov_raw;
  std::vector<Eigen::MatrixXd> cov_cor;
  std::vector<Eigen::MatrixXd> cov_se;
  std::vector<Eigen::MatrixXd> cov_z;
  std::vector<Eigen::VectorXd> mean_raw;
  std::vector<Eigen::VectorXd> mean_cor;
  std::vector<Eigen::VectorXd> mean_se;
  std::vector<Eigen::VectorXd> mean_z;
  double                       srmr = 0.0;
};

post_expected<StandardizedResiduals>
standardized_residuals(spec::LatentStructure pt, const model::MatrixRep& rep,
                       const SampleStats& samp, const Estimates& est);

}  // namespace magmaan::measures
