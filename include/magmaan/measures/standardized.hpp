#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"          // Estimates
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::measures::standardize {

using data::SampleStats;
using estimate::Estimates;

// Standardized parameter values (same indexing as `Estimates::theta`) and
// delta-method SEs of the standardized values.
struct StandardizedSolution {
  Eigen::VectorXd theta;
  Eigen::VectorXd se;
};

// Native FC-SEM row-level report. Unlike `StandardizedSolution`, this includes
// fixed marker rows (`free == 0`) because lavaan reports their standardized
// values and delta-method SEs even though they are not entries in θ.
struct FcSemStandardizedRow {
  std::size_t  row = 0;      // 0-based LatentStructure row index
  std::string  lhs;
  parse::Op    op = parse::Op::Regression;
  std::string  rhs;
  std::int32_t group = 1;
  std::int32_t free = 0;     // 1-based θ index, 0 for fixed

  double est = std::numeric_limits<double>::quiet_NaN();
  double se = 0.0;
  double std_lv = std::numeric_limits<double>::quiet_NaN();
  double std_lv_se = 0.0;
  double std_all = std::numeric_limits<double>::quiet_NaN();
  double std_all_se = 0.0;
};

// `std.lv` standardization: latent variables are scaled to unit total
// model-implied variance. Λ loadings are rescaled by √diag(AΨAᵀ),
// structural Β coefficients by the predictor/outcome latent SD ratio, and
// Ψ rows by the corresponding total latent SDs. Indicator-side parameter
// kinds (Θ, Nu) pass through as identity transforms.
//
// SEs come from the delta method: SE_std = √diag(J · vcov · Jᵀ) where
// J is the (n_free × n_free) Jacobian of the transformation.
post_expected<StandardizedSolution>
standardize_lv(const spec::LatentStructure& pt,
               const model::MatrixRep&   rep,
               const Estimates&          est,
               const Eigen::MatrixXd&    vcov);

// `std.all` standardization: on top of `std.lv` (latent variables → unit
// total variance), observed indicators are also standardized
// to unit variance — every parameter that touches indicator i is divided
// by an additional √σ_ii (where σ_ii is the model-implied variance of
// indicator i, evaluated at θ̂).
//
//   Λ_{r,c} → λ · √Var(η_c) / √σ_rr    (latent + indicator scale)
//   Θ_{r,r} → θ_rr / σ_rr              (residual variance ratio)
//   ν_r     → ν_r / √σ_rr              (intercept in SD units)
//   α_j     → α_j / √Var(η_j)          (latent mean in latent-SD units)
//   β_ij    → β_ij · √Var(η_j)/√Var(η_i)
//   ψ_jk    → ψ_jk / √(Var(η_j)·Var(η_k))
//   Θ off-diagonals pass through as identity in v0.
//
// SEs via delta method on the Jacobian of the transformation, which
// includes terms from ∂σ_ii/∂θ (from `dsigma_dtheta`) for the indicator
// scale, plus the latent-scale terms shared with `std.lv`.
post_expected<StandardizedSolution>
standardize_all(const spec::LatentStructure& pt,
                const model::MatrixRep&   rep,
                const Estimates&          est,
                const Eigen::MatrixXd&    vcov);

// Native FC-SEM counterparts. They use the FcSemEvaluator's sample-backed W/T
// covariance semantics and return values/SEs for free rows in θ order, just as
// the ordinary LISREL standardizers do.
post_expected<StandardizedSolution>
standardize_lv_fcsem(const spec::LatentStructure& pt,
                     const SampleStats&          samp,
                     const Estimates&            est,
                     const Eigen::MatrixXd&      vcov);

post_expected<StandardizedSolution>
standardize_all_fcsem(const spec::LatentStructure& pt,
                      const SampleStats&          samp,
                      const Estimates&            est,
                      const Eigen::MatrixXd&      vcov);

post_expected<std::vector<FcSemStandardizedRow>>
standardized_rows_fcsem(const spec::LatentStructure& pt,
                        const spec::LatentNames&     names,
                        const SampleStats&           samp,
                        const Estimates&             est,
                        const Eigen::MatrixXd&       vcov);

}  // namespace magmaan::measures::standardize
