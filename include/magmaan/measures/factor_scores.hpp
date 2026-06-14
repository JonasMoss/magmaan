#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::measures {

using estimate::Estimates;

// Factor-score method — lavaan's `lavPredict(fit, method = ...)`.
//
//   Regression — Thurstone / regression scores, the conditional mean
//                E[η | y] under the fitted multivariate-normal model:
//                  f̂ = E[η] + (AΨAᵀ)·Λᵀ·Σ̂⁻¹·(y − μ̂)
//   Bartlett   — the GLS / maximum-likelihood scores, conditionally
//                unbiased (E[f̂ | η] = η):
//                  f̂ = (Λᵀ Θ⁻¹ Λ)⁻¹ Λᵀ Θ⁻¹ (y − ν)
//
//   Ebm / Ml / Eap — categorical latent-response scores for ordinal and
//                mixed-ordinal fits. EBM is the posterior mode, ML is the
//                likelihood mode without the latent prior, and EAP is the
//                posterior mean (currently one latent dimension).
enum class FactorScoreMethod : std::uint8_t {
  Regression,
  Bartlett,
  Ebm,
  Ml,
  Eap,
};

// Per-observation latent scores. `scores[b]` is (n_b × m): one row per
// observation, one column per *extended* latent (in the reduced-LISREL
// extended ordering — for a CFA this is exactly the user factors; for a
// path / SEM model it also includes the phantom latents promoted from
// observed variables, which a caller can slice away by name).
struct FactorScores {
  std::vector<Eigen::MatrixXd> scores;
};

// Computes per-observation factor scores for `raw` at θ̂. The model is
// centered on the model-implied mean (mean-structure models) or the sample
// column means (covariance-only models). Complete data only — a non-empty
// `raw.mask` is rejected.
//
// `pt` is taken by value because `factor_scores` resolves fixed.x
// `fixed_value`s from the data internally. Returns `PostError::NumericIssue`
// on a non-PD implied Σ̂ / Θ, a singular ΛᵀΘ⁻¹Λ (Bartlett), or a size
// mismatch.
post_expected<FactorScores>
factor_scores(spec::LatentStructure pt, const model::MatrixRep& rep,
              const data::RawData& raw, const Estimates& est,
              FactorScoreMethod method);

post_expected<FactorScores>
factor_scores_ordinal(spec::LatentStructure pt, const model::MatrixRep& rep,
                      const data::RawData& raw,
                      const data::OrdinalStats& stats,
                      const Estimates& est,
                      FactorScoreMethod method,
                      estimate::OrdinalParameterization parameterization);

post_expected<FactorScores>
factor_scores_mixed_ordinal(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const data::RawData& raw,
                            const data::MixedOrdinalStats& stats,
                            const Estimates& est,
                            FactorScoreMethod method,
                            estimate::OrdinalParameterization parameterization);

}  // namespace magmaan::measures
