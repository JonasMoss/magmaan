#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"  // Estimates
#include "magmaan/spec/partable.hpp"

namespace magmaan::measures::composite {

using estimate::Estimates;

// Recovered weights of one composite (`C <~ x1 + ... + xK`), for one group.
//
// Under the Henseler-Ogasawara reformulation the fitted parameters are the
// K×K loading block `Λ` of the emergent + excrescent latents, not the weights
// themselves. The indicators are an exact linear image of the latents
// (`x = Λ c`), so the latents — and the emergent composite in particular —
// are recovered by inverting: `c = Λ⁻¹ x`. The composite weight vector is the
// emergent row of that inverse, `weight = (Λ⁻¹).row(0)`; `weight[p]` is the
// weight on indicator p. `se` is the delta-method standard error of each
// weight, from `∂Λ⁻¹ = −Λ⁻¹ (∂Λ) Λ⁻¹` and the parameter covariance `vcov`.
struct CompositeWeights {
  std::string              composite;     // emergent latent = user's `<~` name
  std::int32_t             group = 1;     // 1-based group index
  std::vector<std::string> indicators;    // x1..xK, in order
  Eigen::VectorXd          weight;        // length K
  Eigen::VectorXd          se;            // length K, delta-method SE
};

// Recover composite weights and their delta-method standard errors from a
// fitted Henseler-Ogasawara model. Returns one entry per (composite, group),
// in `names.composites` order then ascending group; an empty vector if the
// model has no composites.
//
// `vcov` is the `n_free × n_free` parameter covariance (`inference::vcov`).
// Fails with `PostError::NumericIssue` if a composite's loading block is
// singular (weights not identified) or incomplete.
post_expected<std::vector<CompositeWeights>>
composite_weights(const spec::LatentStructure& pt,
                  const spec::LatentNames&     names,
                  const Estimates&             est,
                  const Eigen::MatrixXd&       vcov);

}  // namespace magmaan::measures::composite
