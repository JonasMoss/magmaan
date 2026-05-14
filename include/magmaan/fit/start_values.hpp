#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/partable/partable.hpp"
#include "magmaan/partable/start_hints.hpp"

namespace magmaan::fit {

// Compute starting values for each free parameter, sized n_free.
//
// Strategy ("simple", matching lavaan's default):
//   - A user-supplied start hint (`starts.hint[k]`, from `start(v)*x` / `v?x`) wins.
//   - Free Λ entries (loadings):       0.7  (lavaan's "simple" default)
//   - Free Θ diagonal (residual var):  0.5 * S_diag from sample
//   - Free Θ off-diagonal (resid cov): 0.0
//   - Free Ψ diagonal (latent var):    0.05
//   - Free Ψ off-diagonal (latent cov):0.0
//
// `starts` is the optional hint vector produced by `lavaanify` (an empty /
// shorter-than-n_free `hint` simply means "no hints"). Multi-block extension
// is straightforward; v0 is single-block.
fit_expected<Eigen::VectorXd>
simple_start_values(const partable::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    const SampleStats& samp,
                    const partable::Starts& starts = {});

}  // namespace magmaan::fit
