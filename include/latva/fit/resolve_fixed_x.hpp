#pragma once

#include "latva/expected.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/partable/partable.hpp"

namespace latva::fit {

// `lavaanify` marks fixed.x rows with `exo=1, free=0, fixed_value=NaN` because
// the value is supposed to come from the data. This helper fills those
// NaN `fixed_value`s with the corresponding sample covariance entries from
// `samp` (using `rep` to map row→cell).
//
// Mutates `pt` in place. Idempotent — safe to call more than once.
fit_expected<void>
resolve_fixed_x_from_sample(partable::ParTable&     pt,
                            const model::MatrixRep& rep,
                            const SampleStats&      samp);

}  // namespace latva::fit
