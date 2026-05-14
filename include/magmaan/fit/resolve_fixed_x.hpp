#pragma once

#include "magmaan/expected.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/partable/partable.hpp"

namespace magmaan::fit {

// `lavaanify` marks fixed.x rows with `exo=1, free=0, fixed_value=NaN` because
// the value is supposed to come from the data. This helper fills those
// NaN `fixed_value`s with the corresponding sample covariance entries from
// `samp` (using `rep` to map row→cell).
//
// Mutates `pt` in place. Idempotent — safe to call more than once.
fit_expected<void>
resolve_fixed_x_from_sample(partable::LatentStructure&     pt,
                            const model::MatrixRep& rep,
                            const SampleStats&      samp);

}  // namespace magmaan::fit
