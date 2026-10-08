#pragma once

#include "magmaan/estimate/layered_start.hpp"

#ifdef MAGMAAN_ENABLE_TEST_PROBES
namespace magmaan::estimate::layered_start_test {
// The unchanged alternating construction, for bitwise non-overlap regression gates.
fit_expected<LayeredStartReport>
alternating_candidate(const spec::LatentStructure&, const model::MatrixRep&,
                      const data::SampleStats&, const spec::Starts& = {});
}  // namespace magmaan::estimate::layered_start_test
#endif
