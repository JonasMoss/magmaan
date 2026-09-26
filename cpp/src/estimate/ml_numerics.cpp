#include "magmaan/estimate/ml_numerics.hpp"
#include <cmath>

namespace magmaan::estimate {


fit_expected<MlStarts> ml_start_values(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const data::SampleStats& samp, const spec::Starts& hints) {
  return start_values(pt, rep, samp, StartPolicy{}, hints);
}

} // namespace magmaan::estimate
