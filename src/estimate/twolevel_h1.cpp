#include "magmaan/estimate/twolevel.hpp"

namespace magmaan::estimate::twolevel {

fit_expected<TwoLevelH1>
twolevel_h1_moments(const ClusterSampleStats& /*cs*/,
                    TwoLevelH1Options /*opts*/) {
  // Phase-0 contract stub. Stream C implements the iterative two-level
  // saturated solver (the analogue of fiml_h1_moments). See the multilevel-SEM
  // plan (Contract 5).
  return std::unexpected(FitError{
      FitError::Kind::NumericIssue,
      "estimate::twolevel: twolevel_h1_moments not yet implemented"});
}

}  // namespace magmaan::estimate::twolevel
