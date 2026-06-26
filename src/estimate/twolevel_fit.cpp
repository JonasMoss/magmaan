#include "magmaan/estimate/twolevel.hpp"

namespace magmaan::estimate::twolevel {

fit_expected<Eigen::VectorXd>
twolevel_start_values(const spec::LatentStructure& /*pt*/,
                      const model::MatrixRep& /*rep*/,
                      const ClusterSampleStats& /*cs*/,
                      const spec::Starts& /*starts*/) {
  // Phase-0 contract stub. Stream C implements the two-level start heuristic.
  // See the multilevel-SEM plan (Contract 6).
  return std::unexpected(FitError{
      FitError::Kind::NumericIssue,
      "estimate::twolevel: twolevel_start_values not yet implemented"});
}

fit_expected<Estimates>
fit_ml_twolevel(spec::LatentStructure /*pt*/, const model::MatrixRep& /*rep*/,
                const ClusterSampleStats& /*cs*/, const Eigen::VectorXd& /*x0*/,
                Bounds /*bounds*/, Backend /*backend*/, OptimOptions /*opts*/) {
  // Phase-0 contract stub. Stream C implements the two-level ML fit composer
  // (mirrors estimate::fit_ml). See the multilevel-SEM plan (Contract 6).
  return std::unexpected(FitError{
      FitError::Kind::NumericIssue,
      "estimate::twolevel: fit_ml_twolevel not yet implemented"});
}

}  // namespace magmaan::estimate::twolevel
