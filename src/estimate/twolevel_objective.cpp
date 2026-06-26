#include "magmaan/estimate/twolevel.hpp"

#include <string>

namespace magmaan::estimate::twolevel {

namespace {
// Phase-0 contract stubs — Stream C implements the Muthén–Asparouhov
// full-information cluster likelihood and its analytic gradient. See the
// multilevel-SEM plan (Contract 4 / 7).
template <class T>
fit_expected<T> not_impl(const char* what) {
  return std::unexpected(FitError{
      FitError::Kind::NumericIssue,
      std::string("estimate::twolevel: ") + what + " not yet implemented"});
}
}  // namespace

fit_expected<TwoLevelCache> twolevel_prepare(const ClusterSampleStats& /*cs*/) {
  return not_impl<TwoLevelCache>("twolevel_prepare");
}

fit_expected<double>
twolevel_value(const ClusterSampleStats& /*cs*/, const TwoLevelCache& /*cache*/,
               const model::ImpliedMoments& /*moments*/,
               const model::MatrixRep& /*rep*/) {
  return not_impl<double>("twolevel_value");
}

fit_expected<TwoLevelValueGradient>
twolevel_value_gradient(const ClusterSampleStats& /*cs*/,
                        const TwoLevelCache& /*cache*/,
                        const model::ImpliedMoments& /*moments*/,
                        const Eigen::MatrixXd& /*J_sigma*/,
                        const Eigen::MatrixXd& /*J_mu*/,
                        const model::MatrixRep& /*rep*/) {
  return not_impl<TwoLevelValueGradient>("twolevel_value_gradient");
}

fit_expected<optim::ScalarProblem>
twolevel_ml_objective(const model::ModelEvaluator& /*ev*/,
                      const ClusterSampleStats& /*cs*/) {
  return not_impl<optim::ScalarProblem>("twolevel_ml_objective");
}

fit_expected<Eigen::MatrixXd>
twolevel_information(const model::ModelEvaluator& /*ev*/,
                     const ClusterSampleStats& /*cs*/,
                     const Eigen::VectorXd& /*theta*/, bool /*expected*/) {
  return not_impl<Eigen::MatrixXd>("twolevel_information");
}

}  // namespace magmaan::estimate::twolevel
