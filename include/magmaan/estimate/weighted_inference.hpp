#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/robust.hpp"

namespace magmaan::estimate {

struct WeightedMomentBlock {
  Eigen::MatrixXd jacobian;  // model moments wrt full free theta
  Eigen::MatrixXd weight;    // estimator weight for this moment block
  Eigen::MatrixXd gamma;     // NACOV/meat for this moment block
  std::int64_t n_obs = 0;
};

struct WeightedRobustResult {
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  Eigen::VectorXd eigvals;
  double chisq_standard = 0.0;
  int df = 0;
  fit::SatorraBentlerResult satorra_bentler;
  fit::MeanVarAdjustedResult mean_var_adjusted;
  fit::ScaledShiftedResult scaled_shifted;
};

post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin);

}  // namespace magmaan::estimate
