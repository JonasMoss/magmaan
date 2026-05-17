#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/pairwise_mixed.hpp"

namespace magmaan::data {

// All ordinal covariance-like matrices use the same moment order:
// thresholds first, then lower-triangle polychorics by columns
// (rho_10, rho_20, ..., rho_p0, rho_21, ...).
struct OrdinalStats {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<Eigen::MatrixXd> NACOV;
  std::vector<Eigen::MatrixXd> W_dwls;
  std::vector<Eigen::MatrixXd> W_wls;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

// Mixed continuous/ordinal categorical statistics. Moment order follows
// lavaan's WLS observed vector for categorical data:
// thresholds, negative continuous means, continuous variances, then all
// lower-triangle covariance/correlation entries by columns.
struct MixedOrdinalStats {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> mean;
  std::vector<std::vector<std::int32_t>> ordered;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<Eigen::VectorXd> moments;
  std::vector<Eigen::MatrixXd> NACOV;
  std::vector<Eigen::MatrixXd> W_dwls;
  std::vector<Eigen::MatrixXd> W_wls;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& X);

post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_data(const std::vector<Eigen::MatrixXd>& X,
                              const std::vector<std::vector<std::int32_t>>& ordered);

}  // namespace magmaan::data
