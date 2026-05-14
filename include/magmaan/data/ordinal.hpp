#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::data {

struct OrdinalStats {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<Eigen::MatrixXd> W_dwls;
  std::vector<Eigen::MatrixXd> W_wls;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& X);

}  // namespace magmaan::data
