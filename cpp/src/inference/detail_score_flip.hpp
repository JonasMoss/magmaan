#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace magmaan::inference::frontier::detail {

struct ScoreFlipGroupGeometry {
  Eigen::MatrixXd GG;
  Eigen::MatrixXd GK;
  Eigen::MatrixXd KK;
};

inline Eigen::MatrixXd score_flip_variance(
    const std::vector<ScoreFlipGroupGeometry>& geometry,
    const Eigen::MatrixXd& nuisance_information_inverse,
    const std::vector<std::int64_t>& n_obs,
    const std::vector<std::int64_t>& sign_sum) {
  const Eigen::Index df = geometry.front().GG.rows();
  const Eigen::Index nuisance = geometry.front().KK.rows();
  Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(df, nuisance);
  for (std::size_t b = 0; b < geometry.size(); ++b) {
    cross.noalias() += static_cast<double>(sign_sum[b]) * geometry[b].GK;
  }
  const Eigen::MatrixXd R = cross * nuisance_information_inverse;
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(df, df);
  for (std::size_t b = 0; b < geometry.size(); ++b) {
    const double nb = static_cast<double>(n_obs[b]);
    const double cb = static_cast<double>(sign_sum[b]);
    V.noalias() += nb * geometry[b].GG;
    V.noalias() -= cb * geometry[b].GK * R.transpose();
    V.noalias() -= cb * R * geometry[b].GK.transpose();
    V.noalias() += nb * R * geometry[b].KK * R.transpose();
  }
  return 0.5 * (V + V.transpose());
}

}  // namespace magmaan::inference::frontier::detail
