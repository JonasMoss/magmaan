#pragma once

#include <random>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::sim {

struct NormalOptions {
  double cholesky_jitter = 0.0;
};

sim_expected<Eigen::MatrixXd>
simulate_normal_matrix(Eigen::Index n,
                       const Eigen::Ref<const Eigen::VectorXd>& mean,
                       const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                       std::mt19937_64& rng,
                       const NormalOptions& options = {});

sim_expected<data::RawData>
simulate_normal_raw(Eigen::Index n,
                    const Eigen::Ref<const Eigen::VectorXd>& mean,
                    const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                    std::mt19937_64& rng,
                    const NormalOptions& options = {});

}  // namespace magmaan::sim
