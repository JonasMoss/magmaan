#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::sim {

enum class ObservedKind : std::uint8_t {
  Continuous,
  Ordinal,
};

struct MixedProjectionSpec {
  std::vector<ObservedKind> kinds;
  std::vector<Eigen::VectorXd> thresholds;
};

struct MixedProjectionResult {
  Eigen::MatrixXd X;
  std::vector<std::int32_t> ordered;
  std::vector<std::int32_t> n_levels;
  std::vector<Eigen::VectorXi> category_counts;
};

sim_expected<Eigen::VectorXd>
thresholds_from_probabilities(
    const Eigen::Ref<const Eigen::VectorXd>& probabilities);

sim_expected<MixedProjectionResult>
project_mixed_matrix(const Eigen::Ref<const Eigen::MatrixXd>& latent,
                     const MixedProjectionSpec& spec);

sim_expected<MixedProjectionResult>
project_ordinal_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& latent,
    const std::vector<Eigen::VectorXd>& thresholds);

}  // namespace magmaan::sim
