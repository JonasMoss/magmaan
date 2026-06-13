#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
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
  // Achieved category proportions per ordinal variable (counts / n_rows);
  // empty for continuous columns. Mirrors `category_counts`.
  std::vector<Eigen::VectorXd> category_proportions;
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

// Wrap a single projected block as a `data::RawData` carrier with descriptive
// metadata. `variable_names` (length p) and `ordinal_level_labels` (per
// variable; empty entry for a continuous column) are optional: when
// `ordinal_level_labels` is empty, default labels "1".."k" are derived for each
// ordinal column from `result.n_levels`. Continuous columns get empty labels.
sim_expected<data::RawData>
raw_data_from_mixed_projection(
    const MixedProjectionResult& result,
    const std::vector<std::string>& variable_names = {},
    const std::vector<std::vector<std::string>>& ordinal_level_labels = {});

}  // namespace magmaan::sim
