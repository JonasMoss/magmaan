#include "magmaan/data/cluster_stats.hpp"

namespace magmaan::data {

post_expected<ClusterSampleStats>
cluster_sample_stats(const Eigen::Ref<const Eigen::MatrixXd>& /*X*/,
                     const std::vector<std::int32_t>& /*cluster_id*/,
                     const std::vector<std::int32_t>& /*within_cols*/,
                     const std::vector<std::int32_t>& /*between_cols*/) {
  // Phase-0 contract stub. Stream B implements the within/between scatter
  // decomposition and the distinct-cluster-size grouping. See the
  // multilevel-SEM plan (Contract 2).
  return std::unexpected(PostError{
      PostError::Kind::NumericIssue,
      "data::cluster_sample_stats: two-level sample statistics not yet "
      "implemented"});
}

}  // namespace magmaan::data
