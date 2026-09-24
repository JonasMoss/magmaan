#pragma once

// Two-level (multilevel) sufficient statistics. Computed from clustered raw
// data, grouped by distinct cluster size — the structural analogue of the FIML
// missingness-pattern cache (estimate::fiml::FIMLCache). Consumed by the
// two-level ML estimation core (estimate::twolevel).
//
// v1 scope: random-intercept two-level models over a *shared* observed variable
// set, i.e. every observed variable decomposes into a within and a between
// part, so the within and between sufficient statistics are both p×p over the
// same p observed variables. Between-level-only variables are deferred.
//
// This is the Stream B contract of the multilevel-SEM plan (Contract 2).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::data {

// Between-level sufficient statistics for one distinct cluster size `d` within a
// group. The cluster mean ȳ_·j is the average of the d level-1 rows in cluster
// j; the between law of these means (per size d) is what the two-level
// likelihood scores against Σ_W + d·Σ_B.
struct ClusterSizePattern {
  std::int64_t    cluster_size = 0;     // d = number of level-1 units per cluster
  std::int64_t    n_clusters   = 0;     // J_d = clusters with exactly d units
  Eigen::VectorXd sum_cluster_mean;     // Σ_j ȳ_·j over size-d clusters       (p × 1)
  Eigen::MatrixXd sum_cluster_mean_cp;  // Σ_j ȳ_·j ȳ_·jᵀ over size-d clusters (p × p)
};

// Per-group two-level sufficient statistics.
struct ClusterGroupStats {
  std::int64_t    n_within   = 0;   // total level-1 units N_g
  std::int64_t    n_clusters = 0;   // total level-2 units J_g
  Eigen::Index    p_within   = 0;   // number of within-level observed variables
  Eigen::Index    p_between  = 0;   // number of between-level observed variables
                                    // (v1: equal to p_within — shared observed set)
  Eigen::VectorXd grand_mean;       // observed grand mean                     (p × 1)
  // Pooled within scatter SSW = Σ_j Σ_i (y_ij − ȳ_·j)(y_ij − ȳ_·j)ᵀ. The
  // raw-SSW convention (no N−J divisor) is FROZEN here; consumers assume raw.
  Eigen::MatrixXd within_scatter;   //                                         (p × p)
  std::vector<ClusterSizePattern> size_patterns;  // one per distinct cluster size
};

// Two-level sufficient statistics across all groups.
struct ClusterSampleStats {
  std::vector<ClusterGroupStats> groups;       // one per group
  // Column orderings aligning the raw-data columns to the MatrixRep within /
  // between block `ov_names` (filled by the caller from the model). v1: equal.
  std::vector<std::int32_t> within_ov_index;
  std::vector<std::int32_t> between_ov_index;
};

// Build two-level sufficient statistics from clustered raw data: `X` is (N × p),
// `cluster_id[r]` the 0-based cluster of row r, and `within_cols` / `between_cols`
// select the within / between observed columns (v1: identical — every observed
// variable decomposes into a within and a between part).
post_expected<ClusterSampleStats>
cluster_sample_stats(const Eigen::Ref<const Eigen::MatrixXd>& X,
                     const std::vector<std::int32_t>& cluster_id,
                     const std::vector<std::int32_t>& within_cols,
                     const std::vector<std::int32_t>& between_cols);

// Multi-group two-level sufficient statistics. magmaan represents groups as a
// vector of per-group raw matrices (`data::RawData::X[g]`), so this takes one
// (n_g × p) matrix and one matching `cluster_id` vector per group, reduces each
// group independently via the single-group `cluster_sample_stats` above, and
// concatenates the per-group results into one ClusterSampleStats whose `groups`
// has one entry per input group (group order preserved). `within_cols` /
// `between_cols` are shared across groups (v1: identical — the shared observed
// variable set). Cluster labels need only be unique *within* each group; the
// per-group reduction never compares labels across groups. The single-group
// builder is the `X_by_group.size() == 1` special case.
post_expected<ClusterSampleStats>
cluster_sample_stats_multigroup(
    const std::vector<Eigen::MatrixXd>& X_by_group,
    const std::vector<std::vector<std::int32_t>>& cluster_id_by_group,
    const std::vector<std::int32_t>& within_cols,
    const std::vector<std::int32_t>& between_cols);

}  // namespace magmaan::data
