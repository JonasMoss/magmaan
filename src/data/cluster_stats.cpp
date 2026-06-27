#include "magmaan/data/cluster_stats.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace magmaan::data {

namespace {

// Reject a column selector that points outside [0, ncol).
bool cols_in_range(const std::vector<std::int32_t>& cols, Eigen::Index ncol) {
  for (std::int32_t c : cols) {
    if (c < 0 || static_cast<Eigen::Index>(c) >= ncol) return false;
  }
  return true;
}

}  // namespace

post_expected<ClusterSampleStats>
cluster_sample_stats(const Eigen::Ref<const Eigen::MatrixXd>& X,
                     const std::vector<std::int32_t>& cluster_id,
                     const std::vector<std::int32_t>& within_cols,
                     const std::vector<std::int32_t>& between_cols) {
  const Eigen::Index n_rows = X.rows();
  const Eigen::Index n_cols = X.cols();

  // ---- input validation ---------------------------------------------------
  if (n_rows == 0) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats: empty data matrix (0 rows)"});
  }
  if (static_cast<std::int64_t>(cluster_id.size()) !=
      static_cast<std::int64_t>(n_rows)) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats: cluster_id length does not match the "
        "number of data rows"});
  }
  // v1: within and between observed sets are identical (shared observed set).
  if (within_cols != between_cols) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats: v1 requires within_cols == between_cols "
        "(shared observed variable set)"});
  }
  if (within_cols.empty()) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats: empty column selector"});
  }
  if (!cols_in_range(within_cols, n_cols) ||
      !cols_in_range(between_cols, n_cols)) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats: column selector out of range"});
  }

  const Eigen::Index p = static_cast<Eigen::Index>(within_cols.size());

  // ---- group rows by cluster id, preserving first-appearance order --------
  // cluster_id values are arbitrary 0-based labels; map each distinct label to
  // a dense, stable cluster index so the output is deterministic.
  std::map<std::int32_t, std::size_t> label_to_idx;
  std::vector<std::vector<Eigen::Index>> cluster_rows;  // per cluster: row list
  for (Eigen::Index r = 0; r < n_rows; ++r) {
    const std::int32_t lab = cluster_id[static_cast<std::size_t>(r)];
    auto [it, inserted] = label_to_idx.try_emplace(lab, cluster_rows.size());
    if (inserted) cluster_rows.emplace_back();
    cluster_rows[it->second].push_back(r);
  }

  const std::int64_t n_clusters =
      static_cast<std::int64_t>(cluster_rows.size());

  // ---- accumulate the sufficient statistics -------------------------------
  // Selected-column view extractor (row r → p-vector over the chosen columns).
  auto row_y = [&](Eigen::Index r) {
    Eigen::VectorXd y(p);
    for (Eigen::Index k = 0; k < p; ++k) {
      y(k) = X(r, static_cast<Eigen::Index>(within_cols[static_cast<std::size_t>(k)]));
    }
    return y;
  };

  Eigen::VectorXd grand_sum = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd ssw = Eigen::MatrixXd::Zero(p, p);  // raw within scatter

  // Per distinct size d: accumulate Σ ȳ and Σ ȳ ȳᵀ and count.
  std::map<std::int64_t, ClusterSizePattern> by_size;

  for (const auto& rows : cluster_rows) {
    const std::int64_t d = static_cast<std::int64_t>(rows.size());
    if (d <= 0) {
      // Cannot occur given the grouping construction, but a guard keeps the
      // raw-SSW divisor contract well-defined.
      return std::unexpected(PostError{
          PostError::Kind::NumericIssue,
          "data::cluster_sample_stats: empty cluster encountered"});
    }

    // Cluster mean ȳ_·j over its d rows (selected columns).
    Eigen::VectorXd ybar = Eigen::VectorXd::Zero(p);
    for (Eigen::Index r : rows) ybar += row_y(r);
    grand_sum += ybar;  // grand_sum accumulates Σ_i y_ij over all units
    ybar /= static_cast<double>(d);

    // Within scatter contribution: Σ_i (y_ij − ȳ_·j)(y_ij − ȳ_·j)ᵀ.
    for (Eigen::Index r : rows) {
      const Eigen::VectorXd dev = row_y(r) - ybar;
      ssw.noalias() += dev * dev.transpose();
    }

    // Size-d between accumulators.
    auto [it, inserted] = by_size.try_emplace(d);
    ClusterSizePattern& pat = it->second;
    if (inserted) {
      pat.cluster_size = d;
      pat.n_clusters = 0;
      pat.sum_cluster_mean = Eigen::VectorXd::Zero(p);
      pat.sum_cluster_mean_cp = Eigen::MatrixXd::Zero(p, p);
    }
    pat.n_clusters += 1;
    pat.sum_cluster_mean += ybar;
    pat.sum_cluster_mean_cp.noalias() += ybar * ybar.transpose();
  }

  // ---- assemble the group result ------------------------------------------
  ClusterGroupStats g;
  g.n_within = static_cast<std::int64_t>(n_rows);
  g.n_clusters = n_clusters;
  g.p_within = p;
  g.p_between = p;  // v1: shared observed set
  g.grand_mean = grand_sum / static_cast<double>(n_rows);
  g.within_scatter = std::move(ssw);

  // size_patterns ordered by ascending cluster size (map iteration order).
  g.size_patterns.reserve(by_size.size());
  for (auto& [d, pat] : by_size) {
    g.size_patterns.push_back(std::move(pat));
  }

  ClusterSampleStats out;
  out.groups.push_back(std::move(g));
  out.within_ov_index = within_cols;
  out.between_ov_index = between_cols;

  return out;
}

post_expected<ClusterSampleStats>
cluster_sample_stats_multigroup(
    const std::vector<Eigen::MatrixXd>& X_by_group,
    const std::vector<std::vector<std::int32_t>>& cluster_id_by_group,
    const std::vector<std::int32_t>& within_cols,
    const std::vector<std::int32_t>& between_cols) {
  if (X_by_group.empty()) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats_multigroup: no groups"});
  }
  if (X_by_group.size() != cluster_id_by_group.size()) {
    return std::unexpected(PostError{
        PostError::Kind::NumericIssue,
        "data::cluster_sample_stats_multigroup: expected one cluster_id vector "
        "per data group"});
  }

  ClusterSampleStats out;
  out.groups.reserve(X_by_group.size());
  // Each group is its own within/between law: reduce it with the single-group
  // builder (which validates the column selectors and shared-set contract per
  // group) and append its lone ClusterGroupStats, preserving group order.
  for (std::size_t g = 0; g < X_by_group.size(); ++g) {
    auto cs_g = cluster_sample_stats(X_by_group[g], cluster_id_by_group[g],
                                     within_cols, between_cols);
    if (!cs_g) return std::unexpected(cs_g.error());
    out.groups.push_back(std::move(cs_g->groups.front()));
  }
  out.within_ov_index = within_cols;
  out.between_ov_index = between_cols;

  return out;
}

}  // namespace magmaan::data
