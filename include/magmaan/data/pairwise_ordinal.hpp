#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::data {

struct OrdinalPairMlOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
  bool   lavaan_adjust_2x2 = true;
};

struct OrdinalPairMlResult {
  double rho = 0.0;
  double negloglik = 0.0;
  int    iterations = 0;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::MatrixXd adjusted_counts;
};

struct OrdinalPairJointMlOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 250;
  double ftol = 1e-10;
  double gtol = 1e-7;
  double fd_step = 1e-5;
  double min_threshold_spacing = 1e-6;
  bool   lavaan_adjust_2x2 = true;
};

struct OrdinalPairJointMlResult {
  Eigen::VectorXd thresholds_i;
  Eigen::VectorXd thresholds_j;
  double rho = 0.0;
  double negloglik = 0.0;
  int    iterations = 0;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::MatrixXd adjusted_counts;
};

struct OrdinalPairObservedTable {
  Eigen::MatrixXd counts;
  std::int64_t n_obs = 0;
  std::int64_t n_missing = 0;
};

struct OrdinalPairObservedMlResult {
  OrdinalPairMlResult fit;
  Eigen::MatrixXd counts;
  std::int64_t n_obs = 0;
  std::int64_t n_missing = 0;
};

struct OrdinalPairObservedJointMlResult {
  OrdinalPairJointMlResult fit;
  Eigen::MatrixXd counts;
  std::int64_t n_obs = 0;
  std::int64_t n_missing = 0;
};

struct OrdinalPairScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd threshold_i;
  Eigen::MatrixXd threshold_j;
};

struct OrdinalPairLabel {
  std::int32_t block = 0;
  std::int32_t i = 0;
  std::int32_t j = 0;
  std::int32_t n_levels_i = 0;
  std::int32_t n_levels_j = 0;
};

struct OrdinalPairDiagnostics {
  OrdinalPairLabel label;
  double rho = 0.0;
  double negloglik = 0.0;
  int    iterations = 0;
  bool   hit_lower = false;
  bool   hit_upper = false;
  std::int64_t n_obs = 0;
  std::int64_t n_missing = 0;
  bool   ridge_applied = false;
  double ridge = 0.0;
  bool   shrinkage_applied = false;
  double shrinkage_intensity = 0.0;
  Eigen::MatrixXd counts;
  Eigen::MatrixXd adjusted_counts;
  Eigen::MatrixXd expected_counts;
  Eigen::MatrixXd residual_counts;
};

struct PairwiseOrdinalBlockDiagnostics {
  std::vector<OrdinalPairDiagnostics> pair_diagnostics;
  Eigen::MatrixXd moment_influence;
  Eigen::MatrixXd gamma;
  double min_eigen_r = 0.0;
};

struct PairwiseOrdinalStats {
  OrdinalStats stats;
  std::vector<PairwiseOrdinalBlockDiagnostics> block_diagnostics;
};

post_expected<Eigen::MatrixXd>
ordinal_pair_table(const Eigen::Ref<const Eigen::VectorXi>& x_i,
                   const Eigen::Ref<const Eigen::VectorXi>& x_j,
                   std::int32_t n_levels_i,
                   std::int32_t n_levels_j);

post_expected<OrdinalPairObservedTable>
ordinal_pair_observed_table(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                            const Eigen::Ref<const Eigen::VectorXd>& x_j,
                            std::int32_t n_levels_i,
                            std::int32_t n_levels_j);

double ordinal_bvn_rect_prob(double lo_i, double hi_i,
                             double lo_j, double hi_j,
                             double rho) noexcept;

double ordinal_bvn_rect_drho(double lo_i, double hi_i,
                             double lo_j, double hi_j,
                             double rho) noexcept;

post_expected<double>
ordinal_pair_negloglik(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                       double rho);

post_expected<OrdinalPairMlResult>
fit_ordinal_pair_rho_ml(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                        const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                        const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                        OrdinalPairMlOptions options = {});

post_expected<OrdinalPairJointMlResult>
fit_ordinal_pair_joint_ml(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                          OrdinalPairJointMlOptions options = {});

post_expected<OrdinalPairObservedMlResult>
fit_ordinal_pair_observed_rho_ml(
    const Eigen::Ref<const Eigen::VectorXd>& x_i,
    const Eigen::Ref<const Eigen::VectorXd>& x_j,
    std::int32_t n_levels_i,
    std::int32_t n_levels_j,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    OrdinalPairMlOptions options = {});

post_expected<OrdinalPairObservedJointMlResult>
fit_ordinal_pair_observed_joint_ml(
    const Eigen::Ref<const Eigen::VectorXd>& x_i,
    const Eigen::Ref<const Eigen::VectorXd>& x_j,
    std::int32_t n_levels_i,
    std::int32_t n_levels_j,
    OrdinalPairJointMlOptions options = {});

post_expected<OrdinalPairScores>
ordinal_pair_scores(const Eigen::Ref<const Eigen::VectorXi>& x_i,
                    const Eigen::Ref<const Eigen::VectorXi>& x_j,
                    double rho,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j);

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& X);

}  // namespace magmaan::data
