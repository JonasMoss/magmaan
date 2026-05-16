#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/h_score.hpp"
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

struct OrdinalPairHWeightedOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
  double x_tol = 1e-10;
  bool   lavaan_adjust_2x2 = true;
  PolychoricHScoreOptions h_score;
};

struct OrdinalPairHWeightedResult {
  double rho = 0.0;
  double objective = 0.0;
  double score = 0.0;
  int    iterations = 0;
  bool   converged = false;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::MatrixXd adjusted_counts;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd expected_counts;
  Eigen::MatrixXd residual_counts;
  Eigen::MatrixXd pearson_residuals;
  Eigen::MatrixXd weights;
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

struct OrdinalPairJointHWeightedOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 250;
  double ftol = 1e-10;
  double gtol = 1e-7;
  double fd_step = 1e-5;
  double min_threshold_spacing = 1e-6;
  bool   lavaan_adjust_2x2 = true;
  PolychoricHScoreOptions h_score;
};

struct OrdinalPairJointHWeightedResult {
  Eigen::VectorXd thresholds_i;
  Eigen::VectorXd thresholds_j;
  double rho = 0.0;
  double objective = 0.0;
  double gradient_inf = 0.0;
  int    iterations = 0;
  bool   converged = false;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::MatrixXd adjusted_counts;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd expected_counts;
  Eigen::MatrixXd residual_counts;
  Eigen::MatrixXd pearson_residuals;
  Eigen::MatrixXd weights;
};

struct OrdinalPairJointDpdOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 250;
  double ftol = 1e-10;
  double gtol = 1e-7;
  double fd_step = 1e-5;
  double min_threshold_spacing = 1e-6;
  bool   lavaan_adjust_2x2 = true;
  double alpha = 0.3;
};

struct OrdinalPairJointDpdResult {
  Eigen::VectorXd thresholds_i;
  Eigen::VectorXd thresholds_j;
  double rho = 0.0;
  double objective = 0.0;
  double gradient_inf = 0.0;
  int    iterations = 0;
  bool   converged = false;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::MatrixXd adjusted_counts;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd expected_counts;
  Eigen::MatrixXd residual_counts;
  Eigen::MatrixXd pearson_residuals;
  Eigen::MatrixXd weights;
};

struct OrdinalPairHWeightedInfluenceOptions {
  // Retained for source compatibility; h-weighted bread is analytic.
  double fd_step = 1e-5;
  PolychoricHScoreOptions h_score;
};

struct OrdinalPairHWeightedInfluence {
  std::int64_t n_obs = 0;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd ratios;
  Eigen::MatrixXd h_values;
  Eigen::MatrixXd dh_values;
  Eigen::MatrixXd weights;
  Eigen::MatrixXd estimating_functions;
  Eigen::MatrixXd score_gamma;
  Eigen::MatrixXd bread;
  Eigen::MatrixXd influence;
  Eigen::MatrixXd gamma;
};

enum class PairwiseOrdinalCorrelationRepairKind {
  None,
  Error,
  Ridge,
  Shrinkage,
};

struct PairwiseOrdinalCorrelationRepairOptions {
  PairwiseOrdinalCorrelationRepairKind kind =
      PairwiseOrdinalCorrelationRepairKind::None;
  double min_eigenvalue = 1e-8;
};

struct PairwiseOrdinalHWeightedStatsOptions {
  OrdinalPairHWeightedOptions rho;
  // Retained for source compatibility; h-weighted influence bread is analytic.
  double influence_fd_step = 1e-5;
  PairwiseOrdinalCorrelationRepairOptions correlation_repair;
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
  double objective = 0.0;
  double score = 0.0;
  int    iterations = 0;
  bool   h_weighted = false;
  bool   converged = true;
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
  Eigen::MatrixXd pearson_residuals;
  Eigen::MatrixXd weights;
};

struct PairwiseOrdinalBlockDiagnostics {
  std::vector<OrdinalPairDiagnostics> pair_diagnostics;
  Eigen::MatrixXd moment_influence;
  Eigen::MatrixXd gamma;
  double min_eigen_r = 0.0;
  double raw_min_eigen_r = 0.0;
  bool   r_repair_applied = false;
  double r_ridge = 0.0;
  double r_shrinkage_intensity = 0.0;
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

post_expected<OrdinalPairHWeightedResult>
fit_ordinal_pair_rho_h_weighted(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    OrdinalPairHWeightedOptions options = {});

post_expected<OrdinalPairJointMlResult>
fit_ordinal_pair_joint_ml(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                          OrdinalPairJointMlOptions options = {});

post_expected<OrdinalPairJointHWeightedResult>
fit_ordinal_pair_joint_h_weighted(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    OrdinalPairJointHWeightedOptions options = {});

post_expected<OrdinalPairJointDpdResult>
fit_ordinal_pair_joint_dpd(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    OrdinalPairJointDpdOptions options = {});

post_expected<OrdinalPairHWeightedInfluence>
ordinal_pair_h_weighted_influence(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    double rho,
    OrdinalPairHWeightedInfluenceOptions options = {});

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

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_h_weighted_from_integer_data(
    const std::vector<Eigen::MatrixXd>& X,
    PairwiseOrdinalHWeightedStatsOptions options = {});

}  // namespace magmaan::data
