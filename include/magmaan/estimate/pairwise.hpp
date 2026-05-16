#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::estimate {

enum class PairwiseCompositeWeighting {
  ObservedPairCount,
  EqualPair,
};

enum class PairwiseCompositeScaling {
  SumNegLogLik,
  AverageNegLogLik,
};

struct PairwiseOrdinalCompositeOptions {
  PairwiseCompositeWeighting weighting = PairwiseCompositeWeighting::ObservedPairCount;
  PairwiseCompositeScaling scaling = PairwiseCompositeScaling::AverageNegLogLik;
  bool lavaan_adjust_2x2 = true;
  double rho_lower = -0.999;
  double rho_upper = 0.999;
};

struct PairwiseOrdinalCompositePair {
  data::OrdinalPairLabel label;
  Eigen::VectorXd thresholds_i;
  Eigen::VectorXd thresholds_j;
  double rho = 0.0;
  double negloglik = 0.0;
  double weighted_negloglik = 0.0;
  double scaling_weight = 0.0;
  std::int64_t n_obs = 0;
  std::int64_t n_missing = 0;
  bool hit_lower = false;
  bool hit_upper = false;
  Eigen::MatrixXd counts;
  Eigen::MatrixXd adjusted_counts;
  Eigen::MatrixXd expected_counts;
  Eigen::MatrixXd residual_counts;
  Eigen::MatrixXd score_contributions;
  Eigen::MatrixXd score_gamma;
};

struct PairwiseOrdinalCompositeBlock {
  std::vector<PairwiseOrdinalCompositePair> pairs;
  double negloglik = 0.0;
  double weighted_negloglik = 0.0;
  double objective = 0.0;
  double scaling_denominator = 0.0;
  std::int64_t n_obs = 0;
  bool reports_chisq = false;
  double chisq = 0.0;
  int df = -1;
};

struct PairwiseOrdinalCompositeResult {
  std::vector<PairwiseOrdinalCompositeBlock> blocks;
  double negloglik = 0.0;
  double weighted_negloglik = 0.0;
  double objective = 0.0;
  double scaling_denominator = 0.0;
  bool reports_chisq = false;
  double chisq = 0.0;
  int df = -1;
};

post_expected<PairwiseOrdinalCompositeResult>
pairwise_ordinal_composite_objective(
    const data::PairwiseOrdinalStats& sample,
    const std::vector<Eigen::VectorXd>& implied_thresholds,
    const std::vector<Eigen::MatrixXd>& implied_correlation,
    PairwiseOrdinalCompositeOptions options = {});

post_expected<PairwiseOrdinalCompositeResult>
pairwise_ordinal_joint_composite_objective(
    const data::PairwiseOrdinalStats& sample,
    PairwiseOrdinalCompositeOptions options = {});

post_expected<PairwiseOrdinalCompositeResult>
pairwise_ordinal_observed_joint_composite_objective(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& n_levels,
    PairwiseOrdinalCompositeOptions options = {});

}  // namespace magmaan::estimate
