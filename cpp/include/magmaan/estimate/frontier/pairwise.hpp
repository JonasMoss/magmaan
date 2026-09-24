#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate::frontier {

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

struct PairwiseOrdinalObservedData {
  std::vector<Eigen::MatrixXd> X;
  data::OrdinalStats stats;
  PairwiseOrdinalCompositeResult saturated;
  PairwiseOrdinalCompositeOptions options;
};

struct PairwiseOrdinalCompositeFit {
  Estimates estimates;
  PairwiseOrdinalCompositeResult objective;
  int df = 0;
};

struct PairwiseOrdinalCompositeGodambe {
  Eigen::MatrixXd bread;
  Eigen::MatrixXd meat;
  Eigen::MatrixXd vcov;
  Eigen::MatrixXd vcov_naive;
  Eigen::VectorXd se;
  Eigen::VectorXd se_naive;
  Eigen::MatrixXd casewise_scores;
  double condition_bread = 0.0;
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

post_expected<PairwiseOrdinalObservedData>
pairwise_ordinal_observed_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& n_levels,
    PairwiseOrdinalCompositeOptions options = {});

fit_expected<PairwiseOrdinalCompositeFit>
fit_pairwise_ordinal_composite(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const PairwiseOrdinalObservedData& data,
    const Eigen::VectorXd& x0 = {},
    Bounds bounds = {},
    Backend backend = Backend::NloptLbfgs,
    optim::OptimOptions opts = {},
    spec::Starts starts = {});

post_expected<PairwiseOrdinalCompositeGodambe>
pairwise_ordinal_composite_godambe(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const PairwiseOrdinalObservedData& data,
    const Estimates& est,
    double fd_step = 1e-5);

post_expected<robust::LRSatorra2000Result>
lr_test_pairwise_ordinal_composite(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const PairwiseOrdinalObservedData& data,
    const PairwiseOrdinalCompositeFit& fit_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const PairwiseOrdinalCompositeFit& fit_H0,
    robust::SatorraAMethod a_method = robust::SatorraAMethod::Exact,
    double fd_step = 1e-5);

}  // namespace magmaan::estimate::frontier
