#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::data {

enum class HuberResidualClipKind {
  None,
  HardHuber,
  PseudoHuber,
  TukeyBiweight,
};

struct HuberResidualClipOptions {
  HuberResidualClipKind kind = HuberResidualClipKind::HardHuber;
  double k = 1.345;
};

struct HuberResidualClipEval {
  double psi = 0.0;
  double dpsi = 0.0;
  double loss = 0.0;
  double weight = 1.0;
};

post_expected<HuberResidualClipEval>
eval_huber_residual_clip(double r, HuberResidualClipOptions options = {});

struct PolyserialPairMlOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
};

struct PolyserialPairMlResult {
  double rho = 0.0;
  double negloglik = 0.0;
  int    iterations = 0;
  bool   hit_lower = false;
  bool   hit_upper = false;
};

struct PolyserialPairDpdOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
  double alpha = 0.3;
  double fd_step = 1e-5;
};

struct PolyserialPairHuberResidualOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
  double fd_step = 1e-5;
  HuberResidualClipOptions clip;
};

struct PolyserialPairDpdResult {
  double rho = 0.0;
  double objective = 0.0;
  int    iterations = 0;
  bool   converged = true;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::VectorXd probabilities;
  Eigen::VectorXd weights;
};

struct PolyserialPairJointDpdOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 120;
  double ftol = 1e-10;
  double gtol = 2e-5;
  double fd_step = 2e-5;
  double min_threshold_spacing = 1e-4;
  double alpha = 0.5;
};

struct PolyserialPairJointDpdResult {
  Eigen::VectorXd thresholds;
  double mean = 0.0;
  double sd = 1.0;
  double rho = 0.0;
  double objective = 0.0;
  int    iterations = 0;
  bool   converged = false;
  // Per-row conditional probabilities for the observed category.
  Eigen::VectorXd probabilities;
  // Per-row joint densities phi(u_i) * Pr(y_i | u_i).
  Eigen::VectorXd joint_densities;
  // Raw DPD attenuation weights: joint_density^alpha.
  Eigen::VectorXd weights;
};

struct PolyserialPairScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd thresholds;
  // Columns: thresholds, rho.
  Eigen::MatrixXd score_contributions;
  Eigen::MatrixXd score_gamma;
  // Raw-metric pair-likelihood scores for the continuous margin's mean and
  // variance, at unit sigma: sigma * dl/dmu and sigma^2 * dl/dsigma^2. The
  // caller divides by sigma / sigma^2 to land in the raw parameter metric.
  Eigen::VectorXd mu_unit;
  Eigen::VectorXd var_unit;
};

struct PolyserialPairDpdScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd thresholds;
  // Columns: thresholds, rho.
  Eigen::MatrixXd score_contributions;
  Eigen::MatrixXd score_gamma;
  // Positive bread: -d mean(estimating_functions) / d(thresholds, rho).
  Eigen::MatrixXd bread;
  Eigen::VectorXd weights;
};

enum class MixedPairKind {
  ordinal_ordinal,
  continuous_ordinal,
  continuous_continuous
};

enum class MixedMomentKind {
  threshold,
  continuous_mean,
  continuous_variance,
  pair
};

struct MixedPairLabel {
  std::int32_t i = 0;
  std::int32_t j = 0;
  std::int32_t moment_index = 0;
  MixedPairKind kind = MixedPairKind::continuous_continuous;
};

struct MixedMomentLabel {
  std::int32_t index = 0;
  MixedMomentKind kind = MixedMomentKind::threshold;
  std::int32_t variable = -1;
  std::int32_t variable_i = -1;
  std::int32_t variable_j = -1;
  std::int32_t threshold_level = -1;
  MixedPairKind pair_kind = MixedPairKind::continuous_continuous;
};

struct ContinuousPairNormalResult {
  double mean_i = 0.0;
  double mean_j = 0.0;
  double var_i = 0.0;
  double var_j = 0.0;
  double cov = 0.0;
  double rho = 0.0;
  double negloglik = 0.0;
  std::int64_t n_obs = 0;
  // Columns: mean_i, mean_j, var_i, var_j, cov.
  Eigen::MatrixXd score_contributions;
  Eigen::MatrixXd score_gamma;
};

struct ContinuousPairNormalScores {
  // Columns: mean_i, mean_j, var_i, var_j, cov.
  Eigen::MatrixXd score_contributions;
  Eigen::MatrixXd score_gamma;
};

post_expected<std::vector<MixedPairLabel>>
mixed_pair_labels(const std::vector<std::int32_t>& ordered,
                  std::int32_t n_thresholds);

post_expected<std::vector<MixedMomentLabel>>
mixed_moment_labels(const std::vector<std::int32_t>& ordered,
                    const std::vector<std::int32_t>& threshold_ov,
                    const std::vector<std::int32_t>& threshold_level);

post_expected<double>
continuous_pair_normal_negloglik(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                                 const Eigen::Ref<const Eigen::VectorXd>& x_j,
                                 double mean_i,
                                 double mean_j,
                                 double var_i,
                                 double var_j,
                                 double cov);

post_expected<ContinuousPairNormalResult>
fit_continuous_pair_normal_ml(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                              const Eigen::Ref<const Eigen::VectorXd>& x_j);

post_expected<ContinuousPairNormalScores>
continuous_pair_normal_scores(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                              const Eigen::Ref<const Eigen::VectorXd>& x_j,
                              double mean_i,
                              double mean_j,
                              double var_i,
                              double var_j,
                              double cov);

post_expected<double>
polyserial_pair_negloglik(const Eigen::Ref<const Eigen::VectorXi>& categories,
                          const Eigen::Ref<const Eigen::VectorXd>& u,
                          const Eigen::Ref<const Eigen::VectorXd>& thresholds,
                          double rho);

post_expected<PolyserialPairMlResult>
fit_polyserial_pair_rho_ml(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    PolyserialPairMlOptions options = {});

post_expected<PolyserialPairDpdResult>
fit_polyserial_pair_rho_dpd(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    PolyserialPairDpdOptions options = {});

post_expected<PolyserialPairJointDpdResult>
fit_polyserial_pair_joint_dpd(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    PolyserialPairJointDpdOptions options = {});

post_expected<PolyserialPairScores>
polyserial_pair_scores(const Eigen::Ref<const Eigen::VectorXi>& categories,
                       const Eigen::Ref<const Eigen::VectorXd>& u,
                       double rho,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds);

post_expected<PolyserialPairDpdScores>
polyserial_pair_dpd_scores(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    double rho,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    PolyserialPairDpdOptions options = {});

}  // namespace magmaan::data
