#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/pairwise_mixed.hpp"

namespace magmaan::data {

// All ordinal covariance-like matrices use the same moment order:
// thresholds first, then lower-triangle polychorics by columns
// (rho_10, rho_20, ..., rho_p0, rho_21, ...).
struct OrdinalStats {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<Eigen::MatrixXd> NACOV;
  std::vector<Eigen::MatrixXd> W_dwls;
  std::vector<Eigen::MatrixXd> W_wls;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

// Mixed continuous/ordinal categorical statistics. Moment order follows
// lavaan's WLS observed vector for categorical data:
// thresholds, negative continuous means, continuous variances, then all
// lower-triangle covariance/correlation entries by columns.
struct MixedOrdinalStats {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> mean;
  std::vector<std::vector<std::int32_t>> ordered;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<Eigen::VectorXd> moments;
  std::vector<Eigen::MatrixXd> NACOV;
  std::vector<Eigen::MatrixXd> W_dwls;
  std::vector<Eigen::MatrixXd> W_wls;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

struct MixedOrdinalPolyserialDpdBlockDiagnostics {
  std::vector<MixedPairLabel> dpd_pairs;
  std::vector<PolyserialPairDpdResult> dpd_fits;
  Eigen::MatrixXd moment_influence;
  Eigen::MatrixXd gamma;
};

struct MixedOrdinalPolyserialDpdStats {
  MixedOrdinalStats stats;
  std::vector<MixedOrdinalPolyserialDpdBlockDiagnostics> block_diagnostics;
};

enum class MixedOrdinalCorrelationRepairKind {
  None,
  Error,
  Ridge,
  Shrinkage,
};

struct MixedOrdinalCorrelationRepairOptions {
  MixedOrdinalCorrelationRepairKind kind =
      MixedOrdinalCorrelationRepairKind::None;
  double min_eigenvalue = 1e-8;
};

struct MixedOrdinalHuberResidualOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 90;
  double ftol = 1e-10;
  double gtol = 2e-5;
  double fd_step = 1e-5;
  double min_threshold_spacing = 1e-6;
  bool   lavaan_adjust_2x2 = true;
  HuberResidualClipOptions clip;
  MixedOrdinalCorrelationRepairOptions correlation_repair;
};

struct MixedOrdinalHuberResidualBlockDiagnostics {
  std::vector<MixedPairLabel> robust_pairs;
  Eigen::VectorXd rho;
  Eigen::VectorXd objective;
  Eigen::MatrixXd moment_influence;
  Eigen::MatrixXd gamma;
  double min_eigen_r = 0.0;
  double raw_min_eigen_r = 0.0;
  bool   r_repair_applied = false;
  double r_ridge = 0.0;
  double r_shrinkage_intensity = 0.0;
};

struct MixedOrdinalHuberResidualStats {
  MixedOrdinalStats stats;
  std::vector<MixedOrdinalHuberResidualBlockDiagnostics> block_diagnostics;
};

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& X);

post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_data(const std::vector<Eigen::MatrixXd>& X,
                              const std::vector<std::vector<std::int32_t>>& ordered);

post_expected<MixedOrdinalPolyserialDpdStats>
mixed_ordinal_stats_polyserial_dpd_from_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& ordered,
    PolyserialPairDpdOptions options = {});

post_expected<MixedOrdinalHuberResidualStats>
mixed_ordinal_stats_huber_residual_from_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& ordered,
    MixedOrdinalHuberResidualOptions options = {});

}  // namespace magmaan::data
