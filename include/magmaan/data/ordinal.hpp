#pragma once

#include <cstddef>
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

  // Per-case stage-1 influence functions g_i (rows) of the moment vector
  // κ̂ = [thresholds; polychorics], one n_b × m_b matrix per block. Their
  // empirical covariance is the NACOV: (1/n_b)·Σ_i g_i g_iᵀ = NACOV[b]. Carried
  // for the infinitesimal-jackknife sandwich (`robust_ordinal_ij`), which needs
  // the influence of the estimated weight Ŵ = diag(NACOV)⁻¹ -- a quantity the
  // fixed-weight observed-bread sandwich omits. Empty when not computed.
  std::vector<Eigen::MatrixXd> moment_influence;

  // 0-based integer category data (n_b × p), one matrix per block, -1 for
  // missing. Carried so the IJ can finite-difference ∂Γ̂/∂κ (the "bread" piece of
  // the weight influence): perturb thresholds/polychorics κ and recompute NACOV
  // from this data via `ordinal_gamma_diag_jacobian_fd`. Empty when not computed.
  std::vector<Eigen::MatrixXi> int_data;

  // Stage-1 estimating-equation Jacobian inverse B_inv (m × m, one per block):
  // Â⁻¹ = N·B_inv with Â the score-Jacobian, so the per-case scores recover as
  // s_i = N⁻¹·B_inv⁻¹·g_i (g_i = moment_influence row). Carried for the IJ's
  // Â-direct ("score-Jacobian variation") channel of the weight influence. Empty
  // when not computed.
  std::vector<Eigen::MatrixXd> moment_bread;

  // Optional pairwise-deletion diagnostics. Empty for ordinary complete/listwise
  // stats. Supports are 0-based variable indices; singleton rows are thresholds,
  // pair rows are polychorics.
  std::vector<std::vector<std::int32_t>> moment_support_i;
  std::vector<std::vector<std::int32_t>> moment_support_j;
  std::vector<std::vector<std::int64_t>> moment_n_obs;
  std::vector<Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>>
      moment_overlap_n_obs;
  std::string pairwise_gamma;
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

// Observed all-ordinal moment metadata without Gamma/NACOV or fit weights.
// This is the non-breaking split point for cache-aware ordinal fitting: legacy
// `OrdinalStats` remains the compatibility adapter that materializes all
// pieces.
struct OrdinalMoments {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

// Observed mixed continuous/ordinal moment metadata without Gamma/NACOV or fit
// weights. Moment order matches `MixedOrdinalStats::moments`.
struct MixedOrdinalMoments {
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::VectorXd> mean;
  std::vector<std::vector<std::int32_t>> ordered;
  std::vector<Eigen::VectorXd> thresholds;
  std::vector<std::vector<std::int32_t>> threshold_ov;
  std::vector<std::vector<std::int32_t>> threshold_level;
  std::vector<Eigen::VectorXd> moments;
  std::vector<std::int64_t> n_obs;
  std::vector<std::vector<std::int32_t>> n_levels;
  std::vector<std::vector<std::string>> ov_names;
};

enum class OrdinalWorkspacePurpose {
  FitOnly,
  FitPlusInference,
  InferenceOnly,
};

enum class OrdinalEstimatorKind {
  ULS,
  DWLS,
  WLS,
};

enum class OrdinalMomentParameterization {
  Delta,
  Theta,
};

// Advisory threshold-structure label for benchmark/plan bookkeeping; the fit
// paths derive the actual threshold design (free / merged / fixed /
// linear-map, including cross-group invariance) from the partable. All three
// modes are supported by the threshold-profiled paths.
enum class OrdinalThresholdMode {
  FreeIdentity,
  LinearMap,
  FixedOrConstrained,
};

enum class OrdinalGammaMaterialization {
  None,
  Diagonal,
  Full,
  Reduced,
};

enum class OrdinalPairwiseGammaKind {
  Overlap,
  Nominal,
};

struct OrdinalWeightPlan {
  OrdinalWorkspacePurpose purpose = OrdinalWorkspacePurpose::FitOnly;
  OrdinalEstimatorKind estimator = OrdinalEstimatorKind::DWLS;
  OrdinalMomentParameterization parameterization =
      OrdinalMomentParameterization::Delta;
  OrdinalThresholdMode threshold_mode = OrdinalThresholdMode::FreeIdentity;
  OrdinalGammaMaterialization materialization =
      OrdinalGammaMaterialization::Diagonal;
};

struct OrdinalGammaCacheBlock {
  Eigen::VectorXd diagonal;
  Eigen::MatrixXd gamma;
  Eigen::MatrixXd w_dwls;
  Eigen::MatrixXd w_wls;
  bool has_diagonal = false;
  bool has_full = false;
  bool has_dwls_weight = false;
  bool has_wls_weight = false;
};

struct OrdinalGammaCache {
  std::vector<OrdinalGammaCacheBlock> blocks;

  std::size_t block_count() const noexcept { return blocks.size(); }
};

struct OrdinalWorkspace {
  OrdinalMoments moments;
  OrdinalGammaCache gamma_cache;
};

struct MixedOrdinalWorkspace {
  MixedOrdinalMoments moments;
  OrdinalGammaCache gamma_cache;
};

OrdinalMoments ordinal_moments_from_stats(const OrdinalStats& stats);
MixedOrdinalMoments
mixed_ordinal_moments_from_stats(const MixedOrdinalStats& stats);

OrdinalGammaCache ordinal_gamma_cache_from_stats(const OrdinalStats& stats);
OrdinalGammaCache
ordinal_gamma_cache_from_stats(const MixedOrdinalStats& stats);
OrdinalGammaCache
ordinal_gamma_cache_from_diagonal(const std::vector<Eigen::VectorXd>& diagonal);

OrdinalWeightPlan ordinal_weight_plan(
    OrdinalWorkspacePurpose purpose,
    OrdinalEstimatorKind estimator,
    OrdinalMomentParameterization parameterization =
        OrdinalMomentParameterization::Delta,
    OrdinalThresholdMode threshold_mode = OrdinalThresholdMode::FreeIdentity);

post_expected<void> ordinal_gamma_cache_ensure_diagonal(
    OrdinalGammaCache& cache);
post_expected<void> ordinal_gamma_cache_ensure_dwls_weights(
    OrdinalGammaCache& cache);
post_expected<void> ordinal_gamma_cache_ensure_wls_weights(
    OrdinalGammaCache& cache);

post_expected<OrdinalWorkspace> ordinal_workspace_from_integer_data(
    const std::vector<Eigen::MatrixXd>& X,
    OrdinalWeightPlan plan = {});

post_expected<MixedOrdinalWorkspace> mixed_ordinal_workspace_from_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& ordered,
    OrdinalWeightPlan plan = {});

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

// `full_wls_weight` controls whether the full-WLS weight (the dense NACOV
// inverse) is materialized; see the mixed-ordinal overload below. DWLS-only
// callers pass `false` to skip the O(m³) inverse, which is also often singular
// at small N with many indicators. When `false`, `W_wls` is left empty.
post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& X,
                                bool full_wls_weight = true);

post_expected<OrdinalStats>
ordinal_stats_from_observed_integer_data(
    const std::vector<Eigen::MatrixXd>& X,
    OrdinalPairwiseGammaKind gamma_kind = OrdinalPairwiseGammaKind::Overlap,
    bool full_wls_weight = true);

// Finite-difference Jacobian D(k,l) = ∂NACOV_kk/∂κ_l of the polychoric NACOV
// diagonal with respect to the stage-1 moments κ = [thresholds; polychorics],
// at fixed integer data `Xcat` (0-based, n × p). This is the "bread" channel of
// the estimated-weight influence the infinitesimal jackknife needs: perturbing
// κ and recomputing the empirical sandwich Γ̂ = n·B_inv·INNER·B_inv'. `levels`
// is the per-variable category count; `thresholds` is the stacked threshold
// vector (matching NACOV's leading block); `R` the p × p polychoric matrix.
// Returns an m × m matrix (m = #thresholds + p(p-1)/2). Complete data only.
post_expected<Eigen::MatrixXd>
ordinal_gamma_diag_jacobian_fd(const Eigen::MatrixXi& Xcat,
                               const std::vector<std::int32_t>& levels,
                               const Eigen::VectorXd& thresholds,
                               const Eigen::MatrixXd& R,
                               double h_rel = 1e-4);

// Finite-difference Jacobian D(vec,l) = ∂vec(NACOV)/∂κ_l of the full
// polychoric NACOV matrix with respect to κ = [thresholds; polychorics], at
// fixed integer data `Xcat`. The vectorization is Eigen's column-major
// `MatrixXd::data()` order, so `Map<MatrixXd>(D.col(l).data(), m, m)` recovers
// the l-th derivative matrix. Complete data only.
post_expected<Eigen::MatrixXd>
ordinal_gamma_jacobian_fd(const Eigen::MatrixXi& Xcat,
                          const std::vector<std::int32_t>& levels,
                          const Eigen::VectorXd& thresholds,
                          const Eigen::MatrixXd& R,
                          double h_rel = 1e-4);

// Per-case DATA-DIRECT influence of the NACOV diagonal: row i, col k is
// IF_i(NACOV_kk) holding κ fixed -- the full sandwich influence of
// Γ̂ = Â⁻¹V̂Â⁻¹ at case i (V̂-direct + Â-direct, i.e. score-product AND
// score-Jacobian variation). Recomputes the per-case scores from `Xcat` at κ
// (marginal threshold scores for the Â₁₁ block, bivariate pair scores for the
// Â₂₁ coupling), so it captures what the influence functions alone cannot.
// Returns n × m. Pair with `ordinal_gamma_diag_jacobian_fd` (the κ-movement
// channel) for the complete IF of Γ̂. Complete data only.
post_expected<Eigen::MatrixXd>
ordinal_gamma_diag_data_influence(const Eigen::MatrixXi& Xcat,
                                  const std::vector<std::int32_t>& levels,
                                  const Eigen::VectorXd& thresholds,
                                  const Eigen::MatrixXd& R);

// Per-case DATA-DIRECT influence of the full NACOV matrix at fixed κ. Returns
// n × (m*m); each row uses the same column-major matrix vectorization as
// `ordinal_gamma_jacobian_fd`. Pair with `ordinal_gamma_jacobian_fd` and the
// stage-1 moment influence rows for the complete IF of Γ̂. Complete data only.
post_expected<Eigen::MatrixXd>
ordinal_gamma_data_influence(const Eigen::MatrixXi& Xcat,
                             const std::vector<std::int32_t>& levels,
                             const Eigen::VectorXd& thresholds,
                             const Eigen::MatrixXd& R);

// `full_wls_weight` controls whether the full-WLS weight (the dense NACOV
// inverse) is materialized. DWLS needs only the diagonal `W_dwls` (an
// element-wise reciprocal of the NACOV diagonal) and the robust sandwich uses
// NACOV directly, so a DWLS-only caller can pass `false` to skip the O(m³)
// inverse — which is also often singular at small N with many indicators. When
// `false`, `W_wls` is left empty and an explicit full-WLS fit will report it.
post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_data(const std::vector<Eigen::MatrixXd>& X,
                              const std::vector<std::vector<std::int32_t>>& ordered,
                              bool full_wls_weight = true);

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
