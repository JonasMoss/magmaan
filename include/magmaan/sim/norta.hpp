#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::sim {

enum class MarginalKind : std::uint8_t {
  StandardNormal,
  StandardizedLognormal,
  TukeyGH,
  Pearson,
};

enum class MomentMatchFamily : std::uint8_t {
  TukeyGH,
  Pearson,
  Johnson,
};

struct ShapeMoments {
  double skewness = 0.0;
  double excess_kurtosis = 0.0;
};

struct MarginalMomentSummary {
  double mean = 0.0;
  double sd = 1.0;
  double skewness = 0.0;
  double excess_kurtosis = 0.0;
};

struct MarginalSpec {
  MarginalKind kind = MarginalKind::StandardNormal;
  double mean = 0.0;
  double sd = 1.0;

  // StandardizedLognormal: exp(sigma_log * Z), centered and scaled before
  // applying mean/sd.
  double sigma_log = 1.0;

  // Tukey g-and-h: ((exp(gZ)-1)/g) * exp(hZ^2/2), with the continuous
  // g -> 0 limit. Moments are determined numerically during calibration.
  double g = 0.0;
  double h = 0.0;

  // Pearson distribution-system parameters in PearsonDS convention. `type`
  // matches PearsonDS: 0 normal, 1 beta, 2 symmetric beta, 3 gamma, 4 Type IV,
  // 5 inverse gamma, 6 beta prime, 7 Student t.
  int pearson_type = 0;
  double pearson_p1 = 0.0;
  double pearson_p2 = 0.0;
  double pearson_p3 = 0.0;
  double pearson_p4 = 0.0;

  static MarginalSpec standard_normal(double mean = 0.0, double sd = 1.0);
  static MarginalSpec standardized_lognormal(double sigma_log,
                                             double mean = 0.0,
                                             double sd = 1.0);
  static MarginalSpec tukey_g_h(double g,
                                double h,
                                double mean = 0.0,
                                double sd = 1.0);
  static MarginalSpec pearson(int type,
                              double p1,
                              double p2 = 0.0,
                              double p3 = 0.0,
                              double p4 = 0.0,
                              double mean = 0.0,
                              double sd = 1.0);
};

struct MomentMatchSpec {
  MomentMatchFamily family = MomentMatchFamily::TukeyGH;
  ShapeMoments shape = {};
  double mean = 0.0;
  double sd = 1.0;
};

struct MomentMatchOptions {
  int quadrature_points = 81;
  int max_iter = 80;
  int grid_points_g = 29;
  int grid_points_h = 25;
  double objective_tol = 1e-8;
  double parameter_tol = 1e-8;
  double finite_diff_step = 1e-4;
  double tukey_g_bound = 3.0;
  double tukey_h_upper = 0.249;
};

struct MomentMatchResult {
  MarginalSpec marginal = {};
  MarginalMomentSummary moments = {};
  double residual_norm = 0.0;
  int iterations = 0;
};

struct NortaOptions {
  int quadrature_points = 31;
  int max_bisection_iter = 80;
  double rho_bound = 0.999;
  double calibration_tol = 1e-8;
  double cholesky_jitter = 1e-10;
};

struct IndependentOptions {
  int quadrature_points = 31;
};

struct NortaCalibration {
  Eigen::MatrixXd latent_corr;
  Eigen::VectorXd marginal_mean;
  Eigen::VectorXd marginal_sd;
};

sim_expected<double>
normal_quantile(double u);

double normal_cdf(double z) noexcept;

sim_expected<double>
marginal_quantile(const MarginalSpec& marginal, double u);

sim_expected<MarginalMomentSummary>
marginal_moment_summary(const MarginalSpec& marginal,
                        int quadrature_points = 81);

sim_expected<MomentMatchResult>
fit_marginal_to_moments(const MomentMatchSpec& spec,
                        const MomentMatchOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_independent_matrix(Eigen::Index n,
                            const std::vector<MarginalSpec>& marginals,
                            std::mt19937_64& rng,
                            const IndependentOptions& options = {});

sim_expected<data::RawData>
simulate_independent_raw(Eigen::Index n,
                         const std::vector<MarginalSpec>& marginals,
                         std::mt19937_64& rng,
                         const IndependentOptions& options = {});

sim_expected<NortaCalibration>
calibrate_norta(const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                const std::vector<MarginalSpec>& marginals,
                const NortaOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_norta_matrix(Eigen::Index n,
                      const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                      const std::vector<MarginalSpec>& marginals,
                      std::mt19937_64& rng,
                      const NortaOptions& options = {});

sim_expected<data::RawData>
simulate_norta_raw(Eigen::Index n,
                   const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                   const std::vector<MarginalSpec>& marginals,
                   std::mt19937_64& rng,
                   const NortaOptions& options = {});

}  // namespace magmaan::sim
