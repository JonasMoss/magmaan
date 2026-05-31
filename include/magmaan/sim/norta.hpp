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

  static MarginalSpec standard_normal(double mean = 0.0, double sd = 1.0);
  static MarginalSpec standardized_lognormal(double sigma_log,
                                             double mean = 0.0,
                                             double sd = 1.0);
  static MarginalSpec tukey_g_h(double g,
                                double h,
                                double mean = 0.0,
                                double sd = 1.0);
};

struct NortaOptions {
  int quadrature_points = 31;
  int max_bisection_iter = 80;
  double rho_bound = 0.999;
  double calibration_tol = 1e-8;
  double cholesky_jitter = 1e-10;
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
