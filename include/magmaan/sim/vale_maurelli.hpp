#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::sim {

struct FleishmanCoefficients {
  double a = 0.0;
  double b = 1.0;
  double c = 0.0;
  double d = 0.0;
};

struct ValeMaurelliOptions {
  int max_iter = 80;
  double coefficient_tol = 1e-10;
  double correlation_tol = 1e-10;
  double rho_bound = 0.999;
  double cholesky_jitter = 1e-10;
};

struct ValeMaurelliCalibration {
  std::vector<FleishmanCoefficients> coefficients;
  Eigen::MatrixXd intermediate_corr;
};

sim_expected<FleishmanCoefficients>
fit_fleishman_coefficients(double skewness,
                           double excess_kurtosis,
                           const ValeMaurelliOptions& options = {});

sim_expected<double>
fleishman_covariance(const FleishmanCoefficients& left,
                     const FleishmanCoefficients& right,
                     double rho);

sim_expected<ValeMaurelliCalibration>
calibrate_vale_maurelli(
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    const ValeMaurelliOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_vale_maurelli_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    std::mt19937_64& rng,
    const ValeMaurelliOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_vale_maurelli_matrix(
    Eigen::Index n,
    const ValeMaurelliCalibration& calibration,
    std::mt19937_64& rng,
    const ValeMaurelliOptions& options = {});

sim_expected<data::RawData>
simulate_vale_maurelli_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    std::mt19937_64& rng,
    const ValeMaurelliOptions& options = {});

}  // namespace magmaan::sim
