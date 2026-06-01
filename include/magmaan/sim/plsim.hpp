#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::sim {

enum class PlsimCovarianceMethod : std::uint8_t {
  Hermite,
  Quadrature,
  HermiteThenQuadrature,
};

struct PlsimOptions {
  int num_segments = 4;
  bool monotone = false;
  int max_iter = 80;
  int quadrature_points = 31;
  int hermite_order = 24;
  double marginal_tol = 1e-8;
  double correlation_tol = 1e-8;
  double rho_bound = 0.999;
  double cholesky_jitter = 1e-10;
};

struct PlsimMarginal {
  Eigen::VectorXd gamma;
  Eigen::VectorXd a;
  Eigen::VectorXd b;
  Eigen::VectorXd hermite_coefficients;
  double mean = 0.0;
  double variance = 1.0;
  double skewness = 0.0;
  double excess_kurtosis = 0.0;
};

struct PlsimCalibration {
  std::vector<PlsimMarginal> marginals;
  Eigen::MatrixXd intermediate_corr;
  Eigen::MatrixXd achieved_corr;
  Eigen::MatrixXi iterations;
};

sim_expected<PlsimMarginal>
fit_plsim_marginal(double skewness,
                   double excess_kurtosis,
                   const PlsimOptions& options = {});

sim_expected<Eigen::VectorXd>
plsim_hermite_coefficients(const PlsimMarginal& marginal, int order);

double plsim_apply(const PlsimMarginal& marginal, double z) noexcept;

sim_expected<double>
plsim_covariance_hermite(const PlsimMarginal& left,
                         const PlsimMarginal& right,
                         double rho,
                         int order);

sim_expected<double>
plsim_covariance_quadrature(const PlsimMarginal& left,
                            const PlsimMarginal& right,
                            double rho,
                            int quadrature_points);

sim_expected<PlsimCalibration>
calibrate_plsim(
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    PlsimCovarianceMethod method = PlsimCovarianceMethod::HermiteThenQuadrature,
    const PlsimOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_plsim_matrix(Eigen::Index n,
                      const PlsimCalibration& calibration,
                      std::mt19937_64& rng,
                      const PlsimOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_plsim_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    std::mt19937_64& rng,
    PlsimCovarianceMethod method = PlsimCovarianceMethod::HermiteThenQuadrature,
    const PlsimOptions& options = {});

sim_expected<data::RawData>
simulate_plsim_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    std::mt19937_64& rng,
    PlsimCovarianceMethod method = PlsimCovarianceMethod::HermiteThenQuadrature,
    const PlsimOptions& options = {});

}  // namespace magmaan::sim
