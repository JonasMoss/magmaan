#pragma once

#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/sim/normal.hpp"

namespace magmaan::sim {

struct StudentTOptions {
  double cholesky_jitter = 0.0;
};

struct ContaminatedNormalSpec {
  double contamination_probability = 0.05;
  double scale_multiplier = 3.0;
};

struct SlashSpec {
  double q = 5.0;
};

struct NormalScaleMixtureSpec {
  std::vector<double> weights;
  std::vector<double> scale_multipliers;
};

struct EllipticalMomentDiagnostics {
  double scale_second_moment = 1.0;
  double scale_fourth_moment = 1.0;
  double normal_covariance_multiplier = 1.0;
  double radial_kurtosis_factor = 1.0;
  double marginal_excess_kurtosis = 0.0;
  bool finite_fourth_moment = true;
};

sim_expected<EllipticalMomentDiagnostics>
student_t_moment_diagnostics(double df);

sim_expected<EllipticalMomentDiagnostics>
scale_mixture_normal_moment_diagnostics(
    const NormalScaleMixtureSpec& mixture);

sim_expected<EllipticalMomentDiagnostics>
contaminated_normal_moment_diagnostics(
    const ContaminatedNormalSpec& contamination);

sim_expected<EllipticalMomentDiagnostics>
slash_moment_diagnostics(const SlashSpec& slash);

sim_expected<Eigen::MatrixXd>
simulate_student_t_matrix(Eigen::Index n,
                          const Eigen::Ref<const Eigen::VectorXd>& mean,
                          const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                          double df,
                          std::mt19937_64& rng,
                          const StudentTOptions& options = {});

sim_expected<data::RawData>
simulate_student_t_raw(Eigen::Index n,
                       const Eigen::Ref<const Eigen::VectorXd>& mean,
                       const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                       double df,
                       std::mt19937_64& rng,
                       const StudentTOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_scale_mixture_normal_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const NormalScaleMixtureSpec& mixture,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<data::RawData>
simulate_scale_mixture_normal_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const NormalScaleMixtureSpec& mixture,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_contaminated_normal_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const ContaminatedNormalSpec& contamination,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<data::RawData>
simulate_contaminated_normal_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const ContaminatedNormalSpec& contamination,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_slash_matrix(Eigen::Index n,
                      const Eigen::Ref<const Eigen::VectorXd>& mean,
                      const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                      const SlashSpec& slash,
                      std::mt19937_64& rng,
                      const NormalOptions& options = {});

sim_expected<data::RawData>
simulate_slash_raw(Eigen::Index n,
                   const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const SlashSpec& slash,
                   std::mt19937_64& rng,
                   const NormalOptions& options = {});

}  // namespace magmaan::sim
