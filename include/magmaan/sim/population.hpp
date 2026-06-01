#pragma once

#include <random>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/sim/elliptical.hpp"
#include "magmaan/sim/normal.hpp"
#include "magmaan/sim/projection.hpp"

namespace magmaan::sim {

struct ContinuousPopulation {
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
};

struct MixedPopulation {
  ContinuousPopulation latent;
  MixedProjectionSpec observed;
};

struct MixedPopulationDraw {
  Eigen::MatrixXd latent;
  MixedProjectionResult observed;
};

sim_expected<Eigen::MatrixXd>
simulate_continuous_population_normal(Eigen::Index n,
                                      const ContinuousPopulation& population,
                                      std::mt19937_64& rng,
                                      const NormalOptions& options = {});

sim_expected<MixedPopulationDraw>
simulate_mixed_population_normal(Eigen::Index n,
                                 const MixedPopulation& population,
                                 std::mt19937_64& rng,
                                 const NormalOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_continuous_population_student_t(
    Eigen::Index n,
    const ContinuousPopulation& population,
    double df,
    std::mt19937_64& rng,
    const StudentTOptions& options = {});

sim_expected<MixedPopulationDraw>
simulate_mixed_population_student_t(Eigen::Index n,
                                    const MixedPopulation& population,
                                    double df,
                                    std::mt19937_64& rng,
                                    const StudentTOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_continuous_population_scale_mixture(
    Eigen::Index n,
    const ContinuousPopulation& population,
    const NormalScaleMixtureSpec& mixture,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<MixedPopulationDraw>
simulate_mixed_population_scale_mixture(
    Eigen::Index n,
    const MixedPopulation& population,
    const NormalScaleMixtureSpec& mixture,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_continuous_population_contaminated_normal(
    Eigen::Index n,
    const ContinuousPopulation& population,
    const ContaminatedNormalSpec& contamination,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<MixedPopulationDraw>
simulate_mixed_population_contaminated_normal(
    Eigen::Index n,
    const MixedPopulation& population,
    const ContaminatedNormalSpec& contamination,
    std::mt19937_64& rng,
    const NormalOptions& options = {});

sim_expected<Eigen::MatrixXd>
simulate_continuous_population_slash(Eigen::Index n,
                                     const ContinuousPopulation& population,
                                     const SlashSpec& slash,
                                     std::mt19937_64& rng,
                                     const NormalOptions& options = {});

sim_expected<MixedPopulationDraw>
simulate_mixed_population_slash(Eigen::Index n,
                                const MixedPopulation& population,
                                const SlashSpec& slash,
                                std::mt19937_64& rng,
                                const NormalOptions& options = {});

}  // namespace magmaan::sim
