#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/sim/population.hpp"
#include "magmaan/sim/projection.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::sim {

struct ModelImpliedPopulation {
  std::vector<MixedPopulation> groups;
  std::vector<std::string> ov_names;
  std::vector<ObservedKind> kinds;
  std::vector<std::int32_t> n_levels;
};

sim_expected<ModelImpliedPopulation>
lower_model_implied(const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    Eigen::Ref<const Eigen::VectorXd> theta);

enum class GeneratorKind : std::uint8_t {
  Normal,
  StudentT,
  ScaleMixture,
  ContaminatedNormal,
  Slash,
};

struct GeneratorSpec {
  GeneratorKind kind = GeneratorKind::Normal;
  double df = 0.0;
  NormalScaleMixtureSpec scale_mixture{};
  ContaminatedNormalSpec contamination{};
  SlashSpec slash{};
  NormalOptions normal_options{};
  StudentTOptions student_t_options{};
};

sim_expected<MixedPopulationDraw>
simulate_model_implied_group(Eigen::Index n,
                             const MixedPopulation& population,
                             const GeneratorSpec& generator,
                             std::mt19937_64& rng);

sim_expected<std::vector<MixedPopulationDraw>>
simulate_model_implied(Eigen::Index n,
                       const ModelImpliedPopulation& population,
                       const GeneratorSpec& generator,
                       std::mt19937_64& rng);

}  // namespace magmaan::sim
