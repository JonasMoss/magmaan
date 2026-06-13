#include "magmaan/sim/model_implied.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::sim {

namespace {

SimError make_err(SimError::Kind k, std::string detail) {
  return SimError{k, std::move(detail)};
}

SimError model_err(const char* caller, const ModelError& e) {
  return make_err(SimError::Kind::NumericIssue,
                  std::string(caller) + ": " + e.detail);
}

std::string ov_label(const model::MatrixRep& rep, std::size_t j) {
  if (!rep.ov_names.empty() && j < rep.ov_names[0].size()) {
    return rep.ov_names[0][j];
  }
  return "column " + std::to_string(j);
}

sim_expected<double>
row_value(const spec::LatentStructure& pt,
          std::size_t row,
          Eigen::Ref<const Eigen::VectorXd> theta) {
  if (pt.free[row] > 0) {
    const auto k = static_cast<Eigen::Index>(pt.free[row] - 1);
    if (k < 0 || k >= theta.size()) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold free index exceeds theta length"));
    }
    if (!std::isfinite(theta(k))) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold value must be finite"));
    }
    return theta(k);
  }
  const double value = pt.fixed_value[row];
  if (!std::isfinite(value)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "lower_model_implied: fixed threshold value must be finite"));
  }
  return value;
}

sim_expected<void>
sort_validate_thresholds(std::vector<double>& thresholds,
                         const std::string& name) {
  std::sort(thresholds.begin(), thresholds.end());
  for (std::size_t k = 0; k < thresholds.size(); ++k) {
    if (!std::isfinite(thresholds[k])) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold for " + name + " must be finite"));
    }
    if (k > 0 && !(thresholds[k - 1] < thresholds[k])) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: duplicate threshold for " + name));
    }
  }
  return {};
}

Eigen::VectorXd to_vector(const std::vector<double>& values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = values[static_cast<std::size_t>(i)];
  }
  return out;
}

}  // namespace

sim_expected<ModelImpliedPopulation>
lower_model_implied(const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    Eigen::Ref<const Eigen::VectorXd> theta) {
  constexpr const char* caller = "lower_model_implied";
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(model_err(caller, ev_or.error()));
  }
  auto moments_or = ev_or->sigma(theta);
  if (!moments_or.has_value()) {
    return std::unexpected(model_err(caller, moments_or.error()));
  }
  const model::ImpliedMoments moments = std::move(*moments_or);

  const std::size_t nb = rep.dims.size();
  if (nb == 0 || moments.sigma.size() != nb || moments.mu.size() != nb) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "lower_model_implied: block counts do not agree"));
  }
  if (rep.ov_names.size() != nb) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "lower_model_implied: MatrixRep observed-name blocks do not agree"));
  }
  const std::size_t p = rep.ov_names[0].size();
  if (p == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "lower_model_implied: model has no observed variables"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    if (static_cast<std::size_t>(rep.dims[b].n_observed) != p ||
        rep.ov_names[b].size() != p) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: all blocks must share observed-variable order"));
    }
  }

  std::vector<std::vector<std::vector<double>>> threshold_values(
      nb, std::vector<std::vector<double>>(p));
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Threshold) continue;
    if (pt.group[i] <= 0) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold row has no positive group"));
    }
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= nb) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold row group exceeds block count"));
    }
    const std::int32_t lhs = pt.lhs_var[i];
    if (lhs < 0 || lhs >= pt.n_vars) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold row references an unknown variable"));
    }
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(lhs)];
    if (ov < 0 || static_cast<std::size_t>(ov) >= p) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: threshold row references a non-observed variable"));
    }
    auto value_or = row_value(pt, i, theta);
    if (!value_or.has_value()) return std::unexpected(value_or.error());
    threshold_values[b][static_cast<std::size_t>(ov)].push_back(*value_or);
  }

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t j = 0; j < p; ++j) {
      if (auto ok = sort_validate_thresholds(
              threshold_values[b][j], ov_label(rep, j));
          !ok.has_value()) {
        return std::unexpected(ok.error());
      }
    }
  }

  ModelImpliedPopulation out;
  out.ov_names = rep.ov_names[0];
  out.kinds.assign(p, ObservedKind::Continuous);
  out.n_levels.assign(p, 0);
  for (std::size_t j = 0; j < p; ++j) {
    const std::size_t count = threshold_values[0][j].size();
    for (std::size_t b = 1; b < nb; ++b) {
      if (threshold_values[b][j].size() != count) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "lower_model_implied: threshold count differs across groups for " +
                ov_label(rep, j)));
      }
    }
    if (count > 0) {
      out.kinds[j] = ObservedKind::Ordinal;
      out.n_levels[j] = static_cast<std::int32_t>(count + 1);
    }
  }

  out.groups.reserve(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto& sigma = moments.sigma[b];
    if (sigma.rows() != sigma.cols() ||
        static_cast<std::size_t>(sigma.rows()) != p ||
        !sigma.allFinite()) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: implied covariance has invalid shape or values"));
    }
    MixedPopulation group;
    group.latent.covariance = sigma;
    if (moments.mu[b].size() == 0) {
      group.latent.mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
    } else if (static_cast<std::size_t>(moments.mu[b].size()) == p &&
               moments.mu[b].allFinite()) {
      group.latent.mean = moments.mu[b];
    } else {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "lower_model_implied: implied mean has invalid shape or values"));
    }

    group.observed.kinds = out.kinds;
    group.observed.thresholds.resize(p);
    for (std::size_t j = 0; j < p; ++j) {
      group.observed.thresholds[j] = to_vector(threshold_values[b][j]);
    }
    out.groups.push_back(std::move(group));
  }
  return out;
}

sim_expected<MixedPopulationDraw>
simulate_model_implied_group(Eigen::Index n,
                             const MixedPopulation& population,
                             const GeneratorSpec& generator,
                             std::mt19937_64& rng) {
  switch (generator.kind) {
    case GeneratorKind::Normal:
      return simulate_mixed_population_normal(
          n, population, rng, generator.normal_options);
    case GeneratorKind::StudentT:
      return simulate_mixed_population_student_t(
          n, population, generator.df, rng, generator.student_t_options);
    case GeneratorKind::ScaleMixture:
      return simulate_mixed_population_scale_mixture(
          n, population, generator.scale_mixture, rng,
          generator.normal_options);
    case GeneratorKind::ContaminatedNormal:
      return simulate_mixed_population_contaminated_normal(
          n, population, generator.contamination, rng,
          generator.normal_options);
    case GeneratorKind::Slash:
      return simulate_mixed_population_slash(
          n, population, generator.slash, rng, generator.normal_options);
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidInput,
      "simulate_model_implied_group: unknown generator kind"));
}

sim_expected<std::vector<MixedPopulationDraw>>
simulate_model_implied(Eigen::Index n,
                       const ModelImpliedPopulation& population,
                       const GeneratorSpec& generator,
                       std::mt19937_64& rng) {
  if (population.groups.empty()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_model_implied: population has no groups"));
  }
  std::vector<MixedPopulationDraw> out;
  out.reserve(population.groups.size());
  for (const auto& group : population.groups) {
    auto draw_or = simulate_model_implied_group(n, group, generator, rng);
    if (!draw_or.has_value()) return std::unexpected(draw_or.error());
    out.push_back(std::move(*draw_or));
  }
  return out;
}

}  // namespace magmaan::sim
