#include "magmaan/sim/projection.hpp"

#include <cmath>
#include <string>
#include <utility>

#include "magmaan/sim/norta.hpp"

namespace magmaan::sim {

namespace {

SimError make_err(SimError::Kind k, std::string detail) {
  return SimError{k, std::move(detail)};
}

sim_expected<void>
validate_thresholds(const Eigen::VectorXd& thresholds,
                    const std::string& caller,
                    Eigen::Index column) {
  if (thresholds.size() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        caller + ": ordinal thresholds must not be empty for column " +
            std::to_string(column)));
  }
  for (Eigen::Index k = 0; k < thresholds.size(); ++k) {
    if (!std::isfinite(thresholds(k))) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          caller + ": ordinal thresholds must be finite for column " +
              std::to_string(column)));
    }
    if (k > 0 && !(thresholds(k - 1) < thresholds(k))) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          caller + ": ordinal thresholds must be strictly increasing for column " +
              std::to_string(column)));
    }
  }
  return {};
}

int ordinal_category(double x, const Eigen::VectorXd& thresholds) noexcept {
  int category = 1;
  for (Eigen::Index k = 0; k < thresholds.size(); ++k) {
    if (x <= thresholds(k)) break;
    ++category;
  }
  return category;
}

}  // namespace

sim_expected<Eigen::VectorXd>
thresholds_from_probabilities(
    const Eigen::Ref<const Eigen::VectorXd>& probabilities) {
  if (probabilities.size() < 2) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "thresholds_from_probabilities: need at least two categories"));
  }
  double total = 0.0;
  for (Eigen::Index k = 0; k < probabilities.size(); ++k) {
    const double p = probabilities(k);
    if (!std::isfinite(p) || p <= 0.0) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "thresholds_from_probabilities: probabilities must be finite and positive"));
    }
    total += p;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "thresholds_from_probabilities: probability total must be positive"));
  }

  Eigen::VectorXd thresholds(probabilities.size() - 1);
  double cumulative = 0.0;
  for (Eigen::Index k = 0; k < thresholds.size(); ++k) {
    cumulative += probabilities(k) / total;
    if (!(cumulative > 0.0) || !(cumulative < 1.0)) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "thresholds_from_probabilities: cumulative probabilities must stay inside (0, 1)"));
    }
    auto q_or = normal_quantile(cumulative);
    if (!q_or.has_value()) return std::unexpected(q_or.error());
    thresholds(k) = *q_or;
  }
  return thresholds;
}

sim_expected<MixedProjectionResult>
project_mixed_matrix(const Eigen::Ref<const Eigen::MatrixXd>& latent,
                     const MixedProjectionSpec& spec) {
  if (latent.rows() <= 0 || latent.cols() <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "project_mixed_matrix: latent matrix must not be empty"));
  }
  if (!latent.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "project_mixed_matrix: latent matrix must be finite"));
  }
  const auto p_size = static_cast<std::size_t>(latent.cols());
  if (spec.kinds.size() != p_size || spec.thresholds.size() != p_size) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "project_mixed_matrix: kind and threshold counts must match column count"));
  }

  MixedProjectionResult out;
  out.X = latent;
  out.ordered.assign(p_size, 0);
  out.n_levels.assign(p_size, 0);
  out.category_counts.resize(p_size);

  for (Eigen::Index j = 0; j < latent.cols(); ++j) {
    const auto idx = static_cast<std::size_t>(j);
    switch (spec.kinds[idx]) {
      case ObservedKind::Continuous:
        if (spec.thresholds[idx].size() != 0) {
          return std::unexpected(make_err(
              SimError::Kind::InvalidInput,
              "project_mixed_matrix: continuous columns must not have thresholds"));
        }
        continue;
      case ObservedKind::Ordinal:
        break;
    }

    if (auto ok = validate_thresholds(
            spec.thresholds[idx], "project_mixed_matrix", j);
        !ok.has_value()) {
      return std::unexpected(ok.error());
    }

    const Eigen::Index n_levels = spec.thresholds[idx].size() + 1;
    out.ordered[idx] = 1;
    out.n_levels[idx] = static_cast<std::int32_t>(n_levels);
    out.category_counts[idx] = Eigen::VectorXi::Zero(n_levels);
    for (Eigen::Index i = 0; i < latent.rows(); ++i) {
      const int category = ordinal_category(latent(i, j), spec.thresholds[idx]);
      out.X(i, j) = static_cast<double>(category);
      out.category_counts[idx](category - 1) += 1;
    }
  }

  return out;
}

sim_expected<MixedProjectionResult>
project_ordinal_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& latent,
    const std::vector<Eigen::VectorXd>& thresholds) {
  MixedProjectionSpec spec;
  spec.kinds.assign(static_cast<std::size_t>(latent.cols()),
                    ObservedKind::Ordinal);
  spec.thresholds = thresholds;
  return project_mixed_matrix(latent, spec);
}

}  // namespace magmaan::sim
