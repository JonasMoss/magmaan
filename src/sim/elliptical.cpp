#include "magmaan/sim/elliptical.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>

#include <Eigen/Cholesky>

namespace magmaan::sim {

namespace {

SimError make_err(SimError::Kind k, std::string detail) {
  return SimError{k, std::move(detail)};
}

sim_expected<void>
validate_mean_covariance(Eigen::Index n,
                         const Eigen::Ref<const Eigen::VectorXd>& mean,
                         const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                         const char* caller) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": n must be positive"));
  }
  if (mean.size() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": mean must not be empty"));
  }
  if (covariance.rows() != covariance.cols() ||
      covariance.rows() != mean.size()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) +
            ": covariance must be square and match mean length"));
  }
  if (!mean.allFinite() || !covariance.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": mean and covariance must be finite"));
  }
  if (!covariance.isApprox(covariance.transpose(), 1e-10)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": covariance must be symmetric"));
  }
  return {};
}

sim_expected<Eigen::MatrixXd>
cholesky_for_covariance(const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                        double multiplier,
                        double jitter,
                        const char* caller) {
  if (!std::isfinite(multiplier) || multiplier <= 0.0 ||
      !std::isfinite(jitter) || jitter < 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": invalid scale or jitter"));
  }
  Eigen::MatrixXd work = multiplier * covariance;
  if (jitter > 0.0) {
    work.diagonal().array() += jitter;
  }
  Eigen::LLT<Eigen::MatrixXd> llt(work);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(
        SimError::Kind::NonPositiveDefinite,
        std::string(caller) + ": covariance is not positive definite"));
  }
  return llt.matrixL();
}

Eigen::MatrixXd draw_standard_normals(Eigen::Index n,
                                      Eigen::Index p,
                                      std::mt19937_64& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd Z(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) {
      Z(i, j) = normal(rng);
    }
  }
  return Z;
}

data::RawData raw_from_matrix(Eigen::MatrixXd X) {
  data::RawData raw;
  raw.X.push_back(std::move(X));
  return raw;
}

sim_expected<double>
validate_scale_mixture(const NormalScaleMixtureSpec& mixture,
                       const char* caller) {
  if (mixture.weights.empty() ||
      mixture.weights.size() != mixture.scale_multipliers.size()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) +
            ": weights and scale_multipliers must be nonempty and equal length"));
  }
  double weight_total = 0.0;
  double second_scale_moment = 0.0;
  for (std::size_t k = 0; k < mixture.weights.size(); ++k) {
    const double weight = mixture.weights[k];
    const double scale = mixture.scale_multipliers[k];
    if (!std::isfinite(weight) || weight <= 0.0 ||
        !std::isfinite(scale) || scale <= 0.0) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          std::string(caller) +
              ": mixture weights and scales must be finite and positive"));
    }
    weight_total += weight;
    second_scale_moment += weight * scale * scale;
  }
  if (!std::isfinite(weight_total) || weight_total <= 0.0 ||
      !std::isfinite(second_scale_moment) || second_scale_moment <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": invalid mixture moments"));
  }
  return second_scale_moment / weight_total;
}

}  // namespace

sim_expected<Eigen::MatrixXd>
simulate_student_t_matrix(Eigen::Index n,
                          const Eigen::Ref<const Eigen::VectorXd>& mean,
                          const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                          double df,
                          std::mt19937_64& rng,
                          const StudentTOptions& options) {
  constexpr const char* caller = "simulate_student_t_matrix";
  if (auto ok = validate_mean_covariance(n, mean, covariance, caller);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (!std::isfinite(df) || df <= 2.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_student_t_matrix: df must be finite and greater than 2"));
  }
  auto L_or = cholesky_for_covariance(
      covariance, (df - 2.0) / df, options.cholesky_jitter, caller);
  if (!L_or.has_value()) return std::unexpected(L_or.error());

  Eigen::MatrixXd X = draw_standard_normals(n, mean.size(), rng);
  X *= L_or->transpose();
  std::chi_squared_distribution<double> chi_square(df);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double w = chi_square(rng);
    if (!std::isfinite(w) || w <= 0.0) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "simulate_student_t_matrix: non-positive chi-square draw"));
    }
    X.row(i) *= std::sqrt(df / w);
  }
  X.rowwise() += mean.transpose();
  return X;
}

sim_expected<data::RawData>
simulate_student_t_raw(Eigen::Index n,
                       const Eigen::Ref<const Eigen::VectorXd>& mean,
                       const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                       double df,
                       std::mt19937_64& rng,
                       const StudentTOptions& options) {
  auto X_or = simulate_student_t_matrix(n, mean, covariance, df, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  return raw_from_matrix(std::move(*X_or));
}

sim_expected<Eigen::MatrixXd>
simulate_scale_mixture_normal_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const NormalScaleMixtureSpec& mixture,
    std::mt19937_64& rng,
    const NormalOptions& options) {
  constexpr const char* caller = "simulate_scale_mixture_normal_matrix";
  if (auto ok = validate_mean_covariance(n, mean, covariance, caller);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto second_scale_or = validate_scale_mixture(mixture, caller);
  if (!second_scale_or.has_value()) {
    return std::unexpected(second_scale_or.error());
  }
  auto L_or = cholesky_for_covariance(
      covariance, 1.0 / *second_scale_or, options.cholesky_jitter, caller);
  if (!L_or.has_value()) return std::unexpected(L_or.error());

  Eigen::MatrixXd X = draw_standard_normals(n, mean.size(), rng);
  X *= L_or->transpose();
  std::discrete_distribution<std::size_t> component(
      mixture.weights.begin(), mixture.weights.end());
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto k = component(rng);
    X.row(i) *= mixture.scale_multipliers[k];
  }
  X.rowwise() += mean.transpose();
  return X;
}

sim_expected<data::RawData>
simulate_scale_mixture_normal_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const NormalScaleMixtureSpec& mixture,
    std::mt19937_64& rng,
    const NormalOptions& options) {
  auto X_or = simulate_scale_mixture_normal_matrix(
      n, mean, covariance, mixture, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  return raw_from_matrix(std::move(*X_or));
}

sim_expected<Eigen::MatrixXd>
simulate_contaminated_normal_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const ContaminatedNormalSpec& contamination,
    std::mt19937_64& rng,
    const NormalOptions& options) {
  if (!std::isfinite(contamination.contamination_probability) ||
      contamination.contamination_probability <= 0.0 ||
      contamination.contamination_probability >= 1.0 ||
      !std::isfinite(contamination.scale_multiplier) ||
      contamination.scale_multiplier <= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_contaminated_normal_matrix: invalid contamination spec"));
  }
  NormalScaleMixtureSpec mixture;
  mixture.weights = {1.0 - contamination.contamination_probability,
                     contamination.contamination_probability};
  mixture.scale_multipliers = {1.0, contamination.scale_multiplier};
  return simulate_scale_mixture_normal_matrix(
      n, mean, covariance, mixture, rng, options);
}

sim_expected<data::RawData>
simulate_contaminated_normal_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const ContaminatedNormalSpec& contamination,
    std::mt19937_64& rng,
    const NormalOptions& options) {
  auto X_or = simulate_contaminated_normal_matrix(
      n, mean, covariance, contamination, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  return raw_from_matrix(std::move(*X_or));
}

sim_expected<Eigen::MatrixXd>
simulate_slash_matrix(Eigen::Index n,
                      const Eigen::Ref<const Eigen::VectorXd>& mean,
                      const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                      const SlashSpec& slash,
                      std::mt19937_64& rng,
                      const NormalOptions& options) {
  constexpr const char* caller = "simulate_slash_matrix";
  if (auto ok = validate_mean_covariance(n, mean, covariance, caller);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (!std::isfinite(slash.q) || slash.q <= 2.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_slash_matrix: q must be finite and greater than 2"));
  }
  auto L_or = cholesky_for_covariance(
      covariance, (slash.q - 2.0) / slash.q, options.cholesky_jitter, caller);
  if (!L_or.has_value()) return std::unexpected(L_or.error());

  Eigen::MatrixXd X = draw_standard_normals(n, mean.size(), rng);
  X *= L_or->transpose();
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double u = std::max(uniform(rng), std::numeric_limits<double>::min());
    X.row(i) /= std::pow(u, 1.0 / slash.q);
  }
  X.rowwise() += mean.transpose();
  return X;
}

sim_expected<data::RawData>
simulate_slash_raw(Eigen::Index n,
                   const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const SlashSpec& slash,
                   std::mt19937_64& rng,
                   const NormalOptions& options) {
  auto X_or = simulate_slash_matrix(n, mean, covariance, slash, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  return raw_from_matrix(std::move(*X_or));
}

}  // namespace magmaan::sim
