#include "magmaan/sim/normal.hpp"

#include <cmath>
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
validate_normal_inputs(Eigen::Index n,
                       const Eigen::Ref<const Eigen::VectorXd>& mean,
                       const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                       const NormalOptions& options) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_normal_matrix: n must be positive"));
  }
  if (mean.size() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_normal_matrix: mean must not be empty"));
  }
  if (sigma.rows() != sigma.cols() || sigma.rows() != mean.size()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_normal_matrix: sigma must be square and match mean length"));
  }
  if (!mean.allFinite() || !sigma.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_normal_matrix: mean and sigma must be finite"));
  }
  if (!std::isfinite(options.cholesky_jitter) ||
      options.cholesky_jitter < 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_normal_matrix: cholesky_jitter must be finite and nonnegative"));
  }
  if (!sigma.isApprox(sigma.transpose(), 1e-10)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_normal_matrix: sigma must be symmetric"));
  }
  return {};
}

sim_expected<Eigen::MatrixXd>
normal_cholesky(const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                double jitter) {
  Eigen::MatrixXd sigma_work = sigma;
  if (jitter > 0.0) {
    sigma_work.diagonal().array() += jitter;
  }
  Eigen::LLT<Eigen::MatrixXd> llt(sigma_work);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(
        SimError::Kind::NonPositiveDefinite,
        "simulate_normal_matrix: sigma is not positive definite"));
  }
  return llt.matrixL();
}

}  // namespace

sim_expected<Eigen::MatrixXd>
simulate_normal_matrix(Eigen::Index n,
                       const Eigen::Ref<const Eigen::VectorXd>& mean,
                       const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                       std::mt19937_64& rng,
                       const NormalOptions& options) {
  if (auto ok = validate_normal_inputs(n, mean, sigma, options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto L_or = normal_cholesky(sigma, options.cholesky_jitter);
  if (!L_or.has_value()) return std::unexpected(L_or.error());

  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd Z(n, mean.size());
  for (Eigen::Index row = 0; row < n; ++row) {
    for (Eigen::Index j = 0; j < mean.size(); ++j) {
      Z(row, j) = normal(rng);
    }
  }

  Eigen::MatrixXd X = Z * L_or->transpose();
  X.rowwise() += mean.transpose();
  return X;
}

sim_expected<data::RawData>
simulate_normal_raw(Eigen::Index n,
                    const Eigen::Ref<const Eigen::VectorXd>& mean,
                    const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                    std::mt19937_64& rng,
                    const NormalOptions& options) {
  auto X_or = simulate_normal_matrix(n, mean, sigma, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

}  // namespace magmaan::sim
