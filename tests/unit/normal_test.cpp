#include <doctest/doctest.h>

#include <random>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/error.hpp"
#include "magmaan/sim/normal.hpp"

namespace {

Eigen::MatrixXd sample_cov(const Eigen::MatrixXd& X) {
  const Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
  return centered.transpose() * centered / static_cast<double>(X.rows());
}

}  // namespace

TEST_CASE("normal simulation respects requested mean and covariance") {
  Eigen::VectorXd mean(3);
  mean << 1.0, -2.0, 0.5;
  Eigen::MatrixXd sigma(3, 3);
  sigma << 1.0, 0.35, -0.20,
           0.35, 2.25, 0.40,
          -0.20, 0.40, 0.75;

  std::mt19937_64 rng(12345);
  auto X_or = magmaan::sim::simulate_normal_matrix(60000, mean, sigma, rng);
  REQUIRE(X_or.has_value());

  const Eigen::VectorXd got_mean = X_or->colwise().mean();
  const Eigen::MatrixXd got_cov = sample_cov(*X_or);
  for (Eigen::Index j = 0; j < mean.size(); ++j) {
    CHECK(got_mean(j) == doctest::Approx(mean(j)).epsilon(0.01));
  }
  CHECK(got_cov(0, 0) == doctest::Approx(sigma(0, 0)).epsilon(0.025));
  CHECK(got_cov(1, 1) == doctest::Approx(sigma(1, 1)).epsilon(0.025));
  CHECK(got_cov(2, 2) == doctest::Approx(sigma(2, 2)).epsilon(0.025));
  CHECK(got_cov(0, 1) == doctest::Approx(sigma(0, 1)).epsilon(0.08));
  CHECK(got_cov(0, 2) == doctest::Approx(sigma(0, 2)).epsilon(0.08));
  CHECK(got_cov(1, 2) == doctest::Approx(sigma(1, 2)).epsilon(0.08));
}

TEST_CASE("normal raw wrapper returns one complete data block") {
  Eigen::VectorXd mean(2);
  mean << 0.0, 1.0;
  Eigen::MatrixXd sigma(2, 2);
  sigma << 1.0, 0.2,
           0.2, 1.5;

  std::mt19937_64 rng(6789);
  auto raw_or = magmaan::sim::simulate_normal_raw(17, mean, sigma, rng);
  REQUIRE(raw_or.has_value());
  REQUIRE(raw_or->X.size() == 1u);
  CHECK(raw_or->X[0].rows() == 17);
  CHECK(raw_or->X[0].cols() == 2);
  CHECK(raw_or->mask.empty());

  auto stats_or = magmaan::data::sample_stats_from_raw(*raw_or);
  REQUIRE(stats_or.has_value());
  CHECK(stats_or->n_obs[0] == 17);
  CHECK(stats_or->mean[0].size() == 2);
  CHECK(stats_or->S[0].rows() == 2);
}

TEST_CASE("normal simulation validates dimensions and positive definiteness") {
  Eigen::VectorXd mean(2);
  mean << 0.0, 0.0;

  Eigen::MatrixXd wrong_dim(3, 3);
  wrong_dim.setIdentity();
  std::mt19937_64 rng(222);
  auto dim_or = magmaan::sim::simulate_normal_matrix(10, mean, wrong_dim, rng);
  REQUIRE_FALSE(dim_or.has_value());
  CHECK(dim_or.error().kind == magmaan::SimError::Kind::InvalidInput);

  Eigen::MatrixXd indefinite(2, 2);
  indefinite << 1.0, 2.0,
                2.0, 1.0;
  auto pd_or = magmaan::sim::simulate_normal_matrix(10, mean, indefinite, rng);
  REQUIRE_FALSE(pd_or.has_value());
  CHECK(pd_or.error().kind == magmaan::SimError::Kind::NonPositiveDefinite);
}
