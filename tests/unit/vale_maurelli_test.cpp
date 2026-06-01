#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/sim/vale_maurelli.hpp"

namespace {

Eigen::MatrixXd corr2(double r) {
  Eigen::MatrixXd R(2, 2);
  R << 1.0, r,
       r, 1.0;
  return R;
}

Eigen::MatrixXd sample_cov(const Eigen::MatrixXd& X) {
  const Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
  return centered.transpose() * centered / static_cast<double>(X.rows());
}

double sample_corr(const Eigen::MatrixXd& X, Eigen::Index a, Eigen::Index b) {
  const Eigen::VectorXd xa = X.col(a).array() - X.col(a).mean();
  const Eigen::VectorXd xb = X.col(b).array() - X.col(b).mean();
  return xa.dot(xb) / std::sqrt(xa.squaredNorm() * xb.squaredNorm());
}

double sample_skewness(const Eigen::MatrixXd& X, Eigen::Index j) {
  const Eigen::ArrayXd centered = X.col(j).array() - X.col(j).mean();
  const double var = centered.square().mean();
  return centered.pow(3).mean() / std::pow(var, 1.5);
}

double sample_excess_kurtosis(const Eigen::MatrixXd& X, Eigen::Index j) {
  const Eigen::ArrayXd centered = X.col(j).array() - X.col(j).mean();
  const double var = centered.square().mean();
  return centered.pow(4).mean() / (var * var) - 3.0;
}

}  // namespace

TEST_CASE("Fleishman coefficients preserve the normal identity case") {
  auto coef_or = magmaan::sim::fit_fleishman_coefficients(0.0, 0.0);
  REQUIRE(coef_or.has_value());
  CHECK(coef_or->a == doctest::Approx(0.0));
  CHECK(coef_or->b == doctest::Approx(1.0));
  CHECK(coef_or->c == doctest::Approx(0.0));
  CHECK(coef_or->d == doctest::Approx(0.0));
}

TEST_CASE("Vale-Maurelli calibration keeps normal correlations unchanged") {
  Eigen::VectorXd skew = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd kurt = Eigen::VectorXd::Zero(2);
  auto cal_or = magmaan::sim::calibrate_vale_maurelli(corr2(0.42), skew, kurt);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->intermediate_corr(0, 1) == doctest::Approx(0.42).epsilon(1e-10));
  REQUIRE(cal_or->coefficients.size() == 2u);
  CHECK(cal_or->coefficients[0].b == doctest::Approx(1.0));
}

TEST_CASE("Fleishman coefficient solve matches requested moments by simulation") {
  auto coef_or = magmaan::sim::fit_fleishman_coefficients(0.75, 1.25);
  if (!coef_or.has_value()) MESSAGE(coef_or.error().detail);
  REQUIRE(coef_or.has_value());

  std::mt19937_64 rng(20260601);
  Eigen::MatrixXd X(120000, 1);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double z = normal(rng);
    X(i, 0) = coef_or->a + z * (coef_or->b + z * (coef_or->c + z * coef_or->d));
  }
  CHECK(std::abs(X.col(0).mean()) < 0.02);
  CHECK(std::abs(sample_cov(X)(0, 0) - 1.0) < 0.03);
  CHECK(std::abs(sample_skewness(X, 0) - 0.75) < 0.06);
  CHECK(std::abs(sample_excess_kurtosis(X, 0) - 1.25) < 0.14);
}

TEST_CASE("Vale-Maurelli solves the pairwise polynomial covariance equation") {
  Eigen::VectorXd skew(2);
  skew << 0.50, -0.25;
  Eigen::VectorXd kurt(2);
  kurt << 0.75, 0.50;
  const Eigen::MatrixXd target = corr2(0.30);

  auto cal_or = magmaan::sim::calibrate_vale_maurelli(target, skew, kurt);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  auto cov_or = magmaan::sim::fleishman_covariance(
      cal_or->coefficients[0], cal_or->coefficients[1],
      cal_or->intermediate_corr(0, 1));
  REQUIRE(cov_or.has_value());
  CHECK(*cov_or == doctest::Approx(0.30).epsilon(1e-10));
}

TEST_CASE("Fleishman coefficient solve covers Rhemtulla nonnormal target") {
  auto coef_or = magmaan::sim::fit_fleishman_coefficients(2.0, 7.0);
  if (!coef_or.has_value()) MESSAGE(coef_or.error().detail);
  REQUIRE(coef_or.has_value());

  std::mt19937_64 rng(20260603);
  Eigen::MatrixXd X(160000, 1);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double z = normal(rng);
    X(i, 0) = coef_or->a + z * (coef_or->b + z * (coef_or->c + z * coef_or->d));
  }
  CHECK(std::abs(sample_skewness(X, 0) - 2.0) < 0.14);
  CHECK(std::abs(sample_excess_kurtosis(X, 0) - 7.0) < 0.55);
}

TEST_CASE("Vale-Maurelli simulation respects target moments and correlations") {
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.30, -0.20,
            0.30, 1.0, 0.15,
           -0.20, 0.15, 1.0;
  Eigen::VectorXd skew(3);
  skew << 0.50, -0.35, 0.0;
  Eigen::VectorXd kurt(3);
  kurt << 0.80, 0.60, 1.0;

  std::mt19937_64 rng(20260602);
  auto X_or = magmaan::sim::simulate_vale_maurelli_matrix(
      80000, target, skew, kurt, rng);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());
  const auto& X = *X_or;
  REQUIRE(X.rows() == 80000);
  REQUIRE(X.cols() == 3);

  CHECK(std::abs(sample_corr(X, 0, 1) - target(0, 1)) < 0.035);
  CHECK(std::abs(sample_corr(X, 0, 2) - target(0, 2)) < 0.035);
  CHECK(std::abs(sample_corr(X, 1, 2) - target(1, 2)) < 0.035);
  for (Eigen::Index j = 0; j < 3; ++j) {
    CHECK(std::abs(sample_skewness(X, j) - skew(j)) < 0.08);
    CHECK(std::abs(sample_excess_kurtosis(X, j) - kurt(j)) < 0.18);
  }
}

TEST_CASE("Vale-Maurelli raw wrapper returns one complete data block") {
  Eigen::VectorXd skew = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd kurt = Eigen::VectorXd::Zero(2);
  std::mt19937_64 rng(7);
  auto raw_or = magmaan::sim::simulate_vale_maurelli_raw(
      30, corr2(0.1), skew, kurt, rng);
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 1u);
  CHECK(raw_or->mask.empty());
  CHECK(raw_or->X[0].rows() == 30);
  CHECK(raw_or->X[0].cols() == 2);
}

TEST_CASE("Vale-Maurelli rejects infeasible marginal moments") {
  auto coef_or = magmaan::sim::fit_fleishman_coefficients(2.0, 0.0);
  REQUIRE_FALSE(coef_or.has_value());
  CHECK(coef_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}
