#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/sim/norta.hpp"

namespace {

Eigen::MatrixXd corr2(double r) {
  Eigen::MatrixXd R(2, 2);
  R << 1.0, r,
       r, 1.0;
  return R;
}

double sample_corr(const Eigen::MatrixXd& X, Eigen::Index a, Eigen::Index b) {
  const Eigen::VectorXd xa = X.col(a).array() - X.col(a).mean();
  const Eigen::VectorXd xb = X.col(b).array() - X.col(b).mean();
  return xa.dot(xb) / std::sqrt(xa.squaredNorm() * xb.squaredNorm());
}

}  // namespace

TEST_CASE("normal quantile round-trips through normal CDF") {
  for (double u : {0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999}) {
    auto z_or = magmaan::sim::normal_quantile(u);
    REQUIRE(z_or.has_value());
    CHECK(magmaan::sim::normal_cdf(*z_or) == doctest::Approx(u).epsilon(1e-13));
  }
}

TEST_CASE("NORTA calibration keeps normal marginals unchanged") {
  const Eigen::MatrixXd target = corr2(0.42);
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};

  auto cal_or = magmaan::sim::calibrate_norta(target, marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) == doctest::Approx(0.42).epsilon(1e-8));
  CHECK(cal_or->latent_corr(1, 0) == doctest::Approx(0.42).epsilon(1e-8));
}

TEST_CASE("NORTA calibration matches analytic lognormal copula correlation") {
  const double sigma = 0.8;
  const double target_r = 0.35;
  const double s2 = sigma * sigma;
  const double latent_r = std::log(1.0 + target_r * (std::exp(s2) - 1.0)) / s2;

  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(sigma),
      magmaan::sim::MarginalSpec::standardized_lognormal(sigma)};

  auto cal_or = magmaan::sim::calibrate_norta(corr2(target_r), marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) == doctest::Approx(latent_r).epsilon(3e-6));
}

TEST_CASE("NORTA reports infeasible lognormal target correlations") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(0.8),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.8)};

  auto cal_or = magmaan::sim::calibrate_norta(corr2(-0.8), marginals);
  REQUIRE_FALSE(cal_or.has_value());
  CHECK(cal_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("independent generator supports Tukey g-and-h marginals") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::tukey_g_h(0.35, 0.05, 1.0, 2.0),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.4, -2.0, 1.5),
      magmaan::sim::MarginalSpec::standard_normal()};

  std::mt19937_64 rng(20260601);
  magmaan::sim::IndependentOptions options;
  options.quadrature_points = 35;
  auto X_or = magmaan::sim::simulate_independent_matrix(30000, marginals, rng, options);
  REQUIRE(X_or.has_value());
  const auto& X = *X_or;
  REQUIRE(X.rows() == 30000);
  REQUIRE(X.cols() == 3);

  CHECK(std::abs(X.col(0).mean() - 1.0) < 0.08);
  CHECK(std::abs(X.col(1).mean() + 2.0) < 0.05);
  CHECK(std::abs(X.col(2).mean()) < 0.03);
  CHECK(std::abs(sample_corr(X, 0, 1)) < 0.03);
  CHECK(std::abs(sample_corr(X, 0, 2)) < 0.03);
  CHECK(std::abs(sample_corr(X, 1, 2)) < 0.03);
}

TEST_CASE("independent raw generator wraps one complete data block") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::tukey_g_h(0.0, 0.1)};

  std::mt19937_64 rng(11);
  auto raw_or = magmaan::sim::simulate_independent_raw(25, marginals, rng);
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 1u);
  CHECK(raw_or->mask.empty());
  CHECK(raw_or->X[0].rows() == 25);
  CHECK(raw_or->X[0].cols() == 2);
}

TEST_CASE("Tukey g-and-h moment matcher recovers Kowalchuk-Headrick shape") {
  magmaan::sim::MomentMatchSpec spec;
  spec.family = magmaan::sim::MomentMatchFamily::TukeyGH;
  spec.shape.skewness = 3.0;
  spec.shape.excess_kurtosis = 20.0;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto fit_or = magmaan::sim::fit_marginal_to_moments(spec, options);
  REQUIRE(fit_or.has_value());
  if (!fit_or.has_value()) return;

  CHECK(fit_or->marginal.kind == magmaan::sim::MarginalKind::TukeyGH);
  CHECK(fit_or->marginal.g == doctest::Approx(0.690678).epsilon(2e-3));
  CHECK(fit_or->marginal.h == doctest::Approx(0.009831).epsilon(3e-2));
  CHECK(fit_or->moments.skewness == doctest::Approx(3.0).epsilon(2e-6));
  CHECK(fit_or->moments.excess_kurtosis == doctest::Approx(20.0).epsilon(2e-6));
}

TEST_CASE("moment-matched Tukey marginals feed NORTA calibration") {
  magmaan::sim::MomentMatchSpec left;
  left.shape.skewness = 3.0;
  left.shape.excess_kurtosis = 20.0;
  magmaan::sim::MomentMatchSpec right;
  right.shape.skewness = -3.0;
  right.shape.excess_kurtosis = 20.0;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto left_or = magmaan::sim::fit_marginal_to_moments(left, options);
  auto right_or = magmaan::sim::fit_marginal_to_moments(right, options);
  REQUIRE(left_or.has_value());
  REQUIRE(right_or.has_value());
  if (!left_or.has_value() || !right_or.has_value()) return;
  CHECK(left_or->marginal.g > 0.0);
  CHECK(right_or->marginal.g < 0.0);

  const std::vector<magmaan::sim::MarginalSpec> marginals{
      left_or->marginal, right_or->marginal};
  auto cal_or = magmaan::sim::calibrate_norta(corr2(0.30), marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) > 0.30);
}

TEST_CASE("moment matcher reserves unsupported family slots") {
  magmaan::sim::MomentMatchSpec spec;
  spec.family = magmaan::sim::MomentMatchFamily::Pearson;
  spec.shape.excess_kurtosis = 1.0;
  auto fit_or = magmaan::sim::fit_marginal_to_moments(spec);
  REQUIRE_FALSE(fit_or.has_value());
  CHECK(fit_or.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("NORTA simulation respects target moments and correlations") {
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.30, -0.20,
            0.30, 1.0, 0.15,
           -0.20, 0.15, 1.0;
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(2.0, 1.5),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.5, -1.0, 2.0),
      magmaan::sim::MarginalSpec::tukey_g_h(0.2, 0.08)};

  std::mt19937_64 rng(20260531);
  magmaan::sim::NortaOptions options;
  options.quadrature_points = 35;
  auto X_or = magmaan::sim::simulate_norta_matrix(20000, target, marginals, rng, options);
  REQUIRE(X_or.has_value());
  const auto& X = *X_or;
  REQUIRE(X.rows() == 20000);
  REQUIRE(X.cols() == 3);

  CHECK(std::abs(X.col(0).mean() - 2.0) < 0.06);
  CHECK(std::abs(X.col(1).mean() + 1.0) < 0.08);
  CHECK(std::abs(X.col(2).mean()) < 0.05);
  CHECK(std::abs(sample_corr(X, 0, 1) - target(0, 1)) < 0.04);
  CHECK(std::abs(sample_corr(X, 0, 2) - target(0, 2)) < 0.04);
  CHECK(std::abs(sample_corr(X, 1, 2) - target(1, 2)) < 0.04);
}
