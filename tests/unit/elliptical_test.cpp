#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/sim/elliptical.hpp"

namespace {

Eigen::MatrixXd sample_cov(const Eigen::MatrixXd& X) {
  const Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
  return centered.transpose() * centered / static_cast<double>(X.rows());
}

void check_mean_covariance(const Eigen::MatrixXd& X,
                           const Eigen::VectorXd& mean,
                           const Eigen::MatrixXd& covariance,
                           double mean_tol,
                           double covariance_tol) {
  const Eigen::VectorXd got_mean = X.colwise().mean();
  const Eigen::MatrixXd got_cov = sample_cov(X);
  REQUIRE(got_mean.size() == mean.size());
  REQUIRE(got_cov.rows() == covariance.rows());
  REQUIRE(got_cov.cols() == covariance.cols());
  for (Eigen::Index j = 0; j < mean.size(); ++j) {
    CHECK(std::abs(got_mean(j) - mean(j)) < mean_tol);
  }
  for (Eigen::Index r = 0; r < covariance.rows(); ++r) {
    for (Eigen::Index c = 0; c < covariance.cols(); ++c) {
      CHECK(std::abs(got_cov(r, c) - covariance(r, c)) < covariance_tol);
    }
  }
}

}  // namespace

TEST_CASE("elliptical moment diagnostics pin radial fourth moments") {
  auto t_or = magmaan::sim::student_t_moment_diagnostics(8.0);
  REQUIRE(t_or.has_value());
  CHECK(t_or->scale_second_moment == doctest::Approx(4.0 / 3.0));
  CHECK(t_or->scale_fourth_moment == doctest::Approx(8.0 / 3.0));
  CHECK(t_or->normal_covariance_multiplier == doctest::Approx(0.75));
  CHECK(t_or->radial_kurtosis_factor == doctest::Approx(1.5));
  CHECK(t_or->marginal_excess_kurtosis == doctest::Approx(1.5));
  CHECK(t_or->finite_fourth_moment);

  magmaan::sim::ContaminatedNormalSpec contamination;
  contamination.contamination_probability = 0.10;
  contamination.scale_multiplier = 4.0;
  auto contam_or =
      magmaan::sim::contaminated_normal_moment_diagnostics(contamination);
  REQUIRE(contam_or.has_value());
  CHECK(contam_or->scale_second_moment == doctest::Approx(2.5));
  CHECK(contam_or->scale_fourth_moment == doctest::Approx(26.5));
  CHECK(contam_or->normal_covariance_multiplier == doctest::Approx(0.4));
  CHECK(contam_or->radial_kurtosis_factor == doctest::Approx(4.24));
  CHECK(contam_or->marginal_excess_kurtosis == doctest::Approx(9.72));
  CHECK(contam_or->finite_fourth_moment);

  auto slash_or = magmaan::sim::slash_moment_diagnostics({.q = 6.0});
  REQUIRE(slash_or.has_value());
  CHECK(slash_or->scale_second_moment == doctest::Approx(1.5));
  CHECK(slash_or->scale_fourth_moment == doctest::Approx(3.0));
  CHECK(slash_or->normal_covariance_multiplier == doctest::Approx(2.0 / 3.0));
  CHECK(slash_or->radial_kurtosis_factor == doctest::Approx(4.0 / 3.0));
  CHECK(slash_or->marginal_excess_kurtosis == doctest::Approx(1.0));
  CHECK(slash_or->finite_fourth_moment);

  magmaan::sim::NormalScaleMixtureSpec mixture;
  mixture.weights = {2.0, 3.0, 5.0};
  mixture.scale_multipliers = {0.5, 1.0, 2.0};
  auto mix_or = magmaan::sim::scale_mixture_normal_moment_diagnostics(mixture);
  REQUIRE(mix_or.has_value());
  CHECK(mix_or->scale_second_moment == doctest::Approx(2.35));
  CHECK(mix_or->scale_fourth_moment == doctest::Approx(8.3125));
  CHECK(mix_or->normal_covariance_multiplier ==
        doctest::Approx(1.0 / 2.35));
  CHECK(mix_or->radial_kurtosis_factor ==
        doctest::Approx(8.3125 / (2.35 * 2.35)));
  CHECK(mix_or->marginal_excess_kurtosis ==
        doctest::Approx(3.0 * (8.3125 / (2.35 * 2.35) - 1.0)));
  CHECK(mix_or->finite_fourth_moment);
}

TEST_CASE("elliptical diagnostics report infinite fourth moments") {
  auto t_or = magmaan::sim::student_t_moment_diagnostics(3.5);
  REQUIRE(t_or.has_value());
  CHECK(t_or->finite_fourth_moment == false);
  CHECK(std::isinf(t_or->scale_fourth_moment));
  CHECK(std::isinf(t_or->marginal_excess_kurtosis));

  auto slash_or = magmaan::sim::slash_moment_diagnostics({.q = 3.5});
  REQUIRE(slash_or.has_value());
  CHECK(slash_or->finite_fourth_moment == false);
  CHECK(std::isinf(slash_or->scale_fourth_moment));
  CHECK(std::isinf(slash_or->marginal_excess_kurtosis));
}

TEST_CASE("elliptical diagnostics validate inputs") {
  auto bad_t = magmaan::sim::student_t_moment_diagnostics(2.0);
  REQUIRE_FALSE(bad_t.has_value());
  CHECK(bad_t.error().kind == magmaan::SimError::Kind::InvalidInput);

  magmaan::sim::NormalScaleMixtureSpec bad_mixture;
  bad_mixture.weights = {0.5, 0.5};
  bad_mixture.scale_multipliers = {1.0};
  auto bad_mix =
      magmaan::sim::scale_mixture_normal_moment_diagnostics(bad_mixture);
  REQUIRE_FALSE(bad_mix.has_value());
  CHECK(bad_mix.error().kind == magmaan::SimError::Kind::InvalidInput);

  auto bad_slash = magmaan::sim::slash_moment_diagnostics({.q = 2.0});
  REQUIRE_FALSE(bad_slash.has_value());
  CHECK(bad_slash.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("elliptical simulations preserve requested covariance") {
  Eigen::VectorXd mean(2);
  mean << 0.75, -1.25;
  Eigen::MatrixXd covariance(2, 2);
  covariance << 1.0, 0.30,
                0.30, 1.80;

  std::mt19937_64 t_rng(2026061401);
  auto t_or = magmaan::sim::simulate_student_t_matrix(
      70000, mean, covariance, 8.0, t_rng);
  REQUIRE(t_or.has_value());
  check_mean_covariance(*t_or, mean, covariance, 0.035, 0.075);

  magmaan::sim::ContaminatedNormalSpec contamination;
  contamination.contamination_probability = 0.10;
  contamination.scale_multiplier = 4.0;
  std::mt19937_64 contam_rng(2026061402);
  auto contam_or = magmaan::sim::simulate_contaminated_normal_matrix(
      70000, mean, covariance, contamination, contam_rng);
  REQUIRE(contam_or.has_value());
  check_mean_covariance(*contam_or, mean, covariance, 0.045, 0.090);

  std::mt19937_64 slash_rng(2026061403);
  auto slash_or = magmaan::sim::simulate_slash_matrix(
      70000, mean, covariance, {.q = 6.0}, slash_rng);
  REQUIRE(slash_or.has_value());
  check_mean_covariance(*slash_or, mean, covariance, 0.035, 0.075);

  magmaan::sim::NormalScaleMixtureSpec mixture;
  mixture.weights = {0.85, 0.15};
  mixture.scale_multipliers = {0.75, 2.50};
  std::mt19937_64 mix_rng(2026061404);
  auto mix_or = magmaan::sim::simulate_scale_mixture_normal_matrix(
      70000, mean, covariance, mixture, mix_rng);
  REQUIRE(mix_or.has_value());
  check_mean_covariance(*mix_or, mean, covariance, 0.035, 0.075);
}
