#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/sim/plsim.hpp"
#include "magmaan/sim/vale_maurelli.hpp"

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

TEST_CASE("PLSIM marginal matches moments where Fleishman fails") {
  magmaan::sim::PlsimOptions options;
  options.hermite_order = 30;
  auto marginal_or = magmaan::sim::fit_plsim_marginal(2.0, 5.0, options);
  if (!marginal_or.has_value()) MESSAGE(marginal_or.error().detail);
  REQUIRE(marginal_or.has_value());

  CHECK(std::abs(marginal_or->mean) < 1e-10);
  CHECK(marginal_or->variance == doctest::Approx(1.0).epsilon(1e-9));
  CHECK(marginal_or->skewness == doctest::Approx(2.0).epsilon(1e-6));
  CHECK(marginal_or->excess_kurtosis == doctest::Approx(5.0).epsilon(1e-6));

  auto fleishman_or = magmaan::sim::fit_fleishman_coefficients(2.0, 5.0);
  REQUIRE_FALSE(fleishman_or.has_value());
}

TEST_CASE("PLSIM Hermite covariance agrees with quadrature for fitted marginals") {
  magmaan::sim::PlsimOptions options;
  options.hermite_order = 36;
  options.quadrature_points = 41;
  auto left_or = magmaan::sim::fit_plsim_marginal(0.75, 1.25, options);
  auto right_or = magmaan::sim::fit_plsim_marginal(-0.35, 0.80, options);
  if (!left_or.has_value()) MESSAGE(left_or.error().detail);
  if (!right_or.has_value()) MESSAGE(right_or.error().detail);
  REQUIRE(left_or.has_value());
  REQUIRE(right_or.has_value());

  auto h_or = magmaan::sim::plsim_covariance_hermite(
      *left_or, *right_or, 0.42, options.hermite_order);
  auto q_or = magmaan::sim::plsim_covariance_quadrature(
      *left_or, *right_or, 0.42, options.quadrature_points);
  REQUIRE(h_or.has_value());
  REQUIRE(q_or.has_value());
  CHECK(*h_or == doctest::Approx(*q_or).epsilon(4e-3));
}

TEST_CASE("PLSIM rectangle covariance agrees with quadrature reference") {
  magmaan::sim::PlsimOptions options;
  options.hermite_order = 36;
  options.quadrature_points = 61;
  auto left_or = magmaan::sim::fit_plsim_marginal(0.75, 1.25, options);
  auto right_or = magmaan::sim::fit_plsim_marginal(-0.35, 0.80, options);
  if (!left_or.has_value()) MESSAGE(left_or.error().detail);
  if (!right_or.has_value()) MESSAGE(right_or.error().detail);
  REQUIRE(left_or.has_value());
  REQUIRE(right_or.has_value());

  auto rect_or = magmaan::sim::plsim_covariance_rectangle(
      *left_or, *right_or, 0.42, options);
  auto quad_or = magmaan::sim::plsim_covariance_quadrature(
      *left_or, *right_or, 0.42, options.quadrature_points);
  REQUIRE(rect_or.has_value());
  REQUIRE(quad_or.has_value());
  CHECK(*rect_or == doctest::Approx(*quad_or).epsilon(2e-3));
}

TEST_CASE("PLSIM calibration solves pairwise covariance with alternative strategies") {
  Eigen::VectorXd skew(2);
  skew << 2.0, 0.75;
  Eigen::VectorXd kurt(2);
  kurt << 5.0, 1.25;
  const Eigen::MatrixXd target = corr2(0.30);

  magmaan::sim::PlsimOptions options;
  options.hermite_order = 30;
  options.quadrature_points = 35;
  options.correlation_tol = 1e-7;

  auto exact_or = magmaan::sim::calibrate_plsim(
      target, skew, kurt, magmaan::sim::PlsimCovarianceMethod::Quadrature, options);
  if (!exact_or.has_value()) MESSAGE(exact_or.error().detail);
  REQUIRE(exact_or.has_value());
  CHECK(exact_or->achieved_corr(0, 1) == doctest::Approx(0.30).epsilon(5e-6));

  auto hermite_or = magmaan::sim::calibrate_plsim(
      target, skew, kurt, magmaan::sim::PlsimCovarianceMethod::Hermite, options);
  if (!hermite_or.has_value()) MESSAGE(hermite_or.error().detail);
  REQUIRE(hermite_or.has_value());

  auto hermite_exact_or = magmaan::sim::calibrate_plsim(
      target, skew, kurt,
      magmaan::sim::PlsimCovarianceMethod::HermiteThenQuadrature, options);
  if (!hermite_exact_or.has_value()) MESSAGE(hermite_exact_or.error().detail);
  REQUIRE(hermite_exact_or.has_value());
  CHECK(hermite_exact_or->achieved_corr(0, 1) ==
        doctest::Approx(exact_or->achieved_corr(0, 1)).epsilon(5e-6));
  CHECK(std::abs(hermite_or->intermediate_corr(0, 1) -
                 exact_or->intermediate_corr(0, 1)) < 0.02);

  auto rect_or = magmaan::sim::calibrate_plsim(
      target, skew, kurt, magmaan::sim::PlsimCovarianceMethod::Rectangle, options);
  if (!rect_or.has_value()) MESSAGE(rect_or.error().detail);
  REQUIRE(rect_or.has_value());
  CHECK(rect_or->achieved_corr(0, 1) == doctest::Approx(0.30).epsilon(5e-6));
}

TEST_CASE("PLSIM simulation respects calibrated population moments") {
  Eigen::VectorXd skew(2);
  skew << 2.0, 0.75;
  Eigen::VectorXd kurt(2);
  kurt << 5.0, 1.25;
  const Eigen::MatrixXd target = corr2(0.25);

  magmaan::sim::PlsimOptions options;
  options.hermite_order = 30;
  options.quadrature_points = 31;

  auto cal_or = magmaan::sim::calibrate_plsim(
      target, skew, kurt,
      magmaan::sim::PlsimCovarianceMethod::HermiteThenQuadrature, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());

  std::mt19937_64 rng(20260604);
  auto X_or = magmaan::sim::simulate_plsim_matrix(90000, *cal_or, rng, options);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());
  const Eigen::MatrixXd& X = *X_or;
  CHECK(std::abs(sample_corr(X, 0, 1) - target(0, 1)) < 0.035);
  CHECK(std::abs(sample_skewness(X, 0) - skew(0)) < 0.12);
  CHECK(std::abs(sample_excess_kurtosis(X, 0) - kurt(0)) < 0.45);
  CHECK(std::abs(sample_skewness(X, 1) - skew(1)) < 0.08);
  CHECK(std::abs(sample_excess_kurtosis(X, 1) - kurt(1)) < 0.18);
}

TEST_CASE("PLSIM raw wrapper returns one complete data block") {
  Eigen::VectorXd skew = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd kurt = Eigen::VectorXd::Zero(2);
  std::mt19937_64 rng(11);
  auto raw_or = magmaan::sim::simulate_plsim_raw(
      30, corr2(0.1), skew, kurt, rng,
      magmaan::sim::PlsimCovarianceMethod::Hermite);
  if (!raw_or.has_value()) MESSAGE(raw_or.error().detail);
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 1u);
  CHECK(raw_or->mask.empty());
  CHECK(raw_or->X[0].rows() == 30);
  CHECK(raw_or->X[0].cols() == 2);
}
