#include <doctest/doctest.h>

#include <Eigen/Core>

#include "magmaan/estimate/weighted_inference.hpp"

TEST_CASE("robust_weighted_moments computes sandwich and U-Gamma for one block") {
  magmaan::estimate::WeightedMomentBlock block;
  block.jacobian.resize(2, 1);
  block.jacobian << 1.0, 0.0;
  block.weight = Eigen::MatrixXd::Identity(2, 2);
  block.gamma = Eigen::MatrixXd::Identity(2, 2);
  block.n_obs = 100;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  auto out = magmaan::estimate::robust_weighted_moments({block}, K, 0.5);
  REQUIRE(out.has_value());
  CHECK(out->df == 1);
  CHECK(out->vcov.rows() == 1);
  CHECK(out->vcov.cols() == 1);
  CHECK(out->vcov(0, 0) == doctest::Approx(0.01));
  CHECK(out->se(0) == doctest::Approx(0.1));
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) == doctest::Approx(1.0));
  CHECK(out->satorra_bentler.scale_c == doctest::Approx(1.0));
  CHECK(out->satorra_bentler.chi2_scaled == doctest::Approx(50.0));
}

TEST_CASE("robust_weighted_moments respects per-block sample weighting and K") {
  magmaan::estimate::WeightedMomentBlock b1;
  b1.jacobian.resize(1, 2);
  b1.jacobian << 1.0, 0.0;
  b1.weight.resize(1, 1);
  b1.weight << 4.0;
  b1.gamma.resize(1, 1);
  b1.gamma << 2.0;
  b1.n_obs = 50;

  magmaan::estimate::WeightedMomentBlock b2;
  b2.jacobian.resize(1, 2);
  b2.jacobian << 0.0, 1.0;
  b2.weight.resize(1, 1);
  b2.weight << 3.0;
  b2.gamma.resize(1, 1);
  b2.gamma << 5.0;
  b2.n_obs = 150;

  Eigen::MatrixXd K(2, 1);
  K << 1.0, 1.0;
  auto out = magmaan::estimate::robust_weighted_moments({b1, b2}, K, 0.25);
  REQUIRE(out.has_value());
  CHECK(out->df == 1);
  REQUIRE(out->vcov.rows() == 2);
  CHECK(out->vcov(0, 0) == doctest::Approx(41.75 / (3.25 * 3.25) / 200.0));
  CHECK(out->vcov(1, 1) == doctest::Approx(out->vcov(0, 0)));
  CHECK(out->vcov(0, 1) == doctest::Approx(out->vcov(0, 0)));
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) >= 0.0);
}
