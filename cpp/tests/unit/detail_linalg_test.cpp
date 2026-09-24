#include <doctest/doctest.h>

#include <limits>

#include <Eigen/Core>
#include <Eigen/QR>

// Private src/ helper; included by relative path because src/ is PRIVATE to the
// magmaan target. The definition is compiled into libmagmaan (detail_linalg.cpp)
// and resolved at link time.
#include "../../src/detail_linalg.hpp"

using magmaan::detail::symmetric_inverse_pd_gated;

TEST_CASE("symmetric_inverse_pd_gated: inverts a well-conditioned SPD matrix") {
  Eigen::MatrixXd A(3, 3);
  A << 4, 1, 0,
       1, 3, 1,
       0, 1, 2;
  const auto r = symmetric_inverse_pd_gated(A);
  REQUIRE(r.ok);
  CHECK(r.finite);
  CHECK(r.decomposed);
  CHECK(r.dim == 3);
  CHECK(r.rank == 3);
  CHECK(r.min_eval > 0.0);
  CHECK(r.rcond > 0.0);
  // inverse ≈ A⁻¹.
  const Eigen::MatrixXd should_be_I = A * r.inverse;
  CHECK((should_be_I - Eigen::MatrixXd::Identity(3, 3)).cwiseAbs().maxCoeff() <
        1e-12);
}

TEST_CASE("symmetric_inverse_pd_gated: rejects a rank-deficient spectrum "
          "(muthen-like)") {
  // Q·diag(10, 1, 1e-18)·Qᵀ — one structurally-zero eigendirection, the same
  // pathology as the muthen_2017_ch2_ex2_1__adf Γ̂ (rcond ~5e-18). The barely-
  // positive eigenvalue passes a plain Cholesky but must be rejected here.
  Eigen::MatrixXd M(3, 3);
  M << 1, 2, 3,
       4, 5, 6,
       7, 8, 10;  // nonsingular ⇒ full-rank orthonormal Q
  const Eigen::MatrixXd Q =
      Eigen::HouseholderQR<Eigen::MatrixXd>(M).householderQ();
  Eigen::VectorXd d(3);
  d << 10.0, 1.0, 1e-18;
  const Eigen::MatrixXd A = Q * d.asDiagonal() * Q.transpose();

  const auto r = symmetric_inverse_pd_gated(A);
  CHECK_FALSE(r.ok);
  CHECK(r.finite);
  CHECK(r.decomposed);
  CHECK(r.dim == 3);
  CHECK(r.rank == 2);                 // the 1e-18 direction is dropped
  CHECK(r.inverse.size() == 0);       // inverse not populated when !ok
  CHECK(r.rcond < 1e-10);
  CHECK(r.tol == doctest::Approx(1e-10 * 10.0));
  CHECK(r.min_eval == doctest::Approx(1e-18).epsilon(0.5));
  CHECK(r.max_eval == doctest::Approx(10.0));
}

TEST_CASE("symmetric_inverse_pd_gated: rejects non-square and non-finite") {
  Eigen::MatrixXd non_square(2, 3);
  non_square.setZero();
  const auto r1 = symmetric_inverse_pd_gated(non_square);
  CHECK_FALSE(r1.ok);
  CHECK_FALSE(r1.finite);

  Eigen::MatrixXd has_nan(2, 2);
  has_nan.setIdentity();
  has_nan(0, 0) = std::numeric_limits<double>::quiet_NaN();
  const auto r2 = symmetric_inverse_pd_gated(has_nan);
  CHECK_FALSE(r2.ok);
  CHECK_FALSE(r2.finite);
}
