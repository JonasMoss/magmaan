#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fiml.hpp"

namespace {

magmaan::data::RawData complete_single_block() {
  // 12 observations of 3 continuous variables — no missingness, no mask.
  // Deterministic values so the MLE is reproducible.
  Eigen::MatrixXd X(12, 3);
  X << 1.0, 2.0, 3.0,
       2.0, 1.0, 4.0,
       3.0, 4.0, 2.0,
       4.0, 3.0, 5.0,
       5.0, 5.0, 1.0,
       6.0, 4.0, 6.0,
       7.0, 7.0, 4.0,
       8.0, 6.0, 7.0,
       2.5, 3.5, 2.5,
       3.5, 2.5, 3.5,
       6.5, 5.5, 5.5,
       4.5, 4.0, 4.0;
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  // No mask → complete-data path inside h1_moments_block.
  return raw;
}

magmaan::data::RawData missing_single_block() {
  const double na = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(8, 3);
  X << 1.0, 2.0, 3.0,
       2.0, 1.0, 4.0,
       3.0, 4.0, 2.0,
       4.0, 3.0, 5.0,
       5.0, 5.0, 1.0,
       6.0, 4.0, 6.0,
       7.0, 7.0, 4.0,
       8.0, 6.0, 7.0;
  X(2, 1) = na;
  X(5, 2) = na;
  X(6, 0) = na;

  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(8, 3);
  M << 1, 1, 1,
       1, 1, 1,
       1, 0, 1,
       1, 1, 1,
       1, 1, 1,
       1, 1, 0,
       0, 1, 1,
       1, 1, 1;

  magmaan::data::RawData raw;
  raw.X.push_back(X);
  raw.mask.push_back(M);
  return raw;
}

magmaan::data::RawData two_complete_blocks() {
  Eigen::MatrixXd X1(6, 2);
  X1 << 1.0, 2.0,
        2.0, 3.0,
        3.0, 1.0,
        4.0, 4.0,
        5.0, 2.0,
        2.0, 5.0;
  Eigen::MatrixXd X2(7, 2);
  X2 << 0.5, 1.5,
        1.5, 0.5,
        2.5, 2.5,
        1.0, 3.0,
        3.0, 1.0,
        2.0, 2.0,
        1.5, 1.5;
  magmaan::data::RawData raw;
  raw.X.push_back(X1);
  raw.X.push_back(X2);
  return raw;
}

}  // namespace

TEST_CASE("saturated_em_moments: complete data reproduces MLE moments") {
  const auto raw = complete_single_block();
  auto out_or = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE(out_or.has_value());
  const auto& out = *out_or;

  REQUIRE(out.mean.size() == 1u);
  REQUIRE(out.cov.size() == 1u);
  REQUIRE(out.n_obs.size() == 1u);
  CHECK(out.n_obs[0] == 12);

  const Eigen::MatrixXd& X = raw.X[0];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();

  // Closed-form complete-data MLE: sample mean and N-divisor covariance.
  Eigen::VectorXd ref_mu = X.colwise().mean();
  Eigen::MatrixXd Xc = X.rowwise() - ref_mu.transpose();
  Eigen::MatrixXd ref_cov = (Xc.transpose() * Xc) / static_cast<double>(n);

  for (Eigen::Index k = 0; k < p; ++k) {
    CHECK(std::abs(out.mean[0](k) - ref_mu(k)) < 1e-10);
  }
  for (Eigen::Index r = 0; r < p; ++r) {
    for (Eigen::Index c = 0; c < p; ++c) {
      CHECK(std::abs(out.cov[0](r, c) - ref_cov(r, c)) < 1e-10);
    }
  }

  // Block η dimension = p + p*(p+1)/2.
  const Eigen::Index q = p + p * (p + 1) / 2;
  CHECK(out.H.rows() == q);
  CHECK(out.H.cols() == q);
  CHECK(out.J.rows() == q);
  CHECK(out.acov.rows() == q);

  // H is symmetric PD; J is symmetric PSD.
  const Eigen::MatrixXd H_asym = out.H - out.H.transpose();
  CHECK(H_asym.cwiseAbs().maxCoeff() < 1e-8);
  Eigen::LLT<Eigen::MatrixXd> llt_H(0.5 * (out.H + out.H.transpose()));
  CHECK(llt_H.info() == Eigen::Success);

  // ACOV is symmetric (the implementation enforces this) and PSD.
  const Eigen::MatrixXd acov_asym = out.acov - out.acov.transpose();
  CHECK(acov_asym.cwiseAbs().maxCoeff() < 1e-12);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(out.acov);
  REQUIRE(eig.info() == Eigen::Success);
  CHECK(eig.eigenvalues().minCoeff() > -1e-10);
}

TEST_CASE("saturated_em_moments: missingness converges and packages cleanly") {
  const auto raw = missing_single_block();
  auto out_or = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE(out_or.has_value());
  const auto& out = *out_or;

  REQUIRE(out.mean.size() == 1u);
  CHECK(out.n_obs[0] == 8);
  const Eigen::Index p = raw.X[0].cols();
  CHECK(out.mean[0].size() == p);
  CHECK(out.cov[0].rows() == p);
  CHECK(out.cov[0].cols() == p);

  // Saturated covariance must be PD.
  Eigen::LLT<Eigen::MatrixXd> llt_S(0.5 * (out.cov[0] + out.cov[0].transpose()));
  CHECK(llt_S.info() == Eigen::Success);

  // H PD, J PSD, ACOV PSD. Use eigendecomposition so we don't reject due to
  // tiny negative round-off on borderline directions.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_H(out.H);
  REQUIRE(eig_H.info() == Eigen::Success);
  CHECK(eig_H.eigenvalues().minCoeff() > 0.0);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_J(out.J);
  REQUIRE(eig_J.info() == Eigen::Success);
  CHECK(eig_J.eigenvalues().minCoeff() > -1e-10);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_A(out.acov);
  REQUIRE(eig_A.info() == Eigen::Success);
  CHECK(eig_A.eigenvalues().minCoeff() > -1e-10);

  // Sandwich identity: H · acov · H ≡ J (to floating-point round-off).
  const Eigen::MatrixXd reconstructed = out.H * out.acov * out.H;
  CHECK((reconstructed - out.J).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("saturated_em_moments: multi-block H and J are block-diagonal") {
  const auto raw = two_complete_blocks();
  auto out_or = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE(out_or.has_value());
  const auto& out = *out_or;

  REQUIRE(out.mean.size() == 2u);
  REQUIRE(out.cov.size() == 2u);

  const Eigen::Index p1 = raw.X[0].cols();
  const Eigen::Index p2 = raw.X[1].cols();
  const Eigen::Index q1 = p1 + p1 * (p1 + 1) / 2;
  const Eigen::Index q2 = p2 + p2 * (p2 + 1) / 2;
  REQUIRE(out.H.rows() == q1 + q2);

  // Off-diagonal blocks of H and J must be zero: groups are independent.
  const Eigen::MatrixXd H_off_tr = out.H.block(0, q1, q1, q2);
  const Eigen::MatrixXd H_off_bl = out.H.block(q1, 0, q2, q1);
  CHECK(H_off_tr.cwiseAbs().maxCoeff() == 0.0);
  CHECK(H_off_bl.cwiseAbs().maxCoeff() == 0.0);

  const Eigen::MatrixXd J_off_tr = out.J.block(0, q1, q1, q2);
  const Eigen::MatrixXd J_off_bl = out.J.block(q1, 0, q2, q1);
  CHECK(J_off_tr.cwiseAbs().maxCoeff() == 0.0);
  CHECK(J_off_bl.cwiseAbs().maxCoeff() == 0.0);

  // Each block's moments match the complete-data MLE for that block.
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const Eigen::MatrixXd& X = raw.X[b];
    const Eigen::VectorXd ref_mu = X.colwise().mean();
    const Eigen::MatrixXd Xc = X.rowwise() - ref_mu.transpose();
    const Eigen::MatrixXd ref_cov =
        (Xc.transpose() * Xc) / static_cast<double>(X.rows());
    CHECK((out.mean[b] - ref_mu).cwiseAbs().maxCoeff() < 1e-10);
    CHECK((out.cov[b] - ref_cov).cwiseAbs().maxCoeff() < 1e-10);
  }
}
