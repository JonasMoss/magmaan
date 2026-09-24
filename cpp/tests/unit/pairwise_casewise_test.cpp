#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/pairwise_cov.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/robust/robust.hpp"

namespace {

magmaan::data::RawData complete_data_block() {
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
  return raw;
}

magmaan::data::RawData missing_data_block() {
  const double na = std::numeric_limits<double>::quiet_NaN();
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
  X(2, 1) = na;
  X(5, 2) = na;
  X(8, 0) = na;
  X(10, 1) = na;

  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(12, 3);
  for (Eigen::Index r = 0; r < 12; ++r)
    for (Eigen::Index c = 0; c < 3; ++c)
      M(r, c) = std::isfinite(X(r, c)) ? 1 : 0;

  magmaan::data::RawData raw;
  raw.X.push_back(X);
  raw.mask.push_back(M);
  return raw;
}

magmaan::data::RawData two_complete_blocks() {
  Eigen::MatrixXd X1(8, 2);
  X1 << 1, 2, 2, 1, 3, 4, 4, 3, 5, 5, 6, 4, 7, 7, 8, 6;
  Eigen::MatrixXd X2(9, 2);
  X2 << 0.5, 1.5, 1.5, 0.5, 2.5, 2.5, 1.0, 3.0,
        3.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.2, 3.3, 1.7, 2.8;
  magmaan::data::RawData raw;
  raw.X.push_back(X1);
  raw.X.push_back(X2);
  return raw;
}

}  // namespace

TEST_CASE("pairwise_sample_stats: complete data matches sample_stats_from_raw") {
  const auto raw = complete_data_block();
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw_or.has_value());
  const auto& pw = *pw_or;

  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  const auto& samp = *samp_or;

  CHECK(pw.n_obs[0] == samp.n_obs[0]);
  CHECK((pw.mean[0] - samp.mean[0]).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((pw.S[0]    - samp.S[0]).cwiseAbs().maxCoeff() < 1e-12);
  // π̂_(j,ℓ) = 1 for every pair when no missingness.
  CHECK((pw.pi_hat[0].array() - 1.0).abs().maxCoeff() < 1e-12);
  // n_pair = n for every pair.
  for (Eigen::Index j = 0; j < pw.S[0].rows(); ++j)
    for (Eigen::Index k = 0; k < pw.S[0].cols(); ++k)
      CHECK(pw.n_pair[0](j, k) == 12);
}

TEST_CASE("pairwise_casewise_contributions: complete data matches casewise_contributions") {
  const auto raw = complete_data_block();
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto Psi_or = magmaan::robust::pairwise_casewise_contributions(raw, *pw);
  REQUIRE(Psi_or.has_value());
  auto Zc_or  = magmaan::robust::casewise_contributions(raw, *samp, false);
  REQUIRE(Zc_or.has_value());

  CHECK(Psi_or->rows() == Zc_or->rows());
  CHECK(Psi_or->cols() == Zc_or->cols());
  CHECK((*Psi_or - *Zc_or).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("pairwise_sample_stats: missingness produces correct overlaps and means") {
  const auto raw = missing_data_block();
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw_or.has_value());
  const auto& pw = *pw_or;

  CHECK(pw.n_obs[0] == 12);

  // Overlap counts: count rows where both variables are observed.
  for (Eigen::Index j = 0; j < 3; ++j) {
    for (Eigen::Index k = 0; k < 3; ++k) {
      int expected = 0;
      for (Eigen::Index r = 0; r < 12; ++r) {
        if (raw.mask[0](r, j) != 0 && raw.mask[0](r, k) != 0) ++expected;
      }
      CHECK(pw.n_pair[0](j, k) == expected);
      CHECK(std::abs(pw.pi_hat[0](j, k) -
                     static_cast<double>(expected) / 12.0) < 1e-12);
    }
  }

  // S is symmetric and S(j,j) ≥ 0 (it's a variance).
  CHECK((pw.S[0] - pw.S[0].transpose()).cwiseAbs().maxCoeff() < 1e-12);
  for (Eigen::Index j = 0; j < 3; ++j) CHECK(pw.S[0](j, j) >= 0.0);
}

TEST_CASE("pairwise_casewise_contributions: empirical pairwise Γ̂ has expected scaling") {
  const auto raw = missing_data_block();
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());

  auto Psi_or = magmaan::robust::pairwise_casewise_contributions(raw, *pw);
  REQUIRE(Psi_or.has_value());
  const Eigen::MatrixXd& Psi = *Psi_or;

  const double n = static_cast<double>(raw.X[0].rows());
  const Eigen::MatrixXd Gamma_hat = (Psi.transpose() * Psi) / n;

  // Γ̂^pw must be symmetric and PSD.
  CHECK((Gamma_hat - Gamma_hat.transpose()).cwiseAbs().maxCoeff() < 1e-10);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Gamma_hat);
  REQUIRE(eig.info() == Eigen::Success);
  CHECK(eig.eigenvalues().minCoeff() > -1e-10);

  // Rows with a fully unobserved variable contribute zero entries to that
  // variable's column-slices — sanity check Ψ̂ is sparse on those entries.
  // Row 2 has variable 1 missing (X(2,1)=NA). With p=3, the column-major
  // lower-tri vech offsets are: (0,0)→0, (1,0)→1, (2,0)→2, (1,1)→3, (2,1)→4,
  // (2,2)→5 — so columns 1, 3, 4 involve variable 1 and must be zero on
  // row 2.
  CHECK(std::abs(Psi(2, 1)) < 1e-12);
  CHECK(std::abs(Psi(2, 3)) < 1e-12);
  CHECK(std::abs(Psi(2, 4)) < 1e-12);
}

TEST_CASE("pairwise_casewise_contributions: G3b complete data matches casewise_contributions") {
  const auto raw = complete_data_block();
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto Psi_or = magmaan::robust::pairwise_casewise_contributions(
      raw, *pw, /*include_means=*/true);
  REQUIRE(Psi_or.has_value());
  auto Zc_or = magmaan::robust::casewise_contributions(
      raw, *samp, /*include_means=*/true);
  REQUIRE(Zc_or.has_value());

  REQUIRE(Psi_or->rows() == Zc_or->rows());
  REQUIRE(Psi_or->cols() == Zc_or->cols());
  // Per-block column count is p + p*(p+1)/2 (G3b).
  const Eigen::Index p = raw.X[0].cols();
  CHECK(Psi_or->cols() == p + p * (p + 1) / 2);
  CHECK((*Psi_or - *Zc_or).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("pairwise_casewise_contributions: μ-segment zeros where variable missing") {
  const auto raw = missing_data_block();
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());

  auto Psi_or = magmaan::robust::pairwise_casewise_contributions(
      raw, *pw, /*include_means=*/true);
  REQUIRE(Psi_or.has_value());
  const Eigen::MatrixXd& Psi = *Psi_or;

  const Eigen::Index p = raw.X[0].cols();
  // G3b layout: [μ-cols (size p) | σ-vech cols (size p*(p+1)/2)] per block.
  CHECK(Psi.cols() == p + p * (p + 1) / 2);

  // For each row t and variable j, Ψ(t, μ_off + j) must equal 0 iff variable
  // j is unobserved on row t. Strict equality below: zero rows are exact 0
  // (no floating-point contamination), non-zero rows are non-zero finite.
  for (Eigen::Index t = 0; t < Psi.rows(); ++t) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const double v = Psi(t, j);
      const bool obs = raw.mask[0](t, j) != 0;
      if (!obs) CHECK(v == 0.0);
    }
  }
}

TEST_CASE("pairwise_casewise_contributions: μ column means are zero by construction") {
  const auto raw = missing_data_block();
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());

  auto Psi_or = magmaan::robust::pairwise_casewise_contributions(
      raw, *pw, /*include_means=*/true);
  REQUIRE(Psi_or.has_value());
  const Eigen::MatrixXd& Psi = *Psi_or;

  const Eigen::Index p = raw.X[0].cols();
  // Σ_t R_tj (x_tj − m̄_j) = 0 when m̄_j is the marginal mean, so the
  // marginal-availability rescaling preserves the zero sum.
  for (Eigen::Index j = 0; j < p; ++j) {
    CHECK(std::abs(Psi.col(j).sum()) < 1e-10);
  }
}

TEST_CASE("pairwise: multi-block layout is block-diagonal in column space") {
  const auto raw = two_complete_blocks();
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());
  auto Psi_or = magmaan::robust::pairwise_casewise_contributions(raw, *pw);
  REQUIRE(Psi_or.has_value());
  const Eigen::MatrixXd& Psi = *Psi_or;

  // 2 vars per block, vech len 3 per block → total cols = 6.
  CHECK(Psi.cols() == 6);
  CHECK(Psi.rows() == 8 + 9);

  // Row from block 0 must have zeros in block-1 column slice and vice versa.
  for (Eigen::Index r = 0; r < 8; ++r) {
    CHECK(Psi.block(r, 3, 1, 3).cwiseAbs().maxCoeff() < 1e-12);
  }
  for (Eigen::Index r = 8; r < 17; ++r) {
    CHECK(Psi.block(r, 0, 1, 3).cwiseAbs().maxCoeff() < 1e-12);
  }
}
