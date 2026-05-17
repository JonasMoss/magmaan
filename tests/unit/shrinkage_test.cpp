#include <doctest/doctest.h>

#include <cstdint>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/data/shrinkage.hpp"

using magmaan::data::CovarianceShrinkageKind;
using magmaan::data::CovarianceShrinkageOptions;
using magmaan::data::SampleStats;
using magmaan::data::shrink_sample_stats;

namespace {

SampleStats one_block(const Eigen::MatrixXd& S, std::int64_t n = 100) {
  SampleStats s;
  s.S.push_back(S);
  s.n_obs.push_back(n);
  return s;
}

double max_abs(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace

// ============================================================================
// Identity behaviour: s = 0 and kind = None leave S untouched.
// ============================================================================

TEST_CASE("shrink_sample_stats: kind=None and s=0 leave S unchanged") {
  Eigen::Matrix3d S;
  S << 1.0, 0.4, 0.6,
       0.4, 1.0, 0.8,
       0.6, 0.8, 1.0;
  auto samp = one_block(S);

  auto none = shrink_sample_stats(samp, {CovarianceShrinkageKind::None, 0.7});
  REQUIRE(none.has_value());
  CHECK(max_abs(none->stats.S[0], S) == 0.0);
  CHECK_FALSE(none->block_diagnostics[0].shrunk);
  CHECK(none->block_diagnostics[0].intensity == 0.0);

  auto zero =
      shrink_sample_stats(samp, {CovarianceShrinkageKind::DiagonalTarget, 0.0});
  REQUIRE(zero.has_value());
  CHECK(max_abs(zero->stats.S[0], S) == 0.0);
  CHECK_FALSE(zero->block_diagnostics[0].shrunk);
}

// ============================================================================
// s = 1 reaches the target exactly.
// ============================================================================

TEST_CASE("shrink_sample_stats: s=1 lands exactly on the target") {
  Eigen::Matrix3d S;
  S << 2.0, 0.5, 0.3,
       0.5, 1.0, 0.4,
       0.3, 0.4, 0.5;
  auto samp = one_block(S);

  SUBCASE("DiagonalTarget -> diag(S)") {
    auto r =
        shrink_sample_stats(samp, {CovarianceShrinkageKind::DiagonalTarget, 1.0});
    REQUIRE(r.has_value());
    Eigen::Matrix3d T = S.diagonal().asDiagonal();
    CHECK(max_abs(r->stats.S[0], T) < 1e-12);
  }
  SUBCASE("IdentityTarget -> (tr S / p) I") {
    auto r =
        shrink_sample_stats(samp, {CovarianceShrinkageKind::IdentityTarget, 1.0});
    REQUIRE(r.has_value());
    const double mu = S.trace() / 3.0;
    Eigen::Matrix3d T = mu * Eigen::Matrix3d::Identity();
    CHECK(max_abs(r->stats.S[0], T) < 1e-12);
  }
  SUBCASE("Ridge -> I") {
    auto r = shrink_sample_stats(samp, {CovarianceShrinkageKind::Ridge, 1.0});
    REQUIRE(r.has_value());
    CHECK(max_abs(r->stats.S[0], Eigen::Matrix3d::Identity()) < 1e-12);
  }
  SUBCASE("ConstantCorrelation -> equal off-diagonal correlations") {
    // All variances 1, correlations 0.4/0.6/0.8 ⇒ average correlation 0.6.
    Eigen::Matrix3d C;
    C << 1.0, 0.4, 0.6,
         0.4, 1.0, 0.8,
         0.6, 0.8, 1.0;
    auto r = shrink_sample_stats(one_block(C),
                                 {CovarianceShrinkageKind::ConstantCorrelation,
                                  1.0});
    REQUIRE(r.has_value());
    Eigen::Matrix3d T;
    T << 1.0, 0.6, 0.6,
         0.6, 1.0, 0.6,
         0.6, 0.6, 1.0;
    CHECK(max_abs(r->stats.S[0], T) < 1e-12);
  }
}

// ============================================================================
// Conditioning: shrinkage lifts the smallest eigenvalue.
// ============================================================================

TEST_CASE("shrink_sample_stats: shrinkage raises the minimum eigenvalue") {
  // Near-singular 2x2 — minimum eigenvalue 0.01.
  Eigen::Matrix2d S;
  S << 1.0, 0.99,
       0.99, 1.0;
  auto r = shrink_sample_stats(one_block(S),
                               {CovarianceShrinkageKind::DiagonalTarget, 0.5});
  REQUIRE(r.has_value());
  const auto& d = r->block_diagnostics[0];
  CHECK(d.raw_min_eigen == doctest::Approx(0.01));
  CHECK(d.min_eigen == doctest::Approx(0.505));   // (1-0.5)*0.01 + 0.5*1.0
  CHECK(d.min_eigen > d.raw_min_eigen);
  CHECK(d.shrunk);
  CHECK(d.intensity == doctest::Approx(0.5));
}

// ============================================================================
// Intensity is clamped to [0, 1].
// ============================================================================

TEST_CASE("shrink_sample_stats: intensity is clamped to [0,1]") {
  Eigen::Matrix2d S;
  S << 1.0, 0.5,
       0.5, 1.0;
  auto samp = one_block(S);

  auto over =
      shrink_sample_stats(samp, {CovarianceShrinkageKind::DiagonalTarget, 1.5});
  REQUIRE(over.has_value());
  CHECK(over->block_diagnostics[0].intensity == doctest::Approx(1.0));

  auto under = shrink_sample_stats(
      samp, {CovarianceShrinkageKind::DiagonalTarget, -0.5});
  REQUIRE(under.has_value());
  CHECK(under->block_diagnostics[0].intensity == doctest::Approx(0.0));
  CHECK(max_abs(under->stats.S[0], S) == 0.0);
}

// ============================================================================
// Means and n_obs pass through unchanged.
// ============================================================================

TEST_CASE("shrink_sample_stats: means and n_obs pass through") {
  Eigen::Matrix2d S;
  S << 1.0, 0.5,
       0.5, 1.0;
  SampleStats samp = one_block(S, 321);
  samp.mean.push_back(Eigen::Vector2d(2.5, -1.0));

  auto r = shrink_sample_stats(samp,
                               {CovarianceShrinkageKind::IdentityTarget, 0.3});
  REQUIRE(r.has_value());
  REQUIRE(r->stats.mean.size() == 1);
  CHECK(r->stats.mean[0] == samp.mean[0]);
  REQUIRE(r->stats.n_obs.size() == 1);
  CHECK(r->stats.n_obs[0] == 321);
}

// ============================================================================
// Estimated (Ledoit-Wolf) intensity.
// ============================================================================

TEST_CASE("shrink_sample_stats: estimated DiagonalTarget intensity") {
  // 2x2, variances 2.0 and 0.5, covariance 0.6 ⇒ correlation r = 0.6.
  // Normal-theory λ* = (1-r²)²/n / r² = (0.64)²/100 / 0.36 = 0.4096/36.
  Eigen::Matrix2d S;
  S << 2.0, 0.6,
       0.6, 0.5;
  auto r = shrink_sample_stats(
      one_block(S, 100),
      {CovarianceShrinkageKind::DiagonalTarget, 0.0, /*estimate=*/true});
  REQUIRE(r.has_value());
  const double lambda = 0.4096 / 36.0;
  CHECK(r->block_diagnostics[0].intensity == doctest::Approx(lambda));
  // DiagonalTarget keeps the variances and scales the covariance by (1-λ*).
  CHECK(r->stats.S[0](0, 0) == doctest::Approx(2.0));
  CHECK(r->stats.S[0](1, 1) == doctest::Approx(0.5));
  CHECK(r->stats.S[0](0, 1) == doctest::Approx(0.6 * (1.0 - lambda)));
}

TEST_CASE("shrink_sample_stats: estimate_intensity rejects non-Diagonal kinds") {
  Eigen::Matrix2d S;
  S << 1.0, 0.5,
       0.5, 1.0;
  auto r = shrink_sample_stats(
      one_block(S),
      {CovarianceShrinkageKind::IdentityTarget, 0.0, /*estimate=*/true});
  CHECK_FALSE(r.has_value());
}
