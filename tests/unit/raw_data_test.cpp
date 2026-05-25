#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"

namespace {

// Random PD covariance generator — same recipe as elsewhere in the suite.
Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p, double diag_boost) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() + Eigen::MatrixXd::Identity(p, p) * diag_boost;
}

// Sample n rows from N(μ, Σ) via the LLT factor.
Eigen::MatrixXd mvn_sample(std::mt19937& rng,
                           Eigen::Index n,
                           const Eigen::VectorXd& mu,
                           const Eigen::MatrixXd& Sigma) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X(n, mu.size());
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(mu.size());
    for (Eigen::Index k = 0; k < mu.size(); ++k) zi(k) = z(rng);
    X.row(i) = (mu + L * zi).transpose();
  }
  return X;
}

}  // namespace

TEST_CASE("sample_stats_from_raw: N-divisor cov + sample mean") {
  std::mt19937 rng(2026);
  const Eigen::Index n = 300, p = 4;
  Eigen::VectorXd mu(p); mu << 1.0, 2.0, 3.0, 4.0;
  const Eigen::MatrixXd Sigma = random_pd(rng, p, 5.0);

  magmaan::data::RawData raw;
  raw.X.push_back(mvn_sample(rng, n, mu, Sigma));

  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  const auto& samp = *samp_or;

  CHECK(samp.S.size()     == 1u);
  CHECK(samp.mean.size()  == 1u);
  CHECK(samp.n_obs.size() == 1u);
  CHECK(samp.S[0].rows()  == p);
  CHECK(samp.n_obs[0]     == n);

  // Recompute by hand: N-divisor cov from raw, sample mean.
  const Eigen::VectorXd mean_hand = raw.X[0].colwise().mean();
  const Eigen::MatrixXd Xc        = raw.X[0].rowwise() - mean_hand.transpose();
  const Eigen::MatrixXd S_hand    = (Xc.transpose() * Xc) / static_cast<double>(n);
  CHECK((samp.mean[0] - mean_hand).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((samp.S[0] - S_hand).cwiseAbs().maxCoeff()       < 1e-12);
}

TEST_CASE("gamma_nt: closed-form formula σ_ik·σ_jl + σ_il·σ_jk") {
  std::mt19937 rng(7);
  const Eigen::Index p = 3;
  const Eigen::MatrixXd Sigma = random_pd(rng, p, 3.0);
  const Eigen::Index pstar = p * (p + 1) / 2;

  auto G_or = magmaan::data::gamma_nt(Sigma);
  REQUIRE(G_or.has_value());
  const auto& G = *G_or;

  CHECK(G.rows() == pstar);
  CHECK(G.cols() == pstar);
  // Symmetric.
  CHECK((G - G.transpose()).cwiseAbs().maxCoeff() < 1e-12);
  // Spot-check a known entry. Lower-tri column-major vech:
  //   a = 0 -> (i,j) = (0,0)        a = 1 -> (1,0)   a = 2 -> (2,0)
  //   a = 3 -> (1,1)                a = 4 -> (2,1)   a = 5 -> (2,2)
  // G[0,0] = σ_00·σ_00 + σ_00·σ_00 = 2·σ_00²
  CHECK(G(0, 0) == doctest::Approx(2.0 * Sigma(0, 0) * Sigma(0, 0)).epsilon(1e-12));
  // G[1,1] involves (1,0)x(1,0): σ_11·σ_00 + σ_10·σ_10
  CHECK(G(1, 1) == doctest::Approx(Sigma(1, 1) * Sigma(0, 0) +
                                   Sigma(1, 0) * Sigma(1, 0)).epsilon(1e-12));
  // G[0,3] involves (0,0)x(1,1): σ_01·σ_01 + σ_01·σ_01 = 2·σ_01²
  CHECK(G(0, 3) == doctest::Approx(2.0 * Sigma(0, 1) * Sigma(0, 1)).epsilon(1e-12));
}

TEST_CASE("empirical_gamma → gamma_nt on large MVN sample") {
  // Under multivariate normality, Γ̂_n → Γ_NT(Σ) as n → ∞. With n=20000,
  // p=3 the relative error should land well under 5% — tight enough that
  // a bug in either function would surface, loose enough to absorb the
  // O(1/√n) sampling noise.
  std::mt19937 rng(2025);
  const Eigen::Index p = 3, n = 20000;
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  const Eigen::MatrixXd Sigma = random_pd(rng, p, 4.0);

  const Eigen::MatrixXd X = mvn_sample(rng, n, mu, Sigma);
  auto G_emp_or = magmaan::data::empirical_gamma(X);
  auto G_nt_or  = magmaan::data::gamma_nt(Sigma);
  REQUIRE(G_emp_or.has_value());
  REQUIRE(G_nt_or.has_value());

  const double max_abs  = (*G_emp_or - *G_nt_or).cwiseAbs().maxCoeff();
  const double scale    = G_nt_or->cwiseAbs().maxCoeff();
  CHECK(max_abs / scale < 0.05);
}

TEST_CASE("empirical_gamma_with_means: bottom-right is bit-equal to "
          "empirical_gamma (regression guard)") {
  // The Zc-casewise pipeline behind empirical_gamma is shared with the
  // new sibling via detail::build_centered_moments. The bottom-right
  // (p* × p*) sub-block of the stacked NACOV must equal the bare
  // empirical_gamma output bit-for-bit on the same input — that is the
  // promise of the refactor.
  std::mt19937 rng(20260525);
  const Eigen::Index n = 400, p = 3;
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  const Eigen::MatrixXd Sigma = random_pd(rng, p, 2.0);
  const Eigen::MatrixXd X = mvn_sample(rng, n, mu, Sigma);

  auto G_or      = magmaan::data::empirical_gamma(X);
  auto G_full_or = magmaan::data::empirical_gamma_with_means(X);
  REQUIRE(G_or.has_value());
  REQUIRE(G_full_or.has_value());

  const Eigen::Index pstar = p * (p + 1) / 2;
  REQUIRE(G_full_or->rows() == p + pstar);
  REQUIRE(G_full_or->cols() == p + pstar);

  const Eigen::MatrixXd br = G_full_or->bottomRightCorner(pstar, pstar);
  CHECK((br - *G_or).cwiseAbs().maxCoeff() == 0.0);   // bit-exact equality
}

TEST_CASE("empirical_gamma_with_means → blockdiag(Σ, Γ_NT) on large MVN") {
  // Under multivariate normality, third moments vanish so the cross
  // block of the full stacked NACOV → 0; the top-left block → Σ; the
  // bottom-right block → Γ_NT(Σ). Same n=20000 budget as the existing
  // empirical_gamma → gamma_nt convergence test.
  std::mt19937 rng(20260526);
  const Eigen::Index p = 3, n = 20000;
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  const Eigen::MatrixXd Sigma = random_pd(rng, p, 4.0);
  const Eigen::MatrixXd X = mvn_sample(rng, n, mu, Sigma);

  auto G_full_or = magmaan::data::empirical_gamma_with_means(X);
  auto G_nt_or   = magmaan::data::gamma_nt(Sigma);
  REQUIRE(G_full_or.has_value());
  REQUIRE(G_nt_or.has_value());
  const Eigen::Index pstar = p * (p + 1) / 2;

  const Eigen::MatrixXd tl = G_full_or->topLeftCorner(p, p);
  const Eigen::MatrixXd br = G_full_or->bottomRightCorner(pstar, pstar);
  const Eigen::MatrixXd cr = G_full_or->topRightCorner(p, pstar);

  // Top-left → Σ; bottom-right → Γ_NT(Σ); cross → 0. Relative tolerances
  // matching the existing convergence test (5%).
  CHECK((tl - Sigma).cwiseAbs().maxCoeff() / Sigma.cwiseAbs().maxCoeff() < 0.05);
  CHECK((br - *G_nt_or).cwiseAbs().maxCoeff() / G_nt_or->cwiseAbs().maxCoeff() < 0.05);
  // Cross-block: absolute bound — the true value is 0, so a few × σ/√n
  // is the noise scale.
  const double cross_scale = Sigma.cwiseAbs().maxCoeff();
  CHECK(cr.cwiseAbs().maxCoeff() / cross_scale < 0.05);
}

TEST_CASE("empirical_gamma_with_means: skewed data → non-zero cross-block") {
  // χ²(1)-style skewed columns: y_i = z_i² − 1 has E=0, Var=2, skew>0.
  // The population cross-block Cov(y, vech(y yᵀ)) is non-trivial (the
  // third central moments are non-zero by construction), so the empirical
  // cross-block should land well above the zero-moment baseline a normal
  // sample of the same size produces.
  std::mt19937 rng(20260527);
  const Eigen::Index n = 10000, p = 2;
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  const Eigen::MatrixXd Sigma = Eigen::MatrixXd::Identity(p, p);

  // Baseline: MVN(0, I) → cross-block should be ≈ 0.
  const Eigen::MatrixXd X_norm = mvn_sample(rng, n, mu, Sigma);
  auto G_norm_or = magmaan::data::empirical_gamma_with_means(X_norm);
  REQUIRE(G_norm_or.has_value());

  // Skewed: y_ij = z_ij² − 1 with z ~ N(0, 1).
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X_skew(n, p);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < p; ++j) {
      const double zi = z(rng);
      X_skew(i, j) = zi * zi - 1.0;
    }
  auto G_skew_or = magmaan::data::empirical_gamma_with_means(X_skew);
  REQUIRE(G_skew_or.has_value());

  const Eigen::Index pstar = p * (p + 1) / 2;
  const double cross_norm = G_norm_or->topRightCorner(p, pstar).cwiseAbs().maxCoeff();
  const double cross_skew = G_skew_or->topRightCorner(p, pstar).cwiseAbs().maxCoeff();

  // Skewed cross block dwarfs the MVN baseline. For χ²(1) the diagonal
  // third moments are 8; with n=10000 the sample value should land within
  // a couple of standard errors of that, way above the MVN baseline (which
  // is O(1/√n) sampling noise around zero).
  CHECK(cross_skew > 10.0 * cross_norm);
  CHECK(cross_skew > 4.0);   // sample of E[y³] ≈ 8 with √n noise
}

TEST_CASE("empirical_gamma: error on degenerate input") {
  Eigen::MatrixXd X(1, 3);  X << 1.0, 2.0, 3.0;
  auto out = magmaan::data::empirical_gamma(X);
  CHECK_FALSE(out.has_value());

  Eigen::MatrixXd Y(5, 0);
  auto out2 = magmaan::data::empirical_gamma(Y);
  CHECK_FALSE(out2.has_value());
}

TEST_CASE("sample_stats_from_raw: error on degenerate / missingness") {
  magmaan::data::RawData raw;
  raw.X.push_back(Eigen::MatrixXd::Zero(1, 3));   // n=1 — covariance undefined
  CHECK_FALSE(magmaan::data::sample_stats_from_raw(raw).has_value());

  raw.X.clear();
  raw.X.push_back(Eigen::MatrixXd::Zero(10, 3));
  raw.mask.push_back(
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(10, 3));
  // Mask present → FIML-phase territory, not yet supported.
  CHECK_FALSE(magmaan::data::sample_stats_from_raw(raw).has_value());
}
