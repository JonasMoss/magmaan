#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/inference/inference.hpp"     // chi2_pvalue (oracle for p-values)
#include "magmaan/robust/robust.hpp"
#include "magmaan/data/raw_data.hpp"      // empirical_gamma (oracle Γ̂)
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/robust.hpp"


namespace {

// ── Synthetic toy: saturated 2-variable covariance model ───────────────────
//
// Identify Σ̂ at H1 with the sample S (so θ̂_H1 = vech(S), Π = I_3, P = V).
// Restriction H0: σ_{12} = 0  ⇒  A_alpha = [0, 1, 0] (selecting the
// off-diagonal entry of vech).  m = 1, df_H0 − df_H1 = 1.
//
// Build all three quantities (C, S, eigvals) two ways and check agreement:
//   (i)  streaming/low-rank — `compute_satorra2000`.
//   (ii) form-the-full-UΓ inline (the "naïve" reference Satorra-2000 is
//        meant to replace) — `naive_satorra_eigvals` below.
// ───────────────────────────────────────────────────────────────────────────

// V_g · vech(M) = vech(Σ⁻¹·M·Σ⁻¹) with diag halved.  Plain restatement of
// the operator inside satorra2000.cpp, kept here so the oracle path can
// build V_g as a dense p* × p* matrix (which the production path never does).
Eigen::MatrixXd dense_V(const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p     = Sigma.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd W = llt.solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd V(pstar, pstar);
  for (Eigen::Index c = 0; c < pstar; ++c) {
    // unit basis vector e_c → M, then vech(W·M·W) with diag halved.
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(p, p);
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j; i < p; ++i) {
        if (k == c) {
          M(i, j) = 1.0;
          if (i != j) M(j, i) = 1.0;
        }
        ++k;
      }
    }
    const Eigen::MatrixXd Z = W * M * W;
    Eigen::Index r = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j; i < p; ++i) {
        V(r++, c) = (i == j) ? 0.5 * Z(i, j) : Z(i, j);
      }
    }
  }
  return V;
}

// Naïve full-UΓ eigenvalues — the textbook computation Satorra-2000 replaces.
// Always builds the dense p* × p* matrices.
Eigen::VectorXd naive_satorra_eigvals(
    const std::vector<magmaan::robust::SatorraGroup>& groups,
    const Eigen::MatrixXd&               A_alpha) {
  const Eigen::Index r1 = A_alpha.cols();
  const Eigen::Index m  = A_alpha.rows();

  // Pooled P = Σ_g w_g · Πᵀ · V · Π   (r1 × r1)
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(r1, r1);
  for (const auto& gr : groups) {
    const Eigen::MatrixXd V = dense_V(gr.Sigma);
    P.noalias() += gr.weight * (gr.Pi_alpha.transpose() * V * gr.Pi_alpha);
  }
  const Eigen::MatrixXd P_inv =
      P.ldlt().solve(Eigen::MatrixXd::Identity(r1, r1));

  Eigen::VectorXd all_eigs(m);
  // Per-group: form U_g = V_g·Π_g·P⁻¹·Aᵀ·C⁻¹·A·P⁻¹·Π_gᵀ·V_g  (p*_g × p*_g),
  // pool through Γ̂ and the group weight, eigendecompose.
  // For multi-group this is block-diag, so we'd need to stack blocks; here
  // we only use this oracle on the single-group toy test.
  REQUIRE(groups.size() == 1);

  const auto& gr   = groups[0];
  const Eigen::MatrixXd V    = dense_V(gr.Sigma);
  const Eigen::MatrixXd C    = A_alpha * P_inv * A_alpha.transpose();
  const Eigen::MatrixXd Cinv = C.ldlt().solve(Eigen::MatrixXd::Identity(m, m));
  const Eigen::MatrixXd U =
      V * gr.Pi_alpha * P_inv * A_alpha.transpose() *
      Cinv * A_alpha * P_inv * gr.Pi_alpha.transpose() * V;

  auto Gamma_or = magmaan::data::empirical_gamma(gr.X);
  REQUIRE(Gamma_or.has_value());
  const Eigen::MatrixXd Gamma = *Gamma_or;

  // Eigvals of U·Γ̂.  U is symmetric (by construction) and Γ̂ is symmetric
  // PSD — but U·Γ̂ is not symmetric in general.  Compute as
  //   eigvals(U·Γ̂) = eigvals(Γ̂^{½}·U·Γ̂^{½})   (similarity transform).
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_g(Gamma);
  const Eigen::VectorXd d_pos =
      es_g.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  const Eigen::MatrixXd sqrt_G = es_g.eigenvectors() * d_pos.asDiagonal() *
                                 es_g.eigenvectors().transpose();
  const Eigen::MatrixXd Msym = sqrt_G * U * sqrt_G;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (Msym + Msym.transpose()));
  const Eigen::VectorXd evs = es.eigenvalues();
  // Return the top m eigenvalues (ascending order; the others should be ~0).
  return evs.tail(m).reverse().reverse();  // tail returns ascending; OK
}

// Synthetic-data generator: zero-mean MVN with given Σ.
Eigen::MatrixXd sample_mvn(std::mt19937& rng, Eigen::Index n,
                            const Eigen::MatrixXd& Sigma) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X(n, Sigma.rows());
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(Sigma.rows());
    for (Eigen::Index k = 0; k < Sigma.rows(); ++k) zi(k) = z(rng);
    X.row(i) = (L * zi).transpose();
  }
  return X;
}

}  // namespace

// ── End-to-end: streaming vs. naïve on a saturated 2-var model ─────────────

TEST_CASE("compute_satorra2000: 2-var saturated, σ₁₂=0 restriction") {
  std::mt19937 rng(0xbeefU);
  const Eigen::Index n = 500;
  Eigen::MatrixXd Sigma_true(2, 2);
  Sigma_true << 1.5, 0.4,
                0.4, 0.8;
  const Eigen::MatrixXd X = sample_mvn(rng, n, Sigma_true);

  // θ̂_H1 = vech(S) — saturated.
  const Eigen::VectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc   = X.rowwise() - mean.transpose();
  const Eigen::MatrixXd S    = (Xc.transpose() * Xc) / static_cast<double>(n);

  // Π_alpha = I_3 (saturated, vech-coords), r1 = 3, p*_g = 3.
  const Eigen::MatrixXd Pi_alpha = Eigen::MatrixXd::Identity(3, 3);
  // A_alpha picks the off-diagonal slot σ_{12}.
  Eigen::MatrixXd A_alpha(1, 3);
  A_alpha << 0, 1, 0;

  magmaan::robust::SatorraGroup gr{
      /*Pi_alpha=*/Pi_alpha,
      /*Sigma   =*/S,                 // Σ̂_g(θ̂_H1) = S since saturated
      /*X       =*/X,
      /*mean    =*/mean,
      /*weight  =*/1.0,               // single group, n/N = 1
      /*n_g     =*/static_cast<std::int32_t>(n),
  };

  auto fast_or = magmaan::robust::compute_satorra2000({gr}, A_alpha,
                                          magmaan::robust::GammaSource::Empirical);
  REQUIRE(fast_or.has_value());
  const Eigen::VectorXd ev_fast   = fast_or->eigenvalues;
  const Eigen::VectorXd ev_naive = naive_satorra_eigvals({gr}, A_alpha);

  INFO("ev_fast   = ", ev_fast.transpose());
  INFO("ev_naive = ", ev_naive.transpose());
  REQUIRE(ev_fast.size() == 1);
  REQUIRE(ev_naive.size() == 1);
  CHECK(ev_fast(0) == doctest::Approx(ev_naive(0)).epsilon(1e-10));
  // The Satorra-Bentler scale `c` = tr(C⁻¹S)/m equals the single eigenvalue.
  CHECK(fast_or->trace_CinvS == doctest::Approx(ev_fast(0)).epsilon(1e-12));
  CHECK(fast_or->trace_CinvS_sq
        == doctest::Approx(ev_fast(0) * ev_fast(0)).epsilon(1e-12));
}

// ── NT short-circuit: Γ = Γ_NT ⇒ S = C ⇒ eigvals ≡ 1 ───────────────────────

TEST_CASE("compute_satorra2000: NT-Γ sanity (all eigvals → 1)") {
  std::mt19937 rng(0xfaceU);
  const Eigen::Index n = 200;
  Eigen::MatrixXd Sigma_true(3, 3);
  Sigma_true << 2.0, 0.3, 0.1,
                0.3, 1.5, 0.2,
                0.1, 0.2, 1.0;
  const Eigen::MatrixXd X = sample_mvn(rng, n, Sigma_true);
  const Eigen::VectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc   = X.rowwise() - mean.transpose();
  const Eigen::MatrixXd S    = (Xc.transpose() * Xc) / static_cast<double>(n);

  // Saturated p=3: p* = 6, r1 = 6.  Restrict two off-diagonals to zero.
  const Eigen::MatrixXd Pi_alpha = Eigen::MatrixXd::Identity(6, 6);
  Eigen::MatrixXd A_alpha(2, 6);
  // vech order (col-major lower-tri): (1,1), (2,1), (3,1), (2,2), (3,2), (3,3)
  // σ_{12} is index 1; σ_{13} is index 2.
  A_alpha << 0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0;

  magmaan::robust::SatorraGroup gr{Pi_alpha, S, X, mean, 1.0,
                      static_cast<std::int32_t>(n)};
  auto r = magmaan::robust::compute_satorra2000({gr}, A_alpha, magmaan::robust::GammaSource::NT);
  REQUIRE(r.has_value());
  CHECK(r->eigenvalues.size() == 2);
  for (Eigen::Index k = 0; k < r->eigenvalues.size(); ++k) {
    CHECK(r->eigenvalues(k) == doctest::Approx(1.0).epsilon(1e-12));
  }
  CHECK(r->trace_CinvS == doctest::Approx(2.0).epsilon(1e-12));
}

// ── m = 0 degenerate ───────────────────────────────────────────────────────

TEST_CASE("compute_satorra2000: degenerate m=0 (H0 ≡ H1)") {
  const Eigen::Index n = 50;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, 2);
  Eigen::VectorXd mean = X.colwise().mean();
  Eigen::MatrixXd Xc   = X.rowwise() - mean.transpose();
  Eigen::MatrixXd S    = (Xc.transpose() * Xc) / static_cast<double>(n);

  magmaan::robust::SatorraGroup gr{Eigen::MatrixXd::Identity(3, 3), S, X, mean, 1.0,
                      static_cast<std::int32_t>(n)};
  const Eigen::MatrixXd A_alpha = Eigen::MatrixXd::Zero(0, 3);
  auto r = magmaan::robust::compute_satorra2000({gr}, A_alpha);
  REQUIRE(r.has_value());
  CHECK(r->C.size() == 0);
  CHECK(r->S.size() == 0);
  CHECK(r->eigenvalues.size() == 0);
  CHECK(r->trace_CinvS == doctest::Approx(0.0));
  CHECK(r->trace_CinvS_sq == doctest::Approx(0.0));
}

// ── Low-level p-value wrap ─────────────────────────────────────────────────

TEST_CASE("lr_test_satorra2000: low-level p-value wrap") {
  // Synthesise a SatorraDiffResult with hand-picked eigenvalues so all four
  // p-value formulas can be cross-checked against closed forms.
  magmaan::robust::SatorraDiffResult sd;
  Eigen::VectorXd eig(3);
  eig << 1.5, 1.0, 0.5;   // Σλ = 3, Σλ² = 3.5
  sd.eigenvalues    = eig;
  sd.trace_CinvS    = eig.sum();           // 3.0
  sd.trace_CinvS_sq = eig.squaredNorm();   // 3.5

  const double T_diff = 7.0;
  auto r_or = magmaan::robust::lr_test_satorra2000(T_diff, sd);
  REQUIRE(r_or.has_value());
  const auto& r = *r_or;

  CHECK(r.df_diff == 3);
  CHECK(r.T_diff  == doctest::Approx(T_diff));
  CHECK(r.p_unscaled == doctest::Approx(magmaan::inference::chi2_pvalue(T_diff, 3)));

  // Scaled: ĉ = 3/3 = 1; T_scaled = 7.
  CHECK(r.scale_c  == doctest::Approx(1.0));
  CHECK(r.T_scaled == doctest::Approx(7.0));
  CHECK(r.p_scaled == doctest::Approx(magmaan::inference::chi2_pvalue(7.0, 3)));

  // Adjusted: d̂₀ = 9/3.5 = 2.5714…; T_adj = 7 · 2.5714 / 3 = 6.0.
  const double d0_expected  = 9.0 / 3.5;
  const double Tadj_expected = T_diff * d0_expected / sd.trace_CinvS;
  CHECK(r.adjust_d0 == doctest::Approx(d0_expected));
  CHECK(r.T_adjusted == doctest::Approx(Tadj_expected));
  // p_adjusted = 1 − pchisq(Tadj, d̂₀).
  const double p_adj_expected =
      1.0 - magmaan::inference::noncentral_chisq_cdf(Tadj_expected, d0_expected, 0.0);
  CHECK(r.p_adjusted == doctest::Approx(p_adj_expected));

  // Mixture p-value matches `imhof_upper` directly.
  CHECK(r.p_mixture == doctest::Approx(magmaan::robust::imhof_upper(eig, T_diff)));
}

TEST_CASE("lr_test_satorra2000: degenerate (m = 0)") {
  magmaan::robust::SatorraDiffResult sd;
  sd.eigenvalues    = Eigen::VectorXd::Zero(0);
  sd.trace_CinvS    = 0.0;
  sd.trace_CinvS_sq = 0.0;
  auto r_or = magmaan::robust::lr_test_satorra2000(0.0, sd);
  REQUIRE(r_or.has_value());
  const auto& r = *r_or;
  CHECK(r.df_diff == 0);
  CHECK(r.p_unscaled == doctest::Approx(1.0));
  CHECK(r.p_scaled   == doctest::Approx(1.0));
  CHECK(r.p_adjusted == doctest::Approx(1.0));
  CHECK(r.p_mixture  == doctest::Approx(1.0));
}

// ── Error: P singular (over-parameterised Π) ──────────────────────────────

TEST_CASE("compute_satorra2000: rejects singular pooled info") {
  Eigen::MatrixXd Pi_alpha(3, 4);    // 4 params, 3 moments — rank-deficient P
  Pi_alpha << 1, 0, 0, 1,
              0, 1, 0, 0,
              0, 0, 1, 0;
  const Eigen::MatrixXd S = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::Index n = 10;
  const Eigen::MatrixXd X    = Eigen::MatrixXd::Random(n, 2);
  const Eigen::VectorXd mean = X.colwise().mean();
  magmaan::robust::SatorraGroup gr{Pi_alpha, S, X, mean, 1.0,
                      static_cast<std::int32_t>(n)};
  Eigen::MatrixXd A_alpha(1, 4);
  A_alpha << 1, 0, 0, -1;
  auto r = magmaan::robust::compute_satorra2000({gr}, A_alpha);
  CHECK_FALSE(r.has_value());
}
