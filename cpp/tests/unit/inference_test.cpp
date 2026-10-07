#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#include "../inference_bundle.hpp"

using magmaan::estimate::Estimates;
using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatrixRep;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::LatentStructure;
using magmaan::test::analytic_observed_inference;
using magmaan::test::expected_inference;
using magmaan::test::fd_observed_inference;
using magmaan::test::InferenceBundle;

namespace {

// Build pt + rep and keep them alive in caller-owned storage. Returns the
// pair by reference via the static-storage trick — same pattern as
// ml_test.cpp's must_build, but we need pt/rep handles here, not the
// evaluator (which inference.compute() rebuilds itself).
struct ModelHandles {
  LatentStructure* pt;
  MatrixRep* rep;
};

ModelHandles must_model(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  static thread_local LatentStructure  s_pt;
  static thread_local MatrixRep s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  return {&s_pt, &s_mr};
}

// Random PD covariance — matches ml_test.cpp's helper.
Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() + Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

// Load lavaan's θ̂ from a fit fixture so we have a meaningful point to
// evaluate inference at without re-running the optimizer.
Eigen::VectorXd theta_from_fixture(const std::string& fixture_path) {
  std::ifstream in(fixture_path);
  REQUIRE(in.is_open());
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& arr = j["theta_hat"];
  Eigen::VectorXd theta(arr.size());
  for (Eigen::Index k = 0; k < theta.size(); ++k)
    theta(k) = arr[static_cast<std::size_t>(k)].get<double>();
  return theta;
}

}  // namespace

TEST_CASE("expected_inference: shapes, symmetry, PSD at θ̂ (1F CFA)") {
  // 1-factor CFA — load lavaan's θ̂ so we evaluate at a real estimate.
  auto h = must_model("f =~ x1 + x2 + x3");
  const Eigen::VectorXd theta = theta_from_fixture(
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0001_one_factor_cfa.fit.json");

  // S = lavaan's sample S from the same fixture so n and the moments are
  // self-consistent. The inference math doesn't depend on S beyond
  // (n − 1)/2 weighting, but reuse it for cleanliness.
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                  .get<double>();

  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());

  Estimates est;
  est.theta = theta;
  est.fmin  = 0.0;  // not used by the shape checks below

  auto inf_or = expected_inference(*h.pt, *h.rep, samp, est);
  REQUIRE(inf_or.has_value());
  const auto& inf = *inf_or;

  const auto n_free = static_cast<Eigen::Index>(theta.size());

  // Shape.
  CHECK(inf.info.rows() == n_free);
  CHECK(inf.info.cols() == n_free);
  CHECK(inf.vcov.rows() == n_free);
  CHECK(inf.vcov.cols() == n_free);
  CHECK(inf.se.size()   == n_free);

  // Symmetry: I = Iᵀ to within numerical noise. vcov inherits this from I⁻¹.
  CHECK((inf.info - inf.info.transpose()).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((inf.vcov - inf.vcov.transpose()).cwiseAbs().maxCoeff() < 1e-10);

  // Positive definite: LLT succeeds.
  Eigen::LLT<Eigen::MatrixXd> llt(inf.info);
  CHECK(llt.info() == Eigen::Success);

  // All SEs strictly positive (no NaN, no zero) for an identified model.
  for (Eigen::Index k = 0; k < n_free; ++k) {
    CHECK(std::isfinite(inf.se(k)));
    CHECK(inf.se(k) > 0.0);
  }

  // df = p(p+1)/2 − n_free.
  const int expected_df = static_cast<int>(p * (p + 1) / 2 - n_free);
  CHECK(inf.df == expected_df);
}

TEST_CASE("df_stat: df = 24 for 3F Holzinger") {
  // The flagship df sanity check from the plan: 9 indicators × 10 / 2 − 21 = 24.
  auto h = must_model(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  const Eigen::VectorXd theta = theta_from_fixture(
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0002_three_factor_hs.fit.json");
  REQUIRE(theta.size() == 21);

  // S can be anything PD with the right dimension; the df formula is
  // structural (counts moments and free params), not numerical.
  std::mt19937 rng(2026);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 9));
  samp.n_obs.push_back(301);

  Estimates est;  est.theta = theta;  est.fmin = 0.0;

  auto inf_or = expected_inference(*h.pt, *h.rep, samp, est);
  REQUIRE(inf_or.has_value());
  CHECK(inf_or->df == 24);
  CHECK(inf_or->info.rows() == 21);
  CHECK(inf_or->se.size()   == 21);
}

TEST_CASE("rls_chi2: matches lavaan browne.residual.nt.model on 3F Holzinger") {
  // Reference: lavaan with `test = "browne.residual.nt.model"` (the RLS /
  // model-based variant — the one named "Browne's residual (NT model-based)
  // test", a.k.a. reweighted least-squares) returns 81.3677 on the 3F
  // Holzinger fit. Our T_ML is 85.3055, so the two stats agree
  // asymptotically but differ in finite samples.
  auto h = must_model(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0002_three_factor_hs.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());

  // Fit so we have θ̂; the statistic needs the model Jacobian, so it takes the
  // structure rather than pre-built moments.
  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();

  auto t_rls = magmaan::inference::rls_chi2(*h.pt, *h.rep, samp, est.theta);
  REQUIRE(t_rls.has_value());
  CHECK(*t_rls == doctest::Approx(81.3677).epsilon(1e-3));
}

TEST_CASE("rls_chi2: zero on saturated 1F CFA") {
  // At saturation Σ̂ = S exactly, so the residual is the zero matrix and
  // F_RLS = 0. Numerically expect ~0 up to LLT/solve roundoff.
  auto h = must_model("f =~ x1 + x2 + x3");

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());

  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();

  auto t_rls = magmaan::inference::rls_chi2(*h.pt, *h.rep, samp, est.theta);
  REQUIRE(t_rls.has_value());
  CHECK(std::abs(*t_rls) < 1e-6);
}

TEST_CASE("nt_moment_quadratic: separates mean and covariance residuals") {
  // With Sigma = diag(2, 4), covariance residual diag(1, -2), and mean
  // residual (1, 2), the two per-observation quadratic forms are
  //   1/2 tr{diag(1/2, -1/2)^2} = 1/4,
  //   (1, 2)' diag(1/2, 1/4) (1, 2) = 3/2.
  // At n = 40 this gives T_cov = 10, T_mean = 60, and T_total = 70.
  SampleStats samp;
  samp.S.push_back((Eigen::Matrix2d() << 3.0, 0.0, 0.0, 2.0).finished());
  samp.mean.push_back((Eigen::Vector2d() << 2.0, 1.0).finished());
  samp.n_obs.push_back(40);

  magmaan::model::ImpliedMoments im;
  im.sigma.push_back((Eigen::Matrix2d() << 2.0, 0.0, 0.0, 4.0).finished());
  im.mu.push_back((Eigen::Vector2d() << 1.0, -1.0).finished());

  auto t = magmaan::inference::frontier::nt_moment_quadratic(samp, im);
  REQUIRE(t.has_value());
  CHECK(t->mean == doctest::Approx(60.0));
  CHECK(t->covariance == doctest::Approx(10.0));
  CHECK(t->statistic == doctest::Approx(70.0));
}

TEST_CASE("nt_moment_quadratic: empty implied means drop the mean block") {
  SampleStats samp;
  samp.S.push_back((Eigen::Matrix2d() << 3.0, 0.0, 0.0, 2.0).finished());
  samp.mean.push_back((Eigen::Vector2d() << 9.0, -7.0).finished());
  samp.n_obs.push_back(40);

  magmaan::model::ImpliedMoments im;
  im.sigma.push_back((Eigen::Matrix2d() << 2.0, 0.0, 0.0, 4.0).finished());

  // Σ = diag(2, 4), S = diag(3, 2) ⇒ A = Σ⁻¹(S − Σ) = diag(½, −½), so the
  // per-observation covariance form is ½·tr(A²) = ¼ and T_cov = 40·¼ = 10.
  constexpr double kCovOnly = 10.0;

  auto full = magmaan::inference::frontier::nt_moment_quadratic(samp, im);
  REQUIRE(full.has_value());
  CHECK(full->mean == 0.0);
  CHECK(full->covariance == doctest::Approx(kCovOnly));
  CHECK(full->statistic == doctest::Approx(kCovOnly));

  // The converse is the same contract: an implied mean vector alone does not
  // make means part of a covariance-only fit.
  SampleStats covariance_fit = samp;
  covariance_fit.mean.clear();
  im.mu.push_back((Eigen::Vector2d() << 1.0, -1.0).finished());
  auto no_sample_means =
      magmaan::inference::frontier::nt_moment_quadratic(covariance_fit, im);
  REQUIRE(no_sample_means.has_value());
  CHECK(no_sample_means->mean == 0.0);
  CHECK(no_sample_means->statistic == doctest::Approx(kCovOnly));
}

TEST_CASE("browne_residual_nt: matches lavaan on 3F Holzinger") {
  // Reference: lavaan with `test = "browne.residual.nt"` returns 77.9034
  // on this fit. This is the model-projected, S⁻¹-weighted residual-based
  // NT test — distinct from both T_ML (85.3055) and T_RLS (81.3677).
  auto h = must_model(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0002_three_factor_hs.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());

  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();
  auto t_or = magmaan::inference::browne_residual_nt(*h.pt, *h.rep, samp, est);
  REQUIRE(t_or.has_value());
  CHECK(*t_or == doctest::Approx(77.9034).epsilon(1e-3));
}

TEST_CASE("browne_residual_nt: zero on saturated 1F CFA") {
  // At saturation, res = 0 so term1 = term2 = 0 and T = 0.
  auto h = must_model("f =~ x1 + x2 + x3");

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());

  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();
  auto t_or = magmaan::inference::browne_residual_nt(*h.pt, *h.rep, samp, est);
  REQUIRE(t_or.has_value());
  CHECK(std::abs(*t_or) < 1e-6);
}

TEST_CASE("browne_residual_adf: empirical Gamma approaches NT on MVN data") {
  auto h = must_model("f =~ x1 + x2 + x3 + x4");

  Eigen::MatrixXd Sigma(4, 4);
  Sigma << 1.00, 0.72, 0.63, 0.54,
           0.72, 1.30, 0.70, 0.60,
           0.63, 0.70, 1.10, 0.66,
           0.54, 0.60, 0.66, 1.20;
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);

  std::mt19937 rng(1729);
  std::normal_distribution<double> zdist(0.0, 1.0);
  magmaan::data::RawData raw;
  raw.X.emplace_back(6000, 4);
  for (Eigen::Index r = 0; r < raw.X[0].rows(); ++r) {
    Eigen::VectorXd z(4);
    for (Eigen::Index c = 0; c < 4; ++c) z(c) = zdist(rng);
    raw.X[0].row(r) = (llt.matrixL() * z).transpose();
  }

  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  auto est = magmaan::test::fit(*h.pt, *h.rep, *samp_or).value();
  auto nt_or = magmaan::inference::browne_residual_nt(*h.pt, *h.rep, *samp_or, est);
  auto adf_or = magmaan::inference::browne_residual_adf(*h.pt, *h.rep, *samp_or, raw, est);
  REQUIRE(nt_or.has_value());
  REQUIRE(adf_or.has_value());
  CHECK(*adf_or == doctest::Approx(*nt_or).epsilon(0.20));
}

TEST_CASE("information_cross_products → information_expected on MVN data") {
  // Information equality: under MVN at the true parameter (and asymptotically
  // at the MLE), the OPG estimator Σ_i s_i s_iᵀ converges to the expected
  // Fisher info Δᵀ W Δ. Sample MVN with a known Σ at moderate N and check
  // the two information matrices agree within sampling noise.
  auto h = must_model("f =~ x1 + x2 + x3 + x4");

  Eigen::MatrixXd Sigma(4, 4);
  Sigma << 1.00, 0.72, 0.63, 0.54,
           0.72, 1.30, 0.70, 0.60,
           0.63, 0.70, 1.10, 0.66,
           0.54, 0.60, 0.66, 1.20;
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);

  std::mt19937 rng(424242);
  std::normal_distribution<double> zdist(0.0, 1.0);
  magmaan::data::RawData raw;
  raw.X.emplace_back(8000, 4);
  for (Eigen::Index r = 0; r < raw.X[0].rows(); ++r) {
    Eigen::VectorXd z(4);
    for (Eigen::Index c = 0; c < 4; ++c) z(c) = zdist(rng);
    raw.X[0].row(r) = (llt.matrixL() * z).transpose();
  }

  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  auto est = magmaan::test::fit(*h.pt, *h.rep, *samp_or).value();

  auto exp_or = magmaan::inference::information_expected(
      *h.pt, *h.rep, *samp_or, est);
  REQUIRE(exp_or.has_value());
  auto xp_or = magmaan::inference::information_cross_products(
      *h.pt, *h.rep, *samp_or, raw, est);
  REQUIRE(xp_or.has_value());

  const Eigen::MatrixXd& I_E  = *exp_or;
  const Eigen::MatrixXd& I_XP = *xp_or;
  REQUIRE(I_E.rows() == I_XP.rows());
  // OPG variance is high; 15% relative Frobenius distance at N = 8000 is a
  // realistic tolerance for a 1F CFA (saturated). Tighten if/when needed.
  const double rel = (I_XP - I_E).norm() / I_E.norm();
  CHECK(rel < 0.15);
}

TEST_CASE("Γ_NT⁻¹·vech(A) matches the trace-identity form casewise_scores uses") {
  // `casewise_scores` applies the normal-theory weight to the σ-segment of Δ
  // without forming Γ_NT: since Γ_NT⁻¹ = ½·Dᵀ(Σ⁻¹ ⊗ Σ⁻¹)D,
  //
  //   (Γ_NT⁻¹·vech(A))[(i,j)] = M(i,j)      i > j
  //                           = ½·M(i,i)    i == j,     M = Σ⁻¹AΣ⁻¹.
  //
  // That turns an O(p⁶) factorization plus O(p⁴·n_free) of solves into O(p³)
  // per column. This checks the substitution directly against a dense
  // `data::gamma_nt` solve — it is the one algebraic claim the rewrite rests
  // on, and the asymptotic OPG-vs-expected test above is far too loose to
  // catch an error in it.
  std::mt19937 rng(20260918);
  std::normal_distribution<double> z(0.0, 1.0);

  for (int p_i : {1, 2, 3, 5, 8}) {
    const Eigen::Index p = p_i;

    Eigen::MatrixXd B(p, p);
    for (Eigen::Index i = 0; i < p; ++i)
      for (Eigen::Index j = 0; j < p; ++j) B(i, j) = z(rng);
    Eigen::MatrixXd Sigma = B * B.transpose();
    Sigma.diagonal().array() += static_cast<double>(p);

    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    REQUIRE(llt.info() == Eigen::Success);
    auto G_or = magmaan::data::gamma_nt(Sigma);
    REQUIRE(G_or.has_value());
    Eigen::LLT<Eigen::MatrixXd> gllt(*G_or);
    REQUIRE(gllt.info() == Eigen::Success);

    const Eigen::Index pstar = p * (p + 1) / 2;

    for (int trial = 0; trial < 3; ++trial) {
      Eigen::MatrixXd A(p, p);
      for (Eigen::Index i = 0; i < p; ++i)
        for (Eigen::Index j = i; j < p; ++j) {
          A(i, j) = z(rng);
          A(j, i) = A(i, j);
        }

      // vech(A) in column-major lower-triangle order, matching
      // `dsigma_dtheta` and `data::gamma_nt`.
      Eigen::VectorXd vA(pstar);
      Eigen::Index    t = 0;
      for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index i = j; i < p; ++i, ++t) vA[t] = A(i, j);

      const Eigen::VectorXd ref = gllt.solve(vA);

      const Eigen::MatrixXd X  = llt.solve(A);
      const Eigen::MatrixXd Xt = X.transpose();
      const Eigen::MatrixXd M  = llt.solve(Xt);
      Eigen::VectorXd       got(pstar);
      t = 0;
      for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index i = j; i < p; ++i, ++t)
          got[t] = (i == j) ? 0.5 * M(i, j) : M(i, j);

      const double denom = ref.norm() > 1e-12 ? ref.norm() : 1.0;
      CHECK((got - ref).norm() / denom < 1e-9);
    }
  }
}

TEST_CASE("browne_residual_adf: zero on saturated model") {
  auto h = must_model("f =~ x1 + x2 + x3");

  magmaan::data::RawData raw;
  raw.X.emplace_back(301, 3);
  std::mt19937 rng(2024);
  std::normal_distribution<double> zdist(0.0, 1.0);
  for (Eigen::Index r = 0; r < raw.X[0].rows(); ++r) {
    raw.X[0](r, 0) = zdist(rng);
    raw.X[0](r, 1) = 0.7 * raw.X[0](r, 0) + zdist(rng);
    raw.X[0](r, 2) = 0.5 * raw.X[0](r, 0) + zdist(rng);
  }
  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  auto est = magmaan::test::fit(*h.pt, *h.rep, *samp_or).value();
  auto adf_or = magmaan::inference::browne_residual_adf(*h.pt, *h.rep, *samp_or, raw, est);
  REQUIRE(adf_or.has_value());
  CHECK(std::abs(*adf_or) < 1e-6);
}

TEST_CASE("chi2_stat: chi2 = 2n · fmin") {
  // Pure arithmetic on the fmin → chi2 plumbing. est.fmin = ½F, so the
  // statistic is T = 2N·fmin = N·F. N (not N−1) matches lavaan's
  // `likelihood = "normal"` default. Doesn't depend on any information
  // matrix — `chi2_stat` reads samp.n_obs and est.fmin only.
  SampleStats samp;
  samp.S.push_back(Eigen::MatrixXd::Identity(3, 3));   // not used by chi2_stat
  samp.n_obs.push_back(301);

  Estimates est;
  est.theta = Eigen::VectorXd::Zero(3);                // unused
  est.fmin  = 0.04321;

  CHECK(magmaan::inference::chi2_stat(samp, est) ==
        doctest::Approx(2.0 * 301.0 * 0.04321).epsilon(1e-12));
}

// ----------------------------------------------------------------------------
// Error / failure paths — the branches golden tests never reach because they
// only ever feed well-identified, well-shaped, PD inputs. Each drives one
// `std::unexpected(...)` return so the error arms are exercised (and pinned).
// ----------------------------------------------------------------------------

TEST_CASE("information_expected: errors when θ̂ size ≠ evaluator n_free") {
  auto h = must_model("f =~ x1 + x2 + x3");   // 6 free params
  std::mt19937 rng(7);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs.push_back(200);

  Estimates est;
  est.theta = Eigen::VectorXd::Zero(2);   // deliberately wrong length
  auto r = magmaan::inference::information_expected(*h.pt, *h.rep, samp, est);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().detail.find("n_free") != std::string::npos);
}

TEST_CASE("information_expected: errors on SampleStats/evaluator block mismatch") {
  auto h = must_model("f =~ x1 + x2 + x3");   // single block
  std::mt19937 rng(8);
  SampleStats samp;                            // two blocks — model has one
  samp.S.push_back(random_pd(rng, 3));
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs = {200, 200};

  Estimates est;
  est.theta = Eigen::VectorXd::Zero(6);
  auto r = magmaan::inference::information_expected(*h.pt, *h.rep, samp, est);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("vcov: singular information matrix errors (InfoMatrixSingular)") {
  // No labels ⇒ no equality constraints ⇒ the plain `invert_spd(info)` arm,
  // including its LLT→LDLT fallback when the matrix isn't PD.
  auto h = must_model("f =~ x1 + x2 + x3");
  // An indefinite info (one negative eigenvalue) is not SPD, so invert_spd's
  // LLT→LDLT path returns InfoMatrixSingular. A rank-deficient *PSD* matrix can
  // slip past LDLT::isPositive(), and an all-zero one trips an Eigen debug
  // assert — an unambiguous negative eigenvalue avoids both pitfalls.
  Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6);
  info(0, 0) = -1.0;
  auto r = magmaan::inference::vcov(info, *h.pt);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == magmaan::PostError::Kind::InfoMatrixSingular);
  CHECK(r.error().detail.find("not invertible") != std::string::npos);
}

TEST_CASE("se: NaN on a negative diagonal, √ otherwise") {
  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(3, 3);
  v(0, 0) = 4.0;
  v(1, 1) = -1.0;   // negative variance ⇒ NaN, never throws
  v(2, 2) = 9.0;
  const Eigen::VectorXd s = magmaan::inference::se(v);
  CHECK(s(0) == doctest::Approx(2.0));
  CHECK(std::isnan(s(1)));
  CHECK(s(2) == doctest::Approx(3.0));
}

TEST_CASE("wald_test: shape-mismatch inputs all error") {
  Estimates est;
  est.theta = Eigen::VectorXd::Zero(3);
  const Eigen::MatrixXd vcov = Eigen::MatrixXd::Identity(3, 3);

  // Empty restriction matrix.
  {
    const Eigen::MatrixXd R(0, 3);
    const Eigen::VectorXd q(0);
    CHECK_FALSE(magmaan::inference::wald_test(R, q, est, vcov).has_value());
  }
  // R.cols() ≠ θ̂.size().
  {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 2);
    R(0, 0) = 1.0;
    const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
    CHECK_FALSE(magmaan::inference::wald_test(R, q, est, vcov).has_value());
  }
  // q.size() ≠ R.rows().
  {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 3);
    R(0, 0) = 1.0;
    const Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    CHECK_FALSE(magmaan::inference::wald_test(R, q, est, vcov).has_value());
  }
  // vcov shape ≠ θ̂.size().
  {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 3);
    R(0, 0) = 1.0;
    const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
    const Eigen::MatrixXd vbad = Eigen::MatrixXd::Identity(2, 2);
    CHECK_FALSE(magmaan::inference::wald_test(R, q, est, vbad).has_value());
  }
}

TEST_CASE("nt_moment_quadratic: SampleStats/ImpliedMoments block mismatch errors") {
  std::mt19937 rng(3);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs.push_back(100);
  magmaan::model::ImpliedMoments im;   // zero implied blocks
  auto r = magmaan::inference::frontier::nt_moment_quadratic(samp, im);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("browne_residual_adf: rejects missing-data RawData") {
  auto h = must_model("f =~ x1 + x2 + x3");

  magmaan::data::RawData raw;
  raw.X.emplace_back(200, 3);
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (Eigen::Index i = 0; i < raw.X[0].rows(); ++i) {
    raw.X[0](i, 0) = nd(rng);
    raw.X[0](i, 1) = 0.7 * raw.X[0](i, 0) + nd(rng);
    raw.X[0](i, 2) = 0.5 * raw.X[0](i, 0) + nd(rng);
  }

  // Derive sample stats + fit from the *complete* data first…
  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  auto est = magmaan::test::fit(*h.pt, *h.rep, *samp_or).value();

  // …then mark the data as having missingness and confirm ADF refuses it.
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(200, 3);
  M.setOnes();
  M(0, 0) = 0;   // one missing cell
  raw.mask.push_back(M);

  auto r = magmaan::inference::browne_residual_adf(*h.pt, *h.rep, *samp_or, raw, est);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().detail.find("missing-data") != std::string::npos);
}

TEST_CASE("vcov / df_stat: nonlinear `==` constraint takes the null-space branch") {
  // The linear (shared-label) constraint path is covered in constraints_test;
  // the nonlinear `a == b*b` path projects onto the null space of the
  // constraint Jacobian and needs θ̂. Fit the *unconstrained* model for a
  // valid θ̂ + PD info (identical parameter layout), then drive vcov/df_stat
  // with the constrained partable.
  std::mt19937 rng(99);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs.push_back(250);

  auto fp_u = magmaan::parse::Parser::parse("f =~ x1 + a*x2 + b*x3");
  REQUIRE(fp_u.has_value());
  auto pt_u  = magmaan::spec::build(*fp_u);            REQUIRE(pt_u.has_value());
  auto rep_u = magmaan::model::build_matrix_rep(*pt_u); REQUIRE(rep_u.has_value());
  auto est   = magmaan::test::fit(*pt_u, *rep_u, samp).value();
  auto info_or = magmaan::inference::information_expected(*pt_u, *rep_u, samp, est);
  REQUIRE(info_or.has_value());

  // Same formula + labels ⇒ identical θ ordering; the extra row is the
  // nonlinear constraint (no new free parameters).
  auto fp_c = magmaan::parse::Parser::parse("f =~ x1 + a*x2 + b*x3\na == b*b");
  REQUIRE(fp_c.has_value());
  auto pt_c = magmaan::spec::build(*fp_c);  REQUIRE(pt_c.has_value());
  REQUIRE(pt_c->nl_constraints.size() == 1);

  // (a) Omitting θ̂ on a nonlinearly-constrained model errors.
  auto miss = magmaan::inference::vcov(*info_or, *pt_c);
  REQUIRE_FALSE(miss.has_value());
  CHECK(miss.error().detail.find("third argument") != std::string::npos);

  // (b) With θ̂ the Z-null-space projection succeeds and stays symmetric.
  auto vc = magmaan::inference::vcov(*info_or, *pt_c, est.theta);
  REQUIRE_MESSAGE(vc.has_value(),
      "nonlinear vcov failed: " << (vc.has_value() ? "" : vc.error().detail));
  CHECK(vc->rows() == est.theta.size());
  CHECK(vc->cols() == est.theta.size());
  CHECK((*vc - vc->transpose()).cwiseAbs().maxCoeff() < 1e-9);

  // (c) df_stat: the nonlinear rank is added to the linear df, and θ̂ is
  // likewise required. Saturated 3-var 1F CFA ⇒ 6 moments − 6 free + 1
  // constraint rank = 1.
  REQUIRE_FALSE(magmaan::inference::df_stat(*pt_c, samp).has_value());
  auto df = magmaan::inference::df_stat(*pt_c, samp, est.theta);
  REQUIRE(df.has_value());
  CHECK(*df == 1);
}

// ----------------------------------------------------------------------------
// Observed information — FD and analytic variants.
// ----------------------------------------------------------------------------

namespace {

// Load (pt, rep, samp, est) from a fit fixture so we can drive any SE
// method at lavaan's θ̂ with the same S lavaan fit against. Bundles the
// boilerplate used by the observed-info tests below.
struct FixtureCtx {
  ModelHandles handles;
  SampleStats  samp;
  Estimates    est;
};

FixtureCtx load_fit_fixture(std::string_view model, const std::string& fixture_path) {
  FixtureCtx ctx{must_model(model), {}, {}};
  std::ifstream in(fixture_path);
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  ctx.samp.S.push_back(std::move(S));
  ctx.samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());

  const auto& th = j["theta_hat"];
  ctx.est.theta.resize(static_cast<Eigen::Index>(th.size()));
  for (Eigen::Index k = 0; k < ctx.est.theta.size(); ++k)
    ctx.est.theta(k) = th[static_cast<std::size_t>(k)].get<double>();
  ctx.est.fmin = j.contains("chi2") ?
      j["chi2"].get<double>() / static_cast<double>(ctx.samp.n_obs[0]) : 0.0;
  return ctx;
}

}  // namespace

TEST_CASE("fd_observed_inference: shape, symmetry, PSD at saturated 1F CFA") {
  auto ctx = load_fit_fixture("f =~ x1 + x2 + x3",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0001_one_factor_cfa.fit.json");

  auto inf_or = fd_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE(inf_or.has_value());
  const auto& inf = *inf_or;
  const Eigen::Index n_free = ctx.est.theta.size();

  CHECK(inf.info.rows() == n_free);
  CHECK(inf.se.size()   == n_free);
  // FD asymmetry is O(h² · ‖∂³F/∂θ³‖); with h=1e-4 a ~1e-6 floor is normal.
  // The implementation symmetrizes explicitly so the residual is the
  // floating-point error of `0.5 (H + Hᵀ)` itself — well under 1e-12.
  CHECK((inf.info - inf.info.transpose()).cwiseAbs().maxCoeff() < 1e-12);

  Eigen::LLT<Eigen::MatrixXd> llt(inf.info);
  CHECK(llt.info() == Eigen::Success);
  for (Eigen::Index k = 0; k < n_free; ++k) CHECK(inf.se(k) > 0.0);
}

TEST_CASE("analytic_observed_inference: shape, symmetry, PSD at saturated 1F CFA") {
  auto ctx = load_fit_fixture("f =~ x1 + x2 + x3",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0001_one_factor_cfa.fit.json");

  auto inf_or = analytic_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE(inf_or.has_value());
  const auto& inf = *inf_or;
  const Eigen::Index n_free = ctx.est.theta.size();

  CHECK(inf.info.rows() == n_free);
  CHECK(inf.se.size()   == n_free);
  CHECK((inf.info - inf.info.transpose()).cwiseAbs().maxCoeff() < 1e-12);

  Eigen::LLT<Eigen::MatrixXd> llt(inf.info);
  CHECK(llt.info() == Eigen::Success);
  for (Eigen::Index k = 0; k < n_free; ++k) CHECK(inf.se(k) > 0.0);
}

TEST_CASE("Observed info: FD ≈ analytic on 1F CFA (saturated)") {
  // At a saturated, converged fit the analytic Hessian is exact and the FD
  // approximation should agree to FD truncation/roundoff (~1e-5 with h=1e-4).
  auto ctx = load_fit_fixture("f =~ x1 + x2 + x3",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0001_one_factor_cfa.fit.json");

  auto fd_or = fd_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  auto an_or = analytic_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE(fd_or.has_value());
  REQUIRE(an_or.has_value());

  const double max_info_diff =
      (fd_or->info - an_or->info).cwiseAbs().maxCoeff();
  const double max_se_diff =
      (fd_or->se - an_or->se).cwiseAbs().maxCoeff();
  // Saturated 1F CFA has tiny info magnitudes; FD truncation noise dominates.
  CHECK(max_info_diff < 1e-3);
  CHECK(max_se_diff   < 1e-5);
}

TEST_CASE("Observed info: FD ≈ analytic on 3F Holzinger (non-saturated)") {
  // For an over-identified model both methods should still agree (analytic
  // is exact; FD has truncation error ~h²·‖∂³F/∂θ³‖).
  auto ctx = load_fit_fixture(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0002_three_factor_hs.fit.json");

  auto fd_or = fd_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  auto an_or = analytic_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));
  REQUIRE_MESSAGE(an_or.has_value(),
      "Analytic failed: " << (an_or.has_value() ? "" : an_or.error().detail));

  // Relative comparison on SE — info entries span several orders of
  // magnitude across rows so absolute comparison would be misleading.
  const Eigen::VectorXd rel = (fd_or->se - an_or->se).cwiseAbs().array() /
                              an_or->se.array().abs();
  CHECK(rel.maxCoeff() < 1e-4);
}

TEST_CASE("Observed info ≈ Expected info at saturated fit (1F CFA)") {
  // Saturated fit: S ≈ Σ̂, G ≈ 0, so the H2 correction and the difference
  // between H1's data-dependent term and the expected-info trace are both
  // O(‖S − Σ̂‖). Lavaan reports chi² ≈ 4e-13 for this fixture (n=301),
  // so F_ML ≈ 1.3e-15 and ‖S − Σ̂‖ ≈ √F_ML ≈ 4e-8. Info entries with N/2
  // scaling drift by ~ N · √F_ML, SEs by relative ~ √F_ML.
  auto ctx = load_fit_fixture("f =~ x1 + x2 + x3",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0001_one_factor_cfa.fit.json");

  auto exp_or = expected_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  auto an_or  = analytic_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE(exp_or.has_value());
  REQUIRE(an_or.has_value());

  // Relative SE diff, the meaningful comparison — SE entries vary by an
  // order of magnitude across params on this fixture.
  const Eigen::VectorXd rel_se =
      (exp_or->se - an_or->se).cwiseAbs().array() / an_or->se.array().abs();
  CHECK(rel_se.maxCoeff() < 1e-6);
}

TEST_CASE("z_test: per-parameter z = θ̂_k / SE_k and chi²(1) p-value") {
  // Synthetic inputs — verify the closed form.
  Estimates est;
  est.theta = Eigen::VectorXd(3);
  est.theta << 1.0, 0.5, -2.0;
  Eigen::VectorXd se_v(3);
  se_v << 0.25, 0.0, 1.0;
  const auto zt = magmaan::inference::z_test(est, se_v);
  CHECK(zt.z(0) == doctest::Approx(4.0));
  CHECK(zt.p_value(0) ==
        doctest::Approx(magmaan::inference::chi2_pvalue(16.0, 1)).epsilon(1e-12));
  CHECK(std::isnan(zt.z(1)));        // SE = 0
  CHECK(std::isnan(zt.p_value(1)));
  CHECK(zt.z(2) == doctest::Approx(-2.0));
  CHECK(zt.p_value(2) ==
        doctest::Approx(magmaan::inference::chi2_pvalue(4.0, 1)).epsilon(1e-12));
}

TEST_CASE("wald_test: single-parameter restriction matches (θ̂_k / SE_k)²") {
  // For a single linear restriction `θ_k = 0`, the Wald statistic
  // reduces to (θ̂_k / SE_k)². Verify on the 1F CFA saturated fit.
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S = {S}; samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  auto est = magmaan::test::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

  // Test θ_0 = 0 (first free param). R = [1, 0, 0, ...], q = [0].
  const Eigen::Index n_free = est.theta.size();
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, n_free);
  R(0, 0) = 1.0;
  Eigen::VectorXd q(1); q(0) = 0.0;

  auto wald_or = magmaan::inference::wald_test(R, q, est, inf.vcov);
  REQUIRE(wald_or.has_value());
  CHECK(wald_or->df == 1);
  const double expected = (est.theta(0) / inf.se(0)) * (est.theta(0) / inf.se(0));
  CHECK(wald_or->chi2 == doctest::Approx(expected).epsilon(1e-10));
}

TEST_CASE("wald_test: rank-deficient R errors out") {
  // R has 2 rows that are identical → R · vcov · Rᵀ is rank-1, not invertible.
  Estimates est;
  est.theta = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd vcov = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd R(2, 3);
  R << 1, 0, 0,
       1, 0, 0;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
  auto w = magmaan::inference::wald_test(R, q, est, vcov);
  REQUIRE_FALSE(w.has_value());
}

TEST_CASE("chi2_pvalue: classic spot checks against R's pchisq") {
  using magmaan::inference::chi2_pvalue;
  // pchisq(3.84, 1, lower.tail=FALSE) ≈ 0.05
  CHECK(chi2_pvalue(3.8414588, 1) == doctest::Approx(0.05).epsilon(1e-5));
  // pchisq(7.815, 3, lower.tail=FALSE) ≈ 0.05
  CHECK(chi2_pvalue(7.8147279, 3) == doctest::Approx(0.05).epsilon(1e-5));
  // pchisq(24, 24, lower.tail=FALSE) ≈ 0.4615869 (R's value)
  CHECK(chi2_pvalue(24.0, 24) == doctest::Approx(0.4615869).epsilon(1e-5));
  // Edge cases
  CHECK(chi2_pvalue(0.0, 5) == doctest::Approx(1.0));
  CHECK(std::isnan(chi2_pvalue(5.0, 0)));
  CHECK(std::isnan(chi2_pvalue(-1.0, 3)));
  // Very large chi2 → p ≈ 0 (asymptote).
  CHECK(chi2_pvalue(100.0, 5) < 1e-15);
}

TEST_CASE("noncentral_chisq_cdf: spot checks against R's pchisq(x, df, ncp)") {
  using magmaan::inference::chi2_pvalue;
  using magmaan::inference::noncentral_chisq_cdf;

  // ncp == 0 ⇒ central χ²(df) CDF (= 1 − chi2_pvalue) for several (x, df).
  for (auto [x, df] : {std::pair{2.0, 3}, std::pair{12.5, 7},
                       std::pair{40.0, 24}, std::pair{0.7, 1}}) {
    CHECK(noncentral_chisq_cdf(x, static_cast<double>(df), 0.0) ==
          doctest::Approx(1.0 - chi2_pvalue(x, df)).epsilon(1e-12));
  }

  // pchisq(x, df, ncp) — values from R 4.x.
  CHECK(noncentral_chisq_cdf(5.0, 3.0, 0.0)   == doctest::Approx(0.828202855703).epsilon(1e-9));
  CHECK(noncentral_chisq_cdf(10.0, 3.0, 4.0)  == doctest::Approx(0.775921995699).epsilon(1e-9));
  CHECK(noncentral_chisq_cdf(20.0, 5.0, 10.0) == doctest::Approx(0.781070388285).epsilon(1e-9));
  CHECK(noncentral_chisq_cdf(2.5, 7.0, 12.0)  == doctest::Approx(0.000741366913154).epsilon(1e-7));
  CHECK(noncentral_chisq_cdf(0.001, 2.0, 3.0) == doctest::Approx(0.000111579021642).epsilon(1e-7));
  // Large ncp: the mode-centered summation must stay accurate (the j=0
  // Poisson weight underflows here).
  CHECK(noncentral_chisq_cdf(2010.0, 10.0, 2000.0) == doctest::Approx(0.504451319691).epsilon(1e-7));
  CHECK(noncentral_chisq_cdf(100.0, 10.0, 2000.0)  < 1e-200);

  // Monotone decreasing in ncp; in [0, 1].
  const double a = noncentral_chisq_cdf(15.0, 5.0, 1.0);
  const double b = noncentral_chisq_cdf(15.0, 5.0, 8.0);
  const double c = noncentral_chisq_cdf(15.0, 5.0, 30.0);
  CHECK(a > b);
  CHECK(b > c);
  CHECK(c >= 0.0);
  CHECK(a <= 1.0);

  // Bad / boundary inputs.
  CHECK(std::isnan(noncentral_chisq_cdf(5.0, 0.0, 1.0)));
  CHECK(std::isnan(noncentral_chisq_cdf(5.0, 3.0, -1.0)));
  CHECK(noncentral_chisq_cdf(0.0, 3.0, 4.0) == 0.0);
  CHECK(noncentral_chisq_cdf(-2.0, 3.0, 4.0) == 0.0);
}

TEST_CASE("Multi-group + mean structure: θ̂/SE match lavaan on HS × school") {
  // Saturated 1F CFA fit across the HolzingerSwineford1939 `school`
  // grouping with `meanstructure = TRUE`. Reference numbers below come
  // from running lavaan offline (lavaan 0.6.22.2560):
  //   fit <- cfa("f =~ x1 + x2 + x3", data = HolzingerSwineford1939,
  //              group = "school", std.lv = FALSE)
  //
  // Per-block sample stats from lavInspect(fit, "sampstat"):
  Eigen::MatrixXd S_pasteur(3, 3);
  S_pasteur << 1.395229515982678, 0.402103192488289, 0.620106704192554,
               0.402103192488289, 1.503749589086128, 0.476757684089415,
               0.620106704192554, 0.476757684089415, 1.345789160092045;
  Eigen::VectorXd mean_pasteur(3);
  mean_pasteur << 4.94123931326923, 5.98397435897436, 2.48717948717949;

  Eigen::MatrixXd S_grant(3, 3);
  S_grant << 1.318647108981901, 0.414310343344828, 0.533749506034483,
             0.414310343344828, 1.226379310344828, 0.478448275862069,
             0.533749506034483, 0.478448275862069, 1.073365041617122;
  Eigen::VectorXd mean_grant(3);
  mean_grant << 4.92988506000000, 6.20000000000000, 1.99568965517241;

  SampleStats samp;
  samp.S     = {S_pasteur, S_grant};
  samp.mean  = {mean_pasteur, mean_grant};
  samp.n_obs = {156, 145};

  // lavaan free-index order within each group:
  //   λ_2, λ_3, θ_1, θ_2, θ_3, ψ, ν_1, ν_2, ν_3
  // Group 1's indices 1-9, group 2's 10-18.
  Eigen::VectorXd theta_lavaan(18);
  theta_lavaan <<
      0.768831669613, 1.185659990272, 0.872224012300, 1.194599907699,
      0.610553442096, 0.523005504066, 4.941239313269, 5.983974358974,
      2.487179487179,
      0.896391066253, 1.154806495919, 0.856448907961, 0.854995234218,
      0.456987633405, 0.462198223430, 4.929885060000, 6.200000000000,
      1.995689655172;
  Eigen::VectorXd se_lavaan(18);
  se_lavaan <<
      0.1958753831134, 0.3309093565655, 0.1706105514509, 0.1582979150226,
      0.2074325764326, 0.1858982644590, 0.0945715546614, 0.0981805498780,
      0.0928808565124,
      0.2002435870686, 0.2807012729274, 0.1447100733691, 0.1306567955377,
      0.1487605952885, 0.1571313388695, 0.0953630858746, 0.0919662358609,
      0.0860378840252;

  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.n_groups      = 2;
  opts.meanstructure = true;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto est_or = magmaan::test::fit(*pt, *mr, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;
  REQUIRE(est.theta.size() == 18);

  // θ̂ within 1e-5 of lavaan.
  const double max_theta_diff =
      (est.theta - theta_lavaan).cwiseAbs().maxCoeff();
  CHECK(max_theta_diff < 1e-5);

  // expected info → SEs within 1e-4 of lavaan.
  auto inf_or = expected_inference(*pt, *mr, samp, est);
  REQUIRE_MESSAGE(inf_or.has_value(),
      "expected_inference failed: " <<
      (inf_or.has_value() ? "" : inf_or.error().detail));
  const double max_se_diff =
      (inf_or->se - se_lavaan).cwiseAbs().maxCoeff();
  CHECK(max_se_diff < 1e-4);

  // df = 0 (saturated: 2 blocks × (6 cov + 3 mean) moments = 18 moments,
  // and 18 free params).
  CHECK(inf_or->df == 0);
  // chi² ≈ 0 at the saturated fit; lavaan reports ~1e-13.
  CHECK(inf_or->chi2 < 1e-6);
}

TEST_CASE("Multi-group: lavaanify + matrix_rep + fit → end-to-end 2-group CFA") {
  // 1F CFA with `n_groups = 2`. Each group has the same S (so the joint
  // optimum coincides with the single-group optimum) and independent
  // parameters (no labels → no cross-group equality). Verify:
  //   (a) lavaanify produces 2 blocks of rows with separate free indices.
  //   (b) build_matrix_rep produces 2 blocks of dims/ov_names/lv_names.
  //   (c) fit() converges and each block's θ̂ matches the single-block fit.
  //   (d) Both expected and FD-observed information return PD info matrices.

  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());

  magmaan::spec::BuildOptions opts;
  opts.n_groups = 2;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  // (a) LatentStructure structure
  std::int32_t n_b1 = 0, n_b2 = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->group[i] == 1) ++n_b1;
    if (pt->group[i] == 2) ++n_b2;
  }
  CHECK(n_b1 == 7);          // 3 =~ + 3 θ + 1 ψ
  CHECK(n_b2 == 7);
  CHECK(pt->n_free() == 12); // 6 free per group × 2 (configural)

  // (b) MatrixRep structure
  REQUIRE(mr->dims.size() == 2);
  REQUIRE(mr->ov_names.size() == 2);
  CHECK(mr->dims[0].n_observed == 3);
  CHECK(mr->dims[1].n_observed == 3);
  CHECK(mr->ov_names[0] == mr->ov_names[1]);   // same vars

  // Build a 2-block SampleStats from the saturated 1F CFA fixture: same
  // S in both blocks → both groups should land at the same θ̂.
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  const auto n = j["n_obs"].get<std::int64_t>();

  SampleStats samp_mg;
  samp_mg.S     = {S, S};
  samp_mg.n_obs = {n, n};

  auto est_mg_or = magmaan::test::fit(*pt, *mr, samp_mg);
  REQUIRE_MESSAGE(est_mg_or.has_value(),
      "multi-group fit failed: " <<
      (est_mg_or.has_value() ? "" : est_mg_or.error().detail));
  const auto& est_mg = *est_mg_or;
  CHECK(est_mg.theta.size() == 12);

  // (c) Each group's 6 params should match the single-block θ̂.
  auto fp_single = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp_single.has_value());
  auto pt_single = magmaan::spec::build(*fp_single);
  REQUIRE(pt_single.has_value());
  auto mr_single = magmaan::model::build_matrix_rep(*pt_single);
  REQUIRE(mr_single.has_value());
  SampleStats samp_single;
  samp_single.S = {S};  samp_single.n_obs = {n};
  auto est_single = magmaan::test::fit(*pt_single, *mr_single, samp_single).value();

  const double max_g1 = (est_mg.theta.head(6) - est_single.theta).cwiseAbs().maxCoeff();
  const double max_g2 = (est_mg.theta.tail(6) - est_single.theta).cwiseAbs().maxCoeff();
  CHECK(max_g1 < 1e-5);
  CHECK(max_g2 < 1e-5);

  // (d) Inference on multi-group fit
  auto exp_or = expected_inference(*pt, *mr, samp_mg, est_mg);
  REQUIRE_MESSAGE(exp_or.has_value(),
      "expected_inference failed: " <<
      (exp_or.has_value() ? "" : exp_or.error().detail));
  CHECK(exp_or->info.rows() == 12);
  // Block-diagonal structure: parameters in different groups have zero info
  // off-diagonal (no shared params in configural model).
  const auto top_right = exp_or->info.topRightCorner(6, 6);
  CHECK(top_right.cwiseAbs().maxCoeff() < 1e-10);

  auto fd_or = fd_observed_inference(*pt, *mr, samp_mg, est_mg);
  REQUIRE_MESSAGE(fd_or.has_value(),
      "fd_observed_inference failed: " <<
      (fd_or.has_value() ? "" : fd_or.error().detail));
  CHECK(fd_or->info.rows() == 12);
}

TEST_CASE("expected_inference on mean-structure CFA: ν SEs match closed form") {
  // 1F CFA + intercepts, saturated fit (ν̂_i = m̄_i, df = 0).
  // Closed-form expected-info result for ν: I[ν, ν] = n · Σ̂⁻¹.
  // vcov[ν, ν] = Σ̂ / n, so SE(ν_i) = √(Σ̂_ii / n).
  auto h = must_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");

  std::mt19937 rng(2026);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 4.2, 5.7, 6.1;
  SampleStats samp;
  samp.S = {S};  samp.mean = {mean};  samp.n_obs = {301};

  auto est_or = magmaan::test::fit(*h.pt, *h.rep, samp);
  REQUIRE(est_or.has_value());

  auto inf_or = expected_inference(*h.pt, *h.rep, samp, *est_or);
  REQUIRE_MESSAGE(inf_or.has_value(),
      "expected_inference failed: " <<
          (inf_or.has_value() ? "" : inf_or.error().detail));
  const auto& inf = *inf_or;
  const Eigen::Index n_free = est_or->theta.size();
  CHECK(inf.info.rows() == n_free);
  CHECK((inf.info - inf.info.transpose()).cwiseAbs().maxCoeff() < 1e-10);

  // ν params: SE = √(Σ̂_ii / n). At the saturated fit Σ̂ = S exactly.
  auto ev = magmaan::model::ModelEvaluator::build(*h.pt, *h.rep).value();
  const auto locs = ev.param_locations();
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Nu) {
      const auto i = locs[k].row;
      const double se_expected = std::sqrt(S(i, i) / 301.0);
      CHECK(inf.se(static_cast<Eigen::Index>(k)) ==
            doctest::Approx(se_expected).epsilon(1e-6));
    }
  }
}

TEST_CASE("fd_observed_inference on mean-structure ≈ expected_inference at saturated") {
  // At a saturated fit (S = Σ̂, m̄ = μ̂), d = 0 so the H1 cov term reduces
  // to tr(WMaWMb) and the mean Hessian to 2·ν_a' W ν_b — exactly what
  // expected information returns. FD and expected should match to FD truncation.
  auto h = must_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");

  std::mt19937 rng(99);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 1.0, -0.5, 2.3;
  SampleStats samp;
  samp.S = {S};  samp.mean = {mean};  samp.n_obs = {200};

  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();

  auto exp_or = expected_inference(*h.pt, *h.rep, samp, est);
  auto fd_or  = fd_observed_inference(*h.pt, *h.rep, samp, est);
  REQUIRE_MESSAGE(exp_or.has_value(),
      "expected_inference failed: " <<
          (exp_or.has_value() ? "" : exp_or.error().detail));
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));

  const Eigen::VectorXd rel = (fd_or->se - exp_or->se).cwiseAbs().array() /
                              exp_or->se.array().abs();
  CHECK(rel.maxCoeff() < 1e-4);
}

TEST_CASE("information_observed_analytic matches FD for mean structure") {
  auto h = must_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");

  std::mt19937 rng(1);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 1.0, 2.0, 3.0;
  SampleStats samp;
  samp.S = {S};  samp.mean = {mean};  samp.n_obs = {100};

  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();
  auto an_or = magmaan::inference::information_observed_analytic(*h.pt, *h.rep, samp, est);
  auto fd_or = magmaan::inference::information_observed_fd(*h.pt, *h.rep, samp, est);
  REQUIRE_MESSAGE(an_or.has_value(),
      "Analytic failed: " << (an_or.has_value() ? "" : an_or.error().detail));
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));

  const Eigen::MatrixXd diff = *an_or - *fd_or;
  const double scale = std::max(1.0, fd_or->cwiseAbs().maxCoeff());
  CHECK(diff.cwiseAbs().maxCoeff() / scale < 1e-4);
}

TEST_CASE("information_observed_analytic matches FD for structural latent means") {
  auto h = must_model("f =~ x1 + x2 + x3\ny ~ f\nf ~ 1\ny ~ 1\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");

  std::mt19937 rng(2027);
  Eigen::MatrixXd S = random_pd(rng, 4);
  Eigen::VectorXd mean(4);  mean << 1.0, 2.0, 3.0, 1.7;
  SampleStats samp;
  samp.S = {S};  samp.mean = {mean};  samp.n_obs = {180};

  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();
  auto an_or = magmaan::inference::information_observed_analytic(*h.pt, *h.rep, samp, est);
  auto fd_or = magmaan::inference::information_observed_fd(*h.pt, *h.rep, samp, est);
  REQUIRE_MESSAGE(an_or.has_value(),
      "Analytic failed: " << (an_or.has_value() ? "" : an_or.error().detail));
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));

  const Eigen::MatrixXd diff = *an_or - *fd_or;
  const double scale = std::max(1.0, fd_or->cwiseAbs().maxCoeff());
  CHECK(diff.cwiseAbs().maxCoeff() / scale < 2e-4);
}

TEST_CASE("Observed info: FD ≈ analytic on path analysis (Reduced LISREL)") {
  // Pure path: `x9 ~ x1 + x2 + x3`. Exercises (B, B), (Λ, B), (Ψ, B)
  // cases of analytic ∂²Σ — none of which fire for Pure CFA. Cross-check
  // that the closed-form Reduced cases agree with FD.
  auto ctx = load_fit_fixture("x9 ~ x1 + x2 + x3",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0019_path_hs.fit.json");

  auto fd_or = fd_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  auto an_or = analytic_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));
  REQUIRE_MESSAGE(an_or.has_value(),
      "Analytic failed: " << (an_or.has_value() ? "" : an_or.error().detail));

  const Eigen::VectorXd rel = (fd_or->se - an_or->se).cwiseAbs().array() /
                              an_or->se.array().abs();
  CHECK(rel.maxCoeff() < 1e-4);
}

namespace {

// Lavaanify doesn't generate multi-group ParTables yet (v0 is single-group),
// so to exercise the multi-block inference paths we duplicate an already-
// built (pt, rep) into a 2-block version: block 0 = original, block 1 =
// identical structure with a fresh slice of free-parameter indices.
struct TwoBlockHandles {
  magmaan::spec::LatentStructure* pt;
  magmaan::model::MatrixRep*   rep;
  std::size_t                n_free_single;
};

TwoBlockHandles duplicate_two_blocks(const ModelHandles& src) {
  using namespace magmaan::spec;
  using namespace magmaan::model;
  static thread_local LatentStructure  s_pt;
  static thread_local MatrixRep s_rep;
  s_pt  = *src.pt;
  s_rep = *src.rep;

  const std::int32_t n_free_single =
      static_cast<std::int32_t>(src.pt->n_free());
  const std::size_t  orig_size  = src.pt->size();

  // Append block-1 LatentStructure rows: independent free indices, group = 2.
  // The variable table (n_vars / var_role / orderings) is shared — already
  // copied above — so the appended rows reuse the same var ids.
  for (std::size_t i = 0; i < orig_size; ++i) {
    s_pt.op.push_back(src.pt->op[i]);
    s_pt.group.push_back(2);
    s_pt.free.push_back(src.pt->free[i] > 0
                            ? src.pt->free[i] + n_free_single
                            : 0);
    s_pt.exo.push_back(src.pt->exo[i]);
    s_pt.fixed_value.push_back(src.pt->fixed_value[i]);
    s_pt.lhs_var.push_back(src.pt->lhs_var[i]);
    s_pt.rhs_var.push_back(src.pt->rhs_var[i]);
  }
  // Identity equality reparameterization over the doubled free set.
  s_pt.eq_groups.resize(static_cast<std::size_t>(2 * n_free_single));
  for (std::int32_t k = 0; k < 2 * n_free_single; ++k)
    s_pt.eq_groups[static_cast<std::size_t>(k)] = k;
  s_pt.has_unenforced_constraints = false;

  // Mirror MatrixRep: block-1 cells point at the same (mat, row, col)
  // but with block index 1 so ModelEvaluator writes to a separate
  // per-block buffer.
  for (std::size_t i = 0; i < orig_size; ++i) {
    Cell c = src.rep->cell_for_row[i];
    c.block = 1;
    s_rep.cell_for_row.push_back(c);
  }
  for (const auto& sc : src.rep->structural_cells) {
    StructuralCell sc2 = sc;
    sc2.block = 1;
    s_rep.structural_cells.push_back(sc2);
  }
  s_rep.dims.push_back(src.rep->dims[0]);
  s_rep.ov_names.push_back(src.rep->ov_names[0]);
  s_rep.lv_names.push_back(src.rep->lv_names[0]);

  return {&s_pt, &s_rep, static_cast<std::size_t>(n_free_single)};
}

}  // namespace

TEST_CASE("Inference: multi-block infrastructure (synthetic 2-block 1F CFA)") {
  // Duplicate the 1F CFA into 2 independent identical blocks and run
  // both observed-info methods. Verify:
  //   (a) info matrix is 2n_free × 2n_free.
  //   (b) cross-block off-diagonal is exactly zero (no shared params).
  //   (c) each diagonal block matches single-block analytic info.
  //   (d) FD multi-block ≈ analytic multi-block.
  auto single = must_model("f =~ x1 + x2 + x3");
  auto two    = duplicate_two_blocks(single);

  // Load single-block S and n_obs from the saturated fixture.
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  const auto n = j["n_obs"].get<std::int64_t>();

  // Build a 2-block SampleStats with the same S on both blocks.
  SampleStats samp_two;
  samp_two.S = {S, S};
  samp_two.n_obs = {n, n};

  // θ̂ for both blocks = lavaan's single-block θ̂.
  const Eigen::VectorXd theta_single = theta_from_fixture(
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0001_one_factor_cfa.fit.json");
  Estimates est_two;
  est_two.theta.resize(2 * theta_single.size());
  est_two.theta.head(theta_single.size()) = theta_single;
  est_two.theta.tail(theta_single.size()) = theta_single;
  est_two.fmin = 0.0;

  auto an_or = analytic_observed_inference(*two.pt, *two.rep, samp_two, est_two);
  auto fd_or = fd_observed_inference(*two.pt, *two.rep, samp_two, est_two);
  REQUIRE_MESSAGE(an_or.has_value(),
      "analytic failed: " << (an_or.has_value() ? "" : an_or.error().detail));
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));

  const Eigen::Index n_free_single = static_cast<Eigen::Index>(two.n_free_single);
  const Eigen::Index n_free_two    = 2 * n_free_single;

  // (a) shape
  CHECK(an_or->info.rows() == n_free_two);
  CHECK(fd_or->info.rows() == n_free_two);

  // (b) cross-block off-diagonal is exactly zero — params are independent
  // between blocks so ∂²F/∂θ_a ∂θ_b = 0 whenever a, b live in different
  // blocks.
  const auto top_right = an_or->info.topRightCorner(n_free_single, n_free_single);
  const auto bot_left  = an_or->info.bottomLeftCorner(n_free_single, n_free_single);
  CHECK(top_right.cwiseAbs().maxCoeff() < 1e-12);
  CHECK(bot_left.cwiseAbs().maxCoeff()  < 1e-12);

  // (c) each diagonal block matches the single-block info at the same θ̂.
  Estimates est_single;
  est_single.theta = theta_single;
  est_single.fmin  = 0.0;
  SampleStats samp_single;  samp_single.S = {S};  samp_single.n_obs = {n};
  auto an_single_or = analytic_observed_inference(*single.pt, *single.rep, samp_single, est_single);
  REQUIRE(an_single_or.has_value());

  const auto& info_2 = an_or->info;
  const auto top_left  = info_2.topLeftCorner(n_free_single, n_free_single);
  const auto bot_right = info_2.bottomRightCorner(n_free_single, n_free_single);
  const double diff_tl =
      (top_left - an_single_or->info).cwiseAbs().maxCoeff();
  const double diff_br =
      (bot_right - an_single_or->info).cwiseAbs().maxCoeff();
  CHECK(diff_tl < 1e-9);
  CHECK(diff_br < 1e-9);

  // (d) FD multi-block ≈ analytic multi-block on SE.
  const Eigen::VectorXd rel = (fd_or->se - an_or->se).cwiseAbs().array() /
                              an_or->se.array().abs();
  CHECK(rel.maxCoeff() < 1e-4);
}

TEST_CASE("Observed info: FD ≈ analytic on CFA + structural (Reduced)") {
  // 0020 mixes Λ (loadings), Β (latent-on-latent regression), Ψ, Θ —
  // the full LISREL machinery and every nonzero (·,·) ∂²Σ case.
  auto ctx = load_fit_fixture(
      "visual =~ x1 + x2 + x3\nx9 ~ visual",
      std::string(MAGMAAN_FIXTURES_DIR) + "/fit/0020_cfa_plus_structural_hs.fit.json");

  auto fd_or = fd_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  auto an_or = analytic_observed_inference(*ctx.handles.pt, *ctx.handles.rep, ctx.samp, ctx.est);
  REQUIRE_MESSAGE(fd_or.has_value(),
      "FD failed: " << (fd_or.has_value() ? "" : fd_or.error().detail));
  REQUIRE_MESSAGE(an_or.has_value(),
      "Analytic failed: " << (an_or.has_value() ? "" : an_or.error().detail));

  const Eigen::VectorXd rel = (fd_or->se - an_or->se).cwiseAbs().array() /
                              an_or->se.array().abs();
  CHECK(rel.maxCoeff() < 1e-4);
}

TEST_CASE("noncentral_chisq_cdf: retained ML2S tail work is unavailable") {
  using magmaan::inference::noncentral_chisq_cdf;
  // Owning experiment 15, optimizer-confirm-v3/fit_43.rds: fmin =
  // 925720248854.02063, N = 130, df = 2. chi2_stat = 2*N*fmin.
  // Old CDF at the first lower-interval midpoint took 15.68 seconds and
  // returned 0.613927 instead of a trustworthy numerical value.
  constexpr double x = 240687264702045.38;
  CHECK(std::isnan(noncentral_chisq_cdf(x, 2.0, x / 2.0)));
  CHECK(std::isnan(noncentral_chisq_cdf(x, 2.0, x)));
  CHECK(std::isnan(noncentral_chisq_cdf(x, 2.0, 4.0 * x)));
  // The same explicit inference call's scaled statistic also enters the
  // robust interval family; retain its endpoint and bisection probes.
  constexpr double scaled_x = 28343200981398004.0;
  CHECK(std::isnan(noncentral_chisq_cdf(scaled_x, 2.0, scaled_x / 2.0)));
  CHECK(std::isnan(noncentral_chisq_cdf(scaled_x, 2.0, scaled_x)));
  CHECK(std::isnan(noncentral_chisq_cdf(scaled_x, 2.0, 4.0 * scaled_x)));
  CHECK(std::isnan(noncentral_chisq_cdf(1.0, 2.0, 1e300)));
  CHECK(noncentral_chisq_cdf(x, 2.0, 0.0) == 1.0);
}

TEST_CASE("noncentral_chisq_cdf: 60-digit Poisson mixture grid") {
  // mpmath 1.3.0, 60 digits: regularized gammainc Poisson mixture,
  // summing until weights < 1e-55, reanchoring gammainc every 256 terms.
  // Seven spot checks with direct gammainc at every term agree at ncp 0..1e5.
  // df = {1,2,10,100,500}, ncp = {0,.1,1,10,100,1000,1e4,1e5};
  // x = max(1e-8, df+ncp + z*sqrt(2*(df+2*ncp))), z = {-12,-6,-2,0,2,6,12}.
  struct Reference { double x, df, ncp, cdf; };
  constexpr Reference references[] = {
      {1e-08, 1.0, 0.0, 7.978845594730578e-05},
      {1e-08, 1.0, 0.0, 7.978845594730578e-05},
      {1e-08, 1.0, 0.0, 7.978845594730578e-05},
      {1.0, 1.0, 0.0, 0.6826894921370859},
      {3.8284271247461903, 1.0, 0.0, 0.9496098015056764},
      {9.485281374238571, 1.0, 0.0, 0.997928731546161},
      {17.970562748477143, 1.0, 0.0, 0.9999775652332464},
      {1e-08, 1.0, 0.1, 7.589712704520577e-05},
      {1e-08, 1.0, 0.1, 7.589712704520577e-05},
      {1e-08, 1.0, 0.1, 7.589712704520577e-05},
      {1.1, 1.0, 0.1, 0.6819722423595479},
      {4.198386676965933, 1.0, 0.1, 0.9494220316498043},
      {10.3951600308978, 1.0, 0.1, 0.9979810457874655},
      {19.690320061795603, 1.0, 0.1, 0.9999801515897883},
      {1e-08, 1.0, 1.0, 4.839414490382867e-05},
      {1e-08, 1.0, 1.0, 4.839414490382867e-05},
      {1e-08, 1.0, 1.0, 4.839414490382867e-05},
      {2.0, 1.0, 1.0, 0.6527565366822697},
      {6.898979485566356, 1.0, 1.0, 0.9479443939735372},
      {16.696938456699066, 1.0, 1.0, 0.9989861169171981},
      {31.393876913398135, 1.0, 1.0, 0.9999979179778227},
      {1e-08, 1.0, 10.0, 5.376103962719851e-07},
      {1e-08, 1.0, 10.0, 5.376103962719851e-07},
      {1e-08, 1.0, 10.0, 5.376103962719851e-07},
      {11.0, 1.0, 10.0, 0.5613319809529725},
      {23.96148139681572, 1.0, 10.0, 0.9584316288641356},
      {49.88444419044716, 1.0, 10.0, 0.9999520255628969},
      {88.76888838089432, 1.0, 10.0, 0.9999999998068271},
      {1e-08, 1.0, 100.0, 1.5389199792630507e-26},
      {1e-08, 1.0, 100.0, 1.5389199792630507e-26},
      {60.900124688473156, 1.0, 100.0, 0.014040718700756467},
      {101.0, 1.0, 100.0, 0.5198892476769775},
      {141.09987531152683, 1.0, 100.0, 0.9698468029922818},
      {221.29962593458055, 1.0, 100.0, 0.9999994590985805},
      {341.5992518691611, 1.0, 100.0, 1.0},
      {241.86364861113395, 1.0, 1000.0, 2.0435347684182855e-58},
      {621.431824305567, 1.0, 1000.0, 1.0839600532288629e-11},
      {874.4772747685223, 1.0, 1000.0, 0.020123022690594877},
      {1001.0, 1.0, 1000.0, 0.5063059925157216},
      {1127.5227252314776, 1.0, 1000.0, 0.9747573239117976},
      {1380.568175694433, 1.0, 1000.0, 0.9999999842798757},
      {1760.136351388866, 1.0, 1000.0, 1.0},
      {7600.940000749981, 1.0, 10000.0, 6.616398505089929e-38},
      {8800.970000374991, 1.0, 10000.0, 3.075445093056053e-10},
      {9600.990000124997, 1.0, 10000.0, 0.021933557799646675},
      {10001.0, 1.0, 10000.0, 0.5019946532260723},
      {10401.009999875003, 1.0, 10000.0, 0.976446787542684},
      {11201.029999625009, 1.0, 10000.0, 0.9999999973091587},
      {12401.05999925002, 1.0, 10000.0, 1.0},
      {92411.51464195365, 1.0, 100000.0, 1.0087418950784455e-34},
      {96206.25732097682, 1.0, 100000.0, 6.958757641670926e-10},
      {98736.08577365894, 1.0, 100000.0, 0.022493356614672572},
      {100001.0, 1.0, 100000.0, 0.5006307812907309},
      {101265.91422634106, 1.0, 100000.0, 0.9769944424514757},
      {103795.74267902318, 1.0, 100000.0, 0.9999999986236167},
      {107590.48535804635, 1.0, 100000.0, 1.0},
      {1e-08, 2.0, 0.0, 4.9999999875e-09},
      {1e-08, 2.0, 0.0, 4.9999999875e-09},
      {1e-08, 2.0, 0.0, 4.9999999875e-09},
      {2.0, 2.0, 0.0, 0.6321205588285577},
      {6.0, 2.0, 0.0, 0.950212931632136},
      {14.0, 2.0, 0.0, 0.9990881180344455},
      {26.0, 2.0, 0.0, 0.999997739670593},
      {1e-08, 2.0, 0.1, 4.756147111207721e-09},
      {1e-08, 2.0, 0.1, 4.756147111207721e-09},
      {1e-08, 2.0, 0.1, 4.756147111207721e-09},
      {2.1, 2.0, 0.1, 0.6319076528175414},
      {6.295235392680606, 2.0, 0.1, 0.950189892615454},
      {14.68570617804182, 2.0, 0.1, 0.9990990529832318},
      {27.27141235608364, 2.0, 0.1, 0.9999978618563108},
      {1e-08, 2.0, 1.0, 3.0326532947723506e-09},
      {1e-08, 2.0, 1.0, 3.0326532947723506e-09},
      {1e-08, 2.0, 1.0, 3.0326532947723506e-09},
      {3.0, 2.0, 1.0, 0.6206436532195436},
      {8.65685424949238, 2.0, 1.0, 0.9502736819224945},
      {19.970562748477143, 2.0, 1.0, 0.9994147439249161},
      {36.941125496954285, 2.0, 1.0, 0.9999995120189255},
      {1e-08, 2.0, 10.0, 3.368973533232469e-11},
      {1e-08, 2.0, 10.0, 3.368973533232469e-11},
      {1e-08, 2.0, 10.0, 3.368973533232469e-11},
      {12.0, 2.0, 10.0, 0.5589920829003435},
      {25.2664991614216, 2.0, 10.0, 0.9589601150388225},
      {51.7994974842648, 2.0, 10.0, 0.9999579363493971},
      {91.5989949685296, 2.0, 10.0, 0.9999999998706108},
      {1e-08, 2.0, 100.0, 9.643750421178917e-31},
      {1e-08, 2.0, 100.0, 9.643750421178917e-31},
      {61.80049751551644, 2.0, 100.0, 0.014079970124288261},
      {102.0, 2.0, 100.0, 0.5198071667188149},
      {142.19950248448356, 2.0, 100.0, 0.9698744953204945},
      {222.59850745345068, 2.0, 100.0, 0.9999994669549118},
      {343.19701490690136, 2.0, 100.0, 1.0},
      {242.67398306129394, 2.0, 1000.0, 2.1782898765670478e-58},
      {622.336991530647, 2.0, 1000.0, 1.08681451344555e-11},
      {875.445663843549, 2.0, 1000.0, 0.020124149653207496},
      {1002.0, 2.0, 1000.0, 0.5063033675594192},
      {1128.554336156451, 2.0, 1000.0, 0.9747583299368598},
      {1381.663008469353, 2.0, 1000.0, 0.9999999842942231},
      {1761.326016938706, 2.0, 1000.0, 1.0},
      {7601.88000299985, 2.0, 10000.0, 6.619821424600929e-38},
      {8801.940001499925, 2.0, 10000.0, 3.0756087276260935e-10},
      {9601.980000499974, 2.0, 10000.0, 0.021933592149321716},
      {10002.0, 2.0, 10000.0, 0.501994570123568},
      {10402.019999500026, 2.0, 10000.0, 0.9764468206781006},
      {11202.059998500075, 2.0, 10000.0, 0.9999999973092624},
      {12402.11999700015, 2.0, 10000.0, 1.0},
      {92412.49566835884, 2.0, 100000.0, 1.0087546576971118e-34},
      {96207.24783417942, 2.0, 100000.0, 6.958768047454886e-10},
      {98737.08261139314, 2.0, 100000.0, 0.022493357687831003},
      {100002.0, 2.0, 100000.0, 0.5006307786625009},
      {101266.91738860686, 2.0, 100000.0, 0.9769944435124862},
      {103796.75216582058, 2.0, 100000.0, 0.9999999986236185},
      {107591.50433164116, 2.0, 100000.0, 1.0},
      {1e-08, 10.0, 0.0, 2.6041666558159723e-44},
      {1e-08, 10.0, 0.0, 2.6041666558159723e-44},
      {1.0557280900008408, 10.0, 0.0, 0.00022060998893778098},
      {10.0, 10.0, 0.0, 0.5595067149347875},
      {18.94427190999916, 10.0, 0.0, 0.9590237503458523},
      {36.83281572999748, 10.0, 0.0, 0.9999395116875759},
      {63.665631459994955, 10.0, 0.0, 0.9999999992710185},
      {1e-08, 10.0, 0.1, 2.4771599494189916e-44},
      {1e-08, 10.0, 0.1, 2.4771599494189916e-44},
      {1.0667281674910285, 10.0, 0.1, 0.00022098293130066088},
      {10.1, 10.0, 0.1, 0.5594995884898118},
      {19.13327183250897, 10.0, 0.1, 0.9590250305439667},
      {37.19981549752691, 10.0, 0.1, 0.9999395682976726},
      {64.29963099505382, 10.0, 0.1, 0.9999999992742918},
      {1e-08, 10.0, 1.0, 1.579506920411832e-44},
      {1e-08, 10.0, 1.0, 1.579506920411832e-44},
      {1.2020410288672885, 10.0, 1.0, 0.0002532573240740949},
      {11.0, 10.0, 1.0, 0.5589343163047967},
      {20.79795897113271, 10.0, 1.0, 0.9591324834425929},
      {40.39387691339813, 10.0, 1.0, 0.9999433173387619},
      {69.78775382679626, 10.0, 1.0, 0.9999999994459823},
      {1e-08, 10.0, 10.0, 1.7546736976785073e-46},
      {1e-08, 10.0, 10.0, 1.7546736976785073e-46},
      {4.508066615170332, 10.0, 10.0, 0.002614238029868565},
      {20.0, 10.0, 10.0, 0.5460705660031602},
      {35.491933384829665, 10.0, 10.0, 0.9620826981270608},
      {66.47580015448901, 10.0, 10.0, 0.9999825081286936},
      {112.95160030897802, 10.0, 10.0, 0.9999999999918913},
      {1e-08, 10.0, 100.0, 5.022786250760517e-66},
      {1e-08, 10.0, 100.0, 5.022786250760517e-66},
      {69.01219693616162, 10.0, 100.0, 0.014378668684134564},
      {110.0, 10.0, 100.0, 0.5191799470779188},
      {150.98780306383838, 10.0, 100.0, 0.9700868372330647},
      {232.96340919151518, 10.0, 100.0, 0.9999995242244333},
      {355.92681838303037, 10.0, 100.0, 1.0},
      {249.15836076092683, 10.0, 1000.0, 3.606472612090083e-58},
      {629.5791803804634, 10.0, 1000.0, 1.109771616480571e-11},
      {883.1930601268211, 10.0, 1000.0, 0.02013312179040287},
      {1010.0, 10.0, 1000.0, 0.5062824665937948},
      {1136.8069398731789, 10.0, 1000.0, 0.97476634130932},
      {1390.4208196195366, 10.0, 1000.0, 0.9999999844080918},
      {1770.8416392390732, 10.0, 1000.0, 1.0},
      {7609.400074981256, 10.0, 10000.0, 6.647253017983668e-38},
      {8809.700037490627, 10.0, 10000.0, 3.076917445756685e-10},
      {9609.900012496875, 10.0, 10000.0, 0.021933866815816123},
      {10010.0, 10.0, 10000.0, 0.5019939056174888},
      {10410.099987503125, 10.0, 10000.0, 0.9764470856373196},
      {11210.299962509373, 10.0, 10000.0, 0.9999999973100913},
      {12410.599925018745, 10.0, 10000.0, 1.0},
      {92420.34388130793, 10.0, 100000.0, 1.008856759379635e-34},
      {96215.17194065396, 10.0, 100000.0, 6.958851290249712e-10},
      {98745.05731355133, 10.0, 100000.0, 0.022493366272690125},
      {100010.0, 10.0, 100000.0, 0.500630757637655},
      {101274.94268644867, 10.0, 100000.0, 0.976994452000172},
      {103804.82805934604, 10.0, 100000.0, 0.9999999986236333},
      {107599.65611869207, 10.0, 100000.0, 1.0},
      {1e-08, 100.0, 0.0, 0.0},
      {15.147186257614294, 100.0, 0.0, 1.829868160846534e-24},
      {71.7157287525381, 100.0, 0.0, 0.014661655202476578},
      {100.0, 100.0, 0.0, 0.5188083154720433},
      {128.2842712474619, 100.0, 0.0, 0.9701935091467563},
      {184.8528137423857, 100.0, 0.0, 0.9999994782807551},
      {269.7056274847714, 100.0, 0.0, 1.0},
      {1e-08, 100.0, 0.1, 0.0},
      {15.162375827905322, 100.0, 0.1, 1.8300529122805706e-24},
      {71.78745860930178, 100.0, 0.1, 0.014661666653317145},
      {100.1, 100.0, 0.1, 0.5188082879507262},
      {128.4125413906982, 100.0, 0.1, 0.9701935187998433},
      {185.03762417209467, 100.0, 0.1, 0.9999994782848348},
      {269.97524834418937, 100.0, 0.1, 1.0},
      {1e-08, 100.0, 1.0, 0.0},
      {15.302858857485802, 100.0, 1.0, 1.848068720135195e-24},
      {72.43428628582859, 100.0, 1.0, 0.014662769310283935},
      {101.0, 100.0, 1.0, 0.5188056414666261},
      {129.5657137141714, 100.0, 1.0, 0.9701944465836881},
      {186.6971411425142, 100.0, 1.0, 0.9999994786735256},
      {272.3942822850284, 100.0, 1.0, 1.0},
      {1e-08, 100.0, 10.0, 0.0},
      {17.048399691021984, 100.0, 10.0, 4.111209510775576e-24},
      {79.01613323034067, 100.0, 10.0, 0.014748112124839643},
      {110.0, 100.0, 10.0, 0.5186032535276203},
      {140.98386676965933, 100.0, 10.0, 0.9702651811649825},
      {202.95160030897802, 100.0, 10.0, 0.9999995058378265},
      {295.90320061795603, 100.0, 10.0, 1.0},
      {1e-08, 100.0, 100.0, 0.0},
      {53.0306154330093, 100.0, 100.0, 1.4024254718866045e-17},
      {151.01020514433645, 100.0, 100.0, 0.016551520724374023},
      {200.0, 100.0, 100.0, 0.5144861249627792},
      {248.98979485566355, 100.0, 100.0, 0.9717186659242374},
      {346.9693845669907, 100.0, 100.0, 0.9999998196516782},
      {493.9387691339814, 100.0, 100.0, 1.0},
      {322.31111619105684, 100.0, 1000.0, 5.189755047134225e-56},
      {711.1555580955285, 100.0, 1000.0, 1.3823966986697901e-11},
      {970.3851860318427, 100.0, 1000.0, 0.020228997998770246},
      {1100.0, 100.0, 1000.0, 0.5060588113172563},
      {1229.6148139681573, 100.0, 1000.0, 0.9748521792159962},
      {1488.8444419044715, 100.0, 1000.0, 0.9999999855859952},
      {1877.688883808943, 100.0, 1000.0, 1.0},
      {7694.007481308389, 100.0, 10000.0, 6.961815647326587e-38},
      {8897.003740654194, 100.0, 10000.0, 3.0915967263725287e-10},
      {9699.00124688473, 100.0, 10000.0, 0.021936940863010835},
      {10100.0, 100.0, 10000.0, 0.5019864681808376},
      {10500.99875311527, 100.0, 10000.0, 0.9764500513043453},
      {11302.996259345806, 100.0, 10000.0, 0.9999999973193544},
      {12505.992518691612, 100.0, 10000.0, 1.0},
      {92508.63648611134, 100.0, 100000.0, 1.0100054930505303e-34},
      {96304.31824305566, 100.0, 100000.0, 6.959787346574633e-10},
      {98834.77274768522, 100.0, 100000.0, 0.022493462802550396},
      {100100.0, 100.0, 100000.0, 0.5006305212297664},
      {101365.22725231478, 100.0, 100000.0, 0.9769945474376748},
      {103895.68175694434, 100.0, 100000.0, 0.9999999986238006},
      {107691.36351388866, 100.0, 100000.0, 1.0},
      {120.5266807797945, 500.0, 0.0, 2.8232293926835933e-74},
      {310.2633403898973, 500.0, 0.0, 1.60842644427919e-12},
      {436.7544467966324, 500.0, 0.0, 0.019233660907914778},
      {500.0, 500.0, 0.0, 0.508410626968991},
      {563.2455532033675, 500.0, 0.0, 0.9739475105105316},
      {689.7366596101027, 500.0, 0.0, 0.999999965744614},
      {879.4733192202054, 500.0, 0.0, 1.0},
      {120.55079370389933, 500.0, 0.1, 2.8232550128896487e-74},
      {310.3253968519497, 500.0, 0.1, 1.608427156109468e-12},
      {436.8417989506499, 500.0, 0.1, 0.01923366111793158},
      {500.1, 500.0, 0.1, 0.5084106264668754},
      {563.3582010493501, 500.0, 0.1, 0.9739475107038523},
      {689.8746031480504, 500.0, 0.1, 0.9999999657446207},
      {879.6492062961007, 500.0, 0.1, 1.0},
      {120.76849157388335, 500.0, 1.0, 2.82578187060671e-74},
      {310.8842457869417, 500.0, 1.0, 1.6084972665847583e-12},
      {437.6280819289806, 500.0, 1.0, 0.019233681787406705},
      {501.0, 500.0, 1.0, 0.5084105770552556},
      {564.3719180710194, 500.0, 1.0, 0.9739475297263746},
      {691.1157542130584, 500.0, 1.0, 0.999999965745275},
      {881.2315084261166, 500.0, 1.0, 1.0},
      {123.01162808166964, 500.0, 10.0, 3.0788630802318166e-74},
      {316.5058140408348, 500.0, 10.0, 1.6151744867448387e-12},
      {445.5019380136116, 500.0, 10.0, 0.01923563253591011},
      {510.0, 500.0, 10.0, 0.5084059188105736},
      {574.4980619863884, 500.0, 10.0, 0.973949321713556},
      {703.4941859591652, 500.0, 10.0, 0.9999999658064164},
      {896.9883719183304, 500.0, 10.0, 1.0},
      {151.001113587127, 500.0, 100.0, 9.328460969812469e-72},
      {375.50055679356353, 500.0, 100.0, 2.100569258616538e-12},
      {525.1668522645211, 500.0, 100.0, 0.0193544127714166},
      {600.0, 500.0, 100.0, 0.5081243017315487},
      {674.8331477354789, 500.0, 100.0, 0.9740572461854259},
      {824.4994432064365, 500.0, 100.0, 0.9999999691644048},
      {1048.998886412873, 500.0, 100.0, 1.0},
      {651.4718625761429, 500.0, 1000.0, 3.1959693855405425e-50},
      {1075.7359312880715, 500.0, 1000.0, 2.854708244824926e-11},
      {1358.5786437626905, 500.0, 1000.0, 0.02056687123635103},
      {1500.0, 500.0, 1000.0, 0.5052662343423624},
      {1641.4213562373095, 500.0, 1000.0, 0.9751579917238298},
      {1924.2640687119285, 500.0, 1000.0, 0.9999999892017439},
      {2348.528137423857, 500.0, 1000.0, 1.0},
      {8070.185192242009, 500.0, 10000.0, 8.497666529836166e-38},
      {9285.092596121005, 500.0, 10000.0, 3.1558768547738984e-10},
      {10095.030865373668, 500.0, 10000.0, 0.02195025812272148},
      {10500.0, 500.0, 10000.0, 0.5019542409673978},
      {10904.969134626332, 500.0, 10000.0, 0.9764629046125153},
      {11714.907403878995, 500.0, 10000.0, 0.9999999973591938},
      {12929.81480775799, 500.0, 10000.0, 1.0},
      {92901.05270448598, 500.0, 100000.0, 1.0151129553412933e-34},
      {96700.52635224299, 500.0, 100000.0, 6.963938172013247e-10},
      {99233.508784081, 500.0, 100000.0, 0.022493890720441917},
      {100500.0, 500.0, 100000.0, 0.5006294732237153},
      {101766.491215919, 500.0, 100000.0, 0.9769949705190712},
      {104299.47364775701, 500.0, 100000.0, 0.9999999986245417},
      {108098.94729551402, 500.0, 100000.0, 1.0},
  };
  for (const auto& r : references) {
    CAPTURE(r.x);
    CAPTURE(r.df);
    CAPTURE(r.ncp);
    const double actual = magmaan::inference::noncentral_chisq_cdf(r.x, r.df, r.ncp);
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(actual - r.cdf) <= 1e-12);
  }
}
