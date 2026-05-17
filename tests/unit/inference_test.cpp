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

  // Fit so we have Σ̂; reuse the converged θ̂ to compute implied moments.
  auto est = magmaan::test::fit(*h.pt, *h.rep, samp).value();
  auto ev  = magmaan::model::ModelEvaluator::build(*h.pt, *h.rep).value();
  auto im_or = ev.sigma(est.theta);
  REQUIRE(im_or.has_value());
  // Copy out of the evaluator's internal buffer — rls_chi2 doesn't mutate
  // it but the API contract is "view into ev"; the copy keeps the test
  // resilient if implementation details shift later.
  magmaan::model::ImpliedMoments im;
  im.sigma.assign(im_or->sigma.begin(), im_or->sigma.end());
  im.mu.assign(im_or->mu.begin(), im_or->mu.end());

  auto t_rls = magmaan::inference::rls_chi2(samp, im);
  REQUIRE(t_rls.has_value());
  CHECK(*t_rls == doctest::Approx(81.3677).epsilon(1e-3));

  auto t_rls_theta = magmaan::inference::rls_chi2(
      *h.pt, *h.rep, samp, est.theta);
  REQUIRE(t_rls_theta.has_value());
  CHECK(*t_rls_theta == doctest::Approx(*t_rls).epsilon(1e-12));
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
  auto ev  = magmaan::model::ModelEvaluator::build(*h.pt, *h.rep).value();
  auto im_or = ev.sigma(est.theta);
  REQUIRE(im_or.has_value());
  magmaan::model::ImpliedMoments im;
  im.sigma.assign(im_or->sigma.begin(), im_or->sigma.end());
  im.mu.assign(im_or->mu.begin(), im_or->mu.end());

  auto t_rls = magmaan::inference::rls_chi2(samp, im);
  REQUIRE(t_rls.has_value());
  CHECK(std::abs(*t_rls) < 1e-6);
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

TEST_CASE("chi2_stat: chi2 = n · fmin") {
  // Pure arithmetic on the fmin → chi2 plumbing. N (not N−1) matches
  // lavaan's `likelihood = "normal"` default. Doesn't depend on any
  // information matrix — `chi2_stat` reads samp.n_obs and est.fmin only.
  SampleStats samp;
  samp.S.push_back(Eigen::MatrixXd::Identity(3, 3));   // not used by chi2_stat
  samp.n_obs.push_back(301);

  Estimates est;
  est.theta = Eigen::VectorXd::Zero(3);                // unused
  est.fmin  = 0.04321;

  CHECK(magmaan::inference::chi2_stat(samp, est) ==
        doctest::Approx(301.0 * 0.04321).epsilon(1e-12));
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
