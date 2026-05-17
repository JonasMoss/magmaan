#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#include "../inference_bundle.hpp"

using magmaan::data::SampleStats;
using magmaan::test::expected_inference;
using magmaan::measures::standardize::standardize_all;
using magmaan::measures::standardize::standardize_lv;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;

namespace {

Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() + Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

}  // namespace

TEST_CASE("standardize_lv: 1F CFA — ψ_ff → 1, λs scaled by √ψ̂_ff") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::mt19937 rng(7);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};
  auto est = magmaan::test::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

  auto std_or = standardize_lv(*pt, *mr, est, inf.vcov);
  REQUIRE(std_or.has_value());
  const auto& sol = *std_or;
  REQUIRE(sol.theta.size() == est.theta.size());
  REQUIRE(sol.se.size()    == est.theta.size());

  // Find ψ_ff (Psi[0,0]) and the two free Λ loadings (λ_2, λ_3).
  auto ev = ModelEvaluator::build(*pt, *mr).value();
  const auto locs = ev.param_locations();
  Eigen::Index psi_idx = -1;
  std::vector<Eigen::Index> lambda_idx;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Psi &&
        locs[k].row == 0 && locs[k].col == 0) {
      psi_idx = static_cast<Eigen::Index>(k);
    }
    if (locs[k].mat == magmaan::model::MatId::Lambda) {
      lambda_idx.push_back(static_cast<Eigen::Index>(k));
    }
  }
  REQUIRE(psi_idx >= 0);
  REQUIRE(lambda_idx.size() == 2);

  const double psi_hat = est.theta(psi_idx);
  const double sqrt_psi = std::sqrt(psi_hat);

  // ψ_ff → 1, SE 0.
  CHECK(sol.theta(psi_idx) == doctest::Approx(1.0));
  CHECK(sol.se(psi_idx)    == doctest::Approx(0.0));

  // For each free λ: std = λ̂ · √ψ̂.
  for (auto k : lambda_idx) {
    const double lam_hat = est.theta(k);
    CHECK(sol.theta(k) == doctest::Approx(lam_hat * sqrt_psi).epsilon(1e-10));
    // SE via delta method with grad J_k = [√ψ at k, λ/(2√ψ) at ψ-index,
    // zero elsewhere]. Compute the closed-form variance and compare.
    const double a = sqrt_psi;
    const double b = lam_hat / (2.0 * sqrt_psi);
    const double var = a * a * inf.vcov(k, k) +
                       2.0 * a * b * inf.vcov(k, psi_idx) +
                       b * b * inf.vcov(psi_idx, psi_idx);
      CHECK(sol.se(k) == doctest::Approx(std::sqrt(var)).epsilon(1e-6));
  }
}

TEST_CASE("standardize_all: 1F CFA — ν_i rescaled by 1/√σ̂_ii, λ by √ψ/√σ") {
  // With meanstructure, ν_i / √σ_ii should match closed-form, and Λ
  // should be scaled by both ψ_cc and σ_rr.
  auto fp = Parser::parse("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::mt19937 rng(2026);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 3.0, 4.0, 5.0;
  SampleStats samp;  samp.S = {S};  samp.mean = {mean};  samp.n_obs = {300};

  auto est = magmaan::test::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

  auto std_or = standardize_all(*pt, *mr, est, inf.vcov);
  REQUIRE_MESSAGE(std_or.has_value(),
      "standardize_all failed: " <<
      (std_or.has_value() ? "" : std_or.error().detail));
  const auto& sol = *std_or;
  REQUIRE(sol.theta.size() == est.theta.size());

  // Get implied Σ at θ̂ to verify the formulas.
  auto ev = ModelEvaluator::build(*pt, *mr).value();
  const auto sm = ev.sigma(est.theta).value();
  const auto locs = ev.param_locations();

  // For ν params: std value = ν̂ / √σ̂_ii. (Saturated → σ̂_ii = S_ii.)
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat != magmaan::model::MatId::Nu) continue;
    const auto i = locs[k].row;
    const double sigma_ii = sm.sigma[0](i, i);
    const double expected = est.theta(static_cast<Eigen::Index>(k)) /
                            std::sqrt(sigma_ii);
    CHECK(sol.theta(static_cast<Eigen::Index>(k)) ==
          doctest::Approx(expected).epsilon(1e-8));
    // SE should be positive and finite.
    CHECK(sol.se(static_cast<Eigen::Index>(k)) > 0.0);
    CHECK(std::isfinite(sol.se(static_cast<Eigen::Index>(k))));
  }
  // For ψ (the lone diagonal Psi free param) — std value is 1, SE is 0.
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Psi &&
        locs[k].row == locs[k].col) {
      CHECK(sol.theta(static_cast<Eigen::Index>(k)) ==
            doctest::Approx(1.0));
      CHECK(sol.se(static_cast<Eigen::Index>(k)) == doctest::Approx(0.0));
    }
  }
}

TEST_CASE("standardize_lv: 2F CFA — factor covariance → correlation, with delta-method SE") {
  // Two correlated factors over the first six Holzinger indicators. lavaan's
  // `auto.cov.lv.x` adds a free `f1 ~~ f2`; the marker convention frees both
  // factor variances. std.lv must rescale that covariance to a correlation
  // (the G7 fix — it used to pass through unscaled).
  auto fp = Parser::parse(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt); REQUIRE(mr.has_value());

  // Use the [x1..x6] submatrix of lavaan's Holzinger sample covariance so the
  // fit has a genuine factor structure (positive factor variances).
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0002_three_factor_hs.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& M = j["sample_cov"][0]["matrix"];
  Eigen::MatrixXd S(6, 6);
  for (Eigen::Index r = 0; r < 6; ++r)
    for (Eigen::Index c = 0; c < 6; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                    .get<double>();
  SampleStats samp;  samp.S = {S};  samp.n_obs = {301};

  auto est = magmaan::test::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();
  auto sol_or = standardize_lv(*pt, *mr, est, inf.vcov);
  REQUIRE_MESSAGE(sol_or.has_value(),
      "standardize_lv failed: " << (sol_or.has_value() ? "" : sol_or.error().detail));
  const auto& sol = *sol_or;

  auto ev = ModelEvaluator::build(*pt, *mr).value();
  const auto locs = ev.param_locations();
  Eigen::Index cov_idx = -1, v0_idx = -1, v1_idx = -1;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    if (L.mat != magmaan::model::MatId::Psi) continue;
    if (L.row == L.col) {
      if (L.row == 0) v0_idx = static_cast<Eigen::Index>(k);
      if (L.row == 1) v1_idx = static_cast<Eigen::Index>(k);
    } else {
      cov_idx = static_cast<Eigen::Index>(k);
    }
  }
  REQUIRE(cov_idx >= 0);
  REQUIRE(v0_idx >= 0);
  REQUIRE(v1_idx >= 0);

  const double psi_rr = est.theta(v0_idx);
  const double psi_cc = est.theta(v1_idx);
  const double psi_rc = est.theta(cov_idx);
  const double prod   = std::sqrt(psi_rr * psi_cc);

  // (a) value = ψ̂_rc / √(ψ̂_rr·ψ̂_cc) — a genuine correlation in (-1, 1).
  CHECK(sol.theta(cov_idx) == doctest::Approx(psi_rc / prod).epsilon(1e-10));
  CHECK(std::abs(sol.theta(cov_idx)) < 1.0);

  // (b) delta-method SE: grad = [1/prod at cov, -ψ_rc/(2·ψ_rr·prod) at v0,
  //     -ψ_rc/(2·ψ_cc·prod) at v1].
  const double g_cov = 1.0 / prod;
  const double g_v0  = -psi_rc / (2.0 * psi_rr * prod);
  const double g_v1  = -psi_rc / (2.0 * psi_cc * prod);
  const double var =
      g_cov * g_cov * inf.vcov(cov_idx, cov_idx) +
      g_v0  * g_v0  * inf.vcov(v0_idx, v0_idx) +
      g_v1  * g_v1  * inf.vcov(v1_idx, v1_idx) +
      2.0 * g_cov * g_v0 * inf.vcov(cov_idx, v0_idx) +
      2.0 * g_cov * g_v1 * inf.vcov(cov_idx, v1_idx) +
      2.0 * g_v0  * g_v1 * inf.vcov(v0_idx, v1_idx);
  CHECK(sol.se(cov_idx) == doctest::Approx(std::sqrt(var)).epsilon(1e-10));

  // (c) std.all reduces to the same transform on a latent-latent covariance
  //     (the extra 1/√σ_rr factors only touch indicators).
  auto sol_all = standardize_all(*pt, *mr, est, inf.vcov);
  REQUIRE(sol_all.has_value());
  CHECK(sol_all->theta(cov_idx) == doctest::Approx(sol.theta(cov_idx)).epsilon(1e-12));
  CHECK(sol_all->se(cov_idx)    == doctest::Approx(sol.se(cov_idx)).epsilon(1e-12));
}

TEST_CASE("standardize_all: structural Beta uses total latent variances") {
  auto fp = Parser::parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9\n"
      "speed ~ visual + textual");
  REQUIRE(fp.has_value());
  auto pt = build(*fp); REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt); REQUIRE(mr.has_value());

  SampleStats samp;
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit_std/0002_cfa_plus_structural_hs.fit.json");
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
  samp.S = {std::move(S)};
  samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  auto est_or = magmaan::test::fit(*pt, *mr, samp);
  REQUIRE(est_or.has_value());
  auto inf = expected_inference(*pt, *mr, samp, *est_or).value();

  auto lv_or = standardize_lv(*pt, *mr, *est_or, inf.vcov);
  auto all_or = standardize_all(*pt, *mr, *est_or, inf.vcov);
  REQUIRE(lv_or.has_value());
  REQUIRE(all_or.has_value());

  auto ev = ModelEvaluator::build(*pt, *mr).value();
  auto am = ev.assembled(est_or->theta).value();
  const auto locs = ev.param_locations();
  Eigen::Index beta_idx = -1;
  magmaan::model::ParamLocation beta_loc;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Beta) {
      beta_idx = static_cast<Eigen::Index>(k);
      beta_loc = locs[k];
      break;
    }
  }
  REQUIRE(beta_idx >= 0);

  const auto& bm = am.blocks[static_cast<std::size_t>(beta_loc.block)];
  const double beta = est_or->theta(beta_idx);
  const double expected =
      beta * std::sqrt(bm.Mid(beta_loc.col, beta_loc.col)) /
      std::sqrt(bm.Mid(beta_loc.row, beta_loc.row));
  CHECK(lv_or->theta(beta_idx) == doctest::Approx(expected).epsilon(1e-10));
  CHECK(all_or->theta(beta_idx) == doctest::Approx(expected).epsilon(1e-10));
  CHECK(all_or->se(beta_idx) > 0.0);
}
