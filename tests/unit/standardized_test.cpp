#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "latva/fit/fit.hpp"
#include "latva/fit/inference.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/fit/standardized.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

using latva::fit::ExpectedInfoSE;
using latva::fit::SampleStats;
using latva::fit::standardize_all;
using latva::fit::standardize_lv;
using latva::model::build_matrix_rep;
using latva::model::ModelEvaluator;
using latva::parse::Parser;
using latva::partable::lavaanify;

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
  auto pt = lavaanify(*fp);  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::mt19937 rng(7);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};
  auto est = latva::fit::fit(*pt, *mr, samp).value();
  auto inf = ExpectedInfoSE{}.compute(*pt, *mr, samp, est).value();

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
    if (locs[k].mat == latva::model::MatId::Psi &&
        locs[k].row == 0 && locs[k].col == 0) {
      psi_idx = static_cast<Eigen::Index>(k);
    }
    if (locs[k].mat == latva::model::MatId::Lambda) {
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
    CHECK(sol.se(k) == doctest::Approx(std::sqrt(var)).epsilon(1e-10));
  }
}

TEST_CASE("standardize_all: 1F CFA — ν_i rescaled by 1/√σ̂_ii, λ by √ψ/√σ") {
  // With meanstructure, ν_i / √σ_ii should match closed-form, and Λ
  // should be scaled by both ψ_cc and σ_rr.
  auto fp = Parser::parse("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::mt19937 rng(2026);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 3.0, 4.0, 5.0;
  SampleStats samp;  samp.S = {S};  samp.mean = {mean};  samp.n_obs = {300};

  auto est = latva::fit::fit(*pt, *mr, samp).value();
  auto inf = ExpectedInfoSE{}.compute(*pt, *mr, samp, est).value();

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
    if (locs[k].mat != latva::model::MatId::Nu) continue;
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
    if (locs[k].mat == latva::model::MatId::Psi &&
        locs[k].row == locs[k].col) {
      CHECK(sol.theta(static_cast<Eigen::Index>(k)) ==
            doctest::Approx(1.0));
      CHECK(sol.se(static_cast<Eigen::Index>(k)) == doctest::Approx(0.0));
    }
  }
}
