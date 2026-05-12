#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "latva/fit/effects.hpp"
#include "latva/fit/fit.hpp"
#include "latva/fit/inference.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

using latva::fit::AnalyticObservedInfoSE;
using latva::fit::compute_defined;
using latva::fit::ExpectedInfoSE;
using latva::fit::SampleStats;
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

TEST_CASE("Effects: := value and delta-method SE on a^2 of a labeled loading") {
  // Single-factor CFA with a labeled non-marker loading. Define a derived
  // parameter `a_sq := a^2` and verify both value and SE match the
  // expected closed-form delta-method result.
  const std::string src = "f =~ x1 + a*x2 + x3\na_sq := a^2";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  latva::partable::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(7);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};

  auto est = latva::fit::fit(*pt, *mr, samp).value();
  auto inf = ExpectedInfoSE{}.compute(*pt, *mr, samp, est).value();

  // Find a's free index — the row labeled "a" in pt.
  Eigen::Index a_idx = -1;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (names.row_label[i] == "a" && pt->free[i] > 0) {
      a_idx = pt->free[i] - 1;
      break;
    }
  }
  REQUIRE(a_idx >= 0);
  const double a_hat   = est.theta(a_idx);
  const double a_se    = inf.se(a_idx);
  const double a_var   = inf.vcov(a_idx, a_idx);

  auto defs_or = compute_defined(*fp, *pt, names, est, inf.vcov);
  REQUIRE(defs_or.has_value());
  REQUIRE(defs_or->entries.size() == 1);
  const auto& dp = defs_or->entries[0];

  CHECK(dp.name == "a_sq");
  CHECK(dp.value == doctest::Approx(a_hat * a_hat).epsilon(1e-12));
  // Delta-method: var(g(θ)) = (∂g/∂θ)' vcov (∂g/∂θ). For g = a², ∂g/∂a = 2a.
  // SE = sqrt((2a)² · var(a)) = |2a| · se(a).
  const double expected_se = std::abs(2.0 * a_hat) * a_se;
  CHECK(dp.se == doctest::Approx(expected_se).epsilon(1e-8));
  // Also verify via the variance: var = (2a)² · var(a).
  CHECK(dp.se * dp.se ==
        doctest::Approx(4.0 * a_hat * a_hat * a_var).epsilon(1e-8));
}

TEST_CASE("Effects: := with a product of two labeled loadings") {
  // Two labeled loadings; `prod := a * b`. Gradient is [b, a], SE comes
  // from a 2x2 sub-block of vcov.
  const std::string src =
      "f =~ x1 + a*x2 + b*x3\nprod := a * b";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  latva::partable::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(42);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};

  auto est = latva::fit::fit(*pt, *mr, samp).value();
  auto inf = ExpectedInfoSE{}.compute(*pt, *mr, samp, est).value();

  Eigen::Index a_idx = -1, b_idx = -1;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->free[i] == 0) continue;
    if (names.row_label[i] == "a") a_idx = pt->free[i] - 1;
    if (names.row_label[i] == "b") b_idx = pt->free[i] - 1;
  }
  REQUIRE(a_idx >= 0);
  REQUIRE(b_idx >= 0);
  const double a_hat = est.theta(a_idx);
  const double b_hat = est.theta(b_idx);

  auto defs = compute_defined(*fp, *pt, names, est, inf.vcov).value();
  REQUIRE(defs.entries.size() == 1);
  CHECK(defs.entries[0].name == "prod");
  CHECK(defs.entries[0].value ==
        doctest::Approx(a_hat * b_hat).epsilon(1e-12));

  // Delta-method: ∇(ab) = [b, a] (at the indices of a and b).
  // var = [b, a] · vcov_2x2 · [b, a]' with vcov_2x2 being the relevant
  // 2x2 sub-block.
  const double var_expected =
      b_hat * b_hat * inf.vcov(a_idx, a_idx) +
      2.0 * a_hat * b_hat * inf.vcov(a_idx, b_idx) +
      a_hat * a_hat * inf.vcov(b_idx, b_idx);
  CHECK(defs.entries[0].se * defs.entries[0].se ==
        doctest::Approx(var_expected).epsilon(1e-8));
}

TEST_CASE("Effects: := referencing an unknown label errors clearly") {
  const std::string src = "f =~ x1 + a*x2 + x3\nbad := a + missing";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  latva::partable::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(1);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {200};
  auto est = latva::fit::fit(*pt, *mr, samp).value();
  auto inf = ExpectedInfoSE{}.compute(*pt, *mr, samp, est).value();

  auto defs_or = compute_defined(*fp, *pt, names, est, inf.vcov);
  REQUIRE_FALSE(defs_or.has_value());
  CHECK(defs_or.error().kind == latva::PostError::Kind::NumericIssue);
}
