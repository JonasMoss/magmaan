#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <Eigen/Core>

#include "magmaan/nt/effects.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/nt/infer.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

#include "../inference_bundle.hpp"

using magmaan::nt::effects::compute_defined;
using magmaan::data::SampleStats;
using magmaan::test::analytic_observed_inference;
using magmaan::test::expected_inference;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::lavaanify;

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
  magmaan::spec::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(7);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};

  auto est = magmaan::estimate::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

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
  magmaan::spec::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(42);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};

  auto est = magmaan::estimate::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

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

TEST_CASE("Effects: chained := (one definition references an earlier one)") {
  // ab := a*b ; sum := ab + a. `sum` chains off `ab` — its gradient must be
  // the accumulated θ-gradient ([b+1] at a, [a] at b), giving the right
  // delta-method SE for the composite.
  const std::string src =
      "f =~ x1 + a*x2 + b*x3\nab := a * b\nsum := ab + a";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(99);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {300};

  auto est = magmaan::estimate::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

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
  REQUIRE(defs.entries.size() == 2);
  // Source order preserved.
  CHECK(defs.entries[0].name == "ab");
  CHECK(defs.entries[1].name == "sum");
  CHECK(defs.entries[0].value == doctest::Approx(a_hat * b_hat).epsilon(1e-12));
  CHECK(defs.entries[1].value ==
        doctest::Approx(a_hat * b_hat + a_hat).epsilon(1e-12));

  // ∇(a·b + a) = [b+1, a] at (a_idx, b_idx).
  const double ga = b_hat + 1.0, gb = a_hat;
  const double var_expected =
      ga * ga * inf.vcov(a_idx, a_idx) +
      2.0 * ga * gb * inf.vcov(a_idx, b_idx) +
      gb * gb * inf.vcov(b_idx, b_idx);
  CHECK(defs.entries[1].se * defs.entries[1].se ==
        doctest::Approx(var_expected).epsilon(1e-8));
}

TEST_CASE("Effects: := resolves .pN. plabels (free and fixed rows)") {
  // Row layout for `f =~ x1 + a*x2 + b*x3`: row0 = f=~x1 (marker, fixed 1.0,
  // .p1.), row1 = f=~x2 labeled `a` (.p2.), row2 = f=~x3 labeled `b` (.p3.).
  const std::string src =
      "f =~ x1 + a*x2 + b*x3\nq := .p2. * .p3.\nm := .p1. + a";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  // Pin the assumed plabel mapping so a lavaanify reorder fails loudly here.
  Eigen::Index a_idx = -1, b_idx = -1;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (names.row_label[i] == "a") { CHECK(names.row_plabel[i] == ".p2."); if (pt->free[i] > 0) a_idx = pt->free[i] - 1; }
    if (names.row_label[i] == "b") { CHECK(names.row_plabel[i] == ".p3."); if (pt->free[i] > 0) b_idx = pt->free[i] - 1; }
    if (names.row_lhs[i] == "f" && names.row_rhs[i] == "x1") {
      CHECK(names.row_plabel[i] == ".p1.");
      CHECK(pt->free[i] == 0);
      CHECK(pt->fixed_value[i] == doctest::Approx(1.0));
    }
  }
  REQUIRE(a_idx >= 0);
  REQUIRE(b_idx >= 0);

  std::mt19937 rng(123);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {250};

  auto est = magmaan::estimate::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();
  const double a_hat = est.theta(a_idx);
  const double b_hat = est.theta(b_idx);

  auto defs = compute_defined(*fp, *pt, names, est, inf.vcov).value();
  REQUIRE(defs.entries.size() == 2);
  CHECK(defs.entries[0].name == "q");
  CHECK(defs.entries[0].value == doctest::Approx(a_hat * b_hat).epsilon(1e-12));
  // m := .p1.(=1, fixed) + a  ⇒  value 1+a, gradient [1] at a only.
  CHECK(defs.entries[1].name == "m");
  CHECK(defs.entries[1].value == doctest::Approx(1.0 + a_hat).epsilon(1e-12));
  CHECK(defs.entries[1].se ==
        doctest::Approx(std::sqrt(inf.vcov(a_idx, a_idx))).epsilon(1e-10));
}

TEST_CASE("Effects: circular := definitions error cleanly") {
  // `c1` and `c2` reference each other — Kahn's sweep can't order them.
  // compute_defined() only needs Estimates(theta) + a matching-size vcov.
  const std::string src =
      "f =~ x1 + a*x2 + x3\nc1 := c2 + 1\nc2 := c1 * 2";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());

  const Eigen::Index nf = pt->n_free();
  magmaan::estimate::Estimates est;
  est.theta = Eigen::VectorXd::Zero(nf);
  const Eigen::MatrixXd vcov = Eigen::MatrixXd::Identity(nf, nf);

  auto defs_or = compute_defined(*fp, *pt, names, est, vcov);
  REQUIRE_FALSE(defs_or.has_value());
  CHECK(defs_or.error().kind == magmaan::PostError::Kind::NumericIssue);

  // A direct self-reference is the n == 1 cycle.
  auto fp2 = Parser::parse("f =~ x1 + a*x2 + x3\nd := d + 1");
  REQUIRE(fp2.has_value());
  magmaan::spec::LatentNames names2;
  auto pt2 = lavaanify(*fp2, {}, nullptr, &names2);
  REQUIRE(pt2.has_value());
  magmaan::estimate::Estimates est2;
  est2.theta = Eigen::VectorXd::Zero(pt2->n_free());
  const Eigen::MatrixXd vcov2 =
      Eigen::MatrixXd::Identity(pt2->n_free(), pt2->n_free());
  auto defs2_or = compute_defined(*fp2, *pt2, names2, est2, vcov2);
  REQUIRE_FALSE(defs2_or.has_value());
  CHECK(defs2_or.error().kind == magmaan::PostError::Kind::NumericIssue);
}

TEST_CASE("Effects: := referencing an unknown label errors clearly") {
  const std::string src = "f =~ x1 + a*x2 + x3\nbad := a + missing";
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LatentNames names;
  auto pt = lavaanify(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(1);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {200};
  auto est = magmaan::estimate::fit(*pt, *mr, samp).value();
  auto inf = expected_inference(*pt, *mr, samp, est).value();

  auto defs_or = compute_defined(*fp, *pt, names, est, inf.vcov);
  REQUIRE_FALSE(defs_or.has_value());
  CHECK(defs_or.error().kind == magmaan::PostError::Kind::NumericIssue);
}
