#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "magmaan/error.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/fit.hpp"
#include "magmaan/fit/inference.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/partable/lavaanify.hpp"

#include "../inference_bundle.hpp"

using magmaan::fit::build_eq_constraints;
using magmaan::fit::SampleStats;
using magmaan::test::expected_inference;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatId;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::partable::lavaanify;
using magmaan::partable::LavaanifyOptions;
using magmaan::partable::LatentStructure;

namespace {

LatentStructure must_lavaanify(std::string_view src, LavaanifyOptions opts = {}) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp, opts);
  REQUIRE_MESSAGE(pt.has_value(),
      "lavaanify failed: " << (pt.has_value() ? "" : pt.error().detail));
  return std::move(*pt);
}

// 3×3 sample covariance (x1,x2,x3) + n from the saturated 1F-CFA fixture.
SampleStats fixture_samp_3() {
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
      S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());
  return samp;
}

// Free index of the cell Λ[row, 0] (the loading of the `row`-th observed
// variable on the sole latent), as a 0-based θ ordinal; -1 if not found.
Eigen::Index lambda_free_idx(const ModelEvaluator& ev, std::int16_t row) {
  const auto locs = ev.param_locations();
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == MatId::Lambda && locs[k].row == row) {
      return static_cast<Eigen::Index>(k);
    }
  }
  return -1;
}

}  // namespace

TEST_CASE("build_eq_constraints: identity reparameterization on a model with no `==` rows") {
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto con_or = build_eq_constraints(pt);
  REQUIRE(con_or.has_value());
  const auto& con = *con_or;
  CHECK_FALSE(con.active());
  CHECK(con.rank == 0);
  CHECK(con.npar == pt.n_free());
  CHECK(con.n_alpha == pt.n_free());
  for (std::int32_t k = 0; k < con.npar; ++k) {
    CHECK(con.group[static_cast<std::size_t>(k)] == k);
  }
}

TEST_CASE("build_eq_constraints: a shared label merges the two tied loadings") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto con_or = build_eq_constraints(pt);
  REQUIRE(con_or.has_value());
  const auto& con = *con_or;
  CHECK(con.active());
  CHECK(con.rank == 1);
  CHECK(con.npar == pt.n_free());
  CHECK(con.n_alpha == pt.n_free() - 1);

  auto rep = build_matrix_rep(pt).value();
  auto ev  = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(con.group[static_cast<std::size_t>(k_x2)] ==
        con.group[static_cast<std::size_t>(k_x3)]);
}

TEST_CASE("fit: an equality constraint is actually enforced (the tied loadings come out equal)") {
  auto samp = fixture_samp_3();
  auto pt   = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto rep  = build_matrix_rep(pt).value();

  auto est_or = magmaan::fit::fit(pt, rep, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "constrained fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;
  // The returned θ̂ is the FULL parameter vector (length n_free), not the
  // reduced α — downstream inference / R bindings expect this.
  CHECK(static_cast<std::int32_t>(est.theta.size()) == pt.n_free());

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(est.theta(k_x2) == doctest::Approx(est.theta(k_x3)).epsilon(1e-12));
}

TEST_CASE("Inference under an equality constraint: df += 1, tied SEs, χ² reflects the cost") {
  auto samp = fixture_samp_3();

  // Unconstrained baseline: saturated 1F CFA, df = 0, χ² ≈ 0.
  auto pt_u  = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep_u = build_matrix_rep(pt_u).value();
  auto est_u = magmaan::fit::fit(pt_u, rep_u, samp).value();
  auto inf_u = expected_inference(pt_u, rep_u, samp, est_u).value();
  CHECK(inf_u.df == 0);
  CHECK(inf_u.chi2 < 1e-6);

  // Constrained: x2 & x3 loadings tied → one fewer effective free param.
  auto pt_c  = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto rep_c = build_matrix_rep(pt_c).value();
  auto est_c = magmaan::fit::fit(pt_c, rep_c, samp).value();
  auto inf_or = expected_inference(pt_c, rep_c, samp, est_c);
  REQUIRE_MESSAGE(inf_or.has_value(),
      "constrained expected_inference failed: " <<
          (inf_or.has_value() ? "" : inf_or.error().detail));
  const auto& inf = *inf_or;

  CHECK(inf.df == inf_u.df + 1);
  CHECK(inf.info.rows() == pt_c.n_free());   // full n_free × n_free info
  CHECK(inf.info.cols() == pt_c.n_free());
  CHECK(inf.vcov.rows() == pt_c.n_free());
  CHECK(inf.se.size()   == pt_c.n_free());
  CHECK(inf.chi2 >= inf_u.chi2);             // constraining can't improve fit
  CHECK(inf.chi2 > 1e-8);                    // on real data it costs something

  auto ev = ModelEvaluator::build(pt_c, rep_c).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(inf.se(k_x2) == doctest::Approx(inf.se(k_x3)).epsilon(1e-10));
}

TEST_CASE("build_eq_constraints: inequality (`<` / `>`) rows error out — not yet enforced") {
  // A `b > 0` line on a labeled factor variance produces a GtConstraint row.
  auto pt = must_lavaanify("f =~ x1 + x2 + x3\nf ~~ b*f\nb > 0");
  auto con_or = build_eq_constraints(pt);
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().kind == magmaan::PostError::Kind::NumericIssue);
}

// === P9 phase 2 — general LINEAR equality constraints ======================

TEST_CASE("build_eq_constraints: pure-merge path unchanged — no lin_constraint rows, 0/1 K") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  CHECK(pt.lin_constraint_R.empty());
  CHECK(pt.lin_constraint_d.empty());
  auto con = build_eq_constraints(pt).value();
  CHECK(con.active());
  CHECK_FALSE(con.group.empty());                 // pure-merge ⇒ diagnostic group filled
  CHECK(con.Kmat.rows() == pt.n_free());
  CHECK(con.Kmat.cols() == con.n_alpha);
  CHECK(con.theta0.isZero());                      // θ₀ = 0 in the merge case
  // K is the 0/1 group-membership matrix.
  for (std::int32_t k = 0; k < con.npar; ++k) {
    for (std::int32_t g = 0; g < con.n_alpha; ++g) {
      const double want = (con.group[static_cast<std::size_t>(k)] == g) ? 1.0 : 0.0;
      CHECK(con.Kmat(k, g) == want);
    }
  }
}

TEST_CASE("build_eq_constraints: general linear `a == 2*b + c`") {
  // x0 is the marker (fixed), so a/b/c (loadings of x1/x2/x3) are all free.
  auto pt = must_lavaanify("f =~ x0 + a*x1 + b*x2 + c*x3\na == 2*b + c");
  CHECK_FALSE(pt.has_unenforced_constraints);     // linear ⇒ resolved, not flagged
  CHECK(pt.lin_constraint_d.size() == 1);
  auto con = build_eq_constraints(pt).value();
  CHECK(con.active());
  CHECK(con.rank == 1);
  CHECK(con.npar == pt.n_free());
  CHECK(con.n_alpha == pt.n_free() - 1);
  CHECK(con.Kmat.cols() == con.n_alpha);
  CHECK(con.group.empty());                        // general path ⇒ no merge partition
  // K's columns are orthonormal (an SVD kernel basis).
  Eigen::MatrixXd KtK = con.Kmat.transpose() * con.Kmat;
  CHECK(KtK.isApprox(Eigen::MatrixXd::Identity(con.n_alpha, con.n_alpha), 1e-9));
}

TEST_CASE("fit: a linear equality constraint `b2 + b3 == 1.5` is enforced") {
  auto samp = fixture_samp_3();
  auto pt   = must_lavaanify("f =~ x1 + b2*x2 + b3*x3\nb2 + b3 == 1.5");
  CHECK_FALSE(pt.has_unenforced_constraints);
  auto rep  = build_matrix_rep(pt).value();

  auto est_or = magmaan::fit::fit(pt, rep, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "constrained fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;
  CHECK(static_cast<std::int32_t>(est.theta.size()) == pt.n_free());

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(est.theta(k_x2) + est.theta(k_x3) == doctest::Approx(1.5).epsilon(1e-9));

  // df increases by 1 vs the unconstrained model.
  auto pt_u  = must_lavaanify("f =~ x1 + b2*x2 + b3*x3");
  auto rep_u = build_matrix_rep(pt_u).value();
  auto est_u = magmaan::fit::fit(pt_u, rep_u, samp).value();
  auto inf_u = expected_inference(pt_u, rep_u, samp, est_u).value();
  auto inf_c = expected_inference(pt, rep, samp, est).value();
  CHECK(inf_c.df == inf_u.df + 1);
  CHECK(inf_c.chi2 >= inf_u.chi2 - 1e-9);
  CHECK(inf_c.se.size() == pt.n_free());           // full n_free-length SEs
}

TEST_CASE("fit: `d == 0` pins a loading to zero") {
  auto samp = fixture_samp_3();
  auto pt   = must_lavaanify("f =~ x1 + d*x2 + x3\nd == 0");
  CHECK_FALSE(pt.has_unenforced_constraints);
  auto rep  = build_matrix_rep(pt).value();
  auto est  = magmaan::fit::fit(pt, rep, samp).value();
  auto ev   = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  CHECK(std::abs(est.theta(k_x2)) < 1e-8);
}

TEST_CASE("build_eq_constraints: an infeasible linear system errors out") {
  auto pt = must_lavaanify("f =~ x1 + d*x2 + x3\nd == 1\nd == 2");
  CHECK_FALSE(pt.has_unenforced_constraints);       // both rows ARE linear...
  auto con_or = build_eq_constraints(pt);           // ...but jointly infeasible
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().kind == magmaan::PostError::Kind::NumericIssue);
  CHECK(con_or.error().detail.find("infeasible") != std::string::npos);
}

TEST_CASE("build_eq_constraints: a genuinely nonlinear `==` is still rejected") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + b*x3\na == b*b");
  CHECK(pt.has_unenforced_constraints);             // b*b ⇒ not affine ⇒ stays flagged
  auto con_or = build_eq_constraints(pt);
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().kind == magmaan::PostError::Kind::NumericIssue);
}

// === effect coding =========================================================

TEST_CASE("lavaanify: effect_coding synthesizes `Σλ == #indicators` and frees everything") {
  LavaanifyOptions opts; opts.effect_coding = true;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);
  CHECK_FALSE(pt.has_unenforced_constraints);
  CHECK(pt.lin_constraint_d.size() == 1);           // one latent, one group
  CHECK(pt.lin_constraint_d[0] == doctest::Approx(3.0));
  // The R row has a 1 on each of the three loading columns and 0 elsewhere.
  REQUIRE(pt.lin_constraint_R.size() == static_cast<std::size_t>(pt.n_free()));
  double rowsum = 0.0; int ones = 0;
  for (double v : pt.lin_constraint_R) { rowsum += v; if (v == 1.0) ++ones; }
  CHECK(rowsum == doctest::Approx(3.0));
  CHECK(ones == 3);
  CHECK(pt.free[0] > 0);                             // first loading is free, not the marker

  // f ~~ f stays free under effect coding.
  auto rep = build_matrix_rep(pt).value();
  auto ev  = ModelEvaluator::build(pt, rep).value();
  bool psi_free = false;
  for (const auto& loc : ev.param_locations())
    if (loc.mat == MatId::Psi && loc.row == loc.col) psi_free = true;
  CHECK(psi_free);
}

TEST_CASE("lavaanify: effect_coding and std_lv are mutually exclusive") {
  LavaanifyOptions opts; opts.effect_coding = true; opts.std_lv = true;
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp, opts);
  REQUIRE_FALSE(pt.has_value());
  CHECK(pt.error().kind == magmaan::PartableError::Kind::BadGroupSpec);
}

TEST_CASE("fit: effect_coding — loadings sum to #indicators; χ²/df match the marker fit") {
  auto samp = fixture_samp_3();
  LavaanifyOptions opts; opts.effect_coding = true;
  auto pt  = must_lavaanify("f =~ x1 + x2 + x3", opts);
  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::fit::fit(pt, rep, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "effect-coding fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;

  auto ev = ModelEvaluator::build(pt, rep).value();
  const double lsum = est.theta(lambda_free_idx(ev, 0)) +
                      est.theta(lambda_free_idx(ev, 1)) +
                      est.theta(lambda_free_idx(ev, 2));
  CHECK(lsum == doctest::Approx(3.0).epsilon(1e-7));

  auto inf_ec = expected_inference(pt, rep, samp, est).value();
  // Marker fit of the same model — bijective reparam ⇒ identical χ² and df.
  auto pt_m  = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep_m = build_matrix_rep(pt_m).value();
  auto est_m = magmaan::fit::fit(pt_m, rep_m, samp).value();
  auto inf_m = expected_inference(pt_m, rep_m, samp, est_m).value();
  CHECK(inf_ec.df == inf_m.df);
  CHECK(inf_ec.chi2 == doctest::Approx(inf_m.chi2).epsilon(1e-6));
}
