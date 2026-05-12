#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "latva/error.hpp"
#include "latva/fit/constraints.hpp"
#include "latva/fit/fit.hpp"
#include "latva/fit/inference.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/model/model_evaluator.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

using latva::fit::build_eq_constraints;
using latva::fit::ExpectedInfoSE;
using latva::fit::SampleStats;
using latva::model::build_matrix_rep;
using latva::model::MatId;
using latva::model::ModelEvaluator;
using latva::parse::Parser;
using latva::partable::lavaanify;
using latva::partable::ParTable;

namespace {

ParTable must_lavaanify(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  return std::move(*pt);
}

// 3×3 sample covariance (x1,x2,x3) + n from the saturated 1F-CFA fixture.
SampleStats fixture_samp_3() {
  std::ifstream in(std::string(LATVA_FIXTURES_DIR) +
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

  auto est_or = latva::fit::fit(pt, rep, samp);
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
  auto est_u = latva::fit::fit(pt_u, rep_u, samp).value();
  auto inf_u = ExpectedInfoSE{}.compute(pt_u, rep_u, samp, est_u).value();
  CHECK(inf_u.df == 0);
  CHECK(inf_u.chi2 < 1e-6);

  // Constrained: x2 & x3 loadings tied → one fewer effective free param.
  auto pt_c  = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto rep_c = build_matrix_rep(pt_c).value();
  auto est_c = latva::fit::fit(pt_c, rep_c, samp).value();
  auto inf_or = ExpectedInfoSE{}.compute(pt_c, rep_c, samp, est_c);
  REQUIRE_MESSAGE(inf_or.has_value(),
      "constrained ExpectedInfoSE failed: " <<
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
  CHECK(con_or.error().kind == latva::PostError::Kind::NumericIssue);
}
