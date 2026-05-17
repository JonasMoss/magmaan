#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/error.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::FitError;
using magmaan::optim::LbfgsBOptimizer;
using magmaan::optim::LbfgsOptimizer;
using magmaan::data::SampleStats;
using magmaan::spec::LavaanifyOptions;

namespace {

// Read a fixture file as a JSON document (skipping the read if missing).
nlohmann::json read_fixture(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.is_open(), "missing fixture: " << path);
  std::stringstream ss;  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  return j;
}

// Pull `sample_cov` (single block) and optionally `sample_mean` from a fit
// fixture into a SampleStats. Mirrors the loader logic in
// `fit_theta_golden_test.cpp` but is single-group only — what we need
// here.
SampleStats load_samp(const nlohmann::json& fix) {
  SampleStats samp;
  const auto& M = fix["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(fix["n_obs"].get<std::int64_t>());
  if (fix.contains("sample_mean") && !fix["sample_mean"].is_null()) {
    const auto& v = fix["sample_mean"][0]["vector"];
    Eigen::VectorXd mu(static_cast<Eigen::Index>(v.size()));
    for (Eigen::Index i = 0; i < mu.size(); ++i)
      mu(i) = v[static_cast<std::size_t>(i)].get<double>();
    samp.mean.push_back(std::move(mu));
  }
  return samp;
}

// Build (LatentStructure, MatrixRep) from a syntax string + options.
struct ModelBuild {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep          rep;
};

ModelBuild build_model(std::string_view src, bool meanstructure) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  LavaanifyOptions opts;
  opts.meanstructure = meanstructure;
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return ModelBuild{std::move(*pt), std::move(*mr)};
}

}  // namespace

// ============================================================================
// Subcase A — end-to-end parity check.
//
// On a fixture that BOTH optimizers can fit (`0026_two_factor_meanstructure_hs`:
// 2F + meanstructure on HS x1..x6, single group), confirm that
// `fit_bounded<ML, LbfgsBOptimizer>` lands at the same θ̂ as
// `fit<ML, LbfgsOptimizer>` to lavaan-comparable tolerance. This is the
// load-bearing assertion that the bounded path is correctly wired
// end-to-end (lavaanify → MatrixRep → ModelEvaluator → ML → LBFGS-B →
// auto-derived bounds).
// ============================================================================

TEST_CASE("LbfgsBOptimizer end-to-end: 2F+means HS — same θ̂ as unbounded LBFGS") {
  auto fix = read_fixture(std::string(MAGMAAN_FIXTURES_DIR) +
                          "/fit/0026_two_factor_meanstructure_hs.fit.json");
  auto m = build_model(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "x1 ~ 1\nx2 ~ 1\nx3 ~ 1\nx4 ~ 1\nx5 ~ 1\nx6 ~ 1",
      /*meanstructure=*/true);
  auto samp = load_samp(fix);

  // Unbounded baseline.
  auto base = magmaan::test::fit(m.pt, m.rep, samp);
  REQUIRE_MESSAGE(base.has_value(),
      "fit<> failed: " << (base.has_value() ? "" : base.error().detail));

  // Bounded — picks LbfgsBOptimizer + auto-bounds from partable by default.
  auto out = magmaan::test::fit_bounded(m.pt, m.rep, samp, {});
  REQUIRE_MESSAGE(out.has_value(),
      "fit_bounded<> failed: " << (out.has_value() ? "" : out.error().detail));

  // Same θ̂ to optimizer-disagreement tolerance. LBFGSSolver
  // (backtracking line search, no delta criterion) and LBFGSBSolver
  // (More–Thuente line search, delta-based ftol stopping) are two
  // distinct algorithms in the same family — empirically they land
  // within ~3-5e-6 on a flat ML section. The fit_theta golden's
  // ≤2e-6 tolerance is LBFGS vs lavaan's nlminb on a smaller model;
  // here both factors push the disagreement up slightly.
  const double max_diff = (out->theta - base->theta).cwiseAbs().maxCoeff();
  CHECK(max_diff < 5e-6);
  // fmin agrees to many more digits than θ̂: the ML surface is flat
  // along the directions where θ̂ wobbles, so the minimum value itself
  // is a much sharper invariant.
  CHECK(std::abs(out->fmin - base->fmin) < 1e-9);

  // (Variance-diagonal bounds are correctness-tested in
  // `lbfgsb_optimizer_test.cpp`'s Heywood case; here we'd need to know
  // which θ-indices are variances, which is what `bounds_from_partable`
  // already encodes — covered by the auto-bounds path under test.)
}

// ============================================================================
// Subcase B — G5b regression demonstration.
//
// Single-group 3F + meanstructure on HS x1..x9 is the documented
// LbfgsOptimizer line-search-failure case (`src/optim/lbfgs_optimizer.cpp:64`).
// We don't have a lavaan fit fixture for this exact model (the corpus
// route-around is `0026_two_factor_meanstructure_hs`), so this test:
//
//   1) builds S from the existing 9×9 `0002_three_factor_hs.fit.json`,
//   2) supplies a sample mean vector composed of the x1..x6 means from
//      `0026_two_factor_meanstructure_hs` (provenance: lavInspect on
//      HolzingerSwineford1939) and published x7–x9 means (HS 1939 / the
//      lavaan vignette: x7≈4.186, x8≈5.527, x9≈5.374),
//   3) asserts that `fit_bounded<ML, LbfgsBOptimizer>` converges.
//
// We deliberately do NOT assert that `fit<ML, LbfgsOptimizer>` fails on
// this exact synthetic-mean setup — the line-search failure depends on
// the data interaction, and with means we've borrowed from a different
// fixture we can't promise the failure path is hit. A future tranche that
// adds a real `0033_three_factor_meanstructure_hs` fixture should
// upgrade this assertion to lavaan-θ̂ parity.
// ============================================================================

TEST_CASE("LbfgsBOptimizer G5b: 3F+means HS — converges where LBFGS historically didn't") {
  auto fix_cov = read_fixture(std::string(MAGMAAN_FIXTURES_DIR) +
                              "/fit/0002_three_factor_hs.fit.json");
  auto m = build_model(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9",
      /*meanstructure=*/true);

  SampleStats samp = load_samp(fix_cov);   // S only — 0002 has sample_mean=null.

  // Synthesize a 9-vector mean. x1..x6 are exact (from 0026's fixture,
  // i.e. lavInspect on HS); x7..x9 are 3-decimal published values for
  // HolzingerSwineford1939. Precision is sufficient — this test asserts
  // convergence, not θ̂ parity.
  Eigen::VectorXd mu(9);
  mu << 4.93576965637874,
        6.08803986710963,
        2.25041528239203,
        3.06090808684385,
        4.34053156146179,
        2.18557190176080,
        4.186,    // HS x7
        5.527,    // HS x8
        5.374;    // HS x9
  samp.mean.push_back(std::move(mu));

  auto out = magmaan::test::fit_bounded(m.pt, m.rep, samp, {});
  REQUIRE_MESSAGE(out.has_value(),
      "fit_bounded<LbfgsBOptimizer> on 3F+means did not converge: " <<
      (out.has_value() ? "" : out.error().detail));
  CHECK(std::isfinite(out->fmin));
  CHECK(out->iterations > 0);
  // 3F+means HS has 30 free parameters; lavaan converges in ~30 iters via
  // nlminb. LBFGS-B with M=10 history typically takes 50-150. 500 is the
  // default max_iter; if we needed > 250 something is off.
  CHECK(out->iterations < 250);
}
