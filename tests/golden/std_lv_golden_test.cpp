#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "latva/fit/fit.hpp"
#include "latva/fit/inference.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

// Stage D of the ParTable split refactor: the `std_lv` knob on
// LavaanifyOptions. These goldens pin a CFA fitted with `std.lv = TRUE`
// (latent variances fixed at 1, all loadings free) against
// `lavaan::cfa(model, data, std.lv = TRUE)` — fixtures under
// tests/fixtures/fit_stdlv/ are produced by the dedicated section in
// tools/regen_oracle.R (kept out of the shared corpus so the marker-mode
// golden layers aren't perturbed).

namespace {

// Models in tests/fixtures/fit_stdlv/. Each `<id>.fit.json` carries:
//   model, n_obs, sample_cov[0].matrix, theta_hat (free-index order),
//   se, chi2, df.
const std::vector<std::string> kStdLvFixtures = {
    "0001_three_factor_hs",
};

latva::fit::SampleStats sample_from_fixture(const nlohmann::json& exp) {
  latva::fit::SampleStats samp;
  const auto& sample_blocks = exp["sample_cov"];
  for (std::size_t b = 0; b < sample_blocks.size(); ++b) {
    const auto& M = sample_blocks[b]["matrix"];
    const Eigen::Index p = static_cast<Eigen::Index>(M.size());
    Eigen::MatrixXd S(p, p);
    for (Eigen::Index r = 0; r < p; ++r)
      for (Eigen::Index c = 0; c < p; ++c)
        S(r, c) = M[static_cast<std::size_t>(r)]
                   [static_cast<std::size_t>(c)].get<double>();
    samp.S.push_back(std::move(S));
    samp.n_obs.push_back(exp["n_obs"].get<std::int64_t>());
  }
  return samp;
}

}  // namespace

TEST_CASE("std.lv goldens — θ̂/SE/χ²/df match lavaan(std.lv=TRUE)") {
  const std::string dir = latva::test::fixtures_dir() + "/fit_stdlv";

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kStdLvFixtures) {
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw = latva::test::read_fixture(path);
    if (!raw.has_value()) { failures.push_back(id + ": missing fixture"); continue; }
    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (exp.is_discarded()) { failures.push_back(id + ": invalid JSON"); continue; }
    ++total;

    const std::string model = exp["input"].get<std::string>();
    auto fp = latva::parse::Parser::parse(model);
    if (!fp.has_value()) { failures.push_back(id + ": parse"); continue; }

    latva::partable::LavaanifyOptions opts;
    opts.std_lv = true;
    auto pt = latva::partable::lavaanify(*fp, opts);
    if (!pt.has_value()) {
      failures.push_back(id + ": lavaanify — " + pt.error().detail);
      continue;
    }
    auto mr = latva::model::build_matrix_rep(*pt);
    if (!mr.has_value()) {
      failures.push_back(id + ": matrix_rep — " + mr.error().detail);
      continue;
    }

    const latva::fit::SampleStats samp = sample_from_fixture(exp);

    auto est_or = latva::fit::fit(*pt, *mr, samp);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    latva::fit::ExpectedInfoSE se_method;
    auto inf_or = se_method.compute(*pt, *mr, samp, est);
    if (!inf_or.has_value()) {
      failures.push_back(id + ": inference — " + inf_or.error().detail);
      continue;
    }
    const auto& inf = *inf_or;

    bool ok = true;
    char buf[256];

    // 1) θ̂ — LBFGS vs lavaan's nlminb converge to slightly different points
    //    on the flat section of the ML surface (objective equal to machine
    //    precision; see fit_theta_golden_test.cpp's note). Under std.lv the
    //    loadings are O(1) rather than O(residual-variance), so the absolute
    //    disagreement scales up to ~4e-6 — 5e-6 is the honest comparable-
    //    optimizer tolerance here. The χ² (≤1e-3) and SE (≤1e-4) checks below
    //    are the substantive ones; revisit if we wire nlminb directly.
    const auto& th = exp["theta_hat"];
    if (static_cast<std::size_t>(est.theta.size()) != th.size()) {
      failures.push_back(id + ": n_free mismatch"); continue;
    }
    double max_th = 0.0;
    for (Eigen::Index k = 0; k < est.theta.size(); ++k)
      max_th = std::max(max_th,
          std::abs(est.theta(k) - th[static_cast<std::size_t>(k)].get<double>()));
    if (max_th > 5e-6) {
      std::snprintf(buf, sizeof(buf), "max |θ̂ - θ̂_lavaan| = %.3e", max_th);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 2) df — exact.
    const int df_lavaan = exp["df"].get<int>();
    if (inf.df != df_lavaan) {
      std::snprintf(buf, sizeof(buf), "df = %d, lavaan = %d", inf.df, df_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 3) χ² — ≤ 1e-3 absolute (matches inference_golden_test).
    const double chi2_lavaan = exp["chi2"].get<double>();
    const double chi2_diff = std::abs(inf.chi2 - chi2_lavaan);
    if (chi2_diff > 1e-3) {
      std::snprintf(buf, sizeof(buf), "|χ² - lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                    chi2_diff, inf.chi2, chi2_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 4) SE — max abs diff ≤ 1e-4.
    const auto& se_arr = exp["se"];
    if (static_cast<std::size_t>(inf.se.size()) != se_arr.size()) {
      failures.push_back(id + ": se length mismatch"); continue;
    }
    double max_se = 0.0;
    for (Eigen::Index k = 0; k < inf.se.size(); ++k)
      max_se = std::max(max_se,
          std::abs(inf.se(k) - se_arr[static_cast<std::size_t>(k)].get<double>()));
    if (max_se > 1e-4) {
      std::snprintf(buf, sizeof(buf), "max |se - lavaan| = %.3e", max_se);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 5) Bijectivity: the same model under the (default) marker convention
    //    must reach the same #df and the same χ² — marker and std.lv are
    //    bijective reparameterizations of one fit.
    auto pt_m = latva::partable::lavaanify(*fp);   // marker (default)
    auto mr_m = latva::model::build_matrix_rep(*pt_m);
    if (pt_m.has_value() && mr_m.has_value()) {
      auto est_m = latva::fit::fit(*pt_m, *mr_m, samp);
      if (est_m.has_value()) {
        auto inf_m = se_method.compute(*pt_m, *mr_m, samp, *est_m);
        if (inf_m.has_value()) {
          if (inf_m->df != inf.df) {
            failures.push_back(id + ": marker/std.lv df disagree"); ok = false;
          }
          if (std::abs(inf_m->chi2 - inf.chi2) > 1e-4) {
            std::snprintf(buf, sizeof(buf),
                "marker χ²=%.6f vs std.lv χ²=%.6f", inf_m->chi2, inf.chi2);
            failures.push_back(id + ": " + buf); ok = false;
          }
        }
      }
    }

    if (ok) ++passed;
  }

  MESSAGE("std.lv goldens: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
