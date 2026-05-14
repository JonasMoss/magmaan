#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/nt/infer.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

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

magmaan::data::SampleStats sample_from_fixture(const nlohmann::json& exp) {
  magmaan::data::SampleStats samp;
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
  const std::string dir = magmaan::test::fixtures_dir() + "/fit_stdlv";

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kStdLvFixtures) {
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) { failures.push_back(id + ": missing fixture"); continue; }
    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (exp.is_discarded()) { failures.push_back(id + ": invalid JSON"); continue; }
    ++total;

    const std::string model = exp["input"].get<std::string>();
    auto fp = magmaan::parse::Parser::parse(model);
    if (!fp.has_value()) { failures.push_back(id + ": parse"); continue; }

    magmaan::spec::LavaanifyOptions opts;
    opts.std_lv = true;
    auto pt = magmaan::spec::lavaanify(*fp, opts);
    if (!pt.has_value()) {
      failures.push_back(id + ": lavaanify — " + pt.error().detail);
      continue;
    }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) {
      failures.push_back(id + ": matrix_rep — " + mr.error().detail);
      continue;
    }

    const magmaan::data::SampleStats samp = sample_from_fixture(exp);

    auto est_or = magmaan::estimate::fit(*pt, *mr, samp);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    auto info_or = magmaan::nt::infer::information_expected(*pt, *mr, samp, est);
    if (!info_or.has_value()) {
      failures.push_back(id + ": information_expected — " + info_or.error().detail);
      continue;
    }
    auto vcov_or = magmaan::nt::infer::vcov(*info_or, *pt);
    if (!vcov_or.has_value()) {
      failures.push_back(id + ": vcov — " + vcov_or.error().detail);
      continue;
    }
    const Eigen::VectorXd se_v = magmaan::nt::infer::se(*vcov_or);
    const double          chi2 = magmaan::nt::infer::chi2_stat(samp, est);
    auto df_or = magmaan::nt::infer::df_stat(*pt, samp);
    if (!df_or.has_value()) {
      failures.push_back(id + ": df_stat — " + df_or.error().detail);
      continue;
    }
    const int df = *df_or;

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
    if (df != df_lavaan) {
      std::snprintf(buf, sizeof(buf), "df = %d, lavaan = %d", df, df_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 3) χ² — ≤ 1e-3 absolute (matches inference_golden_test).
    const double chi2_lavaan = exp["chi2"].get<double>();
    const double chi2_diff = std::abs(chi2 - chi2_lavaan);
    if (chi2_diff > 1e-3) {
      std::snprintf(buf, sizeof(buf), "|χ² - lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                    chi2_diff, chi2, chi2_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 4) SE — max abs diff ≤ 1e-4.
    const auto& se_arr = exp["se"];
    if (static_cast<std::size_t>(se_v.size()) != se_arr.size()) {
      failures.push_back(id + ": se length mismatch"); continue;
    }
    double max_se = 0.0;
    for (Eigen::Index k = 0; k < se_v.size(); ++k)
      max_se = std::max(max_se,
          std::abs(se_v(k) - se_arr[static_cast<std::size_t>(k)].get<double>()));
    if (max_se > 1e-4) {
      std::snprintf(buf, sizeof(buf), "max |se - lavaan| = %.3e", max_se);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 5) Bijectivity: the same model under the (default) marker convention
    //    must reach the same #df and the same χ² — marker and std.lv are
    //    bijective reparameterizations of one fit.
    auto pt_m = magmaan::spec::lavaanify(*fp);   // marker (default)
    auto mr_m = magmaan::model::build_matrix_rep(*pt_m);
    if (pt_m.has_value() && mr_m.has_value()) {
      auto est_m = magmaan::estimate::fit(*pt_m, *mr_m, samp);
      if (est_m.has_value()) {
        const double chi2_m = magmaan::nt::infer::chi2_stat(samp, *est_m);
        auto df_m_or = magmaan::nt::infer::df_stat(*pt_m, samp);
        if (df_m_or.has_value()) {
          if (*df_m_or != df) {
            failures.push_back(id + ": marker/std.lv df disagree"); ok = false;
          }
          if (std::abs(chi2_m - chi2) > 1e-4) {
            std::snprintf(buf, sizeof(buf),
                "marker χ²=%.6f vs std.lv χ²=%.6f", chi2_m, chi2);
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
