#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// Stage D of the ParTable split refactor: the `std_lv` knob on
// BuildOptions. These goldens pin a CFA fitted with `std.lv = TRUE`
// (latent variances fixed at 1, all loadings free) against
// `lavaan::cfa(model, data, std.lv = TRUE)` — fixtures under
// tests/fixtures/fit_stdlv/ are produced by the dedicated section in
// tests/tools/regen_oracle.R (kept out of the shared corpus so the marker-mode
// golden layers aren't perturbed).

namespace {

// Models in tests/fixtures/fit_stdlv/. Each `<id>.fit.json` carries:
//   model, n_obs, sample_cov[b].matrix, theta_hat (free-index order),
//   se, chi2, df. Multi-group fixtures additionally carry n_groups,
//   group_var, group_labels, group_equal, meanstructure, and an `n_obs`
//   *array* (one per group) rather than a scalar.
const std::vector<std::string> kStdLvFixtures = {
    "0001_three_factor_hs",
    "0002_three_factor_hs_2group_loadings",
};

magmaan::data::SampleStats sample_from_fixture(const nlohmann::json& exp) {
  magmaan::data::SampleStats samp;
  const auto& sample_blocks = exp["sample_cov"];
  const auto& n_obs = exp["n_obs"];
  for (std::size_t b = 0; b < sample_blocks.size(); ++b) {
    const auto& M = sample_blocks[b]["matrix"];
    const Eigen::Index p = static_cast<Eigen::Index>(M.size());
    Eigen::MatrixXd S(p, p);
    for (Eigen::Index r = 0; r < p; ++r)
      for (Eigen::Index c = 0; c < p; ++c)
        S(r, c) = M[static_cast<std::size_t>(r)]
                   [static_cast<std::size_t>(c)].get<double>();
    samp.S.push_back(std::move(S));
    // Single-group fixtures store a scalar ntotal; multi-group ones store a
    // per-group array.
    samp.n_obs.push_back(n_obs.is_array() ? n_obs[b].get<std::int64_t>()
                                          : n_obs.get<std::int64_t>());
  }
  return samp;
}

// Fixture `group_equal` strings -> the GroupEqual enum. Returns false on an
// unrecognized family rather than dropping it: a silently ignored group.equal
// would turn this golden into a weaker test without anyone noticing.
bool group_equal_from_string(const std::string& s,
                             magmaan::spec::GroupEqual& out) {
  using GE = magmaan::spec::GroupEqual;
  if (s == "loadings")             { out = GE::Loadings;            return true; }
  if (s == "thresholds")           { out = GE::Thresholds;          return true; }
  if (s == "intercepts")           { out = GE::Intercepts;          return true; }
  if (s == "means")                { out = GE::Means;               return true; }
  if (s == "residuals")            { out = GE::Residuals;           return true; }
  if (s == "residual.covariances") { out = GE::ResidualCovariances; return true; }
  if (s == "lv.variances")         { out = GE::LvVariances;         return true; }
  if (s == "lv.covariances")       { out = GE::LvCovariances;       return true; }
  if (s == "regressions")          { out = GE::Regressions;         return true; }
  return false;
}

// Apply a fixture's group axis to BuildOptions. `err` is set (and false
// returned) when the fixture names a group.equal family we cannot map.
bool apply_group_options(const nlohmann::json& exp,
                         magmaan::spec::BuildOptions& opts,
                         std::string& err) {
  if (!exp.contains("n_groups")) return true;
  opts.n_groups = exp["n_groups"].get<std::int32_t>();
  if (exp.contains("group_labels"))
    for (const auto& g : exp["group_labels"])
      opts.group_labels.push_back(g.get<std::string>());
  if (exp.contains("meanstructure"))
    opts.meanstructure = exp["meanstructure"].get<bool>();
  if (exp.contains("group_equal")) {
    const auto& ge = exp["group_equal"];
    std::vector<std::string> fams;
    if (ge.is_array())
      for (const auto& f : ge) fams.push_back(f.get<std::string>());
    else
      fams.push_back(ge.get<std::string>());
    for (const auto& f : fams) {
      magmaan::spec::GroupEqual g{};
      if (!group_equal_from_string(f, g)) {
        err = "unmapped group_equal family '" + f + "'";
        return false;
      }
      opts.group_equal.push_back(g);
    }
  }
  return true;
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

    magmaan::spec::BuildOptions opts;
    opts.std_lv = true;
    std::string gerr;
    if (!apply_group_options(exp, opts, gerr)) {
      failures.push_back(id + ": " + gerr);
      continue;
    }
    auto pt = magmaan::spec::build(*fp, opts);
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

    auto est_or = magmaan::test::fit(*pt, *mr, samp);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    auto info_or = magmaan::inference::information_expected(*pt, *mr, samp, est);
    if (!info_or.has_value()) {
      failures.push_back(id + ": information_expected — " + info_or.error().detail);
      continue;
    }
    auto vcov_or = magmaan::inference::vcov(*info_or, *pt);
    if (!vcov_or.has_value()) {
      failures.push_back(id + ": vcov — " + vcov_or.error().detail);
      continue;
    }
    const Eigen::VectorXd se_v = magmaan::inference::se(*vcov_or);
    const double          chi2 = magmaan::inference::chi2_stat(samp, est);
    auto df_or = magmaan::inference::df_stat(*pt, samp);
    if (!df_or.has_value()) {
      failures.push_back(id + ": df_stat — " + df_or.error().detail);
      continue;
    }
    const int df = *df_or;

    bool ok = true;
    char buf[256];

    // 1) θ̂ — magmaan's optimizer and lavaan's nlminb converge to slightly
    //    different points on the flat section of the ML surface (objective
    //    equal to machine precision; see fit_theta_golden_test.cpp's note).
    //    Under std.lv the loadings are O(1) rather than O(residual-variance),
    //    so the absolute disagreement is larger than in the marker goldens.
    //    Measured: 3.6e-6 single-group, 8.5e-6 for the 2-group metric-
    //    invariance fit (45 raw parameters under 9 cross-group equalities, so
    //    the search runs in a rotated 36-dim α-space with more flat
    //    directions). 1e-5 covers both.
    //
    //    This bound is deliberately NOT the sharp part of the test. Over the
    //    same two fits the χ² agrees to 2.6e-9 and 3.9e-9 — nine significant
    //    figures of objective agreement against an 8.5e-6 parameter
    //    displacement, which is what "flat direction" means quantitatively.
    //    So the χ² bound below carries the discriminating power and is set
    //    1000x tighter than the marker goldens' 1e-3 rather than being
    //    inherited from them.
    const auto& th = exp["theta_hat"];
    if (static_cast<std::size_t>(est.theta.size()) != th.size()) {
      // Worth reporting both counts: a multi-group std.lv n_free deficit of
      // exactly (G-1)*n_lv is the signature of the latent variances not being
      // released in groups 2..G.
      std::snprintf(buf, sizeof(buf), "n_free = %lld, lavaan = %zu",
                    static_cast<long long>(est.theta.size()), th.size());
      failures.push_back(id + ": " + buf); continue;
    }
    double max_th = 0.0;
    for (Eigen::Index k = 0; k < est.theta.size(); ++k)
      max_th = std::max(max_th,
          std::abs(est.theta(k) - th[static_cast<std::size_t>(k)].get<double>()));
    if (max_th > 1e-5) {
      std::snprintf(buf, sizeof(buf), "max |θ̂ - θ̂_lavaan| = %.3e", max_th);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 2) df — exact.
    const int df_lavaan = exp["df"].get<int>();
    if (df != df_lavaan) {
      std::snprintf(buf, sizeof(buf), "df = %d, lavaan = %d", df, df_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // 3) χ² — ≤ 1e-6 absolute. Observed 2.6e-9 / 3.9e-9, so this keeps ~250x
    //    headroom while still being the sharpest assertion in the test.
    const double chi2_lavaan = exp["chi2"].get<double>();
    const double chi2_diff = std::abs(chi2 - chi2_lavaan);
    if (chi2_diff > 1e-6) {
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
    //    bijective reparameterizations of one fit. The group axis has to be
    //    carried over too, otherwise this silently compares a 2-group std.lv
    //    fit against a 1-group marker fit and the df check is meaningless.
    magmaan::spec::BuildOptions opts_m;      // marker: std_lv left false
    if (!apply_group_options(exp, opts_m, gerr)) {
      failures.push_back(id + ": marker " + gerr);
      continue;
    }
    auto pt_m = magmaan::spec::build(*fp, opts_m);
    auto mr_m = magmaan::model::build_matrix_rep(*pt_m);
    if (pt_m.has_value() && mr_m.has_value()) {
      auto est_m = magmaan::test::fit(*pt_m, *mr_m, samp);
      if (est_m.has_value()) {
        const double chi2_m = magmaan::inference::chi2_stat(samp, *est_m);
        auto df_m_or = magmaan::inference::df_stat(*pt_m, samp);
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
