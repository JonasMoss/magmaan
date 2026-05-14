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
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

// P9 phase 2 — general *linear* `==` equality constraints. These goldens pin a
// 1-factor CFA fitted with `b2 + b3 == 1.5` (lavaan enforces it via its
// Lagrange path; magmaan by an affine reparam θ = θ₀ + Kα) against
// `lavaan::cfa(model, data)`. Fixtures live under tests/fixtures/fit_lincon/,
// produced by the dedicated section in tools/regen_oracle.R (kept out of the
// shared corpus, like fit_stdlv/).

namespace {

const std::vector<std::string> kLinConFixtures = {
    "0001_loading_sum_hs",
};

magmaan::data::SampleStats sample_from_fixture(const nlohmann::json& exp) {
  magmaan::data::SampleStats samp;
  const auto& blocks = exp["sample_cov"];
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& M = blocks[b]["matrix"];
    const Eigen::Index p = static_cast<Eigen::Index>(M.size());
    Eigen::MatrixXd S(p, p);
    for (Eigen::Index r = 0; r < p; ++r)
      for (Eigen::Index c = 0; c < p; ++c)
        S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>();
    samp.S.push_back(std::move(S));
    samp.n_obs.push_back(exp["n_obs"].get<std::int64_t>());
  }
  return samp;
}

// Free index of Λ[row, 0] (loading of the `row`-th observed variable on the
// sole latent), or -1.
Eigen::Index lambda_free_idx(const magmaan::model::ModelEvaluator& ev, std::int16_t row) {
  const auto locs = ev.param_locations();
  for (std::size_t k = 0; k < locs.size(); ++k)
    if (locs[k].mat == magmaan::model::MatId::Lambda && locs[k].row == row)
      return static_cast<Eigen::Index>(k);
  return -1;
}

}  // namespace

TEST_CASE("linear-constraint goldens — θ̂/SE/χ²/df match lavaan(+ `==`)") {
  const std::string dir = magmaan::test::fixtures_dir() + "/fit_lincon";

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kLinConFixtures) {
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) { failures.push_back(id + ": missing fixture"); continue; }
    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (exp.is_discarded()) { failures.push_back(id + ": invalid JSON"); continue; }
    ++total;

    const std::string model = exp["input"].get<std::string>();
    auto fp = magmaan::parse::Parser::parse(model);
    if (!fp.has_value()) { failures.push_back(id + ": parse"); continue; }
    auto pt = magmaan::spec::lavaanify(*fp);
    if (!pt.has_value()) { failures.push_back(id + ": lavaanify — " + pt.error().detail); continue; }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) { failures.push_back(id + ": matrix_rep — " + mr.error().detail); continue; }

    const magmaan::data::SampleStats samp = sample_from_fixture(exp);

    auto est_or = magmaan::estimate::fit(*pt, *mr, samp);
    if (!est_or.has_value()) { failures.push_back(id + ": fit — " + est_or.error().detail); continue; }
    const auto& est = *est_or;

    auto info_or = magmaan::nt::infer::information_expected(*pt, *mr, samp, est);
    if (!info_or.has_value()) { failures.push_back(id + ": information_expected — " + info_or.error().detail); continue; }
    auto vcov_or = magmaan::nt::infer::vcov(*info_or, *pt);
    if (!vcov_or.has_value()) { failures.push_back(id + ": vcov — " + vcov_or.error().detail); continue; }
    const Eigen::VectorXd se_v = magmaan::nt::infer::se(*vcov_or);
    const double          chi2 = magmaan::nt::infer::chi2_stat(samp, est);
    auto df_or = magmaan::nt::infer::df_stat(*pt, samp);
    if (!df_or.has_value()) { failures.push_back(id + ": df_stat — " + df_or.error().detail); continue; }
    const int df = *df_or;

    bool ok = true;
    char buf[256];

    // θ̂ — same comparable-optimizer tolerance as the std.lv / 3F goldens.
    const auto& th = exp["theta_hat"];
    if (static_cast<std::size_t>(est.theta.size()) != th.size()) {
      failures.push_back(id + ": n_free mismatch"); continue;
    }
    double max_th = 0.0;
    for (Eigen::Index k = 0; k < est.theta.size(); ++k)
      max_th = std::max(max_th, std::abs(est.theta(k) - th[static_cast<std::size_t>(k)].get<double>()));
    if (max_th > 5e-6) {
      std::snprintf(buf, sizeof(buf), "max |θ̂ - lavaan| = %.3e", max_th);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // df — exact, and one more than the unconstrained model.
    const int df_lavaan = exp["df"].get<int>();
    if (df != df_lavaan) {
      std::snprintf(buf, sizeof(buf), "df = %d, lavaan = %d", df, df_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }
    if (exp.contains("unconstrained_df") && !exp["unconstrained_df"].is_null()) {
      const int u = exp["unconstrained_df"].get<int>();
      if (df != u + 1) {
        std::snprintf(buf, sizeof(buf), "df %d != unconstrained_df %d + 1", df, u);
        failures.push_back(id + ": " + buf); ok = false;
      }
    }

    // χ² — ≤ 1e-3 absolute.
    const double chi2_lavaan = exp["chi2"].get<double>();
    if (std::abs(chi2 - chi2_lavaan) > 1e-3) {
      std::snprintf(buf, sizeof(buf), "|χ² - lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                    std::abs(chi2 - chi2_lavaan), chi2, chi2_lavaan);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // SE — max abs diff ≤ 1e-4.
    const auto& se_arr = exp["se"];
    if (static_cast<std::size_t>(se_v.size()) != se_arr.size()) {
      failures.push_back(id + ": se length mismatch"); continue;
    }
    double max_se = 0.0;
    for (Eigen::Index k = 0; k < se_v.size(); ++k)
      max_se = std::max(max_se, std::abs(se_v(k) - se_arr[static_cast<std::size_t>(k)].get<double>()));
    if (max_se > 1e-4) {
      std::snprintf(buf, sizeof(buf), "max |se - lavaan| = %.3e", max_se);
      failures.push_back(id + ": " + buf); ok = false;
    }

    // The constraint actually binds: λ_x2 + λ_x3 ≈ 1.5.
    auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr);
    if (ev.has_value()) {
      const Eigen::Index k2 = lambda_free_idx(*ev, 1);
      const Eigen::Index k3 = lambda_free_idx(*ev, 2);
      if (k2 >= 0 && k3 >= 0 &&
          std::abs(est.theta(k2) + est.theta(k3) - 1.5) > 1e-5) {
        std::snprintf(buf, sizeof(buf), "λ_x2 + λ_x3 = %.6f, expected 1.5",
                      est.theta(k2) + est.theta(k3));
        failures.push_back(id + ": " + buf); ok = false;
      }
    }

    if (ok) ++passed;
  }

  MESSAGE("linear-constraint goldens: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
