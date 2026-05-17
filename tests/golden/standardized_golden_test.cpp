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
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// Post-hoc standardized-solution parity vs lavaan::standardizedSolution(fit,
// type = "std.lv" / "std.all"). Distinct from std_lv_golden_test.cpp, which
// pins the std.lv *identification convention* (`lavaanify(std_lv=TRUE)`); here
// we fit in the marker parameterization and apply the post-hoc transforms.
// Fixtures under tests/fixtures/fit_std/ are produced by the dedicated section
// in tools/regen_oracle.R. Each carries, per free θ index (free-index order):
//   par_lhs/par_op/par_rhs/par_block (for debugging), std_lv_est/std_lv_se,
//   std_all_est/std_all_se, plus per-block sample_cov (+ sample_mean for the
//   meanstructure models) and meanstructure/n_groups so the C++ side re-fits.

namespace {

const std::vector<std::string> kStdFixtures = {
    "0001_three_factor_hs",          // cov-only — exercises the std.lv Ψ off-diagonals (G7)
    "0002_cfa_plus_structural_hs",   // structural Beta paths — std.all latent-variance rescaling
    "0003_three_factor_hs_2group",   // configural 2-group + meanstructure — ν rescaling under std.all
};

}  // namespace

TEST_CASE("standardized-solution goldens — std.lv / std.all vs lavaan") {
  const std::string dir = magmaan::test::fixtures_dir() + "/fit_std";

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kStdFixtures) {
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
    opts.n_groups      = exp["n_groups"].get<int>();
    opts.meanstructure = exp["meanstructure"].get<bool>();
    auto pt = magmaan::spec::lavaanify(*fp, opts);
    if (!pt.has_value()) { failures.push_back(id + ": lavaanify — " + pt.error().detail); continue; }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) { failures.push_back(id + ": matrix_rep — " + mr.error().detail); continue; }

    // Assemble (possibly multi-block) SampleStats from the fixture.
    magmaan::data::SampleStats samp;
    const auto n_blocks = static_cast<std::size_t>(exp["sample_cov"].size());
    const bool has_means = exp.contains("sample_mean") && !exp["sample_mean"].is_null();
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const auto& M = exp["sample_cov"][b]["matrix"];
      const Eigen::Index p = static_cast<Eigen::Index>(M.size());
      Eigen::MatrixXd S(p, p);
      for (Eigen::Index r = 0; r < p; ++r)
        for (Eigen::Index c = 0; c < p; ++c)
          S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                        .get<double>();
      samp.S.push_back(std::move(S));
      if (has_means) {
        const auto& v = exp["sample_mean"][b]["vector"];
        Eigen::VectorXd mu(static_cast<Eigen::Index>(v.size()));
        for (Eigen::Index i = 0; i < mu.size(); ++i)
          mu(i) = v[static_cast<std::size_t>(i)].get<double>();
        samp.mean.push_back(std::move(mu));
      }
      const auto& nob = exp["n_obs_per_block"];
      samp.n_obs.push_back(nob.is_array()
          ? nob[b].get<std::int64_t>() : nob.get<std::int64_t>());
    }

    auto est_or = magmaan::test::fit(*pt, *mr, samp);
    if (!est_or.has_value()) { failures.push_back(id + ": fit — " + est_or.error().detail); continue; }
    const auto& est = *est_or;

    auto info_or = magmaan::nt::infer::information_expected(*pt, *mr, samp, est);
    if (!info_or.has_value()) { failures.push_back(id + ": information_expected — " + info_or.error().detail); continue; }
    auto vcov_or = magmaan::nt::infer::vcov(*info_or, *pt);
    if (!vcov_or.has_value()) { failures.push_back(id + ": vcov — " + vcov_or.error().detail); continue; }
    const Eigen::MatrixXd& vcov_m = *vcov_or;

    auto slv_or  = magmaan::nt::standardize::standardize_lv(*pt, *mr, est, vcov_m);
    auto sall_or = magmaan::nt::standardize::standardize_all(*pt, *mr, est, vcov_m);
    if (!slv_or.has_value())  { failures.push_back(id + ": standardize_lv — " + slv_or.error().detail); continue; }
    if (!sall_or.has_value()) { failures.push_back(id + ": standardize_all — " + sall_or.error().detail); continue; }
    const auto& slv  = *slv_or;
    const auto& sall = *sall_or;

    const auto& lv_est  = exp["std_lv_est"];
    const auto& lv_se   = exp["std_lv_se"];
    const auto& all_est = exp["std_all_est"];
    const auto& all_se  = exp["std_all_se"];
    const std::size_t n_free = lv_est.size();
    if (static_cast<std::size_t>(est.theta.size()) != n_free ||
        static_cast<std::size_t>(slv.theta.size())  != n_free ||
        static_cast<std::size_t>(sall.theta.size()) != n_free) {
      failures.push_back(id + ": n_free mismatch"); continue;
    }

    bool ok = true;
    char buf[256];
    // Value tolerance 1e-4 (smooth in θ̂; θ̂ matches lavaan only to ~few·1e-6
    // on the flat ML surface). SE tolerance 1e-3 — the delta-method SE rides
    // on a vcov matched to ~1e-4, and lavaan's standardizedSolution uses the
    // same delta method.
    for (std::size_t k = 0; k < n_free && ok; ++k) {
      const auto kk = static_cast<Eigen::Index>(k);
      const std::string tag = id + " [" +
          exp["par_lhs"][k].get<std::string>() +
          exp["par_op"][k].get<std::string>() +
          exp["par_rhs"][k].get<std::string>() +
          " blk" + std::to_string(exp["par_block"][k].get<int>()) + "]";
      struct Check { const char* name; double ours; double lavaan; double tol; };
      const Check checks[] = {
          {"std.lv  est", slv.theta(kk),  lv_est[k].get<double>(),  1e-4},
          {"std.lv  se",  slv.se(kk),     lv_se[k].get<double>(),   1e-3},
          {"std.all est", sall.theta(kk), all_est[k].get<double>(), 1e-4},
          {"std.all se",  sall.se(kk),    all_se[k].get<double>(),  1e-3},
      };
      for (const auto& c : checks) {
        if (std::abs(c.ours - c.lavaan) > c.tol) {
          std::snprintf(buf, sizeof(buf), "%s %s ours=%.6f lavaan=%.6f (diff=%.2e)",
                        c.name, tag.c_str(), c.ours, c.lavaan,
                        std::abs(c.ours - c.lavaan));
          failures.push_back(buf);
          ok = false;
          break;
        }
      }
    }

    if (ok) ++passed;
  }

  MESSAGE("standardized goldens: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
