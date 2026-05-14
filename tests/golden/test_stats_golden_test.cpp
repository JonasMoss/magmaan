#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
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

// Golden parity for the per-parameter test statistics layered on top of an
// ExpectedInfoSE fit:
//   • `z_test`     vs lavaan `parameterEstimates(fit)$z` / `$pvalue`
//   • `chi2_pvalue` (the engine behind the z p-value)
//   • `wald_test`  vs lavaan `lavTestWald(fit, "<plabel> == 0")` on the first
//     free loading of the 3F Holzinger model.
// `lr_test` is just the subtraction of two already-golden `Inference`s, so it
// gets no separate golden here.

namespace {

// Same under-identified skip list the other inference goldens use — lavaan
// can't invert the information matrix for these, so it has no z/Wald oracle.
const std::set<std::string> kSkipForTestStats = {
    "0010_covariance",
    "0013_string_label",
    "0015_start_call",
    "0018_na_modifier",
};

}  // namespace

TEST_CASE("test-stat goldens — z / p-value / Wald vs lavaan") {
  const auto corpus = magmaan::test::load_corpus();
  const std::string fit_dir = magmaan::test::fixtures_dir() + "/fit";

  int total = 0, passed = 0;
  std::vector<std::string> failures, skipped;

  for (const auto& e : corpus) {
    if (e.n_groups > 1) continue;          // single-group only
    if (kSkipForTestStats.count(e.id)) { skipped.push_back(e.id); continue; }
    const std::string path = fit_dir + "/" + e.id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) continue;
    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (exp.is_discarded()) { failures.push_back(e.id + ": invalid JSON"); continue; }
    if (!exp.contains("pe_z") || exp["pe_z"].is_null()) continue;  // pre-regen fixture
    ++total;

    auto fp = magmaan::parse::Parser::parse(e.model);
    if (!fp.has_value()) { failures.push_back(e.id + ": parse"); continue; }
    magmaan::spec::LavaanifyOptions opts;
    opts.meanstructure = e.meanstructure;
    auto pt = magmaan::spec::lavaanify(*fp, opts);
    if (!pt.has_value()) { failures.push_back(e.id + ": lavaanify"); continue; }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) { failures.push_back(e.id + ": matrix_rep"); continue; }

    const auto& M = exp["sample_cov"][0]["matrix"];
    const Eigen::Index p = static_cast<Eigen::Index>(M.size());
    Eigen::MatrixXd S(p, p);
    for (Eigen::Index r = 0; r < p; ++r)
      for (Eigen::Index c = 0; c < p; ++c)
        S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                      .get<double>();
    magmaan::data::SampleStats samp;
    samp.S.push_back(std::move(S));
    samp.n_obs.push_back(exp["n_obs"].get<std::int64_t>());
    if (exp.contains("sample_mean") && !exp["sample_mean"].is_null()) {
      const auto& v = exp["sample_mean"][0]["vector"];
      Eigen::VectorXd mean(static_cast<Eigen::Index>(v.size()));
      for (Eigen::Index i = 0; i < mean.size(); ++i)
        mean(i) = v[static_cast<std::size_t>(i)].get<double>();
      samp.mean.push_back(std::move(mean));
    }

    auto est_or = magmaan::estimate::fit(*pt, *mr, samp);
    if (!est_or.has_value()) { failures.push_back(e.id + ": fit — " + est_or.error().detail); continue; }
    const auto& est = *est_or;

    auto info_or = magmaan::nt::infer::information_expected(*pt, *mr, samp, est);
    if (!info_or.has_value()) { failures.push_back(e.id + ": information_expected — " + info_or.error().detail); continue; }
    auto vcov_or = magmaan::nt::infer::vcov(*info_or, *pt);
    if (!vcov_or.has_value()) { failures.push_back(e.id + ": vcov — " + vcov_or.error().detail); continue; }
    const Eigen::VectorXd se_v = magmaan::nt::infer::se(*vcov_or);
    const Eigen::MatrixXd& vcov_m = *vcov_or;

    bool ok = true;
    char buf[200];

    // 1) z / two-sided p-value. Our chi2_pvalue(z², 1) == pchisq(z², 1, lower=FALSE)
    //    == lavaan's 2·pnorm(-|z|), so both should agree to round-off; assert
    //    z to 1e-3 (smooth in θ̂; θ̂ matches lavaan only to ~2e-6) and p to 1e-6.
    const auto z_arr = magmaan::nt::infer::z_test(est, se_v);
    const auto& pe_z = exp["pe_z"];
    const auto& pe_p = exp["pe_pvalue"];
    if (static_cast<std::size_t>(z_arr.z.size()) != pe_z.size()) {
      failures.push_back(e.id + ": z length mismatch"); continue;
    }
    for (Eigen::Index k = 0; k < z_arr.z.size(); ++k) {
      const double zl = pe_z[static_cast<std::size_t>(k)].get<double>();
      const double pl = pe_p[static_cast<std::size_t>(k)].get<double>();
      // lavaan reports a signed z; our z_test is also signed (θ̂_k/SE_k).
      if (std::abs(z_arr.z(k) - zl) > 1e-3) {
        std::snprintf(buf, sizeof(buf), "z[%lld] ours=%.6f lavaan=%.6f",
                      static_cast<long long>(k), z_arr.z(k), zl);
        failures.push_back(e.id + ": " + buf); ok = false; break;
      }
      if (std::abs(z_arr.p_value(k) - pl) > 1e-6) {
        std::snprintf(buf, sizeof(buf), "p[%lld] ours=%.3e lavaan=%.3e",
                      static_cast<long long>(k), z_arr.p_value(k), pl);
        failures.push_back(e.id + ": " + buf); ok = false; break;
      }
    }
    if (!ok) continue;

    // 2) Wald test: fix the first free loading to 0 and compare to lavTestWald.
    //    For a single linear restriction, W = (θ̂_k/SE_k)² — should match the
    //    z² above and lavaan's reported stat.
    if (exp.contains("wald_l1_eq0_chi2") && !exp["wald_l1_eq0_chi2"].is_null()) {
      const auto k = static_cast<Eigen::Index>(
          exp["wald_l1_eq0_free_idx"].get<int>() - 1);   // 1-based → 0-based
      if (k < 0 || k >= est.theta.size()) {
        failures.push_back(e.id + ": wald free_idx out of range"); continue;
      }
      Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, est.theta.size());
      R(0, k) = 1.0;
      Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
      auto w_or = magmaan::nt::infer::wald_test(R, q, est, vcov_m);
      if (!w_or.has_value()) { failures.push_back(e.id + ": wald_test — " + w_or.error().detail); continue; }
      const double chi2_l = exp["wald_l1_eq0_chi2"].get<double>();
      const int    df_l   = exp["wald_l1_eq0_df"].get<int>();
      const double p_l     = exp["wald_l1_eq0_pvalue"].get<double>();
      if (w_or->df != df_l) {
        std::snprintf(buf, sizeof(buf), "wald df=%d lavaan=%d", w_or->df, df_l);
        failures.push_back(e.id + ": " + buf); ok = false;
      }
      if (std::abs(w_or->chi2 - chi2_l) > 1e-3) {
        std::snprintf(buf, sizeof(buf), "wald χ² ours=%.6f lavaan=%.6f",
                      w_or->chi2, chi2_l);
        failures.push_back(e.id + ": " + buf); ok = false;
      }
      const double p_ours = magmaan::nt::infer::chi2_pvalue(w_or->chi2, w_or->df);
      if (std::abs(p_ours - p_l) > 1e-6) {
        std::snprintf(buf, sizeof(buf), "wald p ours=%.3e lavaan=%.3e", p_ours, p_l);
        failures.push_back(e.id + ": " + buf); ok = false;
      }
    }

    if (ok) ++passed;
  }

  MESSAGE("test-stat goldens: " << passed << " / " << total << " pass (+ "
          << skipped.size() << " skipped under-identified)");
  for (const auto& s : skipped)  MESSAGE("  SKIP " << s);
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
