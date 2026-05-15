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
#include "magmaan/nt/measures.hpp"
#include "magmaan/nt/infer.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

// Same skip list as inference / fit-theta goldens: under-identified models
// lavaan won't produce a clean fit for (and our optimizer lands somewhere
// arbitrary), so there's no fit-measure oracle worth comparing.
const std::set<std::string> kSkipForFitMeasureGoldens = {
    "0010_covariance",
    "0013_string_label",
    "0015_start_call",
    "0018_na_modifier",
};

bool finite_json(const nlohmann::json& j) {
  return !j.is_null() && j.is_number() && std::isfinite(j.get<double>());
}

}  // namespace

TEST_CASE("fit-measure goldens — CFI/TLI/RMSEA/SRMR/logl/AIC/BIC match lavaan") {
  const auto corpus = magmaan::test::load_corpus();
  const std::string fit_dir = magmaan::test::fixtures_dir() + "/fit";

  int total = 0, passed = 0;
  std::vector<std::string> failures;
  std::vector<std::string> skipped;
  std::vector<std::string> needs_regen;
  std::vector<std::string> processed;

  for (const auto& e : corpus) {
    const std::string path = fit_dir + "/" + e.id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) continue;
    if (kSkipForFitMeasureGoldens.count(e.id)) {
      skipped.push_back(e.id);
      continue;
    }
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(e.id + ": fixture not valid JSON");
      continue;
    }
    // Fixtures predating this oracle regen lack logl/aic/bic/srmr.
    if (!exp.contains("logl") || !exp.contains("aic") ||
        !exp.contains("srmr") || !exp.contains("npar")) {
      needs_regen.push_back(e.id);
      continue;
    }
    ++total;
    processed.push_back(e.id);

    auto fp = magmaan::parse::Parser::parse(e.model);
    if (!fp.has_value()) { failures.push_back(e.id + ": parse"); continue; }
    magmaan::spec::LavaanifyOptions opts;
    opts.n_groups      = e.n_groups;
    opts.meanstructure = e.meanstructure;
    auto pt = magmaan::spec::lavaanify(*fp, opts);
    if (!pt.has_value()) {
      failures.push_back(e.id + ": lavaanify — " + pt.error().detail);
      continue;
    }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) {
      failures.push_back(e.id + ": matrix_rep — " + mr.error().detail);
      continue;
    }

    // Assemble (multi-block) SampleStats from the fixture.
    const auto& sample_blocks = exp["sample_cov"];
    REQUIRE(sample_blocks.is_array());
    const std::size_t n_blocks = sample_blocks.size();
    magmaan::data::SampleStats samp;
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const auto& M = sample_blocks[b]["matrix"];
      const Eigen::Index p = static_cast<Eigen::Index>(M.size());
      Eigen::MatrixXd S(p, p);
      for (Eigen::Index r = 0; r < p; ++r)
        for (Eigen::Index c = 0; c < p; ++c)
          S(r, c) = M[static_cast<std::size_t>(r)]
                     [static_cast<std::size_t>(c)].get<double>();
      samp.S.push_back(std::move(S));
      if (n_blocks == 1) {
        samp.n_obs.push_back(exp["n_obs"].get<std::int64_t>());
      } else {
        samp.n_obs.push_back(
            exp["n_obs_per_block"][b].get<std::int64_t>());
      }
      if (exp.contains("sample_mean") && !exp["sample_mean"].is_null()) {
        const auto& v = exp["sample_mean"][b]["vector"];
        Eigen::VectorXd m(static_cast<Eigen::Index>(v.size()));
        for (Eigen::Index i = 0; i < m.size(); ++i)
          m(i) = v[static_cast<std::size_t>(i)].get<double>();
        samp.mean.push_back(std::move(m));
      }
    }

    auto est_or = magmaan::estimate::fit(*pt, *mr, samp);
    if (!est_or.has_value()) {
      failures.push_back(e.id + ": fit — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    const double chi2 = magmaan::nt::infer::chi2_stat(samp, est);
    auto df_or = magmaan::nt::infer::df_stat(*pt, samp);
    if (!df_or.has_value()) {
      failures.push_back(e.id + ": df_stat — " + df_or.error().detail);
      continue;
    }
    const int df = *df_or;
    const auto bl = magmaan::nt::measures::baseline_chi2(samp);
    const auto fm = magmaan::nt::measures::fit_measures(chi2, df, bl, samp);

    auto fx_or = magmaan::nt::measures::fit_extras(*pt, *mr, samp, est);
    if (!fx_or.has_value()) {
      failures.push_back(e.id + ": fit_extras — " + fx_or.error().detail);
      continue;
    }
    const auto& fx = *fx_or;

    bool ok = true;
    auto fail = [&](const std::string& msg) {
      failures.push_back(e.id + ": " + msg);
      ok = false;
    };
    // abs-or-rel tolerance: passes when |a-b| ≤ tol·max(1,|b|).
    auto close = [](double a, double b, double tol) {
      return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
    };

    // npar — exact integer.
    {
      const int npar_l = exp["npar"].get<int>();
      if (fx.npar != npar_l) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "npar = %d, lavaan = %d",
                      fx.npar, npar_l);
        fail(buf);
      }
    }
    // Practical fit indices — only when the fixture value is finite (lavaan
    // emits NaN for tli/cfi on saturated/under-identified fits; our
    // fit_measures returns NaN for tli when df_u == 0). Compare what's
    // comparable; non-finite-on-either-side is skipped, not a failure.
    auto cmp_field = [&](const char* name, double ours,
                         const char* key, double tol) {
      if (!exp.contains(key)) return;
      const auto& jv = exp[key];
      if (!finite_json(jv)) return;
      const double l = jv.get<double>();
      if (!std::isfinite(ours)) return;  // e.g. tli=NaN on saturated models
      if (!close(ours, l, tol)) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "%s = %.10g, lavaan = %.10g (|Δ| = %.3e)",
                      name, ours, l, std::abs(ours - l));
        fail(buf);
      }
    };
    cmp_field("cfi",   fm.cfi,                "cfi",   1e-5);
    cmp_field("tli",   fm.tli,                "tli",   1e-5);
    cmp_field("rmsea", fm.rmsea,              "rmsea", 1e-5);
    cmp_field("rmsea.ci.lower", fm.rmsea_ci_lower, "rmsea_ci_lower", 1e-4);
    cmp_field("rmsea.ci.upper", fm.rmsea_ci_upper, "rmsea_ci_upper", 1e-4);
    cmp_field("rmsea.pvalue", fm.rmsea_pvalue, "rmsea_pvalue", 1e-5);
    cmp_field("rmsea.close.h0", fm.rmsea_close_h0, "rmsea_close_h0", 1e-12);
    cmp_field("rmsea.notclose.pvalue", fm.rmsea_notclose_pvalue,
              "rmsea_notclose_pvalue", 1e-5);
    cmp_field("rmsea.notclose.h0", fm.rmsea_notclose_h0,
              "rmsea_notclose_h0", 1e-12);
    cmp_field("srmr",  fx.srmr,               "srmr",  1e-4);
    cmp_field("logl",  fx.logl,               "logl",  1e-6);
    cmp_field("unrestricted_logl", fx.unrestricted_logl,
              "unrestricted_logl", 1e-9);
    cmp_field("aic",   fx.aic,                "aic",   1e-6);
    cmp_field("bic",   fx.bic,                "bic",   1e-6);
    cmp_field("bic2",  fx.bic2,               "bic2",  1e-6);

    // logl ≡ unrestricted_logl − χ²/2 (cov-only or free-mean models). Ties
    // the new log-likelihood back to the already-golden χ².
    {
      const double recon = fx.unrestricted_logl - chi2 / 2.0;
      if (!close(fx.logl, recon, 1e-6)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "logl=%.6f vs ulogl-chi2/2=%.6f (|Δ|=%.3e)",
                      fx.logl, recon, std::abs(fx.logl - recon));
        fail(buf);
      }
    }

    if (ok) ++passed;
  }

  if (!needs_regen.empty()) {
    std::string m = "fixtures missing logl/aic/srmr/npar (run tools/regen_oracle.R):";
    for (const auto& id : needs_regen) m += " " + id;
    MESSAGE(m);
  }
  if (!skipped.empty()) {
    std::string m = "skipped (under-identified — no fit-measure oracle):";
    for (const auto& id : skipped) m += " " + id;
    MESSAGE(m);
  }
  {
    std::string m = "fit-measure goldens checked: ";
    for (const auto& id : processed) m += id + " ";
    MESSAGE(m);
  }
  if (!failures.empty()) {
    std::string m = "fit-measure golden failures:\n";
    for (const auto& f : failures) m += "  - " + f + "\n";
    FAIL(m);
  }
  CHECK(passed == total);
  CHECK(total >= 8);   // at least the core HS / path / structural / 2-group set
}
