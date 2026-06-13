#include <doctest/doctest.h>
#include "../test_fit.hpp"

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
#include "magmaan/inference/inference.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

// Mirror the skip list from fit_theta_golden_test.cpp — under-identified
// models lavaan itself can't produce SEs for ("information matrix could
// not be inverted"), so there's no oracle to compare against.
const std::set<std::string> kSkipForInferenceGoldens = {
    "0010_covariance",
    "0013_string_label",
    "0015_start_call",
    "0018_na_modifier",
};

}  // namespace

TEST_CASE("inference goldens — SE/χ²/df match lavaan") {
  const auto corpus = magmaan::test::load_corpus();
  const std::string fit_dir = magmaan::test::fixtures_dir() + "/fit";

  int total = 0, passed = 0;
  std::vector<std::string> failures;
  std::vector<std::string> skipped;
  std::vector<std::string> needs_regen;

  for (const auto& e : corpus) {
    const std::string path = fit_dir + "/" + e.id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) continue;
    if (e.n_groups > 1) continue;   // multi-group has its own golden
    if (kSkipForInferenceGoldens.count(e.id)) {
      skipped.push_back(e.id);
      continue;
    }

    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(e.id + ": fixture not valid JSON");
      continue;
    }
    if (!exp.contains("se") || !exp.contains("chi2") || !exp.contains("df")) {
      // Fixture predates the inference-oracle regen. Skip until the R
      // script has been re-run.
      needs_regen.push_back(e.id);
      continue;
    }
    ++total;

    auto fp = magmaan::parse::Parser::parse(e.model);
    if (!fp.has_value()) { failures.push_back(e.id + ": parse"); continue; }
    // Pass meanstructure through so explicit `~ 1` rows + the auto-add
    // semantics (ν free, α fixed at 0) match lavaan's cfa(meanstructure=TRUE).
    magmaan::spec::BuildOptions opts;
    opts.meanstructure = e.meanstructure;
    auto pt = magmaan::spec::build(*fp, opts);
    if (!pt.has_value()) { failures.push_back(e.id + ": lavaanify"); continue; }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) { failures.push_back(e.id + ": matrix_rep"); continue; }

    // Sample S from lavaan's lavInspect(fit, "sampstat")$cov — same
    // as fit_theta_golden_test.cpp. Plus sample_mean when the fixture has
    // it (single-group meanstructure fixtures populate this).
    const auto& sample_blocks = exp["sample_cov"];
    REQUIRE(sample_blocks.is_array());
    magmaan::data::SampleStats samp;
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
      if (exp.contains("sample_mean") && !exp["sample_mean"].is_null()) {
        const auto& v = exp["sample_mean"][b]["vector"];
        Eigen::VectorXd mean(static_cast<Eigen::Index>(v.size()));
        for (Eigen::Index i = 0; i < mean.size(); ++i)
          mean(i) = v[static_cast<std::size_t>(i)].get<double>();
        samp.mean.push_back(std::move(mean));
      }
    }

    auto est_or = magmaan::test::fit(*pt, *mr, samp);
    if (!est_or.has_value()) {
      failures.push_back(e.id + ": fit failed — kind=" +
                         std::to_string(static_cast<int>(est_or.error().kind)) +
                         " " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    auto info_or = magmaan::inference::information_expected(*pt, *mr, samp, est);
    if (!info_or.has_value()) {
      failures.push_back(e.id + ": information_expected failed — kind=" +
                         std::to_string(static_cast<int>(info_or.error().kind)) +
                         " " + info_or.error().detail);
      continue;
    }
    auto vcov_or = magmaan::inference::vcov(*info_or, *pt);
    if (!vcov_or.has_value()) {
      failures.push_back(e.id + ": vcov failed — " + vcov_or.error().detail);
      continue;
    }
    const Eigen::VectorXd se_v = magmaan::inference::se(*vcov_or);
    const double          chi2 = magmaan::inference::chi2_stat(samp, est);
    auto df_or = magmaan::inference::df_stat(*pt, samp);
    if (!df_or.has_value()) {
      failures.push_back(e.id + ": df_stat failed — " + df_or.error().detail);
      continue;
    }
    const int df = *df_or;

    // 1) df — exact integer match.
    const int df_lavaan = exp["df"].get<int>();
    if (df != df_lavaan) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "df = %d, lavaan = %d", df, df_lavaan);
      failures.push_back(e.id + ": " + buf);
      continue;
    }

    // 2) chi² — ≤ 1e-3 absolute. For saturated models lavaan reports
    // ~1e-13; (n-1)·fmin is the same order of magnitude. 1e-3 covers it.
    const double chi2_lavaan = exp["chi2"].get<double>();
    const double chi2_diff   = std::abs(chi2 - chi2_lavaan);
    if (chi2_diff > 1e-3) {
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "|chi2 - lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                    chi2_diff, chi2, chi2_lavaan);
      failures.push_back(e.id + ": " + buf);
      continue;
    }

    // 3) SE — max absolute diff ≤ 1e-4 over all free params.
    const auto& se_arr = exp["se"];
    if (static_cast<std::size_t>(se_v.size()) != se_arr.size()) {
      failures.push_back(e.id + ": se length mismatch");
      continue;
    }
    double max_se_diff = 0.0;
    for (Eigen::Index k = 0; k < se_v.size(); ++k) {
      const double d = std::abs(se_v(k) -
          se_arr[static_cast<std::size_t>(k)].get<double>());
      if (d > max_se_diff) max_se_diff = d;
    }
    if (max_se_diff > 1e-4) {
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "max |se - lavaan| = %.3e (df=%d, chi2_diff=%.3e)",
                    max_se_diff, df, chi2_diff);
      failures.push_back(e.id + ": " + buf);
      continue;
    }

    // 3b) Browne residual-based normal-theory test (`test = "browne.residual.nt"`)
    //     — full quadratic form with the model space projected out. ≈ 0 for
    //     saturated models, distinct from χ²_ML in finite samples otherwise.
    if (exp.contains("browne_residual_nt") && !exp["browne_residual_nt"].is_null()) {
      auto br_or = magmaan::inference::browne_residual_nt(*pt, *mr, samp, est);
      if (!br_or.has_value()) {
        failures.push_back(e.id + ": browne_residual_nt — " + br_or.error().detail);
        continue;
      }
      const double br_lavaan = exp["browne_residual_nt"].get<double>();
      if (std::abs(*br_or - br_lavaan) > 1e-3) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "browne.residual.nt ours=%.6f, lavaan=%.6f (diff=%.3e)",
                      *br_or, br_lavaan, std::abs(*br_or - br_lavaan));
        failures.push_back(e.id + ": " + buf);
        continue;
      }
    }

    // 3c) RLS / model-based Browne residual (`test = "browne.residual.nt.model"`).
    //     Build the implied Σ̂ from our θ̂ via the evaluator (mirroring fit()'s
    //     fixed.x resolution) and feed it to rls_chi2.
    if (exp.contains("rls_chi2") && !exp["rls_chi2"].is_null()) {
      magmaan::spec::LatentStructure pt_res = *pt;
      (void)magmaan::estimate::resolve_fixed_x_from_sample(pt_res, *mr, samp);
      auto ev_or = magmaan::model::ModelEvaluator::build(pt_res, *mr);
      if (!ev_or.has_value()) {
        failures.push_back(e.id + ": rls_chi2 build_evaluator — " +
                           ev_or.error().detail);
        continue;
      }
      auto im_or = ev_or->sigma(est.theta);
      if (!im_or.has_value()) {
        failures.push_back(e.id + ": rls_chi2 sigma — " + im_or.error().detail);
        continue;
      }
      auto rls_or = magmaan::inference::rls_chi2(samp, *im_or);
      if (!rls_or.has_value()) {
        failures.push_back(e.id + ": rls_chi2 — " + rls_or.error().detail);
        continue;
      }
      const double rls_lavaan = exp["rls_chi2"].get<double>();
      if (std::abs(*rls_or - rls_lavaan) > 1e-3) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "browne.residual.nt.model ours=%.6f, lavaan=%.6f (diff=%.3e)",
                      *rls_or, rls_lavaan, std::abs(*rls_or - rls_lavaan));
        failures.push_back(e.id + ": " + buf);
        continue;
      }
    }

    // 4) Robust normal-theory tests (SB, mean.var.adjusted, scaled.shifted)
    //    via the reduced-symmetric eigenvalue path. Only checked when the
    //    fixture carries oracle values AND the model is non-saturated
    //    (df > 0 in the χ² sense).
    if (df > 0 && exp.contains("sb_chi2") && !exp["sb_chi2"].is_null()) {
      auto uf_or = magmaan::robust::build_u_factor(*pt, *mr, samp, est);
      if (!uf_or.has_value()) {
        failures.push_back(e.id + ": build_u_factor — " +
                           uf_or.error().detail);
        continue;
      }
      auto M_nt_or = magmaan::robust::reduced_gamma_nt(*uf_or);
      if (!M_nt_or.has_value()) {
        failures.push_back(e.id + ": reduced_gamma_nt — " +
                           M_nt_or.error().detail);
        continue;
      }
      auto ev_or = magmaan::robust::ugamma_eigenvalues(*M_nt_or);
      if (!ev_or.has_value()) {
        failures.push_back(e.id + ": ugamma_eigenvalues — " +
                           ev_or.error().detail);
        continue;
      }
      // Sanity smoke: U·Γ_NT has all eigenvalues = 1 (rank-df projector).
      const double max_dev_from_one =
          (ev_or->array() - 1.0).abs().maxCoeff();
      if (max_dev_from_one > 1e-9) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "U·Γ_NT eigenvalue deviation = %.3e (expected 0)",
                      max_dev_from_one);
        failures.push_back(e.id + ": " + buf);
        continue;
      }
      // The SB / mean.var / scaled.shifted *wrapper formulas* are
      // verified independently against lavaan: feed lavaan's stored
      // UΓ̂ eigenvalues (which lavaan computed using empirical Γ̂ from
      // raw data — not Γ_NT(Σ̂)) into our wrapper functions and confirm
      // we recover lavaan's reported stats to 1e-3.
      //
      // Why this split: our reduced_gamma_nt produces all-1 eigenvalues
      // (correct under Γ_NT), while lavaan uses Γ̂_empirical so its
      // eigenvalues sum to roughly df + sampling drift. Both are
      // correct in their own setting; the wrapper formulas should
      // produce identical chi² statistics given the same eigenvalues.
      if (exp.contains("ugamma_eigvals_nt") &&
          !exp["ugamma_eigvals_nt"].is_null()) {
        const auto& ev_arr = exp["ugamma_eigvals_nt"];
        Eigen::VectorXd ev_lavaan(static_cast<Eigen::Index>(ev_arr.size()));
        for (Eigen::Index k = 0; k < ev_lavaan.size(); ++k)
          ev_lavaan(k) = ev_arr[static_cast<std::size_t>(k)].get<double>();
        auto sb  = magmaan::robust::satorra_bentler(chi2, df, ev_lavaan);
        auto mv  = magmaan::robust::mean_var_adjusted(chi2, df, ev_lavaan);
        auto ss  = magmaan::robust::scaled_shifted(chi2, df, ev_lavaan);
        const double sb_lavaan = exp["sb_chi2"].get<double>();
        const double mv_lavaan = exp["mean_var_chi2"].get<double>();
        const double ss_lavaan = exp["scaled_shifted_chi2"].get<double>();
        if (std::abs(sb.chi2_scaled - sb_lavaan) > 1e-3) {
          char buf[160];
          std::snprintf(buf, sizeof(buf),
                        "SB ours=%.6f, lavaan=%.6f (diff=%.3e)",
                        sb.chi2_scaled, sb_lavaan,
                        std::abs(sb.chi2_scaled - sb_lavaan));
          failures.push_back(e.id + ": " + buf);
          continue;
        }
        if (std::abs(mv.chi2_adj - mv_lavaan) > 1e-3) {
          char buf[160];
          std::snprintf(buf, sizeof(buf),
                        "mean.var ours=%.6f, lavaan=%.6f (diff=%.3e)",
                        mv.chi2_adj, mv_lavaan,
                        std::abs(mv.chi2_adj - mv_lavaan));
          failures.push_back(e.id + ": " + buf);
          continue;
        }
        if (std::abs(ss.chi2_adj - ss_lavaan) > 1e-3) {
          char buf[160];
          std::snprintf(buf, sizeof(buf),
                        "scaled.shifted ours=%.6f, lavaan=%.6f (diff=%.3e)",
                        ss.chi2_adj, ss_lavaan,
                        std::abs(ss.chi2_adj - ss_lavaan));
          failures.push_back(e.id + ": " + buf);
          continue;
        }
        // Scaling-factor / shift / Satterthwaite-df *outputs* (not just the
        // resulting χ²): pure functions of Σλ, Σλ², df — tighter tolerance.
        // Note: lavaan's `scaled.shifted` `scaling.factor` is the *divisor*
        // applied as `T/c + b`, i.e. the reciprocal of our `scale_a` (which is
        // the multiplier in `T·a + b`); the shift parameter `b` is the same.
        struct ScaleCheck { const char* name; double ours; double lavaan; };
        const ScaleCheck scale_checks[] = {
            {"sb_scale",        sb.scale_c,     exp["sb_scale"].get<double>()},
            {"mean_var_df_adj", mv.df_adj,      exp["mean_var_df_adj"].get<double>()},
            {"scaled_shifted_a", 1.0/ss.scale_a, exp["scaled_shifted_a"].get<double>()},
            {"scaled_shifted_b", ss.shift_b,    exp["scaled_shifted_b"].get<double>()},
        };
        bool scale_ok = true;
        for (const auto& sc : scale_checks) {
          if (std::abs(sc.ours - sc.lavaan) > 1e-4) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "%s ours=%.8f, lavaan=%.8f (diff=%.3e)",
                          sc.name, sc.ours, sc.lavaan,
                          std::abs(sc.ours - sc.lavaan));
            failures.push_back(e.id + ": " + buf);
            scale_ok = false;
            break;
          }
        }
        if (!scale_ok) continue;
      }

      // Robust ("sandwich") SEs vs lavaan's `estimator = "MLM"`
      // (= se = "robust.sem"). Uses lavaan's stored empirical Γ̂
      // (`gamma_hat`) as the meat so the moments match exactly. Single-
      // group only (the robust-SE v1 surface is single-block).
      if (exp.contains("gamma_hat") && !exp["gamma_hat"].is_null() &&
          exp.contains("se_robust_sem") && !exp["se_robust_sem"].is_null()) {
        const auto& Gj = exp["gamma_hat"];
        const Eigen::Index pstar = static_cast<Eigen::Index>(Gj.size());
        Eigen::MatrixXd Gamma_hat(pstar, pstar);
        for (Eigen::Index r = 0; r < pstar; ++r)
          for (Eigen::Index c = 0; c < pstar; ++c)
            Gamma_hat(r, c) = Gj[static_cast<std::size_t>(r)]
                               [static_cast<std::size_t>(c)].get<double>();
        auto rob_or = magmaan::robust::robust_se(
            *pt, *mr, samp, est, Gamma_hat,
            {magmaan::robust::Information::Expected,
             magmaan::robust::WeightMoments::Structured,
             magmaan::robust::ScoreCovariance::Empirical});
        if (!rob_or.has_value()) {
          failures.push_back(e.id + ": robust_se — " + rob_or.error().detail);
          continue;
        }
        const auto& se_arr_r = exp["se_robust_sem"];
        if (static_cast<std::size_t>(rob_or->se.size()) != se_arr_r.size()) {
          failures.push_back(e.id + ": robust se length mismatch");
          continue;
        }
        double max_rob_diff = 0.0;
        for (Eigen::Index k = 0; k < rob_or->se.size(); ++k) {
          const double d = std::abs(rob_or->se(k) -
              se_arr_r[static_cast<std::size_t>(k)].get<double>());
          if (d > max_rob_diff) max_rob_diff = d;
        }
        if (max_rob_diff > 1e-4) {
          char buf[160];
          std::snprintf(buf, sizeof(buf),
                        "max |robust.sem se - lavaan| = %.3e", max_rob_diff);
          failures.push_back(e.id + ": " + buf);
          continue;
        }

        // Huber-White (`estimator = "MLR"`, `se = "robust.huber.white"`):
        // same meat, Observed (Hessian) bread.
        if (exp.contains("se_robust_huberwhite") &&
            !exp["se_robust_huberwhite"].is_null()) {
          auto hw_or = magmaan::robust::robust_se(
              *pt, *mr, samp, est, Gamma_hat,
              {magmaan::robust::Information::Observed,
               magmaan::robust::WeightMoments::Structured,
               magmaan::robust::ScoreCovariance::Empirical});
          if (!hw_or.has_value()) {
            failures.push_back(e.id + ": robust_se (observed) — " +
                               hw_or.error().detail);
            continue;
          }
          const auto& se_hw = exp["se_robust_huberwhite"];
          double max_hw_diff = 0.0;
          for (Eigen::Index k = 0; k < hw_or->se.size(); ++k) {
            const double d = std::abs(hw_or->se(k) -
                se_hw[static_cast<std::size_t>(k)].get<double>());
            if (d > max_hw_diff) max_hw_diff = d;
          }
          if (max_hw_diff > 1e-4) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "max |robust.huber.white se - lavaan| = %.3e",
                          max_hw_diff);
            failures.push_back(e.id + ": " + buf);
            continue;
          }
        }
      }
    }

    ++passed;
  }

  MESSAGE("inference goldens: " << passed << " / " << total << " pass"
          << " (+ " << skipped.size() << " skipped under-identified, "
          << needs_regen.size() << " awaiting fixture regen)");
  for (const auto& s : skipped)     MESSAGE("  SKIP    " << s);
  for (const auto& s : needs_regen) MESSAGE("  NO-SE   " << s);
  for (const auto& f : failures)    MESSAGE("  FAIL    " << f);

  CHECK(skipped.size() == kSkipForInferenceGoldens.size());
  CHECK(needs_regen.empty());
  CHECK(passed == total);
}
