#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/nt/fiml.hpp"
#include "magmaan/nt/measures.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

const std::vector<std::string> kFimlFixtures = {
    "0001_one_factor_hs_fiml",
    "0002_three_factor_hs_fiml",
    "0003_equal_loading_hs_fiml",
    "0004_multigroup_1f_school_fiml",
    "0005_multigroup_3f_school_fiml",
    "0006_multigroup_equal_loading_school_fiml",
    "0007_structural_hs_fiml",
    "0008_structural_fixedx_false_hs_fiml",
    "0009_path_fixed_x_missing_hs_fiml",
    "0010_three_factor_dense_patterns_hs_fiml",
    "0011_path_fixedx_false_missing_x_y_fiml",
    "0012_path_fixedx_true_complete_x_missing_y_fiml",
    "0013_structural_equal_regression_hs_fiml",
    "0014_multigroup_dense_school_specific_fiml",
    "0015_structural_highdim_hs_fiml",
    "0016_three_factor_complete_rows_fiml",
    "0017_multigroup_path_fixedx_true_complete_x_missing_y_fiml",
};

magmaan::data::RawData raw_from_fixture(const nlohmann::json& exp) {
  magmaan::data::RawData raw;
  const auto& blocks = exp["raw"];
  raw.X.reserve(blocks.size());
  raw.mask.reserve(blocks.size());
  for (const auto& block : blocks) {
    const auto& Xj = block["X"];
    const auto& Mj = block["mask"];
    const Eigen::Index n = static_cast<Eigen::Index>(Xj.size());
    const Eigen::Index p = n > 0 ? static_cast<Eigen::Index>(Xj[0].size()) : 0;

    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < p; ++c) {
        const auto& x = Xj[static_cast<std::size_t>(r)]
                         [static_cast<std::size_t>(c)];
        X(r, c) = x.is_null()
            ? std::numeric_limits<double>::quiet_NaN()
            : x.get<double>();
        M(r, c) = static_cast<std::uint8_t>(
            Mj[static_cast<std::size_t>(r)]
              [static_cast<std::size_t>(c)].get<int>());
      }
    }
    raw.X.push_back(std::move(X));
    raw.mask.push_back(std::move(M));
  }
  return raw;
}

bool finite_json(const nlohmann::json& j) {
  return !j.is_null() && j.is_number() && std::isfinite(j.get<double>());
}

bool finite_scalar_json(const nlohmann::json& j) {
  if (finite_json(j)) return true;
  if (!j.is_array() || j.empty()) return false;
  for (const auto& x : j) {
    if (!finite_json(x)) return false;
  }
  return true;
}

double scalar_from_json(const nlohmann::json& j) {
  if (j.is_array()) {
    double out = 0.0;
    for (const auto& x : j) out += x.get<double>();
    return out;
  }
  return j.get<double>();
}

Eigen::VectorXd vector_from_json(const nlohmann::json& j) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].is_null()
        ? std::numeric_limits<double>::quiet_NaN()
        : j[static_cast<std::size_t>(i)].get<double>();
  }
  return out;
}

}  // namespace

TEST_CASE("FIML goldens — θ̂ matches lavaan missing='fiml'") {
  const std::string dir = magmaan::test::fixtures_dir() + "/fiml";

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kFimlFixtures) {
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw_json = magmaan::test::read_fixture(path);
    if (!raw_json.has_value()) {
      failures.push_back(id + ": missing fixture");
      continue;
    }
    auto exp = nlohmann::json::parse(*raw_json, nullptr,
                                     /*allow_exceptions=*/false);
    if (exp.is_discarded()) {
      failures.push_back(id + ": invalid JSON");
      continue;
    }
    ++total;

    const std::string model = exp["input"].get<std::string>();
    auto fp = magmaan::parse::Parser::parse(model);
    if (!fp.has_value()) {
      failures.push_back(id + ": parse");
      continue;
    }

    magmaan::spec::LavaanifyOptions opts;
    opts.n_groups = exp.value("n_groups", 1);
    opts.meanstructure = exp.value("meanstructure", true);
    opts.fixed_x = exp.value("fixed_x", true);
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

    const magmaan::data::RawData raw = raw_from_fixture(exp);
    if (raw.X.size() != static_cast<std::size_t>(opts.n_groups)) {
      failures.push_back(id + ": raw block count does not match n_groups");
      continue;
    }
    auto est_or = (id == "0015_structural_highdim_hs_fiml")
        ? magmaan::estimate::fit_fiml(
              *pt, *mr, raw, magmaan::nt::fiml::FIML{},
              magmaan::optim::LbfgsOptimizer(
                  magmaan::optim::LbfgsOptions{.max_iter = 4000}))
        : magmaan::estimate::fit_fiml(*pt, *mr, raw);
    if (exp.contains("expect_error")) {
      if (est_or.has_value()) {
        failures.push_back(id + ": expected fit_fiml error");
        continue;
      }
      const std::string want = exp["expect_error"].get<std::string>();
      if (est_or.error().detail.find(want) == std::string::npos) {
        failures.push_back(id + ": unexpected fit_fiml error — " +
                           est_or.error().detail);
        continue;
      }
      ++passed;
      continue;
    }
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit_fiml — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    auto fx_or = magmaan::estimate::fiml_extras(*pt, *mr, raw, est);
    if (!fx_or.has_value()) {
      failures.push_back(id + ": fiml_extras — " + fx_or.error().detail);
      continue;
    }
    const auto& fx = *fx_or;

    auto start_samp_or = magmaan::nt::fiml::fiml_start_sample_stats(raw);
    if (!start_samp_or.has_value()) {
      failures.push_back(id + ": fiml_start_sample_stats — " +
                         start_samp_or.error().detail);
      continue;
    }

    auto bl_or = magmaan::nt::fiml::fiml_baseline_chi2(*pt, raw);
    if (!bl_or.has_value()) {
      failures.push_back(id + ": fiml_baseline_chi2 — " +
                         bl_or.error().detail);
      continue;
    }
    const auto& bl = *bl_or;

    if (!exp.contains("df")) {
      failures.push_back(id + ": missing df");
      continue;
    }
    const int df = exp["df"].get<int>();
    const auto fm = magmaan::nt::measures::fit_measures(fx.chi2, df, bl,
                                                        *start_samp_or);

    const auto& th = exp["theta_hat"];
    if (static_cast<std::size_t>(est.theta.size()) != th.size()) {
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "n_free mismatch: got %td, lavaan fixture has %zu",
                    est.theta.size(), th.size());
      failures.push_back(id + ": " + buf);
      continue;
    }

    double max_diff = 0.0;
    Eigen::Index max_k = 0;
    for (Eigen::Index k = 0; k < est.theta.size(); ++k) {
      const double d = std::abs(est.theta(k) -
          th[static_cast<std::size_t>(k)].get<double>());
      if (d > max_diff) {
        max_diff = d;
        max_k = k;
      }
    }

    if (max_diff > 5e-5) {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "max |θ̂ - lavaan| = %.3e at %td (got %.10f, want %.10f; "
                    "iters=%d, fmin=%.10f)",
                    max_diff, max_k, est.theta(max_k),
                    th[static_cast<std::size_t>(max_k)].get<double>(),
                    est.iterations, est.fmin);
      failures.push_back(id + ": " + buf);
      continue;
    }

    auto cmp = [&](const char* name, double got, double tol) {
      if (!exp.contains(name)) {
        failures.push_back(id + ": missing " + std::string(name));
        return false;
      }
      const double want = exp[name].get<double>();
      const double d = std::abs(got - want);
      if (d <= tol) return true;
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "%s mismatch: got %.12g, want %.12g, |diff| %.3e",
                    name, got, want, d);
      failures.push_back(id + ": " + buf);
      return false;
    };
    bool ok = true;
    ok = cmp("logl", fx.logl, 2e-5) && ok;
    ok = cmp("unrestricted_logl", fx.unrestricted_logl, 2e-5) && ok;
    ok = cmp("chisq", fx.chi2, 2e-5) && ok;
    ok = cmp("aic", fx.aic, 5e-5) && ok;
    ok = cmp("bic", fx.bic, 5e-5) && ok;
    ok = cmp("bic2", fx.bic2, 5e-5) && ok;

    ok = cmp("baseline_chisq", bl.chi2, 2e-5) && ok;
    if (!exp.contains("baseline_df") ||
        bl.df != exp["baseline_df"].get<int>()) {
      failures.push_back(id + ": baseline_df mismatch");
      ok = false;
    }

    auto cmp_finite = [&](const char* label, double got,
                          const char* key, double tol) {
      if (!exp.contains(key)) {
        failures.push_back(id + ": missing " + std::string(key));
        return false;
      }
      const auto& want_j = exp[key];
      if (!finite_json(want_j)) return true;
      if (!std::isfinite(got)) return true;
      const double want = want_j.get<double>();
      const double d = std::abs(got - want);
      if (d <= tol * std::max(1.0, std::abs(want))) return true;
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "%s mismatch: got %.12g, want %.12g, |diff| %.3e",
                    label, got, want, d);
      failures.push_back(id + ": " + buf);
      return false;
    };
    ok = cmp_finite("cfi", fm.cfi, "cfi", 1e-5) && ok;
    ok = cmp_finite("tli", fm.tli, "tli", 1e-5) && ok;
    ok = cmp_finite("rmsea", fm.rmsea, "rmsea", 1e-5) && ok;
    ok = cmp_finite("rmsea_ci_lower", fm.rmsea_ci_lower,
                    "rmsea_ci_lower", 1e-4) && ok;
    ok = cmp_finite("rmsea_ci_upper", fm.rmsea_ci_upper,
                    "rmsea_ci_upper", 1e-4) && ok;
    if (exp.contains("rmsea_pvalue")) {
      ok = cmp_finite("rmsea_pvalue", fm.rmsea_pvalue,
                      "rmsea_pvalue", 1e-5) && ok;
      ok = cmp_finite("rmsea_close_h0", fm.rmsea_close_h0,
                      "rmsea_close_h0", 1e-12) && ok;
      ok = cmp_finite("rmsea_notclose_pvalue", fm.rmsea_notclose_pvalue,
                      "rmsea_notclose_pvalue", 1e-5) && ok;
      ok = cmp_finite("rmsea_notclose_h0", fm.rmsea_notclose_h0,
                      "rmsea_notclose_h0", 1e-12) && ok;
    }

    if (!exp.contains("npar") || fx.npar != exp["npar"].get<int>()) {
      failures.push_back(id + ": npar mismatch");
      ok = false;
    }
    if (!exp.contains("n_obs") || fx.ntotal != exp["n_obs"].get<std::int64_t>()) {
      failures.push_back(id + ": ntotal mismatch");
      ok = false;
    }
    if (exp.contains("n_obs_per_block")) {
      const auto& nobs = exp["n_obs_per_block"];
      for (std::size_t b = 0; b < raw.X.size(); ++b) {
        const std::int64_t want = nobs.is_array()
            ? nobs[b].get<std::int64_t>()
            : nobs.get<std::int64_t>();
        if (raw.X[b].rows() != want) {
          failures.push_back(id + ": n_obs_per_block mismatch");
          ok = false;
          break;
        }
      }
    }
    if (exp.contains("se_robust_huberwhite") &&
        !exp["se_robust_huberwhite"].is_null() &&
        exp.contains("mlr_chisq_scaled") &&
        finite_json(exp["mlr_chisq_scaled"]) &&
        df > 0) {
      auto rob_or = magmaan::nt::fiml::fiml_robust_mlr(
          *pt, *mr, raw, est, df, fx.chi2);
      if (!rob_or.has_value()) {
        failures.push_back(id + ": fiml_robust_mlr — " +
                           rob_or.error().detail);
        ok = false;
      } else {
        const Eigen::VectorXd lavaan_se =
            vector_from_json(exp["se_robust_huberwhite"]);
        if (lavaan_se.size() != rob_or->se.size()) {
          failures.push_back(id + ": robust SE length mismatch");
          ok = false;
        } else {
          double d_se = 0.0;
          for (Eigen::Index k = 0; k < lavaan_se.size(); ++k) {
            if (!std::isfinite(lavaan_se(k))) continue;
            d_se = std::max(d_se, std::abs(lavaan_se(k) - rob_or->se(k)));
          }
          if (d_se > 3e-4) {
            failures.push_back(id + ": robust SE max diff " +
                               std::to_string(d_se));
            ok = false;
          }
        }
        auto robust_cmp = [&](const char* label, double got,
                              const char* key, double tol) {
          if (!exp.contains(key) || !finite_scalar_json(exp[key])) return true;
          const double want = scalar_from_json(exp[key]);
          const double d = std::abs(got - want);
          if (d <= tol * std::max(1.0, std::abs(want))) return true;
          failures.push_back(id + ": " + std::string(label) +
                             " diff " + std::to_string(d));
          return false;
        };
        ok = robust_cmp("mlr_chisq_scaled", rob_or->chisq_scaled,
                        "mlr_chisq_scaled", 5e-3) && ok;
        ok = robust_cmp("mlr_scaling_factor", rob_or->scaling_factor,
                        "mlr_scaling_factor", 5e-3) && ok;
        ok = robust_cmp("mlr_trace_ugamma", rob_or->trace_ugamma,
                        "mlr_trace_ugamma", 5e-3) && ok;
        ok = robust_cmp("mlr_trace_ugamma_h1", rob_or->trace_ugamma_h1,
                        "mlr_trace_ugamma_h1", 5e-3) && ok;
        ok = robust_cmp("mlr_trace_ugamma_h0", rob_or->trace_ugamma_h0,
                        "mlr_trace_ugamma_h0", 5e-3) && ok;
      }
    }
    if (ok) ++passed;
  }

  MESSAGE("FIML goldens: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
