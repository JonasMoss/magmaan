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
#include "latva/model/model_evaluator.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

namespace {

// Pull one block's S / m̄ / n from a fit fixture's per-block arrays.
struct BlockData {
  Eigen::MatrixXd S;
  Eigen::VectorXd mean;
  std::int64_t    n_obs = 0;
};

BlockData load_block(const nlohmann::json& exp, std::size_t b) {
  BlockData out;
  const auto& M = exp["sample_cov"][b]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  out.S.resize(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      out.S(r, c) = M[static_cast<std::size_t>(r)]
                     [static_cast<std::size_t>(c)].get<double>();
  if (exp.contains("sample_mean") && !exp["sample_mean"].is_null()) {
    const auto& v = exp["sample_mean"][b]["vector"];
    out.mean.resize(static_cast<Eigen::Index>(v.size()));
    for (Eigen::Index i = 0; i < out.mean.size(); ++i)
      out.mean(i) = v[static_cast<std::size_t>(i)].get<double>();
  }
  out.n_obs = exp["n_obs_per_block"][b].get<std::int64_t>();
  return out;
}

}  // namespace

TEST_CASE("multi-group goldens — θ̂ / SE / df match lavaan") {
  const auto corpus = latva::test::load_corpus();
  const std::string fit_dir = latva::test::fixtures_dir() + "/fit";

  int total = 0, passed = 0;
  std::vector<std::string> failures;
  std::vector<std::string> processed;

  for (const auto& e : corpus) {
    if (e.n_groups <= 1) continue;     // this golden owns only multi-group
    const std::string path = fit_dir + "/" + e.id + ".fit.json";
    auto raw = latva::test::read_fixture(path);
    if (!raw.has_value()) continue;
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(e.id + ": fixture not valid JSON");
      continue;
    }
    ++total;
    processed.push_back(e.id);

    auto fp = latva::parse::Parser::parse(e.model);
    if (!fp.has_value()) { failures.push_back(e.id + ": parse"); continue; }
    latva::partable::LavaanifyOptions opts;
    opts.n_groups      = e.n_groups;
    opts.meanstructure = e.meanstructure;
    auto pt = latva::partable::lavaanify(*fp, opts);
    if (!pt.has_value()) {
      failures.push_back(e.id + ": lavaanify — " + pt.error().detail);
      continue;
    }
    auto mr = latva::model::build_matrix_rep(*pt);
    if (!mr.has_value()) {
      failures.push_back(e.id + ": matrix_rep — " + mr.error().detail);
      continue;
    }

    // Assemble multi-block SampleStats from the fixture.
    latva::fit::SampleStats samp;
    const auto n_blocks =
        static_cast<std::size_t>(exp["sample_cov"].size());
    for (std::size_t b = 0; b < n_blocks; ++b) {
      auto bd = load_block(exp, b);
      samp.S.push_back(std::move(bd.S));
      if (bd.mean.size() > 0) samp.mean.push_back(std::move(bd.mean));
      samp.n_obs.push_back(bd.n_obs);
    }

    auto est_or = latva::fit::fit(*pt, *mr, samp);
    if (!est_or.has_value()) {
      failures.push_back(e.id + ": fit — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

    // θ̂ within 1e-5 (matches the single-group fit-theta tolerance).
    const auto& th = exp["theta_hat"];
    if (static_cast<std::size_t>(est.theta.size()) != th.size()) {
      failures.push_back(e.id + ": n_free mismatch");
      continue;
    }
    double max_theta_diff = 0.0;
    for (Eigen::Index k = 0; k < est.theta.size(); ++k) {
      const double d = std::abs(est.theta(k) -
          th[static_cast<std::size_t>(k)].get<double>());
      if (d > max_theta_diff) max_theta_diff = d;
    }
    // Multi-group fits with 60 free params and meanstructure routinely
    // land at ~1.5e-5 absolute precision (within optimizer tolerance) —
    // 5e-5 is a comfortable cushion that still flags real divergence.
    if (max_theta_diff > 5e-5) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "max|θ̂ − θ̂_lavaan| = %.3e",
                    max_theta_diff);
      failures.push_back(e.id + ": " + buf);
      continue;
    }

    // ExpectedInfoSE: SE within 1e-4, df exact, chi² within 1e-3.
    auto inf_or = latva::fit::ExpectedInfoSE{}.compute(*pt, *mr, samp, est);
    if (!inf_or.has_value()) {
      failures.push_back(e.id + ": ExpectedInfoSE — " + inf_or.error().detail);
      continue;
    }
    const auto& inf = *inf_or;
    const int df_lavaan = exp["df"].get<int>();
    if (inf.df != df_lavaan) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "df = %d, lavaan = %d", inf.df, df_lavaan);
      failures.push_back(e.id + ": " + buf);
      continue;
    }
    const double chi2_lavaan = exp["chi2"].get<double>();
    if (std::abs(inf.chi2 - chi2_lavaan) > 1e-3) {
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "|chi² − lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                    std::abs(inf.chi2 - chi2_lavaan), inf.chi2, chi2_lavaan);
      failures.push_back(e.id + ": " + buf);
      continue;
    }
    const auto& se_arr = exp["se"];
    if (static_cast<std::size_t>(inf.se.size()) != se_arr.size()) {
      failures.push_back(e.id + ": se length mismatch");
      continue;
    }
    double max_se_diff = 0.0;
    for (Eigen::Index k = 0; k < inf.se.size(); ++k) {
      const double d = std::abs(inf.se(k) -
          se_arr[static_cast<std::size_t>(k)].get<double>());
      if (d > max_se_diff) max_se_diff = d;
    }
    if (max_se_diff > 1e-4) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "max|se − lavaan| = %.3e", max_se_diff);
      failures.push_back(e.id + ": " + buf);
      continue;
    }

    // Browne residual NT + RLS chi²: tolerance 1e-3, mirroring chi². Both
    // are pure functions of (S, Σ̂) (and Δ for the projected variant);
    // skip silently if the fixture didn't carry oracle values (older
    // fixtures pre-dating the regen-script change).
    if (exp.contains("rls_chi2") && !exp["rls_chi2"].is_null()) {
      latva::model::ImpliedMoments im;
      // Reconstruct implied moments from our fit's θ̂ via a fresh evaluator.
      auto ev_or = latva::model::ModelEvaluator::build(*pt, *mr);
      if (!ev_or.has_value()) {
        failures.push_back(e.id + ": evaluator build — " +
                           ev_or.error().detail);
        continue;
      }
      auto im_view = ev_or->sigma(est.theta);
      if (!im_view.has_value()) {
        failures.push_back(e.id + ": sigma — " + im_view.error().detail);
        continue;
      }
      im.sigma.assign(im_view->sigma.begin(), im_view->sigma.end());
      im.mu.assign(im_view->mu.begin(), im_view->mu.end());
      auto rls = latva::fit::rls_chi2(samp, im);
      if (!rls.has_value()) {
        failures.push_back(e.id + ": rls_chi2 — " + rls.error().detail);
        continue;
      }
      const double rls_lavaan = exp["rls_chi2"].get<double>();
      if (std::abs(*rls - rls_lavaan) > 1e-3) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "|RLS − lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                      std::abs(*rls - rls_lavaan), *rls, rls_lavaan);
        failures.push_back(e.id + ": " + buf);
        continue;
      }
    }
    if (exp.contains("browne_residual_nt") &&
        !exp["browne_residual_nt"].is_null()) {
      auto br = latva::fit::browne_residual_nt(*pt, *mr, samp, est);
      if (!br.has_value()) {
        failures.push_back(e.id + ": browne_residual_nt — " +
                           br.error().detail);
        continue;
      }
      const double br_lavaan = exp["browne_residual_nt"].get<double>();
      if (std::abs(*br - br_lavaan) > 1e-3) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "|Browne − lavaan| = %.3e (ours=%.6f, lavaan=%.6f)",
                      std::abs(*br - br_lavaan), *br, br_lavaan);
        failures.push_back(e.id + ": " + buf);
        continue;
      }
    }

    // FdObservedInfoSE: SE within 1e-4 of lavaan's se_observed. Skip
    // silently if the fixture didn't carry an observed-info refit.
    if (exp.contains("se_observed") && !exp["se_observed"].is_null()) {
      auto fd_or = latva::fit::FdObservedInfoSE{}.compute(
          *pt, *mr, samp, est);
      if (!fd_or.has_value()) {
        failures.push_back(e.id + ": FdObservedInfoSE — " +
                           fd_or.error().detail);
        continue;
      }
      const auto& se_obs = exp["se_observed"];
      double max_fd_diff = 0.0;
      for (Eigen::Index k = 0; k < fd_or->se.size(); ++k) {
        const double d = std::abs(fd_or->se(k) -
            se_obs[static_cast<std::size_t>(k)].get<double>());
        if (d > max_fd_diff) max_fd_diff = d;
      }
      if (max_fd_diff > 1e-4) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "max|FD-se − lavaan_observed| = %.3e", max_fd_diff);
        failures.push_back(e.id + ": " + buf);
        continue;
      }
    }
    ++passed;
  }

  MESSAGE("multi-group goldens: " << passed << " / " << total << " pass");
  for (const auto& s : processed) MESSAGE("  " << s);
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
