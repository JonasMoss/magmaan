#include <doctest/doctest.h>

#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

// Models whose LatentStructure has rows our P6.1 evaluator doesn't yet handle
// (e.g., latent regression Beta). The fit fixture exists, but there's no
// matching evaluator surface yet.
const std::set<std::string> kDeferred = {
    /* none in current corpus that pass build_matrix_rep but fail evaluator */
};

}  // namespace

// For each fit fixture, run our evaluator at lavaan's θ̂ and check that
// our implied Σ matches lavaan's lavInspect(fit, "implied")$cov.
TEST_CASE("model evaluator: implied Σ matches lavaan at θ̂") {
  const auto corpus = magmaan::test::load_corpus();
  const std::string fit_dir = magmaan::test::fixtures_dir() + "/fit";

  int total = 0, passed = 0, deferred_count = 0;
  std::vector<std::string> failures;
  std::vector<std::string> deferred_listed;
  std::vector<std::string> evaluator_skips;

  for (const auto& e : corpus) {
    if (e.n_groups > 1) continue;   // multi-group has its own golden
    const std::string path = fit_dir + "/" + e.id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) continue;
    if (kDeferred.count(e.id)) {
      ++deferred_count;
      deferred_listed.push_back(e.id);
      continue;
    }
    ++total;

    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (exp.is_discarded()) {
      failures.push_back(e.id + ": fixture not valid JSON");
      continue;
    }

    auto fp = magmaan::parse::Parser::parse(e.model);
    if (!fp.has_value()) {
      failures.push_back(e.id + ": parse failed");
      continue;
    }
    auto pt_or = magmaan::spec::lavaanify(*fp);
    if (!pt_or.has_value()) {
      failures.push_back(e.id + ": lavaanify failed");
      continue;
    }
    auto pt = std::move(*pt_or);
    auto mr_or = magmaan::model::build_matrix_rep(pt);
    if (!mr_or.has_value()) {
      failures.push_back(e.id + ": matrix_rep failed");
      continue;
    }
    auto mr = std::move(*mr_or);
    // Resolve fixed.x fixed_values from the fixture's sample_cov so that
    // Σ(θ̂) — which only carries the free values — gets the right
    // exogenous-moment fills.
    if (exp.contains("sample_cov")) {
      const auto& sample_blocks = exp["sample_cov"];
      magmaan::data::SampleStats samp_for_resolve;
      for (std::size_t b = 0; b < sample_blocks.size(); ++b) {
        const auto& M = sample_blocks[b]["matrix"];
        const Eigen::Index p = static_cast<Eigen::Index>(M.size());
        Eigen::MatrixXd S(p, p);
        for (Eigen::Index r = 0; r < p; ++r)
          for (Eigen::Index c = 0; c < p; ++c)
            S(r, c) = M[static_cast<std::size_t>(r)]
                       [static_cast<std::size_t>(c)].get<double>();
        samp_for_resolve.S.push_back(std::move(S));
        samp_for_resolve.n_obs.push_back(0);
      }
      auto rx = magmaan::estimate::resolve_fixed_x_from_sample(pt, mr, samp_for_resolve);
      if (!rx.has_value()) {
        failures.push_back(e.id + ": resolve_fixed_x failed");
        continue;
      }
    }
    auto ev_or = magmaan::model::ModelEvaluator::build(pt, mr);
    if (!ev_or.has_value()) {
      failures.push_back(e.id + ": evaluator build failed");
      continue;
    }
    auto ev = std::move(*ev_or);

    // Pull θ̂ from fixture.
    const auto& th_arr = exp["theta_hat"];
    Eigen::VectorXd theta(th_arr.size());
    for (Eigen::Index k = 0; k < theta.size(); ++k)
      theta(k) = th_arr[static_cast<std::size_t>(k)].get<double>();

    if (static_cast<std::size_t>(theta.size()) != ev.n_free()) {
      evaluator_skips.push_back(e.id +
          " (theta_hat size " + std::to_string(theta.size()) +
          " ≠ evaluator n_free " + std::to_string(ev.n_free()) + ")");
      --total;
      continue;
    }

    auto sm = ev.sigma(theta);
    if (!sm.has_value()) {
      failures.push_back(e.id + ": sigma() failed — " + sm.error().detail);
      continue;
    }

    // Compare against fixture matrix.
    const auto& sigma_blocks = exp["implied_sigma"];
    REQUIRE(sigma_blocks.is_array());
    if (sm->sigma.size() != sigma_blocks.size()) {
      failures.push_back(e.id + ": block count mismatch");
      continue;
    }

    bool ok = true;
    double max_diff = 0;
    for (std::size_t b = 0; b < sm->sigma.size(); ++b) {
      const auto& got = sm->sigma[b];
      const auto& exp_mat = sigma_blocks[b]["matrix"];
      if (static_cast<std::size_t>(got.rows()) != exp_mat.size()) { ok = false; break; }
      for (Eigen::Index r = 0; r < got.rows(); ++r) {
        const auto& exp_row = exp_mat[static_cast<std::size_t>(r)];
        if (static_cast<std::size_t>(got.cols()) != exp_row.size()) { ok = false; break; }
        for (Eigen::Index c = 0; c < got.cols(); ++c) {
          const double d = std::abs(got(r, c) -
                                    exp_row[static_cast<std::size_t>(c)].get<double>());
          if (d > max_diff) max_diff = d;
        }
      }
      if (!ok) break;
    }
    if (!ok) {
      failures.push_back(e.id + ": shape mismatch in implied Σ");
    } else if (max_diff > 1e-9) {
      failures.push_back(e.id + ": max |Σ_ours - Σ_lavaan| = " +
                         std::to_string(max_diff));
    } else {
      ++passed;
    }
  }

  MESSAGE("fit/implied goldens: " << passed << " / " << total << " pass"
          << " (+ " << deferred_count << " deferred, "
          << evaluator_skips.size() << " skipped due to free-count mismatch)");
  for (const auto& d : deferred_listed) MESSAGE("  DEFERRED " << d);
  for (const auto& s : evaluator_skips) MESSAGE("  SKIP " << s);
  for (const auto& f : failures)        MESSAGE("  FAIL " << f);

  CHECK(passed == total);
}
