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
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

// Mirror the skip list from the expected-info golden — same under-
// identified models can't produce observed SEs either (lavaan refuses to
// invert the info matrix; we'd refuse for the same reason).
const std::set<std::string> kSkipForObservedGoldens = {
    "0010_covariance",
    "0013_string_label",
    "0015_start_call",
    "0018_na_modifier",
};

// Run a single information_* function on one fixture, comparing the derived
// SEs to `se_observed` in the fixture. Appends a `<id>: <message>` entry to
// `failures` on disagreement and increments `passed` on success.
template <class InfoFn>
bool run_one(const magmaan::test::CorpusEntry&   e,
             const nlohmann::json&             exp,
             InfoFn                            info_fn,
             std::string_view                  method_name,
             std::vector<std::string>&         failures,
             int&                              passed,
             int&                              total) {
  ++total;

  auto fp = magmaan::parse::Parser::parse(e.model);
  if (!fp.has_value()) { failures.push_back(e.id + ": parse"); return true; }
  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = e.meanstructure;
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  if (!pt.has_value()) { failures.push_back(e.id + ": lavaanify"); return true; }
  auto mr = magmaan::model::build_matrix_rep(*pt);
  if (!mr.has_value()) { failures.push_back(e.id + ": matrix_rep"); return true; }

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
    failures.push_back(e.id + " [" + std::string(method_name) +
                       "]: fit failed — " + est_or.error().detail);
    return true;
  }

  auto info_or = info_fn(*pt, *mr, samp, *est_or);
  if (!info_or.has_value()) {
    failures.push_back(e.id + " [" + std::string(method_name) +
                       "]: information failed — " + info_or.error().detail);
    return true;
  }
  auto vcov_or = magmaan::nt::infer::vcov(*info_or, *pt);
  if (!vcov_or.has_value()) {
    failures.push_back(e.id + " [" + std::string(method_name) +
                       "]: vcov failed — " + vcov_or.error().detail);
    return true;
  }
  const Eigen::VectorXd se_v = magmaan::nt::infer::se(*vcov_or);

  const auto& se_arr = exp["se_observed"];
  if (static_cast<std::size_t>(se_v.size()) != se_arr.size()) {
    failures.push_back(e.id + " [" + std::string(method_name) +
                       "]: se length mismatch");
    return true;
  }
  double max_diff = 0.0;
  for (Eigen::Index k = 0; k < se_v.size(); ++k) {
    const double d = std::abs(se_v(k) -
        se_arr[static_cast<std::size_t>(k)].get<double>());
    if (d > max_diff) max_diff = d;
  }
  // 1e-4 absolute matches the expected-info tolerance. Observed SE
  // entries are typically O(0.05–0.2), so this is a strict comparison.
  if (max_diff > 1e-4) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), " [%s]: max |se - lavaan| = %.3e",
                  std::string(method_name).c_str(), max_diff);
    failures.push_back(e.id + buf);
    return true;
  }
  ++passed;
  return true;
}

}  // namespace

TEST_CASE("observed inference goldens — FD + analytic vs lavaan") {
  const auto corpus = magmaan::test::load_corpus();
  const std::string fit_dir = magmaan::test::fixtures_dir() + "/fit";

  int total_fd = 0, passed_fd = 0;
  int total_an = 0, passed_an = 0;
  std::vector<std::string> failures;
  std::vector<std::string> skipped;
  std::vector<std::string> no_oracle;

  for (const auto& e : corpus) {
    const std::string path = fit_dir + "/" + e.id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) continue;
    if (e.n_groups > 1) continue;   // multi-group has its own golden
    if (kSkipForObservedGoldens.count(e.id)) {
      skipped.push_back(e.id);
      continue;
    }

    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(e.id + ": fixture not valid JSON");
      continue;
    }
    if (!exp.contains("se_observed") || exp["se_observed"].is_null()) {
      no_oracle.push_back(e.id);
      continue;
    }

    auto fd_fn = [](auto&&... args) {
      return magmaan::nt::infer::information_observed_fd(std::forward<decltype(args)>(args)...);
    };
    auto an_fn = [](auto&&... args) {
      return magmaan::nt::infer::information_observed_analytic(std::forward<decltype(args)>(args)...);
    };
    run_one(e, exp, fd_fn, "FD",       failures, passed_fd, total_fd);
    run_one(e, exp, an_fn, "analytic", failures, passed_an, total_an);
  }

  MESSAGE("observed inference goldens: FD " << passed_fd << "/" << total_fd
          << ", analytic " << passed_an << "/" << total_an
          << " (+ " << skipped.size() << " skipped under-identified, "
          << no_oracle.size() << " no observed-info oracle)");
  for (const auto& s : skipped)   MESSAGE("  SKIP    " << s);
  for (const auto& s : no_oracle) MESSAGE("  NO-OBS  " << s);
  for (const auto& f : failures)  MESSAGE("  FAIL    " << f);

  CHECK(passed_fd == total_fd);
  CHECK(passed_an == total_an);
}
