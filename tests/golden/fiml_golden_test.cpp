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
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

const std::vector<std::string> kFimlFixtures = {
    "0001_one_factor_hs_fiml",
    "0002_three_factor_hs_fiml",
    "0003_equal_loading_hs_fiml",
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
    opts.meanstructure = true;
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
    auto est_or = magmaan::estimate::fit_fiml(*pt, *mr, raw);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit_fiml — " + est_or.error().detail);
      continue;
    }
    const auto& est = *est_or;

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
    for (Eigen::Index k = 0; k < est.theta.size(); ++k) {
      const double d = std::abs(est.theta(k) -
          th[static_cast<std::size_t>(k)].get<double>());
      max_diff = std::max(max_diff, d);
    }

    if (max_diff <= 5e-6) {
      ++passed;
    } else {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "max |θ̂ - lavaan| = %.3e (iters=%d, fmin=%.10f)",
                    max_diff, est.iterations, est.fmin);
      failures.push_back(id + ": " + buf);
    }
  }

  MESSAGE("FIML goldens: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
