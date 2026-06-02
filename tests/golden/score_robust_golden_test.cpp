#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "../test_fit.hpp"

#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/spec/build.hpp"

// Robust (generalized / SB-scaled) equality-release score test vs an oracle
// assembled from lavaan's internals (tests/tools/regen_robust_score.R). lavaan
// has no robust lavTestScore (it falls back to the ordinary statistic), so the
// oracle is built from delta/wls.v/gamma/ceq.JAC; the scaling is convention-free
// in θ-space, so magmaan — deriving its own delta/weight/Γ̂ from the same raw
// data — must reproduce mi, scaling_factor, and mi_scaled.

namespace {

namespace inf = magmaan::inference;
namespace rob = magmaan::robust;

}  // namespace

TEST_CASE("robust release-score matches the lavaan-internals oracle (c != 1)") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/score/0006_robust_release_mlm.score_robust.json";
  auto raw_json = magmaan::test::read_fixture(path);
  REQUIRE(raw_json.has_value());
  auto exp = nlohmann::json::parse(*raw_json, nullptr, false);
  REQUIRE_FALSE(exp.is_discarded());

  // Model.
  const std::string src = exp["input"].get<std::string>();
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  // Raw data (n × p) → single complete block.
  const auto& jraw = exp["raw"];
  const Eigen::Index n = static_cast<Eigen::Index>(jraw.size());
  const Eigen::Index p = static_cast<Eigen::Index>(jraw[0].size());
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < p; ++j)
      X(i, j) = jraw[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    .get<double>();
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  // Refit ML (= MLM point estimates) and run the robust release-score test.
  auto est = magmaan::test::fit(*pt, *rep, *samp);
  REQUIRE(est.has_value());
  inf::frontier::RobustScoreOptions opts;
  opts.spec = {rob::Information::Expected, rob::WeightMoments::Structured,
               rob::ScoreCovariance::Empirical};
  auto st = inf::frontier::score_tests_robust(*pt, *rep, *samp, raw, *est, opts);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() == 1);

  const auto& want = exp["score_tests_robust"]["rows"][0];
  const double mi_want = want["mi"].get<double>();
  const double c_want = want["scaling_factor"].get<double>();
  const double mis_want = want["mi_scaled"].get<double>();
  const auto& got = st->rows[0];

  // Same raw data, same θ̂ (refit ≈ lavaan's to ~1e-6); the assembly is
  // convention-free, so the agreement is tight.
  CHECK(std::abs(got.mi - mi_want) < 5e-3 * (1.0 + std::abs(mi_want)));
  CHECK(std::abs(got.scaling_factor - c_want) < 5e-3 * (1.0 + std::abs(c_want)));
  CHECK(std::abs(got.mi_scaled - mis_want) < 5e-3 * (1.0 + std::abs(mis_want)));
  CHECK(got.scaling_factor > 1.2);  // genuinely non-trivial scaling (c ≈ 1.34)
}
