#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "../test_fit.hpp"

#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
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

namespace {

struct RobustReleaseFixture {
  nlohmann::json exp;
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  double mi_want = 0.0;
  double c_want = 0.0;
  double mis_want = 0.0;
};

RobustReleaseFixture load_release_fixture(const std::string& name) {
  const std::string path =
      magmaan::test::fixtures_dir() + "/score/" + name + ".score_robust.json";
  auto raw_json = magmaan::test::read_fixture(path);
  REQUIRE(raw_json.has_value());
  auto exp = nlohmann::json::parse(*raw_json, nullptr, false);
  REQUIRE_FALSE(exp.is_discarded());

  const std::string src = exp["input"].get<std::string>();
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  const auto& want = exp["score_tests_robust"]["rows"][0];
  RobustReleaseFixture out{std::move(exp), std::move(*pt), std::move(*rep),
                           want["mi"].get<double>(),
                           want["scaling_factor"].get<double>(),
                           want["mi_scaled"].get<double>()};
  return out;
}

Eigen::MatrixXd raw_matrix(const nlohmann::json& jraw) {
  const Eigen::Index n = static_cast<Eigen::Index>(jraw.size());
  const Eigen::Index p = static_cast<Eigen::Index>(jraw[0].size());
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < p; ++j)
      X(i, j) = jraw[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    .get<double>();
  return X;
}

}  // namespace

TEST_CASE("robust LS release-score matches the lavaan-internals oracle (DWLS)") {
  auto f = load_release_fixture("0007_robust_release_dwls_continuous");

  magmaan::data::RawData raw;
  raw.X.push_back(raw_matrix(f.exp["raw"]));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  // Diagonal-ADF (continuous DWLS) weight, matching lavaan's wls.v up to its
  // gamma denominator convention (a uniform scale, < 2e-3 at n = 600).
  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W(k, k) = 1.0 / (*G)(k, k);
  magmaan::estimate::gmm::Weight weight{W};

  auto est = magmaan::test::fit_gmm(f.pt, f.rep, *samp, weight);
  REQUIRE(est.has_value());
  auto st = inf::frontier::score_tests_robust(f.pt, f.rep, *samp, raw, *est,
                                              weight);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() == 1);
  const auto& got = st->rows[0];

  // magmaan's diag(empirical_gamma)⁻¹ weight, raw-data meat, and refit θ̂ are
  // numerically identical to lavaan's (observed gaps ~1e-9), so these gates
  // are tight.
  CHECK(std::abs(got.mi - f.mi_want) < 1e-6 * (1.0 + std::abs(f.mi_want)));
  CHECK(std::abs(got.scaling_factor - f.c_want) <
        1e-6 * (1.0 + std::abs(f.c_want)));
  CHECK(std::abs(got.mi_scaled - f.mis_want) <
        1e-6 * (1.0 + std::abs(f.mis_want)));
  CHECK(got.scaling_factor > 1.1);  // genuinely non-trivial (c ≈ 1.18)
}

TEST_CASE("robust ordinal release-score matches the lavaan-internals oracle (WLSMV)") {
  auto f = load_release_fixture("0008_robust_release_wlsmv_ordinal");

  auto stats = magmaan::data::ordinal_stats_from_integer_data(
      {raw_matrix(f.exp["raw"])});
  REQUIRE(stats.has_value());
  auto est = magmaan::test::fit_ordinal_bounded(
      f.pt, f.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(est.has_value());

  auto st = magmaan::estimate::frontier::score_tests_ordinal_robust(
      f.pt, f.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() == 1);
  const auto& got = st->rows[0];

  // magmaan's polychoric stats pipeline reproduces lavaan's delta/W/NACOV
  // almost exactly at the pinned version: the observed c gap is ~6e-10 and the
  // mi gap ~4e-4 (refit θ̂ difference only), so these gates are tight.
  CHECK(std::abs(got.mi - f.mi_want) < 5e-3 * (1.0 + std::abs(f.mi_want)));
  CHECK(std::abs(got.scaling_factor - f.c_want) <
        1e-6 * (1.0 + std::abs(f.c_want)));
  CHECK(std::abs(got.mi_scaled - f.mis_want) <
        5e-3 * (1.0 + std::abs(f.mis_want)));
  CHECK(got.scaling_factor < 0.95);  // genuinely non-trivial (c ≈ 0.86)
}

TEST_CASE("robust FIML release-score matches the lavaan-internals oracle (MLR)") {
  const std::string path =
      magmaan::test::fixtures_dir() +
      "/score/0009_robust_release_fiml_mlr.score_robust.json";
  auto raw_json = magmaan::test::read_fixture(path);
  REQUIRE(raw_json.has_value());
  auto exp = nlohmann::json::parse(*raw_json, nullptr, false);
  REQUIRE_FALSE(exp.is_discarded());

  // FIML estimates means, so build with the mean structure on.
  const std::string src = exp["input"].get<std::string>();
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions bo;
  bo.meanstructure = true;
  auto pt = magmaan::spec::build(*fp, bo);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  // Raw data with JSON-null missing cells → NaN + observed mask.
  const auto& jraw = exp["raw"];
  const Eigen::Index n = static_cast<Eigen::Index>(jraw.size());
  const Eigen::Index p = static_cast<Eigen::Index>(jraw[0].size());
  Eigen::MatrixXd X(n, p);
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto& row = jraw[static_cast<std::size_t>(i)];
    for (Eigen::Index j = 0; j < p; ++j) {
      const auto& cell = row[static_cast<std::size_t>(j)];
      if (cell.is_null()) {
        X(i, j) = std::numeric_limits<double>::quiet_NaN();
        M(i, j) = 0;
      } else {
        X(i, j) = cell.get<double>();
        M(i, j) = 1;
      }
    }
  }
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  raw.mask.push_back(std::move(M));

  auto est = magmaan::test::fit_fiml(*pt, *rep, raw);
  REQUIRE(est.has_value());
  auto st = inf::frontier::score_tests_fiml_robust(*pt, *rep, raw, *est);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() == 1);
  const auto& got = st->rows[0];

  const auto& want = exp["score_tests_robust"]["rows"][0];
  const double mi_want = want["mi"].get<double>();
  const double c_want = want["scaling_factor"].get<double>();
  const double mis_want = want["mi_scaled"].get<double>();

  // c is the convention-free θ-space ratio; magmaan's analytic observed
  // information and casewise-deviance meat reproduce lavaan's
  // information.observed / crossprod(lavScores) at the same FIML θ̂.
  CHECK(std::abs(got.mi - mi_want) < 5e-3 * (1.0 + std::abs(mi_want)));
  CHECK(std::abs(got.scaling_factor - c_want) <
        5e-3 * (1.0 + std::abs(c_want)));
  CHECK(std::abs(got.mi_scaled - mis_want) <
        5e-3 * (1.0 + std::abs(mis_want)));
  CHECK(got.scaling_factor > 1.5);  // genuinely non-trivial (c ≈ 2.16)
}

TEST_CASE("robust multi-group release-score matches the lavaan-internals oracle (MLM)") {
  const std::string path =
      magmaan::test::fixtures_dir() +
      "/score/0010_robust_release_multigroup_mlm.score_robust.json";
  auto raw_json = magmaan::test::read_fixture(path);
  REQUIRE(raw_json.has_value());
  auto exp = nlohmann::json::parse(*raw_json, nullptr, false);
  REQUIRE_FALSE(exp.is_discarded());

  // Two-group model with explicit per-group loading labels + `==` rows.
  const std::string src = exp["input"].get<std::string>();
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions bo;
  bo.n_groups = 2;
  auto pt = magmaan::spec::build(*fp, bo);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  // Raw data is a list of two complete blocks.
  const auto& jraw = exp["raw"];
  REQUIRE(jraw.size() == 2);
  magmaan::data::RawData raw;
  raw.X.push_back(raw_matrix(jraw[0]));
  raw.X.push_back(raw_matrix(jraw[1]));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  REQUIRE(samp->S.size() == 2);

  auto est = magmaan::test::fit(*pt, *rep, *samp);  // ML = MLM point estimates
  REQUIRE(est.has_value());
  inf::frontier::RobustScoreOptions opts;
  opts.spec = {rob::Information::Expected, rob::WeightMoments::Structured,
               rob::ScoreCovariance::Empirical};
  auto st = inf::frontier::score_tests_robust(*pt, *rep, *samp, raw, *est, opts);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() == 3);  // a2==b2, a3==b3, a4==b4

  // The oracle releases the first equality (a2==b2); magmaan emits releases in
  // partable constraint order, so row 0 is the same release.
  const auto& want = exp["score_tests_robust"]["rows"][0];
  const double mi_want = want["mi"].get<double>();
  const double c_want = want["scaling_factor"].get<double>();
  const double mis_want = want["mi_scaled"].get<double>();
  const auto& got = st->rows[0];

  // The pooled-information bread and the n_b/N-weighted meat are convention-free
  // in θ-space; the gate is the multiplier-sensitive 5e-3 used by the other
  // multi-group goldens (refit θ̂ difference only).
  CHECK(std::abs(got.mi - mi_want) < 5e-3 * (1.0 + std::abs(mi_want)));
  CHECK(std::abs(got.scaling_factor - c_want) <
        5e-3 * (1.0 + std::abs(c_want)));
  CHECK(std::abs(got.mi_scaled - mis_want) <
        5e-3 * (1.0 + std::abs(mis_want)));
  CHECK(got.scaling_factor > 1.1);  // genuinely non-trivial (c ≈ 1.24)
}
