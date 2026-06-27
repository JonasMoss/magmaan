// Two-level (multilevel) ML lavaan-parity golden test — Stream D of the
// multilevel-SEM plan. Drives the public api end to end on the fixtures emitted
// by tests/tools/regen_oracle_twolevel.R and checks theta-hat, standard SEs,
// chi-square, and df against lavaan::sem(model, data, cluster="cluster").
//
// The level: parser, the cluster statistics, and the two-level core are now
// integrated, so any failure of api::model_from_lavaan / data_from_cluster /
// fit is a real parity failure (the old Phase-0 PENDING sentinel guard is
// gone).

#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"

#include "magmaan/api/sem.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/parse/op.hpp"

namespace {

using magmaan::test::load_json_fixture;
using magmaan::test::matrix_from_json;

constexpr const char* kFixtures[] = {
    "twolevel/twolevel_ri3.json",  // saturated (df=0): theta-hat + SE
    "twolevel/twolevel_1f4.json",  // df=4
    "twolevel/twolevel_2f6.json",  // df=16, well-fitting
};

// (lhs, op, rhs, block) — the lavaan parameter identity, block == level for a
// single-group two-level model (1 = within, 2 = between).
using Key = std::tuple<std::string, std::string, std::string, int>;

std::vector<std::int32_t> cluster_ids_from(const nlohmann::json& arr) {
  std::vector<std::int32_t> out;
  out.reserve(arr.size());
  for (const auto& v : arr) out.push_back(v.get<std::int32_t>());
  return out;
}

}  // namespace

TEST_CASE("twolevel golden: lavaan cluster-ML parity") {
  using namespace magmaan;

  for (const char* rel_c : kFixtures) {
    const std::string rel(rel_c);
    CAPTURE(rel);
    const auto j = load_json_fixture(rel);
    REQUIRE_FALSE(j.is_discarded());

    const std::string syntax = j.at("model").get<std::string>();
    const auto& exp = j.at("expected");

    // 1) model (the level: block axis).
    auto model = api::model_from_lavaan(syntax);
    if (!model) {
      FAIL_CHECK("twolevel model_from_lavaan() failed: ", model.error().detail);
      continue;
    }

    // 2) clustered sufficient statistics.
    Eigen::MatrixXd X = matrix_from_json(j.at("data").at("X"));
    std::vector<std::int32_t> cluster_id =
        cluster_ids_from(j.at("data").at("cluster_id"));
    data::RawData raw;
    raw.X = {std::move(X)};
    auto data = api::data_from_cluster(*model, std::move(raw), cluster_id);
    if (!data) {
      FAIL_CHECK("twolevel data_from_cluster() failed: ", data.error().detail);
      continue;
    }

    // 3) two-level ML fit.
    auto fit = api::fit(*model, *data, api::twolevel_ml());
    if (!fit) {
      FAIL_CHECK("twolevel fit() failed: ", fit.error().detail);
      continue;
    }

    // ---------- full parity (post-integration) ----------

    // chi-square LRT + df.
    auto t = api::test(*fit, api::standard_chi_square());
    if (!t) {
      FAIL_CHECK("twolevel test() failed: ", t.error().detail);
      continue;
    }
    CHECK(t->df == exp.at("df").get<int>());
    CHECK(t->statistic ==
          doctest::Approx(exp.at("chisq").get<double>()).epsilon(1e-4));

    // standard SEs (expected-information inverse).
    auto se = api::standard_errors(*fit, api::expected_information());
    if (!se) {
      FAIL_CHECK("twolevel standard_errors() failed: ", se.error().detail);
      continue;
    }
    const Eigen::VectorXd& theta = fit->estimates().theta;
    const Eigen::VectorXd& se_vec = se->se;
    REQUIRE(se_vec.size() == theta.size());

    // theta-hat + SE matched by (lhs, op, rhs, block). to_lavaan_partable's
    // `free` column is the 1-based theta index per row, so the fitted value /
    // SE for a free row is theta[free-1] / se[free-1].
    const auto lpt =
        compat::lavaan::to_lavaan_partable(model->structure(), model->names());
    std::map<Key, std::pair<double, double>> ours;
    for (std::size_t i = 0; i < lpt.size(); ++i) {
      const std::int32_t f = lpt.free[i];
      if (f <= 0) continue;
      Key k{lpt.lhs[i], std::string(parse::to_string(lpt.op[i])), lpt.rhs[i],
            lpt.block[i]};
      ours[std::move(k)] = {theta[f - 1], se_vec[f - 1]};
    }

    int compared = 0;
    for (const auto& prow : exp.at("params")) {
      if (prow.at("free").get<int>() <= 0) continue;
      Key k{prow.at("lhs").get<std::string>(), prow.at("op").get<std::string>(),
            prow.at("rhs").get<std::string>(), prow.at("block").get<int>()};
      const auto it = ours.find(k);
      if (it == ours.end()) {
        FAIL_CHECK("magmaan has no free param for lavaan row ",
                   std::get<0>(k), " ", std::get<1>(k), " ", std::get<2>(k),
                   " block=", std::get<3>(k));
        continue;
      }
      CHECK(it->second.first ==
            doctest::Approx(prow.at("est").get<double>()).epsilon(1e-4));
      CHECK(it->second.second ==
            doctest::Approx(prow.at("se").get<double>()).epsilon(2e-3));
      ++compared;
    }
    CHECK(compared == exp.at("npar").get<int>());
  }
}

// Multi-group two-level parity by self-consistency against the single-group
// lavaan oracle. lavaan cannot fit a multilevel model with `group=` (lavaanify
// drops the group axis for `level:` models: it emits only n_levels blocks, not
// n_groups * n_levels, and sem(cluster=, group=) then errors), so there is no
// direct multigroup-twolevel oracle. Instead we exploit the fact that two
// groups with *no* cross-group equality constraints are two independent fits:
// fitting the SAME single-group fixture data as a 2-group model must reproduce,
// per group, the lavaan-validated single-group estimates and SEs, with the LRT
// statistic and df both doubling. This pins the multi-group data boundary
// (data_from_cluster building one ClusterGroupStats per group), the generalized
// df formula df = Σ_g [p_g(p_g+1) + p_g] − q, and the core's per-group loop.
TEST_CASE("twolevel multigroup: two-group self-consistency vs single-group oracle") {
  using namespace magmaan;
  constexpr int kNLevels = 2;

  for (const char* rel_c : kFixtures) {
    const std::string rel(rel_c);
    CAPTURE(rel);
    const auto j = load_json_fixture(rel);
    REQUIRE_FALSE(j.is_discarded());

    const std::string syntax = j.at("model").get<std::string>();
    const auto& exp = j.at("expected");

    Eigen::MatrixXd X = matrix_from_json(j.at("data").at("X"));
    std::vector<std::int32_t> cluster_id =
        cluster_ids_from(j.at("data").at("cluster_id"));

    // --- single-group oracle path (already validated above) ----------------
    // Guard every std::expected with `if (!x) { FAIL_CHECK; continue; }`:
    // under -fno-exceptions a failing REQUIRE cannot abort, so a subsequent
    // deref of an empty expected would segfault.
    auto model_s = api::model_from_lavaan(syntax);
    if (!model_s) { FAIL_CHECK("single-group model build failed: ",
                               model_s.error().detail); continue; }
    data::RawData raw_s;
    raw_s.X = {X};
    auto data_s = api::data_from_cluster(*model_s, std::move(raw_s), cluster_id);
    if (!data_s) { FAIL_CHECK("single-group data_from_cluster failed: ",
                              data_s.error().detail); continue; }
    auto fit_s = api::fit(*model_s, *data_s, api::twolevel_ml());
    if (!fit_s) { FAIL_CHECK("single-group fit failed: ",
                             fit_s.error().detail); continue; }
    auto t_s = api::test(*fit_s, api::standard_chi_square());
    if (!t_s) { FAIL_CHECK("single-group test() failed: ",
                           t_s.error().detail); continue; }

    // --- two-group path: same data replicated as group 1 and group 2 -------
    api::ModelOptions opts;
    opts.build.n_groups = 2;
    auto model_m = api::model_from_lavaan(syntax, opts);
    if (!model_m) { FAIL_CHECK("two-group model build failed: ",
                               model_m.error().detail); continue; }

    data::RawData raw_m;
    raw_m.X = {X, X};  // two identical groups
    std::vector<std::vector<std::int32_t>> cluster_ids_m = {cluster_id,
                                                            cluster_id};
    auto data_m =
        api::data_from_cluster(*model_m, std::move(raw_m), cluster_ids_m);
    if (!data_m) { FAIL_CHECK("multigroup data_from_cluster failed: ",
                              data_m.error().detail); continue; }
    auto fit_m = api::fit(*model_m, *data_m, api::twolevel_ml());
    if (!fit_m) { FAIL_CHECK("multigroup fit failed: ",
                             fit_m.error().detail); continue; }

    auto t_m = api::test(*fit_m, api::standard_chi_square());
    if (!t_m) { FAIL_CHECK("multigroup test() failed: ",
                           t_m.error().detail); continue; }

    // df doubles: df = Σ_g [p(p+1)+p] − q = 2·(p(p+1)+p) − 2q_single.
    CHECK(t_m->df == 2 * exp.at("df").get<int>());
    CHECK(t_m->df == 2 * t_s->df);
    // The LRT statistic doubles (two independent identical groups).
    CHECK(t_m->statistic ==
          doctest::Approx(2.0 * t_s->statistic).epsilon(1e-4));
    CHECK(t_m->statistic ==
          doctest::Approx(2.0 * exp.at("chisq").get<double>()).epsilon(1e-3));
    // q doubles too.
    CHECK(fit_m->estimates().theta.size() ==
          2 * fit_s->estimates().theta.size());

    auto se_m = api::standard_errors(*fit_m, api::expected_information());
    if (!se_m) { FAIL_CHECK("multigroup standard_errors() failed: ",
                            se_m.error().detail); continue; }
    const Eigen::VectorXd& theta_m = fit_m->estimates().theta;
    const Eigen::VectorXd& se_vec_m = se_m->se;
    if (se_vec_m.size() != theta_m.size()) {
      FAIL_CHECK("multigroup SE/theta size mismatch"); continue;
    }

    // Per-group estimates/SEs must equal the lavaan single-group oracle. Match
    // each free row of the 2-group partable by (lhs, op, rhs, level), where
    // level = block − (group−1)·n_levels, against the fixture's free rows.
    const auto lpt =
        compat::lavaan::to_lavaan_partable(model_m->structure(),
                                           model_m->names());
    // Fixture lookup keyed by (lhs, op, rhs, level).
    std::map<Key, std::pair<double, double>> oracle;
    for (const auto& prow : exp.at("params")) {
      if (prow.at("free").get<int>() <= 0) continue;
      Key k{prow.at("lhs").get<std::string>(), prow.at("op").get<std::string>(),
            prow.at("rhs").get<std::string>(), prow.at("level").get<int>()};
      oracle[std::move(k)] = {prow.at("est").get<double>(),
                              prow.at("se").get<double>()};
    }

    int compared_m = 0;
    for (std::size_t i = 0; i < lpt.size(); ++i) {
      const std::int32_t f = lpt.free[i];
      if (f <= 0) continue;
      const int level = lpt.block[i] - (lpt.group[i] - 1) * kNLevels;
      Key k{lpt.lhs[i], std::string(parse::to_string(lpt.op[i])), lpt.rhs[i],
            level};
      const auto it = oracle.find(k);
      if (it == oracle.end()) {
        FAIL_CHECK("multigroup free row not in single-group oracle: ",
                   lpt.lhs[i], " ", std::string(parse::to_string(lpt.op[i])),
                   " ", lpt.rhs[i], " level=", level);
        continue;
      }
      CHECK(theta_m[f - 1] == doctest::Approx(it->second.first).epsilon(1e-4));
      CHECK(se_vec_m[f - 1] == doctest::Approx(it->second.second).epsilon(2e-3));
      ++compared_m;
    }
    // Every free param of both groups was checked: 2 × npar.
    CHECK(compared_m == 2 * exp.at("npar").get<int>());
  }
}
