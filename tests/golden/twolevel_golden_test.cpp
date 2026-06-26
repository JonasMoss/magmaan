// Two-level (multilevel) ML lavaan-parity golden test — Stream D of the
// multilevel-SEM plan. Drives the public api end to end on the fixtures emitted
// by tests/tools/regen_oracle_twolevel.R and checks theta-hat, standard SEs,
// chi-square, and df against lavaan::sem(model, data, cluster="cluster").
//
// During the parallel fan-out the level: parser (Stream A), the cluster
// statistics (Stream B), and the two-level core (Stream C) are Phase-0 stubs,
// so api::model_from_lavaan / data_from_cluster / fit return a known sentinel.
// The test verifies that sentinel and reports the case PENDING rather than
// asserting parity; once A/B/C integrate, those early branches are no longer
// taken and the full parity assertions below run.

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

// True while the pipeline is still wired to Phase-0 stubs (or the level: parser
// is not yet integrated). Keyed on the documented sentinel strings.
bool is_pending(const magmaan::api::Error& e) {
  const std::string& d = e.detail;
  const auto has = [&](const char* s) { return d.find(s) != std::string::npos; };
  return has("not yet implemented") || has("not implemented") ||
         has("level") ||   // `level:` block header (Stream A)
         has("':'");       // lexer rejects the `:` after level/group (Stream A)
}

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

    // 1) model (Stream A: the level: block axis).
    auto model = api::model_from_lavaan(syntax);
    if (!model) {
      CHECK_MESSAGE(is_pending(model.error()),
                    "unexpected model error: ", model.error().detail);
      MESSAGE("twolevel PENDING (model / level: parser): ", rel);
      continue;
    }

    // 2) clustered sufficient statistics (Stream B).
    Eigen::MatrixXd X = matrix_from_json(j.at("data").at("X"));
    std::vector<std::int32_t> cluster_id =
        cluster_ids_from(j.at("data").at("cluster_id"));
    data::RawData raw;
    raw.X = {std::move(X)};
    auto data = api::data_from_cluster(*model, std::move(raw), cluster_id);
    if (!data) {
      CHECK_MESSAGE(is_pending(data.error()),
                    "unexpected cluster-stats error: ", data.error().detail);
      MESSAGE("twolevel PENDING (cluster stats): ", rel);
      continue;
    }

    // 3) two-level ML fit (Stream C).
    auto fit = api::fit(*model, *data, api::twolevel_ml());
    if (!fit) {
      CHECK_MESSAGE(is_pending(fit.error()),
                    "unexpected fit error: ", fit.error().detail);
      MESSAGE("twolevel PENDING (two-level core): ", rel);
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
