#include <doctest/doctest.h>

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

// Models whose lavaan matrix representation uses surfaces P5.1 doesn't
// implement (regression-style "reduced LISREL" → Beta on observed,
// CFA-with-structural → Beta on latents, intercept-only → Nu/Alpha).
// They land in P5.2 / v0.4. The test reports them as deferred so they
// don't drown out real failures.
const std::set<std::string> kDeferred = {
    "0012_intercept",           // v0.4: mean structure (Nu / Alpha)
};

// One-sentence diff: row-id keyed comparison of expected cells against
// our cell_for_row entries.
std::string diff_cells(const magmaan::model::MatrixRep& mr,
                       const nlohmann::json& exp) {
  // Build row_id → expected cell map.
  if (!exp.contains("cells") || !exp["cells"].is_array()) return "fixture has no 'cells'";
  std::unordered_map<int, nlohmann::json> exp_by_id;
  for (const auto& c : exp["cells"]) exp_by_id[c["row_id"].get<int>()] = c;

  for (std::size_t i = 0; i < mr.cell_for_row.size(); ++i) {
    const auto& got = mr.cell_for_row[i];
    const int row_id = static_cast<int>(i + 1);  // mirrors LatentStructure.id
    if (!exp_by_id.count(row_id)) {
      return "cell row_id=" + std::to_string(row_id) +
             " missing from fixture (got mat=" +
             std::string(magmaan::model::to_string(got.mat)) + ")";
    }
    const auto& exp_c = exp_by_id[row_id];
    const bool exp_used = exp_c.value("used", true);
    if (got.used != exp_used) {
      return "row_id=" + std::to_string(row_id) +
             " used: got=" + (got.used ? "true" : "false") +
             " expected=" + (exp_used ? "true" : "false");
    }
    if (!exp_used) continue;
    const std::string exp_mat   = exp_c["mat"].get<std::string>();
    const std::string got_mat   = std::string(magmaan::model::to_string(got.mat));
    if (got_mat != exp_mat) {
      return "row_id=" + std::to_string(row_id) +
             " mat: got=" + got_mat + " expected=" + exp_mat;
    }
    const int exp_row = exp_c["row"].get<int>();
    const int exp_col = exp_c["col"].get<int>();
    if (got.row != exp_row || got.col != exp_col) {
      return "row_id=" + std::to_string(row_id) +
             " (mat=" + got_mat + ") cell: got=(" +
             std::to_string(got.row) + "," + std::to_string(got.col) +
             ") expected=(" + std::to_string(exp_row) + "," +
             std::to_string(exp_col) + ")";
    }
  }
  return "";
}

}  // namespace

TEST_CASE("matrix_rep goldens — pure-CFA scope (P5.1)") {
  const auto corpus = magmaan::test::load_corpus();
  REQUIRE(!corpus.empty());
  const std::string mr_dir = magmaan::test::fixtures_dir() + "/matrix_rep";

  int total = 0, passed = 0, deferred_count = 0;
  std::vector<std::string> failures;
  std::vector<std::string> deferred_listed;

  for (const auto& e : corpus) {
    const std::string path = mr_dir + "/" + e.id + ".mrep.json";
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
    auto pt = magmaan::spec::build(*fp);
    if (!pt.has_value()) {
      failures.push_back(e.id + ": lavaanify failed — " + pt.error().detail);
      continue;
    }
    auto mr = magmaan::model::build_matrix_rep(*pt);
    if (!mr.has_value()) {
      failures.push_back(e.id + ": build_matrix_rep failed — " +
                         mr.error().detail);
      continue;
    }

    auto d = diff_cells(*mr, exp);
    if (d.empty()) ++passed;
    else           failures.push_back(e.id + ": " + d);
  }

  MESSAGE("matrix_rep goldens: " << passed << " / " << total << " pass"
          << " (+ " << deferred_count << " deferred to P5.2 / v0.4)");
  for (const auto& d : deferred_listed)
    MESSAGE("  DEFERRED " << d);
  for (const auto& f : failures) MESSAGE("  FAIL " << f);

  CHECK(passed == total);
}
