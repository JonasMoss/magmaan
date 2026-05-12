#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "latva/parse/op.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaan_view.hpp"
#include "latva/partable/lavaanify.hpp"
#include "latva/partable/partable.hpp"
#include "latva/partable/start_hints.hpp"

namespace {

// Translate the lavaan-shaped projection of our model to the same JSON shape
// that tools/regen_oracle.R writes for the `ptable` layer (one row per object,
// fields: id, user, lhs, op, rhs, block, group, free, exo, ustart, label,
// plabel). NaN is written as JSON null to match the R `na = "null"` choice.
nlohmann::json ptable_to_json(std::string_view input,
                              std::string_view corpus_id,
                              const latva::partable::LavaanParTable& pt) {
  using latva::test::op_to_lavaan_string;
  nlohmann::json j;
  j["_meta"] = {
      {"format_version", 1},
      {"fixture_kind",   "ptable"},
      {"corpus_id",      std::string(corpus_id)},
      {"tool",           "latva::lavaanify"}};
  j["input"] = std::string(input);

  nlohmann::json rows = nlohmann::json::array();
  for (std::size_t i = 0; i < pt.size(); ++i) {
    nlohmann::json row;
    row["id"]     = pt.id[i];
    row["user"]   = static_cast<int>(pt.user[i]);
    row["lhs"]    = pt.lhs[i];
    row["op"]     = op_to_lavaan_string(pt.op[i]);
    row["rhs"]    = pt.rhs[i];
    row["block"]  = pt.block[i];
    row["group"]  = pt.group[i];
    row["free"]   = pt.free[i];
    row["exo"]    = static_cast<int>(pt.exo[i]);
    const double u = pt.ustart[i];
    if (std::isnan(u)) row["ustart"] = nullptr;
    else               row["ustart"] = u;
    row["label"]  = pt.label[i];
    row["plabel"] = pt.plabel[i];
    rows.push_back(std::move(row));
  }
  j["rows"] = std::move(rows);
  return j;
}

}  // namespace

// Walks two ptable JSONs row-by-row and field-by-field, returning the
// first mismatch (or empty string if equal). Much more readable than a
// full nlohmann::json::dump() diff.
std::string diff_ptable(const nlohmann::json& got, const nlohmann::json& exp) {
  if (!got.contains("rows") || !exp.contains("rows")) return "missing 'rows'";
  const auto& g = got["rows"];
  const auto& e = exp["rows"];
  if (g.size() != e.size()) {
    return "row count: got=" + std::to_string(g.size()) +
           " expected=" + std::to_string(e.size());
  }
  for (std::size_t i = 0; i < g.size(); ++i) {
    for (auto it = e[i].begin(); it != e[i].end(); ++it) {
      const std::string key = it.key();
      if (!g[i].contains(key) || g[i][key] != *it) {
        std::string s = "row[" + std::to_string(i) + "]." + key +
                        ": got=" +
                        (g[i].contains(key) ? g[i][key].dump() : "<missing>") +
                        " expected=" + it->dump();
        return s;
      }
    }
  }
  return "";
}

TEST_CASE("ptable goldens — every corpus entry that lavaanify can handle") {
  const auto corpus = latva::test::load_corpus();
  REQUIRE(!corpus.empty());
  const std::string ptable_dir = latva::test::fixtures_dir() + "/ptable";

  // Lavaan oddities we deliberately do not mirror, with rationale. These
  // are skipped from the strict pass/fail count but still surface as
  // tagged messages so future drift is visible.
  const std::vector<std::pair<std::string, std::string>> tolerated_divergences = {
      {"0008_defined_param",
       "lavaan emits two identical `:=` rows differing only in plabel "
       "(internal quirk in lavaanify); we emit one."},
  };
  auto is_tolerated = [&](std::string_view id) {
    for (const auto& [tid, why] : tolerated_divergences) if (tid == id) return true;
    return false;
  };

  int total = 0, passed = 0;
  std::vector<std::string> failures;
  std::vector<std::string> tolerated;

  for (const auto& e : corpus) {
    const std::string path = ptable_dir + "/" + e.id + ".ptable.json";
    auto raw = latva::test::read_fixture(path);
    if (!raw.has_value()) continue;        // no oracle for this corpus entry
    if (is_tolerated(e.id)) {
      tolerated.push_back(e.id);
      continue;
    }
    ++total;

    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (exp.is_discarded()) {
      failures.push_back(e.id + ": fixture not valid JSON");
      continue;
    }

    auto fp = latva::parse::Parser::parse(e.model);
    if (!fp.has_value()) {
      failures.push_back(e.id + ": parse failed — " + fp.error().detail);
      continue;
    }
    latva::partable::Starts starts;
    latva::partable::LatentNames names;
    auto pt = latva::partable::lavaanify(*fp, {}, &starts, &names);
    if (!pt.has_value()) {
      failures.push_back(e.id + ": lavaanify failed — " + pt.error().detail);
      continue;
    }

    auto got = ptable_to_json(e.model, e.id,
                              latva::partable::to_lavaan_partable(*pt, names, starts));
    auto d = diff_ptable(latva::test::strip_meta(got),
                         latva::test::strip_meta(exp));
    if (d.empty()) ++passed;
    else           failures.push_back(e.id + ": " + d);
  }

  MESSAGE("ptable goldens: " << passed << " / " << total << " pass"
          << " (+ " << tolerated.size() << " tolerated divergence(s))");
  for (const auto& t : tolerated) {
    for (const auto& [tid, why] : tolerated_divergences) {
      if (tid == t) MESSAGE("  TOLERATED " << t << ": " << why);
    }
  }
  for (const auto& f : failures) MESSAGE("  FAIL " << f);

  CHECK(passed == total);
}
