#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace {

// Translate the lavaan-shaped projection of our model to the same JSON shape
// that tools/regen_oracle.R writes for the `ptable` layer (one row per object,
// fields: id, user, lhs, op, rhs, block, group, free, exo, ustart, label,
// plabel). NaN is written as JSON null to match the R `na = "null"` choice.
nlohmann::json ptable_to_json(std::string_view input,
                              std::string_view corpus_id,
                              const magmaan::lavaan::LavaanParTable& pt) {
  using magmaan::test::op_to_lavaan_string;
  nlohmann::json j;
  j["_meta"] = {
      {"format_version", 1},
      {"fixture_kind",   "ptable"},
      {"corpus_id",      std::string(corpus_id)},
      {"tool",           "magmaan::lavaanify"}};
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

std::string field_string(const nlohmann::json& row, std::string_view field) {
  const auto it = row.find(std::string(field));
  if (it == row.end() || it->is_null()) return "";
  if (it->is_string()) return it->get<std::string>();
  if (it->is_number_integer()) return std::to_string(it->get<int>());
  return it->dump();
}

int field_int(const nlohmann::json& row, std::string_view field) {
  const auto it = row.find(std::string(field));
  if (it == row.end() || it->is_null()) return 0;
  return it->get<int>();
}

bool numeric_equal(const nlohmann::json& got,
                   const nlohmann::json& exp,
                   std::string_view field,
                   double tol) {
  const auto g = got.find(std::string(field));
  const auto e = exp.find(std::string(field));
  const bool g_null = (g == got.end() || g->is_null());
  const bool e_null = (e == exp.end() || e->is_null());
  if (g_null || e_null) return g_null == e_null;
  if (!g->is_number() || !e->is_number()) return *g == *e;
  return std::abs(g->get<double>() - e->get<double>()) <= tol;
}

std::string semantic_key(const nlohmann::json& row, bool include_group) {
  std::string key;
  if (include_group) key += "group=" + field_string(row, "group") + "|";
  key += "op=" + field_string(row, "op");
  key += "|lhs=" + field_string(row, "lhs");
  key += "|rhs=" + field_string(row, "rhs");
  return key;
}

std::string row_label(const nlohmann::json& row) {
  return "[" + semantic_key(row, true) + ", free=" + field_string(row, "free") +
         ", label=" + field_string(row, "label") +
         ", plabel=" + field_string(row, "plabel") + "]";
}

bool is_fixed_zero_row(const nlohmann::json& row) {
  if (field_int(row, "free") != 0) return false;
  const auto fixed_zero = [&](std::string_view field) {
    const auto it = row.find(std::string(field));
    return it != row.end() && it->is_number() &&
           std::abs(it->get<double>()) <= 1e-12;
  };
  return fixed_zero("ustart") || fixed_zero("est");
}

int candidate_score(const nlohmann::json& got, const nlohmann::json& exp) {
  int score = 0;
  for (std::string_view f : {"user", "block", "group", "free", "exo"})
    if (field_string(got, f) == field_string(exp, f)) ++score;
  for (std::string_view f : {"label", "plabel"})
    if (field_string(got, f) == field_string(exp, f)) score += 2;
  if (numeric_equal(got, exp, "ustart", 1e-12)) ++score;
  return score;
}

std::size_t best_match(const nlohmann::json& rows,
                       const std::vector<bool>& used,
                       const nlohmann::json& exp,
                       bool include_group) {
  std::size_t best = rows.size();
  int best_score = -1;
  const std::string key = semantic_key(exp, include_group);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (used[i] || semantic_key(rows[i], include_group) != key) continue;
    const int score = candidate_score(rows[i], exp);
    if (score > best_score) {
      best = i;
      best_score = score;
    }
  }
  return best;
}

std::string summarize_failures(const std::vector<std::pair<std::string, std::string>>& failures) {
  if (failures.empty()) return "";
  std::map<std::string, int> counts;
  for (const auto& [category, detail] : failures) {
    (void)detail;
    ++counts[category];
  }

  std::ostringstream out;
  out << "semantic ptable mismatch:";
  bool first = true;
  for (const auto& [category, count] : counts) {
    out << (first ? " " : ", ") << category << "=" << count;
    first = false;
  }
  const std::size_t n_show = std::min<std::size_t>(failures.size(), 6);
  for (std::size_t i = 0; i < n_show; ++i)
    out << "\n    " << failures[i].first << ": " << failures[i].second;
  if (failures.size() > n_show)
    out << "\n    ... " << (failures.size() - n_show) << " more";
  return out.str();
}

// Compare lavaan-shaped partables by model semantics rather than raw row
// position. The semantic identity is group/op/lhs/rhs; duplicate keys are
// paired deterministically by label/plabel/free/metadata similarity.
std::string diff_ptable_semantic(const nlohmann::json& got, const nlohmann::json& exp) {
  if (!got.contains("rows") || !exp.contains("rows")) return "missing 'rows'";
  const auto& g = got["rows"];
  const auto& e = exp["rows"];
  if (!g.is_array() || !e.is_array()) return "'rows' must be arrays";

  std::vector<bool> used(g.size(), false);
  std::vector<std::pair<std::string, std::string>> failures;

  auto add_field_drift = [&](const std::string& category,
                             const nlohmann::json& got_row,
                             const nlohmann::json& exp_row,
                             std::string_view field) {
    failures.emplace_back(category,
                          std::string(field) + " for " + row_label(exp_row) +
                              ": got=" + field_string(got_row, field) +
                              " expected=" + field_string(exp_row, field));
  };

  for (std::size_t ei = 0; ei < e.size(); ++ei) {
    const auto& exp_row = e[ei];
    std::size_t gi = best_match(g, used, exp_row, true);
    if (gi == g.size()) {
      gi = best_match(g, used, exp_row, false);
      if (gi == g.size()) {
        failures.emplace_back("missing row", row_label(exp_row));
        continue;
      }
      failures.emplace_back("group drift",
                            row_label(exp_row) + ": got group=" +
                                field_string(g[gi], "group") +
                                " expected group=" + field_string(exp_row, "group"));
    }

    used[gi] = true;
    const auto& got_row = g[gi];
    if (field_string(got_row, "label") != field_string(exp_row, "label"))
      add_field_drift("label drift", got_row, exp_row, "label");
    if (field_string(got_row, "plabel") != field_string(exp_row, "plabel"))
      add_field_drift("plabel drift", got_row, exp_row, "plabel");
    if (field_string(got_row, "free") != field_string(exp_row, "free"))
      add_field_drift("free-index drift", got_row, exp_row, "free");
    if (!numeric_equal(got_row, exp_row, "ustart", 1e-12)) {
      failures.emplace_back("ustart drift",
                            row_label(exp_row) + ": got=" +
                                (got_row.contains("ustart") ? got_row["ustart"].dump() : "<missing>") +
                                " expected=" +
                                (exp_row.contains("ustart") ? exp_row["ustart"].dump() : "<missing>"));
    }
    for (std::string_view field : {"user", "block", "exo"}) {
      if (field_string(got_row, field) != field_string(exp_row, field))
        add_field_drift("metadata drift", got_row, exp_row, field);
    }
    if (got_row.contains("est") && exp_row.contains("est") &&
        !numeric_equal(got_row, exp_row, "est", 1e-8)) {
      failures.emplace_back("estimate drift",
                            row_label(exp_row) + ": got=" + got_row["est"].dump() +
                                " expected=" + exp_row["est"].dump());
    }
  }

  for (std::size_t gi = 0; gi < g.size(); ++gi) {
    if (used[gi]) continue;
    failures.emplace_back(is_fixed_zero_row(g[gi]) ? "extra fixed-zero row" : "extra row",
                          row_label(g[gi]));
  }

  return summarize_failures(failures);
}

}  // namespace

TEST_CASE("ptable goldens — every corpus entry that lavaanify can handle") {
  const auto corpus = magmaan::test::load_corpus();
  REQUIRE(!corpus.empty());
  const std::string ptable_dir = magmaan::test::fixtures_dir() + "/ptable";

  // Lavaan oddities we deliberately do not mirror, with rationale. These
  // are skipped from the strict pass/fail count but still surface as
  // tagged messages so future drift is visible.
  const std::vector<std::pair<std::string, std::string>> tolerated_divergences = {
      {"0008_defined_param",
       "lavaan emits two identical `:=` rows differing only in plabel "
       "(internal quirk in lavaanify); we emit one."},
      {"0026_two_factor_meanstructure_hs",
       "lavaan's `lavaanify(...)` auto-adds LV α=0 rows when explicit OV "
       "`~1` rows are present; our `lavaanify({})` only adds them when "
       "`opts.meanstructure=true` is passed. The inference goldens pass "
       "the option explicitly; this single-group ptable round-trip uses "
       "the no-options form."},
      {"0023_scalar_invariance_3f_hs",
       "Same auto-α divergence as 0026 (3F variant). The multi-group "
       "inference goldens pass the option and exercise the full scalar-"
       "invariance + meanstructure path; this single-group ptable round-"
       "trip can't."},
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
    auto raw = magmaan::test::read_fixture(path);
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

    auto fp = magmaan::parse::Parser::parse(e.model);
    if (!fp.has_value()) {
      failures.push_back(e.id + ": parse failed — " + fp.error().detail);
      continue;
    }
    magmaan::spec::Starts starts;
    magmaan::spec::LatentNames names;
    auto pt = magmaan::spec::lavaanify(*fp, {}, &starts, &names);
    if (!pt.has_value()) {
      failures.push_back(e.id + ": lavaanify failed — " + pt.error().detail);
      continue;
    }

    auto got = ptable_to_json(e.model, e.id,
                              magmaan::lavaan::to_lavaan_partable(*pt, names, starts));
    auto d = diff_ptable_semantic(magmaan::test::strip_meta(got),
                                  magmaan::test::strip_meta(exp));
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
