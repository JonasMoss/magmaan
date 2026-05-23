#include <doctest/doctest.h>

#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"

namespace {

int count_status(const nlohmann::json &nodes, const std::string &field,
                 const std::string &value) {
  int out = 0;
  for (const auto &node : nodes) {
    if (node.contains(field) && node[field].get<std::string>() == value) {
      ++out;
    }
  }
  return out;
}

bool has_key_recursive(const nlohmann::json &j, const std::string &key) {
  if (j.is_object()) {
    if (j.contains(key))
      return true;
    for (const auto &item : j.items()) {
      if (has_key_recursive(item.value(), key))
        return true;
    }
  } else if (j.is_array()) {
    for (const auto &item : j) {
      if (has_key_recursive(item, key))
        return true;
    }
  }
  return false;
}

} // namespace

TEST_CASE("Paper corpus scout manifest is well formed") {
  const std::string root = magmaan::test::fixtures_dir();
  auto raw =
      magmaan::test::read_fixture(root + "/paper_corpus/scout_manifest.json");
  REQUIRE(raw.has_value());

  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("_meta"));
  CHECK(j["_meta"]["corpus_id"].get<std::string>() ==
        "magmaan_paper_corpus_scout_v1");
  CHECK(j["_meta"]["fixture_kind"].get<std::string>() ==
        "paper_corpus.scout_manifest");
  CHECK(j["_meta"]["tool"].get<std::string>() ==
        "tests/tools/scout_paper_corpus.R");

  REQUIRE(j.contains("counts"));
  REQUIRE(j.contains("nodes"));
  CHECK(j["nodes"].size() ==
        static_cast<std::size_t>(j["counts"]["seed_nodes"].get<int>()));
  CHECK(j["counts"]["scouted"].get<int>() ==
        static_cast<int>(j["nodes"].size()));
  CHECK(j["counts"]["promote_first"].get<int>() == 2);
  CHECK(j["counts"]["candidate_code_files"].get<int>() > 0);
  CHECK(j["counts"]["candidate_data_files"].get<int>() > 0);
  CHECK(j["counts"]["scanned_code_files"].get<int>() > 0);
  CHECK(j["counts"]["detected_lavaan_calls"].get<int>() > 0);

  std::unordered_set<std::string> ids;
  for (const auto &node : j["nodes"]) {
    REQUIRE(node.contains("node_id"));
    REQUIRE(node.contains("promotion_status"));
    REQUIRE(node.contains("summary"));
    REQUIRE(node.contains("files"));
    ids.insert(node["node_id"].get<std::string>());

    for (const auto &file : node["files"]) {
      REQUIRE(file.contains("name"));
      REQUIRE(file.contains("file_class"));
      REQUIRE(file.contains("scan_status"));
      if (file["scan_status"].get<std::string>() == "scanned") {
        REQUIRE(file.contains("signals"));
      }
    }
  }

  CHECK(ids.contains("hwkem"));
  CHECK(ids.contains("zxqvn"));
  CHECK(count_status(j["nodes"], "promotion_status", "promote_first") == 2);
  CHECK_FALSE(has_key_recursive(j, "content"));
}
