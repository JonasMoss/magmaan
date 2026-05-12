#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"

using latva::test::CorpusEntry;
using latva::test::LexerScan;

namespace {

// Iterates the corpus and either compares each fixture to current lexer
// output (default), or rewrites the fixture from current lexer output
// (when LATVA_REGEN_FIXTURES is set in the environment).
//
// Negative fixtures live alongside positive ones under tests/fixtures/lexer/.
// A fixture is "negative" when it carries an `expected_error` field; the
// regen path emits that field automatically when the lexer fails.
TEST_CASE("lexer goldens — every corpus entry round-trips") {
  const auto corpus = latva::test::load_corpus();
  REQUIRE(!corpus.empty());

  const std::string lexer_dir = latva::test::fixtures_dir() + "/lexer";
  const bool regen = latva::test::regen_mode();

  for (const CorpusEntry& e : corpus) {
    CAPTURE(e.id);
    const std::string path = lexer_dir + "/" + e.id + ".tokens.json";

    LexerScan got = latva::test::scan_all(e.model);
    std::optional<latva::ParseError> err = got.error;
    const auto fixture =
        latva::test::format_lexer_fixture(e.id, e.model, got.tokens, err);

    if (regen) {
      const bool ok = latva::test::write_fixture(path, fixture);
      REQUIRE(ok);
      MESSAGE("regenerated " << path);
      continue;
    }

    auto raw = latva::test::read_fixture(path);
    REQUIRE_MESSAGE(raw.has_value(),
                    "missing fixture: " << path
                    << " (run with LATVA_REGEN_FIXTURES=1 to bootstrap)");
    auto j = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    REQUIRE_MESSAGE(!j.is_discarded(), "fixture is not valid JSON: " << path);

    const std::string diff =
        latva::test::diff_lexer_fixture(e.id, e.model, got, j);
    if (!diff.empty()) {
      FAIL(diff);
    }
  }
}

TEST_CASE("lexer negatives — every input produces the expected error kind") {
  const auto cases = latva::test::load_lexer_negatives();
  REQUIRE(!cases.empty());

  const std::string lexer_dir = latva::test::fixtures_dir() + "/lexer/negative";
  const bool regen = latva::test::regen_mode();

  for (const auto& c : cases) {
    CAPTURE(c.id);
    LexerScan got = latva::test::scan_all(c.input);
    REQUIRE_MESSAGE(got.error.has_value(),
                    "expected lexer error of kind '" << c.expected_kind << "', got success");

    // Per-case fixture (carries the exact span). Bootstraps via regen mode.
    const std::string path = lexer_dir + "/" + c.id + ".tokens.json";
    const auto fixture =
        latva::test::format_lexer_fixture(c.id, c.input, got.tokens, got.error);

    if (regen) {
      const bool ok = latva::test::write_fixture(path, fixture);
      REQUIRE(ok);
      MESSAGE("regenerated " << path);
      continue;
    }

    auto raw = latva::test::read_fixture(path);
    REQUIRE_MESSAGE(raw.has_value(),
                    "missing fixture: " << path
                    << " (run with LATVA_REGEN_FIXTURES=1 to bootstrap)");
    auto j = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    REQUIRE_MESSAGE(!j.is_discarded(), "fixture is not valid JSON: " << path);

    const std::string diff =
        latva::test::diff_lexer_fixture(c.id, c.input, got, j);
    if (!diff.empty()) {
      FAIL(diff);
    }
  }
}

}  // namespace
