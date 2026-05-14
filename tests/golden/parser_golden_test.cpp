#include <doctest/doctest.h>

#include <string>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"

// Compares the FlatPartable our parser produces against lavaan's
// lavParseModelString output, fixture by fixture under tests/fixtures/flat/.
//
// Models we don't yet handle (constraint or define-param surface — P4) are
// SKIPPED here, recognized by their fixture having a non-empty
// "constraints" array and an empty "rows" array. Those models still get
// fixtures generated, so the comparison flips from "skipped" to "compared"
// the moment P4 lands without any fixture-side change.

TEST_CASE("parser flat fixtures match lavaan") {
  const auto corpus = magmaan::test::load_corpus();
  REQUIRE(!corpus.empty());

  const std::string flat_dir = magmaan::test::fixtures_dir() + "/flat";

  for (const auto& e : corpus) {
    CAPTURE(e.id);
    const std::string path = flat_dir + "/" + e.id + ".flat.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE_MESSAGE(raw.has_value(),
                    "missing flat fixture: " << path
                    << " — run `Rscript tools/regen_oracle.R` to regenerate");
    auto exp = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    REQUIRE_MESSAGE(!exp.is_discarded(), "fixture is not valid JSON: " << path);

    auto got = magmaan::parse::Parser::parse(e.model);
    REQUIRE_MESSAGE(got.has_value(),
                    "parser failed on '" << e.model << "': kind="
                    << static_cast<int>(got.error().kind) << " — "
                    << got.error().detail);

    nlohmann::json got_json =
        magmaan::test::flat_partable_to_json(e.model, e.id, *got);

    auto a = magmaan::test::strip_meta(got_json);
    auto b = magmaan::test::strip_meta(exp);
    if (a != b) {
      FAIL("flat fixture mismatch [" << e.id << "]\n"
           << "  got:      " << a.dump(2) << "\n"
           << "  expected: " << b.dump(2));
    }
  }
}
