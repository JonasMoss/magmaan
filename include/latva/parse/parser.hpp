#pragma once

#include <string_view>

#include "latva/expected.hpp"
#include "latva/parse/flat_partable.hpp"

namespace latva::parse {

// Hand-written recursive-descent parser. Produces a FlatPartable from a
// model string. See docs/grammar/grammar.ebnf for the normative grammar.
//
// Built with -fno-exceptions; failure is returned as ParseError. The
// returned FlatPartable owns its source bytes via FlatPartable.source_text.
//
// v0 first slice (this header is stable; impl grows phase-by-phase):
//   supported : =~, ~, ~~, ~ 1 (intercept), n*x, lbl*x, v?x modifiers,
//               + with multi-line continuation, ; / Newline statement sep,
//               # / ! comments
//   not yet   : c(...), start(...), NA, "string"-as-label, := ==
//               < > constraints, multi-LHS, rejected-operator diagnostics
class Parser {
 public:
  // production: model
  static parse_expected<FlatPartable> parse(std::string_view src);
};

}
