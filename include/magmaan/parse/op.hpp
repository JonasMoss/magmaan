#pragma once

#include <cstdint>
#include <string_view>

namespace magmaan::parse {

// Enum mirror of the operators in docs/grammar/grammar.ebnf. Only the v0
// operators have semantic meaning. The lexer emits operator text; the
// parser maps text to this enum.
enum class Op : std::uint8_t {
  Measurement,    // =~
  Regression,     // ~
  Covariance,     // ~~
  Threshold,      // |
  ResponseScale,  // ~*~
  Intercept,      // ~1 / fixed numeric intercept shorthand (synthesized by
                  //     the parser when `~` has a single numeric RHS)
  DefineParam,    // :=  (parsed in P4)
  EqConstraint,   // ==  (parsed in P4)
  LtConstraint,   // <   (parsed in P4)
  GtConstraint,   // >   (parsed in P4)
  Composite,      // <~  (composite / formative). A formula operator like `=~`;
                  //     `spec::build` expands it into a Henseler-Ogasawara
                  //     reflective sub-model before matrix_rep sees the rows.
};

constexpr std::string_view to_string(Op op) noexcept {
  switch (op) {
    case Op::Measurement:   return "=~";
    case Op::Regression:    return "~";
    case Op::Covariance:    return "~~";
    case Op::Threshold:     return "|";
    case Op::ResponseScale: return "~*~";
    case Op::Intercept:     return "~1";
    case Op::DefineParam:   return ":=";
    case Op::EqConstraint:  return "==";
    case Op::LtConstraint:  return "<";
    case Op::GtConstraint:  return ">";
    case Op::Composite:     return "<~";
  }
  return "?";
}

}
