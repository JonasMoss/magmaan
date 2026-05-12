#pragma once

#include <cstdint>
#include <string_view>

namespace latva::parse {

// Enum mirror of the operators in docs/grammar/grammar.ebnf. Only the v0
// operators have semantic meaning. The lexer emits operator text; the
// parser maps text to this enum.
enum class Op : std::uint8_t {
  Measurement,    // =~
  Regression,     // ~
  Covariance,     // ~~
  Intercept,      // ~1 (synthesized by the parser when `~` is followed by
                  //     a single `1` term with no modifier)
  DefineParam,    // :=  (parsed in P4)
  EqConstraint,   // ==  (parsed in P4)
  LtConstraint,   // <   (parsed in P4)
  GtConstraint,   // >   (parsed in P4)
};

constexpr std::string_view to_string(Op op) noexcept {
  switch (op) {
    case Op::Measurement:   return "=~";
    case Op::Regression:    return "~";
    case Op::Covariance:    return "~~";
    case Op::Intercept:     return "~1";
    case Op::DefineParam:   return ":=";
    case Op::EqConstraint:  return "==";
    case Op::LtConstraint:  return "<";
    case Op::GtConstraint:  return ">";
  }
  return "?";
}

}
