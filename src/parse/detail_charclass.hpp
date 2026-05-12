#pragma once

// Character predicates used by the lexer. Kept private to src/parse so the
// public surface stays narrow. ASCII-only; non-ASCII bytes are explicitly
// rejected upstream (UnexpectedChar).

namespace latva::parse::detail {

constexpr bool is_digit(char c) noexcept {
  return c >= '0' && c <= '9';
}

constexpr bool is_alpha(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// id_start = [A-Za-z_.]
constexpr bool is_id_start(char c) noexcept {
  return is_alpha(c) || c == '_' || c == '.';
}

// id_cont = [A-Za-z0-9_.]
constexpr bool is_id_cont(char c) noexcept {
  return is_alpha(c) || is_digit(c) || c == '_' || c == '.';
}

// Whitespace that the lexer silently discards between tokens. Newlines are
// NOT in this set; they are significant tokens.
constexpr bool is_inline_ws(char c) noexcept {
  return c == ' ' || c == '\t';
}

}
