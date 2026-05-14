#include "magmaan/parse/lexer.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/token.hpp"
#include "magmaan/source_span.hpp"
#include "parse/detail_charclass.hpp"

namespace magmaan::parse {

namespace {

using detail::is_digit;
using detail::is_id_cont;
using detail::is_id_start;
using detail::is_inline_ws;

// Build a single-character span at the given lexer position.
SourceSpan one_char_span(std::uint32_t pos, std::uint32_t line,
                         std::uint32_t col) noexcept {
  return SourceSpan{pos, pos + 1, line, col};
}

// Construct a ParseError. The detail string is owned.
ParseError make_error(ParseError::Kind kind, SourceSpan span,
                      std::string detail) noexcept {
  return ParseError{kind, span, std::move(detail)};
}

}  // namespace

Lexer::Lexer(std::string_view src) noexcept
    : src_(src), pos_(0), line_(1), col_(1) {}

SourceSpan Lexer::here() const noexcept {
  return SourceSpan{pos_, pos_, line_, col_};
}

bool Lexer::at_eof() const noexcept { return pos_ >= src_.size(); }

namespace {

// Helper: a tiny cursor that mutates a Lexer's position safely. Keeping
// the bookkeeping in a single place makes the scan_* routines below
// declarative.
struct Cursor {
  std::string_view src;
  std::uint32_t&   pos;
  std::uint32_t&   line;
  std::uint32_t&   col;

  char peek(std::uint32_t offset = 0) const noexcept {
    const std::uint32_t idx = pos + offset;
    return idx < src.size() ? src[idx] : '\0';
  }

  bool eof() const noexcept { return pos >= src.size(); }

  // Consume one character and update line/col. Does not handle CRLF
  // pairs — newline handling is done in the scanner.
  char advance() noexcept {
    const char c = src[pos];
    ++pos;
    if (c == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    return c;
  }

  SourceSpan span_from(std::uint32_t begin_pos,
                       std::uint32_t begin_line,
                       std::uint32_t begin_col) const noexcept {
    return SourceSpan{begin_pos, pos, begin_line, begin_col};
  }
};

// production: WS  (lexer.md "Whitespace and comments")
//
// Discards inline whitespace and comments. Stops at newlines (significant)
// and at the start of any other token.
void skip_ws_and_comments(Cursor& cur) noexcept {
  for (;;) {
    const char c = cur.peek();
    if (is_inline_ws(c)) {
      cur.advance();
      continue;
    }
    if (c == '#' || c == '!') {
      while (!cur.eof() && cur.peek() != '\n' && cur.peek() != '\r') {
        cur.advance();
      }
      continue;
    }
    return;
  }
}

// production: NEWLINE
parse_expected<Token> scan_newline(Cursor& cur) noexcept {
  const auto begin_pos  = cur.pos;
  const auto begin_line = cur.line;
  const auto begin_col  = cur.col;
  const char c          = cur.advance();
  if (c == '\r' && cur.peek() == '\n') {
    cur.advance();
  }
  return Token{TokenKind::Newline, cur.span_from(begin_pos, begin_line, begin_col),
               cur.src.substr(begin_pos, cur.pos - begin_pos)};
}

// production: identifier  /  NA
parse_expected<Token> scan_identifier(Cursor& cur) noexcept {
  const auto begin_pos  = cur.pos;
  const auto begin_line = cur.line;
  const auto begin_col  = cur.col;
  cur.advance();  // id_start
  while (!cur.eof() && is_id_cont(cur.peek())) {
    cur.advance();
  }
  const auto text = cur.src.substr(begin_pos, cur.pos - begin_pos);
  const auto kind = (text == "NA") ? TokenKind::NA : TokenKind::Identifier;
  return Token{kind, cur.span_from(begin_pos, begin_line, begin_col), text};
}

// production: num_lit
//
// Accepts:   digit+ ('.' digit*)? exponent?
//          | '.' digit+ exponent?
// Rejects (MalformedNumber):
//   - `e` / `E` not followed by digits (after optional sign)
//   - a trailing `.` immediately followed by an id_start character (`1.foo`)
parse_expected<Token> scan_number(Cursor& cur) noexcept {
  const auto begin_pos  = cur.pos;
  const auto begin_line = cur.line;
  const auto begin_col  = cur.col;
  bool saw_digit_before_dot = false;
  bool saw_dot              = false;
  bool saw_digit_after_dot  = false;

  if (cur.peek() == '.') {
    saw_dot = true;
    cur.advance();
    while (is_digit(cur.peek())) {
      saw_digit_after_dot = true;
      cur.advance();
    }
  } else {
    while (is_digit(cur.peek())) {
      saw_digit_before_dot = true;
      cur.advance();
    }
    if (cur.peek() == '.') {
      saw_dot = true;
      cur.advance();
      while (is_digit(cur.peek())) {
        saw_digit_after_dot = true;
        cur.advance();
      }
    }
  }

  // A trailing '.' followed by id_start is the `1.foo` shape: reject loudly.
  if (saw_dot && !saw_digit_after_dot && is_id_start(cur.peek())) {
    return std::unexpected(make_error(
        ParseError::Kind::MalformedNumber,
        cur.span_from(begin_pos, begin_line, begin_col),
        "number followed by identifier character; insert whitespace or '*' to disambiguate"));
  }

  // The leading-`.`-only case requires at least one digit somewhere; bare `.`
  // never reaches this function (the dispatcher only calls scan_number when
  // it has confirmed `digit` or `.digit`).
  (void)saw_digit_before_dot;

  // Optional exponent.
  if (cur.peek() == 'e' || cur.peek() == 'E') {
    const auto e_pos  = cur.pos;
    const auto e_line = cur.line;
    const auto e_col  = cur.col;
    cur.advance();  // consume e/E
    if (cur.peek() == '+' || cur.peek() == '-') {
      cur.advance();
    }
    bool saw_exp_digit = false;
    while (is_digit(cur.peek())) {
      saw_exp_digit = true;
      cur.advance();
    }
    if (!saw_exp_digit) {
      return std::unexpected(make_error(
          ParseError::Kind::MalformedNumber,
          SourceSpan{e_pos, cur.pos, e_line, e_col},
          "exponent has no digits"));
    }
  }

  return Token{TokenKind::NumLit,
               cur.span_from(begin_pos, begin_line, begin_col),
               cur.src.substr(begin_pos, cur.pos - begin_pos)};
}

// production: string_lit
parse_expected<Token> scan_string(Cursor& cur) noexcept {
  const auto begin_pos  = cur.pos;
  const auto begin_line = cur.line;
  const auto begin_col  = cur.col;
  cur.advance();  // opening "
  while (!cur.eof()) {
    const char c = cur.peek();
    if (c == '"') {
      cur.advance();
      return Token{TokenKind::StringLit,
                   cur.span_from(begin_pos, begin_line, begin_col),
                   cur.src.substr(begin_pos, cur.pos - begin_pos)};
    }
    if (c == '\n' || c == '\r') {
      return std::unexpected(make_error(
          ParseError::Kind::UnterminatedString,
          SourceSpan{begin_pos, begin_pos + 1, begin_line, begin_col},
          "string literal not closed before end of line"));
    }
    cur.advance();
  }
  return std::unexpected(make_error(
      ParseError::Kind::UnterminatedString,
      SourceSpan{begin_pos, begin_pos + 1, begin_line, begin_col},
      "string literal not closed before end of input"));
}

// production: operator  /  rejected_operator  /  punctuation
//
// Implements longest-match operator scanning per docs/grammar/lexer.md.
parse_expected<Token> scan_operator_or_punct(Cursor& cur) noexcept {
  const auto begin_pos  = cur.pos;
  const auto begin_line = cur.line;
  const auto begin_col  = cur.col;
  const char c0 = cur.peek(0);
  const char c1 = cur.peek(1);
  const char c2 = cur.peek(2);

  auto emit_op = [&](std::uint32_t consume,
                     TokenKind kind = TokenKind::Op) -> parse_expected<Token> {
    for (std::uint32_t i = 0; i < consume; ++i) cur.advance();
    return Token{kind, cur.span_from(begin_pos, begin_line, begin_col),
                 cur.src.substr(begin_pos, cur.pos - begin_pos)};
  };

  // 3-char rejected operator
  if (c0 == '~' && c1 == '*' && c2 == '~') return emit_op(3);

  // 2-char operators (rejected and accepted alike)
  if (c0 == '<' && c1 == '~') return emit_op(2);
  if (c0 == '|' && c1 == '~') return emit_op(2);
  if (c0 == '~' && c1 == '~') return emit_op(2);
  if (c0 == '=' && c1 == '~') return emit_op(2);
  if (c0 == '=' && c1 == '=') return emit_op(2);
  if (c0 == ':' && c1 == '=') return emit_op(2);

  // 1-char operators
  if (c0 == '~') return emit_op(1);
  if (c0 == '<') return emit_op(1);
  if (c0 == '>') return emit_op(1);
  if (c0 == '|') return emit_op(1);

  // Punctuation
  if (c0 == '+') return emit_op(1, TokenKind::Plus);
  if (c0 == '-') return emit_op(1, TokenKind::Minus);
  if (c0 == '*') return emit_op(1, TokenKind::Star);
  if (c0 == '/') return emit_op(1, TokenKind::Slash);
  if (c0 == '^') return emit_op(1, TokenKind::Caret);
  if (c0 == '?') return emit_op(1, TokenKind::Question);
  if (c0 == ',') return emit_op(1, TokenKind::Comma);
  if (c0 == '(') return emit_op(1, TokenKind::LParen);
  if (c0 == ')') return emit_op(1, TokenKind::RParen);
  if (c0 == ';') return emit_op(1, TokenKind::Semicolon);

  // Lone '=' or any other unexpected punctuation. A lone '=' is a frequent
  // typo for '==' or '=~'; the dedicated UnknownOperator kind makes it
  // discoverable in error messages.
  if (c0 == '=') {
    const auto span = one_char_span(begin_pos, begin_line, begin_col);
    cur.advance();
    return std::unexpected(make_error(
        ParseError::Kind::UnknownOperator, span,
        "lone '=' is not a valid operator; did you mean '==' or '=~'?"));
  }

  const auto span = one_char_span(begin_pos, begin_line, begin_col);
  cur.advance();
  std::string msg = "unexpected character '";
  msg.push_back(c0);
  msg += "'";
  return std::unexpected(make_error(ParseError::Kind::UnexpectedChar, span,
                                    std::move(msg)));
}

}  // namespace

parse_expected<Token> Lexer::next() noexcept {
  Cursor cur{src_, pos_, line_, col_};
  skip_ws_and_comments(cur);
  if (cur.eof()) {
    return Token{TokenKind::EndOfFile,
                 SourceSpan{cur.pos, cur.pos, cur.line, cur.col}, {}};
  }
  const char c = cur.peek();

  if (c == '\n' || c == '\r') {
    return scan_newline(cur);
  }
  // Numeric literal: a digit, or `.` followed by a digit.
  if (is_digit(c) || (c == '.' && is_digit(cur.peek(1)))) {
    return scan_number(cur);
  }
  if (is_id_start(c)) {
    return scan_identifier(cur);
  }
  if (c == '"') {
    return scan_string(cur);
  }
  // Non-ASCII (high bit set) bytes are explicitly UnexpectedChar.
  if (static_cast<unsigned char>(c) >= 0x80) {
    const auto span = one_char_span(cur.pos, cur.line, cur.col);
    cur.advance();
    return std::unexpected(make_error(
        ParseError::Kind::UnexpectedChar, span,
        "non-ASCII byte in model source"));
  }
  return scan_operator_or_punct(cur);
}

}  // namespace magmaan::parse
