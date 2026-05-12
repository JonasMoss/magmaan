#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "latva/error.hpp"
#include "latva/parse/lexer.hpp"
#include "latva/parse/token.hpp"

using latva::ParseError;
using latva::SourceSpan;
using latva::parse::Lexer;
using latva::parse::Token;
using latva::parse::TokenKind;

namespace {

// Run the lexer to completion. Returns either the full token stream
// (including the terminating EndOfFile) or the first error.
struct ScanResult {
  std::vector<Token> tokens;
  std::optional<ParseError> error;
};

ScanResult scan_all(std::string_view src) {
  Lexer lex(src);
  ScanResult out;
  for (;;) {
    auto t = lex.next();
    if (!t.has_value()) {
      out.error = t.error();
      return out;
    }
    out.tokens.push_back(*t);
    if (t->kind == TokenKind::EndOfFile) return out;
  }
}

// Convenience: produce just the kinds, ignoring spans/text.
std::vector<TokenKind> kinds_of(const std::vector<Token>& ts) {
  std::vector<TokenKind> out;
  out.reserve(ts.size());
  for (const auto& t : ts) out.push_back(t.kind);
  return out;
}

}  // namespace

TEST_CASE("empty input yields a single EndOfFile token") {
  auto r = scan_all("");
  REQUIRE_FALSE(r.error.has_value());
  REQUIRE(r.tokens.size() == 1);
  CHECK(r.tokens[0].kind == TokenKind::EndOfFile);
  CHECK(r.tokens[0].span == SourceSpan{0, 0, 1, 1});
}

TEST_CASE("identifiers, numbers, and NA") {
  auto r = scan_all("foo 12 .5 NA bar.baz");
  REQUIRE_FALSE(r.error.has_value());
  CHECK(kinds_of(r.tokens) == std::vector<TokenKind>{
                                  TokenKind::Identifier, TokenKind::NumLit,
                                  TokenKind::NumLit, TokenKind::NA,
                                  TokenKind::Identifier, TokenKind::EndOfFile});
  CHECK(r.tokens[0].text == "foo");
  CHECK(r.tokens[1].text == "12");
  CHECK(r.tokens[2].text == ".5");
  CHECK(r.tokens[3].text == "NA");
  CHECK(r.tokens[4].text == "bar.baz");
}

TEST_CASE("numeric literal forms") {
  SUBCASE("integer") {
    auto r = scan_all("42");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "42");
  }
  SUBCASE("trailing dot") {
    auto r = scan_all("5.");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "5.");
  }
  SUBCASE("leading dot") {
    auto r = scan_all(".25");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == ".25");
  }
  SUBCASE("exponent") {
    auto r = scan_all("1.5e-3");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "1.5e-3");
  }
  SUBCASE("uppercase exponent") {
    auto r = scan_all("2E10");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "2E10");
  }
}

TEST_CASE("string literal") {
  auto r = scan_all("\"my label\"");
  REQUIRE_FALSE(r.error.has_value());
  CHECK(r.tokens[0].kind == TokenKind::StringLit);
  CHECK(r.tokens[0].text == "\"my label\"");
}

TEST_CASE("operators are longest-match") {
  SUBCASE("=~ wins over =") {
    auto r = scan_all("=~");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].kind == TokenKind::Op);
    CHECK(r.tokens[0].text == "=~");
  }
  SUBCASE("~~ wins over ~") {
    auto r = scan_all("~~");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "~~");
  }
  SUBCASE("~*~ wins over ~ * ~") {
    auto r = scan_all("~*~");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "~*~");
  }
  SUBCASE("== and := are 2-char ops") {
    auto r = scan_all("== :=");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "==");
    CHECK(r.tokens[1].text == ":=");
  }
}

TEST_CASE("rejected operators tokenize as Op (parser raises later)") {
  auto r = scan_all("|~ <~ ~*~ |");
  REQUIRE_FALSE(r.error.has_value());
  CHECK(r.tokens[0].text == "|~");
  CHECK(r.tokens[1].text == "<~");
  CHECK(r.tokens[2].text == "~*~");
  CHECK(r.tokens[3].text == "|");
  for (std::size_t i = 0; i < 4; ++i) CHECK(r.tokens[i].kind == TokenKind::Op);
}

TEST_CASE("punctuation kinds") {
  auto r = scan_all("+ * ? , ( ) ;");
  REQUIRE_FALSE(r.error.has_value());
  CHECK(kinds_of(r.tokens) == std::vector<TokenKind>{
                                  TokenKind::Plus, TokenKind::Star,
                                  TokenKind::Question, TokenKind::Comma,
                                  TokenKind::LParen, TokenKind::RParen,
                                  TokenKind::Semicolon, TokenKind::EndOfFile});
}

TEST_CASE("comments are stripped, newlines are kept") {
  SUBCASE("hash comment") {
    auto r = scan_all("foo # this is a comment\nbar");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(kinds_of(r.tokens) == std::vector<TokenKind>{
                                    TokenKind::Identifier, TokenKind::Newline,
                                    TokenKind::Identifier, TokenKind::EndOfFile});
  }
  SUBCASE("bang comment") {
    auto r = scan_all("foo ! also a comment\nbar");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[0].text == "foo");
    CHECK(r.tokens[1].kind == TokenKind::Newline);
    CHECK(r.tokens[2].text == "bar");
  }
}

TEST_CASE("CRLF and bare CR both produce a single Newline token") {
  SUBCASE("CRLF") {
    auto r = scan_all("a\r\nb");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[1].kind == TokenKind::Newline);
    CHECK(r.tokens[1].text == "\r\n");
  }
  SUBCASE("bare CR") {
    auto r = scan_all("a\rb");
    REQUIRE_FALSE(r.error.has_value());
    CHECK(r.tokens[1].kind == TokenKind::Newline);
    CHECK(r.tokens[1].text == "\r");
  }
}

TEST_CASE("source spans track lines and columns") {
  auto r = scan_all("foo\n  bar");
  REQUIRE_FALSE(r.error.has_value());
  CHECK(r.tokens[0].span == SourceSpan{0, 3, 1, 1});
  CHECK(r.tokens[1].kind == TokenKind::Newline);
  CHECK(r.tokens[1].span.line == 1);
  // After the newline we should be on line 2.
  CHECK(r.tokens[2].span.line == 2);
  CHECK(r.tokens[2].span.col == 3);  // two leading spaces
  CHECK(r.tokens[2].text == "bar");
}

TEST_CASE("a representative formula tokenizes cleanly") {
  // f =~ 1*x1 + 0.5?x2 + a*x3 ; y ~ 1
  auto r = scan_all("f =~ 1*x1 + 0.5?x2 + a*x3 ; y ~ 1");
  REQUIRE_FALSE(r.error.has_value());
  // Spot-check key kinds and texts.
  CHECK(r.tokens[0].text == "f");
  CHECK(r.tokens[1].text == "=~");
  CHECK(r.tokens[2].text == "1");
  CHECK(r.tokens[3].kind == TokenKind::Star);
  CHECK(r.tokens[4].text == "x1");
  CHECK(r.tokens[5].kind == TokenKind::Plus);
  CHECK(r.tokens[6].text == "0.5");
  CHECK(r.tokens[7].kind == TokenKind::Question);
  CHECK(r.tokens[8].text == "x2");
  CHECK(r.tokens.back().kind == TokenKind::EndOfFile);
}

TEST_CASE("error: lone '=' is UnknownOperator") {
  auto r = scan_all("a = b");
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::UnknownOperator);
  CHECK(r.error->span.line == 1);
  CHECK(r.error->span.col == 3);
}

TEST_CASE("error: unexpected character") {
  auto r = scan_all("foo & bar");
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::UnexpectedChar);
  CHECK(r.error->span == SourceSpan{4, 5, 1, 5});
}

TEST_CASE("error: non-ASCII byte") {
  auto r = scan_all("foo \xc3\xa9 bar");  // UTF-8 'é'
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::UnexpectedChar);
}

TEST_CASE("error: unterminated string at EOF") {
  auto r = scan_all("\"open");
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::UnterminatedString);
  CHECK(r.error->span == SourceSpan{0, 1, 1, 1});
}

TEST_CASE("error: unterminated string at newline") {
  auto r = scan_all("\"open\nbroken\"");
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::UnterminatedString);
  CHECK(r.error->span.line == 1);
  CHECK(r.error->span.col == 1);
}

TEST_CASE("error: malformed number — exponent with no digits") {
  auto r = scan_all("1e");
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::MalformedNumber);
}

TEST_CASE("error: malformed number — `1.foo`") {
  auto r = scan_all("1.foo");
  REQUIRE(r.error.has_value());
  CHECK(r.error->kind == ParseError::Kind::MalformedNumber);
}

TEST_CASE("comments at end of input do not error") {
  auto r = scan_all("a # trailing comment with no newline");
  REQUIRE_FALSE(r.error.has_value());
  CHECK(r.tokens[0].text == "a");
  CHECK(r.tokens.back().kind == TokenKind::EndOfFile);
}

TEST_CASE("multiple consecutive newlines emit multiple tokens") {
  // The parser collapses runs of newlines; the lexer just emits one each.
  auto r = scan_all("a\n\n\nb");
  REQUIRE_FALSE(r.error.has_value());
  int newlines = 0;
  for (const auto& t : r.tokens) {
    if (t.kind == TokenKind::Newline) ++newlines;
  }
  CHECK(newlines == 3);
}

TEST_CASE("at_eof flips after EndOfFile is emitted") {
  Lexer lex("x");
  CHECK_FALSE(lex.at_eof());
  auto t1 = lex.next();
  REQUIRE(t1.has_value());
  CHECK(t1->kind == TokenKind::Identifier);
  auto t2 = lex.next();
  REQUIRE(t2.has_value());
  CHECK(t2->kind == TokenKind::EndOfFile);
  CHECK(lex.at_eof());
}
