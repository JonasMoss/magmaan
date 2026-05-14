#include "magmaan/parse/parser.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/lexer.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/token.hpp"
#include "magmaan/source_span.hpp"

namespace magmaan::parse {

namespace {

// === Parser state ============================================================
// Pre-tokenizes the source so parse-time backtracking is just a position
// restore. The Lexer's per-token errors surface up front, with the same
// SourceSpan that the parser would later attach.

struct State {
  std::vector<Token> tokens;
  std::size_t        pos = 0;

  const Token& peek(std::size_t offset = 0) const noexcept {
    const std::size_t i = pos + offset;
    return i < tokens.size() ? tokens[i] : tokens.back();
  }
  const Token& consume() noexcept { return tokens[pos++]; }
  bool at_eof() const noexcept {
    return tokens[pos].kind == TokenKind::EndOfFile;
  }

  // Index of the next non-Newline token (used to look across multi-line
  // continuation). Does not consume.
  std::size_t pos_skipping_newlines() const noexcept {
    std::size_t i = pos;
    while (i < tokens.size() && tokens[i].kind == TokenKind::Newline) ++i;
    return i;
  }
  void skip_newlines() noexcept {
    while (peek().kind == TokenKind::Newline) ++pos;
  }
};

ParseError make_err(ParseError::Kind kind, SourceSpan span,
                    std::string detail) noexcept {
  return ParseError{kind, span, std::move(detail)};
}

// production: TOKEN-STREAM (drains the lexer into a vector)
parse_expected<std::vector<Token>> tokenize(std::string_view src) noexcept {
  std::vector<Token> out;
  Lexer lex(src);
  for (;;) {
    auto t = lex.next();
    if (!t.has_value()) return std::unexpected(t.error());
    out.push_back(*t);
    if (t->kind == TokenKind::EndOfFile) return out;
  }
}

// === Operator text → Op enum ================================================
//
// Returns std::nullopt for tokens whose text is not a recognized operator
// (or is a deferred operator). Caller decides whether that's a parse error
// or an "unsupported in this slice" error.
std::optional<Op> op_from_text(std::string_view t) noexcept {
  if (t == "=~") return Op::Measurement;
  if (t == "~~") return Op::Covariance;
  if (t == "~")  return Op::Regression;
  if (t == ":=") return Op::DefineParam;
  if (t == "==") return Op::EqConstraint;
  if (t == "<")  return Op::LtConstraint;
  if (t == ">")  return Op::GtConstraint;
  return std::nullopt;
}

bool is_rejected_op(std::string_view t) noexcept {
  return t == "<~" || t == "|~" || t == "~*~" || t == "|";
}

// Parse a numeric literal token's text into a double. The lexer guarantees
// shape; this should always succeed.
double parse_double(std::string_view text) noexcept {
  double v = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), v);
  return v;
}

// === RHS term scratch ========================================================
// Internal-only result of parsing one RHS term. The formula-level code
// decides whether a bare-NumLit term is an intercept (`y ~ 1`) or an error.

struct RawRhsTerm {
  std::uint32_t    mod_idx   = 0;     // 0 = none
  std::string_view ident_text;        // for identifier terms
  bool             is_numlit_one = false;  // true iff the lone term `1`
  SourceSpan       span      = {};
};

// Strip the surrounding double quotes from a StringLit token's text.
// The lexer guarantees a leading '"' and a closing '"' for StringLit.
std::string_view strip_quotes(std::string_view quoted) noexcept {
  if (quoted.size() >= 2 && quoted.front() == '"' && quoted.back() == '"') {
    return quoted.substr(1, quoted.size() - 2);
  }
  return quoted;
}

// production: modifier_atom
//
//   modifier_atom ::= num_lit | NA | identifier | string_lit
parse_expected<ModifierAtom> parse_modifier_atom(State& st) noexcept {
  const Token& t = st.peek();
  if (t.kind == TokenKind::NumLit) {
    st.consume();
    return ModifierAtom{FixedValue{parse_double(t.text)}};
  }
  if (t.kind == TokenKind::NA) {
    st.consume();
    return ModifierAtom{Free{}};
  }
  if (t.kind == TokenKind::Identifier) {
    st.consume();
    return ModifierAtom{Label{t.text}};
  }
  if (t.kind == TokenKind::StringLit) {
    st.consume();
    return ModifierAtom{Label{strip_quotes(t.text)}};
  }
  return std::unexpected(make_err(
      ParseError::Kind::ModifierEvalFailed, t.span,
      std::string("expected a modifier atom (number, NA, identifier, or "
                  "string literal), got ") +
          std::string(to_string(t.kind))));
}

// production: start_call
//
//   start_call ::= 'start' '(' num_lit ')'
//
// Caller has already verified that peek(0) is Identifier "start" and
// peek(1) is LParen.
parse_expected<Modifier> parse_start_call(State& st) noexcept {
  st.consume();   // Identifier "start"
  st.consume();   // LParen
  const Token& v = st.peek();
  if (v.kind != TokenKind::NumLit) {
    return std::unexpected(make_err(
        ParseError::Kind::ModifierEvalFailed, v.span,
        "start(...) requires a numeric literal argument"));
  }
  const double value = parse_double(v.text);
  st.consume();
  if (st.peek().kind != TokenKind::RParen) {
    return std::unexpected(make_err(
        ParseError::Kind::ModifierEvalFailed, st.peek().span,
        "expected ')' after start(...) argument"));
  }
  st.consume();
  return Modifier{StartValue{value}};
}

// production: group_vec
//
//   group_vec ::= 'c' '(' modifier_atom (',' modifier_atom)* ')'
//
// Caller has already verified that peek(0) is Identifier "c" and peek(1)
// is LParen.
parse_expected<Modifier> parse_group_vec(State& st) noexcept {
  st.consume();   // Identifier "c"
  st.consume();   // LParen
  GroupVec gv;
  // First atom is required; empty c() is rejected.
  if (st.peek().kind == TokenKind::RParen) {
    return std::unexpected(make_err(
        ParseError::Kind::GroupVecMismatch, st.peek().span,
        "c(...) modifier requires at least one entry"));
  }
  for (;;) {
    auto atom_or = parse_modifier_atom(st);
    if (!atom_or.has_value()) return std::unexpected(atom_or.error());
    gv.per_group.push_back(std::move(*atom_or));
    if (st.peek().kind == TokenKind::Comma) {
      st.consume();
      continue;
    }
    break;
  }
  if (st.peek().kind != TokenKind::RParen) {
    return std::unexpected(make_err(
        ParseError::Kind::GroupVecMismatch, st.peek().span,
        "expected ',' or ')' in c(...) modifier"));
  }
  st.consume();
  return Modifier{std::move(gv)};
}

// production: modifier
//
//   modifier ::= modifier_atom | group_vec | start_call
//
// Decides between the three by lookahead on peek(0)/peek(1). Caller is
// responsible for confirming the next-after-modifier token is `*` or `?`.
parse_expected<Modifier> parse_modifier(State& st) noexcept {
  const Token& head  = st.peek(0);
  const Token& after = st.peek(1);
  if (head.kind == TokenKind::Identifier && after.kind == TokenKind::LParen) {
    if (head.text == "c")     return parse_group_vec(st);
    if (head.text == "start") return parse_start_call(st);
    // Any other identifier(...) form is a modifier-evaluator failure: only
    // c() and start() are defined modifier functions in v0.
    return std::unexpected(make_err(
        ParseError::Kind::ModifierEvalFailed, head.span,
        std::string("unknown modifier function: '") + std::string(head.text) +
            "(...)' (only c(...) and start(...) are recognized)"));
  }
  auto atom_or = parse_modifier_atom(st);
  if (!atom_or.has_value()) return std::unexpected(atom_or.error());
  return std::visit(
      [](auto&& a) -> Modifier { return Modifier{std::forward<decltype(a)>(a)}; },
      std::move(*atom_or));
}

// True if peek(0)/peek(1) commit us to the modifier-prefixed form. Used by
// parse_rhs_term_with_mod to decide between modifier+identifier and the
// bare-identifier shapes without backtracking.
//
// Any Identifier immediately followed by '(' commits to the modifier path,
// even for unknown function names — parse_modifier raises a precise
// "unknown modifier function" error rather than letting the parser drift
// into a generic "expected statement separator" failure.
bool starts_with_modifier(const State& st) noexcept {
  const Token& h = st.peek(0);
  const Token& a = st.peek(1);
  if (h.kind == TokenKind::Identifier && a.kind == TokenKind::LParen) {
    return true;
  }
  // Atom modifier: head atom token immediately followed by Star/Question.
  const bool atom_head =
      (h.kind == TokenKind::NumLit || h.kind == TokenKind::NA ||
       h.kind == TokenKind::StringLit || h.kind == TokenKind::Identifier);
  const bool sep_after =
      (a.kind == TokenKind::Star || a.kind == TokenKind::Question);
  return atom_head && sep_after;
}

// Result of parsing one RHS term, with the parsed modifier (if any).
struct ParsedRhsTerm {
  RawRhsTerm              term;
  std::optional<Modifier> modifier;
};

// production: rhs_term
//
//   rhs_term ::= modifier modifier_sep identifier
//              | identifier
//              | num_lit                   (only valid as intercept; checked above)
//
// The lookahead in starts_with_modifier() decides which branch we take
// without backtracking.
parse_expected<ParsedRhsTerm> parse_rhs_term_with_mod(State& st) noexcept {
  const Token& head = st.peek();

  if (starts_with_modifier(st)) {
    RawRhsTerm out;
    out.span = head.span;
    auto mod_or = parse_modifier(st);
    if (!mod_or.has_value()) return std::unexpected(mod_or.error());
    const Token& sep = st.peek();
    if (sep.kind != TokenKind::Star && sep.kind != TokenKind::Question) {
      return std::unexpected(make_err(
          ParseError::Kind::ExpectedRhsTerm, sep.span,
          std::string("expected '*' or '?' after modifier, got ") +
              std::string(to_string(sep.kind))));
    }
    st.consume();
    Modifier mod;
    std::swap(mod, *mod_or);
    if (sep.kind == TokenKind::Question) {
      // ? forces start-value interpretation. The atom must be a plain
      // FixedValue — labels, free, c(...), and start(...) are nonsensical
      // as start-value shorthands.
      auto* fixed = std::get_if<FixedValue>(&mod);
      if (fixed == nullptr) {
        return std::unexpected(make_err(
            ParseError::Kind::ModifierEvalFailed, sep.span,
            "'?' separator requires a numeric start value on the left"));
      }
      mod = Modifier{StartValue{fixed->value}};
    }
    // Allow `modifier*1` for label-on-intercept (`x1 ~ n1*1`) — the upstream
    // formula path checks `op==Regression && is_numlit_one && modifier.has_value()`
    // and emits an Op::Intercept row with the modifier preserved.
    if (st.peek().kind == TokenKind::NumLit && st.peek().text == "1") {
      const Token& one = st.consume();
      out.is_numlit_one = true;
      out.span = SourceSpan{out.span.begin, one.span.end,
                            out.span.line, out.span.col};
      return ParsedRhsTerm{out, std::optional<Modifier>{std::move(mod)}};
    }
    if (st.peek().kind != TokenKind::Identifier) {
      return std::unexpected(make_err(
          ParseError::Kind::ExpectedRhsTerm, st.peek().span,
          "expected an identifier after modifier"));
    }
    const Token& id = st.consume();
    out.ident_text = id.text;
    out.span = SourceSpan{out.span.begin, id.span.end,
                          out.span.line, out.span.col};
    return ParsedRhsTerm{out, std::optional<Modifier>{std::move(mod)}};
  }

  RawRhsTerm out;
  out.span = head.span;
  if (head.kind == TokenKind::Identifier) {
    st.consume();
    out.ident_text = head.text;
    return ParsedRhsTerm{out, std::nullopt};
  }
  if (head.kind == TokenKind::NumLit && head.text == "1") {
    st.consume();
    out.is_numlit_one = true;
    return ParsedRhsTerm{out, std::nullopt};
  }
  if (head.kind == TokenKind::NumLit) {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedRhsTerm, head.span,
        "bare numeric literal is only valid as the intercept form `~ 1`"));
  }
  return std::unexpected(make_err(
      ParseError::Kind::ExpectedRhsTerm, head.span,
      std::string("expected RHS term, got ") +
          std::string(to_string(head.kind))));
}

// production: rhs_list
//
// One or more rhs_terms separated by `+`. The `+` may sit on either side
// of a Newline run (multi-line continuation). The list ends when the next
// non-Newline token is not `+`.
parse_expected<std::vector<ParsedRhsTerm>>
parse_rhs_list(State& st) noexcept {
  std::vector<ParsedRhsTerm> out;
  auto first = parse_rhs_term_with_mod(st);
  if (!first.has_value()) return std::unexpected(first.error());
  out.push_back(std::move(*first));

  for (;;) {
    const std::size_t saved = st.pos;
    st.skip_newlines();
    if (st.peek().kind != TokenKind::Plus) {
      st.pos = saved;   // backtrack; newlines belong to the next stmt
      break;
    }
    st.consume();          // consume Plus
    st.skip_newlines();    // allow `+ \n term`
    auto next = parse_rhs_term_with_mod(st);
    if (!next.has_value()) return std::unexpected(next.error());
    out.push_back(std::move(*next));
  }
  return out;
}

// === Pratt expression parser ================================================
//
// Implements docs/grammar/grammar.ebnf:
//   expr   ::= term (('+' | '-') term)*
//   term   ::= factor (('*' | '/') factor)*
//   factor ::= power
//   power  ::= unary ('^' power)?
//   unary  ::= ('+' | '-') unary | primary
//   primary::= num_lit | identifier | '(' expr ')'
//
// Left-associative for + - * /; right-associative for ^.
// No function calls in v0 (a methods developer wanting them must extend
// the AST, which is an ABI break).

struct BindingPower { std::uint8_t lhs; std::uint8_t rhs; };

// Returns std::nullopt if the token isn't a binary operator in expressions.
std::optional<std::pair<BinOp, BindingPower>>
expr_binop_for(TokenKind k) noexcept {
  switch (k) {
    case TokenKind::Plus:  return std::pair{BinOp::Add, BindingPower{1, 2}};
    case TokenKind::Minus: return std::pair{BinOp::Sub, BindingPower{1, 2}};
    case TokenKind::Star:  return std::pair{BinOp::Mul, BindingPower{3, 4}};
    case TokenKind::Slash: return std::pair{BinOp::Div, BindingPower{3, 4}};
    // Right-associative: rhs binding power lower than lhs forces nesting on the right.
    case TokenKind::Caret: return std::pair{BinOp::Pow, BindingPower{6, 5}};
    default: return std::nullopt;
  }
}

constexpr std::uint8_t kUnaryBp = 5;  // between * and ^

parse_expected<Expr> parse_expr(State& st, std::uint8_t min_bp) noexcept;

// production: primary  /  unary
parse_expected<Expr> parse_unary_or_primary(State& st) noexcept {
  const Token& t = st.peek();
  if (t.kind == TokenKind::Plus || t.kind == TokenKind::Minus) {
    const UnOp op = (t.kind == TokenKind::Minus) ? UnOp::Neg : UnOp::Pos;
    st.consume();
    auto inner = parse_expr(st, kUnaryBp);
    if (!inner.has_value()) return std::unexpected(inner.error());
    return Expr{UnNode{op, std::make_unique<Expr>(std::move(*inner))}};
  }
  if (t.kind == TokenKind::NumLit) {
    st.consume();
    return Expr{Num{parse_double(t.text)}};
  }
  if (t.kind == TokenKind::Identifier) {
    st.consume();
    return Expr{Param{t.text}};
  }
  if (t.kind == TokenKind::LParen) {
    st.consume();
    auto inner = parse_expr(st, 0);
    if (!inner.has_value()) return std::unexpected(inner.error());
    if (st.peek().kind != TokenKind::RParen) {
      return std::unexpected(make_err(
          ParseError::Kind::ExpectedRhsTerm, st.peek().span,
          "expected ')' to close parenthesized expression"));
    }
    st.consume();
    return inner;
  }
  return std::unexpected(make_err(
      ParseError::Kind::ExpectedRhsTerm, t.span,
      std::string("expected expression atom (number, identifier, "
                  "parenthesized expr, or unary +/-), got ") +
          std::string(to_string(t.kind))));
}

// production: expr  (precedence-climbing form)
parse_expected<Expr> parse_expr(State& st, std::uint8_t min_bp) noexcept {
  auto lhs_or = parse_unary_or_primary(st);
  if (!lhs_or.has_value()) return std::unexpected(lhs_or.error());
  Expr lhs;
  std::swap(lhs, *lhs_or);

  for (;;) {
    const auto bop = expr_binop_for(st.peek().kind);
    if (!bop.has_value()) break;
    const auto [op, bp] = *bop;
    if (bp.lhs < min_bp) break;
    st.consume();   // the operator token
    auto rhs_or = parse_expr(st, bp.rhs);
    if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());
    lhs = Expr{BinNode{op,
                       std::make_unique<Expr>(std::move(lhs)),
                       std::make_unique<Expr>(std::move(*rhs_or))}};
  }
  return lhs;
}

// True if the next non-Newline token in the statement is a constraint or
// define-param operator. Used by parse_statement to dispatch between
// formula / constraint / define forms before committing to either path.
//
// Scans ahead until it hits a top-level Op token, a statement terminator
// (Newline / Semicolon / EndOfFile), or runs out of tokens. LParens
// increase paren depth so we don't mistake a c(... == ...) — though no
// such modifier exists in v0, the discipline is right.
struct StatementShape {
  enum class Kind { Formula, Constraint, Define, Unknown };
  Kind         kind = Kind::Unknown;
  std::size_t  op_pos = 0;        // index of the dispatching Op token
  Op           op = Op::Measurement;
  std::string_view op_text;
  SourceSpan   op_span;
};

StatementShape classify_statement(const State& st) noexcept {
  StatementShape out;
  std::size_t i = st.pos;
  int paren_depth = 0;
  while (i < st.tokens.size()) {
    const Token& t = st.tokens[i];
    if (paren_depth == 0) {
      if (t.kind == TokenKind::Newline ||
          t.kind == TokenKind::Semicolon ||
          t.kind == TokenKind::EndOfFile) {
        return out;
      }
      if (t.kind == TokenKind::Op) {
        const auto op_opt = op_from_text(t.text);
        if (!op_opt.has_value()) {
          // Could be a rejected operator (handled later); leave as Unknown.
          return out;
        }
        out.op_pos  = i;
        out.op      = *op_opt;
        out.op_text = t.text;
        out.op_span = t.span;
        switch (*op_opt) {
          case Op::Measurement:
          case Op::Regression:
          case Op::Covariance:
            out.kind = StatementShape::Kind::Formula; break;
          case Op::EqConstraint:
          case Op::LtConstraint:
          case Op::GtConstraint:
            out.kind = StatementShape::Kind::Constraint; break;
          case Op::DefineParam:
            out.kind = StatementShape::Kind::Define; break;
          case Op::Intercept:
            // Intercept is parser-synthesized; not lexed as a single op.
            break;
        }
        return out;
      }
    }
    if (t.kind == TokenKind::LParen) ++paren_depth;
    if (t.kind == TokenKind::RParen && paren_depth > 0) --paren_depth;
    ++i;
  }
  return out;
}

// production: constraint
//
//   constraint ::= expr constraint_op expr
parse_expected<void>
parse_constraint(State& st, FlatPartable& flat,
                 const StatementShape& shape) noexcept {
  auto lhs_or = parse_expr(st, 0);
  if (!lhs_or.has_value()) return std::unexpected(lhs_or.error());
  // Sanity: we should be at the constraint op now.
  if (st.peek().kind != TokenKind::Op || st.peek().text != shape.op_text) {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedOperator, st.peek().span,
        std::string("expected constraint operator '") +
            std::string(shape.op_text) + "' here"));
  }
  st.consume();
  auto rhs_or = parse_expr(st, 0);
  if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());

  ConstraintKind k = ConstraintKind::Eq;
  if (shape.op == Op::LtConstraint) k = ConstraintKind::Lt;
  if (shape.op == Op::GtConstraint) k = ConstraintKind::Gt;

  Constraint c{k, std::string_view{}, std::move(*lhs_or), std::move(*rhs_or),
               shape.op_span};
  flat.constraints.push_back(std::move(c));
  return {};
}

// production: define_param
//
//   define_param ::= identifier ':=' expr
parse_expected<void>
parse_define(State& st, FlatPartable& flat,
             const StatementShape& shape) noexcept {
  const Token& name_tok = st.peek();
  if (name_tok.kind != TokenKind::Identifier) {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedLhs, name_tok.span,
        ":= requires a single identifier on the left"));
  }
  st.consume();
  if (st.peek().kind != TokenKind::Op || st.peek().text != ":=") {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedOperator, st.peek().span,
        ":= requires a single identifier on the left"));
  }
  st.consume();   // the := op
  auto rhs_or = parse_expr(st, 0);
  if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());
  Constraint c{ConstraintKind::Define, name_tok.text, Expr{Num{}},
               std::move(*rhs_or), shape.op_span};
  flat.constraints.push_back(std::move(c));
  return {};
}

// production: formula
//
//   formula ::= identifier operator rhs_list
//
// Multi-LHS (`y1 + y2 ~ x`) is a future slice.
parse_expected<void>
parse_formula(State& st, FlatPartable& flat) noexcept {
  const Token& lhs_tok = st.peek();
  if (lhs_tok.kind != TokenKind::Identifier) {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedLhs, lhs_tok.span,
        std::string("expected identifier on LHS, got ") +
            std::string(to_string(lhs_tok.kind))));
  }
  st.consume();
  const std::string_view lhs_text = lhs_tok.text;

  const Token& op_tok = st.peek();
  if (op_tok.kind != TokenKind::Op) {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedOperator, op_tok.span,
        std::string("expected operator, got ") +
            std::string(to_string(op_tok.kind))));
  }
  if (is_rejected_op(op_tok.text)) {
    return std::unexpected(make_err(
        ParseError::Kind::UnsupportedOperator, op_tok.span,
        std::string("operator '") + std::string(op_tok.text) +
            "' is not supported in magmaan v0"));
  }
  auto op_opt = op_from_text(op_tok.text);
  if (!op_opt.has_value()) {
    return std::unexpected(make_err(
        ParseError::Kind::ExpectedOperator, op_tok.span,
        std::string("operator '") + std::string(op_tok.text) +
            "' is not recognized as a formula operator"));
  }
  Op op = *op_opt;
  st.consume();

  auto rhs_or = parse_rhs_list(st);
  if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());
  std::vector<ParsedRhsTerm>& rhs = *rhs_or;

  // Intercept detection: `lhs ~ 1` or `lhs ~ label*1` / `lhs ~ val*1` /
  // `lhs ~ c(...)*1` — exactly one bare-NumLit-one term, with an optional
  // modifier (label, fixed value, per-group `c(...)`) attached. The modifier
  // is preserved into the Intercept row so scalar-invariance models can
  // express shared-label intercepts (`x1 ~ n1*1` ≙ ν_x1 shared across
  // groups via union-find on `n1`).
  if (op == Op::Regression && rhs.size() == 1 && rhs[0].term.is_numlit_one) {
    std::uint32_t mi = 0;
    if (rhs[0].modifier.has_value()) {
      mi = flat.add_modifier(std::move(*rhs[0].modifier));
    }
    flat.rows.push_back(FlatRow{
        lhs_text, Op::Intercept, std::string_view{}, /*block=*/1,
        mi, rhs[0].term.span});
    return {};
  }

  for (auto& r : rhs) {
    if (r.term.is_numlit_one) {
      return std::unexpected(make_err(
          ParseError::Kind::ExpectedRhsTerm, r.term.span,
          "bare '1' is only valid as the lone RHS of `~` (intercept form)"));
    }
    std::uint32_t mi = 0;
    if (r.modifier.has_value()) {
      mi = flat.add_modifier(std::move(*r.modifier));
    }
    flat.rows.push_back(FlatRow{lhs_text, op, r.term.ident_text,
                                /*block=*/1, mi, r.term.span});
  }
  return {};
}

// production: statement
//
// Dispatches between formula / constraint / define-param by classifying
// the upcoming statement: scan ahead for the first top-level Op token and
// branch on its kind.
parse_expected<void>
parse_statement(State& st, FlatPartable& flat) noexcept {
  const StatementShape shape = classify_statement(st);
  switch (shape.kind) {
    case StatementShape::Kind::Constraint:
      return parse_constraint(st, flat, shape);
    case StatementShape::Kind::Define:
      return parse_define(st, flat, shape);
    case StatementShape::Kind::Formula:
    case StatementShape::Kind::Unknown:
      // Unknown shapes (no Op found, or rejected op like `<~`) fall through
      // to parse_formula, which raises the precise error.
      return parse_formula(st, flat);
  }
  return {};
}

// Skip statement separators (Newline, Semicolon) but stop at EOF.
void skip_statement_separators(State& st) noexcept {
  while (st.peek().kind == TokenKind::Newline ||
         st.peek().kind == TokenKind::Semicolon) {
    ++st.pos;
  }
}

// production: model
parse_expected<void>
parse_model(State& st, FlatPartable& flat) noexcept {
  skip_statement_separators(st);
  while (!st.at_eof()) {
    auto r = parse_statement(st, flat);
    if (!r.has_value()) return std::unexpected(r.error());
    // After a statement, demand at least one separator (or EOF).
    if (!st.at_eof() &&
        st.peek().kind != TokenKind::Newline &&
        st.peek().kind != TokenKind::Semicolon) {
      return std::unexpected(make_err(
          ParseError::Kind::ExpectedOperator, st.peek().span,
          std::string("expected statement separator (newline or ';'), got ") +
              std::string(to_string(st.peek().kind))));
    }
    skip_statement_separators(st);
  }
  return {};
}

}  // namespace

parse_expected<FlatPartable> Parser::parse(std::string_view src) {
  FlatPartable flat;
  flat.source_text.assign(src.begin(), src.end());
  // string_views in flat.rows and flat.mods reference flat.source_text.data(),
  // which is heap-stable across moves of the FlatPartable.
  const std::string_view owned = flat.source();

  auto toks_or = tokenize(owned);
  if (!toks_or.has_value()) return std::unexpected(toks_or.error());

  State st{std::move(*toks_or), 0};
  auto r = parse_model(st, flat);
  if (!r.has_value()) return std::unexpected(r.error());
  return flat;
}

}  // namespace magmaan::parse
