#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <variant>

#include "magmaan/error.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"

using magmaan::ParseError;
using magmaan::parse::FixedValue;
using magmaan::parse::FlatPartable;
using magmaan::parse::Free;
using magmaan::parse::GroupVec;
using magmaan::parse::Label;
using magmaan::parse::Modifier;
using magmaan::parse::ModifierAtom;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::parse::StartValue;

namespace {

// Convenience: parse and assert success, returning the FlatPartable.
FlatPartable must_parse(std::string_view src) {
  auto r = Parser::parse(src);
  REQUIRE_MESSAGE(r.has_value(),
                  "parse failed unexpectedly: kind=" << static_cast<int>(r.error().kind)
                  << " — " << r.error().detail);
  return std::move(*r);
}

}  // namespace

TEST_CASE("one-factor CFA: f =~ x1 + x2 + x3") {
  auto fp = must_parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.rows.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(fp.rows[i].lhs == "f");
    CHECK(fp.rows[i].op == Op::Measurement);
    CHECK(fp.rows[i].mod_idx == 0);
  }
  CHECK(fp.rows[0].rhs == "x1");
  CHECK(fp.rows[1].rhs == "x2");
  CHECK(fp.rows[2].rhs == "x3");
}

TEST_CASE("multi-statement: newline-separated") {
  auto fp = must_parse("visual =~ x1 + x2\ntextual =~ x3 + x4");
  REQUIRE(fp.rows.size() == 4);
  CHECK(fp.rows[0].lhs == "visual");
  CHECK(fp.rows[1].lhs == "visual");
  CHECK(fp.rows[2].lhs == "textual");
  CHECK(fp.rows[3].lhs == "textual");
}

TEST_CASE("multi-statement: semicolon-separated") {
  auto fp = must_parse("f =~ x1 + x2; y ~ f");
  REQUIRE(fp.rows.size() == 3);
  CHECK(fp.rows[0].op == Op::Measurement);
  CHECK(fp.rows[2].op == Op::Regression);
  CHECK(fp.rows[2].lhs == "y");
  CHECK(fp.rows[2].rhs == "f");
}

TEST_CASE("multi-line continuation: trailing '+'") {
  auto fp = must_parse("f =~ x1 + x2 +\n     x3 + x4");
  REQUIRE(fp.rows.size() == 4);
  CHECK(fp.rows[0].rhs == "x1");
  CHECK(fp.rows[3].rhs == "x4");
}

TEST_CASE("multi-line continuation: leading '+' on next line") {
  auto fp = must_parse("f =~ x1\n     + x2\n     + x3");
  REQUIRE(fp.rows.size() == 3);
  CHECK(fp.rows[2].rhs == "x3");
}

TEST_CASE("variance: f ~~ f") {
  auto fp = must_parse("f ~~ f");
  REQUIRE(fp.rows.size() == 1);
  CHECK(fp.rows[0].op == Op::Covariance);
  CHECK(fp.rows[0].lhs == "f");
  CHECK(fp.rows[0].rhs == "f");
}

TEST_CASE("regression: y ~ x1 + x2") {
  auto fp = must_parse("y ~ x1 + x2");
  REQUIRE(fp.rows.size() == 2);
  CHECK(fp.rows[0].op == Op::Regression);
  CHECK(fp.rows[0].lhs == "y");
  CHECK(fp.rows[0].rhs == "x1");
  CHECK(fp.rows[1].rhs == "x2");
}

TEST_CASE("intercept: y ~ 1") {
  auto fp = must_parse("y ~ 1");
  REQUIRE(fp.rows.size() == 1);
  CHECK(fp.rows[0].op == Op::Intercept);
  CHECK(fp.rows[0].lhs == "y");
  CHECK(fp.rows[0].rhs.empty());
}

TEST_CASE("intercept does not mis-fire when other terms are present") {
  // `y ~ 1 + x` — the bare `1` cannot mix with other terms in v0.
  auto r = Parser::parse("y ~ 1 + x");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ExpectedRhsTerm);
}

TEST_CASE("modifier: fixed value via n*x") {
  auto fp = must_parse("f =~ 1*x1 + x2");
  REQUIRE(fp.rows.size() == 2);
  REQUIRE(fp.rows[0].mod_idx != 0);
  REQUIRE(fp.rows[0].mod_idx < fp.mods.size());
  const auto* fv = std::get_if<FixedValue>(&fp.mods[fp.rows[0].mod_idx]);
  REQUIRE(fv != nullptr);
  CHECK(fv->value == doctest::Approx(1.0));
  CHECK(fp.rows[1].mod_idx == 0);
}

TEST_CASE("modifier: label via lbl*x") {
  auto fp = must_parse("f =~ a*x1 + a*x2 + b*x3");
  REQUIRE(fp.rows.size() == 3);
  const auto* l0 = std::get_if<Label>(&fp.mods[fp.rows[0].mod_idx]);
  const auto* l1 = std::get_if<Label>(&fp.mods[fp.rows[1].mod_idx]);
  const auto* l2 = std::get_if<Label>(&fp.mods[fp.rows[2].mod_idx]);
  REQUIRE(l0 != nullptr);
  REQUIRE(l1 != nullptr);
  REQUIRE(l2 != nullptr);
  CHECK(l0->text == "a");
  CHECK(l1->text == "a");
  CHECK(l2->text == "b");
}

TEST_CASE("modifier: start-value shorthand v?x") {
  auto fp = must_parse("f =~ 1*x1 + 0.7?x2 + x3");
  REQUIRE(fp.rows.size() == 3);
  CHECK(std::holds_alternative<FixedValue>(fp.mods[fp.rows[0].mod_idx]));
  const auto* sv = std::get_if<StartValue>(&fp.mods[fp.rows[1].mod_idx]);
  REQUIRE(sv != nullptr);
  CHECK(sv->value == doctest::Approx(0.7));
  CHECK(fp.rows[2].mod_idx == 0);
}

TEST_CASE("error: ? requires numeric atom on the left") {
  auto r = Parser::parse("f =~ a?x1");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ModifierEvalFailed);
}

TEST_CASE("error: missing identifier on LHS") {
  auto r = Parser::parse("=~ x1 + x2");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ExpectedLhs);
}

TEST_CASE("error: missing operator after LHS") {
  auto r = Parser::parse("f x1 + x2");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ExpectedOperator);
}

TEST_CASE("error: rejected operator surfaces as UnsupportedOperator") {
  auto r = Parser::parse("y |~ x");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::UnsupportedOperator);
}

// === Constraints + define-param + Pratt expressions =========================

TEST_CASE("constraint: a == b") {
  auto fp = must_parse("a == b");
  REQUIRE(fp.rows.empty());
  REQUIRE(fp.constraints.size() == 1);
  CHECK(fp.constraints[0].kind == magmaan::parse::ConstraintKind::Eq);
  const auto* lhs = std::get_if<magmaan::parse::Param>(&fp.constraints[0].lhs);
  const auto* rhs = std::get_if<magmaan::parse::Param>(&fp.constraints[0].rhs);
  REQUIRE(lhs != nullptr);
  REQUIRE(rhs != nullptr);
  CHECK(lhs->text == "a");
  CHECK(rhs->text == "b");
}

TEST_CASE("constraint: a > 0 and d < 1") {
  auto fp = must_parse("a > 0\nd < 1");
  REQUIRE(fp.constraints.size() == 2);
  CHECK(fp.constraints[0].kind == magmaan::parse::ConstraintKind::Gt);
  CHECK(fp.constraints[1].kind == magmaan::parse::ConstraintKind::Lt);
}

TEST_CASE("constraint: chained statements via ;") {
  auto fp = must_parse("a == b ; c > 0 ; d < 1");
  REQUIRE(fp.constraints.size() == 3);
}

TEST_CASE("define: indirect := a * b") {
  auto fp = must_parse("indirect := a * b");
  REQUIRE(fp.constraints.size() == 1);
  const auto& c = fp.constraints[0];
  CHECK(c.kind == magmaan::parse::ConstraintKind::Define);
  CHECK(c.name == "indirect");
  const auto* bin = std::get_if<magmaan::parse::BinNode>(&c.rhs);
  REQUIRE(bin != nullptr);
  CHECK(bin->op == magmaan::parse::BinOp::Mul);
}

TEST_CASE("Pratt: precedence — a + b * c parses as a + (b * c)") {
  auto fp = must_parse("indirect := a + b * c");
  REQUIRE(fp.constraints.size() == 1);
  const auto* bin = std::get_if<magmaan::parse::BinNode>(&fp.constraints[0].rhs);
  REQUIRE(bin != nullptr);
  CHECK(bin->op == magmaan::parse::BinOp::Add);
  // RHS of Add should be a Mul
  const auto* rhs_bin = std::get_if<magmaan::parse::BinNode>(bin->rhs.get());
  REQUIRE(rhs_bin != nullptr);
  CHECK(rhs_bin->op == magmaan::parse::BinOp::Mul);
}

TEST_CASE("Pratt: parens override — (a + b) * c") {
  auto fp = must_parse("indirect := (a + b) * c");
  REQUIRE(fp.constraints.size() == 1);
  const auto* bin = std::get_if<magmaan::parse::BinNode>(&fp.constraints[0].rhs);
  REQUIRE(bin != nullptr);
  CHECK(bin->op == magmaan::parse::BinOp::Mul);
  const auto* lhs_bin = std::get_if<magmaan::parse::BinNode>(bin->lhs.get());
  REQUIRE(lhs_bin != nullptr);
  CHECK(lhs_bin->op == magmaan::parse::BinOp::Add);
}

TEST_CASE("Pratt: unary minus") {
  auto fp = must_parse("indirect := -a + b");
  REQUIRE(fp.constraints.size() == 1);
  const auto* bin = std::get_if<magmaan::parse::BinNode>(&fp.constraints[0].rhs);
  REQUIRE(bin != nullptr);
  CHECK(bin->op == magmaan::parse::BinOp::Add);
  const auto* unop = std::get_if<magmaan::parse::UnNode>(bin->lhs.get());
  REQUIRE(unop != nullptr);
  CHECK(unop->op == magmaan::parse::UnOp::Neg);
}

TEST_CASE("Pratt: ^ is right-associative") {
  // 2 ^ 3 ^ 2 → 2 ^ (3 ^ 2) = 2^9 = 512 (right-assoc)
  // (vs left-assoc which would be (2^3)^2 = 8^2 = 64)
  auto fp = must_parse("indirect := 2 ^ 3 ^ 2");
  REQUIRE(fp.constraints.size() == 1);
  const auto* bin = std::get_if<magmaan::parse::BinNode>(&fp.constraints[0].rhs);
  REQUIRE(bin != nullptr);
  CHECK(bin->op == magmaan::parse::BinOp::Pow);
  // RHS should also be Pow (right-grouped)
  const auto* rhs_bin = std::get_if<magmaan::parse::BinNode>(bin->rhs.get());
  REQUIRE(rhs_bin != nullptr);
  CHECK(rhs_bin->op == magmaan::parse::BinOp::Pow);
}

TEST_CASE("error: bare := needs identifier on the left") {
  // `a + b := c` — LHS of := must be a single identifier.
  auto r = Parser::parse("a + b := c");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ExpectedOperator);
}

TEST_CASE("error: missing rhs in expression") {
  auto r = Parser::parse("indirect := a *");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ExpectedRhsTerm);
}

TEST_CASE("error: unmatched parens") {
  auto r = Parser::parse("indirect := (a + b");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ExpectedRhsTerm);
}

TEST_CASE("comments are transparent to the parser") {
  auto fp = must_parse("f =~ x1 # marker\n     + x2 # second\n     + x3");
  REQUIRE(fp.rows.size() == 3);
  CHECK(fp.rows[2].rhs == "x3");
}

// === Slice-2 modifier coverage ==============================================

TEST_CASE("modifier: NA produces Free variant") {
  auto fp = must_parse("f =~ NA*x1 + x2");
  REQUIRE(fp.rows.size() == 2);
  REQUIRE(fp.rows[0].mod_idx != 0);
  CHECK(std::holds_alternative<Free>(fp.mods[fp.rows[0].mod_idx]));
  CHECK(fp.rows[1].mod_idx == 0);
}

TEST_CASE("modifier: quoted-string label has quotes stripped") {
  auto fp = must_parse("f =~ \"loading_1\"*x1 + x2");
  REQUIRE(fp.rows.size() == 2);
  const auto* l = std::get_if<Label>(&fp.mods[fp.rows[0].mod_idx]);
  REQUIRE(l != nullptr);
  CHECK(l->text == "loading_1");
}

TEST_CASE("modifier: start(v) produces StartValue") {
  auto fp = must_parse("f =~ start(0.5)*x1 + start(0.7)*x2");
  REQUIRE(fp.rows.size() == 2);
  const auto* s0 = std::get_if<StartValue>(&fp.mods[fp.rows[0].mod_idx]);
  const auto* s1 = std::get_if<StartValue>(&fp.mods[fp.rows[1].mod_idx]);
  REQUIRE(s0 != nullptr);
  REQUIRE(s1 != nullptr);
  CHECK(s0->value == doctest::Approx(0.5));
  CHECK(s1->value == doctest::Approx(0.7));
}

TEST_CASE("modifier: c(...) per-group vector") {
  auto fp = must_parse("f =~ c(1, NA)*x1 + c(0.8, 1.2)*x2");
  REQUIRE(fp.rows.size() == 2);
  const auto* g0 = std::get_if<GroupVec>(&fp.mods[fp.rows[0].mod_idx]);
  const auto* g1 = std::get_if<GroupVec>(&fp.mods[fp.rows[1].mod_idx]);
  REQUIRE(g0 != nullptr);
  REQUIRE(g1 != nullptr);
  REQUIRE(g0->per_group.size() == 2);
  REQUIRE(g1->per_group.size() == 2);
  CHECK(std::get_if<FixedValue>(&g0->per_group[0])->value == doctest::Approx(1.0));
  CHECK(std::holds_alternative<Free>(g0->per_group[1]));
  CHECK(std::get_if<FixedValue>(&g1->per_group[0])->value == doctest::Approx(0.8));
  CHECK(std::get_if<FixedValue>(&g1->per_group[1])->value == doctest::Approx(1.2));
}

TEST_CASE("modifier: c(...) accepts identifiers and string labels") {
  auto fp = must_parse("f =~ c(a, \"b\")*x1");
  REQUIRE(fp.rows.size() == 1);
  const auto* g = std::get_if<GroupVec>(&fp.mods[fp.rows[0].mod_idx]);
  REQUIRE(g != nullptr);
  REQUIRE(g->per_group.size() == 2);
  CHECK(std::get_if<Label>(&g->per_group[0])->text == "a");
  CHECK(std::get_if<Label>(&g->per_group[1])->text == "b");
}

TEST_CASE("error: empty c() rejected") {
  auto r = Parser::parse("f =~ c()*x1");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::GroupVecMismatch);
}

TEST_CASE("error: trailing comma in c(...) rejected") {
  auto r = Parser::parse("f =~ c(1, 2,)*x1");
  REQUIRE_FALSE(r.has_value());
  // The trailing comma trips the modifier_atom parser at ')'.
  CHECK(r.error().kind == ParseError::Kind::ModifierEvalFailed);
}

TEST_CASE("error: start() with non-numeric argument rejected") {
  auto r = Parser::parse("f =~ start(a)*x1");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ModifierEvalFailed);
}

TEST_CASE("error: unknown modifier function rejected") {
  auto r = Parser::parse("f =~ prior(\"normal(0,1)\")*x1");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ModifierEvalFailed);
}

TEST_CASE("error: ? on non-FixedValue modifier rejected") {
  // start(0.5)?x is meaningless; our parse_rhs_term_with_mod restricts
  // ? to plain FixedValue atoms.
  auto r = Parser::parse("f =~ start(0.5)?x1");
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().kind == ParseError::Kind::ModifierEvalFailed);
}

TEST_CASE("identifier 'c' as a bare term is still a valid identifier") {
  // No reserved words at the parser level. `c + x` on the RHS treats
  // 'c' as an ordinary identifier (the lookahead requires LParen to
  // commit to the c(...) modifier).
  auto fp = must_parse("y ~ c + x");
  REQUIRE(fp.rows.size() == 2);
  CHECK(fp.rows[0].rhs == "c");
  CHECK(fp.rows[1].rhs == "x");
}

// === Original slice-1 tests continue below ==================================

TEST_CASE("FlatPartable owns its source bytes") {
  FlatPartable fp;
  {
    std::string scratch = "f =~ x1 + x2";
    auto r = Parser::parse(scratch);
    REQUIRE(r.has_value());
    fp = std::move(*r);
    // scratch goes out of scope here; FlatPartable.source_text holds a copy.
  }
  REQUIRE(fp.rows.size() == 2);
  CHECK(fp.rows[0].lhs == "f");
  CHECK(fp.rows[1].rhs == "x2");
}
