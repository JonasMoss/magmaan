#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/lin_constraints.hpp"

namespace lp = magmaan::parse;
using magmaan::spec::analyze_linear;
using magmaan::spec::LinearForm;

namespace {

lp::ExprPtr num(double v) { return std::make_unique<lp::Expr>(lp::Num{v}); }
lp::ExprPtr par(std::string_view t) { return std::make_unique<lp::Expr>(lp::Param{t}); }
lp::ExprPtr bin(lp::BinOp op, lp::ExprPtr l, lp::ExprPtr r) {
  return std::make_unique<lp::Expr>(lp::BinNode{op, std::move(l), std::move(r)});
}
lp::ExprPtr neg(lp::ExprPtr a) {
  return std::make_unique<lp::Expr>(lp::UnNode{lp::UnOp::Neg, std::move(a)});
}
lp::ExprPtr fn(lp::UnOp op, lp::ExprPtr a) {
  return std::make_unique<lp::Expr>(lp::UnNode{op, std::move(a)});
}

// n_free = 3, free indices (1-based): a→1, b→2, c→3; fixed: k = 5.0.
const std::unordered_map<std::string_view, int>    kFree  = {{"a", 1}, {"b", 2}, {"c", 3}};
const std::unordered_map<std::string_view, double> kFixed = {{"k", 5.0}};

std::optional<LinearForm> an(const lp::Expr& e) { return analyze_linear(e, 3, kFree, kFixed); }

void check_form(const std::optional<LinearForm>& f, double c0, double c1, double c2, double cst) {
  REQUIRE(f.has_value());
  REQUIRE(f->coef.size() == 3);
  CHECK(f->coef[0] == doctest::Approx(c0));
  CHECK(f->coef[1] == doctest::Approx(c1));
  CHECK(f->coef[2] == doctest::Approx(c2));
  CHECK(f->cst     == doctest::Approx(cst));
}

}  // namespace

TEST_CASE("analyze_linear: leaves") {
  check_form(an(*num(2.0)),         0, 0, 0, 2.0);
  check_form(an(*par("a")),         1, 0, 0, 0.0);
  check_form(an(*par("b")),         0, 1, 0, 0.0);
  check_form(an(*par("k")),         0, 0, 0, 5.0);   // fixed param ⇒ a constant
  CHECK_FALSE(an(*par("nope")).has_value());          // unknown identifier
}

TEST_CASE("analyze_linear: affine combinations") {
  // 2*b + c
  check_form(an(*bin(lp::BinOp::Add, bin(lp::BinOp::Mul, num(2.0), par("b")), par("c"))),
             0, 2, 1, 0.0);
  // b - 3
  check_form(an(*bin(lp::BinOp::Sub, par("b"), num(3.0))), 0, 1, 0, -3.0);
  // -b
  check_form(an(*neg(par("b"))), 0, -1, 0, 0.0);
  // b / 2
  check_form(an(*bin(lp::BinOp::Div, par("b"), num(2.0))), 0, 0.5, 0, 0.0);
  // 2 * (a + b)   — constant times an affine subexpr
  check_form(an(*bin(lp::BinOp::Mul, num(2.0), bin(lp::BinOp::Add, par("a"), par("b")))),
             2, 2, 0, 0.0);
  // b ^ 0 / b ^ 1 / 2 ^ 3   — constant exponent
  check_form(an(*bin(lp::BinOp::Pow, par("b"), num(0.0))), 0, 0, 0, 1.0);
  check_form(an(*bin(lp::BinOp::Pow, par("b"), num(1.0))), 0, 1, 0, 0.0);
  check_form(an(*bin(lp::BinOp::Pow, num(2.0), num(3.0))), 0, 0, 0, 8.0);
}

TEST_CASE("analyze_linear: nonlinear forms are rejected") {
  // b * c   — product of two θ-dependent subexprs
  CHECK_FALSE(an(*bin(lp::BinOp::Mul, par("b"), par("c"))).has_value());
  // (a + b) * c
  CHECK_FALSE(an(*bin(lp::BinOp::Mul, bin(lp::BinOp::Add, par("a"), par("b")), par("c"))).has_value());
  // b ^ 2
  CHECK_FALSE(an(*bin(lp::BinOp::Pow, par("b"), num(2.0))).has_value());
  // b / c   — divide by a θ-dependent subexpr
  CHECK_FALSE(an(*bin(lp::BinOp::Div, par("b"), par("c"))).has_value());
  // b / 0   — divide by zero
  CHECK_FALSE(an(*bin(lp::BinOp::Div, par("b"), num(0.0))).has_value());
}

TEST_CASE("analyze_linear: exp/log fold constants, reject θ-dependent args") {
  // exp / log of a constant argument fold to a constant LinearForm.
  check_form(an(*fn(lp::UnOp::Exp, num(2.0))),  0, 0, 0, std::exp(2.0));
  check_form(an(*fn(lp::UnOp::Log, par("k"))),  0, 0, 0, std::log(5.0));  // k=5
  // exp / log of anything θ-dependent is nonlinear.
  CHECK_FALSE(an(*fn(lp::UnOp::Exp, par("b"))).has_value());
  CHECK_FALSE(an(*fn(lp::UnOp::Log,
                     bin(lp::BinOp::Add, par("a"), par("c")))).has_value());
  // log of a non-positive constant is rejected, not folded to NaN.
  CHECK_FALSE(an(*fn(lp::UnOp::Log, num(-1.0))).has_value());
}
