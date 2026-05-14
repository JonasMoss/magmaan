#pragma once

#include <charconv>
#include <cmath>
#include <string>
#include <type_traits>
#include <variant>

#include "magmaan/parse/flat_partable.hpp"

namespace magmaan::parse {

// Expr → canonical text. Mirrors lavaan's `attr(flat, "constraints")[[i]]$rhs`
// shape: tokens concatenated with no whitespace, parens only when associativity
// demands it. e.g.
//   Mul(Add(a, b), c)  →  "(a+b)*c"
//   Add(Mul(a, b), c)  →  "a*b+c"

inline int binop_prec(BinOp op) noexcept {
  switch (op) {
    case BinOp::Add:
    case BinOp::Sub: return 1;
    case BinOp::Mul:
    case BinOp::Div: return 2;
    case BinOp::Pow: return 3;
  }
  return 0;
}

inline char binop_char(BinOp op) noexcept {
  switch (op) {
    case BinOp::Add: return '+';
    case BinOp::Sub: return '-';
    case BinOp::Mul: return '*';
    case BinOp::Div: return '/';
    case BinOp::Pow: return '^';
  }
  return '?';
}

inline std::string format_number_canonical(double v) {
  // Integers print without a decimal point ("0", "1") to match lavaan's
  // R-style output. Non-integers go through to_chars (locale-independent).
  if (std::isfinite(v)) {
    const long long iv = static_cast<long long>(v);
    if (static_cast<double>(iv) == v) return std::to_string(iv);
  }
  char buf[32];
  auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, end);
}

inline std::string expr_to_canonical(const Expr& e, int parent_prec = 0);

inline std::string expr_to_canonical(const Expr& e, int parent_prec) {
  std::string out;
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Num>) {
          out = format_number_canonical(v.value);
        } else if constexpr (std::is_same_v<T, Param>) {
          out = std::string(v.text);
        } else if constexpr (std::is_same_v<T, UnNode>) {
          out += (v.op == UnOp::Neg) ? "-" : "+";
          out += expr_to_canonical(*v.arg, /*parent_prec=*/4);
        } else if constexpr (std::is_same_v<T, BinNode>) {
          const int prec = binop_prec(v.op);
          out += expr_to_canonical(*v.lhs, prec);
          out += binop_char(v.op);
          // For left-associative operators, bumping rhs parent_prec by 1
          // forces parens on equal-precedence right operands. For ^ (right-
          // associative), we keep the same precedence on the right.
          const int rhs_parent = (v.op == BinOp::Pow) ? prec : prec + 1;
          out += expr_to_canonical(*v.rhs, rhs_parent);
          if (prec < parent_prec) out = "(" + out + ")";
        }
      },
      static_cast<const std::variant<Num, Param, BinNode, UnNode>&>(e));
  return out;
}

}  // namespace magmaan::parse
