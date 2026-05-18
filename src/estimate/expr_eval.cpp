#include "magmaan/estimate/expr_eval.hpp"

#include <cmath>
#include <string>
#include <type_traits>
#include <variant>

namespace magmaan::estimate {

namespace {

PostError make_err(std::string detail) {
  return PostError{PostError::Kind::NumericIssue, std::move(detail)};
}

post_expected<ADValue> eval_num(const parse::Num& n, const Scope& sc) {
  ADValue out = zero_ad(sc.n_free);
  out.v = n.value;
  return out;
}

post_expected<ADValue> eval_param(const parse::Param& p, const Scope& sc) {
  // A `:=` name shadows a same-named row label (lavaan strips def names out of
  // the label-resolution list). For constraint expressions `defined` is empty.
  if (auto it = sc.defined.find(p.text); it != sc.defined.end()) {
    return it->second;
  }
  if (auto it = sc.label_to_free.find(p.text); it != sc.label_to_free.end()) {
    const auto k = it->second;  // 1-based free index
    if (k < 1 || static_cast<std::size_t>(k) > sc.n_free) {
      return std::unexpected(make_err(
          std::string("expression references '") + std::string(p.text) +
          "' with free index " + std::to_string(k) + " out of range"));
    }
    ADValue out = zero_ad(sc.n_free);
    out.v = sc.theta(k - 1);
    out.dv(k - 1) = 1.0;
    return out;
  }
  if (auto it = sc.label_to_fixed.find(p.text); it != sc.label_to_fixed.end()) {
    ADValue out = zero_ad(sc.n_free);
    out.v = it->second;  // ∂(fixed)/∂θ = 0
    return out;
  }
  return std::unexpected(make_err(
      std::string("expression references unknown identifier '") +
      std::string(p.text) +
      "'; not a labeled row, a `.pN.` plabel, or a `:=` row"));
}

post_expected<ADValue> eval_bin(const parse::BinNode& b, const Scope& sc) {
  auto lhs_or = eval_expr(*b.lhs, sc);
  if (!lhs_or.has_value()) return std::unexpected(lhs_or.error());
  auto rhs_or = eval_expr(*b.rhs, sc);
  if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());
  const ADValue& a = *lhs_or;
  const ADValue& c = *rhs_or;
  ADValue out;
  out.dv.resize(static_cast<Eigen::Index>(sc.n_free));
  switch (b.op) {
    case parse::BinOp::Add:
      out.v  = a.v + c.v;
      out.dv = a.dv + c.dv;
      break;
    case parse::BinOp::Sub:
      out.v  = a.v - c.v;
      out.dv = a.dv - c.dv;
      break;
    case parse::BinOp::Mul:
      out.v  = a.v * c.v;
      out.dv = c.v * a.dv + a.v * c.dv;
      break;
    case parse::BinOp::Div:
      if (c.v == 0.0) {
        return std::unexpected(make_err("division by zero in expression"));
      }
      out.v  = a.v / c.v;
      out.dv = (a.dv * c.v - a.v * c.dv) / (c.v * c.v);  // (a/c)' = (a'c − ac')/c²
      break;
    case parse::BinOp::Pow:
      // (a^c)' = a^c · (c'·log|a| + c·a'/a). With c constant w.r.t. θ the
      // log|a| term drops out (c.dv == 0); with variable c and a ≤ 0 the
      // log is undefined.
      if (c.dv.cwiseAbs().maxCoeff() != 0.0 && a.v <= 0.0) {
        return std::unexpected(make_err(
            "a^b with a non-constant exponent requires a > 0"));
      }
      out.v = std::pow(a.v, c.v);
      if (a.v == 0.0 && c.v < 1.0) {
        return std::unexpected(make_err(
            "0^b with b < 1 has an unbounded gradient"));
      }
      out.dv = c.v * std::pow(a.v, c.v - 1.0) * a.dv;
      if (c.dv.cwiseAbs().maxCoeff() != 0.0) {
        out.dv += out.v * std::log(a.v) * c.dv;
      }
      break;
  }
  return out;
}

post_expected<ADValue> eval_un(const parse::UnNode& u, const Scope& sc) {
  auto arg_or = eval_expr(*u.arg, sc);
  if (!arg_or.has_value()) return std::unexpected(arg_or.error());
  ADValue out;
  if (u.op == parse::UnOp::Neg) {
    out.v  = -arg_or->v;
    out.dv = -arg_or->dv;
  } else {  // Pos
    out = *arg_or;
  }
  return out;
}

}  // namespace

ADValue zero_ad(std::size_t n_free) {
  return ADValue{0.0, Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_free))};
}

post_expected<ADValue> eval_expr(const parse::Expr& e, const Scope& sc) {
  return std::visit(
      [&](auto&& v) -> post_expected<ADValue> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Num>) {
          return eval_num(v, sc);
        } else if constexpr (std::is_same_v<T, parse::Param>) {
          return eval_param(v, sc);
        } else if constexpr (std::is_same_v<T, parse::BinNode>) {
          return eval_bin(v, sc);
        } else {  // parse::UnNode
          return eval_un(v, sc);
        }
      },
      static_cast<const parse::Expr::variant&>(e));
}

void collect_params(const parse::Expr& e, std::vector<std::string_view>& out) {
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Param>) {
          out.push_back(v.text);
        } else if constexpr (std::is_same_v<T, parse::BinNode>) {
          collect_params(*v.lhs, out);
          collect_params(*v.rhs, out);
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          collect_params(*v.arg, out);
        }
      },
      static_cast<const parse::Expr::variant&>(e));
}

}  // namespace magmaan::estimate
