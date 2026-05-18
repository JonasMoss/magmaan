#include "magmaan/spec/lin_constraints.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"

#include "detail_constraint_text.hpp"

namespace magmaan::spec {

namespace {

LinearForm zero_form(int n_free) {
  LinearForm f;
  f.coef.assign(static_cast<std::size_t>(n_free < 0 ? 0 : n_free), 0.0);
  return f;
}

bool is_constant(const LinearForm& f) noexcept {
  for (double v : f.coef) if (v != 0.0) return false;
  return true;
}

LinearForm scaled(LinearForm f, double s) {
  for (double& v : f.coef) v *= s;
  f.cst *= s;
  return f;
}

// a + s·b  (s is +1 for Add, -1 for Sub).
LinearForm combine(LinearForm a, const LinearForm& b, double s) {
  for (std::size_t k = 0; k < a.coef.size(); ++k) a.coef[k] += s * b.coef[k];
  a.cst += s * b.cst;
  return a;
}

// True if `e` mentions any identifier that is neither a free parameter nor a
// fixed cell — i.e. an unknown reference. `analyze_linear` returns
// `std::nullopt` for *both* a nonlinear expression and an unknown identifier;
// this separates the two so a malformed `==` row is not mistaken for a
// well-formed nonlinear one.
bool expr_has_unknown_id(
    const parse::Expr& e,
    const std::unordered_map<std::string_view, int>&    name_to_free,
    const std::unordered_map<std::string_view, double>& name_to_fixed) {
  return std::visit(
      [&](auto&& v) -> bool {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Num>) {
          return false;
        } else if constexpr (std::is_same_v<T, parse::Param>) {
          return name_to_free.find(v.text) == name_to_free.end() &&
                 name_to_fixed.find(v.text) == name_to_fixed.end();
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          return expr_has_unknown_id(*v.arg, name_to_free, name_to_fixed);
        } else {  // parse::BinNode
          return expr_has_unknown_id(*v.lhs, name_to_free, name_to_fixed) ||
                 expr_has_unknown_id(*v.rhs, name_to_free, name_to_fixed);
        }
      },
      static_cast<const parse::Expr::variant&>(e));
}

// Compile a parsed expression into the name-free `NlExprNode` pool, resolving
// each identifier through the constraint name maps. The caller has already
// run `expr_has_unknown_id`, so every `Param` resolves to a free index or a
// fixed value. Children are appended before their parent; returns the
// parent's pool index.
std::int32_t compile_nl_expr(
    const parse::Expr& e,
    const std::unordered_map<std::string_view, int>&    name_to_free,
    const std::unordered_map<std::string_view, double>& name_to_fixed,
    std::vector<NlExprNode>& pool) {
  return std::visit(
      [&](auto&& v) -> std::int32_t {
        using T = std::decay_t<decltype(v)>;
        NlExprNode node;
        if constexpr (std::is_same_v<T, parse::Num>) {
          node.kind = NlExprNode::Kind::Const;
          node.constant = v.value;
        } else if constexpr (std::is_same_v<T, parse::Param>) {
          node.kind = NlExprNode::Kind::Param;
          if (auto it = name_to_free.find(v.text); it != name_to_free.end()) {
            node.free_idx = it->second - 1;            // 1-based → 0-based
          } else {
            node.free_idx = -1;
            node.constant = name_to_fixed.at(v.text);  // a fixed reference
          }
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          const std::int32_t c =
              compile_nl_expr(*v.arg, name_to_free, name_to_fixed, pool);
          node.kind  = NlExprNode::Kind::Unary;
          node.un_op = v.op;
          node.lhs   = c;
        } else {  // parse::BinNode
          const std::int32_t l =
              compile_nl_expr(*v.lhs, name_to_free, name_to_fixed, pool);
          const std::int32_t r =
              compile_nl_expr(*v.rhs, name_to_free, name_to_fixed, pool);
          node.kind   = NlExprNode::Kind::Binary;
          node.bin_op = v.op;
          node.lhs    = l;
          node.rhs    = r;
        }
        pool.push_back(node);
        return static_cast<std::int32_t>(pool.size() - 1);
      },
      static_cast<const parse::Expr::variant&>(e));
}

}  // namespace

std::optional<LinearForm>
analyze_linear(const parse::Expr& e, int n_free,
               const std::unordered_map<std::string_view, int>&    name_to_free,
               const std::unordered_map<std::string_view, double>& name_to_fixed) {
  return std::visit(
      [&](auto&& v) -> std::optional<LinearForm> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Num>) {
          LinearForm f = zero_form(n_free);
          f.cst = v.value;
          return f;
        } else if constexpr (std::is_same_v<T, parse::Param>) {
          if (auto it = name_to_free.find(v.text); it != name_to_free.end()) {
            const int k = it->second;  // 1-based free index
            if (k < 1 || k > n_free) return std::nullopt;
            LinearForm f = zero_form(n_free);
            f.coef[static_cast<std::size_t>(k - 1)] = 1.0;
            return f;
          }
          if (auto it = name_to_fixed.find(v.text); it != name_to_fixed.end()) {
            LinearForm f = zero_form(n_free);
            f.cst = it->second;
            return f;
          }
          return std::nullopt;  // unknown identifier
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          auto a = analyze_linear(*v.arg, n_free, name_to_free, name_to_fixed);
          if (!a) return std::nullopt;
          return (v.op == parse::UnOp::Neg) ? scaled(std::move(*a), -1.0) : std::move(*a);
        } else {  // parse::BinNode
          auto a = analyze_linear(*v.lhs, n_free, name_to_free, name_to_fixed);
          if (!a) return std::nullopt;
          auto b = analyze_linear(*v.rhs, n_free, name_to_free, name_to_fixed);
          if (!b) return std::nullopt;
          switch (v.op) {
            case parse::BinOp::Add: return combine(std::move(*a), *b, +1.0);
            case parse::BinOp::Sub: return combine(std::move(*a), *b, -1.0);
            case parse::BinOp::Mul:
              if (is_constant(*a)) return scaled(std::move(*b), a->cst);
              if (is_constant(*b)) return scaled(std::move(*a), b->cst);
              return std::nullopt;  // θ·θ — nonlinear
            case parse::BinOp::Div:
              if (!is_constant(*b) || b->cst == 0.0) return std::nullopt;
              return scaled(std::move(*a), 1.0 / b->cst);
            case parse::BinOp::Pow:
              if (!is_constant(*b)) return std::nullopt;          // θ in exponent
              if (b->cst == 1.0) return std::move(*a);
              if (b->cst == 0.0) { LinearForm f = zero_form(n_free); f.cst = 1.0; return f; }
              if (!is_constant(*a)) return std::nullopt;          // θ^c, c ∉ {0,1}
              { LinearForm f = zero_form(n_free); f.cst = std::pow(a->cst, b->cst); return f; }
          }
          return std::nullopt;  // unreachable
        }
      },
      static_cast<const parse::Expr::variant&>(e));
}

void resolve_lin_constraints(LatentStructure& pt, const LatentNames& names) {
  pt.lin_constraint_R.clear();
  pt.lin_constraint_d.clear();
  pt.nonlinear_eq_rows.clear();
  pt.nl_constraints.clear();
  pt.has_inequality_constraints = false;
  pt.has_unenforced_constraints = false;
  const int n_free = pt.n_free();

  // name (row label OR `.pN.` plabel) → 1-based free index (free non-constraint
  // rows) / fixed cell value (fixed non-constraint rows). First-wins on labels,
  // matching `compute_eq_groups`'s `free_by_label.front()`.
  std::unordered_map<std::string_view, int>    name_to_free;
  std::unordered_map<std::string_view, double> name_to_fixed;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.is_constraint_row(i)) continue;
    const std::string_view lbl =
        i < names.row_label.size() ? std::string_view(names.row_label[i]) : std::string_view{};
    const std::string_view pl =
        i < names.row_plabel.size() ? std::string_view(names.row_plabel[i]) : std::string_view{};
    if (pt.free[i] > 0) {
      if (!lbl.empty()) name_to_free.try_emplace(lbl, pt.free[i]);
      if (!pl.empty())  name_to_free.try_emplace(pl,  pt.free[i]);
    } else if (std::isfinite(pt.fixed_value[i])) {
      if (!lbl.empty()) name_to_fixed.try_emplace(lbl, pt.fixed_value[i]);
      if (!pl.empty())  name_to_fixed.try_emplace(pl,  pt.fixed_value[i]);
    }
  }

  for (std::size_t i = 0; i < pt.size(); ++i) {
    const parse::Op op = pt.op[i];
    if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) {
      pt.has_inequality_constraints = true;
      continue;
    }
    if (op != parse::Op::EqConstraint) continue;
    const std::string& lhs_txt = names.row_lhs[i];
    const std::string& rhs_txt = names.row_rhs[i];
    // Already folded into `eq_groups` by `compute_eq_groups` (a bare parameter
    // reference on *both* sides) — skip; mirrors that function's `resolve()`.
    if (detail::is_bare_identifier(lhs_txt) && detail::is_bare_identifier(rhs_txt) &&
        name_to_free.count(lhs_txt) && name_to_free.count(rhs_txt)) {
      continue;
    }
    // Re-parse "<lhs> == <rhs>" to recover the Expr ASTs (no public bare-expr
    // parser; the canonical text round-trips through `Parser::parse`).
    const std::string snippet = lhs_txt + " == " + rhs_txt;
    auto fp = parse::Parser::parse(snippet);
    if (!fp.has_value() || fp->constraints.empty() ||
        fp->constraints[0].kind != parse::ConstraintKind::Eq) {
      pt.has_unenforced_constraints = true;   // malformed: failed to re-parse
      continue;
    }
    const parse::Constraint& c = fp->constraints[0];
    // An unknown identifier means the row is malformed — distinct from a
    // well-formed nonlinear constraint, which `analyze_linear` would also
    // reject. Classify it before asking whether the expression is affine.
    if (expr_has_unknown_id(c.lhs, name_to_free, name_to_fixed) ||
        expr_has_unknown_id(c.rhs, name_to_free, name_to_fixed)) {
      pt.has_unenforced_constraints = true;
      continue;
    }
    auto gl = analyze_linear(c.lhs, n_free, name_to_free, name_to_fixed);
    auto gr = analyze_linear(c.rhs, n_free, name_to_free, name_to_fixed);
    if (!gl || !gr) {
      // Every identifier is known but the expression is not affine — a
      // genuine nonlinear equality constraint (`a == b*c`, `b == (c+d)^2`).
      // Compile `h = lhs − rhs` into the name-free node pool.
      pt.nonlinear_eq_rows.push_back(static_cast<std::int32_t>(i));
      NlConstraint nlc;
      const std::int32_t rl =
          compile_nl_expr(c.lhs, name_to_free, name_to_fixed, nlc.nodes);
      const std::int32_t rr =
          compile_nl_expr(c.rhs, name_to_free, name_to_fixed, nlc.nodes);
      NlExprNode sub;
      sub.kind   = NlExprNode::Kind::Binary;
      sub.bin_op = parse::BinOp::Sub;
      sub.lhs    = rl;
      sub.rhs    = rr;
      nlc.nodes.push_back(sub);
      nlc.root = static_cast<std::int32_t>(nlc.nodes.size() - 1);
      pt.nl_constraints.push_back(std::move(nlc));
      continue;
    }
    // lhs(θ) == rhs(θ)  ⟺  (gl.coef − gr.coef)·θ = gr.cst − gl.cst.
    pt.lin_constraint_d.push_back(gr->cst - gl->cst);
    pt.lin_constraint_R.reserve(pt.lin_constraint_R.size() +
                                static_cast<std::size_t>(n_free < 0 ? 0 : n_free));
    for (int k = 0; k < n_free; ++k) {
      pt.lin_constraint_R.push_back(gl->coef[static_cast<std::size_t>(k)] -
                                    gr->coef[static_cast<std::size_t>(k)]);
    }
  }
}

}  // namespace magmaan::spec
