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
  if (!pt.has_unenforced_constraints) return;  // nothing flagged ⇒ nothing to do
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

  bool has_ineq        = false;
  bool any_eq_flagged  = false;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const parse::Op op = pt.op[i];
    if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) { has_ineq = true; continue; }
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
      any_eq_flagged = true;
      continue;
    }
    const parse::Constraint& c = fp->constraints[0];
    auto gl = analyze_linear(c.lhs, n_free, name_to_free, name_to_fixed);
    auto gr = analyze_linear(c.rhs, n_free, name_to_free, name_to_fixed);
    if (!gl || !gr) { any_eq_flagged = true; continue; }
    // lhs(θ) == rhs(θ)  ⟺  (gl.coef − gr.coef)·θ = gr.cst − gl.cst.
    pt.lin_constraint_d.push_back(gr->cst - gl->cst);
    pt.lin_constraint_R.reserve(pt.lin_constraint_R.size() +
                                static_cast<std::size_t>(n_free < 0 ? 0 : n_free));
    for (int k = 0; k < n_free; ++k) {
      pt.lin_constraint_R.push_back(gl->coef[static_cast<std::size_t>(k)] -
                                    gr->coef[static_cast<std::size_t>(k)]);
    }
  }

  if (!has_ineq && !any_eq_flagged) pt.has_unenforced_constraints = false;
}

}  // namespace magmaan::spec
