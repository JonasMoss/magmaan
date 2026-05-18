#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::spec {

// One side of a constraint, reduced to a linear-in-θ form:
//   value(θ) = Σ_k coef[k]·θ_k + cst    over the `n_free` free parameters
//   (coef[k] is the coefficient on the (k+1)-th free param).
struct LinearForm {
  std::vector<double> coef;   // size n_free
  double              cst = 0.0;
};

// Reduce an `==`-row side (a parsed Expr) to a `LinearForm` if it's affine in θ,
// else `std::nullopt` — genuinely nonlinear (a product / quotient of two
// θ-dependent subexprs, a θ-dependent base raised to a power ≠ 1) or referencing
// an unknown identifier. `name_to_free` maps a row label / `.pN.` plabel to its
// 1-based free index (free non-constraint rows); `name_to_fixed` maps a label /
// plabel to a fixed cell value (fixed non-constraint rows).
std::optional<LinearForm>
analyze_linear(const parse::Expr& e, int n_free,
               const std::unordered_map<std::string_view, int>&    name_to_free,
               const std::unordered_map<std::string_view, double>& name_to_fixed);

// Classify and resolve the model's constraint rows. Recomputes, from scratch:
//   * `pt.lin_constraint_R` / `pt.lin_constraint_d` — one `R·θ = d` row per
//     general-*linear* `==` row that `compute_eq_groups` could not fold into
//     `eq_groups` (`a == 2*b+c`, `Σλ == k`, …);
//   * `pt.nonlinear_eq_rows` — partable indices of well-formed *nonlinear*
//     `==` rows (every identifier known, expression not affine);
//   * `pt.has_inequality_constraints` — a `<` / `>` row is present;
//   * `pt.has_unenforced_constraints` — a `==` row is malformed (failed to
//     re-parse, or names an unknown identifier).
// Non-merge `==` rows carry their canonical text on `names.row_lhs` /
// `names.row_rhs`, which is re-parsed here. Idempotent (clears every output
// first); call right after `compute_eq_groups`.
void resolve_lin_constraints(LatentStructure& pt, const LatentNames& names);

}  // namespace magmaan::spec
