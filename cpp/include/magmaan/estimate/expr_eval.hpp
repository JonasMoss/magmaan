#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"

namespace magmaan::estimate {

// Forward-mode automatic differentiation over the parsed expression AST
// (`parse::Expr`, shared by `:=` definitions and `==` / `<` / `>` constraint
// rows). Each subexpression carries a scalar value and a θ-space gradient
// (size n_free). Used for the delta-method SE of a `:=` defined parameter and
// for the constraint function h(θ) and its Jacobian on nonlinear `==` rows.

struct ADValue {
  double          v = 0.0;
  Eigen::VectorXd dv;   // size n_free; ∂value/∂θ
};

using LabelMap   = std::unordered_map<std::string_view, std::int32_t>;
using FixedMap   = std::unordered_map<std::string_view, double>;
using DefinedMap = std::unordered_map<std::string_view, ADValue>;

// Identifier-resolution context for one evaluation pass. The members are
// references — the underlying maps and the θ vector outlive the call.
//
//   label_to_free  : user label / `.pN.` plabel of a *free* row  → 1-based θ index
//   label_to_fixed : user label / `.pN.` plabel of a *fixed* row → fixed value
//   defined        : earlier `:=` names → their AD value; empty for constraint
//                    expressions, which never reference defined parameters
struct Scope {
  std::size_t            n_free = 0;
  const Eigen::VectorXd& theta;
  const LabelMap&        label_to_free;
  const FixedMap&        label_to_fixed;
  const DefinedMap&      defined;
};

// A zero ADValue with a correctly-sized (n_free) zero gradient.
ADValue zero_ad(std::size_t n_free);

// Evaluate `e` to a value + θ-space gradient under `sc`. Errors
// (`PostError::Kind::NumericIssue`) on an unknown identifier, division by
// zero, or an ill-defined power.
post_expected<ADValue> eval_expr(const parse::Expr& e, const Scope& sc);

// Append every `Param` identifier appearing in `e` to `out` (with repeats —
// the caller deduplicates).
void collect_params(const parse::Expr& e, std::vector<std::string_view>& out);

}  // namespace magmaan::estimate
