#pragma once

#include <string_view>

namespace magmaan::spec::detail {

// True when a canonical Expr text is a bare identifier — no operator / paren /
// whitespace characters. `a`, `.p3.` are bare; `2*b`, `(a+b)`, `-1` are not.
// (`2` passes the char test but won't resolve to any parameter name.)
//
// Used to decide whether a constraint `==` side is a simple parameter
// reference: a bare side that resolves to a free index is folded into
// `eq_groups` (a merge) by `compute_eq_groups` (lavaanify.cpp); anything else
// is re-parsed and reduced to a `lin_constraint_R` row by
// `resolve_lin_constraints` (lin_constraints.cpp). Both must agree on this
// classification — keep them sharing this one function.
inline bool is_bare_identifier(std::string_view s) noexcept {
  if (s.empty()) return false;
  for (char c : s) {
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' ||
        c == '(' || c == ')' || c == ' ' || c == '\t') {
      return false;
    }
  }
  return true;
}

}  // namespace magmaan::spec::detail
