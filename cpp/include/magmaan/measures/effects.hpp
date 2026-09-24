#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"            // Estimates
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::measures::effects {

using estimate::Estimates;

// One user-defined parameter (`name := expr`). `value` is the expression
// evaluated at θ̂; `se` is the delta-method standard error using the
// gradient of `expr` wrt θ and the parameter covariance `vcov`.
struct DefinedParam {
  std::string name;
  double      value = 0.0;
  double      se    = 0.0;
};

struct DefinedParams {
  std::vector<DefinedParam> entries;
};

// Evaluate every `:=` constraint in `flat`, returning the numeric value
// and delta-method SE per entry (entries in source order, matching lavaan's
// `parameterEstimates()` layout).
//
// `pt` + `names` resolve each `Param` identifier in a `:=` expression:
//   - a user-supplied label ("a", on `names.row_label`), or
//   - a `.pN.` plabel (on `names.row_plabel`) — works for free rows *and*
//     fixed/exo rows (a fixed reference contributes value but a zero gradient), or
//   - the name of another `:=` row — chained references; the referenced row's
//     accumulated θ-gradient flows through, so the composite gets the right
//     delta-method SE. `:=` rows are evaluated in dependency order; a cycle
//     (including a self-reference) is a `PostError::NumericIssue`.
// An identifier that matches none of the above is a `PostError::NumericIssue`.
post_expected<DefinedParams>
compute_defined(const parse::FlatPartable& flat,
                const spec::LatentStructure&  pt,
                const spec::LatentNames&      names,
                const Estimates&           est,
                const Eigen::MatrixXd&     vcov);

}  // namespace magmaan::measures::effects
