#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "latva/expected.hpp"
#include "latva/fit/fit.hpp"            // Estimates
#include "latva/parse/flat_partable.hpp"
#include "latva/partable/partable.hpp"

namespace latva::fit {

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
// and delta-method SE per entry.
//
// `pt` is needed to resolve each `Param` identifier in the expression: a
// user-supplied label like "a" maps to a free θ index. v0 supports
// resolution against rows in `pt` that carry a matching `label`; chained
// references (one := referring to another) and references to fixed
// params via their `plabel` aren't supported yet.
post_expected<DefinedParams>
compute_defined(const parse::FlatPartable& flat,
                const partable::ParTable&  pt,
                const Estimates&           est,
                const Eigen::MatrixXd&     vcov);

}  // namespace latva::fit
