#pragma once

#include <Eigen/Core>
#include "magmaan/expected.hpp"

namespace magmaan::estimate::frontier {

// units and objective_diagonal use the same equality-reduced coordinates.
// The diagonal is the Gauss-Newton Hessian of the actual backend scalar
// objective (including its weights and scalar factors), not NT information.
// Invalid/degenerate diagonal entries retain the corresponding sample unit.
// Two-sided multipliers are clamped to [1e-6, 1e6]; downward-only to [1e-3, 1].
// Empty inputs are valid. Invalid units, sizes or final scales fail as values.
fit_expected<Eigen::VectorXd> objective_coordinate_scale(
    const Eigen::VectorXd& units, const Eigen::VectorXd& objective_diagonal,
    bool downward_only);

}  // namespace magmaan::estimate::frontier
