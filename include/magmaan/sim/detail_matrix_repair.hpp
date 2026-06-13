#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/sim/norta.hpp"  // BivariateCopulaCorrelationRepairKind

namespace magmaan::sim {

// Eigenvalue-floor repair of a calibrated correlation matrix. Shared by the
// bivariate-copula calibration (src/sim/norta.cpp) and the ordinal-correlation
// calibration (src/sim/ordinal_correlation.cpp); the definition is single-
// sourced in norta.cpp. `repaired` is false when the raw matrix already meets
// `min_eigenvalue` or when the kind is None.
struct MatrixRepairResult {
  Eigen::MatrixXd corr;
  double raw_min_eigenvalue = 0.0;
  double repaired_min_eigenvalue = 0.0;
  double ridge = 0.0;
  double shrinkage = 0.0;
  bool repaired = false;
};

sim_expected<MatrixRepairResult>
repair_correlation_matrix_if_requested(
    const Eigen::Ref<const Eigen::MatrixXd>& corr,
    BivariateCopulaCorrelationRepairKind kind,
    double min_eigenvalue,
    const char* caller);

}  // namespace magmaan::sim
