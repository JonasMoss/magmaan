#pragma once

#include "detail_linalg.hpp"

namespace magmaan::estimate::gmm {

// Fourth-moment coordinates inherit products of observed-variable units.
// Gate and invert their correlation-scaled matrix, then transport the inverse
// back. Diagnostics describe that scaled gate. This changes neither Gamma nor
// its rank and introduces no ridge or pseudoinverse.
inline detail::SymInverseResult equilibrated_weight_inverse(const Eigen::MatrixXd& gamma) {
  const auto scaled=detail::equilibrate_symmetric(gamma);
  auto inverse=detail::symmetric_inverse_pd_gated(scaled.matrix);
  if(inverse.ok) {
    inverse.inverse=scaled.scale.asDiagonal()*inverse.inverse*scaled.scale.asDiagonal();
    if(!inverse.inverse.allFinite()) inverse.ok=false;
  }
  return inverse;
}

}  // namespace magmaan::estimate::gmm
