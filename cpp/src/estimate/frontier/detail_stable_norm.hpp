#pragma once

#include <Eigen/Core>

namespace magmaan::estimate::frontier::detail {

// Eigen 3.4's stableNorm() wraps its 4096-entry blocks in a Ref that carries
// the evaluator's alignment (EIGEN_MAX_ALIGN_BYTES, 64 here, for plain
// vectors) while it starts the blocks at the packet-default alignment. Builds
// whose packets are narrower than 64 bytes therefore trip Eigen's Debug
// alignment assertion. An unaligned Map runs the same blocked kernel without
// that claim, so the value, and the rounding model the interval bounds rely
// on, are unchanged.
inline double stable_norm(const Eigen::Ref<const Eigen::VectorXd>& v) {
  return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size()).stableNorm();
}

}  // namespace magmaan::estimate::frontier::detail
