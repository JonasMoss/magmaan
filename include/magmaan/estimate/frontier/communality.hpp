#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::estimate::frontier {

// Closed-form estimators of the diagonal H_ii = Var(x_i explained by its
// declared factor), represented on the correlation scale as h2_i = H_ii / S_ii.
// The loading pattern is supplied as one block id per observed variable; each
// block is one simple-structure factor and needs at least three indicators.
enum class CommunalityMethod : std::uint8_t {
  AverageRatio,                // mean_jk r_ij r_ik / r_jk
  RatioOfSums,                 // sum_jk r_ij r_ik / sum_jk r_jk
  InstrumentalLeastSquares,    // sum_jk r_jk r_ij r_ik / sum_jk r_jk^2
  GmmBlock,                    // triad-moment GMM, independent block weights
  GmmFull,                     // triad-moment GMM, one joint selected weight
};

struct HCommunalityResult {
  Eigen::VectorXd h2;      // correlation-scale communalities
  Eigen::VectorXd h_diag;  // covariance-scale diagonal of H
  Eigen::MatrixXd H;       // S with diag(S) replaced by h_diag
};

const char* communality_method_name(CommunalityMethod method);

fit_expected<Eigen::VectorXd>
estimate_h2_communalities(const Eigen::MatrixXd& S,
                          const std::vector<std::int32_t>& block_of_indicator,
                          CommunalityMethod method);

fit_expected<HCommunalityResult>
estimate_h_communalities(const Eigen::MatrixXd& S,
                         const std::vector<std::int32_t>& block_of_indicator,
                         CommunalityMethod method);

}  // namespace magmaan::estimate::frontier
