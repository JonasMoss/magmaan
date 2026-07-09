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
  TriadLeastSquares,           // sum_jk r_jk r_ij r_ik / sum_jk r_jk^2
  AnchorTriadLeastSquares,     // same, adding cross-block anchor rows
  GmmBlock,                    // triad-moment GMM, independent block weights
  GmmFull,                     // triad-moment GMM, one joint selected weight
};

struct HCommunalityResult {
  Eigen::VectorXd h2;      // correlation-scale communalities
  Eigen::VectorXd h_diag;  // covariance-scale diagonal of H
  Eigen::MatrixXd H;       // S with diag(S) replaced by h_diag
};

// Linear least-squares/GMM representation of the triad communality moments
// A h2 ~= b on the correlation scale. W is the criterion weight for the rows of
// A. It is a function of S only; no fourth-moment Gamma enters here.
struct CommunalitySystem {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::MatrixXd W;
};

struct CommunalitySystemDirection {
  CommunalitySystem system;
  Eigen::MatrixXd dA;
  Eigen::VectorXd db;
  Eigen::MatrixXd dW;
};

const char* communality_method_name(CommunalityMethod method);

fit_expected<CommunalitySystem>
communality_system(const Eigen::MatrixXd& S,
                   const std::vector<std::int32_t>& block_of_indicator,
                   CommunalityMethod method);

fit_expected<CommunalitySystemDirection>
communality_system_directional(
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& dS,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method);

fit_expected<Eigen::VectorXd>
solve_communality_system(const CommunalitySystem& system,
                         const Eigen::MatrixXd& R_h2,
                         const Eigen::VectorXd& r_h2);

fit_expected<Eigen::VectorXd>
estimate_h2_communalities(const Eigen::MatrixXd& S,
                          const std::vector<std::int32_t>& block_of_indicator,
                          CommunalityMethod method);

// Directional derivative of `estimate_h2_communalities(S, ..., method)` in the
// symmetric covariance direction `dS`. This is the regular interior derivative:
// GMM weight derivatives are included when the normality-implied triad weight is
// full rank; rank-changing pseudo-inverse points return an error so callers can
// fall back to finite differences.
fit_expected<Eigen::VectorXd>
estimate_h2_communalities_directional(
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& dS,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method);

// Batched regular-interior Jacobian of
// `estimate_h2_communalities(S, ..., method)`, with columns ordered as the
// lower-triangle column-major vech(S). The current batched path covers the
// least-squares/GMM rules used by the configural GLS-aligned Guttman map:
// triad_ls, anchor_triad_ls, and gmm_block. Unsupported methods return an
// error so callers can fall back to directional derivatives or finite
// differences.
fit_expected<Eigen::MatrixXd>
estimate_h2_communalities_jacobian(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method);

// Batched regular-interior Jacobian of the residual-restricted
// least-squares/GMM communality map. R_h2 and r_h2 are the linear constraints
// imposed on the correlation-scale h2 vector at the current S. Nonzero
// coefficients are treated as residual-variance transformed rows, so
// dR_ai/dS_ii = R_ai/S_ii and dr_a/dS_ii follows the same convention used by
// the restricted Guttman estimator; this helper is not for fixed, S-invariant
// h2 constraints.
fit_expected<Eigen::MatrixXd>
estimate_h2_communalities_constrained_jacobian(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2);

fit_expected<Eigen::VectorXd>
estimate_h2_communalities_constrained(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2);

fit_expected<HCommunalityResult>
estimate_h_communalities(const Eigen::MatrixXd& S,
                         const std::vector<std::int32_t>& block_of_indicator,
                         CommunalityMethod method);

fit_expected<HCommunalityResult>
estimate_h_communalities_constrained(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2);

}  // namespace magmaan::estimate::frontier
