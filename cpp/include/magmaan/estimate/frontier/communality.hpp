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
  TriadMean,                  // mean_jk r_ij r_ik / r_jk         (was ar)
  TriadPooled,                // sum_jk r_ij r_ik / sum_jk r_jk   (was rs)
  TriadLeastSquares,          // sum_jk r_jk r_ij r_ik / sum_jk r_jk^2
  ExtendedTriadLeastSquares,  // triad LS + cross-block triads    (was anchor_triad_ls)
  TriadWls,                   // efficiency-weighted triad LS, per block   (was gmm_block)
  TriadWlsJoint,              // efficiency-weighted triad LS, one joint weight (was gmm_full)
};

enum class AdmissibilityPolicy : std::uint8_t { Raw, Hard, Soft };

struct AdmissibilityConfig {
  AdmissibilityPolicy policy = AdmissibilityPolicy::Raw;
  double margin = 1e-4;
  double beta0 = 1.0;
  double rate_exp = 0.5;
};

struct AdmissibilityClamp {
  AdmissibilityPolicy policy = AdmissibilityPolicy::Raw;
  double margin = 1e-4;
  double beta = 0.0;
};

struct HCommunalityResult {
  Eigen::VectorXd h2;      // correlation-scale communalities
  Eigen::VectorXd h_diag;  // covariance-scale diagonal of H
  Eigen::MatrixXd H;       // S with diag(S) replaced by h_diag
  Eigen::Index n_active_clamped = 0;
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
const char* admissibility_policy_name(AdmissibilityPolicy policy);

fit_expected<AdmissibilityClamp>
resolve_admissibility(const AdmissibilityConfig& config, double n_obs);

double admissibility_clamp_value(double x, const AdmissibilityClamp& clamp);
double admissibility_clamp_deriv(double x, const AdmissibilityClamp& clamp);

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

// Batched right-hand side for the regular-interior derivative of the
// unconstrained normal equations underlying `system`, evaluated at h2. Columns
// are ordered as vech(S). The result is dc - dG h2, before any active
// constraints are added by the KKT solve. This is an internal frontier building
// block for restricted Guttman Jacobians that share a joint constrained solve
// across blocks.
fit_expected<Eigen::MatrixXd>
communality_system_jacobian_rhs(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::VectorXd& h2);

fit_expected<Eigen::VectorXd>
estimate_h2_communalities(const Eigen::MatrixXd& S,
                          const std::vector<std::int32_t>& block_of_indicator,
                          CommunalityMethod method,
                          const AdmissibilityClamp& clamp = {});

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
    CommunalityMethod method,
    const AdmissibilityClamp& clamp = {});

// Batched regular-interior Jacobian of
// `estimate_h2_communalities(S, ..., method)`, with columns ordered as the
// lower-triangle column-major vech(S). The current batched path covers the
// least-squares/GMM rules used by the configural GLS-aligned Guttman map:
// triad_ls, extended_triad_ls, and triad_wls. Unsupported methods return an
// error so callers can fall back to directional derivatives or finite
// differences.
fit_expected<Eigen::MatrixXd>
estimate_h2_communalities_jacobian(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const AdmissibilityClamp& clamp = {});

// Batched regular-interior Jacobian of the residual-restricted
// least-squares/GMM communality map. R_h2 and r_h2 are the linear constraints
// imposed on the correlation-scale h2 vector at the current S. Nonzero
// coefficients are treated as residual-variance transformed rows, so
// dR_ai/dS_ii = R_ai/S_ii and dr_a/dS_ii follows the same convention used by
// the restricted Guttman estimator; this helper is not for fixed, S-invariant
// h2 constraints. A non-Raw admissibility map is applied after the KKT solve;
// boundary values therefore need not retain the raw residual-equality rows.
fit_expected<Eigen::MatrixXd>
estimate_h2_communalities_constrained_jacobian(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2,
    const AdmissibilityClamp& clamp = {});

fit_expected<Eigen::VectorXd>
estimate_h2_communalities_constrained(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2,
    const AdmissibilityClamp& clamp = {});

fit_expected<HCommunalityResult>
estimate_h_communalities(const Eigen::MatrixXd& S,
                         const std::vector<std::int32_t>& block_of_indicator,
                         CommunalityMethod method,
                         const AdmissibilityClamp& clamp = {});

fit_expected<HCommunalityResult>
estimate_h_communalities_constrained(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2,
    const AdmissibilityClamp& clamp = {});

}  // namespace magmaan::estimate::frontier
