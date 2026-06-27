#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::measures::frontier::reliability {

enum class Coefficient : std::uint8_t {
  Alpha,
  Lambda6,
  SpearmanGuttmanOmega,
};

struct DeltaResult {
  double          value = 0.0;
  Eigen::VectorXd gradient;
  double          avar = 0.0;  // Var limit for sqrt(n)(rho_hat - rho)
  double          se    = 0.0;  // sqrt(avar / n)
};

post_expected<double>
cronbach_alpha(const Eigen::Ref<const Eigen::MatrixXd>& S);

post_expected<double>
guttman_lambda6(const Eigen::Ref<const Eigen::MatrixXd>& S);

post_expected<double>
spearman_guttman_omega(const Eigen::Ref<const Eigen::MatrixXd>& S);

post_expected<Eigen::VectorXd>
gradient(Coefficient coef, const Eigen::Ref<const Eigen::MatrixXd>& S);

post_expected<double>
value(Coefficient coef, const Eigen::Ref<const Eigen::MatrixXd>& S);

post_expected<DeltaResult>
delta_method(Coefficient coef,
             const Eigen::Ref<const Eigen::MatrixXd>& S,
             const Eigen::Ref<const Eigen::MatrixXd>& gamma,
             std::int64_t n);

// ---------------------------------------------------------------------------
// Multidimensional closed-form reliability (omega-total and omega-hierarchical)
//
// A two-stage Guttman/centroid map turns a multi-factor congeneric covariance
// into closed-form reliability with no iterated CFA. The single-factor case is
// Hancock & An (2020); these extend it to k factors. Both are smooth covariance
// functionals g(S), so inference is the ordinary delta method on full Gamma.
// ---------------------------------------------------------------------------

enum class OmegaTarget : std::uint8_t {
  Total,         // reliability of the (weighted) total score across all factors
  Hierarchical,  // share carried by a Schmid-Leiman general factor (k >= 3)
};

// Simple-structure factor model. `block[i]` in [0, k) names the single factor
// item i loads on; k is inferred as max(block)+1. `weights` (length p) applies
// to OmegaTarget::Total only; empty means unit weights. Communalities are the
// within-block Spearman (1927) ratio-of-sums (Hancock & An 2020), so each block
// needs at least 3 items; OmegaTarget::Hierarchical additionally needs k >= 3.
struct OmegaSpec {
  Eigen::VectorXi block;
  Eigen::VectorXd weights;  // empty => unit weights
};

post_expected<double>
omega_multidim(OmegaTarget target,
               const Eigen::Ref<const Eigen::MatrixXd>& S,
               const OmegaSpec& spec);

// Central finite-difference gradient over vech(S) (lower-triangle, column major).
post_expected<Eigen::VectorXd>
omega_multidim_gradient(OmegaTarget target,
                        const Eigen::Ref<const Eigen::MatrixXd>& S,
                        const OmegaSpec& spec);

post_expected<DeltaResult>
omega_multidim_delta(OmegaTarget target,
                     const Eigen::Ref<const Eigen::MatrixXd>& S,
                     const OmegaSpec& spec,
                     const Eigen::Ref<const Eigen::MatrixXd>& gamma,
                     std::int64_t n);

}  // namespace magmaan::measures::frontier::reliability
