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

}  // namespace magmaan::measures::frontier::reliability
