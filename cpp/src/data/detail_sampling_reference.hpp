#pragma once

#include "magmaan/data/ordinal.hpp"

namespace magmaan::data::validation {
post_expected<Eigen::MatrixXd> mixed_sampling_jacobian_for_validation(
    const Eigen::MatrixXd& X, const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels, const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean, const Eigen::MatrixXd& R, bool sparse,
    double h_rel = 1e-5);
// Dense pre-optimization FD implementation, retained for numerical gates.
post_expected<Eigen::MatrixXd> mixed_sampling_influence_dense_reference(
    const Eigen::MatrixXd& X, const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels, const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean, const Eigen::MatrixXd& R, double h_rel = 1e-5);
}  // namespace magmaan::data::validation
