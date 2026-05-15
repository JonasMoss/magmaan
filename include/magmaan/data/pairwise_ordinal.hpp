#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::data {

struct OrdinalPairMlOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
  bool   lavaan_adjust_2x2 = true;
};

struct OrdinalPairMlResult {
  double rho = 0.0;
  double negloglik = 0.0;
  int    iterations = 0;
  bool   hit_lower = false;
  bool   hit_upper = false;
  Eigen::MatrixXd adjusted_counts;
};

struct OrdinalPairScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd threshold_i;
  Eigen::MatrixXd threshold_j;
};

post_expected<Eigen::MatrixXd>
ordinal_pair_table(const Eigen::Ref<const Eigen::VectorXi>& x_i,
                   const Eigen::Ref<const Eigen::VectorXi>& x_j,
                   std::int32_t n_levels_i,
                   std::int32_t n_levels_j);

double ordinal_bvn_rect_prob(double lo_i, double hi_i,
                             double lo_j, double hi_j,
                             double rho) noexcept;

double ordinal_bvn_rect_drho(double lo_i, double hi_i,
                             double lo_j, double hi_j,
                             double rho) noexcept;

post_expected<double>
ordinal_pair_negloglik(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                       double rho);

post_expected<OrdinalPairMlResult>
fit_ordinal_pair_rho_ml(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                        const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                        const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                        OrdinalPairMlOptions options = {});

post_expected<OrdinalPairScores>
ordinal_pair_scores(const Eigen::Ref<const Eigen::VectorXi>& x_i,
                    const Eigen::Ref<const Eigen::VectorXi>& x_j,
                    double rho,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j);

}  // namespace magmaan::data
