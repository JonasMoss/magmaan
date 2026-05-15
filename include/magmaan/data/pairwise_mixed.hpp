#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"

namespace magmaan::data {

struct PolyserialPairMlOptions {
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int    max_iter = 72;
};

struct PolyserialPairMlResult {
  double rho = 0.0;
  double negloglik = 0.0;
  int    iterations = 0;
  bool   hit_lower = false;
  bool   hit_upper = false;
};

struct PolyserialPairScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd thresholds;
};

post_expected<double>
polyserial_pair_negloglik(const Eigen::Ref<const Eigen::VectorXi>& categories,
                          const Eigen::Ref<const Eigen::VectorXd>& u,
                          const Eigen::Ref<const Eigen::VectorXd>& thresholds,
                          double rho);

post_expected<PolyserialPairMlResult>
fit_polyserial_pair_rho_ml(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    PolyserialPairMlOptions options = {});

post_expected<PolyserialPairScores>
polyserial_pair_scores(const Eigen::Ref<const Eigen::VectorXi>& categories,
                       const Eigen::Ref<const Eigen::VectorXd>& u,
                       double rho,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds);

}  // namespace magmaan::data
