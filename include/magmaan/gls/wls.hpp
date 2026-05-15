#pragma once

#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::gls {

using data::SampleStats;

// Explicit-weight weighted least squares discrepancy.
//
// `weights[b]` is a full symmetric positive-definite matrix aligned to the
// stacked observed-moment vector for block b:
//
//   [ mean ; vech(cov) ]  when that fit has means,
//   [ vech(cov) ]         otherwise.
//
// The residual/Jacobian interface applies the Cholesky whitening of each
// block's weight and the usual sqrt(n_b / N) multi-block scaling.
struct WLS {
  static constexpr std::string_view name = "WLS";

  WLS() = default;
  explicit WLS(std::vector<Eigen::MatrixXd> weights_in)
      : weights(std::move(weights_in)) {}

  bool is_ml() const noexcept { return false; }

  std::vector<Eigen::MatrixXd> weights;

  fit_expected<double>
  value(const SampleStats& s, const model::ImpliedMoments& m) const;

  fit_expected<Eigen::VectorXd>
  gradient(const SampleStats& s, const model::ImpliedMoments& m,
           const Eigen::MatrixXd& J,
           const Eigen::MatrixXd& Jmu = Eigen::MatrixXd()) const;

  fit_expected<Eigen::VectorXd>
  residuals(const SampleStats& s, const model::ImpliedMoments& m) const;

  fit_expected<Eigen::MatrixXd>
  residual_jacobian(const SampleStats& s, const model::ImpliedMoments& m,
                    const Eigen::MatrixXd& J_sigma,
                    const Eigen::MatrixXd& J_mu = Eigen::MatrixXd()) const;
};

}  // namespace magmaan::gls
