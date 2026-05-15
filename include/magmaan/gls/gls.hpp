#pragma once

#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::gls {

using data::SampleStats;

// Generalized least squares discrepancy for continuous complete-data moments.
//
// Covariance blocks use the normal-theory GLS quadratic form
//
//   F_b,cov = 1/2 * vech(Σ_b - S_b)' W_b vech(Σ_b - S_b)
//
// where W_b is induced by S_b^{-1}: W_kl = tr(S_b^{-1} E_k S_b^{-1} E_l)
// for symmetric lower-vech basis matrices E_k. Mean rows, when present, use
// S_b^{-1}. Residual rows are the Cholesky-whitened stacked moments:
// means first per layout, then lower-triangle column-major vech.
struct GLS {
  static constexpr std::string_view name = "GLS";
  bool is_ml() const noexcept { return false; }

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
