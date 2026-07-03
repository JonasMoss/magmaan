#pragma once

// Private SAM helper: the classic two-step ("twostep") standard errors of
// Gong & Samaniego, as used by lavaan `sam(se = "twostep")`. The joint-model
// expected information at the plugged-in SAM estimates is partitioned by
// step-1 (measurement: loadings, residuals, indicator intercepts) vs step-2
// (structural: regressions, latent (co)variances, latent means) free indices:
//
//   V2   = I22.inv
//   V1   = I22.inv I21 Sigma11 I12 I22.inv
//   VCOV = V2 + V1        (V2 alone for se = "standard")
//
// Sigma11 is the block-diagonal stack of the measurement blocks' own parameter
// covariances. magmaan information is total-N scaled, so V2 = I22.inv with no
// extra 1/N (the N factors in V1 cancel), matching lavaan's unit-scaled V2+V1.

#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::frontier::detail {

struct SamSeResult {
  Eigen::MatrixXd vcov;   // n_free × n_free (joint free order)
  Eigen::VectorXd se;     // n_free (joint free order)
};

// `is_step2` and `sigma11_full` are indexed by joint free index (0-based).
// `sigma11_full` holds the block measurement vcovs placed at their joint
// indices (step-1 entries; step-2 rows/cols are ignored). `standard_only`
// selects se = "standard" (V2 only).
inline fit_expected<SamSeResult>
sam_twostep_se(const spec::LatentStructure& pt, const model::MatrixRep& rep,
               const data::SampleStats& samp, const Eigen::VectorXd& theta_joint,
               const std::vector<char>& is_step2,
               const Eigen::MatrixXd& sigma11_full, bool standard_only) {
  Estimates est_joint;
  est_joint.theta = theta_joint;
  auto info_or = inference::information_expected(pt, rep, samp, est_joint);
  if (!info_or.has_value()) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
                                    "sam_twostep_se: joint information failed: " +
                                        info_or.error().detail,
                                    0, 0.0});
  }
  const Eigen::MatrixXd& I = *info_or;
  const Eigen::Index n = I.rows();

  std::vector<Eigen::Index> s1, s2;
  for (Eigen::Index t = 0; t < n; ++t)
    (is_step2[static_cast<std::size_t>(t)] ? s2 : s1).push_back(t);
  const Eigen::Index n2 = static_cast<Eigen::Index>(s2.size());

  auto pick = [&](const Eigen::MatrixXd& A, const std::vector<Eigen::Index>& r,
                  const std::vector<Eigen::Index>& c) {
    Eigen::MatrixXd out(static_cast<Eigen::Index>(r.size()),
                        static_cast<Eigen::Index>(c.size()));
    for (std::size_t i = 0; i < r.size(); ++i)
      for (std::size_t j = 0; j < c.size(); ++j)
        out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            A(r[i], c[j]);
    return out;
  };

  const Eigen::MatrixXd I22 = pick(I, s2, s2);
  const Eigen::MatrixXd I21 = pick(I, s2, s1);
  const Eigen::MatrixXd Sigma11 = pick(sigma11_full, s1, s1);

  Eigen::MatrixXd I22inv;
  {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (I22 + I22.transpose()));
    if (ldlt.info() != Eigen::Success) {
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
                                      "sam_twostep_se: I22 not invertible", 0,
                                      0.0});
    }
    I22inv = ldlt.solve(Eigen::MatrixXd::Identity(n2, n2));
  }

  const Eigen::MatrixXd V2 = I22inv;
  Eigen::MatrixXd VCOV = V2;
  if (!standard_only) {
    const Eigen::MatrixXd V1 =
        I22inv * I21 * Sigma11 * I21.transpose() * I22inv;
    VCOV = V2 + V1;
  }

  SamSeResult out;
  out.vcov = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t i = 0; i < s1.size(); ++i)
    for (std::size_t j = 0; j < s1.size(); ++j)
      out.vcov(s1[i], s1[j]) = Sigma11(static_cast<Eigen::Index>(i),
                                       static_cast<Eigen::Index>(j));
  for (std::size_t i = 0; i < s2.size(); ++i)
    for (std::size_t j = 0; j < s2.size(); ++j)
      out.vcov(s2[i], s2[j]) =
          VCOV(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));

  out.se = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  for (Eigen::Index t = 0; t < n; ++t) {
    const double v = out.vcov(t, t);
    out.se(t) = v >= 0.0 ? std::sqrt(v) : std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

}  // namespace magmaan::estimate::frontier::detail
