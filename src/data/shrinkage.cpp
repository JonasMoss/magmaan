#include "magmaan/data/shrinkage.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"

namespace magmaan::data {

namespace {

PostError make_err(std::string detail) {
  return PostError{PostError::Kind::NumericIssue, std::move(detail)};
}

// Smallest eigenvalue of a symmetric matrix (NaN on solver failure / empty).
double min_eigenvalue(const Eigen::MatrixXd& M) {
  if (M.rows() == 0) return std::numeric_limits<double>::quiet_NaN();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return es.eigenvalues().minCoeff();
}

}  // namespace

post_expected<ShrunkSampleStats>
shrink_sample_stats(const SampleStats& samp, CovarianceShrinkageOptions opts) {
  ShrunkSampleStats out;
  out.stats.mean = samp.mean;     // means pass through unchanged
  out.stats.n_obs = samp.n_obs;   // n_obs pass through unchanged
  out.stats.S.reserve(samp.S.size());
  out.block_diagnostics.reserve(samp.S.size());

  for (std::size_t blk = 0; blk < samp.S.size(); ++blk) {
    const Eigen::MatrixXd& S0 = samp.S[blk];
    if (S0.rows() != S0.cols()) {
      return std::unexpected(make_err(
          "shrink_sample_stats: block " + std::to_string(blk) +
          " covariance is not square"));
    }
    const Eigen::Index p = S0.rows();
    // Symmetrize defensively — a sample covariance should already be symmetric.
    const Eigen::MatrixXd S = 0.5 * (S0 + S0.transpose());

    CovarianceRepairResult diag;
    diag.raw_min_eigen = min_eigenvalue(S);

    // Resolve the shrinkage intensity.
    double s = 0.0;
    if (opts.kind != CovarianceShrinkageKind::None) {
      if (opts.estimate_intensity) {
        if (opts.kind != CovarianceShrinkageKind::DiagonalTarget) {
          return std::unexpected(make_err(
              "shrink_sample_stats: estimate_intensity is supported only for "
              "DiagonalTarget"));
        }
        if (blk >= samp.n_obs.size() || samp.n_obs[blk] < 1) {
          return std::unexpected(make_err(
              "shrink_sample_stats: estimate_intensity needs a positive "
              "n_obs for block " + std::to_string(blk)));
        }
        // Normal-theory Ledoit-Wolf / Schäfer-Strimmer intensity for shrinking
        // the correlation matrix toward the identity:
        //   λ* = Σ_{i<j} Var(r_ij) / Σ_{i<j} r_ij²,  Var(r_ij) ≈ (1-r²)²/n.
        const double n = static_cast<double>(samp.n_obs[blk]);
        double num = 0.0;
        double den = 0.0;
        for (Eigen::Index i = 0; i < p; ++i) {
          if (!(S(i, i) > 0.0)) {
            return std::unexpected(make_err(
                "shrink_sample_stats: non-positive sample variance in block " +
                std::to_string(blk)));
          }
        }
        for (Eigen::Index i = 0; i < p; ++i) {
          for (Eigen::Index j = i + 1; j < p; ++j) {
            const double r = S(i, j) / std::sqrt(S(i, i) * S(j, j));
            num += (1.0 - r * r) * (1.0 - r * r) / n;
            den += r * r;
          }
        }
        s = (den > 0.0) ? std::clamp(num / den, 0.0, 1.0) : 0.0;
      } else {
        s = std::clamp(opts.intensity, 0.0, 1.0);
      }
    }

    // Build the shrinkage target T and form S' = (1-s)·S + s·T.
    Eigen::MatrixXd Sp = S;
    if (opts.kind != CovarianceShrinkageKind::None && s > 0.0) {
      Eigen::MatrixXd T = Eigen::MatrixXd::Zero(p, p);
      switch (opts.kind) {
        case CovarianceShrinkageKind::None:
          break;
        case CovarianceShrinkageKind::Ridge:
          T = Eigen::MatrixXd::Identity(p, p);
          break;
        case CovarianceShrinkageKind::IdentityTarget: {
          const double mu = (p > 0) ? S.trace() / static_cast<double>(p) : 0.0;
          T = mu * Eigen::MatrixXd::Identity(p, p);
          break;
        }
        case CovarianceShrinkageKind::DiagonalTarget:
          T = S.diagonal().asDiagonal();
          break;
        case CovarianceShrinkageKind::ConstantCorrelation: {
          for (Eigen::Index i = 0; i < p; ++i) {
            if (!(S(i, i) > 0.0)) {
              return std::unexpected(make_err(
                  "shrink_sample_stats: non-positive sample variance in "
                  "block " + std::to_string(blk)));
            }
          }
          double rbar = 0.0;
          Eigen::Index npair = 0;
          for (Eigen::Index i = 0; i < p; ++i) {
            for (Eigen::Index j = i + 1; j < p; ++j) {
              rbar += S(i, j) / std::sqrt(S(i, i) * S(j, j));
              ++npair;
            }
          }
          if (npair > 0) rbar /= static_cast<double>(npair);
          for (Eigen::Index i = 0; i < p; ++i) {
            T(i, i) = S(i, i);
            for (Eigen::Index j = i + 1; j < p; ++j) {
              const double t = rbar * std::sqrt(S(i, i) * S(j, j));
              T(i, j) = t;
              T(j, i) = t;
            }
          }
          break;
        }
      }
      Sp = (1.0 - s) * S + s * T;
    }

    diag.min_eigen = min_eigenvalue(Sp);
    diag.intensity = s;
    diag.shrunk = (opts.kind != CovarianceShrinkageKind::None) && (s > 0.0);

    out.stats.S.push_back(std::move(Sp));
    out.block_diagnostics.push_back(diag);
  }

  return out;
}

}  // namespace magmaan::data
