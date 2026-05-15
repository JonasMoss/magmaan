#include "magmaan/estimate/weighted_inference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"

namespace magmaan::estimate {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

post_expected<Eigen::MatrixXd> inverse_sym_pd(const Eigen::MatrixXd& A,
                                              const char* what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " is not square"));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (A + A.transpose()));
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not positive definite"));
  }
  return ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
}

post_expected<Eigen::MatrixXd> symmetric_sqrt_psd(const Eigen::MatrixXd& A,
                                                  const char* what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " is not square"));
  }
  if (A.size() == 0) return Eigen::MatrixXd(A.rows(), A.cols());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " eigendecomposition failed"));
  }
  const double tol = 1e-10 * std::max<double>(1.0, A.cwiseAbs().maxCoeff());
  Eigen::VectorXd vals = es.eigenvalues();
  for (Eigen::Index i = 0; i < vals.size(); ++i) {
    if (vals(i) < -tol) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(what) + " is not positive semidefinite"));
    }
    vals(i) = std::sqrt(std::max(0.0, vals(i)));
  }
  return es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace

post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin) {
  if (blocks.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moments: no moment blocks supplied"));
  }
  if (K.rows() == 0 || K.cols() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moments: empty constraint reparameterization"));
  }
  double N_total = 0.0;
  Eigen::Index total_rows = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& blk = blocks[b];
    if (blk.n_obs <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: non-positive n_obs in block " +
              std::to_string(b)));
    }
    if (blk.jacobian.cols() != K.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: jacobian/K column mismatch in block " +
              std::to_string(b)));
    }
    if (blk.weight.rows() != blk.jacobian.rows() ||
        blk.weight.cols() != blk.jacobian.rows() ||
        blk.gamma.rows() != blk.jacobian.rows() ||
        blk.gamma.cols() != blk.jacobian.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: moment matrix shape mismatch in block " +
              std::to_string(b)));
    }
    N_total += static_cast<double>(blk.n_obs);
    total_rows += blk.jacobian.rows();
  }
  if (!(N_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moments: non-positive total sample size"));
  }

  const Eigen::Index n_alpha = K.cols();
  const int df = static_cast<int>(total_rows - n_alpha);
  if (df < 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "robust_weighted_moments: model has more reduced parameters than moments"));
  }

  Eigen::MatrixXd Dtilde(total_rows, n_alpha);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::Index off = 0;
  for (const auto& blk : blocks) {
    const Eigen::Index mb = blk.jacobian.rows();
    const double sw = std::sqrt(static_cast<double>(blk.n_obs) / N_total);
    Dtilde.block(off, 0, mb, n_alpha) = sw * blk.jacobian * K;
    W.block(off, off, mb, mb) = blk.weight;
    Gamma.block(off, off, mb, mb) = blk.gamma;
    off += mb;
  }

  Eigen::MatrixXd A = Dtilde.transpose() * W * Dtilde;
  A = 0.5 * (A + A.transpose()).eval();
  auto A_inv_or = inverse_sym_pd(A, "robust_weighted_moments bread");
  if (!A_inv_or.has_value()) return std::unexpected(A_inv_or.error());
  const Eigen::MatrixXd& A_inv = *A_inv_or;

  Eigen::MatrixXd B = Dtilde.transpose() * W * Gamma * W * Dtilde;
  B = 0.5 * (B + B.transpose()).eval();

  WeightedRobustResult out;
  Eigen::MatrixXd V_alpha = (A_inv * B * A_inv) / N_total;
  V_alpha = 0.5 * (V_alpha + V_alpha.transpose()).eval();
  out.vcov = K * V_alpha * K.transpose();
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose()).eval();
  out.se.resize(out.vcov.rows());
  const double diag_tol = 1e-12 * std::max<double>(1.0, out.vcov.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < out.se.size(); ++i) {
    const double v = out.vcov(i, i);
    out.se(i) = v >= -diag_tol ? std::sqrt(std::max(0.0, v))
                               : std::numeric_limits<double>::quiet_NaN();
  }

  out.chisq_standard = N_total * fmin;
  out.df = df;
  if (df > 0) {
    Eigen::MatrixXd U = W - W * Dtilde * A_inv * Dtilde.transpose() * W;
    U = 0.5 * (U + U.transpose()).eval();
    auto sqrtG_or = symmetric_sqrt_psd(Gamma, "robust_weighted_moments gamma");
    if (!sqrtG_or.has_value()) return std::unexpected(sqrtG_or.error());
    Eigen::MatrixXd M = (*sqrtG_or) * U * (*sqrtG_or);
    M = 0.5 * (M + M.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: U-Gamma eigendecomposition failed"));
    }
    out.eigvals = es.eigenvalues().tail(df);
    for (Eigen::Index i = 0; i < out.eigvals.size(); ++i) {
      if (out.eigvals(i) < 0.0 && out.eigvals(i) > -1e-10) out.eigvals(i) = 0.0;
    }
  } else {
    out.eigvals.resize(0);
  }
  out.satorra_bentler = fit::satorra_bentler(out.chisq_standard, out.df, out.eigvals);
  out.mean_var_adjusted = fit::mean_var_adjusted(out.chisq_standard, out.df, out.eigvals);
  out.scaled_shifted = fit::scaled_shifted(out.chisq_standard, out.df, out.eigvals);
  return out;
}

}  // namespace magmaan::estimate
