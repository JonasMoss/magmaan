#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include "magmaan/expected.hpp"
#include "magmaan/error.hpp"
#include "detail_whiten_factor.hpp"

namespace magmaan::detail {

struct ThetaThresholdProfile {
  WhitenFactor factor;
  Eigen::MatrixXd threshold_from_corr;
};

// F F' is the full moment weight. Eliminate freely varying standardized
// thresholds once: QR supplies a square root of the Schur complement without
// subtracting nearly equal Gram matrices. Diagonal weights need no elimination,
// and a diagonal F keeps a diagonal Schur complement, so the returned operator
// stays diagonal and never becomes a dense multiply on the gradient path.
inline fit_expected<ThetaThresholdProfile> theta_threshold_profile(
    const WhitenFactor& F, Eigen::Index nth, bool diagonal) {
  const Eigen::Index dim = F.rows();
  if (nth < 0 || nth > dim || !F.valid(dim)) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "theta threshold profile: invalid weight factor", 0, 0.0});
  }
  const Eigen::Index nc = dim - nth;
  ThetaThresholdProfile out;
  out.threshold_from_corr = Eigen::MatrixXd::Zero(nth, nc);
  if (diagonal || nth == 0) {
    switch (F.kind()) {
      case WhitenFactor::Kind::Identity:
        out.factor = WhitenFactor::identity(nc);
        return out;
      case WhitenFactor::Kind::Diagonal:
        out.factor = WhitenFactor::diagonal(F.diag().tail(nc));
        return out;
      case WhitenFactor::Kind::Dense:
        out.factor = WhitenFactor::dense(
            F.dense_matrix().bottomRightCorner(nc, nc).transpose());
        return out;
    }
  }
  const Eigen::MatrixXd Fd = F.to_dense();
  const Eigen::MatrixXd H = Fd.topRows(nth).transpose();
  const Eigen::MatrixXd G = Fd.bottomRows(nc).transpose();
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(H);
  if (qr.rank() != nth) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "theta threshold profile: rank-deficient threshold weight", 0, 0.0});
  }
  out.threshold_from_corr = qr.solve(G);
  const Eigen::MatrixXd rotated = qr.householderQ().adjoint() * G;
  out.factor = WhitenFactor::dense(rotated.bottomRows(nc));
  return out;
}

}  // namespace magmaan::detail
