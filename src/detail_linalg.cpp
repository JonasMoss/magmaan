#include "detail_linalg.hpp"

#include <algorithm>

#include <Eigen/Eigenvalues>

namespace magmaan::detail {

SymInverseResult symmetric_inverse_pd_gated(const Eigen::MatrixXd& A) {
  SymInverseResult r;
  r.dim = A.rows();

  if (A.rows() != A.cols() || !A.allFinite()) {
    return r;  // finite == false, ok == false
  }
  r.finite = true;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) {
    return r;  // decomposed == false, ok == false
  }
  r.decomposed = true;

  const Eigen::VectorXd& evals = es.eigenvalues();
  r.max_eval = evals.maxCoeff();
  r.min_eval = evals.minCoeff();
  r.tol      = 1e-10 * std::max(1.0, r.max_eval);
  r.rank     = (evals.array() > r.tol).count();
  r.rcond    = r.max_eval > 0.0 ? r.min_eval / r.max_eval : 0.0;

  if (r.min_eval <= r.tol) {
    return r;  // rank deficient, ok == false
  }

  r.inverse = es.eigenvectors() * evals.cwiseInverse().asDiagonal() *
              es.eigenvectors().transpose();
  r.ok = true;
  return r;
}

}  // namespace magmaan::detail
