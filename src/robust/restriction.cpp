#include "magmaan/robust/restriction.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;


namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

namespace {

Eigen::MatrixXd pinv_times(const Eigen::Ref<const Eigen::MatrixXd>& A,
                           const Eigen::Ref<const Eigen::MatrixXd>& B,
                           double tol) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd& s = svd.singularValues();
  const double scale = s.size() > 0 ? s(0) : 0.0;
  Eigen::VectorXd inv = Eigen::VectorXd::Zero(s.size());
  const double cutoff = tol * static_cast<double>(std::max(A.rows(), A.cols())) *
                        std::max(1.0, scale);
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    if (s(i) > cutoff) inv(i) = 1.0 / s(i);
  }
  return svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose() * B;
}

}  // namespace

post_expected<RestrictionAlpha>
restriction_alpha_from_K(const EqConstraints& K_H1,
                         const EqConstraints& K_H0,
                         double               tol_range_F,
                         double               tol_singular) {
  const Eigen::Index npar = K_H1.Kmat.rows();
  const Eigen::Index r1   = K_H1.Kmat.cols();
  const Eigen::Index r0   = K_H0.Kmat.cols();

  if (K_H0.Kmat.rows() != npar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: K_H1.npar (" + std::to_string(npar) +
        ") != K_H0.npar (" + std::to_string(K_H0.Kmat.rows()) + ")"));
  }
  if (r0 > r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: r_H0 (" + std::to_string(r0) +
        ") > r_H1 (" + std::to_string(r1) + ") — H0 has *more* free directions"
        " than H1, so the pair is not nested in the expected order."));
  }

  const Eigen::Index m = r1 - r0;
  if (m == 0) {
    // Degenerate but valid: H0 ≡ H1 (no extra restriction). Return an empty
    // (0 × r1) restriction; downstream Satorra returns T_diff = 0, df = 0.
    return RestrictionAlpha{
        /*A=*/Eigen::MatrixXd::Zero(0, r1),
        /*b=*/Eigen::VectorXd::Zero(0)};
  }

  // 1. Project K_H0 onto K_H1's column space.  Both K matrices are
  //    orthonormal-column (built by SVD or as 0/1 group-membership with
  //    unit-norm columns after the QR step in `build_eq_constraints`), so
  //    when range(K_H0) ⊆ range(K_H1) the projector reduces to
  //
  //        M = K_H1ᵀ · K_H0     (r1 × r0,  M is then column-orthonormal).
  //
  //    We use a QR solve instead of plain matrix multiply because either K
  //    can come out of `build_eq_constraints` non-strictly-orthonormal in the
  //    pure-merge case (the 0/1 K has columns of unit norm only after the
  //    helper rescales).
  Eigen::HouseholderQR<Eigen::MatrixXd> qr_K1(K_H1.Kmat);
  const Eigen::MatrixXd M = qr_K1.solve(K_H0.Kmat);  // r1 × r0

  // Range-inclusion check: ‖K_H1·M − K_H0‖_F should be ≈ 0 when H0 nests
  // inside H1.  Tolerance is absolute, scaled to be robust against the
  // numerical noise that JacobiSVD introduces in K_H0.
  const double inclusion_resid = (K_H1.Kmat * M - K_H0.Kmat).norm();
  if (inclusion_resid > tol_range_F) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: range(K_H0) is not contained in "
        "range(K_H1) — residual ‖K_H1·M − K_H0‖_F = " +
        std::to_string(inclusion_resid) +
        " exceeds tol_range_F = " + std::to_string(tol_range_F) +
        ". H0 is not nested in H1."));
  }

  // 2. Build A_α as the orthonormal basis of null(Mᵀ) inside R^{r1}.
  //    Equivalently: the left-singular vectors of Mᵀ corresponding to its
  //    *small* singular values (i.e. its (r1 − r0) trailing left singular
  //    vectors), reshaped as rows of A.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU);
  const Eigen::MatrixXd& U = svd.matrixU();        // r1 × r1
  const Eigen::VectorXd& sv = svd.singularValues();// length min(r1,r0) = r0

  // Defensive: count actual zero singular values.  In pathological cases the
  // user-provided K may be near-rank-deficient and `m` may not match the
  // null-space dimension we predicted from `r1 - r0`.
  Eigen::Index n_zero = r1 - r0;  // expected
  for (Eigen::Index k = 0; k < sv.size(); ++k) {
    if (sv(k) < tol_singular) ++n_zero;
  }
  if (n_zero != m) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: null-space dimension " +
        std::to_string(n_zero) + " from SVD does not match r1−r0 = " +
        std::to_string(m) + " — K_H1 or K_H0 likely rank-deficient."));
  }

  // The last m columns of U span null(Mᵀ); rows of A are those vectors.
  Eigen::MatrixXd A(m, r1);
  A = U.rightCols(m).transpose();

  // 3. Inhomogeneous shift b = A · b₀ where K_H1·b₀ = θ₀_H0 − θ₀_H1
  //    (least-squares solve; b₀ is unique up to a vector in null(K_H1) which
  //    A annihilates anyway, so b is well-defined regardless).
  Eigen::VectorXd b = Eigen::VectorXd::Zero(m);
  // Only bother if either θ₀ is non-zero (pure-merge case has θ₀ ≡ 0).
  if (K_H1.theta0.size() == npar && K_H0.theta0.size() == npar &&
      (K_H0.theta0 - K_H1.theta0).squaredNorm() > 0.0) {
    const Eigen::VectorXd b0 = qr_K1.solve(K_H0.theta0 - K_H1.theta0);
    b = A * b0;
  }

  return RestrictionAlpha{std::move(A), std::move(b)};
}

post_expected<Eigen::MatrixXd>
restriction_alpha_delta_from_jacobians(
    const Eigen::Ref<const Eigen::MatrixXd>& Pi_H1_alpha,
    const Eigen::Ref<const Eigen::MatrixXd>& Pi_H0_alpha,
    int                                      expected_m,
    double                                   tol_singular) {
  if (Pi_H1_alpha.rows() != Pi_H0_alpha.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_delta_from_jacobians: H1/H0 moment Jacobians have "
        "different row counts"));
  }
  const Eigen::Index r1 = Pi_H1_alpha.cols();
  if (expected_m < 0 || expected_m > r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_delta_from_jacobians: expected restriction rank m = " +
        std::to_string(expected_m) + " is outside [0, r_H1 = " +
        std::to_string(r1) + "]"));
  }
  if (expected_m == 0) {
    return Eigen::MatrixXd::Zero(0, r1);
  }

  const Eigen::MatrixXd H = pinv_times(Pi_H1_alpha, Pi_H0_alpha, tol_singular);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU);
  const Eigen::VectorXd& s = svd.singularValues();
  const double scale = s.size() > 0 ? s(0) : 0.0;
  const double cutoff = tol_singular *
                        static_cast<double>(std::max(H.rows(), H.cols())) *
                        std::max(1.0, scale);
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    if (s(i) > cutoff) ++rank;
  }
  const Eigen::Index null_dim = r1 - rank;
  if (null_dim != expected_m) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_delta_from_jacobians: null-space dimension " +
        std::to_string(null_dim) + " from H = pinv(Π_H1)Π_H0 does not match "
        "df_H0−df_H1 = " + std::to_string(expected_m)));
  }
  return svd.matrixU().rightCols(expected_m).transpose();
}

}  // namespace magmaan::robust
