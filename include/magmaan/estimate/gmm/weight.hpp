#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/expected.hpp"

// Structured per-block moment weight for the moment-quadratic family.
//
// Every member of that family — ULS / DWLS / WLS / GLS — minimizes
// F(θ) = ½·r(θ)ᵀ W r(θ) over the stacked [mean ; vech(cov)] moment vector, and
// differs only in W. But W is never an arbitrary q×q matrix:
//
//   ULS   → identity
//   DWLS  → diagonal (diag Γ̂ from the polychoric/NT moment ACOV)
//   GLS   → ½ Dᵀ(A⁻¹ ⊗ A⁻¹) D on the cov block, A⁻¹ on the mean block, A = S
//   WLS   → genuinely dense (Γ̂⁻¹, ADF)
//
// and the Fisher/IRLS reweight is the GLS form with A = Σ(θ_k) instead of S.
// Storing all four as a dense q×q with q = p + p(p+1)/2 turns an O(q) scaling
// or an O(p³) triangular solve into an O(q²) GEMM against the Jacobian on
// every gradient evaluation, and costs q² doubles to hold a matrix that is
// structurally identity, diagonal, or Kronecker. At p = 60 that is a dense
// 1890×1890 factor (29 MB/block) multiplied against a 1890×130 Jacobian every
// iteration, plus a one-off O(q³) Cholesky.
//
// This is the same argument, and the same Identity/Diagonal/Dense design, that
// `src/estimate/detail_whiten_factor.hpp` already applies on the ordinal path;
// the continuous path never got it, and neither path had the Kronecker case.
//
// `BlockWeight` represents the weight W itself (unlike `detail::WhitenFactor`,
// which represents the factor F) and stores whatever internal form makes
// whitening cheapest. `t_apply(s, X)` is the whitening form s·Fᵀ X with
// F Fᵀ = W, which is the only operation the hot path needs.

namespace magmaan::estimate::gmm {

class BlockWeight {
 public:
  enum class Kind { Identity, Diagonal, Dense, NormalTheory };

  BlockWeight() = default;

  // W = I_n (ULS).
  static BlockWeight identity(Eigen::Index n) {
    BlockWeight w;
    w.kind_ = Kind::Identity;
    w.dim_  = n;
    return w;
  }

  // W = diag(d), d > 0 (DWLS). Stores √d, so whitening is a row scaling.
  static BlockWeight diagonal(const Eigen::VectorXd& d) {
    BlockWeight w;
    w.kind_ = Kind::Diagonal;
    w.dim_  = d.size();
    w.diag_ = d.cwiseMax(0.0).cwiseSqrt();
    return w;
  }

  // W dense and symmetric PSD (WLS / ADF / DLS). Factors once here — the same
  // Cholesky-with-clamped-eigendecomposition fallback `gmm::residuals` used to
  // do per call — so the hot path only ever sees the triangular factor.
  static fit_expected<BlockWeight> dense(const Eigen::MatrixXd& W,
                                         FitError::Kind err_kind,
                                         const std::string& detail) {
    BlockWeight w;
    w.kind_ = Kind::Dense;
    w.dim_  = W.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(W);
    if (llt.info() == Eigen::Success) {
      w.dense_ = Eigen::MatrixXd(llt.matrixL());
      return w;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(W);
    if (es.info() != Eigen::Success) {
      return std::unexpected(
          FitError{err_kind, detail + ": weight matrix eigendecomposition failed"});
    }
    Eigen::VectorXd vals = es.eigenvalues();
    const double tol = 1e-10 * std::max<double>(1.0, W.cwiseAbs().maxCoeff());
    for (Eigen::Index i = 0; i < vals.size(); ++i) {
      if (vals(i) < -tol) {
        return std::unexpected(FitError{
            err_kind, detail + ": weight matrix is not positive semidefinite"});
      }
      vals(i) = std::sqrt(std::max(0.0, vals(i)));
    }
    w.dense_ = es.eigenvectors() * vals.asDiagonal();
    return w;
  }

  // The normal-theory / expected-information weight built from a PD p×p matrix
  // A — A = S gives GLS, A = Σ(θ_k) gives the Fisher/IRLS reweight:
  //
  //   W = blockdiag( A⁻¹ , ½·Dᵀ(A⁻¹ ⊗ A⁻¹) D )
  //
  // over [mean ; vech(cov)] (the mean block present only when `has_means`).
  // Stores chol(A) — p×p — and never forms the q×q. Matches
  // `symmetric_vech_gls_weight` scaled by ½ on the cov block and A⁻¹ on the
  // mean block, which is what `normal_theory_weight` materializes today.
  static fit_expected<BlockWeight> normal_theory(const Eigen::MatrixXd& A,
                                                 bool has_means,
                                                 FitError::Kind err_kind,
                                                 const std::string& detail) {
    const Eigen::Index p = A.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(FitError{
          err_kind, detail + ": normal-theory weight source is not positive definite"});
    }
    BlockWeight w;
    w.kind_      = Kind::NormalTheory;
    w.chol_      = Eigen::MatrixXd(llt.matrixL());
    w.has_means_ = has_means;
    w.dim_       = (has_means ? p : 0) + p * (p + 1) / 2;
    return w;
  }

  Kind kind() const { return kind_; }
  Eigen::Index rows() const { return dim_; }
  Eigen::Index cols() const { return dim_; }
  bool is_identity() const { return kind_ == Kind::Identity; }

  bool valid(Eigen::Index n) const {
    if (dim_ != n) return false;
    switch (kind_) {
      case Kind::Identity: return true;
      case Kind::Diagonal: return diag_.size() == n && diag_.allFinite();
      case Kind::Dense:
        return dense_.rows() == n && dense_.cols() == n && dense_.allFinite();
      case Kind::NormalTheory: return chol_.allFinite();
    }
    return false;
  }

  // s · Fᵀ X with F Fᵀ = W, for X a vector or a matrix.
  //
  //   Identity      O(q·ncol)   copy
  //   Diagonal      O(q·ncol)   row scaling
  //   Dense         O(q²·ncol)  GEMM  — the only branch that reaches one
  //   NormalTheory  O(p³·ncol)  two triangular solves per column
  //
  // For the cov block NormalTheory applies d ↦ scaled-vech(L⁻¹ unvech(d) L⁻ᵀ)
  // with diagonal entries scaled by 1/√2, so that ‖·‖² reproduces
  // ½·tr(A⁻¹ D A⁻¹ D); the mean block applies d ↦ L⁻¹ d.
  template <typename Derived>
  typename Derived::PlainObject t_apply(
      double s, const Eigen::MatrixBase<Derived>& X) const {
    switch (kind_) {
      case Kind::Identity: return s * X;
      case Kind::Diagonal: return s * (diag_.asDiagonal() * X);
      case Kind::Dense:    return s * (dense_.transpose() * X);
      case Kind::NormalTheory: return t_apply_nt(s, X);
    }
    return typename Derived::PlainObject();
  }

  // Materialize W. Only for callers that genuinely need the dense form
  // (Satorra-Bentler corrections, Γ spectra); never on a gradient path.
  Eigen::MatrixXd to_dense() const {
    switch (kind_) {
      case Kind::Identity:
        return Eigen::MatrixXd::Identity(dim_, dim_);
      case Kind::Diagonal:
        return Eigen::MatrixXd(diag_.cwiseProduct(diag_).asDiagonal());
      case Kind::Dense:
        return dense_ * dense_.transpose();
      case Kind::NormalTheory:
        return nt_to_dense();
      default: break;
    }
    return Eigen::MatrixXd();
  }

 private:
  template <typename Derived>
  typename Derived::PlainObject t_apply_nt(
      double s, const Eigen::MatrixBase<Derived>& X) const {
    using Plain = typename Derived::PlainObject;
    const Eigen::Index p    = chol_.rows();
    const Eigen::Index off  = has_means_ ? p : 0;
    const Eigen::Index ncol = X.cols();
    const double inv_sqrt2  = 0.7071067811865475244;
    const auto Lv = chol_.template triangularView<Eigen::Lower>();

    Plain out(dim_, ncol);
    Eigen::MatrixXd D(p, p);
    Eigen::MatrixXd M(p, p);
    for (Eigen::Index c = 0; c < ncol; ++c) {
      if (has_means_) {
        Eigen::VectorXd dm = X.col(c).head(p);
        Lv.solveInPlace(dm);
        out.col(c).head(p) = s * dm;
      }
      Eigen::Index k = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j; i < p; ++i, ++k) {
          const double v = X.col(c)(off + k);
          D(i, j) = v;
          D(j, i) = v;
        }
      }
      // M = L⁻¹ D L⁻ᵀ, via Y = L⁻¹D then M = L⁻¹Yᵀ (D, M symmetric).
      Lv.solveInPlace(D);
      M = D.transpose();
      Lv.solveInPlace(M);
      k = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j; i < p; ++i, ++k) {
          out(off + k, c) = s * (i == j ? inv_sqrt2 * M(i, j) : M(i, j));
        }
      }
    }
    return out;
  }

  Eigen::MatrixXd nt_to_dense() const {
    const Eigen::Index p     = chol_.rows();
    const Eigen::Index pstar = p * (p + 1) / 2;
    const Eigen::Index off   = has_means_ ? p : 0;
    const auto Lv = chol_.template triangularView<Eigen::Lower>();
    Eigen::MatrixXd Ainv =
        Lv.solve(Eigen::MatrixXd::Identity(p, p));
    Ainv = Ainv.transpose() * Ainv;  // A⁻¹ = L⁻ᵀL⁻¹

    Eigen::MatrixXd W(dim_, dim_);
    W.setZero();
    if (has_means_) W.block(0, 0, p, p) = Ainv;

    // ½·tr(A⁻¹ E_{k1} A⁻¹ E_{k2}) with E_k the symmetric basis matrix, i.e.
    // exactly ½·symmetric_vech_gls_weight(A⁻¹).
    Eigen::MatrixXd E1 = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(p, p);
    Eigen::Index k1 = 0;
    for (Eigen::Index c1 = 0; c1 < p; ++c1) {
      for (Eigen::Index r1 = c1; r1 < p; ++r1, ++k1) {
        E1.setZero();
        E1(r1, c1) = 1.0;
        E1(c1, r1) = 1.0;
        const Eigen::MatrixXd A1 = Ainv * E1 * Ainv;
        Eigen::Index k2 = 0;
        for (Eigen::Index c2 = 0; c2 < p; ++c2) {
          for (Eigen::Index r2 = c2; r2 < p; ++r2, ++k2) {
            E2.setZero();
            E2(r2, c2) = 1.0;
            E2(c2, r2) = 1.0;
            W(off + k1, off + k2) = 0.5 * (A1 * E2).trace();
          }
        }
      }
    }
    (void)pstar;
    return W;
  }

  Kind kind_ = Kind::Identity;
  Eigen::Index dim_ = 0;
  Eigen::VectorXd diag_;   // Diagonal: √diag(W)
  Eigen::MatrixXd dense_;  // Dense: lower factor L, W = L Lᵀ
  Eigen::MatrixXd chol_;   // NormalTheory: chol(A) lower
  bool has_means_ = false;
};

// NOTE: `gmm::Weight` is still `std::vector<Eigen::MatrixXd>`, declared in
// moment_quadratic.hpp. Retyping it to `std::vector<BlockWeight>` is a
// separate change — it touches the five weight-producing functions, two
// dense-indexing consumers (`robust/weighted_inference.cpp`,
// `estimate/fiml.cpp`), and the R boundary (`r-package/src/prepared.hpp`).
// Tracked in docs/backlog/todo.md under the continuous-whitening entry.

}  // namespace magmaan::estimate::gmm
