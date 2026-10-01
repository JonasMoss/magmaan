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

// Frozen moment metrics across continuous, ordinal and saturated-FIML sources.
// NT is a quadratic metric here; it is not the nonlinear ML discrepancy.
enum class FixedWeightKind { Nt, Uls, Dwls, Wls, Adf = Wls, Dls };

struct FixedWeightOptions {
  double a = 0.5;  // DLS mixes Gamma: (1-a) Gamma_NT + a Gamma_observed.
};

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

  // W = diag(d), d ≥ 0 (DWLS). Keeps diag(W) for `to_dense` and √diag(W) for
  // whitening, so the hot path is a row scaling rather than a GEMM.
  static BlockWeight diagonal(const Eigen::VectorXd& d) {
    BlockWeight w;
    w.kind_   = Kind::Diagonal;
    w.dim_    = d.size();
    w.w_diag_ = d;
    w.f_diag_ = d.cwiseMax(0.0).cwiseSqrt();
    return w;
  }

  // W dense and symmetric PSD (WLS / ADF / DLS). Factors once here — the same
  // Cholesky-with-clamped-eigendecomposition fallback `gmm::residuals` used to
  // do per call — so the hot path only ever sees the triangular factor, and
  // only the factor is stored.
  //
  // `to_dense()` therefore reconstructs L Lᵀ rather than returning the W that
  // was handed in. For the Cholesky path that is W to roundoff. For the
  // clamped-eigendecomposition fallback (W positive *semi*definite, or PD to
  // within the tolerance) it is the PSD projection of W, differing by at most
  // the 1e-10·max|W| clamp. Keeping a second q×q copy purely to make
  // `to_dense()` bit-exact would double the resident cost of exactly the
  // weights that are already the largest — ADF/WLS blocks persist in
  // `EstimatorSpec`, `api::Fit`, and the R handles.
  static fit_expected<BlockWeight> dense(const Eigen::MatrixXd& W,
                                         FitError::Kind err_kind,
                                         const std::string& detail) {
    if (W.rows() != W.cols()) {
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          detail + ": weight matrix is not square"});
    }
    if (!W.allFinite()) {
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          detail + ": weight matrix contains non-finite entries"});
    }
    if (!W.isApprox(W.transpose(), 1e-10)) {
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          detail + ": weight matrix is not symmetric"});
    }
    BlockWeight w;
    w.kind_ = Kind::Dense;
    w.dim_  = W.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(W);
    if (llt.info() == Eigen::Success) {
      w.f_dense_ = Eigen::MatrixXd(llt.matrixL());
      return w;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
        0.5 * (W + W.transpose()));
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
    w.f_dense_ = es.eigenvectors() * vals.asDiagonal();
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
      case Kind::Diagonal: return f_diag_.size() == n && f_diag_.allFinite();
      case Kind::Dense:
        return f_dense_.rows() == n && f_dense_.cols() == n &&
               f_dense_.allFinite();
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
      case Kind::Diagonal: return s * (f_diag_.asDiagonal() * X);
      case Kind::Dense:    return s * (f_dense_.transpose() * X);
      case Kind::NormalTheory: return t_apply_nt(s, X);
    }
    return typename Derived::PlainObject();
  }

  // Materialize W. Only for callers that genuinely need the dense form
  // (Satorra-Bentler corrections, Γ spectra); never on a gradient path.
  // Identity / Diagonal / Dense return W exactly as supplied.
  Eigen::MatrixXd to_dense() const {
    switch (kind_) {
      case Kind::Identity:
        return Eigen::MatrixXd::Identity(dim_, dim_);
      case Kind::Diagonal:
        return Eigen::MatrixXd(w_diag_.asDiagonal());
      case Kind::Dense:
        return f_dense_ * f_dense_.transpose();
      case Kind::NormalTheory:
        return nt_to_dense();
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

    // ½·tr(A⁻¹ E_{k1} A⁻¹ E_{k2}) with E_k the symmetric lower-vech basis
    // matrix. Expanding the basis matrices (each has at most two nonzeros) and
    // using tr(A e_a e_bᵀ A e_c e_dᵀ) = A_bc·A_da collapses the entry to a
    // handful of scalar lookups:
    //
    //   tr(A E1 A E2) = Σ_{(a,b)∈T1} Σ_{(c,d)∈T2} A_bc · A_da
    //
    // with T1 = {(r1,c1)} when r1 == c1 and {(r1,c1),(c1,r1)} otherwise. That
    // is O(1) per entry, so O(p⁴) overall rather than the O(p⁷) an explicit
    // `(A·E1·A·E2).trace()` per pair costs.
    const auto basis_terms = [](Eigen::Index r, Eigen::Index c,
                                Eigen::Index (&t)[2][2]) -> int {
      t[0][0] = r; t[0][1] = c;
      if (r == c) return 1;
      t[1][0] = c; t[1][1] = r;
      return 2;
    };
    Eigen::Index k1 = 0;
    for (Eigen::Index c1 = 0; c1 < p; ++c1) {
      for (Eigen::Index r1 = c1; r1 < p; ++r1, ++k1) {
        Eigen::Index t1[2][2];
        const int n1 = basis_terms(r1, c1, t1);
        Eigen::Index k2 = 0;
        for (Eigen::Index c2 = 0; c2 < p; ++c2) {
          for (Eigen::Index r2 = c2; r2 < p; ++r2, ++k2) {
            Eigen::Index t2[2][2];
            const int n2 = basis_terms(r2, c2, t2);
            double acc = 0.0;
            for (int i = 0; i < n1; ++i) {
              for (int j = 0; j < n2; ++j) {
                acc += Ainv(t1[i][1], t2[j][0]) * Ainv(t2[j][1], t1[i][0]);
              }
            }
            W(off + k1, off + k2) = 0.5 * acc;
          }
        }
      }
    }
    (void)pstar;
    return W;
  }

  Kind kind_ = Kind::Identity;
  Eigen::Index dim_ = 0;
  // Diagonal keeps both forms: it is O(q), so exactness in `to_dense()` is
  // free. Dense keeps only the factor — see the `dense()` note above.
  Eigen::VectorXd w_diag_;   // Diagonal: diag(W)
  Eigen::VectorXd f_diag_;   // Diagonal: √diag(W)
  Eigen::MatrixXd f_dense_;  // Dense: factor L, W = L Lᵀ
  Eigen::MatrixXd chol_;     // NormalTheory: chol(A) lower
  bool has_means_ = false;
};

// Per-block weight, aligned to the stacked [mean ; vech(cov)] moment vector of
// each block. An empty vector means identity in every block (ULS) — the
// historical sentinel, preserved so existing callers keep their meaning.
using Weight = std::vector<BlockWeight>;

// Legacy adapter: wrap per-block dense matrices as Dense blocks. Fails exactly
// where `gmm::residuals` used to fail, when a block is not symmetric PSD.
// Callers that know their structure should use the named `BlockWeight`
// constructors instead — this exists for the paths that genuinely produce a
// dense Γ̂⁻¹ (ADF/WLS) and for the R boundary, which receives a bare matrix.
fit_expected<Weight> dense_weight(const std::vector<Eigen::MatrixXd>& blocks,
                                  FitError::Kind err_kind,
                                  const std::string& detail);

}  // namespace magmaan::estimate::gmm
