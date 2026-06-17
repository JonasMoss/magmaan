#include "magmaan/robust/satorra2000.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

#include "detail_vech.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;


namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// Apply V_g = Γ_NT(Σ_g)⁻¹ to vech(M).  Closed form:
//
//     V_g · vech(M) = vech(W · M · W)   with diagonal halved,    W = Σ_g⁻¹.
//
// We work on a column-bundle (p*_g × n) at a time for SIMD-friendly Eigen
// matrix products, but the operator is column-independent.
//
// `inv_sigma` is Σ_g⁻¹ (caller computes via LLT and passes it in once).
Eigen::MatrixXd apply_V_columns(const Eigen::MatrixXd& inv_sigma,
                                const Eigen::Ref<const Eigen::MatrixXd>& cols) {
  const Eigen::Index p     = inv_sigma.rows();
  const Eigen::Index pstar = detail::vech_len(p);
  const Eigen::Index n     = cols.cols();
  Eigen::MatrixXd out(pstar, n);
  Eigen::MatrixXd Mbuf(p, p);
  for (Eigen::Index c = 0; c < n; ++c) {
    detail::vech_unpack(cols.col(c), p, Mbuf);
    // Z = W · M · W (W symmetric ⇒ Z symmetric)
    const Eigen::MatrixXd Z = inv_sigma * Mbuf * inv_sigma;
    // Pack with diagonal halved.
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j; i < p; ++i) {
        out(k++, c) = (i == j) ? 0.5 * Z(i, j) : Z(i, j);
      }
    }
  }
  return out;
}

// Augmented variant of apply_V_columns for the mean-structured moment vector
// [μ; vech(M)] (mean rows on top).  The μ-block of V = Γ_NT⁻¹ is Σ⁻¹ (the
// covariance of the mean residuals; no factor of 2, no diagonal halving), so
// the first `mu_dim` rows map μ-part ↦ Σ⁻¹ · μ-part, while the remaining p*
// rows reuse the covariance closed form V · vech(M) = vech(W·M·W) diag-halved.
//
// `mu_dim` must be 0 (degenerates to apply_V_columns on the σ-rows only) or
// equal to p = inv_sigma.rows().
Eigen::MatrixXd apply_V_augmented_columns(
    const Eigen::MatrixXd& inv_sigma, Eigen::Index mu_dim,
    const Eigen::Ref<const Eigen::MatrixXd>& cols) {
  if (mu_dim == 0) return apply_V_columns(inv_sigma, cols);
  Eigen::MatrixXd out(cols.rows(), cols.cols());
  out.topRows(mu_dim).noalias() = inv_sigma * cols.topRows(mu_dim);
  out.bottomRows(cols.rows() - mu_dim) =
      apply_V_columns(inv_sigma, cols.bottomRows(cols.rows() - mu_dim));
  return out;
}

}  // namespace

post_expected<SatorraDiffResult>
compute_satorra2000(const std::vector<SatorraGroup>& groups,
                    const Eigen::MatrixXd&           A_alpha,
                    GammaSource                      gamma,
                    GammaComputation                 computation) {
  if (groups.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000: groups vector is empty"));
  }
  const Eigen::Index r1 = A_alpha.cols();
  const Eigen::Index m  = A_alpha.rows();

  // Degenerate H0 ≡ H1.
  if (m == 0) {
    SatorraDiffResult deg;
    deg.C = Eigen::MatrixXd::Zero(0, 0);
    deg.S = Eigen::MatrixXd::Zero(0, 0);
    deg.eigenvalues    = Eigen::VectorXd::Zero(0);
    deg.trace_CinvS    = 0.0;
    deg.trace_CinvS_sq = 0.0;
    return deg;
  }
  if (m > r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000: A_alpha is wider than tall (m > r1) — "
        "expected an m × r1 matrix with m ≤ r1."));
  }

  // ── 1. Per-group preprocessing.  We need V_g = Σ_g⁻¹⊗-style operator;
  //       fetch the per-group inverses up-front.
  std::vector<Eigen::MatrixXd> inv_sigma(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    const auto& gr = groups[g];
    if (gr.Sigma.rows() != gr.Sigma.cols() || gr.Sigma.rows() == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: group " + std::to_string(g) +
          " has invalid Sigma shape"));
    }
    if (gr.Pi_alpha.cols() != r1) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: group " + std::to_string(g) +
          " Pi_alpha has " + std::to_string(gr.Pi_alpha.cols()) +
          " columns, expected r1 = " + std::to_string(r1)));
    }
    const Eigen::Index p     = gr.Sigma.rows();
    const Eigen::Index pstar = detail::vech_len(p);
    if (gr.mu_dim != 0 && gr.mu_dim != p) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: group " + std::to_string(g) + " mu_dim = " +
          std::to_string(gr.mu_dim) + " must be 0 (cov-only) or p_g = " +
          std::to_string(p) + " (augmented [μ; vech(Σ)])"));
    }
    if (gr.Pi_alpha.rows() != gr.mu_dim + pstar) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: group " + std::to_string(g) +
          " Pi_alpha has " + std::to_string(gr.Pi_alpha.rows()) +
          " rows, expected mu_dim + p*_g = " + std::to_string(gr.mu_dim + pstar)));
    }
    Eigen::LLT<Eigen::MatrixXd> llt(gr.Sigma);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: group " + std::to_string(g) +
          " Sigma is not positive definite"));
    }
    inv_sigma[g] = llt.solve(Eigen::MatrixXd::Identity(p, p));
  }

  // ── 2. Pooled expected information  P = Σ_g w_g · Π_α_gᵀ · V_g · Π_α_g
  //
  // A NaN/Inf here typically traces back to a non-finite group `weight`
  // (e.g. `n_g / N_total` with `N_total == 0` because the caller's
  // multi-group split came up empty); we surface that distinctly below
  // because the LDLT failure message would otherwise be cryptic.
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(r1, r1);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    const auto& gr = groups[g];
    if (!std::isfinite(gr.weight)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: group " + std::to_string(g) +
          " has non-finite weight (= n_g / N_total) — typically caused by "
          "an empty group in the caller's split."));
    }
    const Eigen::MatrixXd V_Pi =
        apply_V_augmented_columns(inv_sigma[g], gr.mu_dim, gr.Pi_alpha);
    P.noalias() += gr.weight * (gr.Pi_alpha.transpose() * V_Pi);
  }
  P = 0.5 * (P + P.transpose()).eval();
  Eigen::LDLT<Eigen::MatrixXd> ldlt_P(P);
  if (ldlt_P.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000: pooled expected info P is not invertible "
        "(H1 not identified at θ̂)"));
  }
  // LDLT.info() does not catch PSD-singular matrices.  An additional
  // tiny-pivot check on the diagonal of D rejects under-identified H1.
  //
  // The threshold is *relative* to the pooled scale ‖P‖_∞: a configural
  // multi-group fit has P block-diagonal with weights (n_g/N) ≈ 1/G, so a
  // 1e-12-of-max pivot in a well-identified block is normal.  We use
  // 1e-10 here to leave headroom while still catching genuine singularity.
  {
    const Eigen::VectorXd d_abs = ldlt_P.vectorD().cwiseAbs();
    const double tol_pivot = 1e-10 * d_abs.maxCoeff();
    if (d_abs.minCoeff() <= tol_pivot) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: pooled expected info P is rank-deficient "
          "(smallest |D| pivot " + std::to_string(d_abs.minCoeff()) +
          " ≤ tol " + std::to_string(tol_pivot) + ", max pivot " +
          std::to_string(d_abs.maxCoeff()) +
          ") — H1 not identified or pooled moment Jacobian has a zero column."));
    }
  }

  // ── 3. Y = P⁻¹ · A_αᵀ   (r1 × m)
  const Eigen::MatrixXd Y = ldlt_P.solve(A_alpha.transpose());

  // ── 4. C = A_α · Y     (m × m)
  Eigen::MatrixXd C = A_alpha * Y;
  C = 0.5 * (C + C.transpose()).eval();

  // ── 5. S accumulator.  Per-group casewise reduction:
  //       D_g = V_g · Π_α_g · Y            (p*_g × m)
  //       u_gi = D_gᵀ · (d_gi − s_g)       (m-vector)
  //       S += weight_g / (n_g − 1) · Σᵢ u_gi · u_giᵀ
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m, m);
  std::vector<std::string> warnings;

  // Dense-path scratch: the Dense computation forms the full block-diagonal
  // q×q empirical Γ̂ and the q×q restricted U-matrix instead of the m×m
  // reduction.  Per-group pieces are collected here and assembled after the
  // loop;  q_total = Σ_g p*_g.
  std::vector<Eigen::MatrixXd> dense_gamma;
  std::vector<Eigen::MatrixXd> dense_dstack;
  Eigen::Index q_total = 0;

  if (gamma == GammaSource::NT) {
    // Sanity-check short-circuit.  Under Γ = Γ_NT and V = Γ_NT⁻¹ the formula
    // S = A · P⁻¹ · Πᵀ · V · Γ · V · Π · P⁻¹ · Aᵀ collapses to
    // S = A · P⁻¹ · Aᵀ = C, giving λⱼ ≡ 1 — equivalently the trace ratio
    // tr(C⁻¹S)/m = 1 (no SB correction needed).
    S = C;
  } else {
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const auto& gr = groups[g];
      if (gr.n_g < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "compute_satorra2000: group " + std::to_string(g) +
            " has n_g < 2; centring the casewise residuals is undefined"));
      }
      if (gr.X.rows() != gr.n_g) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "compute_satorra2000: group " + std::to_string(g) +
            " X has " + std::to_string(gr.X.rows()) + " rows, expected n_g = "
            + std::to_string(gr.n_g)));
      }
      if (gr.X.cols() != gr.Sigma.rows()) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "compute_satorra2000: group " + std::to_string(g) +
            " X has " + std::to_string(gr.X.cols()) +
            " columns, expected p_g = " + std::to_string(gr.Sigma.rows())));
      }
      // D_g = V_g · (Π_α_g · Y)            ((mu_dim+p*_g) × m)
      const Eigen::MatrixXd PiY     = gr.Pi_alpha * Y;   // (mu_dim+p*_g) × m
      const Eigen::MatrixXd D_g =
          apply_V_augmented_columns(inv_sigma[g], gr.mu_dim, PiY);
      // s_g = vech(S_g_sample) — but the casewise residual d_gi − s_g cancels
      // the constant centring, so we instead use the per-group mean of d_gi
      // over i.  Numerically equivalent to subtracting vech(S_g) up to a
      // 1/n_g vs 1/(n_g−1) difference, and using the empirical mean keeps
      // round-off out of the rank-1 updates below.
      const Eigen::Index p     = gr.Sigma.rows();
      const Eigen::Index pstar = detail::vech_len(p);
      const Eigen::MatrixXd Xc = gr.X.rowwise() - gr.mean.transpose();

      // Build D row-major: each row i is the augmented casewise residual
      // [ (x_i − x̄) ; vech((x_i − x̄)(x_i − x̄)ᵀ) ], with the mean block present
      // only when mu_dim > 0.  The μ-columns equal Xc directly (already mean
      // zero, so the dbar centring below is a no-op on them); the σ-columns are
      // the vech outer products.
      Eigen::MatrixXd D_rows(gr.n_g, gr.mu_dim + pstar);
      if (gr.mu_dim > 0) D_rows.leftCols(gr.mu_dim) = Xc;
      for (Eigen::Index i = 0; i < gr.n_g; ++i) {
        const Eigen::VectorXd xi = Xc.row(i).transpose();
        const Eigen::MatrixXd outer = xi * xi.transpose();
        D_rows.row(i).tail(pstar) = detail::vech_lower(outer).transpose();
      }
      // Centre by the empirical mean of D_rows (σ-block ≡ vech(S_g_n-divisor);
      // μ-block mean ≈ 0).
      const Eigen::VectorXd dbar = D_rows.colwise().mean();
      D_rows.rowwise() -= dbar.transpose();

      // U_rows = D_rows · D_g   (n_g × m); S += (weight_g / n_g) · U·Uᵀ
      // (`n_g` not `n_g − 1` — matches magmaan's existing `empirical_gamma`
      // and lavaan's `likelihood = "normal"` convention. Some derivations use
      // n_g - 1, but the choice cancels in the scaling factor c = tr(C^-1 S)/m
      // as long as S and the test statistic T agree; standardize on the n
      // divisor everywhere.
      if (computation == GammaComputation::Streaming) {
        const Eigen::MatrixXd U_rows = D_rows * D_g;           // n_g × m
        const double scale = gr.weight / static_cast<double>(gr.n_g);
        S.noalias() += scale * (U_rows.transpose() * U_rows);
      } else if (computation == GammaComputation::Materialized) {
        const Eigen::MatrixXd Gamma_hat =
            (D_rows.transpose() * D_rows) / static_cast<double>(gr.n_g);
        S.noalias() += gr.weight * (D_g.transpose() * Gamma_hat * D_g);
      } else {  // Dense — collect the full q×q pieces, assemble after the loop.
        dense_gamma.push_back(
            (D_rows.transpose() * D_rows) / static_cast<double>(gr.n_g));
        dense_dstack.push_back(std::sqrt(gr.weight) * D_g);
        q_total += gr.mu_dim + pstar;
      }
    }
  }

  // ── 6. The m test eigenvalues.
  Eigen::VectorXd eig;
  if (computation == GammaComputation::Dense && gamma != GammaSource::NT) {
    // Dense reference: form the full block-diagonal q×q empirical Γ̂ and the
    // q×q restricted U-matrix  U_d = D · C⁻¹ · Dᵀ, then read the m non-zero
    // eigenvalues off the q×q product U_d·Γ̂.  This is the O(q³) computation
    // the reduction is built to avoid — kept as an auditable strawman.
    if (q_total < m) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: dense path has q_total < m"));
    }
    Eigen::MatrixXd Gamma_block = Eigen::MatrixXd::Zero(q_total, q_total);
    Eigen::MatrixXd D_stack     = Eigen::MatrixXd::Zero(q_total, m);
    Eigen::Index off = 0;
    for (std::size_t g = 0; g < dense_gamma.size(); ++g) {
      const Eigen::Index qg = dense_gamma[g].rows();
      Gamma_block.block(off, off, qg, qg) = dense_gamma[g];
      D_stack.middleRows(off, qg)         = dense_dstack[g];
      off += qg;
    }
    dense_gamma.clear();
    dense_dstack.clear();
    // S = Dᵀ·Γ̂·D — the same m×m S the reduced paths accumulate; kept for
    // result-struct consistency.
    S = D_stack.transpose() * Gamma_block * D_stack;
    S = 0.5 * (S + S.transpose()).eval();
    Eigen::LDLT<Eigen::MatrixXd> ldlt_C(C);
    if (ldlt_C.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: companion matrix C is not invertible "
          "(dense path)"));
    }
    Eigen::MatrixXd U_d = D_stack * ldlt_C.solve(D_stack.transpose());
    const Eigen::MatrixXd prod = U_d * Gamma_block;   // q×q, non-symmetric
    U_d.resize(0, 0);
    Gamma_block.resize(0, 0);
    Eigen::EigenSolver<Eigen::MatrixXd> es(prod, /*computeEigenvectors=*/false);
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: dense q×q eigensolver failed"));
    }
    Eigen::VectorXd rev   = es.eigenvalues().real();
    const double imag_max = es.eigenvalues().imag().cwiseAbs().maxCoeff();
    const double scale_ref = std::max(1.0, rev.cwiseAbs().maxCoeff());
    if (imag_max > 1e-8 * scale_ref) {
      warnings.emplace_back(
          "compute_satorra2000: dense U·Γ spectrum carries imaginary part " +
          std::to_string(imag_max) + " — real parts used");
    }
    // U_d and Γ̂ are both PSD ⇒ the q eigenvalues are real and non-negative;
    // the m test eigenvalues are the m largest (the rest are ~0).  Ascending.
    std::sort(rev.data(), rev.data() + rev.size());
    eig = rev.tail(m);
  } else {
    // Reduced paths (and the NT short-circuit): the m generalised
    // eigenvalues of  S · v = λ · C · v.
    S = 0.5 * (S + S.transpose()).eval();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        S, C, Eigen::EigenvaluesOnly | Eigen::Ax_lBx);
    if (ges.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000: generalised eigensolver failed (C is likely "
          "singular — the restriction is degenerate against H1's curvature)"));
    }
    eig = ges.eigenvalues();
  }
  // Clip tiny round-off negatives (the eigvals are non-negative in exact
  // arithmetic).
  const double clip_thresh = 1e-12 * std::max(1.0, eig.cwiseAbs().maxCoeff());
  for (Eigen::Index k = 0; k < eig.size(); ++k) {
    if (eig(k) < 0.0) {
      if (eig(k) >= -clip_thresh) {
        eig(k) = 0.0;
      } else {
        warnings.emplace_back(
            "compute_satorra2000: detected eigenvalue " +
            std::to_string(eig(k)) + " below clip threshold " +
            std::to_string(-clip_thresh) + " — clipped to 0");
        eig(k) = 0.0;
      }
    }
  }

  // ── 7. Traces.  tr(C⁻¹·S) = Σ λⱼ; tr((C⁻¹·S)²) = Σ λⱼ²; computing these
  //       directly from the spectrum keeps round-off out of the trace solve.
  SatorraDiffResult out;
  out.C              = std::move(C);
  out.S              = std::move(S);
  out.eigenvalues    = eig;
  out.trace_CinvS    = eig.sum();
  out.trace_CinvS_sq = eig.squaredNorm();
  out.warnings       = std::move(warnings);
  return out;
}

post_expected<SatorraDiffResult>
compute_diff_spectrum_2001(const Eigen::Ref<const Eigen::MatrixXd>& U0,
                           const Eigen::Ref<const Eigen::MatrixXd>& U1,
                           const Eigen::Ref<const Eigen::MatrixXd>& Gamma,
                           int                                      df) {
  const Eigen::Index q = U0.rows();
  if (U0.cols() != q || U1.rows() != q || U1.cols() != q ||
      Gamma.rows() != q || Gamma.cols() != q || q == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_diff_spectrum_2001: U0, U1, Gamma must be square and the same "
        "dimension"));
  }
  if (df <= 0) {
    SatorraDiffResult deg;
    deg.C = Eigen::MatrixXd::Zero(0, 0);
    deg.S = Eigen::MatrixXd::Zero(0, 0);
    deg.eigenvalues    = Eigen::VectorXd::Zero(0);
    deg.trace_CinvS    = 0.0;
    deg.trace_CinvS_sq = 0.0;
    return deg;
  }
  if (static_cast<Eigen::Index>(df) > q) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_diff_spectrum_2001: df (" + std::to_string(df) +
        ") exceeds moment dimension (" + std::to_string(q) + ")"));
  }

  // D = U0 − U1 (symmetric since each U_g is symmetric).
  const Eigen::MatrixXd D = 0.5 * ((U0 - U1) + (U0 - U1).transpose());

  // eig((U0−U1)·Γ) = eig(RᵀDR) with Γ = RRᵀ — the symmetric reduction used
  // throughout the U·Γ machinery (mirrors fiml_ugamma_spectrum_impl).
  Eigen::MatrixXd reduced;
  {
    const Eigen::MatrixXd Gs = 0.5 * (Gamma + Gamma.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(Gs);
    if (llt.info() == Eigen::Success) {
      const Eigen::MatrixXd R = llt.matrixL();
      reduced = R.transpose() * D * R;
    } else {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_g(Gs);
      if (es_g.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "compute_diff_spectrum_2001: common Gamma is not symmetric PSD"));
      }
      const Eigen::VectorXd d = es_g.eigenvalues().cwiseMax(0.0).cwiseSqrt();
      const Eigen::MatrixXd sq =
          es_g.eigenvectors() * d.asDiagonal() * es_g.eigenvectors().transpose();
      reduced = sq * D * sq;
    }
    reduced = 0.5 * (reduced + reduced.transpose()).eval();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(reduced,
                                                    Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_diff_spectrum_2001: eigen-solve of the difference spectrum "
        "failed"));
  }
  // The d test eigenvalues are the d largest (ascending) of the q-spectrum.
  const Eigen::VectorXd all = es.eigenvalues();
  Eigen::VectorXd eig = all.tail(df);

  std::vector<std::string> warnings;
  const double scale = std::max(1.0, eig.cwiseAbs().maxCoeff());
  if (eig.minCoeff() < -1e-8 * scale) {
    warnings.emplace_back(
        "compute_diff_spectrum_2001: method-2001 negative eigenvalue " +
        std::to_string(eig.minCoeff()) + " in the top-" + std::to_string(df) +
        " difference spectrum (U0 − U1 is not PSD); the SB/mixture readouts can "
        "go negative — fall back to method 2000 if a positive statistic is "
        "required.");
  }

  SatorraDiffResult out;
  out.C              = Eigen::MatrixXd::Zero(0, 0);
  out.S              = Eigen::MatrixXd::Zero(0, 0);
  out.eigenvalues    = std::move(eig);
  out.trace_CinvS    = out.eigenvalues.sum();
  out.trace_CinvS_sq = out.eigenvalues.squaredNorm();
  out.warnings       = std::move(warnings);
  return out;
}

}  // namespace magmaan::robust
