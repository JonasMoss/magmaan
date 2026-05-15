#include "magmaan/fit/robust.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/fit/constraints.hpp"        // EqConstraints / build_eq_constraints
#include "magmaan/fit/inference.hpp"          // information_observed_{analytic,fd}
#include "magmaan/fit/raw_data.hpp"
#include "magmaan/fit/resolve_fixed_x.hpp"
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"

namespace magmaan::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

using detail::vech_lower;
using detail::vech_unpack;

// For an ObservedHessian UFactor: C = I − A·H_obs⁻¹·Aᵀ
// (total_rows × total_rows, symmetric).
// `UΓ = L_Γ⁻ᵀ·C·L_Γ⁻¹·Γ` and `eigvals(UΓ) = eigvals(C · L_Γ⁻¹ΓL_Γ⁻ᵀ)`.
Eigen::MatrixXd observed_C(const UFactor& uf) {
  Eigen::MatrixXd C = Eigen::MatrixXd::Identity(uf.total_rows, uf.total_rows);
  C.noalias() -= uf.A * uf.H_obs_inv * uf.A.transpose();
  return 0.5 * (C + C.transpose()).eval();
}

// ────────────────────────────────────────────────────────────────────────────
// G3b helpers — per-block dual-segment ([μ; σ]) operator applications.
// The per-block Γ_NT block-diagonalises as `[M_b 0; 0 Γ_NT_cov(M_b)]`, so
// each helper applies its operator independently to the two segments and
// stitches the result into the block's slice of the stacked output.
// All three helpers are byte-equivalent to the existing σ-only path when
// `has_means == false` (then `blk.mu_off == -1` and the μ branch is skipped).
// ────────────────────────────────────────────────────────────────────────────

// dst_block ← L_Γ_block⁻¹ · src_block.
//   μ-rows: dst[mu_off..mu_off+p, :] = chol(M_b)⁻¹ · src[mu_off..mu_off+p, :]
//   σ-rows: dst[row_offset..row_offset+pstar, :] = chol(Γ_NT_cov(M_b))⁻¹ · src[..]
// `src` and `dst` must share the same dimensions (total_rows × n_cols).
void apply_L_inv_block(const UFactor::Block& blk, bool has_means,
                       const Eigen::Ref<const Eigen::MatrixXd>& src,
                       Eigen::Ref<Eigen::MatrixXd>              dst) {
  if (has_means && blk.mu_off >= 0) {
    dst.middleRows(blk.mu_off, blk.p) =
        blk.llt_M.matrixL().solve(src.middleRows(blk.mu_off, blk.p));
  }
  dst.middleRows(blk.row_offset, blk.pstar) =
      blk.llt_gamma_nt.matrixL().solve(
          src.middleRows(blk.row_offset, blk.pstar));
}

// dst_block ← L_Γ_block⁻ᵀ · src_block (the adjoint of apply_L_inv_block;
// used to form B = L_Γ⁻ᵀ · N).
void apply_L_invT_block(const UFactor::Block& blk, bool has_means,
                        const Eigen::Ref<const Eigen::MatrixXd>& src,
                        Eigen::Ref<Eigen::MatrixXd>              dst) {
  if (has_means && blk.mu_off >= 0) {
    dst.middleRows(blk.mu_off, blk.p) =
        blk.llt_M.matrixL().adjoint().solve(src.middleRows(blk.mu_off, blk.p));
  }
  dst.middleRows(blk.row_offset, blk.pstar) =
      blk.llt_gamma_nt.matrixL().adjoint().solve(
          src.middleRows(blk.row_offset, blk.pstar));
}

// dst_block ← Γ_NT(M_b) · src_block, one column at a time. Used by
// `reduced_gamma_nt` to fill `GB` column-by-column.
//   μ-rows: dst[mu_off..mu_off+p] = M_b · src[mu_off..mu_off+p]
//           (no halving, no factor of 2 — the μ-block of Γ_NT is just M_b).
//   σ-rows: vech_unpack(src) → halve off-diag → M_b·H·M_b → pack 2·vech(Z).
void apply_gamma_nt_block(const UFactor::Block& blk, bool has_means,
                          WeightMoments moments,
                          const Eigen::Ref<const Eigen::VectorXd>& src,
                          Eigen::Ref<Eigen::VectorXd>              dst,
                          Eigen::MatrixXd&                         H_buf) {
  const Eigen::MatrixXd& Mmoments =
      (moments == WeightMoments::Structured) ? blk.Sigma_hat : blk.S;
  if (has_means && blk.mu_off >= 0) {
    dst.segment(blk.mu_off, blk.p).noalias() =
        Mmoments * src.segment(blk.mu_off, blk.p);
  }
  H_buf.resize(blk.p, blk.p);
  vech_unpack(src.segment(blk.row_offset, blk.pstar), blk.p, H_buf);
  // Halve off-diagonals so the symmetrically-unpacked matrix becomes the
  // M_½ form the Γ_NT operator expects (Γ_NT = 2·D⁺(Σ⊗Σ)D⁺ᵀ pulls in the
  // duplication-pseudoinverse halving).
  for (Eigen::Index j = 0; j < blk.p; ++j)
    for (Eigen::Index i = 0; i < blk.p; ++i)
      if (i != j) H_buf(i, j) *= 0.5;
  const Eigen::MatrixXd Z = Mmoments * H_buf * Mmoments;
  // Pack 2·vech(Z) back (no extra diag halving — Z is the output).
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < blk.p; ++j) {
    for (Eigen::Index i = j; i < blk.p; ++i) {
      dst(blk.row_offset + k) = 2.0 * Z(i, j);
      ++k;
    }
  }
}

// Given M̃ (p* × p*, symmetric PSD) and C (p* × p*, symmetric), return a
// symmetric matrix whose eigenvalues are exactly those of C·M̃ — so
// `ugamma_eigenvalues` can be applied unchanged. Via Cholesky M̃ = R·Rᵀ
// → Rᵀ·C·R (eigvals(RᵀCR) = eigvals(CRRᵀ) = eigvals(CM̃)); falls back to
// the eigendecomposition square root M̃^{½}·C·M̃^{½} when M̃ is PSD-singular.
Eigen::MatrixXd symm_product_eigbasis(const Eigen::MatrixXd& Mtilde,
                                      const Eigen::MatrixXd& C) {
  const Eigen::MatrixXd Ms = 0.5 * (Mtilde + Mtilde.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt(Ms);
  if (llt.info() == Eigen::Success) {
    const Eigen::MatrixXd R = llt.matrixL();   // Ms = R·Rᵀ
    Eigen::MatrixXd out = R.transpose() * C * R;
    return 0.5 * (out + out.transpose()).eval();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ms);
  const Eigen::VectorXd d = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  const Eigen::MatrixXd sqrtM =
      es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
  Eigen::MatrixXd out = sqrtM * C * sqrtM;
  return 0.5 * (out + out.transpose()).eval();
}

// Mirror `prepare_evaluator` in inference.cpp — kept duplicated to avoid
// promoting a private helper across translation units. Resolves fixed.x
// `fixed_value`s from the sample (matches `fit()` pattern), then builds and
// shape-checks the evaluator.
post_expected<model::ModelEvaluator>
prepare_evaluator(partable::LatentStructure&       pt,
                  const model::MatrixRep&   rep,
                  const SampleStats&        samp,
                  const Estimates&          est) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "resolve_fixed_x_from_sample failed: " + e.error().detail));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " ≠ evaluator n_free " + std::to_string(ev.n_free())));
  }
  if (samp.S.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "SampleStats and evaluator have different block counts"));
  }
  return ev;
}

}  // namespace

// ============================================================================
// build_u_factor
// ============================================================================

post_expected<UFactor>
build_u_factor(partable::LatentStructure        pt,
               const model::MatrixRep&   rep,
               const SampleStats&        samp,
               const Estimates&          est,
               InferenceSpec             spec) {
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto& ev = *ev_or;

  // Detect mean structure (G3b). `dmu_dtheta` returns either an empty
  // (0×0) matrix — cov-only — or `(Σ_b p_b) × n_free` stacked per block
  // (zero rows for blocks without their own `~1` rows, but the row slot
  // is still reserved). When non-empty, each per-block slice gets prepended
  // to that block's σ-rows in Δ; the per-block Γ_NT block-diagonalises as
  // `[M_b 0; 0 Γ_NT_cov(M_b)]` so the per-block triangular solves apply
  // independently to the two segments.
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  const bool has_means = Jmu_or->size() > 0;
  Eigen::MatrixXd Delta_mu = has_means ? std::move(*Jmu_or)
                                       : Eigen::MatrixXd();

  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: dsigma_dtheta failed: " + J_or.error().detail));
  }
  Eigen::MatrixXd Delta_sigma = std::move(*J_or);

  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: sigma(theta) failed: " + im_or.error().detail));
  }

  // Per-block geometry + LLTs. M_b is Σ̂_b (structured) or S_b (unstructured)
  // per `spec.moments`. Σ̂_b and S_b are both copied into the block — the
  // evaluator's internal Σ̂ buffer is overwritten by later sigma() calls, and
  // `reduced_gamma_nt` needs whichever moments matched the bread. With mean
  // structure, `llt_M` (= LLT(M_b)) is the μ-block's Cholesky.
  UFactor uf;
  uf.moments    = spec.moments;
  uf.has_means  = has_means;
  uf.blocks.resize(samp.S.size());
  double N_total = 0.0;
  for (auto n : samp.n_obs) N_total += static_cast<double>(n);
  Eigen::Index row_cursor = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p     = samp.S[b].rows();
    const Eigen::Index pstar = p * (p + 1) / 2;
    auto&              blk   = uf.blocks[b];
    blk.p          = p;
    blk.pstar      = pstar;
    blk.mu_off     = has_means ? row_cursor : Eigen::Index{-1};
    if (has_means) row_cursor += p;          // μ-segment sits on top
    blk.row_offset = row_cursor;             // σ-segment start (same name kept)
    row_cursor    += pstar;
    blk.n_obs      = static_cast<Eigen::Index>(samp.n_obs[b]);
    blk.Sigma_hat  = im_or->sigma[b];
    blk.S          = samp.S[b];
    const Eigen::MatrixXd& W_moments =
        (spec.moments == WeightMoments::Structured) ? blk.Sigma_hat : blk.S;
    auto G_or = gamma_nt(W_moments);
    if (!G_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "build_u_factor: gamma_nt failed for block " + std::to_string(b) +
              ": " + G_or.error().detail));
    }
    blk.llt_gamma_nt = Eigen::LLT<Eigen::MatrixXd>(*G_or);
    if (blk.llt_gamma_nt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "build_u_factor: Γ_NT(weight moments) not positive definite in "
          "block " + std::to_string(b)));
    }
    if (has_means) {
      blk.llt_M = Eigen::LLT<Eigen::MatrixXd>(W_moments);
      if (blk.llt_M.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
            "build_u_factor: M_b (μ-block of Γ_NT) not positive definite in "
            "block " + std::to_string(b)));
      }
    }
  }
  uf.total_rows = row_cursor;
  // Σ-only total kept for the σ-only external surface (R bindings,
  // legacy callers, the `gamma_hat` overload's shape doc).
  uf.pstar = 0;
  for (const auto& blk : uf.blocks) uf.pstar += blk.pstar;

  // Stack Δ_full = [Δμ_b; Δσ_b] per block. Σ-rows come from `Delta_sigma`'s
  // own per-block stacking (cursor `s_cursor`); μ-rows from `Delta_mu`
  // (cursor `m_cursor`, reserving p_b rows per block per `dmu_dtheta`'s
  // contract, zero rows for blocks without means).
  const Eigen::Index n_free = Delta_sigma.cols();
  if (has_means && Delta_mu.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: Δμ cols " + std::to_string(Delta_mu.cols()) +
            " ≠ Δσ cols " + std::to_string(n_free)));
  }
  Eigen::MatrixXd Delta_full =
      Eigen::MatrixXd::Zero(uf.total_rows, n_free);
  {
    Eigen::Index s_cursor = 0;
    Eigen::Index m_cursor = 0;
    for (const auto& blk : uf.blocks) {
      if (has_means) {
        Delta_full.block(blk.mu_off, 0, blk.p, n_free) =
            Delta_mu.block(m_cursor, 0, blk.p, n_free);
        m_cursor += blk.p;
      }
      Delta_full.block(blk.row_offset, 0, blk.pstar, n_free) =
          Delta_sigma.block(s_cursor, 0, blk.pstar, n_free);
      s_cursor += blk.pstar;
    }
    if (s_cursor != Delta_sigma.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "build_u_factor: Δσ row count " + std::to_string(Delta_sigma.rows())
              + " ≠ Σ_b p*_b = " + std::to_string(s_cursor)));
    }
    if (has_means && m_cursor != Delta_mu.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "build_u_factor: Δμ row count " + std::to_string(Delta_mu.rows())
              + " ≠ Σ_b p_b = " + std::to_string(m_cursor)));
    }
  }

  // Linear equality constraints (shared labels, `a == b`, scalar invariance):
  // θ = K·α, so the relevant Jacobian is Δ_full·K (total_rows × n_alpha) and
  // `df` becomes total_rows − n_alpha. Stacking [μ; σ] happens BEFORE the
  // K_con multiply (K_con acts on columns, not rows). `K_con` is empty when
  // there are none — the identity reparameterization. Mirrors
  // `fit::vcov(info, pt)`.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(con_or.error().kind,
        "build_u_factor: " + con_or.error().detail));
  }
  Eigen::MatrixXd K_con;   // empty ⇒ no constraints
  Eigen::MatrixXd Delta = std::move(Delta_full);
  if (con_or->active()) {
    K_con = con_or->K();
    Delta = (Delta * K_con).eval();
  }
  const Eigen::Index q = Delta.cols();

  // Per-block dual-segment triangular solve A_b = L_Γ_block⁻¹·Δ_b.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(uf.total_rows, q);
  for (const auto& blk : uf.blocks) {
    apply_L_inv_block(blk, has_means, Delta, A);
  }

  // ── Observed-Hessian bread (the MLR convention) ────────────────────────
  // U = L_Γ⁻ᵀ·(I − A·H_obs⁻¹·Aᵀ)·L_Γ⁻¹  (not idempotent). We don't form U;
  // store A and H_obs⁻¹ and let `reduced_gamma_*` assemble the spectrum.
  if (spec.bread == Information::Observed) {
    // Observed Hessian (q × q) in the *per-unit* units of ΔᵀWΔ = AᵀA.
    // `information_observed_analytic` returns the total observed info =
    // N·(per-unit Hessian/2)·2 = N·(per-unit info) at θ̂; dividing by N
    // gives the per-unit form, which equals AᵀA up to the H2 residual term.
    // FD fallback for models the analytic path can't do (e.g. mean structure).
    // The matrix is the *unprojected* npar×npar total observed info; /N
    // gives the per-unit form; project through K to the n_alpha space
    // (= AᵀA with the reparameterized Δ·K). Trivial when K_con is empty.
    Eigen::MatrixXd H_obs;
    if (auto h_or = information_observed_analytic(pt, rep, samp, est);
        h_or.has_value()) {
      H_obs = (*h_or) / N_total;
    } else if (auto fd_or = information_observed_fd(pt, rep, samp, est);
               fd_or.has_value()) {
      H_obs = (*fd_or) / N_total;
    } else {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "build_u_factor: could not compute the observed Hessian"));
    }
    if (K_con.size() > 0) H_obs = (K_con.transpose() * H_obs * K_con).eval();
    H_obs = 0.5 * (H_obs + H_obs.transpose()).eval();
    Eigen::LLT<Eigen::MatrixXd> llt_H(H_obs);
    Eigen::MatrixXd H_inv;
    if (llt_H.info() == Eigen::Success) {
      H_inv = llt_H.solve(Eigen::MatrixXd::Identity(q, q));
    } else {
      Eigen::LDLT<Eigen::MatrixXd> ldlt_H(H_obs);
      if (ldlt_H.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
            "build_u_factor: observed Hessian is not invertible"));
      }
      H_inv = ldlt_H.solve(Eigen::MatrixXd::Identity(q, q));
    }
    uf.kind      = UFactor::Kind::ObservedHessian;
    uf.A         = std::move(A);
    uf.H_obs_inv = std::move(H_inv);
    uf.df        = uf.total_rows - q;  // the χ²-reference df
    if (uf.df <= 0) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "build_u_factor: model is saturated (df ≤ 0)"));
    }
    return uf;
  }

  // ── Expected-info projection bread (`U = B·Bᵀ`) ─────────────────────────
  // QR of A → orthonormal basis N of ker(Aᵀ). Trailing (total_rows − rank(A))
  // columns of Q form the basis. We assume full column rank of A (=
  // full column rank of Δ in the Γ_NT-metric); if not, the trailing
  // columns still span ker(Aᵀ) but `df` is larger than `total_rows − q`.
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
  // Estimate rank via the absolute diagonal of R. Cheap and adequate for
  // SEM sizes — rank-deficient Δ is a "your model is broken" condition
  // and the SE methods would already complain.
  const Eigen::MatrixXd R = qr.matrixQR().topLeftCorner(
      std::min(uf.total_rows, q), q).triangularView<Eigen::Upper>();
  const double tol = std::numeric_limits<double>::epsilon() *
                     static_cast<double>(std::max(uf.total_rows, q)) *
                     R.diagonal().cwiseAbs().maxCoeff();
  Eigen::Index r_eff = 0;
  for (Eigen::Index k = 0; k < std::min(uf.total_rows, q); ++k)
    if (std::abs(R(k, k)) > tol) ++r_eff;
  const Eigen::Index df_global = uf.total_rows - r_eff;
  if (df_global <= 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "build_u_factor: model is saturated (df = " +
            std::to_string(df_global) +
            "); robust UΓ inference has no nontrivial spectrum"));
  }
  uf.df = df_global;

  // Form N = trailing df columns of Q without materialising full total×total Q:
  //   N = Q · [0; I_df]
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(uf.total_rows, df_global);
  E.bottomRows(df_global).setIdentity();
  const Eigen::MatrixXd N = qr.householderQ() * E;

  // Per-block dual-segment adjoint solve: B_b = L_Γ_block⁻ᵀ·N_b.
  uf.B = Eigen::MatrixXd::Zero(uf.total_rows, df_global);
  for (const auto& blk : uf.blocks) {
    apply_L_invT_block(blk, has_means, N, uf.B);
  }

  return uf;
}

// ============================================================================
// reduced_gamma_sample — never forms Γ̂.
// ============================================================================

post_expected<Eigen::MatrixXd>
reduced_gamma_sample(const UFactor&                            uf,
                     const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
                     const Eigen::Ref<const Eigen::VectorXd>&  denom) {
  // Expected Zc shape depends on whether the UFactor carries mean structure.
  // Distinguish "missing means" from a true shape error for a clearer message.
  const Eigen::Index expected_cols =
      uf.has_means ? uf.total_rows : uf.pstar;
  if (Zc.cols() != expected_cols) {
    if (uf.has_means && Zc.cols() == uf.pstar) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "reduced_gamma_sample: Zc has " + std::to_string(Zc.cols()) +
              " columns (σ-only) but the UFactor was built with mean "
              "structure (total_rows = " + std::to_string(uf.total_rows) +
              "); rebuild Zc via casewise_contributions(raw, samp, "
              "/*include_means=*/true)"));
    }
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_sample: Zc has " + std::to_string(Zc.cols()) +
            " columns, expected " + std::to_string(expected_cols)));
  }
  const Eigen::Index nb = static_cast<Eigen::Index>(uf.blocks.size());
  if (denom.size() != 1 && denom.size() != nb) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_sample: denom has length " +
            std::to_string(denom.size()) + "; expected 1 or " +
            std::to_string(nb) + " (one per block)"));
  }
  if (!(denom.minCoeff() > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_sample: every denom entry must be > 0"));
  }
  const auto denom_b = [&](Eigen::Index b) -> double {
    return denom.size() == 1 ? denom(0) : denom(b);
  };
  // Per-block contiguous slice in the stacked layout (cov-only: pure σ-vech;
  // means: μ on top of σ-vech, both stitched into one contiguous slice that
  // matches the corresponding rows of `uf.B`). Γ̂ is block-diagonal, so the
  // cross-block contributions of (Zc_b·B_b)ᵀ(Zc_b·B_b) cancel.
  auto block_slice = [&](const UFactor::Block& blk)
      -> std::pair<Eigen::Index, Eigen::Index> {
    const Eigen::Index start = (uf.has_means && blk.mu_off >= 0)
                                   ? blk.mu_off : blk.row_offset;
    const Eigen::Index size  = (uf.has_means ? blk.p : 0) + blk.pstar;
    return {start, size};
  };

  if (uf.kind == UFactor::Kind::ObservedHessian) {
    // M̃ = diag_b( L_Γ_block⁻¹·Γ̂_b·L_Γ_block⁻ᵀ ). With means present
    // L_Γ_block is block-diagonal across (μ, σ), so the μ-cols of Xb get
    // `llt_M` applied and the σ-cols get `llt_gamma_nt` applied — both
    // independently.
    Eigen::MatrixXd Mtilde =
        Eigen::MatrixXd::Zero(uf.total_rows, uf.total_rows);
    for (Eigen::Index b = 0; b < nb; ++b) {
      const auto& blk = uf.blocks[static_cast<std::size_t>(b)];
      const auto [bstart, bsize] = block_slice(blk);
      Eigen::MatrixXd ZcLw_b(Zc.rows(), bsize);
      if (uf.has_means && blk.mu_off >= 0) {
        const Eigen::MatrixXd Xb_mu = Zc.middleCols(blk.mu_off, blk.p);
        ZcLw_b.leftCols(blk.p) =
            blk.llt_M.matrixL().solve(Xb_mu.transpose()).transpose();
      }
      const Eigen::Index sigma_col_in_block = uf.has_means ? blk.p : 0;
      const Eigen::MatrixXd Xb_sig =
          Zc.middleCols(blk.row_offset, blk.pstar);                 // N × p*_b
      ZcLw_b.middleCols(sigma_col_in_block, blk.pstar) =
          blk.llt_gamma_nt.matrixL().solve(Xb_sig.transpose()).transpose();
      Mtilde.block(bstart, bstart, bsize, bsize)
          .noalias() += (ZcLw_b.transpose() * ZcLw_b) / denom_b(b);
    }
    return symm_product_eigbasis(Mtilde, observed_C(uf));
  }
  // ProjectionExpected.
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(uf.df, uf.df);
  for (Eigen::Index b = 0; b < nb; ++b) {
    const auto& blk = uf.blocks[static_cast<std::size_t>(b)];
    const auto [bstart, bsize] = block_slice(blk);
    const Eigen::MatrixXd ZcB_b =
        Zc.middleCols(bstart, bsize) *
        uf.B.middleRows(bstart, bsize);                              // N × df
    M.noalias() += (ZcB_b.transpose() * ZcB_b) / denom_b(b);
  }
  M = 0.5 * (M + M.transpose()).eval();             // kill upstream asymmetry
  return M;
}

post_expected<Eigen::MatrixXd>
reduced_gamma_sample(const UFactor&                            uf,
                     const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
                     double                                    denom) {
  const Eigen::VectorXd d = Eigen::VectorXd::Constant(1, denom);
  return reduced_gamma_sample(uf, Zc, d);
}

post_expected<Eigen::MatrixXd>
reduced_gamma_sample_streaming(const UFactor&                       uf,
                               const std::vector<Eigen::VectorXd>&  zc_rows,
                               double                               denom) {
  if (uf.kind == UFactor::Kind::ObservedHessian) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_sample_streaming: not supported for the "
        "ObservedHessian U-factor — use the (non-streaming) reduced_gamma_sample"));
  }
  if (!(denom > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_sample_streaming: denom must be > 0"));
  }
  // Row size matches B's row count — cov-only = uf.pstar, means = uf.total_rows.
  const Eigen::Index expected_size =
      uf.has_means ? uf.total_rows : uf.pstar;
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(uf.df, uf.df);
  for (std::size_t i = 0; i < zc_rows.size(); ++i) {
    if (zc_rows[i].size() != expected_size) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "reduced_gamma_sample_streaming: zc_rows[" + std::to_string(i) +
              "] has size " + std::to_string(zc_rows[i].size()) +
              ", expected " + std::to_string(expected_size)));
    }
    const Eigen::VectorXd r = uf.B.transpose() * zc_rows[i];
    M.noalias() += r * r.transpose();
  }
  M /= denom;
  M = 0.5 * (M + M.transpose()).eval();
  return M;
}

post_expected<Eigen::MatrixXd>
casewise_contributions(const RawData& raw, const SampleStats& samp,
                       bool include_means) {
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "casewise_contributions: RawData and SampleStats block count mismatch"));
  }
  // Sum n_obs across blocks. Column count is Σ_b p_b* (cov-only) or
  // Σ_b (p_b + p_b*) (means included, G3b — matches `UFactor::total_rows`).
  Eigen::Index n_rows = 0;
  Eigen::Index n_cols = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const Eigen::Index p = samp.S[b].rows();
    n_rows += raw.X[b].rows();
    n_cols += (include_means ? p : 0) + p * (p + 1) / 2;
  }
  Eigen::MatrixXd Zc = Eigen::MatrixXd::Zero(n_rows, n_cols);

  Eigen::Index row_cursor = 0;
  Eigen::Index col_cursor = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X     = raw.X[b];
    const auto& S     = samp.S[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (p != S.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "casewise_contributions: block " + std::to_string(b) +
              " raw cols (" + std::to_string(p) +
              ") ≠ SampleStats S dim (" + std::to_string(S.rows()) + ")"));
    }
    const Eigen::Index pstar = p * (p + 1) / 2;
    const Eigen::VectorXd mean_b =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : X.colwise().mean().transpose().eval();
    const Eigen::MatrixXd Xc = X.rowwise() - mean_b.transpose();
    const Eigen::VectorXd s_vech = vech_lower(S);
    const Eigen::Index mu_col_off    = include_means ? col_cursor : Eigen::Index{-1};
    const Eigen::Index sigma_col_off =
        include_means ? col_cursor + p : col_cursor;
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd xi    = Xc.row(i).transpose();
      // μ-segment: per-row mean residual `x_i − m̄_b` at the block's μ-slice.
      if (include_means) {
        Zc.block(row_cursor + i, mu_col_off, 1, p) = xi.transpose();
      }
      // σ-segment: centred outer-product vech residual at the block's σ-slice.
      const Eigen::MatrixXd outer = xi * xi.transpose();
      const Eigen::VectorXd d_i   = vech_lower(outer) - s_vech;
      Zc.block(row_cursor + i, sigma_col_off, 1, pstar) = d_i.transpose();
    }
    row_cursor += n;
    col_cursor += (include_means ? p : 0) + pstar;
  }
  return Zc;
}

// ============================================================================
// reduced_gamma_nt — operator-only, never forms Γ_NT.
// ============================================================================

post_expected<Eigen::MatrixXd>
reduced_gamma_nt(const UFactor& uf) {
  // ObservedHessian: with the meat moments matching the bread moments,
  // M̃ = L_Γ⁻¹·Γ_NT(M)·L_Γ⁻ᵀ = I, so eigvals(UΓ_NT) = eigvals(C·I) =
  // eigvals(C) where C = I − A·H_obs⁻¹·Aᵀ. (For the expected-info bread
  // this same C is the rank-df projector NNᵀ, eigvals all 0/1.)
  if (uf.kind == UFactor::Kind::ObservedHessian) {
    return observed_C(uf);
  }

  // ProjectionExpected: apply x ↦ Γ_NT·x column-by-column to B, then form
  // M = Bᵀ·(Γ_NT·B).
  //
  // Σ-block of Γ_NT: Γ_NT·vech(M) = vech(2·M_b·M_½·M_b), where M_½ is the
  // off-diagonal-halved variant of vech_unpack(x). Derivation:
  // Γ_NT = 2·D⁺(M_b⊗M_b)D⁺ᵀ — D⁺ᵀ·vech(M) takes vech(M) into vec-space with
  // off-diagonals halved. Verified on a 2×2 worked example against the
  // closed-form `gamma_nt(M_b)` matrix-multiply path.
  //
  // μ-block of Γ_NT (G3b, when has_means): Γ_NT acts as `M_b` (no halving,
  // no factor of 2) on the μ-segment — the standard `[μ; vech(Σ)]` moment
  // convention. Mirrors `browne_residual_nt`'s `apply_gamma_inv` lambda
  // inverted (there it's `M_b⁻¹` for the weight; here forward `M_b`).
  Eigen::MatrixXd GB = Eigen::MatrixXd::Zero(uf.total_rows, uf.df);
  Eigen::MatrixXd H_buf;     // scratch — symmetric M_½
  for (Eigen::Index c = 0; c < uf.df; ++c) {
    for (const auto& blk : uf.blocks) {
      apply_gamma_nt_block(blk, uf.has_means, uf.moments,
                           uf.B.col(c), GB.col(c), H_buf);
    }
  }
  Eigen::MatrixXd M = uf.B.transpose() * GB;
  M = 0.5 * (M + M.transpose()).eval();
  return M;
}

// ============================================================================
// reduced_gamma_unbiased — Browne's correction (single-block in v0).
// ============================================================================

post_expected<Eigen::MatrixXd>
reduced_gamma_unbiased(const UFactor&            uf,
                       const SampleStats&        samp,
                       const Eigen::MatrixXd&    M_sample,
                       const Eigen::MatrixXd&    M_nt) {
  if (uf.kind != UFactor::Kind::ProjectionExpected) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: only the ProjectionExpected U-factor is "
        "supported (Browne's correction is defined in the df-dimensional "
        "projector basis)"));
  }
  if (uf.blocks.size() != 1 || samp.S.size() != 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: only single-block supported in v0; "
        "Browne's correction is well-defined per block but the v0 surface "
        "doesn't expose per-block stitching"));
  }
  if (M_sample.rows() != uf.df || M_sample.cols() != uf.df ||
      M_nt.rows()     != uf.df || M_nt.cols()     != uf.df) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: M shape mismatch with UFactor.df"));
  }
  const double N = static_cast<double>(samp.n_obs[0]);
  if (!(N > 3.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: Browne's correction requires N > 3"));
  }
  const double a = N * (N - 1.0) / ((N - 2.0) * (N - 3.0));
  const double b = N             / ((N - 2.0) * (N - 3.0));
  const double c = 2.0 / (N - 1.0);

  const Eigen::VectorXd s_vech = vech_lower(samp.S[0]);
  const Eigen::VectorXd Bs     = uf.B.transpose() * s_vech;

  Eigen::MatrixXd M = a * M_sample - b * M_nt + b * c * (Bs * Bs.transpose());
  M = 0.5 * (M + M.transpose()).eval();
  return M;
}

post_expected<Eigen::MatrixXd>
reduced_gamma_unbiased_casewise(
    const UFactor&                            uf,
    const SampleStats&                        samp,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    const Eigen::Ref<const Eigen::VectorXd>&  denom) {
  if (uf.kind != UFactor::Kind::ProjectionExpected) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: only the ProjectionExpected U-factor is "
        "supported"));
  }
  if (uf.has_means) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: Browne's covariance correction currently "
        "expects a covariance-only UFactor"));
  }
  const Eigen::Index nb = static_cast<Eigen::Index>(uf.blocks.size());
  if (samp.S.size() != uf.blocks.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: SampleStats/UFactor block mismatch"));
  }
  if (denom.size() != 1 && denom.size() != nb) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: denom has length " +
            std::to_string(denom.size()) + "; expected 1 or " +
            std::to_string(nb)));
  }
  if (!(denom.minCoeff() > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: every denom entry must be > 0"));
  }
  if (Zc.cols() != uf.pstar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_unbiased: Zc has " + std::to_string(Zc.cols()) +
            " columns, expected " + std::to_string(uf.pstar)));
  }
  const auto denom_b = [&](Eigen::Index b) -> double {
    return denom.size() == 1 ? denom(0) : denom(b);
  };

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(uf.df, uf.df);
  Eigen::MatrixXd H_buf;
  for (Eigen::Index b = 0; b < nb; ++b) {
    const auto& blk = uf.blocks[static_cast<std::size_t>(b)];
    const double N = static_cast<double>(samp.n_obs[static_cast<std::size_t>(b)]);
    if (!(N > 3.0)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "reduced_gamma_unbiased: Browne's correction requires every "
          "block N > 3"));
    }
    const double a = N * (N - 1.0) / ((N - 2.0) * (N - 3.0));
    const double bb = N / ((N - 2.0) * (N - 3.0));
    const double c = 2.0 / (N - 1.0);

    const Eigen::Index bstart = blk.row_offset;
    const Eigen::Index bsize = blk.pstar;
    const Eigen::MatrixXd B_b = uf.B.middleRows(bstart, bsize);
    const Eigen::MatrixXd ZcB_b = Zc.middleCols(bstart, bsize) * B_b;
    const Eigen::MatrixXd M_sample_b =
        (ZcB_b.transpose() * ZcB_b) / denom_b(b);

    Eigen::MatrixXd GB_b = Eigen::MatrixXd::Zero(bsize, uf.df);
    Eigen::VectorXd dst = Eigen::VectorXd::Zero(uf.total_rows);
    for (Eigen::Index col = 0; col < uf.df; ++col) {
      dst.setZero();
      apply_gamma_nt_block(blk, /*has_means=*/false, uf.moments,
                           uf.B.col(col), dst, H_buf);
      GB_b.col(col) = dst.segment(bstart, bsize);
    }
    const Eigen::MatrixXd M_nt_b = B_b.transpose() * GB_b;

    const Eigen::VectorXd s_vech = vech_lower(samp.S[static_cast<std::size_t>(b)]);
    const Eigen::VectorXd Bs = B_b.transpose() * s_vech;
    M.noalias() += a * M_sample_b - bb * M_nt_b +
                   bb * c * (Bs * Bs.transpose());
  }
  M = 0.5 * (M + M.transpose()).eval();
  return M;
}

post_expected<Eigen::MatrixXd>
reduced_gamma_unbiased_casewise(
    const UFactor&                            uf,
    const SampleStats&                        samp,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    double                                    denom) {
  const Eigen::VectorXd d = Eigen::VectorXd::Constant(1, denom);
  return reduced_gamma_unbiased_casewise(uf, samp, Zc, d);
}

// ============================================================================
// Eigenvalues + robust statistics
// ============================================================================

post_expected<Eigen::VectorXd>
ugamma_eigenvalues(const Eigen::Ref<const Eigen::MatrixXd>& M) {
  if (M.rows() != M.cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ugamma_eigenvalues: M must be square"));
  }
  Eigen::MatrixXd Msym = 0.5 * (M + M.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      Msym, Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ugamma_eigenvalues: SelfAdjointEigenSolver did not converge"));
  }
  return es.eigenvalues();
}

SatorraBentlerResult
satorra_bentler(double                                    t_ml,
                int                                       df,
                const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept {
  SatorraBentlerResult out;
  out.df = df;
  if (df <= 0 || eigvals.size() == 0) {
    out.scale_c     = std::numeric_limits<double>::quiet_NaN();
    out.chi2_scaled = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  const double sum_lambda = eigvals.sum();
  const double c          = sum_lambda / static_cast<double>(df);
  out.scale_c             = c;
  out.chi2_scaled         = (c > 0.0) ? t_ml / c
                                      : std::numeric_limits<double>::quiet_NaN();
  return out;
}

MeanVarAdjustedResult
mean_var_adjusted(double                                    t_ml,
                  int                                       df,
                  const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept {
  MeanVarAdjustedResult out;
  if (df <= 0 || eigvals.size() == 0) {
    out.chi2_adj = std::numeric_limits<double>::quiet_NaN();
    out.df_adj   = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  const double s1 = eigvals.sum();
  const double s2 = eigvals.squaredNorm();
  if (!(s2 > 0.0)) {
    out.chi2_adj = std::numeric_limits<double>::quiet_NaN();
    out.df_adj   = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  out.df_adj   = (s1 * s1) / s2;
  out.chi2_adj = t_ml * s1 / s2;
  return out;
}

ScaledShiftedResult
scaled_shifted(double                                    t_ml,
               int                                       df,
               const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept {
  ScaledShiftedResult out;
  out.df = df;
  if (df <= 0 || eigvals.size() == 0) {
    out.chi2_adj = std::numeric_limits<double>::quiet_NaN();
    out.scale_a  = std::numeric_limits<double>::quiet_NaN();
    out.shift_b  = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  const double s1 = eigvals.sum();
  const double s2 = eigvals.squaredNorm();
  if (!(s2 > 0.0)) {
    out.chi2_adj = std::numeric_limits<double>::quiet_NaN();
    out.scale_a  = std::numeric_limits<double>::quiet_NaN();
    out.shift_b  = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  const double a = std::sqrt(static_cast<double>(df) / s2);
  const double b = static_cast<double>(df) - a * s1;
  out.scale_a  = a;
  out.shift_b  = b;
  out.chi2_adj = t_ml * a + b;
  return out;
}

// ============================================================================
// robust_se — sandwich standard errors
// ============================================================================

namespace {

// Setup shared by both robust_se overloads. Block-stacked: WΔ is the
// total_rows × q matrix whose b-th row block is W_b·(Δ_b·K) (the b-th block
// itself stacks μ-rows on top of σ-rows when `has_means` — G3b layout, with
// W block-diagonal across (μ, σ): W_μ = M_b⁻¹, W_σ = Γ_NT_cov(M_b)⁻¹). The
// "bread" is the per-unit GLS/Fisher (or observed) information A1 = Σ_b
// (n_b/N)·(Δ_b·K)ᵀW_b(Δ_b·K), and `K_con` (empty ⇒ identity) reparameterizes
// for linear equality constraints (θ = K·α). With K_con active the SE path
// solves in α-space and expands vcov = K·vcov_α·Kᵀ.
struct RobustSetup {
  double          N_total    = 0.0;  // Σ_b n_obs[b]
  Eigen::Index    pstar      = 0;    // Σ_b p*_b — σ-only total
  Eigen::Index    total_rows = 0;    // Σ_b ((has_means ? p_b : 0) + p*_b)
  Eigen::Index    q          = 0;    // n_alpha (= npar when unconstrained)
  bool            has_means  = false;
  Eigen::MatrixXd K_con;             // npar × n_alpha (0/1); empty ⇒ identity
  Eigen::MatrixXd WDelta;            // total_rows × q
  Eigen::MatrixXd bread;             // q × q  (A1, or H_obs_α)
};

post_expected<RobustSetup>
robust_setup(partable::LatentStructure pt, const model::MatrixRep& rep,
             const SampleStats& samp, const Estimates& est,
             InferenceSpec spec, bool /*gamma_hat_overload*/) {
  if (spec.cov == ScoreCovariance::BrowneUnbiased) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: ScoreCovariance::BrowneUnbiased not yet implemented"));
  }
  // Note: the gamma_hat overload accepts multi-group input but the caller is
  // responsible for assembling a block-diagonal Γ̂ pre-weighted by `n_b/N`
  // per block (the per-unit B1 = Σ_b (n_b/N)·Δ_bᵀW_bΓ̂_bW_bΔ_b structure
  // collapses to `(WΔ)ᵀ Γ̂_full (WΔ)` when each block is so weighted). The
  // raw-data overload applies this weighting implicitly via Zc; this
  // overload exposes the requirement to the caller.

  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto& ev = *ev_or;

  // Detect mean structure (G3b). When present, Δ_full per block is
  // [Δμ_b; Δσ_b] and the per-block triangular solves apply L_M⁻¹ to the
  // μ-segment and L_Γ⁻¹ to the σ-segment independently (the block-diagonal
  // structure of Γ_NT across (μ, σ) decouples them).
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  const bool has_means = Jmu_or->size() > 0;
  Eigen::MatrixXd Delta_mu = has_means ? std::move(*Jmu_or)
                                       : Eigen::MatrixXd();
  auto Delta_or = ev.dsigma_dtheta(est.theta);
  if (!Delta_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: dsigma_dtheta failed: " + Delta_or.error().detail));
  }
  Eigen::MatrixXd Delta_sigma = std::move(*Delta_or);    // p*_total × npar
  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: sigma(theta) failed: " + im_or.error().detail));
  }

  RobustSetup s;
  s.has_means = has_means;

  // Geometry first so block offsets are known before Δ-stacking.
  s.N_total = 0.0;
  for (auto n : samp.n_obs) s.N_total += static_cast<double>(n);
  s.pstar = 0;
  s.total_rows = 0;
  std::vector<UFactor::Block> blocks(samp.S.size());
  Eigen::Index row_cursor = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p       = samp.S[b].rows();
    const Eigen::Index pstar_b = p * (p + 1) / 2;
    auto&              blk     = blocks[b];
    blk.p          = p;
    blk.pstar      = pstar_b;
    blk.mu_off     = has_means ? row_cursor : Eigen::Index{-1};
    if (has_means) row_cursor += p;
    blk.row_offset = row_cursor;             // σ-segment start
    row_cursor    += pstar_b;
    blk.n_obs      = static_cast<Eigen::Index>(samp.n_obs[b]);
    blk.Sigma_hat  = im_or->sigma[b];
    blk.S          = samp.S[b];
    s.pstar       += pstar_b;
  }
  s.total_rows = row_cursor;

  const Eigen::Index n_free = Delta_sigma.cols();
  if (has_means && Delta_mu.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Δμ cols " + std::to_string(Delta_mu.cols()) +
            " ≠ Δσ cols " + std::to_string(n_free)));
  }
  // Stack Δ_full = [Δμ_b; Δσ_b] per block (μ on top of σ, per block) in the
  // global `[block0_μ; block0_σ; block1_μ; block1_σ; ...]` layout. Mirrors
  // `build_u_factor`. K_con (equality-constraint reparam) is applied AFTER
  // stacking — it acts on columns only.
  Eigen::MatrixXd Delta_full =
      Eigen::MatrixXd::Zero(s.total_rows, n_free);
  {
    Eigen::Index s_cursor = 0;
    Eigen::Index m_cursor = 0;
    for (const auto& blk : blocks) {
      if (has_means) {
        Delta_full.block(blk.mu_off, 0, blk.p, n_free) =
            Delta_mu.block(m_cursor, 0, blk.p, n_free);
        m_cursor += blk.p;
      }
      Delta_full.block(blk.row_offset, 0, blk.pstar, n_free) =
          Delta_sigma.block(s_cursor, 0, blk.pstar, n_free);
      s_cursor += blk.pstar;
    }
    if (s_cursor != Delta_sigma.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_se: Δσ row count " + std::to_string(Delta_sigma.rows())
              + " ≠ Σ_b p*_b = " + std::to_string(s_cursor)));
    }
    if (has_means && m_cursor != Delta_mu.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_se: Δμ row count " + std::to_string(Delta_mu.rows())
              + " ≠ Σ_b p_b = " + std::to_string(m_cursor)));
    }
  }

  // Constraints: Δ_full → Δ_full·K (total_rows × n_alpha). Mirrors
  // `fit::vcov(info, pt)` / `build_u_factor`.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(con_or.error().kind,
        "robust_se: " + con_or.error().detail));
  }
  Eigen::MatrixXd Delta = std::move(Delta_full);
  if (con_or->active()) {
    s.K_con = con_or->K();
    Delta = (Delta * s.K_con).eval();
  }
  s.q = Delta.cols();
  if (Delta.rows() != s.total_rows) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Δ row count " + std::to_string(Delta.rows()) +
            " ≠ total_rows = " + std::to_string(s.total_rows)));
  }

  // Per-block triangular solves to build A and WΔ. W is block-diagonal
  // across (μ, σ) so the two segments factor through independent Choleskys.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(s.total_rows, s.q);
  s.WDelta          = Eigen::MatrixXd::Zero(s.total_rows, s.q);
  Eigen::MatrixXd bread_exp = Eigen::MatrixXd::Zero(s.q, s.q);
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto&              blk     = blocks[b];
    const double       w_b     = static_cast<double>(samp.n_obs[b]) / s.N_total;
    const Eigen::MatrixXd& M_b =
        (spec.moments == WeightMoments::Structured) ? blk.Sigma_hat : blk.S;
    auto G_or = gamma_nt(M_b);
    if (!G_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_se: gamma_nt failed for block " + std::to_string(b) + ": " +
              G_or.error().detail));
    }
    blk.llt_gamma_nt = Eigen::LLT<Eigen::MatrixXd>(*G_or);
    if (blk.llt_gamma_nt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "robust_se: Γ_NT(weight moments) not positive definite in block " +
              std::to_string(b)));
    }
    if (has_means) {
      blk.llt_M = Eigen::LLT<Eigen::MatrixXd>(M_b);
      if (blk.llt_M.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
            "robust_se: M_b (μ-block of Γ_NT) not positive definite in "
            "block " + std::to_string(b)));
      }
    }
    // A_b = L_Γ_block⁻¹·Δ_b (dual-segment when has_means).
    apply_L_inv_block(blk, has_means, Delta, A);
    // W_b·Δ_b = L_Γ_block⁻ᵀ·A_b (dual-segment, same μ/σ split).
    apply_L_invT_block(blk, has_means, A, s.WDelta);
    // (n_b/N)·Δ_bᵀW_bΔ_b = (n_b/N)·A_bᵀ·A_b (block-restricted product).
    const Eigen::Index bstart = has_means ? blk.mu_off : blk.row_offset;
    const Eigen::Index bsize  = (has_means ? blk.p : 0) + blk.pstar;
    const auto A_b = A.block(bstart, 0, bsize, s.q);
    bread_exp.noalias() += w_b * (A_b.transpose() * A_b);
  }

  // The "bread": A1 (per-unit GLS/Fisher) for Expected; the per-unit observed
  // Hessian (projected through K) for Observed. `*InfoSE.info`
  // is the *unprojected* total observed info; /N → per-unit; Kᵀ(·)K → α-space.
  if (spec.bread == Information::Expected) {
    s.bread = std::move(bread_exp);
  } else {
    Eigen::MatrixXd H_obs;
    if (auto h_or = information_observed_analytic(pt, rep, samp, est);
        h_or.has_value()) {
      H_obs = (*h_or) / s.N_total;
    } else if (auto fd_or = information_observed_fd(pt, rep, samp, est);
               fd_or.has_value()) {
      H_obs = (*fd_or) / s.N_total;
    } else {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_se: could not compute the observed Hessian"));
    }
    if (s.K_con.size() > 0) H_obs = (s.K_con.transpose() * H_obs * s.K_con).eval();
    s.bread = 0.5 * (H_obs + H_obs.transpose()).eval();
  }
  return s;
}

// vcov_robust = (1/N) · A1⁻¹ · B1 · A1⁻¹, expanded through K when constrained:
//   vcov_α = (1/N)·bread⁻¹·meat·bread⁻¹  (q × q);  vcov = K·vcov_α·Kᵀ.
// With bread = A1 (= Σ_b (n_b/N)Δ_bᵀW_bΔ_b) it collapses to (1/N)·A1⁻¹ (the
// naive expected vcov) when meat = A1; with bread = H_obs it's the
// Huber-White sandwich. `meat` (= B1, q × q) is the per-unit
// score variance Σ_b (n_b/N)Δ_bᵀW_bΓ̂_bW_bΔ_b.
post_expected<RobustSeResult>
sandwich_finish(const RobustSetup& s, const Eigen::MatrixXd& meat) {
  if (meat.rows() != s.q || meat.cols() != s.q) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: meat shape mismatch with the reduced parameter count"));
  }
  const Eigen::MatrixXd Bsym = 0.5 * (s.bread + s.bread.transpose());
  Eigen::MatrixXd Binv;
  if (Eigen::LLT<Eigen::MatrixXd> llt(Bsym); llt.info() == Eigen::Success) {
    Binv = llt.solve(Eigen::MatrixXd::Identity(s.q, s.q));
  } else {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Bsym);
    if (ldlt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "robust_se: the bread matrix is not invertible "
          "(under-identified / collinear / non-PD observed Hessian)"));
    }
    Binv = ldlt.solve(Eigen::MatrixXd::Identity(s.q, s.q));
  }
  Eigen::MatrixXd vcov_q = (Binv * meat * Binv) / s.N_total;   // q × q
  vcov_q = 0.5 * (vcov_q + vcov_q.transpose()).eval();
  RobustSeResult out;
  out.vcov = (s.K_con.size() > 0)
                 ? Eigen::MatrixXd(s.K_con * vcov_q * s.K_con.transpose())
                 : std::move(vcov_q);
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose()).eval();
  out.se.resize(out.vcov.rows());
  for (Eigen::Index k = 0; k < out.vcov.rows(); ++k) {
    const double d = out.vcov(k, k);
    out.se(k) = (d >= 0.0) ? std::sqrt(d)
                           : std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

}  // namespace

post_expected<RobustSeResult>
robust_se(partable::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est,
          const Eigen::MatrixXd&    gamma_hat,
          InferenceSpec             spec) {
  auto setup_or =
      robust_setup(std::move(pt), rep, samp, est, spec, /*gamma_hat=*/true);
  if (!setup_or.has_value()) return std::unexpected(setup_or.error());
  const auto& s = *setup_or;
  // Γ̂ shape matches the WΔ row count: cov-only ⇒ s.pstar, means ⇒ s.total_rows.
  const Eigen::Index expected_dim =
      s.has_means ? s.total_rows : s.pstar;
  if (gamma_hat.rows() != expected_dim || gamma_hat.cols() != expected_dim) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Γ̂ is " + std::to_string(gamma_hat.rows()) + "×" +
            std::to_string(gamma_hat.cols()) + ", expected " +
            std::to_string(expected_dim) + "×" + std::to_string(expected_dim) +
            (s.has_means ? " (μ-rows + σ-rows per block)" : " (σ-only)")));
  }
  // meat = ΔᵀWΓ̂WΔ = (WΔ)ᵀ·Γ̂·(WΔ). For multi-block callers of this overload,
  // Γ̂ must already carry the documented per-block n_b/N scaling.
  const Eigen::MatrixXd meat = s.WDelta.transpose() * gamma_hat * s.WDelta;
  return sandwich_finish(s, meat);
}

post_expected<RobustSeResult>
robust_se(partable::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est,
          const RawData&            raw,
          InferenceSpec             spec) {
  auto setup_or =
      robust_setup(std::move(pt), rep, samp, est, spec, /*gamma_hat=*/false);
  if (!setup_or.has_value()) return std::unexpected(setup_or.error());
  const auto& s = *setup_or;
  auto Zc_or = casewise_contributions(raw, samp, /*include_means=*/s.has_means);
  if (!Zc_or.has_value()) return std::unexpected(Zc_or.error());
  const Eigen::MatrixXd& Zc = *Zc_or;
  const Eigen::Index expected_cols =
      s.has_means ? s.total_rows : s.pstar;
  if (Zc.cols() != expected_cols) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Z_c has " + std::to_string(Zc.cols()) +
            " columns, expected " + std::to_string(expected_cols) +
            (s.has_means ? " ([μ; σ] per block)" : " (σ-only)")));
  }
  // meat = B1 = Σ_b (n_b/N)·Δ_bᵀW_bΓ̂_bW_bΔ_b.  With WΔ stacked block-diagonal
  // and Γ̂ = Z_cᵀZ_c / N_total this is exactly (Z_c·WΔ)ᵀ(Z_c·WΔ) / N_total
  // (each block's rows of Z_c·WΔ pick up only that block's WΔ slice, so the
  // outer-product accumulates Σ_b (W_bΔ_b)ᵀ(n_bΓ̂_b)(W_bΔ_b)).
  const Eigen::MatrixXd ZcWDelta = Zc * s.WDelta;          // N_total × q
  const Eigen::MatrixXd meat = (ZcWDelta.transpose() * ZcWDelta) / s.N_total;
  return sandwich_finish(s, meat);
}

}  // namespace magmaan::fit
