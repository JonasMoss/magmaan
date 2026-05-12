#include "latva/fit/robust.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/fit/constraints.hpp"        // EqConstraints / build_eq_constraints
#include "latva/fit/inference.hpp"          // AnalyticObservedInfoSE / FdObservedInfoSE
#include "latva/fit/raw_data.hpp"
#include "latva/fit/resolve_fixed_x.hpp"
#include "latva/model/model_evaluator.hpp"

namespace latva::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// Pack a symmetric p×p matrix into lower-tri column-major vech.
Eigen::VectorXd vech_lower(const Eigen::Ref<const Eigen::MatrixXd>& M) {
  const Eigen::Index p = M.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::VectorXd out(pstar);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j)
    for (Eigen::Index i = j; i < p; ++i)
      out(k++) = M(i, j);
  return out;
}

// Unpack a lower-tri column-major vech vector back into a symmetric p×p
// matrix. Caller pre-sizes M (cheap when used in a loop with reused buffer).
void vech_unpack(const Eigen::Ref<const Eigen::VectorXd>& x,
                 Eigen::Index p, Eigen::Ref<Eigen::MatrixXd> M) {
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      M(i, j) = x(k);
      if (i != j) M(j, i) = x(k);
      ++k;
    }
  }
}

// For an ObservedHessian UFactor: C = I − A·H_obs⁻¹·Aᵀ  (p* × p*, symmetric).
// `UΓ = L_Γ⁻ᵀ·C·L_Γ⁻¹·Γ` and `eigvals(UΓ) = eigvals(C · L_Γ⁻¹ΓL_Γ⁻ᵀ)`.
Eigen::MatrixXd observed_C(const UFactor& uf) {
  Eigen::MatrixXd C = Eigen::MatrixXd::Identity(uf.pstar, uf.pstar);
  C.noalias() -= uf.A * uf.H_obs_inv * uf.A.transpose();
  return 0.5 * (C + C.transpose()).eval();
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
  if (spec.bread == Information::Observed && samp.S.size() != 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: Information::Observed bread is single-block only "
        "in v1 (the H_obs units are clean only single-block)"));
  }

  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto& ev = *ev_or;

  // v0 gates mean structure off — μ-rows in Δ would need to be prepended
  // per block with the Σ̂_b μ-block in Γ_NT. Path is straightforward; the
  // single-block cov-only goldens are the target for this round.
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  if (Jmu_or->size() > 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: mean structure not yet supported in v0"));
  }

  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: dsigma_dtheta failed: " + J_or.error().detail));
  }
  Eigen::MatrixXd Delta = std::move(*J_or);

  // Linear equality constraints (shared labels, `a == b`): the model is
  // really parameterized by α with θ = K·α, so the relevant Jacobian is
  // Δ·K (p* × n_alpha) and `df` becomes p* − n_alpha (= the unconstrained
  // p* − npar, plus the constraint rank). `K_con` is empty when there are
  // none — the identity reparameterization. Mirrors `finalize_inference`.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(con_or.error().kind,
        "build_u_factor: " + con_or.error().detail));
  }
  Eigen::MatrixXd K_con;   // empty ⇒ no constraints
  if (con_or->active()) {
    K_con = con_or->K();
    Delta = (Delta * K_con).eval();
  }
  const Eigen::Index q = Delta.cols();

  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: sigma(theta) failed: " + im_or.error().detail));
  }

  // Per-block geometry + LLT of the weight matrix Γ_NT(M_b)⁻¹, where M_b
  // is Σ̂_b (structured) or S_b (unstructured) per `spec.moments`. Σ̂_b and
  // S_b are both copied into the block — the evaluator's internal Σ̂
  // buffer is overwritten by later sigma() calls, and `reduced_gamma_nt`
  // needs whichever moments matched the bread.
  UFactor uf;
  uf.moments = spec.moments;
  uf.blocks.resize(samp.S.size());
  Eigen::Index row_cursor = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p     = samp.S[b].rows();
    const Eigen::Index pstar = p * (p + 1) / 2;
    auto&              blk   = uf.blocks[b];
    blk.p          = p;
    blk.pstar      = pstar;
    blk.row_offset = row_cursor;
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
    row_cursor += pstar;
  }
  uf.pstar = row_cursor;
  if (Delta.rows() != uf.pstar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "build_u_factor: Δ row count " + std::to_string(Delta.rows()) +
            " ≠ Σ_b p*_b = " + std::to_string(uf.pstar)));
  }

  // Per-block triangular solve: A_b = L_Γ,b⁻¹·Δ_b. Stack into global A.
  Eigen::MatrixXd A(uf.pstar, q);
  for (const auto& blk : uf.blocks) {
    A.block(blk.row_offset, 0, blk.pstar, q) =
        blk.llt_gamma_nt.matrixL().solve(
            Delta.block(blk.row_offset, 0, blk.pstar, q));
  }

  // ── Observed-Hessian bread (single-block; the MLR convention) ───────────
  // U = L_Γ⁻ᵀ·(I − A·H_obs⁻¹·Aᵀ)·L_Γ⁻¹  (not idempotent). We don't form U;
  // store A and H_obs⁻¹ and let `reduced_gamma_*` assemble the spectrum.
  if (spec.bread == Information::Observed) {
    // Observed Hessian (q × q) in the *per-unit* units of ΔᵀWΔ = AᵀA.
    // `AnalyticObservedInfoSE.info` is the total observed info = N·(per-unit
    // Hessian/2)·2 = N·(per-unit info) at θ̂; dividing by N gives the
    // per-unit form, which equals AᵀA up to the H2 residual term. FD
    // fallback for models the analytic path can't do (e.g. mean structure).
    // `*InfoSE::info` is the *unprojected* npar×npar total observed info;
    // /N gives the per-unit form; project through K to the n_alpha space
    // (= AᵀA with the reparameterized Δ·K). Trivial when K_con is empty.
    Eigen::MatrixXd H_obs;
    if (auto h_or = AnalyticObservedInfoSE{}.compute(pt, rep, samp, est);
        h_or.has_value()) {
      H_obs = h_or->info / static_cast<double>(samp.n_obs[0]);
    } else if (auto fd_or = FdObservedInfoSE{}.compute(pt, rep, samp, est);
               fd_or.has_value()) {
      H_obs = fd_or->info / static_cast<double>(samp.n_obs[0]);
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
    uf.df        = uf.pstar - q;       // the χ²-reference df
    if (uf.df <= 0) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "build_u_factor: model is saturated (df ≤ 0)"));
    }
    return uf;
  }

  // ── Expected-info projection bread (`U = B·Bᵀ`) ─────────────────────────
  // QR of A → orthonormal basis N of ker(Aᵀ). Trailing (p* − rank(A))
  // columns of Q form the basis. We assume full column rank of A (=
  // full column rank of Δ in the Γ_NT-metric); if not, the trailing
  // columns still span ker(Aᵀ) but `df` is larger than `p* − q`.
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
  // Estimate rank via the absolute diagonal of R. Cheap and adequate for
  // SEM sizes — rank-deficient Δ is a "your model is broken" condition
  // and the SE methods would already complain.
  const Eigen::MatrixXd R = qr.matrixQR().topLeftCorner(
      std::min(uf.pstar, q), q).triangularView<Eigen::Upper>();
  const double tol = std::numeric_limits<double>::epsilon() *
                     static_cast<double>(std::max(uf.pstar, q)) *
                     R.diagonal().cwiseAbs().maxCoeff();
  Eigen::Index r_eff = 0;
  for (Eigen::Index k = 0; k < std::min(uf.pstar, q); ++k)
    if (std::abs(R(k, k)) > tol) ++r_eff;
  const Eigen::Index df_global = uf.pstar - r_eff;
  if (df_global <= 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "build_u_factor: model is saturated (df = " +
            std::to_string(df_global) +
            "); robust UΓ inference has no nontrivial spectrum"));
  }
  uf.df = df_global;

  // Form N = trailing df columns of Q without materialising full p*×p* Q:
  //   N = Q · [0; I_df]
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(uf.pstar, df_global);
  E.bottomRows(df_global).setIdentity();
  const Eigen::MatrixXd N = qr.householderQ() * E;

  // Per-block triangular-adjoint solve: B_b = L_Γ,b⁻ᵀ·N_b.
  uf.B.resize(uf.pstar, df_global);
  for (const auto& blk : uf.blocks) {
    uf.B.block(blk.row_offset, 0, blk.pstar, df_global) =
        blk.llt_gamma_nt.matrixL().adjoint().solve(
            N.block(blk.row_offset, 0, blk.pstar, df_global));
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
  if (Zc.cols() != uf.pstar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "reduced_gamma_sample: Zc has " + std::to_string(Zc.cols()) +
            " columns, expected " + std::to_string(uf.pstar)));
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
  // Γ̂ is block-diagonal: each block's vech contributions live in a disjoint
  // column slice of Zc (zeros elsewhere — `casewise_contributions` layout),
  // so `Zc[:, cols_b] · B[rows_b, :]` is nonzero only for rows in block b and
  // (·)ᵀ(·) accumulates exactly B_bᵀ(n_b Γ̂_b)B_b. Divide each by its own n_b
  // (or the recycled scalar) and sum — `M = Σ_b B_bᵀ Γ̂_b B_b`.
  if (uf.kind == UFactor::Kind::ObservedHessian) {
    // M̃ = diag_b( L_Γ,b⁻¹·Γ̂_b·L_Γ,b⁻ᵀ ); off-diagonal blocks vanish for the
    // same disjoint-slice reason.
    Eigen::MatrixXd Mtilde = Eigen::MatrixXd::Zero(uf.pstar, uf.pstar);
    for (Eigen::Index b = 0; b < nb; ++b) {
      const auto& blk = uf.blocks[static_cast<std::size_t>(b)];
      const Eigen::MatrixXd Xb =
          Zc.middleCols(blk.row_offset, blk.pstar);                 // N × p*_b
      // Xb·L_Γ,b⁻ᵀ = (L_Γ,b⁻¹·Xbᵀ)ᵀ.
      const Eigen::MatrixXd ZcLw_b =
          blk.llt_gamma_nt.matrixL().solve(Xb.transpose()).transpose();
      Mtilde.block(blk.row_offset, blk.row_offset, blk.pstar, blk.pstar)
          .noalias() += (ZcLw_b.transpose() * ZcLw_b) / denom_b(b);
    }
    return symm_product_eigbasis(Mtilde, observed_C(uf));
  }
  // ProjectionExpected.
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(uf.df, uf.df);
  for (Eigen::Index b = 0; b < nb; ++b) {
    const auto& blk = uf.blocks[static_cast<std::size_t>(b)];
    const Eigen::MatrixXd ZcB_b =
        Zc.middleCols(blk.row_offset, blk.pstar) *
        uf.B.middleRows(blk.row_offset, blk.pstar);                  // N × df
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
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(uf.df, uf.df);
  for (std::size_t i = 0; i < zc_rows.size(); ++i) {
    if (zc_rows[i].size() != uf.pstar) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "reduced_gamma_sample_streaming: zc_rows[" + std::to_string(i) +
              "] has size " + std::to_string(zc_rows[i].size()) +
              ", expected " + std::to_string(uf.pstar)));
    }
    const Eigen::VectorXd r = uf.B.transpose() * zc_rows[i];
    M.noalias() += r * r.transpose();
  }
  M /= denom;
  M = 0.5 * (M + M.transpose()).eval();
  return M;
}

post_expected<Eigen::MatrixXd>
casewise_contributions(const RawData& raw, const SampleStats& samp) {
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "casewise_contributions: RawData and SampleStats block count mismatch"));
  }
  // Sum n_obs across blocks; sum pstar for column layout.
  Eigen::Index n_rows = 0;
  Eigen::Index n_cols = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    n_rows += raw.X[b].rows();
    n_cols += samp.S[b].rows() * (samp.S[b].rows() + 1) / 2;
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
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd xi    = Xc.row(i).transpose();
      const Eigen::MatrixXd outer = xi * xi.transpose();
      const Eigen::VectorXd d_i   = vech_lower(outer) - s_vech;
      Zc.block(row_cursor + i, col_cursor, 1, pstar) = d_i.transpose();
    }
    row_cursor += n;
    col_cursor += pstar;
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
  // Correct operator: Γ_NT·vech(M) = vech(2·Σ̂·M_½·Σ̂), where M_½ is the
  // off-diagonal-halved variant of M. Derivation: Γ_NT = 2·D⁺(Σ⊗Σ)D⁺ᵀ
  // and D⁺ᵀ·vech(M) takes vech(M) into vec-space with off-diagonals
  // halved (since (DᵀD)⁻¹ has 1 on the diagonal entries and 1/2 on the
  // off-diagonal duplicated entries). Verified on a 2×2 worked example
  // against the closed-form `gamma_nt(Σ)` matrix-multiply path.
  Eigen::MatrixXd GB(uf.pstar, uf.df);
  Eigen::MatrixXd H_buf;     // scratch — symmetric M_½
  for (const auto& blk : uf.blocks) {
    // Meat moments match the bread: Σ̂_b for structured, S_b for unstructured.
    const Eigen::MatrixXd& Mmoments =
        (uf.moments == WeightMoments::Structured) ? blk.Sigma_hat : blk.S;
    H_buf.resize(blk.p, blk.p);
    for (Eigen::Index c = 0; c < uf.df; ++c) {
      vech_unpack(uf.B.block(blk.row_offset, c, blk.pstar, 1).col(0),
                  blk.p, H_buf);
      // Halve off-diagonals so vech_unpack(x) becomes the M_½ form the
      // Γ_NT operator expects (both M(i,j) and M(j,i) for i≠j).
      for (Eigen::Index j = 0; j < blk.p; ++j)
        for (Eigen::Index i = 0; i < blk.p; ++i)
          if (i != j) H_buf(i, j) *= 0.5;
      const Eigen::MatrixXd Z = Mmoments * H_buf * Mmoments;
      // Pack 2·vech(Z) back (no extra diag halving — Z is the result
      // matrix, not an input).
      Eigen::Index k = 0;
      for (Eigen::Index j = 0; j < blk.p; ++j) {
        for (Eigen::Index i = j; i < blk.p; ++i) {
          GB(blk.row_offset + k, c) = 2.0 * Z(i, j);
          ++k;
        }
      }
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
// p*_total × q matrix whose b-th row block is W_b·(Δ_b·K), the "bread" is
// the per-unit GLS/Fisher (or observed) information A1 = Σ_b (n_b/N)·
// (Δ_b·K)ᵀW_b(Δ_b·K), and `K_con` (empty ⇒ identity) reparameterizes for
// linear equality constraints (θ = K·α, df = p* − n_alpha). With K_con
// active the SE path solves in α-space and expands vcov = K·vcov_α·Kᵀ.
struct RobustSetup {
  double          N_total = 0.0;     // Σ_b n_obs[b]
  Eigen::Index    pstar   = 0;       // Σ_b p*_b
  Eigen::Index    q       = 0;       // n_alpha (= npar when unconstrained)
  Eigen::MatrixXd K_con;             // npar × n_alpha (0/1); empty ⇒ identity
  Eigen::MatrixXd WDelta;            // p*_total × q  (b-th block = W_b·(Δ_b·K))
  Eigen::MatrixXd bread;             // q × q  (A1, or H_obs_α)
};

post_expected<RobustSetup>
robust_setup(partable::LatentStructure pt, const model::MatrixRep& rep,
             const SampleStats& samp, const Estimates& est,
             InferenceSpec spec, bool gamma_hat_overload) {
  if (spec.cov == ScoreCovariance::BrowneUnbiased) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: ScoreCovariance::BrowneUnbiased not yet implemented"));
  }
  if (spec.bread == Information::Observed && samp.S.size() != 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: the Observed-info bread is single-block only (the H_obs "
        "units are clean only single-block); use the Expected bread for "
        "multi-group robust SEs"));
  }
  if (gamma_hat_overload && samp.S.size() != 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: the gamma_hat overload is single-block only — use the "
        "raw-data overload for multi-group (per-block weighting is implicit)"));
  }

  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto& ev = *ev_or;

  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  if (Jmu_or->size() > 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: mean structure not yet supported"));
  }
  auto Delta_or = ev.dsigma_dtheta(est.theta);
  if (!Delta_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: dsigma_dtheta failed: " + Delta_or.error().detail));
  }
  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: sigma(theta) failed: " + im_or.error().detail));
  }

  RobustSetup s;
  // Constraints: Δ → Δ·K (p*_total × n_alpha). Mirrors `finalize_inference`.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(con_or.error().kind,
        "robust_se: " + con_or.error().detail));
  }
  Eigen::MatrixXd Delta = std::move(*Delta_or);          // p*_total × npar
  if (con_or->active()) {
    s.K_con = con_or->K();
    Delta = (Delta * s.K_con).eval();                    // p*_total × n_alpha
  }
  s.q = Delta.cols();

  s.N_total = 0.0;
  for (auto n : samp.n_obs) s.N_total += static_cast<double>(n);
  s.pstar = 0;
  for (const auto& Sb : samp.S) s.pstar += Sb.rows() * (Sb.rows() + 1) / 2;
  if (Delta.rows() != s.pstar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Δ row count " + std::to_string(Delta.rows()) +
            " ≠ Σ_b p*_b = " + std::to_string(s.pstar)));
  }

  s.WDelta.resize(s.pstar, s.q);
  Eigen::MatrixXd bread_exp = Eigen::MatrixXd::Zero(s.q, s.q);
  Eigen::Index row_cursor = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p       = samp.S[b].rows();
    const Eigen::Index pstar_b = p * (p + 1) / 2;
    const double       w_b     = static_cast<double>(samp.n_obs[b]) / s.N_total;
    const Eigen::MatrixXd& M_b =
        (spec.moments == WeightMoments::Structured) ? im_or->sigma[b]
                                                    : samp.S[b];
    auto G_or = gamma_nt(M_b);
    if (!G_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_se: gamma_nt failed for block " + std::to_string(b) + ": " +
              G_or.error().detail));
    }
    Eigen::LLT<Eigen::MatrixXd> llt(*G_or);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "robust_se: Γ_NT(weight moments) not positive definite in block " +
              std::to_string(b)));
    }
    const Eigen::MatrixXd Delta_b = Delta.block(row_cursor, 0, pstar_b, s.q);
    const Eigen::MatrixXd A_b      = llt.matrixL().solve(Delta_b);            // L_b⁻¹·Δ_b
    const Eigen::MatrixXd WDelta_b = llt.matrixL().transpose().solve(A_b);    // W_b·Δ_b
    s.WDelta.block(row_cursor, 0, pstar_b, s.q) = WDelta_b;
    bread_exp.noalias() += w_b * (A_b.transpose() * A_b);   // (n_b/N)·Δ_bᵀW_bΔ_b
    row_cursor += pstar_b;
  }

  // The "bread": A1 (per-unit GLS/Fisher) for Expected; the per-unit observed
  // Hessian (single-block, projected through K) for Observed. `*InfoSE.info`
  // is the *unprojected* total observed info; /N → per-unit; Kᵀ(·)K → α-space.
  if (spec.bread == Information::Expected) {
    s.bread = std::move(bread_exp);
  } else {
    Eigen::MatrixXd H_obs;
    if (auto h_or = AnalyticObservedInfoSE{}.compute(pt, rep, samp, est);
        h_or.has_value()) {
      H_obs = h_or->info / s.N_total;
    } else if (auto fd_or = FdObservedInfoSE{}.compute(pt, rep, samp, est);
               fd_or.has_value()) {
      H_obs = fd_or->info / s.N_total;
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
// Huber-White sandwich (single-block). `meat` (= B1, q × q) is the per-unit
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
  if (gamma_hat.rows() != s.pstar || gamma_hat.cols() != s.pstar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Γ̂ is " + std::to_string(gamma_hat.rows()) + "×" +
            std::to_string(gamma_hat.cols()) + ", expected p*×p* = " +
            std::to_string(s.pstar) + "×" + std::to_string(s.pstar)));
  }
  // meat = ΔᵀWΓ̂WΔ = (WΔ)ᵀ·Γ̂·(WΔ).  (Single-block: no n_b/N weight.)
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
  auto Zc_or = casewise_contributions(raw, samp);
  if (!Zc_or.has_value()) return std::unexpected(Zc_or.error());
  const Eigen::MatrixXd& Zc = *Zc_or;
  if (Zc.cols() != s.pstar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_se: Z_c has " + std::to_string(Zc.cols()) +
            " columns, expected p* = " + std::to_string(s.pstar)));
  }
  // meat = B1 = Σ_b (n_b/N)·Δ_bᵀW_bΓ̂_bW_bΔ_b.  With WΔ stacked block-diagonal
  // and Γ̂ = Z_cᵀZ_c / N_total this is exactly (Z_c·WΔ)ᵀ(Z_c·WΔ) / N_total
  // (each block's rows of Z_c·WΔ pick up only that block's WΔ slice, so the
  // outer-product accumulates Σ_b (W_bΔ_b)ᵀ(n_bΓ̂_b)(W_bΔ_b)).
  const Eigen::MatrixXd ZcWDelta = Zc * s.WDelta;          // N_total × q
  const Eigen::MatrixXd meat = (ZcWDelta.transpose() * ZcWDelta) / s.N_total;
  return sandwich_finish(s, meat);
}

}  // namespace latva::fit
