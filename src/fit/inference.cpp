#include "latva/fit/inference.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/fit/constraints.hpp"
#include "latva/fit/ml.hpp"
#include "latva/fit/resolve_fixed_x.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/model/model_evaluator.hpp"

namespace latva::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// vech length and column-major lower-tri index — matches ml.cpp.
inline Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}
inline Eigen::Index vech_index(Eigen::Index p, Eigen::Index r,
                               Eigen::Index c) noexcept {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

// Common chi² + df calculation. Both fixed.x-aware and matches the formula
// in ExpectedInfoSE::compute (kept in sync).
struct Chi2Df { double chi2; int df; };

Chi2Df compute_chi2_df(const partable::LatentStructure& pt,
                       const SampleStats&        samp,
                       const Estimates&          est) {
  double n_total = 0.0;
  int    df_moments = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    n_total    += static_cast<double>(samp.n_obs[b]);
    const int p = static_cast<int>(samp.S[b].rows());
    df_moments += p * (p + 1) / 2;
    // Mean structure: each block also contributes p mean moments when
    // the caller has supplied a sample mean for that block.
    if (b < samp.mean.size() && samp.mean[b].size() > 0) {
      df_moments += p;
    }
  }
  int fixed_x_moments = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == parse::Op::Covariance && pt.exo[i] == 1 && pt.free[i] == 0) {
      ++fixed_x_moments;
    }
  }
  const double chi2 = n_total * est.fmin;
  const int    df   = df_moments - fixed_x_moments -
                      static_cast<int>(est.theta.size());
  return {chi2, df};
}

// Invert an SPD matrix (LLT, LDLT fallback); error if not invertible.
post_expected<Eigen::MatrixXd>
invert_spd(const Eigen::MatrixXd& A, const char* what) {
  const Eigen::Index n = A.rows();
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  if (llt.info() == Eigen::Success) {
    return Eigen::MatrixXd(llt.solve(Eigen::MatrixXd::Identity(n, n)));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not invertible (under-identified model "
        "or collinear free parameters)"));
  }
  return Eigen::MatrixXd(ldlt.solve(Eigen::MatrixXd::Identity(n, n)));
}

Eigen::VectorXd se_from_vcov(const Eigen::MatrixXd& vcov) {
  Eigen::VectorXd se(vcov.rows());
  for (Eigen::Index k = 0; k < vcov.rows(); ++k) {
    const double d = vcov(k, k);
    se(k) = (d >= 0.0) ? std::sqrt(d)
                       : std::numeric_limits<double>::quiet_NaN();
  }
  return se;
}

// Package the (full, npar × npar) information matrix into an Inference,
// applying any linear equality constraints: vcov(θ̂) = K (Kᵀ I K)⁻¹ Kᵀ (the
// constrained covariance — equality-tied params get equal SEs), df += rank
// (= npar − n_alpha). The unconstrained case is K = I, rank = 0 (plain
// info⁻¹). Shared by all three SE methods.
post_expected<Inference>
finalize_inference(Eigen::MatrixXd info, double chi2, int df,
                   const EqConstraints& con,
                   std::vector<std::string> warnings = {}) {
  if (!con.active()) {
    auto vcov_or = invert_spd(info, "information matrix");
    if (!vcov_or.has_value()) return std::unexpected(vcov_or.error());
    Eigen::VectorXd se = se_from_vcov(*vcov_or);
    return Inference{std::move(info), std::move(*vcov_or), std::move(se),
                     chi2, df, std::move(warnings)};
  }
  const Eigen::MatrixXd K = con.K();                            // npar × n_alpha (0/1)
  const Eigen::MatrixXd info_alpha = K.transpose() * info * K;  // n_alpha × n_alpha
  auto vcov_alpha_or = invert_spd(info_alpha,
                                  "reduced (constrained) information matrix");
  if (!vcov_alpha_or.has_value()) return std::unexpected(vcov_alpha_or.error());
  Eigen::MatrixXd vcov_full = K * (*vcov_alpha_or) * K.transpose();  // npar × npar
  Eigen::VectorXd se = se_from_vcov(vcov_full);
  return Inference{std::move(info), std::move(vcov_full), std::move(se),
                   chi2, df + con.rank, std::move(warnings)};
}

// Resolve fixed.x fixed_values, build the evaluator, and validate shapes.
// Returns the evaluator on success. `pt` is taken by mutable reference and
// gets its fixed.x `fixed_value`s filled — each method owns its own copy of
// pt (the public API takes pt by value).
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

// Post-fit Heywood / improper-solution diagnostics: a free *diagonal*
// Ψ (latent) or Θ (residual) parameter that came out negative. lavaan
// emits the equivalent "some estimated ov variances are negative" /
// "some estimated lv variances are negative" warning. v0 scope (per the
// roadmap's G5): just the negative-variance check at θ̂ — no box constraints,
// no implied-correlation > 1 check (a non-PD implied Σ is already a hard
// error upstream; a negative implied variance can only come from a negative
// Ψ/Θ diagonal, which this catches). Never fails — returns an empty vector
// for a clean solution.
std::vector<std::string>
heywood_warnings(const model::ModelEvaluator& ev,
                 const model::MatrixRep&      rep,
                 const Estimates&             est) {
  std::vector<std::string> out;
  const auto locs = ev.param_locations();
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    if (L.row != L.col) continue;            // off-diagonal: a covariance, not a variance
    if (L.mat != model::MatId::Psi && L.mat != model::MatId::Theta) continue;
    if (!(est.theta(static_cast<Eigen::Index>(k)) < 0.0)) continue;
    const std::size_t blk = static_cast<std::size_t>(L.block);
    const auto idx        = static_cast<std::size_t>(L.row);
    std::string vname;
    if (L.mat == model::MatId::Theta) {
      if (blk < rep.ov_names.size() && idx < rep.ov_names[blk].size())
        vname = rep.ov_names[blk][idx];
    } else {
      if (blk < rep.lv_names.size() && idx < rep.lv_names[blk].size())
        vname = rep.lv_names[blk][idx];
    }
    if (vname.empty()) vname = std::to_string(idx + 1);
    const char* what = (L.mat == model::MatId::Theta)
                           ? "residual variance"
                           : "latent variance";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g",
                  est.theta(static_cast<Eigen::Index>(k)));
    std::string msg = "negative " + std::string(what) + " estimate (Heywood "
        "case): " + vname;
    if (rep.dims.size() > 1)
      msg += " (block " + std::to_string(blk + 1) + ")";
    msg += " = " + std::string(buf);
    out.push_back(std::move(msg));
  }
  return out;
}

}  // namespace

post_expected<Inference>
ExpectedInfoSE::compute(partable::LatentStructure        pt,
                        const model::MatrixRep&   rep,
                        const SampleStats&        samp,
                        const Estimates&          est) const {
  // Fill in any fixed.x rows' fixed_value from the sample covariance
  // — mirrors `fit()`. Without this, ModelEvaluator sees NA fixed_values as
  // zero and the implied Σ collapses on path-style models with fixed
  // exogenous variances.
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "resolve_fixed_x_from_sample failed: " + e.error().detail));
  }
  // Re-derive the evaluator. `fit()` did this once already; we don't share
  // it because the caller-side API stays symmetric.
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;

  const std::size_t n_free   = ev.n_free();
  const std::size_t n_blocks = samp.S.size();
  if (static_cast<std::size_t>(est.theta.size()) != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " ≠ evaluator n_free " + std::to_string(n_free)));
  }
  if (samp.S.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "SampleStats and evaluator have different block counts"));
  }

  // Σ(θ̂) and full Jacobian J = ∂vech(Σ)/∂θ.
  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.sigma(θ̂) failed: " + sm_or.error().detail));
  }
  const auto& sm = *sm_or;

  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.dsigma_dtheta(θ̂) failed: " + J_or.error().detail));
  }
  const Eigen::MatrixXd& J = *J_or;

  // dμ/dθ — empty for covariance-only models. When mean structure is
  // present, expected info gains a per-block `ν_a' Σ_b⁻¹ ν_b` term.
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.dmu_dtheta(θ̂) failed: " + Jmu_or.error().detail));
  }
  const Eigen::MatrixXd& Jmu = *Jmu_or;
  const bool has_means = (Jmu.size() > 0);

  // Per-block precompute: Σ_b⁻¹ and weight (n_b − 1)/2. Σ⁻¹ as dense — small
  // p in v0; cheap.
  std::vector<Eigen::MatrixXd> SigmaInv(n_blocks);
  std::vector<double>          weight(n_blocks, 0.0);
  std::vector<Eigen::Index>    p_dim(n_blocks, 0);
  std::vector<Eigen::Index>    vech_off(n_blocks, 0);

  Eigen::Index running = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    // Symmetrize: Λ(I−B)⁻¹Ψ(I−B)⁻ᵀΛᵀ is mathematically symmetric, but
    // float non-associativity in the chained products introduces O(1e-14)
    // asymmetric noise that can flip a near-zero eigenvalue negative on
    // saturated models. Averaging upper/lower triangles restores the
    // symmetric part exactly without affecting well-conditioned cases.
    const Eigen::MatrixXd Sigma_b = 0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
    const Eigen::Index p = Sigma_b.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_b);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "implied Σ for block " + std::to_string(b) +
              " is not positive definite at θ̂"));
    }
    SigmaInv[b] = llt.solve(Eigen::MatrixXd::Identity(p, p));
    // n/2, not (n-1)/2 — matches lavaan's default `likelihood = "normal"`
    // convention (vs Wishart's (n-1)/2). Determines both info scaling and
    // the matching chi² formula below.
    weight[b]   = static_cast<double>(samp.n_obs[b]) / 2.0;
    p_dim[b]    = p;
    vech_off[b] = running;
    running += vech_len(p);
  }
  if (J.rows() != running) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "Jacobian row count " + std::to_string(J.rows()) +
            " ≠ total vech length " + std::to_string(running)));
  }

  // For each free parameter k, materialize per-block T_{k,b} = Σ_b⁻¹ · M_{k,b}
  // where M_{k,b} is the un-vech of J's column k restricted to block b.
  // Storage cost: n_free × Σ_b p_b² doubles — trivial at v0 sizes.
  std::vector<std::vector<Eigen::MatrixXd>> T(n_free,
      std::vector<Eigen::MatrixXd>(n_blocks));
  Eigen::MatrixXd M;  // reused buffer
  for (std::size_t k = 0; k < n_free; ++k) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p = p_dim[b];
      M.setZero(p, p);
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          const double v = J(vech_off[b] + vech_index(p, r, c),
                             static_cast<Eigen::Index>(k));
          M(r, c) = v;
          if (r != c) M(c, r) = v;
        }
      }
      T[k][b].noalias() = SigmaInv[b] * M;
    }
  }

  // Mean-structure precompute: η_{k,b} = Σ_b⁻¹ · ν_{k,b}, where
  // ν_{k,b} is the block-b segment of Jmu's column k. Used in the mean
  // contribution to expected info: ν_a' η_b per block.
  std::vector<Eigen::Index> mu_off(n_blocks, 0);
  std::vector<std::vector<Eigen::VectorXd>> eta;
  if (has_means) {
    Eigen::Index running_p = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) {
      mu_off[b] = running_p;
      running_p += p_dim[b];
    }
    eta.assign(n_free, std::vector<Eigen::VectorXd>(n_blocks));
    for (std::size_t k = 0; k < n_free; ++k) {
      for (std::size_t b = 0; b < n_blocks; ++b) {
        const Eigen::VectorXd nu_kb = Jmu.col(static_cast<Eigen::Index>(k))
            .segment(mu_off[b], p_dim[b]);
        eta[k][b].noalias() = SigmaInv[b] * nu_kb;
      }
    }
  }

  // Pairwise traces. trace(T_a · T_b) = Σ_{i,j} T_a(j,i) · T_b(i,j)
  //                                   = (T_a.transpose().array() * T_b.array()).sum()
  // Mean-structure term: the standard expected-info formula is
  //   I[a,b] = (N/2) tr(W M_a W M_b) + N · ν_a' W ν_b
  //         = (N/2) [tr(W M_a W M_b) + 2 · ν_a' η_b]
  // so the mean dot product carries a factor of 2 inside the (n/2) scale.
  Eigen::MatrixXd info = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(n_free), static_cast<Eigen::Index>(n_free));
  for (std::size_t a = 0; a < n_free; ++a) {
    for (std::size_t b = a; b < n_free; ++b) {
      double acc = 0.0;
      for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        double per_block = (T[a][blk].transpose().array() * T[b][blk].array()).sum();
        if (has_means) {
          const Eigen::VectorXd nu_a = Jmu.col(static_cast<Eigen::Index>(a))
              .segment(mu_off[blk], p_dim[blk]);
          per_block += 2.0 * nu_a.dot(eta[b][blk]);
        }
        acc += weight[blk] * per_block;
      }
      info(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = acc;
      if (a != b)
        info(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) = acc;
    }
  }

  // χ² and df via the shared helper (handles cov + mean moments and the
  // fixed.x adjustment uniformly with the other SE methods).
  const auto [chi2, df] = compute_chi2_df(pt, samp, est);

  // Linear equality constraints (shared labels, `a == b`) — `finalize_inference`
  // projects info → reduced → vcov → expanded, and bumps df by the constraint
  // rank. Trivial (K = I, rank = 0) when there are none.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  return finalize_inference(std::move(info), chi2, df, *con_or,
                            heywood_warnings(ev, rep, est));
}

// ----------------------------------------------------------------------------
// FdObservedInfoSE — observed info via central-difference Hessian of the
// analytic ML gradient. Works for any model `fit()` can fit.
// ----------------------------------------------------------------------------

post_expected<Inference>
FdObservedInfoSE::compute(partable::LatentStructure        pt,
                          const model::MatrixRep&   rep,
                          const SampleStats&        samp,
                          const Estimates&          est) const {
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  const auto& ev = *ev_or;

  const std::size_t n_free   = ev.n_free();
  const std::size_t n_blocks = samp.S.size();
  const Eigen::Index n_i     = static_cast<Eigen::Index>(n_free);
  if (n_free == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "model has no free parameters"));
  }

  // FD per-block Hessian, accumulated with each block's n_b/2 weighting:
  //   info[a,b] = Σ_blocks (n_b/2) · ∂²F_b/∂θ_a ∂θ_b
  //
  // For each free param k, evaluate ∇F_b(θ̂ ± h e_k) per block (via
  // `ML::gradient_block`) and central-difference into H_b's k-th column.
  ML ml;
  auto block_grad_at = [&](const Eigen::VectorXd& theta, std::size_t b)
      -> post_expected<Eigen::VectorXd> {
    auto sm = ev.sigma(theta);
    if (!sm.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ev.sigma(θ±h) failed: " + sm.error().detail));
    }
    auto J = ev.dsigma_dtheta(theta);
    if (!J.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ev.dsigma_dtheta(θ±h) failed: " + J.error().detail));
    }
    // dμ/dθ — empty for covariance-only models, so the mean-term branch
    // of `gradient_block` is skipped automatically.
    auto Jmu = ev.dmu_dtheta(theta);
    if (!Jmu.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ev.dmu_dtheta(θ±h) failed: " + Jmu.error().detail));
    }
    auto g = ml.gradient_block(samp, *sm, *J, b, *Jmu);
    if (!g.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ML::gradient_block failed: " + g.error().detail));
    }
    return *g;
  };

  Eigen::MatrixXd info = Eigen::MatrixXd::Zero(n_i, n_i);
  for (std::size_t blk = 0; blk < n_blocks; ++blk) {
    Eigen::MatrixXd H_blk = Eigen::MatrixXd::Zero(n_i, n_i);
    for (std::size_t k = 0; k < n_free; ++k) {
      Eigen::VectorXd tp = est.theta;
      Eigen::VectorXd tm = est.theta;
      tp(static_cast<Eigen::Index>(k)) += h_step;
      tm(static_cast<Eigen::Index>(k)) -= h_step;
      auto gp = block_grad_at(tp, blk);
      if (!gp.has_value()) return std::unexpected(gp.error());
      auto gm = block_grad_at(tm, blk);
      if (!gm.has_value()) return std::unexpected(gm.error());
      H_blk.col(static_cast<Eigen::Index>(k)) = (*gp - *gm) / (2.0 * h_step);
    }
    H_blk = (0.5 * (H_blk + H_blk.transpose())).eval();
    info  += (static_cast<double>(samp.n_obs[blk]) / 2.0) * H_blk;
  }

  const auto [chi2, df] = compute_chi2_df(pt, samp, est);
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  return finalize_inference(std::move(info), chi2, df, *con_or,
                            heywood_warnings(ev, rep, est));
}

// ----------------------------------------------------------------------------
// AnalyticObservedInfoSE — closed-form Hessian.
//
//   info[a,b] = Σ_blocks (n_b/2) · (H1_b[a,b] + H2_b[a,b])
//   H1_b[a,b] = -tr(W_b M_b^a W_b M_b^b) + 2 tr(W_b M_b^b · W_b S_b W_b · M_b^a)
//   H2_b[a,b] = tr(G_b · ∂²Σ_b / ∂θ_a ∂θ_b),   G_b = W_b − W_b S_b W_b
//
// Per-block accumulation: each free param k lives in exactly one block
// (v0 — no cross-group label equality yet), so M_k restricted to block b
// is zero for params not in b. H1 contributions naturally zero out;
// H2_b contributes only when both params live in block b.
// ----------------------------------------------------------------------------

namespace {

// Element-wise trace: tr(A · B) = Σ_{i,j} A(i,j) · B(j,i)
//                              = (A.array() * B.transpose().array()).sum().
inline double trace_product(const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B) noexcept {
  return (A.array() * B.transpose().array()).sum();
}

// Closed-form H2 entries for the general LISREL parameterization
// Σ = Λ A Ψ Aᵀ Λᵀ + Θ, A = (I − B)⁻¹. The (Λ, ·) cases come from
// cyclic-trace simplification of tr(G · ∂²Σ) where ∂Σ/∂Λ contains Λ (not
// LamA), so the precompute that emerges is GLam = G·Λ rather than G·LamA.
//
// Notation in the formulas below:
//   r_x, c_x   : (row, col) of free param x's matrix entry
//   M[r, c]    : entry (r, c) of matrix M
//   GLamA  = G · LamA  = G · Λ · A   (p × m)
//   GLamMid = G · Λ · Mid             (p × m)
//   K      = Λᵀ G Λ                   (m × m, sym)
//   P      = LamAᵀ G LamA = Aᵀ K A    (m × m, sym)
//   MKA    = Mid · K · A              (m × m)
//
// For Pure CFA (A = I) these all collapse: GLamA → G·Λ, GLamMid → G·Λ·Ψ,
// Mid → Ψ, P → K, MKA → Ψ·K. The Reduced formulas reduce smoothly so a
// single code path handles both forms.
//
// All H2 formulas are symmetric in (a, b) by construction (verified via
// trace-cyclic symmetry on the derivation).

inline double h2_lambda_lambda(const model::ParamLocation& la,
                               const model::ParamLocation& lb,
                               const Eigen::MatrixXd& Mid,
                               const Eigen::MatrixXd& G) noexcept {
  return 2.0 * Mid(la.col, lb.col) * G(la.row, lb.row);
}

inline double h2_lambda_psi(const model::ParamLocation& lam,
                            const model::ParamLocation& psi,
                            const Eigen::MatrixXd& GLamA,
                            const Eigen::MatrixXd& A) noexcept {
  if (psi.row == psi.col) {
    return 2.0 * GLamA(lam.row, psi.row) * A(lam.col, psi.row);
  }
  return 2.0 * (GLamA(lam.row, psi.row) * A(lam.col, psi.col) +
                GLamA(lam.row, psi.col) * A(lam.col, psi.row));
}

inline double h2_lambda_beta(const model::ParamLocation& lam,
                             const model::ParamLocation& beta,
                             const Eigen::MatrixXd& GLamA,
                             const Eigen::MatrixXd& GLamMid,
                             const Eigen::MatrixXd& Mid,
                             const Eigen::MatrixXd& A) noexcept {
  return 2.0 * (GLamA(lam.row, beta.row)    * Mid(beta.col, lam.col) +
                GLamMid(lam.row, beta.col)  * A(lam.col, beta.row));
}

inline double h2_psi_beta(const model::ParamLocation& psi,
                          const model::ParamLocation& beta,
                          const Eigen::MatrixXd& P,
                          const Eigen::MatrixXd& A) noexcept {
  if (psi.row == psi.col) {
    return 2.0 * P(psi.row, beta.row) * A(beta.col, psi.row);
  }
  return 2.0 * (P(psi.col, beta.row) * A(beta.col, psi.row) +
                P(psi.row, beta.row) * A(beta.col, psi.col));
}

inline double h2_beta_beta(const model::ParamLocation& ba,
                           const model::ParamLocation& bb,
                           const Eigen::MatrixXd& A,
                           const Eigen::MatrixXd& MKA,
                           const Eigen::MatrixXd& P,
                           const Eigen::MatrixXd& Mid) noexcept {
  return 2.0 * (A(bb.col, ba.row) * MKA(ba.col, bb.row) +
                A(ba.col, bb.row) * MKA(bb.col, ba.row) +
                Mid(ba.col, bb.col) * P(ba.row, bb.row));
}

}  // namespace

post_expected<Inference>
AnalyticObservedInfoSE::compute(partable::LatentStructure        pt,
                                const model::MatrixRep&   rep,
                                const SampleStats&        samp,
                                const Estimates&          est) const {
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  const auto& ev = *ev_or;

  const std::size_t n_free   = ev.n_free();
  const std::size_t n_blocks = samp.S.size();
  const Eigen::Index n_i     = static_cast<Eigen::Index>(n_free);
  if (n_free == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "model has no free parameters"));
  }
  // Mean-structure analytic observed info isn't wired up — the
  // closed-form ∂²μ/∂θ² cases (Λ×α, Λ×B, α×B, B×B) plus Part B's
  // η/∂²μ cross-terms would need to be added. FdObservedInfoSE handles
  // it correctly; users wanting analytic on a mean-structure model
  // should switch to FD until this lands.
  if (auto Jmu_or = ev.dmu_dtheta(est.theta);
      Jmu_or.has_value() && Jmu_or->size() > 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "AnalyticObservedInfoSE: mean structure (~1) not yet implemented; "
        "use FdObservedInfoSE instead"));
  }

  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.sigma(θ̂) failed: " + sm_or.error().detail));
  }
  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.dsigma_dtheta(θ̂) failed: " + J_or.error().detail));
  }
  auto am_or = ev.assembled(est.theta);
  if (!am_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.assembled(θ̂) failed: " + am_or.error().detail));
  }
  const auto& sm   = *sm_or;
  const auto& J    = *J_or;
  const auto  locs = ev.param_locations();

  Eigen::MatrixXd info = Eigen::MatrixXd::Zero(n_i, n_i);
  Eigen::Index vech_off = 0;

  for (std::size_t blk = 0; blk < n_blocks; ++blk) {
    const auto& bm = am_or->blocks[blk];

    // Σ̂_blk symmetrized, W = Σ̂_blk⁻¹, Z = W S_blk W, G = W − Z.
    const Eigen::MatrixXd Sigma_blk = 0.5 * (sm.sigma[blk] + sm.sigma[blk].transpose());
    const Eigen::Index p = Sigma_blk.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_blk);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "implied Σ at θ̂ is not positive definite in block " +
              std::to_string(blk)));
    }
    const Eigen::MatrixXd W = llt.solve(Eigen::MatrixXd::Identity(p, p));
    const Eigen::MatrixXd& S = samp.S[blk];
    const Eigen::MatrixXd Z  = W * S * W;
    const Eigen::MatrixXd G  = W - Z;

    // M_k_blk = ∂Σ_blk/∂θ_k (un-vech of J's column k restricted to block blk).
    // For params not in block blk, the J segment is zero so M_k_blk, T_k_blk,
    // ZM_k_blk are all zero — H1 contributions vanish naturally.
    std::vector<Eigen::MatrixXd> M(n_free, Eigen::MatrixXd::Zero(p, p));
    std::vector<Eigen::MatrixXd> T(n_free);
    std::vector<Eigen::MatrixXd> ZM(n_free);
    for (std::size_t k = 0; k < n_free; ++k) {
      auto& Mk = M[k];
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          const double v = J(vech_off + vech_index(p, r, c),
                             static_cast<Eigen::Index>(k));
          Mk(r, c) = v;
          if (r != c) Mk(c, r) = v;
        }
      }
      T[k].noalias()  = W * Mk;
      ZM[k].noalias() = Z * Mk;
    }

    // Per-block H2 precomputes.
    const Eigen::MatrixXd GLam    = G * bm.Lambda;
    const Eigen::MatrixXd GLamA   = G * bm.LamA;
    const Eigen::MatrixXd GLamMid = GLam * bm.Mid;
    const Eigen::MatrixXd K       = bm.Lambda.transpose() * G * bm.Lambda;
    const Eigen::MatrixXd P       = bm.LamA.transpose()   * G * bm.LamA;
    const Eigen::MatrixXd MKA     = bm.Mid * K * bm.A;

    const double weight = static_cast<double>(samp.n_obs[blk]) / 2.0;
    const auto blk_i    = static_cast<std::int8_t>(blk);

    for (std::size_t a = 0; a < n_free; ++a) {
      for (std::size_t b = a; b < n_free; ++b) {
        const double h1 = -trace_product(T[b], T[a]) +
                          2.0 * trace_product(T[b], ZM[a]);

        // H2_blk[a, b]: case switch. Contributes only if BOTH params live
        // in block `blk` — otherwise ∂²Σ_blk/∂θ_a ∂θ_b = 0 mechanically
        // (each param affects exactly one block's Σ in v0).
        double h2 = 0.0;
        const auto& la = locs[a];
        const auto& lb = locs[b];
        if (la.block == blk_i && lb.block == blk_i) {
          using model::MatId;
          const auto pair = std::pair{la.mat, lb.mat};
          switch (pair.first) {
            case MatId::Lambda:
              switch (pair.second) {
                case MatId::Lambda: h2 = h2_lambda_lambda(la, lb, bm.Mid, G); break;
                case MatId::Psi:    h2 = h2_lambda_psi(la, lb, GLamA, bm.A); break;
                case MatId::Beta:   h2 = h2_lambda_beta(la, lb, GLamA, GLamMid, bm.Mid, bm.A); break;
                case MatId::Theta:  break;
                case MatId::Nu:     break;
                case MatId::Alpha:  break;
              }
              break;
            case MatId::Psi:
              switch (pair.second) {
                case MatId::Lambda: h2 = h2_lambda_psi(lb, la, GLamA, bm.A); break;
                case MatId::Beta:   h2 = h2_psi_beta(la, lb, P, bm.A); break;
                case MatId::Psi:    break;
                case MatId::Theta:  break;
                case MatId::Nu:     break;
                case MatId::Alpha:  break;
              }
              break;
            case MatId::Beta:
              switch (pair.second) {
                case MatId::Lambda: h2 = h2_lambda_beta(lb, la, GLamA, GLamMid, bm.Mid, bm.A); break;
                case MatId::Psi:    h2 = h2_psi_beta(lb, la, P, bm.A); break;
                case MatId::Beta:   h2 = h2_beta_beta(la, lb, bm.A, MKA, P, bm.Mid); break;
                case MatId::Theta:  break;
                case MatId::Nu:     break;
                case MatId::Alpha:  break;
              }
              break;
            case MatId::Theta:
              break;
            case MatId::Nu:
            case MatId::Alpha:
              // Σ-side ∂²Σ/∂{ν,α}·∂(anything) = 0. Mean-term observed-info
              // contribution lives in a `dmu_dtheta`-based path that's
              // not wired here yet (mean-structure inference is phase 2).
              break;
          }
        }

        const double val = weight * (h1 + h2);
        info(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) += val;
        if (a != b)
          info(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) += val;
      }
    }

    vech_off += vech_len(p);
  }

  const auto [chi2, df] = compute_chi2_df(pt, samp, est);
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  return finalize_inference(std::move(info), chi2, df, *con_or,
                            heywood_warnings(ev, rep, est));
}

// ----------------------------------------------------------------------------
// Wald test for a linear restriction Rθ = q.
// ----------------------------------------------------------------------------

post_expected<WaldTestResult>
wald_test(const Eigen::MatrixXd& R, const Eigen::VectorXd& q,
          const Estimates& est, const Eigen::MatrixXd& vcov) {
  if (R.rows() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "wald_test: restriction matrix R has zero rows"));
  }
  if (R.cols() != est.theta.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "wald_test: R.cols()=" + std::to_string(R.cols()) +
            " ≠ theta.size()=" + std::to_string(est.theta.size())));
  }
  if (q.size() != R.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "wald_test: q.size()=" + std::to_string(q.size()) +
            " ≠ R.rows()=" + std::to_string(R.rows())));
  }
  if (vcov.rows() != est.theta.size() || vcov.cols() != est.theta.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "wald_test: vcov shape ≠ n_free"));
  }
  const Eigen::VectorXd diff   = R * est.theta - q;
  const Eigen::MatrixXd RVRT   = R * vcov * R.transpose();
  // Solve via LLT — R · vcov · Rᵀ is PD when R has full row rank and
  // vcov is PD. LLT failure means rank-deficient restrictions; we don't
  // fall back to LDLT (which would happily produce a pseudo-solution).
  Eigen::LLT<Eigen::MatrixXd> llt(RVRT);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "wald_test: R · vcov · Rᵀ is not positive definite (rank-deficient "
        "restriction or singular vcov)"));
  }
  const Eigen::VectorXd y = llt.solve(diff);
  WaldTestResult out;
  out.chi2 = diff.dot(y);
  out.df   = static_cast<int>(R.rows());
  return out;
}

// ----------------------------------------------------------------------------
// Upper-tail χ²(df) p-value via the regularized upper incomplete gamma
// function Q(a, x) = Γ(a, x) / Γ(a). Returns P(X² > chi2) where X² ~
// χ²(df). Equivalent to pchisq(chi2, df, lower.tail=FALSE) in R.
// ----------------------------------------------------------------------------

namespace {

// Series expansion for the lower-tail regularized P(a, x). Converges
// quickly when x < a + 1.
double gamma_p_series(double a, double x) noexcept {
  double sum = 1.0 / a;
  double term = sum;
  constexpr int max_iter = 200;
  for (int n = 1; n < max_iter; ++n) {
    term *= x / (a + static_cast<double>(n));
    sum  += term;
    if (std::abs(term) < std::abs(sum) * 1e-15) break;
  }
  // P(a, x) = sum · exp(−x + a·log x − lgamma(a)).
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Continued-fraction expansion for the upper-tail regularized Q(a, x).
// Converges quickly when x ≥ a + 1.
double gamma_q_cfrac(double a, double x) noexcept {
  constexpr double FPMIN = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / FPMIN;
  double d = 1.0 / b;
  double h = d;
  constexpr int max_iter = 200;
  for (int n = 1; n < max_iter; ++n) {
    const double an = -static_cast<double>(n) * (static_cast<double>(n) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (std::abs(c) < FPMIN) c = FPMIN;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < 1e-15) break;
  }
  // Q(a, x) = h · exp(−x + a·log x − lgamma(a)).
  return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

}  // namespace

post_expected<double>
rls_chi2(const SampleStats&            samp,
         const model::ImpliedMoments&  implied) {
  if (samp.S.size() != implied.sigma.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "rls_chi2: SampleStats and ImpliedMoments have different block counts"));
  }
  double total = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const auto& S     = samp.S[b];
    const auto& Sigma = implied.sigma[b];
    if (S.rows() != Sigma.rows() || S.cols() != Sigma.cols()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "rls_chi2: block " + std::to_string(b) +
              " S and Σ have different shapes"));
    }
    const Eigen::MatrixXd Sigma_sym =
        0.5 * (Sigma + Sigma.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_sym);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "rls_chi2: block " + std::to_string(b) +
              " implied Σ is not positive definite"));
    }
    // A = Σ⁻¹ · (S − Σ). tr(A²) gives the quadratic form vec(S−Σ)' (Σ⁻¹⊗Σ⁻¹) vec(S−Σ).
    const Eigen::MatrixXd diff = S - Sigma_sym;
    const Eigen::MatrixXd A    = llt.solve(diff);
    const double tr_A2 = (A * A).trace();
    const double F_b   = 0.5 * tr_A2;
    total += static_cast<double>(samp.n_obs[b]) * F_b;
  }
  return total;
}

post_expected<double>
browne_residual_nt(partable::LatentStructure        pt,
                   const model::MatrixRep&   rep,
                   const SampleStats&        samp,
                   const Estimates&          est) {
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto& ev = *ev_or;

  const std::size_t n_blocks = samp.S.size();
  double N_total = 0.0;
  for (std::size_t b = 0; b < n_blocks; ++b)
    N_total += static_cast<double>(samp.n_obs[b]);
  if (!(N_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: total n_obs is zero"));
  }

  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: sigma(theta) failed: " + im_or.error().detail));
  }

  // Mean structure: dmu_dtheta returns either a 0×0 matrix (cov-only) or
  // (Σ_b p_b × n_free, stacked per block, zero rows for blocks without
  // their own means). The implied μ̂_b sits in im_or->mu[b]; sample means
  // in samp.mean[b]. Detect presence and route accordingly.
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  const bool has_means = Jmu_or->size() > 0;

  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: dsigma_dtheta failed: " + J_or.error().detail));
  }
  const Eigen::MatrixXd Delta_sigma = std::move(*J_or);
  const Eigen::MatrixXd Delta_mu    = has_means ? std::move(*Jmu_or)
                                                : Eigen::MatrixXd();
  const Eigen::Index q              = Delta_sigma.cols();

  // Lay out stacked block offsets in the combined (μ; vech(Σ)) vector.
  struct BlockLayout {
    Eigen::Index p;
    Eigen::Index pstar;
    Eigen::Index mu_off;       // row offset in stacked vector (or -1 if no mean)
    Eigen::Index sigma_off;    // row offset in stacked vector
    double       weight;       // n_b / N_total — multiplied into Γ⁻¹ acting
                               // on this block, so STAT = N_total · (...)
                               // collapses to Σ_b n_b · (...) per block.
    Eigen::MatrixXd W;         // S_b⁻¹ — weight matrix for Γ_NT⁻¹
  };
  std::vector<BlockLayout> blk(n_blocks);

  Eigen::Index row_cursor = 0;
  Eigen::Index total_rows = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::Index p = samp.S[b].rows();
    if (p != samp.S[b].cols()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "browne_residual_nt: block " + std::to_string(b) +
              " S is not square"));
    }
    blk[b].p     = p;
    blk[b].pstar = p * (p + 1) / 2;
    blk[b].weight = static_cast<double>(samp.n_obs[b]) / N_total;
    if (im_or->sigma[b].rows() != p) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "browne_residual_nt: block " + std::to_string(b) +
              " implied Σ̂ shape mismatch with S"));
    }
    const Eigen::MatrixXd S_sym =
        0.5 * (samp.S[b] + samp.S[b].transpose());
    Eigen::LLT<Eigen::MatrixXd> llt_S(S_sym);
    if (llt_S.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "browne_residual_nt: block " + std::to_string(b) +
              " sample S is not positive definite"));
    }
    blk[b].W = llt_S.solve(Eigen::MatrixXd::Identity(p, p));
  }
  // Allocate offsets: per-block layout matches lavaan's [μ_b ; vech(Σ_b)]
  // ordering so the mean part sits on top inside each block.
  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (has_means) {
      blk[b].mu_off    = row_cursor;
      row_cursor      += blk[b].p;
    } else {
      blk[b].mu_off    = -1;
    }
    blk[b].sigma_off   = row_cursor;
    row_cursor        += blk[b].pstar;
  }
  total_rows = row_cursor;

  // Sanity: Delta_sigma rows match Σ_b pstar_b; Delta_mu rows match
  // Σ_b p_b (when has_means). The evaluator stacks per-block in the
  // same order as samp.S — `total_rows` for the combined layout matches
  // the sum.
  Eigen::Index expected_sigma_rows = 0;
  Eigen::Index expected_mu_rows    = 0;
  for (const auto& bl : blk) {
    expected_sigma_rows += bl.pstar;
    expected_mu_rows    += bl.p;
  }
  if (Delta_sigma.rows() != expected_sigma_rows) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: Δσ row count " +
            std::to_string(Delta_sigma.rows()) + " ≠ expected " +
            std::to_string(expected_sigma_rows)));
  }
  if (has_means && Delta_mu.rows() != expected_mu_rows) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: Δμ row count " +
            std::to_string(Delta_mu.rows()) + " ≠ expected " +
            std::to_string(expected_mu_rows)));
  }
  if (has_means && samp.mean.size() != n_blocks) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: mean structure requires per-block sample means"));
  }

  // Stack residual + Δ into the combined layout.
  Eigen::VectorXd res = Eigen::VectorXd::Zero(total_rows);
  Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(total_rows, q);
  {
    Eigen::Index s_cursor = 0;  // cursor into Delta_sigma rows
    Eigen::Index m_cursor = 0;  // cursor into Delta_mu rows
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p     = blk[b].p;
      const Eigen::Index pstar = blk[b].pstar;
      // μ residual + Δμ rows.
      if (has_means) {
        const auto& mbar  = samp.mean[b];
        const auto& mu_hat = im_or->mu[b];
        if (mbar.size() != p || mu_hat.size() != p) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "browne_residual_nt: block " + std::to_string(b) +
                  " mean vector shape mismatch"));
        }
        res.segment(blk[b].mu_off, p) = mbar - mu_hat;
        Delta.block(blk[b].mu_off, 0, p, q) =
            Delta_mu.block(m_cursor, 0, p, q);
        m_cursor += p;
      }
      // vech(S − Σ̂) residual + Δσ rows.
      const Eigen::MatrixXd& Sb = samp.S[b];
      const Eigen::MatrixXd& Sh = im_or->sigma[b];
      Eigen::Index k = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j; i < p; ++i) {
          res(blk[b].sigma_off + k) = Sb(i, j) - Sh(i, j);
          ++k;
        }
      }
      Delta.block(blk[b].sigma_off, 0, pstar, q) =
          Delta_sigma.block(s_cursor, 0, pstar, q);
      s_cursor += pstar;
    }
  }

  // Apply weighted Γ_NT⁻¹ to a stacked vector x: for each block, the μ-part
  // (if present) maps x_μ → (n_b/N_total)·W_b·x_μ, the σ-part vech(M) →
  // (n_b/N_total)·vech(W_b·M·W_b) with diagonal halved (the D⁺(W⊗W)D⁺ᵀ
  // structure). The weights collapse `N_total · (...)` to `Σ_b n_b · (...)`.
  auto apply_gamma_inv =
      [&](const Eigen::Ref<const Eigen::VectorXd>& x,
          Eigen::Ref<Eigen::VectorXd> out) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p     = blk[b].p;
      const auto&        W     = blk[b].W;
      const double       w     = blk[b].weight;
      if (has_means) {
        out.segment(blk[b].mu_off, p) =
            w * (W * x.segment(blk[b].mu_off, p));
      }
      // σ-part: vech_reverse → W·M·W → vech, halve diag, scale by w.
      Eigen::MatrixXd M(p, p);
      Eigen::Index k = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j; i < p; ++i) {
          M(i, j) = x(blk[b].sigma_off + k);
          if (i != j) M(j, i) = M(i, j);
          ++k;
        }
      }
      const Eigen::MatrixXd Z = W * M * W;
      k = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j; i < p; ++i) {
          const double z = Z(i, j);
          out(blk[b].sigma_off + k) = w * ((i == j) ? 0.5 * z : z);
          ++k;
        }
      }
    }
  };

  Eigen::VectorXd u(total_rows);
  apply_gamma_inv(res, u);

  Eigen::MatrixXd GinvDelta(total_rows, q);
  Eigen::VectorXd col_buf(total_rows);
  for (Eigen::Index c = 0; c < q; ++c) {
    apply_gamma_inv(Delta.col(c), col_buf);
    GinvDelta.col(c) = col_buf;
  }

  const Eigen::MatrixXd A     = Delta.transpose() * GinvDelta;
  const Eigen::VectorXd b_vec = Delta.transpose() * u;
  const Eigen::MatrixXd A_sym = 0.5 * (A + A.transpose());

  Eigen::LLT<Eigen::MatrixXd> llt_A(A_sym);
  Eigen::VectorXd Ainv_b;
  if (llt_A.info() == Eigen::Success) {
    Ainv_b = llt_A.solve(b_vec);
  } else {
    Eigen::LDLT<Eigen::MatrixXd> ldlt_A(A_sym);
    if (ldlt_A.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "browne_residual_nt: Δ'Γ⁻¹Δ is singular — model under-identified "
          "or Δ rank-deficient"));
    }
    Ainv_b = ldlt_A.solve(b_vec);
  }

  const double term1 = res.dot(u);
  const double term2 = b_vec.dot(Ainv_b);
  return N_total * (term1 - term2);
}

double chi2_pvalue(double chi2, int df) noexcept {
  if (df <= 0)      return std::numeric_limits<double>::quiet_NaN();
  if (chi2 < 0)     return std::numeric_limits<double>::quiet_NaN();
  if (chi2 == 0.0)  return 1.0;
  const double a = 0.5 * static_cast<double>(df);
  const double x = 0.5 * chi2;
  if (x < a + 1.0) {
    return 1.0 - gamma_p_series(a, x);
  }
  return gamma_q_cfrac(a, x);
}

}  // namespace latva::fit
