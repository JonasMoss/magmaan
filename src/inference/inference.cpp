#include "magmaan/inference/inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/robust/robust.hpp"   // robust::casewise_contributions

#include "detail_distribution_math.hpp"
#include "detail_second_order.hpp"
#include "detail_vech.hpp"

namespace magmaan::inference {

using estimate::build_eq_constraints;
using estimate::resolve_fixed_x_from_sample;

using data::RawData;
using data::SampleStats;
using estimate::Estimates;

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

using detail::vech_len;
using detail::vech_index;

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

// Resolve fixed.x fixed_values, build the evaluator, and validate shapes.
// `pt` is taken by mutable reference and gets its fixed.x `fixed_value`s
// filled — each information_* function owns its own copy of pt (the public
// API takes pt by value).
post_expected<model::ModelEvaluator>
prepare_evaluator(spec::LatentStructure&       pt,
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

PostError model_err(const ModelError& e, std::string_view who) {
  return make_err(PostError::Kind::NumericIssue,
                  std::string(who) + ": " + e.detail);
}

post_expected<Eigen::MatrixXd>
expected_info_covariance_only(const SampleStats& samp,
                              const model::ImpliedMoments& sm,
                              const Eigen::MatrixXd& J,
                              Eigen::Index n_free,
                              std::string_view who) {
  const std::size_t n_blocks = samp.S.size();
  if (sm.sigma.size() != n_blocks || samp.n_obs.size() != n_blocks) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) +
            ": SampleStats and implied moments have different block counts"));
  }

  std::vector<Eigen::MatrixXd> SigmaInv(n_blocks);
  std::vector<double>          weight(n_blocks, 0.0);
  std::vector<Eigen::Index>    p_dim(n_blocks, 0);
  std::vector<Eigen::Index>    vech_off(n_blocks, 0);

  Eigen::Index running = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (samp.S[b].rows() != sm.sigma[b].rows() ||
        samp.S[b].cols() != sm.sigma[b].cols()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(who) + ": block " + std::to_string(b) +
              " S and Σ have different shapes"));
    }
    const Eigen::MatrixXd Sigma_b =
        0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
    const Eigen::Index p = Sigma_b.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_b);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(who) + ": implied Σ for block " +
              std::to_string(b) + " is not positive definite at θ̂"));
    }
    SigmaInv[b] = llt.solve(Eigen::MatrixXd::Identity(p, p));
    weight[b]   = static_cast<double>(samp.n_obs[b]) / 2.0;
    p_dim[b]    = p;
    vech_off[b] = running;
    running += vech_len(p);
  }
  if (J.rows() != running || J.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) + ": Jacobian shape " +
            std::to_string(J.rows()) + "x" + std::to_string(J.cols()) +
            " does not match expected " + std::to_string(running) + "x" +
            std::to_string(n_free)));
  }

  std::vector<std::vector<Eigen::MatrixXd>> T(
      static_cast<std::size_t>(n_free), std::vector<Eigen::MatrixXd>(n_blocks));
  Eigen::MatrixXd M;
  for (Eigen::Index k = 0; k < n_free; ++k) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p = p_dim[b];
      M.setZero(p, p);
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          const double v = J(vech_off[b] + vech_index(p, r, c), k);
          M(r, c) = v;
          if (r != c) M(c, r) = v;
        }
      }
      T[static_cast<std::size_t>(k)][b].noalias() = SigmaInv[b] * M;
    }
  }

  Eigen::MatrixXd info = Eigen::MatrixXd::Zero(n_free, n_free);
  for (Eigen::Index a = 0; a < n_free; ++a) {
    for (Eigen::Index b = a; b < n_free; ++b) {
      double acc = 0.0;
      for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const auto& Ta = T[static_cast<std::size_t>(a)][blk];
        const auto& Tb = T[static_cast<std::size_t>(b)][blk];
        const double per_block =
            (Ta.transpose().array() * Tb.array()).sum();
        acc += weight[blk] * per_block;
      }
      info(a, b) = acc;
      if (a != b) info(b, a) = acc;
    }
  }
  return info;
}

}  // namespace

// ============================================================================
// vcov: invert info, apply constraint projection when there are equality
// constraints (shared labels, cross-group invariance, general linear `==`).
//
//   no constraints: vcov = info⁻¹
//   with K:         vcov = K · (Kᵀ I K)⁻¹ · Kᵀ
//
// Equality-tied params get equal SEs by construction.
// ============================================================================

post_expected<Eigen::MatrixXd>
vcov(const Eigen::MatrixXd&            info,
     const spec::LatentStructure&  pt,
     const Eigen::VectorXd&        theta) {
  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const auto& con = *con_or;

  if (pt.nl_constraints.empty()) {
    // Linear (or no) constraints — the K-reparameterization projection.
    if (!con.active()) {
      return invert_spd(info, "information matrix");
    }
    const Eigen::MatrixXd K          = con.K();                   // npar × n_alpha
    const Eigen::MatrixXd info_alpha = K.transpose() * info * K;   // n_alpha²
    auto vcov_alpha_or = invert_spd(info_alpha,
                                    "reduced (constrained) information matrix");
    if (!vcov_alpha_or.has_value()) return std::unexpected(vcov_alpha_or.error());
    return Eigen::MatrixXd(K * (*vcov_alpha_or) * K.transpose());  // npar × npar
  }

  // Nonlinear equality constraints: project onto the null space of the
  // stacked constraint Jacobian [A_eq ; H(θ̂)] — the exact nonlinear analog
  // of the linear K-projection above (the constrained estimate is an interior
  // point of the manifold {θ : h(θ) = 0}).
  if (theta.size() != info.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "vcov: a model with nonlinear equality constraints needs θ̂ — pass "
        "est.theta as the third argument"));
  }
  const auto nl = estimate::build_nl_constraints(pt);
  const Eigen::MatrixXd H    = nl.jacobian(theta);   // m × npar
  const Eigen::Index    npar = info.rows();
  Eigen::MatrixXd C(con.A_eq.rows() + H.rows(), npar);
  if (con.A_eq.rows() > 0) C.topRows(con.A_eq.rows()) = con.A_eq;
  C.bottomRows(H.rows()) = H;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);
  svd.setThreshold(1e-9);
  const Eigen::Index nz = npar - svd.rank();
  if (nz <= 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "vcov: the equality constraints leave no free dimensions"));
  }
  const Eigen::MatrixXd Z      = svd.matrixV().rightCols(nz);   // npar × nz
  const Eigen::MatrixXd info_z = Z.transpose() * info * Z;
  auto vcov_z_or = invert_spd(info_z,
                              "reduced (constrained) information matrix");
  if (!vcov_z_or.has_value()) return std::unexpected(vcov_z_or.error());
  return Eigen::MatrixXd(Z * (*vcov_z_or) * Z.transpose());      // npar × npar
}

Eigen::VectorXd se(const Eigen::MatrixXd& vcov) noexcept {
  Eigen::VectorXd out(vcov.rows());
  for (Eigen::Index k = 0; k < vcov.rows(); ++k) {
    const double d = vcov(k, k);
    out(k) = (d >= 0.0) ? std::sqrt(d)
                        : std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

// ============================================================================
// df_stat: pure function of (pt, samp).
//
//   df = Σ_b p_b(p_b+1)/2 (+ Σ_b p_b if mean structure)
//        − fixed_x_moments
//        − n_free
//        + constraint.rank
// ============================================================================

post_expected<int>
df_stat(const spec::LatentStructure& pt,
        const SampleStats&               samp,
        const Eigen::VectorXd&           theta) {
  int df_moments = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const int p = static_cast<int>(samp.S[b].rows());
    df_moments += p * (p + 1) / 2;
    // Mean structure: each block also contributes p mean moments when
    // the caller has supplied a sample mean for that block.
    if (b < samp.mean.size() && samp.mean[b].size() > 0) {
      df_moments += p;
    }
  }
  if (pt.composite_mode == spec::CompositeMode::FcSem) {
    for (const auto& c : pt.composite_blocks) {
      const int k = static_cast<int>(c.indicator_vars.size());
      df_moments -= static_cast<int>(samp.S.size()) * k * (k + 1) / 2;
    }
  }
  int fixed_x_moments = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if ((pt.op[i] == parse::Op::Covariance ||
         pt.op[i] == parse::Op::Intercept) &&
        pt.exo[i] == 1 && pt.free[i] == 0) {
      ++fixed_x_moments;
    }
  }
  // n_free already reflects the merged free-parameter count (max(pt.free[]));
  // the constraint.rank is added on top — it accounts for general-linear
  // equalities that further reduce dimensionality beyond the eq-group merge.
  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) return std::unexpected(con_or.error());

  // Each independent nonlinear `==` constraint adds one degree of freedom —
  // the rank of the constraint Jacobian H = ∂h/∂θ at θ̂.
  int nl_rank = 0;
  if (!pt.nl_constraints.empty()) {
    if (theta.size() != static_cast<Eigen::Index>(pt.n_free())) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "df_stat: a model with nonlinear equality constraints needs θ̂ — "
          "pass est.theta"));
    }
    const auto nl = estimate::build_nl_constraints(pt);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(nl.jacobian(theta));
    svd.setThreshold(1e-9);
    nl_rank = static_cast<int>(svd.rank());
  }
  return df_moments - fixed_x_moments
       - static_cast<int>(pt.n_free())
       + static_cast<int>(con_or->rank)
       + nl_rank;
}

// ============================================================================
// information_expected — closed-form expected Fisher information for ML.
//
//   I[a, b] = Σ_blocks (n_b/2) · [tr(Σ_b⁻¹ ∂Σ_b/∂θ_a Σ_b⁻¹ ∂Σ_b/∂θ_b)
//                                  + 2 · ν_a' Σ_b⁻¹ ν_b]
// ============================================================================

post_expected<Eigen::MatrixXd>
information_expected(spec::LatentStructure       pt,
                     const model::MatrixRep&         rep,
                     const SampleStats&              samp,
                     const Estimates&                est) {
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  const auto& ev = *ev_or;

  const std::size_t n_free   = ev.n_free();
  const std::size_t n_blocks = samp.S.size();

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

  // Per-block precompute: Σ_b⁻¹ and weight n_b/2. Σ⁻¹ as dense — small
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
  // ν_{k,b} is the block-b segment of Jmu's column k.
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
  // Mean-structure term: ν_a' η_b carries a factor of 2 inside the (n/2) scale.
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

  return info;
}

post_expected<Eigen::MatrixXd>
information_expected_fcsem(spec::LatentStructure       pt,
                           const SampleStats&          samp,
                           const Estimates&            est,
                           double                      rel_step) {
  auto ev_or = model::FcSemEvaluator::build(pt);
  if (!ev_or.has_value()) {
    return std::unexpected(model_err(ev_or.error(),
                                     "information_expected_fcsem build"));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_expected_fcsem: Estimates.theta size " +
            std::to_string(est.theta.size()) + " != evaluator n_free " +
            std::to_string(ev.n_free())));
  }
  if (samp.S.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_expected_fcsem: SampleStats and evaluator have different "
        "block counts"));
  }

  auto sm_or = ev.sigma(samp, est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(model_err(sm_or.error(),
                                     "information_expected_fcsem sigma"));
  }
  auto J_or = ev.dsigma_dtheta(samp, est.theta, rel_step);
  if (!J_or.has_value()) {
    return std::unexpected(model_err(J_or.error(),
                                     "information_expected_fcsem jacobian"));
  }
  return expected_info_covariance_only(
      samp, *sm_or, *J_or, static_cast<Eigen::Index>(ev.n_free()),
      "information_expected_fcsem");
}

// ============================================================================
// information_observed_fd — observed info via central-difference Hessian of
// the analytic ML gradient. Works for any model `fit()` can fit.
// ============================================================================

post_expected<Eigen::MatrixXd>
information_observed_fd(spec::LatentStructure       pt,
                        const model::MatrixRep&         rep,
                        const SampleStats&              samp,
                        const Estimates&                est,
                        double                          h_step) {
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
    auto Jmu = ev.dmu_dtheta(theta);
    if (!Jmu.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ev.dmu_dtheta(θ±h) failed: " + Jmu.error().detail));
    }
    auto g = estimate::ml_gradient_block(samp, *sm, *J, b, *Jmu);
    if (!g.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ml_gradient_block failed: " + g.error().detail));
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
  return info;
}

// ============================================================================
// information_observed_analytic — closed-form Hessian.
//
//   info[a,b] = Σ_blocks (n_b/2) · (H1_b[a,b] + H2_b[a,b])
//   H1_b[a,b] = -tr(W_b M_b^a W_b M_b^b) + 2 tr(W_b M_b^b · W_b S_b W_b · M_b^a)
//   H2_b[a,b] = tr(G_b · ∂²Σ_b / ∂θ_a ∂θ_b),   G_b = W_b − W_b S_b W_b
// ============================================================================

namespace {

// Element-wise trace: tr(A · B) = Σ_{i,j} A(i,j) · B(j,i)
//                              = (A.array() * B.transpose().array()).sum().
inline double trace_product(const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B) noexcept {
  return (A.array() * B.transpose().array()).sum();
}

}  // namespace

post_expected<Eigen::MatrixXd>
information_observed_analytic(spec::LatentStructure       pt,
                              const model::MatrixRep&         rep,
                              const SampleStats&              samp,
                              const Estimates&                est) {
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
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.dmu_dtheta(θ̂) failed: " + Jmu_or.error().detail));
  }
  const Eigen::MatrixXd& Jmu = *Jmu_or;
  const bool has_means = (Jmu.size() > 0);

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
  Eigen::Index mu_off = 0;

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
    if (has_means &&
        (blk >= samp.mean.size() || samp.mean[blk].size() != p ||
         blk >= sm.mu.size() || sm.mu[blk].size() != p)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "information_observed_analytic: sample/implied mean missing or "
          "wrong size in block " + std::to_string(blk)));
    }
    Eigen::VectorXd d;
    Eigen::VectorXd z;
    Eigen::MatrixXd S_eff = samp.S[blk];
    if (has_means) {
      d = samp.mean[blk] - sm.mu[blk];
      z = W * d;
      S_eff.noalias() += d * d.transpose();
    }
    const Eigen::MatrixXd Z  = W * S_eff * W;
    const Eigen::MatrixXd G  = W - Z;

    // M_k_blk = ∂Σ_blk/∂θ_k (un-vech of J's column k restricted to block blk).
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
    const auto sow = detail::SecondOrderWeights::build(G, bm, has_means);

    const double weight = static_cast<double>(samp.n_obs[blk]) / 2.0;
    const auto blk_i    = static_cast<std::int8_t>(blk);

    for (std::size_t a = 0; a < n_free; ++a) {
      for (std::size_t b = a; b < n_free; ++b) {
        const double h1 = -trace_product(T[b], T[a]) +
                          2.0 * trace_product(T[b], ZM[a]);

        double h2 = 0.0;
        const auto& la = locs[a];
        const auto& lb = locs[b];
        if (la.block == blk_i && lb.block == blk_i) {
          h2 = detail::second_sigma_trace(la, lb, sow, bm);
        }

        if (has_means && la.block == blk_i && lb.block == blk_i) {
          const Eigen::VectorXd mu_a =
              Jmu.col(static_cast<Eigen::Index>(a)).segment(mu_off, p);
          const Eigen::VectorXd mu_b =
              Jmu.col(static_cast<Eigen::Index>(b)).segment(mu_off, p);
          const Eigen::VectorXd W_mu_b = W * mu_b;
          const Eigen::VectorXd W_Mb_z = W * (M[b] * z);
          const Eigen::VectorXd mu_ab = detail::second_mu(la, lb, bm,
                                                          sow.A_alpha);
          h2 += 2.0 * mu_a.dot(W_mu_b)
              + 2.0 * mu_b.dot(W * (M[a] * z))
              + 2.0 * mu_a.dot(W_Mb_z)
              - 2.0 * mu_ab.dot(z);
        }

        const double val = weight * (h1 + h2);
        info(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) += val;
        if (a != b)
          info(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) += val;
      }
    }

    vech_off += vech_len(p);
    if (has_means) mu_off += p;
  }

  return info;
}

// ============================================================================
// casewise_scores — the (N × n_free) per-case ML score matrix, row i = s_iᵀ.
//
// Built from the casewise score factorisation
//   s_i = (W_b · Δ_b)ᵀ · z_i,b ,  z_i,b = [x_i − m̄_b ; vech((x_i − m̄_b)(...)ᵀ) − vech(S_b)]
// stacked block-diagonal across groups, returning the stacked Z_c · WΔ. Its
// Gram matrix is the parameter-level OPG `information_cross_products`
//   I_XP = Σ_i s_i s_iᵀ = (Z_c · WΔ)ᵀ · (Z_c · WΔ)
// with per-block (n_b/N) weighting absorbed into the block-diagonal stacking
// (each row of Z_c hits only its block's WΔ slice, so Σ_b n_b · ... falls
// out naturally — total N-scaled, matching `information_expected`'s
// convention so the same `vcov(info, pt)` path works downstream).
// ============================================================================

post_expected<Eigen::MatrixXd>
casewise_scores(spec::LatentStructure       pt,
                const model::MatrixRep&     rep,
                const SampleStats&          samp,
                const RawData&              raw,
                const Estimates&            est) {
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  const auto& ev = *ev_or;

  const auto n_blocks = samp.S.size();

  // Σ̂, Δσ, Δμ at θ̂.
  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_cross_products: sigma(θ̂) failed: " + sm_or.error().detail));
  }
  const auto& sm = *sm_or;

  auto Jsig_or = ev.dsigma_dtheta(est.theta);
  if (!Jsig_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_cross_products: dsigma_dtheta failed: " +
            Jsig_or.error().detail));
  }
  const Eigen::MatrixXd& J_sigma = *Jsig_or;

  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_cross_products: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  const Eigen::MatrixXd& J_mu = *Jmu_or;
  const bool has_means = (J_mu.size() > 0);

  const Eigen::Index n_free = static_cast<Eigen::Index>(ev.n_free());

  // Per-block geometry: μ on top of σ when has_means; σ-only otherwise.
  // Mirrors `robust::casewise_contributions` / `robust_setup`.
  struct BlockGeom {
    Eigen::Index p          = 0;
    Eigen::Index pstar      = 0;
    Eigen::Index mu_off     = -1;     // start row in the stacked layout (-1 ⇒ no means)
    Eigen::Index row_offset = 0;      // σ-segment start row
    Eigen::LLT<Eigen::MatrixXd> llt_gamma_nt;   // Cholesky of Γ_NT(Σ̂_b) (p* × p*)
    Eigen::LLT<Eigen::MatrixXd> llt_sigma;      // Cholesky of Σ̂_b (used for the μ-block of W)
  };
  std::vector<BlockGeom> blocks(n_blocks);

  Eigen::Index total_rows = 0;
  Eigen::Index sigma_vech_total = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::MatrixXd Sigma_b =
        0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
    const Eigen::Index p = Sigma_b.rows();
    auto& blk = blocks[b];
    blk.p          = p;
    blk.pstar      = vech_len(p);
    blk.mu_off     = has_means ? total_rows : Eigen::Index{-1};
    if (has_means) total_rows += p;
    blk.row_offset = total_rows;
    total_rows    += blk.pstar;
    sigma_vech_total += blk.pstar;

    auto G_or = data::gamma_nt(Sigma_b);
    if (!G_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "information_cross_products: gamma_nt failed for block " +
              std::to_string(b) + ": " + G_or.error().detail));
    }
    blk.llt_gamma_nt = Eigen::LLT<Eigen::MatrixXd>(*G_or);
    if (blk.llt_gamma_nt.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "information_cross_products: Γ_NT(Σ̂) is not positive definite in "
          "block " + std::to_string(b)));
    }
    if (has_means) {
      blk.llt_sigma = Eigen::LLT<Eigen::MatrixXd>(Sigma_b);
      if (blk.llt_sigma.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
            "information_cross_products: Σ̂ is not positive definite in "
            "block " + std::to_string(b)));
      }
    }
  }
  if (J_sigma.rows() != sigma_vech_total) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_cross_products: Δσ row count " +
            std::to_string(J_sigma.rows()) + " ≠ Σ_b p_b* = " +
            std::to_string(sigma_vech_total)));
  }
  if (has_means) {
    Eigen::Index p_total = 0;
    for (const auto& blk : blocks) p_total += blk.p;
    if (J_mu.rows() != p_total) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "information_cross_products: Δμ row count " +
              std::to_string(J_mu.rows()) + " ≠ Σ_b p_b = " +
              std::to_string(p_total)));
    }
    if (J_mu.cols() != n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "information_cross_products: Δμ cols " +
              std::to_string(J_mu.cols()) + " ≠ Δσ cols " +
              std::to_string(n_free)));
    }
  }

  // Stack Δ_full = [Δμ_b; Δσ_b] per block in the global (μ on top of σ
  // within each block) layout — matches casewise_contributions.
  Eigen::MatrixXd Delta_full = Eigen::MatrixXd::Zero(total_rows, n_free);
  {
    Eigen::Index s_cursor = 0;
    Eigen::Index m_cursor = 0;
    for (const auto& blk : blocks) {
      if (has_means) {
        Delta_full.block(blk.mu_off, 0, blk.p, n_free) =
            J_mu.block(m_cursor, 0, blk.p, n_free);
        m_cursor += blk.p;
      }
      Delta_full.block(blk.row_offset, 0, blk.pstar, n_free) =
          J_sigma.block(s_cursor, 0, blk.pstar, n_free);
      s_cursor += blk.pstar;
    }
  }

  // WΔ per block: W = Γ_NT(Σ̂_b)⁻¹ on the σ-segment, Σ̂_b⁻¹ on the μ-segment
  // (the μ-block of Γ_NT in the [μ; σ] stacked-moment convention is just Σ̂_b).
  // Two triangular solves per block: A = L⁻¹·Δ, then WΔ = L⁻ᵀ·A.
  Eigen::MatrixXd WDelta = Eigen::MatrixXd::Zero(total_rows, n_free);
  for (const auto& blk : blocks) {
    if (has_means) {
      const auto Dmu_b = Delta_full.middleRows(blk.mu_off, blk.p);
      const Eigen::MatrixXd A_mu = blk.llt_sigma.matrixL().solve(Dmu_b);
      WDelta.middleRows(blk.mu_off, blk.p) =
          blk.llt_sigma.matrixU().solve(A_mu);
    }
    const auto Dsig_b = Delta_full.middleRows(blk.row_offset, blk.pstar);
    const Eigen::MatrixXd A_sig = blk.llt_gamma_nt.matrixL().solve(Dsig_b);
    WDelta.middleRows(blk.row_offset, blk.pstar) =
        blk.llt_gamma_nt.matrixU().solve(A_sig);
  }

  // Z_c — block-stacked casewise contributions (μ-cols then σ-vech cols per
  // block when has_means; σ-vech-only otherwise). Same layout as Delta_full
  // along the moment axis, so Z_c · WΔ gives the per-case score matrix.
  auto Zc_or = robust::casewise_contributions(raw, samp, has_means);
  if (!Zc_or.has_value()) return std::unexpected(Zc_or.error());
  const Eigen::MatrixXd& Zc = *Zc_or;
  if (Zc.cols() != total_rows) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "information_cross_products: Z_c has " + std::to_string(Zc.cols()) +
            " columns, expected " + std::to_string(total_rows)));
  }

  // Per-case scores: row i = s_iᵀ = (Z_c · WΔ)_i. N-scaled (no /N), matching
  // `information_expected` so the same `vcov(info, pt)` path works downstream.
  return Eigen::MatrixXd(Zc * WDelta);                           // N × n_free
}

// information_cross_products — parameter-level OPG I_XP = Σ_i s_i s_iᵀ, the
// Gram matrix of the per-case scores. Mplus calls the SE built from it "MLF".
post_expected<Eigen::MatrixXd>
information_cross_products(spec::LatentStructure       pt,
                           const model::MatrixRep&     rep,
                           const SampleStats&          samp,
                           const RawData&              raw,
                           const Estimates&            est) {
  auto s_or = casewise_scores(std::move(pt), rep, samp, raw, est);
  if (!s_or.has_value()) return std::unexpected(s_or.error());
  const Eigen::MatrixXd& S = *s_or;
  Eigen::MatrixXd info = S.transpose() * S;                      // n_free × n_free
  info = 0.5 * (info + info.transpose()).eval();
  return info;
}

// ============================================================================
// Wald test for a linear restriction Rθ = q.
// ============================================================================

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

// ============================================================================
// Upper-tail χ²(df) p-value via regularized upper incomplete gamma.
// The incomplete-gamma kernels live in detail_distribution_math.hpp (shared with
// the simulation Pearson path and the FMG p-values); callers below guard inputs
// before dispatching, so the kernels' nan-on-bad-input guards are never reached.
// ============================================================================

using detail::gamma_p_series;
using detail::gamma_q_cfrac;

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
    const Eigen::MatrixXd diff = S - Sigma_sym;
    const Eigen::MatrixXd A    = llt.solve(diff);
    const double tr_A2 = (A * A).trace();
    const double F_b   = 0.5 * tr_A2;
    total += static_cast<double>(samp.n_obs[b]) * F_b;
  }
  return total;
}

post_expected<double>
rls_chi2(spec::LatentStructure       pt,
         const model::MatrixRep&     rep,
         const SampleStats&          samp,
         const Eigen::VectorXd&      theta) {
  Estimates est;
  est.theta = theta;
  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto im_or = ev_or->sigma(theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "rls_chi2: sigma(theta) failed: " + im_or.error().detail));
  }
  return rls_chi2(samp, *im_or);
}

post_expected<double>
browne_residual_nt(spec::LatentStructure        pt,
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

  struct BlockLayout {
    Eigen::Index p;
    Eigen::Index pstar;
    Eigen::Index mu_off;
    Eigen::Index sigma_off;
    double       weight;
    Eigen::MatrixXd W;
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
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.exo[i] != 1 || pt.free[i] != 0) continue;
      const std::int32_t g = i < pt.group.size() ? pt.group[i] : 1;
      if (g <= 0 || static_cast<std::size_t>(g) != b + 1) continue;
      auto zero_var = [&](std::int32_t v) {
        if (v < 0 || v >= pt.n_vars) return;
        const auto pos = pt.ov_pos[static_cast<std::size_t>(v)];
        if (pos < 0 || pos >= p) return;
        blk[b].W.row(pos).setZero();
        blk[b].W.col(pos).setZero();
      };
      zero_var(pt.lhs_var[i]);
      zero_var(pt.rhs_var[i]);
    }
  }
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

  Eigen::VectorXd res = Eigen::VectorXd::Zero(total_rows);
  Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(total_rows, q);
  {
    Eigen::Index s_cursor = 0;
    Eigen::Index m_cursor = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p     = blk[b].p;
      const Eigen::Index pstar = blk[b].pstar;
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

  std::vector<char> include_row(static_cast<std::size_t>(total_rows), char{1});
  bool drops_fixed_x_rows = false;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.exo[i] != 1 || pt.free[i] != 0) continue;
    const std::int32_t g = i < pt.group.size() ? pt.group[i] : 1;
    if (g <= 0 || static_cast<std::size_t>(g) > n_blocks) continue;
    const std::size_t b = static_cast<std::size_t>(g - 1);
    auto ov_index = [&](std::int32_t v) -> Eigen::Index {
      if (v < 0 || v >= pt.n_vars) return -1;
      const auto pos = pt.ov_pos[static_cast<std::size_t>(v)];
      return pos >= 0 ? static_cast<Eigen::Index>(pos) : -1;
    };
    if (pt.op[i] == parse::Op::Intercept && has_means) {
      const Eigen::Index row = ov_index(pt.lhs_var[i]);
      if (row >= 0 && row < blk[b].p) {
        include_row[static_cast<std::size_t>(blk[b].mu_off + row)] = char{0};
        drops_fixed_x_rows = true;
      }
    } else if (pt.op[i] == parse::Op::Covariance) {
      Eigen::Index r = ov_index(pt.lhs_var[i]);
      Eigen::Index c = ov_index(pt.rhs_var[i]);
      if (r >= 0 && c >= 0 && r < blk[b].p && c < blk[b].p) {
        if (r < c) std::swap(r, c);
        const Eigen::Index row = blk[b].sigma_off + vech_index(blk[b].p, r, c);
        include_row[static_cast<std::size_t>(row)] = char{0};
        drops_fixed_x_rows = true;
      }
    }
  }

  if (drops_fixed_x_rows) {
    for (Eigen::Index row = 0; row < total_rows; ++row) {
      if (include_row[static_cast<std::size_t>(row)] != char{0}) continue;
      res(row) = 0.0;
      Delta.row(row).setZero();
    }
  }

  // Equality constraints reduce the fitted manifold: project Δ onto the
  // constrained directions (Δ → Δ·K) so the Browne residual is taken against
  // the n_alpha-dimensional model tangent space. Mirrors src/nt/robust.cpp;
  // for an unconstrained model K is absent and Δ is unchanged.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_nt: build_eq_constraints — " +
            con_or.error().detail));
  }
  if (con_or->active()) Delta = (Delta * con_or->Kmat).eval();
  const Eigen::Index q_red = Delta.cols();

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

  Eigen::MatrixXd GinvDelta(total_rows, q_red);
  Eigen::VectorXd col_buf(total_rows);
  for (Eigen::Index c = 0; c < q_red; ++c) {
    apply_gamma_inv(Delta.col(c), col_buf);
    GinvDelta.col(c) = col_buf;
  }

  const Eigen::MatrixXd A     = Delta.transpose() * GinvDelta;
  const Eigen::VectorXd b_vec = Delta.transpose() * u;
  const Eigen::MatrixXd A_sym = 0.5 * (A + A.transpose());

  // term2 = b'A⁻¹b projects the discrepancy onto the model tangent space.
  // A_sym = Δ'Γ⁻¹Δ is a Gram matrix in the Γ⁻¹ metric (symmetric PSD), and
  // b_vec = Δ'u lies in range(A_sym) by construction — Γ⁻¹ has full rank, so
  // range(A_sym) = range(Δ'). LLT is the fast path for the well-conditioned
  // identified case. A just-identified or fixed-x model can leave A_sym with
  // an eigenvalue at the rounding floor; -O3/-march FP reassociation then
  // tips it slightly indefinite and LLT rejects it. Fall back to a
  // self-adjoint eigendecomposition and a range-restricted (Moore-Penrose)
  // solve: identical to A⁻¹ at full rank, and well-defined when A_sym is
  // numerically singular since b_vec carries no weight in the null space.
  double term2 = 0.0;
  Eigen::LLT<Eigen::MatrixXd> llt_A(A_sym);
  if (llt_A.info() == Eigen::Success) {
    term2 = b_vec.dot(llt_A.solve(b_vec));
  } else {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_sym);
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "browne_residual_nt: Δ'Γ⁻¹Δ eigendecomposition failed"));
    }
    const Eigen::VectorXd& lambda = es.eigenvalues();
    const double tol = static_cast<double>(A_sym.rows()) *
                       std::numeric_limits<double>::epsilon() *
                       std::max(lambda.cwiseAbs().maxCoeff(), 1.0);
    const Eigen::VectorXd c = es.eigenvectors().transpose() * b_vec;
    for (Eigen::Index k = 0; k < lambda.size(); ++k) {
      if (lambda(k) > tol) term2 += (c(k) * c(k)) / lambda(k);
    }
  }

  const double term1 = res.dot(u);
  return N_total * (term1 - term2);
}

post_expected<double>
browne_residual_adf(spec::LatentStructure        pt,
                    const model::MatrixRep&   rep,
                    const SampleStats&        samp,
                    const RawData&            raw,
                    const Estimates&          est) {
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: missing-data RawData is not supported"));
  }
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: RawData and SampleStats block count mismatch"));
  }

  auto ev_or = prepare_evaluator(pt, rep, samp, est);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());
  auto& ev = *ev_or;

  const std::size_t n_blocks = samp.S.size();
  double N_total = 0.0;
  for (std::size_t b = 0; b < n_blocks; ++b)
    N_total += static_cast<double>(samp.n_obs[b]);
  if (!(N_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: total n_obs is zero"));
  }

  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: sigma(theta) failed: " + im_or.error().detail));
  }
  auto Jmu_or = ev.dmu_dtheta(est.theta);
  if (!Jmu_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: dmu_dtheta failed: " + Jmu_or.error().detail));
  }
  const bool has_means = Jmu_or->size() > 0;
  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: dsigma_dtheta failed: " + J_or.error().detail));
  }
  const Eigen::MatrixXd Delta_sigma = std::move(*J_or);
  const Eigen::MatrixXd Delta_mu = has_means ? std::move(*Jmu_or)
                                             : Eigen::MatrixXd();
  const Eigen::Index q = Delta_sigma.cols();

  struct BlockLayout {
    Eigen::Index p;
    Eigen::Index pstar;
    Eigen::Index mu_off;
    Eigen::Index sigma_off;
    Eigen::Index size;
    Eigen::MatrixXd W;
  };
  std::vector<BlockLayout> blk(n_blocks);

  Eigen::Index row_cursor = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::Index p = samp.S[b].rows();
    if (p <= 0 || samp.S[b].cols() != p || raw.X[b].cols() != p) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "browne_residual_adf: block " + std::to_string(b) +
              " has incompatible dimensions"));
    }
    if (raw.X[b].rows() != samp.n_obs[b] || raw.X[b].rows() < 2) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "browne_residual_adf: block " + std::to_string(b) +
              " raw row count does not match n_obs or is too small"));
    }
    if (has_means && (samp.mean.size() != n_blocks || samp.mean[b].size() != p)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "browne_residual_adf: mean structure requires per-block sample means"));
    }

    blk[b].p = p;
    blk[b].pstar = p * (p + 1) / 2;
    blk[b].mu_off = has_means ? row_cursor : Eigen::Index{-1};
    if (has_means) row_cursor += p;
    blk[b].sigma_off = row_cursor;
    row_cursor += blk[b].pstar;
    blk[b].size = (has_means ? p : 0) + blk[b].pstar;

    Eigen::VectorXd mbar;
    if (samp.mean.size() == n_blocks && samp.mean[b].size() == p) {
      mbar = samp.mean[b];
    } else {
      mbar = raw.X[b].colwise().mean().transpose();
    }

    Eigen::MatrixXd Z(raw.X[b].rows(), blk[b].size);
    Z.setZero();
    for (Eigen::Index r = 0; r < raw.X[b].rows(); ++r) {
      const Eigen::VectorXd z = raw.X[b].row(r).transpose() - mbar;
      Eigen::Index off = 0;
      if (has_means) {
        Z.block(r, 0, 1, p) = z.transpose();
        off = p;
      }
      Eigen::Index k = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j; i < p; ++i) {
          Z(r, off + k) = z(i) * z(j) - samp.S[b](i, j);
          ++k;
        }
      }
    }
    Eigen::MatrixXd Gamma =
        (Z.transpose() * Z) / static_cast<double>(raw.X[b].rows());
    Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();
    auto Winv_or = invert_spd(Gamma, "browne_residual_adf: empirical Gamma");
    if (!Winv_or.has_value()) return std::unexpected(Winv_or.error());
    blk[b].W = (static_cast<double>(samp.n_obs[b]) / N_total) * (*Winv_or);
  }
  const Eigen::Index total_rows = row_cursor;

  Eigen::Index expected_sigma_rows = 0;
  Eigen::Index expected_mu_rows = 0;
  for (const auto& bl : blk) {
    expected_sigma_rows += bl.pstar;
    expected_mu_rows += bl.p;
  }
  if (Delta_sigma.rows() != expected_sigma_rows ||
      (has_means && Delta_mu.rows() != expected_mu_rows)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "browne_residual_adf: derivative row count mismatch"));
  }

  Eigen::VectorXd res = Eigen::VectorXd::Zero(total_rows);
  Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(total_rows, q);
  Eigen::Index s_cursor = 0;
  Eigen::Index m_cursor = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::Index p = blk[b].p;
    if (has_means) {
      res.segment(blk[b].mu_off, p) = samp.mean[b] - im_or->mu[b];
      Delta.block(blk[b].mu_off, 0, p, q) =
          Delta_mu.block(m_cursor, 0, p, q);
      m_cursor += p;
    }
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j; i < p; ++i) {
        res(blk[b].sigma_off + k) = samp.S[b](i, j) - im_or->sigma[b](i, j);
        ++k;
      }
    }
    Delta.block(blk[b].sigma_off, 0, blk[b].pstar, q) =
        Delta_sigma.block(s_cursor, 0, blk[b].pstar, q);
    s_cursor += blk[b].pstar;
  }

  auto apply_weight =
      [&](const Eigen::Ref<const Eigen::VectorXd>& x,
          Eigen::Ref<Eigen::VectorXd> out) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index start = has_means ? blk[b].mu_off : blk[b].sigma_off;
      out.segment(start, blk[b].size) =
          blk[b].W * x.segment(start, blk[b].size);
    }
  };

  Eigen::VectorXd u(total_rows);
  apply_weight(res, u);
  Eigen::MatrixXd WDelta(total_rows, q);
  Eigen::VectorXd col_buf(total_rows);
  for (Eigen::Index c = 0; c < q; ++c) {
    apply_weight(Delta.col(c), col_buf);
    WDelta.col(c) = col_buf;
  }

  const Eigen::MatrixXd A = Delta.transpose() * WDelta;
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
          "browne_residual_adf: Δ'Γ⁻¹Δ is singular"));
    }
    Ainv_b = ldlt_A.solve(b_vec);
  }
  return N_total * (res.dot(u) - b_vec.dot(Ainv_b));
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

namespace {

double central_chisq_cdf_half(double a, double x) noexcept {
  if (x <= 0.0) return 0.0;
  if (x < a + 1.0) return gamma_p_series(a, x);
  return 1.0 - gamma_q_cfrac(a, x);
}

}  // namespace

double noncentral_chisq_cdf(double x, double df, double ncp) noexcept {
  if (!(df > 0.0) || ncp < 0.0 || !std::isfinite(x) || !std::isfinite(df) ||
      !std::isfinite(ncp)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (x <= 0.0) return 0.0;
  const double a0 = 0.5 * df;
  const double xh = 0.5 * x;
  if (ncp == 0.0) return std::clamp(central_chisq_cdf_half(a0, xh), 0.0, 1.0);

  const double lh     = 0.5 * ncp;
  const double log_lh = std::log(lh);
  const long   j_mode = static_cast<long>(lh);
  auto log_w = [&](long j) {
    return -lh + static_cast<double>(j) * log_lh -
           std::lgamma(static_cast<double>(j) + 1.0);
  };
  double sum = 0.0;
  for (long j = j_mode; ; ++j) {
    const double w = std::exp(log_w(j));
    sum += w * central_chisq_cdf_half(a0 + static_cast<double>(j), xh);
    if (j > j_mode && w < 1e-17) break;
    if (j - j_mode > 200000) break;
  }
  for (long j = j_mode - 1; j >= 0; --j) {
    const double w = std::exp(log_w(j));
    sum += w * central_chisq_cdf_half(a0 + static_cast<double>(j), xh);
    if (w < 1e-17) break;
  }
  return std::clamp(sum, 0.0, 1.0);
}

}  // namespace magmaan::inference
