#include "latva/fit/standardized.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/model/model_evaluator.hpp"

namespace latva::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

post_expected<StandardizedSolution>
standardize_lv(const partable::LatentStructure& pt,
               const model::MatrixRep&   rep,
               const Estimates&          est,
               const Eigen::MatrixXd&    vcov) {
  const Eigen::Index n_free = est.theta.size();
  if (vcov.rows() != n_free || vcov.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_lv: vcov shape doesn't match Estimates.theta size"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_lv: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  auto am_or = ev.assembled(est.theta);
  if (!am_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_lv: ev.assembled failed: " + am_or.error().detail));
  }
  const auto& am = *am_or;
  const auto locs = ev.param_locations();

  // Build a lookup: (block, j) → free index (1-based) of Ψ[j, j] if free,
  // else 0. Used to record the gradient contribution wrt ψ_jj when
  // standardizing a Λ loading that loads on factor j.
  // For multi-block: one lookup per block.
  std::vector<std::vector<std::int32_t>> psi_diag_free(am.blocks.size());
  for (std::size_t b = 0; b < am.blocks.size(); ++b) {
    psi_diag_free[b].assign(static_cast<std::size_t>(am.blocks[b].Psi.rows()), 0);
  }
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    if (L.mat == model::MatId::Psi && L.row == L.col) {
      psi_diag_free[static_cast<std::size_t>(L.block)]
                   [static_cast<std::size_t>(L.row)] =
          static_cast<std::int32_t>(k) + 1;
    }
  }

  // Build the Jacobian J of θ → θ_std and the values theta_std. Each row k
  // of J is the gradient of g_k(θ).
  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_free, n_free);
  Eigen::VectorXd theta_std = est.theta;

  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    const auto  b = static_cast<std::size_t>(L.block);
    const auto& bm = am.blocks[b];
    const Eigen::Index ki = static_cast<Eigen::Index>(k);
    switch (L.mat) {
      case model::MatId::Lambda: {
        // λ_std = λ · √ψ_cc where c = L.col is the factor index.
        const double psi_cc = bm.Psi(L.col, L.col);
        if (psi_cc <= 0.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "standardize_lv: ψ_cc for column " + std::to_string(L.col) +
                  " of block " + std::to_string(L.block) +
                  " is non-positive; cannot standardize"));
        }
        const double sqrt_psi = std::sqrt(psi_cc);
        const double lambda   = est.theta(ki);
        theta_std(ki) = lambda * sqrt_psi;
        // Reset row to zero (we built J from Identity; for non-identity
        // transforms we set the row explicitly).
        J.row(ki).setZero();
        J(ki, ki) = sqrt_psi;
        const std::int32_t psi_k = psi_diag_free[b][static_cast<std::size_t>(L.col)];
        if (psi_k > 0) {
          // ψ_cc is free; add ∂(λ√ψ)/∂ψ = λ / (2√ψ) to the J row.
          J(ki, psi_k - 1) = lambda / (2.0 * sqrt_psi);
        }
        break;
      }
      case model::MatId::Psi:
        if (L.row == L.col) {
          // ψ_jj → 1: constant transform.
          theta_std(ki) = 1.0;
          J.row(ki).setZero();
        }
        // Ψ off-diagonals (factor covariances) pass through as identity
        // for now — a full implementation would rescale them as
        // ψ_jk / √(ψ_jj·ψ_kk), but that's not yet wired up here.
        break;
      case model::MatId::Theta:
      case model::MatId::Beta:
      case model::MatId::Nu:
      case model::MatId::Alpha:
        // Identity transform — J already has 1 on the diagonal from
        // the initial Identity, value already correct in theta_std.
        break;
    }
  }

  StandardizedSolution out;
  out.theta = std::move(theta_std);
  // SE_std = √diag(J · vcov · Jᵀ). Compute J · vcov · Jᵀ explicitly for
  // small v0 sizes; for production-scale code we'd row-wise solve.
  const Eigen::MatrixXd JV  = J * vcov;
  out.se.resize(n_free);
  for (Eigen::Index k = 0; k < n_free; ++k) {
    const double var = JV.row(k).dot(J.row(k));
    out.se(k) = (var > 0.0) ? std::sqrt(var) : 0.0;
  }
  return out;
}

namespace {

// vech-index for column-major lower-tri symmetric storage — matches the
// convention used by `ModelEvaluator::dsigma_dtheta`.
inline Eigen::Index vech_index(Eigen::Index p, Eigen::Index r,
                               Eigen::Index c) noexcept {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}
inline Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}

}  // namespace

post_expected<StandardizedSolution>
standardize_all(const partable::LatentStructure& pt,
                const model::MatrixRep&   rep,
                const Estimates&          est,
                const Eigen::MatrixXd&    vcov) {
  const Eigen::Index n_free = est.theta.size();
  if (vcov.rows() != n_free || vcov.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: vcov shape doesn't match Estimates.theta size"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;

  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ev.sigma failed: " + sm_or.error().detail));
  }
  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ev.dsigma_dtheta failed: " + J_or.error().detail));
  }
  auto am_or = ev.assembled(est.theta);
  if (!am_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ev.assembled failed: " + am_or.error().detail));
  }
  const auto& sm = *sm_or;
  const auto& Js = *J_or;
  const auto& am = *am_or;
  const auto  locs = ev.param_locations();
  const std::size_t n_blocks = am.blocks.size();

  // Lookups: free-index of Ψ[j,j] per (block, j); per-block vech offset
  // into the Jacobian Js (rows = sum_b vech_len(p_b)).
  std::vector<std::vector<std::int32_t>> psi_diag_free(n_blocks);
  std::vector<Eigen::Index> vech_off(n_blocks, 0);
  Eigen::Index running = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    psi_diag_free[b].assign(static_cast<std::size_t>(am.blocks[b].Psi.rows()), 0);
    vech_off[b] = running;
    running += vech_len(static_cast<Eigen::Index>(am.blocks[b].Lambda.rows()));
  }
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    if (L.mat == model::MatId::Psi && L.row == L.col) {
      psi_diag_free[static_cast<std::size_t>(L.block)]
                   [static_cast<std::size_t>(L.row)] =
          static_cast<std::int32_t>(k) + 1;
    }
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_free, n_free);
  Eigen::VectorXd theta_std = est.theta;

  // Helper: add `−scalar · ∂σ_rr/∂θ_*` to J row `k_out`. ∂σ_rr/∂θ comes
  // from row `vech_off[b] + vech_index(p, r, r)` of Js.
  auto sub_dsigma_rr = [&](Eigen::Index k_out, std::size_t b, Eigen::Index r,
                            double scalar) {
    const Eigen::Index p = static_cast<Eigen::Index>(am.blocks[b].Lambda.rows());
    const Eigen::Index row = vech_off[b] + vech_index(p, r, r);
    J.row(k_out) -= scalar * Js.row(row);
  };

  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    const auto  b = static_cast<std::size_t>(L.block);
    const auto& bm = am.blocks[b];
    const Eigen::Index ki = static_cast<Eigen::Index>(k);
    switch (L.mat) {
      case model::MatId::Lambda: {
        const double psi_cc   = bm.Psi(L.col, L.col);
        const double sigma_rr = sm.sigma[b](L.row, L.row);
        if (psi_cc <= 0.0 || sigma_rr <= 0.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "standardize_all: non-positive ψ_cc or σ_rr"));
        }
        const double sp = std::sqrt(psi_cc);
        const double ss = std::sqrt(sigma_rr);
        const double lam = est.theta(ki);
        theta_std(ki) = lam * sp / ss;
        J.row(ki).setZero();
        J(ki, ki) = sp / ss;                          // direct λ
        if (auto pk = psi_diag_free[b][static_cast<std::size_t>(L.col)]; pk > 0) {
          J(ki, pk - 1) += lam / (2.0 * sp * ss);     // via ψ_cc
        }
        // via σ_rr: ∂(lam·sp/ss)/∂σ_rr = − lam·sp / (2·σ·ss); chain through dsigma.
        sub_dsigma_rr(ki, b, L.row, lam * sp / (2.0 * sigma_rr * ss));
        break;
      }
      case model::MatId::Theta:
        if (L.row == L.col) {
          const double sigma_rr = sm.sigma[b](L.row, L.row);
          if (sigma_rr <= 0.0) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "standardize_all: non-positive σ_rr for Θ_rr"));
          }
          const double th = est.theta(ki);
          theta_std(ki) = th / sigma_rr;
          J.row(ki).setZero();
          J(ki, ki) = 1.0 / sigma_rr;
          sub_dsigma_rr(ki, b, L.row, th / (sigma_rr * sigma_rr));
        }
        // Θ off-diagonals: identity (preserved from Identity init).
        break;
      case model::MatId::Psi:
        if (L.row == L.col) {
          theta_std(ki) = 1.0;
          J.row(ki).setZero();
        } else {
          // ψ_rc / √(ψ_rr · ψ_cc): same as std.lv off-diag.
          const double psi_rr = bm.Psi(L.row, L.row);
          const double psi_cc = bm.Psi(L.col, L.col);
          if (psi_rr <= 0.0 || psi_cc <= 0.0) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "standardize_all: non-positive ψ_rr or ψ_cc for off-diag Ψ"));
          }
          const double psi_rc = est.theta(ki);
          const double prod   = std::sqrt(psi_rr * psi_cc);
          theta_std(ki) = psi_rc / prod;
          J.row(ki).setZero();
          J(ki, ki) = 1.0 / prod;
          if (auto pk = psi_diag_free[b][static_cast<std::size_t>(L.row)]; pk > 0) {
            J(ki, pk - 1) -= psi_rc / (2.0 * psi_rr * prod);
          }
          if (auto pk = psi_diag_free[b][static_cast<std::size_t>(L.col)]; pk > 0) {
            J(ki, pk - 1) -= psi_rc / (2.0 * psi_cc * prod);
          }
        }
        break;
      case model::MatId::Nu: {
        const double sigma_rr = sm.sigma[b](L.row, L.row);
        if (sigma_rr <= 0.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "standardize_all: non-positive σ_rr for ν_r"));
        }
        const double ss = std::sqrt(sigma_rr);
        const double nu = est.theta(ki);
        theta_std(ki) = nu / ss;
        J.row(ki).setZero();
        J(ki, ki) = 1.0 / ss;
        sub_dsigma_rr(ki, b, L.row, nu / (2.0 * sigma_rr * ss));
        break;
      }
      case model::MatId::Alpha: {
        const double psi_jj = bm.Psi(L.row, L.row);
        if (psi_jj <= 0.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "standardize_all: non-positive ψ_jj for α_j"));
        }
        const double sp = std::sqrt(psi_jj);
        const double a  = est.theta(ki);
        theta_std(ki) = a / sp;
        J.row(ki).setZero();
        J(ki, ki) = 1.0 / sp;
        if (auto pk = psi_diag_free[b][static_cast<std::size_t>(L.row)]; pk > 0) {
          J(ki, pk - 1) -= a / (2.0 * psi_jj * sp);
        }
        break;
      }
      case model::MatId::Beta:
        // Identity transform for v0 (Reduced-form β rescaling pending).
        break;
    }
  }

  StandardizedSolution out;
  out.theta = std::move(theta_std);
  const Eigen::MatrixXd JV = J * vcov;
  out.se.resize(n_free);
  for (Eigen::Index k = 0; k < n_free; ++k) {
    const double var = JV.row(k).dot(J.row(k));
    out.se(k) = (var > 0.0) ? std::sqrt(var) : 0.0;
  }
  return out;
}

}  // namespace latva::fit
