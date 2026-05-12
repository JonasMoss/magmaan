#include "latva/fit/ml.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/model_evaluator.hpp"

namespace latva::fit {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

// vech length for a p × p symmetric matrix (matches model_evaluator.cpp).
inline Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}

inline Eigen::Index vech_index(Eigen::Index p, Eigen::Index r,
                               Eigen::Index c) noexcept {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

}  // namespace

fit_expected<double>
ML::value(const SampleStats& s, const model::ImpliedMoments& m) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
  // F_ML = Σ_b (n_b/N)·F_b — lavaan's `likelihood = "normal"` convention.
  // Single-group: n_1/N = 1, so unchanged. Multi-group: matches lavaan's
  // θ̂ and N·F_ML = Σ_b n_b·F_b automatically gives the right χ².
  std::int64_t N_total = 0;
  for (auto n : s.n_obs) N_total += n;
  if (N_total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats has non-positive total n_obs"));
  }
  double f = 0.0;
  for (std::size_t b = 0; b < s.S.size(); ++b) {
    const auto& S     = s.S[b];
    const auto& Sigma = m.sigma[b];
    if (S.rows() != Sigma.rows() || S.cols() != Sigma.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "block " + std::to_string(b) +
              ": S and Σ have different shapes"));
    }
    const Eigen::Index p = Sigma.rows();

    Eigen::LLT<Eigen::MatrixXd> llt_sigma(Sigma);
    if (llt_sigma.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "implied Σ for block " + std::to_string(b) +
              " is not positive definite"));
    }
    Eigen::LLT<Eigen::MatrixXd> llt_S(S);
    if (llt_S.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSample,
          "sample S for block " + std::to_string(b) +
              " is not positive definite"));
    }
    // log|Σ| from the Cholesky factor: 2 * sum(log(diag(L))).
    const auto& L_sigma = llt_sigma.matrixL();
    const auto& L_S     = llt_S.matrixL();
    double log_det_sigma = 0.0, log_det_S = 0.0;
    for (Eigen::Index i = 0; i < p; ++i) {
      log_det_sigma += std::log(L_sigma(i, i));
      log_det_S     += std::log(L_S(i, i));
    }
    log_det_sigma *= 2.0;
    log_det_S     *= 2.0;

    // tr(S Σ⁻¹) via Σ⁻¹ S; use the Cholesky to solve.
    const Eigen::MatrixXd SigmaInv_S = llt_sigma.solve(S);
    const double tr_term = SigmaInv_S.trace();

    double F_b = log_det_sigma + tr_term - log_det_S - static_cast<double>(p);
    // Mean-structure term: + (m̄_b − μ_b)' Σ_b⁻¹ (m̄_b − μ_b). Only when
    // both data (s.mean[b]) and model (m.mu[b]) supply a mean vector.
    if (b < s.mean.size() && b < m.mu.size() &&
        s.mean[b].size() > 0 && m.mu[b].size() > 0) {
      if (s.mean[b].size() != m.mu[b].size()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "block " + std::to_string(b) +
                ": sample mean and implied μ have different sizes"));
      }
      const Eigen::VectorXd d = s.mean[b] - m.mu[b];
      F_b += d.dot(llt_sigma.solve(d));
    }
    const double w_b = static_cast<double>(s.n_obs[b]) /
                       static_cast<double>(N_total);
    f += w_b * F_b;
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "F_ML evaluated to non-finite"));
  }
  return f;
}

namespace {

// Fill the vech-doubled weight vector segment for one block, scaled by
// `scale` (typically n_b/N for the weighted-sum gradient, or 1.0 for the
// un-weighted per-block gradient).
//   w[off..off+vech_len) ← scale · vech-doubled(G_b)
// G_b = Σ_b⁻¹ − Σ_b⁻¹ S_b Σ_b⁻¹, symmetrized numerically.
fit_expected<void>
fill_block_weight(const SampleStats& s, const model::ImpliedMoments& m,
                  std::size_t b, Eigen::Index off, double scale,
                  Eigen::VectorXd& w) {
  const auto& S     = s.S[b];
  const auto& Sigma = m.sigma[b];
  const Eigen::Index p = Sigma.rows();

  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "implied Σ for block " + std::to_string(b) +
            " is not positive definite (in gradient)"));
  }
  const Eigen::MatrixXd SigmaInv = llt.solve(Eigen::MatrixXd::Identity(p, p));
  const Eigen::MatrixXd SigmaInv_S      = llt.solve(S);
  const Eigen::MatrixXd SigmaInv_S_Inv  = llt.solve(SigmaInv_S.transpose()).transpose();
  Eigen::MatrixXd G = SigmaInv - SigmaInv_S_Inv;
  G = 0.5 * (G + G.transpose());

  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      const Eigen::Index idx = off + vech_index(p, r, c);
      w(idx) = scale * ((r == c) ? G(r, r) : 2.0 * G(r, c));
    }
  }
  return {};
}

}  // namespace

namespace {

// Mean-structure correction for one block's vech-doubled weight vector
// and the parallel "u" vector that gets multiplied by Jmu^T. Updates w
// in place (subtracts scale · vech-doubled(zz')) and writes u_segment.
//
//   G* = G − (Σ⁻¹ d)(Σ⁻¹ d)'   where d = m̄ − μ
//   correction to w-vech : − scale · vech-doubled(z z')
//   contribution to u    : − 2 · scale · z
fit_expected<void>
apply_mean_correction(const SampleStats& s, const model::ImpliedMoments& m,
                      std::size_t b, Eigen::Index w_off, double scale,
                      Eigen::VectorXd& w, Eigen::Index u_off,
                      Eigen::VectorXd& u) {
  if (b >= s.mean.size() || b >= m.mu.size()) return {};
  if (s.mean[b].size() == 0 || m.mu[b].size() == 0) return {};
  if (s.mean[b].size() != m.mu[b].size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "block " + std::to_string(b) +
            ": sample mean and implied μ have different sizes (gradient)"));
  }
  const auto& Sigma = m.sigma[b];
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "implied Σ for block " + std::to_string(b) +
            " is not positive definite (in gradient mean term)"));
  }
  const Eigen::VectorXd d = s.mean[b] - m.mu[b];
  const Eigen::VectorXd z = llt.solve(d);
  const Eigen::Index p = Sigma.rows();
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      const Eigen::Index idx = w_off + vech_index(p, r, c);
      w(idx) -= scale * ((r == c) ? z(r) * z(r) : 2.0 * z(r) * z(c));
    }
  }
  u.segment(u_off, p) = -2.0 * scale * z;
  return {};
}

}  // namespace

fit_expected<Eigen::VectorXd>
ML::gradient(const SampleStats& s, const model::ImpliedMoments& m,
             const Eigen::MatrixXd& J,
             const Eigen::MatrixXd& Jmu) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
  // ∂F_ML/∂θ_k = Σ_b (n_b/N) · ∂F_b/∂θ_k.
  // Covariance part : J^T · w_vech (vech-doubled G* with scale n_b/N).
  // Mean part       : Jmu^T · u   (u_b = −2 · (n_b/N) · Σ_b⁻¹ d_b).
  std::int64_t N_total = 0;
  for (auto n : s.n_obs) N_total += n;
  Eigen::Index total_vech = 0;
  for (const auto& Sigma : m.sigma) total_vech += vech_len(Sigma.rows());
  Eigen::VectorXd w(total_vech);
  w.setZero();

  const bool has_means = (Jmu.size() > 0);
  Eigen::Index total_p = 0;
  if (has_means) {
    for (const auto& Sigma : m.sigma) total_p += Sigma.rows();
  }
  Eigen::VectorXd u(has_means ? total_p : 0);
  u.setZero();

  Eigen::Index off    = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    const double scale = static_cast<double>(s.n_obs[b]) /
                         static_cast<double>(N_total);
    if (auto e = fill_block_weight(s, m, b, off, scale, w); !e.has_value()) {
      return std::unexpected(e.error());
    }
    if (has_means) {
      if (auto e = apply_mean_correction(s, m, b, off, scale, w, mu_off, u);
          !e.has_value()) {
        return std::unexpected(e.error());
      }
      mu_off += m.sigma[b].rows();
    }
    off += vech_len(m.sigma[b].rows());
  }
  if (J.rows() != total_vech) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "Jacobian row count " + std::to_string(J.rows()) +
            " ≠ total vech length " + std::to_string(total_vech)));
  }
  Eigen::VectorXd g = J.transpose() * w;
  if (has_means) {
    if (Jmu.rows() != total_p) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "Jmu row count " + std::to_string(Jmu.rows()) +
              " ≠ total p " + std::to_string(total_p)));
    }
    g.noalias() += Jmu.transpose() * u;
  }
  return g;
}

fit_expected<Eigen::VectorXd>
ML::gradient_block(const SampleStats& s, const model::ImpliedMoments& m,
                   const Eigen::MatrixXd& J, std::size_t block_idx,
                   const Eigen::MatrixXd& Jmu) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
  if (block_idx >= m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "block_idx " + std::to_string(block_idx) + " ≥ n_blocks " +
            std::to_string(m.sigma.size())));
  }
  // Un-weighted per-block gradient ∂F_b/∂θ — FdObservedInfoSE accumulates
  // (n_b/2) · FD(grad_block_b) which gives the right multi-group info.
  Eigen::Index total_vech = 0;
  Eigen::Index off_target = 0;
  Eigen::Index total_p    = 0;
  Eigen::Index mu_off_target = 0;
  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    if (b == block_idx) {
      off_target    = total_vech;
      mu_off_target = total_p;
    }
    total_vech += vech_len(m.sigma[b].rows());
    total_p    += m.sigma[b].rows();
  }
  Eigen::VectorXd w(total_vech);
  w.setZero();
  if (auto e = fill_block_weight(s, m, block_idx, off_target, 1.0, w);
      !e.has_value()) {
    return std::unexpected(e.error());
  }

  const bool has_means = (Jmu.size() > 0);
  Eigen::VectorXd u(has_means ? total_p : 0);
  u.setZero();
  if (has_means) {
    if (auto e = apply_mean_correction(s, m, block_idx, off_target, 1.0, w,
                                       mu_off_target, u);
        !e.has_value()) {
      return std::unexpected(e.error());
    }
  }

  if (J.rows() != total_vech) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "Jacobian row count " + std::to_string(J.rows()) +
            " ≠ total vech length " + std::to_string(total_vech)));
  }
  Eigen::VectorXd g = J.transpose() * w;
  if (has_means) {
    if (Jmu.rows() != total_p) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "Jmu row count " + std::to_string(Jmu.rows()) +
              " ≠ total p " + std::to_string(total_p)));
    }
    g.noalias() += Jmu.transpose() * u;
  }
  return g;
}

}  // namespace latva::fit
