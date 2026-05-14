#include "magmaan/fit/uls.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"

namespace magmaan::fit {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

using detail::vech_len;
using detail::vech_lower;

// Inspect (s, m) to decide whether the LS layout includes mean rows. Same
// rule as `value` / `gradient`: a block participates in the mean term only
// when *both* `s.mean[b]` and `m.mu[b]` are populated. The layout-wide
// `has_means` flag is on iff any block populates a mean — mixing means and
// no-means across blocks isn't supported (mirrors the existing scalar path,
// which silently zeros u_b for unpopulated blocks). For the residual layout
// we use the same rule: the per-block mean rows are zeroed out for blocks
// that don't have a sample mean, so block-row offsets stay deterministic.
bool ls_has_means(const SampleStats& s, const model::ImpliedMoments& m) {
  for (std::size_t b = 0; b < m.mu.size(); ++b) {
    if (m.mu[b].size() > 0 && b < s.mean.size() && s.mean[b].size() > 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

fit_expected<double>
ULS::value(const SampleStats& s, const model::ImpliedMoments& m) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
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
    // Covariance part: vech residual; each off-diagonal counted once. The
    // explicit MatrixXd cast forces evaluation of the expression template
    // before vech_lower takes its Ref<const MatrixXd>.
    const Eigen::VectorXd r_sigma =
        vech_lower(Eigen::MatrixXd(S - Sigma));
    double F_b = 0.5 * r_sigma.squaredNorm();

    // Mean-structure part: (m̄ − μ)ᵀ(m̄ − μ)/2. Same gating rule as ML —
    // both data and model must supply a mean for the block to participate.
    if (b < s.mean.size() && b < m.mu.size() &&
        s.mean[b].size() > 0 && m.mu[b].size() > 0) {
      if (s.mean[b].size() != m.mu[b].size()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "block " + std::to_string(b) +
                ": sample mean and implied μ have different sizes"));
      }
      const Eigen::VectorXd d = s.mean[b] - m.mu[b];
      F_b += 0.5 * d.squaredNorm();
    }

    const double w_b = static_cast<double>(s.n_obs[b]) /
                       static_cast<double>(N_total);
    f += w_b * F_b;
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "F_ULS evaluated to non-finite"));
  }
  return f;
}

fit_expected<Eigen::VectorXd>
ULS::gradient(const SampleStats& s, const model::ImpliedMoments& m,
              const Eigen::MatrixXd& J,
              const Eigen::MatrixXd& Jmu) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
  std::int64_t N_total = 0;
  for (auto n : s.n_obs) N_total += n;
  if (N_total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats has non-positive total n_obs"));
  }

  // Stacked-vech residual w (rows match J): w_b = (n_b/N)·vech(Σ_b − S_b).
  // Stacked-mean residual u (rows match Jmu): u_b = (n_b/N)·(μ_b − m̄_b).
  Eigen::Index total_vech = 0;
  for (const auto& Sigma : m.sigma) total_vech += vech_len(Sigma.rows());

  const bool has_means = (Jmu.size() > 0);
  Eigen::Index total_p = 0;
  if (has_means) {
    for (const auto& Sigma : m.sigma) total_p += Sigma.rows();
  }

  Eigen::VectorXd w(total_vech);
  w.setZero();
  Eigen::VectorXd u(has_means ? total_p : 0);
  u.setZero();

  Eigen::Index off    = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    const auto& S     = s.S[b];
    const auto& Sigma = m.sigma[b];
    if (S.rows() != Sigma.rows() || S.cols() != Sigma.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "block " + std::to_string(b) +
              ": S and Σ have different shapes (gradient)"));
    }
    const double scale = static_cast<double>(s.n_obs[b]) /
                         static_cast<double>(N_total);
    const Eigen::Index pstar = vech_len(Sigma.rows());
    w.segment(off, pstar) =
        scale * vech_lower(Eigen::MatrixXd(Sigma - S));

    if (has_means) {
      // A model with Jmu but a block missing m̄/μ contributes u_b = 0 —
      // matches ML's "skip silently" gating. Cursor still advances.
      if (b < s.mean.size() && b < m.mu.size() &&
          s.mean[b].size() > 0 && m.mu[b].size() > 0) {
        if (s.mean[b].size() != m.mu[b].size()) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              "block " + std::to_string(b) +
                  ": sample mean and implied μ have different sizes (gradient)"));
        }
        u.segment(mu_off, Sigma.rows()) =
            scale * (m.mu[b] - s.mean[b]);
      }
      mu_off += Sigma.rows();
    }
    off += pstar;
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

// ============================================================================
// LS-shape interface — `residuals` / `residual_jacobian`. Per block:
//
//   r_b = √(n_b/N) · [ μ_b(θ) − m̄_b ; vech(Σ_b(θ) − S_b) ]
//   J_b = √(n_b/N) · [ J_μ,b         ; J_σ,b              ]
//
// Stacked over blocks in (means first, then vech) order so the block-row
// offsets are deterministic. When `has_means` is on but a specific block
// has no sample mean populated, its mean rows are zeroed (mirrors the
// scalar `value` / `gradient` "skip silently" rule).
// ============================================================================

fit_expected<Eigen::VectorXd>
ULS::residuals(const SampleStats& s, const model::ImpliedMoments& m) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
  std::int64_t N_total = 0;
  for (auto n : s.n_obs) N_total += n;
  if (N_total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats has non-positive total n_obs"));
  }

  const bool has_means = ls_has_means(s, m);
  Eigen::Index total_vech = 0;
  Eigen::Index total_p    = 0;
  for (const auto& Sigma : m.sigma) {
    total_vech += vech_len(Sigma.rows());
    if (has_means) total_p += Sigma.rows();
  }
  Eigen::VectorXd r(total_p + total_vech);
  r.setZero();

  Eigen::Index mu_off = 0;
  Eigen::Index sig_off = has_means ? total_p : 0;
  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    const auto& S     = s.S[b];
    const auto& Sigma = m.sigma[b];
    if (S.rows() != Sigma.rows() || S.cols() != Sigma.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "block " + std::to_string(b) +
              ": S and Σ have different shapes (residuals)"));
    }
    const double w_b = static_cast<double>(s.n_obs[b]) /
                       static_cast<double>(N_total);
    const double sw  = std::sqrt(w_b);
    const Eigen::Index pstar = vech_len(Sigma.rows());

    // Vech residuals (model − sample) so Jᵀr = ∇F and ½||r||² = F.
    r.segment(sig_off, pstar) = sw * vech_lower(Eigen::MatrixXd(Sigma - S));
    sig_off += pstar;

    if (has_means) {
      const Eigen::Index p = Sigma.rows();
      if (b < s.mean.size() && b < m.mu.size() &&
          s.mean[b].size() > 0 && m.mu[b].size() > 0) {
        if (s.mean[b].size() != m.mu[b].size()) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              "block " + std::to_string(b) +
                  ": sample mean and implied μ have different sizes (residuals)"));
        }
        r.segment(mu_off, p) = sw * (m.mu[b] - s.mean[b]);
      }
      mu_off += p;
    }
  }

  if (!r.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "ULS residual vector contains non-finite entries"));
  }
  return r;
}

fit_expected<Eigen::MatrixXd>
ULS::residual_jacobian(const SampleStats& s, const model::ImpliedMoments& m,
                       const Eigen::MatrixXd& J_sigma,
                       const Eigen::MatrixXd& J_mu) const {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats and ImpliedMoments have different block counts"));
  }
  std::int64_t N_total = 0;
  for (auto n : s.n_obs) N_total += n;
  if (N_total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats has non-positive total n_obs"));
  }

  const bool has_means = ls_has_means(s, m) && J_mu.size() > 0;
  Eigen::Index total_vech = 0;
  Eigen::Index total_p    = 0;
  for (const auto& Sigma : m.sigma) {
    total_vech += vech_len(Sigma.rows());
    if (has_means) total_p += Sigma.rows();
  }
  if (J_sigma.rows() != total_vech) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "J_sigma row count " + std::to_string(J_sigma.rows()) +
            " ≠ total vech length " + std::to_string(total_vech) +
            " (residual_jacobian)"));
  }
  if (has_means && J_mu.rows() != total_p) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "J_mu row count " + std::to_string(J_mu.rows()) +
            " ≠ total p " + std::to_string(total_p) +
            " (residual_jacobian)"));
  }
  const Eigen::Index n_free = J_sigma.cols();
  if (has_means && J_mu.cols() != n_free) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "J_sigma and J_mu column counts disagree (residual_jacobian)"));
  }

  Eigen::MatrixXd Jr(total_p + total_vech, n_free);
  // Per-block √(n_b/N) scaling on the corresponding row stripes. Doing this
  // inline (rather than a single end-scale by a block-diagonal weight) keeps
  // the rows aligned with the residual ordering computed in `residuals`.
  //   mu_off  : running offset in J_mu / Jr's mean-row region
  //   sig_offJ: running offset in J_sigma's row space (no mean-row prefix)
  //   sig_off : running offset in Jr's row space (after the mean-row prefix)
  Eigen::Index mu_off  = 0;
  Eigen::Index sig_offJ = 0;
  Eigen::Index sig_off  = has_means ? total_p : 0;
  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    const Eigen::Index p     = m.sigma[b].rows();
    const Eigen::Index pstar = vech_len(p);
    const double w_b = static_cast<double>(s.n_obs[b]) /
                       static_cast<double>(N_total);
    const double sw  = std::sqrt(w_b);

    Jr.block(sig_off, 0, pstar, n_free) =
        sw * J_sigma.block(sig_offJ, 0, pstar, n_free);
    sig_offJ += pstar;
    sig_off  += pstar;

    if (has_means) {
      // The per-block sample-mean populated check matches `residuals`: a
      // block missing its sample mean contributes zero rows.
      if (b < s.mean.size() && b < m.mu.size() &&
          s.mean[b].size() > 0 && m.mu[b].size() > 0) {
        Jr.block(mu_off, 0, p, n_free) =
            sw * J_mu.block(mu_off, 0, p, n_free);
      } else {
        Jr.block(mu_off, 0, p, n_free).setZero();
      }
      mu_off += p;
    }
  }
  return Jr;
}

}  // namespace magmaan::fit
