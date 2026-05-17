#include "magmaan/estimate/gmm/moment_quadratic.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"

#include "detail_vech.hpp"

namespace magmaan::gmm {

using data::SampleStats;

namespace {

using detail::vech_index;
using detail::vech_len;
using detail::vech_lower;

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

FitError model_err(const ModelError& e, const char* who) {
  return FitError{FitError::Kind::NonPositiveDefiniteSigma,
                  std::string(who) + ": " + e.detail, 0, 0.0};
}

fit_expected<std::int64_t> total_n_obs(const SampleStats& s) {
  std::int64_t total = 0;
  for (auto n : s.n_obs) total += n;
  if (total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "SampleStats has non-positive total n_obs"));
  }
  return total;
}

bool ls_has_means(const SampleStats& s, const model::ImpliedMoments& m) {
  for (std::size_t b = 0; b < m.mu.size(); ++b) {
    if (m.mu[b].size() > 0 && b < s.mean.size() && s.mean[b].size() > 0) {
      return true;
    }
  }
  return false;
}

fit_expected<void>
validate_common_shapes(const SampleStats& s, const model::ImpliedMoments& m,
                       const char* who) {
  if (s.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) +
            ": SampleStats and ImpliedMoments have different block counts"));
  }
  if (s.n_obs.size() != s.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": n_obs block count does not match S"));
  }
  for (std::size_t b = 0; b < s.S.size(); ++b) {
    const auto& S = s.S[b];
    const auto& Sigma = m.sigma[b];
    if (S.rows() != S.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": block " + std::to_string(b) +
              " sample covariance is not square"));
    }
    if (S.rows() != Sigma.rows() || S.cols() != Sigma.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": block " + std::to_string(b) +
              " S and Sigma have different shapes"));
    }
    if (b < s.mean.size() && b < m.mu.size() &&
        s.mean[b].size() > 0 && m.mu[b].size() > 0 &&
        s.mean[b].size() != m.mu[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": block " + std::to_string(b) +
              " sample mean and implied mu have different sizes"));
    }
  }
  return {};
}

// Per-block moment layout: each block contributes [mean (if any) ; vech(cov)].
struct Layout {
  bool has_means = false;
  Eigen::Index n_rows = 0;
  Eigen::Index n_mu_rows = 0;
  Eigen::Index n_sigma_rows = 0;
  std::vector<Eigen::Index> block_rows;
  std::vector<Eigen::Index> mu_offsets;
  std::vector<Eigen::Index> sigma_offsets;
};

Layout make_layout(const SampleStats& s, const model::ImpliedMoments& m) {
  Layout layout;
  layout.has_means = ls_has_means(s, m);
  layout.block_rows.resize(m.sigma.size());
  layout.mu_offsets.resize(m.sigma.size());
  layout.sigma_offsets.resize(m.sigma.size());

  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    const Eigen::Index p = m.sigma[b].rows();
    layout.mu_offsets[b] = layout.n_mu_rows;
    if (layout.has_means) layout.n_mu_rows += p;
    layout.sigma_offsets[b] = layout.n_sigma_rows;
    layout.n_sigma_rows += vech_len(p);
    layout.block_rows[b] = (layout.has_means ? p : 0) + vech_len(p);
  }
  for (auto n : layout.block_rows) layout.n_rows += n;
  return layout;
}

// W_kl = tr(S⁻¹ E_k S⁻¹ E_l) for symmetric lower-vech basis matrices E_k —
// the normal-theory GLS weight on the covariance moments.
Eigen::MatrixXd symmetric_vech_gls_weight(const Eigen::MatrixXd& Sinv) {
  const Eigen::Index p = Sinv.rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd W(pstar, pstar);
  W.setZero();

  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1; r1 < p; ++r1) {
      const Eigen::Index k1 = vech_index(p, r1, c1);
      Eigen::MatrixXd E1 = Eigen::MatrixXd::Zero(p, p);
      E1(r1, c1) = 1.0;
      E1(c1, r1) = 1.0;
      const Eigen::MatrixXd A1 = Sinv * E1 * Sinv;

      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2; r2 < p; ++r2) {
          const Eigen::Index k2 = vech_index(p, r2, c2);
          Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(p, p);
          E2(r2, c2) = 1.0;
          E2(c2, r2) = 1.0;
          W(k1, k2) = (A1 * E2).trace();
        }
      }
    }
  }
  return W;
}

// Lower-Cholesky factor L of W (W = L Lᵀ). Falls back to a clamped
// eigendecomposition when W is only positive *semi*definite.
fit_expected<Eigen::MatrixXd>
llt_factor_for_weight(const Eigen::MatrixXd& W, FitError::Kind kind,
                      const std::string& detail) {
  if (W.rows() != W.cols()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        detail + ": weight matrix is not square"));
  }
  if (!W.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        detail + ": weight matrix contains non-finite entries"));
  }
  if (!W.isApprox(W.transpose(), 1e-10)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        detail + ": weight matrix is not symmetric"));
  }
  Eigen::LLT<Eigen::MatrixXd> llt(W);
  if (llt.info() != Eigen::Success) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (W + W.transpose()));
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_err(kind,
          detail + ": weight matrix eigendecomposition failed"));
    }
    Eigen::VectorXd vals = es.eigenvalues();
    const double tol = 1e-10 * std::max<double>(1.0, W.cwiseAbs().maxCoeff());
    for (Eigen::Index i = 0; i < vals.size(); ++i) {
      if (vals(i) < -tol) {
        return std::unexpected(make_err(kind,
            detail + ": weight matrix is not positive semidefinite"));
      }
      vals(i) = std::sqrt(std::max(0.0, vals(i)));
    }
    return es.eigenvectors() * vals.asDiagonal();
  }
  return Eigen::MatrixXd(llt.matrixL());
}

// Stacked moment delta for block b: [μ_b(θ) − m̄_b ; vech(Σ_b(θ) − S_b)].
Eigen::VectorXd block_moment_delta(const SampleStats& s,
                                   const model::ImpliedMoments& m,
                                   const Layout& layout, std::size_t b) {
  const Eigen::Index p = m.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd d(layout.block_rows[b]);
  Eigen::Index off = 0;

  if (layout.has_means) {
    d.segment(0, p).setZero();
    if (b < s.mean.size() && b < m.mu.size() &&
        s.mean[b].size() > 0 && m.mu[b].size() > 0) {
      d.segment(0, p) = m.mu[b] - s.mean[b];
    }
    off = p;
  }
  d.segment(off, pstar) = vech_lower(Eigen::MatrixXd(m.sigma[b] - s.S[b]));
  return d;
}

// Whitened residual: per block, √(n_b/N) · Lᵀ·d. Empty `factors` ⇒ identity
// whitening (the ULS fast path — no matmul).
fit_expected<Eigen::VectorXd>
weighted_residuals(const SampleStats& s, const model::ImpliedMoments& m,
                   const Layout& layout,
                   const std::vector<Eigen::MatrixXd>& factors) {
  auto N = total_n_obs(s);
  if (!N.has_value()) return std::unexpected(N.error());
  const bool identity = factors.empty();

  Eigen::VectorXd r(layout.n_rows);
  Eigen::Index out_off = 0;
  for (std::size_t b = 0; b < s.S.size(); ++b) {
    const double sw = std::sqrt(static_cast<double>(s.n_obs[b]) /
                                static_cast<double>(*N));
    const Eigen::VectorXd d = block_moment_delta(s, m, layout, b);
    if (identity) {
      r.segment(out_off, layout.block_rows[b]) = sw * d;
    } else {
      r.segment(out_off, layout.block_rows[b]) =
          sw * (factors[b].transpose() * d);
    }
    out_off += layout.block_rows[b];
  }
  if (!r.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "moment-quadratic residual vector contains non-finite entries"));
  }
  return r;
}

// Whitened residual Jacobian: per block, √(n_b/N) · Lᵀ·[J_μ ; J_σ].
fit_expected<Eigen::MatrixXd>
weighted_jacobian(const SampleStats& s, const model::ImpliedMoments& m,
                  const Eigen::MatrixXd& J_sigma, const Eigen::MatrixXd& J_mu,
                  const Layout& layout,
                  const std::vector<Eigen::MatrixXd>& factors) {
  auto N = total_n_obs(s);
  if (!N.has_value()) return std::unexpected(N.error());
  const bool identity = factors.empty();

  if (J_sigma.rows() != layout.n_sigma_rows) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "moment-quadratic Jacobian: J_sigma row count " +
            std::to_string(J_sigma.rows()) +
            " does not match total vech length " +
            std::to_string(layout.n_sigma_rows)));
  }
  if (layout.has_means && J_mu.rows() != layout.n_mu_rows) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "moment-quadratic Jacobian: J_mu row count " +
            std::to_string(J_mu.rows()) +
            " does not match total mean length " +
            std::to_string(layout.n_mu_rows)));
  }

  const Eigen::Index n_free = J_sigma.cols();
  if (layout.has_means && J_mu.cols() != n_free) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "moment-quadratic Jacobian: J_sigma and J_mu column counts disagree"));
  }

  Eigen::MatrixXd Jr(layout.n_rows, n_free);
  Eigen::Index out_off = 0;
  for (std::size_t b = 0; b < s.S.size(); ++b) {
    const Eigen::Index p = m.sigma[b].rows();
    const Eigen::Index pstar = vech_len(p);
    Eigen::MatrixXd Jb(layout.block_rows[b], n_free);
    Eigen::Index local_off = 0;

    if (layout.has_means) {
      Jb.topRows(p).setZero();
      if (b < s.mean.size() && b < m.mu.size() &&
          s.mean[b].size() > 0 && m.mu[b].size() > 0) {
        Jb.topRows(p) = J_mu.block(layout.mu_offsets[b], 0, p, n_free);
      }
      local_off = p;
    }
    Jb.block(local_off, 0, pstar, n_free) =
        J_sigma.block(layout.sigma_offsets[b], 0, pstar, n_free);

    const double sw = std::sqrt(static_cast<double>(s.n_obs[b]) /
                                static_cast<double>(*N));
    if (identity) {
      Jr.block(out_off, 0, layout.block_rows[b], n_free) = sw * Jb;
    } else {
      Jr.block(out_off, 0, layout.block_rows[b], n_free) =
          sw * (factors[b].transpose() * Jb);
    }
    out_off += layout.block_rows[b];
  }
  return Jr;
}

}  // namespace

fit_expected<optim::GmmProblem>
residuals(const model::ModelEvaluator& ev, const data::SampleStats& samp,
          const Eigen::VectorXd& theta0, const Weight& weight) {
  auto eval0 = ev.evaluate(theta0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(model_err(eval0.error(), "gmm::residuals: theta0"));
  }
  if (auto ok = validate_common_shapes(samp, eval0->moments, "gmm::residuals");
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  const Layout layout = make_layout(samp, eval0->moments);

  // Factor the weight once. Empty ⇒ identity (no whitening at all).
  std::vector<Eigen::MatrixXd> factors;
  if (!weight.empty()) {
    if (weight.size() != samp.S.size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "gmm::residuals: weight block count " +
              std::to_string(weight.size()) +
              " does not match sample block count " +
              std::to_string(samp.S.size())));
    }
    factors.reserve(weight.size());
    for (std::size_t b = 0; b < weight.size(); ++b) {
      if (weight[b].rows() != layout.block_rows[b] ||
          weight[b].cols() != layout.block_rows[b]) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "gmm::residuals: block " + std::to_string(b) +
                " weight dimension does not match moment dimension " +
                std::to_string(layout.block_rows[b])));
      }
      auto L = llt_factor_for_weight(weight[b], FitError::Kind::NumericIssue,
                                     "gmm::residuals: block " +
                                         std::to_string(b));
      if (!L.has_value()) return std::unexpected(L.error());
      factors.push_back(std::move(*L));
    }
  }

  optim::GmmProblem prob;
  prob.n_resid = layout.n_rows;
  prob.n_param = static_cast<Eigen::Index>(ev.n_free());
  prob.expand  = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [&ev, samp, layout, factors](
               const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto e = ev.evaluate(x, false, false);
    if (!e.has_value()) {
      return std::unexpected(model_err(e.error(), "gmm residual evaluate"));
    }
    return weighted_residuals(samp, e->moments, layout, factors);
  };
  prob.J = [&ev, samp, layout, factors](
               const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto e = ev.evaluate(x, true, true);
    if (!e.has_value()) {
      return std::unexpected(model_err(e.error(), "gmm jacobian evaluate"));
    }
    return weighted_jacobian(samp, e->moments, e->J_sigma, e->J_mu, layout,
                             factors);
  };
  return prob;
}

fit_expected<Weight>
normal_theory_weight(const model::ModelEvaluator& ev,
                     const data::SampleStats& samp,
                     const Eigen::VectorXd& theta0) {
  auto eval0 = ev.evaluate(theta0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(
        model_err(eval0.error(), "gmm::normal_theory_weight: theta0"));
  }
  if (auto ok = validate_common_shapes(samp, eval0->moments,
                                       "gmm::normal_theory_weight");
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  const Layout layout = make_layout(samp, eval0->moments);

  Weight W;
  W.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p = samp.S[b].rows();
    if (!samp.S[b].allFinite()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "gmm::normal_theory_weight: block " + std::to_string(b) +
              " sample covariance contains non-finite entries"));
    }
    Eigen::LLT<Eigen::MatrixXd> s_llt(samp.S[b]);
    if (s_llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSample,
          "gmm::normal_theory_weight: block " + std::to_string(b) +
              " sample covariance is not positive definite"));
    }
    const Eigen::MatrixXd Sinv = s_llt.solve(Eigen::MatrixXd::Identity(p, p));
    const Eigen::MatrixXd Wcov = symmetric_vech_gls_weight(Sinv);

    Eigen::MatrixXd Wb(layout.block_rows[b], layout.block_rows[b]);
    Wb.setZero();
    Eigen::Index off = 0;
    if (layout.has_means) {
      Wb.block(0, 0, p, p) = Sinv;
      off = p;
    }
    Wb.block(off, off, vech_len(p), vech_len(p)) = Wcov;
    // Scale the whole block by ½ so the moment quadratic ½·rᵀWr lands on
    // lavaan's GLS objective scale: the χ² is then 2N·fmin uniformly with
    // ULS/WLS — GLS is just a WLS with this weight, not a separate criterion.
    Wb *= 0.5;
    W.push_back(std::move(Wb));
  }
  return W;
}

}  // namespace magmaan::gmm
