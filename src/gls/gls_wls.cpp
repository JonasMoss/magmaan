#include "magmaan/fit/gls.hpp"
#include "magmaan/fit/wls.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"

namespace magmaan::fit {

namespace {

using detail::vech_index;
using detail::vech_len;
using detail::vech_lower;

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
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
    return std::unexpected(make_err(kind,
        detail + ": weight matrix is not positive definite"));
  }
  return llt.matrixL();
}

fit_expected<std::vector<Eigen::MatrixXd>>
gls_block_factors(const SampleStats& s, const model::ImpliedMoments&,
                  const Layout& layout) {
  std::vector<Eigen::MatrixXd> factors;
  factors.reserve(s.S.size());

  for (std::size_t b = 0; b < s.S.size(); ++b) {
    const Eigen::Index p = s.S[b].rows();
    if (!s.S[b].allFinite()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "GLS: block " + std::to_string(b) +
              " sample covariance contains non-finite entries"));
    }
    Eigen::LLT<Eigen::MatrixXd> s_llt(s.S[b]);
    if (s_llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSample,
          "GLS: block " + std::to_string(b) +
              " sample covariance is not positive definite"));
    }
    const Eigen::MatrixXd Sinv =
        s_llt.solve(Eigen::MatrixXd::Identity(p, p));
    const Eigen::MatrixXd Wcov = symmetric_vech_gls_weight(Sinv);

    Eigen::MatrixXd W(layout.block_rows[b], layout.block_rows[b]);
    W.setZero();
    Eigen::Index off = 0;
    if (layout.has_means) {
      W.block(0, 0, p, p) = Sinv;
      off = p;
    }
    const Eigen::Index pstar = vech_len(p);
    W.block(off, off, pstar, pstar) = Wcov;
    auto L = llt_factor_for_weight(
        W, FitError::Kind::NonPositiveDefiniteSample,
        "GLS: block " + std::to_string(b));
    if (!L.has_value()) return std::unexpected(L.error());
    factors.push_back(std::move(*L));
  }
  return factors;
}

fit_expected<std::vector<Eigen::MatrixXd>>
wls_block_factors(const WLS& wls, const SampleStats& s,
                  const model::ImpliedMoments& m, const Layout& layout) {
  if (wls.weights.size() != s.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "WLS: weight block count " + std::to_string(wls.weights.size()) +
            " does not match sample block count " +
            std::to_string(s.S.size())));
  }

  std::vector<Eigen::MatrixXd> factors;
  factors.reserve(wls.weights.size());
  for (std::size_t b = 0; b < wls.weights.size(); ++b) {
    const auto& W = wls.weights[b];
    if (W.rows() != layout.block_rows[b] || W.cols() != layout.block_rows[b]) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "WLS: block " + std::to_string(b) +
              " weight dimension " + std::to_string(W.rows()) + "x" +
              std::to_string(W.cols()) + " does not match moment dimension " +
              std::to_string(layout.block_rows[b])));
    }
    auto L = llt_factor_for_weight(
        W, FitError::Kind::NumericIssue,
        "WLS: block " + std::to_string(b));
    if (!L.has_value()) return std::unexpected(L.error());
    factors.push_back(std::move(*L));
  }
  (void)m;
  return factors;
}

Eigen::VectorXd block_moment_delta(const SampleStats& s,
                                   const model::ImpliedMoments& m,
                                   const Layout& layout,
                                   std::size_t b) {
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

fit_expected<Eigen::VectorXd>
weighted_residuals(const SampleStats& s, const model::ImpliedMoments& m,
                   const Layout& layout,
                   const std::vector<Eigen::MatrixXd>& factors) {
  auto N = total_n_obs(s);
  if (!N.has_value()) return std::unexpected(N.error());

  Eigen::VectorXd r(layout.n_rows);
  Eigen::Index out_off = 0;
  for (std::size_t b = 0; b < s.S.size(); ++b) {
    const double sw = std::sqrt(static_cast<double>(s.n_obs[b]) /
                                static_cast<double>(*N));
    const Eigen::VectorXd d = block_moment_delta(s, m, layout, b);
    r.segment(out_off, layout.block_rows[b]) =
        sw * (factors[b].transpose() * d);
    out_off += layout.block_rows[b];
  }
  if (!r.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "LS weighted residual vector contains non-finite entries"));
  }
  return r;
}

fit_expected<Eigen::MatrixXd>
weighted_jacobian(const SampleStats& s, const model::ImpliedMoments& m,
                  const Eigen::MatrixXd& J_sigma,
                  const Eigen::MatrixXd& J_mu,
                  const Layout& layout,
                  const std::vector<Eigen::MatrixXd>& factors) {
  auto N = total_n_obs(s);
  if (!N.has_value()) return std::unexpected(N.error());

  if (J_sigma.rows() != layout.n_sigma_rows) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "LS weighted Jacobian: J_sigma row count " +
            std::to_string(J_sigma.rows()) +
            " does not match total vech length " +
            std::to_string(layout.n_sigma_rows)));
  }
  if (layout.has_means && J_mu.rows() != layout.n_mu_rows) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "LS weighted Jacobian: J_mu row count " +
            std::to_string(J_mu.rows()) +
            " does not match total mean length " +
            std::to_string(layout.n_mu_rows)));
  }

  const Eigen::Index n_free = J_sigma.cols();
  if (layout.has_means && J_mu.cols() != n_free) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "LS weighted Jacobian: J_sigma and J_mu column counts disagree"));
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
    Jr.block(out_off, 0, layout.block_rows[b], n_free) =
        sw * (factors[b].transpose() * Jb);
    out_off += layout.block_rows[b];
  }
  return Jr;
}

template <class FactorFn>
fit_expected<Eigen::VectorXd>
residuals_impl(const SampleStats& s, const model::ImpliedMoments& m,
               const char* who, FactorFn factor_fn) {
  if (auto ok = validate_common_shapes(s, m, who); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto layout = make_layout(s, m);
  auto factors = factor_fn(layout);
  if (!factors.has_value()) return std::unexpected(factors.error());
  return weighted_residuals(s, m, layout, *factors);
}

template <class FactorFn>
fit_expected<Eigen::MatrixXd>
jacobian_impl(const SampleStats& s, const model::ImpliedMoments& m,
              const Eigen::MatrixXd& J_sigma,
              const Eigen::MatrixXd& J_mu,
              const char* who, FactorFn factor_fn) {
  if (auto ok = validate_common_shapes(s, m, who); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto layout = make_layout(s, m);
  auto factors = factor_fn(layout);
  if (!factors.has_value()) return std::unexpected(factors.error());
  return weighted_jacobian(s, m, J_sigma, J_mu, layout, *factors);
}

fit_expected<double> value_from_residuals(fit_expected<Eigen::VectorXd> r) {
  if (!r.has_value()) return std::unexpected(r.error());
  const double f = 0.5 * r->squaredNorm();
  if (!std::isfinite(f)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "LS weighted objective evaluated to non-finite"));
  }
  return f;
}

fit_expected<Eigen::VectorXd>
gradient_from_residuals(fit_expected<Eigen::VectorXd> r,
                        fit_expected<Eigen::MatrixXd> Jr) {
  if (!r.has_value()) return std::unexpected(r.error());
  if (!Jr.has_value()) return std::unexpected(Jr.error());
  return Jr->transpose() * *r;
}

}  // namespace

fit_expected<double>
GLS::value(const SampleStats& s, const model::ImpliedMoments& m) const {
  return value_from_residuals(residuals(s, m));
}

fit_expected<Eigen::VectorXd>
GLS::gradient(const SampleStats& s, const model::ImpliedMoments& m,
              const Eigen::MatrixXd& J,
              const Eigen::MatrixXd& Jmu) const {
  return gradient_from_residuals(residuals(s, m),
                                residual_jacobian(s, m, J, Jmu));
}

fit_expected<Eigen::VectorXd>
GLS::residuals(const SampleStats& s, const model::ImpliedMoments& m) const {
  return residuals_impl(s, m, "GLS",
      [&](const Layout& layout) {
        return gls_block_factors(s, m, layout);
      });
}

fit_expected<Eigen::MatrixXd>
GLS::residual_jacobian(const SampleStats& s,
                       const model::ImpliedMoments& m,
                       const Eigen::MatrixXd& J_sigma,
                       const Eigen::MatrixXd& J_mu) const {
  return jacobian_impl(s, m, J_sigma, J_mu, "GLS",
      [&](const Layout& layout) {
        return gls_block_factors(s, m, layout);
      });
}

fit_expected<double>
WLS::value(const SampleStats& s, const model::ImpliedMoments& m) const {
  return value_from_residuals(residuals(s, m));
}

fit_expected<Eigen::VectorXd>
WLS::gradient(const SampleStats& s, const model::ImpliedMoments& m,
              const Eigen::MatrixXd& J,
              const Eigen::MatrixXd& Jmu) const {
  return gradient_from_residuals(residuals(s, m),
                                residual_jacobian(s, m, J, Jmu));
}

fit_expected<Eigen::VectorXd>
WLS::residuals(const SampleStats& s, const model::ImpliedMoments& m) const {
  return residuals_impl(s, m, "WLS",
      [&](const Layout& layout) {
        return wls_block_factors(*this, s, m, layout);
      });
}

fit_expected<Eigen::MatrixXd>
WLS::residual_jacobian(const SampleStats& s,
                       const model::ImpliedMoments& m,
                       const Eigen::MatrixXd& J_sigma,
                       const Eigen::MatrixXd& J_mu) const {
  return jacobian_impl(s, m, J_sigma, J_mu, "WLS",
      [&](const Layout& layout) {
        return wls_block_factors(*this, s, m, layout);
      });
}

}  // namespace magmaan::fit
