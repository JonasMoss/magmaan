#include "magmaan/robust/weighted_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/frontier/dls_weight.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/robust/weighted_chisq.hpp"

#include "detail_second_order.hpp"
#include "detail_vech.hpp"

namespace magmaan::estimate {

using data::RawData;
using data::SampleStats;
using robust::MeanVarAdjustedResult;
using robust::SatorraBentlerResult;
using robust::ScaledShiftedResult;

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

PostError fit_to_post(FitError e) {
  return make_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

PostError model_to_post(ModelError e) {
  return make_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

using detail::vech_index;
using detail::vech_len;
using detail::vech_lower;
using detail::vech_unpack;

// Convert h so tr(G X) == h^T vech(X) for symmetric X.
void vech_gradient_to_trace_weight(const Eigen::Ref<const Eigen::VectorXd>& x,
                                   Eigen::Index p,
                                   Eigen::Ref<Eigen::MatrixXd> G) {
  Eigen::Index k = 0;
  G.setZero();
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      if (r == c) {
        G(r, c) = x(k);
      } else {
        const double half = 0.5 * x(k);
        G(r, c) = half;
        G(c, r) = half;
      }
      ++k;
    }
  }
}

post_expected<Eigen::MatrixXd> inverse_sym_pd(const Eigen::MatrixXd& A,
                                              const char* what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " is not square"));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (A + A.transpose()));
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not positive definite"));
  }
  return ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
}

post_expected<Eigen::MatrixXd> symmetric_sqrt_psd(const Eigen::MatrixXd& A,
                                                  const char* what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " is not square"));
  }
  if (A.size() == 0) return Eigen::MatrixXd(A.rows(), A.cols());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " eigendecomposition failed"));
  }
  const double tol = 1e-10 * std::max<double>(1.0, A.cwiseAbs().maxCoeff());
  Eigen::VectorXd vals = es.eigenvalues();
  for (Eigen::Index i = 0; i < vals.size(); ++i) {
    if (vals(i) < -tol) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(what) + " is not positive semidefinite"));
    }
    vals(i) = std::sqrt(std::max(0.0, vals(i)));
  }
  return es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose();
}

bool same_matrix_within(const Eigen::MatrixXd& A,
                        const Eigen::MatrixXd& B,
                        double rel_tol) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
  const double scale = std::max({1.0, A.cwiseAbs().maxCoeff(),
                                B.cwiseAbs().maxCoeff()});
  return (A - B).cwiseAbs().maxCoeff() <= rel_tol * scale;
}

post_expected<Eigen::MatrixXd>
normal_theory_moment_metric(const Eigen::MatrixXd& S, const char* what) {
  auto G_or = data::gamma_nt(S);
  if (!G_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + ": gamma_nt failed: " + G_or.error().detail));
  }
  return inverse_sym_pd(*G_or, what);
}

post_expected<double> total_n(const data::SampleStats& samp) {
  double n = 0.0;
  for (auto nb : samp.n_obs) n += static_cast<double>(nb);
  if (!(n > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: SampleStats has non-positive total n_obs"));
  }
  return n;
}

// Standard (un-robust) LS reference statistic. Uniform 2N·fmin: the GLS
// moment weight now carries the ½ that used to make GLS a special case.
post_expected<double>
robust_ls_standard_chisq(const data::SampleStats& samp,
                         const Estimates& est) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return 2.0 * *n * est.fmin;
}

bool has_mean_rows(const data::SampleStats& samp,
                   const model::ImpliedMoments& moments) {
  for (std::size_t b = 0; b < moments.mu.size(); ++b) {
    if (moments.mu[b].size() > 0 && b < samp.mean.size() &&
        samp.mean[b].size() > 0) {
      return true;
    }
  }
  return false;
}

struct ContinuousLsLayout {
  bool has_means = false;
  std::vector<Eigen::Index> block_rows;
  std::vector<Eigen::Index> mu_offsets;
  std::vector<Eigen::Index> sigma_offsets;
  Eigen::Index total_mu_rows = 0;
  Eigen::Index total_sigma_rows = 0;
};

post_expected<void> validate_moment_shapes(const data::SampleStats& samp,
                                           const model::ImpliedMoments& moments) {
  if (samp.S.size() != moments.sigma.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: SampleStats and ImpliedMoments block counts differ"));
  }
  if (samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: n_obs block count does not match sample covariances"));
  }
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    if (samp.n_obs[b] <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_continuous_ls: non-positive n_obs in block " +
              std::to_string(b)));
    }
    const auto& S = samp.S[b];
    const auto& Sigma = moments.sigma[b];
    if (S.rows() != S.cols() || Sigma.rows() != Sigma.cols() ||
        S.rows() != Sigma.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_continuous_ls: covariance shape mismatch in block " +
              std::to_string(b)));
    }
    if (b < samp.mean.size() && b < moments.mu.size() &&
        samp.mean[b].size() > 0 && moments.mu[b].size() > 0 &&
        samp.mean[b].size() != moments.mu[b].size()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_continuous_ls: mean shape mismatch in block " +
              std::to_string(b)));
    }
  }
  return {};
}

ContinuousLsLayout make_layout(const data::SampleStats& samp,
                               const model::ImpliedMoments& moments) {
  ContinuousLsLayout layout;
  layout.has_means = has_mean_rows(samp, moments);
  layout.block_rows.resize(moments.sigma.size());
  layout.mu_offsets.resize(moments.sigma.size());
  layout.sigma_offsets.resize(moments.sigma.size());
  for (std::size_t b = 0; b < moments.sigma.size(); ++b) {
    const Eigen::Index p = moments.sigma[b].rows();
    layout.mu_offsets[b] = layout.total_mu_rows;
    if (layout.has_means) layout.total_mu_rows += p;
    layout.sigma_offsets[b] = layout.total_sigma_rows;
    layout.total_sigma_rows += vech_len(p);
    layout.block_rows[b] = (layout.has_means ? p : 0) + vech_len(p);
  }
  return layout;
}

post_expected<Eigen::MatrixXd>
continuous_moment_jacobian_block(const ContinuousLsLayout& layout,
                                 const model::ImpliedMoments& moments,
                                 const Eigen::MatrixXd& J_sigma,
                                 const Eigen::MatrixXd& J_mu,
                                 std::size_t b) {
  const Eigen::Index p = moments.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index n_free = J_sigma.cols();
  if (J_sigma.rows() != layout.total_sigma_rows) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: J_sigma row count does not match moment layout"));
  }
  if (layout.has_means) {
    if (J_mu.rows() != layout.total_mu_rows || J_mu.cols() != n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_continuous_ls: J_mu shape does not match moment layout"));
    }
  }

  Eigen::MatrixXd Jb(layout.block_rows[b], n_free);
  Eigen::Index out = 0;
  if (layout.has_means) {
    Jb.topRows(p).setZero();
    if (b < moments.mu.size() && moments.mu[b].size() > 0) {
      Jb.topRows(p) = J_mu.block(layout.mu_offsets[b], 0, p, n_free);
    }
    out = p;
  }
  Jb.block(out, 0, pstar, n_free) =
      J_sigma.block(layout.sigma_offsets[b], 0, pstar, n_free);
  return Jb;
}

Eigen::VectorXd continuous_block_residual(const data::SampleStats& samp,
                                          const model::ImpliedMoments& moments,
                                          const ContinuousLsLayout& layout,
                                          std::size_t b) {
  const Eigen::Index p = moments.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd d(layout.block_rows[b]);
  Eigen::Index off = 0;
  if (layout.has_means) {
    const bool have_mu =
        b < moments.mu.size() && moments.mu[b].size() == p;
    const Eigen::VectorXd mu_model =
        have_mu ? moments.mu[b] : Eigen::VectorXd::Zero(p);
    const Eigen::VectorXd mean_s =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : Eigen::VectorXd::Zero(p);
    d.head(p) = mu_model - mean_s;
    off = p;
  }
  d.segment(off, pstar) = vech_lower(moments.sigma[b] - samp.S[b]);
  return d;
}

// Per-block estimator weight for the robust sandwich. An empty `weight` ⇒
// ULS (identity); otherwise the caller-supplied block (GLS callers pass
// `gmm::normal_theory_weight`). The sandwich is weight-scale-invariant, so
// the absolute scale of a GLS/WLS weight does not affect `vcov`.
post_expected<Eigen::MatrixXd>
weight_block(const gmm::Weight& weight,
             const ContinuousLsLayout& layout,
             std::size_t b) {
  if (weight.empty()) {
    return Eigen::MatrixXd::Identity(layout.block_rows[b],
                                     layout.block_rows[b]);
  }
  if (weight.size() <= b) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: missing weight block " + std::to_string(b)));
  }
  const auto& W = weight[b];
  if (W.rows() != layout.block_rows[b] || W.cols() != layout.block_rows[b]) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: weight dimension mismatch in block " +
            std::to_string(b)));
  }
  return W;
}

Eigen::MatrixXd gamma_nt_directional(const Eigen::MatrixXd& S,
                                     const Eigen::MatrixXd& H) {
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(pstar, pstar);
  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1; r1 < p; ++r1) {
      const Eigen::Index k1 = vech_index(p, r1, c1);
      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2; r2 < p; ++r2) {
          const Eigen::Index k2 = vech_index(p, r2, c2);
          out(k1, k2) =
              H(r1, r2) * S(c1, c2) + S(r1, r2) * H(c1, c2) +
              H(r1, c2) * S(c1, r2) + S(r1, c2) * H(c1, r2);
        }
      }
    }
  }
  return out;
}

post_expected<Eigen::MatrixXd>
normal_theory_weight_correction_block(const ContinuousLsLayout& layout,
                                      const data::SampleStats& samp,
                                      const Eigen::VectorXd& residual,
                                      const Eigen::MatrixXd& moment_influence,
                                      const Eigen::MatrixXd& weight,
                                      std::size_t b) {
  const Eigen::Index p = samp.S[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index mb = layout.block_rows[b];
  const Eigen::Index cov_off = layout.has_means ? p : 0;
  if (residual.size() != mb || moment_influence.cols() != mb ||
      weight.rows() != mb || weight.cols() != mb ||
      cov_off + pstar != mb) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "normal_theory_weight_correction_block: shape mismatch in block " +
            std::to_string(b)));
  }

  Eigen::MatrixXd correction =
      Eigen::MatrixXd::Zero(moment_influence.rows(), mb);
  Eigen::MatrixXd dS(p, p);
  const Eigen::MatrixXd& S = samp.S[b];
  if (layout.has_means) {
    const Eigen::MatrixXd Wmu = weight.topLeftCorner(p, p);
    const Eigen::RowVectorXd lhs_mu = residual.head(p).transpose() * Wmu;
    for (Eigen::Index i = 0; i < moment_influence.rows(); ++i) {
      vech_unpack(moment_influence.row(i).segment(cov_off, pstar).transpose(),
                  p, dS);
      correction.row(i).head(p) = lhs_mu * dS * Wmu;
    }
  }

  const Eigen::MatrixXd Wcov =
      weight.block(cov_off, cov_off, pstar, pstar);
  const Eigen::RowVectorXd lhs_cov =
      residual.segment(cov_off, pstar).transpose() * Wcov;
  for (Eigen::Index i = 0; i < moment_influence.rows(); ++i) {
    vech_unpack(moment_influence.row(i).segment(cov_off, pstar).transpose(),
                p, dS);
    const Eigen::MatrixXd dG = gamma_nt_directional(S, dS);
    correction.row(i).segment(cov_off, pstar) = lhs_cov * dG * Wcov;
  }
  return correction;
}

post_expected<Eigen::MatrixXd>
normal_theory_gamma_correction_block(const ContinuousLsLayout& layout,
                                     const data::SampleStats& samp,
                                     const Eigen::VectorXd& residual,
                                     const Eigen::MatrixXd& moment_influence,
                                     const Eigen::MatrixXd& weight,
                                     std::size_t b) {
  const Eigen::Index p = samp.S[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index mb = layout.block_rows[b];
  const Eigen::Index cov_off = layout.has_means ? p : 0;
  if (residual.size() != mb || moment_influence.cols() != mb ||
      weight.rows() != mb || weight.cols() != mb ||
      cov_off + pstar != mb) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "normal_theory_gamma_correction_block: shape mismatch in block " +
            std::to_string(b)));
  }

  const Eigen::MatrixXd& S = samp.S[b];
  const Eigen::RowVectorXd lhs = residual.transpose() * weight;
  Eigen::MatrixXd correction =
      Eigen::MatrixXd::Zero(moment_influence.rows(), mb);
  Eigen::MatrixXd dS(p, p);
  Eigen::MatrixXd dG = Eigen::MatrixXd::Zero(mb, mb);
  for (Eigen::Index i = 0; i < moment_influence.rows(); ++i) {
    dG.setZero();
    vech_unpack(moment_influence.row(i).segment(cov_off, pstar).transpose(),
                p, dS);
    if (layout.has_means) {
      dG.topLeftCorner(p, p) = dS;
    }
    dG.block(cov_off, cov_off, pstar, pstar) =
        gamma_nt_directional(S, dS);
    correction.row(i) = lhs * dG * weight;
  }
  return correction;
}

Eigen::VectorXd empirical_gamma_mean_derivative_row(
    const Eigen::VectorXd& h,
    const Eigen::VectorXd& x,
    bool has_means) {
  const Eigen::Index p = x.size();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd dz((has_means ? p : 0) + pstar);
  Eigen::Index off = 0;
  if (has_means) {
    dz.head(p) = -h;
    off = p;
  }
  Eigen::Index k = off;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      dz(k++) = -(h(r) * x(c) + x(r) * h(c));
    }
  }
  return dz;
}

post_expected<gmm::Weight>
empirical_wls_weight_from_rows(const std::vector<Eigen::MatrixXd>& rows,
                               const std::vector<Eigen::Index>& block_rows,
                               const char* who) {
  if (rows.size() != block_rows.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) + ": empirical-row block count mismatch"));
  }
  gmm::Weight out;
  out.reserve(rows.size());
  for (std::size_t b = 0; b < rows.size(); ++b) {
    const Eigen::MatrixXd& Z = rows[b];
    if (Z.rows() <= 0 || Z.cols() != block_rows[b]) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(who) + ": empirical-row shape mismatch in block " +
              std::to_string(b)));
    }
    Eigen::MatrixXd Gamma =
        (Z.transpose() * Z) / static_cast<double>(Z.rows());
    Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();
    const std::string what =
        std::string(who) + " empirical Gamma block " + std::to_string(b);
    auto inv_or = inverse_sym_pd(Gamma, what.c_str());
    if (!inv_or.has_value()) return std::unexpected(inv_or.error());
    out.push_back(std::move(*inv_or));
  }
  return out;
}

post_expected<gmm::Weight>
empirical_dwls_weight_from_rows(const std::vector<Eigen::MatrixXd>& rows,
                                const std::vector<Eigen::Index>& block_rows,
                                const char* who) {
  if (rows.size() != block_rows.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) + ": empirical-row block count mismatch"));
  }
  gmm::Weight out;
  out.reserve(rows.size());
  for (std::size_t b = 0; b < rows.size(); ++b) {
    const Eigen::MatrixXd& Z = rows[b];
    if (Z.rows() <= 0 || Z.cols() != block_rows[b]) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(who) + ": empirical-row shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::RowVectorXd gamma =
        Z.array().square().colwise().mean().matrix();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(Z.cols(), Z.cols());
    for (Eigen::Index k = 0; k < gamma.size(); ++k) {
      if (!(gamma(k) > 0.0) || !std::isfinite(gamma(k))) {
        return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
            std::string(who) + ": empirical Gamma diagonal is not positive "
            "in block " + std::to_string(b)));
      }
      W(k, k) = 1.0 / gamma(k);
    }
    out.push_back(std::move(W));
  }
  return out;
}

post_expected<Eigen::MatrixXd>
empirical_weight_correction_block(const ContinuousLsLayout& layout,
                                  const data::SampleStats& samp,
                                  const data::RawData& raw,
                                  const Eigen::VectorXd& residual,
                                  const Eigen::MatrixXd& moment_influence,
                                  const Eigen::MatrixXd& weight,
                                  std::size_t b) {
  const Eigen::Index mb = layout.block_rows[b];
  if (residual.size() != mb || moment_influence.cols() != mb ||
      weight.rows() != mb || weight.cols() != mb ||
      raw.X.size() <= b || raw.X[b].rows() != moment_influence.rows() ||
      raw.X[b].cols() != samp.S[b].rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "empirical_weight_correction_block: shape mismatch in block " +
            std::to_string(b)));
  }

  const Eigen::MatrixXd& X = raw.X[b];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::VectorXd mean_b =
      (b < samp.mean.size() && samp.mean[b].size() == p)
          ? samp.mean[b]
          : X.colwise().mean().transpose().eval();
  const Eigen::MatrixXd Xc = X.rowwise() - mean_b.transpose();
  Eigen::MatrixXd Gamma =
      (moment_influence.transpose() * moment_influence) /
      static_cast<double>(n);
  Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();

  std::vector<Eigen::MatrixXd> L;
  L.reserve(static_cast<std::size_t>(p));
  for (Eigen::Index r = 0; r < p; ++r) {
    Eigen::VectorXd h = Eigen::VectorXd::Zero(p);
    h(r) = 1.0;
    Eigen::MatrixXd Lr = Eigen::MatrixXd::Zero(mb, mb);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd x = Xc.row(i).transpose();
      const Eigen::VectorXd dz =
          empirical_gamma_mean_derivative_row(h, x, layout.has_means);
      const Eigen::RowVectorXd z = moment_influence.row(i);
      Lr.noalias() += dz * z;
      Lr.noalias() += z.transpose() * dz.transpose();
    }
    Lr /= static_cast<double>(n);
    Lr = 0.5 * (Lr + Lr.transpose()).eval();
    L.push_back(std::move(Lr));
  }

  const Eigen::RowVectorXd lhs = residual.transpose() * weight;
  const Eigen::RowVectorXd base = -lhs * Gamma * weight;
  const Eigen::MatrixXd Zw = moment_influence * weight;
  std::vector<Eigen::RowVectorXd> lhs_Lw;
  lhs_Lw.reserve(static_cast<std::size_t>(p));
  for (Eigen::Index r = 0; r < p; ++r) {
    lhs_Lw.push_back(lhs * L[static_cast<std::size_t>(r)] * weight);
  }

  Eigen::MatrixXd correction(n, mb);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::RowVectorXd z = moment_influence.row(i);
    Eigen::RowVectorXd row = base + z.dot(lhs) * Zw.row(i);
    for (Eigen::Index r = 0; r < p; ++r) {
      row.noalias() += Xc(i, r) * lhs_Lw[static_cast<std::size_t>(r)];
    }
    correction.row(i) = row;
  }
  return correction;
}

post_expected<Eigen::MatrixXd>
empirical_diagonal_weight_correction_block(
    const ContinuousLsLayout& layout,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& moment_influence,
    const Eigen::MatrixXd& weight,
    std::size_t b) {
  const Eigen::Index mb = layout.block_rows[b];
  if (residual.size() != mb || moment_influence.cols() != mb ||
      weight.rows() != mb || weight.cols() != mb ||
      raw.X.size() <= b || raw.X[b].rows() != moment_influence.rows() ||
      raw.X[b].cols() != samp.S[b].rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "empirical_diagonal_weight_correction_block: shape mismatch in block " +
            std::to_string(b)));
  }
  const Eigen::MatrixXd Wdiag = weight.diagonal().asDiagonal();
  const double offdiag = (weight - Wdiag).cwiseAbs().maxCoeff();
  if (offdiag > 1e-12 * std::max<double>(1.0, weight.cwiseAbs().maxCoeff())) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "empirical_diagonal_weight_correction_block: weight is not diagonal"));
  }

  const Eigen::MatrixXd& X = raw.X[b];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::VectorXd mean_b =
      (b < samp.mean.size() && samp.mean[b].size() == p)
          ? samp.mean[b]
          : X.colwise().mean().transpose().eval();
  const Eigen::MatrixXd Xc = X.rowwise() - mean_b.transpose();
  const Eigen::RowVectorXd gamma =
      moment_influence.array().square().colwise().mean().matrix();
  for (Eigen::Index k = 0; k < gamma.size(); ++k) {
    if (!(gamma(k) > 0.0) || !std::isfinite(gamma(k))) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          "empirical_diagonal_weight_correction_block: empirical Gamma "
          "diagonal is not positive in block " + std::to_string(b)));
    }
  }

  Eigen::MatrixXd Ldiag = Eigen::MatrixXd::Zero(p, mb);
  for (Eigen::Index r = 0; r < p; ++r) {
    Eigen::VectorXd h = Eigen::VectorXd::Zero(p);
    h(r) = 1.0;
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd x = Xc.row(i).transpose();
      const Eigen::VectorXd dz =
          empirical_gamma_mean_derivative_row(h, x, layout.has_means);
      for (Eigen::Index k = 0; k < mb; ++k) {
        Ldiag(r, k) += 2.0 * dz(k) * moment_influence(i, k);
      }
    }
    Ldiag.row(r) /= static_cast<double>(n);
  }

  const Eigen::RowVectorXd scale =
      residual.transpose().array() / gamma.array().square();
  Eigen::MatrixXd correction(n, mb);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::RowVectorXd ifgamma =
        moment_influence.row(i).array().square().matrix();
    ifgamma -= gamma;
    for (Eigen::Index r = 0; r < p; ++r) {
      ifgamma.noalias() += Xc(i, r) * Ldiag.row(r);
    }
    correction.row(i) = ifgamma.array() * scale.array();
  }
  return correction;
}

post_expected<Eigen::MatrixXd>
dls_weight_correction_block(const ContinuousLsLayout& layout,
                            const data::SampleStats& samp,
                            const data::RawData& raw,
                            const Eigen::VectorXd& residual,
                            const Eigen::MatrixXd& moment_influence,
                            const Eigen::MatrixXd& weight,
                            frontier::DlsWeightOptions opts,
                            std::size_t b) {
  if (!(opts.a >= 0.0 && opts.a <= 1.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "dls_weight_correction_block: mixing scalar a must be in [0, 1]"));
  }

  Eigen::MatrixXd correction =
      Eigen::MatrixXd::Zero(moment_influence.rows(), moment_influence.cols());
  if (opts.a < 1.0) {
    auto nt_or = normal_theory_gamma_correction_block(
        layout, samp, residual, moment_influence, weight, b);
    if (!nt_or.has_value()) return std::unexpected(nt_or.error());
    correction.noalias() += (1.0 - opts.a) * (*nt_or);
  }
  if (opts.a > 0.0) {
    auto emp_or = empirical_weight_correction_block(
        layout, samp, raw, residual, moment_influence, weight, b);
    if (!emp_or.has_value()) return std::unexpected(emp_or.error());
    correction.noalias() += opts.a * (*emp_or);
  }
  return correction;
}

post_expected<std::vector<Eigen::MatrixXd>>
continuous_ls_moment_influence_blocks_from_raw(
    const data::RawData& raw,
    const data::SampleStats& samp,
    const ContinuousLsLayout& layout) {
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: RawData masks are not supported for complete-data LS"));
  }
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: RawData and SampleStats block counts differ"));
  }
  std::vector<Eigen::MatrixXd> rows_out;
  rows_out.reserve(raw.X.size());
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n != samp.n_obs[b] || p != samp.S[b].rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_continuous_ls: raw-data shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::Index pstar = vech_len(p);
    const Eigen::Index rows = layout.block_rows[b];
    Eigen::MatrixXd Z(n, rows);
    Z.setZero();
    const Eigen::VectorXd mean_b =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : X.colwise().mean().transpose().eval();
    const Eigen::MatrixXd Xc = X.rowwise() - mean_b.transpose();
    const Eigen::VectorXd s_vech = vech_lower(samp.S[b]);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd xi = Xc.row(i).transpose();
      Eigen::Index off = 0;
      if (layout.has_means) {
        Z.block(i, 0, 1, p) = xi.transpose();
        off = p;
      }
      const Eigen::VectorXd d_i = vech_lower(Eigen::MatrixXd(xi * xi.transpose())) -
                                  s_vech;
      Z.block(i, off, 1, pstar) = d_i.transpose();
    }
    rows_out.push_back(std::move(Z));
  }
  return rows_out;
}

post_expected<std::vector<Eigen::MatrixXd>>
gamma_blocks_from_raw(const data::RawData& raw,
                      const data::SampleStats& samp,
                      const ContinuousLsLayout& layout) {
  auto rows_or = continuous_ls_moment_influence_blocks_from_raw(raw, samp, layout);
  if (!rows_or.has_value()) return std::unexpected(rows_or.error());
  std::vector<Eigen::MatrixXd> out;
  out.reserve(rows_or->size());
  for (const auto& Z : *rows_or) {
    const double inv_n = 1.0 / static_cast<double>(Z.rows());
    Eigen::MatrixXd Gamma = Z.transpose() * Z * inv_n;
    Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();
    out.push_back(std::move(Gamma));
  }
  return out;
}

// Shared block assembly for `robust_continuous_ls` and the moment-metric
// parameter-space sandwich: Δ_b / W_b / Γ̂_b / n_b at `est.theta`, which may
// belong to an augmented (freed-candidate) partable. `pt` must already be
// fixed-x-resolved.
post_expected<std::vector<WeightedMomentBlock>>
build_continuous_ls_blocks(const spec::LatentStructure& pt,
                           const model::MatrixRep& rep,
                           const data::SampleStats& samp,
                           const Estimates& est,
                           const gmm::Weight& weight,
                           const std::vector<Eigen::MatrixXd>& gamma) {
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: fitted theta length does not match partable"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(model_to_post(ev_or.error()));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(model_to_post(eval.error()));
  }
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = make_layout(samp, eval->moments);
  if (gamma.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: gamma block count does not match sample blocks"));
  }

  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto Jb = continuous_moment_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = weight_block(weight, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    if (gamma[b].rows() != layout.block_rows[b] ||
        gamma[b].cols() != layout.block_rows[b]) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_continuous_ls: gamma dimension mismatch in block " +
              std::to_string(b)));
    }
    blocks.push_back(WeightedMomentBlock{
        .jacobian = std::move(*Jb),
        .weight = std::move(*Wb),
        .gamma = gamma[b],
        .n_obs = samp.n_obs[b]});
  }
  return blocks;
}

post_expected<Eigen::MatrixXd>
continuous_ls_observed_bread_analytic(const spec::LatentStructure& pt,
                                      const model::MatrixRep& rep,
                                      const data::SampleStats& samp,
                                      const Estimates& est,
                                      const gmm::Weight& weight,
                                      const Eigen::MatrixXd& K) {
  if (est.theta.size() != static_cast<Eigen::Index>(pt.n_free())) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_observed_bread: fitted theta length does not match partable"));
  }
  if (K.rows() != est.theta.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_observed_bread: K row count does not match theta length"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  auto assembled = ev_or->assembled(est.theta);
  if (!assembled.has_value()) {
    return std::unexpected(model_to_post(assembled.error()));
  }
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (assembled->blocks.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_observed_bread: assembled block count does not match sample blocks"));
  }
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  const double N_total = *n;
  const auto layout = make_layout(samp, eval->moments);
  const auto locs = ev_or->param_locations();
  const Eigen::Index q = est.theta.size();
  if (locs.size() != static_cast<std::size_t>(q)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_observed_bread: parameter-location count mismatch"));
  }

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto Jb = continuous_moment_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = weight_block(weight, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    const Eigen::VectorXd d =
        continuous_block_residual(samp, eval->moments, layout, b);
    const Eigen::VectorXd h = (*Wb) * d;
    const double w_b = static_cast<double>(samp.n_obs[b]) / N_total;

    H.noalias() += w_b * (Jb->transpose() * (*Wb) * (*Jb));

    const Eigen::Index p = eval->moments.sigma[b].rows();
    const Eigen::Index pstar = vech_len(p);
    const Eigen::Index cov_off = layout.has_means ? p : 0;
    if (h.size() != layout.block_rows[b] || cov_off + pstar != h.size()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "continuous_ls_observed_bread: residual-gradient shape mismatch in block " +
              std::to_string(b)));
    }

    Eigen::MatrixXd G(p, p);
    vech_gradient_to_trace_weight(h.segment(cov_off, pstar), p, G);
    const auto sow = detail::SecondOrderWeights::build(
        std::move(G), assembled->blocks[b], layout.has_means);
    const auto blk = static_cast<std::int8_t>(b);
    std::vector<Eigen::Index> block_params;
    block_params.reserve(locs.size());
    for (Eigen::Index k = 0; k < q; ++k) {
      if (locs[static_cast<std::size_t>(k)].block == blk) {
        block_params.push_back(k);
      }
    }
    for (std::size_t ai = 0; ai < block_params.size(); ++ai) {
      const Eigen::Index a = block_params[ai];
      const auto& la = locs[static_cast<std::size_t>(a)];
      for (std::size_t ci = ai; ci < block_params.size(); ++ci) {
        const Eigen::Index c = block_params[ci];
        const auto& lb = locs[static_cast<std::size_t>(c)];
        double h2 = detail::second_sigma_trace(la, lb, sow,
                                               assembled->blocks[b]);
        if (layout.has_means) {
          h2 += h.head(p).dot(detail::second_mu(la, lb,
                                                assembled->blocks[b],
                                                sow.A_alpha));
        }
        const double val = w_b * h2;
        H(a, c) += val;
        if (a != c) H(c, a) += val;
      }
    }
  }

  if (!H.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_observed_bread: non-finite Hessian"));
  }
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::MatrixXd Halpha = K.transpose() * H * K;
  Halpha = 0.5 * (Halpha + Halpha.transpose()).eval();
  return Halpha;
}

post_expected<Eigen::MatrixXd>
continuous_ls_observed_bread(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             const data::SampleStats& samp,
                             const Estimates& est,
                             const gmm::Weight& weight,
                             const Eigen::MatrixXd& K) {
  return continuous_ls_observed_bread_analytic(pt, rep, samp, est, weight, K);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_impl(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::SampleStats& samp,
                          const Estimates& est,
                          const gmm::Weight& weight,
                          const std::vector<Eigen::MatrixXd>& gamma,
                          robust::Information bread) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  if (con_or->K().rows() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: constraint reparameterization has incompatible shape"));
  }

  auto blocks = build_continuous_ls_blocks(pt, rep, samp, est, weight, gamma);
  if (!blocks.has_value()) return std::unexpected(blocks.error());

  auto chisq = robust_ls_standard_chisq(samp, est);
  if (!chisq.has_value()) return std::unexpected(chisq.error());
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());

  std::optional<Eigen::MatrixXd> bread_override;
  if (bread == robust::Information::Observed) {
    auto ob = continuous_ls_observed_bread(pt, rep, samp, est, weight,
                                           con_or->K());
    if (!ob.has_value()) return std::unexpected(ob.error());
    bread_override = std::move(*ob);
  }
  return robust_weighted_moments(*blocks, con_or->K(), *chisq / *n,
                                 bread_override);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_raw_impl(spec::LatentStructure pt,
                              const model::MatrixRep& rep,
                              const data::SampleStats& samp,
                              const Estimates& est,
                              const gmm::Weight& weight,
                              const data::RawData& raw,
                              robust::Information bread) {
  spec::LatentStructure pt_for_layout = pt;
  if (auto e = resolve_fixed_x_from_sample(pt_for_layout, rep, samp);
      !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto ev_or = model::ModelEvaluator::build(pt_for_layout, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = make_layout(samp, eval->moments);
  auto gamma = gamma_blocks_from_raw(raw, samp, layout);
  if (!gamma.has_value()) return std::unexpected(gamma.error());
  return robust_continuous_ls_impl(std::move(pt), rep, samp, est,
                                   weight, *gamma, bread);
}

enum class ContinuousLsIJWeightMode {
  Fixed,
  SampleNormalTheory,
  SampleEmpiricalWls,
  SampleEmpiricalDwls,
  SampleDls,
};

const char* continuous_ls_ij_name(ContinuousLsIJWeightMode mode) {
  switch (mode) {
    case ContinuousLsIJWeightMode::Fixed:
      return "robust_continuous_ls_fixed_weight_ij";
    case ContinuousLsIJWeightMode::SampleNormalTheory:
      return "robust_continuous_ls_gls_ij";
    case ContinuousLsIJWeightMode::SampleEmpiricalWls:
      return "robust_continuous_ls_wls_ij";
    case ContinuousLsIJWeightMode::SampleEmpiricalDwls:
      return "robust_continuous_ls_dwls_ij";
    case ContinuousLsIJWeightMode::SampleDls:
      return "robust_continuous_ls_dls_ij";
  }
  return "robust_continuous_ls_ij";
}

post_expected<WeightedRobustResult>
robust_continuous_ls_ij_impl(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::SampleStats& samp,
                             const Estimates& est,
                             const gmm::Weight& weight,
                             const data::RawData& raw,
                             ContinuousLsIJWeightMode mode,
                             frontier::DlsWeightOptions dls_opts = {}) {
  const char* who = continuous_ls_ij_name(mode);
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) +
            ": constraint reparameterization has incompatible shape"));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) + ": fitted theta length does not match partable"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = make_layout(samp, eval->moments);
  auto rows_or = continuous_ls_moment_influence_blocks_from_raw(raw, samp, layout);
  if (!rows_or.has_value()) return std::unexpected(rows_or.error());
  if (rows_or->size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(who) + ": influence block count mismatch"));
  }

  gmm::Weight estimated_weight;
  const gmm::Weight* active_weight = &weight;
  if (mode == ContinuousLsIJWeightMode::SampleNormalTheory) {
    auto w_or = gmm::normal_theory_weight(*ev_or, samp, est.theta);
    if (!w_or.has_value()) return std::unexpected(fit_to_post(w_or.error()));
    estimated_weight = std::move(*w_or);
    active_weight = &estimated_weight;
  } else if (mode == ContinuousLsIJWeightMode::SampleEmpiricalWls) {
    auto w_or = empirical_wls_weight_from_rows(
        *rows_or, layout.block_rows, who);
    if (!w_or.has_value()) return std::unexpected(w_or.error());
    estimated_weight = std::move(*w_or);
    active_weight = &estimated_weight;
  } else if (mode == ContinuousLsIJWeightMode::SampleEmpiricalDwls) {
    auto w_or = empirical_dwls_weight_from_rows(
        *rows_or, layout.block_rows, who);
    if (!w_or.has_value()) return std::unexpected(w_or.error());
    estimated_weight = std::move(*w_or);
    active_weight = &estimated_weight;
  } else if (mode == ContinuousLsIJWeightMode::SampleDls) {
    auto w_or = frontier::dls_weight(*ev_or, samp, raw, est.theta, dls_opts);
    if (!w_or.has_value()) return std::unexpected(fit_to_post(w_or.error()));
    estimated_weight = std::move(*w_or);
    active_weight = &estimated_weight;
  }

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto Jb = continuous_moment_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = weight_block(*active_weight, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    Eigen::MatrixXd correction;
    if (mode == ContinuousLsIJWeightMode::SampleNormalTheory) {
      const Eigen::VectorXd d_b =
          continuous_block_residual(samp, eval->moments, layout, b);
      auto corr_or = normal_theory_weight_correction_block(
          layout, samp, d_b, (*rows_or)[b], *Wb, b);
      if (!corr_or.has_value()) return std::unexpected(corr_or.error());
      correction = std::move(*corr_or);
    } else if (mode == ContinuousLsIJWeightMode::SampleEmpiricalWls) {
      const Eigen::VectorXd d_b =
          continuous_block_residual(samp, eval->moments, layout, b);
      auto corr_or = empirical_weight_correction_block(
          layout, samp, raw, d_b, (*rows_or)[b], *Wb, b);
      if (!corr_or.has_value()) return std::unexpected(corr_or.error());
      correction = std::move(*corr_or);
    } else if (mode == ContinuousLsIJWeightMode::SampleEmpiricalDwls) {
      const Eigen::VectorXd d_b =
          continuous_block_residual(samp, eval->moments, layout, b);
      auto corr_or = empirical_diagonal_weight_correction_block(
          layout, samp, raw, d_b, (*rows_or)[b], *Wb, b);
      if (!corr_or.has_value()) return std::unexpected(corr_or.error());
      correction = std::move(*corr_or);
    } else if (mode == ContinuousLsIJWeightMode::SampleDls) {
      const Eigen::VectorXd d_b =
          continuous_block_residual(samp, eval->moments, layout, b);
      auto corr_or = dls_weight_correction_block(
          layout, samp, raw, d_b, (*rows_or)[b], *Wb, dls_opts, b);
      if (!corr_or.has_value()) return std::unexpected(corr_or.error());
      correction = std::move(*corr_or);
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = std::move(*Jb),
        .weight = std::move(*Wb),
        .moment_influence = (*rows_or)[b],
        .weight_correction = std::move(correction),
        .n_obs = samp.n_obs[b]});
  }

  auto ob = continuous_ls_observed_bread(pt, rep, samp, est, *active_weight, K);
  if (!ob.has_value()) return std::unexpected(ob.error());
  auto chisq = robust_ls_standard_chisq(samp, est);
  if (!chisq.has_value()) return std::unexpected(chisq.error());
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return robust_weighted_moment_ij(ij_blocks, K, *chisq / *n, *ob);
}

}  // namespace

post_expected<double>
evaluate_ls_objective(spec::LatentStructure pt,
                      const model::MatrixRep& rep,
                      const data::SampleStats& samp,
                      const Eigen::VectorXd& theta,
                      const gmm::Weight& weight) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  if (static_cast<std::size_t>(theta.size()) != ev_or->n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "evaluate_ls_objective: theta length " + std::to_string(theta.size()) +
            " does not match evaluator n_free " +
            std::to_string(ev_or->n_free())));
  }
  auto prob = gmm::residuals(*ev_or, samp, theta, weight);
  if (!prob.has_value()) return std::unexpected(fit_to_post(prob.error()));
  auto r = prob->r(theta);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  return 0.5 * r->squaredNorm();
}

post_expected<double>
continuous_ls_chisq(data::SampleStats samp,
                    spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const Estimates& est,
                    const gmm::Weight& weight) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (weight.empty()) {
    // ULS — Browne's residual-based normal-theory statistic (N·F_ULS is not
    // itself asymptotically χ²).
    auto br = inference::browne_residual_nt(std::move(pt), rep, samp, est);
    if (!br.has_value()) return std::unexpected(br.error());
    const double n_used = *n - static_cast<double>(samp.S.size());
    if (!(n_used > 0.0)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "continuous_ls_chisq ULS: non-positive effective sample size"));
    }
    return *br * (n_used / *n);
  }
  // GLS / WLS — the moment quadratic's reference statistic is 2N·fmin.
  return 2.0 * *n * est.fmin;
}

post_expected<Eigen::MatrixXd>
observed_moment_bread_fd(
    const std::function<post_expected<Eigen::VectorXd>(const Eigen::VectorXd&)>&
        grad_at,
    const Eigen::VectorXd& theta_hat,
    const Eigen::MatrixXd& K,
    double h_rel) {
  const Eigen::Index q = theta_hat.size();
  if (q == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "observed_moment_bread_fd: empty theta"));
  }
  if (K.rows() != q) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "observed_moment_bread_fd: K row count does not match theta length"));
  }
  if (!(h_rel > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "observed_moment_bread_fd: non-positive step"));
  }
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index k = 0; k < q; ++k) {
    const double hk = h_rel * std::max(1.0, std::abs(theta_hat(k)));
    Eigen::VectorXd tp = theta_hat;
    Eigen::VectorXd tm = theta_hat;
    tp(k) += hk;
    tm(k) -= hk;
    auto gp = grad_at(tp);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = grad_at(tm);
    if (!gm.has_value()) return std::unexpected(gm.error());
    if (gp->size() != q || gm->size() != q) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "observed_moment_bread_fd: gradient length mismatch"));
    }
    H.col(k) = (*gp - *gm) / (2.0 * hk);
  }
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::MatrixXd Halpha = K.transpose() * H * K;
  Halpha = 0.5 * (Halpha + Halpha.transpose()).eval();
  return Halpha;
}

post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin,
                        const std::optional<Eigen::MatrixXd>& bread_override) {
  if (blocks.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moments: no moment blocks supplied"));
  }
  if (K.rows() == 0 || K.cols() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moments: empty constraint reparameterization"));
  }
  double N_total = 0.0;
  Eigen::Index total_rows = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& blk = blocks[b];
    if (blk.n_obs <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: non-positive n_obs in block " +
              std::to_string(b)));
    }
    if (blk.jacobian.cols() != K.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: jacobian/K column mismatch in block " +
              std::to_string(b)));
    }
    if (blk.weight.rows() != blk.jacobian.rows() ||
        blk.weight.cols() != blk.jacobian.rows() ||
        blk.gamma.rows() != blk.jacobian.rows() ||
        blk.gamma.cols() != blk.jacobian.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: moment matrix shape mismatch in block " +
              std::to_string(b)));
    }
    N_total += static_cast<double>(blk.n_obs);
    total_rows += blk.jacobian.rows();
  }
  if (!(N_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moments: non-positive total sample size"));
  }

  const Eigen::Index n_alpha = K.cols();
  const int df = static_cast<int>(total_rows - n_alpha);
  if (df < 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "robust_weighted_moments: model has more reduced parameters than moments"));
  }

  Eigen::MatrixXd Dtilde(total_rows, n_alpha);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::Index off = 0;
  for (const auto& blk : blocks) {
    const Eigen::Index mb = blk.jacobian.rows();
    const double sw = std::sqrt(static_cast<double>(blk.n_obs) / N_total);
    Dtilde.block(off, 0, mb, n_alpha) = sw * blk.jacobian * K;
    W.block(off, off, mb, mb) = blk.weight;
    Gamma.block(off, off, mb, mb) = blk.gamma;
    off += mb;
  }

  Eigen::MatrixXd A;
  if (bread_override.has_value()) {
    if (bread_override->rows() != n_alpha || bread_override->cols() != n_alpha) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: bread_override shape does not match the "
          "reduced parameter count"));
    }
    A = *bread_override;
  } else {
    A = Dtilde.transpose() * W * Dtilde;
  }
  A = 0.5 * (A + A.transpose()).eval();
  auto A_inv_or = inverse_sym_pd(A, "robust_weighted_moments bread");
  if (!A_inv_or.has_value()) return std::unexpected(A_inv_or.error());
  const Eigen::MatrixXd& A_inv = *A_inv_or;

  Eigen::MatrixXd B = Dtilde.transpose() * W * Gamma * W * Dtilde;
  B = 0.5 * (B + B.transpose()).eval();

  WeightedRobustResult out;
  Eigen::MatrixXd V_alpha = (A_inv * B * A_inv) / N_total;
  V_alpha = 0.5 * (V_alpha + V_alpha.transpose()).eval();
  out.vcov = K * V_alpha * K.transpose();
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose()).eval();
  out.se.resize(out.vcov.rows());
  const double diag_tol = 1e-12 * std::max<double>(1.0, out.vcov.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < out.se.size(); ++i) {
    const double v = out.vcov(i, i);
    out.se(i) = v >= -diag_tol ? std::sqrt(std::max(0.0, v))
                               : std::numeric_limits<double>::quiet_NaN();
  }

  out.chisq_standard = N_total * fmin;
  out.df = df;
  if (df > 0) {
    Eigen::MatrixXd U = W - W * Dtilde * A_inv * Dtilde.transpose() * W;
    U = 0.5 * (U + U.transpose()).eval();
    auto sqrtG_or = symmetric_sqrt_psd(Gamma, "robust_weighted_moments gamma");
    if (!sqrtG_or.has_value()) return std::unexpected(sqrtG_or.error());
    Eigen::MatrixXd M = (*sqrtG_or) * U * (*sqrtG_or);
    M = 0.5 * (M + M.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moments: U-Gamma eigendecomposition failed"));
    }
    out.eigvals = es.eigenvalues().tail(df);
    for (Eigen::Index i = 0; i < out.eigvals.size(); ++i) {
      if (out.eigvals(i) < 0.0 && out.eigvals(i) > -1e-10) out.eigvals(i) = 0.0;
    }
  } else {
    out.eigvals.resize(0);
  }
  out.satorra_bentler = robust::satorra_bentler(out.chisq_standard, out.df, out.eigvals);
  out.mean_var_adjusted = robust::mean_var_adjusted(out.chisq_standard, out.df, out.eigvals);
  out.scaled_shifted = robust::scaled_shifted(out.chisq_standard, out.df, out.eigvals);
  return out;
}

post_expected<WeightedProfileRMSEAResult>
weighted_moment_profile_rmsea_two_metric(
    const std::vector<WeightedProfileMomentBlock>& blocks,
    const Eigen::MatrixXd& K,
    double fmin,
    const Eigen::MatrixXd& observed_bread,
    std::size_t n_groups,
    double eig_tol) {
  if (blocks.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_two_metric: no moment blocks supplied"));
  }
  if (K.rows() == 0 || K.cols() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_two_metric: empty constraint "
        "reparameterization"));
  }
  if (!std::isfinite(fmin)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_two_metric: non-finite discrepancy"));
  }
  if (!(eig_tol >= 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_two_metric: negative eigentolerance"));
  }

  double N_total = 0.0;
  std::int64_t N_total_i = 0;
  Eigen::Index total_rows = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& blk = blocks[b];
    if (blk.n_obs <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_two_metric: non-positive n_obs in block " +
              std::to_string(b)));
    }
    if (blk.jacobian.cols() != K.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_two_metric: jacobian/K column mismatch in block " +
              std::to_string(b)));
    }
    if (blk.data_metric.rows() != blk.jacobian.rows() ||
        blk.data_metric.cols() != blk.jacobian.rows() ||
        blk.projection_metric.rows() != blk.jacobian.rows() ||
        blk.projection_metric.cols() != blk.jacobian.rows() ||
        blk.gamma.rows() != blk.jacobian.rows() ||
        blk.gamma.cols() != blk.jacobian.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_two_metric: moment matrix shape "
          "mismatch in block " + std::to_string(b)));
    }
    N_total += static_cast<double>(blk.n_obs);
    N_total_i += blk.n_obs;
    total_rows += blk.jacobian.rows();
  }
  if (!(N_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_two_metric: non-positive total sample "
        "size"));
  }

  const Eigen::Index n_alpha = K.cols();
  if (observed_bread.rows() != n_alpha || observed_bread.cols() != n_alpha) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_two_metric: observed_bread shape does not match "
        "the reduced parameter count"));
  }
  const int df = static_cast<int>(total_rows - n_alpha);
  if (df < 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "weighted_moment_profile_rmsea_two_metric: model has more reduced "
        "parameters than moments"));
  }

  Eigen::MatrixXd Dtilde(total_rows, n_alpha);
  Eigen::MatrixXd V0 = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::MatrixXd Wstar = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::Index off = 0;
  for (const auto& blk : blocks) {
    const Eigen::Index mb = blk.jacobian.rows();
    const double sw = std::sqrt(static_cast<double>(blk.n_obs) / N_total);
    Dtilde.block(off, 0, mb, n_alpha) = sw * blk.jacobian * K;
    V0.block(off, off, mb, mb) = blk.data_metric;
    Wstar.block(off, off, mb, mb) = blk.projection_metric;
    Gamma.block(off, off, mb, mb) = blk.gamma;
    off += mb;
  }

  Eigen::MatrixXd B = 0.5 * (observed_bread + observed_bread.transpose()).eval();
  auto B_inv_or =
      inverse_sym_pd(B, "weighted_moment_profile_rmsea_two_metric observed bread");
  if (!B_inv_or.has_value()) return std::unexpected(B_inv_or.error());

  WeightedProfileRMSEAResult out;
  out.profile_hessian =
      V0 - Wstar * Dtilde * (*B_inv_or) * Dtilde.transpose() * Wstar;
  out.profile_hessian =
      0.5 * (out.profile_hessian + out.profile_hessian.transpose()).eval();
  out.gamma = 0.5 * (Gamma + Gamma.transpose()).eval();

  auto spectrum_or = robust::compute_profile_contrast_spectrum(
      out.profile_hessian, out.gamma, eig_tol);
  if (!spectrum_or.has_value()) return std::unexpected(spectrum_or.error());
  out.eigvals = std::move(spectrum_or->eigenvalues);
  out.bias_trace = spectrum_or->trace_CinvS;
  out.bias_trace_sq = spectrum_or->trace_CinvS_sq;
  out.spectrum_size = static_cast<int>(out.eigvals.size());
  out.warnings = std::move(spectrum_or->warnings);

  out.fmin = fmin;
  out.chisq_standard = N_total * fmin;
  out.df = df;
  out.ntotal = N_total_i;
  out.n_groups = std::max<std::size_t>(1, n_groups);
  if (df > 0) {
    const double G = static_cast<double>(out.n_groups);
    const double denom = static_cast<double>(df);
    out.rmsea =
        std::sqrt(std::max((fmin - out.bias_trace / N_total) * G / denom,
                           0.0));
    out.rmsea_df =
        std::sqrt(std::max((fmin - denom / N_total) * G / denom, 0.0));
  }
  return out;
}

post_expected<WeightedProfileRMSEAResult>
weighted_moment_profile_rmsea(const std::vector<WeightedMomentBlock>& blocks,
                              const Eigen::MatrixXd& K,
                              double fmin,
                              const Eigen::MatrixXd& observed_bread,
                              std::size_t n_groups,
                              double eig_tol) {
  std::vector<WeightedProfileMomentBlock> profile_blocks;
  profile_blocks.reserve(blocks.size());
  for (const auto& blk : blocks) {
    profile_blocks.push_back(WeightedProfileMomentBlock{
        .jacobian = blk.jacobian,
        .data_metric = blk.weight,
        .projection_metric = blk.weight,
        .gamma = blk.gamma,
        .n_obs = blk.n_obs});
  }
  return weighted_moment_profile_rmsea_two_metric(
      profile_blocks, K, fmin, observed_bread, n_groups, eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
weighted_moment_profile_rmsea_estimated_weight(
    const std::vector<WeightedEstimatedWeightProfileBlock>& blocks,
    const Eigen::MatrixXd& K,
    double fmin,
    const Eigen::MatrixXd& observed_bread,
    std::size_t n_groups,
    double eig_tol) {
  if (blocks.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_rmsea_estimated_weight: no moment blocks "
        "supplied"));
  }

  std::vector<WeightedProfileMomentBlock> ext_blocks;
  ext_blocks.reserve(blocks.size());
  Eigen::Index u_rows = 0;       // classical u-moment count Σ m_b
  double N_total = 0.0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& blk = blocks[b];
    const Eigen::Index m = blk.jacobian.rows();
    const Eigen::Index q = blk.jacobian.cols();
    if (m == 0 || q == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_estimated_weight: empty jacobian in "
          "block " + std::to_string(b)));
    }
    if (blk.weight_diag.size() != m || blk.residual.size() != m) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_estimated_weight: weight_diag/residual "
          "length does not match jacobian rows in block " + std::to_string(b)));
    }
    if (blk.gamma.rows() != 2 * m || blk.gamma.cols() != 2 * m) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_estimated_weight: gamma must be the "
          "2m × 2m joint NACOV of (u, γ) in block " + std::to_string(b)));
    }
    if (blk.n_obs <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_estimated_weight: non-positive n_obs "
          "in block " + std::to_string(b)));
    }
    if (!(blk.weight_diag.array() > 0.0).all() ||
        !blk.weight_diag.allFinite() || !blk.residual.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_moment_profile_rmsea_estimated_weight: weight_diag must be "
          "positive and finite, residual finite, in block " + std::to_string(b)));
    }

    // Diagonal channels of the value-function Hessian:
    //   W = diag(1/γ), R = diag(r/γ²), S = diag(r²/γ³).
    const Eigen::ArrayXd g = blk.weight_diag.array();
    const Eigen::ArrayXd r = blk.residual.array();
    const Eigen::VectorXd w_diag = (1.0 / g).matrix();
    const Eigen::VectorXd r_diag = (r / (g * g)).matrix();
    const Eigen::VectorXd s_diag = (r * r / (g * g * g)).matrix();

    WeightedProfileMomentBlock ext;
    ext.jacobian.resize(2 * m, q);
    ext.jacobian.topRows(m) = blk.jacobian;
    ext.jacobian.bottomRows(m) = blk.jacobian;

    ext.data_metric = Eigen::MatrixXd::Zero(2 * m, 2 * m);
    ext.data_metric.block(0, 0, m, m).diagonal() = w_diag;
    ext.data_metric.block(0, m, m, m).diagonal() = r_diag;
    ext.data_metric.block(m, 0, m, m).diagonal() = r_diag;
    ext.data_metric.block(m, m, m, m).diagonal() = s_diag;

    ext.projection_metric = Eigen::MatrixXd::Zero(2 * m, 2 * m);
    ext.projection_metric.block(0, 0, m, m).diagonal() = w_diag;
    ext.projection_metric.block(m, m, m, m).diagonal() = r_diag;

    ext.gamma = blk.gamma;
    ext.n_obs = blk.n_obs;
    ext_blocks.push_back(std::move(ext));

    u_rows += m;
    N_total += static_cast<double>(blk.n_obs);
  }

  auto out = weighted_moment_profile_rmsea_two_metric(
      ext_blocks, K, fmin, observed_bread, n_groups, eig_tol);
  if (!out.has_value()) return out;

  // The two-metric core counts df off the DOUBLED extended dimension; the
  // genuine nominal df lives in the u-moment space alone. Restate it and the
  // df-comparator RMSEA, leaving the spectrum-driven `bias_trace`/`rmsea`
  // untouched.
  const Eigen::Index n_alpha = K.cols();
  const int classical_df = static_cast<int>(u_rows - n_alpha);
  out->df = classical_df;
  if (classical_df > 0 && N_total > 0.0) {
    const double G = static_cast<double>(out->n_groups);
    const double denom = static_cast<double>(classical_df);
    out->rmsea = std::sqrt(
        std::max((fmin - out->bias_trace / N_total) * G / denom, 0.0));
    out->rmsea_df =
        std::sqrt(std::max((fmin - denom / N_total) * G / denom, 0.0));
  } else {
    out->rmsea = 0.0;
    out->rmsea_df = 0.0;
  }
  return out;
}

post_expected<WeightedProfileLRTResult>
weighted_moment_profile_lrt(const WeightedProfileRMSEAResult& h1,
                            const WeightedProfileRMSEAResult& h0,
                            double eig_tol) {
  const Eigen::Index q = h1.profile_hessian.rows();
  if (q == 0 || h1.profile_hessian.cols() != q ||
      h0.profile_hessian.rows() != q || h0.profile_hessian.cols() != q ||
      h1.gamma.rows() != q || h1.gamma.cols() != q ||
      h0.gamma.rows() != q || h0.gamma.cols() != q) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: profile Hessians and Gamma matrices "
        "must be square with the same non-zero dimension"));
  }
  if (!(eig_tol >= 0.0) || !std::isfinite(eig_tol)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: eig_tol must be finite and non-negative"));
  }
  if (h1.ntotal <= 0 || h0.ntotal <= 0 || h1.ntotal != h0.ntotal) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: profile fits must have the same positive "
        "total sample size"));
  }
  if (h1.n_groups == 0 || h0.n_groups == 0 || h1.n_groups != h0.n_groups) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: profile fits must have the same positive "
        "group count"));
  }
  const int df_diff = h0.df - h1.df;
  if (df_diff < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: restricted model has smaller nominal df "
        "than unrestricted model"));
  }
  if (!std::isfinite(h1.fmin) || !std::isfinite(h0.fmin)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: non-finite discrepancy"));
  }
  if (!same_matrix_within(h1.gamma, h0.gamma, 1e-8)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_moment_profile_lrt: profile fits use different Gamma "
        "matrices; nested profile LRT requires a common first-stage covariance"));
  }

  WeightedProfileLRTResult out;
  out.profile_hessian =
      h0.profile_hessian - h1.profile_hessian;
  out.profile_hessian =
      0.5 * (out.profile_hessian + out.profile_hessian.transpose()).eval();
  out.gamma = 0.5 * (h0.gamma + h0.gamma.transpose()).eval();
  out.fmin_diff = h0.fmin - h1.fmin;
  out.T_diff = static_cast<double>(h0.ntotal) * out.fmin_diff;
  out.df_diff = df_diff;
  out.ntotal = h0.ntotal;
  out.n_groups = h0.n_groups;
  out.warnings = h1.warnings;
  out.warnings.insert(out.warnings.end(), h0.warnings.begin(),
                      h0.warnings.end());

  const double stat_tol =
      1e-10 * std::max({1.0, std::abs(h0.fmin), std::abs(h1.fmin)}) *
      static_cast<double>(h0.ntotal);
  if (out.T_diff < -stat_tol) {
    out.warnings.emplace_back(
        "weighted_moment_profile_lrt: restricted discrepancy is smaller than "
        "unrestricted discrepancy by " + std::to_string(-out.T_diff));
  }
  const double T_tail = std::max(0.0, out.T_diff);

  auto spectrum_or = robust::compute_profile_contrast_spectrum(
      out.profile_hessian, out.gamma, eig_tol);
  if (!spectrum_or.has_value()) return std::unexpected(spectrum_or.error());
  out.eigvals = std::move(spectrum_or->eigenvalues);
  out.bias_trace = spectrum_or->trace_CinvS;
  out.bias_trace_sq = spectrum_or->trace_CinvS_sq;
  out.spectrum_size = static_cast<int>(out.eigvals.size());
  out.warnings.insert(out.warnings.end(),
                      spectrum_or->warnings.begin(),
                      spectrum_or->warnings.end());

  out.p_unscaled =
      out.df_diff > 0
          ? inference::chi2_pvalue(T_tail, out.df_diff)
          : (T_tail <= stat_tol ? 1.0
                                : std::numeric_limits<double>::quiet_NaN());
  if (out.spectrum_size <= 0) {
    out.scale_c = std::numeric_limits<double>::quiet_NaN();
    out.T_scaled = std::numeric_limits<double>::quiet_NaN();
    out.p_scaled = std::numeric_limits<double>::quiet_NaN();
    out.T_adjusted = std::numeric_limits<double>::quiet_NaN();
    out.adjust_df = std::numeric_limits<double>::quiet_NaN();
    out.p_adjusted = std::numeric_limits<double>::quiet_NaN();
    out.scaled_shifted = robust::ScaledShiftedResult{
        std::numeric_limits<double>::quiet_NaN(),
        0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
    out.p_scaled_shifted = std::numeric_limits<double>::quiet_NaN();
    out.p_mixture = T_tail <= stat_tol
                        ? 1.0
                        : robust::weighted_chisq_upper(out.eigvals, T_tail);
    if (out.df_diff > 0) {
      out.warnings.emplace_back(
          "weighted_moment_profile_lrt: no positive profile contrast "
          "eigenvalues for a positive nominal df difference");
    }
    return out;
  }

  const auto moments = robust::WeightedChiSquareMoments{
      out.spectrum_size, out.bias_trace, out.bias_trace_sq};
  out.scale_c = out.bias_trace / static_cast<double>(out.spectrum_size);
  out.T_scaled = (out.scale_c > 0.0)
                     ? T_tail / out.scale_c
                     : std::numeric_limits<double>::quiet_NaN();
  out.p_scaled = inference::chi2_pvalue(out.T_scaled, out.spectrum_size);
  if (out.bias_trace_sq > 0.0) {
    out.adjust_df = (out.bias_trace * out.bias_trace) / out.bias_trace_sq;
    out.T_adjusted = T_tail * out.adjust_df / out.bias_trace;
    const double cdf =
        inference::noncentral_chisq_cdf(out.T_adjusted, out.adjust_df, 0.0);
    out.p_adjusted = std::isfinite(cdf)
                         ? std::clamp(1.0 - cdf, 0.0, 1.0)
                         : std::numeric_limits<double>::quiet_NaN();
  } else {
    out.adjust_df = std::numeric_limits<double>::quiet_NaN();
    out.T_adjusted = std::numeric_limits<double>::quiet_NaN();
    out.p_adjusted = std::numeric_limits<double>::quiet_NaN();
  }
  out.scaled_shifted = robust::scaled_shifted(T_tail, moments);
  out.p_scaled_shifted =
      inference::chi2_pvalue(out.scaled_shifted.chi2_adj,
                             out.scaled_shifted.df);
  out.p_mixture = robust::weighted_chisq_upper(out.eigvals, T_tail);
  return out;
}

post_expected<WeightedRobustResult>
robust_weighted_moment_ij(const std::vector<WeightedMomentIJBlock>& blocks,
                          const Eigen::MatrixXd& K,
                          double fmin,
                          const Eigen::MatrixXd& observed_bread) {
  if (blocks.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moment_ij: no moment blocks supplied"));
  }
  if (K.rows() == 0 || K.cols() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moment_ij: empty constraint reparameterization"));
  }
  const Eigen::Index n_alpha = K.cols();
  if (observed_bread.rows() != n_alpha || observed_bread.cols() != n_alpha) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moment_ij: observed bread shape does not match the "
        "reduced parameter count"));
  }

  double N_total = 0.0;
  Eigen::Index total_rows = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& blk = blocks[b];
    if (blk.n_obs <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moment_ij: non-positive n_obs in block " +
              std::to_string(b)));
    }
    if (blk.jacobian.cols() != K.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moment_ij: jacobian/K column mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::Index mb = blk.jacobian.rows();
    if (blk.weight.rows() != mb || blk.weight.cols() != mb ||
        blk.moment_influence.rows() != blk.n_obs ||
        blk.moment_influence.cols() != mb) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moment_ij: moment matrix shape mismatch in block " +
              std::to_string(b)));
    }
    if (blk.weight_correction.size() != 0 &&
        (blk.weight_correction.rows() != blk.n_obs ||
         blk.weight_correction.cols() != mb)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "robust_weighted_moment_ij: correction shape mismatch in block " +
              std::to_string(b)));
    }
    N_total += static_cast<double>(blk.n_obs);
    total_rows += mb;
  }
  if (!(N_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_weighted_moment_ij: non-positive total sample size"));
  }

  const int df = static_cast<int>(total_rows - n_alpha);
  if (df < 0) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "robust_weighted_moment_ij: model has more reduced parameters than "
        "moments"));
  }

  Eigen::MatrixXd A = 0.5 * (observed_bread + observed_bread.transpose()).eval();
  auto A_inv_or = inverse_sym_pd(A, "robust_weighted_moment_ij observed bread");
  if (!A_inv_or.has_value()) return std::unexpected(A_inv_or.error());
  const Eigen::MatrixXd& A_inv = *A_inv_or;

  Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(n_alpha, n_alpha);
  for (const auto& blk : blocks) {
    Eigen::MatrixXd V = blk.moment_influence * blk.weight;
    if (blk.weight_correction.size() != 0) V += blk.weight_correction;
    const Eigen::MatrixXd DbK = blk.jacobian * K;
    meat.noalias() += DbK.transpose() * V.transpose() * V * DbK;
  }
  meat = 0.5 * (meat + meat.transpose()).eval();

  WeightedRobustResult out;
  Eigen::MatrixXd V_alpha = (A_inv * meat * A_inv) / (N_total * N_total);
  V_alpha = 0.5 * (V_alpha + V_alpha.transpose()).eval();
  out.vcov = K * V_alpha * K.transpose();
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose()).eval();
  out.se.resize(out.vcov.rows());
  const double diag_tol =
      1e-12 * std::max<double>(1.0, out.vcov.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < out.se.size(); ++i) {
    const double v = out.vcov(i, i);
    out.se(i) = v >= -diag_tol ? std::sqrt(std::max(0.0, v))
                               : std::numeric_limits<double>::quiet_NaN();
  }
  out.chisq_standard = N_total * fmin;
  out.df = df;
  out.eigvals.resize(0);
  return out;
}

post_expected<Eigen::MatrixXd>
ls_information(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::SampleStats& samp,
               const Estimates& est,
               const gmm::Weight& weight) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ls_information: fitted theta length does not match partable"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = make_layout(samp, eval->moments);

  // Sum the per-block LS information Δ_bᵀ W_b Δ_b, each weighted by n_b. The
  // result is the full-θ npar × npar information; `inference::vcov` projects
  // it through any equality constraints.
  const Eigen::Index q = static_cast<Eigen::Index>(pt.n_free());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto Jb = continuous_moment_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = weight_block(weight, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    const double n_b = static_cast<double>(samp.n_obs[b]);
    H.noalias() += n_b * (Jb->transpose() * (*Wb) * (*Jb));
  }
  H = 0.5 * (H + H.transpose()).eval();
  return H;
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     const gmm::Weight& weight,
                     const std::vector<Eigen::MatrixXd>& gamma,
                     robust::Information bread) {
  return robust_continuous_ls_impl(std::move(pt), rep, samp, est,
                                   weight, gamma, bread);
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     const gmm::Weight& weight,
                     const data::RawData& raw,
                     robust::Information bread) {
  return robust_continuous_ls_raw_impl(std::move(pt), rep, samp, est,
                                       weight, raw, bread);
}

post_expected<WeightedProfileRMSEAResult>
continuous_ls_profile_rmsea(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const data::SampleStats& samp,
                            const Estimates& est,
                            const gmm::Weight& weight,
                            const std::vector<Eigen::MatrixXd>& gamma,
                            double eig_tol) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  if (con_or->K().rows() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_profile_rmsea: constraint reparameterization has "
        "incompatible shape"));
  }

  auto blocks = build_continuous_ls_blocks(pt, rep, samp, est, weight, gamma);
  if (!blocks.has_value()) return std::unexpected(blocks.error());

  auto ob = continuous_ls_observed_bread(pt, rep, samp, est, weight,
                                         con_or->K());
  if (!ob.has_value()) return std::unexpected(ob.error());
  auto chisq = robust_ls_standard_chisq(samp, est);
  if (!chisq.has_value()) return std::unexpected(chisq.error());
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return weighted_moment_profile_rmsea(
      *blocks, con_or->K(), *chisq / *n, *ob, samp.S.size(), eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
continuous_ls_profile_rmsea(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const data::SampleStats& samp,
                            const Estimates& est,
                            const gmm::Weight& weight,
                            const data::RawData& raw,
                            double eig_tol) {
  spec::LatentStructure pt_for_layout = pt;
  if (auto e = resolve_fixed_x_from_sample(pt_for_layout, rep, samp);
      !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto ev_or = model::ModelEvaluator::build(pt_for_layout, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = make_layout(samp, eval->moments);
  auto gamma = gamma_blocks_from_raw(raw, samp, layout);
  if (!gamma.has_value()) return std::unexpected(gamma.error());
  return continuous_ls_profile_rmsea(std::move(pt), rep, samp, est, weight,
                                     *gamma, eig_tol);
}

post_expected<WeightedProfileLRTResult>
continuous_ls_profile_lrt(spec::LatentStructure pt_H1,
                          const model::MatrixRep& rep_H1,
                          const data::SampleStats& samp,
                          const Estimates& est_H1,
                          spec::LatentStructure pt_H0,
                          const model::MatrixRep& rep_H0,
                          const Estimates& est_H0,
                          const gmm::Weight& weight,
                          const std::vector<Eigen::MatrixXd>& gamma,
                          double eig_tol) {
  auto h1 = continuous_ls_profile_rmsea(std::move(pt_H1), rep_H1, samp,
                                        est_H1, weight, gamma, eig_tol);
  if (!h1.has_value()) return std::unexpected(h1.error());
  auto h0 = continuous_ls_profile_rmsea(std::move(pt_H0), rep_H0, samp,
                                        est_H0, weight, gamma, eig_tol);
  if (!h0.has_value()) return std::unexpected(h0.error());
  return weighted_moment_profile_lrt(*h1, *h0, eig_tol);
}

post_expected<WeightedProfileLRTResult>
continuous_ls_profile_lrt(spec::LatentStructure pt_H1,
                          const model::MatrixRep& rep_H1,
                          const data::SampleStats& samp,
                          const Estimates& est_H1,
                          spec::LatentStructure pt_H0,
                          const model::MatrixRep& rep_H0,
                          const Estimates& est_H0,
                          const gmm::Weight& weight,
                          const data::RawData& raw,
                          double eig_tol) {
  auto h1 = continuous_ls_profile_rmsea(pt_H1, rep_H1, samp, est_H1, weight,
                                        raw, eig_tol);
  if (!h1.has_value()) return std::unexpected(h1.error());
  auto h0 = continuous_ls_profile_rmsea(std::move(pt_H0), rep_H0, samp,
                                        est_H0, weight, raw, eig_tol);
  if (!h0.has_value()) return std::unexpected(h0.error());
  return weighted_moment_profile_lrt(*h1, *h0, eig_tol);
}

namespace {

post_expected<std::vector<Eigen::MatrixXd>>
ml_covariance_gamma_blocks_from_raw(const data::RawData& raw,
                                    const data::SampleStats& samp) {
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: RawData masks are not supported for complete-data ML"));
  }
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: RawData and SampleStats block counts differ"));
  }
  std::vector<Eigen::MatrixXd> out;
  out.reserve(raw.X.size());
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    if (raw.X[b].rows() != samp.n_obs[b] ||
        raw.X[b].cols() != samp.S[b].rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ml_profile_rmsea: raw-data shape mismatch in block " +
              std::to_string(b)));
    }
    auto G = data::empirical_gamma(raw.X[b]);
    if (!G.has_value()) return std::unexpected(G.error());
    out.push_back(std::move(*G));
  }
  return out;
}

post_expected<std::vector<WeightedProfileMomentBlock>>
build_ml_profile_blocks(const spec::LatentStructure& pt,
                        const model::MatrixRep& rep,
                        const data::SampleStats& samp,
                        const Estimates& est,
                        const std::vector<Eigen::MatrixXd>& gamma) {
  if (est.theta.size() != static_cast<Eigen::Index>(pt.n_free())) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: fitted theta length does not match partable"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (eval->J_mu.size() != 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: mean-structure ML profile Hessian is not wired in "
        "this covariance-only first pass"));
  }
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (gamma.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: gamma block count does not match sample blocks"));
  }

  std::vector<WeightedProfileMomentBlock> blocks;
  blocks.reserve(samp.S.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p = samp.S[b].rows();
    const Eigen::Index pstar = vech_len(p);
    if (eval->J_sigma.rows() < off + pstar) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ml_profile_rmsea: covariance Jacobian row count is too small"));
    }
    if (gamma[b].rows() != pstar || gamma[b].cols() != pstar) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ml_profile_rmsea: gamma dimension mismatch in block " +
              std::to_string(b)));
    }
    auto V0 = normal_theory_moment_metric(
        samp.S[b], ("ml_profile_rmsea sample metric block " +
                    std::to_string(b)).c_str());
    if (!V0.has_value()) return std::unexpected(V0.error());
    auto Wstar = normal_theory_moment_metric(
        eval->moments.sigma[b], ("ml_profile_rmsea fitted metric block " +
                                 std::to_string(b)).c_str());
    if (!Wstar.has_value()) return std::unexpected(Wstar.error());
    blocks.push_back(WeightedProfileMomentBlock{
        .jacobian = eval->J_sigma.middleRows(off, pstar),
        .data_metric = std::move(*V0),
        .projection_metric = std::move(*Wstar),
        .gamma = gamma[b],
        .n_obs = samp.n_obs[b]});
    off += pstar;
  }
  if (off != eval->J_sigma.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: covariance Jacobian row count does not match "
        "sample covariance blocks"));
  }
  return blocks;
}

}  // namespace

post_expected<WeightedProfileRMSEAResult>
ml_profile_rmsea(spec::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const data::SampleStats& samp,
                 const Estimates& est,
                 const std::vector<Eigen::MatrixXd>& gamma,
                 double eig_tol) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  if (con_or->K().rows() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ml_profile_rmsea: constraint reparameterization has incompatible "
        "shape"));
  }
  auto blocks = build_ml_profile_blocks(pt, rep, samp, est, gamma);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  auto info = inference::information_observed_analytic(pt, rep, samp, est);
  if (!info.has_value()) return std::unexpected(info.error());
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  Eigen::MatrixXd bread =
      con_or->K().transpose() * (*info) * con_or->K() / *n;
  bread = 0.5 * (bread + bread.transpose()).eval();
  return weighted_moment_profile_rmsea_two_metric(
      *blocks, con_or->K(), 2.0 * est.fmin, bread, samp.S.size(), eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
ml_profile_rmsea(spec::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const data::SampleStats& samp,
                 const Estimates& est,
                 const data::RawData& raw,
                 double eig_tol) {
  auto gamma = ml_covariance_gamma_blocks_from_raw(raw, samp);
  if (!gamma.has_value()) return std::unexpected(gamma.error());
  return ml_profile_rmsea(std::move(pt), rep, samp, est, *gamma, eig_tol);
}

post_expected<WeightedProfileLRTResult>
ml_profile_lrt(spec::LatentStructure pt_H1,
               const model::MatrixRep& rep_H1,
               const data::SampleStats& samp,
               const Estimates& est_H1,
               spec::LatentStructure pt_H0,
               const model::MatrixRep& rep_H0,
               const Estimates& est_H0,
               const std::vector<Eigen::MatrixXd>& gamma,
               double eig_tol) {
  auto h1 = ml_profile_rmsea(std::move(pt_H1), rep_H1, samp, est_H1,
                             gamma, eig_tol);
  if (!h1.has_value()) return std::unexpected(h1.error());
  auto h0 = ml_profile_rmsea(std::move(pt_H0), rep_H0, samp, est_H0,
                             gamma, eig_tol);
  if (!h0.has_value()) return std::unexpected(h0.error());
  return weighted_moment_profile_lrt(*h1, *h0, eig_tol);
}

post_expected<WeightedProfileLRTResult>
ml_profile_lrt(spec::LatentStructure pt_H1,
               const model::MatrixRep& rep_H1,
               const data::SampleStats& samp,
               const Estimates& est_H1,
               spec::LatentStructure pt_H0,
               const model::MatrixRep& rep_H0,
               const Estimates& est_H0,
               const data::RawData& raw,
               double eig_tol) {
  auto gamma = ml_covariance_gamma_blocks_from_raw(raw, samp);
  if (!gamma.has_value()) return std::unexpected(gamma.error());
  return ml_profile_lrt(std::move(pt_H1), rep_H1, samp, est_H1,
                        std::move(pt_H0), rep_H0, est_H0, *gamma, eig_tol);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_fixed_weight_ij(spec::LatentStructure pt,
                                     const model::MatrixRep& rep,
                                     const data::SampleStats& samp,
                                     const Estimates& est,
                                     const gmm::Weight& weight,
                                     const data::RawData& raw) {
  return robust_continuous_ls_ij_impl(std::move(pt), rep, samp, est, weight,
                                      raw, ContinuousLsIJWeightMode::Fixed);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_gls_ij(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const data::SampleStats& samp,
                            const Estimates& est,
                            const data::RawData& raw) {
  return robust_continuous_ls_ij_impl(
      std::move(pt), rep, samp, est, gmm::Weight{}, raw,
      ContinuousLsIJWeightMode::SampleNormalTheory);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_wls_ij(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const data::SampleStats& samp,
                            const Estimates& est,
                            const data::RawData& raw) {
  return robust_continuous_ls_ij_impl(
      std::move(pt), rep, samp, est, gmm::Weight{}, raw,
      ContinuousLsIJWeightMode::SampleEmpiricalWls);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_dwls_ij(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::SampleStats& samp,
                             const Estimates& est,
                             const data::RawData& raw) {
  return robust_continuous_ls_ij_impl(
      std::move(pt), rep, samp, est, gmm::Weight{}, raw,
      ContinuousLsIJWeightMode::SampleEmpiricalDwls);
}

post_expected<WeightedRobustResult>
robust_continuous_ls_dls_ij(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const data::SampleStats& samp,
                            const Estimates& est,
                            const data::RawData& raw,
                            frontier::DlsWeightOptions opts) {
  return robust_continuous_ls_ij_impl(
      std::move(pt), rep, samp, est, gmm::Weight{}, raw,
      ContinuousLsIJWeightMode::SampleDls, opts);
}

post_expected<robust::ParamSpaceSandwich>
weighted_param_space_sandwich(const std::vector<WeightedMomentBlock>& blocks) {
  if (blocks.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "weighted_param_space_sandwich: no moment blocks supplied"));
  }
  const Eigen::Index q = blocks[0].jacobian.cols();
  double N_total = 0.0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& blk = blocks[b];
    if (blk.n_obs <= 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_param_space_sandwich: non-positive n_obs in block " +
              std::to_string(b)));
    }
    if (blk.jacobian.cols() != q ||
        blk.weight.rows() != blk.jacobian.rows() ||
        blk.weight.cols() != blk.jacobian.rows() ||
        blk.gamma.rows() != blk.jacobian.rows() ||
        blk.gamma.cols() != blk.jacobian.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "weighted_param_space_sandwich: moment matrix shape mismatch in block " +
              std::to_string(b)));
    }
    N_total += static_cast<double>(blk.n_obs);
  }

  Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(q, q);
  Eigen::MatrixXd B1 = Eigen::MatrixXd::Zero(q, q);
  for (const auto& blk : blocks) {
    const double w_b = static_cast<double>(blk.n_obs) / N_total;
    const Eigen::MatrixXd WD = blk.weight * blk.jacobian;
    A1.noalias() += w_b * (blk.jacobian.transpose() * WD);
    B1.noalias() += w_b * (WD.transpose() * blk.gamma * WD);
  }
  A1 = 0.5 * (A1 + A1.transpose()).eval();
  B1 = 0.5 * (B1 + B1.transpose()).eval();
  return robust::ParamSpaceSandwich{std::move(A1), std::move(B1),
                                    Eigen::MatrixXd(), q};
}

post_expected<robust::ParamSpaceSandwich>
continuous_ls_param_space_sandwich(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::SampleStats& samp,
                                   const Estimates& est,
                                   const gmm::Weight& weight,
                                   const std::vector<Eigen::MatrixXd>& gamma) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto blocks = build_continuous_ls_blocks(pt, rep, samp, est, weight, gamma);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  return weighted_param_space_sandwich(*blocks);
}

post_expected<robust::ParamSpaceSandwich>
continuous_ls_param_space_sandwich(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::SampleStats& samp,
                                   const Estimates& est,
                                   const gmm::Weight& weight,
                                   const data::RawData& raw) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = make_layout(samp, eval->moments);
  auto gamma = gamma_blocks_from_raw(raw, samp, layout);
  if (!gamma.has_value()) return std::unexpected(gamma.error());
  auto blocks = build_continuous_ls_blocks(pt, rep, samp, est, weight, *gamma);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  return weighted_param_space_sandwich(*blocks);
}

post_expected<robust::ParamSpaceSandwich>
continuous_ls_param_space_sandwich(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::SampleStats& samp,
                                   const Estimates& est,
                                   const gmm::Weight& weight,
                                   robust::WeightMoments nt_moments) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  if (auto v = validate_moment_shapes(samp, eval->moments); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (nt_moments == robust::WeightMoments::Pairwise) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_ls_param_space_sandwich: Pairwise Γ_NT moments are not "
        "supported"));
  }
  const auto layout = make_layout(samp, eval->moments);
  // Γ_NT in the [μ ; vech σ] moment layout: under normality the mean and
  // covariance blocks are asymptotically independent, with Var(√n·x̄) = M and
  // Var(√n·vech S) = gamma_nt(M).
  std::vector<Eigen::MatrixXd> gamma;
  gamma.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd& M =
        nt_moments == robust::WeightMoments::Structured ? eval->moments.sigma[b]
                                                        : samp.S[b];
    auto G_cov = data::gamma_nt(M);
    if (!G_cov.has_value()) return std::unexpected(G_cov.error());
    if (!layout.has_means) {
      gamma.push_back(std::move(*G_cov));
      continue;
    }
    const Eigen::Index p = M.rows();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(layout.block_rows[b],
                                              layout.block_rows[b]);
    G.topLeftCorner(p, p) = M;
    G.bottomRightCorner(G_cov->rows(), G_cov->cols()) = *G_cov;
    gamma.push_back(std::move(G));
  }
  auto blocks = build_continuous_ls_blocks(pt, rep, samp, est, weight, gamma);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  return weighted_param_space_sandwich(*blocks);
}

}  // namespace magmaan::estimate
