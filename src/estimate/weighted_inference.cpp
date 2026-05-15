#include "magmaan/estimate/weighted_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/inference.hpp"
#include "magmaan/fit/resolve_fixed_x.hpp"
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"

namespace magmaan::estimate {

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

using fit::detail::vech_index;
using fit::detail::vech_len;
using fit::detail::vech_lower;

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

post_expected<double> total_n(const fit::SampleStats& samp) {
  double n = 0.0;
  for (auto nb : samp.n_obs) n += static_cast<double>(nb);
  if (!(n > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: SampleStats has non-positive total n_obs"));
  }
  return n;
}

bool has_mean_rows(const fit::SampleStats& samp,
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

post_expected<void> validate_moment_shapes(const fit::SampleStats& samp,
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

ContinuousLsLayout make_layout(const fit::SampleStats& samp,
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

Eigen::MatrixXd symmetric_vech_gls_weight(const Eigen::MatrixXd& Sinv) {
  const Eigen::Index p = Sinv.rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(pstar, pstar);
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

post_expected<Eigen::MatrixXd>
gls_weight_block(const fit::SampleStats& samp,
                 const ContinuousLsLayout& layout,
                 std::size_t b) {
  const Eigen::Index p = samp.S[b].rows();
  Eigen::LLT<Eigen::MatrixXd> llt(samp.S[b]);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "robust_continuous_ls GLS: sample covariance is not positive definite in block " +
            std::to_string(b)));
  }
  const Eigen::MatrixXd Sinv = llt.solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(layout.block_rows[b],
                                           layout.block_rows[b]);
  Eigen::Index off = 0;
  if (layout.has_means) {
    W.block(0, 0, p, p) = Sinv;
    off = p;
  }
  W.block(off, off, vech_len(p), vech_len(p)) =
      symmetric_vech_gls_weight(Sinv);
  return W;
}

template <class D>
post_expected<Eigen::MatrixXd>
weight_block(const D&, const fit::SampleStats&,
             const ContinuousLsLayout& layout,
             std::size_t b) {
  return Eigen::MatrixXd::Identity(layout.block_rows[b], layout.block_rows[b]);
}

post_expected<Eigen::MatrixXd>
weight_block(const fit::GLS& gls,
             const fit::SampleStats& samp,
             const ContinuousLsLayout& layout,
             std::size_t b) {
  (void)gls;
  return gls_weight_block(samp, layout, b);
}

post_expected<Eigen::MatrixXd>
weight_block(const fit::WLS& wls,
             const fit::SampleStats&,
             const ContinuousLsLayout& layout,
             std::size_t b) {
  if (wls.weights.size() <= b) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls WLS: missing weight block " +
            std::to_string(b)));
  }
  const auto& W = wls.weights[b];
  if (W.rows() != layout.block_rows[b] || W.cols() != layout.block_rows[b]) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls WLS: weight dimension mismatch in block " +
            std::to_string(b)));
  }
  return W;
}

post_expected<std::vector<Eigen::MatrixXd>>
gamma_blocks_from_raw(const fit::RawData& raw,
                      const fit::SampleStats& samp,
                      const ContinuousLsLayout& layout) {
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: RawData masks are not supported for complete-data LS"));
  }
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: RawData and SampleStats block counts differ"));
  }
  std::vector<Eigen::MatrixXd> out;
  out.reserve(raw.X.size());
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
    Eigen::MatrixXd Gamma = (Z.transpose() * Z) / static_cast<double>(n);
    Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();
    out.push_back(std::move(Gamma));
  }
  return out;
}

post_expected<double>
continuous_ls_chisq(const fit::SampleStats& samp,
                    const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    const fit::Estimates& est,
                    const fit::ULS&) {
  (void)pt;
  (void)rep;
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return 2.0 * *n * est.fmin;
}

post_expected<double>
continuous_ls_chisq(const fit::SampleStats& samp,
                    const spec::LatentStructure&,
                    const model::MatrixRep&,
                    const fit::Estimates& est,
                    const fit::GLS&) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return *n * est.fmin;
}

post_expected<double>
continuous_ls_chisq(const fit::SampleStats& samp,
                    const spec::LatentStructure&,
                    const model::MatrixRep&,
                    const fit::Estimates& est,
                    const fit::WLS&) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return 2.0 * *n * est.fmin;
}

template <class D>
post_expected<WeightedRobustResult>
robust_continuous_ls_impl(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const fit::SampleStats& samp,
                          const fit::Estimates& est,
                          D discrepancy,
                          const std::vector<Eigen::MatrixXd>& gamma) {
  if (auto e = fit::resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
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

  auto con_or = fit::build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  if (con_or->K().rows() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "robust_continuous_ls: constraint reparameterization has incompatible shape"));
  }

  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto Jb = continuous_moment_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = weight_block(discrepancy, samp, layout, b);
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

  auto chisq = continuous_ls_chisq(samp, pt, rep, est, discrepancy);
  if (!chisq.has_value()) return std::unexpected(chisq.error());
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  return robust_weighted_moments(blocks, con_or->K(), *chisq / *n);
}

template <class D>
post_expected<WeightedRobustResult>
robust_continuous_ls_raw_impl(spec::LatentStructure pt,
                              const model::MatrixRep& rep,
                              const fit::SampleStats& samp,
                              const fit::Estimates& est,
                              D discrepancy,
                              const fit::RawData& raw) {
  spec::LatentStructure pt_for_layout = pt;
  if (auto e = fit::resolve_fixed_x_from_sample(pt_for_layout, rep, samp);
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
                                   std::move(discrepancy), *gamma);
}

}  // namespace

post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin) {
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

  Eigen::MatrixXd A = Dtilde.transpose() * W * Dtilde;
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
  out.satorra_bentler = fit::satorra_bentler(out.chisq_standard, out.df, out.eigvals);
  out.mean_var_adjusted = fit::mean_var_adjusted(out.chisq_standard, out.df, out.eigvals);
  out.scaled_shifted = fit::scaled_shifted(out.chisq_standard, out.df, out.eigvals);
  return out;
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::ULS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma) {
  return robust_continuous_ls_impl(std::move(pt), rep, samp, est,
                                   discrepancy, gamma);
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::GLS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma) {
  return robust_continuous_ls_impl(std::move(pt), rep, samp, est,
                                   discrepancy, gamma);
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::WLS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma) {
  return robust_continuous_ls_impl(std::move(pt), rep, samp, est,
                                   std::move(discrepancy), gamma);
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::ULS discrepancy,
                     const fit::RawData& raw) {
  return robust_continuous_ls_raw_impl(std::move(pt), rep, samp, est,
                                       discrepancy, raw);
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::GLS discrepancy,
                     const fit::RawData& raw) {
  return robust_continuous_ls_raw_impl(std::move(pt), rep, samp, est,
                                       discrepancy, raw);
}

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::WLS discrepancy,
                     const fit::RawData& raw) {
  return robust_continuous_ls_raw_impl(std::move(pt), rep, samp, est,
                                       std::move(discrepancy), raw);
}

}  // namespace magmaan::estimate
