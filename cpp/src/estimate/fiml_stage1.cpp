#include "fiml_internal.hpp"

namespace magmaan::estimate::fiml {
namespace internal {



post_expected<SaturatedMoments>
saturated_em_moments_impl(const RawData& raw,
                          const FIMLPack& pack,
                          const FIMLH1& h1,
                          double h_step,
                          SaturatedHessianKind hessian_kind,
                          const char* caller) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": h_step must be > 0"));
  }
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(fit_to_post(ok.error(), caller));
  }

  const FIMLCache& cache = pack.cache;
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), caller));
  }

  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": empty or inconsistent block layout"));
  }

  // Block-stacked η = (μ_1, vech(Σ_1), μ_2, vech(Σ_2), …); column-major
  // lower-triangle vech matches `fiml_saturated_scores_block` and
  // `robust::casewise_contributions(include_means = true)`.
  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = cache.block_p[b];
    q_b[b] = p + detail::vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
  }
  const Eigen::Index Q = off_b.back();

  SaturatedMoments out;
  out.mean.resize(B);
  out.cov.resize(B);
  out.n_obs.resize(B);
  out.warnings = h1.warnings;
  out.solver_recorded = h1.solver_recorded;
  out.solver_options = h1.solver_options;
  out.solver_blocks = h1.solver_blocks;
  out.H = Eigen::MatrixXd::Zero(Q, Q);
  out.raw_gradient = Eigen::VectorXd::Zero(Q);
  out.J = Eigen::MatrixXd::Zero(Q, Q);

  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::VectorXd& mu_b = h1.mu[b];
    const Eigen::MatrixXd& Sigma_b = h1.sigma[b];

    auto scores_or = fiml_saturated_scores_block(raw, b, mu_b, Sigma_b);
    if (!scores_or.has_value()) {
      return std::unexpected(scores_or.error());
    }
    auto H_or = (hessian_kind == SaturatedHessianKind::Analytic)
        ? fiml_saturated_hessian_analytic_block(cache, b, mu_b, Sigma_b)
        : fiml_saturated_hessian_fd_block(raw, b, mu_b, Sigma_b, h_step);
    if (!H_or.has_value()) {
      return std::unexpected(H_or.error());
    }

    // The block helpers report scores as deviance gradients (∂(−2logL)/∂η) and
    // H as the Hessian of mean deviance. Convert to log-likelihood scale:
    //   information  H_b = (n_b / 2) · H_dev_mean
    //   score cov    J_b = (1 / 4) · scoresᵀ scores
    const Eigen::MatrixXd& scores_dev = *scores_or;
    const Eigen::MatrixXd& H_dev_mean = *H_or;
    const double n_b = static_cast<double>(raw.X[b].rows());
    const Eigen::Index q = q_b[b];
    const Eigen::Index off = off_b[b];
    out.H.block(off, off, q, q) = (n_b / 2.0) * H_dev_mean;
    out.raw_gradient.segment(off, q) = 0.5 * scores_dev.colwise().sum().transpose();
    out.J.block(off, off, q, q) =
        0.25 * (scores_dev.transpose() * scores_dev);

    out.mean[b] = mu_b;
    out.cov[b]  = Sigma_b;
    out.n_obs[b] = static_cast<std::int64_t>(raw.X[b].rows());
  }

  out.raw_H = out.H;
  out.raw_hessian_analytic = hessian_kind == SaturatedHessianKind::Analytic;
  auto H_repair = floor_symmetric_post(
      out.H, /*absolute_floor=*/0.0, saturated_info_rel_floor,
      std::string(caller) + ": aggregated saturated information");
  if (!H_repair.has_value()) return std::unexpected(H_repair.error());
  out.information_repaired = H_repair->applied;
  out.information_ridge = H_repair->ridge;
  out.information_min_eigen = H_repair->min_eigen_before;
  if (H_repair->applied) {
    add_unique_warning(out.warnings,
        std::string(caller) +
        ": aggregated saturated information was regularized; min eigenvalue = " +
        fmt_sci(H_repair->min_eigen_before) +
        ", diagonal ridge = " + fmt_sci(H_repair->ridge));
  }

  auto Hinv_or = invert_symmetric(
      out.H, std::string(caller) + ": aggregated saturated information");
  if (!Hinv_or.has_value()) {
    return std::unexpected(Hinv_or.error());
  }
  const Eigen::MatrixXd Hinv = *Hinv_or;
  out.acov = Hinv * out.J * Hinv;
  out.acov = (0.5 * (out.acov + out.acov.transpose())).eval();
  return out;
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence_impl(const RawData& raw,
                                   const FIMLCache& cache,
                                   const FIMLH1& h1,
                                   const SaturatedMoments& sm,
                                   const char* caller) {
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(fit_to_post(ok.error(), caller));
  }
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), caller));
  }

  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B || sm.mean.size() != B ||
      sm.cov.size() != B || sm.n_obs.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": empty or inconsistent block layout"));
  }

  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index n = raw.X[b].rows();
    const Eigen::Index p = cache.block_p[b];
    if (raw.X[b].cols() != p || h1.mu[b].size() != p ||
        h1.sigma[b].rows() != p || h1.sigma[b].cols() != p ||
        sm.mean[b].size() != p || sm.cov[b].rows() != p ||
        sm.cov[b].cols() != p || sm.n_obs[b] != n) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": malformed saturated block " +
              std::to_string(b)));
    }
    q_b[b] = p + detail::vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
    n_total += n;
  }
  const Eigen::Index Q = off_b.back();
  if (sm.H.rows() != Q || sm.H.cols() != Q || sm.acov.rows() != Q ||
      sm.acov.cols() != Q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": saturated information shape mismatch"));
  }

  Eigen::MatrixXd H_for_inv = sm.H;
  auto H_repair = floor_symmetric_post(
      H_for_inv, /*absolute_floor=*/0.0, saturated_info_rel_floor,
      std::string(caller) + ": aggregated saturated information");
  if (!H_repair.has_value()) return std::unexpected(H_repair.error());
  auto Hinv_or = invert_symmetric(H_for_inv,
      std::string(caller) + ": aggregated saturated information");
  if (!Hinv_or.has_value()) return std::unexpected(Hinv_or.error());
  const Eigen::MatrixXd& Hinv = *Hinv_or;

  Eigen::MatrixXd influence = Eigen::MatrixXd::Zero(n_total, Q);
  Eigen::Index row_off = 0;
  for (std::size_t b = 0; b < B; ++b) {
    auto scores_or = fiml_saturated_scores_block(raw, b, h1.mu[b], h1.sigma[b]);
    if (!scores_or.has_value()) return std::unexpected(scores_or.error());

    const Eigen::Index n = raw.X[b].rows();
    const Eigen::Index q = q_b[b];
    const Eigen::Index off = off_b[b];
    if (scores_or->rows() != n || scores_or->cols() != q) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": saturated score shape mismatch in block " +
              std::to_string(b)));
    }

    const Eigen::MatrixXd scores_log = -0.5 * (*scores_or);
    influence.middleRows(row_off, n).noalias() =
        scores_log * Hinv.middleRows(off, q);
    row_off += n;
  }
  if (!influence.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": non-finite influence rows"));
  }
  return influence;
}

post_expected<SaturatedMoments>
saturated_em_moments_from_raw(const RawData& raw,
                              double h_step,
                              SaturatedHessianKind hessian_kind,
                              const char* caller,
                              FIMLH1Options options) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(),
        std::string(caller) + ": fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or, options);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(),
        std::string(caller) + ": H1 EM"));
  }
  return saturated_em_moments_impl(raw, *pack_or, *h1_or, h_step,
                                   hessian_kind, caller);
}





std::string stage1_target_name(Stage1RegularizationTarget target) {
  switch (target) {
    case Stage1RegularizationTarget::Diagonal:
      return "diagonal";
    case Stage1RegularizationTarget::ScaledIdentity:
      return "scaled_identity";
    case Stage1RegularizationTarget::Identity:
      return "identity";
  }
  return "unknown";
}

post_expected<void>
validate_stage1_regularization_options(
    const Stage1RegularizationOptions& options) {
  if (!options.enabled) return {};
  if (std::isfinite(options.intensity) &&
      (options.intensity < 0.0 || options.intensity > 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: fixed intensity must lie in [0, 1]"));
  }
  if (!(options.min_eigenvalue >= 0.0) ||
      !std::isfinite(options.min_eigenvalue)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: min_eigenvalue must be finite and >= 0"));
  }
  if (!(options.jacobian_step > 0.0) ||
      !std::isfinite(options.jacobian_step)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: jacobian_step must be finite and > 0"));
  }
  if (std::isfinite(options.condition_max) &&
      !(options.condition_max >= 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: condition_max must be >= 1"));
  }
  return {};
}

post_expected<Stage1CovMetrics>
stage1_cov_metrics(const Eigen::MatrixXd& S, const std::string& what) {
  if (S.rows() != S.cols() || S.rows() <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        what + " must be a non-empty square matrix"));
  }
  if (!S.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        what + " contains non-finite entries"));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      0.5 * (S + S.transpose()), Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success || es.eigenvalues().size() == 0 ||
      !es.eigenvalues().allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        what + " eigendecomposition failed"));
  }
  Stage1CovMetrics out;
  out.min_eigen = es.eigenvalues().minCoeff();
  out.max_eigen = es.eigenvalues().maxCoeff();
  out.condition = (out.min_eigen > 0.0)
      ? out.max_eigen / out.min_eigen
      : std::numeric_limits<double>::infinity();
  return out;
}

bool stage1_metrics_meet(const Stage1CovMetrics& m,
                         const Stage1RegularizationOptions& options) {
  if (!std::isfinite(m.min_eigen) || !std::isfinite(m.max_eigen) ||
      !(m.min_eigen > 0.0)) {
    return false;
  }
  if (m.min_eigen < options.min_eigenvalue) return false;
  if (std::isfinite(options.condition_max) &&
      !(m.condition <= options.condition_max)) {
    return false;
  }
  return true;
}

post_expected<Eigen::MatrixXd>
stage1_regularization_target(const Eigen::MatrixXd& S,
                             Stage1RegularizationTarget target,
                             const std::string& what) {
  const Eigen::Index p = S.rows();
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(p, p);
  switch (target) {
    case Stage1RegularizationTarget::Diagonal:
      T = S.diagonal().asDiagonal();
      break;
    case Stage1RegularizationTarget::ScaledIdentity: {
      const double scale = S.trace() / static_cast<double>(p);
      if (!(scale > 0.0) || !std::isfinite(scale)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "regularize_saturated_stage1: scaled-identity target for " + what +
            " is not positive"));
      }
      T = scale * Eigen::MatrixXd::Identity(p, p);
      break;
    }
    case Stage1RegularizationTarget::Identity:
      T = Eigen::MatrixXd::Identity(p, p);
      break;
  }
  if (!T.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: non-finite target for " + what));
  }
  auto mt = stage1_cov_metrics(T, "regularize_saturated_stage1 target " + what);
  if (!mt.has_value()) return std::unexpected(mt.error());
  if (!(mt->min_eigen > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: target for " + what +
        " is not positive definite"));
  }
  return T;
}

Eigen::MatrixXd stage1_convex_cov(const Eigen::MatrixXd& S,
                                  const Eigen::MatrixXd& T,
                                  double intensity) {
  if (intensity <= 0.0) return S;
  if (intensity >= 1.0) return T;
  return (1.0 - intensity) * S + intensity * T;
}

post_expected<Stage1CovApply>
stage1_regularize_covariance_matrix(
    const Eigen::MatrixXd& S0,
    const Stage1RegularizationOptions& options,
    const std::string& what) {
  const Eigen::MatrixXd S = 0.5 * (S0 + S0.transpose());
  auto raw = stage1_cov_metrics(S, "regularize_saturated_stage1 " + what);
  if (!raw.has_value()) return std::unexpected(raw.error());

  Stage1CovApply out;
  out.cov = S;
  auto& d = out.diagnostic;
  d.target = stage1_target_name(options.target);
  d.raw_min_eigen = raw->min_eigen;
  d.raw_max_eigen = raw->max_eigen;
  d.raw_condition = raw->condition;

  if (!options.enabled) {
    d.min_eigen = raw->min_eigen;
    d.max_eigen = raw->max_eigen;
    d.condition = raw->condition;
    return out;
  }

  const bool fixed_intensity = std::isfinite(options.intensity);
  double s = fixed_intensity ? options.intensity : 0.0;
  if (!fixed_intensity && !stage1_metrics_meet(*raw, options)) {
    auto target = stage1_regularization_target(S, options.target, what);
    if (!target.has_value()) return std::unexpected(target.error());

    const Eigen::MatrixXd S1 = stage1_convex_cov(S, *target, 1.0);
    auto m1 = stage1_cov_metrics(S1,
        "regularize_saturated_stage1 " + what + " target");
    if (!m1.has_value()) return std::unexpected(m1.error());
    if (!stage1_metrics_meet(*m1, options)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "regularize_saturated_stage1: target cannot satisfy conditioning "
          "constraints for " + what));
    }

    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 80; ++iter) {
      const double mid = 0.5 * (lo + hi);
      const Eigen::MatrixXd Smid = stage1_convex_cov(S, *target, mid);
      auto mmid = stage1_cov_metrics(Smid,
          "regularize_saturated_stage1 " + what + " candidate");
      if (!mmid.has_value()) return std::unexpected(mmid.error());
      if (stage1_metrics_meet(*mmid, options)) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    s = hi;
    out.cov = stage1_convex_cov(S, *target, s);
  } else if (fixed_intensity && s > 0.0) {
    auto target = stage1_regularization_target(S, options.target, what);
    if (!target.has_value()) return std::unexpected(target.error());
    out.cov = stage1_convex_cov(S, *target, s);
  }

  auto fin = stage1_cov_metrics(out.cov,
      "regularize_saturated_stage1 " + what + " result");
  if (!fin.has_value()) return std::unexpected(fin.error());
  if (!stage1_metrics_meet(*fin, options)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: regularized covariance for " + what +
        " is not positive definite or does not meet conditioning constraints"));
  }
  d.min_eigen = fin->min_eigen;
  d.max_eigen = fin->max_eigen;
  d.condition = fin->condition;
  d.intensity = s;
  d.applied = options.enabled && (s > 0.0);
  return out;
}

post_expected<Eigen::MatrixXd>
stage1_covariance_jacobian(const Eigen::MatrixXd& S,
                           const Stage1RegularizationOptions& options,
                           const Stage1RegularizationBlockDiagnostic& diag,
                           const std::string& what) {
  const Eigen::Index p = S.rows();
  const Eigen::Index ps = vech_len(p);
  if (!options.enabled || !diag.applied) {
    return Eigen::MatrixXd::Identity(ps, ps);
  }

  Eigen::MatrixXd J(ps, ps);
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      const Eigen::Index col = vech_index(p, i, j);
      double h = options.jacobian_step *
          std::max(1.0, std::abs(S(i, j)));
      post_expected<Stage1CovApply> plus =
          std::unexpected(make_post_err(PostError::Kind::NumericIssue, ""));
      post_expected<Stage1CovApply> minus =
          std::unexpected(make_post_err(PostError::Kind::NumericIssue, ""));
      for (int shrink = 0; shrink < 8; ++shrink) {
        Eigen::MatrixXd Sp = S;
        Eigen::MatrixXd Sm = S;
        Sp(i, j) += h;
        Sm(i, j) -= h;
        if (i != j) {
          Sp(j, i) += h;
          Sm(j, i) -= h;
        }
        plus = stage1_regularize_covariance_matrix(
            Sp, options, what + " jacobian +");
        minus = stage1_regularize_covariance_matrix(
            Sm, options, what + " jacobian -");
        if (plus.has_value() && minus.has_value()) break;
        h *= 0.25;
      }
      if (!plus.has_value()) return std::unexpected(plus.error());
      if (!minus.has_value()) return std::unexpected(minus.error());
      J.col(col) =
          (vech_lower(plus->cov) - vech_lower(minus->cov)) / (2.0 * h);
    }
  }
  if (!J.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: non-finite covariance Jacobian for " +
        what));
  }
  return J;
}

}  // namespace

post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw, double h_step) {
  return saturated_em_moments_from_raw(raw, h_step,
                                       SaturatedHessianKind::Analytic,
                                       "saturated_em_moments");
}

post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw, FIMLH1Options options,
                     double h_step) {
  return saturated_em_moments_from_raw(raw, h_step,
                                       SaturatedHessianKind::Analytic,
                                       "saturated_em_moments",
                                       options);
}

post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw,
                     const FIMLPack& pack,
                     const FIMLH1& h1) {
  return saturated_em_moments_impl(raw, pack, h1, /*h_step=*/1e-4,
                                   SaturatedHessianKind::Analytic,
                                   "saturated_em_moments");
}

post_expected<Stage1RegularizedMoments>
regularize_saturated_stage1(const SaturatedMoments& sm,
                            Stage1RegularizationOptions options) {
  if (auto ok = validate_stage1_regularization_options(options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  const std::size_t B = sm.cov.size();
  if (sm.mean.size() != B || sm.n_obs.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: inconsistent block counts"));
  }
  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = sm.cov[b].rows();
    if (p <= 0 || sm.cov[b].cols() != p || sm.mean[b].size() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "regularize_saturated_stage1: malformed block " +
          std::to_string(b)));
    }
    q_b[b] = p + vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
  }
  const Eigen::Index Q = off_b.back();
  if (sm.acov.rows() != Q || sm.acov.cols() != Q || !sm.acov.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "regularize_saturated_stage1: saturated ACOV shape mismatch"));
  }

  Stage1RegularizedMoments out;
  out.moments = sm;
  out.block_diagnostics.reserve(B);
  Eigen::MatrixXd D = Eigen::MatrixXd::Identity(Q, Q);
  bool any_applied = false;

  for (std::size_t b = 0; b < B; ++b) {
    const std::string what = "block " + std::to_string(b);
    auto cov_or =
        stage1_regularize_covariance_matrix(sm.cov[b], options, what);
    if (!cov_or.has_value()) return std::unexpected(cov_or.error());
    out.moments.cov[b] = std::move(cov_or->cov);
    out.block_diagnostics.push_back(cov_or->diagnostic);
    any_applied = any_applied || cov_or->diagnostic.applied;

    auto Jcov_or = stage1_covariance_jacobian(
        sm.cov[b], options, cov_or->diagnostic, what);
    if (!Jcov_or.has_value()) return std::unexpected(Jcov_or.error());
    const Eigen::Index p = sm.cov[b].rows();
    const Eigen::Index ps = vech_len(p);
    const Eigen::Index off = off_b[b] + p;
    D.block(off, off, ps, ps) = std::move(*Jcov_or);
  }

  if (any_applied) {
    // Raw likelihood derivatives describe the untransformed endpoint only.
    out.moments.raw_H.resize(0, 0);
    out.moments.raw_hessian_analytic = false;
    out.moments.raw_gradient.resize(0);
    out.moments.acov = D * sm.acov * D.transpose();
    out.moments.acov =
        (0.5 * (out.moments.acov + out.moments.acov.transpose())).eval();
    if (!out.moments.acov.allFinite()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "regularize_saturated_stage1: transformed ACOV is non-finite"));
    }
    add_unique_warning(out.moments.warnings,
        "regularize_saturated_stage1: Stage-1 covariance was regularized; "
        "ACOV was delta-method transformed");
  }
  return out;
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence(const RawData& raw, double h_step) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(),
        "saturated_em_moment_influence: fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(),
        "saturated_em_moment_influence: H1 EM"));
  }
  auto sm_or = saturated_em_moments_impl(
      raw, *pack_or, *h1_or, h_step, SaturatedHessianKind::Analytic,
      "saturated_em_moment_influence");
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  return saturated_em_moment_influence_impl(
      raw, pack_or->cache, *h1_or, *sm_or, "saturated_em_moment_influence");
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence(const RawData& raw,
                              const FIMLPack& pack,
                              const FIMLH1& h1) {
  auto sm_or = saturated_em_moments_impl(
      raw, pack, h1, /*h_step=*/1e-4, SaturatedHessianKind::Analytic,
      "saturated_em_moment_influence");
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  return saturated_em_moment_influence_impl(
      raw, pack.cache, h1, *sm_or, "saturated_em_moment_influence");
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence(const RawData& raw,
                              const FIMLPack& pack,
                              const FIMLH1& h1,
                              const SaturatedMoments& sm) {
  return saturated_em_moment_influence_impl(
      raw, pack.cache, h1, sm, "saturated_em_moment_influence");
}

post_expected<data::MixedOrdinalStats>
mixed_ordinal_stats_hybrid_fiml_from_observed_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& ordered,
    bool full_wls_weight,
    double h_step) {
  auto stats_or =
      data::mixed_ordinal_stats_from_observed_data(X, ordered, full_wls_weight);
  if (!stats_or.has_value()) return std::unexpected(stats_or.error());
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_hybrid_fiml_from_observed_data: h_step must be > 0"));
  }

  data::MixedOrdinalStats stats = std::move(*stats_or);
  stats.gamma_diag_influence.clear();
  stats.gamma_full_influence.clear();
  stats.gamma_diag_influence.reserve(X.size());
  stats.gamma_full_influence.reserve(X.size());

  for (std::size_t b = 0; b < X.size(); ++b) {
    const Eigen::MatrixXd& Xb = X[b];
    const Eigen::Index n = Xb.rows();
    const Eigen::Index p = Xb.cols();
    std::vector<Eigen::Index> cont;
    cont.reserve(static_cast<std::size_t>(p));
    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        cont_pos[static_cast<std::size_t>(j)] =
            static_cast<Eigen::Index>(cont.size());
        cont.push_back(j);
      }
    }
    const Eigen::Index q = static_cast<Eigen::Index>(cont.size());
    if (q == 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: block " +
              std::to_string(b) + " has no continuous variables"));
    }

    std::vector<Eigen::Index> keep_rows;
    keep_rows.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index r = 0; r < n; ++r) {
      bool any = false;
      for (Eigen::Index k = 0; k < q; ++k) {
        any = any || std::isfinite(Xb(r, cont[static_cast<std::size_t>(k)]));
      }
      if (any) keep_rows.push_back(r);
    }
    if (keep_rows.size() < 2) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: block " +
              std::to_string(b) +
              " has fewer than 2 rows with continuous observations"));
    }

    RawData raw_cont;
    const Eigen::Index nf = static_cast<Eigen::Index>(keep_rows.size());
    Eigen::MatrixXd Xc(nf, q);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask(nf, q);
    for (Eigen::Index rr = 0; rr < nf; ++rr) {
      const Eigen::Index src = keep_rows[static_cast<std::size_t>(rr)];
      for (Eigen::Index k = 0; k < q; ++k) {
        const double v = Xb(src, cont[static_cast<std::size_t>(k)]);
        const bool obs = std::isfinite(v);
        Xc(rr, k) = obs ? v : std::numeric_limits<double>::quiet_NaN();
        mask(rr, k) = obs ? 1 : 0;
      }
    }
    raw_cont.X.push_back(std::move(Xc));
    raw_cont.mask.push_back(std::move(mask));

    auto pack_or = fiml_pack(raw_cont);
    if (!pack_or.has_value()) {
      return std::unexpected(fit_to_post(pack_or.error(),
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: fiml_pack"));
    }
    auto h1_or = fiml_h1_moments(raw_cont, *pack_or);
    if (!h1_or.has_value()) {
      return std::unexpected(fit_to_post(h1_or.error(),
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: H1 EM"));
    }
    auto sm_or = saturated_em_moments_impl(
        raw_cont, *pack_or, *h1_or, h_step, SaturatedHessianKind::Analytic,
        "mixed_ordinal_stats_hybrid_fiml_from_observed_data");
    if (!sm_or.has_value()) return std::unexpected(sm_or.error());
    auto infl_or = saturated_em_moment_influence(
        raw_cont, *pack_or, *h1_or, *sm_or);
    if (!infl_or.has_value()) return std::unexpected(infl_or.error());
    if (sm_or->mean.size() != 1 || sm_or->cov.size() != 1 ||
        sm_or->mean[0].size() != q || sm_or->cov[0].rows() != q ||
        sm_or->cov[0].cols() != q) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: malformed "
          "continuous saturated moments"));
    }

    const Eigen::Index qmom = q + detail::vech_len(q);
    if (infl_or->rows() != nf || infl_or->cols() != qmom) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: continuous "
          "influence shape mismatch"));
    }
    Eigen::MatrixXd cont_if = Eigen::MatrixXd::Zero(n, qmom);
    for (Eigen::Index rr = 0; rr < nf; ++rr) {
      const Eigen::Index src = keep_rows[static_cast<std::size_t>(rr)];
      cont_if.row(src) = static_cast<double>(n) * infl_or->row(rr);
    }

    Eigen::MatrixXd R_old = stats.R[b];
    Eigen::MatrixXd IF_old = stats.moment_influence[b];
    Eigen::MatrixXd& R = stats.R[b];
    Eigen::VectorXd& mean = stats.mean[b];
    Eigen::VectorXd& moments = stats.moments[b];
    Eigen::MatrixXd& G = stats.moment_influence[b];
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index mb = moments.size();
    if (G.rows() != n || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: mixed "
          "influence shape mismatch"));
    }

    std::vector<Eigen::Index> mean_col(static_cast<std::size_t>(p), -1);
    std::vector<Eigen::Index> var_col(static_cast<std::size_t>(p), -1);
    Eigen::Index pos = nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (cont_pos[static_cast<std::size_t>(j)] >= 0) mean_col[static_cast<std::size_t>(j)] = pos++;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (cont_pos[static_cast<std::size_t>(j)] >= 0) var_col[static_cast<std::size_t>(j)] = pos++;
    }
    const Eigen::Index assoc_start = pos;

    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
      if (cp < 0) continue;
      const Eigen::Index mc = mean_col[static_cast<std::size_t>(j)];
      const Eigen::Index vc = var_col[static_cast<std::size_t>(j)];
      mean(j) = sm_or->mean[0](cp);
      R(j, j) = sm_or->cov[0](cp, cp);
      moments(mc) = -mean(j);
      moments(vc) = R(j, j);
      G.col(mc) = -cont_if.col(cp);
      G.col(vc) = cont_if.col(q + detail::vech_index(q, cp, cp));
    }

    Eigen::Index assoc = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const Eigen::Index col = assoc_start + assoc;
        const Eigen::Index ci = cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index cj = cont_pos[static_cast<std::size_t>(j)];
        if (ci >= 0 && cj >= 0) {
          R(i, j) = R(j, i) = sm_or->cov[0](ci, cj);
          moments(col) = R(i, j);
          G.col(col) = cont_if.col(q + detail::vech_index(q, ci, cj));
        } else if (ci >= 0 || cj >= 0) {
          const Eigen::Index c = ci >= 0 ? i : j;
          const Eigen::Index cp = ci >= 0 ? ci : cj;
          const double old_var = R_old(c, c);
          const double new_var = R(c, c);
          if (!(old_var > 0.0) || !(new_var > 0.0)) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_hybrid_fiml_from_observed_data: "
                "non-positive continuous variance"));
          }
          const double old_sd = std::sqrt(old_var);
          const double new_sd = std::sqrt(new_var);
          const double rho = R_old(i, j) / old_sd;
          const Eigen::Index vc = var_col[static_cast<std::size_t>(c)];
          const Eigen::VectorXd rho_if =
              (IF_old.col(col) - (rho / (2.0 * old_sd)) * IF_old.col(vc)) /
              old_sd;
          R(i, j) = R(j, i) = rho * new_sd;
          moments(col) = R(i, j);
          G.col(col) = new_sd * rho_if +
              (rho / (2.0 * new_sd)) *
                  cont_if.col(q + detail::vech_index(q, cp, cp));
        }
        ++assoc;
      }
    }

    Eigen::MatrixXd NACOV = (G.transpose() * G) / static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose()).eval();
    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mb, mb);
    for (Eigen::Index k = 0; k < mb; ++k) {
      const double v = NACOV(k, k);
      if (!(v > 0.0) || !std::isfinite(v)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_hybrid_fiml_from_observed_data: non-positive "
            "hybrid NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    Eigen::MatrixXd W_wls;
    if (full_wls_weight) {
      Eigen::LLT<Eigen::MatrixXd> llt(NACOV);
      if (llt.info() == Eigen::Success) {
        W_wls = llt.solve(Eigen::MatrixXd::Identity(mb, mb));
        W_wls = 0.5 * (W_wls + W_wls.transpose()).eval();
      }
    }

    Eigen::MatrixXd gamma_diag_if(n, mb);
    Eigen::MatrixXd gamma_full_if(n, mb * mb);
    const Eigen::VectorXd diag = NACOV.diagonal();
    for (Eigen::Index r = 0; r < n; ++r) {
      const Eigen::VectorXd gr = G.row(r).transpose();
      gamma_diag_if.row(r) = (gr.array().square() - diag.array()).matrix();
      const Eigen::MatrixXd outer = gr * gr.transpose() - NACOV;
      for (Eigen::Index cc = 0; cc < mb; ++cc) {
        for (Eigen::Index rr = 0; rr < mb; ++rr) {
          gamma_full_if(r, rr + cc * mb) = outer(rr, cc);
        }
      }
    }

    stats.NACOV[b] = std::move(NACOV);
    stats.W_dwls[b] = std::move(W_dwls);
    stats.W_wls[b] = std::move(W_wls);
    stats.gamma_diag_influence.push_back(std::move(gamma_diag_if));
    stats.gamma_full_influence.push_back(std::move(gamma_full_if));
  }
  return stats;
}


}  // namespace magmaan::estimate::fiml
