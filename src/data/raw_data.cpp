#include "magmaan/data/raw_data.hpp"

#include <cstddef>
#include <string>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

#include "detail_vech.hpp"

namespace magmaan::data {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

using detail::vech_lower;

// Shared Zc-casewise machinery. Both `empirical_gamma` (vech-only ADF
// fourth-moment matrix) and `empirical_gamma_with_means` (full stacked
// NACOV including the mean–vech cross-block, per Browne 1984 with
// meanstructure) need the same per-row centered moments. Computing them
// once here keeps the casewise pipeline single-sourced — any future change
// to the centering convention propagates to both consumers automatically.
struct CenteredMoments {
  Eigen::MatrixXd Xc;   // (n × p)     row i = x_i − m̄
  Eigen::MatrixXd Dc;   // (n × p*)    row i = vech((x_i − m̄)(x_i − m̄)ᵀ) − vech(S_N)
};

post_expected<CenteredMoments>
build_centered_moments(const Eigen::Ref<const Eigen::MatrixXd>& X,
                       const char* fname) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(fname) + ": need at least 2 observations"));
  }
  if (p == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(fname) + ": data matrix has 0 columns"));
  }
  const Eigen::Index pstar = p * (p + 1) / 2;

  CenteredMoments cm;
  const Eigen::VectorXd mean = X.colwise().mean();
  cm.Xc = X.rowwise() - mean.transpose();

  // For each row build d_i = vech(Xc.row(i) · Xc.row(i)ᵀ). Subtract the
  // sample mean of D rows to match Browne 1984's empirical NACOV — that
  // mean equals vech(S_N) by construction, so the centered Dc carries the
  // moment-deviation rows (1/n)·Σ Dc·Dcᵀ = Γ̂_ADF integrates to.
  cm.Dc.resize(n, pstar);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = cm.Xc.row(i).transpose();
    const Eigen::MatrixXd outer = xi * xi.transpose();
    cm.Dc.row(i) = vech_lower(outer).transpose();
  }
  const Eigen::VectorXd dbar = cm.Dc.colwise().mean();
  cm.Dc.rowwise() -= dbar.transpose();
  return cm;
}

}  // namespace

post_expected<SampleStats>
sample_stats_from_raw(const RawData& raw) {
  if (raw.X.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "sample_stats_from_raw: RawData.X is empty"));
  }
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "sample_stats_from_raw: missingness mask not yet supported (FIML phase)"));
  }

  SampleStats out;
  out.S.reserve(raw.X.size());
  out.mean.reserve(raw.X.size());
  out.n_obs.reserve(raw.X.size());

  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X       = raw.X[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "sample_stats_from_raw: block " + std::to_string(b) +
              " has < 2 rows; covariance undefined"));
    }
    if (p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "sample_stats_from_raw: block " + std::to_string(b) +
              " has 0 columns"));
    }
    const Eigen::VectorXd mean = X.colwise().mean();
    const Eigen::MatrixXd Xc   = X.rowwise() - mean.transpose();
    // N-divisor cov matches lavaan's likelihood = "normal" convention.
    Eigen::MatrixXd S = (Xc.transpose() * Xc) / static_cast<double>(n);
    out.S.push_back(std::move(S));
    out.mean.push_back(std::move(mean));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
  }
  return out;
}

post_expected<Eigen::MatrixXd>
empirical_gamma(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  auto cm = build_centered_moments(X, "empirical_gamma");
  if (!cm) return std::unexpected(cm.error());
  const double inv_n = 1.0 / static_cast<double>(X.rows());
  return Eigen::MatrixXd(cm->Dc.transpose() * cm->Dc * inv_n);
}

post_expected<Eigen::MatrixXd>
empirical_gamma_with_means(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  auto cm = build_centered_moments(X, "empirical_gamma_with_means");
  if (!cm) return std::unexpected(cm.error());
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::Index pstar = p * (p + 1) / 2;
  const double inv_n = 1.0 / static_cast<double>(n);

  // Stacked NACOV per Browne 1984 with meanstructure (matches lavaan's
  // `lav_samplestats_gamma.R:374-407`): rows of the unstacked Z matrix
  // are [y_i ; vech((y_i − ȳ)(y_i − ȳ)ᵀ)]. After colMeans subtraction,
  // Zc = [Xc ; Dc] (Xc is centered y, Dc is centered vech outer product).
  // The four sub-blocks of (1/n)·Zcᵀ·Zc:
  //   top-left     : (1/n)·Xcᵀ·Xc     = S_N (N-divisor sample cov)
  //   top-right    : (1/n)·Xcᵀ·Dc     = empirical third-moment cross-block
  //   bottom-left  : transpose of top-right
  //   bottom-right : (1/n)·Dcᵀ·Dc     = Γ̂_ADF (the existing `empirical_gamma`)
  //
  // Under multivariate normality the cross-block vanishes (third moments
  // = 0 by symmetry); under any skewed distribution it carries the
  // population third moments that lavaan plugs in for proper ADF/WLS.
  Eigen::MatrixXd Gamma(p + pstar, p + pstar);
  Gamma.topLeftCorner(p, p)            = cm->Xc.transpose() * cm->Xc * inv_n;
  Gamma.topRightCorner(p, pstar)       = cm->Xc.transpose() * cm->Dc * inv_n;
  Gamma.bottomLeftCorner(pstar, p)     = Gamma.topRightCorner(p, pstar).transpose();
  Gamma.bottomRightCorner(pstar, pstar) = cm->Dc.transpose() * cm->Dc * inv_n;
  return Gamma;
}

post_expected<Eigen::MatrixXd>
gamma_nt(const Eigen::Ref<const Eigen::MatrixXd>& Sigma) {
  const Eigen::Index p = Sigma.rows();
  if (Sigma.cols() != p) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "gamma_nt: Σ is not square"));
  }
  const Eigen::Index pstar = p * (p + 1) / 2;

  // Build the (i,j) index pair list once so we can address Γ_NT[a, b]
  // directly without recomputing i,j from a flat index.
  struct IJ { Eigen::Index i; Eigen::Index j; };
  std::vector<IJ> idx; idx.reserve(static_cast<std::size_t>(pstar));
  for (Eigen::Index j = 0; j < p; ++j)
    for (Eigen::Index i = j; i < p; ++i)
      idx.push_back({i, j});

  Eigen::MatrixXd Gamma(pstar, pstar);
  for (Eigen::Index a = 0; a < pstar; ++a) {
    const auto& A = idx[static_cast<std::size_t>(a)];
    for (Eigen::Index b = 0; b < pstar; ++b) {
      const auto& B = idx[static_cast<std::size_t>(b)];
      Gamma(a, b) = Sigma(A.i, B.i) * Sigma(A.j, B.j) +
                    Sigma(A.i, B.j) * Sigma(A.j, B.i);
    }
  }
  return Gamma;
}

}  // namespace magmaan::data
