#include "latva/fit/raw_data.hpp"

#include <cstddef>
#include <string>

#include <Eigen/Core>

#include "latva/error.hpp"
#include "latva/expected.hpp"

namespace latva::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// Lower-tri column-major vech pack of a p×p matrix into a p*=p(p+1)/2
// vector. Matches `ModelEvaluator::dsigma_dtheta` and `browne_residual_nt`.
Eigen::VectorXd vech_lower(const Eigen::Ref<const Eigen::MatrixXd>& M) {
  const Eigen::Index p     = M.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::VectorXd out(pstar);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      out(k++) = M(i, j);
    }
  }
  return out;
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
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "empirical_gamma: need at least 2 observations"));
  }
  if (p == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "empirical_gamma: data matrix has 0 columns"));
  }
  const Eigen::Index pstar = p * (p + 1) / 2;

  // Centered data and N-divisor sample cov.
  const Eigen::VectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc   = X.rowwise() - mean.transpose();
  const Eigen::MatrixXd S    = (Xc.transpose() * Xc) / static_cast<double>(n);
  const Eigen::VectorXd s_vech = vech_lower(S);

  // For each row build d_i = vech(x_i x_iᵀ) (centered already, so
  // x_i is Xc.row(i)). Accumulate Γ̂ = (1/n) Σ (d_i - vech(S))(d_i - vech(S))ᵀ.
  Eigen::MatrixXd D(n, pstar);   // row i = d_i
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    const Eigen::MatrixXd outer = xi * xi.transpose();
    D.row(i) = vech_lower(outer).transpose();
  }
  // Subtract sample mean of D rows — note this equals vech(S) by
  // construction, so equivalent to subtracting s_vech. Use the
  // empirical mean of D to keep the formula numerically clean
  // (avoids drift if S has slight asymmetry).
  const Eigen::VectorXd dbar = D.colwise().mean();
  D.rowwise() -= dbar.transpose();
  Eigen::MatrixXd Gamma = (D.transpose() * D) / static_cast<double>(n);
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

}  // namespace latva::fit
