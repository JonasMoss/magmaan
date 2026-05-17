#include "magmaan/estimate/dls_weight.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"

namespace magmaan::estimate {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

}  // namespace

fit_expected<gmm::Weight>
dls_weight(const model::ModelEvaluator& ev, const data::SampleStats& samp,
           const data::RawData& raw, const Eigen::VectorXd& theta0,
           DlsWeightOptions opts) {
  if (!(opts.a >= 0.0 && opts.a <= 1.0)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "dls_weight: mixing scalar a must be in [0, 1]"));
  }

  // Evaluate at theta0 once — only to fix the moment layout (block
  // dimensions, whether the model carries mean structure).
  auto eval0 = ev.evaluate(theta0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "dls_weight: theta0: " + eval0.error().detail));
  }
  const model::ImpliedMoments& m = eval0->moments;

  if (samp.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "dls_weight: SampleStats and model block counts differ"));
  }
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "dls_weight: RawData and SampleStats block counts differ"));
  }
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "dls_weight: the ADF weight needs complete raw data (no missingness)"));
  }
  if (samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "dls_weight: n_obs block count does not match sample covariances"));
  }

  // Mean structure is present only when both the model and the sample carry
  // means — the same rule `gmm::residuals` uses for the moment layout.
  bool has_means = false;
  for (std::size_t b = 0; b < m.sigma.size(); ++b) {
    if (b < m.mu.size() && m.mu[b].size() > 0 && b < samp.mean.size() &&
        samp.mean[b].size() > 0) {
      has_means = true;
    }
  }

  gmm::Weight W;
  W.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::MatrixXd& X = raw.X[b];
    const Eigen::Index p = S.rows();
    if (S.rows() != S.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "dls_weight: block " + std::to_string(b) +
          " sample covariance is not square"));
    }
    if (X.cols() != p || X.rows() != samp.n_obs[b]) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "dls_weight: block " + std::to_string(b) +
          " raw data shape does not match the sample covariance / n_obs"));
    }

    // Sinv — the normal-theory weight on the mean block.
    Eigen::LLT<Eigen::MatrixXd> s_llt(S);
    if (s_llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSample,
          "dls_weight: block " + std::to_string(b) +
          " sample covariance is not positive definite"));
    }
    const Eigen::MatrixXd Sinv = s_llt.solve(Eigen::MatrixXd::Identity(p, p));

    // Γ_NT (from S) and Γ_ADF (from raw data) over vech(cov); mix them.
    auto g_nt = data::gamma_nt(S);
    if (!g_nt.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "dls_weight: block " + std::to_string(b) + ": " +
          g_nt.error().detail));
    }
    auto g_adf = data::empirical_gamma(X);
    if (!g_adf.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "dls_weight: block " + std::to_string(b) + ": " +
          g_adf.error().detail));
    }
    const Eigen::MatrixXd gamma_dls =
        (1.0 - opts.a) * (*g_nt) + opts.a * (*g_adf);

    // W_DLS,cov = Γ_DLS⁻¹. At a = 0 this equals gmm::normal_theory_weight's
    // covariance block (its 0.5 scaling cancels the symmetric-basis factor 2).
    Eigen::LLT<Eigen::MatrixXd> g_llt(gamma_dls);
    if (g_llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "dls_weight: block " + std::to_string(b) +
          " mixed Γ is not positive definite (the raw data may have too few "
          "rows for the ADF weight)"));
    }
    const Eigen::Index pstar = gamma_dls.rows();
    const Eigen::MatrixXd Wcov =
        g_llt.solve(Eigen::MatrixXd::Identity(pstar, pstar));

    // Assemble the [mean ; vech(cov)] block. The mean block carries the
    // normal-theory weight 0.5·Sinv regardless of `a` (Browne DLS is
    // covariance-only); the mean/covariance cross-block stays zero.
    const Eigen::Index dim = (has_means ? p : 0) + pstar;
    Eigen::MatrixXd Wb = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::Index off = 0;
    if (has_means) {
      Wb.block(0, 0, p, p) = 0.5 * Sinv;
      off = p;
    }
    Wb.block(off, off, pstar, pstar) = Wcov;
    W.push_back(std::move(Wb));
  }
  return W;
}

}  // namespace magmaan::estimate
