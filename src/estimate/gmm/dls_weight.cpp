#include "magmaan/estimate/gmm/dls_weight.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"

#include "detail_vech.hpp"

namespace magmaan::estimate::frontier {

namespace {

using detail::vech_lower;

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

fit_expected<void>
validate_dls_raw_shapes(const data::SampleStats& samp, const data::RawData& raw,
                        const char* who) {
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": RawData and SampleStats block counts differ"));
  }
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": the ADF weight needs complete raw data "
            "(no missingness)"));
  }
  if (samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": n_obs block count does not match sample "
            "covariances"));
  }
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::MatrixXd& X = raw.X[b];
    const Eigen::Index p = S.rows();
    if (S.rows() != S.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": block " + std::to_string(b) +
              " sample covariance is not square"));
    }
    if (X.cols() != p || X.rows() != samp.n_obs[b]) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": block " + std::to_string(b) +
              " raw data shape does not match the sample covariance / n_obs"));
    }
  }
  return {};
}

Eigen::MatrixXd centered_vech_outer_rows(const Eigen::MatrixXd& X) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::Index pstar = p * (p + 1) / 2;
  const Eigen::VectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - mean.transpose();

  Eigen::MatrixXd D(n, pstar);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    D.row(i) = vech_lower(Eigen::MatrixXd(xi * xi.transpose())).transpose();
  }
  D.rowwise() -= D.colwise().mean();
  return D;
}

struct EbBlockMoments {
  double observed_delta_norm2 = 0.0;
  double estimated_noise_norm2 = 0.0;
};

fit_expected<EbBlockMoments>
eb_block_moments(const Eigen::MatrixXd& S, const Eigen::MatrixXd& X,
                 std::size_t b) {
  auto g_nt = data::gamma_nt(S);
  if (!g_nt.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "empirical_bayes_dls_mixing_scalar: block " + std::to_string(b) +
            ": " + g_nt.error().detail));
  }
  auto g_adf = data::empirical_gamma(X);
  if (!g_adf.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "empirical_bayes_dls_mixing_scalar: block " + std::to_string(b) +
            ": " + g_adf.error().detail));
  }

  const Eigen::MatrixXd D = centered_vech_outer_rows(X);
  const Eigen::Index n = D.rows();
  const Eigen::Index pstar = D.cols();
  const Eigen::MatrixXd delta = *g_adf - *g_nt;

  EbBlockMoments out;
  for (Eigen::Index j = 0; j < pstar; ++j) {
    for (Eigen::Index i = j; i < pstar; ++i) {
      const double d = delta(i, j);
      out.observed_delta_norm2 += d * d;

      const Eigen::ArrayXd zij = D.col(i).array() * D.col(j).array();
      const double mean_z = (*g_adf)(i, j);
      const double var_z =
          (zij - mean_z).square().sum() / static_cast<double>(n);
      out.estimated_noise_norm2 += var_z / static_cast<double>(n);
    }
  }
  return out;
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
  auto shapes = validate_dls_raw_shapes(samp, raw, "dls_weight");
  if (!shapes.has_value()) return std::unexpected(shapes.error());

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

fit_expected<EmpiricalBayesDlsMixingScalar>
empirical_bayes_dls_mixing_scalar(const data::SampleStats& samp,
                                  const data::RawData& raw,
                                  EmpiricalBayesDlsOptions opts) {
  if (!(std::isfinite(opts.min_a) && std::isfinite(opts.max_a) &&
        opts.min_a >= 0.0 && opts.max_a <= 1.0 && opts.min_a <= opts.max_a)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "empirical_bayes_dls_mixing_scalar: min_a/max_a must define a finite "
        "range within [0, 1]"));
  }

  auto shapes = validate_dls_raw_shapes(
      samp, raw, "empirical_bayes_dls_mixing_scalar");
  if (!shapes.has_value()) return std::unexpected(shapes.error());

  EmpiricalBayesDlsMixingScalar out;
  out.block_observed_delta_norm2.reserve(samp.S.size());
  out.block_estimated_noise_norm2.reserve(samp.S.size());

  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto block = eb_block_moments(samp.S[b], raw.X[b], b);
    if (!block.has_value()) return std::unexpected(block.error());
    out.observed_delta_norm2 += block->observed_delta_norm2;
    out.estimated_noise_norm2 += block->estimated_noise_norm2;
    out.block_observed_delta_norm2.push_back(block->observed_delta_norm2);
    out.block_estimated_noise_norm2.push_back(block->estimated_noise_norm2);
  }

  const double raw_signal =
      out.observed_delta_norm2 - out.estimated_noise_norm2;
  out.signal_norm2 = std::max(0.0, raw_signal);
  if (out.observed_delta_norm2 > 0.0) {
    out.a = out.signal_norm2 / out.observed_delta_norm2;
  }
  out.a = std::clamp(out.a, opts.min_a, opts.max_a);
  return out;
}

fit_expected<gmm::Weight>
empirical_bayes_dls_weight(const model::ModelEvaluator& ev,
                           const data::SampleStats& samp,
                           const data::RawData& raw,
                           const Eigen::VectorXd& theta0,
                           EmpiricalBayesDlsOptions opts) {
  auto eb = empirical_bayes_dls_mixing_scalar(samp, raw, opts);
  if (!eb.has_value()) return std::unexpected(eb.error());
  return dls_weight(ev, samp, raw, theta0, DlsWeightOptions{eb->a});
}

}  // namespace magmaan::estimate::frontier
