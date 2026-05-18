#include "magmaan/measures/residuals.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::measures {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

post_expected<ResidualMoments>
residuals(spec::LatentStructure pt, const model::MatrixRep& rep,
          const SampleStats& samp, const Estimates& est) {
  // Mirror fit_extras(): resolve fixed.x from the sample, rebuild the
  // evaluator, then subtract the implied moments from the sample moments.
  // Without the fixed.x fill the evaluator sees NA fixed_values as 0 and the
  // implied Σ collapses on path-style models.
  if (auto e = estimate::resolve_fixed_x_from_sample(pt, rep, samp);
      !e.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "resolve_fixed_x_from_sample failed: " + e.error().detail));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " ≠ evaluator n_free " + std::to_string(ev.n_free())));
  }
  if (samp.S.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "SampleStats and evaluator have different block counts"));
  }

  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.sigma(θ̂) failed: " + sm_or.error().detail));
  }
  const auto& sm = *sm_or;

  ResidualMoments out;
  out.cov.reserve(samp.S.size());
  out.mean.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    // Symmetrize both sides (float non-associativity) before subtracting.
    const Eigen::MatrixXd S = 0.5 * (samp.S[b] + samp.S[b].transpose());
    const Eigen::MatrixXd Sigma =
        0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
    out.cov.push_back(S - Sigma);

    // Mean residual only when the model implies a mean for this block and
    // the caller supplied a matching sample mean (mirrors fit_extras's
    // `has_means_b` test).
    const Eigen::Index p = S.rows();
    const bool has_means_b =
        (sm.mu.size() > b && sm.mu[b].size() == p) &&
        (samp.mean.size() > b && samp.mean[b].size() == p);
    if (has_means_b) {
      out.mean.push_back(samp.mean[b] - sm.mu[b]);
    } else {
      out.mean.emplace_back();
    }
  }
  return out;
}

post_expected<StandardizedResiduals>
standardized_residuals(spec::LatentStructure pt, const model::MatrixRep& rep,
                       const SampleStats& samp, const Estimates& est) {
  // Raw moment residuals + the SRMR — reuse the single definitions of each
  // rather than re-deriving Σ̂(θ̂) here.
  auto raw = residuals(pt, rep, samp, est);
  if (!raw.has_value()) return std::unexpected(raw.error());
  auto extras = fit_extras(pt, rep, samp, est);
  if (!extras.has_value()) return std::unexpected(extras.error());

  StandardizedResiduals out;
  out.srmr     = extras->srmr;
  out.cov_raw  = std::move(raw->cov);
  out.mean_raw = std::move(raw->mean);
  out.cov_cor.reserve(out.cov_raw.size());
  out.mean_cor.reserve(out.cov_raw.size());

  for (std::size_t b = 0; b < out.cov_raw.size(); ++b) {
    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::MatrixXd& r = out.cov_raw[b];
    const Eigen::Index p = r.rows();

    // Correlation-metric residual (Bentler standardization by the *sample*
    // SDs) — the same metric the SRMR sums over.
    Eigen::MatrixXd cor = Eigen::MatrixXd::Zero(p, p);
    for (Eigen::Index c = 0; c < p; ++c) {
      cor(c, c) = (S(c, c) != 0.0) ? r(c, c) / S(c, c) : 0.0;
      for (Eigen::Index rr = c + 1; rr < p; ++rr) {
        const double denom = std::sqrt(S(rr, rr) * S(c, c));
        const double v = (denom != 0.0) ? r(rr, c) / denom : 0.0;
        cor(rr, c) = v;
        cor(c, rr) = v;
      }
    }
    out.cov_cor.push_back(std::move(cor));

    // Mean residual in SD units — only when the block carries a mean.
    if (out.mean_raw[b].size() == p && p > 0) {
      Eigen::VectorXd mc(p);
      for (Eigen::Index i = 0; i < p; ++i)
        mc(i) = (S(i, i) != 0.0) ? out.mean_raw[b](i) / std::sqrt(S(i, i))
                                 : 0.0;
      out.mean_cor.push_back(std::move(mc));
    } else {
      out.mean_cor.emplace_back();
    }
  }
  return out;
}

}  // namespace magmaan::measures
