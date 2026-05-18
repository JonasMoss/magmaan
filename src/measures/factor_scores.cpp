#include "magmaan/measures/factor_scores.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::measures {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

post_expected<FactorScores>
factor_scores(spec::LatentStructure pt, const model::MatrixRep& rep,
              const data::RawData& raw, const Estimates& est,
              FactorScoreMethod method) {
  // Factor scores are per-observation — complete data only.
  for (const auto& m : raw.mask) {
    if (m.size() != 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "factor_scores() does not support missing data"));
    }
  }

  // Resolve fixed.x against the moment summary, then build the evaluator —
  // the same prelude as fit_extras() / residuals().
  auto samp_or = data::sample_stats_from_raw(raw);
  if (!samp_or.has_value()) return std::unexpected(samp_or.error());
  const data::SampleStats& samp = *samp_or;

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
  if (raw.X.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "RawData and evaluator have different block counts"));
  }

  auto asm_or = ev.assembled(est.theta);
  if (!asm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.assembled(θ̂) failed: " + asm_or.error().detail));
  }
  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.sigma(θ̂) failed: " + sm_or.error().detail));
  }
  const auto& blocks = asm_or->blocks;
  const auto& sm = *sm_or;

  FactorScores out;
  out.scores.reserve(raw.X.size());
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const model::BlockMatrices& B = blocks[b];
    const Eigen::MatrixXd& X = raw.X[b];
    const Eigen::Index p = B.Lambda.rows();
    const Eigen::Index m = B.Lambda.cols();
    if (X.cols() != p) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "RawData block " + std::to_string(b) +
              " column count does not match the model's observed count"));
    }
    const bool has_means = (sm.mu.size() > b && sm.mu[b].size() == p);

    // C : the (m × p) score-coefficient matrix; obs_center : subtracted from
    // each observed row; lat_offset : added to each resulting score row.
    Eigen::MatrixXd C(m, p);
    Eigen::VectorXd obs_center(p);
    Eigen::VectorXd lat_offset = Eigen::VectorXd::Zero(m);

    if (method == FactorScoreMethod::Regression) {
      // f̂ = E[η] + (AΨAᵀ)·Λᵀ·Σ̂⁻¹·(y − μ̂).  Cov(η,y) = (AΨAᵀ)·Λᵀ = Mid·Λᵀ.
      const Eigen::MatrixXd Sigma =
          0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
      Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "implied Σ for block " + std::to_string(b) +
                " is not positive definite at θ̂"));
      }
      const Eigen::MatrixXd Sinv =
          llt.solve(Eigen::MatrixXd::Identity(p, p));
      C.noalias() = B.Mid * B.Lambda.transpose() * Sinv;
      if (has_means) {
        obs_center = sm.mu[b];
        if (B.Alpha.size() == m) lat_offset.noalias() = B.A * B.Alpha;
      } else {
        obs_center = X.colwise().mean().transpose();
      }
    } else {
      // Bartlett: f̂ = (Λᵀ Θ⁻¹ Λ)⁻¹ Λᵀ Θ⁻¹ (y − ν).
      const Eigen::MatrixXd Theta =
          0.5 * (B.Theta + B.Theta.transpose());
      Eigen::LLT<Eigen::MatrixXd> tllt(Theta);
      if (tllt.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "residual covariance Θ for block " + std::to_string(b) +
                " is not positive definite — Bartlett scores are undefined"));
      }
      const Eigen::MatrixXd ThiL = tllt.solve(B.Lambda);   // Θ⁻¹·Λ  (p × m)
      const Eigen::MatrixXd M = B.Lambda.transpose() * ThiL;  // ΛᵀΘ⁻¹Λ (m×m)
      Eigen::LLT<Eigen::MatrixXd> mllt(M);
      if (mllt.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ΛᵀΘ⁻¹Λ for block " + std::to_string(b) +
                " is singular — Bartlett scores are undefined"));
      }
      C.noalias() = mllt.solve(ThiL.transpose());          // M⁻¹·ΛᵀΘ⁻¹ (m×p)
      if (has_means && B.Nu.size() == p) {
        obs_center = B.Nu;
      } else {
        obs_center = X.colwise().mean().transpose();
      }
    }

    Eigen::MatrixXd s = (X.rowwise() - obs_center.transpose()) * C.transpose();
    s.rowwise() += lat_offset.transpose();
    out.scores.push_back(std::move(s));
  }
  return out;
}

}  // namespace magmaan::measures
