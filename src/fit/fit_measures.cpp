#include "latva/fit/fit_measures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/fit/constraints.hpp"
#include "latva/fit/resolve_fixed_x.hpp"
#include "latva/model/model_evaluator.hpp"

namespace latva::fit {

namespace {

constexpr double two_pi = 6.283185307179586476925286766559;

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// log|A| via the Cholesky factor; assumes `llt` succeeded.
double log_det_from_llt(const Eigen::LLT<Eigen::MatrixXd>& llt) noexcept {
  double s = 0.0;
  const auto L = llt.matrixL();
  for (Eigen::Index i = 0; i < L.rows(); ++i) s += std::log(L(i, i));
  return 2.0 * s;
}

}  // namespace

BaselineFit baseline_chi2(const SampleStats& samp) noexcept {
  BaselineFit out;
  double total = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const auto& S = samp.S[b];
    const auto p = S.rows();
    // log|S| via LLT — falls back to "skip block" if S is non-PD (the
    // optimizer would already have flagged this).
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) continue;
    double log_det_S = 0.0;
    for (Eigen::Index i = 0; i < p; ++i)
      log_det_S += std::log(llt.matrixL()(i, i));
    log_det_S *= 2.0;
    double log_det_diag = 0.0;
    for (Eigen::Index i = 0; i < p; ++i) log_det_diag += std::log(S(i, i));
    const double F_b = log_det_diag - log_det_S;
    total += static_cast<double>(samp.n_obs[b]) * F_b;
    out.df += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
  }
  out.chi2 = total;
  return out;
}

FitMeasures fit_measures(const Inference&   user,
                         const BaselineFit& baseline,
                         const SampleStats& samp) noexcept {
  FitMeasures out;
  const double T_u  = user.chi2;
  const double T_b  = baseline.chi2;
  const double df_u = user.df;
  const double df_b = baseline.df;

  // CFI = 1 − max(0, T_u − df_u) / max(0, T_b − df_b)
  const double num_u = std::max(0.0, T_u - df_u);
  const double num_b = std::max(0.0, T_b - df_b);
  out.cfi = (num_b > 0.0) ? std::max(0.0, 1.0 - num_u / num_b) : 1.0;

  // TLI: (T_b/df_b − T_u/df_u) / (T_b/df_b − 1). Undefined when df_b = 0
  // or df_u = 0; conventional return value is NaN (or 1 for a perfect
  // fit at the saturated case). We return NaN to flag the singular case.
  if (df_b > 0 && df_u > 0) {
    const double ratio_b = T_b / df_b;
    const double ratio_u = T_u / df_u;
    const double denom   = ratio_b - 1.0;
    out.tli = (std::abs(denom) > 0.0) ? (ratio_b - ratio_u) / denom
                                      : std::numeric_limits<double>::quiet_NaN();
  } else {
    out.tli = std::numeric_limits<double>::quiet_NaN();
  }

  // RMSEA = √(max(0, (T_u − df_u) / (df_u · N))) · √G. The √G multi-group
  // correction is lavaan's convention (Steiger; `lav_fit_rmsea`'s `* sqrt(G)`)
  // — with one group it's the textbook formula.
  std::int64_t N_total = 0;
  for (auto n : samp.n_obs) N_total += n;
  if (df_u > 0 && N_total > 0) {
    const double num = std::max(0.0, T_u - df_u);
    const double ng  = static_cast<double>(std::max<std::size_t>(1, samp.S.size()));
    out.rmsea = std::sqrt(num / (df_u * static_cast<double>(N_total))) * std::sqrt(ng);
  } else {
    out.rmsea = 0.0;
  }
  return out;
}

post_expected<FitExtras>
fit_extras(partable::LatentStructure        pt,
           const model::MatrixRep&   rep,
           const SampleStats&        samp,
           const Estimates&          est) {
  // Mirror fit() / ExpectedInfoSE::compute: resolve fixed.x from the sample,
  // then rebuild the evaluator. Without the fixed.x fill, ModelEvaluator
  // would see NA fixed_values as 0 and the implied Σ would collapse on
  // path-style models.
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
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

  // npar = #free parameters after equality merging (lavaan: max(free), which
  // for shared-label invariance models equals the post-merge count).
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const int npar = static_cast<int>(con_or->n_alpha);

  FitExtras out;
  out.npar = npar;

  std::int64_t N_total = 0;
  for (auto n : samp.n_obs) N_total += n;
  out.ntotal = N_total;

  // Fixed.x exogenous observed variables — lavaan reports `logl` (and hence
  // AIC/BIC) *conditional* on these, i.e. with the saturated marginal logl of
  // the fixed.x block subtracted from both `logl` and `unrestricted_logl`.
  // Identify them via the name-free partable mirror: `exo == 1` rows name the
  // fixed.x variables (`lhs_var` / `rhs_var`); `ov_pos[var]` is the column in
  // the observed ordering (= the row/col of `samp.S[b]`, same for every block).
  std::vector<Eigen::Index> exo_idx;
  {
    std::unordered_set<std::int32_t> exo_vars;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.exo[i] != 1) continue;
      if (i < pt.lhs_var.size() && pt.lhs_var[i] >= 0) exo_vars.insert(pt.lhs_var[i]);
      if (i < pt.rhs_var.size() && pt.rhs_var[i] >= 0) exo_vars.insert(pt.rhs_var[i]);
    }
    std::unordered_set<Eigen::Index> seen;
    for (std::int32_t v : exo_vars) {
      if (v < 0 || static_cast<std::size_t>(v) >= pt.ov_pos.size()) continue;
      const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
      if (pos < 0) continue;
      const Eigen::Index idx = static_cast<Eigen::Index>(pos);
      if (seen.insert(idx).second) exo_idx.push_back(idx);
    }
    std::sort(exo_idx.begin(), exo_idx.end());
  }

  double logl       = 0.0;   // H0 model
  double logl_unres = 0.0;   // saturated (h1)
  double srmr_acc   = 0.0;   // Σ_b (n_b/N)·srmr_b

  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd S = 0.5 * (samp.S[b] + samp.S[b].transpose());
    const Eigen::Index p = S.rows();
    if (p == 0) continue;
    const double n_b = static_cast<double>(samp.n_obs[b]);

    // Implied Σ̂_b — symmetrize (float non-associativity), Cholesky → PD
    // check + log|Σ̂| + Σ̂⁻¹.
    const Eigen::MatrixXd Sigma =
        0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
    Eigen::LLT<Eigen::MatrixXd> llt_sig(Sigma);
    if (llt_sig.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "implied Σ for block " + std::to_string(b) +
              " is not positive definite at θ̂"));
    }
    const double log_det_sigma = log_det_from_llt(llt_sig);
    const Eigen::MatrixXd Sigma_inv =
        llt_sig.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::LLT<Eigen::MatrixXd> llt_S(S);
    if (llt_S.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "sample S for block " + std::to_string(b) +
              " is not positive definite"));
    }
    const double log_det_S = log_det_from_llt(llt_S);

    // tr(S · Σ̂⁻¹) — both symmetric, so the trace is the elementwise dot.
    const double tr_S_Sinv = S.cwiseProduct(Sigma_inv).sum();

    // Mean structure for this block? (model has ~1 rows ⇒ μ̂ populated, and
    // the caller supplied a sample mean of matching size).
    const bool has_means_b =
        (sm.mu.size() > b && sm.mu[b].size() == p) &&
        (samp.mean.size() > b && samp.mean[b].size() == p);
    double mahal = 0.0;  // (m̄−μ̂)ᵀ Σ̂⁻¹ (m̄−μ̂)
    Eigen::VectorXd mean_res;  // (m̄−μ̂)/√(s_ii) — for SRMR's mean term
    if (has_means_b) {
      const Eigen::VectorXd d = samp.mean[b] - sm.mu[b];
      mahal = d.dot(Sigma_inv * d);
      mean_res.resize(p);
      for (Eigen::Index i = 0; i < p; ++i)
        mean_res(i) = d(i) / std::sqrt(S(i, i));
    }

    // log-likelihoods (lavaan `likelihood = "normal"` ⇒ N divisor).
    logl       += -0.5 * n_b * (static_cast<double>(p) * std::log(two_pi)
                                + log_det_sigma + tr_S_Sinv + mahal);
    logl_unres += -0.5 * n_b * (static_cast<double>(p) * std::log(two_pi)
                                + log_det_S + static_cast<double>(p));

    // Conditional-on-fixed.x adjustment: subtract this block's fixed.x
    // exogenous sub-block's saturated marginal logl from both. (The exo
    // means are always saturated under fixed.x, so their mean term is 0.)
    if (!exo_idx.empty() && exo_idx.back() < p) {
      const Eigen::Index px = static_cast<Eigen::Index>(exo_idx.size());
      Eigen::MatrixXd Sxx(px, px);
      for (Eigen::Index r = 0; r < px; ++r)
        for (Eigen::Index c = 0; c < px; ++c)
          Sxx(r, c) = S(exo_idx[static_cast<std::size_t>(r)],
                        exo_idx[static_cast<std::size_t>(c)]);
      Eigen::LLT<Eigen::MatrixXd> llt_xx(Sxx);
      if (llt_xx.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "fixed.x exogenous sub-block for block " + std::to_string(b) +
                " is not positive definite"));
      }
      const double marg = -0.5 * n_b *
          (static_cast<double>(px) * std::log(two_pi)
           + log_det_from_llt(llt_xx) + static_cast<double>(px));
      logl       -= marg;
      logl_unres -= marg;
    }

    // SRMR (Bentler type): standardize the residual S − Σ̂ by the *sample*
    // SDs. vech-sum over the lower triangle including the diagonal; pstar
    // = p(p+1)/2 (+ p mean residuals when mean structure).
    double sum_sq = 0.0;
    for (Eigen::Index c = 0; c < p; ++c) {
      const double dc = (S(c, c) - Sigma(c, c)) / S(c, c);  // diagonal
      sum_sq += dc * dc;
      for (Eigen::Index r = c + 1; r < p; ++r) {
        const double rij =
            (S(r, c) - Sigma(r, c)) / std::sqrt(S(r, r) * S(c, c));
        sum_sq += rij * rij;
      }
    }
    double pstar = static_cast<double>(p) * static_cast<double>(p + 1) / 2.0;
    if (has_means_b) {
      sum_sq += mean_res.squaredNorm();
      pstar  += static_cast<double>(p);
    }
    const double srmr_b = (pstar > 0.0) ? std::sqrt(sum_sq / pstar) : 0.0;
    if (N_total > 0) srmr_acc += (n_b / static_cast<double>(N_total)) * srmr_b;
  }

  out.logl              = logl;
  out.unrestricted_logl = logl_unres;
  out.srmr              = srmr_acc;
  out.aic  = -2.0 * logl + 2.0 * static_cast<double>(npar);
  out.bic  = (N_total > 0)
                 ? -2.0 * logl + static_cast<double>(npar)
                       * std::log(static_cast<double>(N_total))
                 : std::numeric_limits<double>::quiet_NaN();
  out.bic2 = (N_total > 0)
                 ? -2.0 * logl + static_cast<double>(npar)
                       * std::log((static_cast<double>(N_total) + 2.0) / 24.0)
                 : std::numeric_limits<double>::quiet_NaN();
  return out;
}

}  // namespace latva::fit
