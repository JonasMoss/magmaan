#include "magmaan/robust/lr_test_satorra.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/inference/inference.hpp"        // chi2_pvalue, noncentral_chisq_cdf
#include "magmaan/robust/restriction.hpp"      // restriction_alpha_from_K
#include "magmaan/robust/weighted_chisq.hpp"   // imhof_upper
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"                // vech_len

namespace magmaan::nt::robust {

using infer::chi2_pvalue;
using infer::noncentral_chisq_cdf;

using estimate::EqConstraints;


namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

LRSatorra2000Result degenerate_result(double T_diff) {
  LRSatorra2000Result out;
  out.T_diff      = T_diff;
  out.df_diff     = 0;
  out.p_unscaled  = 1.0;
  out.eigenvalues = Eigen::VectorXd::Zero(0);
  out.scale_c     = 1.0;
  out.T_scaled    = T_diff;
  out.p_scaled    = 1.0;
  out.adjust_d0   = 0.0;
  out.T_adjusted  = T_diff;
  out.p_adjusted  = 1.0;
  out.p_mixture   = 1.0;
  if (std::abs(T_diff) > 1e-8) {
    out.warnings.emplace_back(
        "lr_test_satorra2000: df_diff = 0 but T_diff = " +
        std::to_string(T_diff) + " ≠ 0 — H0 and H1 should give identical "
        "test statistics.  Result is degenerate.");
  }
  return out;
}

}  // namespace

post_expected<LRSatorra2000Result>
lr_test_satorra2000(double                  T_diff,
                    const SatorraDiffResult& sd) {
  LRSatorra2000Result out;
  out.T_diff      = T_diff;
  out.df_diff     = static_cast<int>(sd.eigenvalues.size());
  out.eigenvalues = sd.eigenvalues;
  out.warnings    = sd.warnings;

  if (out.df_diff <= 0) {
    return degenerate_result(T_diff);
  }
  const double m = static_cast<double>(out.df_diff);

  // Unscaled tail.
  out.p_unscaled = chi2_pvalue(T_diff, out.df_diff);

  // SB-style scale: ĉ = (Σ λ) / m.  Floor at a tiny positive number so a
  // pathological all-zero spectrum collapses cleanly to the unscaled tail.
  out.scale_c = sd.trace_CinvS / m;
  if (out.scale_c <= 0.0) {
    out.warnings.emplace_back(
        "lr_test_satorra2000: tr(C⁻¹S) = " +
        std::to_string(sd.trace_CinvS) +
        " ≤ 0 — Satorra-Bentler scale factor floored at 1.");
    out.scale_c  = 1.0;
    out.T_scaled = T_diff;
  } else {
    out.T_scaled = T_diff / out.scale_c;
  }
  out.p_scaled = chi2_pvalue(out.T_scaled, out.df_diff);

  // Mean-and-variance adjustment: d̂₀ = (Σλ)² / (Σλ²); T_adj = T · d̂₀ / Σλ.
  if (sd.trace_CinvS_sq > 0.0) {
    out.adjust_d0  = (sd.trace_CinvS * sd.trace_CinvS) / sd.trace_CinvS_sq;
    out.T_adjusted = T_diff * out.adjust_d0 / sd.trace_CinvS;
    // χ²(d̂₀) tail at a real-valued df: use the noncentral CDF with ncp = 0.
    const double cdf = noncentral_chisq_cdf(out.T_adjusted, out.adjust_d0, 0.0);
    out.p_adjusted   = 1.0 - cdf;
  } else {
    out.adjust_d0  = m;
    out.T_adjusted = T_diff;
    out.p_adjusted = out.p_unscaled;
  }

  // Exact mixture tail.
  out.p_mixture = imhof_upper(sd.eigenvalues, T_diff);
  return out;
}

post_expected<LRSatorra2000Result>
lr_test_satorra2000_from_data(
    const spec::LatentStructure&     pt,
    const model::MatrixRep&              rep,
    const Eigen::VectorXd&               theta_H1_full,
    const EqConstraints&                 K_H1,
    const EqConstraints&                 K_H0,
    const std::vector<Eigen::MatrixXd>&  X_per_group,
    const std::vector<Eigen::VectorXd>&  mean_per_group,
    const std::vector<std::int32_t>&     n_per_group,
    const std::vector<double>&           weight_per_group,
    double                               T_H0,
    double                               T_H1,
    int                                  df_H0,
    int                                  df_H1,
    GammaSource                          gamma) {
  // ── 1. Derive A_α (the restriction in H1's α-space) ───────────────────────
  auto restr_or = restriction_alpha_from_K(K_H1, K_H0);
  if (!restr_or.has_value()) {
    return std::unexpected(restr_or.error());
  }
  const Eigen::MatrixXd& A_alpha = restr_or->A;

  // The Satorra-2000 df_diff must equal df_H0 − df_H1; if the caller's df
  // accounting and the K-matrix-derived restriction rank disagree, surface
  // it.
  const int df_diff_from_K = static_cast<int>(A_alpha.rows());
  const int df_diff_from_T = df_H0 - df_H1;
  if (df_diff_from_K != df_diff_from_T) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: df_diff mismatch — "
        "restriction_alpha_from_K returned m = " +
        std::to_string(df_diff_from_K) + " but df_H0 − df_H1 = " +
        std::to_string(df_diff_from_T) + " (check that both fits use the "
        "same lavaanified partable and that the chi²/df pair is consistent)."));
  }

  // ── 2. Evaluate Π and Σ at θ̂_H1, per group ─────────────────────────────
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: ModelEvaluator::build failed: " +
        ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  auto im_or = ev.sigma(theta_H1_full);
  if (!im_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: sigma(θ̂_H1) failed: " +
        im_or.error().detail));
  }
  auto dsig_or = ev.dsigma_dtheta(theta_H1_full);
  if (!dsig_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: dsigma_dtheta(θ̂_H1) failed: " +
        dsig_or.error().detail));
  }
  const auto&             im     = *im_or;
  const Eigen::MatrixXd&  Pi_th  = *dsig_or;    // stacked p*_all × npar
  const std::size_t       G      = im.sigma.size();
  if (G == 0 || G != X_per_group.size() || G != mean_per_group.size() ||
      G != n_per_group.size() || G != weight_per_group.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: per-group vectors have mismatched "
        "sizes (expected " + std::to_string(G) + " blocks)."));
  }
  if (Pi_th.cols() != theta_H1_full.size() ||
      Pi_th.cols() != K_H1.Kmat.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: dsigma_dtheta column count " +
        std::to_string(Pi_th.cols()) +
        " disagrees with K_H1.npar " + std::to_string(K_H1.Kmat.rows()) +
        " or theta size " + std::to_string(theta_H1_full.size())));
  }

  // ── 3. Build per-group SatorraGroup ─────────────────────────────────────
  std::vector<SatorraGroup> groups;
  groups.reserve(G);
  Eigen::Index row_off = 0;
  for (std::size_t g = 0; g < G; ++g) {
    const Eigen::Index p     = im.sigma[g].rows();
    const Eigen::Index pstar = detail::vech_len(p);
    if (row_off + pstar > Pi_th.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "lr_test_satorra2000_from_data: Π stacked layout overruns at group "
          + std::to_string(g)));
    }
    // Π_g · K_H1  (p*_g × r1)
    Eigen::MatrixXd Pi_g       = Pi_th.middleRows(row_off, pstar);
    Eigen::MatrixXd Pi_alpha_g = Pi_g * K_H1.Kmat;
    row_off += pstar;

    SatorraGroup sg;
    sg.Pi_alpha = std::move(Pi_alpha_g);
    sg.Sigma    = im.sigma[g];
    sg.X        = X_per_group[g];
    sg.mean     = mean_per_group[g];
    sg.weight   = weight_per_group[g];
    sg.n_g      = n_per_group[g];
    groups.push_back(std::move(sg));
  }

  // ── 4. Run the core then the p-value wrap ───────────────────────────────
  auto sd_or = compute_satorra2000(groups, A_alpha, gamma);
  if (!sd_or.has_value()) {
    return std::unexpected(sd_or.error());
  }
  return lr_test_satorra2000(T_H0 - T_H1, *sd_or);
}

}  // namespace magmaan::nt::robust
