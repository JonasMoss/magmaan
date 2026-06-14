#include "magmaan/robust/lr_test_satorra.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/error.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/inference/inference.hpp"        // chi2_pvalue, noncentral_chisq_cdf
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/restriction.hpp"      // restriction_alpha_from_K
#include "magmaan/robust/weighted_chisq.hpp"   // imhof_upper
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"                // vech_len

namespace magmaan::robust {

using inference::chi2_pvalue;
using inference::noncentral_chisq_cdf;

using estimate::EqConstraints;


namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

post_expected<Eigen::MatrixXd>
local_constraint_basis(const spec::LatentStructure& pt,
                       const EqConstraints&         con,
                       const Eigen::VectorXd&       theta,
                       std::string                  label) {
  const Eigen::Index npar = con.Kmat.rows();
  if (theta.size() != npar || con.npar != npar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": theta/K dimension mismatch for local constraint basis"));
  }
  if (pt.nl_constraints.empty()) {
    return con.Kmat;
  }

  const estimate::NonlinearEqConstraints nl =
      estimate::build_nl_constraints(pt);
  const Eigen::MatrixXd H = nl.jacobian(theta);
  if (H.cols() != npar || !H.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": nonlinear equality constraint Jacobian is malformed or "
        "not finite at theta"));
  }
  if (con.A_eq.cols() != npar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": linear equality constraint matrix has wrong column count"));
  }

  Eigen::MatrixXd C(con.A_eq.rows() + H.rows(), npar);
  if (con.A_eq.rows() > 0) {
    C.topRows(con.A_eq.rows()) = con.A_eq;
  }
  C.bottomRows(H.rows()) = H;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);
  svd.setThreshold(1e-9);
  const Eigen::Index nz = npar - svd.rank();
  if (nz < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": constraint Jacobian rank exceeds parameter dimension"));
  }
  return Eigen::MatrixXd(svd.matrixV().rightCols(nz));
}

Eigen::VectorXd denom_from_sample(const data::SampleStats& samp) {
  Eigen::VectorXd denom(static_cast<Eigen::Index>(samp.n_obs.size()));
  for (Eigen::Index i = 0; i < denom.size(); ++i) {
    denom(i) = static_cast<double>(samp.n_obs[static_cast<std::size_t>(i)]);
  }
  return denom;
}

struct SingleModelScale {
  double scale_c = quiet_nan();
  Eigen::VectorXd eigenvalues;
  std::vector<std::string> warnings;
};

post_expected<SingleModelScale>
single_model_satorra_bentler_scale(const spec::LatentStructure& pt,
                                   const model::MatrixRep&      rep,
                                   const Eigen::VectorXd&       theta,
                                   const data::SampleStats&     samp,
                                   const data::RawData&         raw,
                                   int                          df,
                                   GammaSource                  gamma,
                                   std::string                  label) {
  SingleModelScale out;
  if (df <= 0) {
    out.scale_c = 1.0;
    out.eigenvalues = Eigen::VectorXd::Zero(0);
    return out;
  }

  estimate::Estimates est;
  est.theta = theta;
  auto uf_or = build_u_factor(pt, rep, samp, est,
                              {Information::Expected,
                               WeightMoments::Structured,
                               ScoreCovariance::Empirical});
  if (!uf_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": build_u_factor failed: " + uf_or.error().detail));
  }

  post_expected<Eigen::MatrixXd> M_or;
  if (gamma == GammaSource::NT) {
    M_or = reduced_gamma_nt(*uf_or);
  } else {
    auto zc_or = casewise_contributions(raw, samp, uf_or->has_means);
    if (!zc_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          label + ": casewise_contributions failed: " + zc_or.error().detail));
    }
    const Eigen::VectorXd denom = denom_from_sample(samp);
    M_or = reduced_gamma_sample(*uf_or, *zc_or, denom);
  }
  if (!M_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": reduced Gamma failed: " + M_or.error().detail));
  }

  auto ev_or = ugamma_eigenvalues(*M_or);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        label + ": ugamma_eigenvalues failed: " + ev_or.error().detail));
  }
  out.eigenvalues = std::move(*ev_or);
  out.scale_c = satorra_bentler(1.0, df, out.eigenvalues).scale_c;
  if (!std::isfinite(out.scale_c)) {
    out.warnings.emplace_back(label + ": single-model scaling factor is not finite");
  }
  return out;
}

LRSatorraBentlerDiffResult degenerate_sb_diff(double T_diff,
                                              std::string method) {
  LRSatorraBentlerDiffResult out;
  out.T_diff = T_diff;
  out.df_diff = 0;
  out.scale_c = quiet_nan();
  out.T_scaled = T_diff;
  out.p_value = 1.0;
  out.c_H0 = 1.0;
  out.c_H1 = 1.0;
  if (std::abs(T_diff) > 1e-8) {
    out.warnings.emplace_back(
        method + ": df_diff = 0 but T_diff = " + std::to_string(T_diff) +
        " — H0 and H1 should give identical test statistics. Result is "
        "degenerate.");
  }
  return out;
}

LRSatorraBentlerDiffResult
finalize_sb_diff(double T_H0, double T_H1, int df_H0, int df_H1,
                 double c_H0, double c_H1, double c_hybrid,
                 double scale_c, std::string method) {
  LRSatorraBentlerDiffResult out;
  out.T_diff = T_H0 - T_H1;
  out.df_diff = df_H0 - df_H1;
  out.c_H0 = c_H0;
  out.c_H1 = c_H1;
  out.c_hybrid = c_hybrid;

  if (out.df_diff == 0) {
    return degenerate_sb_diff(out.T_diff, method);
  }

  out.scale_c = scale_c;
  if (!std::isfinite(scale_c) || scale_c <= 0.0) {
    out.T_scaled = quiet_nan();
    out.p_value = quiet_nan();
    out.warnings.emplace_back(method + ": scaling factor is non-positive or "
                              "not finite");
    return out;
  }
  out.T_scaled = out.T_diff / scale_c;
  out.p_value = chi2_pvalue(out.T_scaled, out.df_diff);
  return out;
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
  out.scaled_shifted = ScaledShiftedResult{
      .chi2_adj = T_diff,
      .df       = 0,
      .scale_a  = 1.0,
      .shift_b  = 0.0};
  out.p_scaled_shifted = 1.0;
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

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001(double T_H0, double T_H1,
                            int df_H0, int df_H1,
                            double c_H0, double c_H1) {
  const int m = df_H0 - df_H1;
  if (m < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra_bentler2001: df_H0 − df_H1 is negative; H1 must "
        "be the less-restricted model"));
  }
  if (m == 0) {
    return degenerate_sb_diff(T_H0 - T_H1, "lr_test_satorra_bentler2001");
  }

  if (df_H0 == 0) {
    c_H0 = 1.0;
  }
  if (df_H1 == 0) {
    c_H1 = 1.0;
  } else if (df_H1 > 0 && std::abs(T_H1) < std::sqrt(std::numeric_limits<double>::epsilon())) {
    c_H1 = 0.0;
  }

  const double cd = (static_cast<double>(df_H0) * c_H0 -
                     static_cast<double>(df_H1) * c_H1) /
                    static_cast<double>(m);
  return finalize_sb_diff(T_H0, T_H1, df_H0, df_H1, c_H0, c_H1,
                          quiet_nan(), cd, "lr_test_satorra_bentler2001");
}

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010(double T_H0, double T_H1,
                            int df_H0, int df_H1,
                            double c_H0, double c_M10) {
  const int m = df_H0 - df_H1;
  if (m < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra_bentler2010: df_H0 − df_H1 is negative; H1 must "
        "be the less-restricted model"));
  }
  if (m == 0) {
    return degenerate_sb_diff(T_H0 - T_H1, "lr_test_satorra_bentler2010");
  }

  if (df_H0 == 0) {
    c_H0 = 1.0;
  }
  if (df_H1 == 0) {
    c_M10 = 1.0;
  }

  const double cd = (static_cast<double>(df_H0) * c_H0 -
                     static_cast<double>(df_H1) * c_M10) /
                    static_cast<double>(m);
  return finalize_sb_diff(T_H0, T_H1, df_H0, df_H1, c_H0, c_M10,
                          c_M10, cd, "lr_test_satorra_bentler2010");
}

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001_from_data(
    const spec::LatentStructure& pt_H1,
    const model::MatrixRep&      rep_H1,
    const Eigen::VectorXd&       theta_H1_full,
    const spec::LatentStructure& pt_H0,
    const model::MatrixRep&      rep_H0,
    const Eigen::VectorXd&       theta_H0_full,
    const data::RawData&         raw,
    double                       T_H0,
    double                       T_H1,
    int                          df_H0,
    int                          df_H1,
    GammaSource                  gamma) {
  auto samp_or = data::sample_stats_from_raw(raw);
  if (!samp_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra_bentler2001_from_data: sample_stats_from_raw failed: " +
        samp_or.error().detail));
  }
  auto c1_or = single_model_satorra_bentler_scale(
      pt_H1, rep_H1, theta_H1_full, *samp_or, raw, df_H1, gamma,
      "lr_test_satorra_bentler2001_from_data/H1");
  if (!c1_or.has_value()) {
    return std::unexpected(c1_or.error());
  }
  auto c0_or = single_model_satorra_bentler_scale(
      pt_H0, rep_H0, theta_H0_full, *samp_or, raw, df_H0, gamma,
      "lr_test_satorra_bentler2001_from_data/H0");
  if (!c0_or.has_value()) {
    return std::unexpected(c0_or.error());
  }
  auto out_or = lr_test_satorra_bentler2001(
      T_H0, T_H1, df_H0, df_H1, c0_or->scale_c, c1_or->scale_c);
  if (!out_or.has_value()) {
    return std::unexpected(out_or.error());
  }
  out_or->warnings.insert(out_or->warnings.end(),
                          c1_or->warnings.begin(), c1_or->warnings.end());
  out_or->warnings.insert(out_or->warnings.end(),
                          c0_or->warnings.begin(), c0_or->warnings.end());
  return out_or;
}

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010_from_data(
    const spec::LatentStructure& pt_H1,
    const model::MatrixRep&      rep_H1,
    const Eigen::VectorXd&       theta_H0_full,
    const spec::LatentStructure& pt_H0,
    const model::MatrixRep&      rep_H0,
    const Eigen::VectorXd&       theta_H0_for_H0,
    const data::RawData&         raw,
    double                       T_H0,
    double                       T_H1,
    int                          df_H0,
    int                          df_H1,
    GammaSource                  gamma) {
  if (theta_H0_full.size() != theta_H0_for_H0.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra_bentler2010_from_data: H0 theta cannot be injected "
        "into H1 because the full parameter vectors have different sizes"));
  }
  auto samp_or = data::sample_stats_from_raw(raw);
  if (!samp_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra_bentler2010_from_data: sample_stats_from_raw failed: " +
        samp_or.error().detail));
  }
  auto c0_or = single_model_satorra_bentler_scale(
      pt_H0, rep_H0, theta_H0_for_H0, *samp_or, raw, df_H0, gamma,
      "lr_test_satorra_bentler2010_from_data/H0");
  if (!c0_or.has_value()) {
    return std::unexpected(c0_or.error());
  }
  auto c10_or = single_model_satorra_bentler_scale(
      pt_H1, rep_H1, theta_H0_full, *samp_or, raw, df_H1, gamma,
      "lr_test_satorra_bentler2010_from_data/M10");
  if (!c10_or.has_value()) {
    return std::unexpected(c10_or.error());
  }
  auto out_or = lr_test_satorra_bentler2010(
      T_H0, T_H1, df_H0, df_H1, c0_or->scale_c, c10_or->scale_c);
  if (!out_or.has_value()) {
    return std::unexpected(out_or.error());
  }
  out_or->warnings.insert(out_or->warnings.end(),
                          c0_or->warnings.begin(), c0_or->warnings.end());
  out_or->warnings.insert(out_or->warnings.end(),
                          c10_or->warnings.begin(), c10_or->warnings.end());
  return out_or;
}

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
  out.scaled_shifted = scaled_shifted(
      T_diff, WeightedChiSquareMoments{out.df_diff, sd.trace_CinvS,
                                       sd.trace_CinvS_sq});
  out.p_scaled_shifted =
      chi2_pvalue(out.scaled_shifted.chi2_adj, out.scaled_shifted.df);

  out.p_mixture = imhof_upper(sd.eigenvalues, T_diff);
  return out;
}

post_expected<LRSatorra2000Result>
lr_test_satorra2000_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&              rep_H1,
    const Eigen::VectorXd&               theta_H1_full,
    const EqConstraints&                 K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&              rep_H0,
    const Eigen::VectorXd&               theta_H0_full,
    const EqConstraints&                 K_H0,
    const std::vector<Eigen::MatrixXd>&  X_per_group,
    const std::vector<Eigen::VectorXd>&  mean_per_group,
    const std::vector<std::int32_t>&     n_per_group,
    const std::vector<double>&           weight_per_group,
    double                               T_H0,
    double                               T_H1,
    int                                  df_H0,
    int                                  df_H1,
    Satorra2000Options                   options) {
  const int df_diff_from_T = df_H0 - df_H1;
  if (df_diff_from_T < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: df_H0 − df_H1 is negative; H1 must "
        "be the less-restricted model"));
  }

  // ── 1. Evaluate Π and Σ at θ̂_H1, per group ─────────────────────────────
  auto ev_or = model::ModelEvaluator::build(pt_H1, rep_H1);
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
  const Eigen::MatrixXd   Pi_H1_alpha_all = Pi_th * K_H1.Kmat;
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

  // ── 2. Derive A_α (the restriction in H1's α-space) ─────────────────────
  Eigen::MatrixXd A_alpha;
  if (options.a_method == SatorraAMethod::Exact) {
    auto restr_or = restriction_alpha_from_K(K_H1, K_H0);
    if (!restr_or.has_value()) {
      return std::unexpected(restr_or.error());
    }
    A_alpha = std::move(restr_or->A);
  } else {
    auto ev0_or = model::ModelEvaluator::build(pt_H0, rep_H0);
    if (!ev0_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "lr_test_satorra2000_from_data: H0 ModelEvaluator::build failed: " +
          ev0_or.error().detail));
    }
    auto dsig0_or = ev0_or->dsigma_dtheta(theta_H0_full);
    if (!dsig0_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "lr_test_satorra2000_from_data: dsigma_dtheta(θ̂_H0) failed: " +
          dsig0_or.error().detail));
    }
    if (dsig0_or->rows() != Pi_th.rows() ||
        dsig0_or->cols() != K_H0.Kmat.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "lr_test_satorra2000_from_data: H0 moment Jacobian shape disagrees "
          "with H1 or K_H0"));
    }
    auto A_or = restriction_alpha_delta_from_jacobians(
        Pi_H1_alpha_all, (*dsig0_or) * K_H0.Kmat, df_diff_from_T);
    if (!A_or.has_value()) {
      return std::unexpected(A_or.error());
    }
    A_alpha = std::move(*A_or);
  }

  // The Satorra-2000 df_diff must equal df_H0 − df_H1; if the caller's df
  // accounting and the derived restriction rank disagree, surface it.
  const int df_diff_from_A = static_cast<int>(A_alpha.rows());
  if (df_diff_from_A != df_diff_from_T) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_from_data: df_diff mismatch — derived A has m = " +
        std::to_string(df_diff_from_A) + " but df_H0 − df_H1 = " +
        std::to_string(df_diff_from_T) + " (check that both fits use the "
        "same lavaanified partable and that the chi²/df pair is consistent)."));
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
    Eigen::MatrixXd Pi_alpha_g = Pi_H1_alpha_all.middleRows(row_off, pstar);
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
  auto sd_or = compute_satorra2000(groups, A_alpha, options.gamma,
                                   options.computation);
  if (!sd_or.has_value()) {
    return std::unexpected(sd_or.error());
  }
  return lr_test_satorra2000(T_H0 - T_H1, *sd_or);
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
  return lr_test_satorra2000_from_data(
      pt, rep, theta_H1_full, K_H1, pt, rep, theta_H1_full, K_H0,
      X_per_group, mean_per_group, n_per_group, weight_per_group,
      T_H0, T_H1, df_H0, df_H1,
      Satorra2000Options{.a_method = SatorraAMethod::Exact, .gamma = gamma});
}

post_expected<SatorraDiffResult>
compute_satorra2000_from_sandwich(
    const Eigen::Ref<const Eigen::MatrixXd>& A1,
    const Eigen::Ref<const Eigen::MatrixXd>& B1,
    const Eigen::Ref<const Eigen::MatrixXd>& A_alpha) {
  const Eigen::Index r1 = A1.rows();
  const Eigen::Index m = A_alpha.rows();

  if (A1.cols() != r1 || B1.rows() != r1 || B1.cols() != r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000_from_sandwich: A1/B1 dimensions must match"));
  }
  if (A_alpha.cols() != r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000_from_sandwich: A_alpha column count " +
        std::to_string(A_alpha.cols()) + " does not match r1 = " +
        std::to_string(r1)));
  }
  if (m == 0) {
    SatorraDiffResult deg;
    deg.C = Eigen::MatrixXd::Zero(0, 0);
    deg.S = Eigen::MatrixXd::Zero(0, 0);
    deg.eigenvalues = Eigen::VectorXd::Zero(0);
    deg.trace_CinvS = 0.0;
    deg.trace_CinvS_sq = 0.0;
    return deg;
  }
  if (m > r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000_from_sandwich: A_alpha has more rows than H1 "
        "alpha directions"));
  }

  Eigen::MatrixXd P = 0.5 * (A1 + A1.transpose());
  Eigen::LDLT<Eigen::MatrixXd> ldlt_P(P);
  if (ldlt_P.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000_from_sandwich: H1 alpha-space bread A1 is not "
        "invertible"));
  }
  {
    const Eigen::VectorXd d_abs = ldlt_P.vectorD().cwiseAbs();
    const double tol_pivot = 1e-10 * d_abs.maxCoeff();
    if (d_abs.minCoeff() <= tol_pivot) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "compute_satorra2000_from_sandwich: H1 alpha-space bread A1 is "
          "rank-deficient (smallest |D| pivot " +
          std::to_string(d_abs.minCoeff()) + " <= tol " +
          std::to_string(tol_pivot) + ")"));
    }
  }

  const Eigen::MatrixXd Y = ldlt_P.solve(A_alpha.transpose());
  Eigen::MatrixXd C = A_alpha * Y;
  C = 0.5 * (C + C.transpose()).eval();

  const Eigen::MatrixXd Bsym = 0.5 * (B1 + B1.transpose());
  Eigen::MatrixXd S = Y.transpose() * Bsym * Y;
  S = 0.5 * (S + S.transpose()).eval();

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
      S, C, Eigen::EigenvaluesOnly | Eigen::Ax_lBx);
  if (ges.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_satorra2000_from_sandwich: generalized eigensolver failed "
        "(restriction C may be singular against H1 eta-space curvature)"));
  }
  Eigen::VectorXd eig = ges.eigenvalues();

  std::vector<std::string> warnings;
  const double clip_thresh = 1e-12 * std::max(1.0, eig.cwiseAbs().maxCoeff());
  for (Eigen::Index k = 0; k < eig.size(); ++k) {
    if (eig(k) < 0.0) {
      if (eig(k) >= -clip_thresh) {
        eig(k) = 0.0;
      } else {
        warnings.emplace_back(
            "compute_satorra2000_from_sandwich: detected eigenvalue " +
            std::to_string(eig(k)) + " below clip threshold " +
            std::to_string(-clip_thresh) + " -- clipped to 0");
        eig(k) = 0.0;
      }
    }
  }

  SatorraDiffResult out;
  out.C = std::move(C);
  out.S = std::move(S);
  out.eigenvalues = eig;
  out.trace_CinvS = eig.sum();
  out.trace_CinvS_sq = eig.squaredNorm();
  out.warnings = std::move(warnings);
  return out;
}

post_expected<SatorraDiffResult>
compute_fiml_satorra2000(
    const Eigen::Ref<const Eigen::MatrixXd>& Delta1_alpha,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& Gamma,
    const Eigen::Ref<const Eigen::MatrixXd>& A_alpha) {
  const Eigen::Index Q = Delta1_alpha.rows();
  if (V.rows() != Q || V.cols() != Q || Gamma.rows() != Q ||
      Gamma.cols() != Q) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "compute_fiml_satorra2000: V/Gamma dimensions must match Delta rows"));
  }
  const Eigen::MatrixXd Vsym = 0.5 * (V + V.transpose());
  const Eigen::MatrixXd Gsym = 0.5 * (Gamma + Gamma.transpose());
  const Eigen::MatrixXd VD = Vsym * Delta1_alpha;
  Eigen::MatrixXd A1 = Delta1_alpha.transpose() * VD;
  A1 = 0.5 * (A1 + A1.transpose()).eval();
  Eigen::MatrixXd B1 = VD.transpose() * Gsym * VD;
  B1 = 0.5 * (B1 + B1.transpose()).eval();
  return compute_satorra2000_from_sandwich(A1, B1, A_alpha);
}

post_expected<LRSatorra2000Result>
lr_test_satorra2000_fiml_from_data_impl(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const EqConstraints&             K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const EqConstraints&             K_H0,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    const estimate::fiml::FIMLPack*  pack,
    const estimate::fiml::FIMLH1*    h1,
    GammaSource                      gamma,
    SatorraAMethod                   a_method,
    double                           h_step) {
  const int df_diff_from_T = df_H0 - df_H1;
  if (df_diff_from_T < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_fiml_from_data: df_H0 - df_H1 is negative; "
        "H1 must be the less-restricted model"));
  }
  if (!(h_step > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_fiml_from_data: h_step must be > 0"));
  }
  const bool has_nonlinear =
      !pt_H1.nl_constraints.empty() || !pt_H0.nl_constraints.empty();
  std::vector<std::string> warnings;
  if (has_nonlinear && a_method == SatorraAMethod::Exact) {
    warnings.emplace_back(
        "lr_test_satorra2000_fiml_from_data: nonlinear equality constraints "
        "have no global affine exact restriction map; using the local tangent "
        "restriction map at the fitted points");
  }

  const bool use_pack = pack != nullptr && h1 != nullptr;
  auto sm_or = use_pack
      ? estimate::fiml::saturated_em_moments(raw, *pack, *h1)
      : estimate::fiml::saturated_em_moments(raw, h_step);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  const estimate::fiml::SaturatedMoments& sm = *sm_or;
  const Eigen::MatrixXd& V = sm.H;

  Eigen::MatrixXd Gamma;
  if (gamma == GammaSource::NT) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt_V(0.5 * (V + V.transpose()));
    if (ldlt_V.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "lr_test_satorra2000_fiml_from_data: saturated information V is "
          "not invertible"));
    }
    Gamma = ldlt_V.solve(Eigen::MatrixXd::Identity(V.rows(), V.cols()));
    Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();
  } else {
    Gamma = sm.acov;
  }

  estimate::Estimates est_H1;
  est_H1.theta = theta_H1_full;
  auto D1_or = use_pack
      ? estimate::fiml::fiml_eta_jacobian(
            pt_H1, rep_H1, raw, est_H1, *pack)
      : estimate::fiml::fiml_eta_jacobian(
            pt_H1, rep_H1, raw, est_H1, estimate::fiml::FIML{});
  if (!D1_or.has_value()) return std::unexpected(D1_or.error());
  if (D1_or->Delta_theta.rows() != V.rows() ||
      D1_or->Delta_theta.cols() != K_H1.Kmat.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_fiml_from_data: H1 eta Jacobian shape "
        "disagrees with saturated information or K_H1"));
  }
  auto B1_or = local_constraint_basis(
      pt_H1, K_H1, theta_H1_full,
      "lr_test_satorra2000_fiml_from_data: H1");
  if (!B1_or.has_value()) return std::unexpected(B1_or.error());
  const Eigen::MatrixXd Delta1_alpha = D1_or->Delta_theta * (*B1_or);

  Eigen::MatrixXd A_alpha;
  if (a_method == SatorraAMethod::Exact && !has_nonlinear) {
    auto restr_or = restriction_alpha_from_K(K_H1, K_H0);
    if (!restr_or.has_value()) return std::unexpected(restr_or.error());
    A_alpha = std::move(restr_or->A);
  } else {
    estimate::Estimates est_H0;
    est_H0.theta = theta_H0_full;
    auto D0_or = use_pack
        ? estimate::fiml::fiml_eta_jacobian(
              pt_H0, rep_H0, raw, est_H0, *pack)
        : estimate::fiml::fiml_eta_jacobian(
              pt_H0, rep_H0, raw, est_H0, estimate::fiml::FIML{});
    if (!D0_or.has_value()) return std::unexpected(D0_or.error());
    if (D0_or->Delta_theta.rows() != V.rows() ||
        D0_or->Delta_theta.cols() != K_H0.Kmat.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "lr_test_satorra2000_fiml_from_data: H0 eta Jacobian shape "
          "disagrees with saturated information or K_H0"));
    }
    auto B0_or = local_constraint_basis(
        pt_H0, K_H0, theta_H0_full,
        "lr_test_satorra2000_fiml_from_data: H0");
    if (!B0_or.has_value()) return std::unexpected(B0_or.error());
    const Eigen::MatrixXd Delta0_alpha = D0_or->Delta_theta * (*B0_or);
    auto A_or = restriction_alpha_delta_from_jacobians(
        Delta1_alpha, Delta0_alpha, df_diff_from_T);
    if (!A_or.has_value()) return std::unexpected(A_or.error());
    A_alpha = std::move(*A_or);
  }

  const int df_diff_from_A = static_cast<int>(A_alpha.rows());
  if (df_diff_from_A != df_diff_from_T) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_fiml_from_data: df_diff mismatch -- derived A "
        "has m = " + std::to_string(df_diff_from_A) +
        " but df_H0 - df_H1 = " + std::to_string(df_diff_from_T)));
  }

  auto sd_or = compute_fiml_satorra2000(Delta1_alpha, V, Gamma, A_alpha);
  if (!sd_or.has_value()) return std::unexpected(sd_or.error());
  auto out_or = lr_test_satorra2000(T_H0 - T_H1, *sd_or);
  if (!out_or.has_value()) return std::unexpected(out_or.error());
  out_or->warnings.insert(out_or->warnings.end(),
                          warnings.begin(), warnings.end());
  return out_or;
}

post_expected<LRSatorra2000Result>
lr_test_satorra2000_fiml_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const EqConstraints&             K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const EqConstraints&             K_H0,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    GammaSource                      gamma,
    SatorraAMethod                   a_method,
    double                           h_step) {
  return lr_test_satorra2000_fiml_from_data_impl(
      pt_H1, rep_H1, theta_H1_full, K_H1,
      pt_H0, rep_H0, theta_H0_full, K_H0,
      raw, T_H0, T_H1, df_H0, df_H1,
      nullptr, nullptr, gamma, a_method, h_step);
}

post_expected<LRSatorra2000Result>
lr_test_satorra2000_fiml_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const EqConstraints&             K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const EqConstraints&             K_H0,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    const estimate::fiml::FIMLPack&  pack,
    const estimate::fiml::FIMLH1&    h1,
    GammaSource                      gamma,
    SatorraAMethod                   a_method,
    double                           h_step) {
  return lr_test_satorra2000_fiml_from_data_impl(
      pt_H1, rep_H1, theta_H1_full, K_H1,
      pt_H0, rep_H0, theta_H0_full, K_H0,
      raw, T_H0, T_H1, df_H0, df_H1,
      &pack, &h1, gamma, a_method, h_step);
}

}  // namespace magmaan::robust
