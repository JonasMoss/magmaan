#include "fiml_internal.hpp"

namespace magmaan::estimate::fiml {
namespace internal {

post_expected<double>
normal_theory_chisq_from_moments(
    const std::vector<Eigen::VectorXd>& mu_model,
    const std::vector<Eigen::MatrixXd>& sigma_model,
    const std::vector<Eigen::VectorXd>& mu_sat,
    const std::vector<Eigen::MatrixXd>& sigma_sat,
    const std::vector<std::int64_t>& n_obs,
    std::string_view label) {
  if (sigma_model.size() != sigma_sat.size() ||
      mu_model.size() != mu_sat.size() ||
      sigma_model.size() != n_obs.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": moment block count mismatch"));
  }

  double out = 0.0;
  for (std::size_t b = 0; b < sigma_sat.size(); ++b) {
    const Eigen::MatrixXd Sigma =
        0.5 * (sigma_model[b] + sigma_model[b].transpose());
    const Eigen::MatrixXd S = 0.5 * (sigma_sat[b] + sigma_sat[b].transpose());
    if (Sigma.rows() != Sigma.cols() || S.rows() != S.cols() ||
        Sigma.rows() != S.rows() || mu_model[b].size() != Sigma.rows() ||
        mu_sat[b].size() != Sigma.rows()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": moment dimension mismatch"));
    }
    Eigen::LLT<Eigen::MatrixXd> llt_sigma(Sigma);
    Eigen::LLT<Eigen::MatrixXd> llt_s(S);
    if (llt_sigma.info() != Eigen::Success || llt_s.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": covariance is not positive definite"));
    }
    const Eigen::VectorXd d = mu_sat[b] - mu_model[b];
    const double f = log_det_from_llt(llt_sigma) - log_det_from_llt(llt_s) +
        llt_sigma.solve(S).trace() + d.dot(llt_sigma.solve(d)) -
        static_cast<double>(S.rows());
    if (!std::isfinite(f)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": non-finite discrepancy"));
    }
    out += static_cast<double>(n_obs[b]) * f;
  }
  if (out < 0.0 && out > -1e-8) out = 0.0;
  return out;
}

std::vector<Eigen::MatrixXd>
independence_covariances_from_baseline(
    const std::vector<Eigen::VectorXd>& vars,
    const std::vector<Eigen::MatrixXd>& covs,
    const std::vector<Eigen::Index>& exo_idx) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(vars.size());
  for (std::size_t b = 0; b < vars.size(); ++b) {
    const Eigen::Index p = vars[b].size();
    Eigen::MatrixXd Sigma = vars[b].asDiagonal();
    if (b < covs.size() && covs[b].rows() == p && covs[b].cols() == p) {
      for (Eigen::Index a = 0;
           a < static_cast<Eigen::Index>(exo_idx.size()); ++a) {
        const Eigen::Index ia = exo_idx[static_cast<std::size_t>(a)];
        if (ia < 0 || ia >= p) continue;
        for (Eigen::Index c = a + 1;
             c < static_cast<Eigen::Index>(exo_idx.size()); ++c) {
          const Eigen::Index ic = exo_idx[static_cast<std::size_t>(c)];
          if (ic < 0 || ic >= p) continue;
          Sigma(ia, ic) = covs[b](ia, ic);
          Sigma(ic, ia) = covs[b](ic, ia);
        }
      }
    }
    out.push_back(std::move(Sigma));
  }
  return out;
}

int baseline_df_from_block_p(const std::vector<Eigen::Index>& block_p,
                             const std::vector<Eigen::Index>& exo_idx) {
  int out = 0;
  const int px = static_cast<int>(exo_idx.size());
  for (Eigen::Index p : block_p) {
    out += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
    out -= px * (px - 1) / 2;
  }
  return out;
}

Eigen::MatrixXd baseline_eta_delta_from_block_p(
    const std::vector<Eigen::Index>& block_p,
    const std::vector<Eigen::Index>& exo_idx) {
  Eigen::Index n_rows = 0;
  Eigen::Index n_cols = 0;
  const Eigen::Index n_exo = static_cast<Eigen::Index>(exo_idx.size());
  const Eigen::Index n_exo_cov = n_exo * (n_exo - 1) / 2;
  for (Eigen::Index p : block_p) {
    n_rows += p + vech_len(p);
    n_cols += p + p + n_exo_cov;
  }

  Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(n_rows, n_cols);
  Eigen::Index row0 = 0;
  Eigen::Index col0 = 0;
  for (Eigen::Index p : block_p) {
    for (Eigen::Index j = 0; j < p; ++j) {
      Delta(row0 + j, col0 + j) = 1.0;
      Delta(row0 + p + vech_index(p, j, j), col0 + p + j) = 1.0;
    }
    Eigen::Index cov_col = col0 + 2 * p;
    for (Eigen::Index a = 0; a < n_exo; ++a) {
      const Eigen::Index ia = exo_idx[static_cast<std::size_t>(a)];
      if (ia < 0 || ia >= p) continue;
      for (Eigen::Index b = a + 1; b < n_exo; ++b) {
        const Eigen::Index ib = exo_idx[static_cast<std::size_t>(b)];
        if (ib >= 0 && ib < p) {
          const Eigen::Index r = std::max(ia, ib);
          const Eigen::Index c = std::min(ia, ib);
          Delta(row0 + p + vech_index(p, r, c), cov_col) = 1.0;
        }
        ++cov_col;
      }
    }
    row0 += p + vech_len(p);
    col0 += 2 * p + n_exo_cov;
  }
  return Delta;
}

post_expected<Eigen::MatrixXd>
fiml_complete_h1_information_from_moments(
    const RawData& raw,
    const std::vector<Eigen::MatrixXd>& sigma,
    std::string_view label) {
  const std::size_t B = raw.X.size();
  if (B == 0 || sigma.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": block count mismatch"));
  }

  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = raw.X[b].cols();
    if (sigma[b].rows() != p || sigma[b].cols() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": covariance dimension mismatch"));
    }
    q_b[b] = p + vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
  }
  const Eigen::Index Q = off_b.back();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Q, Q);

  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = raw.X[b].cols();
    const Eigen::Index pstar = vech_len(p);
    Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (sigma[b] + sigma[b].transpose()));
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": covariance is not positive definite"));
    }
    Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(p, p));
    A = 0.5 * (A + A.transpose()).eval();

    Eigen::MatrixXd Hdev = Eigen::MatrixXd::Zero(p + pstar, p + pstar);
    Hdev.topLeftCorner(p, p) = 2.0 * A;

    std::vector<SigmaBasis> basis;
    basis.reserve(static_cast<std::size_t>(pstar));
    for (Eigen::Index c = 0; c < p; ++c) {
      for (Eigen::Index r0 = c; r0 < p; ++r0) {
        SigmaBasis e;
        e.full_index = p + vech_index(p, r0, c);
        e.a = r0;
        e.b = c;
        basis.push_back(e);
      }
    }
    for (std::size_t kk = 0; kk < basis.size(); ++kk) {
      const SigmaBasis& k = basis[kk];
      for (std::size_t ll = 0; ll <= kk; ++ll) {
        const SigmaBasis& l = basis[ll];
        const double h = basis_trace_xy(A, A, k, l);
        Hdev(k.full_index, l.full_index) += h;
        if (kk != ll) Hdev(l.full_index, k.full_index) += h;
      }
    }

    const double n_b = static_cast<double>(raw.X[b].rows());
    const Eigen::Index q = q_b[b];
    const Eigen::Index off = off_b[b];
    H.block(off, off, q, q) = (n_b / 2.0) * Hdev;
  }

  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

std::vector<Eigen::Index> block_p_from_saturated(const SaturatedMoments& sm) {
  std::vector<Eigen::Index> out;
  out.reserve(sm.cov.size());
  for (const Eigen::MatrixXd& S : sm.cov) out.push_back(S.rows());
  return out;
}

post_expected<double>
residual_projector_trace(const Eigen::MatrixXd& V,
                         const Eigen::MatrixXd& G,
                         const Eigen::MatrixXd& Delta,
                         std::string_view label) {
  if (V.rows() != V.cols() || G.rows() != G.cols() ||
      V.rows() != G.rows() || Delta.rows() != V.rows()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": eta-space dimension mismatch"));
  }
  const Eigen::MatrixXd VD = V * Delta;
  Eigen::MatrixXd DtVD = Delta.transpose() * VD;
  DtVD = 0.5 * (DtVD + DtVD.transpose()).eval();
  auto DtVDinv_or = invert_symmetric(DtVD,
                                     std::string(label) + ": DtVD");
  if (!DtVDinv_or.has_value()) return std::unexpected(DtVDinv_or.error());
  const double tr_vg = V.cwiseProduct(G.transpose()).sum();
  Eigen::MatrixXd DtGVD = VD.transpose() * (G * VD);
  DtGVD = 0.5 * (DtGVD + DtGVD.transpose()).eval();
  const double tr = tr_vg - ((*DtVDinv_or) * DtGVD).trace();
  if (!std::isfinite(tr)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": non-finite trace"));
  }
  return tr;
}

post_expected<double>
fiml_corrected_v3_trace(const Eigen::MatrixXd& H_model,
                        const Eigen::MatrixXd& H_complete,
                        const Eigen::MatrixXd& J_model,
                        const Eigen::MatrixXd& Delta,
                        const Eigen::MatrixXd& H_info,
                        std::string_view label) {
  if (H_model.rows() != H_model.cols() ||
      H_complete.rows() != H_complete.cols() ||
      J_model.rows() != J_model.cols() ||
      H_info.rows() != H_info.cols() ||
      H_model.rows() != H_complete.rows() ||
      H_model.rows() != J_model.rows() ||
      H_model.rows() != H_info.rows() ||
      Delta.rows() != H_model.rows()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": eta-space dimension mismatch"));
  }

  auto Hm_inv_or = invert_symmetric(H_model,
                                    std::string(label) + ": Hm inverse");
  if (!Hm_inv_or.has_value()) return std::unexpected(Hm_inv_or.error());
  const Eigen::MatrixXd& Hm_inv = *Hm_inv_or;

  const Eigen::MatrixXd HinfoD = H_info * Delta;
  Eigen::MatrixXd DtHinfoD = Delta.transpose() * HinfoD;
  DtHinfoD = 0.5 * (DtHinfoD + DtHinfoD.transpose()).eval();
  auto E_or = invert_symmetric(DtHinfoD,
                               std::string(label) + ": information inverse");
  if (!E_or.has_value()) return std::unexpected(E_or.error());
  const Eigen::MatrixXd& E = *E_or;

  const Eigen::MatrixXd gamma = Hm_inv * J_model * Hm_inv;
  const double tr11 = H_complete.cwiseProduct(gamma.transpose()).sum();

  const Eigen::MatrixXd HcD = H_complete * Delta;
  const Eigen::MatrixXd tr12_mat =
      Delta.transpose() * J_model * Hm_inv * HcD;
  const double tr12 = tr12_mat.cwiseProduct(E).sum();

  const Eigen::MatrixXd JD_E =
      (Delta.transpose() * J_model * Delta) * E;
  const Eigen::MatrixXd HcD_E = (Delta.transpose() * HcD) * E;
  const double tr22 = JD_E.cwiseProduct(HcD_E.transpose()).sum();

  const double out = tr11 - 2.0 * tr12 + tr22;
  if (!std::isfinite(out)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": non-finite trace"));
  }
  return out;
}

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures_impl(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const RawData& raw,
                                 const Estimates& est,
                                 int df,
                                 const FIMLPack& pack,
                                 const FIMLH1& h1,
                                 const SaturatedMoments* sm_precomputed) {
  if (df <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_corrected_fit_measures: requires df > 0"));
  }
  if (auto e = validate_h1_blocks(pack.cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
  }
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  if (auto e = resolve_fixed_x_from_sample(pt, rep, pack.start_stats);
      !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "resolve_fixed_x_from_sample"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto implied_or = ev_or->evaluate(est.theta, true, true);
  if (!implied_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::evaluate failed: " + implied_or.error().detail));
  }

  SaturatedMoments sm_owned;
  if (!sm_precomputed) {
    auto sm_or = saturated_em_moments(raw, pack, h1);
    if (!sm_or.has_value()) return std::unexpected(sm_or.error());
    sm_owned = std::move(*sm_or);
  }
  const SaturatedMoments& sm = sm_precomputed ? *sm_precomputed : sm_owned;

  auto xx3_or = normal_theory_chisq_from_moments(
      implied_or->moments.mu, implied_or->moments.sigma, h1.mu, h1.sigma, sm.n_obs,
      "fiml_corrected_fit_measures: user XX3");
  if (!xx3_or.has_value()) return std::unexpected(xx3_or.error());

  auto H_complete_or = fiml_complete_h1_information_from_moments(
      raw, h1.sigma,
      "fiml_corrected_fit_measures: complete-data H1 information");
  if (!H_complete_or.has_value()) return std::unexpected(H_complete_or.error());

  auto delta_or = fiml_projector_delta_impl(
      pt, rep, raw, est, pack, sm.H.rows());
  if (!delta_or.has_value()) return std::unexpected(delta_or.error());

  auto tr_or = fiml_corrected_v3_trace(
      sm.H, *H_complete_or, sm.J, *delta_or, sm.H,
      "fiml_corrected_fit_measures: user");
  if (!tr_or.has_value()) return std::unexpected(tr_or.error());

  const std::vector<Eigen::Index> exo_idx = observed_exogenous_indices(pt);
  std::vector<Eigen::VectorXd> means_null;
  std::vector<Eigen::VectorXd> vars_null;
  if (auto ok = independence_moments_from_raw(raw, means_null, vars_null);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  const std::vector<Eigen::MatrixXd> sigma_null =
      independence_covariances_from_baseline(vars_null, pack.start_stats.S,
                                             exo_idx);
  auto xx3_null_or = normal_theory_chisq_from_moments(
      means_null, sigma_null, h1.mu, h1.sigma, sm.n_obs,
      "fiml_corrected_fit_measures: baseline XX3");
  if (!xx3_null_or.has_value()) return std::unexpected(xx3_null_or.error());

  const int df_null = baseline_df_from_block_p(pack.cache.block_p, exo_idx);
  if (df_null <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_corrected_fit_measures: baseline df must be > 0"));
  }
  const Eigen::MatrixXd D_null =
      baseline_eta_delta_from_block_p(pack.cache.block_p, exo_idx);
  auto tr_null_or = fiml_corrected_v3_trace(
      sm.H, *H_complete_or, sm.J, D_null, sm.H,
      "fiml_corrected_fit_measures: baseline");
  if (!tr_null_or.has_value()) return std::unexpected(tr_null_or.error());

  FIMLCorrectedFitMeasures out;
  out.xx3 = *xx3_or;
  out.df3 = df;
  out.c_hat3 = *tr_or / static_cast<double>(df);
  out.xx3_scaled = (out.c_hat3 > 0.0)
      ? out.xx3 / out.c_hat3
      : std::numeric_limits<double>::quiet_NaN();
  out.xx3_null = *xx3_null_or;
  out.df3_null = df_null;
  out.c_hat3_null = *tr_null_or / static_cast<double>(df_null);
  out.xx3_null_scaled = (out.c_hat3_null > 0.0)
      ? out.xx3_null / out.c_hat3_null
      : std::numeric_limits<double>::quiet_NaN();

  measures::RobustFitMeasureInputs inputs;
  inputs.chi2 = out.xx3;
  inputs.df = out.df3;
  inputs.chi2_scaled = out.xx3_scaled;
  inputs.scaling_factor = out.c_hat3;
  inputs.baseline_chi2 = out.xx3_null;
  inputs.baseline_df = out.df3_null;
  inputs.baseline_chi2_scaled = out.xx3_null_scaled;
  inputs.baseline_scaling_factor = out.c_hat3_null;
  inputs.n_total = pack.cache.n_total;
  inputs.n_groups = pack.cache.block_p.size();
  out.indices = measures::robust_fit_measures(inputs);
  return out;
}

}  // namespace

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const RawData& raw,
                            const Estimates& est,
                            int df,
                            FIML discrepancy) {
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 moments"));
  }
  return fiml_corrected_fit_measures_impl(std::move(pt), rep, raw, est, df,
                                          *pack_or, *h1_or, nullptr);
}

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const RawData& raw,
                            const Estimates& est,
                            int df,
                            const FIMLPack& pack,
                            const FIMLH1& h1) {
  return fiml_corrected_fit_measures_impl(std::move(pt), rep, raw, est, df,
                                          pack, h1, nullptr);
}

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const RawData& raw,
                            const Estimates& est,
                            int df,
                            const FIMLPack& pack,
                            const FIMLH1& h1,
                            const SaturatedMoments& sm) {
  return fiml_corrected_fit_measures_impl(std::move(pt), rep, raw, est, df,
                                          pack, h1, &sm);
}

post_expected<Eigen::MatrixXd>
fiml_residual_projector(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const RawData& raw,
                        const Estimates& est,
                        const Eigen::Ref<const Eigen::MatrixXd>& V,
                        FIML discrepancy) {
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  return fiml_residual_projector_impl(pt, rep, raw, est, *pack_or, V);
}

namespace internal {

post_expected<BaselineFit>
fiml_baseline_chi2_impl(const RawData& raw,
                        const std::vector<Eigen::Index>& exo_idx,
                        const FIMLPack& pack,
                        const FIMLH1& h1) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& start_samp = pack.start_stats;

  std::vector<Eigen::VectorXd> means;
  std::vector<Eigen::VectorXd> vars;
  if (auto ok = independence_moments_from_raw(raw, means, vars); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto baseline_or = independence_value_from_patterns(cache, means, vars,
                                                      start_samp.S, exo_idx);
  if (!baseline_or.has_value()) {
    return std::unexpected(baseline_or.error());
  }

  BaselineFit out;
  out.chi2 = static_cast<double>(cache.n_total) * (*baseline_or - h1.value);
  if (out.chi2 < 0.0 && out.chi2 > -1e-8) out.chi2 = 0.0;
  for (Eigen::Index p : cache.block_p) {
    out.df += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
    const int px = static_cast<int>(exo_idx.size());
    out.df -= px * (px - 1) / 2;
  }
  return out;
}

post_expected<BaselineFit>
fiml_baseline_chi2_from_raw(const RawData& raw,
                            const std::vector<Eigen::Index>& exo_idx) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 likelihood"));
  }
  return fiml_baseline_chi2_impl(raw, exo_idx, *pack_or, *h1_or);
}

}  // namespace

post_expected<BaselineFit>
fiml_baseline_chi2(const RawData& raw,
                   FIML discrepancy) {
  (void)discrepancy;
  return fiml_baseline_chi2_from_raw(raw, {});
}

post_expected<BaselineFit>
fiml_baseline_chi2(const spec::LatentStructure& pt,
                   const RawData& raw,
                   FIML discrepancy) {
  (void)discrepancy;
  return fiml_baseline_chi2_from_raw(raw, observed_exogenous_indices(pt));
}

post_expected<BaselineFit>
fiml_baseline_chi2(const spec::LatentStructure& pt,
                   const RawData& raw,
                   const FIMLPack& pack,
                   const FIMLH1& h1) {
  return fiml_baseline_chi2_impl(raw, observed_exogenous_indices(pt),
                                 pack, h1);
}


}  // namespace magmaan::estimate::fiml
