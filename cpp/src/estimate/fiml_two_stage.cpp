#include "fiml_internal.hpp"

namespace magmaan::estimate::fiml {
namespace internal {

// Non-NT Stage-2 weighted two-stage inference. The DWLS / ADF / DLS weight is
// not the normal-theory weight that build_u_factor bakes in, so the robust
// sandwich runs through the explicit-weight moment-quadratic path. `est` must
// minimize ½ rᵀ W r for the SAME weight (the caller fits Stage 2 with
// `two_stage_stage2_weight_blocks(sm, kind, dls)`); the meat is the per-block
// Γ_FIML in the n-scaled test-statistic convention.
post_expected<TwoStageEMMLInference>
two_stage_em_weighted_inference_from_sm(spec::LatentStructure pt,
                                        const model::MatrixRep& rep,
                                        const Estimates& est,
                                        const SaturatedMoments& sm,
                                        TwoStageWeight kind,
                                        TwoStageDlsOptions dls,
                                        TwoStageBread bread) {
  SampleStats samp = sample_stats_from_saturated(sm);

  auto weight_or = two_stage_stage2_weight_structured(sm, kind, dls);
  if (!weight_or.has_value()) return std::unexpected(weight_or.error());

  auto gamma_full_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_full_or.has_value()) return std::unexpected(gamma_full_or.error());
  std::vector<Eigen::MatrixXd> gamma;
  gamma.reserve(sm.cov.size());
  Eigen::Index goff = 0;
  for (std::size_t b = 0; b < sm.cov.size(); ++b) {
    const Eigen::Index q = sm.cov[b].rows() + vech_len(sm.cov[b].rows());
    gamma.push_back(gamma_full_or->block(goff, goff, q, q));
    goff += q;
  }

  auto rr_or = robust_continuous_ls(std::move(pt), rep, samp, est,
                                    *weight_or, gamma, robust_bread(bread));
  if (!rr_or.has_value()) return std::unexpected(rr_or.error());

  TwoStageEMMLInference out;
  out.vcov = std::move(rr_or->vcov);
  out.se = std::move(rr_or->se);
  out.eigvals = std::move(rr_or->eigvals);
  out.df = rr_or->df;
  out.chisq = rr_or->chisq_standard;
  out.scaling_factor = rr_or->satorra_bentler.scale_c;
  out.chisq_scaled = rr_or->satorra_bentler.chi2_scaled;
  out.trace_ugamma = (out.eigvals.size() > 0) ? out.eigvals.sum() : 0.0;
  out.ntotal = 0;
  for (auto n : samp.n_obs) out.ntotal += n;
  return out;
}

post_expected<TwoStageEMMLInference>
two_stage_complete_data_weighted_ij_from_sm(spec::LatentStructure pt,
                                            const model::MatrixRep& rep,
                                            const RawData& raw,
                                            const Estimates& est,
                                            const SaturatedMoments& sm,
                                            TwoStageWeight kind,
                                            TwoStageDlsOptions dls,
                                            TwoStageBread bread) {
  if (bread != TwoStageBread::Observed || !raw_has_no_missing_mask(raw) ||
      kind == TwoStageWeight::Nt || kind == TwoStageWeight::Uls) {
    return two_stage_em_weighted_inference_from_sm(
        std::move(pt), rep, est, sm, kind, dls, bread);
  }

  auto base_or = two_stage_em_weighted_inference_from_sm(
      pt, rep, est, sm, kind, dls, bread);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  SampleStats samp = sample_stats_from_saturated(sm);
  post_expected<WeightedRobustResult> ij_or =
      std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: unsupported Stage-2 weight"));
  switch (kind) {
    case TwoStageWeight::Dwls:
      ij_or = estimate::robust_continuous_ls_dwls_ij(
          std::move(pt), rep, samp, est, raw);
      break;
    case TwoStageWeight::Adf:
      ij_or = estimate::robust_continuous_ls_wls_ij(
          std::move(pt), rep, samp, est, raw);
      break;
    case TwoStageWeight::Dls: {
      estimate::frontier::DlsWeightOptions opts;
      opts.a = dls.a;
      ij_or = estimate::robust_continuous_ls_dls_ij(
          std::move(pt), rep, samp, est, raw, opts);
      break;
    }
    case TwoStageWeight::Uls:
    case TwoStageWeight::Nt:
      break;
  }
  if (!ij_or.has_value()) return std::unexpected(ij_or.error());

  TwoStageEMMLInference out = std::move(*base_or);
  out.vcov = std::move(ij_or->vcov);
  out.se = std::move(ij_or->se);
  return out;
}



// Per-case ML2S infinitesimal-jackknife blocks plus K and the observed bread,
// shared by the missing-data SE sandwich and the casewise-influence accessor so
// both decompose the SAME complete-sandwich meat. Each block's moment_influence
// is the Stage-1 saturated-moment per-case influence (`saturated_em_moment_influence`,
// which reduces to centered mean/covariance rows under complete data); the
// weight_correction is the data-dependent-weight `IF(Ŵ)` term
// (`ml2s_weight_correction_block`, zero for the NT weight, which lavaan
// robust.two.stage treats as fixed). Always the observed bread.
post_expected<Ml2sIjAssembly>
build_ml2s_ij_blocks(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     const FIMLPack& pack,
                     const FIMLH1& h1,
                     const SaturatedMoments& sm,
                     TwoStageWeight kind,
                     TwoStageDlsOptions dls, bool estimated_weight) {
  SampleStats samp = sample_stats_from_saturated(sm);
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "two_stage_em_ml_inference: fixed.x resolution"));
  }
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  Eigen::MatrixXd K = con_or->K();
  if (K.rows() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: constraint reparameterization has "
        "incompatible shape"));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: fitted theta length does not match "
        "partable"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval_or = ev_or->evaluate(est.theta, true, true);
  if (!eval_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ModelEvaluator::evaluate failed: " +
            eval_or.error().detail));
  }
  if (auto v = ml2s_validate_moment_shapes(samp, eval_or->moments);
      !v.has_value()) {
    return std::unexpected(v.error());
  }
  const Ml2sMomentLayout layout =
      ml2s_make_layout(samp, eval_or->moments);

  auto weight_or = two_stage_stage2_weight_structured(sm, kind, dls);
  if (!weight_or.has_value()) return std::unexpected(weight_or.error());
  auto influence_or = saturated_em_moment_influence(raw, pack, h1, sm);
  if (!influence_or.has_value()) return std::unexpected(influence_or.error());

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(samp.S.size());
  Eigen::Index row_off = 0;
  Eigen::Index col_off = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p = samp.S[b].rows();
    const Eigen::Index qb = p + vech_len(p);
    const Eigen::Index n = raw.X[b].rows();
    if (layout.block_rows[b] != qb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: ML2S estimated-weight IJ requires "
          "mean and covariance moment rows"));
    }
    if (row_off + n > influence_or->rows() ||
        col_off + qb > influence_or->cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: saturated influence shape mismatch"));
    }

    auto Jb = ml2s_moment_jacobian_block(
        layout, eval_or->moments, eval_or->J_sigma, eval_or->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = ml2s_weight_block(*weight_or, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    const Eigen::VectorXd residual =
        ml2s_block_residual(samp, eval_or->moments, layout, b);
    Eigen::MatrixXd moment_rows =
        static_cast<double>(n) *
        influence_or->block(row_off, col_off, n, qb);
    post_expected<Eigen::MatrixXd> correction_or = Eigen::MatrixXd{};
    if (estimated_weight) correction_or = ml2s_weight_correction_block(
        raw, pack.cache, samp, residual, moment_rows, *Wb, layout, b, kind,
        dls);
    if (!correction_or.has_value()) return std::unexpected(correction_or.error());

    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = std::move(*Jb),
        .weight = std::move(*Wb),
        .moment_influence = std::move(moment_rows),
        .weight_correction = std::move(*correction_or),
        .n_obs = samp.n_obs[b]});
    row_off += n;
    col_off += qb;
  }

  auto ob_or = ml2s_observed_bread(pt, rep, samp, est, *weight_or, K);
  if (!ob_or.has_value()) return std::unexpected(ob_or.error());
  return Ml2sIjAssembly{std::move(ij_blocks), std::move(K), std::move(*ob_or)};
}

post_expected<TwoStageEMMLInference>
two_stage_missing_data_weighted_ij_from_sm(spec::LatentStructure pt,
                                           const model::MatrixRep& rep,
                                           const RawData& raw,
                                           const Estimates& est,
                                           const FIMLPack& pack,
                                           const FIMLH1& h1,
                                           const SaturatedMoments& sm,
                                           TwoStageWeight kind,
                                           TwoStageDlsOptions dls,
                                           TwoStageBread bread) {
  if (bread != TwoStageBread::Observed || raw_has_no_missing_mask(raw) ||
      kind == TwoStageWeight::Nt || kind == TwoStageWeight::Uls) {
    return two_stage_em_weighted_inference_from_sm(
        std::move(pt), rep, est, sm, kind, dls, bread);
  }

  auto base_or = two_stage_em_weighted_inference_from_sm(
      pt, rep, est, sm, kind, dls, bread);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1, sm,
                                     kind, dls);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  auto ij_or = robust_weighted_moment_ij(
      asm_or->blocks, asm_or->K, 2.0 * est.fmin, asm_or->observed_bread);
  if (!ij_or.has_value()) return std::unexpected(ij_or.error());

  TwoStageEMMLInference out = std::move(*base_or);
  out.vcov = std::move(ij_or->vcov);
  out.se = std::move(ij_or->se);
  return out;
}

// Core two-stage ML inference from already-computed Stage-1 saturated moments.
// `raw` is intentionally absent: the only thing the inference ever needs from
// the data are the saturated moments and their ACOV, both carried by `sm`.
// Callers that hold a Stage-1 `SaturatedMoments` should route here directly
// rather than recomputing the EM + observed information + ACOV. `kind == Nt`
// is the normal-theory path; non-NT weights dispatch to the explicit-weight
// robust sandwich above.
post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference_from_sm(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const Estimates& est,
                                  const SaturatedMoments& sm,
                                  TwoStageWeight kind,
                                  TwoStageDlsOptions dls,
                                  TwoStageBread bread) {
  if (kind != TwoStageWeight::Nt) {
    return two_stage_em_weighted_inference_from_sm(std::move(pt), rep, est, sm,
                                                   kind, dls, bread);
  }
  SampleStats samp = sample_stats_from_saturated(sm);

  auto df_or = inference::df_stat(pt, samp, est.theta);
  if (!df_or.has_value()) return std::unexpected(df_or.error());

  // The Satorra-Bentler weight uses Unstructured (sample/saturated h1) moments,
  // not Structured (model-implied). lavaan's two.stage / robust.two.stage path
  // hard-forces h1.information = "unstructured", and magmaan's FIML FMG spectrum
  // follows the same convention; with Unstructured the SE and test-statistic
  // scaling match lavaan robust.two.stage to machine precision. Structured here
  // is a finite-sample inconsistency (off by a few percent under non-normality).
  auto gamma_se_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/true);
  if (!gamma_se_or.has_value()) return std::unexpected(gamma_se_or.error());
  auto se_or = robust::robust_se(
      pt, rep, samp, est, *gamma_se_or,
      robust::InferenceSpec{robust_bread(bread),
                            robust::WeightMoments::Unstructured,
                            robust::ScoreCovariance::Empirical});
  if (!se_or.has_value()) return std::unexpected(se_or.error());

  TwoStageEMMLInference out;
  out.vcov = std::move(se_or->vcov);
  out.se = std::move(se_or->se);
  out.df = *df_or;
  out.chisq = inference::chi2_stat(samp, est);
  out.ntotal = 0;
  for (auto n : samp.n_obs) out.ntotal += n;

  if (out.df <= 0) return out;

  auto gamma_test_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_test_or.has_value()) return std::unexpected(gamma_test_or.error());
  auto uf_or = robust::build_u_factor(
      std::move(pt), rep, samp, est,
      robust::InferenceSpec{robust_bread(bread),
                            robust::WeightMoments::Unstructured,
                            robust::ScoreCovariance::Empirical});
  if (!uf_or.has_value()) return std::unexpected(uf_or.error());
  if (static_cast<int>(uf_or->df) != out.df) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: U-factor df (" +
            std::to_string(uf_or->df) + ") != df_stat (" +
            std::to_string(out.df) + ")"));
  }
  auto M_or = robust::reduced_gamma_sample_from_gamma(*uf_or, *gamma_test_or);
  if (!M_or.has_value()) return std::unexpected(M_or.error());
  auto ev_or = robust::ugamma_eigenvalues(*M_or);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());

  out.eigvals = std::move(*ev_or);
  out.trace_ugamma = out.eigvals.sum();
  out.scaling_factor = out.trace_ugamma / static_cast<double>(out.df);
  out.chisq_scaled = (out.scaling_factor > 0.0)
      ? out.chisq / out.scaling_factor
      : std::numeric_limits<double>::quiet_NaN();
  return out;
}

post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference_impl(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const RawData& raw,
                               const Estimates& est,
                               const FIMLPack& pack,
                               const FIMLH1& h1,
                               TwoStageWeight kind,
                               TwoStageDlsOptions dls,
                               TwoStageBread bread) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  if (kind != TwoStageWeight::Nt && bread == TwoStageBread::Observed) {
    if (raw_has_no_missing_mask(raw)) {
      return two_stage_complete_data_weighted_ij_from_sm(
          std::move(pt), rep, raw, est, *sm_or, kind, dls, bread);
    }
    return two_stage_missing_data_weighted_ij_from_sm(
        std::move(pt), rep, raw, est, pack, h1, *sm_or, kind, dls, bread);
  }
  return two_stage_em_ml_inference_from_sm(std::move(pt), rep, est, *sm_or,
                                           kind, dls, bread);
}

}  // namespace

post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const Estimates& est,
                          const SaturatedMoments& sm,
                          TwoStageWeight kind,
                          TwoStageDlsOptions dls,
                          TwoStageBread bread) {
  return two_stage_em_ml_inference_from_sm(std::move(pt), rep, est, sm,
                                           kind, dls, bread);
}

post_expected<CasewiseInfluenceIJ>
two_stage_casewise_influence_ij(spec::LatentStructure pt,
                                const model::MatrixRep& rep,
                                const RawData& raw,
                                const Estimates& est,
                                const FIMLPack& pack,
                                const FIMLH1& h1,
                                TwoStageWeight kind,
                                TwoStageDlsOptions dls) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());

  // Mirror the SE dispatch: a non-NT Stage-2 weight on complete data routes
  // through the continuous-LS casewise accessor (the same route the SE takes to
  // `continuous_ls_*_ij`, so the per-case rows reproduce that vcov). Everything
  // else — non-NT missing data, and the NT weight (correction zero) — goes
  // through the shared IJ-block assembly with the observed bread.
  if (kind != TwoStageWeight::Nt && kind != TwoStageWeight::Uls &&
      raw_has_no_missing_mask(raw)) {
    SampleStats samp = sample_stats_from_saturated(*sm_or);
    ContinuousLsIJWeightMode mode = ContinuousLsIJWeightMode::SampleEmpiricalWls;
    if (kind == TwoStageWeight::Dwls)
      mode = ContinuousLsIJWeightMode::SampleEmpiricalDwls;
    else if (kind == TwoStageWeight::Dls)
      mode = ContinuousLsIJWeightMode::SampleDls;
    estimate::frontier::DlsWeightOptions opts;
    opts.a = dls.a;
    return continuous_ls_casewise_influence_ij(
        std::move(pt), rep, samp, est, gmm::Weight{}, raw, mode, opts);
  }

  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1,
                                     *sm_or, kind, dls);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  return casewise_influence_from_ij_blocks(asm_or->blocks, asm_or->K,
                                           asm_or->observed_bread);
}

post_expected<WeightedMomentRBMParts>
two_stage_rbm_parts(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const RawData& raw,
                    const Estimates& est,
                    const FIMLPack& pack,
                    const FIMLH1& h1,
                    TwoStageWeight kind,
                    TwoStageDlsOptions dls, bool estimated_weight) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1,
                                     *sm_or, kind, dls, estimated_weight);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  return weighted_moment_rbm_parts(asm_or->blocks, asm_or->K,
                                   asm_or->observed_bread);
}

post_expected<WeightedMomentRBMParts>
two_stage_rbm_parts(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const RawData& raw,
                    const Estimates& est,
                    const FIMLPack& pack,
                    const FIMLH1& h1,
                    const SaturatedMoments& sm,
                    TwoStageWeight kind,
                    TwoStageDlsOptions dls, bool estimated_weight) {
  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1,
                                     sm, kind, dls, estimated_weight);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  return weighted_moment_rbm_parts(asm_or->blocks, asm_or->K,
                                   asm_or->observed_bread);
}


}  // namespace magmaan::estimate::fiml
