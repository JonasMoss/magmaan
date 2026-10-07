#include "magmaan/api/conventions.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include <cmath>
#include <numeric>

namespace magmaan::api {
namespace {
using measures::BaselineFit;
using measures::RobustFitMeasureInputs;
using measures::RobustFitMeasures;
void add(ConventionFitMeasures& out, std::string name, double value) {
  out.indices.push_back({std::move(name), value, InferenceReason::Available, {}});
}
Error failure(std::string detail) { return make_error(ErrorStage::PostFit, std::move(detail)); }
Result<ConventionFitMeasures> refused(const PolicyFitState& state) {
  return std::unexpected(failure(state.penalized ? std::string(penalized_detail) :
      "the fit did not pass its convergence verdict"));
}
void family(ConventionFitMeasures& out, const measures::FitMeasures& fm, std::string suffix = "") {
  add(out, "cfi" + suffix, fm.cfi); add(out, "tli" + suffix, fm.tli);
  add(out, "rmsea" + suffix, fm.rmsea);
  add(out, "rmsea.ci.lower" + suffix, fm.rmsea_ci_lower);
  add(out, "rmsea.ci.upper" + suffix, fm.rmsea_ci_upper);
  add(out, "rmsea.pvalue" + suffix, fm.rmsea_pvalue);
  add(out, "rmsea.close.h0" + suffix, fm.rmsea_close_h0);
  add(out, "rmsea.notclose.pvalue" + suffix, fm.rmsea_notclose_pvalue);
  add(out, "rmsea.notclose.h0" + suffix, fm.rmsea_notclose_h0);
}
void standard(ConventionFitMeasures& out, const ConventionTest& user, const ConventionTest& base,
    std::int64_t n, std::size_t groups) {
  add(out, "chisq", user.unscaled_statistic); add(out, "df", user.df);
  add(out, "pvalue", inference::chi2_pvalue(user.unscaled_statistic, user.df));
  add(out, "baseline.chisq", base.unscaled_statistic); add(out, "baseline.df", base.df);
  add(out, "baseline.pvalue", inference::chi2_pvalue(base.unscaled_statistic, base.df));
  family(out, measures::fit_measures(user.unscaled_statistic, user.df,
      {base.unscaled_statistic, base.df}, n, groups));
  add(out, "ntotal", static_cast<double>(n));
}
void scaled(ConventionFitMeasures& out, const ConventionTest& u, const ConventionTest& b,
    const RobustFitMeasureInputs& in, bool shifted) {
  const auto rf = measures::robust_fit_measures(in);
  add(out, "chisq.scaled", u.statistic); add(out, "df.scaled", u.df);
  add(out, "pvalue.scaled", u.p_value); add(out, "chisq.scaling.factor", u.scale);
  add(out, "baseline.chisq.scaled", b.statistic); add(out, "baseline.df.scaled", b.df);
  add(out, "baseline.pvalue.scaled", b.p_value); add(out, "baseline.chisq.scaling.factor", b.scale);
  if (shifted) {
    add(out, "chisq.shift.parameter", u.shift); add(out, "baseline.chisq.shift.parameter", b.shift);
  }
  auto fm = measures::fit_measures(u.statistic, u.df, {b.statistic, b.df}, in.n_total, in.n_groups);
  // Non-shifted tests use trace(U Gamma) as RMSEA's df, while incremental
  // indices use the scaled statistic and nominal df.
  if (!shifted) {
    fm.rmsea = rf.rmsea_scaled; fm.rmsea_ci_lower = rf.rmsea_ci_lower_scaled;
    fm.rmsea_ci_upper = rf.rmsea_ci_upper_scaled; fm.rmsea_pvalue = rf.rmsea_pvalue_scaled;
    fm.rmsea_notclose_pvalue = rf.rmsea_notclose_pvalue_scaled;
  }
  family(out, fm, ".scaled");
}
void robust(ConventionFitMeasures& out, const RobustFitMeasureInputs& in) {
  const auto rf = measures::robust_fit_measures(in);
  measures::FitMeasures fm;
  fm.cfi = rf.cfi_robust; fm.tli = rf.tli_robust; fm.rmsea = rf.rmsea_robust;
  fm.rmsea_ci_lower = rf.rmsea_ci_lower_robust; fm.rmsea_ci_upper = rf.rmsea_ci_upper_robust;
  fm.rmsea_pvalue = rf.rmsea_pvalue_robust; fm.rmsea_notclose_pvalue = rf.rmsea_notclose_pvalue_robust;
  family(out, fm, ".robust");
}
RobustFitMeasureInputs inputs(const ConventionTest& u, const ConventionTest& b,
    std::int64_t n, std::size_t groups) {
  RobustFitMeasureInputs in;
  in.chi2 = u.unscaled_statistic; in.df = u.df; in.chi2_scaled = u.statistic; in.scaling_factor = u.scale;
  in.baseline_chi2 = b.unscaled_statistic; in.baseline_df = b.df;
  in.baseline_chi2_scaled = b.statistic; in.baseline_scaling_factor = b.scale;
  in.n_total = n; in.n_groups = groups;
  return in;
}
Result<Model> independence(Eigen::Index p, int groups, bool means,
    const std::vector<std::int32_t>* levels = nullptr) {
  std::string syntax;
  for (Eigen::Index j = 0; j < p; ++j) {
    const std::string v = "v" + std::to_string(j);
    syntax += v + " ~~ " + v + "\n";
    if (levels) {
      syntax += v + " | ";
      for (int k = 1; k < (*levels)[static_cast<std::size_t>(j)]; ++k) syntax += (k > 1 ? " + " : "") + std::string("t") + std::to_string(k);
      syntax += "\n";
    }
  }
  ModelOptions opts; opts.build.n_groups = groups;
  opts.build.meanstructure = means; opts.build.fixed_x = false;
  return model_from_lavaan(syntax, opts);
}
} // namespace

Result<ConventionFitMeasures> convention_fit_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::Estimates& est, LavaanConvention c, const PolicyFitState& state, bool missing) {
  if (state.penalized || !state.converged) return refused(state);
  if (c != LavaanConvention::ML && c != LavaanConvention::MLM && c != LavaanConvention::MLR)
    return std::unexpected(failure("this convention requires a different fitted estimator"));
  if (missing && c == LavaanConvention::MLM)
    return std::unexpected(failure("FIML compatibility covers ML and MLR"));
  if (raw.X.empty() || std::any_of(pt.exo.begin(), pt.exo.end(), [](auto x) { return x != 0; }))
    return std::unexpected(failure("fit-measures compatibility requires raw observations and random x"));
  auto evaluator = model::ModelEvaluator::build(pt, rep);
  if (!evaluator) return std::unexpected(failure(evaluator.error().detail));
  auto eval = evaluator->evaluate(est.theta, true, true);
  if (!eval) return std::unexpected(failure(eval.error().detail));
  const bool means = eval->J_mu.rows() > 0;
  auto pack = estimate::fiml::fiml_pack(raw);
  if (!pack) return std::unexpected(failure(pack.error().detail));
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack);
  if (!h1) return std::unexpected(failure(h1.error().detail));
  auto model = independence(raw.X[0].cols(), static_cast<int>(raw.X.size()), means);
  if (!model) return std::unexpected(model.error());
  auto data = missing ? data_from_raw(*model, raw) : data_from_sample_stats(*model,
      {h1->sigma, means ? h1->mu : std::vector<Eigen::VectorXd>{}, pack->start_stats.n_obs});
  if (!data) return std::unexpected(data.error());
  auto base = fit(*model, *data, missing ? fiml() : ml());
  if (!base) return std::unexpected(base.error());
  const auto bs = policy_fit_state(base->estimates());
  if (!bs.converged) return std::unexpected(failure("independence baseline did not converge"));
  ConventionInference user_test, base_test;
  data::SampleStats sample{h1->sigma, means ? h1->mu : std::vector<Eigen::VectorXd>{}, pack->start_stats.n_obs};
  if (missing) {
    user_test = lavaan_inference_fiml(pt, rep, raw, *pack, est, c, state);
    base_test = lavaan_inference_fiml(model->structure(), model->matrix_rep(), raw, *pack, base->estimates(), c, bs);
  } else {
    auto prepared_data = robust::frontier::prepare_ntml_data(raw, means);
    if (!prepared_data) return std::unexpected(failure(prepared_data.error().detail));
    auto u = robust::frontier::prepare_ntml_fit(*prepared_data, pt, rep, est);
    auto b = robust::frontier::prepare_ntml_fit(*prepared_data, model->structure(), model->matrix_rep(), base->estimates());
    if (!u) return std::unexpected(failure(u.error().detail));
    if (!b) return std::unexpected(failure(b.error().detail));
    user_test = lavaan_inference_ml(**u, c, state); base_test = lavaan_inference_ml(**b, c, bs);
  }
  if (user_test.test.reason != InferenceReason::Available) return std::unexpected(failure(user_test.test.detail));
  if (base_test.test.reason != InferenceReason::Available) return std::unexpected(failure(base_test.test.detail));
  const auto& u = user_test.test; const auto& b = base_test.test;
  ConventionFitMeasures out; out.convention = convention_name(c);
  standard(out, u, b, pack->cache.n_total, raw.X.size());
  if (missing) {
    auto fx = estimate::fiml::fiml_extras(pt, rep, raw, est, *pack, *h1);
    if (!fx) return std::unexpected(failure(fx.error().detail));
    add(out, "srmr", fx->srmr); add(out, "logl", fx->logl); add(out, "unrestricted.logl", fx->unrestricted_logl);
    add(out, "aic", fx->aic); add(out, "bic", fx->bic); add(out, "bic2", fx->bic2); add(out, "npar", fx->npar);
  } else {
    auto fx = measures::fit_extras(pt, rep, sample, est);
    if (!fx) return std::unexpected(failure(fx.error().detail));
    add(out, "srmr", fx->srmr); add(out, "logl", fx->logl); add(out, "unrestricted.logl", fx->unrestricted_logl);
    add(out, "aic", fx->aic); add(out, "bic", fx->bic); add(out, "bic2", fx->bic2); add(out, "npar", fx->npar);
  }
  if (c != LavaanConvention::ML) {
    auto in = inputs(u, b, pack->cache.n_total, raw.X.size());
    scaled(out, u, b, in, false);
    if (missing) {
      auto corrected = estimate::fiml::fiml_corrected_fit_measures(pt, rep, raw, est, u.df, *pack, *h1);
      if (!corrected) return std::unexpected(failure(corrected.error().detail));
      in.chi2 = corrected->xx3; in.df = corrected->df3;
      in.chi2_scaled = corrected->xx3_scaled; in.scaling_factor = corrected->c_hat3;
      in.baseline_chi2 = corrected->xx3_null; in.baseline_df = corrected->df3_null;
      in.baseline_chi2_scaled = corrected->xx3_null_scaled; in.baseline_scaling_factor = corrected->c_hat3_null;
    }
    robust(out, in);
  }
  return out;
}

Result<ConventionFitMeasures> convention_fit_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const estimate::Estimates& est, estimate::OrdinalWeightKind weight,
    estimate::OrdinalParameterization parameterization,
    LavaanConvention c, const PolicyFitState& state) {
  if (state.penalized || !state.converged) return refused(state);
  auto user = lavaan_inference_ordinal(pt, rep, stats, est, weight, parameterization, c, state);
  if (user.test.reason != InferenceReason::Available) return std::unexpected(failure(user.test.detail));
  if (stats.R.empty()) return std::unexpected(failure("empty ordinal moments"));
  auto model = independence(stats.R[0].rows(), static_cast<int>(stats.R.size()), true, &stats.n_levels[0]);
  if (!model) return std::unexpected(model.error());
  auto data = data_from_ordinal(*model, stats);
  if (!data) return std::unexpected(data.error());
  auto estimator = weight == estimate::OrdinalWeightKind::ULS ? ordinal_dwls() :
      weight == estimate::OrdinalWeightKind::WLS ? ordinal_wls() : ordinal_dwls();
  estimator.ordinal_weight = weight;
  estimator = estimator.parameterization(parameterization);
  auto base = fit(*model, *data, estimator);
  if (!base) return std::unexpected(base.error());
  const auto bs = policy_fit_state(base->estimates());
  if (!bs.converged) return std::unexpected(failure("ordinal independence baseline did not converge"));
  auto baseline = lavaan_inference_ordinal(model->structure(), model->matrix_rep(), stats,
      base->estimates(), weight, parameterization, c, bs);
  if (baseline.test.reason != InferenceReason::Available) return std::unexpected(failure(baseline.test.detail));
  const auto n = std::accumulate(stats.n_obs.begin(), stats.n_obs.end(), std::int64_t{0});
  ConventionFitMeasures out; out.convention = convention_name(c);
  standard(out, user.test, baseline.test, n - static_cast<std::int64_t>(stats.R.size()), stats.R.size());
  if (c != LavaanConvention::WLS) {
    for (auto& index : out.indices) if (index.index == "pvalue" || index.index == "baseline.pvalue")
      index.estimate = std::numeric_limits<double>::quiet_NaN();
  }
  // Residual indices retain the original moments/counts, not the reporting n-1.
  auto fm = estimate::fit_measures_ordinal(pt, rep, stats, est, weight, parameterization);
  if (!fm) return std::unexpected(failure(fm.error().detail));
  add(out, "srmr", fm->srmr);
  if (c == LavaanConvention::WLSMV || c == LavaanConvention::ULSMV) {
    auto in = inputs(user.test, baseline.test, n - static_cast<std::int64_t>(stats.R.size()), stats.R.size());
    scaled(out, user.test, baseline.test, in, true);
    auto cat_stats = stats;
    if (weight == estimate::OrdinalWeightKind::ULS)
      for (auto& w : cat_stats.W_dwls) w.setIdentity();
    auto cat = estimate::catml_dwls_rmsea_ordinal(pt, rep, cat_stats, est, parameterization);
    if (!cat) return std::unexpected(failure(cat.error().detail));
    auto cat_base = estimate::catml_dwls_rmsea_ordinal(model->structure(), model->matrix_rep(), cat_stats,
        base->estimates(), parameterization);
    if (!cat_base) return std::unexpected(failure(cat_base.error().detail));
    in.n_total = n;
    in.chi2 = cat->xx3; in.df = cat->df3; in.scaling_factor = cat->c_hat3;
    in.chi2_scaled = cat->xx3_scaled; in.baseline_chi2 = cat_base->xx3;
    in.baseline_df = cat_base->df3; in.baseline_scaling_factor = cat_base->c_hat3;
    in.baseline_chi2_scaled = cat_base->xx3_scaled;
    robust(out, in);
  }
  return out;
}

Result<ConventionFitMeasures> convention_fit_measures(const Fit& fitted, LavaanConvention c) {
  if (const auto* raw = fitted.data().raw())
    return convention_fit_measures(fitted.model().structure(), fitted.model().matrix_rep(), *raw,
        fitted.estimates(), c, policy_fit_state(fitted.estimates()), fitted.estimator() == EstimatorKind::FIML);
  if (const auto* stats = fitted.data().ordinal())
    return convention_fit_measures(fitted.model().structure(), fitted.model().matrix_rep(), *stats,
        fitted.estimates(), fitted.estimator_spec().ordinal_weight, fitted.estimator_spec().ordinal_parameterization,
        c, policy_fit_state(fitted.estimates()));
  return std::unexpected(failure("this fitted data/estimator composition has no convention fit-measures adapter"));
}
} // namespace magmaan::api
