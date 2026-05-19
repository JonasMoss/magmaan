#include "magmaan/api/sem.hpp"

#include <cmath>
#include <limits>
#include <numeric>

#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::api {

namespace {

Result<Eigen::VectorXd> start_values(const spec::LatentStructure &pt,
                                     const model::MatrixRep &rep,
                                     const data::SampleStats &stats,
                                     const spec::Starts &starts,
                                     const StartSpec &spec) {
  if (spec.kind == StartKind::Explicit) {
    if (spec.theta.size() != pt.n_free()) {
      return std::unexpected(make_error(
          ErrorStage::Fit,
          "explicit start vector length does not match model n_free"));
    }
    return spec.theta;
  }

  fit_expected<Eigen::VectorXd> out;
  switch (spec.kind) {
  case StartKind::Simple:
    out = estimate::simple_start_values(pt, rep, stats, starts);
    break;
  case StartKind::Fabin:
    out = estimate::fabin_start_values(pt, rep, stats, starts);
    break;
  case StartKind::Guttman:
    out = estimate::guttman_start_values(pt, rep, stats, starts);
    break;
  case StartKind::Bentler1982:
    out = estimate::bentler1982_start_values(pt, rep, stats, starts);
    break;
  case StartKind::JamesStein:
    out = estimate::jamesstein_start_values(pt, rep, stats, starts);
    break;
  case StartKind::Explicit:
    break;
  }
  if (!out) {
    return std::unexpected(make_error(ErrorStage::Fit, out.error()));
  }
  return *out;
}

Result<Eigen::VectorXd> ordinal_start_values(const Model &model,
                                             const data::OrdinalStats &stats,
                                             const EstimatorSpec &spec) {
  if (spec.start_spec.kind == StartKind::Explicit) {
    return spec.start_spec.theta;
  }
  auto x0 = estimate::ordinal_start_values(
      model.structure(), model.matrix_rep(), stats, model.starts());
  if (!x0) {
    return std::unexpected(make_error(ErrorStage::Fit, x0.error()));
  }
  return *x0;
}

Result<Eigen::VectorXd> mixed_ordinal_start_values(
    const Model &model, const data::MixedOrdinalStats &stats,
    const EstimatorSpec &spec) {
  if (spec.start_spec.kind == StartKind::Explicit) {
    return spec.start_spec.theta;
  }
  auto x0 = estimate::mixed_ordinal_start_values(
      model.structure(), model.matrix_rep(), stats, model.starts());
  if (!x0) {
    return std::unexpected(make_error(ErrorStage::Fit, x0.error()));
  }
  return *x0;
}

Result<estimate::Bounds> bounds_for(const spec::LatentStructure &pt,
                                    const EstimatorSpec &spec) {
  switch (spec.bounds_mode) {
  case BoundsMode::None:
    return estimate::Bounds{};
  case BoundsMode::Explicit:
    return spec.bounds;
  case BoundsMode::AutoVariance: {
    auto bounds = estimate::bounds_from_partable(pt);
    if (!bounds) {
      return std::unexpected(make_error(ErrorStage::Fit, bounds.error()));
    }
    return *bounds;
  }
  }
  return estimate::Bounds{};
}

Result<estimate::Bounds> ordinal_bounds_for(const EstimatorSpec &spec) {
  if (spec.bounds_mode == BoundsMode::Explicit) {
    return spec.bounds;
  }
  return estimate::Bounds{};
}

estimate::Backend backend_from(const OptimizerSpec &optimizer) {
  return optimizer.kind == OptimizerKind::Ceres ? estimate::Backend::Ceres
                                                : estimate::Backend::Lbfgs;
}

bool is_continuous_ls(EstimatorKind kind) noexcept {
  return kind == EstimatorKind::ULS || kind == EstimatorKind::GLS ||
         kind == EstimatorKind::WLS;
}

bool is_ordinal_estimator(const EstimatorSpec &spec) noexcept {
  return spec.kind == EstimatorKind::DWLS ||
         (spec.kind == EstimatorKind::WLS && spec.weight.empty());
}

Result<estimate::gmm::Weight> ls_weight_for_fit(const Fit &fit) {
  const auto *stats = fit.data().sample_stats();
  if (!stats) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "continuous least-squares post-fit call requires sample statistics"));
  }

  switch (fit.estimator()) {
  case EstimatorKind::ULS:
    return estimate::gmm::Weight{};
  case EstimatorKind::WLS:
    if (fit.estimator_spec().weight.empty()) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "continuous WLS post-fit call requires the fitted WLS weight"));
    }
    return fit.estimator_spec().weight;
  case EstimatorKind::GLS: {
    auto ev = model::ModelEvaluator::build(fit.model().structure(),
                                           fit.model().matrix_rep());
    if (!ev) {
      return std::unexpected(make_error(ErrorStage::PostFit, ev.error()));
    }
    auto weight =
        estimate::gmm::normal_theory_weight(*ev, *stats, fit.estimates().theta);
    if (!weight) {
      return std::unexpected(make_error(ErrorStage::PostFit, weight.error()));
    }
    return *weight;
  }
  default:
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "post-fit call requires a continuous ULS/GLS/WLS fit"));
  }
}

Result<int> fiml_df(const Fit &fit, const data::RawData &raw) {
  auto stats = estimate::fiml::fiml_start_sample_stats(raw);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  auto df = inference::df_stat(fit.model().structure(), *stats,
                               fit.estimates().theta);
  if (!df) {
    return std::unexpected(make_error(ErrorStage::PostFit, df.error()));
  }
  return *df;
}

double total_n(const std::vector<std::int64_t> &n_obs) noexcept {
  double out = 0.0;
  for (const std::int64_t n : n_obs) {
    out += static_cast<double>(n);
  }
  return out;
}

std::vector<double> block_weights(const std::vector<std::int64_t> &n_obs) {
  const double n_total = total_n(n_obs);
  std::vector<double> out;
  out.reserve(n_obs.size());
  for (const std::int64_t n : n_obs) {
    out.push_back(n_total > 0.0 ? static_cast<double>(n) / n_total : 0.0);
  }
  return out;
}

std::vector<std::int32_t> block_n32(const std::vector<std::int64_t> &n_obs) {
  std::vector<std::int32_t> out;
  out.reserve(n_obs.size());
  for (const std::int64_t n : n_obs) {
    out.push_back(static_cast<std::int32_t>(n));
  }
  return out;
}

Result<void> require_complete_ml(const Fit &fit, std::string_view call) {
  if (fit.estimator() != EstimatorKind::ML || !fit.data().sample_stats()) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        std::string(call) + " currently requires a complete-data ML fit"));
  }
  return {};
}

Result<void> require_not_ordinal(const Fit &fit, std::string_view call) {
  if (fit.data().ordinal() || fit.data().mixed_ordinal()) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        std::string(call) +
            " is not exposed for ordinal or mixed-ordinal fits"));
  }
  return {};
}

template <class T>
Result<T> post_result(post_expected<T> result) {
  if (!result) {
    return std::unexpected(make_error(ErrorStage::PostFit, result.error()));
  }
  return *result;
}

} // namespace

Error make_error(ErrorStage stage, std::string detail) {
  return Error{stage, std::move(detail), std::monostate{}};
}

Error make_error(ErrorStage stage, ParseError error) {
  return Error{stage, error.detail, std::move(error)};
}

Error make_error(ErrorStage stage, PartableError error) {
  return Error{stage, error.detail, std::move(error)};
}

Error make_error(ErrorStage stage, ModelError error) {
  return Error{stage, error.detail, std::move(error)};
}

Error make_error(ErrorStage stage, FitError error) {
  return Error{stage, error.detail, std::move(error)};
}

Error make_error(ErrorStage stage, PostError error) {
  return Error{stage, error.detail, std::move(error)};
}

Result<Model> Model::from_lavaan(std::string_view syntax,
                                 ModelOptions options) {
  auto flat = parse::Parser::parse(syntax);
  if (!flat) {
    return std::unexpected(make_error(ErrorStage::Parse, flat.error()));
  }

  spec::Starts starts;
  spec::LatentNames names;
  auto structure = spec::build(*flat, options.build, &starts, &names);
  if (!structure) {
    return std::unexpected(make_error(ErrorStage::Model, structure.error()));
  }

  auto rep = model::build_matrix_rep(*structure, &names);
  if (!rep) {
    return std::unexpected(make_error(ErrorStage::Model, rep.error()));
  }

  return Model(std::string(syntax), std::move(*flat), std::move(*structure),
               std::move(names), std::move(starts), std::move(*rep), options);
}

Model::Model(std::string source, parse::FlatPartable flat,
             spec::LatentStructure structure, spec::LatentNames names,
             spec::Starts starts, model::MatrixRep rep, ModelOptions options)
    : source_(std::move(source)),
      flat_(std::make_shared<parse::FlatPartable>(std::move(flat))),
      structure_(std::move(structure)), names_(std::move(names)),
      starts_(std::move(starts)), rep_(std::move(rep)),
      options_(std::move(options)) {}

Result<Model> model_from_lavaan(std::string_view syntax,
                                ModelOptions options) {
  return Model::from_lavaan(syntax, std::move(options));
}

Data Data::from_sample_stats(data::SampleStats stats) {
  return Data(std::move(stats));
}

Data Data::from_raw(data::RawData raw) { return Data(std::move(raw)); }

Data Data::from_ordinal(data::OrdinalStats stats) {
  return Data(std::move(stats));
}

Data Data::from_mixed_ordinal(data::MixedOrdinalStats stats) {
  return Data(std::move(stats));
}

DataKind Data::kind() const noexcept {
  switch (storage_.index()) {
  case 0:
    return DataKind::SampleStats;
  case 1:
    return DataKind::RawContinuous;
  case 2:
    return DataKind::Ordinal;
  default:
    return DataKind::MixedOrdinal;
  }
}

const data::SampleStats *Data::sample_stats() const noexcept {
  return std::get_if<data::SampleStats>(&storage_);
}

const data::RawData *Data::raw() const noexcept {
  return std::get_if<data::RawData>(&storage_);
}

const data::OrdinalStats *Data::ordinal() const noexcept {
  return std::get_if<data::OrdinalStats>(&storage_);
}

const data::MixedOrdinalStats *Data::mixed_ordinal() const noexcept {
  return std::get_if<data::MixedOrdinalStats>(&storage_);
}

Data::Data(Storage storage) : storage_(std::move(storage)) {}

Result<Data> data_from_sample_stats(const Model &, data::SampleStats stats) {
  return Data::from_sample_stats(std::move(stats));
}

Result<Data> data_from_raw(const Model &, data::RawData raw) {
  return Data::from_raw(std::move(raw));
}

Result<Data> data_from_ordinal(const Model &, data::OrdinalStats stats) {
  return Data::from_ordinal(std::move(stats));
}

Result<Data> data_from_mixed_ordinal(const Model &,
                                     data::MixedOrdinalStats stats) {
  return Data::from_mixed_ordinal(std::move(stats));
}

namespace frontier {

Result<Data> data_from_ordinal_h_weighted(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalHWeightedStatsOptions options) {
  auto stats =
      data::pairwise_ordinal_stats_h_weighted_from_integer_data(blocks,
                                                                options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_ordinal(std::move(stats->stats));
}

Result<Data> data_from_ordinal_dpd(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalDpdStatsOptions options) {
  auto stats =
      data::pairwise_ordinal_stats_dpd_from_integer_data(blocks, options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_ordinal(std::move(stats->stats));
}

Result<Data> data_from_ordinal_huber_residual(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalHuberResidualStatsOptions options) {
  auto stats =
      data::pairwise_ordinal_stats_huber_residual_from_integer_data(blocks,
                                                                    options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_ordinal(std::move(stats->stats));
}

Result<Data> data_from_mixed_ordinal_polyserial_dpd(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    const std::vector<std::vector<std::int32_t>> &ordered,
    data::PolyserialPairDpdOptions options) {
  auto stats =
      data::mixed_ordinal_stats_polyserial_dpd_from_data(blocks, ordered,
                                                         options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_mixed_ordinal(std::move(stats->stats));
}

Result<Data> data_from_mixed_ordinal_huber_residual(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    const std::vector<std::vector<std::int32_t>> &ordered,
    data::MixedOrdinalHuberResidualOptions options) {
  auto stats =
      data::mixed_ordinal_stats_huber_residual_from_data(blocks, ordered,
                                                         options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_mixed_ordinal(std::move(stats->stats));
}

} // namespace frontier

EstimatorSpec EstimatorSpec::optimizer(OptimizerSpec optimizer) const {
  auto out = *this;
  out.optimizer_spec = std::move(optimizer);
  return out;
}

EstimatorSpec EstimatorSpec::starts(StartSpec start) const {
  auto out = *this;
  out.start_spec = std::move(start);
  return out;
}

EstimatorSpec EstimatorSpec::with_bounds(estimate::Bounds b) const {
  auto out = *this;
  out.bounds_mode = BoundsMode::Explicit;
  out.bounds = std::move(b);
  return out;
}

EstimatorSpec EstimatorSpec::auto_variance_bounds() const {
  auto out = *this;
  out.bounds_mode = BoundsMode::AutoVariance;
  out.bounds = {};
  return out;
}

EstimatorSpec EstimatorSpec::parameterization(
    estimate::OrdinalParameterization parameterization) const {
  auto out = *this;
  out.ordinal_parameterization = parameterization;
  return out;
}

EstimatorSpec ml() {
  EstimatorSpec out;
  out.kind = EstimatorKind::ML;
  return out;
}

EstimatorSpec fiml() {
  EstimatorSpec out;
  out.kind = EstimatorKind::FIML;
  return out;
}

EstimatorSpec uls() {
  EstimatorSpec out;
  out.kind = EstimatorKind::ULS;
  return out;
}

EstimatorSpec gls() {
  EstimatorSpec out;
  out.kind = EstimatorKind::GLS;
  return out;
}

EstimatorSpec wls(estimate::gmm::Weight weight) {
  EstimatorSpec out;
  out.kind = EstimatorKind::WLS;
  out.weight = std::move(weight);
  return out;
}

EstimatorSpec ordinal_dwls() {
  EstimatorSpec out;
  out.kind = EstimatorKind::DWLS;
  out.ordinal_weight = estimate::OrdinalWeightKind::DWLS;
  return out;
}

EstimatorSpec ordinal_wls() {
  EstimatorSpec out;
  out.kind = EstimatorKind::WLS;
  out.ordinal_weight = estimate::OrdinalWeightKind::WLS;
  return out;
}

EstimatorSpec dwls() { return ordinal_dwls(); }

InformationSpec expected_information() {
  return InformationSpec{InformationKind::Expected, 1e-4};
}

InformationSpec observed_information_fd(double h_step) {
  return InformationSpec{InformationKind::ObservedFiniteDifference, h_step};
}

InformationSpec observed_information_analytic() {
  return InformationSpec{InformationKind::ObservedAnalytic, 1e-4};
}

Fit::Fit(std::shared_ptr<const Model> model, std::shared_ptr<const Data> data,
         estimate::Estimates estimates, EstimatorSpec estimator)
    : model_(std::move(model)), data_(std::move(data)),
      estimates_(std::move(estimates)), estimator_(std::move(estimator)) {}

Result<Fit> fit(std::shared_ptr<const Model> model,
                std::shared_ptr<const Data> data, EstimatorSpec estimator) {
  if (!model || !data) {
    return std::unexpected(
        make_error(ErrorStage::Fit, "fit requires non-null model and data"));
  }

  spec::LatentStructure pt = model->structure();
  const auto &rep = model->matrix_rep();

  if (estimator.kind == EstimatorKind::FIML) {
    const auto *raw = data->raw();
    if (!raw) {
      return std::unexpected(make_error(ErrorStage::UnsupportedCombination,
                                        "FIML requires raw continuous data"));
    }
    if (estimator.bounds_mode != BoundsMode::None) {
      return std::unexpected(
          make_error(ErrorStage::UnsupportedCombination,
                     "FIML facade fitting does not accept bounds"));
    }
    if (estimator.optimizer_spec.kind != OptimizerKind::Lbfgs) {
      return std::unexpected(
          make_error(ErrorStage::UnsupportedCombination,
                     "FIML currently supports only the L-BFGS optimizer"));
    }

    auto start_stats = estimate::fiml::fiml_start_sample_stats(*raw);
    if (!start_stats) {
      return std::unexpected(make_error(ErrorStage::Data, start_stats.error()));
    }
    auto x0 = start_values(pt, rep, *start_stats, model->starts(),
                           estimator.start_spec);
    if (!x0) {
      return std::unexpected(x0.error());
    }

    auto est = estimate::fiml::fit_fiml(pt, rep, *raw, *x0, {},
                                        estimator.optimizer_spec.lbfgs);
    if (!est) {
      return std::unexpected(make_error(ErrorStage::Fit, est.error()));
    }
    return Fit(std::move(model), std::move(data), std::move(*est),
               std::move(estimator));
  }

  if (const auto *stats = data->ordinal()) {
    if (!is_ordinal_estimator(estimator)) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "ordinal data require ordinal_dwls(), ordinal_wls(), or dwls()"));
    }
    auto x0 = ordinal_start_values(*model, *stats, estimator);
    if (!x0) {
      return std::unexpected(x0.error());
    }
    auto bounds = ordinal_bounds_for(estimator);
    if (!bounds) {
      return std::unexpected(bounds.error());
    }
    auto est = estimate::fit_ordinal_bounded(
        pt, rep, *stats, *bounds, estimator.ordinal_weight, *x0,
        backend_from(estimator.optimizer_spec), estimator.optimizer_spec.lbfgs,
        estimator.ordinal_parameterization);
    if (!est) {
      return std::unexpected(make_error(ErrorStage::Fit, est.error()));
    }
    return Fit(std::move(model), std::move(data), std::move(*est),
               std::move(estimator));
  }

  if (const auto *stats = data->mixed_ordinal()) {
    if (!is_ordinal_estimator(estimator)) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "mixed ordinal data require ordinal_dwls(), ordinal_wls(), or dwls()"));
    }
    if (estimator.ordinal_parameterization ==
        estimate::OrdinalParameterization::Theta) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "mixed ordinal theta parameterization is not supported"));
    }
    auto x0 = mixed_ordinal_start_values(*model, *stats, estimator);
    if (!x0) {
      return std::unexpected(x0.error());
    }
    auto bounds = ordinal_bounds_for(estimator);
    if (!bounds) {
      return std::unexpected(bounds.error());
    }
    auto est = estimate::fit_mixed_ordinal_bounded(
        pt, rep, *stats, *bounds, estimator.ordinal_weight, *x0,
        backend_from(estimator.optimizer_spec), estimator.optimizer_spec.lbfgs,
        estimator.ordinal_parameterization);
    if (!est) {
      return std::unexpected(make_error(ErrorStage::Fit, est.error()));
    }
    return Fit(std::move(model), std::move(data), std::move(*est),
               std::move(estimator));
  }

  const auto *stats = data->sample_stats();
  if (!stats) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "continuous ML/ULS/GLS/WLS require complete-data sample statistics"));
  }
  if (estimator.kind == EstimatorKind::DWLS) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "DWLS requires ordinal or mixed ordinal data"));
  }

  auto x0 =
      start_values(pt, rep, *stats, model->starts(), estimator.start_spec);
  if (!x0) {
    return std::unexpected(x0.error());
  }

  auto bounds = bounds_for(pt, estimator);
  if (!bounds) {
    return std::unexpected(bounds.error());
  }

  fit_expected<estimate::Estimates> est;
  switch (estimator.kind) {
  case EstimatorKind::ML:
    if (estimator.optimizer_spec.kind != OptimizerKind::Lbfgs) {
      return std::unexpected(
          make_error(ErrorStage::UnsupportedCombination,
                     "ML currently supports only the L-BFGS optimizer"));
    }
    est = estimate::fit_ml(pt, rep, *stats, *x0, *bounds,
                           estimate::Backend::Lbfgs,
                           estimator.optimizer_spec.lbfgs);
    break;
  case EstimatorKind::ULS:
    est = estimate::fit_gmm(pt, rep, *stats, *x0, {}, *bounds,
                            backend_from(estimator.optimizer_spec),
                            estimator.optimizer_spec.lbfgs);
    break;
  case EstimatorKind::GLS:
    est = estimate::fit_gls(pt, rep, *stats, *x0, *bounds,
                            backend_from(estimator.optimizer_spec),
                            estimator.optimizer_spec.lbfgs);
    break;
  case EstimatorKind::WLS:
    if (estimator.weight.empty()) {
      return std::unexpected(
          make_error(ErrorStage::UnsupportedCombination,
                     "continuous WLS requires an explicit weight matrix"));
    }
    est = estimate::fit_gmm(pt, rep, *stats, *x0, estimator.weight, *bounds,
                            backend_from(estimator.optimizer_spec),
                            estimator.optimizer_spec.lbfgs);
    break;
  case EstimatorKind::FIML:
  case EstimatorKind::DWLS:
    break;
  }

  if (!est) {
    return std::unexpected(make_error(ErrorStage::Fit, est.error()));
  }
  return Fit(std::move(model), std::move(data), std::move(*est),
             std::move(estimator));
}

Result<Fit> fit(const Model &model, const Data &data,
                EstimatorSpec estimator) {
  return fit(std::make_shared<Model>(model), std::make_shared<Data>(data),
             std::move(estimator));
}

TestSpec standard_chi_square() { return TestSpec{}; }

Result<StandardErrors> standard_errors(const Fit &fit, InformationSpec spec) {
  // The non-robust (information-inverse) standard errors. The information
  // matrix is estimator-specific; `inference::vcov` then inverts it and folds
  // in any equality constraints — the one shared post-step. Robust SEs are the
  // separate fiml_robust_mlr() / robust_continuous_ls() / robust_ordinal()
  // accessors.
  post_expected<Eigen::MatrixXd> info;
  if (fit.estimator() == EstimatorKind::ML && fit.data().sample_stats()) {
    const auto *stats = fit.data().sample_stats();
    switch (spec.kind) {
    case InformationKind::Expected:
      info = inference::information_expected(fit.model().structure(),
                                             fit.model().matrix_rep(), *stats,
                                             fit.estimates());
      break;
    case InformationKind::ObservedFiniteDifference:
      info = inference::information_observed_fd(fit.model().structure(),
                                                fit.model().matrix_rep(),
                                                *stats, fit.estimates(),
                                                spec.h_step);
      break;
    case InformationKind::ObservedAnalytic:
      info = inference::information_observed_analytic(fit.model().structure(),
                                                      fit.model().matrix_rep(),
                                                      *stats, fit.estimates());
      break;
    }
  } else if (fit.estimator() == EstimatorKind::FIML && fit.data().raw()) {
    // FIML exposes a single information notion — the observed −∂²logl/∂θ².
    // `spec.kind` does not apply; `spec.h_step` tunes the FD Hessian.
    info = estimate::fiml::fiml_observed_information(
        fit.model().structure(), fit.model().matrix_rep(), *fit.data().raw(),
        fit.estimates(), {}, spec.h_step);
  } else if (is_continuous_ls(fit.estimator()) && fit.data().sample_stats()) {
    auto weight = ls_weight_for_fit(fit);
    if (!weight) {
      return std::unexpected(weight.error());
    }
    info = estimate::ls_information(fit.model().structure(),
                                    fit.model().matrix_rep(),
                                    *fit.data().sample_stats(),
                                    fit.estimates(), *weight);
  } else {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "standard_errors() supports complete-data ML, FIML, and continuous "
        "least-squares fits"));
  }
  if (!info) {
    return std::unexpected(make_error(ErrorStage::PostFit, info.error()));
  }

  auto vc = inference::vcov(*info, fit.model().structure(),
                            fit.estimates().theta);
  if (!vc) {
    return std::unexpected(make_error(ErrorStage::PostFit, vc.error()));
  }

  return StandardErrors{std::move(*info), *vc, inference::se(*vc)};
}

Result<estimate::fiml::FIMLRobustMLR> fiml_robust_mlr(const Fit &fit,
                                                      double h_step) {
  const auto *raw = fit.data().raw();
  if (fit.estimator() != EstimatorKind::FIML || !raw) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "fiml_robust_mlr() requires a FIML fit over raw continuous data"));
  }
  auto t = test(fit, standard_chi_square());
  if (!t) {
    return std::unexpected(t.error());
  }
  auto robust = estimate::fiml::fiml_robust_mlr(
      fit.model().structure(), fit.model().matrix_rep(), *raw, fit.estimates(),
      t->df, t->statistic, {}, h_step);
  return post_result(std::move(robust));
}

Result<robust::RobustSeResult>
robust_se(const Fit &fit, const data::RawData &raw,
          robust::InferenceSpec spec) {
  auto ok = require_complete_ml(fit, "robust_se()");
  if (!ok) {
    return std::unexpected(ok.error());
  }
  auto robust = robust::robust_se(fit.model().structure(),
                                  fit.model().matrix_rep(),
                                  *fit.data().sample_stats(), fit.estimates(),
                                  raw, spec);
  return post_result(std::move(robust));
}

Result<robust::RobustSeResult>
robust_se(const Fit &fit, const Eigen::MatrixXd &gamma,
          robust::InferenceSpec spec) {
  auto ok = require_complete_ml(fit, "robust_se()");
  if (!ok) {
    return std::unexpected(ok.error());
  }
  auto robust = robust::robust_se(fit.model().structure(),
                                  fit.model().matrix_rep(),
                                  *fit.data().sample_stats(), fit.estimates(),
                                  gamma, spec);
  return post_result(std::move(robust));
}

Result<estimate::WeightedRobustResult>
robust_continuous_ls(const Fit &fit, const data::RawData &raw) {
  if (!is_continuous_ls(fit.estimator()) || !fit.data().sample_stats()) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "robust_continuous_ls() requires a continuous ULS/GLS/WLS fit"));
  }
  auto weight = ls_weight_for_fit(fit);
  if (!weight) {
    return std::unexpected(weight.error());
  }
  auto robust = estimate::robust_continuous_ls(
      fit.model().structure(), fit.model().matrix_rep(),
      *fit.data().sample_stats(), fit.estimates(), *weight, raw);
  return post_result(std::move(robust));
}

Result<estimate::WeightedRobustResult>
robust_continuous_ls(const Fit &fit,
                     const std::vector<Eigen::MatrixXd> &gamma) {
  if (!is_continuous_ls(fit.estimator()) || !fit.data().sample_stats()) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "robust_continuous_ls() requires a continuous ULS/GLS/WLS fit"));
  }
  auto weight = ls_weight_for_fit(fit);
  if (!weight) {
    return std::unexpected(weight.error());
  }
  auto robust = estimate::robust_continuous_ls(
      fit.model().structure(), fit.model().matrix_rep(),
      *fit.data().sample_stats(), fit.estimates(), *weight, gamma);
  return post_result(std::move(robust));
}

Result<estimate::OrdinalRobustResult> robust_ordinal(const Fit &fit) {
  if (const auto *stats = fit.data().ordinal()) {
    if (!is_ordinal_estimator(fit.estimator_spec())) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "robust_ordinal() requires an ordinal DWLS/WLS fit"));
    }
    auto robust = estimate::robust_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight,
        fit.estimator_spec().ordinal_parameterization);
    return post_result(std::move(robust));
  }
  if (const auto *stats = fit.data().mixed_ordinal()) {
    if (!is_ordinal_estimator(fit.estimator_spec())) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "robust_ordinal() requires a mixed ordinal DWLS/WLS fit"));
    }
    auto robust = estimate::robust_mixed_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight,
        fit.estimator_spec().ordinal_parameterization);
    return post_result(std::move(robust));
  }
  return std::unexpected(make_error(
      ErrorStage::UnsupportedCombination,
      "robust_ordinal() requires ordinal or mixed ordinal data"));
}

Result<TestResult> test(const Fit &fit, TestSpec spec) {
  if (spec.kind != TestKind::StandardChiSquare) {
    return std::unexpected(make_error(ErrorStage::UnsupportedCombination,
                                      "unsupported test statistic"));
  }

  if (const auto *stats = fit.data().sample_stats()) {
    post_expected<double> chi2_or;
    double chi2 = std::numeric_limits<double>::quiet_NaN();
    if (fit.estimator() == EstimatorKind::ML) {
      chi2 = inference::chi2_stat(*stats, fit.estimates());
    } else if (is_continuous_ls(fit.estimator())) {
      auto weight = ls_weight_for_fit(fit);
      if (!weight) {
        return std::unexpected(weight.error());
      }
      chi2_or = estimate::continuous_ls_chisq(
          *stats, fit.model().structure(), fit.model().matrix_rep(),
          fit.estimates(), *weight);
      if (!chi2_or) {
        return std::unexpected(make_error(ErrorStage::PostFit,
                                          chi2_or.error()));
      }
      chi2 = *chi2_or;
    } else {
      return std::unexpected(
          make_error(ErrorStage::UnsupportedCombination,
                     "standard chi-square is not available for this fit"));
    }

    auto df = inference::df_stat(fit.model().structure(), *stats,
                               fit.estimates().theta);
    if (!df) {
      return std::unexpected(make_error(ErrorStage::PostFit, df.error()));
    }
    return TestResult{"standard", chi2, *df, inference::chi2_pvalue(chi2, *df)};
  }

  if (const auto *raw = fit.data().raw()) {
    auto df = fiml_df(fit, *raw);
    if (!df) {
      return std::unexpected(df.error());
    }
    auto extras = estimate::fiml::fiml_extras(
        fit.model().structure(), fit.model().matrix_rep(), *raw,
        fit.estimates());
    if (!extras) {
      return std::unexpected(make_error(ErrorStage::PostFit, extras.error()));
    }
    return TestResult{"fiml-likelihood", extras->chi2, *df,
                      inference::chi2_pvalue(extras->chi2, *df)};
  }

  if (fit.data().ordinal() || fit.data().mixed_ordinal()) {
    auto robust = robust_ordinal(fit);
    if (!robust) {
      return std::unexpected(robust.error());
    }
    return TestResult{"ordinal-standard", robust->chisq_standard, robust->df,
                      inference::chi2_pvalue(robust->chisq_standard,
                                             robust->df)};
  }

  return std::unexpected(
      make_error(ErrorStage::UnsupportedCombination,
                 "standard chi-square is not available for this data type"));
}

Result<FitMeasuresResult> fit_measures(const Fit &fit) {
  if (const auto *stats = fit.data().sample_stats()) {
    // Complete-data ML and continuous least-squares (ULS/GLS/WLS): the
    // baseline χ², CFI/TLI/RMSEA, SRMR and information criteria are all
    // estimator-agnostic functions of T_user/df/Σ̂(θ̂)/S. `test()` already
    // produces the right model χ² per estimator.
    if (fit.estimator() != EstimatorKind::ML &&
        !is_continuous_ls(fit.estimator())) {
      return std::unexpected(make_error(
          ErrorStage::UnsupportedCombination,
          "fit_measures() exposes complete-data ML, continuous "
          "least-squares, and FIML fits"));
    }
    auto t = test(fit, standard_chi_square());
    if (!t) {
      return std::unexpected(t.error());
    }
    auto baseline = measures::baseline_chi2(fit.model().structure(), *stats);
    auto indices =
        measures::fit_measures(t->statistic, t->df, baseline, *stats);
    auto extras = measures::fit_extras(fit.model().structure(),
                                       fit.model().matrix_rep(), *stats,
                                       fit.estimates());
    if (!extras) {
      return std::unexpected(make_error(ErrorStage::PostFit, extras.error()));
    }
    return FitMeasuresResult{baseline, indices, std::move(*extras),
                             std::nullopt, std::nullopt};
  }

  if (const auto *raw = fit.data().raw()) {
    auto stats = estimate::fiml::fiml_start_sample_stats(*raw);
    if (!stats) {
      return std::unexpected(make_error(ErrorStage::Data, stats.error()));
    }
    auto t = test(fit, standard_chi_square());
    if (!t) {
      return std::unexpected(t.error());
    }
    auto baseline =
        estimate::fiml::fiml_baseline_chi2(fit.model().structure(), *raw);
    if (!baseline) {
      return std::unexpected(make_error(ErrorStage::PostFit, baseline.error()));
    }
    auto indices =
        measures::fit_measures(t->statistic, t->df, *baseline, *stats);
    auto extras = estimate::fiml::fiml_extras(
        fit.model().structure(), fit.model().matrix_rep(), *raw,
        fit.estimates());
    if (!extras) {
      return std::unexpected(make_error(ErrorStage::PostFit, extras.error()));
    }
    return FitMeasuresResult{*baseline, indices, std::nullopt,
                             std::move(*extras), std::nullopt};
  }

  if (const auto *stats = fit.data().ordinal()) {
    auto fm = estimate::fit_measures_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight,
        fit.estimator_spec().ordinal_parameterization);
    if (!fm) {
      return std::unexpected(make_error(ErrorStage::PostFit, fm.error()));
    }
    return FitMeasuresResult{fm->baseline, fm->indices, std::nullopt,
                             std::nullopt, fm->srmr};
  }

  if (fit.data().mixed_ordinal()) {
    // Mixed ordinal fit measures need the mixed polychoric/polyserial
    // independence baseline; keep that unsupported until the mixed surface has
    // its own parity gate.
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "fit_measures() does not yet cover mixed ordinal DWLS/WLS fits"));
  }
  return std::unexpected(
      make_error(ErrorStage::UnsupportedCombination,
                 "fit measures are not available for this data type"));
}

Result<measures::ResidualMoments> residuals(const Fit &fit) {
  // S − Σ̂(θ̂) needs only θ̂ and the sample moments, so it is exposed for any
  // fit carried over sample statistics (complete-data ML or least-squares).
  // FIML / ordinal fits carry raw / categorical data instead and are not
  // covered here.
  const auto *stats = fit.data().sample_stats();
  if (!stats) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "residuals() requires a fit over sample statistics "
        "(complete-data ML or least-squares)"));
  }
  auto out = measures::residuals(fit.model().structure(),
                                 fit.model().matrix_rep(), *stats,
                                 fit.estimates());
  return post_result(std::move(out));
}

Result<measures::StandardizedResiduals> standardized_residuals(const Fit &fit) {
  const auto *stats = fit.data().sample_stats();
  if (!stats) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "standardized_residuals() requires a fit over sample statistics "
        "(complete-data ML or least-squares)"));
  }
  auto out = measures::standardized_residuals(fit.model().structure(),
                                              fit.model().matrix_rep(), *stats,
                                              fit.estimates());
  return post_result(std::move(out));
}

Result<measures::FactorScores>
factor_scores(const Fit &fit, const data::RawData &raw,
              measures::FactorScoreMethod method) {
  // Factor scores are per-observation, so the caller supplies the raw data
  // explicitly (mirroring robust_se(fit, raw, ...)) — a fit carried over
  // sample statistics can still score separately-held observations. Ordinal
  // fits parameterize thresholds rather than a continuous Λ/Θ split and are
  // not covered.
  auto ok = require_not_ordinal(fit, "factor_scores()");
  if (!ok) {
    return std::unexpected(ok.error());
  }
  auto out = measures::factor_scores(fit.model().structure(),
                                     fit.model().matrix_rep(), raw,
                                     fit.estimates(), method);
  return post_result(std::move(out));
}

Result<inference::ScoreTestTable>
modification_indices(const Fit &fit,
                     inference::ModificationIndexOptions options) {
  if (const auto *stats = fit.data().sample_stats()) {
    if (fit.estimator() == EstimatorKind::ML) {
      auto out = inference::modification_indices(
          fit.model().structure(), fit.model().matrix_rep(), *stats,
          fit.estimates(), options);
      return post_result(std::move(out));
    }
    if (is_continuous_ls(fit.estimator())) {
      auto weight = ls_weight_for_fit(fit);
      if (!weight) {
        return std::unexpected(weight.error());
      }
      auto out = inference::modification_indices(
          fit.model().structure(), fit.model().matrix_rep(), *stats,
          fit.estimates(), *weight, options);
      return post_result(std::move(out));
    }
  }
  if (const auto *raw = fit.data().raw()) {
    auto out = inference::modification_indices_fiml(
        fit.model().structure(), fit.model().matrix_rep(), *raw,
        fit.estimates(), options);
    return post_result(std::move(out));
  }
  if (const auto *stats = fit.data().ordinal()) {
    auto out = estimate::modification_indices_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight, options,
        fit.estimator_spec().ordinal_parameterization);
    return post_result(std::move(out));
  }
  if (const auto *stats = fit.data().mixed_ordinal()) {
    auto out = estimate::modification_indices_mixed_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight, options,
        fit.estimator_spec().ordinal_parameterization);
    return post_result(std::move(out));
  }
  return std::unexpected(make_error(
      ErrorStage::UnsupportedCombination,
      "modification_indices() is not available for this fit"));
}

Result<inference::ScoreTestTable> score_tests(const Fit &fit) {
  if (const auto *stats = fit.data().sample_stats()) {
    if (fit.estimator() == EstimatorKind::ML) {
      auto out = inference::score_tests(fit.model().structure(),
                                        fit.model().matrix_rep(), *stats,
                                        fit.estimates());
      return post_result(std::move(out));
    }
    if (is_continuous_ls(fit.estimator())) {
      auto weight = ls_weight_for_fit(fit);
      if (!weight) {
        return std::unexpected(weight.error());
      }
      auto out = inference::score_tests(fit.model().structure(),
                                        fit.model().matrix_rep(), *stats,
                                        fit.estimates(), *weight);
      return post_result(std::move(out));
    }
  }
  if (const auto *raw = fit.data().raw()) {
    auto out = inference::score_tests_fiml(
        fit.model().structure(), fit.model().matrix_rep(), *raw,
        fit.estimates());
    return post_result(std::move(out));
  }
  if (const auto *stats = fit.data().ordinal()) {
    auto out = estimate::score_tests_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight,
        fit.estimator_spec().ordinal_parameterization);
    return post_result(std::move(out));
  }
  if (const auto *stats = fit.data().mixed_ordinal()) {
    auto out = estimate::score_tests_mixed_ordinal(
        fit.model().structure(), fit.model().matrix_rep(), *stats,
        fit.estimates(), fit.estimator_spec().ordinal_weight,
        fit.estimator_spec().ordinal_parameterization);
    return post_result(std::move(out));
  }
  return std::unexpected(make_error(ErrorStage::UnsupportedCombination,
                                    "score_tests() is not available for this fit"));
}

inference::ZTestResult z_test(const Fit &fit, const Eigen::VectorXd &se) {
  return inference::z_test(fit.estimates(), se);
}

Result<inference::WaldTestResult> wald_test(const Fit &fit,
                                            const Eigen::MatrixXd &R,
                                            const Eigen::MatrixXd &vcov,
                                            const Eigen::VectorXd &q) {
  auto out = inference::wald_test(R, q, fit.estimates(), vcov);
  return post_result(std::move(out));
}

Result<measures::standardize::StandardizedSolution>
standardize_lv(const Fit &fit, const Eigen::MatrixXd &vcov) {
  auto ok = require_not_ordinal(fit, "standardize_lv()");
  if (!ok) {
    return std::unexpected(ok.error());
  }
  auto out = measures::standardize::standardize_lv(
      fit.model().structure(), fit.model().matrix_rep(), fit.estimates(), vcov);
  return post_result(std::move(out));
}

Result<measures::standardize::StandardizedSolution>
standardize_all(const Fit &fit, const Eigen::MatrixXd &vcov) {
  auto ok = require_not_ordinal(fit, "standardize_all()");
  if (!ok) {
    return std::unexpected(ok.error());
  }
  auto out = measures::standardize::standardize_all(
      fit.model().structure(), fit.model().matrix_rep(), fit.estimates(), vcov);
  return post_result(std::move(out));
}

Result<measures::effects::DefinedParams>
compute_defined(const Fit &fit, const Eigen::MatrixXd &vcov) {
  auto ok = require_not_ordinal(fit, "compute_defined()");
  if (!ok) {
    return std::unexpected(ok.error());
  }
  auto out = measures::effects::compute_defined(
      fit.model().flat_partable(), fit.model().structure(), fit.model().names(),
      fit.estimates(), vcov);
  return post_result(std::move(out));
}

Result<LrTestResult> lr_test(const Fit &h1, const Fit &h0) {
  auto t1 = test(h1, standard_chi_square());
  if (!t1) {
    return std::unexpected(t1.error());
  }
  auto t0 = test(h0, standard_chi_square());
  if (!t0) {
    return std::unexpected(t0.error());
  }
  const double diff = t0->statistic - t1->statistic;
  const int df_diff = t0->df - t1->df;
  if (df_diff < 0) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "lr_test() expects h1 to be less restricted than h0"));
  }
  return LrTestResult{diff, df_diff, inference::chi2_pvalue(diff, df_diff)};
}

Result<robust::LRSatorra2000Result>
lr_test_satorra2000(const Fit &h1, const Fit &h0, const data::RawData &raw,
                    robust::GammaSource gamma) {
  return lr_test_satorra2000(
      h1, h0, raw,
      robust::Satorra2000Options{.a_method = robust::SatorraAMethod::Exact,
                                 .gamma = gamma});
}

Result<robust::LRSatorra2000Result>
lr_test_satorra2000(const Fit &h1, const Fit &h0, const data::RawData &raw,
                    robust::Satorra2000Options options) {
  auto ok1 = require_complete_ml(h1, "lr_test_satorra2000()");
  if (!ok1) {
    return std::unexpected(ok1.error());
  }
  auto ok0 = require_complete_ml(h0, "lr_test_satorra2000()");
  if (!ok0) {
    return std::unexpected(ok0.error());
  }
  auto t1 = test(h1, standard_chi_square());
  if (!t1) {
    return std::unexpected(t1.error());
  }
  auto t0 = test(h0, standard_chi_square());
  if (!t0) {
    return std::unexpected(t0.error());
  }
  auto k1 = estimate::build_eq_constraints(h1.model().structure());
  if (!k1) {
    return std::unexpected(make_error(ErrorStage::PostFit, k1.error()));
  }
  auto k0 = estimate::build_eq_constraints(h0.model().structure());
  if (!k0) {
    return std::unexpected(make_error(ErrorStage::PostFit, k0.error()));
  }
  auto samp = data::sample_stats_from_raw(raw);
  if (!samp) {
    return std::unexpected(make_error(ErrorStage::Data, samp.error()));
  }
  const auto n_per_group = block_n32(samp->n_obs);
  auto result = robust::lr_test_satorra2000_from_data(
      h1.model().structure(), h1.model().matrix_rep(), h1.estimates().theta,
      *k1, h0.model().structure(), h0.model().matrix_rep(), h0.estimates().theta,
      *k0, raw.X, samp->mean, n_per_group, block_weights(samp->n_obs),
      t0->statistic, t1->statistic, t0->df, t1->df, options);
  return post_result(std::move(result));
}

Result<robust::LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001(const Fit &h1, const Fit &h0,
                            const data::RawData &raw,
                            robust::GammaSource gamma) {
  auto ok1 = require_complete_ml(h1, "lr_test_satorra_bentler2001()");
  if (!ok1) {
    return std::unexpected(ok1.error());
  }
  auto ok0 = require_complete_ml(h0, "lr_test_satorra_bentler2001()");
  if (!ok0) {
    return std::unexpected(ok0.error());
  }
  auto t1 = test(h1, standard_chi_square());
  if (!t1) {
    return std::unexpected(t1.error());
  }
  auto t0 = test(h0, standard_chi_square());
  if (!t0) {
    return std::unexpected(t0.error());
  }
  auto result = robust::lr_test_satorra_bentler2001_from_data(
      h1.model().structure(), h1.model().matrix_rep(), h1.estimates().theta,
      h0.model().structure(), h0.model().matrix_rep(), h0.estimates().theta,
      raw, t0->statistic, t1->statistic, t0->df, t1->df, gamma);
  return post_result(std::move(result));
}

Result<robust::LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010(const Fit &h1, const Fit &h0,
                            const data::RawData &raw,
                            robust::GammaSource gamma) {
  auto ok1 = require_complete_ml(h1, "lr_test_satorra_bentler2010()");
  if (!ok1) {
    return std::unexpected(ok1.error());
  }
  auto ok0 = require_complete_ml(h0, "lr_test_satorra_bentler2010()");
  if (!ok0) {
    return std::unexpected(ok0.error());
  }
  auto t1 = test(h1, standard_chi_square());
  if (!t1) {
    return std::unexpected(t1.error());
  }
  auto t0 = test(h0, standard_chi_square());
  if (!t0) {
    return std::unexpected(t0.error());
  }
  auto result = robust::lr_test_satorra_bentler2010_from_data(
      h1.model().structure(), h1.model().matrix_rep(), h0.estimates().theta,
      h0.model().structure(), h0.model().matrix_rep(), h0.estimates().theta,
      raw, t0->statistic, t1->statistic, t0->df, t1->df, gamma);
  return post_result(std::move(result));
}

Analysis::Analysis(Model model, Data data)
    : model_(std::make_shared<Model>(std::move(model))),
      data_(std::make_shared<Data>(std::move(data))) {}

Analysis Analysis::fit(EstimatorSpec estimator) const {
  auto out = *this;
  if (out.error_) {
    return out;
  }
  auto f = api::fit(out.model_, out.data_, std::move(estimator));
  if (!f) {
    out.error_ = f.error();
    return out;
  }
  out.fit_ = std::move(*f);
  return out;
}

Analysis Analysis::standard_errors(InformationSpec spec) const {
  auto out = *this;
  if (out.error_) {
    return out;
  }
  if (!out.fit_) {
    out.error_ =
        make_error(ErrorStage::PostFit, "standard_errors() requires fit() first");
    return out;
  }
  auto se = api::standard_errors(*out.fit_, spec);
  if (!se) {
    out.error_ = se.error();
    return out;
  }
  out.standard_errors_ = std::move(*se);
  return out;
}

Analysis Analysis::test(TestSpec spec) const {
  auto out = *this;
  if (out.error_) {
    return out;
  }
  if (!out.fit_) {
    out.error_ = make_error(ErrorStage::PostFit, "test() requires fit() first");
    return out;
  }
  auto t = api::test(*out.fit_, spec);
  if (!t) {
    out.error_ = t.error();
    return out;
  }
  out.test_ = std::move(*t);
  return out;
}

Analysis Analysis::fit_measures() const {
  auto out = *this;
  if (out.error_) {
    return out;
  }
  if (!out.fit_) {
    out.error_ =
        make_error(ErrorStage::PostFit, "fit_measures() requires fit() first");
    return out;
  }
  auto measures = api::fit_measures(*out.fit_);
  if (!measures) {
    out.error_ = measures.error();
    return out;
  }
  out.fit_measures_ = std::move(*measures);
  return out;
}

Result<Summary> Analysis::summary() const {
  if (error_) {
    return std::unexpected(*error_);
  }
  if (!fit_) {
    return std::unexpected(
        make_error(ErrorStage::PostFit, "summary() requires fit() first"));
  }
  return Summary{*fit_, standard_errors_, test_, fit_measures_};
}

Analysis analyze(Model model, Data data) {
  return Analysis(std::move(model), std::move(data));
}

} // namespace magmaan::api
