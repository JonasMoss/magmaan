#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/error.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/nt/fiml.hpp"
#include "magmaan/nt/infer.hpp"
#include "magmaan/nt/measures.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::api {

enum class ErrorStage : std::uint8_t {
  Parse,
  Model,
  Data,
  Fit,
  PostFit,
  UnsupportedCombination,
};

using UnderlyingError = std::variant<std::monostate, ParseError, PartableError,
                                     ModelError, FitError, PostError>;

struct Error {
  ErrorStage stage = ErrorStage::UnsupportedCombination;
  std::string detail;
  UnderlyingError underlying;
};

template <class T> using Result = std::expected<T, Error>;

inline Error make_error(ErrorStage stage, std::string detail) {
  return Error{stage, std::move(detail), std::monostate{}};
}

inline Error make_error(ErrorStage stage, ParseError error) {
  return Error{stage, error.detail, std::move(error)};
}

inline Error make_error(ErrorStage stage, PartableError error) {
  return Error{stage, error.detail, std::move(error)};
}

inline Error make_error(ErrorStage stage, ModelError error) {
  return Error{stage, error.detail, std::move(error)};
}

inline Error make_error(ErrorStage stage, FitError error) {
  return Error{stage, error.detail, std::move(error)};
}

inline Error make_error(ErrorStage stage, PostError error) {
  return Error{stage, error.detail, std::move(error)};
}

struct ModelOptions {
  spec::LavaanifyOptions lavaanify;
};

class Model {
public:
  static Result<Model> from_lavaan(std::string_view syntax,
                                   ModelOptions options = {}) {
    auto flat = parse::Parser::parse(syntax);
    if (!flat) {
      return std::unexpected(make_error(ErrorStage::Parse, flat.error()));
    }

    spec::Starts starts;
    spec::LatentNames names;
    auto structure = spec::lavaanify(*flat, options.lavaanify, &starts, &names);
    if (!structure) {
      return std::unexpected(make_error(ErrorStage::Model, structure.error()));
    }

    auto rep = model::build_matrix_rep(*structure, &names);
    if (!rep) {
      return std::unexpected(make_error(ErrorStage::Model, rep.error()));
    }

    return Model(std::string(syntax), std::move(*structure), std::move(names),
                 std::move(starts), std::move(*rep), options);
  }

  const std::string &source() const noexcept { return source_; }
  const spec::LatentStructure &structure() const noexcept { return structure_; }
  const spec::LatentNames &names() const noexcept { return names_; }
  const spec::Starts &starts() const noexcept { return starts_; }
  const model::MatrixRep &matrix_rep() const noexcept { return rep_; }
  const ModelOptions &options() const noexcept { return options_; }

private:
  Model(std::string source, spec::LatentStructure structure,
        spec::LatentNames names, spec::Starts starts, model::MatrixRep rep,
        ModelOptions options)
      : source_(std::move(source)), structure_(std::move(structure)),
        names_(std::move(names)), starts_(std::move(starts)),
        rep_(std::move(rep)), options_(std::move(options)) {}

  std::string source_;
  spec::LatentStructure structure_;
  spec::LatentNames names_;
  spec::Starts starts_;
  model::MatrixRep rep_;
  ModelOptions options_;
};

inline Result<Model> model_from_lavaan(std::string_view syntax,
                                       ModelOptions options = {}) {
  return Model::from_lavaan(syntax, std::move(options));
}

enum class DataKind : std::uint8_t {
  SampleStats,
  RawContinuous,
  Ordinal,
  MixedOrdinal,
};

class Data {
public:
  using Storage = std::variant<data::SampleStats, data::RawData,
                               data::OrdinalStats, data::MixedOrdinalStats>;

  static Data from_sample_stats(data::SampleStats stats) {
    return Data(std::move(stats));
  }

  static Data from_raw(data::RawData raw) { return Data(std::move(raw)); }

  static Data from_ordinal(data::OrdinalStats stats) {
    return Data(std::move(stats));
  }

  static Data from_mixed_ordinal(data::MixedOrdinalStats stats) {
    return Data(std::move(stats));
  }

  DataKind kind() const noexcept {
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

  const data::SampleStats *sample_stats() const noexcept {
    return std::get_if<data::SampleStats>(&storage_);
  }

  const data::RawData *raw() const noexcept {
    return std::get_if<data::RawData>(&storage_);
  }

  const data::OrdinalStats *ordinal() const noexcept {
    return std::get_if<data::OrdinalStats>(&storage_);
  }

  const data::MixedOrdinalStats *mixed_ordinal() const noexcept {
    return std::get_if<data::MixedOrdinalStats>(&storage_);
  }

private:
  explicit Data(Storage storage) : storage_(std::move(storage)) {}

  Storage storage_;
};

inline Result<Data> data_from_sample_stats(const Model &,
                                           data::SampleStats stats) {
  return Data::from_sample_stats(std::move(stats));
}

inline Result<Data> data_from_raw(const Model &, data::RawData raw) {
  return Data::from_raw(std::move(raw));
}

inline Result<Data> data_from_ordinal(const Model &, data::OrdinalStats stats) {
  return Data::from_ordinal(std::move(stats));
}

inline Result<Data> data_from_mixed_ordinal(const Model &,
                                            data::MixedOrdinalStats stats) {
  return Data::from_mixed_ordinal(std::move(stats));
}

inline Result<Data> data_from_ordinal_h_weighted(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalHWeightedStatsOptions options = {}) {
  auto stats = data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      blocks, options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_ordinal(std::move(stats->stats));
}

inline Result<Data> data_from_ordinal_dpd(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalDpdStatsOptions options = {}) {
  auto stats = data::pairwise_ordinal_stats_dpd_from_integer_data(blocks,
                                                                  options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_ordinal(std::move(stats->stats));
}

inline Result<Data> data_from_mixed_ordinal_polyserial_dpd(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    const std::vector<std::vector<std::int32_t>> &ordered,
    data::PolyserialPairDpdOptions options = {}) {
  auto stats = data::mixed_ordinal_stats_polyserial_dpd_from_data(
      blocks, ordered, options);
  if (!stats) {
    return std::unexpected(make_error(ErrorStage::Data, stats.error()));
  }
  return Data::from_mixed_ordinal(std::move(stats->stats));
}

enum class OptimizerKind : std::uint8_t {
  Lbfgs,
  Ceres,
};

struct OptimizerSpec {
  OptimizerKind kind = OptimizerKind::Lbfgs;
  optim::LbfgsOptions lbfgs;
};

inline OptimizerSpec lbfgs(optim::LbfgsOptions options = {}) {
  return OptimizerSpec{OptimizerKind::Lbfgs, options};
}

inline OptimizerSpec ceres(optim::LbfgsOptions options = {}) {
  return OptimizerSpec{OptimizerKind::Ceres, options};
}

enum class StartKind : std::uint8_t {
  Simple,
  Fabin,
  Guttman,
  Bentler1982,
  JamesStein,
  Explicit,
};

struct StartSpec {
  StartKind kind = StartKind::Simple;
  Eigen::VectorXd theta;
};

inline StartSpec simple_starts() { return StartSpec{StartKind::Simple, {}}; }
inline StartSpec fabin_starts() { return StartSpec{StartKind::Fabin, {}}; }
inline StartSpec guttman_starts() { return StartSpec{StartKind::Guttman, {}}; }
inline StartSpec bentler1982_starts() {
  return StartSpec{StartKind::Bentler1982, {}};
}
inline StartSpec jamesstein_starts() {
  return StartSpec{StartKind::JamesStein, {}};
}
inline StartSpec explicit_starts(Eigen::VectorXd theta) {
  return StartSpec{StartKind::Explicit, std::move(theta)};
}

enum class BoundsMode : std::uint8_t {
  None,
  Explicit,
  AutoVariance,
};

enum class EstimatorKind : std::uint8_t {
  ML,
  FIML,
  ULS,
  GLS,
  WLS,
  DWLS,
};

struct EstimatorSpec {
  EstimatorKind kind = EstimatorKind::ML;
  OptimizerSpec optimizer_spec = lbfgs();
  StartSpec start_spec = simple_starts();
  BoundsMode bounds_mode = BoundsMode::None;
  estimate::Bounds bounds;
  gmm::Weight weight;
  estimate::OrdinalWeightKind ordinal_weight =
      estimate::OrdinalWeightKind::DWLS;
  estimate::OrdinalParameterization ordinal_parameterization =
      estimate::OrdinalParameterization::Delta;

  EstimatorSpec optimizer(OptimizerSpec optimizer) const {
    auto out = *this;
    out.optimizer_spec = std::move(optimizer);
    return out;
  }

  EstimatorSpec starts(StartSpec start) const {
    auto out = *this;
    out.start_spec = std::move(start);
    return out;
  }

  EstimatorSpec with_bounds(estimate::Bounds b) const {
    auto out = *this;
    out.bounds_mode = BoundsMode::Explicit;
    out.bounds = std::move(b);
    return out;
  }

  EstimatorSpec auto_variance_bounds() const {
    auto out = *this;
    out.bounds_mode = BoundsMode::AutoVariance;
    out.bounds = {};
    return out;
  }

  EstimatorSpec
  parameterization(estimate::OrdinalParameterization parameterization) const {
    auto out = *this;
    out.ordinal_parameterization = parameterization;
    return out;
  }
};

inline EstimatorSpec ml() { return EstimatorSpec{EstimatorKind::ML}; }
inline EstimatorSpec fiml() { return EstimatorSpec{EstimatorKind::FIML}; }
inline EstimatorSpec uls() { return EstimatorSpec{EstimatorKind::ULS}; }
inline EstimatorSpec gls() { return EstimatorSpec{EstimatorKind::GLS}; }

inline EstimatorSpec wls(gmm::Weight weight) {
  auto out = EstimatorSpec{EstimatorKind::WLS};
  out.weight = std::move(weight);
  return out;
}

inline EstimatorSpec dwls() {
  auto out = EstimatorSpec{EstimatorKind::DWLS};
  out.ordinal_weight = estimate::OrdinalWeightKind::DWLS;
  return out;
}

inline Result<Eigen::VectorXd> start_values(const spec::LatentStructure &pt,
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

inline Result<estimate::Bounds> bounds_for(const spec::LatentStructure &pt,
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

inline estimate::Backend backend_from(const OptimizerSpec &optimizer) {
  return optimizer.kind == OptimizerKind::Ceres ? estimate::Backend::Ceres
                                                : estimate::Backend::Lbfgs;
}

class Fit;

struct StandardErrors {
  Eigen::MatrixXd information;
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
};

struct TestResult {
  std::string name;
  double statistic = 0.0;
  int df = 0;
  double p_value = 0.0;
};

struct FitMeasuresResult {
  nt::measures::BaselineFit baseline;
  nt::measures::FitMeasures indices;
  std::optional<nt::measures::FitExtras> complete_data_extras;
  std::optional<nt::fiml::FIMLExtras> fiml_extras;
};

class Fit {
public:
  const Model &model() const noexcept { return *model_; }
  const Data &data() const noexcept { return *data_; }
  const estimate::Estimates &estimates() const noexcept { return estimates_; }
  EstimatorKind estimator() const noexcept { return estimator_; }

private:
  friend Result<Fit> fit(std::shared_ptr<const Model>,
                         std::shared_ptr<const Data>, EstimatorSpec);

  Fit(std::shared_ptr<const Model> model, std::shared_ptr<const Data> data,
      estimate::Estimates estimates, EstimatorKind estimator)
      : model_(std::move(model)), data_(std::move(data)),
        estimates_(std::move(estimates)), estimator_(estimator) {}

  std::shared_ptr<const Model> model_;
  std::shared_ptr<const Data> data_;
  estimate::Estimates estimates_;
  EstimatorKind estimator_ = EstimatorKind::ML;
};

inline Result<Fit> fit(std::shared_ptr<const Model> model,
                       std::shared_ptr<const Data> data,
                       EstimatorSpec estimator) {
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

    auto start_stats = nt::fiml::fiml_start_sample_stats(*raw);
    if (!start_stats) {
      return std::unexpected(make_error(ErrorStage::Data, start_stats.error()));
    }
    auto x0 = start_values(pt, rep, *start_stats, model->starts(),
                           estimator.start_spec);
    if (!x0)
      return std::unexpected(x0.error());

    auto est = nt::fiml::fit_fiml(pt, rep, *raw, *x0, {},
                                  estimator.optimizer_spec.lbfgs);
    if (!est)
      return std::unexpected(make_error(ErrorStage::Fit, est.error()));
    return Fit(std::move(model), std::move(data), std::move(*est),
               estimator.kind);
  }

  if (estimator.kind == EstimatorKind::DWLS) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "ordinal DWLS is reserved for the facade but not wired in this "
        "prototype"));
  }

  const auto *stats = data->sample_stats();
  if (!stats) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "continuous ML/ULS/GLS/WLS require complete-data sample statistics"));
  }

  auto x0 =
      start_values(pt, rep, *stats, model->starts(), estimator.start_spec);
  if (!x0)
    return std::unexpected(x0.error());

  auto bounds = bounds_for(pt, estimator);
  if (!bounds)
    return std::unexpected(bounds.error());

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
                     "WLS requires an explicit weight matrix"));
    }
    est = estimate::fit_gmm(pt, rep, *stats, *x0, estimator.weight, *bounds,
                            backend_from(estimator.optimizer_spec),
                            estimator.optimizer_spec.lbfgs);
    break;
  case EstimatorKind::FIML:
  case EstimatorKind::DWLS:
    break;
  }

  if (!est)
    return std::unexpected(make_error(ErrorStage::Fit, est.error()));
  return Fit(std::move(model), std::move(data), std::move(*est),
             estimator.kind);
}

inline Result<Fit> fit(const Model &model, const Data &data,
                       EstimatorSpec estimator) {
  return fit(std::make_shared<Model>(model), std::make_shared<Data>(data),
             std::move(estimator));
}

enum class InformationKind : std::uint8_t {
  Expected,
  ObservedFiniteDifference,
  ObservedAnalytic,
};

struct InformationSpec {
  InformationKind kind = InformationKind::Expected;
  double h_step = 1e-4;
};

inline InformationSpec expected_information() {
  return InformationSpec{InformationKind::Expected, 1e-4};
}

inline InformationSpec observed_information_fd(double h_step = 1e-4) {
  return InformationSpec{InformationKind::ObservedFiniteDifference, h_step};
}

inline InformationSpec observed_information_analytic() {
  return InformationSpec{InformationKind::ObservedAnalytic, 1e-4};
}

inline Result<StandardErrors> standard_errors(const Fit &fit,
                                              InformationSpec spec) {
  const auto *stats = fit.data().sample_stats();
  if (!stats) {
    return std::unexpected(make_error(
        ErrorStage::UnsupportedCombination,
        "standard information-based SEs currently require complete-data "
        "sample statistics"));
  }

  post_expected<Eigen::MatrixXd> info;
  switch (spec.kind) {
  case InformationKind::Expected:
    info = nt::infer::information_expected(fit.model().structure(),
                                           fit.model().matrix_rep(), *stats,
                                           fit.estimates());
    break;
  case InformationKind::ObservedFiniteDifference:
    info = nt::infer::information_observed_fd(fit.model().structure(),
                                              fit.model().matrix_rep(), *stats,
                                              fit.estimates(), spec.h_step);
    break;
  case InformationKind::ObservedAnalytic:
    info = nt::infer::information_observed_analytic(fit.model().structure(),
                                                    fit.model().matrix_rep(),
                                                    *stats, fit.estimates());
    break;
  }
  if (!info) {
    return std::unexpected(make_error(ErrorStage::PostFit, info.error()));
  }

  auto vc = nt::infer::vcov(*info, fit.model().structure());
  if (!vc) {
    return std::unexpected(make_error(ErrorStage::PostFit, vc.error()));
  }

  return StandardErrors{std::move(*info), *vc, nt::infer::se(*vc)};
}

enum class TestKind : std::uint8_t {
  StandardChiSquare,
};

struct TestSpec {
  TestKind kind = TestKind::StandardChiSquare;
};

inline TestSpec standard_chi_square() { return TestSpec{}; }

inline Result<TestResult> test(const Fit &fit, TestSpec spec) {
  if (spec.kind != TestKind::StandardChiSquare) {
    return std::unexpected(make_error(ErrorStage::UnsupportedCombination,
                                      "unsupported test statistic"));
  }

  if (const auto *stats = fit.data().sample_stats()) {
    const double chi2 = nt::infer::chi2_stat(*stats, fit.estimates());
    auto df = nt::infer::df_stat(fit.model().structure(), *stats);
    if (!df) {
      return std::unexpected(make_error(ErrorStage::PostFit, df.error()));
    }
    return TestResult{"standard", chi2, *df, nt::infer::chi2_pvalue(chi2, *df)};
  }

  if (const auto *raw = fit.data().raw()) {
    auto stats = nt::fiml::fiml_start_sample_stats(*raw);
    if (!stats) {
      return std::unexpected(make_error(ErrorStage::Data, stats.error()));
    }
    auto df = nt::infer::df_stat(fit.model().structure(), *stats);
    if (!df) {
      return std::unexpected(make_error(ErrorStage::PostFit, df.error()));
    }
    auto extras =
        nt::fiml::fiml_extras(fit.model().structure(), fit.model().matrix_rep(),
                              *raw, fit.estimates());
    if (!extras) {
      return std::unexpected(make_error(ErrorStage::PostFit, extras.error()));
    }
    return TestResult{"fiml-likelihood", extras->chi2, *df,
                      nt::infer::chi2_pvalue(extras->chi2, *df)};
  }

  return std::unexpected(
      make_error(ErrorStage::UnsupportedCombination,
                 "standard chi-square is not available for this data type"));
}

inline Result<FitMeasuresResult> fit_measures(const Fit &fit) {
  if (const auto *stats = fit.data().sample_stats()) {
    auto t = test(fit, standard_chi_square());
    if (!t)
      return std::unexpected(t.error());
    auto baseline = nt::measures::baseline_chi2(*stats);
    auto indices =
        nt::measures::fit_measures(t->statistic, t->df, baseline, *stats);
    auto extras = nt::measures::fit_extras(fit.model().structure(),
                                           fit.model().matrix_rep(), *stats,
                                           fit.estimates());
    if (!extras) {
      return std::unexpected(make_error(ErrorStage::PostFit, extras.error()));
    }
    return FitMeasuresResult{baseline, indices, std::move(*extras),
                             std::nullopt};
  }

  if (const auto *raw = fit.data().raw()) {
    auto stats = nt::fiml::fiml_start_sample_stats(*raw);
    if (!stats) {
      return std::unexpected(make_error(ErrorStage::Data, stats.error()));
    }
    auto t = test(fit, standard_chi_square());
    if (!t)
      return std::unexpected(t.error());
    auto baseline = nt::fiml::fiml_baseline_chi2(fit.model().structure(), *raw);
    if (!baseline) {
      return std::unexpected(make_error(ErrorStage::PostFit, baseline.error()));
    }
    auto indices =
        nt::measures::fit_measures(t->statistic, t->df, *baseline, *stats);
    auto extras =
        nt::fiml::fiml_extras(fit.model().structure(), fit.model().matrix_rep(),
                              *raw, fit.estimates());
    if (!extras) {
      return std::unexpected(make_error(ErrorStage::PostFit, extras.error()));
    }
    return FitMeasuresResult{*baseline, indices, std::nullopt,
                             std::move(*extras)};
  }

  return std::unexpected(
      make_error(ErrorStage::UnsupportedCombination,
                 "fit measures are not available for this data type"));
}

struct Summary {
  Fit fit;
  std::optional<StandardErrors> standard_errors;
  std::optional<TestResult> test;
  std::optional<FitMeasuresResult> fit_measures;
};

class Analysis {
public:
  Analysis(Model model, Data data)
      : model_(std::make_shared<Model>(std::move(model))),
        data_(std::make_shared<Data>(std::move(data))) {}

  Analysis fit(EstimatorSpec estimator) const {
    auto out = *this;
    if (out.error_)
      return out;
    auto f = api::fit(out.model_, out.data_, std::move(estimator));
    if (!f) {
      out.error_ = f.error();
      return out;
    }
    out.fit_ = std::move(*f);
    return out;
  }

  Analysis standard_errors(InformationSpec spec) const {
    auto out = *this;
    if (out.error_)
      return out;
    if (!out.fit_) {
      out.error_ = make_error(ErrorStage::PostFit,
                              "standard_errors() requires fit() first");
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

  Analysis test(TestSpec spec) const {
    auto out = *this;
    if (out.error_)
      return out;
    if (!out.fit_) {
      out.error_ =
          make_error(ErrorStage::PostFit, "test() requires fit() first");
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

  Analysis fit_measures() const {
    auto out = *this;
    if (out.error_)
      return out;
    if (!out.fit_) {
      out.error_ = make_error(ErrorStage::PostFit,
                              "fit_measures() requires fit() first");
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

  Result<Summary> summary() const {
    if (error_)
      return std::unexpected(*error_);
    if (!fit_) {
      return std::unexpected(
          make_error(ErrorStage::PostFit, "summary() requires fit() first"));
    }
    return Summary{*fit_, standard_errors_, test_, fit_measures_};
  }

private:
  std::shared_ptr<const Model> model_;
  std::shared_ptr<const Data> data_;
  std::optional<Fit> fit_;
  std::optional<StandardErrors> standard_errors_;
  std::optional<TestResult> test_;
  std::optional<FitMeasuresResult> fit_measures_;
  std::optional<Error> error_;
};

inline Analysis analyze(Model model, Data data) {
  return Analysis(std::move(model), std::move(data));
}

} // namespace magmaan::api
