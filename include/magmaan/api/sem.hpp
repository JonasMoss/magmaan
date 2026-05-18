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
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/measures/effects.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/spec/build.hpp"
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

Error make_error(ErrorStage stage, std::string detail);
Error make_error(ErrorStage stage, ParseError error);
Error make_error(ErrorStage stage, PartableError error);
Error make_error(ErrorStage stage, ModelError error);
Error make_error(ErrorStage stage, FitError error);
Error make_error(ErrorStage stage, PostError error);

struct ModelOptions {
  spec::BuildOptions build;
};

class Model {
public:
  static Result<Model> from_lavaan(std::string_view syntax,
                                   ModelOptions options = {});

  const std::string &source() const noexcept { return source_; }
  const parse::FlatPartable &flat_partable() const noexcept { return *flat_; }
  const spec::LatentStructure &structure() const noexcept { return structure_; }
  const spec::LatentNames &names() const noexcept { return names_; }
  const spec::Starts &starts() const noexcept { return starts_; }
  const model::MatrixRep &matrix_rep() const noexcept { return rep_; }
  const ModelOptions &options() const noexcept { return options_; }

private:
  Model(std::string source, parse::FlatPartable flat,
        spec::LatentStructure structure, spec::LatentNames names,
        spec::Starts starts, model::MatrixRep rep, ModelOptions options);

  std::string source_;
  std::shared_ptr<const parse::FlatPartable> flat_;
  spec::LatentStructure structure_;
  spec::LatentNames names_;
  spec::Starts starts_;
  model::MatrixRep rep_;
  ModelOptions options_;
};

Result<Model> model_from_lavaan(std::string_view syntax,
                                ModelOptions options = {});

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

  static Data from_sample_stats(data::SampleStats stats);
  static Data from_raw(data::RawData raw);
  static Data from_ordinal(data::OrdinalStats stats);
  static Data from_mixed_ordinal(data::MixedOrdinalStats stats);

  DataKind kind() const noexcept;
  const data::SampleStats *sample_stats() const noexcept;
  const data::RawData *raw() const noexcept;
  const data::OrdinalStats *ordinal() const noexcept;
  const data::MixedOrdinalStats *mixed_ordinal() const noexcept;

private:
  explicit Data(Storage storage);

  Storage storage_;
};

Result<Data> data_from_sample_stats(const Model &, data::SampleStats stats);
Result<Data> data_from_raw(const Model &, data::RawData raw);
Result<Data> data_from_ordinal(const Model &, data::OrdinalStats stats);
Result<Data> data_from_mixed_ordinal(const Model &,
                                     data::MixedOrdinalStats stats);
Result<Data> data_from_ordinal_h_weighted(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalHWeightedStatsOptions options = {});
Result<Data> data_from_ordinal_dpd(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalDpdStatsOptions options = {});
Result<Data> data_from_ordinal_huber_residual(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    data::PairwiseOrdinalHuberResidualStatsOptions options = {});
Result<Data> data_from_mixed_ordinal_polyserial_dpd(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    const std::vector<std::vector<std::int32_t>> &ordered,
    data::PolyserialPairDpdOptions options = {});
Result<Data> data_from_mixed_ordinal_huber_residual(
    const Model &, const std::vector<Eigen::MatrixXd> &blocks,
    const std::vector<std::vector<std::int32_t>> &ordered,
    data::MixedOrdinalHuberResidualOptions options = {});

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
  estimate::gmm::Weight weight;
  estimate::OrdinalWeightKind ordinal_weight =
      estimate::OrdinalWeightKind::DWLS;
  estimate::OrdinalParameterization ordinal_parameterization =
      estimate::OrdinalParameterization::Delta;

  EstimatorSpec optimizer(OptimizerSpec optimizer) const;
  EstimatorSpec starts(StartSpec start) const;
  EstimatorSpec with_bounds(estimate::Bounds b) const;
  EstimatorSpec auto_variance_bounds() const;
  EstimatorSpec
  parameterization(estimate::OrdinalParameterization parameterization) const;
};

EstimatorSpec ml();
EstimatorSpec fiml();
EstimatorSpec uls();
EstimatorSpec gls();
EstimatorSpec wls(estimate::gmm::Weight weight);
EstimatorSpec ordinal_dwls();
EstimatorSpec ordinal_wls();
EstimatorSpec dwls();

enum class InformationKind : std::uint8_t {
  Expected,
  ObservedFiniteDifference,
  ObservedAnalytic,
};

struct InformationSpec {
  InformationKind kind = InformationKind::Expected;
  double h_step = 1e-4;
};

InformationSpec expected_information();
InformationSpec observed_information_fd(double h_step = 1e-4);
InformationSpec observed_information_analytic();

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
  measures::BaselineFit baseline;
  measures::FitMeasures indices;
  std::optional<measures::FitExtras> complete_data_extras;
  std::optional<estimate::fiml::FIMLExtras> fiml_extras;
};

struct LrTestResult {
  double chi2_diff = 0.0;
  int df_diff = 0;
  double p_value = 0.0;
};

class Fit {
public:
  const Model &model() const noexcept { return *model_; }
  const Data &data() const noexcept { return *data_; }
  const estimate::Estimates &estimates() const noexcept { return estimates_; }
  EstimatorKind estimator() const noexcept { return estimator_.kind; }
  const EstimatorSpec &estimator_spec() const noexcept { return estimator_; }

private:
  friend Result<Fit> fit(std::shared_ptr<const Model>,
                         std::shared_ptr<const Data>, EstimatorSpec);

  Fit(std::shared_ptr<const Model> model, std::shared_ptr<const Data> data,
      estimate::Estimates estimates, EstimatorSpec estimator);

  std::shared_ptr<const Model> model_;
  std::shared_ptr<const Data> data_;
  estimate::Estimates estimates_;
  EstimatorSpec estimator_;
};

Result<Fit> fit(std::shared_ptr<const Model> model,
                std::shared_ptr<const Data> data, EstimatorSpec estimator);
Result<Fit> fit(const Model &model, const Data &data, EstimatorSpec estimator);

enum class TestKind : std::uint8_t {
  StandardChiSquare,
};

struct TestSpec {
  TestKind kind = TestKind::StandardChiSquare;
};

TestSpec standard_chi_square();

Result<StandardErrors> standard_errors(const Fit &fit, InformationSpec spec);
Result<estimate::fiml::FIMLRobustMLR> fiml_robust_mlr(const Fit &fit,
                                                      double h_step = 1e-4);
Result<robust::RobustSeResult>
robust_se(const Fit &fit, const data::RawData &raw,
          robust::InferenceSpec spec = {robust::Information::Expected,
                                        robust::WeightMoments::Structured,
                                        robust::ScoreCovariance::Empirical});
Result<robust::RobustSeResult>
robust_se(const Fit &fit, const Eigen::MatrixXd &gamma,
          robust::InferenceSpec spec = {robust::Information::Expected,
                                        robust::WeightMoments::Structured,
                                        robust::ScoreCovariance::Empirical});
Result<estimate::WeightedRobustResult>
robust_continuous_ls(const Fit &fit, const data::RawData &raw);
Result<estimate::WeightedRobustResult>
robust_continuous_ls(const Fit &fit,
                     const std::vector<Eigen::MatrixXd> &gamma);
Result<estimate::OrdinalRobustResult> robust_ordinal(const Fit &fit);

Result<TestResult> test(const Fit &fit, TestSpec spec);
Result<FitMeasuresResult> fit_measures(const Fit &fit);
Result<inference::ScoreTestTable>
modification_indices(const Fit &fit,
                     inference::ModificationIndexOptions options = {});
Result<inference::ScoreTestTable> score_tests(const Fit &fit);
inference::ZTestResult z_test(const Fit &fit, const Eigen::VectorXd &se);
Result<inference::WaldTestResult> wald_test(const Fit &fit,
                                            const Eigen::MatrixXd &R,
                                            const Eigen::MatrixXd &vcov,
                                            const Eigen::VectorXd &q);
Result<measures::standardize::StandardizedSolution>
standardize_lv(const Fit &fit, const Eigen::MatrixXd &vcov);
Result<measures::standardize::StandardizedSolution>
standardize_all(const Fit &fit, const Eigen::MatrixXd &vcov);
Result<measures::effects::DefinedParams>
compute_defined(const Fit &fit, const Eigen::MatrixXd &vcov);
Result<LrTestResult> lr_test(const Fit &h1, const Fit &h0);
Result<robust::LRSatorra2000Result>
lr_test_satorra2000(const Fit &h1, const Fit &h0, const data::RawData &raw,
                    robust::GammaSource gamma = robust::GammaSource::Empirical);
Result<robust::LRSatorra2000Result>
lr_test_satorra2000(const Fit &h1, const Fit &h0, const data::RawData &raw,
                    robust::Satorra2000Options options);
Result<robust::LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001(
    const Fit &h1, const Fit &h0, const data::RawData &raw,
    robust::GammaSource gamma = robust::GammaSource::Empirical);
Result<robust::LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010(
    const Fit &h1, const Fit &h0, const data::RawData &raw,
    robust::GammaSource gamma = robust::GammaSource::Empirical);

struct Summary {
  Fit fit;
  std::optional<StandardErrors> standard_errors;
  std::optional<TestResult> test;
  std::optional<FitMeasuresResult> fit_measures;
};

class Analysis {
public:
  Analysis(Model model, Data data);

  Analysis fit(EstimatorSpec estimator) const;
  Analysis standard_errors(InformationSpec spec) const;
  Analysis test(TestSpec spec) const;
  Analysis fit_measures() const;
  Result<Summary> summary() const;

private:
  std::shared_ptr<const Model> model_;
  std::shared_ptr<const Data> data_;
  std::optional<Fit> fit_;
  std::optional<StandardErrors> standard_errors_;
  std::optional<TestResult> test_;
  std::optional<FitMeasuresResult> fit_measures_;
  std::optional<Error> error_;
};

Analysis analyze(Model model, Data data);

} // namespace magmaan::api
