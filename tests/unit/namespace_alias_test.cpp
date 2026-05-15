#include <type_traits>

#include <doctest/doctest.h>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/gls/gls.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/lavaan/partable_view.hpp"
#include "magmaan/nt/effects.hpp"
#include "magmaan/nt/fiml.hpp"
#include "magmaan/nt/infer.hpp"
#include "magmaan/nt/measures.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/nt/robust.hpp"
#include "magmaan/nt/standardize.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/concepts.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/spec/lavaanify.hpp"
#include "magmaan/spec/lin_constraints.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

// Focused old-header coverage for the transition window.
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/bounds.hpp"
#include "magmaan/fit/effects.hpp"
#include "magmaan/fit/fit.hpp"
#include "magmaan/fit/fit_measures.hpp"
#include "magmaan/fit/fiml.hpp"
#include "magmaan/fit/gls.hpp"
#include "magmaan/fit/inference.hpp"
#include "magmaan/fit/lbfgsb_optimizer.hpp"
#include "magmaan/fit/lbfgs_optimizer.hpp"
#include "magmaan/fit/ml.hpp"
#include "magmaan/fit/raw_data.hpp"
#include "magmaan/fit/robust.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/score.hpp"
#include "magmaan/fit/standardized.hpp"
#include "magmaan/fit/uls.hpp"
#include "magmaan/fit/wls.hpp"
#include "magmaan/partable/lavaan_view.hpp"
#include "magmaan/partable/lavaanify.hpp"
#include "magmaan/partable/lin_constraints.hpp"
#include "magmaan/partable/partable.hpp"

TEST_CASE("transitional public namespace aliases expose existing API types") {
  static_assert(std::is_same_v<magmaan::spec::LatentStructure,
                               magmaan::partable::LatentStructure>);
  static_assert(std::is_same_v<magmaan::spec::LatentNames,
                               magmaan::partable::LatentNames>);
  static_assert(std::is_same_v<magmaan::spec::Starts,
                               magmaan::partable::Starts>);
  static_assert(std::is_same_v<magmaan::spec::LavaanifyOptions,
                               magmaan::partable::LavaanifyOptions>);
  static_assert(std::is_same_v<magmaan::spec::LinearForm,
                               magmaan::partable::LinearForm>);

  static_assert(std::is_same_v<magmaan::lavaan::LavaanParTable,
                               magmaan::partable::LavaanParTable>);
  static_assert(std::is_same_v<magmaan::lavaan::ParsedLavaanParTable,
                               magmaan::partable::ParsedLavaanParTable>);

  static_assert(std::is_same_v<magmaan::data::SampleStats,
                               magmaan::fit::SampleStats>);
  static_assert(std::is_same_v<magmaan::data::RawData,
                               magmaan::fit::RawData>);

  static_assert(std::is_same_v<magmaan::estimate::Estimates,
                               magmaan::fit::Estimates>);
  static_assert(std::is_same_v<magmaan::estimate::Bounds,
                               magmaan::fit::Bounds>);
  static_assert(std::is_same_v<magmaan::estimate::EqConstraints,
                               magmaan::fit::EqConstraints>);

  static_assert(std::is_same_v<magmaan::optim::LbfgsOptimizer,
                               magmaan::fit::LbfgsOptimizer>);
  static_assert(std::is_same_v<magmaan::optim::LbfgsBOptimizer,
                               magmaan::fit::LbfgsBOptimizer>);
  static_assert(magmaan::optim::Optimizer<magmaan::optim::LbfgsOptimizer>);

  static_assert(std::is_same_v<magmaan::nt::ml::ML, magmaan::fit::ML>);
  static_assert(std::is_same_v<magmaan::nt::fiml::FIML,
                               magmaan::fit::FIML>);
  static_assert(std::is_same_v<magmaan::gls::GLS, magmaan::fit::GLS>);
  static_assert(std::is_same_v<magmaan::gls::ULS, magmaan::fit::ULS>);
  static_assert(std::is_same_v<magmaan::gls::WLS, magmaan::fit::WLS>);
  static_assert(std::is_same_v<magmaan::nt::infer::WaldTestResult,
                               magmaan::fit::WaldTestResult>);
  static_assert(std::is_same_v<magmaan::nt::infer::ScoreTestTable,
                               magmaan::fit::ScoreTestTable>);
  static_assert(std::is_same_v<magmaan::nt::measures::FitMeasures,
                               magmaan::fit::FitMeasures>);
  static_assert(std::is_same_v<magmaan::nt::standardize::StandardizedSolution,
                               magmaan::fit::StandardizedSolution>);
  static_assert(std::is_same_v<magmaan::nt::robust::SatorraBentlerResult,
                               magmaan::fit::SatorraBentlerResult>);
  static_assert(std::is_same_v<magmaan::fit::DefinedParams,
                               magmaan::nt::effects::DefinedParams>);

  CHECK(true);
}
