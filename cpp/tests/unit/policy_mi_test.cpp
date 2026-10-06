#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "magmaan/api/policy_mi.hpp"
#include "magmaan/api/sem.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include <random>
#include <algorithm>
#include "magmaan/data/ordinal.hpp"

TEST_CASE("policy modification indices select observed empirical ML and affine releases") {
  using namespace magmaan;
  auto syntax = parse::Parser::parse("f =~ x1 + a*x2 + a*x3 + x4");
  REQUIRE(syntax);
  spec::BuildOptions build_options;
  build_options.meanstructure = true;
  auto pt = spec::build(*syntax, build_options);
  REQUIRE(pt);
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep);
  data::RawData raw;
  raw.X.emplace_back(400, 4);
  std::mt19937 rng(991);
  std::normal_distribution<double> normal;
  for (int i = 0; i < 400; ++i) {
    const double factor = normal(rng);
    for (int j = 0; j < 4; ++j) raw.X[0](i,j) = factor + normal(rng);
  }
  auto sample = data::sample_stats_from_raw(raw);
  REQUIRE(sample);
  auto estimates = test::fit(*pt, *rep, *sample);
  REQUIRE(estimates);
  auto result = api::policy_modification_indices(*pt, *rep, *sample, raw, *estimates, {});
  REQUIRE(result.reason == api::InferenceReason::Available);
  inference::frontier::RobustScoreOptions opts;
  opts.base.candidates = inference::ScoreCandidateSet::WithAbsentRows;
  opts.spec = {robust::Information::Observed, robust::WeightMoments::Likelihood, robust::ScoreCovariance::Empirical};
  auto fixed = inference::frontier::modification_indices_robust(*pt, *rep, *sample, raw, *estimates, opts);
  auto released = inference::frontier::score_tests_robust(*pt, *rep, *sample, raw, *estimates, opts);
  REQUIRE(fixed); REQUIRE(released);
  for (const auto& expected : fixed->rows) {
    auto found = std::find_if(result.table.rows.begin(), result.table.rows.end(), [&](const auto& row) {
      return row.candidate.kind == expected.candidate.kind && row.candidate.row == expected.candidate.row;
    });
    REQUIRE(found != result.table.rows.end());
    CHECK(found->mi_scaled == expected.mi_scaled);
    CHECK(found->epc_all == expected.epc_all);
  }
  REQUIRE(!released->rows.empty());
  const auto& last = result.table.rows.back();
  CHECK(last.candidate.kind == inference::ScoreCandidateKind::EqualityRelease);
  CHECK(last.mi_scaled == released->rows.back().mi_scaled);
  CHECK(last.p_value == released->rows.back().p_value);
  api::ModelOptions facade_options;
  facade_options.build.meanstructure = true;
  auto facade_model = api::Model::from_lavaan("f =~ x1 + a*x2 + a*x3 + x4", facade_options);
  REQUIRE(facade_model);
  auto facade = api::fit(*facade_model, api::Data::from_sample_stats(*sample), api::ml());
  REQUIRE(facade);
  auto facade_result = api::policy_modification_indices(*facade, raw);
  CHECK(facade_result.reason == api::InferenceReason::Available);
  CHECK(api::policy_modification_indices(*facade).reason == api::InferenceReason::UnsupportedModel);
  api::PolicyFitState state;
  state.converged = false;
  CHECK(api::policy_modification_indices(*pt, *rep, *sample, raw, *estimates, state).reason == api::InferenceReason::NotConverged);
  state.penalized = true;
  CHECK(api::policy_modification_indices(*pt, *rep, *sample, raw, *estimates, state).reason == api::InferenceReason::Penalized);
}

TEST_CASE("policy modification indices use exact all-ordinal first-stage sampling rows") {
  using namespace magmaan;
  for (const int groups : {1, 2}) {
    auto syntax = parse::Parser::parse("f =~ x1 + a*x2 + a*x3 + x4\nx1 | t1+t2\nx2 | t1+t2\nx3 | t1+t2\nx4 | t1+t2\nx1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4");
    REQUIRE(syntax);
    spec::BuildOptions options; options.n_groups = groups;
    auto pt = spec::build(*syntax, options); REQUIRE(pt);
    auto rep = model::build_matrix_rep(*pt); REQUIRE(rep);
    std::mt19937 rng(9912); std::normal_distribution<double> normal;
    std::vector<Eigen::MatrixXd> integer(static_cast<std::size_t>(groups), Eigen::MatrixXd(350, 4));
    for (auto& block : integer) for (int i = 0; i < block.rows(); ++i) {
      const double f = normal(rng), residual = normal(rng);
      for (int j = 0; j < 4; ++j) {
        const double y = .7 * f + .8 * normal(rng) + (j < 2 ? .3 * residual : 0);
        block(i,j) = 1 + (y > -.4) + (y > .6);
      }
    }
    auto stats = data::ordinal_stats_from_integer_data(integer, true); REQUIRE(stats);
    auto estimates = test::fit_ordinal_bounded(*pt, *rep, *stats, {}, estimate::OrdinalWeightKind::DWLS); REQUIRE(estimates);
    const auto policy = api::policy_modification_indices(*pt, *rep, *stats, *estimates,
        estimate::OrdinalParameterization::Delta, {});
    REQUIRE(policy.reason == api::InferenceReason::Available);
    auto exact = *stats;
    for (std::size_t b = 0; b < exact.R.size(); ++b) {
      auto influence = data::ordinal_moment_sampling_influence(exact.int_data[b], exact.n_levels[b],
          exact.thresholds[b], exact.R[b]); REQUIRE(influence);
      exact.sampling_moment_influence.push_back(influence->rows);
    }
    api::PolicyModificationOptions opts;
    auto fixed = estimate::frontier::modification_indices_ordinal_robust(*pt, *rep, exact,
        *estimates, estimate::OrdinalWeightKind::DWLS, opts.candidates,
        estimate::OrdinalParameterization::Delta, true, robust::Information::Observed);
    auto release = estimate::frontier::score_tests_ordinal_robust(*pt, *rep, exact,
        *estimates, estimate::OrdinalWeightKind::DWLS, estimate::OrdinalParameterization::Delta,
        true, robust::Information::Observed);
    REQUIRE(fixed); REQUIRE(release);
    REQUIRE(policy.table.rows.size() == fixed->rows.size() + release->rows.size());
    for (std::size_t i = 0; i < fixed->rows.size(); ++i) {
      CHECK(policy.table.rows[i].mi_scaled == fixed->rows[i].mi_scaled);
      CHECK(policy.table.rows[i].epc_all == fixed->rows[i].epc_all);
    }
    for (std::size_t i = 0; i < release->rows.size(); ++i)
      CHECK(policy.table.rows[fixed->rows.size() + i].mi_scaled == release->rows[i].mi_scaled);
  }
}
