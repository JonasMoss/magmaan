#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/api/policy.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/frontier/fmg.hpp"

namespace {

namespace api = magmaan::api;
using magmaan::estimate::OrdinalParameterization;
using magmaan::estimate::OrdinalWeightKind;

// Four three-category items from one factor plus a nuisance factor shared by
// x1 and x2, so a one-factor model is misspecified and the estimated-weight
// influence is leading order.
Eigen::MatrixXd misspecified_block(std::uint32_t seed, Eigen::Index n, double cut1, double cut2) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  const double loading[4] = {0.75, 0.70, 0.65, 0.60};
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = z(rng), nuisance = z(rng);
    for (int j = 0; j < 4; ++j) {
      const double shared = j < 2 ? 0.45 : 0.0;
      const double residual = std::sqrt(1.0 - loading[j] * loading[j] - shared * shared);
      const double y = loading[j] * eta + shared * nuisance + residual * z(rng);
      X(i, j) = 1.0 + (y > cut1) + (y > cut2);
    }
  }
  return X;
}

struct OrdinalModel {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

OrdinalModel ordinal_model(const std::string& syntax, int groups,
                           std::vector<magmaan::spec::GroupEqual> equal = {}) {
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions options;
  options.n_groups = groups;
  options.group_equal = std::move(equal);
  auto pt = magmaan::spec::build(*fp, options);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}

const std::string one_factor =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
    "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";

magmaan::optim::OptimOptions tight() {
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  opts.ftol = 1e-14;
  opts.gtol = 1e-10;
  return opts;
}

magmaan::estimate::Estimates fit_dwls(const OrdinalModel& m, const magmaan::data::OrdinalStats& stats,
                                      OrdinalParameterization parameterization) {
  auto fit = magmaan::test::fit_ordinal_bounded(m.pt, m.rep, stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, tight(), parameterization);
  REQUIRE_MESSAGE(fit.has_value(), "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));
  return *fit;
}

}
namespace {

const std::string tau_equivalent =
    "f =~ x1 + a*x2 + a*x3 + a*x4\n"
    "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
    "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";

struct NestedDwls {
  api::PolicyNested out;
  magmaan::estimate::Estimates null_est, alt_est;
  OrdinalModel null_model, alt_model;
};

NestedDwls nested_dwls(const magmaan::data::OrdinalStats& stats, int groups,
                       OrdinalParameterization parameterization) {
  NestedDwls r{{}, {}, {}, ordinal_model(tau_equivalent, groups), ordinal_model(one_factor, groups)};
  r.null_est = fit_dwls(r.null_model, stats, parameterization);
  r.alt_est = fit_dwls(r.alt_model, stats, parameterization);
  r.out = api::policy_nested_dwls(r.null_model.pt, r.null_model.rep, r.null_est, {},
                                  r.alt_model.pt, r.alt_model.rep, r.alt_est, {}, stats,
                                  parameterization);
  return r;
}

}

TEST_CASE("DWLS policy snapshots: bit identity and immutable ingredient reuse") {
  for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta})
    for (int groups : {1, 2}) {
      std::vector<Eigen::MatrixXd> blocks;
      for (int g = 0; g < groups; ++g)
        blocks.push_back(misspecified_block(1123u + static_cast<std::uint32_t>(g), 400, -0.4, 0.6));
      auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true); REQUIRE(stats);
      const auto pair = nested_dwls(*stats, groups, parameterization);
      api::DwlsPolicyFit null(pair.null_model.pt, pair.null_model.rep, *stats, pair.null_est, parameterization);
      api::DwlsPolicyFit alt(pair.alt_model.pt, pair.alt_model.rep, *stats, pair.alt_est, parameterization);
      CHECK(api::policy_ingredient_builds(alt) == 0);
      const auto fresh = api::policy_inference_dwls(pair.alt_model.pt, pair.alt_model.rep,
          *stats, pair.alt_est, parameterization, {});
      const auto cached = api::policy_inference_dwls(alt, {});
      REQUIRE(cached.covariance_reason == api::InferenceReason::Available);
      CHECK((cached.covariance.array() == fresh.covariance.array()).all());
      CHECK(cached.score.statistic == fresh.score.statistic);
      CHECK(cached.score.p_all == fresh.score.p_all);
      CHECK((cached.score.eigenvalues.array() == fresh.score.eigenvalues.array()).all());
      CHECK(api::policy_ingredient_builds(alt) == 1);
      for (int i = 0; i < 2; ++i) {
        const auto got = api::policy_nested_dwls(null, {}, alt, {});
        REQUIRE(got.lr.reason == api::InferenceReason::Available);
        CHECK(got.lr.statistic == pair.out.lr.statistic);
        CHECK(got.lr.p_sb == pair.out.lr.p_sb);
        CHECK(got.lr.p_peba4 == pair.out.lr.p_peba4);
        CHECK((got.lr.eigenvalues.array() == pair.out.lr.eigenvalues.array()).all());
        CHECK(api::policy_ingredient_builds(alt) == 2);
      }
      auto other = *stats; other.W_dwls[0](0,0) += 1;
      api::DwlsPolicyFit wrong(pair.null_model.pt, pair.null_model.rep, other, pair.null_est, parameterization);
      CHECK(api::policy_nested_dwls(wrong, {}, alt, {}).lr.reason == api::InferenceReason::NotNested);
      api::PolicyFitState penalized; penalized.penalized = true;
      CHECK(api::policy_nested_dwls(wrong, penalized, alt, {}).lr.reason == api::InferenceReason::Penalized);
      api::PolicyFitState failed; failed.converged = false;
      CHECK(api::policy_inference_dwls(alt, failed).covariance_reason == api::InferenceReason::NotConverged);
      CHECK(api::policy_ingredient_builds(alt) == 2);
      const api::PolicyFitState boundary{true,true,false,false};
      const auto flagged = api::policy_inference_dwls(alt,boundary);
      const auto fresh_flagged = api::policy_inference_dwls(pair.alt_model.pt,
          pair.alt_model.rep,*stats,pair.alt_est,parameterization,boundary);
      CHECK(flagged.psd_boundary == fresh_flagged.psd_boundary);
      CHECK(flagged.verdict_disagreement == fresh_flagged.verdict_disagreement);

    }
}
