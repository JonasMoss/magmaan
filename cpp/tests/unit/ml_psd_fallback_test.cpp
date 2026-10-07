#include <doctest/doctest.h>

#include <utility>

#include "magmaan/estimate/frontier/ml_psd_fallback.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {
struct Case {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::data::SampleStats sample;
  Eigen::VectorXd start;
};

Case one_factor(bool improper = false, bool identified = true) {
  auto syntax = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(syntax.has_value());
  magmaan::spec::BuildOptions build;
  build.auto_fix_first = identified;
  auto pt = magmaan::spec::build(*syntax, build);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  Eigen::Matrix3d covariance;
  if (improper) covariance << 1, .7, .7, .7, 1, .3, .7, .3, 1;
  else covariance << 1, .5, .4, .5, 1, .32, .4, .32, 1;
  magmaan::data::SampleStats sample;
  sample.S.push_back(covariance);
  sample.n_obs.push_back(300);
  auto start = magmaan::estimate::simple_start_values(*pt, *rep, sample, {});
  REQUIRE(start.has_value());
  return {std::move(*pt), std::move(*rep), std::move(sample), std::move(*start)};
}
}  // namespace

using namespace magmaan::estimate;
using namespace magmaan::estimate::frontier;

TEST_CASE("PSD fallback keeps an accepted ordinary fit without running PSD") {
  auto c = one_factor();
  MlPsdFallbackOptions options;
  // A skipped backend must never be invoked.
  options.psd_backend = Backend::Ceres;
  auto policy = fit_ml_psd_fallback(c.pt, c.rep, c.sample, c.start, options);
  REQUIRE(policy.ordinary.has_value());
  REQUIRE(policy.accepted_fit() != nullptr);
  CHECK(policy.accepted_fit() == &*policy.ordinary);
  CHECK_FALSE(policy.psd.has_value());
  CHECK_FALSE(policy.warm_start_used);
  CHECK(policy.reason == PsdFallbackReason::None);
  auto direct = fit_ml(c.pt, c.rep, c.sample, c.start);
  REQUIRE(direct.has_value());
  CHECK(policy.ordinary->theta.isApprox(direct->theta, 1e-12));
  CHECK(policy.ordinary->f_evals == direct->f_evals);
}

TEST_CASE("PSD fallback repairs an accurate improper fit from its estimates") {
  auto c = one_factor(true);
  auto policy = fit_ml_psd_fallback(c.pt, c.rep, c.sample, c.start);
  REQUIRE(policy.ordinary.has_value());
  CHECK(fit_verdict(*policy.ordinary).status == FitCheck::Passed);
  CHECK_FALSE(policy.ordinary->diagnostics.admissibility.admissible);
  CHECK(policy.reason == PsdFallbackReason::OrdinaryInadmissible);
  CHECK(policy.warm_start_used);
  REQUIRE(policy.psd.has_value());
  REQUIRE_MESSAGE(policy.psd->has_value(),
      (policy.psd->has_value() ? "" : policy.psd->error().detail));
  REQUIRE(policy.accepted_fit() != nullptr);
  CHECK(policy.accepted_fit() == &**policy.psd);
  auto direct = fit_ml_psd(c.pt, c.rep, c.sample, policy.ordinary->theta);
  REQUIRE(direct.has_value());
  CHECK(policy.psd->value().theta.isApprox(direct->theta, 1e-12));
  CHECK(policy.psd->value().f_evals == direct->f_evals);
  CHECK(policy.psd->value().fmin > policy.ordinary->fmin);
}

TEST_CASE("PSD fallback retains an ordinary budget stop and uses the original start") {
  auto c = one_factor();
  MlPsdFallbackOptions options;
  options.ordinary.nlopt.max_eval = 1;
  auto policy = fit_ml_psd_fallback(c.pt, c.rep, c.sample, c.start, options);
  REQUIRE(policy.ordinary.has_value());
  CHECK(fit_verdict(*policy.ordinary).status != FitCheck::Passed);
  CHECK(policy.reason == PsdFallbackReason::OrdinaryRejected);
  CHECK_FALSE(policy.warm_start_used);
  REQUIRE(policy.psd.has_value());
  REQUIRE(policy.accepted_fit() != nullptr);
  auto direct = fit_ml_psd(c.pt, c.rep, c.sample, c.start);
  REQUIRE(direct.has_value());
  CHECK(policy.accepted_fit()->theta.isApprox(direct->theta, 1e-12));
}

TEST_CASE("PSD fallback rejects an admissible but inaccurate ordinary return") {
  auto c = one_factor();
  MlPsdFallbackOptions options;
  options.ordinary.nlopt.ftol_rel = 1;
  options.ordinary.nlopt.xtol_rel = 1;
  auto policy = fit_ml_psd_fallback(c.pt, c.rep, c.sample, c.start, options);
  REQUIRE(policy.ordinary.has_value());
  REQUIRE(policy.ordinary->diagnostics.admissibility.admissible);
  CHECK(fit_verdict(*policy.ordinary).status != FitCheck::Passed);
  CHECK(policy.reason == PsdFallbackReason::OrdinaryRejected);
  CHECK(policy.warm_start_used);
  REQUIRE(policy.psd.has_value());
  CHECK(policy.accepted_fit() != nullptr);
}

TEST_CASE("PSD fallback retains both failed budget candidates without accepting them") {
  auto c = one_factor();
  MlPsdFallbackOptions options;
  options.ordinary.nlopt.max_eval = 1;
  options.psd.nlopt.max_eval = 1;
  auto policy = fit_ml_psd_fallback(c.pt, c.rep, c.sample, c.start, options);
  REQUIRE(policy.ordinary.has_value());
  CHECK(fit_verdict(*policy.ordinary).status != FitCheck::Passed);
  REQUIRE(policy.psd.has_value());
  REQUIRE(policy.psd->has_value());
  CHECK(fit_verdict(**policy.psd).status != FitCheck::Passed);
  CHECK(policy.accepted_fit() == nullptr);
}

TEST_CASE("PSD fallback retains a returned PSD fit that fails accuracy") {
  // A free factor scale leaves a null direction in the information matrix.
  auto c = one_factor(false, false);
  MlPsdFallbackOptions options;
  options.ordinary.nlopt.max_eval = 1;
  auto policy = fit_ml_psd_fallback(c.pt, c.rep, c.sample, c.start, options);
  REQUIRE(policy.psd.has_value());
  REQUIRE(policy.psd->has_value());
  CHECK(fit_verdict(**policy.psd).status != FitCheck::Passed);
  CHECK(policy.accepted_fit() == nullptr);
}
