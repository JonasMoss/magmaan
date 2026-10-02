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

OrdinalModel ordinal_model(const std::string& syntax, int groups) {
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions options;
  options.n_groups = groups;
  auto pt = magmaan::spec::build(*fp, options);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}

const std::string kOneFactor =
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

double relative(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).norm() / b.norm();
}

}  // namespace

TEST_CASE("DWLS policy: IJ covariance and one fit-function global test") {
  const Eigen::MatrixXd X = misspecified_block(20261002u, 400, -0.4, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  const auto m = ordinal_model(kOneFactor, 1);
  const auto est = fit_dwls(m, *stats, OrdinalParameterization::Delta);
  const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                              OrdinalParameterization::Delta, {});
  REQUIRE(out.covariance_reason == api::InferenceReason::Available);
  auto ij = magmaan::estimate::robust_ordinal_ij(m.pt, m.rep, *stats, est, OrdinalWeightKind::DWLS);
  auto fixed = magmaan::estimate::robust_ordinal(m.pt, m.rep, *stats, est, OrdinalWeightKind::DWLS);
  auto fixed_observed = magmaan::estimate::robust_ordinal(m.pt, m.rep, *stats, est,
      OrdinalWeightKind::DWLS, OrdinalParameterization::Delta, magmaan::robust::Information::Observed);
  REQUIRE(ij.has_value()); REQUIRE(fixed.has_value()); REQUIRE(fixed_observed.has_value());
  CHECK((out.covariance - ij->vcov).norm() == 0.0);
  // The weight influence is not negligible under this misspecification.
  CHECK(relative(ij->vcov, fixed_observed->vcov) > 1e-3);

  REQUIRE(out.score.reason == api::InferenceReason::Available);
  CHECK(out.score.label == "fit_function");
  CHECK(out.score.statistic == fixed->chisq_standard);
  CHECK(out.score.df == fixed->df);
  CHECK(out.score.df == 2);
  Eigen::VectorXd eig = Eigen::VectorXd::Zero(fixed->df);
  eig.tail(fixed->eigvals.size()) = fixed->eigvals;
  std::sort(eig.data(), eig.data() + eig.size());
  CHECK((out.score.eigenvalues - eig).norm() == 0.0);
  CHECK(out.score.sb_scale == doctest::Approx(eig.mean()));
  CHECK(out.score.p_sb == doctest::Approx(
      magmaan::inference::chi2_pvalue(out.score.statistic / out.score.sb_scale, out.score.df)));
  CHECK(std::isfinite(out.score.p_peba4));
  CHECK(out.lr.reason == api::InferenceReason::Inapplicable);
  CHECK(api::reason_name(out.lr.reason) == "inapplicable");

  api::PolicyFitState failed;
  failed.converged = false;
  const auto refused = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                                  OrdinalParameterization::Delta, failed);
  CHECK(refused.covariance_reason == api::InferenceReason::NotConverged);
  CHECK(refused.score.reason == api::InferenceReason::NotConverged);
}

TEST_CASE("DWLS policy: at exact fit the IJ covariance is the fixed-weight sandwich") {
  // A saturated three-item model reproduces the polychorics, so the residuals,
  // and with them the weight influence and the observed-minus-expected bread,
  // vanish.
  const Eigen::MatrixXd X4 = misspecified_block(77u, 500, -0.3, 0.7);
  const Eigen::MatrixXd X = X4.leftCols(3);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  const auto m = ordinal_model(
      "f =~ x1 + x2 + x3\nx1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\n", 1);
  const auto est = fit_dwls(m, *stats, OrdinalParameterization::Delta);
  const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                              OrdinalParameterization::Delta, {});
  REQUIRE(out.covariance_reason == api::InferenceReason::Available);
  CHECK(out.score.reason == api::InferenceReason::Saturated);
  CHECK(out.lr.reason == api::InferenceReason::Inapplicable);
  auto fixed = magmaan::estimate::robust_ordinal(m.pt, m.rep, *stats, est, OrdinalWeightKind::DWLS);
  REQUIRE(fixed.has_value());
  CHECK(relative(out.covariance, fixed->vcov) < 1e-6);
}

TEST_CASE("DWLS policy IJ covariance agrees with the delete-one jackknife") {
  struct Design {
    const char* name;
    int groups;
    OrdinalParameterization parameterization;
  };
  for (const Design design : {Design{"delta, one group", 1, OrdinalParameterization::Delta},
                              Design{"delta, two groups", 2, OrdinalParameterization::Delta},
                              Design{"theta, one group", 1, OrdinalParameterization::Theta}}) {
    CAPTURE(design.name);
    std::vector<Eigen::MatrixXd> blocks;
    for (int g = 0; g < design.groups; ++g)
      blocks.push_back(misspecified_block(4000u + static_cast<std::uint32_t>(g), 600 + 40 * g,
                                          -0.4 + 0.2 * g, 0.6));
    auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
    REQUIRE(stats.has_value());
    const auto m = ordinal_model(kOneFactor, design.groups);
    const auto est = fit_dwls(m, *stats, design.parameterization);
    const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                                design.parameterization, {});
    REQUIRE(out.covariance_reason == api::InferenceReason::Available);
    auto fixed_observed = magmaan::estimate::robust_ordinal(m.pt, m.rep, *stats, est,
        OrdinalWeightKind::DWLS, design.parameterization, magmaan::robust::Information::Observed);
    REQUIRE(fixed_observed.has_value());

    // Stratified delete-one jackknife: refit without each case, warm-started
    // at the full-sample estimate.
    const Eigen::Index k = est.theta.size();
    Eigen::MatrixXd jackknife = Eigen::MatrixXd::Zero(k, k);
    for (int g = 0; g < design.groups; ++g) {
      const Eigen::Index n = blocks[static_cast<std::size_t>(g)].rows();
      Eigen::MatrixXd thetas(n, k);
      for (Eigen::Index i = 0; i < n; ++i) {
        std::vector<Eigen::MatrixXd> reduced = blocks;
        Eigen::MatrixXd& x = reduced[static_cast<std::size_t>(g)];
        Eigen::MatrixXd y(n - 1, x.cols());
        y.topRows(i) = x.topRows(i);
        y.bottomRows(n - 1 - i) = x.bottomRows(n - 1 - i);
        x = std::move(y);
        auto s = magmaan::data::ordinal_stats_from_integer_data(reduced, true);
        REQUIRE(s.has_value());
        auto refit = magmaan::estimate::fit_ordinal_bounded(m.pt, m.rep, *s, {},
            OrdinalWeightKind::DWLS, est.theta, magmaan::estimate::Backend::NloptLbfgs,
            tight(), design.parameterization);
        REQUIRE(refit.has_value());
        thetas.row(i) = refit->theta.transpose();
      }
      const Eigen::RowVectorXd mean = thetas.colwise().mean();
      const Eigen::MatrixXd centered = thetas.rowwise() - mean;
      jackknife += (static_cast<double>(n - 1) / static_cast<double>(n)) *
                   centered.transpose() * centered;
    }
    const Eigen::VectorXd j = jackknife.diagonal();
    const double ij_error = (out.covariance.diagonal() - j).norm() / j.norm();
    const double fixed_error = (fixed_observed->vcov.diagonal() - j).norm() / j.norm();
    MESSAGE(std::string(design.name) << ": IJ vs jackknife " << ij_error << ", fixed weight vs jackknife "
            << fixed_error);
    // The two estimators of the same asymptotic covariance agree to O(1/n).
    // The fixed-weight sandwich omits the weight influence; its gap does not
    // shrink with n. On the delta scale here that gap dominates the sampling
    // noise; on the theta scale it is smaller than the n = 600 noise (it
    // separates by n = 2400: IJ 1%, fixed 4%).
    CHECK(ij_error < 0.04);
    if (design.parameterization == OrdinalParameterization::Delta) CHECK(ij_error < fixed_error);
  }
}
