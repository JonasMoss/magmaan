#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/error.hpp"
#include "magmaan/sim/norta.hpp"
#include "magmaan/sim/ordinal_correlation.hpp"
#include "magmaan/sim/population.hpp"
#include "magmaan/sim/projection.hpp"

using magmaan::sim::calibrate_ordinal_correlation;
using magmaan::sim::ObservedCorrelationMetric;
using magmaan::sim::ObservedKind;
using magmaan::sim::OrdinalCorrelationOptions;
using magmaan::sim::OrdinalMarginalSpec;
using magmaan::sim::ordinal_correlation_population;
using magmaan::sim::ordinal_pair_observed_corr;
using magmaan::sim::simulate_mixed_population_normal;
using magmaan::sim::simulate_ordinal_correlation_normal;
using BivariateRepair = magmaan::sim::BivariateCopulaCorrelationRepairKind;

namespace {

OrdinalMarginalSpec ordinal_marginal(std::vector<double> probs) {
  OrdinalMarginalSpec spec;
  spec.kind = ObservedKind::Ordinal;
  spec.proportions = Eigen::Map<Eigen::VectorXd>(probs.data(),
                                                 static_cast<Eigen::Index>(probs.size()));
  return spec;
}

OrdinalMarginalSpec continuous_marginal() {
  OrdinalMarginalSpec spec;
  spec.kind = ObservedKind::Continuous;
  return spec;
}

double sample_pearson(const Eigen::MatrixXd& x, Eigen::Index a, Eigen::Index b) {
  const Eigen::VectorXd ca = x.col(a).array() - x.col(a).mean();
  const Eigen::VectorXd cb = x.col(b).array() - x.col(b).mean();
  return ca.dot(cb) / std::sqrt(ca.squaredNorm() * cb.squaredNorm());
}

}  // namespace

TEST_CASE("polychoric calibration sets latent equal to target") {
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.4, 0.2,
            0.4, 1.0, 0.5,
            0.2, 0.5, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.25, 0.50, 0.25}),
      ordinal_marginal({0.30, 0.40, 0.30}),
      ordinal_marginal({0.50, 0.50})};

  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::Polychoric;
  auto cal_or = calibrate_ordinal_correlation(target, marginals, options);
  REQUIRE(cal_or.has_value());
  const auto& cal = *cal_or;

  CHECK(cal.latent_corr.isApprox(target, 1e-12));
  CHECK(cal.achieved_corr.isApprox(target, 1e-12));
  CHECK(cal.max_abs_error == doctest::Approx(0.0).epsilon(1e-12));
  CHECK_FALSE(cal.repair_applied);

  // Thresholds reproduce the requested proportions.
  REQUIRE(cal.thresholds[0].size() == 2);
  CHECK(magmaan::sim::normal_cdf(cal.thresholds[0](0)) ==
        doctest::Approx(0.25).epsilon(1e-12));
  CHECK(magmaan::sim::normal_cdf(cal.thresholds[0](1)) ==
        doctest::Approx(0.75).epsilon(1e-12));
  REQUIRE(cal.thresholds[2].size() == 1);
  CHECK(magmaan::sim::normal_cdf(cal.thresholds[2](0)) ==
        doctest::Approx(0.50).epsilon(1e-12));
}

TEST_CASE("polyserial forward map is linear in rho and invertible") {
  Eigen::VectorXd th(2);
  th << -0.6744897501960817, 0.6744897501960817;  // 25/50/25 split
  Eigen::VectorXd none;  // continuous side has no thresholds

  const double c0 = ordinal_pair_observed_corr(
                        ObservedCorrelationMetric::Polyserial,
                        ObservedKind::Ordinal, th, ObservedKind::Continuous,
                        none, 0.0)
                        .value();
  CHECK(c0 == doctest::Approx(0.0));

  const double c5 = ordinal_pair_observed_corr(
                        ObservedCorrelationMetric::Polyserial,
                        ObservedKind::Ordinal, th, ObservedKind::Continuous,
                        none, 0.5)
                        .value();
  const double c25 = ordinal_pair_observed_corr(
                         ObservedCorrelationMetric::Polyserial,
                         ObservedKind::Ordinal, th, ObservedKind::Continuous,
                         none, 0.25)
                         .value();
  CHECK(c5 == doctest::Approx(2.0 * c25).epsilon(1e-12));  // linearity

  // Calibrating a mixed pair to a polyserial target inverts the map exactly.
  Eigen::MatrixXd target(2, 2);
  target << 1.0, 0.45,
            0.45, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.25, 0.50, 0.25}), continuous_marginal()};
  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::Polyserial;
  auto cal_or = calibrate_ordinal_correlation(target, marginals, options);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->achieved_corr(0, 1) == doctest::Approx(0.45).epsilon(1e-10));
}

TEST_CASE("Pearson-of-codes forward map is monotone and invertible") {
  Eigen::VectorXd ti(2);
  ti << -0.4307272992953674, 0.5244005127080407;  // 0.33/0.37/0.30
  Eigen::VectorXd tj(1);
  tj << 0.0;  // 0.50/0.50

  double prev = -2.0;
  for (double rho = -0.9; rho <= 0.9001; rho += 0.3) {
    const double c =
        ordinal_pair_observed_corr(ObservedCorrelationMetric::PearsonCodes,
                                   ObservedKind::Ordinal, ti,
                                   ObservedKind::Ordinal, tj, rho)
            .value();
    CHECK(c > prev);  // strictly increasing -> clean bisection bracket
    prev = c;
  }

  Eigen::MatrixXd target(2, 2);
  target << 1.0, 0.3,
            0.3, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.33, 0.37, 0.30}), ordinal_marginal({0.50, 0.50})};
  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::PearsonCodes;
  auto cal_or = calibrate_ordinal_correlation(target, marginals, options);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->pairs[0].converged);
  CHECK(cal_or->achieved_corr(0, 1) == doctest::Approx(0.3).epsilon(1e-6));
}

TEST_CASE("PD-repair policy on an indefinite calibrated matrix") {
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.9, 0.9,
            0.9, 1.0, -0.9,
            0.9, -0.9, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.5, 0.5}), ordinal_marginal({0.5, 0.5}),
      ordinal_marginal({0.5, 0.5})};

  OrdinalCorrelationOptions none;
  none.metric = ObservedCorrelationMetric::Polychoric;
  none.matrix_repair = BivariateRepair::None;
  auto none_or = calibrate_ordinal_correlation(target, marginals, none);
  REQUIRE(none_or.has_value());
  CHECK(none_or->raw_min_eigenvalue < 0.0);
  CHECK_FALSE(none_or->repair_applied);

  OrdinalCorrelationOptions ridge = none;
  ridge.matrix_repair = BivariateRepair::Ridge;
  ridge.matrix_repair_min_eigenvalue = 1e-3;
  auto ridge_or = calibrate_ordinal_correlation(target, marginals, ridge);
  REQUIRE(ridge_or.has_value());
  CHECK(ridge_or->repair_applied);
  CHECK(ridge_or->repaired_min_eigenvalue >= 1e-3 - 1e-9);

  OrdinalCorrelationOptions err = none;
  err.matrix_repair = BivariateRepair::Error;
  err.matrix_repair_min_eigenvalue = 1e-3;
  auto err_or = calibrate_ordinal_correlation(target, marginals, err);
  REQUIRE_FALSE(err_or.has_value());
  CHECK(err_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("calibration rejects malformed targets") {
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.5, 0.5}), ordinal_marginal({0.5, 0.5})};
  Eigen::MatrixXd asym(2, 2);
  asym << 1.0, 0.4,
          0.2, 1.0;
  auto bad = calibrate_ordinal_correlation(asym, marginals, {});
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("Pearson-of-codes round-trip recovers the target in large samples") {
  Eigen::MatrixXd target(2, 2);
  target << 1.0, 0.4,
            0.4, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.30, 0.40, 0.30}),
      ordinal_marginal({0.25, 0.50, 0.25})};
  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::PearsonCodes;

  std::mt19937_64 rng(20260613ULL);
  auto draw_or = simulate_ordinal_correlation_normal(200000, target, marginals,
                                                     rng, options);
  REQUIRE(draw_or.has_value());
  CHECK(std::abs(sample_pearson(draw_or->observed.X, 0, 1) - 0.4) < 0.01);
}

TEST_CASE("polyserial round-trip recovers the target in large samples") {
  Eigen::MatrixXd target(2, 2);
  target << 1.0, 0.5,
            0.5, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.25, 0.50, 0.25}), continuous_marginal()};
  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::Polyserial;

  std::mt19937_64 rng(778899ULL);
  auto draw_or = simulate_ordinal_correlation_normal(200000, target, marginals,
                                                     rng, options);
  REQUIRE(draw_or.has_value());
  // polyserial == Pearson(codes, continuous).
  CHECK(std::abs(sample_pearson(draw_or->observed.X, 0, 1) - 0.5) < 0.01);
}

TEST_CASE("polychoric round-trip recovers the target in large samples") {
  Eigen::MatrixXd target(2, 2);
  target << 1.0, 0.5,
            0.5, 1.0;
  std::vector<OrdinalMarginalSpec> marginals = {
      ordinal_marginal({0.30, 0.40, 0.30}),
      ordinal_marginal({0.20, 0.30, 0.50})};
  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::Polychoric;

  std::mt19937_64 rng(424242ULL);
  auto draw_or = simulate_ordinal_correlation_normal(200000, target, marginals,
                                                     rng, options);
  REQUIRE(draw_or.has_value());
  const Eigen::MatrixXd& x = draw_or->observed.X;

  auto fit_or = magmaan::data::fit_ordinal_pair_observed_joint_ml(
      x.col(0), x.col(1), 3, 3);
  REQUIRE(fit_or.has_value());
  CHECK(std::abs(fit_or->fit.rho - 0.5) < 0.02);
}
