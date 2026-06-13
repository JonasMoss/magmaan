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
using magmaan::sim::calibrate_ordinal_correlation_multigroup;
using magmaan::sim::multigroup_category_proportions;
using magmaan::sim::ObservedCorrelationMetric;
using magmaan::sim::ObservedKind;
using magmaan::sim::OrdinalCorrelationOptions;
using magmaan::sim::OrdinalMarginalSpec;
using magmaan::sim::ordinal_correlation_population;
using magmaan::sim::ordinal_pair_observed_corr;
using magmaan::sim::raw_data_from_mixed_projections;
using magmaan::sim::simulate_mixed_population_normal;
using magmaan::sim::simulate_ordinal_correlation_normal;
using magmaan::sim::simulate_ordinal_correlation_multigroup_normal;
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

TEST_CASE("multi-group ordinal correlation calibration decomposes by group") {
  Eigen::MatrixXd target1(3, 3);
  target1 << 1.0, 0.25, 0.30,
             0.25, 1.0, 0.20,
             0.30, 0.20, 1.0;
  Eigen::MatrixXd target2(3, 3);
  target2 << 1.0, -0.15, 0.45,
            -0.15, 1.0, 0.10,
             0.45, 0.10, 1.0;
  std::vector<std::vector<OrdinalMarginalSpec>> marginals = {
      {ordinal_marginal({0.20, 0.50, 0.30}), continuous_marginal(),
       ordinal_marginal({0.40, 0.60})},
      {ordinal_marginal({0.30, 0.40, 0.30}), continuous_marginal(),
       ordinal_marginal({0.55, 0.45})}};
  OrdinalCorrelationOptions options;
  options.metric = ObservedCorrelationMetric::PearsonCodes;

  auto mg_or = calibrate_ordinal_correlation_multigroup(
      {target1, target2}, marginals, {}, options);
  REQUIRE(mg_or.has_value());
  REQUIRE(mg_or->groups.size() == 2);
  CHECK(mg_or->group_labels == std::vector<std::string>{"1", "2"});

  auto g1_or = calibrate_ordinal_correlation(target1, marginals[0], options);
  auto g2_or = calibrate_ordinal_correlation(target2, marginals[1], options);
  REQUIRE(g1_or.has_value());
  REQUIRE(g2_or.has_value());
  CHECK(mg_or->groups[0].latent_corr.isApprox(g1_or->latent_corr, 0.0));
  CHECK(mg_or->groups[0].achieved_corr.isApprox(g1_or->achieved_corr, 0.0));
  CHECK(mg_or->groups[1].latent_corr.isApprox(g2_or->latent_corr, 0.0));
  CHECK(mg_or->groups[1].achieved_corr.isApprox(g2_or->achieved_corr, 0.0));

  auto labeled_or = calibrate_ordinal_correlation_multigroup(
      {target1, target2}, marginals, {"school_a", "school_b"}, options);
  REQUIRE(labeled_or.has_value());
  CHECK(labeled_or->group_labels ==
        std::vector<std::string>{"school_a", "school_b"});
}

TEST_CASE("multi-group ordinal correlation calibration validates shared shape") {
  Eigen::MatrixXd target(2, 2);
  target << 1.0, 0.2,
            0.2, 1.0;
  std::vector<std::vector<OrdinalMarginalSpec>> ok = {
      {ordinal_marginal({0.50, 0.50}), continuous_marginal()},
      {ordinal_marginal({0.40, 0.60}), continuous_marginal()}};
  auto size_bad = calibrate_ordinal_correlation_multigroup({target}, ok);
  REQUIRE_FALSE(size_bad.has_value());
  CHECK(size_bad.error().kind == magmaan::SimError::Kind::InvalidInput);

  std::vector<std::vector<OrdinalMarginalSpec>> kind_bad = {
      {ordinal_marginal({0.50, 0.50}), continuous_marginal()},
      {continuous_marginal(), continuous_marginal()}};
  auto kind_or = calibrate_ordinal_correlation_multigroup(
      {target, target}, kind_bad);
  REQUIRE_FALSE(kind_or.has_value());
  CHECK(kind_or.error().kind == magmaan::SimError::Kind::InvalidInput);

  std::vector<std::vector<OrdinalMarginalSpec>> level_bad = {
      {ordinal_marginal({0.50, 0.50}), continuous_marginal()},
      {ordinal_marginal({0.20, 0.30, 0.50}), continuous_marginal()}};
  auto level_or = calibrate_ordinal_correlation_multigroup(
      {target, target}, level_bad);
  REQUIRE_FALSE(level_or.has_value());
  CHECK(level_or.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("multi-group ordinal correlation draw honors unequal n and proportions") {
  Eigen::MatrixXd target1(3, 3);
  target1 << 1.0, 0.20, 0.25,
             0.20, 1.0, 0.15,
             0.25, 0.15, 1.0;
  Eigen::MatrixXd target2(3, 3);
  target2 << 1.0, 0.35, -0.10,
             0.35, 1.0, 0.25,
            -0.10, 0.25, 1.0;
  std::vector<std::vector<OrdinalMarginalSpec>> marginals = {
      {ordinal_marginal({0.25, 0.50, 0.25}), continuous_marginal(),
       ordinal_marginal({0.30, 0.70})},
      {ordinal_marginal({0.35, 0.35, 0.30}), continuous_marginal(),
       ordinal_marginal({0.60, 0.40})}};
  auto cal_or = calibrate_ordinal_correlation_multigroup(
      {target1, target2}, marginals, {"g1", "g2"});
  REQUIRE(cal_or.has_value());

  std::mt19937_64 rng1(12345ULL);
  auto draw1_or = simulate_ordinal_correlation_multigroup_normal(
      {40, 55}, *cal_or, rng1);
  REQUIRE(draw1_or.has_value());
  std::mt19937_64 rng2(12345ULL);
  auto draw2_or = simulate_ordinal_correlation_multigroup_normal(
      {40, 55}, *cal_or, rng2);
  REQUIRE(draw2_or.has_value());
  REQUIRE(draw1_or->size() == 2);
  CHECK((*draw1_or)[0].observed.X.rows() == 40);
  CHECK((*draw1_or)[1].observed.X.rows() == 55);
  CHECK((*draw1_or)[0].observed.X.isApprox((*draw2_or)[0].observed.X, 0.0));
  CHECK((*draw1_or)[1].observed.X.isApprox((*draw2_or)[1].observed.X, 0.0));

  for (const auto& group : *draw1_or) {
    REQUIRE(group.observed.category_proportions.size() == 3);
    CHECK(group.observed.category_proportions[0].size() == 3);
    CHECK(group.observed.category_proportions[1].size() == 0);
    CHECK(group.observed.category_proportions[2].size() == 2);
    CHECK(group.observed.category_proportions[0].sum() ==
          doctest::Approx(1.0));
    CHECK(group.observed.category_proportions[2].sum() ==
          doctest::Approx(1.0));
  }

  auto props_or = multigroup_category_proportions(
      cal_or->group_labels, *draw1_or);
  REQUIRE(props_or.has_value());
  REQUIRE(props_or->size() == 2);
  CHECK((*props_or)[0].group_label == "g1");
  CHECK((*props_or)[1].group_label == "g2");
  CHECK((*props_or)[0].category_proportions[0].size() == 3);

  auto bad_n = simulate_ordinal_correlation_multigroup_normal(
      {40}, *cal_or, rng1);
  REQUIRE_FALSE(bad_n.has_value());
  CHECK(bad_n.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("raw_data_from_mixed_projections composes multi-block metadata") {
  Eigen::MatrixXd latent1(4, 2);
  latent1 << -1.0, 10.0,
             -0.2, 11.0,
              0.3, 12.0,
              1.1, 13.0;
  Eigen::MatrixXd latent2(3, 2);
  latent2 << -1.2, 20.0,
              0.1, 21.0,
              1.4, 22.0;
  magmaan::sim::MixedProjectionSpec spec;
  spec.kinds = {ObservedKind::Ordinal, ObservedKind::Continuous};
  Eigen::VectorXd th(2);
  th << -0.5, 0.5;
  spec.thresholds = {th, Eigen::VectorXd{}};
  auto p1_or = magmaan::sim::project_mixed_matrix(latent1, spec);
  auto p2_or = magmaan::sim::project_mixed_matrix(latent2, spec);
  REQUIRE(p1_or.has_value());
  REQUIRE(p2_or.has_value());

  auto raw_or = raw_data_from_mixed_projections(
      {*p1_or, *p2_or}, {"a", "b"}, {"y_ord", "x_cont"},
      {{"low", "mid", "high"}, {}});
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 2);
  CHECK(raw_or->X[0].rows() == 4);
  CHECK(raw_or->X[1].rows() == 3);
  CHECK(raw_or->group_labels == std::vector<std::string>{"a", "b"});
  CHECK(raw_or->variable_names ==
        std::vector<std::string>{"y_ord", "x_cont"});
  REQUIRE(raw_or->ordinal_level_labels.size() == 2);
  CHECK(raw_or->ordinal_level_labels[0] ==
        std::vector<std::string>{"low", "mid", "high"});
  CHECK(raw_or->ordinal_level_labels[1].empty());

  auto default_labels_or = raw_data_from_mixed_projections({*p1_or, *p2_or});
  REQUIRE(default_labels_or.has_value());
  CHECK(default_labels_or->group_labels ==
        std::vector<std::string>{"1", "2"});
  CHECK(default_labels_or->ordinal_level_labels[0] ==
        std::vector<std::string>{"1", "2", "3"});

  auto bad_labels = raw_data_from_mixed_projections(
      {*p1_or, *p2_or}, {"only_one"});
  REQUIRE_FALSE(bad_labels.has_value());
  CHECK(bad_labels.error().kind == magmaan::SimError::Kind::InvalidInput);

  auto p_bad = *p2_or;
  p_bad.X.conservativeResize(Eigen::NoChange, 1);
  auto bad_cols = raw_data_from_mixed_projections({*p1_or, p_bad});
  REQUIRE_FALSE(bad_cols.has_value());
  CHECK(bad_cols.error().kind == magmaan::SimError::Kind::InvalidInput);
}
