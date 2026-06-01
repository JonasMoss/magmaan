#include <doctest/doctest.h>

#include <random>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/sim/population.hpp"

namespace {

Eigen::MatrixXd corr2(double r) {
  Eigen::MatrixXd R(2, 2);
  R << 1.0, r,
       r, 1.0;
  return R;
}

}  // namespace

TEST_CASE("copula population t-copula path projects mixed observed columns") {
  magmaan::sim::TCopulaSpec copula;
  copula.df = 6.0;
  copula.corr = corr2(0.35);

  Eigen::VectorXd binary_threshold(1);
  binary_threshold << 0.0;

  magmaan::sim::CopulaPopulation population;
  population.marginals = {
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal(2.0, 1.5)};
  population.observed.kinds = {
      magmaan::sim::ObservedKind::Ordinal,
      magmaan::sim::ObservedKind::Continuous};
  population.observed.thresholds = {binary_threshold, Eigen::VectorXd{}};

  std::mt19937_64 rng(20260607);
  auto draw_or = magmaan::sim::simulate_mixed_population_t_copula(
      400, copula, population, rng);
  if (!draw_or.has_value()) MESSAGE(draw_or.error().detail);
  REQUIRE(draw_or.has_value());
  const auto& draw = *draw_or;
  REQUIRE(draw.latent.rows() == 400);
  REQUIRE(draw.latent.cols() == 2);
  REQUIRE(draw.observed.X.rows() == 400);
  REQUIRE(draw.observed.X.cols() == 2);
  CHECK(draw.observed.ordered == std::vector<std::int32_t>{1, 0});
  CHECK(draw.observed.n_levels == std::vector<std::int32_t>{2, 0});
  CHECK(draw.observed.X.col(1).isApprox(draw.latent.col(1), 0.0));
  CHECK(draw.observed.category_counts[0].sum() == 400);
  CHECK(draw.observed.X.col(0).minCoeff() >= 1.0);
  CHECK(draw.observed.X.col(0).maxCoeff() <= 2.0);
}

TEST_CASE("copula population bivariate path projects ordinal columns") {
  Eigen::VectorXd probs(3);
  probs << 0.25, 0.50, 0.25;
  auto thresholds_or = magmaan::sim::thresholds_from_probabilities(probs);
  REQUIRE(thresholds_or.has_value());

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 90;
  auto copula_or = magmaan::sim::bivariate_copula_from_tau(
      magmaan::sim::BivariateCopulaFamily::Clayton, 0.4, options);
  REQUIRE(copula_or.has_value());

  magmaan::sim::CopulaPopulation population;
  population.marginals = {
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};
  population.observed.kinds = {
      magmaan::sim::ObservedKind::Ordinal,
      magmaan::sim::ObservedKind::Ordinal};
  population.observed.thresholds = {*thresholds_or, *thresholds_or};

  std::mt19937_64 rng(20260608);
  auto draw_or = magmaan::sim::simulate_mixed_population_bivariate_copula(
      500, *copula_or, population, rng, options);
  if (!draw_or.has_value()) MESSAGE(draw_or.error().detail);
  REQUIRE(draw_or.has_value());
  const auto& draw = *draw_or;
  CHECK(draw.latent.rows() == 500);
  CHECK(draw.latent.cols() == 2);
  CHECK(draw.observed.ordered == std::vector<std::int32_t>{1, 1});
  CHECK(draw.observed.n_levels == std::vector<std::int32_t>{3, 3});
  CHECK(draw.observed.category_counts[0].sum() == 500);
  CHECK(draw.observed.category_counts[1].sum() == 500);
  CHECK(draw.observed.X.col(0).minCoeff() >= 1.0);
  CHECK(draw.observed.X.col(0).maxCoeff() <= 3.0);
  CHECK(draw.observed.X.col(1).minCoeff() >= 1.0);
  CHECK(draw.observed.X.col(1).maxCoeff() <= 3.0);
}
