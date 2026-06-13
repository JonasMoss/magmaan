#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/sim/model_implied.hpp"
#include "magmaan/sim/population.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

namespace {

struct BuiltModel {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames names;
  magmaan::model::MatrixRep rep;
};

BuiltModel must_build(std::string_view src,
                      magmaan::spec::BuildOptions opts = {}) {
  auto flat = magmaan::parse::Parser::parse(src);
  REQUIRE(flat.has_value());
  BuiltModel out;
  auto pt = magmaan::spec::build(*flat, opts, nullptr, &out.names);
  REQUIRE(pt.has_value());
  out.pt = std::move(*pt);
  auto rep = magmaan::model::build_matrix_rep(out.pt, &out.names);
  REQUIRE(rep.has_value());
  out.rep = std::move(*rep);
  return out;
}

Eigen::VectorXd empty_theta() {
  return Eigen::VectorXd{};
}

double normal_cdf(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

void require_lowered(const magmaan::sim_expected<
                         magmaan::sim::ModelImpliedPopulation>& lowered) {
  if (!lowered.has_value()) MESSAGE(lowered.error().detail);
  REQUIRE(lowered.has_value());
}

}  // namespace

TEST_CASE("model-implied lowering recovers single-group continuous moments") {
  const BuiltModel model = must_build(
      "x1 ~~ 1.5*x1\n"
      "x2 ~~ 2.0*x2\n"
      "x1 ~~ 0.3*x2\n"
      "x1 ~ 0.4*1\n"
      "x2 ~ -0.2*1\n");
  const Eigen::VectorXd theta = empty_theta();

  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);
  REQUIRE(lowered->groups.size() == 1);
  CHECK(lowered->ov_names == std::vector<std::string>{"x1", "x2"});
  CHECK(lowered->kinds == std::vector<magmaan::sim::ObservedKind>{
      magmaan::sim::ObservedKind::Continuous,
      magmaan::sim::ObservedKind::Continuous});
  CHECK(lowered->n_levels == std::vector<std::int32_t>{0, 0});

  Eigen::MatrixXd expected_sigma(2, 2);
  expected_sigma << 1.5, 0.3,
                    0.3, 2.0;
  Eigen::VectorXd expected_mu(2);
  expected_mu << 0.4, -0.2;
  CHECK(lowered->groups[0].latent.covariance.isApprox(expected_sigma, 0.0));
  CHECK(lowered->groups[0].latent.mean.isApprox(expected_mu, 0.0));
}

TEST_CASE("model-implied lowering preserves multi-group blocks") {
  magmaan::spec::BuildOptions opts;
  opts.n_groups = 2;
  const BuiltModel model = must_build(
      "x1 ~~ c(1.0, 2.0)*x1\n"
      "x2 ~~ c(1.5, 2.5)*x2\n"
      "x1 ~~ c(0.2, -0.1)*x2\n"
      "x1 ~ c(0.0, 1.0)*1\n"
      "x2 ~ c(0.5, -0.5)*1\n",
      opts);
  const Eigen::VectorXd theta = empty_theta();

  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);
  REQUIRE(lowered->groups.size() == 2);

  auto ev = magmaan::model::ModelEvaluator::build(model.pt, model.rep);
  REQUIRE(ev.has_value());
  auto moments = ev->sigma(theta);
  REQUIRE(moments.has_value());
  for (std::size_t b = 0; b < lowered->groups.size(); ++b) {
    CHECK(lowered->groups[b].latent.covariance.isApprox(
        moments->sigma[b], 0.0));
    CHECK(lowered->groups[b].latent.mean.isApprox(moments->mu[b], 0.0));
  }
}

TEST_CASE("model-implied ordinal thresholds reproduce category probabilities") {
  const BuiltModel model = must_build(
      "x1 ~~ 1*x1\n"
      "x1 | -0.5*t1 + 0.75*t2\n");
  const Eigen::VectorXd theta = empty_theta();
  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);
  CHECK(lowered->kinds == std::vector<magmaan::sim::ObservedKind>{
      magmaan::sim::ObservedKind::Ordinal});
  CHECK(lowered->n_levels == std::vector<std::int32_t>{3});

  std::mt19937_64 rng(20260613);
  magmaan::sim::GeneratorSpec generator;
  auto draws = magmaan::sim::simulate_model_implied(
      30000, *lowered, generator, rng);
  if (!draws.has_value()) MESSAGE(draws.error().detail);
  REQUIRE(draws.has_value());
  REQUIRE(draws->size() == 1);
  const auto& counts = (*draws)[0].observed.category_counts[0];
  REQUIRE(counts.size() == 3);
  const double n = static_cast<double>(counts.sum());
  const double p1 = normal_cdf(-0.5);
  const double p2 = normal_cdf(0.75) - normal_cdf(-0.5);
  const double p3 = 1.0 - normal_cdf(0.75);
  CHECK(counts(0) / n == doctest::Approx(p1).epsilon(0.04));
  CHECK(counts(1) / n == doctest::Approx(p2).epsilon(0.04));
  CHECK(counts(2) / n == doctest::Approx(p3).epsilon(0.04));
}

TEST_CASE("model-implied mixed draw carries continuous and ordinal columns") {
  const BuiltModel model = must_build(
      "x1 ~~ 1.2*x1\n"
      "x2 ~~ 1*x2\n"
      "x1 ~~ 0.2*x2\n"
      "x2 | -0.25*t1 + 0.8*t2\n");
  const Eigen::VectorXd theta = empty_theta();
  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);

  std::mt19937_64 rng(20260614);
  magmaan::sim::GeneratorSpec generator;
  auto draw = magmaan::sim::simulate_model_implied_group(
      1000, lowered->groups[0], generator, rng);
  if (!draw.has_value()) MESSAGE(draw.error().detail);
  REQUIRE(draw.has_value());
  CHECK(draw->observed.ordered == std::vector<std::int32_t>{0, 1});
  CHECK(draw->observed.n_levels == std::vector<std::int32_t>{0, 3});
  CHECK(draw->observed.X.col(0).isApprox(draw->latent.col(0), 0.0));
  CHECK(draw->observed.X.col(1).minCoeff() >= 1.0);
  CHECK(draw->observed.X.col(1).maxCoeff() <= 3.0);
}

TEST_CASE("model-implied all-continuous draw is the normal population path") {
  const BuiltModel model = must_build(
      "x1 ~~ 1.5*x1\n"
      "x2 ~~ 2.0*x2\n"
      "x1 ~~ 0.3*x2\n");
  const Eigen::VectorXd theta = empty_theta();
  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);

  std::mt19937_64 rng_model(20260615);
  magmaan::sim::GeneratorSpec generator;
  auto model_draw = magmaan::sim::simulate_model_implied_group(
      50, lowered->groups[0], generator, rng_model);
  REQUIRE(model_draw.has_value());

  std::mt19937_64 rng_direct(20260615);
  auto direct_draw = magmaan::sim::simulate_mixed_population_normal(
      50, lowered->groups[0], rng_direct);
  REQUIRE(direct_draw.has_value());
  CHECK(model_draw->latent.isApprox(direct_draw->latent, 0.0));
  CHECK(model_draw->observed.X.isApprox(direct_draw->observed.X, 0.0));
}

TEST_CASE("model-implied fully fixed ordinal model accepts empty theta") {
  const BuiltModel model = must_build(
      "x1 ~~ 1*x1\n"
      "x1 | -0.2*t1 + 0.9*t2\n");
  const Eigen::VectorXd theta = empty_theta();
  CHECK(model.pt.n_free() == 0);
  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);
  REQUIRE(lowered->groups.size() == 1);
  CHECK(lowered->groups[0].observed.thresholds[0](0) ==
        doctest::Approx(-0.2));
  CHECK(lowered->groups[0].observed.thresholds[0](1) ==
        doctest::Approx(0.9));
}

TEST_CASE("model-implied lowering sorts authored threshold rows") {
  const BuiltModel model = must_build(
      "x1 ~~ 1*x1\n"
      "x1 | 1*t1 + -1*t2\n");
  const Eigen::VectorXd theta = empty_theta();
  auto lowered = magmaan::sim::lower_model_implied(
      model.pt, model.rep, theta);
  require_lowered(lowered);
  const Eigen::VectorXd& thresholds =
      lowered->groups[0].observed.thresholds[0];
  REQUIRE(thresholds.size() == 2);
  CHECK(thresholds(0) == doctest::Approx(-1.0));
  CHECK(thresholds(1) == doctest::Approx(1.0));
}

TEST_CASE("model-implied simulation surfaces sim errors") {
  const BuiltModel free_model = must_build("x1 ~~ x1\n");
  const Eigen::VectorXd empty = empty_theta();
  auto bad_theta = magmaan::sim::lower_model_implied(
      free_model.pt, free_model.rep, empty);
  REQUIRE_FALSE(bad_theta.has_value());
  CHECK(bad_theta.error().kind == magmaan::SimError::Kind::NumericIssue);

  const BuiltModel non_pd = must_build(
      "x1 ~~ 1*x1\n"
      "x2 ~~ 1*x2\n"
      "x1 ~~ 2*x2\n");
  auto lowered = magmaan::sim::lower_model_implied(
      non_pd.pt, non_pd.rep, empty);
  require_lowered(lowered);
  std::mt19937_64 rng(20260616);
  magmaan::sim::GeneratorSpec generator;
  auto draw = magmaan::sim::simulate_model_implied_group(
      20, lowered->groups[0], generator, rng);
  REQUIRE_FALSE(draw.has_value());
  CHECK(draw.error().kind == magmaan::SimError::Kind::NonPositiveDefinite);
}
