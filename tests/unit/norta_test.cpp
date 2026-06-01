#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/error.hpp"
#include "magmaan/sim/norta.hpp"

namespace {

Eigen::MatrixXd corr2(double r) {
  Eigen::MatrixXd R(2, 2);
  R << 1.0, r,
       r, 1.0;
  return R;
}

double sample_corr(const Eigen::MatrixXd& X, Eigen::Index a, Eigen::Index b) {
  const Eigen::VectorXd xa = X.col(a).array() - X.col(a).mean();
  const Eigen::VectorXd xb = X.col(b).array() - X.col(b).mean();
  return xa.dot(xb) / std::sqrt(xa.squaredNorm() * xb.squaredNorm());
}

Eigen::MatrixXd sample_cov(const Eigen::MatrixXd& X) {
  const Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
  return centered.transpose() * centered / static_cast<double>(X.rows());
}

double sample_skewness(const Eigen::MatrixXd& X, Eigen::Index j) {
  const Eigen::ArrayXd centered = X.col(j).array() - X.col(j).mean();
  const double var = centered.square().mean();
  return centered.pow(3).mean() / std::pow(var, 1.5);
}

double sample_excess_kurtosis(const Eigen::MatrixXd& X, Eigen::Index j) {
  const Eigen::ArrayXd centered = X.col(j).array() - X.col(j).mean();
  const double var = centered.square().mean();
  return centered.pow(4).mean() / (var * var) - 3.0;
}

void check_ig_moment_system(const Eigen::MatrixXd& root,
                            const Eigen::VectorXd& generator_skewness,
                            const Eigen::VectorXd& generator_excess_kurtosis,
                            const Eigen::VectorXd& target_skewness,
                            const Eigen::VectorXd& target_excess_kurtosis) {
  Eigen::VectorXd got_skew(root.rows());
  Eigen::VectorXd got_kurt(root.rows());
  for (Eigen::Index i = 0; i < root.rows(); ++i) {
    const double variance = root.row(i).squaredNorm();
    const double sd = std::sqrt(variance);
    got_skew(i) = 0.0;
    got_kurt(i) = 0.0;
    for (Eigen::Index j = 0; j < root.cols(); ++j) {
      const double a = root(i, j);
      const double a2 = a * a;
      got_skew(i) += a2 * a / (sd * sd * sd) * generator_skewness(j);
      got_kurt(i) += a2 * a2 / (variance * variance) *
                     generator_excess_kurtosis(j);
    }
  }
  CHECK((got_skew - target_skewness).norm() == doctest::Approx(0.0).epsilon(1e-10));
  CHECK((got_kurt - target_excess_kurtosis).norm() == doctest::Approx(0.0).epsilon(1e-10));
}

magmaan::sim::BivariateCopulaFamily copula_family_from_string(
    const std::string& family) {
  if (family == "indep") return magmaan::sim::BivariateCopulaFamily::Independence;
  if (family == "clayton") return magmaan::sim::BivariateCopulaFamily::Clayton;
  if (family == "gumbel") return magmaan::sim::BivariateCopulaFamily::Gumbel;
  if (family == "frank") return magmaan::sim::BivariateCopulaFamily::Frank;
  if (family == "joe") return magmaan::sim::BivariateCopulaFamily::Joe;
  return magmaan::sim::BivariateCopulaFamily::Independence;
}

void assign_copula_from_json(magmaan::sim::BivariateCopulaSpec& copula,
                             const nlohmann::json& spec) {
  copula.family = copula_family_from_string(spec["family"].get<std::string>());
  if (spec["parameters"].is_number()) {
    copula.theta = spec["parameters"].get<double>();
  } else {
    copula.theta = 0.0;
  }
}

}  // namespace

TEST_CASE("normal quantile round-trips through normal CDF") {
  for (double u : {0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999}) {
    auto z_or = magmaan::sim::normal_quantile(u);
    REQUIRE(z_or.has_value());
    CHECK(magmaan::sim::normal_cdf(*z_or) == doctest::Approx(u).epsilon(1e-13));
  }
}

TEST_CASE("NORTA calibration keeps normal marginals unchanged") {
  const Eigen::MatrixXd target = corr2(0.42);
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};

  auto cal_or = magmaan::sim::calibrate_norta(target, marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) == doctest::Approx(0.42).epsilon(1e-8));
  CHECK(cal_or->latent_corr(1, 0) == doctest::Approx(0.42).epsilon(1e-8));
}

TEST_CASE("NORTA calibration matches analytic lognormal copula correlation") {
  const double sigma = 0.8;
  const double target_r = 0.35;
  const double s2 = sigma * sigma;
  const double latent_r = std::log(1.0 + target_r * (std::exp(s2) - 1.0)) / s2;

  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(sigma),
      magmaan::sim::MarginalSpec::standardized_lognormal(sigma)};

  auto cal_or = magmaan::sim::calibrate_norta(corr2(target_r), marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) == doctest::Approx(latent_r).epsilon(3e-6));
}

TEST_CASE("NORTA reports infeasible lognormal target correlations") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(0.8),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.8)};

  auto cal_or = magmaan::sim::calibrate_norta(corr2(-0.8), marginals);
  REQUIRE_FALSE(cal_or.has_value());
  CHECK(cal_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("independent generator supports Tukey g-and-h marginals") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::tukey_g_h(0.35, 0.05, 1.0, 2.0),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.4, -2.0, 1.5),
      magmaan::sim::MarginalSpec::standard_normal()};

  std::mt19937_64 rng(20260601);
  magmaan::sim::IndependentOptions options;
  options.quadrature_points = 35;
  auto X_or = magmaan::sim::simulate_independent_matrix(30000, marginals, rng, options);
  REQUIRE(X_or.has_value());
  const auto& X = *X_or;
  REQUIRE(X.rows() == 30000);
  REQUIRE(X.cols() == 3);

  CHECK(std::abs(X.col(0).mean() - 1.0) < 0.08);
  CHECK(std::abs(X.col(1).mean() + 2.0) < 0.05);
  CHECK(std::abs(X.col(2).mean()) < 0.03);
  CHECK(std::abs(sample_corr(X, 0, 1)) < 0.03);
  CHECK(std::abs(sample_corr(X, 0, 2)) < 0.03);
  CHECK(std::abs(sample_corr(X, 1, 2)) < 0.03);
}

TEST_CASE("independent raw generator wraps one complete data block") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::tukey_g_h(0.0, 0.1)};

  std::mt19937_64 rng(11);
  auto raw_or = magmaan::sim::simulate_independent_raw(25, marginals, rng);
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 1u);
  CHECK(raw_or->mask.empty());
  CHECK(raw_or->X[0].rows() == 25);
  CHECK(raw_or->X[0].cols() == 2);
}

TEST_CASE("IG calibration solves generator moments with Cholesky root") {
  const Eigen::MatrixXd sigma = corr2(0.30);
  Eigen::VectorXd target_skewness(2);
  target_skewness << 0.20, 0.45;
  Eigen::VectorXd target_excess_kurtosis(2);
  target_excess_kurtosis << 0.50, 0.90;

  magmaan::sim::IgOptions options;
  options.root = magmaan::sim::IgRootKind::Cholesky;
  options.generator_family = magmaan::sim::MomentMatchFamily::TukeyGH;
  options.moment_match.quadrature_points = 81;

  auto cal_or = magmaan::sim::calibrate_ig(
      sigma, target_skewness, target_excess_kurtosis, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  const auto& cal = *cal_or;
  CHECK((cal.root * cal.root.transpose() - sigma).norm() ==
        doctest::Approx(0.0).epsilon(1e-12));
  CHECK(cal.generator_marginals.size() == 2u);
  CHECK(cal.generator_marginals[0].kind == magmaan::sim::MarginalKind::TukeyGH);
  CHECK(cal.generator_marginals[1].kind == magmaan::sim::MarginalKind::TukeyGH);
  check_ig_moment_system(cal.root,
                         cal.generator_skewness,
                         cal.generator_excess_kurtosis,
                         target_skewness,
                         target_excess_kurtosis);

  for (Eigen::Index j = 0; j < cal.generator_skewness.size(); ++j) {
    auto summary_or = magmaan::sim::marginal_moment_summary(
        cal.generator_marginals[static_cast<std::size_t>(j)], 81);
    REQUIRE(summary_or.has_value());
    CHECK(summary_or->skewness ==
          doctest::Approx(cal.generator_skewness(j)).epsilon(1e-6));
    CHECK(summary_or->excess_kurtosis ==
          doctest::Approx(cal.generator_excess_kurtosis(j)).epsilon(1e-6));
  }
}

TEST_CASE("IG calibration exposes symmetric square-root option") {
  const Eigen::MatrixXd sigma = corr2(0.30);
  Eigen::VectorXd target_skewness(2);
  target_skewness << 0.20, 0.45;
  Eigen::VectorXd target_excess_kurtosis(2);
  target_excess_kurtosis << 0.50, 0.90;

  magmaan::sim::IgOptions cholesky_options;
  cholesky_options.root = magmaan::sim::IgRootKind::Cholesky;
  cholesky_options.moment_match.quadrature_points = 81;
  auto chol_or = magmaan::sim::calibrate_ig(
      sigma, target_skewness, target_excess_kurtosis, cholesky_options);
  if (!chol_or.has_value()) MESSAGE(chol_or.error().detail);
  REQUIRE(chol_or.has_value());
  if (!chol_or.has_value()) return;

  magmaan::sim::IgOptions symmetric_options;
  symmetric_options.root = magmaan::sim::IgRootKind::Symmetric;
  symmetric_options.moment_match.quadrature_points = 81;
  auto sym_or = magmaan::sim::calibrate_ig(
      sigma, target_skewness, target_excess_kurtosis, symmetric_options);
  if (!sym_or.has_value()) MESSAGE(sym_or.error().detail);
  REQUIRE(sym_or.has_value());
  if (!sym_or.has_value()) return;

  CHECK((sym_or->root - sym_or->root.transpose()).norm() ==
        doctest::Approx(0.0).epsilon(1e-12));
  CHECK((sym_or->root * sym_or->root.transpose() - sigma).norm() ==
        doctest::Approx(0.0).epsilon(1e-12));
  CHECK((chol_or->root - sym_or->root).norm() > 1e-3);
  check_ig_moment_system(sym_or->root,
                         sym_or->generator_skewness,
                         sym_or->generator_excess_kurtosis,
                         target_skewness,
                         target_excess_kurtosis);
}

TEST_CASE("Tukey g-and-h moment matcher recovers Kowalchuk-Headrick shape") {
  magmaan::sim::MomentMatchSpec spec;
  spec.family = magmaan::sim::MomentMatchFamily::TukeyGH;
  spec.shape.skewness = 3.0;
  spec.shape.excess_kurtosis = 20.0;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto fit_or = magmaan::sim::fit_marginal_to_moments(spec, options);
  REQUIRE(fit_or.has_value());
  if (!fit_or.has_value()) return;

  CHECK(fit_or->marginal.kind == magmaan::sim::MarginalKind::TukeyGH);
  CHECK(fit_or->marginal.g == doctest::Approx(0.690678).epsilon(2e-3));
  CHECK(fit_or->marginal.h == doctest::Approx(0.009831).epsilon(3e-2));
  CHECK(fit_or->moments.skewness == doctest::Approx(3.0).epsilon(2e-6));
  CHECK(fit_or->moments.excess_kurtosis == doctest::Approx(20.0).epsilon(2e-6));
}

TEST_CASE("moment-matched Tukey marginals feed NORTA calibration") {
  magmaan::sim::MomentMatchSpec left;
  left.shape.skewness = 3.0;
  left.shape.excess_kurtosis = 20.0;
  magmaan::sim::MomentMatchSpec right;
  right.shape.skewness = -3.0;
  right.shape.excess_kurtosis = 20.0;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto left_or = magmaan::sim::fit_marginal_to_moments(left, options);
  auto right_or = magmaan::sim::fit_marginal_to_moments(right, options);
  REQUIRE(left_or.has_value());
  REQUIRE(right_or.has_value());
  if (!left_or.has_value() || !right_or.has_value()) return;
  CHECK(left_or->marginal.g > 0.0);
  CHECK(right_or->marginal.g < 0.0);

  const std::vector<magmaan::sim::MarginalSpec> marginals{
      left_or->marginal, right_or->marginal};
  auto cal_or = magmaan::sim::calibrate_norta(corr2(0.30), marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) > 0.30);
}

TEST_CASE("Pearson moment matcher matches PearsonDS goldens") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/sim/pearson_moment_match.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  for (const auto& c : fixture["cases"]) {
    magmaan::sim::MomentMatchSpec spec;
    spec.family = magmaan::sim::MomentMatchFamily::Pearson;
    spec.mean = c["mean"].get<double>();
    spec.sd = c["sd"].get<double>();
    spec.shape.skewness = c["skewness"].get<double>();
    spec.shape.excess_kurtosis = c["excess_kurtosis"].get<double>();

    auto fit_or = magmaan::sim::fit_marginal_to_moments(spec);
    const bool supported = c["supported"].get<bool>();
    if (!supported) {
      REQUIRE_FALSE(fit_or.has_value());
      CHECK(fit_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
      continue;
    }
    REQUIRE(fit_or.has_value());
    if (!fit_or.has_value()) continue;
    const auto& got = fit_or->marginal;
    CHECK(got.kind == magmaan::sim::MarginalKind::Pearson);
    CHECK(got.pearson_type == c["pearson_type"].get<int>());
    const double got_params[] = {
        got.pearson_p1, got.pearson_p2, got.pearson_p3, got.pearson_p4};
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(got_params[i] == doctest::Approx(
          c["pearson_params"][i].get<double>()).epsilon(2e-11));
    }
    const auto& probs = c["probabilities"];
    const auto& quantiles = c["quantiles"];
    for (std::size_t i = 0; i < probs.size(); ++i) {
      auto q_or = magmaan::sim::marginal_quantile(
          got, probs[i].get<double>());
      REQUIRE(q_or.has_value());
      CHECK(*q_or == doctest::Approx(
          quantiles[i].get<double>()).epsilon(5e-7).scale(1.0));
    }
  }
}

TEST_CASE("Pearson Type IV marginals feed independent generation") {
  magmaan::sim::MomentMatchSpec spec;
  spec.family = magmaan::sim::MomentMatchFamily::Pearson;
  spec.shape.skewness = 1.0;
  spec.shape.excess_kurtosis = 2.0;

  auto fit_or = magmaan::sim::fit_marginal_to_moments(spec);
  REQUIRE(fit_or.has_value());
  REQUIRE(fit_or->marginal.kind == magmaan::sim::MarginalKind::Pearson);
  REQUIRE(fit_or->marginal.pearson_type == 4);

  std::mt19937_64 rng(20260602);
  const std::vector<magmaan::sim::MarginalSpec> marginals{fit_or->marginal};
  auto X_or = magmaan::sim::simulate_independent_matrix(16, marginals, rng);
  REQUIRE(X_or.has_value());
  REQUIRE(X_or->rows() == 16);
  REQUIRE(X_or->cols() == 1);
  CHECK(X_or->array().isFinite().all());
}

TEST_CASE("Johnson moment matcher matches SuppDists goldens") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/sim/johnson_moment_match.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  for (const auto& c : fixture["cases"]) {
    const std::string id = c["id"].get<std::string>();
    CAPTURE(id);
    magmaan::sim::MomentMatchSpec spec;
    spec.family = magmaan::sim::MomentMatchFamily::Johnson;
    spec.mean = c["mean"].get<double>();
    spec.sd = c["sd"].get<double>();
    spec.shape.skewness = c["skewness"].get<double>();
    spec.shape.excess_kurtosis = c["excess_kurtosis"].get<double>();

    auto fit_or = magmaan::sim::fit_marginal_to_moments(spec);
    REQUIRE(fit_or.has_value());
    if (!fit_or.has_value()) continue;
    const auto& got = fit_or->marginal;
    CHECK(got.kind == magmaan::sim::MarginalKind::Johnson);
    CHECK(got.johnson_type == c["johnson_type"].get<int>());
    CHECK(got.johnson_gamma == doctest::Approx(
        c["johnson_gamma"].get<double>()).epsilon(1e-7).scale(1.0));
    CHECK(got.johnson_delta == doctest::Approx(
        c["johnson_delta"].get<double>()).epsilon(1e-7).scale(1.0));

    const auto& probs = c["probabilities"];
    const auto& quantiles = c["quantiles"];
    for (std::size_t i = 0; i < probs.size(); ++i) {
      auto q_or = magmaan::sim::marginal_quantile(got, probs[i].get<double>());
      REQUIRE(q_or.has_value());
      CHECK(*q_or == doctest::Approx(
          quantiles[i].get<double>()).epsilon(1e-7).scale(1.0));
    }
  }
}

TEST_CASE("Johnson moment matcher fits SU and SB shape moments") {
  magmaan::sim::MomentMatchSpec spec;
  spec.family = magmaan::sim::MomentMatchFamily::Johnson;
  spec.shape.skewness = 0.8;
  spec.shape.excess_kurtosis = 1.5;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto fit_or = magmaan::sim::fit_marginal_to_moments(spec, options);
  if (!fit_or.has_value()) MESSAGE(fit_or.error().detail);
  REQUIRE(fit_or.has_value());
  if (!fit_or.has_value()) return;
  CHECK(fit_or->marginal.kind == magmaan::sim::MarginalKind::Johnson);
  CHECK((fit_or->marginal.johnson_type == 2 ||
         fit_or->marginal.johnson_type == 3));
  CHECK(fit_or->moments.skewness == doctest::Approx(0.8).epsilon(2e-6));
  CHECK(fit_or->moments.excess_kurtosis == doctest::Approx(1.5).epsilon(2e-6));

  spec.shape.skewness = 0.0;
  spec.shape.excess_kurtosis = -0.8;
  auto bounded_or = magmaan::sim::fit_marginal_to_moments(spec, options);
  if (!bounded_or.has_value()) MESSAGE(bounded_or.error().detail);
  REQUIRE(bounded_or.has_value());
  if (!bounded_or.has_value()) return;
  CHECK(bounded_or->marginal.kind == magmaan::sim::MarginalKind::Johnson);
  CHECK(bounded_or->marginal.johnson_type == 3);
  CHECK(bounded_or->moments.skewness == doctest::Approx(0.0).epsilon(2e-6));
  CHECK(bounded_or->moments.excess_kurtosis ==
        doctest::Approx(-0.8).epsilon(2e-6));
}

TEST_CASE("moment-matched Johnson marginals feed NORTA calibration") {
  magmaan::sim::MomentMatchSpec left;
  left.family = magmaan::sim::MomentMatchFamily::Johnson;
  left.shape.skewness = 0.8;
  left.shape.excess_kurtosis = 1.5;
  magmaan::sim::MomentMatchSpec right = left;
  right.shape.skewness = -0.8;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto left_or = magmaan::sim::fit_marginal_to_moments(left, options);
  auto right_or = magmaan::sim::fit_marginal_to_moments(right, options);
  REQUIRE(left_or.has_value());
  REQUIRE(right_or.has_value());
  if (!left_or.has_value() || !right_or.has_value()) return;

  const std::vector<magmaan::sim::MarginalSpec> marginals{
      left_or->marginal, right_or->marginal};
  auto cal_or = magmaan::sim::calibrate_norta(corr2(0.30), marginals);
  REQUIRE(cal_or.has_value());
  CHECK(cal_or->latent_corr(0, 1) > 0.0);
}

TEST_CASE("Fleishman moment matcher fits polynomial shape moments") {
  const double c = 0.12;
  const double d = 0.08;
  const double b = -3.0 * d + std::sqrt(1.0 - 2.0 * c * c - 6.0 * d * d);
  auto source_summary_or = magmaan::sim::marginal_moment_summary(
      magmaan::sim::MarginalSpec::fleishman(b, c, d), 81);
  REQUIRE(source_summary_or.has_value());

  magmaan::sim::MomentMatchSpec spec;
  spec.family = magmaan::sim::MomentMatchFamily::Fleishman;
  spec.shape.skewness = source_summary_or->skewness;
  spec.shape.excess_kurtosis = source_summary_or->excess_kurtosis;

  magmaan::sim::MomentMatchOptions options;
  options.quadrature_points = 81;
  auto fit_or = magmaan::sim::fit_marginal_to_moments(spec, options);
  if (!fit_or.has_value()) MESSAGE(fit_or.error().detail);
  REQUIRE(fit_or.has_value());
  if (!fit_or.has_value()) return;
  CHECK(fit_or->marginal.kind == magmaan::sim::MarginalKind::Fleishman);
  CHECK(fit_or->moments.skewness ==
        doctest::Approx(spec.shape.skewness).epsilon(2e-6));
  CHECK(fit_or->moments.excess_kurtosis ==
        doctest::Approx(spec.shape.excess_kurtosis).epsilon(2e-6));

  auto q_or = magmaan::sim::marginal_quantile(fit_or->marginal, 0.5);
  REQUIRE_FALSE(q_or.has_value());
  CHECK(q_or.error().kind == magmaan::SimError::Kind::InvalidMarginal);
}

TEST_CASE("IG calibration can use Fleishman generator marginals") {
  const Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd target_skewness(2);
  target_skewness << 0.45, -0.30;
  Eigen::VectorXd target_excess_kurtosis(2);
  target_excess_kurtosis << 0.80, 0.55;

  magmaan::sim::IgOptions options;
  options.generator_family = magmaan::sim::MomentMatchFamily::Fleishman;
  options.moment_match.quadrature_points = 81;

  auto cal_or = magmaan::sim::calibrate_ig(
      sigma, target_skewness, target_excess_kurtosis, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  REQUIRE(cal_or->generator_marginals.size() == 2);
  CHECK(cal_or->generator_marginals[0].kind ==
        magmaan::sim::MarginalKind::Fleishman);
  CHECK(cal_or->generator_marginals[1].kind ==
        magmaan::sim::MarginalKind::Fleishman);
  check_ig_moment_system(cal_or->root,
                         cal_or->generator_skewness,
                         cal_or->generator_excess_kurtosis,
                         target_skewness,
                         target_excess_kurtosis);
}

TEST_CASE("IG simulation respects covariance and target marginal shapes") {
  const Eigen::MatrixXd sigma = corr2(0.25);
  Eigen::VectorXd target_skewness(2);
  target_skewness << 0.20, 0.45;
  Eigen::VectorXd target_excess_kurtosis(2);
  target_excess_kurtosis << 0.50, 0.90;

  magmaan::sim::IgOptions options;
  options.root = magmaan::sim::IgRootKind::Cholesky;
  options.generator_family = magmaan::sim::MomentMatchFamily::TukeyGH;
  options.moment_match.quadrature_points = 81;

  std::mt19937_64 rng(20260602);
  auto X_or = magmaan::sim::simulate_ig_matrix(
      60000, sigma, target_skewness, target_excess_kurtosis, rng, options);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());
  if (!X_or.has_value()) return;
  const auto& X = *X_or;
  REQUIRE(X.rows() == 60000);
  REQUIRE(X.cols() == 2);

  const Eigen::MatrixXd cov = sample_cov(X);
  CHECK(std::abs(cov(0, 0) - sigma(0, 0)) < 0.04);
  CHECK(std::abs(cov(1, 1) - sigma(1, 1)) < 0.04);
  CHECK(std::abs(cov(0, 1) - sigma(0, 1)) < 0.04);
  CHECK(std::abs(sample_skewness(X, 0) - target_skewness(0)) < 0.08);
  CHECK(std::abs(sample_skewness(X, 1) - target_skewness(1)) < 0.08);
  CHECK(std::abs(sample_excess_kurtosis(X, 0) - target_excess_kurtosis(0)) < 0.20);
  CHECK(std::abs(sample_excess_kurtosis(X, 1) - target_excess_kurtosis(1)) < 0.20);
}

TEST_CASE("NORTA simulation respects target moments and correlations") {
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.30, -0.20,
            0.30, 1.0, 0.15,
           -0.20, 0.15, 1.0;
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(2.0, 1.5),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.5, -1.0, 2.0),
      magmaan::sim::MarginalSpec::tukey_g_h(0.2, 0.08)};

  std::mt19937_64 rng(20260531);
  magmaan::sim::NortaOptions options;
  options.quadrature_points = 35;
  auto X_or = magmaan::sim::simulate_norta_matrix(20000, target, marginals, rng, options);
  REQUIRE(X_or.has_value());
  const auto& X = *X_or;
  REQUIRE(X.rows() == 20000);
  REQUIRE(X.cols() == 3);

  CHECK(std::abs(X.col(0).mean() - 2.0) < 0.06);
  CHECK(std::abs(X.col(1).mean() + 1.0) < 0.08);
  CHECK(std::abs(X.col(2).mean()) < 0.05);
  CHECK(std::abs(sample_corr(X, 0, 1) - target(0, 1)) < 0.04);
  CHECK(std::abs(sample_corr(X, 0, 2) - target(0, 2)) < 0.04);
  CHECK(std::abs(sample_corr(X, 1, 2) - target(1, 2)) < 0.04);
}

TEST_CASE("t-copula simulation preserves t marginals and copula correlation") {
  Eigen::MatrixXd corr(3, 3);
  corr << 1.0, 0.45, -0.25,
          0.45, 1.0, 0.20,
         -0.25, 0.20, 1.0;
  constexpr double df = 7.0;
  const double t_scale = std::sqrt((df - 2.0) / df);
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::pearson(7, df, 0.0, t_scale, 0.0),
      magmaan::sim::MarginalSpec::pearson(7, df, 0.0, t_scale, 0.0),
      magmaan::sim::MarginalSpec::pearson(7, df, 0.0, t_scale, 0.0)};

  magmaan::sim::TCopulaSpec copula;
  copula.df = df;
  copula.corr = corr;
  std::mt19937_64 rng(20260603);
  auto X_or = magmaan::sim::simulate_t_copula_matrix(
      30000, copula, marginals, rng);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());
  if (!X_or.has_value()) return;
  const auto& X = *X_or;
  REQUIRE(X.rows() == 30000);
  REQUIRE(X.cols() == 3);

  CHECK(std::abs(X.col(0).mean()) < 0.04);
  CHECK(std::abs(sample_corr(X, 0, 1) - corr(0, 1)) < 0.04);
  CHECK(std::abs(sample_corr(X, 0, 2) - corr(0, 2)) < 0.04);
  CHECK(std::abs(sample_corr(X, 1, 2) - corr(1, 2)) < 0.04);
  CHECK(std::abs(sample_excess_kurtosis(X, 0) - 2.0) < 0.65);
}

TEST_CASE("t-copula raw generator validates quantile marginals") {
  magmaan::sim::TCopulaSpec copula;
  copula.df = 5.0;
  copula.corr = corr2(0.30);
  std::mt19937_64 rng(20260604);
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.5)};

  auto raw_or = magmaan::sim::simulate_t_copula_raw(20, copula, marginals, rng);
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 1u);
  CHECK(raw_or->mask.empty());
  CHECK(raw_or->X[0].rows() == 20);
  CHECK(raw_or->X[0].cols() == 2);

  const std::vector<magmaan::sim::MarginalSpec> bad_marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::fleishman(1.0, 0.0, 0.0)};
  auto bad_or = magmaan::sim::simulate_t_copula_matrix(
      20, copula, bad_marginals, rng);
  REQUIRE_FALSE(bad_or.has_value());
  CHECK(bad_or.error().kind == magmaan::SimError::Kind::InvalidMarginal);
}

TEST_CASE("bivariate Archimedean copulas generate expected dependence direction") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};

  struct Case {
    magmaan::sim::BivariateCopulaFamily family;
    double theta;
    double lower;
    double upper;
  };
  const Case cases[] = {
      {magmaan::sim::BivariateCopulaFamily::Independence, 0.0, -0.04, 0.04},
      {magmaan::sim::BivariateCopulaFamily::Clayton, 2.0, 0.55, 1.0},
      {magmaan::sim::BivariateCopulaFamily::Gumbel, 2.0, 0.55, 1.0},
      {magmaan::sim::BivariateCopulaFamily::Frank, 5.0, 0.40, 1.0},
      {magmaan::sim::BivariateCopulaFamily::Frank, -5.0, -1.0, -0.40},
      {magmaan::sim::BivariateCopulaFamily::Joe, 2.0, 0.45, 1.0},
  };

  std::mt19937_64 rng(20260605);
  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 31;
  options.max_bisection_iter = 80;
  for (const auto& c : cases) {
    magmaan::sim::BivariateCopulaSpec copula;
    copula.family = c.family;
    copula.theta = c.theta;
    auto X_or = magmaan::sim::simulate_bivariate_copula_matrix(
        12000, copula, marginals, rng, options);
    if (!X_or.has_value()) MESSAGE(X_or.error().detail);
    REQUIRE(X_or.has_value());
    const double r = sample_corr(*X_or, 0, 1);
    CHECK(r > c.lower);
    CHECK(r < c.upper);
    CHECK(std::abs(X_or->col(0).mean()) < 0.04);
    CHECK(std::abs(X_or->col(1).mean()) < 0.04);
  }
}

TEST_CASE("bivariate copula conditional helpers match rvinecopulib goldens") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/sim/bivariate_copula_hfunc.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 90;
  for (const auto& c : fixture["cases"]) {
    const std::string id = c["id"].get<std::string>();
    CAPTURE(id);
    magmaan::sim::BivariateCopulaSpec copula;
    copula.family = copula_family_from_string(c["family"].get<std::string>());
    if (c["parameters"].is_number()) {
      copula.theta = c["parameters"].get<double>();
    }

    for (const auto& point : c["points"]) {
      const double u = point["u"].get<double>();
      const double v = point["v"].get<double>();
      const double p = point["p"].get<double>();
      auto h_or = magmaan::sim::bivariate_copula_conditional_cdf(
          copula, u, v);
      REQUIRE(h_or.has_value());
      CHECK(*h_or == doctest::Approx(
          point["conditional_cdf"].get<double>()).epsilon(1e-10));

      auto q_or = magmaan::sim::bivariate_copula_conditional_quantile(
          copula, u, p, options);
      REQUIRE(q_or.has_value());
      CHECK(*q_or == doctest::Approx(
          point["conditional_quantile"].get<double>()).epsilon(1e-10));
    }
  }
}

TEST_CASE("bivariate copula Kendall tau helpers use standard parameterization") {
  struct TauCase {
    magmaan::sim::BivariateCopulaFamily family;
    double theta;
    double tau;
  };
  const TauCase cases[] = {
      {magmaan::sim::BivariateCopulaFamily::Independence, 0.0, 0.0},
      {magmaan::sim::BivariateCopulaFamily::Clayton, 2.0, 0.5},
      {magmaan::sim::BivariateCopulaFamily::Gumbel, 2.0, 0.5},
      {magmaan::sim::BivariateCopulaFamily::Frank, 5.0, 0.4567009581601169},
      {magmaan::sim::BivariateCopulaFamily::Frank, -5.0, -0.4567009581601169},
      {magmaan::sim::BivariateCopulaFamily::Joe, 2.0, 0.3550659331517736},
  };

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 90;
  for (const auto& c : cases) {
    magmaan::sim::BivariateCopulaSpec copula;
    copula.family = c.family;
    copula.theta = c.theta;
    auto tau_or = magmaan::sim::bivariate_copula_tau(copula);
    REQUIRE(tau_or.has_value());
    CHECK(*tau_or == doctest::Approx(c.tau).epsilon(2e-10));

    auto back_or = magmaan::sim::bivariate_copula_from_tau(
        c.family, c.tau, options);
    REQUIRE(back_or.has_value());
    auto tau_back_or = magmaan::sim::bivariate_copula_tau(*back_or);
    REQUIRE(tau_back_or.has_value());
    CHECK(*tau_back_or == doctest::Approx(c.tau).epsilon(2e-10));
  }

  auto bad_or = magmaan::sim::bivariate_copula_from_tau(
      magmaan::sim::BivariateCopulaFamily::Clayton, -0.25, options);
  REQUIRE_FALSE(bad_or.has_value());
  CHECK(bad_or.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("bivariate copula calibration targets observed Pearson correlation") {
  const std::vector<magmaan::sim::MarginalSpec> normal_marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 17;
  options.max_bisection_iter = 45;
  options.calibration_tol = 8e-4;

  auto cal_or = magmaan::sim::calibrate_bivariate_copula_correlation(
      magmaan::sim::BivariateCopulaFamily::Frank, 0.35,
      normal_marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  CHECK(cal_or->target_corr == doctest::Approx(0.35));
  CHECK(cal_or->achieved_corr == doctest::Approx(0.35).epsilon(3e-3));
  CHECK(cal_or->iterations > 0);
  CHECK(cal_or->lower_bound_corr < 0.0);
  CHECK(cal_or->upper_bound_corr > 0.0);

  auto achieved_or = magmaan::sim::bivariate_copula_observed_corr(
      cal_or->copula, normal_marginals, options);
  REQUIRE(achieved_or.has_value());
  CHECK(*achieved_or == doctest::Approx(cal_or->achieved_corr).epsilon(1e-12));

  auto impossible_or = magmaan::sim::calibrate_bivariate_copula_correlation(
      magmaan::sim::BivariateCopulaFamily::Clayton, -0.25,
      normal_marginals, options);
  REQUIRE_FALSE(impossible_or.has_value());
  CHECK(impossible_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("bivariate copula matrix calibration works pairwise") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.35),
      magmaan::sim::MarginalSpec::standard_normal()};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.30, -0.20,
            0.30, 1.0, 0.25,
           -0.20, 0.25, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 17;
  options.max_bisection_iter = 45;
  options.calibration_tol = 1e-3;

  auto cal_or = magmaan::sim::calibrate_bivariate_copula_correlation_matrix(
      magmaan::sim::BivariateCopulaFamily::Frank, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;

  CHECK(cal_or->theta.rows() == 3);
  CHECK(cal_or->theta.cols() == 3);
  CHECK(cal_or->achieved_corr.rows() == 3);
  CHECK(cal_or->achieved_corr.cols() == 3);
  for (Eigen::Index i = 0; i < 3; ++i) {
    CHECK(cal_or->theta(i, i) == doctest::Approx(0.0));
    CHECK(cal_or->achieved_corr(i, i) == doctest::Approx(1.0));
    for (Eigen::Index j = i + 1; j < 3; ++j) {
      CHECK(cal_or->theta(i, j) == doctest::Approx(cal_or->theta(j, i)));
      CHECK(cal_or->achieved_corr(i, j) ==
            doctest::Approx(target(i, j)).epsilon(5e-3));
      CHECK(cal_or->achieved_corr(i, j) ==
            doctest::Approx(cal_or->achieved_corr(j, i)));
      CHECK(cal_or->lower_bound_corr(i, j) <= target(i, j));
      CHECK(cal_or->upper_bound_corr(i, j) >= target(i, j));
      CHECK(cal_or->iterations(i, j) > 0);
    }
  }

  auto impossible_or = magmaan::sim::calibrate_bivariate_copula_correlation_matrix(
      magmaan::sim::BivariateCopulaFamily::Clayton, target, marginals, options);
  REQUIRE_FALSE(impossible_or.has_value());
  CHECK(impossible_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("bivariate copula matrix calibration reports and repairs indefiniteness") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.75, 0.75,
            0.75, 1.0, -0.75,
            0.75, -0.75, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 17;
  options.max_bisection_iter = 45;
  options.calibration_tol = 1e-3;

  auto raw_or = magmaan::sim::calibrate_bivariate_copula_correlation_matrix(
      magmaan::sim::BivariateCopulaFamily::Frank, target, marginals, options);
  if (!raw_or.has_value()) MESSAGE(raw_or.error().detail);
  REQUIRE(raw_or.has_value());
  if (!raw_or.has_value()) return;
  CHECK(raw_or->max_abs_error < 5e-3);
  CHECK(raw_or->raw_min_eigenvalue < 0.0);
  CHECK(raw_or->repaired_min_eigenvalue == doctest::Approx(raw_or->raw_min_eigenvalue));
  CHECK_FALSE(raw_or->repair_applied);
  CHECK(raw_or->repaired_corr.isApprox(raw_or->achieved_corr, 0.0));

  auto error_options = options;
  error_options.matrix_repair =
      magmaan::sim::BivariateCopulaCorrelationRepairKind::Error;
  error_options.matrix_repair_min_eigenvalue = 1e-4;
  auto error_or = magmaan::sim::calibrate_bivariate_copula_correlation_matrix(
      magmaan::sim::BivariateCopulaFamily::Frank, target, marginals, error_options);
  REQUIRE_FALSE(error_or.has_value());
  CHECK(error_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);

  auto repair_options = error_options;
  repair_options.matrix_repair =
      magmaan::sim::BivariateCopulaCorrelationRepairKind::Shrinkage;
  auto repaired_or = magmaan::sim::calibrate_bivariate_copula_correlation_matrix(
      magmaan::sim::BivariateCopulaFamily::Frank, target, marginals, repair_options);
  if (!repaired_or.has_value()) MESSAGE(repaired_or.error().detail);
  REQUIRE(repaired_or.has_value());
  if (!repaired_or.has_value()) return;
  CHECK(repaired_or->repair_applied);
  CHECK(repaired_or->repair_shrinkage > 0.0);
  CHECK(repaired_or->repaired_min_eigenvalue >=
        repair_options.matrix_repair_min_eigenvalue * 0.999);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(repaired_or->repaired_corr);
  REQUIRE(es.info() == Eigen::Success);
  CHECK(es.eigenvalues().minCoeff() ==
        doctest::Approx(repaired_or->repaired_min_eigenvalue).epsilon(1e-10));
}

TEST_CASE("C-vine 3 copula simulation composes root and conditional pairs") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.35)};

  magmaan::sim::CVine3CopulaSpec copula;
  copula.copula_01.family = magmaan::sim::BivariateCopulaFamily::Frank;
  copula.copula_01.theta = 5.0;
  copula.copula_02.family = magmaan::sim::BivariateCopulaFamily::Frank;
  copula.copula_02.theta = -5.0;
  copula.copula_12_given_0.family =
      magmaan::sim::BivariateCopulaFamily::Independence;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 31;
  options.max_bisection_iter = 80;
  std::mt19937_64 rng(20260607);
  auto X_or = magmaan::sim::simulate_cvine3_copula_matrix(
      18000, copula, marginals, rng, options);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());
  CHECK(sample_corr(*X_or, 0, 1) > 0.35);
  CHECK(sample_corr(*X_or, 0, 2) < -0.30);
  CHECK(std::abs(X_or->col(0).mean()) < 0.04);
  CHECK(std::abs(X_or->col(1).mean()) < 0.04);
  CHECK(std::abs(X_or->col(2).mean()) < 0.06);
}

TEST_CASE("C-vine 3 inverse Rosenblatt matches rvinecopulib goldens") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/sim/cvine3_inverse_rosenblatt.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 90;
  for (const auto& c : fixture["cases"]) {
    const std::string id = c["id"].get<std::string>();
    CAPTURE(id);
    magmaan::sim::CVine3CopulaSpec copula;
    assign_copula_from_json(copula.copula_01, c["copula_01"]);
    assign_copula_from_json(copula.copula_02, c["copula_02"]);
    assign_copula_from_json(copula.copula_12_given_0, c["copula_12_given_0"]);

    Eigen::MatrixXd independent(c["points"].size(), 3);
    Eigen::MatrixXd expected(c["points"].size(), 3);
    for (Eigen::Index row = 0; row < independent.rows(); ++row) {
      const auto& point = c["points"][static_cast<std::size_t>(row)];
      for (Eigen::Index col = 0; col < 3; ++col) {
        independent(row, col) =
            point["independent_u"][static_cast<std::size_t>(col)].get<double>();
        expected(row, col) =
            point["copula_u"][static_cast<std::size_t>(col)].get<double>();
      }
    }

    auto got_or = magmaan::sim::cvine3_copula_inverse_rosenblatt(
        independent, copula, options);
    if (!got_or.has_value()) MESSAGE(got_or.error().detail);
    REQUIRE(got_or.has_value());
    CHECK(got_or->isApprox(expected, 2e-10));
  }
}

TEST_CASE("generic C-vine inverse Rosenblatt matches C-vine 3 specialization") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/sim/cvine3_inverse_rosenblatt.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 90;
  for (const auto& c : fixture["cases"]) {
    const std::string id = c["id"].get<std::string>();
    CAPTURE(id);
    magmaan::sim::CVine3CopulaSpec cvine3;
    assign_copula_from_json(cvine3.copula_01, c["copula_01"]);
    assign_copula_from_json(cvine3.copula_02, c["copula_02"]);
    assign_copula_from_json(
        cvine3.copula_12_given_0, c["copula_12_given_0"]);

    magmaan::sim::CVineCopulaSpec generic;
    generic.pair_copulas = {
        {},
        {cvine3.copula_01},
        {cvine3.copula_02, cvine3.copula_12_given_0}};

    Eigen::MatrixXd independent(c["points"].size(), 3);
    for (Eigen::Index row = 0; row < independent.rows(); ++row) {
      const auto& point = c["points"][static_cast<std::size_t>(row)];
      for (Eigen::Index col = 0; col < 3; ++col) {
        independent(row, col) =
            point["independent_u"][static_cast<std::size_t>(col)].get<double>();
      }
    }

    auto specialized_or = magmaan::sim::cvine3_copula_inverse_rosenblatt(
        independent, cvine3, options);
    auto generic_or = magmaan::sim::cvine_copula_inverse_rosenblatt(
        independent, generic, options);
    REQUIRE(specialized_or.has_value());
    if (!generic_or.has_value()) MESSAGE(generic_or.error().detail);
    REQUIRE(generic_or.has_value());
    CHECK(generic_or->isApprox(*specialized_or, 2e-12));
  }
}

TEST_CASE("generic C-vine inverse Rosenblatt matches rvinecopulib 4D golden") {
  const std::string path = magmaan::test::fixtures_dir() +
                           "/sim/cvine4_inverse_rosenblatt.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 90;
  for (const auto& c : fixture["cases"]) {
    const std::string id = c["id"].get<std::string>();
    CAPTURE(id);
    magmaan::sim::CVineCopulaSpec copula;
    copula.pair_copulas = {{}, {{}}, {{}, {}}, {{}, {}, {}}};
    assign_copula_from_json(copula.pair_copulas[1][0], c["copula_10"]);
    assign_copula_from_json(copula.pair_copulas[2][0], c["copula_20"]);
    assign_copula_from_json(
        copula.pair_copulas[2][1], c["copula_21_given_0"]);
    assign_copula_from_json(copula.pair_copulas[3][0], c["copula_30"]);
    assign_copula_from_json(
        copula.pair_copulas[3][1], c["copula_31_given_0"]);
    assign_copula_from_json(
        copula.pair_copulas[3][2], c["copula_32_given_01"]);

    Eigen::MatrixXd independent(c["points"].size(), 4);
    Eigen::MatrixXd expected(c["points"].size(), 4);
    for (Eigen::Index row = 0; row < independent.rows(); ++row) {
      const auto& point = c["points"][static_cast<std::size_t>(row)];
      for (Eigen::Index col = 0; col < 4; ++col) {
        independent(row, col) =
            point["independent_u"][static_cast<std::size_t>(col)].get<double>();
        expected(row, col) =
            point["copula_u"][static_cast<std::size_t>(col)].get<double>();
      }
    }

    auto got_or = magmaan::sim::cvine_copula_inverse_rosenblatt(
        independent, copula, options);
    if (!got_or.has_value()) MESSAGE(got_or.error().detail);
    REQUIRE(got_or.has_value());
    CHECK(got_or->isApprox(expected, 2e-10));
  }
}

TEST_CASE("generic C-vine sampler supports four variables") {
  magmaan::sim::BivariateCopulaSpec frank_pos;
  frank_pos.family = magmaan::sim::BivariateCopulaFamily::Frank;
  frank_pos.theta = 3.0;
  magmaan::sim::BivariateCopulaSpec frank_neg;
  frank_neg.family = magmaan::sim::BivariateCopulaFamily::Frank;
  frank_neg.theta = -2.0;
  magmaan::sim::BivariateCopulaSpec clayton;
  clayton.family = magmaan::sim::BivariateCopulaFamily::Clayton;
  clayton.theta = 1.2;

  magmaan::sim::CVineCopulaSpec copula;
  copula.pair_copulas = {
      {},
      {frank_pos},
      {frank_neg, frank_pos},
      {clayton, magmaan::sim::BivariateCopulaSpec{}, frank_neg}};

  Eigen::MatrixXd independent(2, 4);
  independent << 0.11, 0.22, 0.33, 0.44,
                 0.81, 0.72, 0.63, 0.54;
  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 80;
  auto U_or = magmaan::sim::cvine_copula_inverse_rosenblatt(
      independent, copula, options);
  if (!U_or.has_value()) MESSAGE(U_or.error().detail);
  REQUIRE(U_or.has_value());
  CHECK(U_or->rows() == 2);
  CHECK(U_or->cols() == 4);
  CHECK(U_or->col(0).isApprox(independent.col(0), 0.0));
  CHECK(U_or->array().minCoeff() > 0.0);
  CHECK(U_or->array().maxCoeff() < 1.0);

  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.20),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.30)};
  std::mt19937_64 rng(20260613);
  auto X_or = magmaan::sim::simulate_cvine_copula_matrix(
      200, copula, marginals, rng, options);
  REQUIRE(X_or.has_value());
  CHECK(X_or->rows() == 200);
  CHECK(X_or->cols() == 4);

  copula.pair_copulas[3].pop_back();
  auto bad_or = magmaan::sim::simulate_cvine_copula_matrix(
      20, copula, marginals, rng, options);
  REQUIRE_FALSE(bad_or.has_value());
  CHECK(bad_or.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("C-vine 3 conditional pair copula can drive residual dependence") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};

  magmaan::sim::CVine3CopulaSpec copula;
  copula.copula_12_given_0.family = magmaan::sim::BivariateCopulaFamily::Frank;
  copula.copula_12_given_0.theta = 5.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.max_bisection_iter = 80;
  std::mt19937_64 rng(20260608);
  auto X_or = magmaan::sim::simulate_cvine3_copula_matrix(
      18000, copula, marginals, rng, options);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());
  CHECK(std::abs(sample_corr(*X_or, 0, 1)) < 0.04);
  CHECK(std::abs(sample_corr(*X_or, 0, 2)) < 0.04);
  CHECK(sample_corr(*X_or, 1, 2) > 0.35);

  auto raw_or = magmaan::sim::simulate_cvine3_copula_raw(
      20, copula, marginals, rng, options);
  REQUIRE(raw_or.has_value());
  REQUIRE(raw_or->X.size() == 1u);
  CHECK(raw_or->X[0].rows() == 20);
  CHECK(raw_or->X[0].cols() == 3);
}

TEST_CASE("C-vine 3 calibration fits an observed correlation matrix") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.30),
      magmaan::sim::MarginalSpec::standard_normal()};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.25, -0.20,
            0.25, 1.0, 0.10,
           -0.20, 0.10, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 13;
  options.max_bisection_iter = 45;
  options.calibration_tol = 1.5e-3;

  auto cal_or = magmaan::sim::calibrate_cvine3_copula_correlation(
      magmaan::sim::BivariateCopulaFamily::Frank, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  CHECK(cal_or->root_01.achieved_corr == doctest::Approx(target(0, 1)).epsilon(5e-3));
  CHECK(cal_or->root_02.achieved_corr == doctest::Approx(target(0, 2)).epsilon(5e-3));
  CHECK(cal_or->conditional_iterations > 0);
  CHECK(cal_or->max_abs_error < 7e-3);
  CHECK(cal_or->achieved_corr(0, 1) == doctest::Approx(target(0, 1)).epsilon(7e-3));
  CHECK(cal_or->achieved_corr(0, 2) == doctest::Approx(target(0, 2)).epsilon(7e-3));
  CHECK(cal_or->achieved_corr(1, 2) == doctest::Approx(target(1, 2)).epsilon(7e-3));

  auto achieved_or = magmaan::sim::cvine3_copula_observed_corr(
      cal_or->copula, marginals, options);
  REQUIRE(achieved_or.has_value());
  CHECK(achieved_or->isApprox(cal_or->achieved_corr, 1e-12));
}

TEST_CASE("C-vine 3 calibration can select the root variable") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(0.25),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.40)};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.18, -0.16,
            0.18, 1.0, 0.22,
           -0.16, 0.22, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 13;
  options.max_bisection_iter = 45;
  options.calibration_tol = 1.5e-3;

  auto cal_or = magmaan::sim::calibrate_cvine3_copula_correlation_select_root(
      magmaan::sim::BivariateCopulaFamily::Frank, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;

  bool seen[3] = {false, false, false};
  for (Eigen::Index i = 0; i < 3; ++i) {
    const int idx = cal_or->variable_order(i);
    REQUIRE(idx >= 0);
    REQUIRE(idx < 3);
    CHECK_FALSE(seen[idx]);
    seen[idx] = true;
  }
  CHECK(cal_or->root_index == cal_or->variable_order(0));
  CHECK(cal_or->target_corr.isApprox(target, 0.0));
  CHECK(cal_or->max_abs_error < 8e-3);
  CHECK(cal_or->achieved_corr(0, 1) == doctest::Approx(target(0, 1)).epsilon(8e-3));
  CHECK(cal_or->achieved_corr(0, 2) == doctest::Approx(target(0, 2)).epsilon(8e-3));
  CHECK(cal_or->achieved_corr(1, 2) == doctest::Approx(target(1, 2)).epsilon(8e-3));
}

TEST_CASE("C-vine 3 calibration supports per-edge family choices") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.25),
      magmaan::sim::MarginalSpec::standard_normal()};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.22, 0.18,
            0.22, 1.0, 0.14,
            0.18, 0.14, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 13;
  options.max_bisection_iter = 45;
  options.calibration_tol = 1.5e-3;

  const magmaan::sim::CVine3FamilySpec families{
      magmaan::sim::BivariateCopulaFamily::Clayton,
      magmaan::sim::BivariateCopulaFamily::Gumbel,
      magmaan::sim::BivariateCopulaFamily::Frank};
  auto cal_or = magmaan::sim::calibrate_cvine3_copula_correlation(
      families, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  CHECK(cal_or->copula.copula_01.family ==
        magmaan::sim::BivariateCopulaFamily::Clayton);
  CHECK(cal_or->copula.copula_02.family ==
        magmaan::sim::BivariateCopulaFamily::Gumbel);
  CHECK(cal_or->copula.copula_12_given_0.family ==
        magmaan::sim::BivariateCopulaFamily::Frank);
  CHECK(cal_or->max_abs_error < 8e-3);
}

TEST_CASE("C-vine 3 calibration can select edge families") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standard_normal()};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.20, -0.18,
            0.20, 1.0, 0.12,
           -0.18, 0.12, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 11;
  options.max_bisection_iter = 40;
  options.calibration_tol = 2e-3;

  const std::vector<magmaan::sim::BivariateCopulaFamily> family_set{
      magmaan::sim::BivariateCopulaFamily::Clayton,
      magmaan::sim::BivariateCopulaFamily::Frank};
  auto cal_or = magmaan::sim::calibrate_cvine3_copula_correlation_select_families(
      family_set, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  CHECK(cal_or->max_abs_error < 1.2e-2);
  CHECK(cal_or->copula.copula_02.family ==
        magmaan::sim::BivariateCopulaFamily::Frank);

  const std::vector<magmaan::sim::BivariateCopulaFamily> one_sided{
      magmaan::sim::BivariateCopulaFamily::Clayton};
  auto impossible_or =
      magmaan::sim::calibrate_cvine3_copula_correlation_select_families(
          one_sided, target, marginals, options);
  REQUIRE_FALSE(impossible_or.has_value());
  CHECK(impossible_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("C-vine 3 calibration can select root and edge families") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(0.25),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.35)};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.19, -0.17,
            0.19, 1.0, 0.15,
           -0.17, 0.15, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 11;
  options.max_bisection_iter = 40;
  options.calibration_tol = 2e-3;

  const std::vector<magmaan::sim::BivariateCopulaFamily> family_set{
      magmaan::sim::BivariateCopulaFamily::Clayton,
      magmaan::sim::BivariateCopulaFamily::Frank};
  auto cal_or =
      magmaan::sim::calibrate_cvine3_copula_correlation_select_structure(
          family_set, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;

  bool seen[3] = {false, false, false};
  for (Eigen::Index i = 0; i < 3; ++i) {
    const int idx = cal_or->variable_order(i);
    REQUIRE(idx >= 0);
    REQUIRE(idx < 3);
    CHECK_FALSE(seen[idx]);
    seen[idx] = true;
  }
  CHECK(cal_or->root_index == cal_or->variable_order(0));
  CHECK(cal_or->target_corr.isApprox(target, 0.0));
  CHECK(cal_or->max_abs_error < 1.2e-2);
  CHECK(cal_or->achieved_corr(0, 1) ==
        doctest::Approx(target(0, 1)).epsilon(1.2e-2));
  CHECK(cal_or->achieved_corr(0, 2) ==
        doctest::Approx(target(0, 2)).epsilon(1.2e-2));
  CHECK(cal_or->achieved_corr(1, 2) ==
        doctest::Approx(target(1, 2)).epsilon(1.2e-2));

  const std::vector<magmaan::sim::BivariateCopulaFamily> one_sided{
      magmaan::sim::BivariateCopulaFamily::Clayton};
  auto impossible_or =
      magmaan::sim::calibrate_cvine3_copula_correlation_select_structure(
          one_sided, target, marginals, options);
  REQUIRE_FALSE(impossible_or.has_value());
  CHECK(impossible_or.error().kind == magmaan::SimError::Kind::CalibrationFailed);
}

TEST_CASE("C-vine 3 calibration simulation restores original variable order") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(0.25),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.35)};

  magmaan::sim::CVine3CorrelationCalibration calibration;
  calibration.variable_order = Eigen::Vector3i(2, 0, 1);
  calibration.root_index = 2;
  calibration.copula.copula_01.family =
      magmaan::sim::BivariateCopulaFamily::Frank;
  calibration.copula.copula_01.theta = 2.0;
  calibration.copula.copula_02.family =
      magmaan::sim::BivariateCopulaFamily::Clayton;
  calibration.copula.copula_02.theta = 1.2;
  calibration.copula.copula_12_given_0.family =
      magmaan::sim::BivariateCopulaFamily::Frank;
  calibration.copula.copula_12_given_0.theta = -1.5;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 11;
  options.max_bisection_iter = 45;

  std::mt19937_64 rng_cal(20260610);
  auto got_or = magmaan::sim::simulate_cvine3_copula_matrix(
      64, calibration, marginals, rng_cal, options);
  if (!got_or.has_value()) MESSAGE(got_or.error().detail);
  REQUIRE(got_or.has_value());

  const std::vector<magmaan::sim::MarginalSpec> ordered_marginals{
      marginals[2], marginals[0], marginals[1]};
  std::mt19937_64 rng_direct(20260610);
  auto ordered_or = magmaan::sim::simulate_cvine3_copula_matrix(
      64, calibration.copula, ordered_marginals, rng_direct, options);
  REQUIRE(ordered_or.has_value());
  Eigen::MatrixXd expected(64, 3);
  expected.col(2) = ordered_or->col(0);
  expected.col(0) = ordered_or->col(1);
  expected.col(1) = ordered_or->col(2);
  CHECK(got_or->isApprox(expected, 0.0));

  auto raw_or = magmaan::sim::simulate_cvine3_copula_raw(
      8, calibration, marginals, rng_cal, options);
  REQUIRE(raw_or.has_value());
  REQUIRE(raw_or->X.size() == 1u);
  CHECK(raw_or->X[0].rows() == 8);
  CHECK(raw_or->X[0].cols() == 3);

  calibration.variable_order = Eigen::Vector3i(0, 0, 2);
  auto bad_or = magmaan::sim::simulate_cvine3_copula_matrix(
      8, calibration, marginals, rng_cal, options);
  REQUIRE_FALSE(bad_or.has_value());
  CHECK(bad_or.error().kind == magmaan::SimError::Kind::InvalidInput);
}

TEST_CASE("C-vine 3 calibrated simulation tracks target correlations") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standardized_lognormal(0.20),
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.35)};
  Eigen::MatrixXd target(3, 3);
  target << 1.0, 0.18, -0.16,
            0.18, 1.0, 0.14,
           -0.16, 0.14, 1.0;

  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = 13;
  options.max_bisection_iter = 45;
  options.calibration_tol = 1.5e-3;

  const std::vector<magmaan::sim::BivariateCopulaFamily> family_set{
      magmaan::sim::BivariateCopulaFamily::Clayton,
      magmaan::sim::BivariateCopulaFamily::Frank};
  auto cal_or =
      magmaan::sim::calibrate_cvine3_copula_correlation_select_structure(
          family_set, target, marginals, options);
  if (!cal_or.has_value()) MESSAGE(cal_or.error().detail);
  REQUIRE(cal_or.has_value());
  if (!cal_or.has_value()) return;
  REQUIRE(cal_or->max_abs_error < 1.0e-2);

  std::mt19937_64 rng(20260612);
  auto X_or = magmaan::sim::simulate_cvine3_copula_matrix(
      30000, *cal_or, marginals, rng, options);
  if (!X_or.has_value()) MESSAGE(X_or.error().detail);
  REQUIRE(X_or.has_value());

  Eigen::MatrixXd empirical = Eigen::MatrixXd::Identity(3, 3);
  empirical(0, 1) = empirical(1, 0) = sample_corr(*X_or, 0, 1);
  empirical(0, 2) = empirical(2, 0) = sample_corr(*X_or, 0, 2);
  empirical(1, 2) = empirical(2, 1) = sample_corr(*X_or, 1, 2);
  CHECK(empirical(0, 1) == doctest::Approx(target(0, 1)).epsilon(2.5e-2));
  CHECK(empirical(0, 2) == doctest::Approx(target(0, 2)).epsilon(2.5e-2));
  CHECK(empirical(1, 2) == doctest::Approx(target(1, 2)).epsilon(2.5e-2));
  CHECK((empirical - cal_or->achieved_corr).cwiseAbs().maxCoeff() < 3.0e-2);
}

TEST_CASE("bivariate copula raw generator validates parameters and marginals") {
  const std::vector<magmaan::sim::MarginalSpec> marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::standardized_lognormal(0.5)};

  magmaan::sim::BivariateCopulaSpec copula;
  copula.family = magmaan::sim::BivariateCopulaFamily::Clayton;
  copula.theta = 1.5;
  std::mt19937_64 rng(20260606);
  auto raw_or = magmaan::sim::simulate_bivariate_copula_raw(
      20, copula, marginals, rng);
  REQUIRE(raw_or.has_value());
  CHECK(raw_or->X.size() == 1u);
  CHECK(raw_or->mask.empty());
  CHECK(raw_or->X[0].rows() == 20);
  CHECK(raw_or->X[0].cols() == 2);

  copula.theta = -1.0;
  auto bad_theta_or = magmaan::sim::simulate_bivariate_copula_matrix(
      20, copula, marginals, rng);
  REQUIRE_FALSE(bad_theta_or.has_value());
  CHECK(bad_theta_or.error().kind == magmaan::SimError::Kind::InvalidInput);

  copula.theta = 1.5;
  const std::vector<magmaan::sim::MarginalSpec> bad_marginals{
      magmaan::sim::MarginalSpec::standard_normal(),
      magmaan::sim::MarginalSpec::fleishman(1.0, 0.0, 0.0)};
  auto bad_marginal_or = magmaan::sim::simulate_bivariate_copula_matrix(
      20, copula, bad_marginals, rng);
  REQUIRE_FALSE(bad_marginal_or.has_value());
  CHECK(bad_marginal_or.error().kind == magmaan::SimError::Kind::InvalidMarginal);
}
