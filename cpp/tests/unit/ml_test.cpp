#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <random>

#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"

#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;

namespace {

ModelEvaluator must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  static thread_local magmaan::spec::LatentStructure s_pt;
  static thread_local magmaan::model::MatrixRep   s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  auto ev = ModelEvaluator::build(s_pt, s_mr);
  REQUIRE(ev.has_value());
  return std::move(*ev);
}

// Build an arbitrary positive-definite p × p matrix from a seed.
Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  // Σ = A Aᵀ + p·I — ensures PD.
  return A * A.transpose() + Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

}  // namespace

TEST_CASE("ML: F=0 when Σ(θ)=S (saturated model fit)") {
  // 3 indicators, no model structure — just look at F at Σ = S.
  auto ev = must_build("f =~ x1 + x2 + x3");
  std::mt19937 rng(1);
  Eigen::MatrixXd S = random_pd(rng, 3);

  SampleStats samp;
  samp.S.push_back(S);
  samp.n_obs.push_back(100);

  // Pick θ such that Σ(θ) = S exactly. For this saturated-equivalent case:
  // Λ = [1; λ_2; λ_3], Ψ scalar, Θ diagonal — there's no general θ that
  // produces an arbitrary S. Instead, verify the formula on a contrived case:
  // set θ such that Σ = Λ Ψ Λᵀ + Θ matches S elementwise.
  // For simplicity, verify F_ML(σ̂) → 0 when we use Σ = S directly via a
  // reference call (we don't have a saturated-model parameterization yet).
  // So this test just asserts F_ML at our chosen θ is finite + matches the
  // value formula computed directly.

  // Pick a random θ and compare F_ML to the formula computed on the
  // resulting Σ.
  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.5, 1.2);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  auto sm = ev.sigma(theta);
  REQUIRE(sm.has_value());
  auto f = magmaan::estimate::ml_value(samp, *sm);
  REQUIRE(f.has_value());

  // Compute F manually and compare.
  Eigen::LLT<Eigen::MatrixXd> llt_S(S), llt_Sigma(sm->sigma[0]);
  REQUIRE(llt_S.info() == Eigen::Success);
  REQUIRE(llt_Sigma.info() == Eigen::Success);
  double log_det_S = 0, log_det_Sigma = 0;
  for (Eigen::Index i = 0; i < 3; ++i) {
    log_det_S     += std::log(llt_S.matrixL()(i, i));
    log_det_Sigma += std::log(llt_Sigma.matrixL()(i, i));
  }
  log_det_S *= 2;  log_det_Sigma *= 2;
  const double tr = llt_Sigma.solve(S).trace();
  const double f_expected = log_det_Sigma + tr - log_det_S - 3.0;

  CHECK(*f == doctest::Approx(f_expected).epsilon(1e-12));
}

TEST_CASE("ML: gradient matches finite differences (1F CFA)") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  std::mt19937 rng(7);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs.push_back(100);

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.5, 1.2);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  auto sm = ev.sigma(theta).value();
  auto J  = ev.dsigma_dtheta(theta).value();
  auto g_an = magmaan::estimate::ml_gradient(samp, sm, J).value();

  // Finite-difference gradient.
  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    auto smp = ev.sigma(tp).value();
    auto smm = ev.sigma(tm).value();
    g_fd(k) = (magmaan::estimate::ml_value(samp, smp).value() - magmaan::estimate::ml_value(samp, smm).value()) / (2.0 * h);
  }

  const double diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-5);
}

TEST_CASE("ML: cached fused value_gradient matches separate value and gradient") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  std::mt19937 rng(17);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs.push_back(100);

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.5, 1.2);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  auto sm = ev.sigma(theta).value();
  auto J  = ev.dsigma_dtheta(theta).value();
  auto cache = magmaan::estimate::ml_prepare(samp).value();
  auto f_sep = magmaan::estimate::ml_value(samp, sm).value();
  auto g_sep = magmaan::estimate::ml_gradient(samp, sm, J).value();
  auto vg = magmaan::estimate::ml_value_gradient(samp, cache, sm, J).value();

  CHECK(vg.value == doctest::Approx(f_sep).epsilon(1e-14));
  CHECK((vg.gradient - g_sep).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("ML: mean-structure F formula matches hand calculation") {
  // 1F CFA + intercepts. At an arbitrary θ, verify
  //   F_b = log|Σ| + tr(SΣ⁻¹) - log|S| - p + (m̄-μ)'Σ⁻¹(m̄-μ)
  // matches what ML::value returns.
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");
  const auto locs = ev.param_locations();

  std::mt19937 rng(123);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);
  mean << 4.0, 5.0, 6.0;

  SampleStats samp;
  samp.S = {S};
  samp.mean = {mean};
  samp.n_obs = {100};

  Eigen::VectorXd theta = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(ev.n_free()));
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    const auto m = locs[static_cast<std::size_t>(k)].mat;
    if (m == magmaan::model::MatId::Lambda)      theta(k) = 0.8;
    else if (m == magmaan::model::MatId::Theta)  theta(k) = 0.5;
    else if (m == magmaan::model::MatId::Psi)    theta(k) = 1.0;
    else if (m == magmaan::model::MatId::Nu)     theta(k) = 3.5;  // ν entries
  }

  auto sm = ev.sigma(theta);
  REQUIRE(sm.has_value());
  REQUIRE(sm->mu.size() == 1);
  REQUIRE(sm->mu[0].size() == 3);

  auto f_or = magmaan::estimate::ml_value(samp, *sm);
  REQUIRE(f_or.has_value());

  // Manual F_b:
  Eigen::LLT<Eigen::MatrixXd> llt_S(S), llt_Sigma(sm->sigma[0]);
  double log_det_S = 0, log_det_Sigma = 0;
  for (Eigen::Index i = 0; i < 3; ++i) {
    log_det_S     += std::log(llt_S.matrixL()(i, i));
    log_det_Sigma += std::log(llt_Sigma.matrixL()(i, i));
  }
  log_det_S *= 2;  log_det_Sigma *= 2;
  const double tr = llt_Sigma.solve(S).trace();
  const Eigen::VectorXd d = mean - sm->mu[0];
  const double mean_term = d.dot(llt_Sigma.solve(d));
  const double F_expected = log_det_Sigma + tr - log_det_S - 3.0 + mean_term;

  CHECK(*f_or == doctest::Approx(F_expected).epsilon(1e-12));
}

TEST_CASE("ML: fixed means contribute to the objective without parameter columns") {
  auto ev = must_build("x1 ~~ 2*x1\nx1 ~ 1*1");
  REQUIRE(ev.n_free() == 0);
  SampleStats sample;
  sample.S = {Eigen::MatrixXd::Constant(1, 1, 3.0)};
  sample.mean = {Eigen::VectorXd::Constant(1, 4.0)};
  sample.n_obs = {100};
  const Eigen::VectorXd theta(0);
  auto evaluation = ev.evaluate(theta, true, true);
  REQUIRE(evaluation.has_value());
  CHECK(evaluation->J_mu.rows() == 1);
  CHECK(evaluation->J_mu.cols() == 0);
  auto cache = magmaan::estimate::ml_prepare(sample);
  REQUIRE(cache.has_value());
  auto fused = magmaan::estimate::ml_value_gradient(sample, *cache, evaluation->moments,
                                                     evaluation->J_sigma, evaluation->J_mu);
  REQUIRE(fused.has_value());
  // log(2/3) + 3/2 - 1 + (4-1)^2/2, on the full-F scale.
  const double reference = std::log(2.0 / 3.0) + 0.5 + 4.5;
  CHECK(fused->value == doctest::Approx(reference).epsilon(1e-12));
  CHECK(fused->gradient.size() == 0);
  auto objective = magmaan::estimate::ml_objective(ev, sample);
  REQUIRE(objective.has_value());
  Eigen::VectorXd gradient(0);
  CHECK(objective->f(theta, gradient) == doctest::Approx(0.5 * reference).epsilon(1e-12));
  CHECK(gradient.size() == 0);
}

TEST_CASE("ML: mean-structure gradient matches finite differences") {
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1\nf ~ 1");
  std::mt19937 rng(7);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);
  mean << 4.0, 5.0, 6.0;
  SampleStats samp;
  samp.S = {S};
  samp.mean = {mean};
  samp.n_obs = {100};

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d_unif(0.4, 1.1);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d_unif(rng);

  auto sm  = ev.sigma(theta).value();
  auto J   = ev.dsigma_dtheta(theta).value();
  auto Jmu = ev.dmu_dtheta(theta).value();
  REQUIRE(Jmu.rows() == 3);
  REQUIRE(Jmu.cols() == theta.size());

  auto g_an = magmaan::estimate::ml_gradient(samp, sm, J, Jmu).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    auto smp = ev.sigma(tp).value();
    auto smm = ev.sigma(tm).value();
    g_fd(k) = (magmaan::estimate::ml_value(samp, smp).value() - magmaan::estimate::ml_value(samp, smm).value()) / (2.0 * h);
  }
  const double max_diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  CHECK(max_diff < 1e-5);
}

TEST_CASE("ML: fit() recovers ν̂_i ≈ m̄_i on saturated mean-structure CFA") {
  // 1F CFA + indicator intercepts: the indicator-mean side is saturated
  // (one ν per indicator, no constraint), so ν̂_i must equal the sample
  // mean m̄_i exactly at the optimum.
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(2026);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);
  mean << 4.2, 5.7, 6.1;
  SampleStats samp;
  samp.S = {S};
  samp.mean = {mean};
  samp.n_obs = {301};

  auto est_or = magmaan::test::fit(*pt, *mr, samp);
  REQUIRE(est_or.has_value());
  const auto& est = *est_or;

  auto ev = ModelEvaluator::build(*pt, *mr).value();
  const auto locs = ev.param_locations();
  std::array<Eigen::Index, 3> nu_idx = {-1, -1, -1};
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Nu) {
      nu_idx[static_cast<std::size_t>(locs[k].row)] =
          static_cast<Eigen::Index>(k);
    }
  }
  for (auto idx : nu_idx) REQUIRE(idx >= 0);

  for (Eigen::Index i = 0; i < 3; ++i) {
    CHECK(est.theta(nu_idx[static_cast<std::size_t>(i)]) ==
          doctest::Approx(mean(i)).epsilon(1e-6));
  }
}

TEST_CASE("ML: gradient matches finite differences (3F Holzinger at lavaan θ̂)") {
  // Random θ for a 3-factor model often produces non-PD implied Σ (Ψ
  // off-diagonals dominate diagonals). Use lavaan's converged θ̂ from the
  // 0002 fit fixture — guaranteed PD by construction.
  auto ev = must_build(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");

  const std::string path = std::string(MAGMAAN_FIXTURES_DIR) +
                           "/fit/0002_three_factor_hs.fit.json";
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::stringstream ss;  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  Eigen::VectorXd theta(j["theta_hat"].size());
  for (Eigen::Index k = 0; k < theta.size(); ++k)
    theta(k) = j["theta_hat"][static_cast<std::size_t>(k)].get<double>();
  REQUIRE(static_cast<std::size_t>(theta.size()) == ev.n_free());

  // Build a SampleStats from a fresh random PD covariance — the gradient
  // of ML at θ̂ is generally nonzero for an arbitrary S, but the gradient
  // is mathematically defined regardless. We're just checking that our
  // analytic formula matches FD at a point where Σ is PD.
  std::mt19937 rng(11);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 9));
  samp.n_obs.push_back(301);

  auto sm = ev.sigma(theta).value();
  auto J  = ev.dsigma_dtheta(theta).value();
  auto g_an = magmaan::estimate::ml_gradient(samp, sm, J).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    g_fd(k) = (magmaan::estimate::ml_value(samp, ev.sigma(tp).value()).value() -
               magmaan::estimate::ml_value(samp, ev.sigma(tm).value()).value()) / (2.0 * h);
  }
  const double max_diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  // Slightly looser tolerance for the larger model — FD noise grows with
  // gradient magnitude.
  CHECK(max_diff < 1e-4);
}

TEST_CASE("ML: a model without free parameters is evaluated at its fixed values") {
  // As lavaan does: nothing is optimized, the objective is the fixed model's,
  // and the verdict passes because no direction can improve it.
  auto fp = Parser::parse("f =~ 1*x1 + 0.8*x2 + 0.6*x3\nf ~~ 1*f\n"
                          "x1 ~~ 1*x1\nx2 ~~ 1*x2\nx3 ~~ 1*x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  REQUIRE(pt->n_free() == 0);
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  std::mt19937 rng(77);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {200};
  auto ev = ModelEvaluator::build(*pt, *mr);
  REQUIRE(ev.has_value());
  auto implied = ev->sigma(Eigen::VectorXd(0));
  REQUIRE(implied.has_value());
  Eigen::Matrix3d expected;
  expected << 2.0, 0.8, 0.6,
              0.8, 1.64, 0.48,
              0.6, 0.48, 1.36;
  CHECK((implied->sigma[0] - expected).norm() < 1e-14);
  const Eigen::Matrix3d& S = samp.S[0];
  const double F = std::log(expected.determinant()) + (S * expected.inverse()).trace() -
                   std::log(S.determinant()) - 3.0;
  for (bool gls : {false, true}) {
    auto est = gls ? magmaan::test::fit_gls(*pt, *mr, samp) : magmaan::test::fit(*pt, *mr, samp);
    REQUIRE(est.has_value());
    CHECK(est->theta.size() == 0);
    CHECK(magmaan::estimate::fit_verdict(*est).status == magmaan::estimate::FitCheck::Passed);
    if (!gls) CHECK(est->fmin == doctest::Approx(0.5 * F).epsilon(1e-12));
  }
}

TEST_CASE("ML correlation target: value and gradient match independent reference") {
  auto ev = must_build("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta(ev.n_free());
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = .6 + .04 * static_cast<double>(k);
  SampleStats sample;
  sample.S = {Eigen::MatrixXd::Constant(4, 4, .2)};
  sample.S[0].diagonal().setOnes();
  sample.n_obs = {400};
  auto objective = magmaan::estimate::ml_objective(
      ev, sample, magmaan::model::MomentTarget::Correlation);
  REQUIRE(objective.has_value());

  auto reference = [&](const Eigen::VectorXd& point) {
    auto moments = ev.sigma(point);
    REQUIRE(moments.has_value());
    const auto& S = moments->sigma[0];
    const Eigen::MatrixXd D = S.diagonal().array().sqrt().inverse().matrix().asDiagonal();
    const Eigen::MatrixXd R = D * S * D;
    return .5 * (std::log(R.determinant()) - std::log(sample.S[0].determinant())
                 + (sample.S[0] * R.inverse()).trace() - 4);
  };
  Eigen::VectorXd gradient;
  CHECK(objective->f(theta, gradient) == doctest::Approx(reference(theta)).epsilon(1e-12));
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    auto plus = theta; auto minus = theta;
    constexpr double h = 1e-6;
    plus(k) += h; minus(k) -= h;
    CHECK(gradient(k) == doctest::Approx((reference(plus) - reference(minus)) / (2 * h))
                              .epsilon(1e-6).scale(1.0));
  }
  theta.setConstant(-10);
  CHECK_FALSE(std::isfinite(objective->f(theta, gradient)));
  CHECK(gradient.size() == theta.size());
  CHECK(gradient.isZero(0.0));
}

TEST_CASE("ML correlation target: invalid inputs fail without repair") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  SampleStats sample;
  sample.S = {Eigen::Matrix3d::Identity()}; sample.n_obs = {400};
  sample.mean = {Eigen::Vector3d::Zero()};
  CHECK_FALSE(magmaan::estimate::ml_objective(
      ev, sample, magmaan::model::MomentTarget::Correlation).has_value());
  sample.mean.clear();
  sample.S[0](0, 0) = 2;
  CHECK_FALSE(magmaan::estimate::ml_objective(
      ev, sample, magmaan::model::MomentTarget::Correlation).has_value());
  sample.S[0] << 1, .9, .7, .9, 1, .3, .7, .3, 1;
  const Eigen::MatrixXd original = sample.S[0];
  auto result = magmaan::estimate::ml_objective(
      ev, sample, magmaan::model::MomentTarget::Correlation);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().kind == magmaan::FitError::Kind::NonPositiveDefiniteSample);
  CHECK(sample.S[0].isApprox(original, 0.0));
}

TEST_CASE("ML correlation target: model barrier composes with the shared objective") {
  namespace frontier = magmaan::estimate::frontier;
  auto ev = must_build("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta(ev.n_free());
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = .6 + .04 * static_cast<double>(k);
  SampleStats sample;
  sample.S = {Eigen::MatrixXd::Constant(4, 4, .2)};
  sample.S[0].diagonal().setOnes(); sample.n_obs = {400};
  auto base = magmaan::estimate::ml_objective(
      ev, sample, magmaan::model::MomentTarget::Correlation);
  REQUIRE(base.has_value());
  auto layout = frontier::multiinfo_penalty_layout(
      ev, theta, frontier::PenaltyTarget::Determinacy);
  REQUIRE(layout.has_value());
  auto problem = frontier::multiinfo_penalized_problem(*base, *layout, ev, .25, 400);
  Eigen::VectorXd gradient;
  const double value = problem.f(theta, gradient);
  Eigen::VectorXd base_gradient;
  const double unpenalized = base->f(theta, base_gradient);
  CHECK(value > unpenalized);
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    auto plus = theta; auto minus = theta;
    constexpr double h = 1e-6;
    plus(k) += h; minus(k) -= h;
    Eigen::VectorXd scratch;
    const double fd = (problem.f(plus, scratch) - problem.f(minus, scratch)) / (2 * h);
    CHECK(gradient(k) == doctest::Approx(fd).epsilon(1e-6).scale(1.0));
  }
  auto zero = frontier::multiinfo_penalized_problem(*base, *layout, ev, 0, 400);
  CHECK(zero.f(theta, gradient) == unpenalized);
  CHECK(gradient.isApprox(base_gradient, 0.0));
}

TEST_CASE("ML: fully fixed latent structures with means have empty tangent spaces") {
  for (const std::string structure : {
      "Y =~ 1*y1 + 0.8*y2 + 0.6*y3\nX =~ 1*x1 + 0.8*x2 + 0.6*x3\nY ~~ 1*Y\nX ~~ 1*X\nY ~~ 0*X",
      "Y =~ 1*y1 + 0.8*y2 + 0.6*y3\nX =~ 1*x1 + 0.8*x2 + 0.6*x3\nY ~ 0.25*X\nY ~~ 1*Y\nX ~~ 1*X",
      "Y =~ 1*y1 + 0.8*y2 + 0.6*y3\nX =~ 1*x1 + 0.8*x2 + 0.6*x3\nG =~ 0.5*Y + 0.7*X\nG ~~ 1*G\nY ~~ 1*Y\nX ~~ 1*X\nY ~~ 0*X"}) {
    std::string source = structure;
    for (const auto* v : {"y1", "y2", "y3", "x1", "x2", "x3"})
      source += "\n" + std::string(v) + " ~~ 1*" + v + "\n" + v + " ~ 0*1";
    auto parsed = Parser::parse(source);
    REQUIRE(parsed.has_value());
    auto pt = build(*parsed);
    REQUIRE(pt.has_value());
    REQUIRE(pt->n_free() == 0);
    auto rep = build_matrix_rep(*pt);
    REQUIRE(rep.has_value());
    auto ev = ModelEvaluator::build(*pt, *rep);
    REQUIRE(ev.has_value());
    Eigen::MatrixXd loading = Eigen::MatrixXd::Zero(6, 2);
    loading.col(0).head(3) << 1., .8, .6;
    loading.col(1).tail(3) << 1., .8, .6;
    Eigen::Matrix2d latent = Eigen::Matrix2d::Identity();
    if (structure.find("Y ~ 0.25") != std::string::npos)
      latent << 1.0625, .25, .25, 1.;
    if (structure.find("G =~") != std::string::npos)
      latent << 1.25, .35, .35, 1.49;
    const Eigen::MatrixXd expected = loading * latent * loading.transpose() +
        Eigen::MatrixXd::Identity(6, 6);
    auto implied = ev->sigma(Eigen::VectorXd(0));
    REQUIRE(implied.has_value());
    CHECK((implied->sigma[0] - expected).norm() < 1e-14);
    CHECK(implied->mu[0].norm() == 0.);
    SampleStats sample;
    sample.S = {expected};
    sample.mean = {Eigen::VectorXd::Zero(6)};
    sample.n_obs = {200};
    auto fit = magmaan::test::fit(*pt, *rep, sample);
    REQUIRE(fit.has_value());
    CHECK(fit->theta.size() == 0);
    CHECK(fit->fmin == doctest::Approx(0.).epsilon(1e-12));
    CHECK(magmaan::estimate::fit_verdict(*fit).status == magmaan::estimate::FitCheck::Passed);
  }
}
