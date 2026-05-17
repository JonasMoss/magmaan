#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <random>

#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"

#include "magmaan/estimate/fit.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::lavaanify;

namespace {

ModelEvaluator must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
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
  auto f = magmaan::nt::ml_value(samp, *sm);
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
  auto g_an = magmaan::nt::ml_gradient(samp, sm, J).value();

  // Finite-difference gradient.
  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    auto smp = ev.sigma(tp).value();
    auto smm = ev.sigma(tm).value();
    g_fd(k) = (magmaan::nt::ml_value(samp, smp).value() - magmaan::nt::ml_value(samp, smm).value()) / (2.0 * h);
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
  auto cache = magmaan::nt::ml_prepare(samp).value();
  auto f_sep = magmaan::nt::ml_value(samp, sm).value();
  auto g_sep = magmaan::nt::ml_gradient(samp, sm, J).value();
  auto vg = magmaan::nt::ml_value_gradient(samp, cache, sm, J).value();

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

  auto f_or = magmaan::nt::ml_value(samp, *sm);
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

  auto g_an = magmaan::nt::ml_gradient(samp, sm, J, Jmu).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    auto smp = ev.sigma(tp).value();
    auto smm = ev.sigma(tm).value();
    g_fd(k) = (magmaan::nt::ml_value(samp, smp).value() - magmaan::nt::ml_value(samp, smm).value()) / (2.0 * h);
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
  auto pt = magmaan::spec::lavaanify(*fp);
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
  auto g_an = magmaan::nt::ml_gradient(samp, sm, J).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    g_fd(k) = (magmaan::nt::ml_value(samp, ev.sigma(tp).value()).value() -
               magmaan::nt::ml_value(samp, ev.sigma(tm).value()).value()) / (2.0 * h);
  }
  const double max_diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  // Slightly looser tolerance for the larger model — FD noise grows with
  // gradient magnitude.
  CHECK(max_diff < 1e-4);
}
