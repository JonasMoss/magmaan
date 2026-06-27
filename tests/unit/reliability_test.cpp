#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/measures/reliability.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace rel = magmaan::measures::frontier::reliability;

namespace {

Eigen::MatrixXd one_factor_sigma() {
  Eigen::VectorXd lambda(4);
  lambda << 0.8, 0.7, 0.6, 0.5;
  Eigen::VectorXd psi(4);
  psi << 0.36, 0.51, 0.64, 0.75;
  Eigen::MatrixXd S = lambda * lambda.transpose();
  S.diagonal() += psi;
  return S;
}

Eigen::MatrixXd perturb_vech(const Eigen::MatrixXd& S, Eigen::Index k,
                             double delta) {
  Eigen::MatrixXd out = S;
  Eigen::Index pos = 0;
  for (Eigen::Index j = 0; j < S.cols(); ++j) {
    for (Eigen::Index i = j; i < S.rows(); ++i) {
      if (pos == k) {
        out(i, j) += delta;
        if (i != j) out(j, i) += delta;
        return out;
      }
      ++pos;
    }
  }
  return out;
}

double finite_diff(rel::Coefficient coef, const Eigen::MatrixXd& S,
                   Eigen::Index k) {
  const Eigen::Index p = S.rows();
  Eigen::Index row = 0;
  Eigen::Index col = 0;
  Eigen::Index pos = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      if (pos == k) {
        row = i;
        col = j;
      }
      ++pos;
    }
  }
  const double h = 1e-6 * std::max(1.0, std::abs(S(row, col)));
  auto fp = rel::value(coef, perturb_vech(S, k, h));
  auto fm = rel::value(coef, perturb_vech(S, k, -h));
  REQUIRE(fp.has_value());
  REQUIRE(fm.has_value());
  return (*fp - *fm) / (2.0 * h);
}

}  // namespace

TEST_CASE("reliability covariance coefficients match closed forms") {
  Eigen::MatrixXd S(4, 4);
  S << 1.00, 0.42, 0.35, 0.28,
       0.42, 1.10, 0.33, 0.25,
       0.35, 0.33, 0.90, 0.22,
       0.28, 0.25, 0.22, 0.80;

  auto alpha = rel::cronbach_alpha(S);
  REQUIRE(alpha.has_value());
  const double p = 4.0;
  const double alpha_hand = (p / (p - 1.0)) * (1.0 - S.trace() / S.sum());
  CHECK(*alpha == doctest::Approx(alpha_hand).epsilon(1e-14));

  auto lambda6 = rel::guttman_lambda6(S);
  REQUIRE(lambda6.has_value());
  const Eigen::MatrixXd Sinv = S.inverse();
  double err_sum = 0.0;
  for (Eigen::Index i = 0; i < S.rows(); ++i) err_sum += 1.0 / Sinv(i, i);
  CHECK(*lambda6 == doctest::Approx(1.0 - err_sum / S.sum()).epsilon(1e-14));
}

TEST_CASE("Spearman-Guttman omega recovers one-factor omega in population") {
  const Eigen::MatrixXd S = one_factor_sigma();
  auto omega = rel::spearman_guttman_omega(S);
  REQUIRE(omega.has_value());

  Eigen::VectorXd lambda(4);
  lambda << 0.8, 0.7, 0.6, 0.5;
  Eigen::VectorXd psi(4);
  psi << 0.36, 0.51, 0.64, 0.75;
  const double common = lambda.sum() * lambda.sum();
  const double true_omega = common / (common + psi.sum());
  CHECK(*omega == doctest::Approx(true_omega).epsilon(1e-13));
}

TEST_CASE("reliability gradients agree with central finite differences") {
  Eigen::MatrixXd S(4, 4);
  S << 1.00, 0.42, 0.35, 0.28,
       0.42, 1.10, 0.33, 0.25,
       0.35, 0.33, 0.90, 0.22,
       0.28, 0.25, 0.22, 0.80;

  for (rel::Coefficient coef :
       {rel::Coefficient::Alpha, rel::Coefficient::Lambda6,
        rel::Coefficient::SpearmanGuttmanOmega}) {
    auto g = rel::gradient(coef, S);
    REQUIRE(g.has_value());
    for (Eigen::Index k = 0; k < g->size(); ++k) {
      CHECK((*g)(k) == doctest::Approx(finite_diff(coef, S, k)).epsilon(2e-6));
    }
  }
}

TEST_CASE("reliability delta method uses Gamma on the vech scale") {
  const Eigen::MatrixXd S = one_factor_sigma();
  auto gamma = magmaan::data::gamma_nt(S);
  REQUIRE(gamma.has_value());

  auto out = rel::delta_method(rel::Coefficient::Lambda6, S, *gamma, 500);
  REQUIRE(out.has_value());
  CHECK(out->value == doctest::Approx(*rel::guttman_lambda6(S)).epsilon(1e-14));
  CHECK(out->gradient.size() == 10);
  CHECK(out->avar >= 0.0);
  CHECK(out->se == doctest::Approx(std::sqrt(out->avar / 500.0)).epsilon(1e-14));
}

namespace {

// A second-order (higher-order) congeneric population: three first-order
// factors, three indicators each, one second-order general factor with loadings
// gam. Standardized indicators (item variances 1) and unit-variance first-order
// factors, so Phi = gam gam' + diag(1 - gam^2) has unit diagonal.
struct HigherOrderPop {
  Eigen::MatrixXd Sigma;
  Eigen::VectorXi block;
  double omega_total_true = 0.0;
  double omega_h_true      = 0.0;
};

HigherOrderPop higher_order_pop() {
  const std::array<std::array<double, 3>, 3> lam = {{{0.80, 0.70, 0.75},
                                                     {0.70, 0.60, 0.65},
                                                     {0.75, 0.80, 0.70}}};
  Eigen::Vector3d gam(0.70, 0.60, 0.65);

  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(9, 3);
  Eigen::VectorXd theta(9);
  Eigen::VectorXi block(9);
  Eigen::Index idx = 0;
  for (std::size_t f = 0; f < 3; ++f) {
    for (std::size_t j = 0; j < 3; ++j) {
      const double lij = lam[f][j];
      const auto fi = static_cast<Eigen::Index>(f);
      L(idx, fi) = lij;
      theta(idx) = 1.0 - lij * lij;  // standardized indicators
      block(idx) = static_cast<int>(f);
      ++idx;
    }
  }
  Eigen::Matrix3d Phi = gam * gam.transpose();
  for (Eigen::Index f = 0; f < 3; ++f) Phi(f, f) = 1.0;  // unit-variance first-order factors

  Eigen::MatrixXd Sigma = L * Phi * L.transpose();
  Sigma.diagonal() += theta;

  const double T = Sigma.sum();
  double g_load_sum = 0.0;  // sum_i lambda_i * gam_{f(i)} = total-score loading on G
  for (std::size_t f = 0; f < 3; ++f) {
    for (std::size_t j = 0; j < 3; ++j) {
      g_load_sum += lam[f][j] * gam(static_cast<Eigen::Index>(f));
    }
  }
  HigherOrderPop pop;
  pop.Sigma = Sigma;
  pop.block = block;
  pop.omega_total_true = (T - theta.sum()) / T;          // 1'(Sigma-Theta)1 / 1'Sigma1
  pop.omega_h_true      = g_load_sum * g_load_sum / T;    // (1'Lambda gam)^2 / 1'Sigma1
  return pop;
}

}  // namespace

TEST_CASE("multidimensional omega recovers truth on the higher-order manifold") {
  const HigherOrderPop pop = higher_order_pop();
  rel::OmegaSpec spec;
  spec.block = pop.block;

  auto ot = rel::omega_multidim(rel::OmegaTarget::Total, pop.Sigma, spec);
  REQUIRE(ot.has_value());
  CHECK(*ot == doctest::Approx(pop.omega_total_true).epsilon(1e-9));

  auto oh = rel::omega_multidim(rel::OmegaTarget::Hierarchical, pop.Sigma, spec);
  REQUIRE(oh.has_value());
  CHECK(*oh == doctest::Approx(pop.omega_h_true).epsilon(1e-9));

  // omega_h < omega_total: the general factor carries less than the full common part.
  CHECK(*oh < *ot);
}

TEST_CASE("single-block omega_total recovers one-factor omega (Hancock-An reduction)") {
  const Eigen::MatrixXd S = one_factor_sigma();
  rel::OmegaSpec spec;
  spec.block = Eigen::VectorXi::Zero(S.rows());  // one factor

  Eigen::VectorXd lambda(4);
  lambda << 0.8, 0.7, 0.6, 0.5;
  Eigen::VectorXd psi(4);
  psi << 0.36, 0.51, 0.64, 0.75;
  const double common = lambda.sum() * lambda.sum();
  const double true_omega = common / (common + psi.sum());

  auto ot = rel::omega_multidim(rel::OmegaTarget::Total, S, spec);
  REQUIRE(ot.has_value());
  CHECK(*ot == doctest::Approx(true_omega).epsilon(1e-12));

  // Hierarchical with one factor is rejected (needs k >= 3).
  auto oh = rel::omega_multidim(rel::OmegaTarget::Hierarchical, S, spec);
  CHECK_FALSE(oh.has_value());
}

TEST_CASE("multidimensional omega delta method uses Gamma on the vech scale") {
  const HigherOrderPop pop = higher_order_pop();
  rel::OmegaSpec spec;
  spec.block = pop.block;
  auto gamma = magmaan::data::gamma_nt(pop.Sigma);
  REQUIRE(gamma.has_value());

  for (rel::OmegaTarget target :
       {rel::OmegaTarget::Total, rel::OmegaTarget::Hierarchical}) {
    auto out = rel::omega_multidim_delta(target, pop.Sigma, spec, *gamma, 1000);
    REQUIRE(out.has_value());
    auto val = rel::omega_multidim(target, pop.Sigma, spec);
    REQUIRE(val.has_value());
    CHECK(out->value == doctest::Approx(*val).epsilon(1e-12));
    CHECK(out->gradient.size() == 45);  // vech of 9x9
    CHECK(out->avar >= 0.0);
    CHECK(out->se == doctest::Approx(std::sqrt(out->avar / 1000.0)).epsilon(1e-14));
    CHECK(out->se > 0.0);
  }
}

TEST_CASE("omega_from_fit recovers higher-order truth from a fitted CFA") {
  const HigherOrderPop pop = higher_order_pop();

  // Second-order CFA: three first-order factors (three indicators each) and a
  // general factor with no direct indicators (its Λ column is all zero, so the
  // hierarchical path detects it). Fitting ML to the population covariance
  // reproduces it exactly, so the model-implied omega equals the closed-form
  // truth (omega is identification-invariant; marker identification is fine).
  auto fp = magmaan::parse::Parser::parse(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f3 =~ x7 + x8 + x9\n"
      "g  =~ f1 + f2 + f3");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);             REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  magmaan::data::SampleStats samp;
  samp.S = {pop.Sigma};
  samp.n_obs = {std::int64_t{1000}};

  auto est_or = magmaan::test::fit(*pt, *mr, samp);
  REQUIRE(est_or.has_value());
  const magmaan::estimate::Estimates& est = *est_or;

  auto gamma = magmaan::data::gamma_nt(pop.Sigma);
  REQUIRE(gamma.has_value());

  // The point estimate is independent of the SE path, so every FitWeight gives
  // the same value (the truth); each weight exercises a distinct robust vcov.
  for (rel::FitWeight w :
       {rel::FitWeight::ML, rel::FitWeight::GLS, rel::FitWeight::ULS}) {
    auto ot = rel::omega_from_fit(rel::OmegaTarget::Total, w, *pt, *mr, samp, est,
                                  *gamma, 1000);
    REQUIRE(ot.has_value());
    CHECK(ot->value == doctest::Approx(pop.omega_total_true).epsilon(1e-5));
    CHECK(ot->gradient.size() == static_cast<Eigen::Index>(est.theta.size()));
    CHECK(ot->se > 0.0);
    CHECK(ot->avar == doctest::Approx(ot->se * ot->se * 1000.0).epsilon(1e-10));

    auto oh = rel::omega_from_fit(rel::OmegaTarget::Hierarchical, w, *pt, *mr,
                                  samp, est, *gamma, 1000);
    REQUIRE(oh.has_value());
    CHECK(oh->value == doctest::Approx(pop.omega_h_true).epsilon(1e-5));
    CHECK(oh->se > 0.0);

    // The general factor carries less than the full common part.
    CHECK(oh->value < ot->value);
  }
}
