#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/measures/reliability.hpp"

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
