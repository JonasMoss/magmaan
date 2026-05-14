#include <doctest/doctest.h>

#include <limits>
#include <random>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/gls/gls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::data::SampleStats;
using magmaan::gls::GLS;
using magmaan::gls::WLS;
using magmaan::model::ModelEvaluator;
using magmaan::model::build_matrix_rep;
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
  static thread_local magmaan::model::MatrixRep s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  auto ev = ModelEvaluator::build(s_pt, s_mr);
  REQUIRE(ev.has_value());
  return std::move(*ev);
}

Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() +
         Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

Eigen::VectorXd theta_for(const ModelEvaluator& ev, std::mt19937& rng) {
  Eigen::VectorXd theta(static_cast<Eigen::Index>(ev.n_free()));
  std::uniform_real_distribution<double> d(0.4, 1.2);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);
  return theta;
}

}  // namespace

TEST_CASE("GLS: covariance value matches hand GLS formula") {
  Eigen::Matrix2d S;
  S << 2.0, 0.3,
       0.3, 1.5;
  Eigen::Matrix2d Sigma;
  Sigma << 1.7, 0.5,
           0.5, 1.2;

  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {100};
  magmaan::model::ImpliedMoments im;
  im.sigma = {Sigma};

  const Eigen::Matrix2d Sinv =
      S.llt().solve(Eigen::Matrix2d::Identity());
  const Eigen::Matrix2d D = Sigma - S;
  const double manual = 0.5 * (Sinv * D * Sinv * D).trace();

  GLS gls;
  auto f = gls.value(samp, im);
  REQUIRE(f.has_value());
  CHECK(*f == doctest::Approx(manual).epsilon(1e-12));
}

TEST_CASE("WLS: value matches explicit weighted moment quadratic") {
  Eigen::Matrix2d S;
  S << 2.0, 0.4,
       0.4, 1.0;
  Eigen::Matrix2d Sigma;
  Sigma << 1.5, 0.2,
           0.2, 1.3;
  Eigen::Vector2d mean;
  mean << 1.0, 2.0;
  Eigen::Vector2d mu;
  mu << 1.5, 1.8;

  Eigen::VectorXd d(5);
  d << (mu - mean), Sigma(0, 0) - S(0, 0), Sigma(1, 0) - S(1, 0),
      Sigma(1, 1) - S(1, 1);
  Eigen::MatrixXd W = Eigen::MatrixXd::Identity(5, 5);
  W.diagonal() << 2.0, 3.0, 4.0, 5.0, 6.0;

  SampleStats samp;
  samp.S = {S};
  samp.mean = {mean};
  samp.n_obs = {77};
  magmaan::model::ImpliedMoments im;
  im.sigma = {Sigma};
  im.mu = {mu};

  WLS wls({W});
  auto f = wls.value(samp, im);
  REQUIRE(f.has_value());
  const double manual = 0.5 * (d.transpose() * W * d)(0, 0);
  CHECK(*f == doctest::Approx(manual).epsilon(1e-12));
}

TEST_CASE("GLS and WLS: residual and gradient identities") {
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");
  std::mt19937 rng(101);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  Eigen::Vector3d mean;
  mean << 1.0, 2.0, 3.0;
  samp.mean = {mean};
  samp.n_obs = {150};

  const Eigen::VectorXd theta = theta_for(ev, rng);
  auto sm = ev.sigma(theta).value();
  auto J = ev.dsigma_dtheta(theta).value();
  auto Jmu = ev.dmu_dtheta(theta).value();

  const Eigen::Index pstar = 6;
  Eigen::MatrixXd W = Eigen::MatrixXd::Identity(3 + pstar, 3 + pstar);
  W.diagonal().array() += 0.5;

  GLS gls;
  WLS wls({W});
  for (int which = 0; which < 2; ++which) {
    auto F = which == 0 ? gls.value(samp, sm) : wls.value(samp, sm);
    auto r = which == 0 ? gls.residuals(samp, sm) : wls.residuals(samp, sm);
    auto g = which == 0 ? gls.gradient(samp, sm, J, Jmu)
                        : wls.gradient(samp, sm, J, Jmu);
    auto Jr = which == 0 ? gls.residual_jacobian(samp, sm, J, Jmu)
                         : wls.residual_jacobian(samp, sm, J, Jmu);
    REQUIRE(F.has_value());
    REQUIRE(r.has_value());
    REQUIRE(g.has_value());
    REQUIRE(Jr.has_value());
    CHECK(0.5 * r->squaredNorm() == doctest::Approx(*F).epsilon(1e-12));
    CHECK((Jr->transpose() * *r - *g).cwiseAbs().maxCoeff() < 1e-12);
  }
}

TEST_CASE("GLS and WLS: residual Jacobian matches finite differences") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  std::mt19937 rng(202);
  SampleStats samp;
  samp.S = {random_pd(rng, 3)};
  samp.n_obs = {200};

  const Eigen::VectorXd theta = theta_for(ev, rng);
  auto sm = ev.sigma(theta).value();
  auto J = ev.dsigma_dtheta(theta).value();

  Eigen::MatrixXd W = Eigen::MatrixXd::Identity(6, 6);
  W.diagonal() << 1.2, 1.4, 1.6, 1.8, 2.0, 2.2;

  GLS gls;
  WLS wls({W});
  const double h = 1e-6;
  for (int which = 0; which < 2; ++which) {
    auto Jr = which == 0 ? gls.residual_jacobian(samp, sm, J)
                         : wls.residual_jacobian(samp, sm, J);
    REQUIRE(Jr.has_value());
    Eigen::MatrixXd fd(Jr->rows(), Jr->cols());
    for (Eigen::Index k = 0; k < theta.size(); ++k) {
      Eigen::VectorXd tp = theta;
      Eigen::VectorXd tm = theta;
      tp(k) += h;
      tm(k) -= h;
      auto rp = which == 0 ? gls.residuals(samp, ev.sigma(tp).value())
                           : wls.residuals(samp, ev.sigma(tp).value());
      auto rm = which == 0 ? gls.residuals(samp, ev.sigma(tm).value())
                           : wls.residuals(samp, ev.sigma(tm).value());
      REQUIRE(rp.has_value());
      REQUIRE(rm.has_value());
      fd.col(k) = (*rp - *rm) / (2.0 * h);
    }
    CHECK((*Jr - fd).cwiseAbs().maxCoeff() < 1e-5);
  }
}

TEST_CASE("WLS: validates weight block count, dimensions, finite values, and PD") {
  Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {50};
  magmaan::model::ImpliedMoments im;
  im.sigma = {S};

  WLS wrong_blocks(std::vector<Eigen::MatrixXd>{});
  CHECK_FALSE(wrong_blocks.value(samp, im).has_value());

  WLS wrong_dim({Eigen::MatrixXd::Identity(2, 2)});
  CHECK_FALSE(wrong_dim.value(samp, im).has_value());

  Eigen::MatrixXd nonfinite = Eigen::MatrixXd::Identity(3, 3);
  nonfinite(1, 1) = std::numeric_limits<double>::quiet_NaN();
  WLS bad_finite({nonfinite});
  CHECK_FALSE(bad_finite.value(samp, im).has_value());

  Eigen::MatrixXd nonpd = Eigen::MatrixXd::Identity(3, 3);
  nonpd(2, 2) = -1.0;
  WLS bad_pd({nonpd});
  CHECK_FALSE(bad_pd.value(samp, im).has_value());
}
