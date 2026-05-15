#include <doctest/doctest.h>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/snlls.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/gls/gls.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::Bounds;
using magmaan::gls::GLS;
using magmaan::gls::ULS;
using magmaan::gls::WLS;
using magmaan::model::ModelEvaluator;
using magmaan::model::build_matrix_rep;
using magmaan::optim::CeresBoundedOptimizer;
using magmaan::optim::LbfgsBOptimizer;
using magmaan::optim::LbfgsBOptions;
using magmaan::parse::Parser;
using magmaan::spec::lavaanify;

namespace {

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

Handles handles_for(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

Eigen::Matrix3d make_1f_S() {
  Eigen::Vector3d lambda_true(1.0, 0.85, 0.70);
  const double psi_true = 2.0;
  Eigen::Vector3d theta_true(0.6, 0.5, 0.8);
  return lambda_true * lambda_true.transpose() * psi_true +
         theta_true.asDiagonal().toDenseMatrix();
}

Eigen::Matrix4d make_misspecified_1f_S() {
  Eigen::Matrix4d S;
  S << 2.0, 0.7, 0.4, 0.2,
       0.7, 1.8, 0.3, 0.25,
       0.4, 0.3, 1.6, 0.5,
       0.2, 0.25, 0.5, 1.4;
  return S;
}

LbfgsBOptimizer snlls_lbfgsb() {
  return LbfgsBOptimizer(LbfgsBOptions{
      .max_iter = 2000, .ftol = 1e-13, .gtol = 1e-8});
}

template <class D>
void check_alpha_basis_matches_residual_basis(const D& discrepancy) {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_misspecified_1f_S()};
  samp.n_obs = {301};

  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto cls = magmaan::estimate::detail_snlls::classify(h.pt, *ev, *x0);
  REQUIRE(cls.has_value());

  Eigen::VectorXd beta = cls->beta0;
  beta.array() += 0.05;
  const Eigen::VectorXd theta_base =
      magmaan::estimate::detail_snlls::expand_beta(*cls, beta);

  auto prof = magmaan::estimate::detail_snlls::profile_at(
      *ev, samp, discrepancy, *cls, beta);
  REQUIRE(prof.has_value());

  auto eval0 = ev->evaluate(theta_base, false, false);
  REQUIRE(eval0.has_value());
  auto r0 = discrepancy.residuals(samp, eval0->moments);
  REQUIRE(r0.has_value());

  Eigen::MatrixXd H_basis(r0->size(), cls->K_alpha.cols());
  for (Eigen::Index j = 0; j < cls->K_alpha.cols(); ++j) {
    auto eval_j = ev->evaluate(theta_base + cls->K_alpha.col(j), false, false);
    REQUIRE(eval_j.has_value());
    auto rj = discrepancy.residuals(samp, eval_j->moments);
    REQUIRE(rj.has_value());
    H_basis.col(j) = *rj - *r0;
  }
  CHECK((prof->H - H_basis).cwiseAbs().maxCoeff() < 1e-10);
}

template <class D>
void check_profile_gradient_finite_difference(const D& discrepancy) {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_misspecified_1f_S()};
  samp.n_obs = {301};

  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto cls = magmaan::estimate::detail_snlls::classify(h.pt, *ev, *x0);
  REQUIRE(cls.has_value());

  Eigen::VectorXd beta = cls->beta0;
  beta.array() += 0.05;
  auto prof = magmaan::estimate::detail_snlls::profile_at(
      *ev, samp, discrepancy, *cls, beta);
  REQUIRE(prof.has_value());
  auto grad = magmaan::estimate::detail_snlls::profile_gradient_at(
      *ev, samp, discrepancy, *cls, *prof);
  REQUIRE(grad.has_value());

  Eigen::VectorXd fd(beta.size());
  constexpr double eps = 1e-6;
  for (Eigen::Index j = 0; j < beta.size(); ++j) {
    Eigen::VectorXd bp = beta;
    Eigen::VectorXd bm = beta;
    bp(j) += eps;
    bm(j) -= eps;
    auto fp = magmaan::estimate::detail_snlls::profile_at(
        *ev, samp, discrepancy, *cls, bp);
    auto fm = magmaan::estimate::detail_snlls::profile_at(
        *ev, samp, discrepancy, *cls, bm);
    REQUIRE(fp.has_value());
    REQUIRE(fm.has_value());
    fd(j) = (fp->fmin - fm->fmin) / (2.0 * eps);
  }
  CHECK((*grad - fd).cwiseAbs().maxCoeff() < 1e-5);
}

}  // namespace

TEST_CASE("SNLLS: analytic alpha basis matches residual basis evaluations") {
  check_alpha_basis_matches_residual_basis(ULS{});
  check_alpha_basis_matches_residual_basis(GLS{});
  check_alpha_basis_matches_residual_basis(WLS({Eigen::MatrixXd::Identity(10, 10)}));
}

TEST_CASE("SNLLS: profiled scalar gradient matches finite differences") {
  check_profile_gradient_finite_difference(ULS{});
  check_profile_gradient_finite_difference(GLS{});
  check_profile_gradient_finite_difference(WLS({Eigen::MatrixXd::Identity(10, 10)}));
}

TEST_CASE("SNLLS: ULS profiles linear block and recovers a feasible 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto est = magmaan::estimate::fit_snlls_bounded(
      h.pt, h.rep, samp, Bounds{}, ULS{}, snlls_lbfgsb());
  if (!est.has_value()) {
    MESSAGE("SNLLS ULS failed: " << est.error().detail);
  }
  REQUIRE(est.has_value());
  CHECK(est->snlls.n_nonlinear == 2);
  CHECK(est->snlls.n_linear == 4);
  CHECK(est->snlls.admissible);
  CHECK(est->fmin < 1e-10);

  auto ev = ModelEvaluator::build(h.pt, h.rep).value();
  auto sm = ev.sigma(est->theta).value();
  CHECK((samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff() < 1e-5);
}

TEST_CASE("SNLLS: GLS and WLS agree with full LS on a feasible 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto gls = magmaan::estimate::fit_snlls_bounded(
      h.pt, h.rep, samp, Bounds{}, GLS{}, snlls_lbfgsb());
  REQUIRE(gls.has_value());
  CHECK(gls->fmin < 1e-10);

  WLS wls({Eigen::MatrixXd::Identity(6, 6)});
  auto wls_est = magmaan::estimate::fit_snlls_bounded(
      h.pt, h.rep, samp, Bounds{}, wls, snlls_lbfgsb());
  REQUIRE(wls_est.has_value());
  CHECK(wls_est->fmin < 1e-10);
}

TEST_CASE("SNLLS: strict compatibility rejects cross-block equality") {
  auto h = handles_for("f =~ x1 + a*x2 + x3\nf ~~ a*f");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto est = magmaan::estimate::fit_snlls_bounded(
      h.pt, h.rep, samp, Bounds{}, ULS{}, snlls_lbfgsb());
  REQUIRE_FALSE(est.has_value());
  if (!est.has_value()) {
    CHECK(est.error().detail.find("SNLLS compatibility") != std::string::npos);
  }
}

TEST_CASE("SNLLS: covariance-only model is solved by profiling alone") {
  auto h = handles_for("x1 ~~ x1\nx2 ~~ x2\nx1 ~~ x2");
  SampleStats samp;
  Eigen::Matrix2d S;
  S << 2.0, 0.4,
       0.4, 1.5;
  samp.S = {S};
  samp.n_obs = {100};

  auto est = magmaan::estimate::fit_snlls_bounded(
      h.pt, h.rep, samp, Bounds{}, ULS{}, snlls_lbfgsb());
  REQUIRE(est.has_value());
  CHECK(est->iterations == 0);
  CHECK(est->snlls.n_nonlinear == 0);
  CHECK(est->snlls.n_linear == 3);
  CHECK(est->fmin < 1e-12);
}

#ifdef MAGMAAN_WITH_CERES
TEST_CASE("SNLLS: Ceres backend recovers a feasible 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto est = magmaan::estimate::fit_snlls_bounded(
      h.pt, h.rep, samp, Bounds{}, ULS{}, CeresBoundedOptimizer{});
  if (!est.has_value()) {
    MESSAGE("SNLLS Ceres failed: " << est.error().detail);
  }
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);

  auto ev = ModelEvaluator::build(h.pt, h.rep).value();
  auto sm = ev.sigma(est->theta).value();
  CHECK((samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff() < 1e-5);
}
#endif
