#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/coordinates.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/frontier/newton_accuracy.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

using magmaan::data::SampleStats;
using magmaan::estimate::Estimates;
using magmaan::estimate::frontier::newton_accuracy_from;
using magmaan::estimate::frontier::newton_accuracy_ml;
using magmaan::estimate::frontier::NewtonAccuracyStatus;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::spec::LatentStructure;

struct Model {
  LatentStructure pt;
  MatrixRep rep;
  SampleStats samp;
  Eigen::VectorXd theta0;  // exact ML solution: S = Sigma(theta0)
};

// Builds the model, takes simple start values as the population point (after
// projecting onto the equality constraints), and sets S to its implied
// covariance, so theta0 is the exact ML solution.
Model exact_model(std::string_view syntax, std::int64_t n = 250) {
  Model m;
  auto flat = magmaan::parse::Parser::parse(syntax);
  REQUIRE(flat.has_value());
  magmaan::spec::BuildOptions opts;
  opts.fixed_x = false;
  auto pt = magmaan::spec::build(*flat, opts);
  REQUIRE(pt.has_value());
  m.pt = std::move(*pt);
  auto rep = magmaan::model::build_matrix_rep(m.pt);
  REQUIRE(rep.has_value());
  m.rep = std::move(*rep);
  // Any positive-definite S serves for the start values.
  const auto p = static_cast<Eigen::Index>(m.rep.ov_names[0].size());
  Eigen::MatrixXd S0 = Eigen::MatrixXd::Constant(p, p, 0.4);
  S0.diagonal().array() = 1.0;
  m.samp.S = {S0};
  m.samp.n_obs = {n};
  auto x0 = magmaan::estimate::simple_start_values(m.pt, m.rep, m.samp, {});
  REQUIRE(x0.has_value());
  auto con = magmaan::estimate::build_eq_constraints(m.pt, /*allow_nonlinear=*/true);
  REQUIRE(con.has_value());
  m.theta0 = con->expand(con->contract(*x0));
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto e = ev->evaluate(m.theta0, false, false);
  REQUIRE(e.has_value());
  m.samp.S = {e->moments.sigma[0]};
  return m;
}

Estimates at(const Eigen::VectorXd& theta) {
  Estimates e;
  e.theta = theta;
  return e;
}

}  // namespace

TEST_CASE("Newton accuracy: exact quadratic identities") {
  Eigen::MatrixXd H(2, 2);
  H << 4, 1, 1, 2;
  Eigen::VectorXd e(2);
  e << 0.1, -0.2;
  const Eigen::VectorXd g = H * e;
  const auto a = newton_accuracy_from(g, H);
  REQUIRE(a.status == NewtonAccuracyStatus::Available);
  CHECK(std::abs(a.distance * a.distance - e.dot(H * e)) < 1e-13);
  CHECK(std::abs(a.predicted_gain - 0.5 * e.dot(H * e)) < 1e-13);
  CHECK(std::abs(a.max_step - 0.2) < 1e-13);

  // Invariant under a linear change of coordinates.
  Eigen::MatrixXd T(2, 2);
  T << 0.01, 2, 0, 100;
  const auto b = newton_accuracy_from(T.transpose() * g, T.transpose() * H * T);
  CHECK(std::abs(a.distance - b.distance) < 1e-12);

  // Total-likelihood scaling: 100 times the data gives 10 times the distance.
  const auto c = newton_accuracy_from(100 * g, 100 * H);
  CHECK(std::abs(c.distance - 10 * a.distance) < 1e-12);

  // Budget.
  CHECK_FALSE(a.passed);
  magmaan::estimate::frontier::NewtonAccuracyOptions loose;
  loose.budget = 1.0;
  CHECK(newton_accuracy_from(g, H, loose).passed);

  H(1, 1) = -1;
  CHECK(newton_accuracy_from(g, H).status == NewtonAccuracyStatus::NonpositiveCurvature);
  H << 1, 1, 1, 1;
  CHECK(newton_accuracy_from(g, H).status != NewtonAccuracyStatus::Available);
}

TEST_CASE("Newton accuracy: exact solution and a small perturbation") {
  for (std::string_view syntax :
       {std::string_view{"f =~ x1 + x2 + x3 + x4"},
        std::string_view{"f =~ x1 + a*x2 + a*x3 + x4\nx2 ~~ v*x2\nx4 ~~ v*x4"}}) {
    CAPTURE(syntax);
    const Model m = exact_model(syntax);
    const auto exact = newton_accuracy_ml(m.pt, m.rep, m.samp, at(m.theta0));
    REQUIRE(exact.status == NewtonAccuracyStatus::Available);
    CHECK(exact.distance < 1e-8);
    CHECK(exact.passed);
    CHECK(exact.covariance_interior);

    // Move along the constraint surface: d must match sqrt(delta' I delta).
    auto con = magmaan::estimate::build_eq_constraints(m.pt);
    REQUIRE(con.has_value());
    Eigen::VectorXd alpha = con->contract(m.theta0);
    Eigen::VectorXd delta = Eigen::VectorXd::LinSpaced(alpha.size(), 1.0, -1.0);
    delta *= 2e-4 / delta.norm();
    const Eigen::VectorXd theta = con->expand(alpha + delta);
    auto info = magmaan::inference::information_observed_analytic(
        m.pt, m.rep, m.samp, at(m.theta0));
    REQUIRE(info.has_value());
    const Eigen::MatrixXd I = con->K().transpose() * (*info) * con->K();
    const double predicted = std::sqrt(delta.dot(I * delta));
    const auto moved = newton_accuracy_ml(m.pt, m.rep, m.samp, at(theta));
    REQUIRE(moved.status == NewtonAccuracyStatus::Available);
    CHECK(std::abs(moved.distance / predicted - 1.0) < 1e-3);
    CHECK(moved.n_reduced == con->n_alpha);
  }
}

TEST_CASE("Newton accuracy: boundary covariance and unsupported constraints") {
  Model m = exact_model("f =~ x1 + x2 + x3 + x4");
  // Put the first residual variance at zero: the covariance domain's boundary.
  Eigen::VectorXd theta = m.theta0;
  for (std::size_t r = 0; r < m.pt.op.size(); ++r) {
    const auto v = m.pt.lhs_var[r];
    if (m.pt.op[r] == magmaan::parse::Op::Covariance && m.pt.free[r] > 0 && v >= 0 &&
        v == m.pt.rhs_var[r] && m.pt.ov_pos[static_cast<std::size_t>(v)] >= 0) {
      theta[m.pt.free[r] - 1] = 0.0;
      break;
    }
  }
  const auto boundary = newton_accuracy_ml(m.pt, m.rep, m.samp, at(theta));
  CHECK_FALSE(boundary.covariance_interior);

  Model nl = exact_model("f =~ x1 + a*x2 + b*x3 + x4\na == b^2");
  CHECK(newton_accuracy_ml(nl.pt, nl.rep, nl.samp, at(nl.theta0)).status ==
        NewtonAccuracyStatus::Unsupported);
}

TEST_CASE("Newton accuracy: multi-group distance matches the information metric") {
  // Two groups of unequal size: the total score and total information must
  // weight the groups alike, so a small displacement from the exact solution
  // has d = sqrt(delta' I delta).
  auto flat = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(flat.has_value());
  magmaan::spec::BuildOptions opts;
  opts.fixed_x = false;
  opts.n_groups = 2;
  auto pt = magmaan::spec::build(*flat, opts);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  SampleStats samp;
  Eigen::MatrixXd S0 = Eigen::MatrixXd::Constant(4, 4, 0.4);
  S0.diagonal().array() = 1.0;
  samp.S = {S0, S0};
  samp.n_obs = {150, 350};
  auto x0 = magmaan::estimate::simple_start_values(*pt, *rep, samp, {});
  REQUIRE(x0.has_value());
  auto con = magmaan::estimate::build_eq_constraints(*pt);
  REQUIRE(con.has_value());
  Eigen::VectorXd theta0 = con->expand(con->contract(*x0));
  // Make the groups differ, so neither group's moments can stand in for both.
  for (std::size_t r = 0; r < pt->op.size(); ++r) {
    if (pt->group[r] == 1 && pt->free[r] > 0 &&
        pt->op[r] == magmaan::parse::Op::Covariance) {
      theta0[pt->free[r] - 1] *= 1.5;
    }
  }
  auto ev = ModelEvaluator::build(*pt, *rep);
  REQUIRE(ev.has_value());
  auto e = ev->evaluate(theta0, false, false);
  REQUIRE(e.has_value());
  samp.S = {e->moments.sigma[0], e->moments.sigma[1]};

  const auto exact = newton_accuracy_ml(*pt, *rep, samp, at(theta0));
  REQUIRE(exact.status == NewtonAccuracyStatus::Available);
  CHECK(exact.distance < 1e-8);

  Eigen::VectorXd delta = Eigen::VectorXd::LinSpaced(theta0.size(), 1.0, -1.0);
  delta *= 2e-4 / delta.norm();
  auto info = magmaan::inference::information_observed_analytic(
      *pt, *rep, samp, at(theta0));
  REQUIRE(info.has_value());
  const double predicted = std::sqrt(delta.dot(*info * delta));
  const auto moved = newton_accuracy_ml(*pt, *rep, samp, at(theta0 + delta));
  REQUIRE(moved.status == NewtonAccuracyStatus::Available);
  CHECK(std::abs(moved.distance / predicted - 1.0) < 1e-3);
}

TEST_CASE("Newton accuracy: ML fits attach it and the common verdict uses it") {
  using magmaan::estimate::fit_verdict;
  using magmaan::estimate::FitCheck;
  using magmaan::estimate::StationarityCriterion;
  using magmaan::estimate::StationarityDomain;
  Model m = exact_model("f =~ x1 + x2 + x3 + x4", 300);
  // Perturb S so the solution is not the start.
  Eigen::MatrixXd E(4, 4);
  E << 0.05, 0.02, -0.01, 0.00,
       0.02, -0.03, 0.01, 0.02,
      -0.01, 0.01, 0.04, -0.02,
       0.00, 0.02, -0.02, 0.01;
  m.samp.S[0] += E;

  auto ordinary = magmaan::estimate::fit_ml(m.pt, m.rep, m.samp, m.theta0);
  REQUIRE(ordinary.has_value());
  const auto& na = ordinary->diagnostics.newton_accuracy;
  CHECK(na.checked);
  REQUIRE(na.status == NewtonAccuracyStatus::Available);
  CHECK(na.passed);
  const auto recomputed = newton_accuracy_ml(m.pt, m.rep, m.samp, *ordinary);
  CHECK(std::abs(recomputed.distance - na.distance) <= 1e-12 + 1e-9 * na.distance);
  auto verdict = fit_verdict(*ordinary);
  CHECK(verdict.criterion == StationarityCriterion::Newton);
  CHECK(verdict.status == FitCheck::Passed);

  // The same fit with a failed accuracy check fails the verdict, even though
  // its first-order residual passes.
  auto degraded = *ordinary;
  degraded.diagnostics.newton_accuracy.passed = false;
  degraded.diagnostics.newton_accuracy.distance = 0.5;
  CHECK(degraded.diagnostics.geometric_stationarity.ambient_stationary);
  CHECK(fit_verdict(degraded).status == FitCheck::Failed);
  degraded.diagnostics.newton_accuracy.status =
      NewtonAccuracyStatus::NonpositiveCurvature;
  CHECK(fit_verdict(degraded).status == FitCheck::Failed);
  // Unsupported (nonlinear equalities): the first-order check decides.
  degraded.diagnostics.newton_accuracy.status = NewtonAccuracyStatus::Unsupported;
  CHECK(fit_verdict(degraded).criterion == StationarityCriterion::FirstOrder);
  CHECK(fit_verdict(degraded).status == FitCheck::Passed);
  // An active box bound: the Newton step is infeasible, first order decides.
  degraded.diagnostics.newton_accuracy.status = NewtonAccuracyStatus::Available;
  degraded.diagnostics.active_bounds_full.at_lower = {0};
  CHECK(fit_verdict(degraded).criterion == StationarityCriterion::FirstOrder);

  // PSD fit at an interior solution: same point, Newton decides.
  auto psd = magmaan::estimate::frontier::fit_ml_psd(m.pt, m.rep, m.samp, m.theta0);
  REQUIRE(psd.has_value());
  CHECK(psd->diagnostics.newton_accuracy.checked);
  CHECK(psd->diagnostics.newton_accuracy.covariance_interior);
  verdict = fit_verdict(*psd);
  CHECK(verdict.domain == StationarityDomain::Psd);
  CHECK(verdict.criterion == StationarityCriterion::Newton);
  CHECK(verdict.status == FitCheck::Passed);
}

TEST_CASE("Newton accuracy: a PSD boundary solution is checked on its face") {
  using magmaan::estimate::fit_verdict;
  using magmaan::estimate::FitCheck;
  using magmaan::estimate::StationarityCriterion;
  // A Heywood sample: x1's residual variance is negative under ordinary ML
  // and zero under PSD ML.
  Model m = exact_model("f =~ x1 + x2 + x3", 200);
  Eigen::Matrix3d S;
  S << 1.00, 0.80, 0.70,
       0.80, 1.00, 0.45,
       0.70, 0.45, 1.00;
  m.samp.S = {S};
  auto ordinary = magmaan::estimate::fit_ml(m.pt, m.rep, m.samp, m.theta0);
  REQUIRE(ordinary.has_value());
  CHECK_FALSE(ordinary->diagnostics.admissibility.admissible);
  auto verdict = fit_verdict(*ordinary);
  CHECK(verdict.criterion == StationarityCriterion::Newton);
  CHECK(verdict.status == FitCheck::Passed);

  auto psd = magmaan::estimate::frontier::fit_ml_psd(m.pt, m.rep, m.samp, m.theta0);
  REQUIRE(psd.has_value());
  CHECK(psd->diagnostics.geometric_stationarity.covariance_nullity > 0);
  const auto& na = psd->diagnostics.newton_accuracy;
  CHECK(na.psd_domain);
  CHECK_FALSE(na.covariance_interior);
  CHECK(na.null_directions == 1);
  CHECK(na.constrained_directions == 1);
  CHECK(na.min_multiplier > 0.0);
  REQUIRE(na.status == NewtonAccuracyStatus::Available);
  verdict = fit_verdict(*psd);
  CHECK(verdict.criterion == StationarityCriterion::Newton);
  CHECK(verdict.status == FitCheck::Passed);
}

namespace {

// Two correlated factors whose population correlation is 1.1: the ordinary
// fit reproduces it, the PSD fit has a singular factor covariance matrix.
Model correlated_factors_above_one(std::int64_t n = 400) {
  Model m;
  auto flat = magmaan::parse::Parser::parse("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6");
  REQUIRE(flat.has_value());
  magmaan::spec::BuildOptions opts;
  opts.fixed_x = false;
  auto pt = magmaan::spec::build(*flat, opts);
  REQUIRE(pt.has_value());
  m.pt = std::move(*pt);
  auto rep = magmaan::model::build_matrix_rep(m.pt);
  REQUIRE(rep.has_value());
  m.rep = std::move(*rep);
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L.col(0).head(3) << 1.0, 0.8, 0.7;
  L.col(1).tail(3) << 1.0, 0.9, 0.6;
  Eigen::Matrix2d Phi;
  Phi << 1.0, 1.1, 1.1, 1.0;
  Eigen::MatrixXd Sigma = L * Phi * L.transpose();
  Sigma.diagonal().array() += 0.6;
  m.samp.S = {Sigma};
  m.samp.n_obs = {n};
  auto x0 = magmaan::estimate::simple_start_values(m.pt, m.rep, m.samp, {});
  REQUIRE(x0.has_value());
  m.theta0 = *x0;
  return m;
}

double total_objective(const Model& m, const Eigen::VectorXd& theta) {
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto obj = magmaan::estimate::ml_objective(*ev, m.samp);
  REQUIRE(obj.has_value());
  Eigen::VectorXd g;
  return static_cast<double>(m.samp.n_obs[0]) * obj->f(theta, g);
}

}  // namespace

TEST_CASE("Newton accuracy: the PSD check equals the interior check at an interior point") {
  Model m = exact_model("f =~ x1 + x2 + x3 + x4", 300);
  Eigen::VectorXd theta = m.theta0;
  theta.array() += 1e-3;
  const auto a = newton_accuracy_ml(m.pt, m.rep, m.samp, at(theta));
  const auto b = magmaan::estimate::frontier::newton_accuracy_ml_psd(
      m.pt, m.rep, m.samp, at(theta));
  REQUIRE(a.status == NewtonAccuracyStatus::Available);
  REQUIRE(b.status == NewtonAccuracyStatus::Available);
  CHECK(b.psd_domain);
  CHECK(b.null_directions == 0);
  CHECK(std::abs(a.distance - b.distance) <= 1e-12 * (1.0 + a.distance));
}

TEST_CASE("Newton accuracy: predicted gain matches the objective along a face") {
  Model m = correlated_factors_above_one();
  auto tight = magmaan::estimate::frontier::ml_psd_optim_options();
  tight.max_iter = 20000;
  tight.nlopt.max_eval = 20000;
  tight.nlopt.ftol_rel = 1e-15;
  tight.nlopt.xtol_rel = 1e-13;
  auto psd = magmaan::estimate::frontier::fit_ml_psd(
      m.pt, m.rep, m.samp, m.theta0, magmaan::estimate::Backend::NloptSlsqp, tight);
  REQUIRE(psd.has_value());
  const Eigen::VectorXd th = psd->theta;
  const auto at_fit = magmaan::estimate::frontier::newton_accuracy_ml_psd(
      m.pt, m.rep, m.samp, *psd);
  REQUIRE(at_fit.status == NewtonAccuracyStatus::Available);
  CHECK(at_fit.null_directions == 1);
  CHECK(at_fit.constrained_directions == 1);
  CHECK(at_fit.distance < 1e-3);
  const auto retained = magmaan::estimate::frontier::audit_newton_ml(
      m.pt, m.rep, m.samp, th, magmaan::estimate::StationarityDomain::Psd);
  REQUIRE(retained.diagnostics.status == NewtonAccuracyStatus::Available);
  const auto& geometry = retained.geometry;
  const Eigen::MatrixXd B = geometry.equality_basis * geometry.tangent_basis;
  CHECK(geometry.curvature_correction.norm() > 0.0);
  CHECK(geometry.reduced_hessian.isApprox(
      B.transpose() * (retained.derivatives.hessian + geometry.curvature_correction) * B));
  // Projection onto the face cancels the normal gradient near the optimum.
  CHECK((geometry.reduced_gradient - B.transpose() * retained.derivatives.gradient).norm()
        < 1e-12 * std::max(1.0, B.norm() * retained.derivatives.gradient.norm()));
  CHECK(retained.diagnostics.distance == doctest::Approx(at_fit.distance));
  const auto reassessed = magmaan::estimate::frontier::assess_newton_accuracy(retained);
  CHECK(reassessed.psd_domain);
  CHECK(reassessed.constrained_directions == at_fit.constrained_directions);
  CHECK(reassessed.distance == doctest::Approx(at_fit.distance));


  // Locate the factor covariance parameters and the rank-one factor of Psi.
  Eigen::Index v1 = -1, v2 = -1, c12 = -1;
  for (std::size_t r = 0; r < m.pt.size(); ++r) {
    if (m.pt.free[r] <= 0 || m.pt.op[r] != magmaan::parse::Op::Covariance) continue;
    const auto a = m.pt.lhs_var[r], b = m.pt.rhs_var[r];
    if (m.pt.ov_pos[static_cast<std::size_t>(a)] >= 0) continue;
    const Eigen::Index k = m.pt.free[r] - 1;
    if (a != b) c12 = k;
    else if (v1 < 0) v1 = k;
    else v2 = k;
  }
  REQUIRE(v1 >= 0);
  REQUIRE(v2 >= 0);
  REQUIRE(c12 >= 0);
  Eigen::Matrix2d Psi;
  Psi << th(v1), th(c12), th(c12), th(v2);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Psi);
  REQUIRE(es.eigenvalues()(0) < 1e-7 * es.eigenvalues()(1));
  const Eigen::Vector2d f0 = std::sqrt(es.eigenvalues()(1)) * es.eigenvectors().col(1);

  // A curve inside the rank-one face: Psi(t) = F(t) F(t)', other
  // parameters linear in t. The face curves, so the predicted gain is right
  // only with the face's curvature term.
  const Eigen::Vector2d df(0.3, -0.5);
  Eigen::VectorXd eta = Eigen::VectorXd::LinSpaced(th.size(), 0.5, -0.4);
  eta(v1) = eta(v2) = eta(c12) = 0.0;
  const double f_hat = total_objective(m, th);
  auto ratio = [&](double t) {
    Eigen::VectorXd x = th + t * eta;
    const Eigen::Vector2d F = f0 + t * df;
    x(v1) = F(0) * F(0);
    x(v2) = F(1) * F(1);
    x(c12) = F(0) * F(1);
    const auto a = magmaan::estimate::frontier::newton_accuracy_ml_psd(
        m.pt, m.rep, m.samp, at(x));
    REQUIRE(a.status == NewtonAccuracyStatus::Available);
    CHECK(a.constrained_directions == 1);
    return a.predicted_gain / (total_objective(m, x) - f_hat);
  };
  const double r1 = ratio(2e-3), r2 = ratio(1e-3);
  CAPTURE(r1);
  CAPTURE(r2);
  CHECK(std::abs(r2 - 1.0) < 0.02);
  CHECK(std::abs(r2 - 1.0) < std::abs(r1 - 1.0) + 1e-3);
}

TEST_CASE("Newton accuracy: a fixed zero variance holds its row on the face") {
  Model m = exact_model("f =~ x1 + x2 + x3 + x4\nx4 ~~ 0*x4", 300);
  auto psd = magmaan::estimate::frontier::fit_ml_psd(m.pt, m.rep, m.samp, m.theta0);
  REQUIRE(psd.has_value());
  const auto& na = psd->diagnostics.newton_accuracy;
  CHECK(na.checked);
  CHECK(na.psd_domain);
  CHECK(na.status == NewtonAccuracyStatus::Available);
}

TEST_CASE("Newton artifacts: reuse factorization and reassess retained solution") {
  using namespace magmaan::estimate::frontier;
  Eigen::Matrix2d H;
  H << 4, 1, 1, 2;
  Eigen::Vector2d displacement(0.1, -0.2);
  const auto system = prepare_newton_system(H);
  REQUIRE(system.status == NewtonAccuracyStatus::Available);
  const auto first = solve_newton_system(system, H * displacement);
  REQUIRE(first.status == NewtonAccuracyStatus::Available);
  CHECK((first.step + displacement).norm() < 1e-14);
  CHECK((H * first.step + H * displacement).norm() < 1e-14);
  const auto second = solve_newton_system(system, 2 * H * displacement);
  CHECK((second.step - 2 * first.step).norm() < 1e-14);
  CHECK(second.distance == doctest::Approx(2 * first.distance));
  CHECK_FALSE(assess_newton_accuracy(first).passed);
  NewtonAccuracyOptions loose;
  loose.budget = 1.0;
  CHECK(assess_newton_accuracy(first, loose).passed);
  loose.max_condition = 1.0;
  CHECK(assess_newton_accuracy(first, loose).status == NewtonAccuracyStatus::IllConditioned);
  CHECK(first.status == NewtonAccuracyStatus::Available);
  CHECK(solve_newton_system(system, Eigen::VectorXd::Zero(3)).status ==
        NewtonAccuracyStatus::Unavailable);
  loose.budget = std::numeric_limits<double>::quiet_NaN();
  CHECK(assess_newton_accuracy(first, loose).status == NewtonAccuracyStatus::Unavailable);
  const auto empty = solve_newton_system(prepare_newton_system(Eigen::MatrixXd(0, 0)),
                                         Eigen::VectorXd(0));
  CHECK(assess_newton_accuracy(empty).passed);
}

TEST_CASE("Newton artifacts: full Hessian remains reusable after equality reduction") {
  using namespace magmaan::estimate::frontier;
  const Model m = exact_model("f =~ x1 + a*x2 + a*x3 + x4\nx2 ~~ v*x2\nx4 ~~ v*x4");
  auto con = magmaan::estimate::build_eq_constraints(m.pt);
  REQUIRE(con.has_value());
  const Eigen::VectorXd theta = m.theta0 + con->K() *
      Eigen::VectorXd::Constant(con->K().cols(), 1e-4);
  NewtonAccuracyOptions options;
  options.budget = 0.02;
  const auto audit = audit_newton_ml(m.pt, m.rep, m.samp, theta,
      magmaan::estimate::StationarityDomain::Ambient, options);
  CHECK(assess_newton_accuracy(audit).budget == 0.02);
  REQUIRE(audit.diagnostics.status == NewtonAccuracyStatus::Available);
  const auto& d = audit.derivatives;
  const auto& g = audit.geometry;
  CHECK(d.theta.isApprox(theta));
  CHECK(d.n_obs == 250);
  const auto info = magmaan::inference::information_observed_analytic(m.pt, m.rep, m.samp, at(theta));
  REQUIRE(info.has_value());
  CHECK(d.hessian.isApprox(*info));
  CHECK(g.curvature_correction.isZero());
  CHECK(g.reduced_hessian.isApprox(con->K().transpose() * d.hessian * con->K()));
  const Eigen::VectorXd full_step = g.equality_basis * g.tangent_basis * audit.solution.step;
  CHECK((con->A_eq * full_step).norm() < 1e-12);
  CHECK((g.reduced_hessian * audit.solution.step + g.reduced_gradient).norm() < 1e-10);
  // Downstream inference accepts the retained full information directly.
  const auto covariance = magmaan::inference::vcov(d.hessian, m.pt);
  REQUIRE(covariance.has_value());
  CHECK(covariance->rows() == theta.size());
  const auto psd_geometry = prepare_newton_geometry(m.pt, m.rep, d,
      magmaan::estimate::StationarityDomain::Psd);
  REQUIRE(psd_geometry.status == NewtonAccuracyStatus::Available);
  CHECK(psd_geometry.reduced_hessian.isApprox(g.reduced_hessian));
  CHECK(psd_geometry.reduced_gradient.isApprox(g.reduced_gradient));
  CHECK(newton_accuracy_ml(m.pt, m.rep, m.samp, at(theta)).distance ==
        doctest::Approx(audit.diagnostics.distance));
}

TEST_CASE("Newton artifacts: rejected curvature retains derivatives and geometry") {
  using namespace magmaan::estimate::frontier;
  const Model m = exact_model("f =~ x1 + x2 + x3 + x4");
  auto derivatives = evaluate_newton_ml(m.pt, m.rep, m.samp, m.theta0);
  REQUIRE(derivatives.status == NewtonAccuracyStatus::Available);
  // Inject unsuitable curvature as a caller supplying its own matrix can do.
  derivatives.hessian = -Eigen::MatrixXd::Identity(m.theta0.size(), m.theta0.size());
  const auto geometry = prepare_newton_geometry(m.pt, m.rep, derivatives);
  REQUIRE(geometry.status == NewtonAccuracyStatus::Available);
  const auto system = prepare_newton_system(geometry.reduced_hessian);
  const auto solution = solve_newton_system(system, geometry.reduced_gradient);
  CHECK(assess_newton_accuracy(solution).status == NewtonAccuracyStatus::NonpositiveCurvature);
  CHECK(derivatives.hessian.diagonal().maxCoeff() == -1.0);
  CHECK(geometry.reduced_hessian.rows() > 0);
  CHECK_FALSE(assess_newton_accuracy(solution).passed);
  CHECK(evaluate_newton_ml(m.pt, m.rep, m.samp, Eigen::VectorXd::Zero(1)).status ==
        NewtonAccuracyStatus::Unavailable);
}


TEST_CASE("PSD Newton accuracy: same point is unit invariant with reusable native artifacts") {
  using namespace magmaan::estimate;
  using namespace magmaan::estimate::frontier;
  Model m = correlated_factors_above_one();
  auto options = ml_psd_optim_options();
  options.nlopt.max_eval = 20000;
  options.nlopt.ftol_rel = 1e-15;
  options.nlopt.xtol_rel = 1e-13;
  auto fit = fit_ml_psd(m.pt, m.rep, m.samp, m.theta0, Backend::NloptSlsqp, options);
  REQUIRE(fit.has_value());
  const auto baseline = audit_newton_ml(m.pt, m.rep, m.samp, fit->theta, StationarityDomain::Psd);
  REQUIRE(baseline.diagnostics.passed);
  REQUIRE(baseline.diagnostics.unit_normalized);
  REQUIRE(baseline.diagnostics.null_directions == 1);
  auto units = parameter_units(m.pt, m.rep, m.samp);
  REQUIRE(units.has_value());
  for (int scenario = 0; scenario < 3; ++scenario) {
    CAPTURE(scenario);
    Eigen::VectorXd factors = Eigen::VectorXd::Constant(6, scenario == 0 ? 0.01 : 100.0);
    if (scenario == 2) factors << 0.01, 100.0, 2.0, 100.0, 0.01, 0.5;
    auto sample = m.samp;
    sample.S[0] = factors.asDiagonal() * m.samp.S[0] * factors.asDiagonal();
    auto other_units = parameter_units(m.pt, m.rep, sample);
    REQUIRE(other_units.has_value());
    const Eigen::VectorXd ratio = other_units->cwiseQuotient(*units);
    const Eigen::VectorXd theta = fit->theta.cwiseProduct(ratio);
    const auto audit = audit_newton_ml(m.pt, m.rep, sample, theta, StationarityDomain::Psd);
    REQUIRE(audit.diagnostics.passed);
    CHECK(audit.diagnostics.unit_normalized);
    CHECK(audit.diagnostics.null_directions == baseline.diagnostics.null_directions);
    CHECK(audit.diagnostics.constrained_directions == baseline.diagnostics.constrained_directions);
    CHECK(audit.diagnostics.condition == doctest::Approx(baseline.diagnostics.condition).epsilon(1e-7));
    CHECK(std::abs(audit.diagnostics.distance - baseline.diagnostics.distance) < 1e-7);
    CHECK(audit.derivatives.theta.isApprox(theta, 1e-14));
    const auto native = evaluate_newton_ml(m.pt, m.rep, sample, theta);
    REQUIRE(native.status == NewtonAccuracyStatus::Available);
    CHECK(audit.derivatives.gradient.isApprox(native.gradient, 1e-7));
    CHECK(audit.derivatives.hessian.isApprox(native.hessian, 1e-8));
    const Eigen::MatrixXd B = audit.geometry.equality_basis * audit.geometry.tangent_basis;
    CHECK(audit.geometry.reduced_gradient.isApprox(B.transpose() * audit.derivatives.gradient, 1e-7));
    CHECK(audit.geometry.reduced_hessian.isApprox(
        B.transpose() * (audit.derivatives.hessian + audit.geometry.curvature_correction) * B, 1e-8));
    CHECK(assess_newton_accuracy(audit).unit_normalized);
    auto strict = audit.options;
    strict.max_condition = 1.0;
    CHECK_FALSE(assess_newton_accuracy(audit, strict).passed);
  }
}

TEST_CASE("PSD Newton normalization preserves rejection and its supported scope") {
  using namespace magmaan::estimate;
  using namespace magmaan::estimate::frontier;
  SUBCASE("std.lv interior and infeasible variance") {
    auto m = exact_model("f =~ NA*x1 + x2 + x3 + x4\nf ~~ 1*f");
    const auto good = audit_newton_ml(m.pt, m.rep, m.samp, m.theta0, StationarityDomain::Psd);
    CHECK(good.diagnostics.unit_normalized);
    CHECK(good.diagnostics.passed);
    auto theta = m.theta0;
    for (std::size_t i = 0; i < m.pt.size(); ++i) {
      const auto& c = m.rep.cell_for_row[i];
      if (c.used && c.mat == magmaan::model::MatId::Theta && c.row == 0 && c.col == 0 && m.pt.free[i] > 0)
        theta(m.pt.free[i] - 1) = -0.01;
    }
    const auto bad = audit_newton_ml(m.pt, m.rep, m.samp, theta, StationarityDomain::Psd);
    CHECK(bad.diagnostics.unit_normalized);
    CHECK_FALSE(bad.diagnostics.passed);
    auto sample = m.samp;
    sample.S[0](0, 0) = 0.0;
    CHECK_FALSE(audit_newton_ml(m.pt, m.rep, sample, m.theta0, StationarityDomain::Psd).diagnostics.passed);
  }
  SUBCASE("equality constraints and ambient audit retain their existing path") {
    auto m = exact_model("f =~ x1 + a*x2 + a*x3 + x4");
    const auto psd = audit_newton_ml(m.pt, m.rep, m.samp, m.theta0, StationarityDomain::Psd);
    CHECK_FALSE(psd.diagnostics.unit_normalized);
    CHECK(psd.diagnostics.passed);
    const auto ambient = audit_newton_ml(m.pt, m.rep, m.samp, m.theta0);
    CHECK_FALSE(ambient.diagnostics.unit_normalized);
    CHECK(ambient.diagnostics.passed);
  }
}
