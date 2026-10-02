#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/frontier/gauge.hpp"
#include "magmaan/estimate/frontier/sphere.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/parameter_map.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#define REQUIRE_OK(value)                                                     \
  do {                                                                        \
    INFO("error: " << ((value).has_value() ? "" : (value).error().detail));   \
    REQUIRE((value).has_value());                                             \
    if (!(value).has_value()) return;                                         \
  } while (false)

namespace {

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::estimate::frontier::analyze_gauge;
using magmaan::estimate::frontier::GaugeKind;
using magmaan::estimate::frontier::GaugePlan;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::GroupEqual;
using magmaan::spec::LatentStructure;

namespace fr = magmaan::estimate::frontier;

LatentStructure lavaanify(std::string_view syntax, BuildOptions options = {}) {
  auto flat = Parser::parse(syntax);
  REQUIRE(flat.has_value());
  auto pt = magmaan::spec::build(*flat, options);
  REQUIRE_MESSAGE(pt.has_value(), "lavaanify failed: "
      << (pt.has_value() ? std::string{} : pt.error().detail));
  return std::move(*pt);
}

GaugePlan plan_of(const LatentStructure& pt) {
  auto plan = analyze_gauge(pt);
  REQUIRE_MESSAGE(plan.has_value(), (plan.has_value() ? "" : plan.error().detail));
  return *plan;
}

bool reason_contains(const GaugePlan& plan, std::string_view needle) {
  for (const auto& p : plan.passthrough)
    if (p.reason.find(needle) != std::string::npos) return true;
  return false;
}

// Deterministic symmetric perturbation, small enough to keep S positive
// definite, so the fits are not exact.
Eigen::MatrixXd perturb(Eigen::MatrixXd S, double eps) {
  const Eigen::Index p = S.rows();
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j <= i; ++j) {
      const double e = eps * std::sin(1.7 * static_cast<double>(i + 1) +
                                      2.3 * static_cast<double>(j + 1));
      S(i, j) += e;
      if (i != j) S(j, i) += e;
    }
  return S;
}

// One factor, four indicators.
Eigen::MatrixXd one_factor_sigma() {
  Eigen::Vector4d lam(1.0, 0.8, 0.6, 0.7);
  Eigen::Vector4d theta(0.5, 0.6, 0.7, 0.4);
  Eigen::MatrixXd S = 1.2 * lam * lam.transpose();
  S.diagonal() += theta;
  return perturb(S, 0.03);
}

// Two factors, X -> Y, three indicators each (the Ernst et al. design).
Eigen::MatrixXd two_factor_sigma() {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L.col(0).head(3) << 1.0, 0.8, 0.6;
  L.col(1).tail(3) << 1.0, 0.8, 0.6;
  Eigen::Matrix2d Phi;
  const double beta = 0.4;
  Phi << 1.0, beta, beta, beta * beta + 1.0;
  Eigen::MatrixXd S = L * Phi * L.transpose();
  S.diagonal().array() += 1.0;
  return perturb(S, 0.04);
}

SampleStats stats(std::vector<Eigen::MatrixXd> S, std::int64_t n = 300) {
  SampleStats s;
  s.S = std::move(S);
  s.n_obs.assign(s.S.size(), n);
  return s;
}

struct Fitted {
  LatentStructure pt;
  MatrixRep rep;
  SampleStats samp;
  Eigen::VectorXd x0;
};

Fitted setup(std::string_view syntax, std::vector<Eigen::MatrixXd> S,
             BuildOptions opts = {}) {
  opts.fixed_x = false;
  Fitted f{lavaanify(syntax, opts), {}, stats(std::move(S)), {}};
  auto rep = build_matrix_rep(f.pt);
  REQUIRE(rep.has_value());
  f.rep = std::move(*rep);
  auto x0 = magmaan::estimate::simple_start_values(f.pt, f.rep, f.samp, {});
  REQUIRE(x0.has_value());
  f.x0 = std::move(*x0);
  return f;
}

Eigen::MatrixXd implied(const LatentStructure& pt, const MatrixRep& rep,
                        const Eigen::VectorXd& theta, std::size_t block = 0) {
  auto ev = ModelEvaluator::build(pt, rep);
  REQUIRE(ev.has_value());
  auto e = ev->evaluate(theta, false, false);
  REQUIRE(e.has_value());
  return e->moments.sigma[block];
}

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  REQUIRE(a.size() == b.size());
  return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace

// ---------------------------------------------------------------------------
// Gauge analysis
// ---------------------------------------------------------------------------

TEST_CASE("gauge: a marker CFA is one affine unit with l = first loading") {
  const auto pt = lavaanify("f =~ x1 + x2 + x3 + x4");
  const auto plan = plan_of(pt);
  REQUIRE(plan.units.size() == 1);
  const auto& u = plan.units[0];
  CHECK(u.kind == GaugeKind::Affine);
  CHECK(u.basis.cols() == 4);
  REQUIRE(u.level.size() == 4);
  CHECK(u.level(0) == doctest::Approx(1.0));
  CHECK(u.level.tail(3).cwiseAbs().maxCoeff() < 1e-12);
  CHECK(plan.passthrough.empty());
}

TEST_CASE("gauge: an explicit 1*x1 marker reads the same as the automatic one") {
  const auto a = plan_of(lavaanify("f =~ x1 + x2 + x3"));
  const auto b = plan_of(lavaanify("f =~ 1*x1 + x2 + x3"));
  REQUIRE(a.units.size() == 1);
  REQUIRE(b.units.size() == 1);
  CHECK((a.units[0].level - b.units[0].level).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("gauge: std.lv is one linear unit that releases the variance") {
  BuildOptions o;
  o.std_lv = true;
  const auto pt = lavaanify("f =~ x1 + x2 + x3 + x4", o);
  const auto plan = plan_of(pt);
  REQUIRE(plan.units.size() == 1);
  const auto& u = plan.units[0];
  CHECK(u.kind == GaugeKind::Linear);
  CHECK(u.basis.cols() == 4);
  REQUIRE(u.variance_row >= 0);
  CHECK(u.variance_value == doctest::Approx(1.0));
}

TEST_CASE("gauge: effect coding is affine with l = mean loading") {
  BuildOptions o;
  o.effect_coding = true;
  const auto pt = lavaanify("f =~ x1 + x2 + x3 + x4", o);
  const auto plan = plan_of(pt);
  REQUIRE(plan.units.size() == 1);
  const auto& u = plan.units[0];
  CHECK(u.kind == GaugeKind::Affine);
  CHECK(u.basis.cols() == 4);
  for (Eigen::Index j = 0; j < 4; ++j) CHECK(u.level(j) == doctest::Approx(0.25));
}

TEST_CASE("gauge: a fixed loading ratio stays as a linear restriction") {
  const auto pt = lavaanify("f =~ x1 + 0.5*x2 + x3 + x4");
  const auto plan = plan_of(pt);
  REQUIRE(plan.units.size() == 1);
  const auto& W = plan.units[0].basis;
  CHECK(W.cols() == 3);
  Eigen::Vector4d in(1.0, 0.5, 0.3, -0.2);
  Eigen::Vector4d out(1.0, 0.0, 0.0, 0.0);
  CHECK((in - W * (W.transpose() * in)).norm() < 1e-10);
  CHECK((out - W * (W.transpose() * out)).norm() > 0.1);
}

TEST_CASE("gauge: tau-equivalent loadings fix the direction and pass through") {
  const auto plan = plan_of(lavaanify("f =~ 1*x1 + 1*x2 + 1*x3 + 1*x4"));
  CHECK(plan.units.empty());
  CHECK(reason_contains(plan, "direction is fixed"));
}

TEST_CASE("gauge: a cross-factor loading equality demotes both latents") {
  const auto plan = plan_of(lavaanify(
      "f1 =~ x1 + a*x2 + x3\n f2 =~ x4 + a*x5 + x6"));
  CHECK(plan.units.empty());
  CHECK(plan.passthrough.size() == 2);
  CHECK(reason_contains(plan, "across latents"));
}

TEST_CASE("gauge: std.lv with a user-fixed loading fixes the scale twice") {
  BuildOptions o;
  o.std_lv = true;
  const auto plan = plan_of(lavaanify("f =~ 0.5*x1 + x2 + x3 + x4", o));
  CHECK(plan.units.empty());
  CHECK(reason_contains(plan, "fixed nonzero value"));
}

TEST_CASE("gauge: higher-order factor and its marker factor pass through") {
  // g's scale is set by its marker loading on f1, so f1 cannot move either.
  // f2's scale is set by its own marker only, so it stays a unit.
  const auto pt = lavaanify(
      "f1 =~ x1 + x2 + x3\n f2 =~ x4 + x5 + x6\n g =~ f1 + f2");
  const auto plan = plan_of(pt);
  REQUIRE(plan.units.size() == 1);
  CHECK(plan.passthrough.size() == 2);
  CHECK(reason_contains(plan, "higher-order"));
  CHECK(reason_contains(plan, "fixed nonzero value"));
}

TEST_CASE("gauge: a structural regression keeps both latents as units") {
  const auto plan = plan_of(lavaanify(
      "X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X"));
  CHECK(plan.units.size() == 2);
  CHECK(plan.passthrough.empty());
}

TEST_CASE("gauge: a nonlinear loading constraint passes through") {
  const auto plan = plan_of(lavaanify(
      "f =~ x1 + a*x2 + b*x3 + x4\n a == b^2"));
  CHECK(plan.units.empty());
  CHECK(reason_contains(plan, "nonlinear"));
}

TEST_CASE("gauge: multi-group configural, metric and partial invariance") {
  const std::string_view m = "f =~ x1 + x2 + x3 + x4";
  BuildOptions o;
  o.n_groups = 2;
  {
    const auto plan = plan_of(lavaanify(m, o));
    CHECK(plan.units.size() == 2);
  }
  o.group_equal = {GroupEqual::Loadings};
  {
    const auto plan = plan_of(lavaanify(m, o));
    REQUIRE(plan.units.size() == 1);
    CHECK(plan.units[0].blocks.size() == 2);
    CHECK(plan.units[0].kind == GaugeKind::Affine);
  }
  o.group_partial = {"f =~ x3"};
  {
    const auto plan = plan_of(lavaanify(m, o));
    CHECK(plan.units.empty());
    CHECK(reason_contains(plan, "partial invariance"));
  }
}

TEST_CASE("gauge: std.lv metric invariance is one tied linear unit") {
  BuildOptions o;
  o.n_groups = 2;
  o.std_lv = true;
  o.group_equal = {GroupEqual::Loadings};
  const auto plan = plan_of(lavaanify("f =~ x1 + x2 + x3 + x4", o));
  REQUIRE(plan.units.size() == 1);
  CHECK(plan.units[0].kind == GaugeKind::Linear);
  CHECK(plan.units[0].blocks.size() == 2);
}

// ---------------------------------------------------------------------------
// Chart translation
// ---------------------------------------------------------------------------

TEST_CASE("reidentify: marker x1 -> std.lv -> marker x3 -> marker x1") {
  auto f = setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()});
  auto est = magmaan::estimate::fit_ml(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(est);
  const Eigen::MatrixXd sigma = implied(f.pt, f.rep, est->theta);

  BuildOptions std_opts;
  std_opts.std_lv = true;
  std_opts.fixed_x = false;
  const auto pt_std = lavaanify("f =~ x1 + x2 + x3 + x4", std_opts);
  auto to_std = fr::reidentify(f.pt, est->theta, pt_std);
  REQUIRE_OK(to_std);
  CHECK((implied(pt_std, f.rep, to_std->theta) - sigma).cwiseAbs().maxCoeff() < 1e-10);

  BuildOptions nf;
  nf.fixed_x = false;
  const auto pt_x3 = lavaanify("f =~ NA*x1 + x2 + 1*x3 + x4", nf);
  auto to_x3 = fr::reidentify(pt_std, to_std->theta, pt_x3);
  REQUIRE_OK(to_x3);
  CHECK((implied(pt_x3, f.rep, to_x3->theta) - sigma).cwiseAbs().maxCoeff() < 1e-10);

  auto back = fr::reidentify(pt_x3, to_x3->theta, f.pt);
  REQUIRE_OK(back);
  CHECK(max_abs_diff(back->theta, est->theta) < 1e-10);
}

TEST_CASE("reidentify: refuses a partable that describes another model") {
  auto f = setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()});
  auto est = magmaan::estimate::fit_ml(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(est);
  BuildOptions nf;
  nf.fixed_x = false;
  const auto restricted = lavaanify("f =~ x1 + 0.5*x2 + x3 + x4", nf);
  CHECK_FALSE(fr::reidentify(f.pt, est->theta, restricted).has_value());
}

// ---------------------------------------------------------------------------
// Parameter map and sphere objective derivatives
// ---------------------------------------------------------------------------

TEST_CASE("parameter map: gradient pulls back through a nonlinear map") {
  magmaan::optim::ScalarProblem base;
  base.n_param = 2;
  base.f = [](const Eigen::VectorXd& t, Eigen::VectorXd& g) {
    g.resize(2);
    g << 2.0 * (t(0) - 1.0), 4.0 * (t(1) + 0.5);
    return (t(0) - 1.0) * (t(0) - 1.0) + 2.0 * (t(1) + 0.5) * (t(1) + 0.5);
  };
  base.expand = [](const Eigen::VectorXd& t) { return t; };
  magmaan::optim::ParameterMap map;
  map.n_param = 2;
  map.expand = [](const Eigen::VectorXd& u) {
    Eigen::VectorXd t(2);
    t << u(0) * u(1), std::sin(u(0));
    return t;
  };
  map.jacobian = [](const Eigen::VectorXd& u) {
    Eigen::MatrixXd J(2, 2);
    J << u(1), u(0), std::cos(u(0)), 0.0;
    return J;
  };
  map.penalty_residual = [](const Eigen::VectorXd& u) {
    Eigen::VectorXd r(1);
    r << u.squaredNorm() - 1.0;
    return r;
  };
  map.penalty_jacobian = [](const Eigen::VectorXd& u) {
    Eigen::MatrixXd J(1, 2);
    J << 2.0 * u(0), 2.0 * u(1);
    return J;
  };
  const auto prob = magmaan::optim::reparameterize(base, map);
  Eigen::VectorXd u(2);
  u << 0.7, -0.4;
  Eigen::VectorXd g;
  prob.f(u, g);
  for (Eigen::Index k = 0; k < 2; ++k) {
    Eigen::VectorXd up = u, dn = u, scratch;
    const double h = 1e-6;
    up(k) += h;
    dn(k) -= h;
    const double fd = (prob.f(up, scratch) - prob.f(dn, scratch)) / (2 * h);
    CHECK(g(k) == doctest::Approx(fd).epsilon(1e-7));
  }
}

TEST_CASE("sphere: ML objective gradient matches finite differences") {
  auto f = setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                 {two_factor_sigma()});
  auto sp = fr::ml_sphere_problem(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sp);
  Eigen::VectorXd u = sp->start;
  for (Eigen::Index k = 0; k < u.size(); ++k)
    u(k) += 0.03 * std::sin(1.3 * static_cast<double>(k + 1));
  Eigen::VectorXd g;
  sp->problem.f(u, g);
  REQUIRE(g.size() == u.size());
  for (Eigen::Index k = 0; k < u.size(); ++k) {
    Eigen::VectorXd up = u, dn = u, scratch;
    const double h = 1e-6;
    up(k) += h;
    dn(k) -= h;
    const double fd =
        (sp->problem.f(up, scratch) - sp->problem.f(dn, scratch)) / (2 * h);
    CHECK(g(k) == doctest::Approx(fd).epsilon(1e-5).scale(1.0));
  }
}

// ---------------------------------------------------------------------------
// Sphere fits reproduce the ordinary estimate whenever it exists
// ---------------------------------------------------------------------------

namespace {

void check_same_as_ordinary_ml(const Fitted& f, double tol = 1e-5) {
  auto ord = magmaan::estimate::fit_ml(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(ord);
  auto sph = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  REQUIRE(sph->user_chart);
  CHECK(sph->report.pin_residual < 1e-6);
  CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-8));
  CHECK(max_abs_diff(sph->estimates.theta, ord->theta) < tol);
  CHECK(sph->report.residual.fixed_rows < 1e-10);
  CHECK(sph->report.residual.linear_constraints < 1e-10);
  INFO(sph->report.native_audit.detail);
  CHECK(sph->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
  CHECK(sph->report.native_audit.computations.geometry.tangent_basis.cols() ==
        sph->report.driven.size() - static_cast<Eigen::Index>(sph->report.plan.units.size()));
}

}  // namespace

TEST_CASE("sphere ML: one-factor marker CFA") {
  check_same_as_ordinary_ml(setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()}));
}

TEST_CASE("sphere audit: tangent curvature matches retracted objective differences away from stationarity") {
  auto f = setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X", {two_factor_sigma()});
  auto problem = fr::ml_sphere_problem(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(problem);
  Eigen::VectorXd point = problem->start;
  for (Eigen::Index k = 0; k < point.size(); ++k) point(k) += 0.04 * std::sin(static_cast<double>(k + 1));
  auto audit = fr::audit_ml_sphere(f.pt, f.rep, f.samp, point);
  REQUIRE_OK(audit);
  const auto& a = audit->computations;
  REQUIRE(a.derivatives.status == fr::NewtonAccuracyStatus::Available);
  const auto& basis = a.geometry.tangent_basis;
  CHECK((basis.transpose() * basis - Eigen::MatrixXd::Identity(basis.cols(), basis.cols())).cwiseAbs().maxCoeff() < 1e-12);
  Eigen::VectorXd direction(basis.cols());
  for (Eigen::Index k = 0; k < direction.size(); ++k) direction(k) = std::cos(0.7 * static_cast<double>(k + 1));
  direction.normalize();
  auto value = [&](double step) {
    Eigen::VectorXd u = a.derivatives.theta + step * basis * direction;
    Eigen::Index offset = u.size();
    for (const auto& unit : problem->plan.units) offset -= unit.basis.cols();
    for (const auto& unit : problem->plan.units) {
      u.segment(offset, unit.basis.cols()).normalize();
      offset += unit.basis.cols();
    }
    Eigen::VectorXd scratch;
    return a.derivatives.n_obs * problem->problem.f(u, scratch);
  };
  const double h = 1e-4;
  const double first = (value(h) - value(-h)) / (2 * h);
  const double second = (value(h) + value(-h) - 2 * value(0)) / (h * h);
  CHECK(first == doctest::Approx(direction.dot(a.geometry.reduced_gradient)).epsilon(1e-5).scale(1));
  CHECK(second == doctest::Approx(direction.dot(a.geometry.reduced_hessian * direction)).epsilon(1e-4).scale(1));
}

TEST_CASE("sphere audit: radial pin and radius do not supply model curvature") {
  auto f = setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()});
  fr::SphereOptions options;
  options.polish = false;
  auto fit = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0, {}, Backend::NloptLbfgs,
                              magmaan::estimate::ml_optim_options(), options);
  REQUIRE_OK(fit);
  const auto& original = fit->report.native_audit.computations;
  Eigen::VectorXd radial = fit->report.driven;
  const Eigen::Index dim = fit->report.plan.units.front().basis.cols();
  radial.tail(dim) *= 7.0;
  options.pin_weight = 100.0;
  auto repeated = fr::audit_ml_sphere(f.pt, f.rep, f.samp, radial, {}, options,
                                     fit->report.fmin_internal);
  REQUIRE_OK(repeated);
  CHECK(fr::assess_convergence(*repeated).status == magmaan::estimate::FitCheck::Passed);
  CHECK(repeated->computations.solution.distance == doctest::Approx(original.solution.distance).epsilon(1e-6).scale(1e-8));
  CHECK((repeated->computations.geometry.reduced_hessian -
         original.geometry.reduced_hessian).cwiseAbs().maxCoeff() < 1e-9);
  auto bad = fr::audit_ml_sphere(f.pt, f.rep, f.samp, radial, {}, options,
                                fit->report.fmin_internal + 1.0);
  REQUIRE_OK(bad);
  auto policy = fr::newton_convergence_policy();
  policy.require_objective_consistency = true;
  CHECK(fr::assess_convergence(*bad, policy).objective_consistency.status == magmaan::estimate::FitCheck::Failed);
  CHECK(fr::assess_convergence(*bad, policy).status == magmaan::estimate::FitCheck::Failed);
  radial.tail(dim).setZero();
  auto zero = fr::audit_ml_sphere(f.pt, f.rep, f.samp, radial);
  REQUIRE_OK(zero);
  CHECK(fr::assess_convergence(*zero).status == magmaan::estimate::FitCheck::Failed);
}

TEST_CASE("sphere audit: a stationary zero-variance ridge fails tangent curvature") {
  auto f = setup("f =~ x1 + x2 + x3 + x4", {Eigen::Matrix4d::Identity()});
  for (std::size_t row = 0; row < f.pt.size(); ++row) {
    if (f.pt.free[row] > 0 && f.pt.op[row] == magmaan::parse::Op::Covariance) {
      const auto var = static_cast<std::size_t>(f.pt.lhs_var[row]);
      f.x0(f.pt.free[row] - 1) = f.pt.ov_pos[var] < 0 ? 0.0 : 1.0;
    }
  }
  fr::SphereOptions options;
  options.start = fr::SphereStart::User;
  auto problem = fr::ml_sphere_problem(f.pt, f.rep, f.samp, f.x0, options);
  REQUIRE_OK(problem);
  auto audit = fr::audit_ml_sphere(f.pt, f.rep, f.samp, problem->start);
  REQUIRE_OK(audit);
  CHECK(audit->computations.geometry.reduced_gradient.norm() < 1e-10);
  CHECK(audit->computations.solution.status == fr::NewtonAccuracyStatus::NonpositiveCurvature);
  CHECK(fr::assess_convergence(*audit).status == magmaan::estimate::FitCheck::Failed);
}

TEST_CASE("sphere audit: a collapsed loading direction can be a stationary saddle") {
  Eigen::Matrix4d covariance = Eigen::Matrix4d::Identity();
  covariance(1, 2) = covariance(2, 1) = 0.8;
  auto f = setup("f =~ x1 + x2 + x3 + x4", {covariance});
  for (std::size_t row = 0; row < f.pt.size(); ++row) {
    if (f.pt.free[row] <= 0) continue;
    if (f.pt.op[row] == magmaan::parse::Op::Measurement) f.x0(f.pt.free[row] - 1) = 0.0;
    if (f.pt.op[row] == magmaan::parse::Op::Covariance) {
      const auto pos = f.pt.ov_pos[static_cast<std::size_t>(f.pt.lhs_var[row])];
      f.x0(f.pt.free[row] - 1) = pos < 0 ? 0.1 : (pos == 0 ? 0.9 : 1.0);
    }
  }
  fr::SphereOptions options;
  options.start = fr::SphereStart::User;
  auto problem = fr::ml_sphere_problem(f.pt, f.rep, f.samp, f.x0, options);
  REQUIRE_OK(problem);
  auto audit = fr::audit_ml_sphere(f.pt, f.rep, f.samp, problem->start);
  REQUIRE_OK(audit);
  const auto& a = audit->computations;
  CHECK(a.geometry.reduced_gradient.norm() < 1e-10);
  // Check the nominated negative-curvature direction against actual objective
  // values on the sphere, rather than trusting the factorization status alone.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(a.geometry.reduced_hessian);
  REQUIRE(eig.info() == Eigen::Success);
  REQUIRE(eig.eigenvalues()(0) < 0);
  const Eigen::VectorXd direction = a.geometry.tangent_basis * eig.eigenvectors().col(0);
  Eigen::VectorXd scratch;
  double best = problem->problem.f(problem->start, scratch);
  for (double delta : {-0.001, 0.001}) {
    Eigen::VectorXd step = a.derivatives.theta + delta * direction;
    step.tail(4).normalize();
    best = std::min(best, problem->problem.f(step, scratch));
  }
  CHECK(best < problem->problem.f(problem->start, scratch));
  CHECK(a.solution.status == fr::NewtonAccuracyStatus::NonpositiveCurvature);
  CHECK(fr::assess_convergence(*audit).status == magmaan::estimate::FitCheck::Failed);
}

TEST_CASE("sphere audit: active bounds and nonlinear equalities remain explicitly unchecked") {
  SUBCASE("active box") {
    auto f = setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()});
    magmaan::estimate::Bounds bounds;
    bounds.lower = Eigen::VectorXd::Constant(f.pt.n_free(), -std::numeric_limits<double>::infinity());
    bounds.upper = -bounds.lower;
    for (std::size_t row = 0; row < f.pt.size(); ++row) {
      if (f.pt.free[row] > 0 && f.pt.op[row] == magmaan::parse::Op::Covariance &&
          f.pt.ov_pos[static_cast<std::size_t>(f.pt.lhs_var[row])] >= 0) {
        const auto index = f.pt.free[row] - 1;
        bounds.lower(index) = bounds.upper(index) = f.x0(index);
        break;
      }
    }
    fr::SphereOptions options;
    options.start = fr::SphereStart::User;
    auto problem = fr::ml_sphere_problem(f.pt, f.rep, f.samp, f.x0, options);
    REQUIRE_OK(problem);
    auto audit = fr::audit_ml_sphere(f.pt, f.rep, f.samp, problem->start, bounds);
    REQUIRE_OK(audit);
    CHECK(fr::assess_convergence(*audit).status == magmaan::estimate::FitCheck::Unchecked);
    CHECK(audit->detail.find("active box") != std::string::npos);
  }
  SUBCASE("nonlinear equality") {
    auto f = setup("f =~ x1 + x2 + x3 + x4\n x1 ~~ a*x1\n x2 ~~ b*x2\n a == b*b", {one_factor_sigma()});
    // Feasibility is established separately from curvature support.
    for (std::size_t row = 0; row < f.pt.size(); ++row) {
      if (f.pt.free[row] > 0 && f.pt.op[row] == magmaan::parse::Op::Covariance &&
          f.pt.ov_pos[static_cast<std::size_t>(f.pt.lhs_var[row])] >= 0)
        f.x0(f.pt.free[row] - 1) = 1.0;
    }
    fr::SphereOptions options;
    options.start = fr::SphereStart::User;
    auto problem = fr::ml_sphere_problem(f.pt, f.rep, f.samp, f.x0, options);
    REQUIRE_OK(problem);
    auto audit = fr::audit_ml_sphere(f.pt, f.rep, f.samp, problem->start);
    REQUIRE_OK(audit);
    CHECK(fr::assess_convergence(*audit).status == magmaan::estimate::FitCheck::Unchecked);
    CHECK(audit->detail.find("nonlinear equality") != std::string::npos);
  }
}

TEST_CASE("sphere ML: one-factor with a mean structure") {
  BuildOptions o;
  o.meanstructure = true;
  auto f = setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()}, o);
  f.samp.mean = {Eigen::Vector4d(0.1, -0.2, 0.3, 0.05)};
  auto x0 = magmaan::estimate::simple_start_values(f.pt, f.rep, f.samp, {});
  REQUIRE(x0.has_value());
  f.x0 = *x0;
  check_same_as_ordinary_ml(f);
}

TEST_CASE("sphere ML: two-factor SEM (Ernst design)") {
  check_same_as_ordinary_ml(setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                                  {two_factor_sigma()}));
}

TEST_CASE("sphere ML: std.lv user chart") {
  BuildOptions o;
  o.std_lv = true;
  check_same_as_ordinary_ml(setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                                  {two_factor_sigma()}, o));
}

TEST_CASE("sphere ML: effect coding") {
  BuildOptions o;
  o.effect_coding = true;
  check_same_as_ordinary_ml(setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()}, o));
}

TEST_CASE("sphere audit: regular ML acceptance survives heterogeneous indicator units") {
  const auto covariance = one_factor_sigma();
  Eigen::Vector4d scale(0.01, 100.0, 2.0, 0.3);
  auto original = setup("f =~ x1 + x2 + x3 + x4", {covariance});
  auto changed = setup("f =~ x1 + x2 + x3 + x4",
                       {scale.asDiagonal() * covariance * scale.asDiagonal()});
  auto first = fr::fit_ml_sphere(original.pt, original.rep, original.samp, original.x0);
  auto second = fr::fit_ml_sphere(changed.pt, changed.rep, changed.samp, changed.x0);
  REQUIRE_OK(first);
  REQUIRE_OK(second);
  CHECK(first->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
  CHECK(second->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
  CHECK(second->report.fmin_internal == doctest::Approx(first->report.fmin_internal).epsilon(1e-9));
}

TEST_CASE("sphere ML and GLS: sample-unit scaling uses the internal latent chart") {
  // Frozen regular N=100 development covariance. With mixed indicator units,
  // the former released-marker scale stops far from the optimum (or fails
  // its line search). Both targets are invariant to this change of units.
  Eigen::Matrix<double, 6, 6> covariance;
  covariance <<
    2.118383711263836, .9696068218577427, .4291974363012256, .3123623074203148, .3302763415688515, .2522176722272337,
    .9696068218577427, 1.923872404396635, .6025127712649045, .008771535716100964, .2645935837282908, .0582359508142738,
    .4291974363012256, .6025127712649045, 1.296549585945821, .007511998082285165, .1843541227085678, -.1183772846292678,
    .3123623074203148, .008771535716100964, .007511998082285165, 2.140010405988564, .8881556521407293, .9195580160775313,
    .3302763415688515, .2645935837282908, .1843541227085678, .8881556521407293, 1.671330543307811, .3823610873140895,
    .2522176722272337, .0582359508142738, -.1183772846292678, .9195580160775313, .3823610873140895, 1.220635536650093;
  Eigen::Matrix<double, 6, 1> scale;
  scale << .01, 100, 2, .3, 10, .1;
  constexpr const char* syntax = "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nY ~ X";
  std::vector<Backend> backends{Backend::NloptLbfgs};
#ifdef MAGMAAN_WITH_PORT
  backends.push_back(Backend::Port);
#endif
  for (const auto backend : backends) {
    for (const bool gls : {false, true}) {
      CAPTURE(static_cast<int>(backend));
      CAPTURE(gls);
      auto original = setup(syntax, {covariance});
      auto changed = setup(syntax, {scale.asDiagonal() * covariance * scale.asDiagonal()});
      original.samp.n_obs[0] = changed.samp.n_obs[0] = 100;
      auto options = magmaan::estimate::ml_optim_options();
      options.coordinate_scaling = magmaan::optim::CoordinateScaling::SampleUnits;
      fr::SphereOptions sphere;
      sphere.polish = false;
      sphere.start = fr::SphereStart::User;
      auto fit = [&](Fitted& f) {
        auto start = magmaan::estimate::ml_start_values(f.pt, f.rep, f.samp);
        REQUIRE(start.has_value());
        return gls ? fr::fit_gls_sphere(f.pt, f.rep, f.samp, start->theta, {}, backend, options, sphere)
                   : fr::fit_ml_sphere(f.pt, f.rep, f.samp, start->theta, {}, backend, options, sphere);
      };
      const auto first = fit(original);
      const auto second = fit(changed);
      REQUIRE_OK(first);
      REQUIRE_OK(second);
      CHECK(first->report.driven_scaled);
      CHECK(second->report.driven_scaled);
      CHECK(first->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
      CHECK(second->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
      CHECK(second->report.fmin_internal == doctest::Approx(first->report.fmin_internal).epsilon(1e-8));
      const Eigen::MatrixXd restored = scale.cwiseInverse().asDiagonal() *
          implied(changed.pt, changed.rep, second->estimates.theta) * scale.cwiseInverse().asDiagonal();
      CHECK((restored - implied(original.pt, original.rep, first->estimates.theta)).cwiseAbs().maxCoeff() < 1e-5);
    }
  }
}

TEST_CASE("sphere ML: fixed loading ratio") {
  check_same_as_ordinary_ml(setup("f =~ x1 + 0.9*x2 + x3 + x4", {one_factor_sigma()}));
}

TEST_CASE("sphere ML: multi-group metric invariance") {
  BuildOptions o;
  o.n_groups = 2;
  o.group_equal = {GroupEqual::Loadings};
  Eigen::MatrixXd S2 = one_factor_sigma();
  S2 *= 1.3;
  S2 = perturb(S2, 0.02);
  check_same_as_ordinary_ml(setup("f =~ x1 + x2 + x3 + x4",
                                  {one_factor_sigma(), S2}, o));
}

TEST_CASE("sphere ML: std.lv multi-group metric invariance") {
  BuildOptions o;
  o.n_groups = 2;
  o.std_lv = true;
  o.group_equal = {GroupEqual::Loadings};
  Eigen::MatrixXd S2 = perturb(1.3 * one_factor_sigma(), 0.02);
  check_same_as_ordinary_ml(setup("f =~ x1 + x2 + x3 + x4",
                                  {one_factor_sigma(), S2}, o));
}

TEST_CASE("sphere LS: ULS and GLS reproduce the ordinary estimates") {
  auto f = setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                 {two_factor_sigma()});
  {
    auto ord = magmaan::estimate::fit_gmm(f.pt, f.rep, f.samp, f.x0);
    REQUIRE_OK(ord);
    auto sph = fr::fit_gmm_sphere(f.pt, f.rep, f.samp, f.x0);
    REQUIRE_OK(sph);
    REQUIRE(sph->user_chart);
    CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-8));
    CHECK(max_abs_diff(sph->estimates.theta, ord->theta) < 1e-5);
  }
  {
    auto ord = magmaan::estimate::fit_gls(f.pt, f.rep, f.samp, f.x0);
    REQUIRE_OK(ord);
    auto sph = fr::fit_gls_sphere(f.pt, f.rep, f.samp, f.x0);
    REQUIRE_OK(sph);
    REQUIRE(sph->user_chart);
    CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-8));
    CHECK(max_abs_diff(sph->estimates.theta, ord->theta) < 1e-5);
  }
}

TEST_CASE("sphere ML: second-order model mixes sphere and user-chart latents") {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(9, 3);
  for (Eigen::Index k = 0; k < 3; ++k) L.col(k).segment(3 * k, 3) << 1.0, 0.8, 0.6;
  Eigen::Matrix3d Phi;
  Phi << 1.0, 0.5, 0.4, 0.5, 1.2, 0.45, 0.4, 0.45, 0.9;
  Eigen::MatrixXd S = L * Phi * L.transpose();
  S.diagonal().array() += 0.8;
  auto f = setup("f1 =~ x1 + x2 + x3\n f2 =~ x4 + x5 + x6\n"
                 " f3 =~ x7 + x8 + x9\n g =~ f1 + f2 + f3",
                 {perturb(S, 0.03)});
  const auto plan = plan_of(f.pt);
  CHECK(plan.units.size() == 2);
  check_same_as_ordinary_ml(f);
}

TEST_CASE("sphere ML: a model with no gauge unit runs in the user chart") {
  check_same_as_ordinary_ml(setup("f =~ 1*x1 + 1*x2 + 1*x3 + 1*x4",
                                  {one_factor_sigma()}));
}

TEST_CASE("sphere ML: a marker at a pole is reported outside the user chart") {
  // x1 is uncorrelated with x2..x4, so the best fit has a zero x1 loading and
  // the marker-chart estimate does not exist.
  Eigen::MatrixXd S = one_factor_sigma();
  for (Eigen::Index j = 1; j < 4; ++j) {
    S(0, j) = 0.0;
    S(j, 0) = 0.0;
  }
  auto f = setup("f =~ x1 + x2 + x3 + x4", {S});
  auto sph = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  CHECK_FALSE(sph->user_chart);
  REQUIRE(sph->report.scales.singular_units.size() == 1);
  CHECK(std::abs(sph->report.scales.direction_level(0)) < 1e-6);
  CHECK(sph->report.internal_theta.allFinite());
  CHECK(sph->report.internal_theta.cwiseAbs().maxCoeff() < 10.0);
}

TEST_CASE("sphere ML: an endogenous latent with no residual variance is the std.lv pole") {
  // Y = beta X exactly, so the residual variance std.lv fixes to 1 is zero at
  // the optimum: the std.lv estimate does not exist, the marker one does.
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L.col(0).head(3) << 1.0, 0.8, 0.6;
  L.col(1).tail(3) << 1.0, 0.8, 0.6;
  const double beta = 0.8;
  Eigen::Matrix2d Phi;
  Phi << 1.0, beta, beta, beta * beta;
  Eigen::MatrixXd S = L * Phi * L.transpose();
  S.diagonal().array() += 0.5;
  const std::string_view m = "X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X";

  BuildOptions o;
  o.std_lv = true;
  auto f = setup(m, {S}, o);
  auto sph = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  CHECK_FALSE(sph->user_chart);
  REQUIRE(sph->report.scales.singular_units.size() == 1);
  CHECK(sph->report.scales.direction_level(1) < 1e-6);
  CHECK(sph->report.fmin_internal < 1e-10);

  auto marker = setup(m, {S});
  auto to_marker = fr::reidentify(sph->report.internal_pt,
                                  sph->report.internal_theta, marker.pt);
  REQUIRE_OK(to_marker);
  CHECK(to_marker->theta.cwiseAbs().maxCoeff() < 10.0);
}

TEST_CASE("sphere FIML: reproduces the ordinary FIML estimate with missing data") {
  std::mt19937 rng(20260923);
  std::normal_distribution<double> z;
  const Eigen::Index n = 240;
  const Eigen::Vector4d lam(1.0, 0.8, 0.6, 0.7);
  const Eigen::Vector4d sd(0.7, 0.8, 0.85, 0.6);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = 1.1 * z(rng);
    for (Eigen::Index j = 0; j < 4; ++j)
      X(i, j) = 0.2 * static_cast<double>(j) + lam(j) * eta + sd(j) * z(rng);
  }
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M =
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (i % 7 == 3) { X(i, 1) = std::numeric_limits<double>::quiet_NaN(); M(i, 1) = 0; }
    if (i % 11 == 5) { X(i, 3) = std::numeric_limits<double>::quiet_NaN(); M(i, 3) = 0; }
  }
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  raw.mask.push_back(M);

  BuildOptions o;
  o.meanstructure = true;
  o.fixed_x = false;
  const auto pt = lavaanify("f =~ x1 + x2 + x3 + x4", o);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());
  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE_OK(samp);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, *samp, {});
  REQUIRE_OK(x0);
  auto ord = magmaan::estimate::fit_fiml(pt, *rep, raw, *x0);
  REQUIRE_OK(ord);
  auto sph = fr::fit_fiml_sphere(pt, *rep, raw, *x0);
  REQUIRE_OK(sph);
  REQUIRE(sph->user_chart);
  CHECK(sph->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
  CHECK(sph->report.plan.units.size() == 1);
  CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-8));
  CHECK(max_abs_diff(sph->estimates.theta, ord->theta) < 1e-4);
}

TEST_CASE("sphere PSD-ML: reproduces fit_ml_psd at an interior solution") {
  auto f = setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                 {two_factor_sigma()});
  auto ord = magmaan::estimate::frontier::fit_ml_psd(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(ord);
  auto sph = fr::fit_ml_psd_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  REQUIRE(sph->user_chart);
  CHECK(sph->report.native_verdict.status == magmaan::estimate::FitCheck::Passed);
  CHECK(sph->report.native_audit.computations.geometry.covariance_interior);
  CHECK(sph->report.pin_residual < 1e-6);
  CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-7));
  CHECK(max_abs_diff(sph->estimates.theta, ord->theta) < 1e-4);
}

TEST_CASE("sphere PSD-ML: a Heywood-prone sample stays admissible and bounded") {
  // One indicator nearly collinear with the factor pushes its residual
  // variance to the PSD boundary.
  Eigen::MatrixXd S = one_factor_sigma();
  S(0, 0) = 0.9 * S(0, 0);
  S(0, 1) *= 1.35;
  S(1, 0) = S(0, 1);
  auto f = setup("f =~ x1 + x2 + x3 + x4", {S});
  auto ord = magmaan::estimate::frontier::fit_ml_psd(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(ord);
  auto sph = fr::fit_ml_psd_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  REQUIRE(sph->user_chart);
  CHECK(sph->report.native_verdict.status == magmaan::estimate::FitCheck::Unchecked);
  CHECK(sph->report.native_audit.detail.find("sphere/PSD") != std::string::npos);
  CHECK(sph->estimates.fmin <= ord->fmin + 1e-7);
  CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-5));
  CHECK(sph->report.internal_theta.cwiseAbs().maxCoeff() < 20.0);
}

TEST_CASE("reidentify: the sphere solution at a marker pole reports in another chart") {
  Eigen::MatrixXd S = one_factor_sigma();
  for (Eigen::Index j = 1; j < 4; ++j) {
    S(0, j) = 0.0;
    S(j, 0) = 0.0;
  }
  auto f = setup("f =~ x1 + x2 + x3 + x4", {S});
  auto sph = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  REQUIRE_FALSE(sph->user_chart);
  const Eigen::MatrixXd sigma =
      implied(sph->report.internal_pt, f.rep, sph->report.internal_theta);

  // The marker chart on x1 cannot hold the point; a marker on x2 can.
  CHECK_FALSE(fr::reidentify(sph->report.internal_pt, sph->report.internal_theta,
                             f.pt).has_value());
  BuildOptions nf;
  nf.fixed_x = false;
  const auto pt_x2 = lavaanify("f =~ NA*x1 + 1*x2 + x3 + x4", nf);
  auto to_x2 = fr::reidentify(sph->report.internal_pt,
                              sph->report.internal_theta, pt_x2);
  REQUIRE_OK(to_x2);
  CHECK((implied(pt_x2, f.rep, to_x2->theta) - sigma).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("reidentify: a sphere fit's internal point maps back to the user estimate") {
  auto f = setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                 {two_factor_sigma()});
  auto sph = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(sph);
  REQUIRE(sph->user_chart);
  auto back = fr::reidentify(sph->report.internal_pt, sph->report.internal_theta, f.pt);
  REQUIRE_OK(back);
  // The internal point is the sphere optimizer's; the estimate is polished in
  // the user chart, which moves it within optimizer tolerance.
  CHECK(max_abs_diff(back->theta, sph->estimates.theta) < 1e-6);
  CHECK(sph->report.polish_shift < 1e-6);
}

TEST_CASE("sphere start: the canonical start makes the fit identification-invariant") {
  // One model under four identifications. The canonical start is the same
  // point for all four, so the sphere solutions coincide, and translating the
  // marker fit into each identification reproduces that identification's fit.
  BuildOptions std_lv;
  std_lv.std_lv = true;
  BuildOptions effect;
  effect.effect_coding = true;
  const std::string_view marker = "X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X";
  const std::string_view marker2 =
      "X =~ NA*x1 + 1*x2 + x3\n Y =~ NA*y1 + 1*y2 + y3\n Y ~ X";
  std::vector<Fitted> fits;
  fits.push_back(setup(marker, {two_factor_sigma()}));
  fits.push_back(setup(marker2, {two_factor_sigma()}));
  fits.push_back(setup(marker, {two_factor_sigma()}, std_lv));
  fits.push_back(setup(marker, {two_factor_sigma()}, effect));
  std::vector<fr::SphereFit> sph;
  for (auto& f : fits) {
    auto r = fr::fit_ml_sphere(f.pt, f.rep, f.samp, f.x0);
    REQUIRE_OK(r);
    REQUIRE(r->user_chart);
    CHECK(r->report.start_used == "canonical");
    sph.push_back(std::move(*r));
  }
  // The driven start and so the sphere solution agree up to the orientation
  // of each unit's coordinates, which the optimizer does not see. Compare by
  // partable row (the internal free indices are ordered differently).
  auto sphere_rows = [](const fr::SphereFit& x) {
    return fr::row_values(x.report.internal_pt, x.report.internal_theta);
  };
  const Eigen::VectorXd ref = sphere_rows(sph[0]);
  for (std::size_t k = 1; k < sph.size(); ++k) {
    const Eigen::VectorXd rk = sphere_rows(sph[k]);
    double worst = 0.0;
    for (Eigen::Index i = 0; i < std::min(ref.size(), rk.size()); ++i) {
      if (sph[0].report.internal_pt.is_constraint_row(static_cast<std::size_t>(i)))
        continue;
      if (std::isfinite(ref(i)) && std::isfinite(rk(i)))
        worst = std::max(worst, std::abs(std::abs(rk(i)) - std::abs(ref(i))));
    }
    CHECK(worst < 1e-10);
  }
  for (std::size_t k = 1; k < fits.size(); ++k) {
    auto moved = fr::reidentify(sph[0].report.internal_pt,
                                sph[0].report.internal_theta, fits[k].pt);
    REQUIRE_OK(moved);
    CHECK(max_abs_diff(moved->theta, sph[k].estimates.theta) < 1e-6);
  }

  // The user start remains available and is reported as such.
  fr::SphereOptions user;
  user.start = fr::SphereStart::User;
  auto r = fr::fit_ml_sphere(fits[3].pt, fits[3].rep, fits[3].samp, fits[3].x0, {},
                             magmaan::estimate::Backend::NloptLbfgs,
                             magmaan::estimate::ml_optim_options(), user);
  REQUIRE_OK(r);
  CHECK(r->report.start_used == "user");
}

TEST_CASE("sphere start: least squares starts from the sphere ML solution") {
  auto f = setup("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X",
                 {two_factor_sigma()});
  auto gls = fr::fit_gls_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(gls);
  CHECK(gls->report.start_used == "canonical (via ML)");
  auto ord = magmaan::estimate::fit_gls(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(ord);
  CHECK(gls->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-8));
  auto uls = fr::fit_gmm_sphere(f.pt, f.rep, f.samp, f.x0);
  REQUIRE_OK(uls);
  CHECK(uls->report.start_used == "canonical (via ML)");
}

TEST_CASE("sphere start: canonical in the reduced LISREL form (mean structure plus regression)") {
  // With a mean structure and a structural regression the observed variables
  // are phantom latents and loadings sit in Beta; the canonical start must
  // still find each indicator's sample variance.
  BuildOptions o;
  o.meanstructure = true;
  o.fixed_x = false;
  const auto pt = lavaanify("X =~ x1 + x2 + x3\n Y =~ y1 + y2 + y3\n Y ~ X", o);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());
  SampleStats ss = stats({two_factor_sigma()});
  ss.mean.push_back(Eigen::VectorXd::Zero(6));
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, ss, {});
  REQUIRE(x0.has_value());
  auto r = fr::fit_ml_sphere(pt, *rep, ss, *x0);
  REQUIRE_OK(r);
  CHECK(r->report.start_used == "canonical");
  auto ord = magmaan::estimate::fit_ml(pt, *rep, ss, *x0);
  REQUIRE_OK(ord);
  CHECK(r->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-8));
}

namespace {

// Three waves of an effect-coded factor with loadings tied across waves and
// linear growth on the wave factors: no latent is a gauge unit (cross-latent
// ties and higher-order loadings), so the sphere route drives the ordinary
// problem. `c` rescales the data.
Eigen::MatrixXd longitudinal_growth_sigma(double c) {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(9, 3);
  for (int t = 0; t < 3; ++t) L.block(3 * t, t, 3, 1) << 0.8, 1.0, 1.2;
  Eigen::MatrixXd G(3, 2);
  G << 1, 0, 1, 1, 1, 2;
  Eigen::Matrix2d Phi;
  Phi << 1.0, 0.1, 0.1, 0.2;
  Eigen::MatrixXd PsiF = G * Phi * G.transpose();
  PsiF.diagonal().array() += 0.3;
  Eigen::MatrixXd S = L * PsiF * L.transpose();
  S.diagonal().array() += 0.4;
  return c * c * perturb(S, 0.03);
}

constexpr const char* longitudinal_growth_model =
    "F1 =~ l1*a1 + l2*b1 + l3*c1\n"
    "F2 =~ l1*a2 + l2*b2 + l3*c2\n"
    "F3 =~ l1*a3 + l2*b3 + l3*c3\n"
    "l1 + l2 + l3 == 3\n"
    "i =~ 1*F1 + 1*F2 + 1*F3\n"
    "s =~ 0*F1 + 1*F2 + 2*F3\n";

}  // namespace

TEST_CASE("sphere ML: without gauge units the driven run is scaled like fit_ml") {
  // At c = 0.1 the unscaled driven run (coordinate_scaling none) fails with an
  // L-BFGS line-search error from this start, while fit_ml converges.
  BuildOptions o;
  o.auto_fix_first = false;
  for (double c : {1.0, 0.1}) {
    CAPTURE(c);
    Fitted f = setup(longitudinal_growth_model, {longitudinal_growth_sigma(c)}, o);
    auto x0 = magmaan::estimate::ml_start_values(f.pt, f.rep, f.samp);
    REQUIRE(x0.has_value());
    auto ord = magmaan::estimate::fit_ml(f.pt, f.rep, f.samp, x0->theta);
    REQUIRE_OK(ord);
    auto sph = fr::fit_ml_sphere(f.pt, f.rep, f.samp, x0->theta);
    REQUIRE_OK(sph);
    CHECK(sph->report.plan.units.empty());
    CHECK(sph->report.driven_scaled);
    CHECK(sph->report.driven_audit.stationary);
    CHECK(sph->estimates.fmin == doctest::Approx(ord->fmin).epsilon(1e-9));
    CHECK(max_abs_diff(sph->estimates.theta, ord->theta) < 1e-5 * c * c);
  }
}
