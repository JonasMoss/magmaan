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
}

}  // namespace

TEST_CASE("sphere ML: one-factor marker CFA") {
  check_same_as_ordinary_ml(setup("f =~ x1 + x2 + x3 + x4", {one_factor_sigma()}));
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
  CHECK(max_abs_diff(back->theta, sph->estimates.theta) < 1e-9);
}
