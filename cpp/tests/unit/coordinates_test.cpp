#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/coordinates.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/frontier/layered_start.hpp"
#include "magmaan/estimate/ml_numerics.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::estimate::build_eq_constraints;
using magmaan::estimate::coordinate_map;
using magmaan::estimate::parameter_units;
using magmaan::model::MatId;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::optim::CoordinateScaling;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentStructure;

namespace {

struct Built {
  LatentStructure pt;
  MatrixRep rep;
};

Built build(std::string_view src, BuildOptions opts = {}) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr)};
}

// Population values that depend only on the matrix cell, so tied parameters
// share a value and every free parameter is nonzero.
Eigen::VectorXd truth(const Built& b) {
  Eigen::VectorXd t = Eigen::VectorXd::Zero(b.pt.n_free());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    if (b.pt.free[i] <= 0) continue;
    const auto& c = b.rep.cell_for_row[i];
    double v = 0.0;
    switch (c.mat) {
      case MatId::Lambda: v = 0.6 + 0.15 * (c.row % 4); break;
      case MatId::Beta: v = 0.7; break;
      case MatId::Psi: v = c.row == c.col ? 0.8 + 0.1 * c.col : 0.3; break;
      case MatId::Theta: v = c.row == c.col ? 0.3 + 0.05 * c.row : 0.1; break;
      case MatId::Nu: v = 2.0 + 0.5 * c.row; break;
      case MatId::Alpha: v = 0.7; break;
    }
    t(b.pt.free[i] - 1) = v;
  }
  return t;
}

// Deterministic symmetric perturbation, so fits are not exact.
Eigen::MatrixXd perturb(const Eigen::MatrixXd& S, double eps) {
  Eigen::MatrixXd E(S.rows(), S.cols());
  for (Eigen::Index i = 0; i < S.rows(); ++i)
    for (Eigen::Index j = 0; j < S.cols(); ++j)
      E(i, j) = std::sin(1.7 * static_cast<double>(i + 1) * static_cast<double>(j + 1));
  E = 0.5 * (E + E.transpose());
  const Eigen::VectorXd d = S.diagonal().cwiseSqrt();
  return S + eps * d.asDiagonal() * E * d.asDiagonal();
}

SampleStats population(const Built& b, const Eigen::VectorXd& theta, bool means,
                       double eps = 0.04) {
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  auto e = ev->evaluate(theta, false, false);
  REQUIRE(e.has_value());
  SampleStats s;
  for (const auto& S : e->moments.sigma) s.S.push_back(perturb(S, eps));
  if (means) s.mean = e->moments.mu;
  s.n_obs.assign(s.S.size(), 400);
  return s;
}

SampleStats rescale(SampleStats s, const Eigen::VectorXd& d, double shift = 0.0) {
  for (auto& S : s.S) S = d.asDiagonal() * S * d.asDiagonal();
  for (auto& m : s.mean) m = d.asDiagonal() * m + Eigen::VectorXd::Constant(m.size(), shift);
  return s;
}

magmaan::estimate::Estimates fit(const Built& b, const SampleStats& s,
                                 const Eigen::VectorXd& x0, Backend backend,
                                 CoordinateScaling kind) {
  auto opts = magmaan::estimate::ml_optim_options();
  opts.coordinate_scaling = kind;
  auto r = magmaan::estimate::fit_ml(b.pt, b.rep, s, x0, {}, backend, opts);
  if (!r) {
    FAIL_CHECK(r.error().detail);
    magmaan::estimate::Estimates failed;
    failed.theta = x0;
    failed.fmin = std::numeric_limits<double>::quiet_NaN();
    return failed;
  }
  return *r;
}

bool accepted(const magmaan::estimate::Estimates& e) {
  return magmaan::estimate::fit_verdict(e).status ==
         magmaan::estimate::FitCheck::Passed;
}

constexpr std::string_view phantom_model =
    "ETA1 =~ y1 + y2 + y3\n"
    "ETA2 =~ y4 + y5 + y6\n"
    "P1 =~ 0*y1\nP2 =~ 0*y1\n"
    "ETA1 ~ P1\nETA2 ~ P2\n"
    "ETA1 ~~ 0*ETA1\nETA2 ~~ 0*ETA2\n"
    "P1 ~~ 1*P1\nP2 ~~ 1*P2\nP1 ~~ P2\n"
    "y1 ~~ y4\n";

constexpr std::string_view higher_order_model =
    "F1 =~ y1 + y2 + y3\nF2 =~ y4 + y5 + y6\nF3 =~ y7 + y8 + y9\n"
    "G =~ F1 + F2 + F3\n";

constexpr std::string_view path_model = "y2 ~ y1\ny3 ~ y1 + y2\n";

Eigen::VectorXd units_d(Eigen::Index p) {
  static const double values[] = {1e3, 1e-2, 1.0, 30.0, 1e-3, 5.0, 0.2, 400.0, 0.07};
  Eigen::VectorXd d(p);
  for (Eigen::Index i = 0; i < p; ++i) d(i) = values[i % 9];
  return d;
}

// Under a unit change every fitted parameter changes by a fixed factor. The
// coordinate scales must change by the same factor, so that a scaled optimizer
// retraces the same path from a transported start.
void check_equivariant(const Built& b, bool means) {
  const Eigen::VectorXd t = truth(b);
  const SampleStats s = population(b, t, means);
  const Eigen::VectorXd d = units_d(s.S[0].rows());
  const SampleStats sd = rescale(s, d);
  const auto a = fit(b, s, t, Backend::Port, CoordinateScaling::SampleUnits);
  REQUIRE(accepted(a));
  auto ua = parameter_units(b.pt, b.rep, s);
  auto uc = parameter_units(b.pt, b.rep, sd);
  REQUIRE(ua.has_value());
  REQUIRE(uc.has_value());
  const Eigen::VectorXd factor = uc->cwiseQuotient(*ua);
  auto rel_gap = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return (x - y).cwiseAbs().cwiseQuotient(y.cwiseAbs().cwiseMax(1e-12)).maxCoeff();
  };

  // The unit ratios are the parameters' transformation: the transported
  // optimum is the rescaled optimum.
  const Eigen::VectorXd transported = a.theta.cwiseProduct(factor);
  const auto refit = fit(b, sd, transported, Backend::Port, CoordinateScaling::SampleUnits);
  CHECK(refit.fmin == doctest::Approx(a.fmin).epsilon(1e-9));
  CHECK(rel_gap(refit.theta, transported) < 1e-5);

  auto con = build_eq_constraints(b.pt);
  REQUIRE(con.has_value());
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  for (auto kind : {CoordinateScaling::SampleUnits, CoordinateScaling::Information}) {
    CAPTURE(static_cast<int>(kind));
    auto ma = coordinate_map(kind, true, b.pt, b.rep, *ev, *con, s, con->contract(a.theta));
    auto mc = coordinate_map(kind, true, b.pt, b.rep, *ev, *con, sd,
                             con->contract(transported));
    REQUIRE(ma.has_value());
    REQUIRE(mc.has_value());
    const Eigen::VectorXd reduced =
        con->contract(transported).cwiseQuotient(con->contract(a.theta));
    CHECK(rel_gap(mc->scale.cwiseQuotient(ma->scale), reduced) < 1e-6);

    // From transported starts the scaled optimizers retrace the same path.
    for (auto backend : {Backend::Port, Backend::NloptLbfgs}) {
      CAPTURE(static_cast<int>(backend));
      auto start = magmaan::estimate::ml_start_values(b.pt, b.rep, s);
      REQUIRE(start.has_value());
      const auto x = fit(b, s, start->theta, backend, kind);
      const auto y = fit(b, sd, start->theta.cwiseProduct(factor), backend, kind);
      CHECK(y.fmin == doctest::Approx(x.fmin).epsilon(1e-10));
      // PORT retraces the path exactly; L-BFGS line searches may differ by
      // round-off near the optimum.
      if (backend == Backend::Port) CHECK(std::abs(y.iterations - x.iterations) <= 1);
    }
  }

  // The layered start is itself unit-equivariant.
  auto ls = magmaan::estimate::frontier::layered_start_values(b.pt, b.rep, s);
  auto ld = magmaan::estimate::frontier::layered_start_values(b.pt, b.rep, sd);
  REQUIRE(ls.has_value());
  REQUIRE(ld.has_value());
  CHECK(rel_gap(*ld, ls->cwiseProduct(factor)) < 1e-4);
}

}  // namespace

TEST_CASE("coordinate units: phantom latents take their indicator's standard deviation") {
  // Observed regressions live on phantom latents whose unit loading is a
  // structural cell, not a partable row.
  BuildOptions o; o.fixed_x = false;
  Built b = build(path_model, o);
  SampleStats s;
  Eigen::Matrix3d S;
  S << 1e4, 30.0, 20.0, 30.0, 1.0, 0.3, 20.0, 0.3, 4.0;
  s.S = {S};
  s.n_obs = {300};
  auto u = parameter_units(b.pt, b.rep, s);
  REQUIRE(u.has_value());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (b.pt.free[i] <= 0 || c.mat != MatId::Beta) continue;
    // Phantom order follows the observed order here: y1, y2, y3.
    const Eigen::Vector3d sd = S.diagonal().cwiseSqrt();
    const double expected = sd(c.row) / sd(c.col);
    CHECK((*u)(b.pt.free[i] - 1) == doctest::Approx(expected).epsilon(1e-12));
  }
}

TEST_CASE("coordinate scales transform like the parameters under unit changes") {
  SUBCASE("phantom-scaled latents") { check_equivariant(build(phantom_model), false); }
  SUBCASE("higher-order markers") { check_equivariant(build(higher_order_model), false); }
  SUBCASE("observed regressions") {
    BuildOptions o; o.fixed_x = false;
    check_equivariant(build(path_model, o), false);
  }
  SUBCASE("std.lv latents with means") {
    BuildOptions o; o.std_lv = true; o.meanstructure = true;
    check_equivariant(build("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6\n", o), true);
  }
  SUBCASE("cross-group loading equality ties a free latent variance") {
    BuildOptions o; o.n_groups = 2;
    check_equivariant(build("f =~ c(NA,NA)*c(l1,l1)*y1 + c(l2,l2)*y2 + c(l3,l3)*y3 + c(l4,l4)*y4\n"
                            "f ~~ c(1,NA)*f\n", o), false);
  }
}

TEST_CASE("coordinate map centers only location coordinates") {
  BuildOptions o; o.meanstructure = true;
  Built b = build("f =~ y1 + y2 + y3 + y4\ny1 ~ 0*1\nf ~ 1\n", o);
  const Eigen::VectorXd t = truth(b);
  const SampleStats s = population(b, t, true);
  auto con = build_eq_constraints(b.pt);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(con.has_value());
  REQUIRE(ev.has_value());
  auto m = coordinate_map(CoordinateScaling::Information, true, b.pt, b.rep, *ev, *con, s, t);
  REQUIRE(m.has_value());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    if (b.pt.free[i] <= 0) continue;
    const auto& c = b.rep.cell_for_row[i];
    const Eigen::Index k = b.pt.free[i] - 1;
    const bool location = c.mat == MatId::Nu || c.mat == MatId::Alpha;
    CHECK(m->center(k) == (location ? t(k) : 0.0));
  }
  auto off = coordinate_map(CoordinateScaling::Information, false, b.pt, b.rep, *ev, *con, s, t);
  REQUIRE(off.has_value());
  CHECK(off->center.cwiseAbs().maxCoeff() == 0.0);
  auto none = coordinate_map(CoordinateScaling::None, true, b.pt, b.rep, *ev, *con, s, t);
  REQUIRE(none.has_value());
  CHECK(none->scale.isOnes());
  CHECK(none->center.isZero());
}

TEST_CASE("scaled PORT and L-BFGS reach the same optimum at any units and locations") {
  BuildOptions o; o.meanstructure = true;
  Built b = build(std::string(phantom_model), o);
  const Eigen::VectorXd t = truth(b);
  const SampleStats s = population(b, t, true, 0.06);
  const Eigen::VectorXd d = units_d(6);
  const SampleStats sd = rescale(s, d, 1e4);
  for (auto backend : {Backend::Port, Backend::NloptLbfgs}) {
    for (auto kind : {CoordinateScaling::SampleUnits, CoordinateScaling::Information}) {
      CAPTURE(static_cast<int>(backend));
      CAPTURE(static_cast<int>(kind));
      const auto a = fit(b, s, t, backend, kind);
      REQUIRE(accepted(a));
      CHECK(a.coordinate_scaling == kind);
      // An equivariant start: the original start transported to the new units.
      auto ua = parameter_units(b.pt, b.rep, s);
      auto uc = parameter_units(b.pt, b.rep, sd);
      REQUIRE(ua.has_value());
      REQUIRE(uc.has_value());
      Eigen::VectorXd x0 = t.cwiseProduct(uc->cwiseQuotient(*ua));
      for (std::size_t i = 0; i < b.pt.size(); ++i) {
        const auto& c = b.rep.cell_for_row[i];
        if (b.pt.free[i] > 0 && c.mat == MatId::Nu) x0(b.pt.free[i] - 1) += 1e4;
      }
      const auto c = fit(b, sd, x0, backend, kind);
      CHECK(accepted(c));
      CHECK(c.fmin == doctest::Approx(a.fmin).epsilon(1e-8));
    }
  }
}

TEST_CASE("scaled fits keep bounds and nonlinear equality constraints") {
  Built b = build("f =~ x1 + a*x2 + b*x3 + x4\na == b*b\n");
  const Eigen::VectorXd t = truth(b);
  SampleStats s = population(b, t, false);
  s = rescale(s, Eigen::Vector4d(100.0, 0.1, 3.0, 20.0));
  auto start = magmaan::estimate::ml_start_values(b.pt, b.rep, s);
  REQUIRE(start.has_value());
  magmaan::estimate::Bounds box;
  box.lower = Eigen::VectorXd::Constant(b.pt.n_free(), -1e6);
  box.upper = Eigen::VectorXd::Constant(b.pt.n_free(), 1e6);
  for (auto kind : {CoordinateScaling::None, CoordinateScaling::SampleUnits,
                    CoordinateScaling::Information}) {
    CAPTURE(static_cast<int>(kind));
    auto opts = magmaan::estimate::ml_optim_options();
    opts.coordinate_scaling = kind;
    auto r = magmaan::estimate::fit_ml(b.pt, b.rep, s, start->theta, box,
                                       Backend::NloptSlsqp, opts);
    REQUIRE(r.has_value());
    CHECK(r->coordinate_scaling == kind);
    CHECK(r->diagnostics.nl_eq_satisfied);
    CHECK(r->audit.f_finite);
  }
}
