#include <doctest/doctest.h>

#include <cmath>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/layered_start.hpp"
#include "magmaan/estimate/start_pipeline.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::build_eq_constraints;
using magmaan::estimate::layered_start_report;
using magmaan::model::MatId;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentStructure;
using magmaan::spec::Starts;

namespace {

struct Built {
  LatentStructure pt;
  MatrixRep rep;
  Starts starts;
};

Built build(std::string_view src, BuildOptions opts = {}) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  Starts st;
  auto pt = magmaan::spec::build(*fp, opts, &st);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr), std::move(st)};
}

// A population parameter vector: values depend only on the matrix cell, so
// parameters tied across groups receive the same value.
Eigen::VectorXd truth(const Built& b, double path = 1.3) {
  Eigen::VectorXd t = Eigen::VectorXd::Zero(b.pt.n_free());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    if (b.pt.free[i] <= 0) continue;
    const auto& c = b.rep.cell_for_row[i];
    double v = 0.0;
    switch (c.mat) {
      case MatId::Lambda: v = 0.6 + 0.15 * (c.row % 4); break;
      case MatId::Beta: v = path; break;
      case MatId::Psi: v = c.row == c.col ? 0.8 + 0.1 * c.col : 0.3; break;
      case MatId::Theta: v = c.row == c.col ? 0.3 + 0.05 * c.row : 0.1; break;
      case MatId::Nu: v = 0.5 * c.row; break;
      case MatId::Alpha: v = 0.7; break;
    }
    t(b.pt.free[i] - 1) = v;
  }
  return t;
}

std::vector<Eigen::MatrixXd> sigma(const Built& b, const Eigen::VectorXd& theta) {
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  auto m = ev->sigma(theta);
  REQUIRE(m.has_value());
  return m->sigma;
}

std::vector<Eigen::VectorXd> mu(const Built& b, const Eigen::VectorXd& theta) {
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  auto e = ev->evaluate(theta, false, false);
  REQUIRE(e.has_value());
  return e->moments.mu;
}

// Deterministic symmetric perturbation, so the start is not an exact fit.
Eigen::MatrixXd perturb(const Eigen::MatrixXd& S, double eps) {
  Eigen::MatrixXd E(S.rows(), S.cols());
  for (Eigen::Index i = 0; i < S.rows(); ++i)
    for (Eigen::Index j = 0; j < S.cols(); ++j)
      E(i, j) = std::sin(1.7 * static_cast<double>(i + 1) * static_cast<double>(j + 1));
  E = 0.5 * (E + E.transpose());
  const Eigen::VectorXd d = S.diagonal().cwiseSqrt();
  return S + eps * d.asDiagonal() * E * d.asDiagonal();
}

SampleStats stats(std::vector<Eigen::MatrixXd> S, std::vector<Eigen::VectorXd> m = {}) {
  SampleStats s;
  s.n_obs.assign(S.size(), 400);
  s.S = std::move(S);
  s.mean = std::move(m);
  return s;
}

Eigen::VectorXd layered(const Built& b, const SampleStats& s) {
  auto r = layered_start_report(b.pt, b.rep, s, b.starts);
  REQUIRE(r.has_value());
  return r->theta;
}

double max_rel_diff(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  return (A - B).cwiseAbs().maxCoeff() / std::max(1e-300, B.cwiseAbs().maxCoeff());
}

bool pd(const Eigen::MatrixXd& S) {
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  return llt.info() == Eigen::Success;
}

double free_value(const Built& b, MatId mat, int row, int col, int block = 0) {
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (b.pt.free[i] > 0 && c.used && c.mat == mat && c.row == row && c.col == col && c.block == block)
      return static_cast<double>(b.pt.free[i] - 1);
  }
  return -1.0;
}

constexpr std::string_view phantom_model =
    "ETA1 =~ y1 + y2 + y3\n"
    "ETA2 =~ y4 + y5 + y6\n"
    "P1 =~ 0*y1\nP2 =~ 0*y1\n"
    "ETA1 ~ P1\nETA2 ~ P2\n"
    "ETA1 ~~ 0*ETA1\nETA2 ~~ 0*ETA2\n"
    "P1 ~~ 1*P1\nP2 ~~ 1*P2\nP1 ~~ P2\n"
    "y1 ~~ y4\n";

}  // namespace

TEST_CASE("layered start: observed-only moments have no latent solve") {
  BuildOptions options;
  SUBCASE("variance only") { options.meanstructure = false; }
  SUBCASE("mean and variance") { options.meanstructure = true; }
  const Built b = build("x ~~ x", options);
  REQUIRE(b.rep.dims.front().n_latent == 0);
  const Eigen::MatrixXd covariance = Eigen::MatrixXd::Constant(1, 1, 2.25);
  const Eigen::VectorXd mean = Eigen::VectorXd::Constant(1, 1.5);
  const SampleStats sample = options.meanstructure ? stats({covariance}, {mean})
                                                 : stats({covariance});
  const auto report = layered_start_report(b.pt, b.rep, sample, b.starts);
  REQUIRE(report.has_value());
  REQUIRE(report->theta.allFinite());
  CHECK(sigma(b, report->theta).front().isApprox(covariance, 1e-12));
  if (options.meanstructure)
    CHECK(mu(b, report->theta).front().isApprox(mean, 1e-12));
  CHECK(std::isfinite(report->structural_cost_initial));
  CHECK(std::isfinite(report->structural_cost_final));
}

TEST_CASE("layered start: a phantom-scaled model is recovered from its population covariance") {
  Built b = build(phantom_model);
  const Eigen::VectorXd t = truth(b, 1.5);
  const auto S = sigma(b, t);
  const Eigen::VectorXd x = layered(b, stats(S));
  CHECK((x - t).cwiseAbs().maxCoeff() < 1e-6);
  // The variance-carrying paths are nonzero: the start is off the reflection trap.
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (b.pt.free[i] > 0 && c.mat == MatId::Beta) CHECK(x(b.pt.free[i] - 1) > 1.0);
  }
}

TEST_CASE("layered start: higher-order loadings and phantom regressions give the same start") {
  Built a = build("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6\nf3 =~ y7 + y8 + y9\n"
                  "g =~ NA*f1 + f2 + f3\ng ~~ 1*g\n");
  Built r = build("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6\nf3 =~ y7 + y8 + y9\n"
                  "g =~ 0*y1\nf1 ~ g\nf2 ~ g\nf3 ~ g\ng ~~ 1*g\n");
  const Eigen::VectorXd t = truth(a, 0.8);
  const auto S = sigma(a, t);
  const SampleStats s = stats({perturb(S[0], 0.04)});
  const Eigen::VectorXd xa = layered(a, s), xr = layered(r, s);
  CHECK(max_rel_diff(sigma(a, xa)[0], sigma(r, xr)[0]) < 1e-6);
  CHECK(pd(sigma(r, xr)[0]));
  int paths = 0;
  for (std::size_t i = 0; i < r.pt.size(); ++i) {
    const auto& c = r.rep.cell_for_row[i];
    if (r.pt.free[i] > 0 && c.mat == MatId::Beta) { CHECK(xr(r.pt.free[i] - 1) > 0.1); ++paths; }
  }
  CHECK(paths == 3);
  // Population input: the second-order model is recovered exactly.
  const Eigen::VectorXd xp = layered(a, stats(S));
  CHECK(max_rel_diff(sigma(a, xp)[0], S[0]) < 1e-6);
}

TEST_CASE("layered start: marker, std.lv and effect coding imply the same start covariance") {
  constexpr std::string_view src = "f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6\nf2 ~ f1\ny3 ~~ y6\n";
  Built marker = build(src);
  BuildOptions sl; sl.std_lv = true;
  Built stdlv = build(src, sl);
  BuildOptions ec; ec.effect_coding = true;
  Built effect = build(src, ec);
  const auto S = sigma(marker, truth(marker, 0.6));
  const SampleStats s = stats({perturb(S[0], 0.05)});
  const Eigen::MatrixXd sm = sigma(marker, layered(marker, s))[0];
  const Eigen::VectorXd xe = layered(effect, s);
  CHECK(max_rel_diff(sigma(stdlv, layered(stdlv, s))[0], sm) < 1e-6);
  CHECK(max_rel_diff(sigma(effect, xe)[0], sm) < 1e-6);
  auto con = build_eq_constraints(effect.pt);
  REQUIRE(con.has_value());
  CHECK((con->A_eq * xe - con->b_eq).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("layered start: rescaling observed variables rescales the implied start") {
  BuildOptions o; o.meanstructure = true;
  Built b = build(std::string(phantom_model) + "ETA2 ~~ 0*ETA1\n", o);
  const Eigen::VectorXd t = truth(b, 1.1);
  const auto S = sigma(b, t);
  const auto m = mu(b, t);
  const Eigen::MatrixXd Sp = perturb(S[0], 0.05);
  Eigen::VectorXd d(6);
  d << 10.0, 0.1, 3.0, 1.0, 250.0, 0.02;
  const Eigen::MatrixXd Sd = d.asDiagonal() * Sp * d.asDiagonal();
  const Eigen::VectorXd md = d.asDiagonal() * m[0];
  const Eigen::VectorXd x1 = layered(b, stats({Sp}, {m[0]}));
  const Eigen::VectorXd x2 = layered(b, stats({Sd}, {md}));
  const Eigen::MatrixXd s1 = d.asDiagonal() * sigma(b, x1)[0] * d.asDiagonal();
  CHECK(max_rel_diff(sigma(b, x2)[0], s1) < 1e-6);
  const Eigen::VectorXd m1 = d.asDiagonal() * mu(b, x1)[0];
  CHECK((mu(b, x2)[0] - m1).cwiseAbs().maxCoeff() / md.cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("layered start: multigroup phantom scaling with shared loadings") {
  // Group 1 fixes the phantom path at 1; group 2 frees it; loadings are shared.
  BuildOptions o; o.n_groups = 2;
  Built b = build("f =~ c(NA,NA)*c(l1,l1)*y1 + c(l2,l2)*y2 + c(l3,l3)*y3 + c(l4,l4)*y4\n"
                  "p =~ c(0,0)*y1\nf ~ c(1,NA)*p\nf ~~ c(0,0)*f\np ~~ c(1,1)*p\n", o);
  const Eigen::VectorXd t = truth(b, 1.4);
  const auto S = sigma(b, t);
  const Eigen::VectorXd x = layered(b, stats(S));
  CHECK((x - t).cwiseAbs().maxCoeff() < 1e-6);
  const Eigen::VectorXd xp = layered(b, stats({perturb(S[0], 0.05), perturb(S[1], -0.04)}));
  auto con = build_eq_constraints(b.pt);
  REQUIRE(con.has_value());
  CHECK((con->A_eq * xp - con->b_eq).cwiseAbs().maxCoeff() < 1e-8);
  const double k = free_value(b, MatId::Beta, 0, 1, 1);
  REQUIRE(k >= 0.0);
  CHECK(xp(static_cast<Eigen::Index>(k)) > 0.5);
  for (const auto& s : sigma(b, xp)) CHECK(pd(s));
}

TEST_CASE("layered start: latent means and intercepts are recovered") {
  BuildOptions o; o.meanstructure = true;
  Built b = build("f =~ y1 + y2 + y3 + y4\ny1 ~ 0*1\nf ~ 1\n", o);
  const Eigen::VectorXd t = truth(b);
  const auto S = sigma(b, t);
  const auto m = mu(b, t);
  const Eigen::VectorXd x = layered(b, stats(S, m));
  CHECK((x - t).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("layered start: user start hints win") {
  Built b = build("f =~ y1 + start(0.3)*y2 + y3\ng =~ y4 + y5 + y6\ng ~ f\n");
  const auto S = sigma(b, truth(b));
  const Eigen::VectorXd x = layered(b, stats(S));
  const double k = free_value(b, MatId::Lambda, 1, 0);
  REQUIRE(k >= 0.0);
  CHECK(x(static_cast<Eigen::Index>(k)) == doctest::Approx(0.3));
}

TEST_CASE("layered start: equalities pin loading blocks like explicit constants") {
  BuildOptions o;
  o.meanstructure = true;
  const Built constrained = build(
      "f =~ NA*a*y1 + b*y2 + c*y3\n"
      "g =~ NA*d*y4 + e*y5 + h*y6\n"
      "a + b + c == 3\na == b\nb == c\n"
      "d + e + h == 3\nd == e\ne == h\n", o);
  const Built fixed = build(
      "f =~ 1*y1 + 1*y2 + 1*y3\n"
      "g =~ 1*y4 + 1*y5 + 1*y6\n", o);
  const auto con = build_eq_constraints(constrained.pt);
  REQUIRE(con.has_value());
  for (double scale : {1.0, 0.001, 1000.0}) {
    const auto pop = sigma(fixed, truth(fixed));
    const SampleStats s = stats({scale * scale * perturb(pop[0], 0.01)},
                                {scale * mu(fixed, truth(fixed))[0]});
    const Eigen::VectorXd x = layered(constrained, s);
    const Eigen::VectorXd xf = layered(fixed, s);
    CHECK((con->A_eq * x - con->b_eq).norm() < 1e-8);
    CHECK(max_rel_diff(sigma(constrained, x)[0], sigma(fixed, xf)[0]) < 1e-9);
    CHECK((mu(constrained, x)[0] - mu(fixed, xf)[0]).norm() < 1e-8 * scale);
    CHECK(pd(sigma(constrained, x)[0]));
  }
}

TEST_CASE("layered start: the start pipeline exposes the layered method untransported") {
  Built b = build(phantom_model);
  const auto S = sigma(b, truth(b));
  magmaan::estimate::StartPolicy policy{magmaan::estimate::StartMethod::Layered,
                                        magmaan::estimate::StartTransport::AutoStdLv};
  auto v = magmaan::estimate::start_values(b.pt, b.rep, stats(S), policy, b.starts);
  REQUIRE(v.has_value());
  CHECK(v->branch == magmaan::estimate::StartBranch::Native);
  CHECK(v->theta.size() == b.pt.n_free());
}

TEST_CASE("simple and FABIN starts: a zero disturbance is not a std.lv scale") {
  // ETA1 is scaled by its phantom parent, not by a fixed variance. Loadings in
  // that marker-free chart must not scale with the indicators' units.
  Built b = build("ETA1 =~ NA*y1 + y2 + y3 + y4\nP1 =~ 0*y1\nETA1 ~ P1\n"
                  "ETA1 ~~ 0*ETA1\nP1 ~~ 1*P1\n");
  const auto S = sigma(b, truth(b, 1.2));
  const SampleStats s1 = stats(S), s2 = stats({100.0 * S[0]});
  auto a1 = magmaan::estimate::simple_start_values(b.pt, b.rep, s1);
  auto a2 = magmaan::estimate::simple_start_values(b.pt, b.rep, s2);
  auto f1 = magmaan::estimate::fabin_start_values(b.pt, b.rep, s1);
  auto f2 = magmaan::estimate::fabin_start_values(b.pt, b.rep, s2);
  REQUIRE(a1.has_value()); REQUIRE(a2.has_value());
  REQUIRE(f1.has_value()); REQUIRE(f2.has_value());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (b.pt.free[i] <= 0 || c.mat != MatId::Lambda) continue;
    const auto k = b.pt.free[i] - 1;
    CHECK((*a1)(k) == doctest::Approx((*a2)(k)));
    CHECK((*f1)(k) == doctest::Approx((*f2)(k)));
  }
}
