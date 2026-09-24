#include <doctest/doctest.h>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// Coverage for the no-optimizer `evaluate_at` helper. Three properties:
//   1. At a converged θ the audit reports stationary with a tiny grad_inf.
//   2. At a displaced θ the audit reports non-stationary with a big grad_inf.
//   3. The verdict tracks the estimator family (ULS, GLS, ML each tested).
// The helper takes a full-θ vector, so it does not need to know whether the
// fit that produced θ was SNLLS or ordinary.

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::estimate::Estimator;
using magmaan::estimate::evaluate_at;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::build;

namespace {

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

Handles handles_for(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

// A clean 1F CFA with Σ = λλᵀψ + Θ — guarantees a finite, stationary ML/GLS
// optimum the fit will reach so we can audit at it.
Eigen::Matrix4d make_1f_S() {
  Eigen::Vector4d lambda(1.0, 0.85, 0.72, 0.65);
  const double psi = 1.5;
  Eigen::Vector4d theta(0.55, 0.45, 0.65, 0.50);
  return lambda * lambda.transpose() * psi +
         theta.asDiagonal().toDenseMatrix();
}

}  // namespace

TEST_CASE("evaluate_at: ULS audit at the converged θ is stationary") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto fit = magmaan::estimate::fit_gmm(h.pt, h.rep, samp, *x0);
  REQUIRE(fit.has_value());
  REQUIRE(fit->audit.stationary);

  auto ev = evaluate_at(h.pt, h.rep, samp, fit->theta, Estimator::ULS);
  REQUIRE(ev.has_value());
  CHECK(ev->audit.stationary);
  CHECK(ev->audit.f_finite);
  CHECK(ev->audit.grad_inf_norm < 1e-6);
  CHECK(ev->iterations == 0);  // no optimizer ran
  CHECK(ev->f_evals    == 1);
  CHECK(ev->g_evals    == 1);
  CHECK(ev->theta.size() == fit->theta.size());
}

TEST_CASE("evaluate_at: ULS audit at a displaced θ is non-stationary") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {300};
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto fit = magmaan::estimate::fit_gmm(h.pt, h.rep, samp, *x0);
  REQUIRE(fit.has_value());

  Eigen::VectorXd theta_off = fit->theta;
  theta_off.array() *= 1.5;  // big displacement → big projected gradient
  auto ev = evaluate_at(h.pt, h.rep, samp, theta_off, Estimator::ULS);
  REQUIRE(ev.has_value());
  CHECK_FALSE(ev->audit.stationary);
  CHECK(ev->audit.grad_inf_norm > 1e-3);  // far above lavaan's 1e-3 bar
}

TEST_CASE("evaluate_at: GLS picks up the same converged θ as fit_gls") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {300};
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto fit = magmaan::estimate::fit_gls(h.pt, h.rep, samp, *x0);
  REQUIRE(fit.has_value());
  REQUIRE(fit->audit.stationary);

  auto ev = evaluate_at(h.pt, h.rep, samp, fit->theta, Estimator::GLS);
  REQUIRE(ev.has_value());
  CHECK(ev->audit.stationary);
  CHECK(ev->audit.grad_inf_norm < 1e-6);
}

TEST_CASE("evaluate_at: ML picks up the converged θ from fit_ml") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {300};
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto fit = magmaan::estimate::fit_ml(h.pt, h.rep, samp, *x0);
  REQUIRE(fit.has_value());
  REQUIRE(fit->audit.stationary);

  auto ev = evaluate_at(h.pt, h.rep, samp, fit->theta, Estimator::ML);
  REQUIRE(ev.has_value());
  CHECK(ev->audit.stationary);
  CHECK(ev->audit.grad_inf_norm < 1e-6);
}

TEST_CASE("evaluate_at: WLS with empty weight is an error") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {300};
  Eigen::VectorXd theta(h.pt.n_free());
  theta.setOnes();
  auto ev = evaluate_at(h.pt, h.rep, samp, theta, Estimator::WLS);
  CHECK_FALSE(ev.has_value());
}

TEST_CASE("evaluate_at: theta size mismatch is an error") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {300};
  Eigen::VectorXd theta(h.pt.n_free() + 3);
  theta.setOnes();
  auto ev = evaluate_at(h.pt, h.rep, samp, theta, Estimator::ULS);
  CHECK_FALSE(ev.has_value());
}
