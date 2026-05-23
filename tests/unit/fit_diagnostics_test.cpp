#include <doctest/doctest.h>

#include <cmath>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::estimate::Bounds;
using magmaan::estimate::build_eq_constraints;
using magmaan::estimate::build_nl_constraints;
using magmaan::estimate::DiagnosticsOptions;
using magmaan::estimate::EqConstraints;
using magmaan::estimate::finalize_fit_diagnostics;
using magmaan::estimate::FitDiagnostics;
using magmaan::estimate::NonlinearEqConstraints;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;

namespace {

// Same pattern as model_evaluator_test.cpp:22-40 — stash LatentStructure and
// MatrixRep in thread_local statics so the evaluator's references outlive
// the helper return.
struct ModelBits {
  ModelEvaluator         ev;
  EqConstraints          con;
  NonlinearEqConstraints nl;
};

ModelBits build_bits(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  static thread_local magmaan::spec::LatentStructure s_pt;
  static thread_local magmaan::model::MatrixRep      s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  auto ev = ModelEvaluator::build(s_pt, s_mr);
  REQUIRE_MESSAGE(ev.has_value(),
                  "ModelEvaluator::build failed: " << ev.error().detail);
  auto con_or = build_eq_constraints(s_pt, /*allow_nonlinear=*/true);
  REQUIRE(con_or.has_value());
  auto nl_built = build_nl_constraints(s_pt);
  return ModelBits{std::move(*ev), std::move(*con_or), std::move(nl_built)};
}

// Build a θ at which Σ(θ) is PD: latent-variance positive, residuals
// positive, loadings ≠ 0. We pick conservative defaults that work for the
// 1F CFA fixture below.
Eigen::VectorXd pd_theta(const ModelEvaluator& ev) {
  Eigen::VectorXd theta = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(ev.n_free()));
  // The exact ordering doesn't matter for PD: with all-ones, the implied
  // Σ for a 1F CFA is `J J' + I` (loadings 1, residuals 1, ψ 1), which is
  // p.d. by Sherman-Morrison.
  return theta;
}

}  // namespace

TEST_CASE("finalize_fit_diagnostics: clean 1F CFA, no constraints, no bounds") {
  auto bits  = build_bits("f =~ x1 + x2 + x3");
  auto theta = pd_theta(bits.ev);
  Bounds empty_bounds;  // .empty() == true

  FitDiagnostics d = finalize_fit_diagnostics(
      theta, bits.ev, bits.con, bits.nl, empty_bounds);

  CHECK(d.sigma_pd_all);
  CHECK(d.sigma_pd_per_block.size() == 1u);
  CHECK(d.sigma_pd_per_block[0]);
  CHECK(d.lin_eq_residual_inf == doctest::Approx(0.0));
  CHECK(d.lin_eq_satisfied);
  CHECK(d.nl_eq_residual.size() == 0);
  CHECK(d.nl_eq_residual_inf == doctest::Approx(0.0));
  CHECK(d.nl_eq_satisfied);
  CHECK(d.active_bounds_full.at_lower.empty());
  CHECK(d.active_bounds_full.at_upper.empty());
  CHECK_FALSE(d.snlls_profile_fallback);
}

TEST_CASE("finalize_fit_diagnostics: linear equality constraint residual ≈ 0 by construction") {
  // Two loadings tied via a shared label — pure-merge K. The
  // K-reparameterization enforces θ_a = θ_b exactly, so A_eq · θ - b_eq
  // should be zero at any θ that lies on the manifold.
  auto bits = build_bits("f =~ x1 + a*x2 + a*x3");
  REQUIRE(bits.con.active());
  REQUIRE(bits.con.A_eq.rows() > 0);

  // Construct a θ that lies on the constraint manifold by expanding from α.
  Eigen::VectorXd alpha = Eigen::VectorXd::Ones(bits.con.n_alpha);
  Eigen::VectorXd theta = bits.con.expand(alpha);

  Bounds empty_bounds;
  FitDiagnostics d = finalize_fit_diagnostics(
      theta, bits.ev, bits.con, bits.nl, empty_bounds);

  CHECK(d.lin_eq_residual_inf < 1e-12);
  CHECK(d.lin_eq_satisfied);
}

TEST_CASE("finalize_fit_diagnostics: active lower bound is reported") {
  // A simple model with explicit bounds: one parameter pinned to its 0
  // lower bound should show up in active_bounds_full.at_lower.
  auto bits = build_bits("f =~ x1 + x2 + x3");
  auto theta = pd_theta(bits.ev);
  // Force the first parameter exactly onto its lower bound by setting it to 0
  // and giving bounds where lower[0] = 0.
  theta[0] = 0.0;
  Bounds b;
  b.lower = Eigen::VectorXd::Constant(theta.size(),
                                       -std::numeric_limits<double>::infinity());
  b.upper = Eigen::VectorXd::Constant(theta.size(),
                                       std::numeric_limits<double>::infinity());
  b.lower[0] = 0.0;

  FitDiagnostics d = finalize_fit_diagnostics(
      theta, bits.ev, bits.con, bits.nl, b);

  REQUIRE(d.active_bounds_full.at_lower.size() == 1u);
  CHECK(d.active_bounds_full.at_lower[0] == 0);
  CHECK(d.active_bounds_full.at_upper.empty());
}

TEST_CASE("finalize_fit_diagnostics: non-PD Σ at a bad θ is detected") {
  // Force a non-PD implied Σ by setting all residual variances to 0 and
  // all loadings to 0 — Σ = ΛΨΛᵀ + Θ = 0, fails Cholesky. (Even at θ=0,
  // Σ may be identically zero, which LLT::info() flags as non-PD.)
  auto bits  = build_bits("f =~ x1 + x2 + x3");
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(bits.ev.n_free()));
  Bounds empty_bounds;

  FitDiagnostics d = finalize_fit_diagnostics(
      theta, bits.ev, bits.con, bits.nl, empty_bounds);

  CHECK_FALSE(d.sigma_pd_all);
  // We don't assert per-block strictly: the contract is just "the test
  // surfaces the issue", which `sigma_pd_all=false` does cleanly.
}
