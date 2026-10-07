#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <Eigen/Cholesky>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::estimate::Bounds;
using magmaan::estimate::audit_geometric_stationarity;
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
  const magmaan::spec::LatentStructure* pt;
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
  return ModelBits{&s_pt, std::move(*ev), std::move(*con_or),
                   std::move(nl_built)};
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
      theta, *bits.pt, bits.ev, bits.con, bits.nl, empty_bounds);

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
  CHECK(d.admissibility.checked);
  CHECK(d.admissibility.covariance_matrices_psd);
  CHECK(d.admissibility.implied_sigma_pd);
  CHECK(d.admissibility.admissible);
  REQUIRE(d.admissibility.theta_blocks.size() == 1u);
  REQUIRE(d.admissibility.psi_blocks.size() == 1u);
  CHECK(d.admissibility.theta_blocks[0].psd);
  CHECK(d.admissibility.psi_blocks[0].psd);
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
      theta, *bits.pt, bits.ev, bits.con, bits.nl, empty_bounds);

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
      theta, *bits.pt, bits.ev, bits.con, bits.nl, b);

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
      theta, *bits.pt, bits.ev, bits.con, bits.nl, empty_bounds);

  CHECK_FALSE(d.sigma_pd_all);
  CHECK_FALSE(d.admissibility.admissible);
  // We don't assert per-block strictly: the contract is just "the test
  // surfaces the issue", which `sigma_pd_all=false` does cleanly.
}

TEST_CASE("finalize_fit_diagnostics: negative residual variance is inadmissible "
          "even when implied Sigma is PD") {
  auto bits = build_bits("f =~ x1 + x2 + x3");
  auto theta = pd_theta(bits.ev);
  const auto locs = bits.ev.param_locations();

  Eigen::Index target = -1;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Theta &&
        locs[k].row == locs[k].col) {
      target = static_cast<Eigen::Index>(k);
      break;
    }
  }
  REQUIRE(target >= 0);
  theta(target) = -0.25;

  FitDiagnostics d = finalize_fit_diagnostics(
      theta, *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});

  CHECK(d.sigma_pd_all);
  CHECK(d.admissibility.checked);
  CHECK_FALSE(d.admissibility.covariance_matrices_psd);
  CHECK_FALSE(d.admissibility.admissible);
  REQUIRE(d.admissibility.theta_blocks.size() == 1u);
  const auto& block = d.admissibility.theta_blocks[0];
  CHECK_FALSE(block.psd);
  CHECK(block.min_eigenvalue == doctest::Approx(-0.25));
  REQUIRE(block.negative_variance_rows.size() == 1u);
  const auto row = static_cast<std::size_t>(block.negative_variance_rows[0]);
  CHECK((*bits.pt).op[row] == magmaan::parse::Op::Covariance);
  CHECK((*bits.pt).lhs_var[row] == (*bits.pt).rhs_var[row]);
}

TEST_CASE("finalize_fit_diagnostics: joint latent PSD failure is detected "
          "despite valid pairwise correlations") {
  auto bits = build_bits(
      "f1 =~ x1 + x2\n"
      "f2 =~ x3 + x4\n"
      "f3 =~ x5 + x6\n"
      "f1 ~~ f2\n"
      "f1 ~~ f3\n"
      "f2 ~~ f3");
  Eigen::VectorXd theta =
      Eigen::VectorXd::Ones(static_cast<Eigen::Index>(bits.ev.n_free()));
  const auto locs = bits.ev.param_locations();

  int covariance = 0;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& loc = locs[k];
    if (loc.mat == magmaan::model::MatId::Theta &&
        loc.row == loc.col) {
      theta(static_cast<Eigen::Index>(k)) = 5.0;
    } else if (loc.mat == magmaan::model::MatId::Psi &&
               loc.row != loc.col) {
      theta(static_cast<Eigen::Index>(k)) =
          (covariance++ == 2) ? -0.9 : 0.9;
    }
  }
  REQUIRE(covariance == 3);

  FitDiagnostics d = finalize_fit_diagnostics(
      theta, *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});

  CHECK(d.sigma_pd_all);
  CHECK_FALSE(d.admissibility.admissible);
  REQUIRE(d.admissibility.psi_blocks.size() == 1u);
  const auto& block = d.admissibility.psi_blocks[0];
  CHECK_FALSE(block.psd);
  CHECK(block.min_eigenvalue < -0.1);
  CHECK(block.negative_variance_rows.empty());
  CHECK(block.invalid_correlation_rows.empty());
  CHECK(block.covariance_rows.size() >= 6u);
}

TEST_CASE("geometric stationarity uses the primitive PSD normal cone") {
  auto bits = build_bits("f =~ x1 + x2 + x3");
  Eigen::VectorXd theta = pd_theta(bits.ev);
  const auto locations = bits.ev.param_locations();
  Eigen::Index variance = -1;
  for (std::size_t k = 0; k < locations.size(); ++k) {
    if (locations[k].mat == magmaan::model::MatId::Theta &&
        locations[k].row == locations[k].col) {
      variance = static_cast<Eigen::Index>(k);
      break;
    }
  }
  REQUIRE(variance >= 0);
  theta(variance) = 0.0;

  Eigen::VectorXd outward = Eigen::VectorXd::Zero(theta.size());
  outward(variance) = 2.0;
  const auto stationary = audit_geometric_stationarity(
      theta, outward, *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});

  CHECK(stationary.checked);
  CHECK(stationary.feasible);
  CHECK(stationary.covariance_feasible);
  CHECK_FALSE(stationary.ambient_stationary);
  CHECK(stationary.ambient_residual_inf == doctest::Approx(2.0));
  CHECK(stationary.cone_stationary);
  CHECK(stationary.cone_residual_inf == doctest::Approx(0.0).scale(1.0));
  CHECK(stationary.covariance_active_blocks == 1);
  CHECK(stationary.covariance_nullity == 1);

  Eigen::VectorXd inward = -outward;
  const auto nonstationary = audit_geometric_stationarity(
      theta, inward, *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  CHECK(nonstationary.feasible);
  CHECK_FALSE(nonstationary.ambient_stationary);
  CHECK_FALSE(nonstationary.cone_stationary);
  CHECK(nonstationary.cone_residual_inf == doctest::Approx(2.0));
}

TEST_CASE("geometric stationarity reduces to the ambient audit in the "
          "positive-definite interior") {
  auto bits = build_bits("f =~ x1 + x2 + x3");
  const Eigen::VectorXd theta = pd_theta(bits.ev);
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(theta.size());

  const auto d = audit_geometric_stationarity(
      theta, gradient, *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});

  CHECK(d.checked);
  CHECK(d.feasible);
  CHECK(d.ambient_stationary);
  CHECK(d.cone_stationary);
  CHECK(d.ambient_residual_inf == doctest::Approx(d.cone_residual_inf));
  CHECK(d.covariance_active_blocks == 0);
  CHECK(d.covariance_nullity == 0);
}

TEST_CASE("geometric stationarity thresholds the metric-dual L2 norm") {
  auto bits = build_bits("f =~ x1 + x2 + x3");
  const Eigen::VectorXd theta = pd_theta(bits.ev);
  const auto locations = bits.ev.param_locations();
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(theta.size());
  int loading_gradients = 0;
  for (std::size_t k = 0; k < locations.size(); ++k) {
    if (locations[k].mat != magmaan::model::MatId::Lambda) continue;
    gradient(static_cast<Eigen::Index>(k)) = 8e-4;
    if (++loading_gradients == 2) break;
  }
  REQUIRE(loading_gradients == 2);

  const auto d = audit_geometric_stationarity(
      theta, gradient, *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});

  CHECK(d.ambient_residual_inf < d.stationarity_tol);
  CHECK(d.ambient_residual_l2 > d.stationarity_tol);
  CHECK_FALSE(d.ambient_stationary);
  CHECK(d.cone_residual_l2 == doctest::Approx(d.ambient_residual_l2));
  CHECK_FALSE(d.cone_stationary);
}

TEST_CASE("common fit verdict is independent of backend and legacy audit") {
  using namespace magmaan::estimate;
  auto bits = build_bits("f =~ x1 + x2 + x3");
  Estimates est;
  est.theta = pd_theta(bits.ev);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(est.theta.size());
  CHECK(fit_verdict(est).status == FitCheck::Unchecked);
  audit_full_model_fit(est.diagnostics, est.theta, zero, 0.25, 0.25,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  for (auto status : {magmaan::optim::OptimStatus::Converged,
                      magmaan::optim::OptimStatus::SingularConvergence,
                      magmaan::optim::OptimStatus::BudgetExhausted,
                      magmaan::optim::OptimStatus::Unknown}) {
    est.optimizer_status = status;
    CHECK_FALSE(est.audit.f_consistent); // Missing L1 is not a veto.
    CHECK(fit_verdict(est).status == FitCheck::Passed);
  }
  est.diagnostics.geometric_stationarity.ambient_projection_converged = false;
  CHECK(fit_verdict(est).status == FitCheck::Unchecked);
  CHECK(fit_verdict(est).objective == FitCheck::Passed);
  est.optimizer_status = magmaan::optim::OptimStatus::Converged;
  audit_full_model_fit(est.diagnostics, est.theta,
                       Eigen::VectorXd::Ones(est.theta.size()), 0.25, 0.25,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  CHECK(fit_verdict(est).status == FitCheck::Failed);
  CHECK(fit_verdict(est).objective == FitCheck::Passed);
  CHECK(fit_verdict(est).stationarity == FitCheck::Failed);
  audit_full_model_fit(est.diagnostics, est.theta, zero, 0.25, 0.5,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  CHECK(fit_verdict(est).status == FitCheck::Failed);
  CHECK(fit_verdict(est).objective == FitCheck::Failed);
  CHECK(fit_verdict(est).stationarity == FitCheck::Passed);
  audit_full_model_fit(est.diagnostics, est.theta, zero, 0.25,
                       std::numeric_limits<double>::quiet_NaN(),
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  CHECK(fit_verdict(est).status == FitCheck::Failed);
}

TEST_CASE("common fit verdict uses declared PSD domain at a boundary") {
  using namespace magmaan::estimate;
  auto bits = build_bits("f =~ x1 + x2 + x3");
  Eigen::VectorXd theta = pd_theta(bits.ev);
  Eigen::Index variance = -1;
  const auto locations = bits.ev.param_locations();
  for (std::size_t k = 0; k < locations.size(); ++k) {
    if (locations[k].mat == magmaan::model::MatId::Theta &&
        locations[k].row == locations[k].col) {
      variance = static_cast<Eigen::Index>(k);
      break;
    }
  }
  REQUIRE(variance >= 0);
  theta(variance) = 0.0;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(theta.size());
  gradient(variance) = 2.0;
  FitDiagnostics d;
  audit_full_model_fit(d, theta, gradient, 1.0, 1.0,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  CHECK(common_fit_verdict(d).status == FitCheck::Failed);
  CHECK(common_fit_verdict(d).domain == StationarityDomain::Ambient);
  audit_full_model_fit(d, theta, gradient, 1.0, 1.0,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{},
                       StationarityDomain::Psd);
  CHECK(common_fit_verdict(d).status == FitCheck::Passed);
  theta(variance) = -0.1;
  gradient.setZero();
  audit_full_model_fit(d, theta, gradient, 1.0, 1.0,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{},
                       StationarityDomain::Psd);
  CHECK(common_fit_verdict(d).status == FitCheck::Failed);
  audit_full_model_fit(d, theta, gradient, 1.0, 1.0,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  CHECK(common_fit_verdict(d).status == FitCheck::Passed);
  CHECK_FALSE(d.geometric_stationarity.covariance_feasible);
}

TEST_CASE("common fit verdict normalizes total objectives and gradients together") {
  using namespace magmaan::estimate;
  auto bits = build_bits("f =~ x1 + x2 + x3");
  const Eigen::VectorXd theta = pd_theta(bits.ev);
  const Eigen::VectorXd gradient = Eigen::VectorXd::Constant(theta.size(), 1e-5);
  FitDiagnostics reference;
  audit_full_model_fit(reference, theta, gradient, 0.25, 0.25,
                       *bits.pt, bits.ev, bits.con, bits.nl, Bounds{});
  for (double n : {1.0, 1000.0}) {
    FitDiagnostics total;
    audit_full_model_fit(total, theta, n * gradient, n * 0.25, n * 0.25,
                         *bits.pt, bits.ev, bits.con, bits.nl, Bounds{},
                         StationarityDomain::Ambient, {}, 1.0 / n);
    CHECK(common_fit_verdict(total).status == FitCheck::Passed);
    CHECK(total.objective.multiplier == doctest::Approx(1.0 / n));
    CHECK(total.objective.reported == doctest::Approx(reference.objective.reported));
    CHECK(total.geometric_stationarity.ambient_residual_l2 ==
          doctest::Approx(reference.geometric_stationarity.ambient_residual_l2));
  }
}


TEST_CASE("admissibility oracle: keyed complete-data improper and proper endpoints") {
  std::ifstream input(std::string(MAGMAAN_FIXTURES_DIR) +
                      "/admissibility/reference.json");
  REQUIRE(input.is_open());
  std::stringstream buffer;
  buffer << input.rdbuf();
  const auto fixture = nlohmann::json::parse(buffer.str(), nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());
  CHECK(fixture["provenance"]["lavaan_version"] == "0.7.2");
  CHECK(fixture["options"]["estimator"] == "ML");
  CHECK(fixture["options"]["sample.nobs"] == 500);
  CHECK(fixture["options"]["meanstructure"] == false);
  CHECK(fixture["options"]["sample.cov.rescale"] == false);
  CHECK(fixture["options"]["std.lv"] == false);
  REQUIRE(fixture["cases"].size() == 2u);
  for (const auto& oracle : fixture["cases"]) {
    const bool improper = oracle["case_id"] == "improper";
    CAPTURE(improper);
    // These are retained lavaan backend statuses, not a C++ optimizer claim.
    CHECK(oracle["converged"] == true);
    CHECK(oracle["post_check"] == !improper);
    CHECK(oracle["warnings"].empty() == !improper);
    if (improper) {
      CHECK(oracle["warnings"][0].get<std::string>().find(
                "estimated ov variances are negative") != std::string::npos);
    }
    auto parsed = Parser::parse(oracle["model"].get<std::string>());
    REQUIRE(parsed.has_value());
    magmaan::spec::LatentNames names;
    magmaan::spec::Starts starts;
    auto structure = build(*parsed, {}, &starts, &names);
    REQUIRE(structure.has_value());
    auto rep = magmaan::model::build_matrix_rep(*structure);
    REQUIRE(rep.has_value());
    auto evaluator = ModelEvaluator::build(*structure, *rep);
    REQUIRE(evaluator.has_value());
    auto constraints = build_eq_constraints(*structure);
    REQUIRE(constraints.has_value());
    auto nonlinear = build_nl_constraints(*structure);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(evaluator->n_free()));
    std::size_t marker_residual_row = structure->free.size();
    REQUIRE(oracle["parameters"].size() == structure->free.size());
    for (std::size_t row = 0; row < structure->free.size(); ++row) {
      const auto op = magmaan::parse::to_string(structure->op[row]);
      const nlohmann::json* matched = nullptr;
      for (const auto& parameter : oracle["parameters"]) {
        if (parameter["lhs"] == names.row_lhs[row] &&
            parameter["op"] == op && parameter["rhs"] == names.row_rhs[row] &&
            parameter["group"] == structure->group[row]) {
          REQUIRE(matched == nullptr);
          matched = &parameter;
        }
      }
      REQUIRE(matched != nullptr);
      const double estimate = (*matched)["est"].get<double>();
      CHECK(((*matched)["free"].get<int>() > 0) == (structure->free[row] > 0));
      if (structure->free[row] > 0) {
        theta(structure->free[row] - 1) = estimate;
      } else {
        CHECK(estimate == doctest::Approx(structure->fixed_value[row]));
      }
      // Independent analytic expectations by variable identity, never row order.
      double expected = 0.0;
      if (op == "=~") {
        const int index = names.row_rhs[row].back() - '1';
        expected = oracle["analytic"]["lambda"][static_cast<std::size_t>(index)].get<double>();
      } else if (names.row_lhs[row] == "f") {
        expected = oracle["analytic"]["psi"].get<double>();
      } else {
        const int index = names.row_lhs[row].back() - '1';
        expected = oracle["analytic"]["residual"][static_cast<std::size_t>(index)].get<double>();
      }
      CHECK(std::abs(estimate - expected) < 2e-7);
      if (names.row_lhs[row] == "x1" && op == "~~" &&
          names.row_rhs[row] == "x1") marker_residual_row = row;
    }
    REQUIRE(marker_residual_row < structure->free.size());
    CHECK(theta(structure->free[marker_residual_row] - 1) ==
          doctest::Approx(improper ? -17.0 / 15.0 : 0.25).epsilon(2e-7));
    auto implied = evaluator->sigma(theta);
    REQUIRE(implied.has_value());
    REQUIRE(implied->sigma.size() == 1u);
    const auto& sigma = implied->sigma[0];
    CHECK(Eigen::LLT<Eigen::MatrixXd>(sigma).info() == Eigen::Success);
    for (Eigen::Index r = 0; r < 3; ++r) {
      for (Eigen::Index c = 0; c < 3; ++c) {
        CHECK(std::abs(sigma(r, c) - oracle["implied_cov"][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>()) < 1e-12);
        CHECK(std::abs(sigma(r, c) - oracle["sample_cov"][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>()) < 2e-7);
      }
    }
    CHECK(std::abs(oracle["criterion"]["value"].get<double>()) < 1e-12);
    const auto diagnostics = finalize_fit_diagnostics(
        theta, *structure, *evaluator, *constraints, nonlinear, Bounds{});
    CHECK(diagnostics.sigma_pd_all);
    CHECK(diagnostics.admissibility.checked);
    CHECK(diagnostics.admissibility.implied_sigma_pd);
    CHECK(diagnostics.admissibility.admissible == !improper);
    CHECK(diagnostics.admissibility.covariance_matrices_psd == !improper);
    REQUIRE(diagnostics.admissibility.theta_blocks.size() == 1u);
    REQUIRE(diagnostics.admissibility.psi_blocks.size() == 1u);
    CHECK(diagnostics.admissibility.psi_blocks[0].psd);
    const auto& residual_block = diagnostics.admissibility.theta_blocks[0];
    CHECK(residual_block.psd == !improper);
    if (improper) {
      REQUIRE(residual_block.negative_variance_rows.size() == 1u);
      CHECK(residual_block.negative_variance_rows[0] == marker_residual_row);
    } else {
      CHECK(residual_block.negative_variance_rows.empty());
    }
  }
}
