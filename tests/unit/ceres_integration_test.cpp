#include <doctest/doctest.h>

#ifdef MAGMAAN_WITH_CERES

#include <random>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::estimate::Bounds;
using magmaan::optim::CeresBoundedOptimizer;
using magmaan::optim::CeresOptions;
using magmaan::optim::LbfgsBOptimizer;
using magmaan::optim::LbfgsBOptions;
using magmaan::data::SampleStats;
using magmaan::gls::ULS;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::lavaanify;

// ============================================================================
// `CeresBoundedOptimizer + ULS` end-to-end via the LS path. Both test cases
// below previously documented the single-residual `r₀ = √(2F+ε)` stall on
// shallow LS landscapes (rank-1 Jacobian → LM damping dominates → fit crawls
// from F ≈ 4 to F ≈ 0.26 over 5000 iterations). With the multi-residual
// `LsCostFunction` adapter — one residual per (block-weighted) vech entry,
// chained through `LsDiscrepancy::residuals` / `residual_jacobian` —
// `JᵀJ` is the natural Gauss–Newton normal matrix and LM converges cleanly.
// ML stays on the scalar (sqrt-trick) path; only LS-shape discrepancies
// route through `minimize_ls`.
// ============================================================================

namespace {

Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() +
         Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

}  // namespace

TEST_CASE("CeresBoundedOptimizer + ULS — multi-residual LS adapter converges "
          "on a saturated 1F-feasible cov") {
  // The cov side has 6 vech moments and 6 free params (3 λ + 1 ψ + 3 θ
  // diagonals, minus the marker λ_1 = 1 → 6 free), so the 1F manifold is
  // dim-saturated. `S` is constructed inside the manifold so the optimum is
  // F = 0; with the multi-residual LS path, LM lands on it.
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  Eigen::Vector3d lambda_true(1.0, 0.85, 0.70);
  const double    psi_true = 2.0;
  Eigen::Vector3d theta_true(0.6, 0.5, 0.8);
  Eigen::Matrix3d S =
      lambda_true * lambda_true.transpose() * psi_true +
      theta_true.asDiagonal().toDenseMatrix();

  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {301};

  CeresBoundedOptimizer opt;  // default opts; max_iter=500
  auto est_or = magmaan::estimate::fit_bounded(*pt, *mr, samp, Bounds{},
                                        ULS{}, opt);
  if (!est_or.has_value()) {
    MESSAGE("fit_bounded<ULS, CeresBoundedOptimizer> failed: "
            << "kind=" << static_cast<int>(est_or.error().kind)
            << " detail=" << est_or.error().detail);
  }
  REQUIRE(est_or.has_value());
  CHECK(est_or->fmin < 1e-10);
  // Auto-bounds put a lower bound of 0 on the residual variances; the bound
  // must hold at the optimum (within the optimizer's interior tolerance).
  // The last three free indices of the 1F model are the θ-residual
  // diagonals (per `bounds_from_partable`'s row order).
  CHECK(est_or->theta.tail(3).minCoeff() >= -1e-12);
}

TEST_CASE("LbfgsBOptimizer + ULS — Heywood-prone S: LS-adapter path honors "
          "bounds, fit converges to finite F") {
  // A random PD `S` outside the 1F manifold: ULS unconstrained would drift
  // θ_resid negative (Heywood case). The auto-derived lower bound of 0 on
  // residual variances should keep the fit on the boundary, with finite F.
  //
  // Runs via the LBFGS-B LS adapter rather than Ceres: at a Heywood-degenerate
  // optimum the loadings can grow unboundedly to compensate for a pinned
  // residual variance, which produces an arbitrarily ill-conditioned
  // Jacobian. LBFGS-B's projected-gradient path handles this gracefully
  // (the projection zeros the active-set direction); Ceres' LM termination
  // criteria can't easily distinguish "stuck at a bound" from "still making
  // progress" without an interior-point reformulation. Both paths route
  // through the LS adapter (`minimize_ls`); only the optimizer differs.
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(2026);
  Eigen::MatrixXd S = random_pd(rng, 3);

  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {301};

  // ULS landscape is shallow at convergence — match the existing ULS
  // bounded path's tolerance combo (see `uls_test.cpp` cov+mean recovery).
  LbfgsBOptimizer opt(LbfgsBOptions{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9});
  auto est_or = magmaan::estimate::fit_bounded(*pt, *mr, samp, Bounds{},
                                        ULS{}, opt);
  if (!est_or.has_value()) {
    MESSAGE("fit_bounded<ULS, LbfgsBOptimizer> failed: "
            << "kind=" << static_cast<int>(est_or.error().kind)
            << " detail=" << est_or.error().detail);
  }
  REQUIRE(est_or.has_value());
  CHECK(std::isfinite(est_or->fmin));
  CHECK(est_or->theta.tail(3).minCoeff() >= -1e-12);
}

#endif  // MAGMAAN_WITH_CERES
