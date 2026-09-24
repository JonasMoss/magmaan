#include <doctest/doctest.h>
#include "../test_fit.hpp"

#ifdef MAGMAAN_WITH_CERES

#include <random>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::estimate::Backend;
using magmaan::estimate::Bounds;
using magmaan::optim::OptimOptions;
using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::build;

// ============================================================================
// End-to-end tests for the moment-quadratic LS path (`fit_gmm` / `fit_gls`).
// Covers:
//
//   • LS-aware Ceres backend recovers ground-truth on a 1F-feasible cov.
//   • Ceres / NLopt L-BFGS θ̂ parity (cross-backend).
//   • Active equality constraints (shared loadings) compose with auto-bounds
//     on residual variances — both constraints respected at the optimum.
// ============================================================================

namespace {

Eigen::Matrix3d make_1f_S() {
  Eigen::Vector3d lambda_true(1.0, 0.85, 0.70);
  const double    psi_true = 2.0;
  Eigen::Vector3d theta_true(0.6, 0.5, 0.8);
  return lambda_true * lambda_true.transpose() * psi_true +
         theta_true.asDiagonal().toDenseMatrix();
}

}  // namespace

TEST_CASE("LS path: ULS / Ceres recovers θ̂ on a 1F-feasible cov") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto est = magmaan::test::fit_gmm(*pt, *mr, samp, {}, Bounds{},
                                    Backend::Ceres);
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);

  // Σ(θ̂) should reproduce S to high precision (the manifold is dim-saturated).
  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr).value();
  auto sm = ev.sigma(est->theta).value();
  const double max_resid = (samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff();
  CHECK(max_resid < 1e-5);
}

TEST_CASE("LS path: Ceres / NLopt L-BFGS θ̂ parity on a 1F-feasible cov") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  // NLopt L-BFGS on ULS still needs the shallow-LS tolerance combo from uls_test.
  const OptimOptions nlopt_opts{.max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9};

  auto est_ceres = magmaan::test::fit_gmm(*pt, *mr, samp, {}, Bounds{},
                                          Backend::Ceres);
  auto est_lb    = magmaan::test::fit_gmm(*pt, *mr, samp, {}, Bounds{},
                                          Backend::NloptLbfgs, nlopt_opts);
  REQUIRE(est_ceres.has_value());
  REQUIRE(est_lb.has_value());

  // Both backends sit on the same in-manifold optimum (F = 0). θ̂ can differ
  // up to the manifold's degenerate directions; check that *Σ̂* matches.
  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr).value();
  auto sm_c = ev.sigma(est_ceres->theta).value();
  auto sm_l = ev.sigma(est_lb->theta).value();
  const double diff = (sm_c.sigma[0] - sm_l.sigma[0]).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-4);
}

TEST_CASE("LS path: GLS / Ceres recovers Sigma on a 1F-feasible cov") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto est = magmaan::test::fit_gls(*pt, *mr, samp, Bounds{}, Backend::Ceres);
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);

  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr).value();
  auto sm = ev.sigma(est->theta).value();
  CHECK((samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff() < 1e-5);
}

TEST_CASE("LS path: WLS / Ceres recovers Sigma on a 1F-feasible cov") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  magmaan::estimate::gmm::Weight wls{Eigen::MatrixXd::Identity(6, 6)};
  auto est = magmaan::test::fit_gmm(*pt, *mr, samp, wls, Bounds{},
                                    Backend::Ceres);
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);

  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr).value();
  auto sm = ev.sigma(est->theta).value();
  CHECK((samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff() < 1e-5);
}

TEST_CASE("LS path: equality constraints + bounds compose (shared loadings + "
          "Heywood-prone S)") {
  // Shared label `a` forces λ_2 = λ_3. Auto-bounds put `θ_resid ≥ 0`. The
  // S below is random PD (outside the 1F-with-equal-loadings manifold), so
  // ULS unbounded would Heywood-drift; with bounds, residuals pin to ≥ 0,
  // and the equality must hold to within `tol_eq`.
  auto fp = Parser::parse("f =~ x1 + a*x2 + a*x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  std::mt19937 rng(2027);
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(3, 3);
  for (Eigen::Index i = 0; i < 3; ++i)
    for (Eigen::Index j = 0; j < 3; ++j) A(i, j) = d(rng);
  Eigen::MatrixXd S = A * A.transpose() + 3.0 * Eigen::MatrixXd::Identity(3, 3);

  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {301};

  auto est = magmaan::test::fit_gmm(*pt, *mr, samp, {}, Bounds{},
                                    Backend::Ceres);
  if (!est.has_value()) {
    MESSAGE("eq+bounds fit failed: kind="
            << static_cast<int>(est.error().kind)
            << " detail=" << est.error().detail);
  }
  REQUIRE(est.has_value());
  CHECK(std::isfinite(est->fmin));

  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr).value();
  const auto locs = ev.param_locations();
  std::vector<Eigen::Index> lam_idx;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Lambda) {
      lam_idx.push_back(static_cast<Eigen::Index>(k));
    }
  }
  // With a shared `a` label the two free Lambda rows collapse to ONE free θ
  // index (the eq_groups merge); equality is then trivially satisfied.
  if (lam_idx.size() == 2) {
    CHECK(est->theta(lam_idx[0]) ==
          doctest::Approx(est->theta(lam_idx[1])).epsilon(1e-6));
  }

  // Auto-bounds: residual-variance lower bound (≥ 0).
  std::vector<Eigen::Index> th_idx;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Theta &&
        locs[k].row == locs[k].col) {
      th_idx.push_back(static_cast<Eigen::Index>(k));
    }
  }
  REQUIRE(!th_idx.empty());
  for (auto idx : th_idx) {
    CHECK(est->theta(idx) >= -1e-12);
  }
}

#endif  // MAGMAAN_WITH_CERES
