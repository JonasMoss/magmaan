#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::estimate::Backend;
using magmaan::estimate::Bounds;
using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::build;

// ============================================================================
// Optimizer cross-check: re-fit the same ML problem with optimizers of
// different algorithm classes and confirm they land on the same optimum.
//
//   • L-BFGS       — line search (the default; validated against lavaan by
//                    the parity golden tests).
//   • Port         — PORT drmngb model-Hessian trust region (= R nlminb,
//                    TOMS 611); when MAGMAAN_WITH_PORT.
//   • PortNls      — PORT drn2gb NL2SOL adaptive trust region (= R nls,
//                    TOMS 573); LS path only, when MAGMAAN_WITH_PORT.
//   • Nlopt SLSQP  — sequential quadratic programming (when MAGMAAN_WITH_NLOPT).
//
// L-BFGS is the reference because the lavaan-parity goldens already pin it to
// lavaan's nlminb result; if SLSQP and Port agree with L-BFGS, they agree
// with lavaan transitively. Tolerances are generous enough to absorb each
// solver's own default stopping criteria while still proving the optimizers
// reach the *same* minimum, not merely the same basin.
// ============================================================================

namespace {

// A 4-indicator covariance built from a 1-factor structure, then mildly
// perturbed off the 1-factor manifold so the ML fit has a genuine interior
// optimum (F_ML > 0, all variances positive — no Heywood, so an unbounded
// fit is well-posed).
Eigen::Matrix4d make_offmanifold_S() {
  Eigen::Vector4d lambda(1.0, 0.9, 0.8, 0.7);
  const double    psi = 1.5;
  Eigen::Vector4d theta(0.5, 0.6, 0.55, 0.7);
  Eigen::Matrix4d S = lambda * lambda.transpose() * psi +
                      theta.asDiagonal().toDenseMatrix();
  S(0, 1) += 0.05;  S(1, 0) += 0.05;   // extra x1–x2 covariance
  S(2, 3) += 0.04;  S(3, 2) += 0.04;   // extra x3–x4 covariance
  return S;
}

}  // namespace

#ifdef MAGMAAN_WITH_PORT
TEST_CASE("optimizer cross-check: PORT drmngb matches L-BFGS on ML") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S = {make_offmanifold_S()};
  samp.n_obs = {300};

  // The PORT backend honors bounds natively; an empty Bounds means unbounded.
  auto est_lbfgs = magmaan::test::fit(*pt, *mr, samp, Bounds{}, Backend::Lbfgs);
  auto est_port  = magmaan::test::fit(*pt, *mr, samp, Bounds{}, Backend::Port);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_port.has_value());

  CHECK(est_lbfgs->fmin > 0.0);
  CHECK(est_port->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-3));
  CHECK((est_port->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
}
#endif  // MAGMAAN_WITH_PORT

#ifdef MAGMAAN_WITH_PORT
TEST_CASE("optimizer cross-check: PORT NL2SOL matches L-BFGS on a feasible ULS fit") {
  // Use a feasible-on-the-manifold S so the ULS optimum is unique and
  // gradient-based / Gauss-Newton solvers must agree. The off-manifold S
  // above creates a non-trivially-shaped ML problem with a single interior
  // optimum, which is fine for L-BFGS / Port / SLSQP — but PortNls's
  // Gauss-Newton-flavoured Hessian model is sensitive to the residual
  // structure, so we put it on the LS path where it belongs.
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  Eigen::Vector4d lambda(1.0, 0.9, 0.8, 0.7);
  const double    psi = 1.0;
  Eigen::Vector4d theta(0.5, 0.6, 0.55, 0.7);
  Eigen::Matrix4d S = lambda * lambda.transpose() * psi +
                      theta.asDiagonal().toDenseMatrix();
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  // ULS path: fit_gmm with empty weight ≡ ULS.
  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());
  auto est_lbfgs = magmaan::estimate::fit_gmm(
      *pt, *mr, samp, *x0, {}, {}, Backend::Lbfgs);
  auto est_port_nls = magmaan::estimate::fit_gmm(
      *pt, *mr, samp, *x0, {}, {}, Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_port_nls.has_value());

  CHECK(est_port_nls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-4));
  CHECK((est_port_nls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
}
#endif  // MAGMAAN_WITH_PORT

#ifdef MAGMAAN_WITH_NLOPT
TEST_CASE("optimizer dispatch: Backend::NloptBobyqa rejects empty bounds") {
  // BOBYQA requires *finite* bounds. The dispatcher (`optim::nlopt_bobyqa`)
  // enforces this with a clear error message; NLopt itself tolerates the
  // ±infinity sentinels opportunistically, so the contract lives at the
  // dispatcher layer, not the adapter.
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  Eigen::Matrix3d S;  S.setIdentity();  S *= 2.0;
  samp.S     = {S};
  samp.n_obs = {300};

  auto est = magmaan::test::fit(*pt, *mr, samp, Bounds{},
                                Backend::NloptBobyqa);
  REQUIRE_FALSE(est.has_value());
  CHECK(est.error().kind == magmaan::FitError::Kind::NumericIssue);
}

TEST_CASE("optimizer cross-check: NLopt gradient backends match L-BFGS on ML") {
  // Three new gradient-based NLopt algorithms (TNEWTON, VAR2, NLopt's own
  // LBFGS) all on the same unbounded ML problem. SLSQP is the existing
  // cross-check; the new ones add curvature-scheme diversity. BOBYQA is
  // skipped here because it requires finite bounds — see the dedicated
  // BOBYQA tests in tests/unit/nlopt_optimizer_test.cpp.
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S = {make_offmanifold_S()};
  samp.n_obs = {300};

  auto est_lbfgs = magmaan::test::fit(*pt, *mr, samp, Bounds{}, Backend::Lbfgs);
  REQUIRE(est_lbfgs.has_value());

  for (auto backend : {Backend::NloptSlsqp, Backend::NloptTnewton,
                       Backend::NloptVar2, Backend::NloptLbfgs}) {
    CAPTURE(static_cast<int>(backend));
    auto est = magmaan::test::fit(*pt, *mr, samp, Bounds{}, backend);
    REQUIRE(est.has_value());
    CHECK(est->fmin == doctest::Approx(est_lbfgs->fmin).epsilon(1e-3));
    CHECK((est->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
  }
}
#endif  // MAGMAAN_WITH_NLOPT
