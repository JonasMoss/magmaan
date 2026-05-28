#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <Eigen/Core>

#include <string>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/optim/problem.hpp"

using magmaan::estimate::Backend;
using magmaan::estimate::Bounds;
using magmaan::estimate::IrlsOptions;
using magmaan::optim::OptimStatus;
using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::build;

// ============================================================================
// IRLS-ML cross-check: the iteratively-reweighted-GLS path (Fisher scoring,
// the inner step solves a GLS subproblem at W(θ_k) = ½D'(Σ⁻¹⊗Σ⁻¹)D) must
// reach the same ML optimum as direct minimisation of F_ML via L-BFGS. The
// algorithm differs, the optimum does not.
// ============================================================================

namespace {

// 1F CFA covariance perturbed off the model manifold so F_ML > 0 at the
// optimum (the off-manifold residual is what makes the cross-check non-
// trivial — at F_ML = 0 every solver is correct by chance).
Eigen::Matrix4d make_offmanifold_S() {
  Eigen::Vector4d lambda(1.0, 0.9, 0.8, 0.7);
  const double    psi = 1.5;
  Eigen::Vector4d theta(0.5, 0.6, 0.55, 0.7);
  Eigen::Matrix4d S = lambda * lambda.transpose() * psi +
                      theta.asDiagonal().toDenseMatrix();
  S(0, 1) += 0.05;  S(1, 0) += 0.05;
  S(2, 3) += 0.04;  S(3, 2) += 0.04;
  return S;
}

}  // namespace

TEST_CASE("IRLS-ML cross-check: matches NLopt L-BFGS on a 1F CFA fit") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_lbfgs = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                             Backend::NloptLbfgs);
  auto est_irls  = magmaan::estimate::fit_ml_irls(*pt, *mr, samp, *x0,
                                                  Bounds{}, Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_irls.has_value());

  CHECK(est_lbfgs->fmin > 0.0);
  CHECK(est_irls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-5));
  CHECK((est_irls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
  // The audit should certify the IRLS terminal iterate as stationary at the
  // lavaan-default `optim.dx.tol = 1e-3` threshold (Absolute mode default).
  CHECK(est_irls->audit.stationary);
}

TEST_CASE("IRLS-ML cross-check: matches NLopt L-BFGS with a mean structure") {
  // Same 1F CFA, but with free indicator intercepts. The mean structure
  // exercises the Σ-block and μ-block jointly in the Fisher weight
  // `normal_theory_weight` builds (cov block weighted by ½D'(Σ⁻¹⊗Σ⁻¹)D,
  // mean block weighted by S⁻¹).
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                          "x1 ~ 1\nx2 ~ 1\nx3 ~ 1\nx4 ~ 1");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  Eigen::Vector4d mean(4.0, 5.0, 6.0, 5.5);
  samp.mean  = {mean};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_lbfgs = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                             Backend::NloptLbfgs);
  auto est_irls  = magmaan::estimate::fit_ml_irls(*pt, *mr, samp, *x0,
                                                  Bounds{}, Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_irls.has_value());

  CHECK(est_irls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-5));
  CHECK((est_irls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
}

TEST_CASE("IRLS-ML-SNLLS cross-check: matches NLopt L-BFGS on a 1F CFA fit") {
  // Same off-manifold 1F CFA as the full-GLS IRLS cross-check. Golub–Pereyra
  // profiles α (Θ, Ψ) out; the outer optimizer drives β (Λ) only. Outer Fisher
  // reweight on Σ(θ) sits unchanged on top.
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_lbfgs = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                             Backend::NloptLbfgs);
  auto est_irls  = magmaan::estimate::fit_ml_irls_snlls(*pt, *mr, samp, *x0,
                                                       Bounds{},
                                                       Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_irls.has_value());

  CHECK(est_lbfgs->fmin > 0.0);
  CHECK(est_irls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-5));
  CHECK((est_irls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
  CHECK(est_irls->audit.stationary);
  // SNLLS telemetry must surface — n_nonlinear / n_linear from the last
  // inner GP solve.
  CHECK(est_irls->n_nonlinear >= 0);
  CHECK(est_irls->n_linear >= 0);
  CHECK(est_irls->n_nonlinear + est_irls->n_linear ==
        static_cast<std::int32_t>(est_irls->theta.size()));
}

TEST_CASE("IRLS-ML-SNLLS cross-check: matches NLopt L-BFGS with a mean structure") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                          "x1 ~ 1\nx2 ~ 1\nx3 ~ 1\nx4 ~ 1");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  Eigen::Vector4d mean(4.0, 5.0, 6.0, 5.5);
  samp.mean  = {mean};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_lbfgs = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                             Backend::NloptLbfgs);
  auto est_irls  = magmaan::estimate::fit_ml_irls_snlls(*pt, *mr, samp, *x0,
                                                       Bounds{},
                                                       Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_irls.has_value());

  CHECK(est_irls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-5));
  CHECK((est_irls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
}

TEST_CASE("IRLS-ML mean structure: latent mean coupling matches NLopt L-BFGS") {
  // Unlike the free-indicator-intercept case above, `f ~ 1` leaves the mean
  // residual coupled to loadings and the latent mean. The IRLS inner covariance
  // target must therefore include the frozen outer d d' term; otherwise the
  // inner GLS score disagrees with the ML score away from the optimum.
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                          "f ~ 1");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  Eigen::Vector4d mean(2.0, -0.2, 1.4, 0.7);
  samp.mean  = {mean};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_lbfgs = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                             Backend::NloptLbfgs);
  auto est_irls  = magmaan::estimate::fit_ml_irls(*pt, *mr, samp, *x0,
                                                  Bounds{}, Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_irls.has_value());

  CHECK(est_irls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-5));
  CHECK((est_irls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
  CHECK(est_irls->audit.stationary);
}

TEST_CASE("IRLS-ML-SNLLS == IRLS-ML at the optimum (separability cross-check)") {
  // On a separable model the Golub–Pereyra reduction shouldn't shift the
  // fixed point of the outer reweight — full-GLS inner and SNLLS-GLS inner
  // must reach the same θ̂ to numerical precision. Ernst-shaped two-factor
  // SEM is the natural target.
  auto fp = Parser::parse("f1 =~ x1 + x2 + x3\n"
                          "f2 =~ x4 + x5 + x6\n"
                          "f2 ~ f1");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  // Ernst-style population Σ for the two-factor model.
  Eigen::VectorXd lam(6);
  lam << 1.0, 0.8, 0.6, 1.0, 0.8, 0.6;
  // f1 var = 1, f2 = β·f1 + ε ⇒ Var(f2) = β² + Var(ε); set β = 0.25.
  const double beta = 0.25;
  const double var_eps = 1.0 - beta * beta;  // standardize Var(f2) = 1.
  Eigen::MatrixXd phi(2, 2);
  phi(0, 0) = 1.0;            phi(0, 1) = beta;
  phi(1, 0) = beta;           phi(1, 1) = beta * beta + var_eps;
  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(6, 2);
  Lambda.block(0, 0, 3, 1) = lam.head(3);
  Lambda.block(3, 1, 3, 1) = lam.tail(3);
  Eigen::MatrixXd Theta = Eigen::MatrixXd::Identity(6, 6) * 0.5;
  Eigen::MatrixXd S = Lambda * phi * Lambda.transpose() + Theta;
  SampleStats samp;
  samp.S     = {S};
  samp.n_obs = {500};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_full  = magmaan::estimate::fit_ml_irls(*pt, *mr, samp, *x0, Bounds{},
                                                  Backend::PortNls);
  auto est_snlls = magmaan::estimate::fit_ml_irls_snlls(*pt, *mr, samp, *x0,
                                                       Bounds{},
                                                       Backend::PortNls);
  REQUIRE(est_full.has_value());
  REQUIRE(est_snlls.has_value());

  CHECK(est_snlls->fmin ==
        doctest::Approx(est_full->fmin).epsilon(1e-5));
  CHECK((est_snlls->theta - est_full->theta).cwiseAbs().maxCoeff() < 5e-3);
}

TEST_CASE("Fisher-ML reaches the direct ML optimum on the Ernst n=25 replicate") {
  auto fp = Parser::parse("f1 =~ x1 + x2 + x3\n"
                          "f2 =~ x4 + x5 + x6\n"
                          "f2 ~ f1");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  Eigen::Matrix<double, 6, 6> S;
  S << 2.5661681014360447, 1.0123531463435886, 0.9068012773014319,
       0.67423897116130305, 0.051923154768001037, 0.54705557103799585,
       1.0123531463435886, 1.8421060208134847, 0.27997096108442582,
       0.9510778710066834, 0.48357874457719119, 0.72748489105184444,
       0.9068012773014319, 0.27997096108442582, 1.1331147592015471,
       0.083426833652732499, 0.0091395654434734477,
       -0.36498061246352559,
       0.67423897116130305, 0.9510778710066834, 0.083426833652732499,
       1.9310457158587859, 1.4635971252541156, 0.94572320314657965,
       0.051923154768001037, 0.48357874457719119, 0.0091395654434734477,
       1.4635971252541156, 2.4604609182382076, 0.53745781007679938,
       0.54705557103799585, 0.72748489105184444, -0.36498061246352559,
       0.94572320314657965, 0.53745781007679938, 1.5458629798366414;
  SampleStats samp;
  samp.S     = {S};
  samp.n_obs = {25};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 200;
  opts.ftol     = 1e-12;
  opts.gtol     = 1e-8;
  auto est_ref = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                           Backend::NloptLbfgs, opts);
  auto est_fisher = magmaan::estimate::fit_ml_fisher(*pt, *mr, samp, *x0,
                                                     Bounds{}, opts);
  auto est_fisher_snlls = magmaan::estimate::fit_ml_fisher_snlls(
      *pt, *mr, samp, *x0, Bounds{}, opts);
  REQUIRE(est_ref.has_value());
  REQUIRE(est_fisher.has_value());
  REQUIRE(est_fisher_snlls.has_value());

  CHECK(est_fisher->fmin ==
        doctest::Approx(est_ref->fmin).epsilon(1e-5));
  CHECK((est_fisher->theta - est_ref->theta).cwiseAbs().maxCoeff() < 1e-2);
  CHECK(est_fisher->audit.stationary);
  CHECK(est_fisher_snlls->fmin ==
        doctest::Approx(est_ref->fmin).epsilon(1e-5));
  CHECK((est_fisher_snlls->theta - est_ref->theta).cwiseAbs().maxCoeff() <
        1e-2);
  CHECK(est_fisher_snlls->audit.stationary);
  CHECK(est_fisher_snlls->n_nonlinear >= 0);
  CHECK(est_fisher_snlls->n_linear > 0);
  CHECK(est_fisher_snlls->n_nonlinear + est_fisher_snlls->n_linear ==
        static_cast<std::int32_t>(est_fisher_snlls->theta.size()));
}

TEST_CASE("IRLS-ML honors shared-label equality constraints") {
  auto fp = Parser::parse("f =~ x1 + a*x2 + a*x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());

  auto est_lbfgs = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                             Backend::NloptLbfgs);
  auto est_irls  = magmaan::estimate::fit_ml_irls(*pt, *mr, samp, *x0,
                                                  Bounds{}, Backend::PortNls);
  REQUIRE(est_lbfgs.has_value());
  REQUIRE(est_irls.has_value());

  CHECK(est_irls->fmin ==
        doctest::Approx(est_lbfgs->fmin).epsilon(1e-5));
  CHECK((est_irls->theta - est_lbfgs->theta).cwiseAbs().maxCoeff() < 5e-3);
  CHECK(est_irls->diagnostics.lin_eq_satisfied);
  CHECK(est_irls->audit.stationary);
}

TEST_CASE("IRLS-ML-SNLLS rejects box bounds explicitly") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());
  auto bounds = magmaan::estimate::variance_bounds(*pt);
  REQUIRE(bounds.has_value());

  auto est = magmaan::estimate::fit_ml_irls_snlls(
      *pt, *mr, samp, *x0, *bounds, Backend::PortNls);
  REQUIRE_FALSE(est.has_value());
  CHECK(est.error().detail.find("box bounds") != std::string::npos);
}

TEST_CASE("IRLS-ML reports converged when a zero-budget start is stationary") {
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());
  auto est_ref = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                           Backend::NloptLbfgs);
  REQUIRE(est_ref.has_value());

  IrlsOptions iopts;
  iopts.max_outer = 0;
  auto est_irls = magmaan::estimate::fit_ml_irls(
      *pt, *mr, samp, est_ref->theta, Bounds{}, Backend::PortNls, {}, iopts);
  REQUIRE(est_irls.has_value());

  CHECK(est_irls->optimizer_status == OptimStatus::Converged);
  CHECK(est_irls->audit.stationary);
  CHECK(est_irls->iterations == 0);
}

TEST_CASE("IRLS-ML converges from an intentionally bad start") {
  // Same problem as the first cross-check, but starting from a parameter
  // vector well away from the simple start. IRLS's Armijo backtracking on
  // F_ML must still reach the L-BFGS optimum.
  auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  SampleStats samp;
  samp.S     = {make_offmanifold_S()};
  samp.n_obs = {300};

  // Reference fit from the canonical start.
  auto x0 = magmaan::estimate::simple_start_values(*pt, *mr, samp, {});
  REQUIRE(x0.has_value());
  auto est_ref = magmaan::estimate::fit_ml(*pt, *mr, samp, *x0, Bounds{},
                                           Backend::NloptLbfgs);
  REQUIRE(est_ref.has_value());

  // Perturb every coordinate by a hefty additive offset — still PD-compatible
  // (loadings/variances stay positive) but well off the simple start.
  Eigen::VectorXd x_bad = *x0;
  for (Eigen::Index k = 0; k < x_bad.size(); ++k) {
    x_bad(k) = x_bad(k) * 0.4 + 0.3;
  }

  IrlsOptions iopts;
  iopts.max_outer = 100;
  auto est_irls = magmaan::estimate::fit_ml_irls(
      *pt, *mr, samp, x_bad, Bounds{}, Backend::PortNls, {}, iopts);
  REQUIRE(est_irls.has_value());

  CHECK(est_irls->fmin ==
        doctest::Approx(est_ref->fmin).epsilon(1e-4));
  CHECK((est_irls->theta - est_ref->theta).cwiseAbs().maxCoeff() < 1e-2);
}
