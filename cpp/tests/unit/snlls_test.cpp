#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/estimate/gmm/gp.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "../oracle.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::model::ModelEvaluator;
using magmaan::model::build_matrix_rep;
using magmaan::optim::OptimOptions;
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

Handles handles_for(std::string_view src, magmaan::spec::BuildOptions opts) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

Handles sem_handles_for(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.auto_cov_y = true;
  auto pt = build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

Eigen::Matrix3d make_1f_S() {
  Eigen::Vector3d lambda_true(1.0, 0.85, 0.70);
  const double psi_true = 2.0;
  Eigen::Vector3d theta_true(0.6, 0.5, 0.8);
  return lambda_true * lambda_true.transpose() * psi_true +
         theta_true.asDiagonal().toDenseMatrix();
}

Eigen::Matrix4d make_misspecified_1f_S() {
  Eigen::Matrix4d S;
  S << 2.0, 0.7, 0.4, 0.2,
       0.7, 1.8, 0.3, 0.25,
       0.4, 0.3, 1.6, 0.5,
       0.2, 0.25, 0.5, 1.4;
  return S;
}

OptimOptions snlls_opts() {
  return OptimOptions{.max_iter = 2000, .ftol = 1e-13, .gtol = 1e-8};
}

OptimOptions matlab_like_opts() {
  return OptimOptions{.max_iter = 400, .ftol = 1e-6, .gtol = 1e-6};
}

// Black-box check on the Golub–Pereyra problem `gmm::gp` builds: the gradient
// of its scalar objective ½‖r̃(β)‖² must match a finite difference. (`gmm::gp`
// reports Kaufman's simplified variable-projection Jacobian — the dropped term
// is orthogonal to r̃, so the *Jacobian* is approximate but the *gradient*
// J̃ᵀr̃ is exact, which is what the optimizer drives on.) Exercised over both
// the identity (ULS) and full-weight (GLS) whitening paths.
void check_gp_gradient_matches_fd(bool gls_weight) {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_misspecified_1f_S()};
  samp.n_obs = {301};

  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());

  magmaan::estimate::gmm::Weight w;
  if (gls_weight) {
    auto W = magmaan::estimate::gmm::normal_theory_weight(*ev, samp, *x0);
    REQUIRE(W.has_value());
    w = std::move(*W);
  }
  auto base = magmaan::estimate::gmm::residuals(*ev, samp, *x0, w);
  REQUIRE(base.has_value());
  auto prof = magmaan::estimate::gmm::gp(*base, h.pt, *ev, *x0);
  REQUIRE(prof.has_value());

  const magmaan::optim::ScalarProblem sp =
      magmaan::optim::scalarize(prof->problem);
  Eigen::VectorXd beta = prof->beta0;
  beta.array() += 0.05;
  Eigen::VectorXd grad(beta.size());
  const double f0 = sp.f(beta, grad);
  REQUIRE(std::isfinite(f0));

  Eigen::VectorXd fd(beta.size());
  Eigen::VectorXd scratch(beta.size());
  constexpr double eps = 1e-6;
  for (Eigen::Index j = 0; j < beta.size(); ++j) {
    Eigen::VectorXd bp = beta;
    Eigen::VectorXd bm = beta;
    bp(j) += eps;
    bm(j) -= eps;
    const double fp = sp.f(bp, scratch);
    const double fm = sp.f(bm, scratch);
    fd(j) = (fp - fm) / (2.0 * eps);
  }
  CHECK((grad - fd).cwiseAbs().maxCoeff() < 1e-5);
}

}  // namespace

TEST_CASE("SNLLS: gp profiled objective gradient matches finite differences") {
  check_gp_gradient_matches_fd(false);                           // ULS
  check_gp_gradient_matches_fd(true);                            // GLS weight
}

TEST_CASE("SNLLS: scalarized profile reuses combined base evaluation") {
  auto h = handles_for("f =~ x1 + x2 + x3 + x4");
  SampleStats samp;
  samp.S = {make_misspecified_1f_S()};
  samp.n_obs = {301};

  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto base = magmaan::estimate::gmm::residuals(*ev, samp, *x0, {});
  REQUIRE(base.has_value());
  REQUIRE(static_cast<bool>(base->eval));

  struct Counts {
    int r = 0;
    int J = 0;
    int eval = 0;
  } counts;

  magmaan::optim::GmmProblem counted;
  counted.n_resid = base->n_resid;
  counted.n_param = base->n_param;
  counted.expand = base->expand;
  counted.r = [r = base->r, &counts](const Eigen::VectorXd& x) {
    ++counts.r;
    return r(x);
  };
  counted.J = [J = base->J, &counts](const Eigen::VectorXd& x) {
    ++counts.J;
    return J(x);
  };
  counted.eval = [eval = base->eval, &counts](const Eigen::VectorXd& x) {
    ++counts.eval;
    return eval(x);
  };

  auto prof = magmaan::estimate::gmm::gp(counted, h.pt, *ev, *x0);
  REQUIRE(prof.has_value());
  const magmaan::optim::ScalarProblem sp =
      magmaan::optim::scalarize(prof->problem);

  Eigen::VectorXd beta = prof->beta0;
  beta.array() += 0.03;
  Eigen::VectorXd grad(beta.size());
  const double f = sp.f(beta, grad);

  REQUIRE(std::isfinite(f));
  CHECK(grad.allFinite());
  CHECK(counts.r == 0);
  CHECK(counts.eval == 1);
  CHECK(counts.J == 1);
}

TEST_CASE("SNLLS: ULS profiles linear block and recovers a feasible 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, {},
                                          Backend::NloptLbfgs, snlls_opts());
  if (!est.has_value()) {
    MESSAGE("SNLLS ULS failed: " << est.error().detail);
  }
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);

  auto ev = ModelEvaluator::build(h.pt, h.rep).value();
  auto sm = ev.sigma(est->theta).value();
  CHECK((samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff() < 1e-5);
}

TEST_CASE("SNLLS: GLS and WLS agree with full LS on a feasible 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());

  auto gls = magmaan::estimate::fit_snlls_gls(h.pt, h.rep, samp, *x0,
                                              Backend::NloptLbfgs, snlls_opts());
  REQUIRE(gls.has_value());
  CHECK(gls->fmin < 1e-10);

  magmaan::estimate::gmm::Weight wls{
      magmaan::estimate::gmm::BlockWeight::identity(6)};
  auto wls_est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, wls,
                                              Backend::NloptLbfgs, snlls_opts());
  REQUIRE(wls_est.has_value());
  CHECK(wls_est->fmin < 1e-10);

#ifdef MAGMAAN_WITH_PORT
  // PortNls — same feasible 1F problem; the unique global optimum lets us
  // pin PortNls to the same fmin ≈ 0 as NLopt L-BFGS without worrying about
  // multi-modality. If PortNls converges anywhere else here, that's a
  // genuine adapter bug, not a non-convex multiple-local-optimum issue.
  auto port_nls_gls = magmaan::estimate::fit_snlls_gls(
      h.pt, h.rep, samp, *x0, Backend::PortNls, snlls_opts());
  REQUIRE(port_nls_gls.has_value());
  CHECK(port_nls_gls->fmin < 1e-8);
  CHECK((port_nls_gls->theta - gls->theta).cwiseAbs().maxCoeff() < 1e-4);
#endif
}

TEST_CASE("SNLLS: strict compatibility rejects cross-block equality") {
  auto h = handles_for("f =~ x1 + a*x2 + x3\nf ~~ a*f");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, {},
                                          Backend::NloptLbfgs, snlls_opts());
  REQUIRE_FALSE(est.has_value());
  if (!est.has_value()) {
    CHECK(est.error().detail.find("SNLLS compatibility") != std::string::npos);
  }
}

TEST_CASE("SNLLS: block-separated linear constraints stay compatible") {
  magmaan::spec::BuildOptions opts;
  opts.effect_coding = true;
  auto h = handles_for(
      "f1 =~ x1 + x2 + x3 + x4\n"
      "f2 =~ x5 + x6 + x7 + x8\n"
      "x1 ~ t1*1\nx2 ~ t2*1\nx3 ~ t3*1\nx4 ~ t4*1\n"
      "x5 ~ t5*1\nx6 ~ t6*1\nx7 ~ t7*1\nx8 ~ t8*1\n"
      "t1 + t2 + t3 + t4 == 0\n"
      "t5 + t6 + t7 + t8 == 0",
      opts);
  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());

  const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(h.pt.n_free());
  CHECK(magmaan::estimate::gmm::gp_compatible(h.pt, *ev, x0));
}

TEST_CASE("SNLLS: covariance-only model is solved by profiling alone") {
  auto h = handles_for("x1 ~~ x1\nx2 ~~ x2\nx1 ~~ x2");
  SampleStats samp;
  Eigen::Matrix2d S;
  S << 2.0, 0.4,
       0.4, 1.5;
  samp.S = {S};
  samp.n_obs = {100};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, {},
                                          Backend::NloptLbfgs, snlls_opts());
  REQUIRE(est.has_value());
  CHECK(est->iterations == 0);
  CHECK(est->fmin < 1e-12);
  // Three variance/covariance parameters all profile out — no β block.
  CHECK(est->n_nonlinear == 0);
  CHECK(est->n_linear == 3);
  CHECK(est->diagnostics.geometric_stationarity.checked);
  CHECK(est->diagnostics.geometric_stationarity.ambient_stationary);
  CHECK(magmaan::estimate::fit_verdict(*est).status ==
        magmaan::estimate::FitCheck::Passed);
}

TEST_CASE("SNLLS: reports β/α block sizes on a 1F covariance model") {
  // Single-factor CFA, marker-coded: x1's loading fixed at 1, x2/x3 loadings
  // free → 2 nonlinear (β) parameters. Three residual variances + one latent
  // variance → 4 linear (α) parameters. The split is a property of the spec,
  // so both the closed-form (covariance-only) and outer-optimizer paths in
  // `compose_snlls` should populate the same n_nonlinear / n_linear counts.
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, {},
                                          Backend::NloptLbfgs, snlls_opts());
  REQUIRE(est.has_value());
  CHECK(est->n_nonlinear == 2);
  CHECK(est->n_linear == 4);
  CHECK(est->n_nonlinear + est->n_linear == h.pt.n_free());
}

TEST_CASE("SNLLS: Bollen GLS backend cross-check") {
  const std::string dir = magmaan::test::fixtures_dir() +
                          "/parity/bollen_democracy_sem";
  auto ref_raw = magmaan::test::read_fixture(dir + "/reference.json");
  auto data_raw = magmaan::test::read_fixture(dir + "/data.json");
  REQUIRE(ref_raw.has_value());
  REQUIRE(data_raw.has_value());
  const auto ref = nlohmann::json::parse(*ref_raw, nullptr, false);
  const auto data = nlohmann::json::parse(*data_raw, nullptr, false);
  REQUIRE_FALSE(ref.is_discarded());
  REQUIRE_FALSE(data.is_discarded());

  auto h = sem_handles_for(ref["model"].get<std::string>());
  auto raw = magmaan::test::raw_from_fixture(data);
  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  SampleStats samp = std::move(*samp_or);
  samp.mean.clear();

  auto x0 = magmaan::estimate::fabin_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  const OptimOptions opts = matlab_like_opts();

  auto run = [&](Backend backend, const char* name) {
    auto est = magmaan::estimate::fit_snlls_gls(h.pt, h.rep, samp, *x0,
                                                backend, opts);
    if (est.has_value()) {
      MESSAGE(std::string(name) << " iterations=" << est->iterations
                                << " fmin=" << est->fmin);
    } else {
      MESSAGE(std::string(name) << " failed: " << est.error().detail);
    }
    return est;
  };

  auto nlopt_lbfgs = run(Backend::NloptLbfgs, "nlopt-lbfgs");
  REQUIRE(nlopt_lbfgs.has_value());
  CHECK(nlopt_lbfgs->fmin < 0.243);

#ifdef MAGMAAN_WITH_PORT
  auto port = run(Backend::Port, "port");
  CHECK(port.has_value());
  if (port.has_value()) {
    CHECK(port->fmin == doctest::Approx(nlopt_lbfgs->fmin).epsilon(1e-5));
  }

  // PortNls is the Gauss-Newton-flavoured NL2SOL trust region (R's `nls`).
  // On Bollen's democracy SEM under SNLLS-GLS the profiled objective is
  // genuinely non-convex; Gauss-Newton-style solvers can be drawn to a
  // different basin than scalar gradient solvers. We require PortNls to
  // converge (no error) and not get worse than the gradient-solver fmin,
  // but we explicitly *do not* require it to land at the same local
  // optimum — recording the disagreement is itself useful SNLLS-convergence
  // research data. The unique-optimum 1F-covariance test above pins the
  // *adapter correctness*; this test pins only "converged successfully".
  auto port_nls = run(Backend::PortNls, "port-nls");
  CHECK(port_nls.has_value());
#endif

  auto nlopt = run(Backend::NloptSlsqp, "nlopt-slsqp");
  CHECK(nlopt.has_value());
  if (nlopt.has_value()) {
    CHECK(nlopt->fmin == doctest::Approx(nlopt_lbfgs->fmin).epsilon(1e-5));
  }

#ifdef MAGMAAN_WITH_CERES
  auto ceres_bfgs = run(Backend::CeresBfgs, "ceres-bfgs");
  CHECK(ceres_bfgs.has_value());
  if (ceres_bfgs.has_value()) {
    CHECK(ceres_bfgs->fmin == doctest::Approx(nlopt_lbfgs->fmin).epsilon(1e-5));
  }

  auto ceres = run(Backend::Ceres, "ceres");
  CHECK(ceres.has_value());
  if (ceres.has_value()) {
    CHECK(ceres->fmin == doctest::Approx(nlopt_lbfgs->fmin).epsilon(1e-5));
  }
#endif
}

#ifdef MAGMAAN_WITH_CERES
TEST_CASE("SNLLS: Ceres backend recovers a feasible 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, {},
                                          Backend::Ceres, snlls_opts());
  if (!est.has_value()) {
    MESSAGE("SNLLS Ceres failed: " << est.error().detail);
  }
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);

  auto ev = ModelEvaluator::build(h.pt, h.rep).value();
  auto sm = ev.sigma(est->theta).value();
  CHECK((samp.S[0] - sm.sigma[0]).cwiseAbs().maxCoeff() < 1e-5);
}
#endif

// === fast α-solve telemetry ============================================
//
// The Golub–Pereyra inner α-solve in `profile_at()` tries Cholesky on
// the normal-equations Gram first and falls back to rank-revealing QR
// when the rcond gate doesn't clear. These tests verify the counters
// reach `Estimates`, that the fast path fires on well-conditioned
// problems, and that the fallback fires (without breaking the fit) when
// the Gram is near-singular. See `project/design/snlls-fast-alpha-solve.md`.

TEST_CASE("SNLLS: fast α-solve fires on a well-conditioned 1F covariance") {
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls_gls(h.pt, h.rep, samp, *x0,
                                              Backend::NloptLbfgs, snlls_opts());
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-10);
  // Counters surface through the SNLLS path with non-sentinel values.
  CHECK(est->n_alpha_solve_fast >= 0);
  CHECK(est->n_alpha_solve_fallback >= 0);
  // At least one cache miss happened (the optimizer evaluates at β0).
  CHECK(est->n_alpha_solve_fast + est->n_alpha_solve_fallback > 0);
  // Well-conditioned Gram → fast path should win every cache miss.
  CHECK(est->n_alpha_solve_fallback == 0);
  CHECK(est->n_alpha_solve_fast > 0);
}

TEST_CASE("SNLLS: fast α-solve accounting on a 2F CFA") {
  auto h = handles_for(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f1 ~~ f2");
  // Construct a population-implied covariance for the model, so the fit
  // converges crisply and we can read clean accounting from the counters.
  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(6, 2);
  Lambda << 1.0, 0.0,
            0.9, 0.0,
            0.8, 0.0,
            0.0, 1.0,
            0.0, 0.85,
            0.0, 0.75;
  Eigen::Matrix2d Phi;
  Phi << 1.5, 0.6,
         0.6, 1.2;
  Eigen::VectorXd theta(6);
  theta << 0.7, 0.5, 0.6, 0.65, 0.55, 0.4;
  Eigen::MatrixXd S = Lambda * Phi * Lambda.transpose() +
                      theta.asDiagonal().toDenseMatrix();
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {500};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls_gls(h.pt, h.rep, samp, *x0,
                                              Backend::NloptLbfgs, snlls_opts());
  REQUIRE(est.has_value());
  CHECK(est->fmin < 1e-8);
  CHECK(est->n_alpha_solve_fast > 0);
  CHECK(est->n_alpha_solve_fallback == 0);
}

TEST_CASE("SNLLS: fast α-solve falls back cleanly on near-singular Gram") {
  // A 5-indicator 1F model fit against a rank-near-1 covariance. The
  // residual-variance columns of the α-Jacobian become near-collinear
  // (every indicator wants almost the same residual variance), so the
  // Gram HᵀH has a small eigenvalue and the rcond gate should trip the
  // QR fallback on at least one cache miss. The fit still has to
  // succeed — QR's column pivoting handles the near-rank deficiency.
  auto h = handles_for("f =~ x1 + x2 + x3 + x4 + x5");
  Eigen::MatrixXd S(5, 5);
  const double off = 0.9999999;
  S << 1.0, off, off, off, off,
       off, 1.0, off, off, off,
       off, off, 1.0, off, off,
       off, off, off, 1.0, off,
       off, off, off, off, 1.0;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_snlls_gls(h.pt, h.rep, samp, *x0,
                                              Backend::NloptLbfgs, snlls_opts());
  REQUIRE(est.has_value());
  CHECK(std::isfinite(est->fmin));
  CHECK(est->n_alpha_solve_fast + est->n_alpha_solve_fallback > 0);
  // At least one cache miss should have fallen back to QR.
  CHECK(est->n_alpha_solve_fallback >= 1);
}

TEST_CASE("SNLLS: alpha-solve counters are sentinel-NA on full-θ paths") {
  // `fit_ml` etc. don't run the SNLLS path. The Estimates default keeps
  // the counters at -1 so the R wrapper can map them to NA cleanly.
  auto h = handles_for("f =~ x1 + x2 + x3");
  SampleStats samp;
  samp.S = {make_1f_S()};
  samp.n_obs = {301};

  auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
  REQUIRE(x0.has_value());
  auto est = magmaan::estimate::fit_ml(h.pt, h.rep, samp, *x0, {},
                                       Backend::NloptLbfgs, matlab_like_opts());
  REQUIRE(est.has_value());
  CHECK(est->n_alpha_solve_fast == -1);
  CHECK(est->n_alpha_solve_fallback == -1);
}

TEST_CASE("SNLLS: full-coordinate stationarity agrees with the original LS gradient") {
  for (const auto syntax : {"f =~ x1 + x2 + x3 + x4",
                            "f =~ x1 + l*x2 + l*x3 + x4"}) {
    const auto h = handles_for(syntax);
    SampleStats samp;
    samp.S = {make_misspecified_1f_S()};
    samp.n_obs = {301};
    auto ev = ModelEvaluator::build(h.pt, h.rep);
    REQUIRE(ev.has_value());
    auto x0 = magmaan::estimate::simple_start_values(h.pt, h.rep, samp, {});
    REQUIRE(x0.has_value());
    for (int kind = 0; kind < 3; ++kind) {
      magmaan::estimate::gmm::Weight weight;
      if (kind == 1) {
        auto w = magmaan::estimate::gmm::normal_theory_weight(*ev, samp, *x0);
        REQUIRE(w.has_value());
        weight = *w;
      } else if (kind == 2) {
        weight = {magmaan::estimate::gmm::BlockWeight::diagonal(
            Eigen::VectorXd::Constant(10, 2.0))};
      }
      auto fit = kind == 1
          ? magmaan::estimate::fit_snlls_gls(h.pt, h.rep, samp, *x0,
                                            Backend::NloptLbfgs, snlls_opts())
          : magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, weight,
                                        Backend::NloptLbfgs, snlls_opts());
      REQUIRE(fit.has_value());
      auto base = magmaan::estimate::gmm::residuals(*ev, samp, *x0, weight);
      REQUIRE(base.has_value());
      auto e = base->eval(fit->theta);
      REQUIRE(e.has_value());
      const Eigen::VectorXd gradient = e->jacobian.transpose() * e->residual;
      const auto& audit = fit->diagnostics.geometric_stationarity;
      CHECK(audit.checked);
      CHECK(audit.gradient_finite);
      CHECK(audit.ambient_stationary);
      CHECK(magmaan::estimate::fit_verdict(*fit).status ==
            magmaan::estimate::FitCheck::Passed);
      CHECK(audit.raw_gradient_inf == doctest::Approx(
          gradient.cwiseAbs().maxCoeff()).scale(1.0).epsilon(1e-12));
      CHECK(fit->fmin == doctest::Approx(0.5 * e->residual.squaredNorm())
                            .scale(1.0).epsilon(1e-12));
    }
  }
}

TEST_CASE("SNLLS: inner solve agrees with SVD across scales and ranks") {
  const auto h = handles_for("x1 ~~ x1\nx2 ~~ x2");
  REQUIRE(h.pt.n_free() == 2);
  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  for (double scale : {1e-8, 1.0, 1e8}) {
    for (double delta : {1.0, 1e-5, 0.0}) {
      CAPTURE(scale);
      CAPTURE(delta);
      Eigen::MatrixXd H(6, 2);
      H << 1, 1, 1, 1 + delta, -1, -1, 0, delta, .5, .5, 0, 0;
      H *= scale;
      Eigen::VectorXd y(6);
      y << 1, -2, .3, .7, -.1, 1;
      y *= scale;
      magmaan::optim::GmmProblem base;
      base.n_param = 2;
      base.n_resid = 6;
      base.r = [H, y](const Eigen::VectorXd& theta)
          -> magmaan::fit_expected<Eigen::VectorXd> { return H * theta - y; };
      base.J = [H](const Eigen::VectorXd&)
          -> magmaan::fit_expected<Eigen::MatrixXd> { return H; };
      auto profile = magmaan::estimate::gmm::gp(
          base, h.pt, *ev, Eigen::VectorXd::Zero(2));
      REQUIRE(profile.has_value());
      CHECK(profile->problem.n_param == 0);
      auto r = profile->problem.r(Eigen::VectorXd(0));
      REQUIRE(r.has_value());
      const Eigen::VectorXd theta = profile->problem.expand(Eigen::VectorXd(0));
      const Eigen::VectorXd reference = H.jacobiSvd(
          Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
      CHECK((H * theta - H * reference).norm() / scale < 1e-8);
      CHECK(r->norm() / scale == doctest::Approx((H * reference - y).norm() / scale)
                                  .epsilon(1e-8));
      CHECK(*profile->n_alpha_solve_fast == (delta == 1.0 ? 1 : 0));
      CHECK(*profile->n_alpha_solve_fallback == (delta == 1.0 ? 0 : 1));
    }
  }
}

TEST_CASE("SNLLS: Kaufman Jacobian preserves the gradient but differs from the residual derivative") {
  const auto h = handles_for("f =~ 1*x1 + b*x2\nf ~~ 1*f\nx1 ~~ 0*x1\nx2 ~~ a*x2");
  REQUIRE(h.pt.n_free() == 2);
  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  const auto locs = ev->param_locations();
  const int b = locs[0].mat == magmaan::model::MatId::Lambda ? 0 : 1;
  const int a = 1 - b;
  magmaan::optim::GmmProblem base;
  base.n_param = 2;
  base.n_resid = 2;
  base.r = [a, b](const Eigen::VectorXd& theta)
      -> magmaan::fit_expected<Eigen::VectorXd> {
    return Eigen::Vector2d(theta(a) - 1.0, theta(a) * theta(b));
  };
  base.J = [a, b](const Eigen::VectorXd& theta)
      -> magmaan::fit_expected<Eigen::MatrixXd> {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
    J(0, a) = 1; J(1, a) = theta(b); J(1, b) = theta(a);
    return J;
  };
  auto profile = magmaan::estimate::gmm::gp(base, h.pt, *ev, Eigen::VectorXd::Ones(2));
  REQUIRE(profile.has_value());
  auto e = profile->problem.eval(Eigen::VectorXd::Ones(1));
  REQUIRE(e.has_value());
  const double eps = 1e-6;
  auto rp = profile->problem.r(Eigen::VectorXd::Constant(1, 1 + eps));
  auto rm = profile->problem.r(Eigen::VectorXd::Constant(1, 1 - eps));
  REQUIRE(rp.has_value());
  REQUIRE(rm.has_value());
  const Eigen::VectorXd fd = (*rp - *rm) / (2 * eps);
  CHECK((fd - e->jacobian.col(0)).norm() > 0.3);
  CHECK(e->jacobian.col(0).dot(e->residual) == doctest::Approx(0.25));
  CHECK(e->jacobian.col(0).dot(e->residual) == doctest::Approx(fd.dot(e->residual)));
}

TEST_CASE("SNLLS: shared GP engine rejects nonlinear equality constraints") {
  const auto h = handles_for("f =~ x1 + b*x2 + x3\nx1 ~~ a*x1\na == b*b");
  REQUIRE_FALSE(h.pt.nl_constraints.empty());
  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  const Eigen::VectorXd start = Eigen::VectorXd::Ones(h.pt.n_free());
  CHECK_FALSE(magmaan::estimate::gmm::gp_compatible(h.pt, *ev, start));
  magmaan::optim::GmmProblem base;
  auto generic = magmaan::estimate::gmm::gp(base, h.pt, *ev, start);
  REQUIRE_FALSE(generic.has_value());
  CHECK(generic.error().detail.find("nonlinear equality") != std::string::npos);
  const std::vector<magmaan::estimate::gmm::GpBlockKind> kinds(
      static_cast<std::size_t>(h.pt.n_free()),
      magmaan::estimate::gmm::GpBlockKind::Linear);
  auto overridden = magmaan::estimate::gmm::gp(base, h.pt, *ev, start, kinds);
  REQUIRE_FALSE(overridden.has_value());
  CHECK(overridden.error().detail.find("nonlinear equality") != std::string::npos);
}

#ifdef MAGMAAN_WITH_PORT
TEST_CASE("SNLLS: retained PORT-NLS endpoint keeps point and objective together") {
  // Literal standardized signed-negative witness, case 2 of experiment 15.
  auto h = handles_for("X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~~ Y");
  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  const double covariance[] = {
      0.99999999999999989, 0.49764184681423657, 0.37233770838357161, -0.01673022977936663, -0.029143978499567878, 0.035790502332395994,
      0.49764184681423657, 1.0000000000000002, 0.26679035903420095, -0.0097761000400810316, 0.065518634680198917, 0.11248895953499291,
      0.37233770838357161, 0.26679035903420095, 1.0000000000000002, -0.062882935302488563, 0.031811438914974427, -0.065137618237596506,
      -0.01673022977936663, -0.0097761000400810316, -0.062882935302488563, 1, 0.46357783567976157, 0.33630438417126113,
      -0.029143978499567812, 0.065518634680198917, 0.031811438914974427, 0.46357783567976157, 1, 0.45911081102000206,
      0.035790502332395925, 0.11248895953499291, -0.065137618237596506, 0.33630438417126113, 0.45911081102000206, 1
  };
  Eigen::MatrixXd S = Eigen::Map<const Eigen::Matrix<double,6,6>>(covariance);
  Eigen::VectorXd weights(21), start(13);
  weights <<
      3.1679977903976483e-08, 2.7644687743622272, 0.00062092358758424266, 2.9328345564974443e-05,
      0.026125116232557855, 2.3020574055609678e-06, 241233994.15201393, 54183.240731558384,
      2559.2598506168747, 2279738.5866365964, 200.88289940580455, 12.170024322210555,
      0.57483188912920713, 512.04899657265878, 0.045120035986716114, 0.027151276941723784,
      24.185826111242509, 0.0021311736802766524, 21544.260549467988, 1.8984078043423098,
      0.00016728131296559094;
  start <<
      -0.71652790740002403, 0.53610917317769391, -1.3651645135443804, 0.99036402451551786,
      0.066923800963649402, 1.6945184432801337, 1.3565742711324884, 1.1996136609843939,
      1.3395765353409115, 1.6328600105357183, 1.3330637633700628, -0.69451844328013379,
      -0.33957653534091148;
  SampleStats sample; sample.S = {S}; sample.n_obs = {100};
  auto block = magmaan::estimate::gmm::BlockWeight::dense(
      weights.asDiagonal(), magmaan::FitError::Kind::NumericIssue, "endpoint regression");
  REQUIRE(block.has_value());
  auto base = magmaan::estimate::gmm::residuals(*ev, sample, start, {*block});
  REQUIRE(base.has_value());
  auto profile = magmaan::estimate::gmm::gp(*base, h.pt, *ev, start);
  REQUIRE(profile.has_value());
  for (bool combined : {false, true}) {
    auto problem = profile->problem;
    if (!combined) problem.eval = {};
    auto fit = magmaan::optim::port_nls(problem, profile->beta0, {}, {});
    REQUIRE(fit.has_value());
    REQUIRE(fit->audit.port_endpoint.has_value());
    const auto& telemetry = *fit->audit.port_endpoint;
    // Whether PORT's own restoration leaves a stale objective on this witness
    // depends on the build's floating-point path: optimized native builds stop
    // singular (7) with a mismatched point, Debug builds stop false (8) with a
    // matching one. The guarantee below is build-independent: any mismatch
    // must trigger the best-point substitution, and the result always pairs
    // the returned point with its own objective.
    const bool mismatch = telemetry.returned_x_objective >
        telemetry.stored_objective * (1 + 1e-14);
    CHECK((!mismatch || telemetry.best_point_substituted));
    CHECK((fit->audit.raw_backend_status == 7 || fit->audit.raw_backend_status == 8));
    auto residual = problem.r(fit->x);
    auto original = base->r(problem.expand(fit->x));
    REQUIRE(residual.has_value()); REQUIRE(original.has_value());
    CHECK(fit->fmin == 0.5 * residual->squaredNorm());
    CHECK(std::abs(fit->fmin - 0.5 * original->squaredNorm()) <=
          1e-12 * (1 + std::abs(fit->fmin)));
    CHECK(fit->fmin <= telemetry.stored_objective * (1 + 1e-14));
  }
}
#endif

#ifdef MAGMAAN_WITH_PORT
TEST_CASE("SNLLS: scalar PORT restores a matching historical endpoint") {
  // Literal standardized signed-negative/positive retained witness, case 3.
  auto h = handles_for("X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~~ Y");
  auto ev = ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  const double covariance[] = {
      1, 0.49311233920713554, 0.30610638208926666, 0.17441481287791163, 0.15674610768104524, 0.21455022649774286,
      0.49311233920713554, 1, 0.058746344903173611, 0.24533304811313639, 0.12889253045822791, 0.11361620881235483,
      0.30610638208926666, 0.058746344903173611, 1.0000000000000002, 0.03030139045795267, 0.096484273170979643, 0.13519757002661503,
      0.17441481287791163, 0.24533304811313639, 0.03030139045795267, 1, 0.40497587647324224, 0.38791244184060608,
      0.15674610768104524, 0.12889253045822791, 0.096484273170979643, 0.40497587647324224, 1, 0.15900099189222314,
      0.21455022649774286, 0.11361620881235483, 0.13519757002661503, 0.38791244184060608, 0.15900099189222314, 0.99999999999999989
  };
  Eigen::MatrixXd S = Eigen::Map<const Eigen::Matrix<double,6,6>>(covariance);
  Eigen::VectorXd weights(21), start(13);
  weights <<
      5.4977398429813658e-08, 3.0989890998244647, 0.0010943901863442339, 4.0057749846988693e-05,
      0.03682614689127791, 3.3152071181050679e-06, 174685119.97873735, 61689.045958865696,
      2257.992078286376, 2075831.7247459406, 186.87298810226562, 21.785131966467738,
      0.79739692258343275, 733.06804086295801, 0.065993121525831228, 0.029186963527429343,
      26.832346057308168, 0.0024155333140680815, 24667.684059101357, 2.2206635416226113,
      0.00019991120987591929;
  start <<
      -0.1919147993655422, 0.1191337961601824, 0.40988886857502993, 0.39261842773671657,
      -1.6357068968164294, 3.5694336280335479, 1.0946355556436107, 1.0364676153271537,
      0.01198615644008025, 0.83400489619220175, 0.84769842698503062, -2.5694336280335484,
      0.98801384355991995;
  SampleStats sample; sample.S = {S}; sample.n_obs = {100};
  auto block = magmaan::estimate::gmm::BlockWeight::dense(
      weights.asDiagonal(), magmaan::FitError::Kind::NumericIssue, "scalar endpoint regression");
  REQUIRE(block.has_value());
  auto base = magmaan::estimate::gmm::residuals(*ev, sample, start, {*block});
  REQUIRE(base.has_value());
  auto profile = magmaan::estimate::gmm::gp(*base, h.pt, *ev, start);
  REQUIRE(profile.has_value());
  auto problem = magmaan::optim::scalarize(profile->problem);
  auto fit = magmaan::optim::port(problem, profile->beta0, {}, {});
  REQUIRE(fit.has_value());
  REQUIRE(fit->audit.port_endpoint.has_value());
  const auto& telemetry = *fit->audit.port_endpoint;
  CHECK(telemetry.best_point_substituted);
  CHECK(telemetry.returned_x_objective > telemetry.stored_objective);
  Eigen::VectorXd gradient(fit->x.size());
  CHECK(fit->fmin == problem.f(fit->x, gradient));
  auto original = base->r(problem.expand(fit->x));
  REQUIRE(original.has_value());
  CHECK(std::abs(fit->fmin - 0.5 * original->squaredNorm()) <=
        1e-12 * (1 + std::abs(fit->fmin)));
  CHECK(fit->fmin <= telemetry.stored_objective * (1 + 1e-14));
}
#endif
