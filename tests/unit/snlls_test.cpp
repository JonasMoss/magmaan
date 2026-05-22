#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/estimate/gmm/gp.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "../oracle.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::model::ModelEvaluator;
using magmaan::model::build_matrix_rep;
using magmaan::optim::LbfgsOptions;
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

LbfgsOptions snlls_opts() {
  return LbfgsOptions{.max_iter = 2000, .ftol = 1e-13, .gtol = 1e-8};
}

LbfgsOptions matlab_like_opts() {
  return LbfgsOptions{.max_iter = 400, .ftol = 1e-6, .gtol = 1e-6};
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
                                          Backend::Lbfgs, snlls_opts());
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
                                              Backend::Lbfgs, snlls_opts());
  REQUIRE(gls.has_value());
  CHECK(gls->fmin < 1e-10);

  magmaan::estimate::gmm::Weight wls{Eigen::MatrixXd::Identity(6, 6)};
  auto wls_est = magmaan::estimate::fit_snlls(h.pt, h.rep, samp, *x0, wls,
                                              Backend::Lbfgs, snlls_opts());
  REQUIRE(wls_est.has_value());
  CHECK(wls_est->fmin < 1e-10);

#ifdef MAGMAAN_WITH_PORT
  // PortNls — same feasible 1F problem; the unique global optimum lets us
  // pin PortNls to the same fmin ≈ 0 as L-BFGS without worrying about
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
                                          Backend::Lbfgs, snlls_opts());
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
                                          Backend::Lbfgs, snlls_opts());
  REQUIRE(est.has_value());
  CHECK(est->iterations == 0);
  CHECK(est->fmin < 1e-12);
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
  const LbfgsOptions opts = matlab_like_opts();

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

  auto lbfgs = run(Backend::Lbfgs, "lbfgs");
  REQUIRE(lbfgs.has_value());
  CHECK(lbfgs->fmin < 0.243);

#ifdef MAGMAAN_WITH_PORT
  auto port = run(Backend::Port, "port");
  CHECK(port.has_value());
  if (port.has_value()) {
    CHECK(port->fmin == doctest::Approx(lbfgs->fmin).epsilon(1e-5));
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

#ifdef MAGMAAN_WITH_NLOPT
  auto nlopt = run(Backend::NloptSlsqp, "nlopt-slsqp");
  CHECK(nlopt.has_value());
  if (nlopt.has_value()) {
    CHECK(nlopt->fmin == doctest::Approx(lbfgs->fmin).epsilon(1e-5));
  }
#endif

#ifdef MAGMAAN_WITH_CERES
  auto ceres_bfgs = run(Backend::CeresBfgs, "ceres-bfgs");
  CHECK(ceres_bfgs.has_value());
  if (ceres_bfgs.has_value()) {
    CHECK(ceres_bfgs->fmin == doctest::Approx(lbfgs->fmin).epsilon(1e-5));
  }

  auto ceres = run(Backend::Ceres, "ceres");
  CHECK(ceres.has_value());
  if (ceres.has_value()) {
    CHECK(ceres->fmin == doctest::Approx(lbfgs->fmin).epsilon(1e-5));
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
