#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <Eigen/Core>

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
