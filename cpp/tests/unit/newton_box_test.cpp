#include <doctest/doctest.h>
#include <limits>
#include <cmath>
#include "magmaan/estimate/frontier/convergence.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using namespace magmaan;
namespace nf = estimate::frontier;
namespace {
struct Model { spec::LatentStructure pt; model::MatrixRep rep; };
Model make_model(std::string_view text) {
  auto syntax = parse::Parser::parse(text); REQUIRE(syntax.has_value());
  spec::BuildOptions opts; opts.fixed_x = false;
  auto pt = spec::build(*syntax, opts); REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}
nf::NewtonDerivatives derivatives(const Eigen::VectorXd& x, const Eigen::VectorXd& g,
                                  const Eigen::MatrixXd& H) {
  nf::NewtonDerivatives d; d.theta = x; d.gradient = g; d.hessian = H;
  d.objective = 0; d.n_obs = 1; d.status = estimate::NewtonAccuracyStatus::Available;
  return d;
}
}

TEST_CASE("Newton boxes: lower and upper boundaries release improving directions") {
  auto m = make_model("x ~~ x");
  const auto x = Eigen::VectorXd::Ones(1); const auto H = Eigen::MatrixXd::Identity(1, 1);
  estimate::Bounds box; box.lower = x; box.upper = 3*x;
  auto outward = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, 2*x, H),
      estimate::StationarityDomain::Ambient, {}, box);
  REQUIRE(outward.diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(outward.diagnostics.box_constrained); CHECK(outward.diagnostics.passed);
  CHECK(outward.solution.step.norm() == doctest::Approx(0));
  CHECK(outward.box.multipliers[0] == doctest::Approx(2));
  auto psd_interior = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, 2*x, H),
      estimate::StationarityDomain::Psd, {}, box);
  CHECK(psd_interior.diagnostics.passed);
  CHECK(psd_interior.diagnostics.box_constrained);
  CHECK(psd_interior.geometry.covariance_interior);
  auto inward = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, -x, H),
      estimate::StationarityDomain::Ambient, {}, box);
  REQUIRE(inward.diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  CHECK_FALSE(inward.diagnostics.passed);
  CHECK(inward.solution.step[0] == doctest::Approx(1));
  CHECK(inward.solution.predicted_gain == doctest::Approx(.5));
  auto upper = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(3*x, -2*x, H),
      estimate::StationarityDomain::Ambient, {}, box);
  CHECK(upper.diagnostics.passed);
  CHECK(upper.box.multipliers[1] == doctest::Approx(2));
  auto weak = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, 0*x, H),
      estimate::StationarityDomain::Ambient, {}, box);
  CHECK(weak.diagnostics.passed);
  CHECK(weak.box.multipliers.norm() == doctest::Approx(0));
}

TEST_CASE("Newton boxes: coupled quadratic hits previously inactive bounds") {
  Eigen::Matrix2d H; H << 2, 1, 1, 2;
  Eigen::MatrixXd A(4, 2); A << 1, 0, 0, 1, -1, 0, 0, -1;
  Eigen::Vector4d b(0, 0, -1, -1);
  Eigen::Vector2d g(-4, -4);
  auto r = nf::solve_newton_box(nf::prepare_newton_system(H), g, A, b);
  REQUIRE(r.solution.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(r.solution.step.isApprox(Eigen::Vector2d::Ones(), 1e-10));
  CHECK(r.solution.predicted_gain == doctest::Approx(5));
  CHECK(r.solution.distance == doctest::Approx(std::sqrt(10.)));
  CHECK(r.solution.solve_residual < 1e-10);
  CHECK((H*r.solution.step + g - A.transpose()*r.multipliers).norm() < 1e-10);
  nf::NewtonAccuracyOptions limited; limited.box_max_iter = 1;
  auto unfinished = nf::solve_newton_box(nf::prepare_newton_system(H), g, A, b, limited);
  CHECK(unfinished.solution.status == estimate::NewtonAccuracyStatus::Unavailable);
  CHECK_FALSE(nf::assess_newton_accuracy(unfinished.solution).passed);
}

TEST_CASE("Newton boxes: equality-reduced and fixed-coordinate constraints") {
  auto m = make_model("x ~~ a*x\ny ~~ a*y");
  auto con = estimate::build_eq_constraints(m.pt); REQUIRE(con.has_value());
  const auto n = static_cast<Eigen::Index>(m.pt.n_free());
  const auto x = Eigen::VectorXd::Ones(n);
  estimate::Bounds box; box.lower = x; box.upper = 2*x;
  auto a = nf::audit_newton_derivatives(m.pt, m.rep,
      derivatives(x, -x, Eigen::MatrixXd::Identity(n,n)), estimate::StationarityDomain::Ambient, {}, box);
  REQUIRE(a.diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  const Eigen::VectorXd step = a.geometry.equality_basis * a.geometry.tangent_basis * a.solution.step;
  CHECK(step.isApprox(x, 1e-10));
  CHECK((con->A_eq * step).norm() < 1e-10);
  box.upper = box.lower;
  auto fixed = nf::audit_newton_derivatives(m.pt, m.rep,
      derivatives(x, -x, -Eigen::MatrixXd::Identity(n,n)), estimate::StationarityDomain::Ambient, {}, box);
  CHECK(fixed.diagnostics.passed);
  CHECK(fixed.geometry.reduced_hessian.size() == 0);
  CHECK(fixed.derivatives.hessian.rows() == n);
}

TEST_CASE("Newton boxes: redundant PSD variance bounds share cone geometry") {
  auto m = make_model("x ~~ x + y\ny ~~ y");
  const auto n = static_cast<Eigen::Index>(m.pt.n_free());
  auto x = Eigen::VectorXd::Zero(n).eval(); auto g = x;
  estimate::Bounds box;
  box.lower = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
  box.upper = -box.lower;
  for (std::size_t row = 0; row < m.pt.size(); ++row) {
    const auto& c = m.rep.cell_for_row[row];
    if (m.pt.free[row] > 0 && c.used && c.row == c.col) {
      g[m.pt.free[row]-1] = 1; box.lower[m.pt.free[row]-1] = 0;
    }
  }
  auto a = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, g, Eigen::MatrixXd::Identity(n,n)),
      estimate::StationarityDomain::Psd, {}, box);
  REQUIRE(a.diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(a.diagnostics.psd_domain); CHECK(a.diagnostics.box_constrained);
  CHECK(a.diagnostics.passed);
  CHECK_FALSE(a.geometry.covariance_interior);
  box.upper[0] = 1;
  auto inactive_upper = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, g, Eigen::MatrixXd::Identity(n,n)),
      estimate::StationarityDomain::Psd, {}, box);
  CHECK(inactive_upper.diagnostics.passed);
  auto concave = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, 0*g, -Eigen::MatrixXd::Identity(n,n)),
      estimate::StationarityDomain::Psd, {}, box);
  CHECK(concave.diagnostics.status == estimate::NewtonAccuracyStatus::NonpositiveCurvature);
  // A genuine additional active upper bound needs joint box/cone multipliers.
  box.upper[0] = 0;
  auto combined = nf::audit_newton_derivatives(m.pt, m.rep, derivatives(x, g, Eigen::MatrixXd::Identity(n,n)),
      estimate::StationarityDomain::Psd, {}, box);
  CHECK(combined.diagnostics.status == estimate::NewtonAccuracyStatus::Unsupported);
  CHECK(combined.derivatives.hessian.rows() == n);
}

TEST_CASE("Newton boxes: explicit policy accepts constrained evidence without migrating compatibility") {
  auto m = make_model("x ~~ x");
  optim::ScalarProblem p; p.n_param = 1;
  p.f = [](const Eigen::VectorXd& x, Eigen::VectorXd& g) { g = x; return .5*x.squaredNorm(); };
  nf::ConvergenceRequest req; req.newton = true;
  req.bounds.lower = Eigen::VectorXd::Ones(1); req.bounds.upper = 3*req.bounds.lower;
  auto report = nf::audit_convergence(m.pt, m.rep, p, req.bounds.lower, 1, 1, req, .5);
  REQUIRE(report.has_value());
  CHECK(nf::assess_convergence(*report, nf::newton_convergence_policy()).status == estimate::FitCheck::Passed);
  CHECK(estimate::common_fit_verdict(report->evidence).criterion == estimate::StationarityCriterion::FirstOrder);
  const auto saved = report->computations.derivatives.hessian;
  auto policy = nf::newton_convergence_policy(); policy.newton.budget = 0;
  CHECK(nf::assess_convergence(*report, policy).status == estimate::FitCheck::Passed);
  CHECK(report->computations.derivatives.hessian.isApprox(saved));
}

TEST_CASE("Newton boxes: convex solutions match exhaustive two-dimensional face minima") {
  Eigen::MatrixXd A(4,2); A << 1,0, 0,1, -1,0, 0,-1;
  Eigen::Vector4d b(-.3,-.4,-.6,-.7);
  Eigen::Vector2d lower(-.3,-.4), upper(.6,.7);
  for (int k = 0; k < 40; ++k) {
    CAPTURE(k);
    const double rho = .9 * std::sin(.7*k);
    Eigen::Matrix2d H; H << 1,rho,rho,2;
    Eigen::Vector2d g(5*std::sin(.9*k),5*std::cos(1.1*k));
    double best = std::numeric_limits<double>::infinity();
    Eigen::Vector2d expected = Eigen::Vector2d::Zero();
    // Enumerate the interior, all four edges and all four corners.
    for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) {
      Eigen::Vector2d x(a == 1 ? lower[0] : upper[0], c == 1 ? lower[1] : upper[1]);
      if (!a && !c) {
        x[0] = (-2*g[0] + rho*g[1])/(2-rho*rho);
        x[1] = (rho*g[0] - g[1])/(2-rho*rho);
      } else if (!a) x[0] = -g[0]-rho*x[1];
      else if (!c) x[1] = (-g[1]-rho*x[0])/2;
      if ((x.array() < lower.array()-1e-12).any() || (x.array() > upper.array()+1e-12).any()) continue;
      const double f = g.dot(x)+.5*x.dot(H*x);
      if (f < best) { best = f; expected = x; }
    }
    auto result = nf::solve_newton_box(nf::prepare_newton_system(H), g, A, b);
    INFO(result.detail);
    REQUIRE(result.solution.status == estimate::NewtonAccuracyStatus::Available);
    CHECK((result.solution.step-expected).norm() < 1e-9);
    CHECK(result.solution.predicted_gain == doctest::Approx(-best).epsilon(1e-9));
    CHECK(result.solution.solve_residual < 1e-10);
    Eigen::Matrix2d T = Eigen::Vector2d(1e-3,1e3).asDiagonal();
    auto scaled = nf::solve_newton_box(nf::prepare_newton_system(T*H*T), T*g, A*T, b);
    REQUIRE(scaled.solution.status == estimate::NewtonAccuracyStatus::Available);
    CHECK((T*scaled.solution.step-expected).norm() < 1e-9);
  }
}
