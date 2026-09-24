#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/frontier/newton_accuracy.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

using magmaan::data::SampleStats;
using magmaan::estimate::Estimates;
using magmaan::estimate::frontier::newton_accuracy_from;
using magmaan::estimate::frontier::newton_accuracy_ml;
using magmaan::estimate::frontier::NewtonAccuracyStatus;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::spec::LatentStructure;

struct Model {
  LatentStructure pt;
  MatrixRep rep;
  SampleStats samp;
  Eigen::VectorXd theta0;  // exact ML solution: S = Sigma(theta0)
};

// Builds the model, takes simple start values as the population point (after
// projecting onto the equality constraints), and sets S to its implied
// covariance, so theta0 is the exact ML solution.
Model exact_model(std::string_view syntax, std::int64_t n = 250) {
  Model m;
  auto flat = magmaan::parse::Parser::parse(syntax);
  REQUIRE(flat.has_value());
  magmaan::spec::BuildOptions opts;
  opts.fixed_x = false;
  auto pt = magmaan::spec::build(*flat, opts);
  REQUIRE(pt.has_value());
  m.pt = std::move(*pt);
  auto rep = magmaan::model::build_matrix_rep(m.pt);
  REQUIRE(rep.has_value());
  m.rep = std::move(*rep);
  // Any positive-definite S serves for the start values.
  const auto p = static_cast<Eigen::Index>(m.rep.ov_names[0].size());
  Eigen::MatrixXd S0 = Eigen::MatrixXd::Constant(p, p, 0.4);
  S0.diagonal().array() = 1.0;
  m.samp.S = {S0};
  m.samp.n_obs = {n};
  auto x0 = magmaan::estimate::simple_start_values(m.pt, m.rep, m.samp, {});
  REQUIRE(x0.has_value());
  auto con = magmaan::estimate::build_eq_constraints(m.pt, /*allow_nonlinear=*/true);
  REQUIRE(con.has_value());
  m.theta0 = con->expand(con->contract(*x0));
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto e = ev->evaluate(m.theta0, false, false);
  REQUIRE(e.has_value());
  m.samp.S = {e->moments.sigma[0]};
  return m;
}

Estimates at(const Eigen::VectorXd& theta) {
  Estimates e;
  e.theta = theta;
  return e;
}

}  // namespace

TEST_CASE("Newton accuracy: exact quadratic identities") {
  Eigen::MatrixXd H(2, 2);
  H << 4, 1, 1, 2;
  Eigen::VectorXd e(2);
  e << 0.1, -0.2;
  const Eigen::VectorXd g = H * e;
  const auto a = newton_accuracy_from(g, H);
  REQUIRE(a.status == NewtonAccuracyStatus::Available);
  CHECK(std::abs(a.distance * a.distance - e.dot(H * e)) < 1e-13);
  CHECK(std::abs(a.predicted_gain - 0.5 * e.dot(H * e)) < 1e-13);
  CHECK(std::abs(a.max_step - 0.2) < 1e-13);

  // Invariant under a linear change of coordinates.
  Eigen::MatrixXd T(2, 2);
  T << 0.01, 2, 0, 100;
  const auto b = newton_accuracy_from(T.transpose() * g, T.transpose() * H * T);
  CHECK(std::abs(a.distance - b.distance) < 1e-12);

  // Total-likelihood scaling: 100 times the data gives 10 times the distance.
  const auto c = newton_accuracy_from(100 * g, 100 * H);
  CHECK(std::abs(c.distance - 10 * a.distance) < 1e-12);

  // Budget.
  CHECK_FALSE(a.passed);
  magmaan::estimate::frontier::NewtonAccuracyOptions loose;
  loose.budget = 1.0;
  CHECK(newton_accuracy_from(g, H, loose).passed);

  H(1, 1) = -1;
  CHECK(newton_accuracy_from(g, H).status == NewtonAccuracyStatus::NonpositiveCurvature);
  H << 1, 1, 1, 1;
  CHECK(newton_accuracy_from(g, H).status != NewtonAccuracyStatus::Available);
}

TEST_CASE("Newton accuracy: exact solution and a small perturbation") {
  for (std::string_view syntax :
       {std::string_view{"f =~ x1 + x2 + x3 + x4"},
        std::string_view{"f =~ x1 + a*x2 + a*x3 + x4\nx2 ~~ v*x2\nx4 ~~ v*x4"}}) {
    CAPTURE(syntax);
    const Model m = exact_model(syntax);
    const auto exact = newton_accuracy_ml(m.pt, m.rep, m.samp, at(m.theta0));
    REQUIRE(exact.status == NewtonAccuracyStatus::Available);
    CHECK(exact.distance < 1e-8);
    CHECK(exact.passed);
    CHECK(exact.covariance_interior);

    // Move along the constraint surface: d must match sqrt(delta' I delta).
    auto con = magmaan::estimate::build_eq_constraints(m.pt);
    REQUIRE(con.has_value());
    Eigen::VectorXd alpha = con->contract(m.theta0);
    Eigen::VectorXd delta = Eigen::VectorXd::LinSpaced(alpha.size(), 1.0, -1.0);
    delta *= 2e-4 / delta.norm();
    const Eigen::VectorXd theta = con->expand(alpha + delta);
    auto info = magmaan::inference::information_observed_analytic(
        m.pt, m.rep, m.samp, at(m.theta0));
    REQUIRE(info.has_value());
    const Eigen::MatrixXd I = con->K().transpose() * (*info) * con->K();
    const double predicted = std::sqrt(delta.dot(I * delta));
    const auto moved = newton_accuracy_ml(m.pt, m.rep, m.samp, at(theta));
    REQUIRE(moved.status == NewtonAccuracyStatus::Available);
    CHECK(std::abs(moved.distance / predicted - 1.0) < 1e-3);
    CHECK(moved.n_reduced == con->n_alpha);
  }
}

TEST_CASE("Newton accuracy: boundary covariance and unsupported constraints") {
  Model m = exact_model("f =~ x1 + x2 + x3 + x4");
  // Put the first residual variance at zero: the covariance domain's boundary.
  Eigen::VectorXd theta = m.theta0;
  for (std::size_t r = 0; r < m.pt.op.size(); ++r) {
    const auto v = m.pt.lhs_var[r];
    if (m.pt.op[r] == magmaan::parse::Op::Covariance && m.pt.free[r] > 0 && v >= 0 &&
        v == m.pt.rhs_var[r] && m.pt.ov_pos[static_cast<std::size_t>(v)] >= 0) {
      theta[m.pt.free[r] - 1] = 0.0;
      break;
    }
  }
  const auto boundary = newton_accuracy_ml(m.pt, m.rep, m.samp, at(theta));
  CHECK_FALSE(boundary.covariance_interior);

  Model nl = exact_model("f =~ x1 + a*x2 + b*x3 + x4\na == b^2");
  CHECK(newton_accuracy_ml(nl.pt, nl.rep, nl.samp, at(nl.theta0)).status ==
        NewtonAccuracyStatus::Unsupported);
}
