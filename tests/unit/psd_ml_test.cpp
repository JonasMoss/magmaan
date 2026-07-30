#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "../../src/estimate/detail_psd_probe.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::estimate::build_eq_constraints;
using magmaan::estimate::build_nl_constraints;
using magmaan::estimate::simple_start_values;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatId;
using magmaan::optim::OptimOptions;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentStructure;

LatentStructure lavaanify(std::string_view syntax,
                          BuildOptions options = {}) {
  auto flat = Parser::parse(syntax);
  REQUIRE(flat.has_value());
  auto pt = magmaan::spec::build(*flat, options);
  REQUIRE_MESSAGE(pt.has_value(), "lavaanify failed: "
      << (pt.has_value() ? std::string{} : pt.error().detail));
  return std::move(*pt);
}

Eigen::MatrixXd one_factor_covariance(const Eigen::VectorXd& loadings,
                                      const Eigen::VectorXd& residuals,
                                      double latent_variance) {
  return latent_variance * loadings * loadings.transpose() +
      residuals.asDiagonal().toDenseMatrix();
}

SampleStats sample_stats(Eigen::MatrixXd covariance,
                         std::int64_t n = 400) {
  SampleStats out;
  out.S.push_back(std::move(covariance));
  out.n_obs.push_back(n);
  return out;
}

OptimOptions strict_options() {
  OptimOptions out;
  out.max_iter = 5000;
  out.ftol = 1e-13;
  out.gtol = 1e-8;
  return out;
}

void check_psd_terminal(const magmaan::estimate::Estimates& fit) {
  CHECK(fit.diagnostics.admissibility.admissible);
  CHECK(fit.audit.constrained);
  CHECK(fit.audit.constraint_violation_inf <= 1e-6);
  CHECK(fit.audit.stationary);
}

}  // namespace

TEST_CASE("PSD ML lifted objective and equality Jacobian match central "
          "finite differences") {
  BuildOptions options;
  options.meanstructure = true;
  auto pt = lavaanify(
      "f =~ 1*x1 + a*x2 + b*x3\n"
      "x1 ~~ v*x1\n"
      "x2 ~~ v*x2\n"
      "x1 ~~ c*x2\n"
      "a + b == 1.6\n"
      "a == b^2",
      options);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Matrix3d covariance;
  covariance << 1.8, 0.72, 0.55,
                0.72, 1.5, 0.46,
                0.55, 0.46, 1.3;
  auto samp = sample_stats(covariance, 300);
  Eigen::Vector3d mean;
  mean << 0.4, -0.2, 0.7;
  samp.mean.push_back(mean);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  auto probe = magmaan::estimate::psd_test::psd_ml_derivative_probe(
      pt, *rep, samp, *start);
  REQUIRE_MESSAGE(probe.has_value(), "PSD derivative probe failed: "
      << (probe.has_value() ? std::string{} : probe.error().detail));
  REQUIRE(probe->n_alpha > 0);
  REQUIRE(probe->n_lift > 0);
  REQUIRE(probe->constraints.size() > 0);
  REQUIRE(probe->analytic_gradient.size() == probe->x.size());
  REQUIRE(probe->analytic_constraint_jacobian.rows() ==
          probe->constraints.size());
  REQUIRE(probe->analytic_constraint_jacobian.cols() == probe->x.size());

  CHECK((probe->analytic_gradient -
         probe->finite_difference_gradient).cwiseAbs().maxCoeff() < 3e-6);
  CHECK((probe->analytic_constraint_jacobian -
         probe->finite_difference_constraint_jacobian)
            .cwiseAbs().maxCoeff() < 3e-7);
}

TEST_CASE("PSD ML honors shared covariances and a general linear equality") {
  auto pt = lavaanify(
      "f =~ 1*x1 + a*x2 + b*x3 + d*x4\n"
      "x1 ~~ c*x2\n"
      "x3 ~~ c*x4\n"
      "a + b == 1.5");
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d loadings;
  loadings << 1.0, 0.8, 0.7, 0.9;
  Eigen::Vector4d residuals;
  residuals << 0.6, 0.7, 0.5, 0.8;
  Eigen::Matrix4d covariance =
      one_factor_covariance(loadings, residuals, 1.2);
  covariance(0, 1) += 0.1;
  covariance(1, 0) += 0.1;
  covariance(2, 3) += 0.1;
  covariance(3, 2) += 0.1;
  auto samp = sample_stats(covariance, 500);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  auto probe = magmaan::estimate::psd_test::psd_ml_derivative_probe(
      pt, *rep, samp, *start);
  REQUIRE_MESSAGE(probe.has_value(),
      "shared/general-linear PSD derivative probe failed: "
          << (probe.has_value() ? std::string{} : probe.error().detail));
  CHECK((probe->analytic_gradient -
         probe->finite_difference_gradient).cwiseAbs().maxCoeff() < 3e-6);
  CHECK((probe->analytic_constraint_jacobian -
         probe->finite_difference_constraint_jacobian)
            .cwiseAbs().maxCoeff() < 3e-7);

  auto fit = magmaan::estimate::frontier::fit_ml_psd(
      pt, *rep, samp, *start, Backend::NloptSlsqp, strict_options());
  REQUIRE_MESSAGE(fit.has_value(), "shared/general-linear PSD fit failed: "
      << (fit.has_value() ? std::string{} : fit.error().detail));
  check_psd_terminal(*fit);

  const auto con = build_eq_constraints(
      pt, /*allow_nonlinear=*/true);
  REQUIRE(con.has_value());
  REQUIRE(con->active());
  CHECK((con->A_eq * fit->theta - con->b_eq)
            .cwiseAbs().maxCoeff() < 1e-9);

  std::vector<Eigen::Index> covariance_parameters;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& cell = rep->cell_for_row[i];
    if (cell.used && cell.mat == MatId::Theta &&
        cell.row != cell.col && pt.free[i] > 0) {
      covariance_parameters.push_back(pt.free[i] - 1);
    }
  }
  REQUIRE(covariance_parameters.size() == 2);
  CHECK(fit->theta(covariance_parameters[0]) ==
        doctest::Approx(fit->theta(covariance_parameters[1])).epsilon(1e-10));
  CHECK(fit->fmin < 1e-8);
}

TEST_CASE("PSD ML honors a nonlinear equality") {
  auto pt = lavaanify(
      "f =~ 1*x1 + a*x2 + b*x3\n"
      "a == b^2");
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector3d loadings;
  loadings << 1.0, 0.64, 0.8;
  Eigen::Vector3d residuals;
  residuals << 0.5, 0.6, 0.7;
  auto samp = sample_stats(
      one_factor_covariance(loadings, residuals, 1.2), 500);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  auto fit = magmaan::estimate::frontier::fit_ml_psd(
      pt, *rep, samp, *start, Backend::NloptSlsqp, strict_options());
  REQUIRE_MESSAGE(fit.has_value(), "nonlinear-equality PSD fit failed: "
      << (fit.has_value() ? std::string{} : fit.error().detail));
  check_psd_terminal(*fit);

  const auto nonlinear = build_nl_constraints(pt);
  REQUIRE(nonlinear.active());
  CHECK(nonlinear.h(fit->theta).cwiseAbs().maxCoeff() < 1e-7);
  CHECK(fit->fmin < 1e-8);
}

TEST_CASE("PSD ML handles independent multi-group covariance blocks") {
  BuildOptions options;
  options.n_groups = 2;
  auto pt = lavaanify("f =~ x1 + x2 + x3", options);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());
  REQUIRE(rep->dims.size() == 2);

  Eigen::Vector3d loadings_1;
  loadings_1 << 1.0, 0.8, 0.65;
  Eigen::Vector3d residuals_1;
  residuals_1 << 0.5, 0.6, 0.7;
  Eigen::Vector3d loadings_2;
  loadings_2 << 1.0, 0.75, 0.7;
  Eigen::Vector3d residuals_2;
  residuals_2 << 0.4, 0.7, 0.6;
  SampleStats samp;
  samp.S = {
      one_factor_covariance(loadings_1, residuals_1, 1.2),
      one_factor_covariance(loadings_2, residuals_2, 1.5)};
  samp.n_obs = {240, 180};
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  auto probe = magmaan::estimate::psd_test::psd_ml_derivative_probe(
      pt, *rep, samp, *start);
  REQUIRE_MESSAGE(probe.has_value(), "multi-group PSD derivative probe failed: "
      << (probe.has_value() ? std::string{} : probe.error().detail));
  CHECK((probe->analytic_gradient -
         probe->finite_difference_gradient).cwiseAbs().maxCoeff() < 3e-6);
  CHECK((probe->analytic_constraint_jacobian -
         probe->finite_difference_constraint_jacobian)
            .cwiseAbs().maxCoeff() < 3e-7);

  auto fit = magmaan::estimate::frontier::fit_ml_psd(
      pt, *rep, samp, *start, Backend::NloptSlsqp, strict_options());
  REQUIRE_MESSAGE(fit.has_value(), "multi-group PSD fit failed: "
      << (fit.has_value() ? std::string{} : fit.error().detail));
  check_psd_terminal(*fit);
  CHECK(fit->fmin < 1e-8);
  REQUIRE(fit->diagnostics.admissibility.theta_blocks.size() == 2);
  REQUIRE(fit->diagnostics.admissibility.psi_blocks.size() == 2);
}

TEST_CASE("PSD ML mean-structure fit agrees with ordinary interior ML") {
  BuildOptions options;
  options.meanstructure = true;
  auto pt = lavaanify("f =~ x1 + x2 + x3", options);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector3d loadings;
  loadings << 1.0, 0.8, 0.65;
  Eigen::Vector3d residuals;
  residuals << 0.5, 0.6, 0.7;
  auto samp = sample_stats(
      one_factor_covariance(loadings, residuals, 1.2), 400);
  Eigen::Vector3d mean;
  mean << 1.2, -0.4, 0.8;
  samp.mean.push_back(mean);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  auto ordinary = magmaan::estimate::fit_ml(
      pt, *rep, samp, *start, {}, Backend::NloptLbfgs, strict_options());
  REQUIRE_MESSAGE(ordinary.has_value(), "ordinary mean ML fit failed: "
      << (ordinary.has_value() ? std::string{}
                               : ordinary.error().detail));
  REQUIRE(ordinary->diagnostics.admissibility.admissible);

  auto fit = magmaan::estimate::frontier::fit_ml_psd(
      pt, *rep, samp, ordinary->theta,
      Backend::NloptSlsqp, strict_options());
  REQUIRE_MESSAGE(fit.has_value(), "mean-structure PSD fit failed: "
      << (fit.has_value() ? std::string{} : fit.error().detail));
  check_psd_terminal(*fit);
  CHECK(fit->fmin == doctest::Approx(ordinary->fmin).epsilon(1e-8));
  CHECK((fit->theta - ordinary->theta).cwiseAbs().maxCoeff() < 2e-5);
}
