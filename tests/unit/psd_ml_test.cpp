#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "../../src/estimate/detail_psd_probe.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/optimizers.hpp"
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

magmaan::data::OrdinalStats ordinal_stats(Eigen::Matrix3d correlation) {
  magmaan::data::OrdinalStats out;
  out.R.push_back(std::move(correlation));
  Eigen::VectorXd thresholds(6);
  thresholds << -0.5, 0.5, -0.4, 0.6, -0.6, 0.4;
  out.thresholds.push_back(std::move(thresholds));
  out.threshold_ov.push_back({0, 0, 1, 1, 2, 2});
  out.threshold_level.push_back({1, 2, 1, 2, 1, 2});
  out.NACOV.push_back(Eigen::MatrixXd::Identity(9, 9));
  out.W_dwls.push_back(Eigen::MatrixXd::Identity(9, 9));
  out.W_wls.push_back(Eigen::MatrixXd::Identity(9, 9));
  out.n_obs.push_back(400);
  out.n_levels.push_back({3, 3, 3});
  out.ov_names.push_back({"x1", "x2", "x3"});
  return out;
}

LatentStructure ordinal_one_factor() {
  return lavaanify(
      "f =~ 1*x1 + l2*x2 + l3*x3\n"
      "x1 | t11 + t12\n"
      "x2 | t21 + t22\n"
      "x3 | t31 + t32");
}

OptimOptions strict_options() {
  OptimOptions out;
  out.max_iter = 5000;
  out.ftol = 1e-13;
  out.gtol = 1e-8;
  return out;
}

Eigen::MatrixXd standardized_design(Eigen::Index n, Eigen::Index p) {
  Eigen::MatrixXd Z(n, p);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index c = 0; c < p; ++c) {
      const double rr = static_cast<double>(r + 1);
      const double cc = static_cast<double>(c + 1);
      Z(r, c) = std::sin(0.37 * rr * cc) +
                std::cos(0.19 * (rr + 1.0) * (cc + 2.0));
    }
  }
  Z.rowwise() -= Z.colwise().mean();
  Eigen::LLT<Eigen::MatrixXd> llt((Z.transpose() * Z) /
                                  static_cast<double>(n));
  REQUIRE(llt.info() == Eigen::Success);
  return Z * llt.matrixU().solve(
                 Eigen::MatrixXd::Identity(p, p));
}

magmaan::data::RawData raw_with_covariance(
    const Eigen::MatrixXd& covariance, const Eigen::VectorXd& mean,
    Eigen::Index n, bool introduce_missing) {
  Eigen::LLT<Eigen::MatrixXd> llt(covariance);
  REQUIRE(llt.info() == Eigen::Success);
  Eigen::MatrixXd X =
      (standardized_design(n, covariance.rows()) *
       llt.matrixL().transpose()).rowwise() + mean.transpose();
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  if (introduce_missing) {
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask =
        Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(
            n, covariance.cols());
    for (Eigen::Index r = 0; r < n; ++r) {
      if (r % 11 == 0) {
        const Eigen::Index c = (r / 11) % covariance.cols();
        mask(r, c) = 0;
        raw.X[0](r, c) = std::numeric_limits<double>::quiet_NaN();
      }
    }
    raw.mask.push_back(std::move(mask));
  }
  return raw;
}

void check_psd_terminal(const magmaan::estimate::Estimates& fit) {
  CHECK(fit.diagnostics.admissibility.admissible);
  CHECK(fit.audit.constrained);
  CHECK(fit.audit.constraint_violation_inf <= 1e-6);
  CHECK(fit.audit.stationary);
}

}  // namespace

TEST_CASE("PSD ordinal delta and theta Jacobians match central finite "
          "differences") {
  auto pt = ordinal_one_factor();
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());
  Eigen::Matrix3d correlation;
  correlation << 1.0, 0.56, 0.48,
                 0.56, 1.0, 0.336,
                 0.48, 0.336, 1.0;
  const auto stats = ordinal_stats(correlation);
  auto start = magmaan::estimate::ordinal_start_values(pt, *rep, stats, {});
  REQUIRE(start.has_value());

  for (const auto parameterization : {
           magmaan::estimate::OrdinalParameterization::Delta,
           magmaan::estimate::OrdinalParameterization::Theta}) {
    auto probe = magmaan::estimate::psd_test::psd_ordinal_derivative_probe(
        pt, *rep, stats, *start,
        magmaan::estimate::OrdinalWeightKind::DWLS, parameterization);
    REQUIRE_MESSAGE(probe.has_value(),
        "PSD ordinal derivative probe failed: "
            << (probe.has_value() ? std::string{} : probe.error().detail));
    REQUIRE(probe->n_lift > 0);
    CHECK((probe->analytic_gradient - probe->finite_difference_gradient)
              .cwiseAbs().maxCoeff() < 5e-6);
    CHECK((probe->analytic_constraint_jacobian -
           probe->finite_difference_constraint_jacobian)
              .cwiseAbs().maxCoeff() < 3e-7);
  }
}

TEST_CASE("PSD ordinal ULS DWLS and WLS consume polychorics unchanged") {
  auto pt = ordinal_one_factor();
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());
  Eigen::Matrix3d correlation;
  correlation << 1.0, 0.56, 0.48,
                 0.56, 1.0, 0.336,
                 0.48, 0.336, 1.0;
  auto stats = ordinal_stats(correlation);
  const Eigen::MatrixXd original_R = stats.R[0];
  auto start = magmaan::estimate::ordinal_start_values(pt, *rep, stats, {});
  REQUIRE(start.has_value());

  for (const auto parameterization : {
           magmaan::estimate::OrdinalParameterization::Delta,
           magmaan::estimate::OrdinalParameterization::Theta}) {
    for (const auto weights : {
             magmaan::estimate::OrdinalWeightKind::ULS,
             magmaan::estimate::OrdinalWeightKind::DWLS,
             magmaan::estimate::OrdinalWeightKind::WLS}) {
      auto ordinary = magmaan::estimate::fit_ordinal_bounded(
          pt, *rep, stats, {}, weights, *start, Backend::NloptLbfgs,
          strict_options(), parameterization);
      REQUIRE_MESSAGE(ordinary.has_value(), "ordinary ordinal fit failed: "
          << (ordinary.has_value() ? std::string{} : ordinary.error().detail));
      if (!ordinary.has_value()) return;
      auto fit = magmaan::estimate::frontier::fit_ordinal_psd(
          pt, *rep, stats, {}, weights, *start, Backend::NloptSlsqp,
          strict_options(), parameterization);
      REQUIRE_MESSAGE(fit.has_value(), "PSD ordinal fit failed: "
          << (fit.has_value() ? std::string{} : fit.error().detail));
      if (!fit.has_value()) return;
      check_psd_terminal(*fit);
      CHECK(fit->fmin < 1e-9);
      CHECK(fit->fmin == doctest::Approx(ordinary->fmin).epsilon(1e-7));
      CHECK((fit->theta - ordinary->theta).cwiseAbs().maxCoeff() < 2e-4);
      CHECK(stats.R[0].isApprox(original_R, 0.0));
    }
  }
}

TEST_CASE("PSD ordinal ULS accepts an indefinite estimated polychoric matrix") {
  auto pt = ordinal_one_factor();
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());
  Eigen::Matrix3d indefinite;
  indefinite << 1.0, 0.9, 0.7,
                0.9, 1.0, 0.3,
                0.7, 0.3, 1.0;
  REQUIRE(indefinite.determinant() < 0.0);
  auto stats = ordinal_stats(indefinite);
  const Eigen::MatrixXd original_R = stats.R[0];

  Eigen::Matrix3d start_R;
  start_R << 1.0, 0.4, 0.3,
             0.4, 1.0, 0.2,
             0.3, 0.2, 1.0;
  const auto start_stats = ordinal_stats(start_R);
  auto start = magmaan::estimate::ordinal_start_values(
      pt, *rep, start_stats, {});
  REQUIRE(start.has_value());
  auto options = strict_options();
  options.max_iter = 20000;
  options.ftol = 1e-10;
  options.gtol = 1e-6;
  auto fit = magmaan::estimate::frontier::fit_ordinal_psd(
      pt, *rep, stats, {}, magmaan::estimate::OrdinalWeightKind::ULS,
      *start, Backend::NloptSlsqp, options);
  REQUIRE_MESSAGE(fit.has_value(), "indefinite-polychoric PSD fit failed: "
      << (fit.has_value() ? std::string{} : fit.error().detail));
  if (!fit.has_value()) return;
  CHECK(fit->diagnostics.admissibility.admissible);
  CHECK(stats.R[0].isApprox(original_R, 0.0));
}

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

TEST_CASE("PSD FIML derivatives and all-observed reduction agree with ML") {
  BuildOptions options;
  options.meanstructure = true;
  auto pt = lavaanify("f =~ x1 + x2 + x3 + x4", options);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d loadings;
  loadings << 1.0, 0.8, 0.65, 0.9;
  Eigen::Vector4d residuals;
  residuals << 0.5, 0.6, 0.7, 0.55;
  const Eigen::Matrix4d covariance =
      one_factor_covariance(loadings, residuals, 1.2);
  Eigen::Vector4d mean;
  mean << 0.4, -0.2, 0.7, 1.1;
  auto raw = raw_with_covariance(covariance, mean, 300, false);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto start = simple_start_values(pt, *rep, pack->start_stats, {});
  REQUIRE(start.has_value());

  auto probe = magmaan::estimate::psd_test::psd_fiml_derivative_probe(
      pt, *rep, raw, *pack, *start);
  REQUIRE_MESSAGE(probe.has_value(), "PSD FIML derivative probe failed: "
      << (probe.has_value() ? std::string{} : probe.error().detail));
  CHECK((probe->analytic_gradient -
         probe->finite_difference_gradient).cwiseAbs().maxCoeff() < 4e-6);
  CHECK((probe->analytic_constraint_jacobian -
         probe->finite_difference_constraint_jacobian)
            .cwiseAbs().maxCoeff() < 3e-7);

  auto ml = magmaan::estimate::frontier::fit_ml_psd(
      pt, *rep, pack->start_stats, *start, Backend::NloptSlsqp,
      strict_options());
  REQUIRE(ml.has_value());
  auto ordinary_fiml = magmaan::estimate::fit_fiml(
      pt, *rep, raw, *start, *pack, Backend::NloptLbfgsSlsqpFallback,
      strict_options());
  REQUIRE(ordinary_fiml.has_value());
  auto fiml = magmaan::estimate::fiml::frontier::fit_fiml_psd(
      pt, *rep, raw, *start, magmaan::estimate::fiml::FIML{},
      Backend::NloptSlsqp, strict_options());
  REQUIRE_MESSAGE(fiml.has_value(), "all-observed PSD FIML failed: "
      << (fiml.has_value() ? std::string{} : fiml.error().detail));
  check_psd_terminal(*fiml);
  CHECK(fiml->fmin == doctest::Approx(ordinary_fiml->fmin).epsilon(1e-8));
  CHECK((fiml->theta - ordinary_fiml->theta).cwiseAbs().maxCoeff() < 2e-5);
  CHECK((fiml->theta - ml->theta).cwiseAbs().maxCoeff() < 2e-5);
}

TEST_CASE("PSD FIML agrees with an interior missing-data fit") {
  BuildOptions options;
  options.meanstructure = true;
  auto pt = lavaanify("f =~ x1 + x2 + x3 + x4", options);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d loadings;
  loadings << 1.0, 0.82, 0.68, 0.9;
  Eigen::Vector4d residuals;
  residuals << 0.55, 0.65, 0.75, 0.6;
  Eigen::Vector4d mean;
  mean << 0.3, -0.1, 0.6, 1.0;
  auto raw = raw_with_covariance(
      one_factor_covariance(loadings, residuals, 1.15), mean, 500, true);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto start = simple_start_values(pt, *rep, pack->start_stats, {});
  REQUIRE(start.has_value());

  auto probe = magmaan::estimate::psd_test::psd_fiml_derivative_probe(
      pt, *rep, raw, *pack, *start);
  REQUIRE(probe.has_value());
  CHECK((probe->analytic_gradient -
         probe->finite_difference_gradient).cwiseAbs().maxCoeff() < 5e-6);

  auto ordinary = magmaan::estimate::fit_fiml(
      pt, *rep, raw, *start, *pack, Backend::NloptLbfgsSlsqpFallback,
      strict_options());
  REQUIRE_MESSAGE(ordinary.has_value(), "ordinary FIML failed: "
      << (ordinary.has_value() ? std::string{} : ordinary.error().detail));
  REQUIRE(ordinary->diagnostics.admissibility.admissible);
  auto psd = magmaan::estimate::fiml::frontier::fit_fiml_psd(
      pt, *rep, raw, ordinary->theta, *pack, Backend::NloptSlsqp,
      strict_options());
  REQUIRE_MESSAGE(psd.has_value(), "interior PSD FIML failed: "
      << (psd.has_value() ? std::string{} : psd.error().detail));
  check_psd_terminal(*psd);
  CHECK(psd->fmin == doctest::Approx(ordinary->fmin).epsilon(1e-8));
  CHECK((psd->theta - ordinary->theta).cwiseAbs().maxCoeff() < 3e-5);
}

TEST_CASE("PSD FIML repairs a missing-data negative-residual solution") {
  BuildOptions options;
  options.meanstructure = true;
  auto pt = lavaanify("f =~ x1 + x2 + x3", options);
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Matrix3d covariance;
  covariance << 1.0, 0.7, 0.7,
                0.7, 1.0, 0.3,
                0.7, 0.3, 1.0;
  auto raw = raw_with_covariance(
      covariance, Eigen::Vector3d::Zero(), 600, false);
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask =
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(
          raw.X[0].rows(), raw.X[0].cols());
  mask(0, 0) = 0;
  raw.X[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
  raw.mask.push_back(std::move(mask));
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto start = simple_start_values(pt, *rep, pack->start_stats, {});
  REQUIRE(start.has_value());

  auto ordinary = magmaan::estimate::fit_fiml(
      pt, *rep, raw, *start, *pack, Backend::NloptLbfgsSlsqpFallback,
      strict_options());
  REQUIRE(ordinary.has_value());
  REQUIRE_FALSE(ordinary->diagnostics.admissibility.admissible);

  auto psd = magmaan::estimate::fiml::frontier::fit_fiml_psd(
      pt, *rep, raw, ordinary->theta, *pack, Backend::NloptSlsqp,
      strict_options());
  REQUIRE_MESSAGE(psd.has_value(), "PSD FIML repair failed: "
      << (psd.has_value() ? std::string{} : psd.error().detail));
  check_psd_terminal(*psd);
  CHECK(psd->fmin > ordinary->fmin + 1e-7);
}

TEST_CASE("PSD fixed-weight GMM objective and equality Jacobian match central "
          "finite differences") {
  BuildOptions options;
  options.meanstructure = true;
  auto pt = lavaanify(
      "f =~ 1*x1 + a*x2 + b*x3\n"
      "x1 ~~ v*x1\n"
      "x2 ~~ v*x2\n"
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

  magmaan::estimate::gmm::Weight weight;
  Eigen::VectorXd diagonal(9);
  diagonal << 0.7, 1.1, 1.4, 0.8, 1.2, 1.6, 0.9, 1.3, 1.7;
  weight.push_back(diagonal.asDiagonal());
  auto probe = magmaan::estimate::psd_test::psd_gmm_derivative_probe(
      pt, *rep, samp, *start, weight);
  REQUIRE_MESSAGE(probe.has_value(), "PSD GMM derivative probe failed: "
      << (probe.has_value() ? std::string{} : probe.error().detail));
  CHECK((probe->analytic_gradient -
         probe->finite_difference_gradient).cwiseAbs().maxCoeff() < 3e-6);
  CHECK((probe->analytic_constraint_jacobian -
         probe->finite_difference_constraint_jacobian)
            .cwiseAbs().maxCoeff() < 3e-7);
}

TEST_CASE("PSD ULS GLS and fixed WLS agree with ordinary interior fits") {
  auto pt = lavaanify("f =~ x1 + x2 + x3 + x4");
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d loadings;
  loadings << 1.0, 0.8, 0.65, 0.9;
  Eigen::Vector4d residuals;
  residuals << 0.5, 0.6, 0.7, 0.55;
  auto samp = sample_stats(
      one_factor_covariance(loadings, residuals, 1.2), 500);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  magmaan::estimate::gmm::Weight weight;
  SUBCASE("ULS") {
    auto ordinary = magmaan::estimate::fit_gmm(
        pt, *rep, samp, *start, {}, {}, Backend::NloptLbfgs,
        strict_options());
    REQUIRE(ordinary.has_value());
    auto psd = magmaan::estimate::frontier::fit_gmm_psd(
        pt, *rep, samp, ordinary->theta, {}, Backend::NloptSlsqp,
        strict_options());
    REQUIRE_MESSAGE(psd.has_value(), "PSD ULS failed: "
        << (psd.has_value() ? std::string{} : psd.error().detail));
    check_psd_terminal(*psd);
    CHECK(psd->fmin == doctest::Approx(ordinary->fmin).epsilon(1e-8));
    CHECK((psd->theta - ordinary->theta).cwiseAbs().maxCoeff() < 2e-5);
  }
  SUBCASE("GLS") {
    auto ordinary = magmaan::estimate::fit_gls(
        pt, *rep, samp, *start, {}, Backend::NloptLbfgs,
        strict_options());
    REQUIRE(ordinary.has_value());
    auto psd = magmaan::estimate::frontier::fit_gls_psd(
        pt, *rep, samp, ordinary->theta, Backend::NloptSlsqp,
        strict_options());
    REQUIRE_MESSAGE(psd.has_value(), "PSD GLS failed: "
        << (psd.has_value() ? std::string{} : psd.error().detail));
    check_psd_terminal(*psd);
    CHECK(psd->fmin == doctest::Approx(ordinary->fmin).epsilon(1e-8));
    CHECK((psd->theta - ordinary->theta).cwiseAbs().maxCoeff() < 2e-5);
  }
  SUBCASE("fixed WLS") {
    Eigen::VectorXd diagonal = Eigen::VectorXd::LinSpaced(10, 0.6, 1.8);
    weight.push_back(diagonal.asDiagonal());
    auto ordinary = magmaan::estimate::fit_gmm(
        pt, *rep, samp, *start, weight, {}, Backend::NloptLbfgs,
        strict_options());
    REQUIRE(ordinary.has_value());
    auto psd = magmaan::estimate::frontier::fit_gmm_psd(
        pt, *rep, samp, ordinary->theta, weight, Backend::NloptSlsqp,
        strict_options());
    REQUIRE_MESSAGE(psd.has_value(), "PSD fixed WLS failed: "
        << (psd.has_value() ? std::string{} : psd.error().detail));
    check_psd_terminal(*psd);
    CHECK(psd->fmin == doctest::Approx(ordinary->fmin).epsilon(1e-8));
    CHECK((psd->theta - ordinary->theta).cwiseAbs().maxCoeff() < 2e-5);
  }
}

TEST_CASE("PSD ULS replaces an exact-fit negative-residual solution") {
  auto pt = lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Matrix3d covariance;
  covariance << 1.0, 0.7, 0.7,
                0.7, 1.0, 0.3,
                0.7, 0.3, 1.0;
  auto samp = sample_stats(covariance, 8);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  auto ordinary = magmaan::estimate::fit_gmm(
      pt, *rep, samp, *start, {}, {}, Backend::NloptLbfgs,
      strict_options());
  REQUIRE(ordinary.has_value());
  REQUIRE_FALSE(ordinary->diagnostics.admissibility.admissible);
  CHECK(ordinary->fmin < 1e-12);

  auto psd = magmaan::estimate::frontier::fit_gmm_psd(
      pt, *rep, samp, ordinary->theta, {}, Backend::NloptSlsqp,
      strict_options());
  REQUIRE_MESSAGE(psd.has_value(), "PSD ULS repair failed: "
      << (psd.has_value() ? std::string{} : psd.error().detail));
  check_psd_terminal(*psd);
  CHECK(psd->fmin > ordinary->fmin + 1e-6);
}

TEST_CASE("PSD fitted-weight GMM matches the ordinary interior fixed point") {
  auto pt = lavaanify("f =~ x1 + x2 + x3 + x4");
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d loadings;
  loadings << 1.0, 0.82, 0.68, 0.9;
  Eigen::Vector4d residuals;
  residuals << 0.55, 0.65, 0.6, 0.5;
  Eigen::Matrix4d covariance =
      one_factor_covariance(loadings, residuals, 1.1);
  covariance(0, 1) += 0.12;
  covariance(1, 0) += 0.12;
  covariance(2, 3) -= 0.08;
  covariance(3, 2) -= 0.08;
  auto samp = sample_stats(covariance, 450);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  magmaan::estimate::frontier::GmmFittedWeightOptions fitted_opts;
  fitted_opts.max_outer = 30;
  fitted_opts.theta_tol = 1e-8;
  fitted_opts.fmin_tol = 1e-11;
  auto ordinary = magmaan::estimate::frontier::fit_gmm_fitted_weight(
      pt, *rep, samp, *start, fitted_opts, {}, Backend::NloptSlsqp,
      strict_options());
  REQUIRE_MESSAGE(ordinary.has_value(), "ordinary fitted-weight GMM failed: "
      << (ordinary.has_value() ? std::string{} : ordinary.error().detail));
  REQUIRE(ordinary->diagnostics.admissibility.admissible);

  auto psd = magmaan::estimate::frontier::fit_gmm_fitted_weight_psd(
      pt, *rep, samp, *start, fitted_opts, Backend::NloptSlsqp,
      strict_options());
  REQUIRE_MESSAGE(psd.has_value(), "PSD fitted-weight GMM failed: "
      << (psd.has_value() ? std::string{} : psd.error().detail));
  check_psd_terminal(*psd);
  CHECK(psd->iterations > 1);
  CHECK(psd->fmin == doctest::Approx(ordinary->fmin).epsilon(2e-7));
  CHECK((psd->theta - ordinary->theta).cwiseAbs().maxCoeff() < 5e-5);

  auto ev = magmaan::model::ModelEvaluator::build(pt, *rep);
  REQUIRE(ev.has_value());
  auto final_weight = magmaan::estimate::gmm::expected_information_weight(
      *ev, samp, psd->theta);
  REQUIRE(final_weight.has_value());
  auto final_problem = magmaan::estimate::gmm::residuals(
      *ev, samp, psd->theta, *final_weight);
  REQUIRE(final_problem.has_value());
  const auto final_objective = magmaan::optim::scalarize(*final_problem);
  Eigen::VectorXd final_gradient(psd->theta.size());
  CHECK(psd->fmin == doctest::Approx(
      final_objective.f(psd->theta, final_gradient)).epsilon(1e-12));

  auto one_step_opts = fitted_opts;
  one_step_opts.max_outer = 1;
  auto one_step = magmaan::estimate::frontier::fit_gmm_fitted_weight_psd(
      pt, *rep, samp, *start, one_step_opts, Backend::NloptSlsqp,
      strict_options());
  REQUIRE(one_step.has_value());
  CHECK(one_step->optimizer_status ==
        magmaan::optim::OptimStatus::BudgetExhausted);
  CHECK((one_step->theta - psd->theta).cwiseAbs().maxCoeff() > 1e-6);
}

TEST_CASE("PSD fitted-weight GMM repairs a negative-residual fixed point") {
  auto pt = lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt);
  REQUIRE(rep.has_value());

  Eigen::Matrix3d covariance;
  covariance << 1.0, 0.7, 0.7,
                0.7, 1.0, 0.3,
                0.7, 0.3, 1.0;
  auto samp = sample_stats(covariance, 80);
  auto start = simple_start_values(pt, *rep, samp, {});
  REQUIRE(start.has_value());

  magmaan::estimate::frontier::GmmFittedWeightOptions fitted_opts;
  fitted_opts.max_outer = 30;
  fitted_opts.theta_tol = 1e-8;
  fitted_opts.fmin_tol = 1e-11;
  auto ordinary = magmaan::estimate::frontier::fit_gmm_fitted_weight(
      pt, *rep, samp, *start, fitted_opts, {}, Backend::NloptSlsqp,
      strict_options());
  REQUIRE(ordinary.has_value());
  REQUIRE_FALSE(ordinary->diagnostics.admissibility.admissible);

  auto psd = magmaan::estimate::frontier::fit_gmm_fitted_weight_psd(
      pt, *rep, samp, ordinary->theta, fitted_opts,
      Backend::NloptSlsqp, strict_options());
  REQUIRE_MESSAGE(psd.has_value(), "PSD fitted-weight repair failed: "
      << (psd.has_value() ? std::string{} : psd.error().detail));
  check_psd_terminal(*psd);
  CHECK(psd->fmin > ordinary->fmin + 1e-6);
}
