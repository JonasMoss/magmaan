#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include "magmaan/robust/satorra2000.hpp"

#include "magmaan/api/policy.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/frontier/fmg.hpp"

namespace {

namespace api = magmaan::api;
using magmaan::estimate::OrdinalParameterization;
using magmaan::estimate::OrdinalWeightKind;

// Four three-category items from one factor plus a nuisance factor shared by
// x1 and x2, so a one-factor model is misspecified and the estimated-weight
// influence is leading order.
Eigen::MatrixXd misspecified_block(std::uint32_t seed, Eigen::Index n, double cut1, double cut2) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  const double loading[4] = {0.75, 0.70, 0.65, 0.60};
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = z(rng), nuisance = z(rng);
    for (int j = 0; j < 4; ++j) {
      const double shared = j < 2 ? 0.45 : 0.0;
      const double residual = std::sqrt(1.0 - loading[j] * loading[j] - shared * shared);
      const double y = loading[j] * eta + shared * nuisance + residual * z(rng);
      X(i, j) = 1.0 + (y > cut1) + (y > cut2);
    }
  }
  return X;
}

struct OrdinalModel {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

OrdinalModel ordinal_model(const std::string& syntax, int groups,
                           std::vector<magmaan::spec::GroupEqual> equal = {}) {
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions options;
  options.n_groups = groups;
  options.group_equal = std::move(equal);
  auto pt = magmaan::spec::build(*fp, options);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}

const std::string kOneFactor =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
    "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";

magmaan::optim::OptimOptions tight() {
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  opts.ftol = 1e-14;
  opts.gtol = 1e-10;
  return opts;
}

magmaan::estimate::Estimates fit_dwls(const OrdinalModel& m, const magmaan::data::OrdinalStats& stats,
                                      OrdinalParameterization parameterization) {
  auto fit = magmaan::test::fit_ordinal_bounded(m.pt, m.rep, stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, tight(), parameterization);
  REQUIRE_MESSAGE(fit.has_value(), "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));
  return *fit;
}

double relative(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).norm() / b.norm();
}

std::string category_syntax(std::string syntax, int categories) {
  for (int j = 1; j <= 4; ++j) {
    const auto name = "x" + std::to_string(j);
    syntax += "\n" + name + " | t1";
    for (int k = 2; k < categories; ++k) syntax += " + t" + std::to_string(k);
    syntax += "\n" + name + " ~*~ 1*" + name;
  }
  return syntax;
}

magmaan::data::OrdinalStats exact_gamma(magmaan::data::OrdinalStats stats) {
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    auto sampling = magmaan::data::ordinal_moment_sampling_influence(stats.int_data[b],
        stats.n_levels[b], stats.thresholds[b], stats.R[b]);
    REQUIRE(sampling);
    stats.NACOV[b] = sampling->rows.transpose() * sampling->rows / double(stats.n_obs[b]);
  }
  return stats;
}

}  // namespace

TEST_CASE("DWLS policy: IJ covariance and one fit-function global test") {
  const Eigen::MatrixXd X = misspecified_block(20261002u, 400, -0.4, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  const auto m = ordinal_model(kOneFactor, 1);
  const auto est = fit_dwls(m, *stats, OrdinalParameterization::Delta);
  const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                              OrdinalParameterization::Delta, {});
  REQUIRE(out.covariance_reason == api::InferenceReason::Available);
  auto ij = magmaan::estimate::robust_ordinal_ij(m.pt, m.rep, *stats, est, OrdinalWeightKind::DWLS, OrdinalParameterization::Delta, nullptr, magmaan::estimate::OrdinalFirstStage::Exact);
  auto fixed = magmaan::estimate::robust_ordinal(m.pt, m.rep, exact_gamma(*stats), est, OrdinalWeightKind::DWLS);
  auto fixed_observed = magmaan::estimate::robust_ordinal(m.pt, m.rep, *stats, est,
      OrdinalWeightKind::DWLS, OrdinalParameterization::Delta, magmaan::robust::Information::Observed);
  REQUIRE(ij.has_value()); REQUIRE(fixed.has_value()); REQUIRE(fixed_observed.has_value());
  CHECK((out.covariance - ij->vcov).norm() == 0.0);
  // The weight influence is not negligible under this misspecification.
  CHECK(relative(ij->vcov, fixed_observed->vcov) > 1e-3);

  REQUIRE(out.score.reason == api::InferenceReason::Available);
  CHECK(out.score.label == "fit_function");
  CHECK(out.score.statistic == fixed->chisq_standard);
  CHECK(out.score.df == fixed->df);
  CHECK(out.score.df == 2);
  Eigen::VectorXd eig = Eigen::VectorXd::Zero(fixed->df);
  eig.tail(fixed->eigvals.size()) = fixed->eigvals;
  std::sort(eig.data(), eig.data() + eig.size());
  CHECK((out.score.eigenvalues - eig).norm() == 0.0);
  CHECK(out.score.reference == "all");
  CHECK(out.score.p_all == doctest::Approx(magmaan::robust::frontier::fmg_test(
      out.score.statistic, out.score.df, eig,
      {magmaan::robust::frontier::FmgMethod::All, 0.0, true}).p_value).epsilon(1e-7));
  CHECK(std::isnan(out.score.p_sb));
  CHECK(std::isnan(out.score.p_peba4));
  CHECK(std::isnan(out.score.sb_scale));
  CHECK(out.score.peba_blocks == 0);
  CHECK(out.lr.reference == "sb_peba4");
  CHECK(out.lr.reason == api::InferenceReason::Inapplicable);
  CHECK(api::reason_name(out.lr.reason) == "inapplicable");

  api::PolicyFitState failed;
  failed.converged = false;
  const auto refused = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                                  OrdinalParameterization::Delta, failed);
  CHECK(refused.covariance_reason == api::InferenceReason::NotConverged);
  CHECK(refused.score.reason == api::InferenceReason::NotConverged);
}

TEST_CASE("DWLS global All reference: delta and theta, one and two groups") {
  for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
    for (int groups : {1, 2}) {
      for (int categories : {2, 5}) {
        CAPTURE(categories);
        CAPTURE(groups);
        CAPTURE(static_cast<int>(parameterization));
        std::vector<Eigen::MatrixXd> blocks;
        for (int g = 0; g < groups; ++g)
          blocks.push_back(misspecified_block(20261003u + static_cast<std::uint32_t>(g), 400, -0.4, 0.6));
        unsigned seed = 6900u;
        for (auto& block : blocks) {
          // Deterministic latent-normal draws with all categories occupied.
          std::mt19937 rng(seed++); std::normal_distribution<double> z;
          for (Eigen::Index i = 0; i < block.rows(); ++i) {
            const double f = z(rng);
            for (int j = 0; j < 4; ++j) {
              const double y = .7*f + std::sqrt(.51)*z(rng);
              int category = 1;
              for (int k = 1; k < categories; ++k) category += y > -.9 + 1.8*k/categories;
              block(i,j) = category;
            }
          }
        }
        auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
        REQUIRE(stats.has_value());
        const auto m = ordinal_model(category_syntax("f =~ x1 + x2 + x3 + x4", categories), groups);
        const auto est = fit_dwls(m, *stats, parameterization);
        const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est, parameterization, {});
        REQUIRE(out.score.reason == api::InferenceReason::Available);
        auto ij = magmaan::estimate::robust_ordinal_ij(m.pt, m.rep, *stats, est,
            OrdinalWeightKind::DWLS, parameterization, nullptr, magmaan::estimate::OrdinalFirstStage::Exact);
        auto global = magmaan::estimate::robust_ordinal(m.pt, m.rep, exact_gamma(*stats), est,
            OrdinalWeightKind::DWLS, parameterization);
        REQUIRE(ij); REQUIRE(global);
        CHECK((out.covariance - ij->vcov).norm() == 0.0);
        CHECK((out.score.eigenvalues - global->eigvals).norm() <= 1e-10);
        auto missing = *stats; missing.int_data.clear();
        CHECK(api::policy_inference_dwls(m.pt, m.rep, missing, est, parameterization, {}).covariance_reason
            == api::InferenceReason::UnsupportedModel);
        CHECK(out.score.reference == "all");
        CHECK(out.lr.reference == "sb_peba4");
        const auto explicit_all = magmaan::robust::frontier::fmg_test(out.score.statistic,
            out.score.df, out.score.eigenvalues,
            {magmaan::robust::frontier::FmgMethod::All, 0.0, true});
        CHECK(std::abs(out.score.p_all - explicit_all.p_value) <= 1e-7);
        CHECK(std::isnan(out.score.p_sb));
        CHECK(std::isnan(out.score.p_peba4));
        CHECK(out.score.peba_blocks == 0);
      }
    }
  }
}

TEST_CASE("DWLS policy: at exact fit the IJ covariance is the fixed-weight sandwich") {
  // A saturated three-item model reproduces the polychorics, so the residuals,
  // and with them the weight influence and the observed-minus-expected bread,
  // vanish.
  const Eigen::MatrixXd X4 = misspecified_block(77u, 500, -0.3, 0.7);
  const Eigen::MatrixXd X = X4.leftCols(3);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  const auto m = ordinal_model(
      "f =~ x1 + x2 + x3\nx1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\n", 1);
  const auto est = fit_dwls(m, *stats, OrdinalParameterization::Delta);
  const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                              OrdinalParameterization::Delta, {});
  REQUIRE(out.covariance_reason == api::InferenceReason::Available);
  CHECK(out.score.reason == api::InferenceReason::Saturated);
  CHECK(out.lr.reason == api::InferenceReason::Inapplicable);
  auto fixed = magmaan::estimate::robust_ordinal(m.pt, m.rep, exact_gamma(*stats), est, OrdinalWeightKind::DWLS);
  REQUIRE(fixed.has_value());
  CHECK(relative(out.covariance, fixed->vcov) < 1e-6);
}

TEST_CASE("DWLS exact policy covariance transports from delta to theta") {
  // Matching moment-map Jacobians gives the local delta-to-theta coordinate
  // derivative. The observed IJ covariance must obey this reparameterization
  // even when the one-factor model is misspecified.
  for (const Eigen::Index n : {Eigen::Index{600}, Eigen::Index{2400}}) {
    CAPTURE(n);
    auto stats = magmaan::data::ordinal_stats_from_integer_data(
        {misspecified_block(4000u, n, -0.4, 0.6)}, true);
    REQUIRE(stats);
    const auto m = ordinal_model(kOneFactor, 1);
    const auto est = fit_dwls(m, *stats, OrdinalParameterization::Theta);
    const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
        OrdinalParameterization::Theta, {});
    REQUIRE(out.covariance_reason == api::InferenceReason::Available);
    const auto delta_est = fit_dwls(m, *stats, OrdinalParameterization::Delta);
    const auto delta_out = api::policy_inference_dwls(m.pt, m.rep, *stats, delta_est,
        OrdinalParameterization::Delta, {});
    REQUIRE(delta_out.covariance_reason == api::InferenceReason::Available);
    auto delta_obj = magmaan::estimate::frontier::ordinal_ls_objective(m.pt, m.rep, *stats,
        delta_est, OrdinalWeightKind::DWLS, OrdinalParameterization::Delta);
    auto theta_obj = magmaan::estimate::frontier::ordinal_ls_objective(m.pt, m.rep, *stats,
        est, OrdinalWeightKind::DWLS, OrdinalParameterization::Theta);
    REQUIRE(delta_obj); REQUIRE(theta_obj);
    auto jd = delta_obj->problem.J(delta_est.theta);
    auto jt = theta_obj->problem.J(est.theta);
    REQUIRE(jd); REQUIRE(jt);
    const Eigen::MatrixXd transport = jt->colPivHouseholderQr().solve(*jd);
    const Eigen::MatrixXd transported = transport * delta_out.covariance * transport.transpose();
    MESSAGE("delta/theta transport error " << relative(out.covariance, transported)
        << " Jacobian residual " << relative((*jt * transport).eval(), *jd)
        << " objective difference " << est.fmin - delta_est.fmin);
    CHECK(relative(out.covariance, transported) < 1e-6);
    CHECK(relative((*jt * transport).eval(), *jd) < 1e-8);
  }
}

TEST_CASE("DWLS policy IJ covariance agrees with the delete-one jackknife") {
  struct Design {
    const char* name;
    int groups;
    OrdinalParameterization parameterization;
  };
  for (const Design design : {Design{"delta, one group", 1, OrdinalParameterization::Delta},
                              Design{"delta, two groups", 2, OrdinalParameterization::Delta},
                              Design{"theta, one group", 1, OrdinalParameterization::Theta}}) {
    CAPTURE(design.name);
    std::vector<Eigen::MatrixXd> blocks;
    for (int g = 0; g < design.groups; ++g)
      blocks.push_back(misspecified_block(4000u + static_cast<std::uint32_t>(g),
          (design.parameterization == OrdinalParameterization::Theta ? 2400 : 600) + 40 * g,
                                          -0.4 + 0.2 * g, 0.6));
    auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
    REQUIRE(stats.has_value());
    const auto m = ordinal_model(kOneFactor, design.groups);
    const auto est = fit_dwls(m, *stats, design.parameterization);
    const auto out = api::policy_inference_dwls(m.pt, m.rep, *stats, est,
                                                design.parameterization, {});
    REQUIRE(out.covariance_reason == api::InferenceReason::Available);
    auto fixed_observed = magmaan::estimate::robust_ordinal(m.pt, m.rep, *stats, est,
        OrdinalWeightKind::DWLS, design.parameterization, magmaan::robust::Information::Observed);
    REQUIRE(fixed_observed.has_value());

    // Stratified delete-one jackknife: refit without each case, warm-started
    // at the full-sample estimate.
    const Eigen::Index k = est.theta.size();
    Eigen::MatrixXd jackknife = Eigen::MatrixXd::Zero(k, k);
    for (int g = 0; g < design.groups; ++g) {
      const Eigen::Index n = blocks[static_cast<std::size_t>(g)].rows();
      Eigen::MatrixXd thetas(n, k);
      for (Eigen::Index i = 0; i < n; ++i) {
        std::vector<Eigen::MatrixXd> reduced = blocks;
        Eigen::MatrixXd& x = reduced[static_cast<std::size_t>(g)];
        Eigen::MatrixXd y(n - 1, x.cols());
        y.topRows(i) = x.topRows(i);
        y.bottomRows(n - 1 - i) = x.bottomRows(n - 1 - i);
        x = std::move(y);
        auto s = magmaan::data::ordinal_stats_from_integer_data(reduced, true);
        REQUIRE(s.has_value());
        auto refit = magmaan::estimate::fit_ordinal_bounded(m.pt, m.rep, *s, {},
            OrdinalWeightKind::DWLS, est.theta, magmaan::estimate::Backend::NloptLbfgs,
            tight(), design.parameterization);
        REQUIRE(refit.has_value());
        thetas.row(i) = refit->theta.transpose();
      }
      const Eigen::RowVectorXd mean = thetas.colwise().mean();
      const Eigen::MatrixXd centered = thetas.rowwise() - mean;
      jackknife += (static_cast<double>(n - 1) / static_cast<double>(n)) *
                   centered.transpose() * centered;
    }
    const Eigen::VectorXd j = jackknife.diagonal();
    const double ij_error = (out.covariance.diagonal() - j).norm() / j.norm();
    const double fixed_error = (fixed_observed->vcov.diagonal() - j).norm() / j.norm();
    MESSAGE(std::string(design.name) << ": IJ vs jackknife " << ij_error << ", fixed weight vs jackknife "
            << fixed_error);
    // The two estimators of the same asymptotic covariance agree to O(1/n).
    // The fixed-weight sandwich omits the weight influence; its gap does not
    // shrink with n. On the delta scale here that gap dominates the sampling
    // noise; on the theta scale it is smaller than the n = 600 noise (it
    // separates by n = 2400: IJ 1%, fixed 4%). Theta therefore uses n = 2400
    // with the same 4% gate; delta retains n = 600. Delta/theta IJ transport
    // agrees within 1e-6 and tighter refits are unchanged: the theta gap is
    // finite-sample reparameterization nonlinearity (5.23% -> 1.32% for 4x n).
    // OPG's 0.31% theta gap at n = 600 was a cancellation; its delta gaps
    // were 2.3-2.9%.
    CHECK(ij_error < 0.04);
    if (design.parameterization == OrdinalParameterization::Delta) CHECK(ij_error < fixed_error);
  }
}

namespace {

const std::string kTauEquivalent =
    "f =~ x1 + a*x2 + a*x3 + a*x4\n"
    "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
    "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";

// Items whose population satisfies the tau-equivalent null (x2-x4 share one
// loading), so the null and the congeneric alternative both fit.
Eigen::MatrixXd tau_equivalent_block(std::uint32_t seed, Eigen::Index n) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  const double loading[4] = {0.80, 0.65, 0.65, 0.65};
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = z(rng);
    for (int j = 0; j < 4; ++j) {
      const double y = loading[j] * eta + std::sqrt(1.0 - loading[j] * loading[j]) * z(rng);
      X(i, j) = 1.0 + (y > -0.3) + (y > 0.7);
    }
  }
  return X;
}

struct NestedDwls {
  api::PolicyNested out;
  magmaan::estimate::Estimates null_est, alt_est;
  OrdinalModel null_model, alt_model;
};

NestedDwls nested_dwls(const magmaan::data::OrdinalStats& stats, int groups,
                       OrdinalParameterization parameterization) {
  NestedDwls r{{}, {}, {}, ordinal_model(kTauEquivalent, groups), ordinal_model(kOneFactor, groups)};
  r.null_est = fit_dwls(r.null_model, stats, parameterization);
  r.alt_est = fit_dwls(r.alt_model, stats, parameterization);
  r.out = api::policy_nested_dwls(r.null_model.pt, r.null_model.rep, r.null_est, {},
                                  r.alt_model.pt, r.alt_model.rep, r.alt_est, {}, stats,
                                  parameterization);
  return r;
}

Eigen::VectorXd check_common_law(const NestedDwls& r, const magmaan::data::OrdinalStats& stats_value,
                                 OrdinalParameterization parameterization) {
  const auto* stats = &stats_value;
  auto p1 = r.alt_model.pt, p0 = r.null_model.pt;
  // Prepare under the fit's parameterization: DELTA and THETA release
  // group-2+ response scales through different coordinates.
  REQUIRE(magmaan::estimate::prepare_ordinal_partable(p1, *stats, parameterization));
  REQUIRE(magmaan::estimate::prepare_ordinal_partable(p0, *stats, parameterization));
  auto c1 = magmaan::estimate::build_eq_constraints(p1);
  auto c0 = magmaan::estimate::build_eq_constraints(p0);
  REQUIRE(c1); REQUIRE(c0);
  auto embed = magmaan::robust::embed_nested_null(p1, r.alt_model.rep, p0,
      r.null_model.rep, r.null_est.theta, *c1, *c0);
  REQUIRE(embed);
  const auto& t = r.out.lr;
  CHECK(t.reference == "all");
  CHECK(t.p_all == doctest::Approx(magmaan::robust::frontier::fmg_test(
      t.statistic, t.df, t.eigenvalues,
      {magmaan::robust::frontier::FmgMethod::All, 0.0, true}).p_value).epsilon(1e-12));
  REQUIRE(t.eigenvalues.size() == t.df);
  CHECK(t.eigenvalues.allFinite());
  CHECK(t.eigenvalues.minCoeff() >= 0.0);
  auto tangent = api::frontier::moment_nested_tangent(r.null_model.pt, r.null_model.rep,
      r.null_est, r.alt_model.pt, r.alt_model.rep, r.alt_est, *stats, parameterization);
  REQUIRE_MESSAGE(tangent.has_value(), (tangent.has_value() ? "" : tangent.error().detail));
  auto parts = magmaan::estimate::frontier::ordinal_ls_newton_parts_prepared(p1,
      r.alt_model.rep, *stats, r.alt_est.theta, OrdinalWeightKind::DWLS, parameterization);
  auto ij = magmaan::estimate::robust_ordinal_ij(r.alt_model.pt, r.alt_model.rep, *stats,
      r.alt_est, OrdinalWeightKind::DWLS, parameterization, nullptr, magmaan::estimate::OrdinalFirstStage::Exact);
  REQUIRE(parts); REQUIRE(ij);
  const double N = std::accumulate(stats->n_obs.begin(), stats->n_obs.end(), 0.0);
  const auto& K = c1->K();
  const Eigen::MatrixXd H = K.transpose() * parts->hessian * K / N;
  const Eigen::MatrixXd L = (K.transpose() * K).ldlt().solve(K.transpose());
  const Eigen::MatrixXd V = N * L * ij->vcov * L.transpose();
  const Eigen::MatrixXd B = H * V * H.transpose();
  auto tangent_spectrum = magmaan::robust::compute_satorra2000_from_sandwich(H, B, tangent->A);
  REQUIRE(tangent_spectrum);
  const Eigen::VectorXd eig = tangent_spectrum->eigenvalues;
  CHECK((t.eigenvalues - eig).norm() <= 1e-10 * eig.norm());
  // Independently evaluate [H^-1 - T(T'HT)^-1T']B. Whiten by H and
  // form the orthogonal complement instead of subtracting two nearly equal
  // dense inverses (which loses precision in the released-scale coordinates).
  Eigen::LLT<Eigen::MatrixXd> chol(H);
  REQUIRE(chol.info() == Eigen::Success);
  const Eigen::MatrixXd Z = chol.matrixU() * tangent->tangent;
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(Z);
  const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(H.rows(), H.rows());
  const Eigen::MatrixXd Y = chol.matrixU().solve(Q.rightCols(t.df));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ep(Y.transpose() * B * Y);
  REQUIRE(ep.info() == Eigen::Success);
  CHECK((ep.eigenvalues() - eig).norm() <= 1e-10 * eig.norm());
  return eig;
}

}  // namespace

TEST_CASE("DWLS nested policy: fit-function difference with the parameter-space estimated-weight IJ law") {
  const Eigen::MatrixXd X = misspecified_block(5150u, 600, -0.4, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  const auto r = nested_dwls(*stats, 1, OrdinalParameterization::Delta);
  const auto& t = r.out.lr;
  REQUIRE(t.reason == api::InferenceReason::Available);
  CHECK(t.label == "fit_function_difference");
  CHECK(t.df == 2);
  CHECK(r.out.score.reason == api::InferenceReason::UnsupportedModel);

  auto profile = magmaan::estimate::ordinal_dwls_profile_lrt(r.alt_model.pt, r.alt_model.rep,
      *stats, r.alt_est, r.null_model.pt, r.null_model.rep, r.null_est);
  REQUIRE(profile.has_value());
  CHECK(t.statistic == doctest::Approx(profile->T_diff));
  // The statistic is the difference of the two global fit-function statistics.
  auto g0 = magmaan::estimate::robust_ordinal(r.null_model.pt, r.null_model.rep, *stats,
                                              r.null_est, OrdinalWeightKind::DWLS);
  auto g1 = magmaan::estimate::robust_ordinal(r.alt_model.pt, r.alt_model.rep, *stats,
                                              r.alt_est, OrdinalWeightKind::DWLS);
  REQUIRE(g0.has_value()); REQUIRE(g1.has_value());
  CHECK(t.statistic == doctest::Approx(g0->chisq_standard - g1->chisq_standard).epsilon(1e-8));
  const auto eig = check_common_law(r, *stats, OrdinalParameterization::Delta);
  CHECK(t.sb_scale == doctest::Approx(eig.sum() / 2.0));
  CHECK(t.p_sb == doctest::Approx(magmaan::inference::chi2_pvalue(t.statistic / t.sb_scale, 2)));
  CHECK(std::isfinite(t.p_peba4));
  const int m = static_cast<int>(t.eigenvalues.size());
  const int width = (m + 3) / 4;
  CHECK(t.peba_blocks == (m + width - 1) / width);
  MESSAGE("spectrum size " << profile->spectrum_size << ", negative " << profile->negative_spectrum_size);

  // The roles matter: the alternative does not restrict the null.
  const auto swapped = api::policy_nested_dwls(r.alt_model.pt, r.alt_model.rep, r.alt_est, {},
      r.null_model.pt, r.null_model.rep, r.null_est, {}, *stats, OrdinalParameterization::Delta);
  CHECK(swapped.lr.reason != api::InferenceReason::Available);
}

TEST_CASE("DWLS nested policy: delta and theta match their common-point laws; two groups compose") {
  const Eigen::MatrixXd X = misspecified_block(5151u, 600, -0.4, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  const auto delta = nested_dwls(*stats, 1, OrdinalParameterization::Delta);
  const auto theta = nested_dwls(*stats, 1, OrdinalParameterization::Theta);
  REQUIRE(delta.out.lr.reason == api::InferenceReason::Available);
  REQUIRE(theta.out.lr.reason == api::InferenceReason::Available);
  CHECK(theta.out.lr.statistic == doctest::Approx(delta.out.lr.statistic).epsilon(1e-6));
  REQUIRE(theta.out.lr.eigenvalues.size() == delta.out.lr.eigenvalues.size());
  check_common_law(delta, *stats, OrdinalParameterization::Delta);
  check_common_law(theta, *stats, OrdinalParameterization::Theta);
  // Away from the nested null, the affine restrictions at H1 describe
  // different tangents under nonlinear delta/theta response scaling. The
  // r-term law must match the direct Exact IJ law in each coordinate system;
  // equal statistics do not imply equal reference spectra there.

  auto grouped = magmaan::data::ordinal_stats_from_integer_data(
      {misspecified_block(5152u, 500, -0.4, 0.6), misspecified_block(5153u, 450, -0.2, 0.6)}, true);
  REQUIRE(grouped.has_value());
  const auto two = nested_dwls(*grouped, 2, OrdinalParameterization::Delta);
  REQUIRE(two.out.lr.reason == api::InferenceReason::Available);
  CHECK(two.out.lr.df == 5);  // x2-x4 equal within and across groups: 6 loadings -> 1
  auto g0 = magmaan::estimate::robust_ordinal(two.null_model.pt, two.null_model.rep, *grouped,
                                              two.null_est, OrdinalWeightKind::DWLS);
  auto g1 = magmaan::estimate::robust_ordinal(two.alt_model.pt, two.alt_model.rep, *grouped,
                                              two.alt_est, OrdinalWeightKind::DWLS);
  REQUIRE(g0.has_value()); REQUIRE(g1.has_value());
  CHECK(two.out.lr.statistic ==
        doctest::Approx(g0->chisq_standard - g1->chisq_standard).epsilon(1e-8));
}

namespace {
Eigen::MatrixXd cross_loading_block(std::uint32_t seed, Eigen::Index n) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X(n, 6);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double f1 = z(rng), f2 = 0.3 * f1 + std::sqrt(0.91) * z(rng);
    for (int j = 0; j < 6; ++j) {
      const double cross = j == 1 ? 0.3 : 0.0;
      const double residual = std::sqrt(1.0 - 0.7 * 0.7 - cross * cross - 2.0 * 0.7 * cross * 0.3);
      const double y = 0.7 * (j < 3 ? f1 : f2) + cross * f2 + residual * z(rng);
      X(i, j) = 1.0 + (y > -0.4) + (y > 0.6);
    }
  }
  return X;
}
}  // namespace

TEST_CASE("DWLS nested policy: two-group theta invariance laws match common-point profiles") {
  auto stats = magmaan::data::ordinal_stats_from_integer_data(
      {cross_loading_block(5162u, 500), cross_loading_block(5163u, 450)}, true);
  REQUIRE(stats);
  // The fitted CFA omits x2's population cross-loading on f2.
  const std::string syntax = "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x4 | t1 + t2\nx5 | t1 + t2\nx6 | t1 + t2\n";
  using magmaan::spec::GroupEqual;
  for (const bool thresholds : {false, true}) {
    const std::vector<GroupEqual> eq1 = thresholds ? std::vector<GroupEqual>{GroupEqual::Thresholds}
                                                   : std::vector<GroupEqual>{};
    auto eq0 = eq1; eq0.push_back(GroupEqual::Loadings);
    NestedDwls r;
    r.alt_model = ordinal_model(syntax, 2, eq1);
    r.null_model = ordinal_model(syntax, 2, eq0);
    r.alt_est = fit_dwls(r.alt_model, *stats, OrdinalParameterization::Theta);
    r.null_est = fit_dwls(r.null_model, *stats, OrdinalParameterization::Theta);
    r.out = api::policy_nested_dwls(r.null_model.pt, r.null_model.rep, r.null_est, {},
        r.alt_model.pt, r.alt_model.rep, r.alt_est, {}, *stats, OrdinalParameterization::Theta);
    REQUIRE_MESSAGE(r.out.lr.reason == api::InferenceReason::Available, r.out.lr.detail);
    check_common_law(r, *stats, OrdinalParameterization::Theta);
  }
}

TEST_CASE("DWLS nested policy: under a true null the IJ law approaches Satorra-2000") {
  // The weight channel is driven by the residuals, which vanish under correct
  // specification, so the estimated-weight spectrum converges to the
  // fixed-weight Satorra-2000 spectrum (exact restriction map) as n grows.
  for (const auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
  double previous = 1.0;
  for (const Eigen::Index n : {Eigen::Index{4000}, Eigen::Index{64000}}) {
    CAPTURE(n);
    auto stats = magmaan::data::ordinal_stats_from_integer_data({tau_equivalent_block(61u, n)}, true);
    REQUIRE(stats.has_value());
    const auto r = nested_dwls(*stats, 1, parameterization);
    REQUIRE(r.out.lr.reason == api::InferenceReason::Available);
    auto g0 = magmaan::estimate::robust_ordinal(r.null_model.pt, r.null_model.rep, *stats,
                                                r.null_est, OrdinalWeightKind::DWLS, parameterization);
    auto g1 = magmaan::estimate::robust_ordinal(r.alt_model.pt, r.alt_model.rep, *stats,
                                                r.alt_est, OrdinalWeightKind::DWLS, parameterization);
    REQUIRE(g0.has_value()); REQUIRE(g1.has_value());
    // Match the sampling Gamma while retaining the fitted OPG weights. With
    // OPG Gamma the comparison also contains random information-equality noise.
    const auto matched_stats = exact_gamma(*stats);
    CHECK((matched_stats.W_dwls[0] - stats->W_dwls[0]).norm() == 0.0);
    auto opg_satorra = magmaan::estimate::lr_test_satorra2000_ordinal(r.alt_model.pt, r.alt_model.rep,
        *stats, r.alt_est, r.null_model.pt, r.null_model.rep, r.null_est, OrdinalWeightKind::DWLS,
        g0->chisq_standard, g1->chisq_standard, g0->df, g1->df,
        magmaan::robust::SatorraAMethod::Exact, parameterization);
    REQUIRE(opg_satorra);
    MESSAGE("n " << n << ": OPG Gamma trace gap " <<
        std::abs(r.out.lr.eigenvalues.sum() - opg_satorra->eigenvalues.sum()) /
        opg_satorra->eigenvalues.sum());
    auto satorra = magmaan::estimate::lr_test_satorra2000_ordinal(r.alt_model.pt, r.alt_model.rep,
        matched_stats, r.alt_est, r.null_model.pt, r.null_model.rep, r.null_est, OrdinalWeightKind::DWLS,
        g0->chisq_standard, g1->chisq_standard, g0->df, g1->df,
        magmaan::robust::SatorraAMethod::Exact, parameterization);
    REQUIRE(satorra.has_value());
    const Eigen::VectorXd& e = r.out.lr.eigenvalues;
    REQUIRE(e.size() >= satorra->eigenvalues.size());
    const double gap = std::abs(e.sum() - satorra->eigenvalues.sum()) / satorra->eigenvalues.sum();
    MESSAGE("n " << n << ": trace gap " << gap << ", spectrum sizes " << e.size() << " vs "
            << satorra->eigenvalues.size());
    CHECK(gap < previous);
    previous = gap;
  }
  CHECK(previous < 0.02);
  }
}


TEST_CASE("DWLS moment nesting: Wu-Estabrook threshold steps") {
  using magmaan::spec::GroupEqual;
  for (const int categories : {3, 5}) {
    std::vector<Eigen::MatrixXd> blocks;
    for (unsigned seed : {7301u, 7302u}) {
      std::mt19937 rng(seed);
      std::normal_distribution<double> z;
      Eigen::MatrixXd X(1000, 4);
      for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const double f = z(rng);
        for (int j = 0; j < 4; ++j) {
          const double y = 0.7 * f + std::sqrt(0.51) * z(rng);
          int category = 1;
          for (int k = 1; k < categories; ++k)
            category += y > (categories == 3 ? (k == 1 ? -0.4 : 0.6) : -1.5 + k * 0.6);
          X(i,j) = category;
        }
      }
      blocks.push_back(X);
    }
    auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
    REQUIRE(stats);
    std::string syntax = "f =~ x1 + x2 + x3 + x4\n";
    for (int j = 1; j <= 4; ++j) {
      syntax += "x" + std::to_string(j) + " | t1";
      for (int k = 2; k < categories; ++k) syntax += " + t" + std::to_string(k);
      syntax += "\n";
    }
    const auto alt = ordinal_model(syntax, 2);
    const auto nul = ordinal_model(syntax, 2, {GroupEqual::Thresholds});
    const auto e1 = fit_dwls(alt, *stats, OrdinalParameterization::Theta);
    const auto e0 = fit_dwls(nul, *stats, OrdinalParameterization::Theta);
    const auto tangent = api::frontier::moment_nested_tangent(nul.pt, nul.rep, e0,
        alt.pt, alt.rep, e1, *stats, OrdinalParameterization::Theta);
    REQUIRE_MESSAGE(tangent.has_value(), (tangent.has_value() ? "" : tangent.error().detail));
    CHECK(tangent->A.rows() == 4 * (categories - 3));
    auto o0 = magmaan::estimate::frontier::ordinal_ls_objective(nul.pt, nul.rep,
        *stats, e0, OrdinalWeightKind::DWLS, OrdinalParameterization::Theta);
    auto o1 = magmaan::estimate::frontier::ordinal_ls_objective(alt.pt, alt.rep,
        *stats, e1, OrdinalWeightKind::DWLS, OrdinalParameterization::Theta);
    REQUIRE(o0); REQUIRE(o1);
    CHECK((*o0->problem.r(e0.theta) - *o1->problem.r(tangent->embedding)).norm() < 1e-8);
    const auto out = api::policy_nested_dwls(nul.pt, nul.rep, e0, api::policy_fit_state(e0), alt.pt,
        alt.rep, e1, api::policy_fit_state(e1), *stats, OrdinalParameterization::Theta);
    MESSAGE("categories=" << categories << " reason=" << api::reason_name(out.lr.reason)
        << " detail=" << out.lr.detail);
    const auto loading_alt = ordinal_model(syntax, 2, {GroupEqual::Loadings});
    const auto loading_null = ordinal_model(syntax, 2, {GroupEqual::Thresholds, GroupEqual::Loadings});
    const auto loading_e1 = fit_dwls(loading_alt, *stats, OrdinalParameterization::Theta);
    const auto loading_e0 = fit_dwls(loading_null, *stats, OrdinalParameterization::Theta);
    auto loading_tangent = api::frontier::moment_nested_tangent(loading_null.pt,
        loading_null.rep, loading_e0, loading_alt.pt, loading_alt.rep, loading_e1,
        *stats, OrdinalParameterization::Theta);
    MESSAGE("threshold+loadings inside loadings-only categories=" << categories
        << " embedding=" << loading_tangent.has_value()
        << " detail=" << (loading_tangent ? "" : loading_tangent.error().detail));
    REQUIRE_FALSE(loading_tangent.has_value());
    CHECK(loading_tangent.error().kind == magmaan::PostError::Kind::NotNested);
    const auto refused = api::policy_nested_dwls(loading_null.pt, loading_null.rep,
        loading_e0, {}, loading_alt.pt, loading_alt.rep, loading_e1, {}, *stats,
        OrdinalParameterization::Theta);
    CHECK(refused.lr.reason == api::InferenceReason::NotNested);
    if (categories == 3) CHECK(out.lr.reason == api::InferenceReason::EquivalentModels);
    else {
      REQUIRE(out.lr.reason == api::InferenceReason::Available);
      CHECK(out.lr.df == 8);
      CHECK(out.lr.reference == "all");
      CHECK(out.lr.p_all == doctest::Approx(magmaan::robust::frontier::fmg_test(
          out.lr.statistic, out.lr.df, out.lr.eigenvalues,
          {magmaan::robust::frontier::FmgMethod::All, 0.0, true}).p_value).epsilon(1e-12));

      CHECK(out.lr.eigenvalues.allFinite());
      auto prepared = alt.pt;
      REQUIRE(magmaan::estimate::prepare_ordinal_partable(prepared, *stats, OrdinalParameterization::Theta));
      auto constraints = magmaan::estimate::build_eq_constraints(prepared); REQUIRE(constraints);
      const auto& K = constraints->K();
      auto parts = magmaan::estimate::frontier::ordinal_ls_newton_parts_prepared(prepared,
          alt.rep, *stats, e1.theta, OrdinalWeightKind::DWLS, OrdinalParameterization::Theta);
      auto ij = magmaan::estimate::robust_ordinal_ij(alt.pt, alt.rep, *stats, e1,
          OrdinalWeightKind::DWLS, OrdinalParameterization::Theta, nullptr,
          magmaan::estimate::OrdinalFirstStage::Exact);
      REQUIRE(parts); REQUIRE(ij);
      const double N = std::accumulate(stats->n_obs.begin(), stats->n_obs.end(), 0.0);
      const Eigen::MatrixXd H = K.transpose() * parts->hessian * K / N;
      const Eigen::MatrixXd L = (K.transpose() * K).ldlt().solve(K.transpose());
      const Eigen::MatrixXd B = H * (N * L * ij->vcov * L.transpose()) * H.transpose();
      auto direct = magmaan::robust::compute_satorra2000_from_sandwich(H, B, tangent->A);
      REQUIRE(direct);
      CHECK((out.lr.eigenvalues - direct->eigenvalues).norm() <= 1e-10);
    }
  }
}


TEST_CASE("DWLS moment nesting: 100-replicate true-null moment diagnostic") {
  const std::string syntax = "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2 + t3 + t4\nx2 | t1 + t2 + t3 + t4\n"
      "x3 | t1 + t2 + t3 + t4\nx4 | t1 + t2 + t3 + t4\n";
  const auto alt = ordinal_model(syntax, 2);
  const auto nul = ordinal_model(syntax, 2, {magmaan::spec::GroupEqual::Thresholds});
  Eigen::VectorXd statistics(100), traces(100), variances(100);
  std::mt19937 rng(68261004u);
  std::normal_distribution<double> z;
  for (int rep = 0; rep < 100; ++rep) {
    std::vector<Eigen::MatrixXd> blocks;
    for (int group = 0; group < 2; ++group) {
      Eigen::MatrixXd X(1000, 4);
      for (Eigen::Index i = 0; i < X.rows(); ++i) {
        const double f = z(rng);
        for (int j = 0; j < 4; ++j) {
          const double y = 0.7 * f + std::sqrt(0.51) * z(rng);
          X(i,j) = 1.0 + (y > -0.9) + (y > -0.3) + (y > 0.3) + (y > 0.9);
        }
      }
      blocks.push_back(X);
    }
    auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
    REQUIRE(stats);
    const auto e1 = fit_dwls(alt, *stats, OrdinalParameterization::Theta);
    const auto e0 = fit_dwls(nul, *stats, OrdinalParameterization::Theta);
    const auto out = api::policy_nested_dwls(nul.pt, nul.rep, e0, api::policy_fit_state(e0), alt.pt,
        alt.rep, e1, api::policy_fit_state(e1), *stats, OrdinalParameterization::Theta);
    REQUIRE_MESSAGE(out.lr.reason == api::InferenceReason::Available, out.lr.detail);
    REQUIRE(out.lr.df == 8);
    statistics(rep) = out.lr.statistic;
    traces(rep) = out.lr.eigenvalues.sum();
    variances(rep) = 2.0 * out.lr.eigenvalues.squaredNorm();
  }
  const double mean = statistics.mean();
  const Eigen::ArrayXd centered = statistics.array() - mean;
  const double variance = centered.square().sum() / 99.0;
  const double mean_se = std::sqrt(variances.mean() / 100.0);
  const double variance_se = std::sqrt((centered.pow(4).mean() - variance * variance) / 100.0);
  MESSAGE("MC100 seed=68261004 n/group=1000: mean=" << mean << " trace=" << traces.mean()
      << " variance=" << variance << " 2sum(lambda^2)=" << variances.mean()
      << " mean_SE=" << mean_se << " variance_SE=" << variance_se);
  // Three Monte Carlo SEs, specified before running: a smoke check, not a
  // finite-sample calibration decision or a fresh-seed size confirmation.
  CHECK(std::abs(mean - traces.mean()) <= 3.0 * mean_se);
  CHECK(std::abs(variance - variances.mean()) <= 3.0 * variance_se);
}

TEST_CASE("DWLS exact policy: binary and five-category nested laws in both parameterizations") {
  for (int categories : {2, 5}) {
    for (int groups : {1, 2}) {
      for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
        CAPTURE(categories); CAPTURE(groups); CAPTURE(static_cast<int>(parameterization));
        std::vector<Eigen::MatrixXd> blocks;
        for (int g = 0; g < groups; ++g) {
          std::mt19937 rng(6969u + static_cast<unsigned>(g)); std::normal_distribution<double> z;
          Eigen::MatrixXd X(600,4);
          for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const double f = z(rng);
            for (int j = 0; j < 4; ++j) {
              const double y = .7*f + std::sqrt(.51)*z(rng);
              int category = 1;
              for (int k = 1; k < categories; ++k) category += y > -1.5 + 3.0*k/categories;
              X(i,j) = category;
            }
          }
          blocks.push_back(X);
        }
        auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks,true); REQUIRE(stats);
        const auto alt = ordinal_model(category_syntax("f =~ x1 + x2 + x3 + x4", categories), groups);
        const auto nul = ordinal_model(category_syntax("f =~ x1 + a*x2 + a*x3 + a*x4", categories), groups);
        NestedDwls r{{}, {}, {}, nul, alt};
        r.null_est = fit_dwls(nul,*stats,parameterization);
        r.alt_est = fit_dwls(alt,*stats,parameterization);
        r.out = api::policy_nested_dwls(nul.pt,nul.rep,r.null_est,{},alt.pt,alt.rep,
            r.alt_est,{},*stats,parameterization);
        REQUIRE_MESSAGE(r.out.lr.reason == api::InferenceReason::Available, r.out.lr.detail);
        check_common_law(r,*stats,parameterization);
      }
    }
  }
}

TEST_CASE("DWLS exact first stage: binary pairwise saturation agrees with OPG at large N") {
  // Each binary pair has three free cell probabilities and three first-stage
  // coordinates. Information equality holds empirically at its saturated MLE;
  // remaining error is solver/finite-difference error, rather than a 1/sqrt(N)
  // discrepancy. Unsaturated multi-category normal pairs converge at that rate.
  std::mt19937 rng(6974u); std::normal_distribution<double> z;
  Eigen::MatrixXd X(4000,4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double f = z(rng);
    for (int j = 0; j < 4; ++j) X(i,j) = 1.0 + (.7*f + std::sqrt(.51)*z(rng) > .2);
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X},true); REQUIRE(stats);
  auto sampling = magmaan::data::ordinal_moment_sampling_influence(stats->int_data[0],
      stats->n_levels[0],stats->thresholds[0],stats->R[0]); REQUIRE(sampling);
  CHECK(relative(sampling->gamma,stats->NACOV[0]) < 1e-6);
}

TEST_CASE("DWLS policy fit measures: Exact comparator and unchanged OPG default") {
  const auto x = misspecified_block(101u, 400, -0.4, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({x}, true);
  REQUIRE(stats);
  const auto m = ordinal_model(kOneFactor, 1);
  const auto est = fit_dwls(m, *stats, OrdinalParameterization::Delta);
  const auto policy = api::policy_fit_measures(m.pt,m.rep,*stats,est,{});
  using magmaan::estimate::OrdinalFirstStage;
  auto rm = magmaan::estimate::ordinal_rmsea_misspec_inference(m.pt,m.rep,*stats,est,
      OrdinalParameterization::Delta,true,0.9,1e-10,OrdinalFirstStage::Exact);
  auto ct = magmaan::estimate::ordinal_cfi_tli_misspec_inference(m.pt,m.rep,*stats,est,
      OrdinalParameterization::Delta,true,0.9,1e-10,OrdinalFirstStage::Exact);
  auto cr = magmaan::estimate::ordinal_crmr_misspec_inference(m.pt,m.rep,*stats,est,
      OrdinalParameterization::Delta,true,false,0.9,1e-10,OrdinalFirstStage::Exact);
  REQUIRE(rm); REQUIRE(ct); REQUIRE(cr);
  CHECK(policy.user.trace == rm->bias_trace);
  CHECK(policy.user.trace == ct->gendf_user);
  CHECK(policy.baseline.trace == ct->gendf_baseline);
  for (const auto& i : policy.indices) {
    REQUIRE_MESSAGE(i.reason == api::InferenceReason::Available,i.detail);
    if (i.index == "rmsea") CHECK(i.estimate == rm->point);
    if (i.index == "cfi") CHECK(i.estimate == ct->cfi);
    if (i.index == "crmr") CHECK(i.estimate == cr->point_bias_corrected);
  }
  auto defaults = magmaan::estimate::ordinal_fit_measures_misspec_inference(m.pt,m.rep,*stats,est);
  auto opg = magmaan::estimate::ordinal_fit_measures_misspec_inference(m.pt,m.rep,*stats,est,
      OrdinalParameterization::Delta,true,0.9,1e-10,OrdinalFirstStage::OPG);
  REQUIRE(defaults); REQUIRE(opg);
  CHECK(defaults->rmsea == opg->rmsea);
  CHECK(defaults->cfi == opg->cfi);
  CHECK(defaults->tli == opg->tli);
  CHECK(defaults->crmr == opg->crmr);
}
