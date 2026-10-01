#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/coordinates.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#define REQUIRE_OK(value)                                                     \
  do {                                                                        \
    INFO("error: " << ((value).has_value() ? "" : (value).error().detail));   \
    REQUIRE((value).has_value());                                             \
    if (!(value).has_value()) return;                                         \
  } while (false)

namespace {

using magmaan::data::SampleStats;
using magmaan::estimate::frontier::fit_ml_multiinfo;
using magmaan::estimate::frontier::multiinfo_penalty;
using magmaan::estimate::frontier::multiinfo_penalty_layout;
using magmaan::estimate::frontier::multiinfo_penalty_report;
using magmaan::estimate::frontier::MultiInfoPenaltyLayout;
using magmaan::estimate::frontier::MultiInfoPenaltyOptions;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatId;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentStructure;

LatentStructure lavaanify(std::string_view syntax, BuildOptions options = {}) {
  auto flat = Parser::parse(syntax);
  REQUIRE(flat.has_value());
  auto pt = magmaan::spec::build(*flat, options);
  REQUIRE_MESSAGE(pt.has_value(), "lavaanify failed: "
      << (pt.has_value() ? std::string{} : pt.error().detail));
  return std::move(*pt);
}

BuildOptions no_fixed_x() {
  BuildOptions out;
  out.fixed_x = false;
  return out;
}

// Deterministic, interior parameter values by matrix type; `jitter` varies
// them across parameters so FD checks do not sit on symmetric points.
Eigen::VectorXd interior_theta(const ModelEvaluator& ev, double jitter = 0.07) {
  const auto locs = ev.param_locations();
  Eigen::VectorXd theta(static_cast<Eigen::Index>(locs.size()));
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    const double wobble = jitter * static_cast<double>(k % 5);
    const auto& l = locs[static_cast<std::size_t>(k)];
    switch (l.mat) {
      case MatId::Lambda: theta(k) = 0.7 + wobble; break;
      case MatId::Beta:   theta(k) = 0.35 + 0.5 * wobble; break;
      case MatId::Psi:    theta(k) = l.row == l.col ? 1.0 + wobble : 0.25; break;
      case MatId::Theta:  theta(k) = l.row == l.col ? 0.5 + wobble : 0.1; break;
      case MatId::Nu:
      case MatId::Alpha:  theta(k) = 0.1 * wobble; break;
    }
  }
  return theta;
}

Eigen::Index find_param(const ModelEvaluator& ev, MatId mat, int row, int col) {
  const auto locs = ev.param_locations();
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& l = locs[k];
    if (l.mat != mat) continue;
    if ((l.row == row && l.col == col) || (l.row == col && l.col == row)) {
      return static_cast<Eigen::Index>(k);
    }
  }
  return -1;
}

// Independent route: LISREL blocks, not the RAM E matrix, then an explicit
// correlation matrix and an LU log-determinant.
double brute_force_penalty(const ModelEvaluator& ev,
                           const MultiInfoPenaltyLayout& layout,
                           const Eigen::VectorXd& theta) {
  auto am = ev.assembled(theta);
  REQUIRE(am.has_value());
  double total = 0.0;
  for (std::size_t b = 0; b < layout.blocks.size(); ++b) {
    const auto& blk = layout.blocks[b];
    const auto& bm = am->blocks[b];
    const Eigen::Index m = blk.m, p = blk.p;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(m + p, m + p);
    if (m > 0) {
      const Eigen::MatrixXd A =
          (Eigen::MatrixXd::Identity(m, m) - bm.Beta).inverse();
      const Eigen::MatrixXd Phi = A * bm.Psi * A.transpose();
      C.topLeftCorner(m, m) = Phi;
      C.bottomLeftCorner(p, m) = bm.Lambda * A * bm.Psi * A.transpose();
      C.topRightCorner(m, p) = C.bottomLeftCorner(p, m).transpose();
      C.bottomRightCorner(p, p) =
          bm.Lambda * Phi * bm.Lambda.transpose() + bm.Theta;
    } else {
      C = bm.Theta;
    }
    const Eigen::Index k = static_cast<Eigen::Index>(blk.keep.size());
    Eigen::MatrixXd R(k, k);
    for (Eigen::Index i = 0; i < k; ++i) {
      for (Eigen::Index j = 0; j < k; ++j) {
        const auto a = blk.keep[static_cast<std::size_t>(i)];
        const auto c = blk.keep[static_cast<std::size_t>(j)];
        R(i, j) = C(a, c) / std::sqrt(C(a, a) * C(c, c));
      }
    }
    total += std::log(R.partialPivLu().determinant());
  }
  return total;
}

SampleStats stats_from(Eigen::MatrixXd S, std::int64_t n) {
  SampleStats out;
  out.S.push_back(std::move(S));
  out.n_obs.push_back(n);
  return out;
}

// Population Σ of a model at interior_theta.
Eigen::MatrixXd population_sigma(const LatentStructure& pt,
                                 const MatrixRep& rep) {
  auto ev = ModelEvaluator::build(pt, rep);
  REQUIRE(ev.has_value());
  auto sm = ev->sigma(interior_theta(*ev));
  REQUIRE(sm.has_value());
  return sm->sigma[0];
}

Eigen::MatrixXd implied_sigma(const LatentStructure& pt, const MatrixRep& rep,
                              const Eigen::VectorXd& theta) {
  auto ev = ModelEvaluator::build(pt, rep);
  REQUIRE(ev.has_value());
  auto sm = ev->sigma(theta);
  REQUIRE(sm.has_value());
  return sm->sigma[0];
}

constexpr std::string_view kTwoFactorCorr =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ x4 + x5 + x6\n"
    "x1 ~~ x4\n";

constexpr std::string_view kRecursiveSem =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ y1 + y2 + y3\n"
    "f3 =~ y4 + y5 + y6\n"
    "f2 ~ f1\n"
    "f3 ~ f1 + f2\n"
    "y1 ~~ y4\n"
    "y2 ~~ y5\n";

constexpr std::string_view kObservedPath =
    "y1 ~ x1 + x2\n"
    "y2 ~ y1 + x1\n";

constexpr std::string_view kNonrecursive =
    "y1 ~ y2 + x1\n"
    "y2 ~ y1 + x2\n";

}  // namespace

TEST_CASE("multiinfo penalty: value matches brute-force complete-data correlation") {
  for (const auto syntax : {kTwoFactorCorr, kRecursiveSem, kObservedPath,
                            kNonrecursive}) {
    CAPTURE(syntax);
    const auto pt = lavaanify(syntax, no_fixed_x());
    auto rep = build_matrix_rep(pt);
    REQUIRE_OK(rep);
    auto ev = ModelEvaluator::build(pt, *rep);
    REQUIRE_OK(ev);
    const Eigen::VectorXd theta = interior_theta(*ev);
    auto layout = multiinfo_penalty_layout(*ev, theta);
    REQUIRE_OK(layout);
    auto pen = multiinfo_penalty(*layout, *ev, theta, false);
    REQUIRE_OK(pen);
    CHECK(pen->value <= 0.0);
    CHECK(pen->value == doctest::Approx(brute_force_penalty(*ev, *layout, theta))
                            .epsilon(1e-12));
  }
}

TEST_CASE("multiinfo penalty: one-factor closed form sum log(1 - h^2)") {
  const auto pt = lavaanify("f =~ x1 + x2 + x3 + x4");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta);
  REQUIRE_OK(layout);
  auto pen = multiinfo_penalty(*layout, *ev, theta, false);
  REQUIRE_OK(pen);
  auto am = ev->assembled(theta);
  REQUIRE_OK(am);
  const auto& bm = am->blocks[0];
  const double phi = bm.Psi(0, 0);
  double expected = 0.0;
  for (Eigen::Index j = 0; j < 4; ++j) {
    const double common = bm.Lambda(j, 0) * bm.Lambda(j, 0) * phi;
    const double h2 = common / (common + bm.Theta(j, j));
    expected += std::log(1.0 - h2);
  }
  CHECK(pen->value == doctest::Approx(expected).epsilon(1e-12));
}

TEST_CASE("multiinfo penalty: recursive identity and report decomposition") {
  const auto pt = lavaanify(kRecursiveSem, no_fixed_x());
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta);
  REQUIRE_OK(layout);
  CHECK(layout->recursive());
  auto report = multiinfo_penalty_report(*layout, *ev, theta);
  REQUIRE_OK(report);
  REQUIRE(report->block_value.size() == 1);
  double terms = 0.0;
  for (const auto& t : report->terms) terms += t.log_one_minus_r2;
  // P = Σ log(1 − R²_i) + log det Corr(S_KK) = log det S_KK − Σ log C_ii.
  CHECK(report->block_value[0] ==
        doctest::Approx(terms + report->residual_log_det_corr[0]).epsilon(1e-12));
  CHECK(report->residual_log_det_corr[0] < 0.0);  // correlated residuals
}

TEST_CASE("multiinfo penalty: analytic gradient matches central differences") {
  for (const auto syntax : {kTwoFactorCorr, kRecursiveSem, kObservedPath,
                            kNonrecursive}) {
    CAPTURE(syntax);
    BuildOptions options = no_fixed_x();
    options.meanstructure = true;  // ν/α entries must carry zero gradient
    const auto pt = lavaanify(syntax, options);
    auto rep = build_matrix_rep(pt);
    REQUIRE_OK(rep);
    auto ev = ModelEvaluator::build(pt, *rep);
    REQUIRE_OK(ev);
    const Eigen::VectorXd theta = interior_theta(*ev);
    auto layout = multiinfo_penalty_layout(*ev, theta);
    REQUIRE_OK(layout);
    auto pen = multiinfo_penalty(*layout, *ev, theta, true);
    REQUIRE_OK(pen);
    REQUIRE(pen->gradient.size() == theta.size());
    const double h = 1e-6;
    for (Eigen::Index k = 0; k < theta.size(); ++k) {
      Eigen::VectorXd up = theta, down = theta;
      up(k) += h;
      down(k) -= h;
      auto fu = multiinfo_penalty(*layout, *ev, up, false);
      auto fd = multiinfo_penalty(*layout, *ev, down, false);
      REQUIRE_OK(fu);
      REQUIRE_OK(fd);
      const double numeric = (fu->value - fd->value) / (2.0 * h);
      CAPTURE(k);
      CHECK(pen->gradient(k) == doctest::Approx(numeric).epsilon(1e-6).scale(1.0));
    }
  }
}

TEST_CASE("multiinfo penalty: zero-residual indicators and phantoms leave K") {
  // auto_fix_single fixes the lone indicator's residual at zero: y1 ≡ f2.
  const auto pt = lavaanify(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ y1\n"
      "f2 ~ f1\n");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta);
  REQUIRE_OK(layout);
  const auto& blk = layout->blocks[0];
  // 2 latents + x1..x3; y1 (observed, Θ ≡ 0) is excluded.
  CHECK(blk.keep.size() == static_cast<std::size_t>(blk.m + 3));
  auto pen = multiinfo_penalty(*layout, *ev, theta, true);
  REQUIRE_OK(pen);
  CHECK(std::isfinite(pen->value));
  CHECK(pen->value <= 0.0);
}

TEST_CASE("multiinfo penalty: fixed.x covariates stay in K and barrier regressions on them") {
  // Excluding x would leave Var(y) = b'Σ_x b + ψ_y > 0 as ψ_y → 0, so the
  // equation y ~ x would carry no barrier at all.
  auto pt = lavaanify("y ~ x1 + x2\n");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  Eigen::Matrix3d S;
  S << 1.0, 0.3, 0.2,
       0.3, 1.0, 0.1,
       0.2, 0.1, 1.0;
  const SampleStats samp = stats_from(S, 200);
  auto resolved = magmaan::estimate::resolve_fixed_x_from_sample(pt, *rep, samp);
  REQUIRE_OK(resolved);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta);
  REQUIRE_OK(layout);
  // All three latent slots (y phantom, x1, x2 phantoms) are in K; the
  // observed copies (Θ ≡ 0) are not.
  CHECK(layout->blocks[0].keep.size() == 3);
  const Eigen::Index psi_y = find_param(*ev, MatId::Psi, 0, 0);
  REQUIRE(psi_y >= 0);
  theta(psi_y) = 1e-10;
  auto pen = multiinfo_penalty(*layout, *ev, theta, false);
  REQUIRE_OK(pen);
  CHECK(pen->value < -15.0);
}

TEST_CASE("multiinfo penalty: nonrecursive model is flagged, finite, nonpositive") {
  const auto pt = lavaanify(kNonrecursive, no_fixed_x());
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta);
  REQUIRE_OK(layout);
  CHECK_FALSE(layout->recursive());
  auto pen = multiinfo_penalty(*layout, *ev, theta, false);
  REQUIRE_OK(pen);
  CHECK(std::isfinite(pen->value));
  CHECK(pen->value <= 0.0);
}

TEST_CASE("penalized ML: marker and std.lv identification give the same Sigma") {
  const auto marker = lavaanify(kTwoFactorCorr);
  BuildOptions std_options;
  std_options.std_lv = true;
  const auto std_lv = lavaanify(kTwoFactorCorr, std_options);
  auto rep_m = build_matrix_rep(marker);
  auto rep_s = build_matrix_rep(std_lv);
  REQUIRE_OK(rep_m);
  REQUIRE_OK(rep_s);
  Eigen::MatrixXd S = population_sigma(marker, *rep_m);
  S(1, 5) += 0.08;  // perturb so the model does not fit exactly
  S(5, 1) += 0.08;
  S(2, 3) -= 0.05;
  S(3, 2) -= 0.05;
  const SampleStats samp = stats_from(S, 60);
  auto x_m = magmaan::estimate::simple_start_values(marker, *rep_m, samp);
  auto x_s = magmaan::estimate::simple_start_values(std_lv, *rep_s, samp);
  REQUIRE_OK(x_m);
  REQUIRE_OK(x_s);
  auto fit_m = fit_ml_multiinfo(marker, *rep_m, samp, *x_m);
  auto fit_s = fit_ml_multiinfo(std_lv, *rep_s, samp, *x_s);
  REQUIRE_OK(fit_m);
  REQUIRE_OK(fit_s);
  const Eigen::MatrixXd sig_m = implied_sigma(marker, *rep_m, fit_m->estimates.theta);
  const Eigen::MatrixXd sig_s = implied_sigma(std_lv, *rep_s, fit_s->estimates.theta);
  CHECK((sig_m - sig_s).cwiseAbs().maxCoeff() < 1e-6);
  CHECK(fit_m->penalty.value == doctest::Approx(fit_s->penalty.value).epsilon(1e-6));
  CHECK(fit_m->estimates.fmin == doctest::Approx(fit_s->estimates.fmin).epsilon(1e-6));
}

TEST_CASE("penalized ML: equivariant under rescaling observed columns") {
  const auto pt = lavaanify(kTwoFactorCorr);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  Eigen::MatrixXd S = population_sigma(pt, *rep);
  S(0, 5) += 0.1;
  S(5, 0) += 0.1;
  Eigen::VectorXd d(6);
  d << 2.0, 0.5, 3.0, 1.5, 0.25, 4.0;
  const Eigen::MatrixXd S_scaled = d.asDiagonal() * S * d.asDiagonal();
  const SampleStats a = stats_from(S, 80);
  const SampleStats b = stats_from(S_scaled, 80);
  auto xa = magmaan::estimate::simple_start_values(pt, *rep, a);
  auto xb = magmaan::estimate::simple_start_values(pt, *rep, b);
  REQUIRE_OK(xa);
  REQUIRE_OK(xb);
  auto fa = fit_ml_multiinfo(pt, *rep, a, *xa);
  auto fb = fit_ml_multiinfo(pt, *rep, b, *xb);
  REQUIRE_OK(fa);
  REQUIRE_OK(fb);
  const Eigen::MatrixXd sa = implied_sigma(pt, *rep, fa->estimates.theta);
  const Eigen::MatrixXd sb = implied_sigma(pt, *rep, fb->estimates.theta);
  const Eigen::MatrixXd back = d.cwiseInverse().asDiagonal() * sb *
                               d.cwiseInverse().asDiagonal();
  CHECK((sa - back).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("penalized ML: interior perturbation is O(1/N)") {
  const auto pt = lavaanify(kRecursiveSem);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  const Eigen::MatrixXd S = population_sigma(pt, *rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd truth = interior_theta(*ev);
  std::vector<double> scaled, chi2;
  for (const std::int64_t n : {800, 3200, 12800}) {
    const SampleStats samp = stats_from(S, n);
    auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
    REQUIRE_OK(x0);
    auto fit = fit_ml_multiinfo(pt, *rep, samp, *x0);
    REQUIRE_OK(fit);
    const double diff = (fit->estimates.theta - truth).norm();
    scaled.push_back(static_cast<double>(n) * diff);
    chi2.push_back(2.0 * static_cast<double>(n) * fit->estimates.fmin);
  }
  CHECK(scaled[0] > 0.1);
  CHECK(scaled[1] == doctest::Approx(scaled[0]).epsilon(0.10));
  CHECK(scaled[2] == doctest::Approx(scaled[1]).epsilon(0.03));
  // The unpenalized χ² at θ̃ (exact-fit S) is O(1/N): it quarters per step.
  CHECK(chi2[1] < 0.3 * chi2[0]);
  CHECK(chi2[2] < 0.3 * chi2[1]);
}

TEST_CASE("penalized ML: weight zero reproduces ordinary ML") {
  const auto pt = lavaanify(kTwoFactorCorr);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  Eigen::MatrixXd S = population_sigma(pt, *rep);
  S(1, 4) += 0.07;
  S(4, 1) += 0.07;
  const SampleStats samp = stats_from(S, 120);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
  REQUIRE_OK(x0);
  MultiInfoPenaltyOptions zero;
  zero.weight = 0.0;
  auto pen = fit_ml_multiinfo(pt, *rep, samp, *x0, zero);
  auto ml = magmaan::estimate::fit_ml(pt, *rep, samp, *x0);
  REQUIRE_OK(pen);
  REQUIRE_OK(ml);
  CHECK((pen->estimates.theta - ml->theta).cwiseAbs().maxCoeff() < 1e-6);
  CHECK(pen->estimates.fmin == doctest::Approx(ml->fmin).epsilon(1e-9));
  CHECK(pen->penalized_fmin == doctest::Approx(pen->estimates.fmin).epsilon(1e-12));
}

TEST_CASE("penalized ML: one-factor Heywood case is pulled inside") {
  // Just-identified: ML λ₁² = s12·s13/s23 = 1.08 > 1, so θ₁ = −0.08.
  const auto pt = lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  Eigen::Matrix3d S;
  S << 1.0, 0.9, 0.6,
       0.9, 1.0, 0.5,
       0.6, 0.5, 1.0;
  const SampleStats samp = stats_from(S, 50);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
  REQUIRE_OK(x0);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::Index t1 = find_param(*ev, MatId::Theta, 0, 0);
  REQUIRE(t1 >= 0);
  auto ml = magmaan::estimate::fit_ml(pt, *rep, samp, *x0);
  REQUIRE_OK(ml);
  CHECK(ml->theta(t1) < -0.05);
  for (const double eta : {2.0, 3.0}) {
    MultiInfoPenaltyOptions options;
    options.eta = eta;
    auto pen = fit_ml_multiinfo(pt, *rep, samp, *x0, options);
    REQUIRE_OK(pen);
    CHECK(pen->estimates.optimizer_status == magmaan::optim::OptimStatus::Converged);
    for (const Eigen::Index k : {find_param(*ev, MatId::Theta, 0, 0),
                                 find_param(*ev, MatId::Theta, 1, 1),
                                 find_param(*ev, MatId::Theta, 2, 2)}) {
      CHECK(pen->estimates.theta(k) > 0.0);
    }
    CHECK(std::isfinite(pen->penalty.value));
    CHECK(pen->estimates.fmin > ml->fmin);
  }
}

TEST_CASE("penalized ML: factor correlation above one is pulled inside") {
  const auto pt = lavaanify(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  // Loadings .7, factor correlation 1.06: inadmissible population.
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(6, 6);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (i == j) continue;
      const bool same = (i < 3) == (j < 3);
      S(i, j) = 0.49 * (same ? 1.0 : 1.06);
    }
  }
  const SampleStats samp = stats_from(S, 80);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
  REQUIRE_OK(x0);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  auto correlation = [&](const Eigen::VectorXd& theta) {
    auto am = ev->assembled(theta);
    REQUIRE(am.has_value());
    const auto& P = am->blocks[0].Psi;
    return P(0, 1) / std::sqrt(P(0, 0) * P(1, 1));
  };
  auto ml = magmaan::estimate::fit_ml(pt, *rep, samp, *x0);
  REQUIRE_OK(ml);
  CHECK(correlation(ml->theta) > 1.0);
  auto pen = fit_ml_multiinfo(pt, *rep, samp, *x0);
  REQUIRE_OK(pen);
  const double r = correlation(pen->estimates.theta);
  CHECK(r < 1.0);
  CHECK(r > 0.8);
}

TEST_CASE("penalized FIML matches penalized ML on complete data") {
  BuildOptions options;
  options.meanstructure = true;
  const auto pt = lavaanify(kTwoFactorCorr, options);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  const Eigen::MatrixXd Sigma = population_sigma(pt, *rep);
  const Eigen::MatrixXd L = Sigma.llt().matrixL();
  std::mt19937 rng(20260922);
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd X(90, 6);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    Eigen::VectorXd z(6);
    for (Eigen::Index j = 0; j < 6; ++j) z(j) = normal(rng);
    X.row(i) = (L * z).transpose();
  }
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE_OK(samp);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, *samp);
  REQUIRE_OK(x0);
  auto ml = fit_ml_multiinfo(pt, *rep, *samp, *x0);
  REQUIRE_OK(ml);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE_OK(pack);
  auto fiml = magmaan::estimate::fiml::frontier::fit_fiml_multiinfo(
      pt, *rep, raw, *x0, *pack);
  REQUIRE_OK(fiml);
  CHECK((ml->estimates.theta - fiml->estimates.theta).cwiseAbs().maxCoeff() < 1e-4);
  CHECK(ml->penalty.value == doctest::Approx(fiml->penalty.value).epsilon(1e-5));
}

// ---- Determinacy target (latent-determinacy barrier) -----------------------

namespace {

using magmaan::estimate::frontier::PenaltyTarget;

constexpr std::string_view kMixedSem =
    "f =~ x1 + x2 + x3\n"
    "f ~ z\n"
    "y ~ f\n";

MultiInfoPenaltyOptions determinacy_options() {
  MultiInfoPenaltyOptions out;
  out.target = PenaltyTarget::Determinacy;
  return out;
}

struct LatentGivenObserved {
  double log_det_q = 0.0;
  double three_term = 0.0;   // log det Corr(Φ_L) + log det Var(y | η_L) − log det Σ
  double joint_minus_q = 0.0;  // log det Corr(Φ_L, y) − log det Q
  double log_det_corr_sigma = 0.0;
};

// Independent route: LISREL blocks, the first `genuine` latents, explicit
// conditional covariance, LU determinants.
LatentGivenObserved brute_force_determinacy(const ModelEvaluator& ev,
                                            const Eigen::VectorXd& theta,
                                            Eigen::Index genuine) {
  auto am = ev.assembled(theta);
  REQUIRE(am.has_value());
  const auto& bm = am->blocks[0];
  const Eigen::Index m = bm.Psi.rows();
  const auto logdet = [](const Eigen::MatrixXd& X) {
    return std::log(X.partialPivLu().determinant());
  };
  const auto corr = [](const Eigen::MatrixXd& X) {
    const Eigen::VectorXd s = X.diagonal().cwiseSqrt().cwiseInverse();
    return Eigen::MatrixXd(s.asDiagonal() * X * s.asDiagonal());
  };
  const Eigen::MatrixXd A = (Eigen::MatrixXd::Identity(m, m) - bm.Beta).inverse();
  const Eigen::MatrixXd Phi = A * bm.Psi * A.transpose();
  const Eigen::MatrixXd Sigma = bm.Lambda * Phi * bm.Lambda.transpose() + bm.Theta;
  LatentGivenObserved out;
  out.log_det_corr_sigma = logdet(corr(Sigma));
  if (genuine == 0) return out;
  const Eigen::MatrixXd PhiL = Phi.topLeftCorner(genuine, genuine);
  const Eigen::MatrixXd C = (Phi * bm.Lambda.transpose()).topRows(genuine);
  const Eigen::MatrixXd V = PhiL - C * Sigma.inverse() * C.transpose();
  const Eigen::VectorXd s = PhiL.diagonal().cwiseSqrt().cwiseInverse();
  const Eigen::MatrixXd Q = s.asDiagonal() * V * s.asDiagonal();
  out.log_det_q = logdet(Q);
  out.three_term = logdet(corr(PhiL)) +
                   logdet(Sigma - C.transpose() * PhiL.inverse() * C) -
                   logdet(Sigma);
  Eigen::MatrixXd joint(genuine + Sigma.rows(), genuine + Sigma.rows());
  joint << PhiL, C, C.transpose(), Sigma;
  out.joint_minus_q = logdet(corr(joint)) - out.log_det_q;
  return out;
}

struct DeterminacyCase {
  std::string_view syntax;
  Eigen::Index genuine;
};

}  // namespace

TEST_CASE("determinacy penalty: value and identities match brute force") {
  const DeterminacyCase cases[] = {{kTwoFactorCorr, 2}, {kRecursiveSem, 3},
                                   {kMixedSem, 1}, {kObservedPath, 0},
                                   {kNonrecursive, 0}};
  for (const auto& c : cases) {
    CAPTURE(c.syntax);
    const auto pt = lavaanify(c.syntax, no_fixed_x());
    auto rep = build_matrix_rep(pt);
    REQUIRE_OK(rep);
    auto ev = ModelEvaluator::build(pt, *rep);
    REQUIRE_OK(ev);
    const Eigen::VectorXd theta = interior_theta(*ev);
    auto layout = multiinfo_penalty_layout(*ev, theta, PenaltyTarget::Determinacy);
    REQUIRE_OK(layout);
    REQUIRE(layout->blocks[0].keep.size() == static_cast<std::size_t>(c.genuine));
    for (Eigen::Index j = 0; j < c.genuine; ++j) {
      CHECK(layout->blocks[0].keep[static_cast<std::size_t>(j)] == j);
    }
    auto pen = multiinfo_penalty(*layout, *ev, theta, false);
    REQUIRE_OK(pen);
    const auto bf = brute_force_determinacy(*ev, theta, c.genuine);
    CHECK(pen->value <= 0.0);
    CHECK(pen->value == doctest::Approx(bf.log_det_q).epsilon(1e-12).scale(1.0));
    CHECK(bf.three_term == doctest::Approx(bf.log_det_q).epsilon(1e-10).scale(1.0));
    if (c.genuine > 0) {
      // log det Corr(η_L, y) = log det Q + log det Corr(Σ).
      CHECK(bf.joint_minus_q ==
            doctest::Approx(bf.log_det_corr_sigma).epsilon(1e-10));
    }
    auto report = multiinfo_penalty_report(*layout, *ev, theta);
    REQUIRE_OK(report);
    CHECK(report->target == PenaltyTarget::Determinacy);
    double terms = 0.0;
    for (const auto& t : report->terms) {
      CHECK(t.latent);
      CHECK(t.log_one_minus_r2 <= 0.0);
      terms += t.log_one_minus_r2;
    }
    CHECK(report->block_value[0] ==
          doctest::Approx(terms + report->residual_log_det_corr[0]).epsilon(1e-12).scale(1.0));
    CHECK(report->block_value[0] ==
          doctest::Approx(-2.0 * (report->total_correlation[0] +
                                  report->mutual_information[0]))
              .epsilon(1e-12).scale(1.0));
  }
}

TEST_CASE("determinacy penalty: joint barrier equals determinacy plus log det Corr(Sigma)") {
  // In a CFA, K is every latent and indicator, so the two targets differ by
  // the observed-margin term alone.
  const auto pt = lavaanify(kTwoFactorCorr, no_fixed_x());
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto joint_layout = multiinfo_penalty_layout(*ev, theta);
  auto det_layout = multiinfo_penalty_layout(*ev, theta, PenaltyTarget::Determinacy);
  REQUIRE_OK(joint_layout);
  REQUIRE_OK(det_layout);
  auto joint = multiinfo_penalty(*joint_layout, *ev, theta, false);
  auto det = multiinfo_penalty(*det_layout, *ev, theta, false);
  REQUIRE_OK(joint);
  REQUIRE_OK(det);
  const auto bf = brute_force_determinacy(*ev, theta, 2);
  CHECK(joint->value - det->value ==
        doctest::Approx(bf.log_det_corr_sigma).epsilon(1e-10));
}

TEST_CASE("determinacy penalty: one-factor closed form -log(1 + phi a' Theta^-1 a)") {
  const auto pt = lavaanify("f =~ x1 + x2 + x3 + x4");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta, PenaltyTarget::Determinacy);
  REQUIRE_OK(layout);
  auto pen = multiinfo_penalty(*layout, *ev, theta, false);
  REQUIRE_OK(pen);
  auto am = ev->assembled(theta);
  REQUIRE_OK(am);
  const auto& bm = am->blocks[0];
  double tau = 0.0;
  for (Eigen::Index j = 0; j < 4; ++j) {
    tau += bm.Lambda(j, 0) * bm.Lambda(j, 0) / bm.Theta(j, j);
  }
  CHECK(pen->value ==
        doctest::Approx(-std::log1p(bm.Psi(0, 0) * tau)).epsilon(1e-12));
}

TEST_CASE("determinacy penalty: analytic gradient matches central differences") {
  for (const auto syntax : {kTwoFactorCorr, kRecursiveSem, kMixedSem}) {
    CAPTURE(syntax);
    BuildOptions options = no_fixed_x();
    options.meanstructure = true;
    const auto pt = lavaanify(syntax, options);
    auto rep = build_matrix_rep(pt);
    REQUIRE_OK(rep);
    auto ev = ModelEvaluator::build(pt, *rep);
    REQUIRE_OK(ev);
    const Eigen::VectorXd theta = interior_theta(*ev);
    auto layout = multiinfo_penalty_layout(*ev, theta, PenaltyTarget::Determinacy);
    REQUIRE_OK(layout);
    auto pen = multiinfo_penalty(*layout, *ev, theta, true);
    REQUIRE_OK(pen);
    REQUIRE(pen->gradient.size() == theta.size());
    const double h = 1e-6;
    for (Eigen::Index k = 0; k < theta.size(); ++k) {
      Eigen::VectorXd up = theta, down = theta;
      up(k) += h;
      down(k) -= h;
      auto fu = multiinfo_penalty(*layout, *ev, up, false);
      auto fd = multiinfo_penalty(*layout, *ev, down, false);
      REQUIRE_OK(fu);
      REQUIRE_OK(fd);
      CAPTURE(k);
      CHECK(pen->gradient(k) ==
            doctest::Approx((fu->value - fd->value) / (2.0 * h)).epsilon(1e-6).scale(1.0));
    }
  }
}

TEST_CASE("determinacy penalty: exactly measured latents are observed data") {
  // auto_fix_single fixes y1's residual at zero, so f2 ≡ y1 is not latent.
  const auto pt = lavaanify(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ y1\n"
      "f2 ~ f1\n");
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  const Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta, PenaltyTarget::Determinacy);
  REQUIRE_OK(layout);
  REQUIRE(layout->blocks[0].keep.size() == 1);
  CHECK(layout->blocks[0].keep[0] == 0);
  auto pen = multiinfo_penalty(*layout, *ev, theta, true);
  REQUIRE_OK(pen);
  CHECK(std::isfinite(pen->value));
  CHECK(pen->value < 0.0);

  // An error-free indicator of two latents makes Var(η | y) singular always.
  const auto shared = lavaanify(
      "f1 =~ x1 + x2 + x3 + x7\n"
      "f2 =~ x4 + x5 + x6 + x7\n"
      "x7 ~~ 0*x7\n");
  auto rep2 = build_matrix_rep(shared);
  REQUIRE_OK(rep2);
  auto ev2 = ModelEvaluator::build(shared, *rep2);
  REQUIRE_OK(ev2);
  auto bad = multiinfo_penalty_layout(*ev2, interior_theta(*ev2),
                                      PenaltyTarget::Determinacy);
  CHECK_FALSE(bad.has_value());
}

TEST_CASE("determinacy penalty: improper points with Sigma PD are outside the domain") {
  const auto pt = lavaanify(kMixedSem, no_fixed_x());
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  auto ev = ModelEvaluator::build(pt, *rep);
  REQUIRE_OK(ev);
  Eigen::VectorXd theta = interior_theta(*ev);
  auto layout = multiinfo_penalty_layout(*ev, theta, PenaltyTarget::Determinacy);
  REQUIRE_OK(layout);
  REQUIRE_OK(multiinfo_penalty(*layout, *ev, theta, false));
  // Negative residual variance of the observed outcome y (a phantom Ψ cell)
  // and a negative unique variance, each with Σ still positive definite.
  auto am = ev->assembled(theta);
  REQUIRE_OK(am);
  const Eigen::Index y_slot = 1;  // lv order [f, y, z]
  const Eigen::Index psi_y = find_param(*ev, MatId::Psi, y_slot, y_slot);
  const Eigen::Index theta_1 = find_param(*ev, MatId::Theta, 0, 0);
  REQUIRE(psi_y >= 0);
  REQUIRE(theta_1 >= 0);
  for (const Eigen::Index k : {psi_y, theta_1}) {
    CAPTURE(k);
    Eigen::VectorXd bad = theta;
    bad(k) = -0.02;
    auto sm = ev->sigma(bad);
    REQUIRE_OK(sm);
    Eigen::LLT<Eigen::MatrixXd> llt(sm->sigma[0]);
    REQUIRE(llt.info() == Eigen::Success);  // Σ still PD
    CHECK_FALSE(multiinfo_penalty(*layout, *ev, bad, false).has_value());
    Eigen::VectorXd near = theta;
    near(k) = 1e-9;
    auto pen = multiinfo_penalty(*layout, *ev, near, false);
    REQUIRE_OK(pen);
    CHECK(pen->value < -10.0);
  }
}

TEST_CASE("determinacy-penalized ML: manifest model returns ordinary ML exactly") {
  auto pt = lavaanify(kObservedPath);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  Eigen::Matrix4d S;
  S << 1.0, 0.4, 0.3, 0.2,
       0.4, 1.0, 0.3, 0.25,
       0.3, 0.3, 1.0, 0.1,
       0.2, 0.25, 0.1, 1.0;
  const SampleStats samp = stats_from(S, 40);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
  REQUIRE_OK(x0);
  auto pen = fit_ml_multiinfo(pt, *rep, samp, *x0, determinacy_options());
  auto ml = magmaan::estimate::fit_ml(pt, *rep, samp, *x0);
  REQUIRE_OK(pen);
  REQUIRE_OK(ml);
  CHECK(pen->penalty.value == 0.0);
  CHECK(pen->penalty.terms.empty());
  CHECK((pen->estimates.theta - ml->theta).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("determinacy-penalized ML: marker and std.lv give the same Sigma") {
  const auto marker = lavaanify(kTwoFactorCorr);
  BuildOptions std_options;
  std_options.std_lv = true;
  const auto std_lv = lavaanify(kTwoFactorCorr, std_options);
  auto rep_m = build_matrix_rep(marker);
  auto rep_s = build_matrix_rep(std_lv);
  REQUIRE_OK(rep_m);
  REQUIRE_OK(rep_s);
  Eigen::MatrixXd S = population_sigma(marker, *rep_m);
  S(1, 5) += 0.08;
  S(5, 1) += 0.08;
  S(2, 3) -= 0.05;
  S(3, 2) -= 0.05;
  const SampleStats samp = stats_from(S, 60);
  auto x_m = magmaan::estimate::simple_start_values(marker, *rep_m, samp);
  auto x_s = magmaan::estimate::simple_start_values(std_lv, *rep_s, samp);
  REQUIRE_OK(x_m);
  REQUIRE_OK(x_s);
  auto fit_m = fit_ml_multiinfo(marker, *rep_m, samp, *x_m, determinacy_options());
  auto fit_s = fit_ml_multiinfo(std_lv, *rep_s, samp, *x_s, determinacy_options());
  REQUIRE_OK(fit_m);
  REQUIRE_OK(fit_s);
  const Eigen::MatrixXd sig_m = implied_sigma(marker, *rep_m, fit_m->estimates.theta);
  const Eigen::MatrixXd sig_s = implied_sigma(std_lv, *rep_s, fit_s->estimates.theta);
  CHECK((sig_m - sig_s).cwiseAbs().maxCoeff() < 1e-6);
  CHECK(fit_m->penalty.value == doctest::Approx(fit_s->penalty.value).epsilon(1e-6));
}

TEST_CASE("determinacy-penalized ML: equivariant under rescaling observed columns") {
  const auto pt = lavaanify(kTwoFactorCorr);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  Eigen::MatrixXd S = population_sigma(pt, *rep);
  S(0, 5) += 0.1;
  S(5, 0) += 0.1;
  Eigen::VectorXd d(6);
  d << 2.0, 0.5, 3.0, 1.5, 0.25, 4.0;
  const Eigen::MatrixXd S_scaled = d.asDiagonal() * S * d.asDiagonal();
  const SampleStats a = stats_from(S, 80);
  const SampleStats b = stats_from(S_scaled, 80);
  auto xa = magmaan::estimate::simple_start_values(pt, *rep, a);
  auto xb = magmaan::estimate::simple_start_values(pt, *rep, b);
  REQUIRE_OK(xa);
  REQUIRE_OK(xb);
  auto fa = fit_ml_multiinfo(pt, *rep, a, *xa, determinacy_options());
  auto fb = fit_ml_multiinfo(pt, *rep, b, *xb, determinacy_options());
  REQUIRE_OK(fa);
  REQUIRE_OK(fb);
  const Eigen::MatrixXd sa = implied_sigma(pt, *rep, fa->estimates.theta);
  const Eigen::MatrixXd sb = implied_sigma(pt, *rep, fb->estimates.theta);
  const Eigen::MatrixXd back = d.cwiseInverse().asDiagonal() * sb *
                               d.cwiseInverse().asDiagonal();
  CHECK((sa - back).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("determinacy-penalized ML: Heywood case and factor correlation pulled inside") {
  {
    const auto pt = lavaanify("f =~ x1 + x2 + x3");
    auto rep = build_matrix_rep(pt);
    REQUIRE_OK(rep);
    Eigen::Matrix3d S;
    S << 1.0, 0.9, 0.6,
         0.9, 1.0, 0.5,
         0.6, 0.5, 1.0;
    const SampleStats samp = stats_from(S, 50);
    auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
    REQUIRE_OK(x0);
    auto ev = ModelEvaluator::build(pt, *rep);
    REQUIRE_OK(ev);
    auto pen = fit_ml_multiinfo(pt, *rep, samp, *x0, determinacy_options());
    REQUIRE_OK(pen);
    CHECK(pen->estimates.optimizer_status == magmaan::optim::OptimStatus::Converged);
    for (int i = 0; i < 3; ++i) {
      CHECK(pen->estimates.theta(find_param(*ev, MatId::Theta, i, i)) > 0.0);
    }
  }
  {
    const auto pt = lavaanify(
        "f1 =~ x1 + x2 + x3\n"
        "f2 =~ x4 + x5 + x6\n");
    auto rep = build_matrix_rep(pt);
    REQUIRE_OK(rep);
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(6, 6);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (i == j) continue;
        S(i, j) = 0.49 * (((i < 3) == (j < 3)) ? 1.0 : 1.06);
      }
    }
    const SampleStats samp = stats_from(S, 80);
    auto x0 = magmaan::estimate::simple_start_values(pt, *rep, samp);
    REQUIRE_OK(x0);
    auto ev = ModelEvaluator::build(pt, *rep);
    REQUIRE_OK(ev);
    auto pen = fit_ml_multiinfo(pt, *rep, samp, *x0, determinacy_options());
    REQUIRE_OK(pen);
    auto am = ev->assembled(pen->estimates.theta);
    REQUIRE_OK(am);
    const auto& P = am->blocks[0].Psi;
    const double r = P(0, 1) / std::sqrt(P(0, 0) * P(1, 1));
    CHECK(r < 1.0);
    CHECK(r > 0.8);
  }
}

TEST_CASE("determinacy-penalized FIML matches penalized ML on complete data") {
  BuildOptions options;
  options.meanstructure = true;
  const auto pt = lavaanify(kTwoFactorCorr, options);
  auto rep = build_matrix_rep(pt);
  REQUIRE_OK(rep);
  const Eigen::MatrixXd Sigma = population_sigma(pt, *rep);
  const Eigen::MatrixXd L = Sigma.llt().matrixL();
  std::mt19937 rng(20260924);
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd X(90, 6);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    Eigen::VectorXd z(6);
    for (Eigen::Index j = 0; j < 6; ++j) z(j) = normal(rng);
    X.row(i) = (L * z).transpose();
  }
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE_OK(samp);
  auto x0 = magmaan::estimate::simple_start_values(pt, *rep, *samp);
  REQUIRE_OK(x0);
  auto ml = fit_ml_multiinfo(pt, *rep, *samp, *x0, determinacy_options());
  REQUIRE_OK(ml);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE_OK(pack);
  auto fiml = magmaan::estimate::fiml::frontier::fit_fiml_multiinfo(
      pt, *rep, raw, *x0, *pack, determinacy_options());
  REQUIRE_OK(fiml);
  CHECK((ml->estimates.theta - fiml->estimates.theta).cwiseAbs().maxCoeff() < 1e-4);
  CHECK(ml->penalty.value == doctest::Approx(fiml->penalty.value).epsilon(1e-5));
}


TEST_CASE("shared barriers preserve fixed quadratic criteria and zero-weight reduction") {
  const auto pt = lavaanify(kTwoFactorCorr);
  auto rep = build_matrix_rep(pt); REQUIRE_OK(rep);
  auto S = population_sigma(pt, *rep); S(0, 5) += .08; S(5, 0) += .08;
  const auto sample = stats_from(S, 150);
  auto start = magmaan::estimate::simple_start_values(pt, *rep, sample); REQUIRE_OK(start);
  auto evaluator = ModelEvaluator::build(pt, *rep); REQUIRE_OK(evaluator);
  auto gls = magmaan::estimate::gmm::normal_theory_weight(*evaluator, sample, *start); REQUIRE_OK(gls);
  const Eigen::Index q = S.rows() * (S.rows() + 1) / 2;
  std::vector<magmaan::estimate::gmm::Weight> weights{{}, *gls,
      {magmaan::estimate::gmm::BlockWeight::diagonal(Eigen::VectorXd::LinSpaced(q, .7, 1.3))}};
  for (auto target : {magmaan::estimate::frontier::PenaltyTarget::Joint,
                       magmaan::estimate::frontier::PenaltyTarget::Determinacy}) {
    MultiInfoPenaltyOptions options; options.target = target;
    for (const auto& weight : weights) {
      auto fit = magmaan::estimate::frontier::fit_gmm_multiinfo(pt, *rep, sample, *start, weight, options);
      REQUIRE_OK(fit);
      auto residuals = magmaan::estimate::gmm::residuals(*evaluator, sample, fit->estimates.theta, weight);
      REQUIRE_OK(residuals);
      auto scalar = magmaan::optim::scalarize(*residuals);
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(start->size());
      CHECK(fit->estimates.fmin == doctest::Approx(scalar.f(fit->estimates.theta, gradient)).epsilon(1e-9));
      CHECK(fit->penalized_fmin == doctest::Approx(fit->estimates.fmin - fit->weight / fit->n_total * fit->penalty.value).epsilon(1e-10));
      CHECK(fit->estimates.sample_normalized);
      auto zero = options; zero.weight = 0;
      auto reduced = magmaan::estimate::frontier::fit_gmm_multiinfo(pt, *rep, sample, *start, weight, zero);
      auto ordinary = magmaan::estimate::fit_gmm(pt, *rep, sample, *start, weight);
      REQUIRE_OK(reduced); REQUIRE_OK(ordinary);
      CHECK(reduced->estimates.fmin == doctest::Approx(ordinary->fmin).epsilon(1e-6));
    }
  }
}

TEST_CASE("normalized barriers preserve affine constraints and penalized objectives") {
  const auto pt = lavaanify("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  auto rep = build_matrix_rep(pt); REQUIRE_OK(rep);
  const auto sample = stats_from(population_sigma(pt, *rep), 100);
  auto start = magmaan::estimate::simple_start_values(pt, *rep, sample); REQUIRE_OK(start);
  auto normalized = magmaan::estimate::normalize_ml_model(pt, *rep, sample); REQUIRE_OK(normalized);
  const Eigen::VectorXd normalized_start = start->cwiseQuotient(normalized->parameter_units);
  magmaan::optim::OptimOptions opts = magmaan::estimate::ml_optim_options();
  auto original = fit_ml_multiinfo(pt, *rep, sample, *start, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  opts.normalize_sample = false;
  auto transformed = fit_ml_multiinfo(normalized->structure, normalized->representation,
      normalized->sample, normalized_start, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_OK(original); REQUIRE_OK(transformed);
  CHECK((original->estimates.theta - transformed->estimates.theta.cwiseProduct(normalized->parameter_units)).norm() < 1e-10);
  CHECK(original->penalized_fmin == doctest::Approx(transformed->penalized_fmin).epsilon(1e-12));
  CHECK(original->estimates.diagnostics.lin_eq_residual_inf < 1e-10);
}

TEST_CASE("zero-weight barriers retain ordinary improper solutions") {
  const auto pt = lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt); REQUIRE_OK(rep);
  Eigen::Matrix3d S; S << 1,.9,.6, .9,1,.5, .6,.5,1;
  const auto sample = stats_from(S, 100);
  auto start = magmaan::estimate::simple_start_values(pt, *rep, sample); REQUIRE_OK(start);
  for (auto target : {magmaan::estimate::frontier::PenaltyTarget::Joint,
                      magmaan::estimate::frontier::PenaltyTarget::Determinacy}) {
    MultiInfoPenaltyOptions options; options.target = target; options.weight = 0;
    auto fit = fit_ml_multiinfo(pt, *rep, sample, *start, options);
    REQUIRE_OK(fit);
    CHECK(fit->estimates.fmin < 1e-10);
    CHECK(fit->penalized_fmin == fit->estimates.fmin);
    CHECK(std::isnan(fit->penalty.value));
    auto am = ModelEvaluator::build(pt, *rep); REQUIRE_OK(am);
    auto matrices = am->assembled(fit->estimates.theta); REQUIRE_OK(matrices);
    CHECK(matrices->blocks[0].Theta.diagonal().minCoeff() < 0);
  }
}

TEST_CASE("ordinal barriers ignore threshold cells and preserve ML saturated thresholds") {
  auto pt = lavaanify("f =~ x1 + x2 + x3 + x4\nx1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2");
  magmaan::data::OrdinalStats stats;
  Eigen::Vector4d load(.75, .65, .55, .45);
  Eigen::MatrixXd R = load * load.transpose(); R.diagonal().setOnes();
  R(0, 1) += .02; R(1, 0) += .02;
  stats.R = {R}; stats.n_obs = {400}; stats.n_levels = {{3,3,3,3}};
  Eigen::VectorXd thresholds(8); thresholds << -.5,.5,-.4,.6,-.3,.7,-.2,.8;
  stats.thresholds = {thresholds}; stats.threshold_ov = {{0,0,1,1,2,2,3,3}};
  stats.threshold_level = {{1,2,1,2,1,2,1,2}};
  stats.NACOV = {Eigen::MatrixXd::Identity(14,14)};
  stats.W_dwls = stats.W_wls = stats.NACOV;
  REQUIRE_OK(magmaan::estimate::prepare_ordinal_delta_partable(pt, stats));
  auto rep = build_matrix_rep(pt); REQUIRE_OK(rep);
  auto start = magmaan::estimate::ordinal_start_values(pt, *rep, stats, {}); REQUIRE_OK(start);
  auto evaluator = ModelEvaluator::build(pt, *rep); REQUIRE_OK(evaluator);
  for (auto target : {magmaan::estimate::frontier::PenaltyTarget::Joint,
                       magmaan::estimate::frontier::PenaltyTarget::Determinacy}) {
    auto layout = multiinfo_penalty_layout(*evaluator, *start, target); REQUIRE_OK(layout);
    auto penalty = multiinfo_penalty(*layout, *evaluator, *start, true); REQUIRE_OK(penalty);
    auto hessian = magmaan::estimate::frontier::multiinfo_penalty_hessian(*layout, *evaluator, *start); REQUIRE_OK(hessian);
    for (std::size_t r = 0; r < pt.size(); ++r) if (pt.op[r] == magmaan::parse::Op::Threshold) {
      const auto k = pt.free[r] - 1;
      CHECK(penalty->gradient(k) == 0);
      CHECK(hessian->row(k).isZero(0));
    }
    MultiInfoPenaltyOptions options; options.target = target;
    for (bool ml : {false, true}) {
      auto fit = magmaan::estimate::frontier::fit_ordinal_multiinfo(pt, *rep, stats, *start,
          ml, magmaan::estimate::OrdinalWeightKind::DWLS,
          magmaan::estimate::OrdinalParameterization::Delta, options);
      REQUIRE_OK(fit);
      CHECK(std::isfinite(fit->penalized_fmin));
      if (ml) { REQUIRE(fit->estimates.association); CHECK(fit->estimates.association->df == 2); }
      Eigen::Index k = 0;
      for (std::size_t r = 0; r < pt.size(); ++r) if (pt.op[r] == magmaan::parse::Op::Threshold)
        CHECK(fit->estimates.theta(pt.free[r] - 1) == doctest::Approx(thresholds(k++)).epsilon(1e-10));
    }
    for (auto weight : {magmaan::estimate::OrdinalWeightKind::ULS,
                        magmaan::estimate::OrdinalWeightKind::DWLS}) {
      auto fit_only = stats;
      fit_only.NACOV.clear();
      if (weight == magmaan::estimate::OrdinalWeightKind::ULS) {
        fit_only.W_dwls.clear(); fit_only.W_wls.clear();
      }
      auto full = magmaan::estimate::frontier::fit_ordinal_multiinfo(pt, *rep, stats, *start,
          false, weight, magmaan::estimate::OrdinalParameterization::Delta, options);
      auto lean = magmaan::estimate::frontier::fit_ordinal_multiinfo(pt, *rep, fit_only, *start,
          false, weight, magmaan::estimate::OrdinalParameterization::Delta, options);
      REQUIRE_OK(full); REQUIRE_OK(lean);
      CHECK((full->estimates.theta - lean->estimates.theta).norm() < 1e-10);
      auto psd = magmaan::estimate::frontier::fit_ordinal_psd(pt, *rep, fit_only, {}, weight, *start);
      REQUIRE_OK(psd);
      CHECK(std::isfinite(psd->fmin));
    }
  }
}
