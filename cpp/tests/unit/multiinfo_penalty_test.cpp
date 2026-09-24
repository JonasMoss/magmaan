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
