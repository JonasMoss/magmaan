#include <doctest/doctest.h>

#include <array>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"

#include "magmaan/estimate/fit.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::data::SampleStats;
using magmaan::gls::ULS;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::lavaanify;

namespace {

ModelEvaluator must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  static thread_local magmaan::spec::LatentStructure s_pt;
  static thread_local magmaan::model::MatrixRep          s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  auto ev = ModelEvaluator::build(s_pt, s_mr);
  REQUIRE(ev.has_value());
  return std::move(*ev);
}

Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() +
         Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

}  // namespace

TEST_CASE("ULS: F=0 when Σ(θ)=S and μ(θ)=m̄ (saturated)") {
  // Hand-construct the saturated case: Σ = S, μ = m̄ by feeding the
  // discrepancy ImpliedMoments that already match the sample. F must be
  // exactly 0 (single-block: w_b = 1). Multi-block: same — Σ_b w_b · 0 = 0.
  std::mt19937 rng(1);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd m(3);  m << 1.5, 2.5, 3.5;

  SampleStats samp;
  samp.S = {S};
  samp.mean = {m};
  samp.n_obs = {200};

  magmaan::model::ImpliedMoments im;
  im.sigma = {S};
  im.mu    = {m};

  ULS uls;
  auto f = uls.value(samp, im);
  REQUIRE(f.has_value());
  CHECK(*f == doctest::Approx(0.0).epsilon(1e-15));
}

TEST_CASE("ULS: F matches hand formula (cov-only, 3-var)") {
  // Pick arbitrary S and Σ; assert F = ½·||vech(S - Σ)||².
  std::mt19937 rng(2);
  Eigen::MatrixXd S     = random_pd(rng, 3);
  Eigen::MatrixXd Sigma = random_pd(rng, 3);

  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {137};

  magmaan::model::ImpliedMoments im;
  im.sigma = {Sigma};

  // Manual: vech (column-major lower, each off-diag once) → squared norm/2.
  double manual = 0.0;
  for (Eigen::Index j = 0; j < 3; ++j) {
    for (Eigen::Index i = j; i < 3; ++i) {
      const double d = S(i, j) - Sigma(i, j);
      manual += d * d;
    }
  }
  manual *= 0.5;

  ULS uls;
  auto f = uls.value(samp, im);
  REQUIRE(f.has_value());
  CHECK(*f == doctest::Approx(manual).epsilon(1e-14));
}

TEST_CASE("ULS: gradient matches finite differences (1F CFA)") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  std::mt19937 rng(7);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 3));
  samp.n_obs.push_back(100);

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.5, 1.2);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  ULS uls;
  auto sm   = ev.sigma(theta).value();
  auto J    = ev.dsigma_dtheta(theta).value();
  auto g_an = uls.gradient(samp, sm, J).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    g_fd(k) = (uls.value(samp, ev.sigma(tp).value()).value() -
               uls.value(samp, ev.sigma(tm).value()).value()) / (2.0 * h);
  }
  const double diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-7);  // ULS gradient is linear-ish in (S−Σ); very FD-clean.
}

TEST_CASE("ULS: gradient matches finite differences (1F + meanstructure)") {
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1\nf ~ 1");
  std::mt19937 rng(13);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 4.0, 5.0, 6.0;
  SampleStats samp;
  samp.S    = {S};
  samp.mean = {mean};
  samp.n_obs = {150};

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.4, 1.1);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  ULS uls;
  auto sm   = ev.sigma(theta).value();
  auto J    = ev.dsigma_dtheta(theta).value();
  auto Jmu  = ev.dmu_dtheta(theta).value();
  REQUIRE(Jmu.rows() == 3);
  REQUIRE(Jmu.cols() == theta.size());

  auto g_an = uls.gradient(samp, sm, J, Jmu).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    g_fd(k) = (uls.value(samp, ev.sigma(tp).value()).value() -
               uls.value(samp, ev.sigma(tm).value()).value()) / (2.0 * h);
  }
  const double diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-7);
}

TEST_CASE("ULS: gradient matches finite differences (3F at lavaan θ̂, HS)") {
  // Reuse the 0002 fixture's θ̂ — Σ is guaranteed PD, but ULS doesn't need
  // PD-Σ; we just want a "reasonable" θ where the FD step is well-conditioned.
  auto ev = must_build(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");

  const std::string path = std::string(MAGMAAN_FIXTURES_DIR) +
                           "/fit/0002_three_factor_hs.fit.json";
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::stringstream ss;  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  Eigen::VectorXd theta(j["theta_hat"].size());
  for (Eigen::Index k = 0; k < theta.size(); ++k)
    theta(k) = j["theta_hat"][static_cast<std::size_t>(k)].get<double>();
  REQUIRE(static_cast<std::size_t>(theta.size()) == ev.n_free());

  std::mt19937 rng(11);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 9));
  samp.n_obs.push_back(301);

  ULS uls;
  auto sm = ev.sigma(theta).value();
  auto J  = ev.dsigma_dtheta(theta).value();
  auto g_an = uls.gradient(samp, sm, J).value();

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    g_fd(k) = (uls.value(samp, ev.sigma(tp).value()).value() -
               uls.value(samp, ev.sigma(tm).value()).value()) / (2.0 * h);
  }
  const double max_diff = (g_an - g_fd).cwiseAbs().maxCoeff();
  CHECK(max_diff < 1e-5);  // 27-vech-element residual is bigger; FD noise grows.
}

TEST_CASE("ULS: multi-group F is the (n_b/N)-weighted sum of per-block F") {
  // Same Σ ≠ S in two blocks, with different n_obs. Verify
  //   F_total = (n1/N)·F_b1 + (n2/N)·F_b2.
  std::mt19937 rng(99);
  Eigen::MatrixXd S1 = random_pd(rng, 3);
  Eigen::MatrixXd S2 = random_pd(rng, 3);
  Eigen::MatrixXd Sigma1 = random_pd(rng, 3);
  Eigen::MatrixXd Sigma2 = random_pd(rng, 3);

  SampleStats samp;
  samp.S     = {S1, S2};
  samp.n_obs = {120, 80};

  magmaan::model::ImpliedMoments im;
  im.sigma = {Sigma1, Sigma2};

  ULS uls;
  auto f = uls.value(samp, im);
  REQUIRE(f.has_value());

  auto F_block = [](const Eigen::MatrixXd& S, const Eigen::MatrixXd& Si) {
    double a = 0.0;
    for (Eigen::Index j = 0; j < S.rows(); ++j)
      for (Eigen::Index i = j; i < S.rows(); ++i) {
        const double d = S(i, j) - Si(i, j);
        a += d * d;
      }
    return 0.5 * a;
  };
  const double F1 = F_block(S1, Sigma1);
  const double F2 = F_block(S2, Sigma2);
  const double expected = (120.0 / 200.0) * F1 + (80.0 / 200.0) * F2;
  CHECK(*f == doctest::Approx(expected).epsilon(1e-14));
}

TEST_CASE("ULS: fit<ULS>() recovers ground-truth on 1F + meanstructure (in-manifold)") {
  // 1F CFA with indicator intercepts (no factor mean): mean side has
  // 3 free ν and 3 sample-mean moments → saturated; cov side as the
  // sister test → saturated when S is in the 1F manifold. Build both
  // from ground-truth params.
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  Eigen::Vector3d lambda_true(1.0, 0.85, 0.70);
  const double    psi_true = 2.0;
  Eigen::Vector3d theta_true(0.6, 0.5, 0.8);
  Eigen::Matrix3d S =
      lambda_true * lambda_true.transpose() * psi_true +
      theta_true.asDiagonal().toDenseMatrix();
  Eigen::Vector3d nu_true(4.2, 5.7, 6.1);

  SampleStats samp;
  samp.S    = {S};
  samp.mean = {nu_true};  // α = 0 ⇒ μ = ν, so sample mean ≡ ν_true.
  samp.n_obs = {301};

  magmaan::optim::LbfgsOptimizer opt(magmaan::optim::LbfgsOptions{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9});
  auto est_or = magmaan::estimate::fit(*pt, *mr, samp, ULS{}, opt);
  if (!est_or.has_value()) {
    MESSAGE("fit<ULS>() failed: kind=" << static_cast<int>(est_or.error().kind)
            << " detail=" << est_or.error().detail
            << " iter=" << est_or.error().iterations
            << " f_value=" << est_or.error().f_value);
  }
  REQUIRE(est_or.has_value());
  CHECK(est_or->fmin < 1e-8);

  // ν̂_i ≡ m̄_i at the optimum (saturated mean side).
  auto ev = ModelEvaluator::build(*pt, *mr).value();
  const auto locs = ev.param_locations();
  std::array<Eigen::Index, 3> nu_idx = {-1, -1, -1};
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == magmaan::model::MatId::Nu) {
      nu_idx[static_cast<std::size_t>(locs[k].row)] =
          static_cast<Eigen::Index>(k);
    }
  }
  for (auto idx : nu_idx) REQUIRE(idx >= 0);
  for (Eigen::Index i = 0; i < 3; ++i) {
    CHECK(est_or->theta(nu_idx[static_cast<std::size_t>(i)]) ==
          doctest::Approx(nu_true(i)).epsilon(1e-4));
  }
}

TEST_CASE("ULS: residuals/value identity (½||r||² = F) on 1F + meanstructure") {
  // Sanity check on the LS-shape representation: ½||r||² must equal the
  // scalar F to floating-point noise. Single block; means populated. Same
  // fixture as the `gradient FD` cov+mean test.
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1\nf ~ 1");
  std::mt19937 rng(21);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 2.0, 3.0, 4.0;
  SampleStats samp;
  samp.S     = {S};
  samp.mean  = {mean};
  samp.n_obs = {150};

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.4, 1.1);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  ULS uls;
  auto sm  = ev.sigma(theta).value();
  auto F   = uls.value(samp, sm).value();
  auto r   = uls.residuals(samp, sm).value();
  CHECK(0.5 * r.squaredNorm() == doctest::Approx(F).epsilon(1e-12));
}

TEST_CASE("ULS: residual_jacobian Jᵀr identity matches gradient (1F + means)") {
  // Jᵀr should match the existing analytic gradient byte-for-byte (up to
  // float roundoff). This pins the sign convention r = model − sample and
  // the row-stripe scaling √(n_b/N).
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1\nf ~ 1");
  std::mt19937 rng(23);
  Eigen::MatrixXd S = random_pd(rng, 3);
  Eigen::VectorXd mean(3);  mean << 1.0, 1.5, 2.0;
  SampleStats samp;
  samp.S     = {S};
  samp.mean  = {mean};
  samp.n_obs = {120};

  Eigen::VectorXd theta(ev.n_free());
  std::uniform_real_distribution<double> d(0.4, 1.2);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  ULS uls;
  auto sm  = ev.sigma(theta).value();
  auto J   = ev.dsigma_dtheta(theta).value();
  auto Jmu = ev.dmu_dtheta(theta).value();
  auto g_old = uls.gradient(samp, sm, J, Jmu).value();
  auto r     = uls.residuals(samp, sm).value();
  auto Jr    = uls.residual_jacobian(samp, sm, J, Jmu).value();
  const Eigen::VectorXd g_ls = Jr.transpose() * r;
  const double diff = (g_old - g_ls).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-12);
}

TEST_CASE("ULS: residual_jacobian matches FD of residuals (3F at lavaan θ̂)") {
  // FD against `residuals` directly — the LS-path optimizer never calls
  // `value`/`gradient`, so this is the real correctness check.
  auto ev = must_build(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");

  const std::string path = std::string(MAGMAAN_FIXTURES_DIR) +
                           "/fit/0002_three_factor_hs.fit.json";
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::stringstream ss;  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());

  Eigen::VectorXd theta(j["theta_hat"].size());
  for (Eigen::Index k = 0; k < theta.size(); ++k)
    theta(k) = j["theta_hat"][static_cast<std::size_t>(k)].get<double>();

  std::mt19937 rng(17);
  SampleStats samp;
  samp.S.push_back(random_pd(rng, 9));
  samp.n_obs.push_back(301);

  ULS uls;
  auto sm = ev.sigma(theta).value();
  auto J  = ev.dsigma_dtheta(theta).value();
  auto Jr_an = uls.residual_jacobian(samp, sm, J).value();

  const double h = 1e-6;
  Eigen::MatrixXd Jr_fd(Jr_an.rows(), Jr_an.cols());
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    auto rp = uls.residuals(samp, ev.sigma(tp).value()).value();
    auto rm = uls.residuals(samp, ev.sigma(tm).value()).value();
    Jr_fd.col(k) = (rp - rm) / (2.0 * h);
  }
  const double diff = (Jr_an - Jr_fd).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-5);
}

TEST_CASE("ULS: residuals multi-group √(n_b/N) per-block weighting") {
  // Verify the per-block √(n_b/N) scaling: stacked ½||r||² over two blocks
  // must equal the weighted-sum F from `value`.
  std::mt19937 rng(29);
  Eigen::MatrixXd S1 = random_pd(rng, 3);
  Eigen::MatrixXd S2 = random_pd(rng, 3);
  Eigen::MatrixXd Sigma1 = random_pd(rng, 3);
  Eigen::MatrixXd Sigma2 = random_pd(rng, 3);

  SampleStats samp;
  samp.S     = {S1, S2};
  samp.n_obs = {120, 80};

  magmaan::model::ImpliedMoments im;
  im.sigma = {Sigma1, Sigma2};

  ULS uls;
  auto F = uls.value(samp, im).value();
  auto r = uls.residuals(samp, im).value();
  CHECK(0.5 * r.squaredNorm() == doctest::Approx(F).epsilon(1e-12));
}

TEST_CASE("ULS: fit<ULS>() recovers ground-truth Σ on a 1F-feasible covariance") {
  // 1F CFA — 6 free params, 6 vech moments. The model manifold
  // Σ(θ) = λλᵀψ + diag(θ_resid) is dim-saturated but a strict subset of
  // PSD(3): rank(Σ − diag) = 1 and off-diagonals share the sign of
  // λ_iλ_jψ. Generate S INSIDE this manifold so the optimum is F = 0.
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  // Ground-truth θ inside the 1F manifold (positive λ, ψ, θ_resid).
  Eigen::Vector3d lambda_true(1.0, 0.85, 0.70);  // λ_1 = 1 marker
  const double    psi_true = 2.0;
  Eigen::Vector3d theta_true(0.6, 0.5, 0.8);
  Eigen::Matrix3d S =
      lambda_true * lambda_true.transpose() * psi_true +
      theta_true.asDiagonal().toDenseMatrix();

  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {301};

  // ULS is shallower than ML near the optimum (no log-det curvature) —
  // LBFGS needs more iterations than the default 1000.
  magmaan::optim::LbfgsOptimizer opt(magmaan::optim::LbfgsOptions{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9});
  auto est_or = magmaan::estimate::fit(*pt, *mr, samp, ULS{}, opt);
  if (!est_or.has_value()) {
    MESSAGE("fit<ULS>() failed: kind=" << static_cast<int>(est_or.error().kind)
            << " detail=" << est_or.error().detail
            << " iter=" << est_or.error().iterations
            << " f_value=" << est_or.error().f_value);
  }
  REQUIRE(est_or.has_value());
  CHECK(est_or->fmin < 1e-8);

  // Σ̂(θ̂) should match S elementwise to high precision.
  auto ev = ModelEvaluator::build(*pt, *mr).value();
  auto sm = ev.sigma(est_or->theta).value();
  const double max_resid = (S - sm.sigma[0]).cwiseAbs().maxCoeff();
  CHECK(max_resid < 1e-4);
}
