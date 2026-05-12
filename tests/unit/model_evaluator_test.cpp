#include <doctest/doctest.h>

#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "latva/model/matrix_rep.hpp"
#include "latva/model/model_evaluator.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

using latva::model::build_matrix_rep;
using latva::model::ImpliedMoments;
using latva::model::ModelEvaluator;
using latva::parse::Parser;
using latva::partable::lavaanify;

namespace {

ModelEvaluator must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  // LatentStructure and MatrixRep need to outlive the evaluator. Stash them in
  // statics with std::move so the references in the evaluator stay valid.
  // (Tests are single-threaded; this is safe.)
  static thread_local latva::partable::LatentStructure s_pt;
  static thread_local latva::model::MatrixRep   s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  auto ev = ModelEvaluator::build(s_pt, s_mr);
  REQUIRE_MESSAGE(ev.has_value(),
                  "ModelEvaluator::build failed: " << ev.error().detail);
  return std::move(*ev);
}

// Finite-difference Jacobian of vech(Σ) wrt θ. Used to cross-check the
// closed-form jacobian.
Eigen::MatrixXd finite_diff_jacobian(const ModelEvaluator& ev,
                                     const Eigen::VectorXd& theta,
                                     double h = 1e-6) {
  const auto p_blocks = ev.n_blocks();
  // Total vech length.
  auto sigma0 = ev.sigma(theta).value();
  Eigen::Index total_vech = 0;
  for (std::size_t b = 0; b < p_blocks; ++b) {
    const Eigen::Index p = sigma0.sigma[b].rows();
    total_vech += p * (p + 1) / 2;
  }
  Eigen::MatrixXd J(total_vech, theta.size());
  J.setZero();

  auto vech_pack = [&](const ImpliedMoments& m) {
    Eigen::VectorXd v(total_vech);
    Eigen::Index off = 0;
    for (std::size_t b = 0; b < m.sigma.size(); ++b) {
      const auto& S = m.sigma[b];
      const Eigen::Index p = S.rows();
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          v(off++) = S(r, c);
        }
      }
    }
    return v;
  };

  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta;  tp(k) += h;
    Eigen::VectorXd tm = theta;  tm(k) -= h;
    auto Sp = ev.sigma(tp).value();
    auto Sm = ev.sigma(tm).value();
    J.col(k) = (vech_pack(Sp) - vech_pack(Sm)) / (2.0 * h);
  }
  return J;
}

}  // namespace

TEST_CASE("ModelEvaluator: 1F CFA — Σ has the right shape and equals Λ Ψ Λᵀ + Θ by hand") {
  // f =~ x1 + x2 + x3
  // After lavaanify: 7 rows, 6 free params (2 free loadings + 3 residual vars + 1 factor var).
  // Free assignment order (row-by-row, skipping the auto-fixed marker):
  //   free=1: λ_2,0  (f =~ x2)
  //   free=2: λ_3,0  (f =~ x3)  — 0-based row 2 in Λ
  //   free=3: θ_0,0  (x1 ~~ x1)
  //   free=4: θ_1,1
  //   free=5: θ_2,2
  //   free=6: ψ_0,0  (f ~~ f)
  // Wait: row indexing in Λ is 0-based: row 0 → x1 (auto-fixed at λ=1), row 1 → x2, row 2 → x3.
  auto ev = must_build("f =~ x1 + x2 + x3");
  REQUIRE(ev.n_free() == 6);

  Eigen::VectorXd theta(6);
  theta << 0.8, 1.2, 0.5, 0.6, 0.7, 1.0;
  // Λ = [1, 0.8, 1.2]ᵀ (column 0 only).
  // Ψ = [1.0]
  // Θ = diag(0.5, 0.6, 0.7)
  // Σ = Λ Ψ Λᵀ + Θ
  //   = [1*1*1+0.5,  1*1*0.8,    1*1*1.2;
  //      0.8*1*1,    0.8*1*0.8+0.6, 0.8*1*1.2;
  //      1.2*1*1,    1.2*1*0.8,  1.2*1*1.2+0.7]

  auto sm = ev.sigma(theta);
  REQUIRE(sm.has_value());
  REQUIRE(sm->sigma.size() == 1);
  const auto& S = sm->sigma[0];
  REQUIRE(S.rows() == 3);
  REQUIRE(S.cols() == 3);

  CHECK(S(0,0) == doctest::Approx(1.0 * 1.0 * 1.0 + 0.5));
  CHECK(S(1,0) == doctest::Approx(0.8));
  CHECK(S(2,0) == doctest::Approx(1.2));
  CHECK(S(1,1) == doctest::Approx(0.8 * 0.8 + 0.6));
  CHECK(S(2,1) == doctest::Approx(0.8 * 1.2));
  CHECK(S(2,2) == doctest::Approx(1.2 * 1.2 + 0.7));
  // Symmetry
  CHECK(S(0,1) == S(1,0));
  CHECK(S(0,2) == S(2,0));
  CHECK(S(1,2) == S(2,1));
}

TEST_CASE("ModelEvaluator: analytic Jacobian matches finite differences (1F CFA)") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> d(0.3, 1.5);

  for (int trial = 0; trial < 5; ++trial) {
    Eigen::VectorXd theta(ev.n_free());
    for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

    auto J_an = ev.dsigma_dtheta(theta).value();
    auto J_fd = finite_diff_jacobian(ev, theta);
    REQUIRE(J_an.rows() == J_fd.rows());
    REQUIRE(J_an.cols() == J_fd.cols());

    const double diff = (J_an - J_fd).cwiseAbs().maxCoeff();
    CHECK(diff < 1e-6);
  }
}

TEST_CASE("ModelEvaluator: analytic Jacobian matches finite differences (3F Holzinger)") {
  auto ev = must_build(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  REQUIRE(ev.n_free() > 9);  // 6 free loadings + 9 residual vars + 3 lv vars + 3 lv covs = 21

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d(0.3, 1.5);
  Eigen::VectorXd theta(ev.n_free());
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);

  auto J_an = ev.dsigma_dtheta(theta).value();
  auto J_fd = finite_diff_jacobian(ev, theta);
  const double diff = (J_an - J_fd).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-6);
}

TEST_CASE("ModelEvaluator: residual covariance off-diagonal — Σ and Jacobian") {
  // x1 ~~ x2 model. After lavaanify: 3 rows
  //   x1~~x2 (free=1), x1~~x1 (free=2), x2~~x2 (free=3)
  // Σ is just Θ (no latents).
  auto ev = must_build("x1 ~~ x2");
  REQUIRE(ev.n_free() == 3);

  Eigen::VectorXd theta(3);
  theta << 0.5, 1.5, 2.0;  // cov(x1,x2)=0.5, var(x1)=1.5, var(x2)=2.0

  auto sm = ev.sigma(theta).value();
  const auto& S = sm.sigma[0];
  CHECK(S(0,0) == doctest::Approx(1.5));
  CHECK(S(1,1) == doctest::Approx(2.0));
  CHECK(S(0,1) == doctest::Approx(0.5));
  CHECK(S(1,0) == doctest::Approx(0.5));

  auto J_an = ev.dsigma_dtheta(theta).value();
  auto J_fd = finite_diff_jacobian(ev, theta);
  CHECK((J_an - J_fd).cwiseAbs().maxCoeff() < 1e-7);
}

TEST_CASE("ModelEvaluator: mean structure — μ = ν (+ Λ·α) at θ") {
  // CFA with explicit indicator intercepts (no latent mean → α = 0 → μ = ν).
  auto ev = must_build("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1");

  const auto locs = ev.param_locations();

  // Find each ν_i's index by location: Nu, row=i.
  std::array<std::ptrdiff_t, 3> nu_idx = {-1, -1, -1};
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == latva::model::MatId::Nu) {
      REQUIRE(locs[k].row >= 0);
      REQUIRE(locs[k].row < 3);
      nu_idx[static_cast<std::size_t>(locs[k].row)] = static_cast<std::ptrdiff_t>(k);
    }
  }
  for (auto idx : nu_idx) REQUIRE(idx >= 0);

  // Construct θ with arbitrary covariance entries and recognizable mean values.
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(ev.n_free()));
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    if (locs[static_cast<std::size_t>(k)].mat == latva::model::MatId::Lambda)
      theta(k) = 0.8;
    else if (locs[static_cast<std::size_t>(k)].mat == latva::model::MatId::Theta)
      theta(k) = 0.5;
    else if (locs[static_cast<std::size_t>(k)].mat == latva::model::MatId::Psi)
      theta(k) = 1.0;
  }
  theta(nu_idx[0]) = 3.0;
  theta(nu_idx[1]) = 4.0;
  theta(nu_idx[2]) = 5.0;

  auto sm = ev.sigma(theta);
  REQUIRE(sm.has_value());
  REQUIRE(sm->mu.size() == 1);
  REQUIRE(sm->mu[0].size() == 3);
  CHECK(sm->mu[0](0) == doctest::Approx(3.0));
  CHECK(sm->mu[0](1) == doctest::Approx(4.0));
  CHECK(sm->mu[0](2) == doctest::Approx(5.0));

  // assembled() exposes Nu/Alpha for downstream consumers.
  auto am = ev.assembled(theta);
  REQUIRE(am.has_value());
  REQUIRE(am->blocks.size() == 1);
  CHECK(am->blocks[0].Nu.size() == 3);
  CHECK(am->blocks[0].Nu(0) == doctest::Approx(3.0));
  // Alpha is sized (m=1) but never written → stays zero.
  CHECK(am->blocks[0].Alpha.size() == 1);
  CHECK(am->blocks[0].Alpha(0) == doctest::Approx(0.0));
}

TEST_CASE("ModelEvaluator: mean structure — μ depends on Λ·α when latent mean is free") {
  // CFA with latent mean: μ = ν + Λ·α. With ν=0 and α free, μ = Λ·α.
  // Without intercepts in the partable, has_means would only fire if there's
  // *any* mean-structure row in the block — here `f ~ 1` is the only one.
  auto ev = must_build("f =~ x1 + x2 + x3\nf ~ 1");

  const auto locs = ev.param_locations();
  std::ptrdiff_t alpha_idx = -1;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == latva::model::MatId::Alpha) {
      alpha_idx = static_cast<std::ptrdiff_t>(k);
    }
  }
  REQUIRE(alpha_idx >= 0);

  Eigen::VectorXd theta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(ev.n_free()));
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    if (locs[static_cast<std::size_t>(k)].mat == latva::model::MatId::Lambda)
      theta(k) = 0.7;
    else if (locs[static_cast<std::size_t>(k)].mat == latva::model::MatId::Theta)
      theta(k) = 0.5;
    else if (locs[static_cast<std::size_t>(k)].mat == latva::model::MatId::Psi)
      theta(k) = 1.0;
  }
  theta(alpha_idx) = 2.0;

  auto sm = ev.sigma(theta);
  REQUIRE(sm.has_value());
  REQUIRE(sm->mu.size() == 1);
  // Λ = [1, 0.7, 0.7]ᵀ (first loading is the auto-fixed marker at 1).
  // μ = Λ · α = [1·2, 0.7·2, 0.7·2] = [2.0, 1.4, 1.4].
  CHECK(sm->mu[0](0) == doctest::Approx(2.0));
  CHECK(sm->mu[0](1) == doctest::Approx(1.4));
  CHECK(sm->mu[0](2) == doctest::Approx(1.4));
}

TEST_CASE("ModelEvaluator: no mean structure → ImpliedMoments.mu is empty") {
  // The default — no `~1` rows — keeps μ empty so downstream consumers
  // (ML, inference) can detect "covariance-only" via `mu.empty()`.
  auto ev = must_build("f =~ x1 + x2 + x3");
  Eigen::VectorXd theta = Eigen::VectorXd::Constant(
      static_cast<Eigen::Index>(ev.n_free()), 0.7);
  auto sm = ev.sigma(theta);
  REQUIRE(sm.has_value());
  REQUIRE(sm->mu.size() == 1);
  CHECK(sm->mu[0].size() == 0);
}

TEST_CASE("ModelEvaluator: rejects mismatched theta size") {
  auto ev = must_build("f =~ x1 + x2 + x3");
  Eigen::VectorXd wrong(2);
  wrong << 1, 2;
  auto sm = ev.sigma(wrong);
  CHECK_FALSE(sm.has_value());
}
