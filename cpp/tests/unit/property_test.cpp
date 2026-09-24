// Structural property tests — invariants that hold for *any* well-formed
// model, independent of a particular lavaan fixture. These guard the seams
// that the parity goldens depend on but never isolate: moment-vector
// ordering, analytic Jacobians on structural (non-CFA) models, multi-group
// block stacking and (n_b/N) weighting, and the equality-constraint affine
// reparameterization algebra.
//
// A break here is a structural mistake; catching it as a sharp local
// property failure beats debugging it as a diffuse lavaan-parity drift.
//
//   ./build/fast/tests/magmaan_test_estimate -tc="Property:*"

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::build_eq_constraints;
using magmaan::estimate::EqConstraints;
using magmaan::model::build_matrix_rep;
using magmaan::model::ImpliedMoments;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentStructure;

namespace {

LatentStructure must_lavaanify(std::string_view src, BuildOptions opts = {}) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp, opts);
  REQUIRE_MESSAGE(pt.has_value(),
                  "lavaanify failed: "
                      << (pt.has_value() ? "" : pt.error().detail));
  return std::move(*pt);
}

// Builds an evaluator and parks LatentStructure + MatrixRep in thread-local
// statics so the references inside the evaluator stay valid. One build per
// test case — a second call in the same test would invalidate the first.
ModelEvaluator must_build(std::string_view src, BuildOptions opts = {}) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp, opts);
  REQUIRE_MESSAGE(pt.has_value(),
                  "lavaanify failed: "
                      << (pt.has_value() ? "" : pt.error().detail));
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  static thread_local LatentStructure s_pt;
  static thread_local magmaan::model::MatrixRep s_mr;
  s_pt = std::move(*pt);
  s_mr = std::move(*mr);
  auto ev = ModelEvaluator::build(s_pt, s_mr);
  REQUIRE_MESSAGE(ev.has_value(),
                  "ModelEvaluator::build failed: "
                      << (ev.has_value() ? "" : ev.error().detail));
  return std::move(*ev);
}

[[maybe_unused]] Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() +
         Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

Eigen::VectorXd random_theta(const ModelEvaluator& ev, std::mt19937& rng) {
  Eigen::VectorXd theta(static_cast<Eigen::Index>(ev.n_free()));
  std::uniform_real_distribution<double> d(0.35, 1.15);
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = d(rng);
  return theta;
}

// Stacks vech(Σ_b) over blocks — column-major lower triangle, matching
// dsigma_dtheta()'s row order.
Eigen::VectorXd vech_stack(const ImpliedMoments& m) {
  Eigen::Index total = 0;
  for (const auto& S : m.sigma) total += S.rows() * (S.rows() + 1) / 2;
  Eigen::VectorXd v(total);
  Eigen::Index off = 0;
  for (const auto& S : m.sigma) {
    const Eigen::Index p = S.rows();
    for (Eigen::Index c = 0; c < p; ++c)
      for (Eigen::Index r = c; r < p; ++r) v(off++) = S(r, c);
  }
  return v;
}

// Stacks μ_b over blocks (each block contributes p_b entries).
Eigen::VectorXd mu_stack(const ImpliedMoments& m) {
  Eigen::Index total = 0;
  for (const auto& mu : m.mu) total += mu.size();
  Eigen::VectorXd v(total);
  Eigen::Index off = 0;
  for (const auto& mu : m.mu)
    for (Eigen::Index k = 0; k < mu.size(); ++k) v(off++) = mu(k);
  return v;
}

// Central-difference Jacobian of a vector-valued θ ↦ f(θ).
template <class Fn>
Eigen::MatrixXd fd_jacobian(Fn&& f, const Eigen::VectorXd& theta,
                            double h = 1e-6) {
  const Eigen::VectorXd f0 = f(theta);
  Eigen::MatrixXd J(f0.size(), theta.size());
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta, tm = theta;
    tp(k) += h;
    tm(k) -= h;
    J.col(k) = (f(tp) - f(tm)) / (2.0 * h);
  }
  return J;
}

}  // namespace

// === moment-vector ordering ================================================

// === analytic Jacobians on structural (non-CFA) models =====================

TEST_CASE("Property: dσ/dθ matches finite differences on a latent regression") {
  // model_evaluator_test FD-checks pure CFA; this exercises a non-zero Β
  // (the (I−B)⁻¹ path) — a structural regression among latents.
  auto ev = must_build(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f2 ~ f1");
  std::mt19937 rng(13);
  for (int trial = 0; trial < 4; ++trial) {
    const Eigen::VectorXd theta = random_theta(ev, rng);
    const Eigen::MatrixXd J_an = ev.dsigma_dtheta(theta).value();
    const Eigen::MatrixXd J_fd = fd_jacobian(
        [&](const Eigen::VectorXd& t) { return vech_stack(ev.sigma(t).value()); },
        theta);
    REQUIRE(J_an.rows() == J_fd.rows());
    REQUIRE(J_an.cols() == J_fd.cols());
    CHECK((J_an - J_fd).cwiseAbs().maxCoeff() < 1e-6);
  }
}

TEST_CASE("Property: dμ/dθ matches finite differences on a mean-structure CFA") {
  // dmu_dtheta() has no FD cross-check elsewhere — the ML/LS mean-term
  // gradient tests only exercise it indirectly through Jᵀr identities.
  auto ev = must_build(
      "f =~ x1 + x2 + x3\n"
      "x1 ~ 1\nx2 ~ 1\nx3 ~ 1\n"
      "f ~ 1");
  std::mt19937 rng(31);
  for (int trial = 0; trial < 4; ++trial) {
    const Eigen::VectorXd theta = random_theta(ev, rng);
    const Eigen::MatrixXd Jmu_an = ev.dmu_dtheta(theta).value();
    const Eigen::MatrixXd Jmu_fd = fd_jacobian(
        [&](const Eigen::VectorXd& t) { return mu_stack(ev.sigma(t).value()); },
        theta);
    REQUIRE(Jmu_an.rows() == Jmu_fd.rows());
    REQUIRE(Jmu_an.cols() == Jmu_fd.cols());
    CHECK((Jmu_an - Jmu_fd).cwiseAbs().maxCoeff() < 1e-6);
  }
}

// === multi-group block stacking ============================================

TEST_CASE("Property: multi-group dσ/dθ is block-stacked and FD-correct") {
  // Two independent groups → dvech(Σ)/dθ stacks block 0's vech rows above
  // block 1's. Verify the analytic Jacobian against finite differences over
  // the joint θ.
  BuildOptions opts;
  opts.n_groups = 2;
  auto ev = must_build("f =~ x1 + x2 + x3", opts);
  REQUIRE(ev.n_blocks() == 2);

  std::mt19937 rng(202);
  const Eigen::VectorXd theta = random_theta(ev, rng);
  const Eigen::MatrixXd J_an = ev.dsigma_dtheta(theta).value();
  // 2 blocks × vech(3×3) = 12 rows.
  CHECK(J_an.rows() == 12);
  CHECK(J_an.cols() == static_cast<Eigen::Index>(ev.n_free()));

  const Eigen::MatrixXd J_fd = fd_jacobian(
      [&](const Eigen::VectorXd& t) { return vech_stack(ev.sigma(t).value()); },
      theta);
  CHECK((J_an - J_fd).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("Property: unconstrained multi-group Jacobian is block-diagonal") {
  // No cross-group labels ⇒ every free parameter belongs to exactly one
  // group. Block b's vech rows must then be insensitive to any parameter
  // located in another block: the analytic Jacobian is block-structured.
  BuildOptions opts;
  opts.n_groups = 2;
  auto ev = must_build("f =~ x1 + x2 + x3", opts);

  std::mt19937 rng(77);
  const Eigen::VectorXd theta = random_theta(ev, rng);
  const Eigen::MatrixXd J = ev.dsigma_dtheta(theta).value();
  const auto locs = ev.param_locations();

  const Eigen::Index vech_per_block = 6;   // vech(3×3)
  for (Eigen::Index row = 0; row < J.rows(); ++row) {
    const int row_block = static_cast<int>(row / vech_per_block);
    for (Eigen::Index k = 0; k < J.cols(); ++k) {
      if (locs[static_cast<std::size_t>(k)].block != row_block) {
        CHECK(std::abs(J(row, k)) < 1e-12);
      }
    }
  }
}

// === multi-group (n_b/N) weighting =========================================

// === equality-constraint affine reparameterization =========================

TEST_CASE("Property: EqConstraints expand/contract/reduce_gradient algebra") {
  // θ = θ₀ + K·α is the affine reparameterization. Check the three transform
  // methods are mutually consistent and that expand∘contract is a genuine
  // projection onto the constraint manifold.
  auto pt = must_lavaanify("f =~ x0 + a*x1 + b*x2 + c*x3\na == 2*b + c");
  auto con = build_eq_constraints(pt).value();
  REQUIRE(con.active());

  std::mt19937 rng(55);
  std::uniform_real_distribution<double> d(-1.5, 1.5);
  Eigen::VectorXd alpha(con.n_alpha);
  for (Eigen::Index k = 0; k < alpha.size(); ++k) alpha(k) = d(rng);

  // expand(α) == θ₀ + K·α.
  const Eigen::VectorXd theta = con.expand(alpha);
  CHECK((theta - (con.theta0 + con.Kmat * alpha)).cwiseAbs().maxCoeff() < 1e-12);

  // contract is the left inverse of expand on the manifold: a point produced
  // by expand round-trips back to its α.
  const Eigen::VectorXd alpha_rt = con.contract(theta);
  CHECK((alpha_rt - alpha).cwiseAbs().maxCoeff() < 1e-9);

  // expand∘contract is idempotent — a projection onto {θ₀ + K·α}.
  Eigen::VectorXd off_manifold(con.npar);
  for (Eigen::Index k = 0; k < off_manifold.size(); ++k) off_manifold(k) = d(rng);
  const Eigen::VectorXd proj1 = con.expand(con.contract(off_manifold));
  const Eigen::VectorXd proj2 = con.expand(con.contract(proj1));
  CHECK((proj1 - proj2).cwiseAbs().maxCoeff() < 1e-9);

  // reduce_gradient(g) == Kᵀg — the chain rule ∂F/∂α = Kᵀ ∂F/∂θ.
  Eigen::VectorXd grad(con.npar);
  for (Eigen::Index k = 0; k < grad.size(); ++k) grad(k) = d(rng);
  CHECK((con.reduce_gradient(grad) - con.Kmat.transpose() * grad)
            .cwiseAbs()
            .maxCoeff() < 1e-12);
}

TEST_CASE("Property: EqConstraints affine surface satisfies A_eq·θ = b_eq") {
  // The K-reparameterization and the affine system A_eq·θ = b_eq describe the
  // same manifold. Every expand(α) must therefore satisfy the affine system
  // exactly, for arbitrary α.
  for (std::string_view model :
       {"f =~ x1 + a*x2 + a*x3",                       // pure-merge path
        "f =~ x0 + a*x1 + b*x2 + c*x3\na == 2*b + c"}) {  // general-linear path
    CAPTURE(model);
    auto pt = must_lavaanify(model);
    auto con = build_eq_constraints(pt).value();
    REQUIRE(con.active());
    REQUIRE(con.A_eq.rows() == con.rank);

    std::mt19937 rng(88);
    std::uniform_real_distribution<double> d(-2.0, 2.0);
    for (int trial = 0; trial < 5; ++trial) {
      Eigen::VectorXd alpha(con.n_alpha);
      for (Eigen::Index k = 0; k < alpha.size(); ++k) alpha(k) = d(rng);
      const Eigen::VectorXd theta = con.expand(alpha);
      const Eigen::VectorXd resid = con.A_eq * theta - con.b_eq;
      CHECK(resid.cwiseAbs().maxCoeff() < 1e-9);
    }
  }
}

TEST_CASE("Property: EqConstraints K basis — merge is 0/1, general is orthonormal") {
  // Pure-merge path: K is the 0/1 group-membership matrix, so KᵀK is diagonal
  // with the group sizes on the diagonal. General-linear path: K is an SVD
  // kernel basis, so KᵀK = I. Both paths must have full column rank n_alpha.
  SUBCASE("pure-merge — shared label") {
    auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3");
    auto con = build_eq_constraints(pt).value();
    REQUIRE_FALSE(con.group.empty());
    const Eigen::MatrixXd KtK = con.Kmat.transpose() * con.Kmat;
    CHECK(KtK.isDiagonal(1e-12));
    // diagonal entry g == size of group g.
    for (Eigen::Index g = 0; g < KtK.rows(); ++g) {
      Eigen::Index members = 0;
      for (auto gid : con.group)
        if (gid == g) ++members;
      CHECK(KtK(g, g) == doctest::Approx(static_cast<double>(members)));
    }
  }
  SUBCASE("general-linear — a == 2*b + c") {
    auto pt = must_lavaanify("f =~ x0 + a*x1 + b*x2 + c*x3\na == 2*b + c");
    auto con = build_eq_constraints(pt).value();
    CHECK(con.group.empty());
    const Eigen::MatrixXd KtK = con.Kmat.transpose() * con.Kmat;
    CHECK(KtK.isApprox(
        Eigen::MatrixXd::Identity(con.n_alpha, con.n_alpha), 1e-9));
  }
}
