#include <doctest/doctest.h>

#include <Eigen/Core>
#include <Eigen/QR>

#include "magmaan/estimate/constraints.hpp"
#include "magmaan/robust/robust.hpp"


namespace {

// Hand-build an EqConstraints struct from a K matrix (and optional θ₀).
// Skips the `build_eq_constraints` pipeline so tests can express arbitrary
// nested pairs without spinning up a full lavaanify.
magmaan::estimate::EqConstraints make_eq(const Eigen::MatrixXd& K,
                          const Eigen::VectorXd& theta0 = {}) {
  magmaan::estimate::EqConstraints eq;
  eq.Kmat   = K;
  eq.theta0 = theta0.size() == K.rows() ? theta0
                                        : Eigen::VectorXd::Zero(K.rows());
  eq.npar    = static_cast<std::int32_t>(K.rows());
  eq.n_alpha = static_cast<std::int32_t>(K.cols());
  eq.rank    = static_cast<std::int32_t>(K.rows() - K.cols());
  return eq;
}

// Column-orthonormalise a matrix via thin QR.  The pure-merge K from
// `build_eq_constraints` is already orthonormal (0/1 columns scaled by
// √(group size)), but the helpers we use here construct K by hand.
Eigen::MatrixXd qr_orthonormalise(const Eigen::MatrixXd& K) {
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(K);
  Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(K.rows(),
                                                                      K.cols());
  return Q;
}

}  // namespace

// ── Sanity: identity H1, one extra restriction in H0 ───────────────────────

TEST_CASE("restriction_alpha_from_K: unconstrained H1, single equality in H0") {
  // H1: no constraints — K_H1 = I_3 (so α₁ = θ_full).
  const Eigen::MatrixXd K1 = Eigen::MatrixXd::Identity(3, 3);
  // H0: fix θ_0 = 0, so K_H0 spans { (0, θ_1, θ_2) }, i.e. e_2 ⊕ e_3.
  Eigen::MatrixXd K0(3, 2);
  K0 << 0, 0,
        1, 0,
        0, 1;

  auto r = magmaan::robust::restriction_alpha_from_K(make_eq(K1), make_eq(K0));
  REQUIRE(r.has_value());
  CHECK(r->A.rows() == 1);
  CHECK(r->A.cols() == 3);
  // A's row should be ±e_1 (i.e. picking the constrained direction).
  CHECK(std::abs(r->A(0, 0)) == doctest::Approx(1.0).epsilon(1e-12));
  CHECK(std::abs(r->A(0, 1)) < 1e-12);
  CHECK(std::abs(r->A(0, 2)) < 1e-12);
  CHECK(r->b.size() == 1);
  CHECK(r->b(0) == doctest::Approx(0.0).epsilon(1e-12));
}

// ── Equality already in H1, additional restriction in H0 ──────────────────

TEST_CASE("restriction_alpha_from_K: nested chain — H0 strictly inside H1") {
  // npar = 4.  H1 imposes θ_0 = θ_1 (one merge); 3 free α₁ directions.
  // H0 additionally imposes θ_2 = θ_3.
  // K_H1: columns 0..2 span the "θ_0 = θ_1" submanifold (with θ_2, θ_3 free).
  Eigen::MatrixXd K1(4, 3);
  K1 << 1, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;
  // K_H0: additionally θ_2 = θ_3.
  Eigen::MatrixXd K0(4, 2);
  K0 << 1, 0,
        1, 0,
        0, 1,
        0, 1;
  // The helper expects column-orthonormal K — but the columns above have
  // unit *direction* and norm > 1.  Orthonormalise to match what
  // `build_eq_constraints` would produce.
  K1 = qr_orthonormalise(K1);
  K0 = qr_orthonormalise(K0);

  auto r = magmaan::robust::restriction_alpha_from_K(make_eq(K1), make_eq(K0));
  REQUIRE(r.has_value());
  CHECK(r->A.rows() == 1);
  CHECK(r->A.cols() == 3);
  // A should have orthonormal rows.
  CHECK((r->A * r->A.transpose() - Eigen::MatrixXd::Identity(1, 1)).norm()
        < 1e-12);
  // The restriction it encodes, lifted back to θ-space, must annihilate
  // K_H0 (since K_H0 spans the admissible directions under H0) and pick up
  // the additional constraint direction.
  CHECK((r->A * (K1.transpose() * K0)).norm() < 1e-12);
}

// ── Degenerate: identical models (m = 0) ──────────────────────────────────

TEST_CASE("restriction_alpha_from_K: H0 ≡ H1 returns an empty restriction") {
  const Eigen::MatrixXd K = Eigen::MatrixXd::Identity(5, 3);
  auto r = magmaan::robust::restriction_alpha_from_K(make_eq(K), make_eq(K));
  REQUIRE(r.has_value());
  CHECK(r->A.rows() == 0);
  CHECK(r->A.cols() == 3);
  CHECK(r->b.size() == 0);
}

// ── Non-nested → NumericIssue ─────────────────────────────────────────────

TEST_CASE("restriction_alpha_from_K: rejects non-nested pair") {
  // H1: span {e_1, e_2}.  H0: span {e_2, e_3}.  Not nested.
  Eigen::MatrixXd K1(3, 2);
  K1 << 1, 0,
        0, 1,
        0, 0;
  Eigen::MatrixXd K0(3, 2);
  K0 << 0, 0,
        1, 0,
        0, 1;
  // Equal dimensions do not imply identical affine models.
  auto equal_dimension = magmaan::robust::restriction_alpha_from_K(make_eq(K1), make_eq(K0));
  CHECK_FALSE(equal_dimension.has_value());
  Eigen::MatrixXd K0_lower(3, 1);
  K0_lower << 0, 0, 1;
  auto r = magmaan::robust::restriction_alpha_from_K(make_eq(K1), make_eq(K0_lower));
  CHECK_FALSE(r.has_value());
}

// ── r_H0 > r_H1 → NumericIssue (wrong nesting order) ──────────────────────

TEST_CASE("restriction_alpha_from_K: rejects wrong-direction nesting") {
  const Eigen::MatrixXd K1 = Eigen::MatrixXd::Identity(3, 2);  // r1 = 2
  const Eigen::MatrixXd K0 = Eigen::MatrixXd::Identity(3, 3);  // r0 = 3 > r1
  auto r = magmaan::robust::restriction_alpha_from_K(make_eq(K1), make_eq(K0));
  CHECK_FALSE(r.has_value());
}

// ── Random nested pair: K_H0 = K_H1 · M_explicit ──────────────────────────

TEST_CASE("restriction_alpha_from_K: random K_H1·M = K_H0 round-trip") {
  // Build a random orthonormal K_H1 (npar=8, r1=5) and a random M (5×3),
  // form K_H0 = K_H1·M and re-orthonormalise.  The recovered A·Aᵀ = I_m and
  // A·(K_H1ᵀ·K_H0) = 0 (annihilates the H0-admissible directions).
  Eigen::MatrixXd raw1 = Eigen::MatrixXd::Random(8, 5);
  Eigen::MatrixXd K1   = qr_orthonormalise(raw1);
  Eigen::MatrixXd M    = Eigen::MatrixXd::Random(5, 3);
  Eigen::MatrixXd K0   = qr_orthonormalise(K1 * M);

  auto r = magmaan::robust::restriction_alpha_from_K(make_eq(K1), make_eq(K0));
  REQUIRE(r.has_value());
  CHECK(r->A.rows() == 2);  // m = r1 − r0 = 5 − 3
  CHECK(r->A.cols() == 5);
  CHECK((r->A * r->A.transpose() - Eigen::MatrixXd::Identity(2, 2)).norm()
        < 1e-10);
  // Annihilates the projection of K_H0 onto K_H1.
  CHECK((r->A * (K1.transpose() * K0)).norm() < 1e-10);
}

TEST_CASE("restriction_alpha_from_K: affine offset outside H1 is not nested") {
  Eigen::MatrixXd K1(3, 2), K0(3, 1);
  K1 << 1, 0, 0, 1, 0, 0;
  K0 << 1, 0, 0;
  Eigen::Vector3d offset(0, 0, 1);
  auto result = magmaan::robust::restriction_alpha_from_K(make_eq(K1),make_eq(K0,offset));
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().kind == magmaan::PostError::Kind::NotNested);
}

TEST_CASE("restriction_alpha_from_K: a nonzero fixed path retains its affine value") {
  const Eigen::MatrixXd K1=Eigen::MatrixXd::Identity(3,3);
  Eigen::MatrixXd K0(3,2);
  K0 << 0,0, 1,0, 0,1;
  const Eigen::Vector3d offset(0.25,0,0);
  auto result=magmaan::robust::restriction_alpha_from_K(make_eq(K1),make_eq(K0,offset));
  REQUIRE(result.has_value());
  CHECK((result->A*offset-result->b).norm()<1e-12);
  CHECK((result->A*K0).norm()<1e-12);
}

// TASK-168. The transported restriction must (a) reduce to the exact rows when
// the fit lies on the null, (b) transform with the larger model's coordinate
// change at its fit, whatever the (different) coordinate change at the null
// point, and (c) ignore invertible changes of moment coordinates when the
// metric transforms with them. Row spaces are compared through the
// annihilated directions, since the rows are orthonormal only up to rotation.
TEST_CASE("Transported nested restriction is invariant to reparameterization") {
  using magmaan::robust::restriction_null_directions;
  using magmaan::robust::transported_restriction_alpha;
  const Eigen::Index q = 12, r1 = 5, m = 2;
  std::srand(16804);
  const Eigen::MatrixXd fit = Eigen::MatrixXd::Random(q, r1);
  const Eigen::MatrixXd at_null = fit + 0.2 * Eigen::MatrixXd::Random(q, r1);
  const Eigen::MatrixXd root = Eigen::MatrixXd::Random(q, q);
  const Eigen::MatrixXd V = root * root.transpose() + Eigen::MatrixXd::Identity(q, q);
  const Eigen::MatrixXd exact = Eigen::HouseholderQR<Eigen::MatrixXd>(
      Eigen::MatrixXd::Random(r1, r1)).householderQ() *
      Eigen::MatrixXd::Identity(r1, m);
  const Eigen::MatrixXd A_exact = exact.transpose();
  const Eigen::MatrixXd N0 = restriction_null_directions(A_exact);
  REQUIRE(N0.cols() == r1 - m);
  CHECK((A_exact * N0).norm() < 1e-12);

  // (a) Same point: the exact row space.
  auto same = transported_restriction_alpha(V * fit, fit, fit, N0);
  REQUIRE(same.has_value());
  CHECK(same->rows() == m);
  CHECK((*same * N0).norm() < 1e-10);

  auto base = transported_restriction_alpha(V * fit, fit, at_null, N0);
  REQUIRE(base.has_value());
  const Eigen::MatrixXd kept = restriction_null_directions(*base);

  // (b) Different coordinate changes at the fit and at the null point.
  const Eigen::MatrixXd J_fit = Eigen::MatrixXd::Identity(r1, r1) +
                                0.3 * Eigen::MatrixXd::Random(r1, r1);
  const Eigen::MatrixXd J_null = Eigen::MatrixXd::Identity(r1, r1) +
                                 0.3 * Eigen::MatrixXd::Random(r1, r1);
  const Eigen::MatrixXd N0_new = J_null.fullPivLu().solve(N0);
  auto moved = transported_restriction_alpha(V * fit * J_fit, fit * J_fit,
                                             at_null * J_null, N0_new);
  REQUIRE(moved.has_value());
  const Eigen::MatrixXd kept_new = J_fit.fullPivLu().solve(kept);
  CHECK((*moved * kept_new).norm() / kept_new.norm() < 1e-10);
  // The exact rows carried by J_null are not the ones the fit needs.
  CHECK(((A_exact * J_null) * kept_new).norm() / kept_new.norm() > 1e-3);

  // (c) Moment coordinates: Δ -> DΔ with metric D^-T V D^-1.
  const Eigen::MatrixXd D = Eigen::MatrixXd::Identity(q, q) +
                            0.2 * Eigen::MatrixXd::Random(q, q);
  const Eigen::MatrixXd Dinv = D.fullPivLu().inverse();
  const Eigen::MatrixXd V_new = Dinv.transpose() * V * Dinv;
  auto rescaled = transported_restriction_alpha(V_new * D * fit, D * fit, D * at_null, N0);
  REQUIRE(rescaled.has_value());
  CHECK((*rescaled * kept).norm() < 1e-9);
}
