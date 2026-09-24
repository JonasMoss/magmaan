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
  // Even though dim(H0) = dim(H1), H0 is not a sub-manifold.  With m = 0
  // the helper short-circuits *before* checking inclusion, so make r0 < r1
  // to exercise the inclusion check.
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
