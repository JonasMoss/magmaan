#include <doctest/doctest.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "../test_fit.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// Phase-1 stabilizer bound builders. The data-derived builders mirror lavaan's
// `optim.bounds` algorithm (lavaan/R/lav_partable_bounds.R); the expected
// numbers below were cross-checked against lavaan 0.6.22's `cfa(..., bounds=)`
// — see the oracle traced in the test comments. magmaan uses the sample
// covariance directly as lavaan's internal (biased) `OV.VAR`, so a lavaan
// run reproduces these once its `sample.cov` input is pre-scaled by N/(N-1).

using magmaan::data::SampleStats;
using magmaan::estimate::ActiveBoundDiagnostics;
using magmaan::estimate::Bounds;
using magmaan::spec::LatentStructure;
using magmaan::spec::VarRole;
using Op = magmaan::parse::Op;
namespace est = magmaan::estimate;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

LatentStructure must_parse(std::string_view src) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  return std::move(*pt);
}

magmaan::model::MatrixRep must_rep(const LatentStructure& pt) {
  auto mr = magmaan::model::build_matrix_rep(pt);
  REQUIRE(mr.has_value());
  return std::move(*mr);
}

// Single-block SampleStats with a diagonal sample covariance — the off-diagonal
// is irrelevant to the bound builders (they read only variances).
SampleStats diag_samp(std::initializer_list<double> d) {
  Eigen::VectorXd v(static_cast<Eigen::Index>(d.size()));
  Eigen::Index k = 0;
  for (double x : d) v(k++) = x;
  SampleStats s;
  Eigen::MatrixXd S = v.asDiagonal();
  s.S.push_back(std::move(S));
  s.n_obs.push_back(200);
  return s;
}

// 0-based θ index of the first free row matching `pred`, or -1.
template <class Pred>
Eigen::Index free_of(const LatentStructure& pt, Pred pred) {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] > 0 && pred(i)) {
      return static_cast<Eigen::Index>(pt.free[i] - 1);
    }
  }
  return -1;
}

bool is_latent(const LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == VarRole::Latent;
}

}  // namespace

// ============================================================================
// standard_bounds — lavaan bounds = "standard".
// ============================================================================

TEST_CASE("standard_bounds: 1F CFA reproduces the lavaan standard box") {
  // f =~ x1 + x2 + x3, marker convention (λ_x1 = 1). Sample variances picked
  // so the loading bounds are exact: var(x1)=1.0 (marker), var(x2)=1.6,
  // var(x3)=0.4. lavaan oracle: cfa("f=~x1+x2+x3", sample.cov=S*N/(N-1),
  // sample.nobs=N, bounds="standard").
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto samp = diag_samp({1.0, 1.6, 0.4});
  auto sb = est::standard_bounds(pt, samp);
  REQUIRE(sb.has_value());
  const Bounds& b = *sb;

  auto loading = [&](int ov) {
    return free_of(pt, [&](std::size_t i) {
      return pt.op[i] == Op::Measurement &&
             pt.ov_pos[static_cast<std::size_t>(pt.rhs_var[i])] == ov;
    });
  };
  auto resvar = [&](int ov) {
    return free_of(pt, [&](std::size_t i) {
      return pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i] &&
             !is_latent(pt, pt.lhs_var[i]) &&
             pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])] == ov;
    });
  };
  auto latvar = free_of(pt, [&](std::size_t i) {
    return pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i] &&
           is_latent(pt, pt.lhs_var[i]);
  });

  // Loadings: ±sqrt(var(indicator) / (0.1 · var(marker))) = ±sqrt(var / 0.1).
  CHECK(b.lower(loading(1)) == doctest::Approx(-4.0));   // sqrt(1.6/0.1)
  CHECK(b.upper(loading(1)) == doctest::Approx(+4.0));
  CHECK(b.lower(loading(2)) == doctest::Approx(-2.0));   // sqrt(0.4/0.1)
  CHECK(b.upper(loading(2)) == doctest::Approx(+2.0));

  // Residual variances: [0, var] with the marker capped at (1-0.1)·var and the
  // upper bumped 0.5% (standard ov.var upper factor 1.0).
  CHECK(b.lower(resvar(0)) == doctest::Approx(0.0));
  CHECK(b.upper(resvar(0)) == doctest::Approx(0.9 * 1.0 * 1.005));   // marker
  CHECK(b.upper(resvar(1)) == doctest::Approx(1.6 * 1.005));
  CHECK(b.upper(resvar(2)) == doctest::Approx(0.4 * 1.005));

  // Latent variance: [0.1·var(marker), var(marker)], no enlargement.
  CHECK(b.lower(latvar) == doctest::Approx(0.1));
  CHECK(b.upper(latvar) == doctest::Approx(1.0));
}

TEST_CASE("standard_bounds: factor and residual covariances get a box") {
  // lavaan oracle (bounds="standard"): f1~~f2 ∈ ±0.998, x2~~x5 ∈ ±1.604784.
  auto pt = must_parse(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f1 ~~ f2\n"
      "x2 ~~ x5");
  auto samp = diag_samp({1.0, 1.6, 0.4, 1.0, 1.6, 0.4});
  auto sb = est::standard_bounds(pt, samp);
  REQUIRE(sb.has_value());
  const Bounds& b = *sb;

  auto cov_ll = free_of(pt, [&](std::size_t i) {
    return pt.op[i] == Op::Covariance && pt.lhs_var[i] != pt.rhs_var[i] &&
           is_latent(pt, pt.lhs_var[i]) && is_latent(pt, pt.rhs_var[i]);
  });
  auto cov_oo = free_of(pt, [&](std::size_t i) {
    return pt.op[i] == Op::Covariance && pt.lhs_var[i] != pt.rhs_var[i] &&
           !is_latent(pt, pt.lhs_var[i]) && !is_latent(pt, pt.rhs_var[i]);
  });
  REQUIRE(cov_ll >= 0);
  REQUIRE(cov_oo >= 0);

  // f1~~f2: ±sqrt(ub(f1)·ub(f2)) shrunk 0.1% — ub(fk) = var(marker) = 1.0.
  CHECK(b.upper(cov_ll) == doctest::Approx(0.998));
  CHECK(b.lower(cov_ll) == doctest::Approx(-0.998));
  // x2~~x5: ±ucov with ucov = sqrt(ub(θ_x2)·ub(θ_x5)) = 1.6·1.005 = 1.608,
  // shrunk by the signed range delta (range = 2·ucov, factor 0.999) ⇒
  // ucov·0.998 = 1.604784.
  CHECK(b.upper(cov_oo) == doctest::Approx(1.608 * 0.998));
  CHECK(b.lower(cov_oo) == doctest::Approx(-1.608 * 0.998));
}

// ============================================================================
// wide_bounds — lavaan bounds = "wide".
// ============================================================================

TEST_CASE("wide_bounds: 1F CFA reproduces the lavaan wide box") {
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto samp = diag_samp({1.0, 1.6, 0.4});
  auto wb = est::wide_bounds(pt, samp);
  REQUIRE(wb.has_value());
  const Bounds& b = *wb;

  auto loading = [&](int ov) {
    return free_of(pt, [&](std::size_t i) {
      return pt.op[i] == Op::Measurement &&
             pt.ov_pos[static_cast<std::size_t>(pt.rhs_var[i])] == ov;
    });
  };
  auto resvar = [&](int ov) {
    return free_of(pt, [&](std::size_t i) {
      return pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i] &&
             !is_latent(pt, pt.lhs_var[i]) &&
             pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])] == ov;
    });
  };
  auto latvar = free_of(pt, [&](std::size_t i) {
    return pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i] &&
           is_latent(pt, pt.lhs_var[i]);
  });

  // Loadings: raw box ±sqrt(var/0.1) widened 10% (loading factor 1.1).
  CHECK(b.upper(loading(1)) == doctest::Approx(4.8));   // 4.0 + (8.0·0.1)/2
  CHECK(b.upper(loading(2)) == doctest::Approx(2.4));   // 2.0 + (4.0·0.1)/2

  // Residual variances: lower -5%·range, upper +20%·range (marker x1: 0.9·var).
  CHECK(b.lower(resvar(0)) == doctest::Approx(-0.045));  // -0.05·0.9
  CHECK(b.upper(resvar(0)) == doctest::Approx(1.08));    // 0.9·1.20
  CHECK(b.lower(resvar(1)) == doctest::Approx(-0.08));   // -0.05·1.6
  CHECK(b.upper(resvar(1)) == doctest::Approx(1.92));    // 1.6·1.20

  // Latent variance: lower unchanged (factor 1.0), upper +30%·range.
  CHECK(b.lower(latvar) == doctest::Approx(0.1));
  CHECK(b.upper(latvar) == doctest::Approx(1.0 + 0.9 * 0.30));  // 1.27
}

// ============================================================================
// loading_bounds — loadings only.
// ============================================================================

TEST_CASE("loading_bounds: bounds the loadings, leaves variances unbounded") {
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto samp = diag_samp({1.0, 1.6, 0.4});
  auto lb = est::loading_bounds(pt, samp);
  REQUIRE(lb.has_value());
  const Bounds& b = *lb;

  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    const Eigen::Index k = pt.free[i] - 1;
    if (pt.op[i] == Op::Measurement) {
      // Same box as standard_bounds' loadings (factor 1.0).
      const int ov = pt.ov_pos[static_cast<std::size_t>(pt.rhs_var[i])];
      const double raw = (ov == 1) ? 4.0 : 2.0;
      CHECK(b.lower(k) == doctest::Approx(-raw));
      CHECK(b.upper(k) == doctest::Approx(+raw));
    } else {
      // Variances stay unbounded.
      CHECK(b.lower(k) == -kInf);
      CHECK(b.upper(k) == +kInf);
    }
  }
}

// ============================================================================
// variance_bounds — lavaan bounds = "pos.var" (and bounds_from_partable alias).
// ============================================================================

TEST_CASE("variance_bounds: variance diagonals get lower 0, alias matches") {
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto vb = est::variance_bounds(pt);
  auto alias = est::bounds_from_partable(pt);
  REQUIRE(vb.has_value());
  REQUIRE(alias.has_value());
  CHECK(vb->lower == alias->lower);
  CHECK(vb->upper == alias->upper);

  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    const Eigen::Index k = pt.free[i] - 1;
    const bool is_var =
        pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i];
    CHECK(vb->lower(k) == (is_var ? 0.0 : -kInf));
    CHECK(vb->upper(k) == +kInf);
  }
}

// ============================================================================
// intersect_bounds.
// ============================================================================

TEST_CASE("intersect_bounds: tighter of each coordinate; empty is identity") {
  Bounds a;
  a.lower = Eigen::Vector3d(-1.0, -kInf, 0.0);
  a.upper = Eigen::Vector3d(+kInf, 2.0, 5.0);
  Bounds b;
  b.lower = Eigen::Vector3d(-2.0, 0.5, 0.0);
  b.upper = Eigen::Vector3d(1.0, +kInf, 3.0);

  auto x = est::intersect_bounds(a, b);
  REQUIRE(x.has_value());
  CHECK(x->lower(0) == -1.0);   // max(-1, -2)
  CHECK(x->lower(1) == 0.5);    // max(-inf, 0.5)
  CHECK(x->lower(2) == 0.0);
  CHECK(x->upper(0) == 1.0);    // min(+inf, 1.0)
  CHECK(x->upper(1) == 2.0);
  CHECK(x->upper(2) == 3.0);

  // An empty operand returns the other unchanged.
  auto e1 = est::intersect_bounds(Bounds{}, a);
  auto e2 = est::intersect_bounds(b, Bounds{});
  REQUIRE(e1.has_value());
  REQUIRE(e2.has_value());
  CHECK(e1->lower == a.lower);
  CHECK(e2->upper == b.upper);

  // Mismatched non-empty sizes are an error.
  Bounds small;
  small.lower = Eigen::Vector2d(0.0, 0.0);
  small.upper = Eigen::Vector2d(1.0, 1.0);
  CHECK_FALSE(est::intersect_bounds(a, small).has_value());
}

TEST_CASE("intersect_bounds: stack loading_bounds onto variance_bounds") {
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto samp = diag_samp({1.0, 1.6, 0.4});
  auto vb = est::variance_bounds(pt);
  auto lb = est::loading_bounds(pt, samp);
  REQUIRE(vb.has_value());
  REQUIRE(lb.has_value());
  auto both = est::intersect_bounds(*vb, *lb);
  REQUIRE(both.has_value());

  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    const Eigen::Index k = pt.free[i] - 1;
    if (pt.op[i] == Op::Measurement) {
      CHECK(both->lower(k) < 0.0);          // loading box from loading_bounds
      CHECK(both->upper(k) > 0.0);
    } else {
      CHECK(both->lower(k) == 0.0);         // variance floor from variance_bounds
    }
  }
}

// ============================================================================
// project_start_into_bounds.
// ============================================================================

TEST_CASE("project_start_into_bounds: clamps and reports moved coordinates") {
  Bounds b;
  b.lower = Eigen::Vector3d(0.0, -1.0, -kInf);
  b.upper = Eigen::Vector3d(+kInf, 1.0, 2.0);
  Eigen::Vector3d x0(-0.5, 5.0, 1.0);

  auto p = est::project_start_into_bounds(x0, b);
  REQUIRE(p.has_value());
  CHECK(p->x0(0) == 0.0);     // clamped up to lower
  CHECK(p->x0(1) == 1.0);     // clamped down to upper
  CHECK(p->x0(2) == 1.0);     // already feasible
  REQUIRE(p->clamped.size() == 2);
  CHECK(p->clamped[0] == 0);
  CHECK(p->clamped[1] == 1);

  // Empty bounds — start passes through unchanged.
  auto none = est::project_start_into_bounds(x0, Bounds{});
  REQUIRE(none.has_value());
  CHECK(none->x0 == x0);
  CHECK(none->clamped.empty());

  // Size mismatch is an error.
  Eigen::Vector2d wrong(0.0, 0.0);
  CHECK_FALSE(est::project_start_into_bounds(wrong, b).has_value());
}

// ============================================================================
// active_bounds.
// ============================================================================

TEST_CASE("active_bounds: flags coordinates pinned to a finite bound") {
  Bounds b;
  b.lower = Eigen::Vector4d(0.0, -1.0, -kInf, 0.0);
  b.upper = Eigen::Vector4d(+kInf, 1.0, 2.0, +kInf);
  Eigen::Vector4d theta(0.0, 1.0, 0.5, 3.0);

  auto d = est::active_bounds(theta, b, 1e-6);
  REQUIRE(d.has_value());
  REQUIRE(d->at_lower.size() == 1);
  CHECK(d->at_lower[0] == 0);          // θ_0 sits on its lower bound
  REQUIRE(d->at_upper.size() == 1);
  CHECK(d->at_upper[0] == 1);          // θ_1 sits on its upper bound
  CHECK(d->any_active());

  // Empty bounds — nothing active.
  auto none = est::active_bounds(theta, Bounds{});
  REQUIRE(none.has_value());
  CHECK_FALSE(none->any_active());
}

// ============================================================================
// End-to-end — a named builder used as a Heywood stabilizer in an ML fit.
// ============================================================================

TEST_CASE("variance_bounds: stabilizes a Heywood-prone 1F ML fit") {
  // A just-identified 1F model whose data implies a negative residual
  // variance for x1: σ12·σ13/σ23 = ψ = 0.49/0.3 ≈ 1.633 > σ11 = 1, so the
  // unbounded ML solution has θ_x1 = 1 - 1.633 < 0. variance_bounds() should
  // keep every residual variance feasible, with x1's pinned at its lower 0.
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto rep = must_rep(pt);

  SampleStats samp;
  Eigen::Matrix3d S;
  S << 1.0, 0.7, 0.7,
       0.7, 1.0, 0.3,
       0.7, 0.3, 1.0;
  samp.S.push_back(S);
  samp.n_obs.push_back(200);

  auto vb = est::variance_bounds(pt);
  REQUIRE(vb.has_value());

  auto out = magmaan::test::fit(pt, rep, samp, *vb);
  REQUIRE_MESSAGE(out.has_value(),
      "bounded ML fit failed: " << (out.has_value() ? "" : out.error().detail));

  // Every residual variance lands feasible (≥ 0).
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    if (pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i]) {
      CHECK(out->theta(pt.free[i] - 1) >= -1e-8);
    }
  }

  // The Heywood coordinate is pinned to its bound — active-bound diagnostics
  // make that visible.
  auto diag = est::active_bounds(out->theta, *vb, 1e-6);
  REQUIRE(diag.has_value());
  CHECK(diag->any_active());
}

TEST_CASE("variance_bounds: stabilizes a Heywood-prone 1F ULS fit") {
  // Same Heywood-prone S as the ML slice — a just-identified 1F model whose
  // unconstrained ULS optimum also drives θ_x1 negative. The named builder
  // keeps the least-squares fit feasible too.
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto rep = must_rep(pt);

  SampleStats samp;
  Eigen::Matrix3d S;
  S << 1.0, 0.7, 0.7,
       0.7, 1.0, 0.3,
       0.7, 0.3, 1.0;
  samp.S.push_back(S);
  samp.n_obs.push_back(200);

  auto vb = est::variance_bounds(pt);
  REQUIRE(vb.has_value());

  auto out = magmaan::test::fit_gmm(pt, rep, samp, {}, *vb);
  REQUIRE_MESSAGE(out.has_value(),
      "bounded ULS fit failed: " << (out.has_value() ? "" : out.error().detail));

  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    if (pt.op[i] == Op::Covariance && pt.lhs_var[i] == pt.rhs_var[i]) {
      CHECK(out->theta(pt.free[i] - 1) >= -1e-8);
    }
  }
  auto diag = est::active_bounds(out->theta, *vb, 1e-6);
  REQUIRE(diag.has_value());
  CHECK(diag->any_active());
}
