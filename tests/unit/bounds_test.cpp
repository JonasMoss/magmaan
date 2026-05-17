#include <doctest/doctest.h>

#include <limits>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::estimate::Bounds;
using magmaan::estimate::bounds_from_partable;

namespace {

magmaan::spec::LatentStructure must_parse(std::string_view src,
                                            int n_groups = 1) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LavaanifyOptions opts;
  opts.n_groups = n_groups;
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  return std::move(*pt);
}

}  // namespace

TEST_CASE("bounds_from_partable: 1F CFA — variance diagonals get lower=0") {
  auto pt = must_parse("f =~ x1 + x2 + x3");
  auto b_or = bounds_from_partable(pt);
  REQUIRE(b_or.has_value());
  const Bounds& b = *b_or;

  // Every free variance-diagonal (Ψ_ff, Θ_11, Θ_22, Θ_33) has lower = 0.
  // Every loading λ_2, λ_3 stays unbounded.
  CHECK(b.lower.size() == pt.n_free());
  CHECK(b.upper.size() == pt.n_free());

  // Sanity: at least *some* coordinates are zero-lower (the variance ones),
  // and at least some are -inf-lower (the loadings).
  int n_zero = 0, n_inf = 0;
  for (Eigen::Index i = 0; i < b.lower.size(); ++i) {
    if (b.lower(i) == 0.0)                          ++n_zero;
    if (b.lower(i) == -std::numeric_limits<double>::infinity()) ++n_inf;
  }
  CHECK(n_zero >= 1);   // ψ + 3 θ — at least one zero-lower
  CHECK(n_inf  >= 1);   // λ_2, λ_3 — at least one ±inf-lower

  // Every upper is +inf — nothing has an upper bound in the variance-only
  // auto-derivation.
  for (Eigen::Index i = 0; i < b.upper.size(); ++i) {
    CHECK(b.upper(i) == std::numeric_limits<double>::infinity());
  }
}

TEST_CASE("bounds_from_partable: counts match Ψ + Θ free diagonals") {
  // 2F CFA with intercepts: Ψ has 2 free diagonals (ψ_v_v, ψ_t_t) + the
  // off-diagonal (free covariance — NOT a diagonal). Θ has 6 free diagonals.
  // → 8 zero-lower entries. Loadings (4 free) + ν (6 free) + ψ_off-diag (1) →
  // 11 ±inf-lower entries.
  auto pt = must_parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "x1 ~ 1\nx2 ~ 1\nx3 ~ 1\nx4 ~ 1\nx5 ~ 1\nx6 ~ 1");
  auto b_or = bounds_from_partable(pt);
  REQUIRE(b_or.has_value());
  const Bounds& b = *b_or;

  int n_zero = 0;
  for (Eigen::Index i = 0; i < b.lower.size(); ++i) {
    if (b.lower(i) == 0.0) ++n_zero;
  }
  CHECK(n_zero == 8);
}

TEST_CASE("bounds_from_partable: empty partable returns empty Bounds") {
  // Edge case — a parser-valid but parameter-free model is unusual but
  // bounds_from_partable must handle n_free == 0 cleanly.
  magmaan::spec::LatentStructure pt;
  auto b_or = bounds_from_partable(pt);
  REQUIRE(b_or.has_value());
  CHECK(b_or->empty());
}

TEST_CASE("bounds_from_partable: multi-group, shared labels — bound applies "
          "once per merged θ index") {
  // Multi-group invariance: residual variances tied across groups via labels.
  // The shared θ index gets the lower=0 bound exactly once (idempotent —
  // setting it twice from two rows is fine).
  auto pt = must_parse(
      "visual =~ x1 + l2*x2 + l3*x3", /*n_groups=*/2);
  auto b_or = bounds_from_partable(pt);
  REQUIRE(b_or.has_value());
  CHECK(b_or->lower.size() == pt.n_free());
  CHECK(b_or->upper.size() == pt.n_free());
  // Variance diagonals exist across both groups; some θ indices must be
  // lower-bounded at 0.
  bool any_zero = false;
  for (Eigen::Index i = 0; i < b_or->lower.size(); ++i) {
    if (b_or->lower(i) == 0.0) { any_zero = true; break; }
  }
  CHECK(any_zero);
}
