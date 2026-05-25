#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>
#include <string_view>

#include "magmaan/error.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::PartableError;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::BuildOptions;
using magmaan::compat::lavaan::LavaanParTable;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;
using magmaan::spec::Starts;
using magmaan::compat::lavaan::to_lavaan_partable;

namespace {

// The tests inspect the lavaan-shaped projection (`lhs` / `rhs` / `user` /
// `block` / `label` / `plabel` / `ustart`) — it's the most direct view of
// what lavaanify produced. The structure + names + starts are bundled back
// up via to_lavaan_partable.
LavaanParTable must_lavaanify(std::string_view src, BuildOptions opts = {}) {
  auto fp = Parser::parse(src);
  REQUIRE_MESSAGE(fp.has_value(), "parser failed: " << fp.error().detail);
  Starts starts;
  LatentNames names;
  auto pt = build(*fp, opts, &starts, &names);
  REQUIRE_MESSAGE(pt.has_value(),
                  "lavaanify failed: kind=" << static_cast<int>(pt.error().kind)
                  << " — " << pt.error().detail);
  return to_lavaan_partable(*pt, names, starts);
}

}  // namespace

TEST_CASE("lavaanify: 1-factor CFA produces 4 rows (3 loadings + 1 var auto.fix.first marker)") {
  // f =~ x1 + x2 + x3
  //   → loadings rows for x1 (auto-fixed to 1), x2, x3
  //   → residual variances for x1, x2, x3
  //   → factor variance for f
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  REQUIRE(pt.size() == 7);

  // Row 0: f =~ x1, auto-fixed to 1
  CHECK(pt.lhs[0]    == "f");
  CHECK(pt.op[0]     == Op::Measurement);
  CHECK(pt.rhs[0]    == "x1");
  CHECK(pt.user[0]   == 1);
  CHECK(pt.free[0]   == 0);
  CHECK(pt.ustart[0] == 1.0);

  // Rows 1, 2: free loadings
  CHECK(pt.free[1] > 0);
  CHECK(pt.free[2] > 0);
  CHECK(std::isnan(pt.ustart[1]));

  // Rows 3-5: residual variances (auto-added, free)
  CHECK(pt.lhs[3] == "x1");
  CHECK(pt.op[3]  == Op::Covariance);
  CHECK(pt.rhs[3] == "x1");
  CHECK(pt.user[3] == 0);
  CHECK(pt.free[3] > 0);

  // Row 6: factor variance (auto-added, free)
  CHECK(pt.lhs[6] == "f");
  CHECK(pt.op[6]  == Op::Covariance);
  CHECK(pt.rhs[6] == "f");
  CHECK(pt.user[6] == 0);
  CHECK(pt.free[6] > 0);

  // n_free should equal the count of free rows (2 free loadings + 4 variances)
  CHECK(pt.n_free() == 6);
}

TEST_CASE("lavaanify: std_lv frees the first loading and fixes the LV variance at 1") {
  // f =~ x1 + x2 + x3, std.lv = TRUE
  //   → all three loadings free; f ~~ f fixed at 1.0 (auto-added)
  BuildOptions opts;
  opts.std_lv = true;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);
  REQUIRE(pt.size() == 7);

  // Row 0: f =~ x1 — now FREE (no marker fix under std.lv)
  CHECK(pt.lhs[0]  == "f");
  CHECK(pt.op[0]   == Op::Measurement);
  CHECK(pt.rhs[0]  == "x1");
  CHECK(pt.free[0] > 0);
  CHECK(std::isnan(pt.ustart[0]));
  CHECK(pt.free[1] > 0);
  CHECK(pt.free[2] > 0);

  // Row 6: f ~~ f — FIXED at 1.0, auto-added (user=0)
  CHECK(pt.lhs[6]    == "f");
  CHECK(pt.op[6]     == Op::Covariance);
  CHECK(pt.rhs[6]    == "f");
  CHECK(pt.user[6]   == 0);
  CHECK(pt.free[6]   == 0);
  CHECK(pt.ustart[6] == 1.0);

  // 3 free loadings + 3 residual variances; LV variance fixed → 6 free params.
  CHECK(pt.n_free() == 6);

  // Same model under the (default) marker convention: same #rows, same
  // #free params — a bijective reparameterization.
  auto marker = must_lavaanify("f =~ x1 + x2 + x3");
  CHECK(marker.size()   == pt.size());
  CHECK(marker.n_free() == pt.n_free());
  CHECK(marker.free[0]  == 0);   // marker fixes the first loading...
  CHECK(pt.free[0]      != 0);   // ...std.lv does not.
}

TEST_CASE("lavaanify: std_lv ignores auto_fix_first and leaves a user-fixed LV variance alone") {
  // f =~ x1 + x2 + x3; f ~~ 2*f, std.lv = TRUE (auto_fix_first left on — std.lv wins)
  BuildOptions opts;
  opts.std_lv = true;
  opts.auto_fix_first = true;   // forced off by std_lv
  auto pt = must_lavaanify("f =~ x1 + x2 + x3\nf ~~ 2*f", opts);

  CHECK(pt.free[0] > 0);   // first loading still free

  bool found = false;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == Op::Covariance && pt.lhs[i] == "f" && pt.rhs[i] == "f") {
      found = true;
      CHECK(pt.free[i]   == 0);
      CHECK(pt.ustart[i] == 2.0);   // the user's value, not std.lv's 1.0
    }
  }
  CHECK(found);
}

TEST_CASE("lavaanify: ids are 1-based, plabels match ids") {
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  for (std::size_t i = 0; i < pt.size(); ++i) {
    CHECK(pt.id[i] == static_cast<std::int32_t>(i + 1));
    CHECK(pt.plabel[i] == ".p" + std::to_string(i + 1) + ".");
  }
}

TEST_CASE("lavaanify: shared label produces auto-equality constraint row with user=2") {
  // f =~ x1 + a*x2 + a*x3
  //   → 3 loading rows; x2 and x3 carry label "a"
  //   → residual variances + factor variance (4 rows)
  //   → ONE auto-equality row: .p2. == .p3., user=2
  auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  REQUIRE(pt.size() == 8);

  // Last row should be the auto-equality constraint.
  const std::size_t last = pt.size() - 1;
  CHECK(pt.op[last]    == Op::EqConstraint);
  CHECK(pt.user[last]  == 2);
  CHECK(pt.block[last] == 0);
  CHECK(pt.group[last] == 0);
  CHECK(pt.lhs[last]   == ".p2.");
  CHECK(pt.rhs[last]   == ".p3.");

  // The two source rows keep distinct free indices.
  CHECK(pt.free[1] != pt.free[2]);
  CHECK(pt.free[1] > 0);
  CHECK(pt.free[2] > 0);
  CHECK(pt.label[1] == "a");
  CHECK(pt.label[2] == "a");
}

TEST_CASE("lavaanify: equal() modifier ties a loading to a named parameter") {
  // equal("f=~x2")*x3 ties the x3 loading to the x2 loading. magmaan
  // expresses the tie as a shared label — the same fitted model as the
  // shared-label form `f =~ x1 + a*x2 + a*x3`.
  auto pt = must_lavaanify("f =~ x1 + x2 + equal(\"f=~x2\")*x3");
  REQUIRE(pt.size() == 8);

  // The x2 and x3 loadings both carry the resolved label, distinct free idx.
  CHECK(pt.label[1] == "f=~x2");
  CHECK(pt.label[2] == "f=~x2");
  CHECK(pt.free[1] != pt.free[2]);
  CHECK(pt.free[1] > 0);
  CHECK(pt.free[2] > 0);

  // ... and a synthesized auto-equality row ties their plabels.
  const std::size_t last = pt.size() - 1;
  CHECK(pt.op[last]  == Op::EqConstraint);
  CHECK(pt.lhs[last] == ".p2.");
  CHECK(pt.rhs[last] == ".p3.");
}

TEST_CASE("lavaanify: equal() referencing an unknown parameter is rejected") {
  auto fp = Parser::parse("f =~ x1 + x2 + equal(\"f=~nope\")*x3");
  REQUIRE(fp.has_value());
  Starts starts;
  LatentNames names;
  auto pt = build(*fp, BuildOptions{}, &starts, &names);
  REQUIRE_FALSE(pt.has_value());
  CHECK(pt.error().kind == PartableError::Kind::UnknownLabelInConstraint);
}

TEST_CASE("lavaanify: regression triggers fixed.x mirror") {
  // y ~ x1 + x2 + x3
  //   → 3 regression rows (free)
  //   → residual variance for y (free)
  //   → x1~~x1, x1~~x2, x1~~x3, x2~~x2, x2~~x3, x3~~x3 (exo=1, free=0)
  auto pt = must_lavaanify("y ~ x1 + x2 + x3");

  // Find the x1~~x1 row.
  bool found_x1_var = false;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.lhs[i] == "x1" && pt.op[i] == Op::Covariance && pt.rhs[i] == "x1") {
      CHECK(pt.exo[i]  == 1);
      CHECK(pt.free[i] == 0);
      CHECK(pt.user[i] == 0);
      found_x1_var = true;
      break;
    }
  }
  CHECK(found_x1_var);

  // Find the x1~~x2 covariance row.
  bool found_x1_x2 = false;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.lhs[i] == "x1" && pt.op[i] == Op::Covariance && pt.rhs[i] == "x2") {
      CHECK(pt.exo[i]  == 1);
      CHECK(pt.free[i] == 0);
      found_x1_x2 = true;
      break;
    }
  }
  CHECK(found_x1_x2);

  // The 3 regression rows themselves are free with exo=0.
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == Op::Regression) {
      CHECK(pt.free[i] > 0);
      CHECK(pt.exo[i]  == 0);
    }
  }
}

TEST_CASE("lavaanify: define-param row appears with op=':=' and label=name") {
  auto pt = must_lavaanify("indirect := a * b");
  REQUIRE(pt.size() == 1);
  CHECK(pt.op[0]    == Op::DefineParam);
  CHECK(pt.lhs[0]   == "indirect");
  CHECK(pt.rhs[0]   == "a*b");        // canonicalized: no spaces
  CHECK(pt.user[0]  == 1);
  CHECK(pt.label[0] == "indirect");
  CHECK(pt.block[0] == 0);
  CHECK(pt.group[0] == 0);
  CHECK(pt.free[0]  == 0);
}

TEST_CASE("lavaanify: explicit constraint rows have block=0, group=0") {
  auto pt = must_lavaanify("a == b ; c > 0 ; d < 1");
  REQUIRE(pt.size() == 3);
  CHECK(pt.op[0] == Op::EqConstraint);
  CHECK(pt.op[1] == Op::GtConstraint);
  CHECK(pt.op[2] == Op::LtConstraint);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(pt.block[i] == 0);
    CHECK(pt.group[i] == 0);
    CHECK(pt.user[i]  == 1);
    CHECK(pt.free[i]  == 0);
  }
}

TEST_CASE("lavaanify: meanstructure auto-adds ν (free) and α (fixed) rows") {
  BuildOptions opts;
  opts.meanstructure = true;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);

  // Should now contain ν rows (one per ov) plus α row (one per lv).
  // Original 7 rows (3 =~ + 3 θ + 1 ψ) → +3 ν + 1 α = 11 rows.
  CHECK(pt.size() == 11);

  int n_nu = 0, n_alpha = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Intercept) continue;
    // Intercept rows for ov (x1/x2/x3) are free; for lv (f) fixed at 0.
    if (pt.lhs[i] == "f") {
      ++n_alpha;
      CHECK(pt.free[i] == 0);          // fixed
      CHECK(pt.ustart[i] == doctest::Approx(0.0));
    } else {
      ++n_nu;
      CHECK(pt.free[i] > 0);           // free
    }
  }
  CHECK(n_nu    == 3);
  CHECK(n_alpha == 1);
}

TEST_CASE("lavaanify: growth defaults fix OV intercepts and free LV means") {
  BuildOptions opts;
  opts.meanstructure = true;
  opts.auto_cov_y = true;
  opts.int_ov_free = false;
  opts.int_lv_free = true;

  auto pt = must_lavaanify(
      "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4\n"
      "s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4\n"
      "i ~ x1 + x2\n"
      "s ~ x1 + x2\n"
      "t1 ~ c1\n"
      "t2 ~ c2\n"
      "t3 ~ c3\n"
      "t4 ~ c4",
      opts);

  auto row_free = [&](std::string_view lhs, Op op,
                      std::string_view rhs) -> int {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.lhs[i] == lhs && pt.op[i] == op && pt.rhs[i] == rhs) {
        return pt.free[i];
      }
    }
    return -1;
  };
  auto row_ustart = [&](std::string_view lhs, Op op,
                        std::string_view rhs) -> double {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.lhs[i] == lhs && pt.op[i] == op && pt.rhs[i] == rhs) {
        return pt.ustart[i];
      }
    }
    return std::numeric_limits<double>::quiet_NaN();
  };

  CHECK(row_free("t1", Op::Intercept, "") == 0);
  CHECK(row_free("t2", Op::Intercept, "") == 0);
  CHECK(row_free("t3", Op::Intercept, "") == 0);
  CHECK(row_free("t4", Op::Intercept, "") == 0);
  CHECK(row_ustart("t1", Op::Intercept, "") == doctest::Approx(0.0));
  CHECK(row_ustart("t4", Op::Intercept, "") == doctest::Approx(0.0));

  CHECK(row_free("i", Op::Intercept, "") > 0);
  CHECK(row_free("s", Op::Intercept, "") > 0);
  CHECK(row_free("i", Op::Covariance, "s") > 0);
}

TEST_CASE("lavaanify: auto_cov_y pairs terminal endogenous OV-Y "
          "(newsom_2015_ex5_6a pattern)") {
  // Three-wave cross-lagged panel path model with no latents — lavaan's
  // sem() default `auto.cov.y = TRUE` adds the residual covariance between
  // the two wave-3 (terminal) endogenous Ys (`w3posaff ~~ w3unw`) but does
  // NOT correlate the wave-2 Ys (`w2posaff`, `w2unw`) because they are
  // non-terminal (they predict the wave-3 outcomes). magmaan's
  // `auto_cov_y` must produce the same single auto-row.
  BuildOptions opts;
  opts.meanstructure = true;
  opts.auto_cov_y = true;
  auto pt = must_lavaanify(
      "w2posaff ~ w1unw\n"
      "w3posaff ~ w2unw\n"
      "w2unw ~ w1posaff\n"
      "w3unw ~ w2posaff\n"
      "w2posaff ~ w1posaff\n"
      "w3posaff ~ w2posaff\n"
      "w2unw ~ w1unw\n"
      "w3unw ~ w2unw",
      opts);

  auto count_cov = [&](std::string_view a, std::string_view b) {
    int n = 0;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] != Op::Covariance) continue;
      if ((pt.lhs[i] == a && pt.rhs[i] == b) ||
          (pt.lhs[i] == b && pt.rhs[i] == a)) ++n;
    }
    return n;
  };

  // Terminal pair: auto-added (1 row, free).
  CHECK(count_cov("w3posaff", "w3unw") == 1);
  // Non-terminal pair: must NOT be auto-added.
  CHECK(count_cov("w2posaff", "w2unw") == 0);
  // Cross-wave pairs that are also regressions are guarded out by
  // `regression_already_present`.
  CHECK(count_cov("w2posaff", "w3posaff") == 0);
  CHECK(count_cov("w2unw", "w3unw") == 0);
}

TEST_CASE("lavaanify: auto_cov_y respects explicit user covariance row") {
  // When the user writes `w3posaff ~~ 0*w3unw` to constrain the auto-row,
  // magmaan must NOT add a duplicate free auto-row (same parity as lavaan).
  BuildOptions opts;
  opts.meanstructure = true;
  opts.auto_cov_y = true;
  auto pt = must_lavaanify(
      "w2posaff ~ w1unw\n"
      "w3posaff ~ w2unw\n"
      "w3unw ~ w2posaff\n"
      "w2unw ~ w1posaff\n"
      "w3posaff ~~ 0*w3unw",
      opts);
  int n = 0, free_sum = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Covariance) continue;
    if ((pt.lhs[i] == "w3posaff" && pt.rhs[i] == "w3unw") ||
        (pt.lhs[i] == "w3unw" && pt.rhs[i] == "w3posaff")) {
      ++n;
      free_sum += pt.free[i];
    }
  }
  CHECK(n == 1);          // not duplicated
  CHECK(free_sum == 0);   // user fixed it at 0
}

TEST_CASE("lavaanify: meanstructure does not duplicate user-supplied ~1 rows") {
  BuildOptions opts;
  opts.meanstructure = true;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3\nx1 ~ 1\nf ~ 1", opts);

  // x1 ~ 1 was user-supplied → keep as user row (free).
  // f ~ 1 was user-supplied → user wants α free (no auto-fix).
  // x2 ~ 1 and x3 ~ 1 are auto-added free.
  int x1_count = 0, f_count = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Intercept) continue;
    if (pt.lhs[i] == "x1") {
      ++x1_count;
      CHECK(pt.free[i] > 0);   // user free, not auto-fixed
    }
    if (pt.lhs[i] == "f") {
      ++f_count;
      CHECK(pt.free[i] > 0);   // user explicitly freed
    }
  }
  CHECK(x1_count == 1);   // not duplicated
  CHECK(f_count  == 1);
}

TEST_CASE("lavaanify: bare numeric intercept shorthand fixes the row") {
  BuildOptions opts;
  opts.meanstructure = true;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3\nx1 ~ 0\nf ~ 0", opts);

  int x1_count = 0, f_count = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Intercept) continue;
    if (pt.lhs[i] == "x1") {
      ++x1_count;
      CHECK(pt.free[i] == 0);
      CHECK(pt.ustart[i] == doctest::Approx(0.0));
      CHECK(pt.user[i] == 1);
    }
    if (pt.lhs[i] == "f") {
      ++f_count;
      CHECK(pt.free[i] == 0);
      CHECK(pt.ustart[i] == doctest::Approx(0.0));
      CHECK(pt.user[i] == 1);
    }
  }
  CHECK(x1_count == 1);
  CHECK(f_count == 1);
}

TEST_CASE("lavaanify: c(...) modifier with wrong arity is rejected") {
  // n_groups defaults to 1, so a 2-atom c() doesn't match.
  auto fp = Parser::parse("f =~ c(1, 1)*x1 + c(NA, NA)*x2");
  REQUIRE(fp.has_value());
  auto pt = build(*fp);
  REQUIRE_FALSE(pt.has_value());
  CHECK(pt.error().kind == PartableError::Kind::BadGroupSpec);
}

TEST_CASE("lavaanify: c(...) modifier indexes per group with n_groups = 2") {
  // c(1, NA) on the marker means: group 1's λ_1 fixed at 1, group 2's free.
  // c(NA, NA) on x2's loading means: both groups free.
  BuildOptions opts;  opts.n_groups = 2;
  auto pt = must_lavaanify("f =~ c(1, NA)*x1 + c(NA, NA)*x2 + x3", opts);

  // Find the two `f =~ x1` rows (one per block) and confirm their fixed-ness.
  int group1_x1_free = -1, group2_x1_free = -1;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.lhs[i] == "f" && pt.rhs[i] == "x1") {
      if (pt.block[i] == 1) group1_x1_free = pt.free[i];
      if (pt.block[i] == 2) group2_x1_free = pt.free[i];
    }
  }
  REQUIRE(group1_x1_free >= 0);
  REQUIRE(group2_x1_free >= 0);
  CHECK(group1_x1_free == 0);   // group 1's marker fixed via c(1, NA)
  CHECK(group2_x1_free  > 0);   // group 2's loading free via NA
}

TEST_CASE("lavaanify: n_groups = 2 replicates rows with block / group set") {
  BuildOptions opts;
  opts.n_groups = 2;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);

  // Each group gets the same template (3 =~ rows + 3 auto.var Θ + 1 auto.var Ψ
  // = 7 rows). Two groups → 14 rows total. No constraint rows for this model.
  CHECK(pt.size() == 14);
  std::int32_t n_b1 = 0, n_b2 = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.block[i] == 1) ++n_b1;
    if (pt.block[i] == 2) ++n_b2;
  }
  CHECK(n_b1 == 7);
  CHECK(n_b2 == 7);

  // Each group's marker indicator (λ_1) is auto-fixed; the other 6 rows
  // per group are free. Total n_free = 12 (no cross-group equality since
  // no labels were used).
  CHECK(pt.n_free() == 12);
}

TEST_CASE("lavaanify: n_groups < 1 is rejected") {
  auto fp = Parser::parse("f =~ x1 + x2");
  REQUIRE(fp.has_value());
  BuildOptions opts;
  opts.n_groups = 0;
  auto pt = build(*fp, opts);
  REQUIRE_FALSE(pt.has_value());
  CHECK(pt.error().kind == PartableError::Kind::BadGroupSpec);
}

TEST_CASE("lavaanify: group identity rides on LatentNames / the lavaan view") {
  // Single group: n_groups() == 1, no group_var / group_labels.
  {
    auto pt = must_lavaanify("f =~ x1 + x2 + x3");
    CHECK(pt.n_groups() == 1);
    CHECK(pt.group_var.empty());
    CHECK(pt.group_labels.empty());
  }
  // Two groups, named, with explicit labels: round-tripped verbatim.
  {
    BuildOptions opts;
    opts.n_groups     = 2;
    opts.group_var    = "school";
    opts.group_labels = {"Pasteur", "Grant-White"};
    auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);
    CHECK(pt.n_groups() == 2);
    CHECK(pt.group_var == "school");
    REQUIRE(pt.group_labels.size() == 2);
    CHECK(pt.group_labels[0] == "Pasteur");
    CHECK(pt.group_labels[1] == "Grant-White");
  }
  // Two groups, name omitted ⇒ defaults to "group"; labels auto-generated.
  {
    BuildOptions opts;  opts.n_groups = 2;
    auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);
    CHECK(pt.n_groups() == 2);
    CHECK(pt.group_var == "group");
    REQUIRE(pt.group_labels.size() == 2);
    CHECK(pt.group_labels[0] == "1");
    CHECK(pt.group_labels[1] == "2");
  }
  // group_labels with wrong arity is rejected.
  {
    auto fp = Parser::parse("f =~ x1 + x2 + x3");
    REQUIRE(fp.has_value());
    BuildOptions opts;  opts.n_groups = 2;  opts.group_labels = {"only-one"};
    auto pt = build(*fp, opts);
    REQUIRE_FALSE(pt.has_value());
    CHECK(pt.error().kind == PartableError::Kind::BadGroupSpec);
  }
}

TEST_CASE("lavaanify: c(...) supplies per-group fixed values") {
  // c(0.8, 1.2) on x2's loading ⇒ group-1 loading fixed at 0.8, group-2 at
  // 1.2 (both rows fixed, distinct values).
  BuildOptions opts;  opts.n_groups = 2;  opts.group_var = "school";
  auto pt = must_lavaanify("f =~ x1 + c(0.8, 1.2)*x2 + x3", opts);
  double g1 = std::nan(""), g2 = std::nan("");
  std::int32_t f1 = -1, f2 = -1;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.lhs[i] == "f" && pt.rhs[i] == "x2") {
      if (pt.block[i] == 1) { g1 = pt.ustart[i]; f1 = pt.free[i]; }
      if (pt.block[i] == 2) { g2 = pt.ustart[i]; f2 = pt.free[i]; }
    }
  }
  REQUIRE(!std::isnan(g1));
  REQUIRE(!std::isnan(g2));
  CHECK(g1 == doctest::Approx(0.8));
  CHECK(g2 == doctest::Approx(1.2));
  CHECK(f1 == 0);   // fixed in group 1
  CHECK(f2 == 0);   // fixed in group 2
}

TEST_CASE("lavaanify: repeated `lhs op rhs` term merges modifiers into one row") {
  // `f =~ NA*x1 + a*x1` names the same loading cell twice: `NA*` frees it,
  // `a*` labels it. lavaan merges the two terms into a single parameter; so
  // must we. Emitting two rows would put two free parameters on one matrix
  // cell — the assemble step overwrites one, leaving a phantom parameter that
  // moves nothing yet carries a nonzero analytic gradient (TODO.md item #1).
  auto pt = must_lavaanify("f =~ NA*x1 + a*x1 + x2 + x3");
  int loading_x1_rows = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.lhs[i] == "f" && pt.op[i] == Op::Measurement && pt.rhs[i] == "x1") {
      ++loading_x1_rows;
      CHECK(pt.free[i] > 0);          // `NA*` freed it (no auto.fix.first)
      CHECK(pt.label[i] == "a");      // `a*` labelled it
    }
  }
  CHECK(loading_x1_rows == 1);
  // The marker is free, so x2 is *not* auto-fixed — x1/x2/x3 loadings all free.
  CHECK(pt.n_free() == 7);            // 3 loadings + 3 resid var + 1 factor var
}

TEST_CASE("lavaanify: repeated term merges per-group modifiers (NA + c(...))") {
  // The cross-group measurement-invariance idiom: `NA*x1 + c(a1,a2)*x1` frees
  // the marker and gives it a per-group label. Each group must end with one
  // free, labelled loading row for that cell — not two.
  BuildOptions opts;  opts.n_groups = 2;  opts.group_var = "grp";
  auto pt = must_lavaanify("f =~ NA*x1 + c(a1,a2)*x1 + x2 + x3", opts);
  int g1 = 0, g2 = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.lhs[i] != "f" || pt.op[i] != Op::Measurement || pt.rhs[i] != "x1")
      continue;
    CHECK(pt.free[i] > 0);
    if (pt.block[i] == 1) { ++g1; CHECK(pt.label[i] == "a1"); }
    if (pt.block[i] == 2) { ++g2; CHECK(pt.label[i] == "a2"); }
  }
  CHECK(g1 == 1);
  CHECK(g2 == 1);
}

TEST_CASE("lavaanify: empty model rejected") {
  // An empty parse would actually fail at the parser level, so we craft a
  // FlatPartable manually with no rows and no constraints.
  magmaan::parse::FlatPartable fp;
  auto pt = build(fp);
  REQUIRE_FALSE(pt.has_value());
  CHECK(pt.error().kind == PartableError::Kind::EmptyModel);
}

TEST_CASE("lavaanify: 3-factor CFA produces auto-covariances among latents") {
  auto pt = must_lavaanify("visual =~ x1 + x2 + x3\n"
                           "textual =~ x4 + x5 + x6\n"
                           "speed =~ x7 + x8 + x9");
  // Look for visual ~~ textual, visual ~~ speed, textual ~~ speed
  int lv_cov_count = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Covariance) continue;
    if (pt.lhs[i] == "visual"  && pt.rhs[i] == "textual") ++lv_cov_count;
    if (pt.lhs[i] == "visual"  && pt.rhs[i] == "speed")   ++lv_cov_count;
    if (pt.lhs[i] == "textual" && pt.rhs[i] == "speed")   ++lv_cov_count;
  }
  CHECK(lv_cov_count == 3);
}

TEST_CASE("lavaanify: orthogonal fixes the auto latent covariances at 0") {
  // lavaan cfa(..., orthogonal = TRUE): the latent-covariance rows are still
  // emitted but pinned (free = 0, ustart = 0).
  BuildOptions opts;
  opts.orthogonal = true;
  auto pt = must_lavaanify("visual =~ x1 + x2 + x3\n"
                           "textual =~ x4 + x5 + x6\n"
                           "speed =~ x7 + x8 + x9", opts);
  int lv_cov_count = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Covariance) continue;
    const bool is_lv_cov =
        (pt.lhs[i] == "visual"  && pt.rhs[i] == "textual") ||
        (pt.lhs[i] == "visual"  && pt.rhs[i] == "speed")   ||
        (pt.lhs[i] == "textual" && pt.rhs[i] == "speed");
    if (!is_lv_cov) continue;
    ++lv_cov_count;
    CHECK(pt.free[i] == 0);       // pinned, not estimated
    CHECK(pt.ustart[i] == 0.0);   // fixed at 0
  }
  CHECK(lv_cov_count == 3);       // rows still present, mirroring lavaan
  // 6 free loadings + 9 residual var + 3 latent var = 18 free parameters
  // (the 3 latent covariances are no longer free).
  int n_free = 0;
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (pt.free[i] > n_free) n_free = pt.free[i];
  CHECK(n_free == 18);
}

TEST_CASE("lavaanify: a user-written exogenous-latent covariance is not auto-duplicated") {
  // `visual ~~ cvt*textual` is also a pair auto.cov.lv.x would generate. The
  // user row must win — a duplicate auto row would be a second free parameter
  // for the same moment, leaving the information matrix rank-deficient.
  auto pt = must_lavaanify("visual =~ x1 + x2 + x3\n"
                           "textual =~ x4 + x5 + x6\n"
                           "speed =~ x7 + x8 + x9\n"
                           "visual ~~ cvt*textual");
  int vt_count = 0;
  int vt_idx = -1;
  int other_cov = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != Op::Covariance) continue;
    if ((pt.lhs[i] == "visual"  && pt.rhs[i] == "textual") ||
        (pt.lhs[i] == "textual" && pt.rhs[i] == "visual")) {
      ++vt_count;
      vt_idx = static_cast<int>(i);
    }
    if (pt.lhs[i] == "visual"  && pt.rhs[i] == "speed") ++other_cov;
    if (pt.lhs[i] == "textual" && pt.rhs[i] == "speed") ++other_cov;
  }
  CHECK(vt_count == 1);                            // not duplicated
  REQUIRE(vt_idx >= 0);
  CHECK(pt.label[static_cast<std::size_t>(vt_idx)] == "cvt");  // the user row
  CHECK(pt.user[static_cast<std::size_t>(vt_idx)] == 1);
  CHECK(pt.free[static_cast<std::size_t>(vt_idx)] > 0);
  CHECK(other_cov == 2);                           // the other pairs still auto-added

  // 6 free loadings + 9 residual var + 3 latent var + 3 latent cov = 21 free
  // parameters — no phantom 22nd from a duplicate covariance.
  int n_free = 0;
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (pt.free[i] > n_free) n_free = pt.free[i];
  CHECK(n_free == 21);
}
