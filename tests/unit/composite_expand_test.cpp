#include <doctest/doctest.h>

#include <cstddef>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/error.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::PartableError;
using magmaan::compat::lavaan::LavaanParTable;
using magmaan::compat::lavaan::to_lavaan_partable;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::BuildOptions;
using magmaan::spec::CompositeMode;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;
using magmaan::spec::Starts;

namespace {

struct Built {
  LavaanParTable pt;
  LatentStructure s;
  LatentNames    names;
};

BuildOptions ho_options() {
  BuildOptions opts;
  opts.composite_mode = CompositeMode::HenselerOgasawara;
  return opts;
}

// Parse + build a composite model, asserting success, and bundle the
// lavaan-shaped projection with the LatentNames (which carries `composites`).
Built must_build(std::string_view src, BuildOptions opts) {
  auto fp = Parser::parse(src);
  REQUIRE_MESSAGE(fp.has_value(), "parser failed: " << fp.error().detail);
  Starts starts;
  LatentNames names;
  auto pt = build(*fp, opts, &starts, &names);
  REQUIRE_MESSAGE(pt.has_value(),
                  "build failed: kind=" << static_cast<int>(pt.error().kind)
                                        << " — " << pt.error().detail);
  LatentStructure s = *pt;
  return Built{to_lavaan_partable(*pt, names, starts), std::move(s),
               std::move(names)};
}

Built must_build(std::string_view src) {
  return must_build(src, ho_options());
}

PartableError::Kind build_error(std::string_view src, BuildOptions opts) {
  auto fp = Parser::parse(src);
  REQUIRE_MESSAGE(fp.has_value(), "parser failed: " << fp.error().detail);
  auto pt = build(*fp, opts);
  REQUIRE_FALSE(pt.has_value());
  return pt.error().kind;
}

PartableError::Kind build_error(std::string_view src) {
  return build_error(src, ho_options());
}

std::size_t count_free(const LavaanParTable& pt) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (pt.free[i] > 0) ++n;
  return n;
}

std::size_t count_op(const LavaanParTable& pt, Op op) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (pt.op[i] == op) ++n;
  return n;
}

}  // namespace

TEST_CASE("composite: <~ requires an explicit composite mode") {
  CHECK(build_error("C <~ x1 + x2 + x3\n y ~ C", BuildOptions{}) ==
        PartableError::Kind::CompositeModeRequired);
}

TEST_CASE("composite: C <~ x1+x2+x3 expands to a Henseler-Ogasawara block") {
  // `y ~ C` connects the composite to the structural model — required for
  // identification (see the connectivity test below).
  const Built b = must_build("C <~ x1 + x2 + x3\n y ~ C");

  // 9 loadings (3×3 block) + 3 zero residuals + 3 latent variances
  // + 3 orthogonality covariances, then `y ~ C` and the auto `y ~~ y`.
  CHECK(b.pt.size() == 20);
  CHECK(count_op(b.pt, Op::Measurement) == 9);
  CHECK(count_op(b.pt, Op::Regression) == 1);
  // 5 free loadings + 3 free latent variances + `y ~ C` + `y ~~ y`.
  CHECK(count_free(b.pt) == 10);

  // The emergent column is scaled by fixing its last loading (C =~ x3) to 1;
  // the other emergent loadings stay free.
  CHECK(b.pt.lhs[0] == "C");
  CHECK(b.pt.op[0] == Op::Measurement);
  CHECK(b.pt.rhs[0] == "x1");
  CHECK(b.pt.free[0] > 0);
  CHECK(b.pt.lhs[2] == "C");
  CHECK(b.pt.rhs[2] == "x3");
  CHECK(b.pt.free[2] == 0);
  CHECK(b.pt.ustart[2] == 1.0);

  // Indicator residuals are fixed to 0 — a composite is an exact weighted sum.
  CHECK(b.pt.lhs[9] == "x1");
  CHECK(b.pt.op[9] == Op::Covariance);
  CHECK(b.pt.rhs[9] == "x1");
  CHECK(b.pt.free[9] == 0);
  CHECK(b.pt.ustart[9] == 0.0);

  // CompositeInfo metadata.
  REQUIRE(b.names.composites.size() == 1);
  CHECK(b.names.composites[0].composite == "C");
  CHECK(b.names.composites[0].indicators ==
        std::vector<std::string>{"x1", "x2", "x3"});
  REQUIRE(b.names.composites[0].excrescent.size() == 2);
  CHECK(b.names.composites[0].excrescent[0].rfind(".exc.", 0) == 0);
}

TEST_CASE("composite: K=2 has no cascading zeros") {
  const Built b = must_build("C <~ x1 + x2\n y ~ C");
  // 4 loadings + 2 zero residuals + 2 variances + 1 orthogonality,
  // then `y ~ C` and the auto `y ~~ y`.
  CHECK(b.pt.size() == 11);
  CHECK(count_op(b.pt, Op::Measurement) == 4);
  // 2 free loadings + 2 free latent variances + `y ~ C` + `y ~~ y`.
  CHECK(count_free(b.pt) == 6);
  REQUIRE(b.names.composites.size() == 1);
  CHECK(b.names.composites[0].excrescent.size() == 1);
}

TEST_CASE("composite: emergent latent connects to the structural model") {
  // A composite predicting an observed outcome must build (Reduced form).
  const Built b = must_build("C <~ x1 + x2 + x3\n y ~ C");
  CHECK(count_op(b.pt, Op::Regression) == 1);
  REQUIRE(b.names.composites.size() == 1);
}

TEST_CASE("composite: an unconnected composite is rejected") {
  // A standalone composite has no `~`/`~~` link to an external variable.
  CHECK(build_error("C <~ x1 + x2 + x3") ==
        PartableError::Kind::UnidentifiedComposite);
  // A relation only to one of its own indicators does not count.
  CHECK(build_error("C <~ x1 + x2 + x3\n C ~ x1") ==
        PartableError::Kind::UnidentifiedComposite);
}

TEST_CASE("composite: fewer than 2 indicators is rejected") {
  CHECK(build_error("C <~ x1") ==
        PartableError::Kind::CompositeTooFewIndicators);
}

TEST_CASE("composite: overlapping indicator sets are rejected") {
  CHECK(build_error("C <~ x1 + x2\n D <~ x2 + x3") ==
        PartableError::Kind::CompositeOverlap);
}

TEST_CASE("composite: a name cannot be both composite and factor") {
  CHECK(build_error("C <~ x1 + x2\n C =~ y1 + y2") ==
        PartableError::Kind::CompositeOverlap);
}

TEST_CASE("fc-sem composite: native mode preserves <~ rows and T metadata") {
  BuildOptions opts;
  opts.composite_mode = CompositeMode::FcSem;
  const Built b = must_build("C <~ x1 + x2 + x3\n y ~ C", opts);

  CHECK(b.s.composite_mode == CompositeMode::FcSem);
  CHECK(b.names.composite_mode == CompositeMode::FcSem);
  REQUIRE(b.names.composites.size() == 1);
  CHECK(b.names.composites[0].composite == "C");
  CHECK(b.names.composites[0].indicators ==
        std::vector<std::string>{"x1", "x2", "x3"});
  CHECK(b.names.composites[0].excrescent.empty());
  REQUIRE(b.s.composite_blocks.size() == 1);
  CHECK(b.s.composite_blocks[0].indicator_vars.size() == 3);

  CHECK(count_op(b.pt, Op::Composite) == 3);
  CHECK(count_op(b.pt, Op::Measurement) == 0);

  CHECK(b.pt.lhs[0] == "C");
  CHECK(b.pt.op[0] == Op::Composite);
  CHECK(b.pt.rhs[0] == "x1");
  CHECK(b.pt.free[0] == 0);
  CHECK(b.pt.ustart[0] == 1.0);
  CHECK(b.pt.free[1] > 0);
  CHECK(b.pt.free[2] > 0);

  std::size_t fixed_t = 0;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const bool is_t_row =
        b.pt.op[i] == Op::Covariance &&
        ((b.pt.lhs[i] == "x1" || b.pt.lhs[i] == "x2" ||
          b.pt.lhs[i] == "x3") &&
         (b.pt.rhs[i] == "x1" || b.pt.rhs[i] == "x2" ||
          b.pt.rhs[i] == "x3"));
    if (!is_t_row) continue;
    CHECK(b.pt.free[i] == 0);
    CHECK(std::isnan(b.pt.ustart[i]));
    ++fixed_t;
  }
  CHECK(fixed_t == 6);
}

TEST_CASE("fc-sem composite: native mode auto-adds endogenous observed covariance") {
  BuildOptions opts;
  opts.composite_mode = CompositeMode::FcSem;
  const Built b = must_build("C <~ x1 + x2 + x3\n x4 ~ C\n x5 ~ C", opts);

  bool found = false;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    if (b.pt.op[i] == Op::Covariance && b.pt.lhs[i] == "x4" &&
        b.pt.rhs[i] == "x5") {
      found = true;
      CHECK(b.pt.free[i] > 0);
      CHECK(b.pt.user[i] == 0);
    }
  }
  CHECK(found);
}

TEST_CASE("fc-sem composite: known identification checks run before evaluator work") {
  BuildOptions opts;
  opts.composite_mode = CompositeMode::FcSem;

  CHECK(build_error("C <~ x1", opts) ==
        PartableError::Kind::CompositeTooFewIndicators);
  CHECK(build_error("C <~ x1 + x2\n D <~ x2 + x3", opts) ==
        PartableError::Kind::CompositeOverlap);
  CHECK(build_error("C <~ x1 + x2 + x3", opts) ==
        PartableError::Kind::UnidentifiedComposite);
  CHECK(build_error("C <~ x1 + x2\n F =~ x1 + y1\n y ~ C", opts) ==
        PartableError::Kind::CompositeOverlap);
}
