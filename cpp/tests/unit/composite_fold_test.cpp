#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "magmaan/compat/lavaan/composite_fold.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::compat::lavaan::fold_composites;
using magmaan::compat::lavaan::LavaanParTable;
using magmaan::compat::lavaan::to_lavaan_partable;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::BuildOptions;
using magmaan::spec::CompositeMode;
using magmaan::spec::LatentNames;

namespace {

struct Folded {
  LavaanParTable full;
  LavaanParTable folded;
};

Folded must_fold(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE_MESSAGE(fp.has_value(), "parser failed: " << fp.error().detail);
  LatentNames names;
  BuildOptions opts;
  opts.composite_mode = CompositeMode::HenselerOgasawara;
  auto pt = build(*fp, opts, nullptr, &names);
  REQUIRE_MESSAGE(pt.has_value(), "build failed: " << pt.error().detail);
  LavaanParTable full = to_lavaan_partable(*pt, names);
  LavaanParTable folded = fold_composites(full, names.composites);
  return Folded{std::move(full), std::move(folded)};
}

bool mentions_excrescent(const LavaanParTable& t) {
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (t.lhs[i].rfind(".exc.", 0) == 0) return true;
    if (t.rhs[i].rfind(".exc.", 0) == 0) return true;
  }
  return false;
}

std::size_t count_op(const LavaanParTable& t, Op op) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < t.size(); ++i)
    if (t.op[i] == op) ++n;
  return n;
}

}  // namespace

TEST_CASE("fold_composites: an H-O block folds back to `<~` rows") {
  Folded f = must_fold("C <~ x1 + x2 + x3\n y ~ C");
  // Expanded: 9 loadings + 3 zero residuals + 3 latent variances
  // + 3 orthogonality + `y ~ C` + `y ~~ y` = 20.
  CHECK(f.full.size() == 20);
  CHECK(mentions_excrescent(f.full));  // sanity — excrescent latents present

  // Folded: drop 9 loadings + 2 excrescent variances + 3 orthogonality;
  // keep 3 zero residuals + `C ~~ C` + `y ~ C` + `y ~~ y` (6); add 3 `<~`.
  CHECK(f.folded.size() == 9);
  CHECK(count_op(f.folded, Op::Composite) == 3);
  CHECK(count_op(f.folded, Op::Measurement) == 0);   // H-O loadings are gone
  CHECK(count_op(f.folded, Op::Regression) == 1);    // y ~ C survives
  CHECK_FALSE(mentions_excrescent(f.folded));        // excrescent latents hidden

  // The `<~` rows are spliced where the loading block was — first three rows.
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(f.folded.op[i] == Op::Composite);
    CHECK(f.folded.lhs[i] == "C");
    CHECK(f.folded.free[i] == 0);  // weights are derived, not free parameters
  }
  CHECK(f.folded.rhs[0] == "x1");
  CHECK(f.folded.rhs[1] == "x2");
  CHECK(f.folded.rhs[2] == "x3");

  // ids are renumbered contiguously.
  for (std::size_t i = 0; i < f.folded.size(); ++i)
    CHECK(f.folded.id[i] == static_cast<std::int32_t>(i + 1));
}

TEST_CASE("fold_composites: a model with no composites is an unchanged copy") {
  Folded f = must_fold("f =~ x1 + x2 + x3");
  CHECK(f.folded.size() == f.full.size());
  CHECK(count_op(f.folded, Op::Composite) == 0);
}
