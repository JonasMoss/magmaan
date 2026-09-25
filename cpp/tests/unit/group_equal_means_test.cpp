#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <vector>

#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// lavaan's group.equal = "intercepts" frees the latent means in groups 2+,
// because equal indicator intercepts identify them. The expectations below
// were read from lavaan 0.7.2 parTable() output for the same calls.

using magmaan::compat::lavaan::LavaanParTable;
using magmaan::compat::lavaan::to_lavaan_partable;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::GroupEqual;
using magmaan::spec::LatentNames;
using magmaan::spec::Starts;

namespace {

constexpr std::string_view kCfa = "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6";

LavaanParTable build_pt(std::string_view syntax, std::vector<GroupEqual> equal,
                        std::vector<std::string> partial = {}) {
  auto parsed = Parser::parse(syntax);
  REQUIRE(parsed.has_value());
  BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = 2;
  opts.group_equal = std::move(equal);
  opts.group_partial = std::move(partial);
  Starts starts;
  LatentNames names;
  auto built = magmaan::spec::build(*parsed, opts, &starts, &names);
  REQUIRE(built.has_value());
  return to_lavaan_partable(*built, names, starts);
}

// "fixed" or "free" for the latent mean of `lv` in `group`.
std::string lv_mean(const LavaanParTable& pt, const std::string& lv, int group) {
  for (std::size_t i = 0; i < pt.lhs.size(); ++i) {
    if (pt.op[i] == Op::Intercept && pt.lhs[i] == lv && pt.group[i] == group) {
      return pt.free[i] > 0 ? "free" : "fixed";
    }
  }
  return "missing";
}

}  // namespace

TEST_CASE("group.equal intercepts frees the latent means in later groups") {
  auto pt = build_pt(kCfa, {GroupEqual::Loadings, GroupEqual::Intercepts});
  CHECK(lv_mean(pt, "visual", 1) == "fixed");
  CHECK(lv_mean(pt, "textual", 1) == "fixed");
  CHECK(lv_mean(pt, "visual", 2) == "free");
  CHECK(lv_mean(pt, "textual", 2) == "free");
  // Intercepts alone, without equal loadings, releases them too.
  auto alone = build_pt(kCfa, {GroupEqual::Intercepts});
  CHECK(lv_mean(alone, "visual", 2) == "free");
}

TEST_CASE("equal means, user-fixed means and loadings alone keep them fixed") {
  auto means = build_pt(kCfa, {GroupEqual::Loadings, GroupEqual::Intercepts,
                               GroupEqual::Means});
  CHECK(lv_mean(means, "visual", 2) == "fixed");
  CHECK(lv_mean(means, "textual", 2) == "fixed");
  auto loadings = build_pt(kCfa, {GroupEqual::Loadings});
  CHECK(lv_mean(loadings, "visual", 2) == "fixed");
  auto user = build_pt(std::string(kCfa) + "\nvisual ~ 0*1",
                       {GroupEqual::Loadings, GroupEqual::Intercepts});
  CHECK(lv_mean(user, "visual", 2) == "fixed");
  CHECK(lv_mean(user, "textual", 2) == "free");
}

TEST_CASE("group.partial on an intercept does not change the mean release") {
  auto pt = build_pt(kCfa, {GroupEqual::Loadings, GroupEqual::Intercepts}, {"x1~1"});
  CHECK(lv_mean(pt, "visual", 2) == "free");
  CHECK(lv_mean(pt, "textual", 2) == "free");
}
