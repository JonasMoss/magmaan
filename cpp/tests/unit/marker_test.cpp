#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <cctype>
#include <unordered_map>
#include "magmaan/estimate/marker.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/parse/parser.hpp"

namespace {
// A builder-generated label: prefix, digits, closing dot (".p3.", ".eqg2.").
bool generated_label(const std::string& s, const std::string& prefix) {
  if (s.size() <= prefix.size() + 1 || s.rfind(prefix, 0) != 0 || s.back() != '.') return false;
  for (std::size_t i = prefix.size(); i + 1 < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
  return true;
}
} // namespace

TEST_CASE("lavaan marker adaptation and builder match frozen 0.7.2 components") {
  using namespace magmaan;
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) + "/fitting/lavaan_marker_0_7_2.json");
  REQUIRE(in.good()); if (!in.good()) return;
  auto root = nlohmann::json::parse(in, nullptr, false);
  REQUIRE(!root.is_discarded()); if (root.is_discarded()) return;
  for (const auto& family : {"rule", "build"}) for (const auto& c : root[family]) {
    CAPTURE(c["name"]);
    auto flat = parse::Parser::parse(c["syntax"].get<std::string>());
    REQUIRE(flat); if (!flat) return;
    spec::BuildOptions options;
    options.n_groups = c["n_groups"].get<int>();
    if (!c["group_equal"].is_null()) options.group_equal = {spec::GroupEqual::Loadings};
    if (!c["marker"].is_null())
      for (auto it = c["marker"].begin(); it != c["marker"].end(); ++it)
        options.marker[it.key()] = it.value().get<std::string>();
    spec::Starts hints;
    spec::LatentNames names;
    auto model = spec::build(*flat, options, &hints, &names);
    REQUIRE(model); if (!model) return;
    auto pt = compat::lavaan::to_lavaan_partable(*model, names, hints);
    if (std::string(family) == "build") {
      REQUIRE(pt.size() == c["pt"].size()); if (pt.size() != c["pt"].size()) return;
      // magmaan names generated group-equality labels .eqgN.; lavaan reuses the
      // first group's plabel. Both synthesize the same plabel == rows, so the
      // generated labels must correspond one to one; all other labels match.
      std::unordered_map<std::string, std::string> to_magmaan, to_lavaan;
      for (std::size_t i = 0; i < pt.size(); ++i) {
        CAPTURE(i);
        const auto& r = c["pt"][i];
        CHECK(pt.lhs[i] == r["lhs"].get<std::string>());
        CHECK(pt.rhs[i] == r["rhs"].get<std::string>());
        CHECK(parse::to_string(pt.op[i]) == r["op"].get<std::string>());
        CHECK(pt.free[i] == r["free"].get<int>());
        const auto expected = r["label"].get<std::string>();
        if (generated_label(expected, ".p")) {
          CHECK(generated_label(pt.label[i], ".eqg"));
          CHECK(to_magmaan.try_emplace(expected, pt.label[i]).first->second == pt.label[i]);
          CHECK(to_lavaan.try_emplace(pt.label[i], expected).first->second == expected);
        } else {
          CHECK(pt.label[i] == expected);
        }
        CHECK(pt.plabel[i] == r["plabel"].get<std::string>());
        if (r["ustart"].is_null()) CHECK(std::isnan(pt.ustart[i]));
        else CHECK(pt.ustart[i] == doctest::Approx(r["ustart"].get<double>()));
      }
    } else {
      std::vector<Eigen::MatrixXd> cov;
      for (const auto& block : c["cov"]) {
        const auto n = static_cast<Eigen::Index>(block.size());
        Eigen::MatrixXd s(n, n);
        for (Eigen::Index i = 0; i < s.rows(); ++i) for (Eigen::Index j = 0; j < s.cols(); ++j)
          s(i,j) = block[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get<double>();
        cov.push_back(s);
      }
      auto result = estimate::lavaan_marker_adapt(pt, cov,
          c["names"].get<std::vector<std::vector<std::string>>>());
      REQUIRE(result); if (!result) return;
      CHECK(result->info.size() == c["info"].size());
      for (std::size_t i = 0; i < std::min(result->info.size(), c["info"].size()); ++i) {
        const auto& a = result->info[i]; const auto& b = c["info"][i];
        CHECK(a.lv == b["lv"].get<std::string>());
        CHECK(a.old_marker == b["old"].get<std::string>());
        CHECK(a.new_marker == b["new"].get<std::string>());
        CHECK(a.r_old == doctest::Approx(b["r_old"].get<double>()));
        CHECK(a.r_new == doctest::Approx(b["r_new"].get<double>()));
      }
    }
  }
}
