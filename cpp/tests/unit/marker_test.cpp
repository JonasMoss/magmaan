#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include "magmaan/estimate/marker.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/parse/parser.hpp"

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
      for (std::size_t i = 0; i < pt.size(); ++i) {
        CAPTURE(i);
        const auto& r = c["pt"][i];
        CHECK(pt.lhs[i] == r["lhs"].get<std::string>());
        CHECK(pt.rhs[i] == r["rhs"].get<std::string>());
        CHECK(parse::to_string(pt.op[i]) == r["op"].get<std::string>());
        CHECK(pt.free[i] == r["free"].get<int>());
        CHECK(pt.label[i] == r["label"].get<std::string>());
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
