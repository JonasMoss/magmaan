#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/compat/eqs/model.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/eqs_parser.hpp"

namespace {
std::string key(std::string lhs, std::string op, std::string rhs) {
  if (op == "~~" && rhs < lhs) std::swap(lhs, rhs);
  return lhs + "|" + op + "|" + rhs;
}
Eigen::MatrixXd matrix(const nlohmann::json& data) {
  Eigen::MatrixXd result(static_cast<Eigen::Index>(data.size()),
                         static_cast<Eigen::Index>(data[0].size()));
  for (Eigen::Index r = 0; r < result.rows(); ++r)
    for (Eigen::Index c = 0; c < result.cols(); ++c)
      result(r,c) = data[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>();
  return result;
}
}

TEST_CASE("EQS: model rows, starts, ML estimates, implied covariance and inference match independent lavaan fixtures") {
  auto raw = magmaan::test::read_fixture(magmaan::test::fixtures_dir() + "/eqs.json");
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());
  for (const auto& test : fixture["cases"]) {
    CAPTURE(test["id"]);
    auto flat = magmaan::parse::EqsParser::parse(test["eqs"].get<std::string>());
    REQUIRE_MESSAGE(flat.has_value(), (flat ? "" : flat.error().detail));
    magmaan::spec::LatentNames names;
    magmaan::spec::Starts starts;
    auto structure = magmaan::spec::build(*flat, magmaan::compat::eqs::build_options(), &starts, &names);
    REQUIRE(structure.has_value());
    auto pt = magmaan::compat::lavaan::to_lavaan_partable(*structure, names, starts);
    std::map<std::string, nlohmann::json> expected;
    for (const auto& row : test["rows"])
      expected.emplace(key(row["lhs"],row["op"],row["rhs"]),row);
    REQUIRE(pt.size() == expected.size());
    for (std::size_t i = 0; i < pt.size(); ++i) {
      const auto k = key(pt.lhs[i],std::string(magmaan::parse::to_string(pt.op[i])),pt.rhs[i]);
      CAPTURE(k);
      REQUIRE(expected.contains(k));
      const auto& row = expected.at(k);
      CHECK((pt.free[i] > 0) == row["free"].get<bool>());
      if (row["ustart"].is_null()) CHECK(std::isnan(pt.ustart[i]));
      else CHECK(pt.ustart[i] == doctest::Approx(row["ustart"].get<double>()).epsilon(1e-12));
    }
    auto rep = magmaan::model::build_matrix_rep(*structure, &names);
    REQUIRE(rep.has_value());
    // These hand-specified models enumerate V1..Vp in observation order.
    magmaan::data::SampleStats sample;
    sample.S.push_back(matrix(test["sample_cov"]));
    sample.n_obs.push_back(test["n"].get<int>());
    auto start = magmaan::estimate::simple_start_values(*structure,*rep,sample,starts);
    REQUIRE(start.has_value());
    auto fit = magmaan::estimate::fit_ml(*structure,*rep,sample,*start);
    REQUIRE_MESSAGE(fit.has_value(), (fit ? "" : fit.error().detail));
    auto info = magmaan::inference::information_expected(*structure,*rep,sample,*fit);
    REQUIRE(info.has_value());
    auto vcov = magmaan::inference::vcov(*info,*structure);
    REQUIRE(vcov.has_value());
    auto se = magmaan::inference::se(*vcov);
    for (std::size_t i = 0; i < pt.size(); ++i) {
      const auto k = key(pt.lhs[i],std::string(magmaan::parse::to_string(pt.op[i])),pt.rhs[i]);
      CAPTURE(k);
      const auto& row = expected.at(k);
      if (pt.free[i] > 0) {
        const auto index = pt.free[i] - 1;
        CHECK(fit->theta[index] == doctest::Approx(row["est"].get<double>()).epsilon(1e-5).scale(1));
        CHECK(se[index] == doctest::Approx(row["se"].get<double>()).epsilon(1e-5).scale(1));
      }
    }
    auto evaluator = magmaan::model::ModelEvaluator::build(*structure,*rep);
    REQUIRE(evaluator.has_value());
    auto implied = evaluator->sigma(fit->theta);
    REQUIRE(implied.has_value());
    CHECK((implied->sigma[0] - matrix(test["implied"])).cwiseAbs().maxCoeff() < 1e-5);
    auto df = magmaan::inference::df_stat(*structure,sample);
    REQUIRE(df.has_value());
    CHECK(*df == test["df"].get<int>());
    CHECK(magmaan::inference::chi2_stat(sample,*fit) ==
          doctest::Approx(test["chisq"].get<double>()).epsilon(1e-5).scale(1));
  }
}
