#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

namespace {

constexpr const char* kCompositeFixtures[] = {
    "composite/0001_pure_composite_hs.fit.json",
    "composite/0002_composite_factor_hs.fit.json",
    "composite/0003_composite_structural_hs.fit.json",
};

using magmaan::data::SampleStats;
using magmaan::estimate::Estimates;
using magmaan::model::FcSemEvaluator;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::CompositeMode;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;
using magmaan::spec::Starts;
using magmaan::test::load_json_fixture;
using magmaan::test::matrix_from_json;

struct Built {
  LatentStructure pt;
  LatentNames names;
  Starts starts;
};

Built build_fcsem(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE_MESSAGE(fp.has_value(), "parse failed: " << fp.error().detail);

  BuildOptions opts;
  opts.composite_mode = CompositeMode::FcSem;
  Starts starts;
  LatentNames names;
  auto pt = magmaan::spec::build(*fp, opts, &starts, &names);
  REQUIRE_MESSAGE(pt.has_value(), "build failed: " << pt.error().detail);
  return Built{std::move(*pt), std::move(names), std::move(starts)};
}

std::vector<std::string> built_observed_order(const Built& b) {
  std::vector<std::string> out;
  out.reserve(b.pt.ov_order.size());
  for (const auto id : b.pt.ov_order) {
    REQUIRE(id >= 0);
    out.push_back(b.names.var_name[static_cast<std::size_t>(id)]);
  }
  return out;
}

std::vector<std::string> fixture_observed_order(const nlohmann::json& j,
                                                std::string_view key) {
  REQUIRE(j[std::string(key)].is_array());
  REQUIRE_FALSE(j[std::string(key)].empty());
  REQUIRE(j[std::string(key)][0].contains("names"));
  return j[std::string(key)][0]["names"].get<std::vector<std::string>>();
}

Eigen::MatrixXd reorder_matrix(const Eigen::MatrixXd& raw,
                               const std::vector<std::string>& source,
                               const std::vector<std::string>& target) {
  REQUIRE(raw.rows() == static_cast<Eigen::Index>(source.size()));
  REQUIRE(raw.cols() == static_cast<Eigen::Index>(source.size()));
  Eigen::MatrixXd out(target.size(), target.size());
  for (std::size_t r = 0; r < target.size(); ++r) {
    const auto sr = std::find(source.begin(), source.end(), target[r]);
    REQUIRE(sr != source.end());
    for (std::size_t c = 0; c < target.size(); ++c) {
      const auto sc = std::find(source.begin(), source.end(), target[c]);
      REQUIRE(sc != source.end());
      out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
          raw(static_cast<Eigen::Index>(std::distance(source.begin(), sr)),
              static_cast<Eigen::Index>(std::distance(source.begin(), sc)));
    }
  }
  return out;
}

Eigen::MatrixXd fixture_matrix_in_built_order(const Built& b,
                                              const nlohmann::json& j,
                                              std::string_view key) {
  return reorder_matrix(matrix_from_json(j[std::string(key)][0]["matrix"]),
                        fixture_observed_order(j, key),
                        built_observed_order(b));
}

SampleStats sample_stats_from_fixture(const Built& b, const nlohmann::json& j) {
  SampleStats samp;
  samp.S = {fixture_matrix_in_built_order(b, j, "sample_cov")};
  samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  return samp;
}

Op op_from_json(const std::string& op) {
  if (op == "<~") return Op::Composite;
  if (op == "=~") return Op::Measurement;
  if (op == "~") return Op::Regression;
  FAIL("unsupported fixture op: " << op);
  return Op::Regression;
}

bool close(double got, double expected, double tol) {
  return std::abs(got - expected) <= tol * std::max(1.0, std::abs(expected));
}

void check_close(std::string_view label, double got, double expected,
                 double tol) {
  INFO(label);
  CHECK_MESSAGE(close(got, expected, tol),
                label << ": got " << got << ", lavaan " << expected);
}

const magmaan::measures::standardize::FcSemStandardizedRow*
find_reported_row(
    const std::vector<magmaan::measures::standardize::FcSemStandardizedRow>&
        rows,
    const nlohmann::json& expected) {
  const std::string lhs = expected["lhs"].get<std::string>();
  const std::string rhs = expected["rhs"].get<std::string>();
  const Op op = op_from_json(expected["op"].get<std::string>());
  const std::int32_t group = expected.value("group", 1);
  const auto it = std::find_if(rows.begin(), rows.end(), [&](const auto& got) {
    return got.lhs == lhs && got.rhs == rhs && got.op == op &&
           got.group == group;
  });
  if (it == rows.end()) return nullptr;
  return &*it;
}

void check_reported_row(
    const std::vector<magmaan::measures::standardize::FcSemStandardizedRow>&
        rows,
    const nlohmann::json& expected) {
  const std::string label = expected["lhs"].get<std::string>() +
                            expected["op"].get<std::string>() +
                            expected["rhs"].get<std::string>();
  const auto* got = find_reported_row(rows, expected);
  REQUIRE_MESSAGE(got != nullptr, "missing reported row " << label);

  check_close(label + " est", got->est, expected["est"].get<double>(), 5e-4);
  check_close(label + " se", got->se, expected["se"].get<double>(), 4e-3);
  check_close(label + " std.lv", got->std_lv,
              expected["std_lv"].get<double>(), 8e-4);
  check_close(label + " std.lv.se", got->std_lv_se,
              expected["std_lv_se"].get<double>(), 4e-3);
  check_close(label + " std.all", got->std_all,
              expected["std_all"].get<double>(), 8e-4);
  check_close(label + " std.all.se", got->std_all_se,
              expected["std_all_se"].get<double>(), 4e-3);
}

Estimates fit_fixture(const Built& b, const SampleStats& samp) {
  auto x0 = magmaan::estimate::simple_fcsem_start_values(b.pt, samp);
  REQUIRE_MESSAGE(x0.has_value(), "starts failed: " << x0.error().detail);
  REQUIRE(x0->size() == b.pt.n_free());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 4000;
  auto est = magmaan::estimate::fit_ml_fcsem(
      b.pt, samp, *x0, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(est.has_value(), "fit failed: " << est.error().detail);
  return *est;
}

Eigen::MatrixXd vcov_fcsem(const Built& b, const SampleStats& samp,
                           const Estimates& est) {
  auto info = magmaan::inference::information_expected_fcsem(b.pt, samp, est);
  REQUIRE_MESSAGE(info.has_value(),
                  "information failed: " << info.error().detail);
  auto vc = magmaan::inference::vcov(*info, b.pt, est.theta);
  REQUIRE_MESSAGE(vc.has_value(), "vcov failed: " << vc.error().detail);
  return *vc;
}

}  // namespace

TEST_CASE("native FC-SEM composite ML goldens match lavaan public surface") {
  for (const char* fixture : kCompositeFixtures) {
    SUBCASE(fixture) {
      const auto j = load_json_fixture(fixture);
      REQUIRE_FALSE(j.is_discarded());

      const Built b = build_fcsem(j["input"].get<std::string>());
      const SampleStats samp = sample_stats_from_fixture(b, j);
      const Estimates est = fit_fixture(b, samp);
      const Eigen::MatrixXd vcov = vcov_fcsem(b, samp, est);

      auto ev = FcSemEvaluator::build(b.pt);
      REQUIRE_MESSAGE(ev.has_value(),
                      "FcSemEvaluator::build failed: " << ev.error().detail);
      auto sigma = ev->sigma(samp, est.theta);
      REQUIRE_MESSAGE(sigma.has_value(),
                      "sigma failed: " << sigma.error().detail);
      const Eigen::MatrixXd lavaan_sigma =
          fixture_matrix_in_built_order(b, j, "implied_sigma");
      CHECK_MESSAGE((sigma->sigma[0] - lavaan_sigma).cwiseAbs().maxCoeff() <
                        1e-5,
                    "implied Sigma mismatch");

      const double expected_f =
          j["chi2"].get<double>() /
          (2.0 * static_cast<double>(j["n_obs"].get<std::int64_t>()));
      check_close("fmin", est.fmin, expected_f, 1e-6);

      const double chi2 = magmaan::inference::chi2_stat(samp, est);
      check_close("chi2", chi2, j["chi2"].get<double>(), 2e-5);

      auto df = magmaan::inference::df_stat(b.pt, samp);
      REQUIRE_MESSAGE(df.has_value(), "df_stat failed: " << df.error().detail);
      CHECK(*df == j["df"].get<int>());

      auto fx = magmaan::measures::fit_extras_fcsem(b.pt, samp, est);
      REQUIRE_MESSAGE(fx.has_value(),
                      "fit_extras_fcsem failed: " << fx.error().detail);
      CHECK(fx->npar == j["npar"].get<int>());
      CHECK(fx->ntotal == j["n_obs"].get<std::int64_t>());

      const auto baseline = magmaan::measures::baseline_chi2(samp);
      const auto fm = magmaan::measures::fit_measures(chi2, *df, baseline,
                                                      samp);
      check_close("cfi", fm.cfi, j["cfi"].get<double>(), 2e-5);
      check_close("tli", fm.tli, j["tli"].get<double>(), 2e-5);
      check_close("rmsea", fm.rmsea, j["rmsea"].get<double>(), 2e-5);
      check_close("rmsea_ci_lower", fm.rmsea_ci_lower,
                  j["rmsea_ci_lower"].get<double>(), 3e-5);
      check_close("rmsea_ci_upper", fm.rmsea_ci_upper,
                  j["rmsea_ci_upper"].get<double>(), 3e-5);
      check_close("rmsea_pvalue", fm.rmsea_pvalue,
                  j["rmsea_pvalue"].get<double>(), 3e-5);
      check_close("srmr", fx->srmr, j["srmr"].get<double>(), 3e-5);
      check_close("logl", fx->logl, j["logl"].get<double>(), 3e-5);
      check_close("unrestricted_logl", fx->unrestricted_logl,
                  j["unrestricted_logl"].get<double>(), 3e-5);
      check_close("aic", fx->aic, j["aic"].get<double>(), 3e-5);
      check_close("bic", fx->bic, j["bic"].get<double>(), 3e-5);
      check_close("bic2", fx->bic2, j["bic2"].get<double>(), 3e-5);

      auto rows = magmaan::measures::standardize::standardized_rows_fcsem(
          b.pt, b.names, samp, est, vcov);
      REQUIRE_MESSAGE(rows.has_value(),
                      "standardized_rows_fcsem failed: "
                          << rows.error().detail);
      CHECK(rows->size() == j["weights"].size() + j["rows"].size());

      for (const auto& row : j["weights"]) check_reported_row(*rows, row);
      for (const auto& row : j["rows"]) check_reported_row(*rows, row);
    }
  }
}
