#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "../test_fit.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

using magmaan::test::matrix_from_json;

using magmaan::test::vector_from_json;

double max_abs_diff(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  if (a.size() == 0)
    return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

double max_abs_diff(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols())
    return std::numeric_limits<double>::infinity();
  if (a.size() == 0)
    return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

bool close(double a, double b, double tol) {
  return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

int count_status(const nlohmann::json &nodes, const std::string &field,
                 const std::string &value) {
  int out = 0;
  for (const auto &node : nodes) {
    if (node.contains(field) && node[field].get<std::string>() == value) {
      ++out;
    }
  }
  return out;
}

bool has_key_recursive(const nlohmann::json &j, const std::string &key) {
  if (j.is_object()) {
    if (j.contains(key))
      return true;
    for (const auto &item : j.items()) {
      if (has_key_recursive(item.value(), key))
        return true;
    }
  } else if (j.is_array()) {
    for (const auto &item : j) {
      if (has_key_recursive(item, key))
        return true;
    }
  }
  return false;
}

magmaan::data::SampleStats sample_stats_from_fit(const nlohmann::json &fit) {
  magmaan::data::SampleStats samp;
  samp.S = {matrix_from_json(fit["sample_cov"])};
  if (fit.contains("sample_mean") && !fit["sample_mean"].is_null()) {
    samp.mean = {vector_from_json(fit["sample_mean"])};
  }
  samp.n_obs = {fit["n_obs"].get<std::int64_t>()};
  return samp;
}

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

std::optional<Handles> handles_from_case(const nlohmann::json &c,
                                         std::vector<std::string> &failures) {
  const std::string id = c["id"].get<std::string>();
  auto fp = magmaan::parse::Parser::parse(c["model"].get<std::string>());
  if (!fp.has_value()) {
    failures.push_back(id + ": parse - " + fp.error().detail);
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = c.value("meanstructure", false);
  opts.fixed_x = c.value("fixed_x", true);
  auto pt = magmaan::spec::build(*fp, opts);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify - " + pt.error().detail);
    return std::nullopt;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) {
    failures.push_back(id + ": matrix_rep - " + rep.error().detail);
    return std::nullopt;
  }
  return Handles{std::move(*pt), std::move(*rep)};
}

magmaan::optim::OptimOptions paper_corpus_opts() {
  return magmaan::optim::OptimOptions{
      .max_iter = 7000,
      .ftol = 1e-13,
      .gtol = 1e-8,
      .history = 10,
  };
}

void check_implied(const std::string &label,
                   const magmaan::spec::LatentStructure &pt,
                   const magmaan::model::MatrixRep &rep,
                   const magmaan::estimate::Estimates &est,
                   const nlohmann::json &fit,
                   std::vector<std::string> &failures) {
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    failures.push_back(label + ": ModelEvaluator - " + ev.error().detail);
    return;
  }
  auto im = ev->sigma(est.theta);
  if (!im.has_value()) {
    failures.push_back(label + ": implied moments - " + im.error().detail);
    return;
  }
  const double d_sigma =
      max_abs_diff(im->sigma[0], matrix_from_json(fit["sigma"]));
  if (d_sigma > 5e-4) {
    failures.push_back(label + ": max|Sigma - lavaan| = " +
                       std::to_string(d_sigma));
  }
}

} // namespace

TEST_CASE("Paper corpus scout manifest is well formed") {
  const std::string root = magmaan::test::fixtures_dir();
  auto raw =
      magmaan::test::read_fixture(root + "/paper_corpus/scout_manifest.json");
  REQUIRE(raw.has_value());

  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("_meta"));
  CHECK(j["_meta"]["corpus_id"].get<std::string>() ==
        "magmaan_paper_corpus_scout_v1");
  CHECK(j["_meta"]["fixture_kind"].get<std::string>() ==
        "paper_corpus.scout_manifest");
  CHECK(j["_meta"]["tool"].get<std::string>() ==
        "external/paper_corpus/scripts/scout_osf.R");

  REQUIRE(j.contains("counts"));
  REQUIRE(j.contains("nodes"));
  CHECK(j["nodes"].size() ==
        static_cast<std::size_t>(j["counts"]["seed_nodes"].get<int>()));
  CHECK(j["counts"]["scouted"].get<int>() ==
        static_cast<int>(j["nodes"].size()));
  CHECK(j["counts"]["promote_first"].get<int>() >= 1);
  CHECK(j["counts"]["promoted"].get<int>() >= 1);
  CHECK(j["counts"]["candidate_code_files"].get<int>() > 0);
  CHECK(j["counts"]["candidate_data_files"].get<int>() > 0);
  CHECK(j["counts"]["scanned_code_files"].get<int>() > 0);
  CHECK(j["counts"]["detected_lavaan_calls"].get<int>() > 0);

  std::unordered_set<std::string> ids;
  for (const auto &node : j["nodes"]) {
    REQUIRE(node.contains("node_id"));
    REQUIRE(node.contains("promotion_status"));
    REQUIRE(node.contains("summary"));
    REQUIRE(node.contains("files"));
    ids.insert(node["node_id"].get<std::string>());

    for (const auto &file : node["files"]) {
      REQUIRE(file.contains("name"));
      REQUIRE(file.contains("file_class"));
      REQUIRE(file.contains("scan_status"));
      if (file["scan_status"].get<std::string>() == "scanned") {
        REQUIRE(file.contains("signals"));
      }
    }
  }

  CHECK(ids.contains("hwkem"));
  CHECK(ids.contains("zxqvn"));
  CHECK(count_status(j["nodes"], "promotion_status", "promote_first") >= 1);
  CHECK(count_status(j["nodes"], "promotion_status", "promoted_core_ml") == 1);
  CHECK_FALSE(has_key_recursive(j, "content"));
}

TEST_CASE("zxqvn paper corpus fixture is well formed") {
  const std::string path =
      magmaan::test::fixtures_dir() + "/paper_corpus/zxqvn_reference.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());

  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("_meta"));
  CHECK(j["_meta"]["corpus_id"].get<std::string>() ==
        "magmaan_paper_corpus_zxqvn_v1");
  CHECK(j["_meta"]["fixture_kind"].get<std::string>() ==
        "paper_corpus.reference");
  CHECK(j["_meta"]["tool"].get<std::string>() ==
        "external/paper_corpus/scripts/export_magmaan.R");
  REQUIRE(j.contains("cases"));
  REQUIRE(j["cases"].size() == 1);

  const auto &c = j["cases"][0];
  CHECK(c["id"].get<std::string>() == "zxqvn_affect_mediation");
  CHECK(c["source"].get<std::string>() == "zxqvn");
  CHECK(c["fits"].contains("ML"));
  CHECK(c["data_summary"]["wide_rows"].get<int>() == 995);
  CHECK(c["data_summary"]["long_rows"].get<int>() == 6944);
  CHECK(c["data_summary"]["cluster_count"].get<int>() == 992);
  CHECK(c["clustered_reference"]["point_estimates_match_core"].get<bool>());
  CHECK_FALSE(has_key_recursive(j, "content"));
}

TEST_CASE("zxqvn paper corpus ML fit matches lavaan") {
  const std::string path =
      magmaan::test::fixtures_dir() + "/paper_corpus/zxqvn_reference.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());

  std::vector<std::string> failures;
  const auto opt = paper_corpus_opts();
  const auto &c = j["cases"][0];
  const auto &fit = c["fits"]["ML"];
  const std::string label = c["id"].get<std::string>() + "/ML";

  auto handles = handles_from_case(c, failures);
  REQUIRE(handles.has_value());
  auto samp = sample_stats_from_fit(fit);
  auto est_or = magmaan::test::fit(handles->pt, handles->rep, samp,
                                   magmaan::estimate::Bounds{},
                                   magmaan::estimate::Backend::NloptLbfgs, opt);
  if (!est_or.has_value()) {
    FAIL(est_or.error().detail);
  }

  const Eigen::VectorXd lavaan_theta = vector_from_json(fit["theta_hat"]);
  const double d_theta = max_abs_diff(est_or->theta, lavaan_theta);
  if (d_theta > 1e-3) {
    failures.push_back(label + ": max|theta - lavaan| = " +
                       std::to_string(d_theta));
  }
  auto df_or = magmaan::inference::df_stat(handles->pt, samp);
  if (!df_or.has_value()) {
    failures.push_back(label + ": df_stat - " + df_or.error().detail);
  } else if (*df_or != fit["df"].get<int>()) {
    failures.push_back(label + ": df = " + std::to_string(*df_or) +
                       ", lavaan = " + std::to_string(fit["df"].get<int>()));
  }
  if (!c.value("fixed_x", true)) {
    check_implied(label, handles->pt, handles->rep, *est_or, fit, failures);
  }

  const double lavaan_f = fit["fmin"].get<double>();
  if (!close(est_or->fmin, lavaan_f, 2e-4)) {
    failures.push_back(label + ": fmin = " + std::to_string(est_or->fmin) +
                       ", lavaan fmin = " + std::to_string(lavaan_f));
  }
  const double chisq = 2.0 * static_cast<double>(samp.n_obs[0]) * est_or->fmin;
  if (!close(chisq, fit["chisq"].get<double>(), 5e-3)) {
    failures.push_back(label + ": chisq = " + std::to_string(chisq) +
                       ", lavaan = " +
                       std::to_string(fit["chisq"].get<double>()));
  }

  for (const auto &f : failures)
    MESSAGE("  FAIL " << f);
  CHECK(failures.empty());
}
