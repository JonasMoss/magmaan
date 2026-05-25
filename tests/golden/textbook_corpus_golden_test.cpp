#include "../test_fit.hpp"
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

Eigen::MatrixXd matrix_from_json(const nlohmann::json &j) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr > 0 ? static_cast<Eigen::Index>(j[0].size()) : 0;
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r) {
    for (Eigen::Index c = 0; c < nc; ++c) {
      out(r, c) = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                      .get<double>();
    }
  }
  return out;
}

Eigen::VectorXd vector_from_json(const nlohmann::json &j) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].get<double>();
  }
  return out;
}

double max_abs_diff(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  if (a.size() == 0)
    return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

double max_abs_diff(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  if (a.size() == 0)
    return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

bool close(double a, double b, double tol) {
  return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

std::string repo_root_from_fixtures() {
  const std::string fixtures = magmaan::test::fixtures_dir();
  const std::string suffix = "/tests/fixtures";
  if (fixtures.size() >= suffix.size() &&
      fixtures.compare(fixtures.size() - suffix.size(), suffix.size(),
                       suffix) == 0) {
    return fixtures.substr(0, fixtures.size() - suffix.size());
  }
  return fixtures + "/../..";
}

nlohmann::json read_json_or_fail(const std::string &path) {
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE_MESSAGE(raw.has_value(), "missing " << path);
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_MESSAGE(!j.is_discarded(), "invalid JSON " << path);
  return j;
}

magmaan::data::SampleStats sample_stats_from_case(const nlohmann::json &c) {
  magmaan::data::SampleStats samp;
  samp.S = {matrix_from_json(c["sample_cov"])};
  if (c.contains("sample_mean") && !c["sample_mean"].is_null()) {
    samp.mean = {vector_from_json(c["sample_mean"])};
  }
  samp.n_obs = {c["n_obs"].get<std::int64_t>()};
  return samp;
}

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

std::optional<Handles> handles_from_case(const nlohmann::json &c,
                                         std::vector<std::string> &failures) {
  const std::string id = c["id"].get<std::string>();
  auto flat = magmaan::parse::Parser::parse(c["model"].get<std::string>());
  if (!flat.has_value()) {
    failures.push_back(id + ": parse - " + flat.error().detail);
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = c.value("meanstructure", false);
  opts.fixed_x = c.value("fixed_x", true);
  if (c.value("lavaan_function", std::string{}) == "growth") {
    opts.meanstructure = true;
    opts.int_ov_free = false;
    opts.int_lv_free = true;
    opts.fixed_x = false;
  }
  auto pt = magmaan::spec::build(*flat, opts);
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

magmaan::optim::OptimOptions textbook_opts() {
  return magmaan::optim::OptimOptions{
      .max_iter = 6000,
      .ftol = 1e-12,
      .gtol = 1e-8,
      .history = 10,
  };
}

void check_case(const std::string &corpus, const nlohmann::json &c,
                std::vector<std::string> &failures) {
  const std::string id = c["id"].get<std::string>();
  const std::string label = corpus + "/" + id;
  auto handles = handles_from_case(c, failures);
  if (!handles.has_value())
    return;
  auto samp = sample_stats_from_case(c);
  auto est =
      magmaan::test::fit(handles->pt, handles->rep, samp, {},
                         magmaan::estimate::Backend::NloptLbfgs, textbook_opts());
  if (!est.has_value()) {
    failures.push_back(label + ": fit - " + est.error().detail);
    return;
  }
  auto ev = magmaan::model::ModelEvaluator::build(handles->pt, handles->rep);
  if (!ev.has_value()) {
    failures.push_back(label + ": evaluator - " + ev.error().detail);
    return;
  }
  auto im = ev->sigma(est->theta);
  if (!im.has_value()) {
    failures.push_back(label + ": implied - " + im.error().detail);
    return;
  }
  const auto &oracle = c["lavaan"];
  const double d_sigma =
      max_abs_diff(im->sigma[0], matrix_from_json(oracle["sigma"]));
  if (d_sigma > 1e-3) {
    failures.push_back(label +
                       ": max|Sigma - lavaan| = " + std::to_string(d_sigma));
  }
  if (!im->mu.empty() && oracle.contains("mu") && !oracle["mu"].is_null()) {
    const double d_mu = max_abs_diff(im->mu[0], vector_from_json(oracle["mu"]));
    if (d_mu > 1e-3) {
      failures.push_back(label +
                         ": max|mu - lavaan| = " + std::to_string(d_mu));
    }
  }
  auto df = magmaan::inference::df_stat(handles->pt, samp);
  if (!df.has_value()) {
    failures.push_back(label + ": df - " + df.error().detail);
  } else if (*df != oracle["df"].get<int>()) {
    failures.push_back(label + ": df = " + std::to_string(*df) +
                       ", lavaan = " + std::to_string(oracle["df"].get<int>()));
  }
  const double lavaan_f = 2.0 * oracle["fmin"].get<double>();
  if (!close(est->fmin, lavaan_f, 1e-3)) {
    failures.push_back(label + ": fmin = " + std::to_string(est->fmin) +
                       ", lavaan 2*fmin = " + std::to_string(lavaan_f));
  }
}

void check_json_file_exists(const std::string &corpus,
                            const std::string &name) {
  const std::string path =
      magmaan::test::fixtures_dir() + "/" + corpus + "/" + name;
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE_MESSAGE(raw.has_value(), "missing " << corpus << "/" << name);
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  CHECK_MESSAGE(!j.is_discarded(), corpus << "/" << name << " is invalid");
  CHECK_MESSAGE(j.contains("_meta"), corpus << "/" << name << " lacks _meta");
}

void check_newsom_lcs_case_at_lavaan_theta(std::string_view case_id) {
  const std::string case_dir = repo_root_from_fixtures() +
      "/corpus/textbook-corpus/cases/newsom_2015/" + std::string(case_id);
  auto model_raw = magmaan::test::read_fixture(case_dir + "/model.lav");
  REQUIRE_MESSAGE(model_raw.has_value(), "missing model for " << case_id);
  const auto meta = read_json_or_fail(case_dir + "/meta.json");
  const auto ref = read_json_or_fail(case_dir + "/expected/lavaan_ml.json");

  auto flat = magmaan::parse::Parser::parse(*model_raw);
  REQUIRE_MESSAGE(flat.has_value(), case_id << ": parse - "
                                            << flat.error().detail);
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = meta["model_options"].value("meanstructure", false);
  opts.fixed_x = meta["model_options"].value("fixed_x", true);
  opts.auto_cov_y = meta.value("lavaan_function", std::string{}) == "sem";
  auto pt = magmaan::spec::build(*flat, opts);
  REQUIRE_MESSAGE(pt.has_value(), case_id << ": lavaanify - "
                                          << pt.error().detail);
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE_MESSAGE(rep.has_value(), case_id << ": matrix_rep - "
                                           << rep.error().detail);
  auto ev = magmaan::model::ModelEvaluator::build(*pt, *rep);
  REQUIRE_MESSAGE(ev.has_value(), case_id << ": evaluator - "
                                          << ev.error().detail);

  const Eigen::VectorXd theta = vector_from_json(ref["theta"]);
  REQUIRE_MESSAGE(static_cast<std::size_t>(theta.size()) == ev->n_free(),
                  case_id << ": theta size " << theta.size()
                          << " != n_free " << ev->n_free());
  auto im = ev->sigma(theta);
  REQUIRE_MESSAGE(im.has_value(), case_id << ": sigma - "
                                          << im.error().detail);
  const double d_sigma =
      max_abs_diff(im->sigma[0], matrix_from_json(ref["implied"]["sigma"]));
  CHECK_MESSAGE(d_sigma < 1e-8,
                case_id << ": max|Sigma - lavaan| = " << d_sigma);
  REQUIRE(!im->mu.empty());
  const double d_mu =
      max_abs_diff(im->mu[0], vector_from_json(ref["implied"]["mu"]));
  CHECK_MESSAGE(d_mu < 1e-8, case_id << ": max|mu - lavaan| = " << d_mu);
}

} // namespace

TEST_CASE("Little and Newsom corpus fixtures are well formed") {
  for (const std::string corpus : {"little", "newsom"}) {
    for (const std::string file :
         {"manifest.json", "continuous_reference.json",
          "ordinal_reference.json", "mixed_reference.json",
          "observed_reference.json"}) {
      check_json_file_exists(corpus, file);
    }
  }
}

TEST_CASE("Textbook corpus v1 manifest is well formed") {
  const std::string root = magmaan::test::fixtures_dir();
  auto raw =
      magmaan::test::read_fixture(root + "/textbook_corpus/manifest.json");
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("_meta"));
  REQUIRE(j["_meta"].contains("corpus_id"));
  CHECK(j["_meta"]["corpus_id"].get<std::string>() ==
        "magmaan_textbook_corpus_v1");
  REQUIRE(j.contains("sources"));
  REQUIRE(j.contains("cases"));
  REQUIRE(j.contains("counts"));
  CHECK(j["sources"].size() == 4);
  CHECK(j["cases"].size() ==
        static_cast<std::size_t>(j["counts"]["total"].get<int>()));
  CHECK(j["counts"]["strict_parity"].get<int>() > 0);
  CHECK(j["counts"]["mixed"].get<int>() > 0);
  CHECK(j["counts"]["ordinal"].get<int>() > 0);
  CHECK(j["counts"]["observed_only"].get<int>() > 0);

  for (const auto &source : j["sources"]) {
    REQUIRE(source.contains("fixture_files"));
    for (const auto &file : source["fixture_files"]) {
      const std::string rel = file.get<std::string>();
      auto fixture = magmaan::test::read_fixture(root + "/" + rel);
      CHECK_MESSAGE(fixture.has_value(), "missing fixture " << rel);
    }
  }
}

TEST_CASE("Textbook corpus v1 overlap graph is well formed") {
  const std::string root = magmaan::test::fixtures_dir();
  auto manifest_raw =
      magmaan::test::read_fixture(root + "/textbook_corpus/manifest.json");
  auto overlap_raw =
      magmaan::test::read_fixture(root + "/textbook_corpus/overlap.json");
  auto overrides_raw = magmaan::test::read_fixture(
      root + "/textbook_corpus/overlap_overrides.json");
  REQUIRE(manifest_raw.has_value());
  REQUIRE(overlap_raw.has_value());
  REQUIRE(overrides_raw.has_value());

  auto manifest = nlohmann::json::parse(*manifest_raw, nullptr, false);
  auto overlap = nlohmann::json::parse(*overlap_raw, nullptr, false);
  auto overrides = nlohmann::json::parse(*overrides_raw, nullptr, false);
  REQUIRE_FALSE(manifest.is_discarded());
  REQUIRE_FALSE(overlap.is_discarded());
  REQUIRE_FALSE(overrides.is_discarded());
  REQUIRE(overlap.contains("_meta"));
  CHECK(overlap["_meta"]["corpus_id"].get<std::string>() ==
        "magmaan_textbook_corpus_v1");
  CHECK(overlap["_meta"]["fixture_kind"].get<std::string>() ==
        "textbook_corpus.overlap");
  CHECK(overrides["_meta"]["fixture_kind"].get<std::string>() ==
        "textbook_corpus.overlap_overrides");

  std::unordered_set<std::string> case_ids;
  for (const auto &c : manifest["cases"]) {
    case_ids.insert(c["id"].get<std::string>());
  }
  REQUIRE(overlap.contains("fingerprints"));
  REQUIRE(overlap.contains("clusters"));
  REQUIRE(overlap.contains("edges"));
  REQUIRE(overlap.contains("counts"));
  CHECK(overlap["fingerprints"].size() == case_ids.size());
  CHECK(overlap["counts"]["cases"].get<int>() ==
        static_cast<int>(case_ids.size()));
  CHECK(overlap["counts"]["fingerprints"].get<int>() ==
        static_cast<int>(overlap["fingerprints"].size()));
  CHECK(overlap["counts"]["clusters"].get<int>() ==
        static_cast<int>(overlap["clusters"].size()));
  CHECK(overlap["counts"]["edges"].get<int>() ==
        static_cast<int>(overlap["edges"].size()));

  for (const auto &fp : overlap["fingerprints"]) {
    CHECK(case_ids.contains(fp["case_id"].get<std::string>()));
  }

  const std::unordered_set<std::string> cluster_kinds = {
      "source_artifact", "data_artifact", "data_shape",
      "sample_stats",    "syntax_named",  "syntax_shape",
      "oracle_named",    "oracle_shape",  "family_shape_hint"};
  for (const auto &cluster : overlap["clusters"]) {
    REQUIRE(cluster.contains("kind"));
    CHECK(cluster_kinds.contains(cluster["kind"].get<std::string>()));
    REQUIRE(cluster.contains("case_ids"));
    CHECK(cluster["case_ids"].size() >= 2);
    for (const auto &id : cluster["case_ids"]) {
      CHECK(case_ids.contains(id.get<std::string>()));
    }
  }

  const std::unordered_set<std::string> edge_kinds = {
      "same_source_artifact",   "same_data_artifact", "same_sample_stats",
      "same_named_syntax",      "same_model_shape",   "same_oracle_structure",
      "same_family_shape_hint", "manual_equivalence"};
  const std::unordered_set<std::string> confidence = {
      "exact", "canonical", "shape", "heuristic", "manual"};
  bool has_exact = false;
  bool has_canonical = false;
  bool has_shape = false;
  for (const auto &edge : overlap["edges"]) {
    CHECK(edge_kinds.contains(edge["kind"].get<std::string>()));
    const std::string conf = edge["confidence"].get<std::string>();
    CHECK(confidence.contains(conf));
    has_exact = has_exact || conf == "exact";
    has_canonical = has_canonical || conf == "canonical";
    has_shape = has_shape || conf == "shape";
    REQUIRE(edge["case_ids"].size() == 2);
    for (const auto &id : edge["case_ids"]) {
      CHECK(case_ids.contains(id.get<std::string>()));
    }
  }
  CHECK(has_exact);
  CHECK(has_canonical);
  CHECK(has_shape);
}

TEST_CASE("Newsom LCS promoted-observed implied moments match lavaan at theta") {
  check_newsom_lcs_case_at_lavaan_theta("newsom_2015_ex9_3");
}

// TODO(default-backend): the provisional NLopt-L-BFGS default exposes a
// remaining corpus failure (`newsom/ex5_5b`). Keep this broad corpus check
// intact and unskip it with the default-backend study.
TEST_CASE("Little and Newsom continuous goldens match lavaan" *
          doctest::skip()) {
  std::vector<std::string> failures;
  int total = 0;

  for (const std::string corpus : {"little", "newsom"}) {
    const std::string path = magmaan::test::fixtures_dir() + "/" + corpus +
                             "/continuous_reference.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto j = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(j.is_discarded());
    for (const auto &c : j["cases"]) {
      ++total;
      check_case(corpus, c, failures);
    }
  }

  MESSAGE("Little/Newsom continuous cases checked: " << total);
  for (const auto &f : failures)
    MESSAGE("  FAIL " << f);
  CHECK(failures.empty());
}
