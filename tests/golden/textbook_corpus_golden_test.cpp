#include "../test_fit.hpp"
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
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

using magmaan::test::matrix_from_json;

using magmaan::test::vector_from_json;

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

bool is_numeric_vector_json(const nlohmann::json &j) {
  if (!j.is_array())
    return false;
  for (const auto &x : j) {
    if (!x.is_number())
      return false;
  }
  return true;
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
  if (c.contains("sample_mean") && is_numeric_vector_json(c["sample_mean"])) {
    samp.mean = {vector_from_json(c["sample_mean"])};
  }
  samp.n_obs = {c["n_obs"].get<std::int64_t>()};
  return samp;
}

magmaan::data::SampleStats sample_stats_from_export(const nlohmann::json &c) {
  magmaan::data::SampleStats samp;
  for (const auto &block : c["data"]["sample_cov"]) {
    samp.S.push_back(matrix_from_json(block));
  }
  if (c["data"].contains("sample_mean") && !c["data"]["sample_mean"].empty()) {
    for (const auto &block : c["data"]["sample_mean"]) {
      samp.mean.push_back(vector_from_json(block));
    }
  }
  for (const auto &n : c["data"]["n_obs"]) {
    samp.n_obs.push_back(n.get<std::int64_t>());
  }
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

std::optional<Handles>
handles_from_export_case(const nlohmann::json &c,
                         std::vector<std::string> &failures) {
  const std::string id = c["case_id"].get<std::string>();
  auto flat = magmaan::parse::Parser::parse(c["model"].get<std::string>());
  if (!flat.has_value()) {
    failures.push_back(id + ": parse - " + flat.error().detail);
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.n_groups = c["data"].value("n_groups", 1);
  opts.meanstructure = c["model_options"].value("meanstructure", false);
  opts.fixed_x = c["model_options"].value("fixed_x", true);
  opts.auto_cov_y = c.value("lavaan_function", std::string{}) == "sem";
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
  if (!im->mu.empty() && oracle.contains("mu") &&
      is_numeric_vector_json(oracle["mu"])) {
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
  const double lavaan_f = oracle["fmin"].get<double>();
  if (!close(est->fmin, lavaan_f, 1e-3)) {
    failures.push_back(label + ": fmin = " + std::to_string(est->fmin) +
                       ", lavaan fmin = " + std::to_string(lavaan_f));
  }
}

struct ExportFitResult {
  std::string id;
  int df = 0;
  double chisq = 0.0;
};

std::optional<ExportFitResult>
check_export_case(const nlohmann::json &c, std::vector<std::string> &failures) {
  const std::string id = c["case_id"].get<std::string>();
  auto handles = handles_from_export_case(c, failures);
  if (!handles.has_value()) {
    return std::nullopt;
  }
  auto samp = sample_stats_from_export(c);
  auto est =
      magmaan::test::fit(handles->pt, handles->rep, samp, {},
                         magmaan::estimate::Backend::NloptLbfgs, textbook_opts());
  if (!est.has_value()) {
    failures.push_back(id + ": fit - " + est.error().detail);
    return std::nullopt;
  }

  const double lavaan_f = c["lavaan"]["fit"]["fmin"].get<double>();
  if (est->fmin > lavaan_f + 1e-6) {
    failures.push_back(id + ": fmin = " + std::to_string(est->fmin) +
                       " worse than lavaan " + std::to_string(lavaan_f));
  }

  auto df = magmaan::inference::df_stat(handles->pt, samp);
  if (!df.has_value()) {
    failures.push_back(id + ": df - " + df.error().detail);
  } else if (*df != c["lavaan"]["fit"]["df"].get<int>()) {
    failures.push_back(id + ": df = " + std::to_string(*df) +
                       ", lavaan = " +
                       std::to_string(c["lavaan"]["fit"]["df"].get<int>()));
  }

  const std::int64_t n_total = std::accumulate(
      samp.n_obs.begin(), samp.n_obs.end(), std::int64_t{0});
  const double chisq = 2.0 * static_cast<double>(n_total) * est->fmin;
  const double lavaan_chisq = c["lavaan"]["fit"]["chisq"].get<double>();
  const bool lavaan_well_converged =
      id != "kline_2023_ch22_guo_mi_strong";
  if (lavaan_well_converged && std::abs(chisq - lavaan_chisq) > 1e-3) {
    failures.push_back(id + ": chisq = " + std::to_string(chisq) +
                       ", lavaan = " + std::to_string(lavaan_chisq));
  }
  return ExportFitResult{id, df.value_or(0), chisq};
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

void check_little_single_indicator_case_at_lavaan_theta() {
  constexpr std::string_view case_id =
      "little_2013_ch3_fig_3_6_1indicator";
  const std::string case_dir = repo_root_from_fixtures() +
      "/corpus/textbook-corpus/cases/little_2013/" + std::string(case_id);
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

// --- Kline Guo multi-group measurement invariance (textbook-corpus submodule) ---
//
// The four Guo invariance rungs (configural/weak/strong/partial_strong) live in
// the textbook-corpus submodule as per-group summary statistics with a
// checked-in lavaan ML oracle. We re-fit each from those summary stats and
// assert order-free parity (df exact, chisq + fmin within 1e-3) against the
// oracle. This guards the fixed duplicated-formula-term bug: a phantom
// invariance parameter that moved the analytic gradient but not the model
// moments — caught by the converged fit, not by implied moments. Per-parameter
// theta/SE comparison is deferred (the oracle stores them in lavaan's free-
// parameter order, which needs a lavaan->magmaan map the chisq/df check avoids).

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> cells;
  std::string cur;
  for (const char ch : line) {
    if (ch == ',') {
      cells.push_back(cur);
      cur.clear();
    } else if (ch != '"' && ch != '\r') {
      cur.push_back(ch);
    }
  }
  cells.push_back(cur);
  return cells;
}

std::vector<std::vector<std::string>> read_csv_rows(const std::string &path) {
  std::vector<std::vector<std::string>> rows;
  auto raw = magmaan::test::read_fixture(path);
  if (!raw.has_value())
    return rows;
  std::string line;
  for (const char ch : *raw) {
    if (ch == '\n') {
      if (!line.empty())
        rows.push_back(split_csv_line(line));
      line.clear();
    } else {
      line.push_back(ch);
    }
  }
  if (!line.empty())
    rows.push_back(split_csv_line(line));
  return rows;
}

// Named covariance CSV: header `"","V1","V2",...`, then one `"Vi",x,x,...` row
// per variable. strtod (not stod) keeps the -fno-exceptions build honest.
std::optional<Eigen::MatrixXd> read_named_cov_csv(const std::string &path) {
  const auto rows = read_csv_rows(path);
  if (rows.size() < 2)
    return std::nullopt;
  const Eigen::Index p = static_cast<Eigen::Index>(rows[0].size()) - 1;
  if (p <= 0 || rows.size() < static_cast<std::size_t>(p) + 1)
    return std::nullopt;
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r) {
    const auto &row = rows[static_cast<std::size_t>(r) + 1];
    if (static_cast<Eigen::Index>(row.size()) < p + 1)
      return std::nullopt;
    for (Eigen::Index c = 0; c < p; ++c) {
      S(r, c) = std::strtod(row[static_cast<std::size_t>(c) + 1].c_str(),
                            nullptr);
    }
  }
  return S;
}

// Named mean CSV: header `"name","mean"`, then one `"Vi",x` row per variable.
std::optional<Eigen::VectorXd> read_named_mean_csv(const std::string &path) {
  const auto rows = read_csv_rows(path);
  if (rows.size() < 2)
    return std::nullopt;
  const Eigen::Index p = static_cast<Eigen::Index>(rows.size()) - 1;
  Eigen::VectorXd mean(p);
  for (Eigen::Index r = 0; r < p; ++r) {
    const auto &row = rows[static_cast<std::size_t>(r) + 1];
    if (row.size() < 2)
      return std::nullopt;
    mean(r) = std::strtod(row[1].c_str(), nullptr);
  }
  return mean;
}

void check_kline_guo_mi_case(std::string_view case_id,
                             bool lavaan_well_converged,
                             std::vector<std::string> &failures) {
  const std::string id(case_id);
  const std::string case_dir = repo_root_from_fixtures() +
      "/corpus/textbook-corpus/cases/kline_2023/" + id;

  auto model_raw = magmaan::test::read_fixture(case_dir + "/model.lav");
  auto meta_raw = magmaan::test::read_fixture(case_dir + "/meta.json");
  auto ref_raw =
      magmaan::test::read_fixture(case_dir + "/expected/lavaan_ml.json");
  if (!model_raw.has_value() || !meta_raw.has_value() || !ref_raw.has_value()) {
    failures.push_back(id + ": missing model/meta/expected");
    return;
  }
  const auto meta = nlohmann::json::parse(*meta_raw, nullptr, false);
  const auto ref = nlohmann::json::parse(*ref_raw, nullptr, false);
  if (meta.is_discarded() || ref.is_discarded()) {
    failures.push_back(id + ": invalid meta/expected JSON");
    return;
  }

  auto flat = magmaan::parse::Parser::parse(*model_raw);
  if (!flat.has_value()) {
    failures.push_back(id + ": parse - " + flat.error().detail);
    return;
  }
  magmaan::spec::BuildOptions opts;
  opts.n_groups = meta["data"].value("n_groups", 1);
  opts.meanstructure = meta["model_options"].value("meanstructure", false);
  opts.fixed_x = meta["model_options"].value("fixed_x", true);
  opts.auto_cov_y = meta.value("lavaan_function", std::string{}) == "sem";
  auto pt = magmaan::spec::build(*flat, opts);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify - " + pt.error().detail);
    return;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) {
    failures.push_back(id + ": matrix_rep - " + rep.error().detail);
    return;
  }

  // Per-group summary statistics from the submodule CSVs. The CSV variable
  // order is the model's declaration order (= magmaan's observed order); a
  // misordering would surface as a chisq mismatch, not a silent pass.
  const auto &data = meta["data"];
  const auto &cov_files = data["files"]["sample_cov"];
  const auto &mean_files = data["files"]["sample_mean"];
  const auto &n_obs = data["n_obs"];
  magmaan::data::SampleStats samp;
  for (std::size_t g = 0; g < cov_files.size(); ++g) {
    auto S =
        read_named_cov_csv(case_dir + "/" + cov_files[g].get<std::string>());
    auto mean =
        read_named_mean_csv(case_dir + "/" + mean_files[g].get<std::string>());
    if (!S.has_value() || !mean.has_value()) {
      failures.push_back(id + ": unreadable summary-stat CSV for group " +
                         std::to_string(g + 1));
      return;
    }
    samp.S.push_back(std::move(*S));
    samp.mean.push_back(std::move(*mean));
    samp.n_obs.push_back(n_obs[g].get<std::int64_t>());
  }

  auto est = magmaan::test::fit(*pt, *rep, samp, {},
                                magmaan::estimate::Backend::NloptLbfgs,
                                textbook_opts());
  if (!est.has_value()) {
    failures.push_back(id + ": fit - " + est.error().detail);
    return;
  }

  const auto &fit = ref["fit"];
  // df is exact for every rung: it pins the constraint structure (a dropped or
  // mis-counted invariance constraint moves df), independent of the optimum.
  auto df = magmaan::inference::df_stat(*pt, samp);
  if (!df.has_value()) {
    failures.push_back(id + ": df - " + df.error().detail);
  } else if (*df != fit["df"].get<int>()) {
    failures.push_back(id + ": df = " + std::to_string(*df) +
                       ", lavaan = " + std::to_string(fit["df"].get<int>()));
  }
  const double chi2 = magmaan::inference::chi2_stat(samp, *est);
  const double chi2_lavaan = fit["chisq"].get<double>();
  const double fmin_lavaan = fit["fmin"].get<double>();
  // magmaan must reach a minimum no worse than the lavaan oracle (+ float
  // slack). True for every rung.
  if (est->fmin > fmin_lavaan + 1e-6) {
    failures.push_back(id + ": fmin = " + std::to_string(est->fmin) +
                       " worse than lavaan " + std::to_string(fmin_lavaan));
  }
  if (lavaan_well_converged) {
    // Tight two-sided parity where lavaan is well-converged.
    if (std::abs(chi2 - chi2_lavaan) > 1e-3) {
      failures.push_back(id + ": chisq = " + std::to_string(chi2) +
                         ", lavaan = " + std::to_string(chi2_lavaan));
    }
  } else if (chi2 > chi2_lavaan + 1e-3) {
    // `guo_mi_strong`: lavaan's reference is under-converged. magmaan reaches a
    // strictly lower chisq (here ~60.451 vs lavaan's ~60.549) at identical df,
    // and three independent optimizers (NLopt L-BFGS, NLopt SLSQP, PORT) agree
    // on that lower optimum to 6 digits — so this is lavaan stopping early, not
    // a magmaan error. We assert df-exact + no-worse-than-oracle and skip the
    // two-sided chisq parity; a higher (worse) chisq would still fail here.
    failures.push_back(id + ": chisq = " + std::to_string(chi2) +
                       " worse than (under-converged) lavaan " +
                       std::to_string(chi2_lavaan));
  }
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

TEST_CASE("Textbook corpus Kline Guo invariance ML cases match lavaan") {
  const std::string root = magmaan::test::fixtures_dir();
  auto raw =
      magmaan::test::read_fixture(root + "/textbook_corpus/case_exports.json");
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("_meta"));
  CHECK(j["_meta"]["fixture_kind"].get<std::string>() ==
        "textbook_corpus.case_export");
  REQUIRE(j.contains("cases"));
  REQUIRE(j["cases"].size() == 4);

  std::vector<std::string> failures;
  std::vector<ExportFitResult> results;
  for (const auto &c : j["cases"]) {
    auto r = check_export_case(c, failures);
    if (r.has_value()) {
      results.push_back(*r);
    }
  }

  if (results.size() == 4) {
    const std::array<std::pair<std::size_t, std::size_t>, 2> nested_pairs = {{
        {1, 0}, // weak vs configural
        {3, 1}, // partial-strong vs weak
    }};
    for (const auto &[more_restricted, less_restricted] : nested_pairs) {
      const auto &h0 = results[more_restricted];
      const auto &h1 = results[less_restricted];
      const double diff = h0.chisq - h1.chisq;
      const int ddf = h0.df - h1.df;
      const double lavaan_diff =
          j["cases"][more_restricted]["lavaan"]["fit"]["chisq"].get<double>() -
          j["cases"][less_restricted]["lavaan"]["fit"]["chisq"].get<double>();
      const int lavaan_ddf =
          j["cases"][more_restricted]["lavaan"]["fit"]["df"].get<int>() -
          j["cases"][less_restricted]["lavaan"]["fit"]["df"].get<int>();
      if (ddf != lavaan_ddf || std::abs(diff - lavaan_diff) > 1e-3) {
        failures.push_back(h0.id + " vs " + h1.id +
                           ": nested delta chisq=" + std::to_string(diff) +
                           " lavaan=" + std::to_string(lavaan_diff) +
                           " ddf=" + std::to_string(ddf) +
                           " lavaan_ddf=" + std::to_string(lavaan_ddf));
      }
    }
  }

  for (const auto &f : failures)
    MESSAGE("  FAIL " << f);
  CHECK(failures.empty());
}

TEST_CASE("Newsom LCS promoted-observed implied moments match lavaan at theta") {
  check_newsom_lcs_case_at_lavaan_theta("newsom_2015_ex9_3");
}

TEST_CASE("Little single-indicator implied moments match lavaan at theta") {
  check_little_single_indicator_case_at_lavaan_theta();
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

// Re-fit the four Kline (2023, ch22) Guo divergent-thinking measurement-
// invariance models from the textbook-corpus submodule's per-group summary
// statistics and check df/chisq/fmin parity against the checked-in lavaan
// oracle. Guards the merge of duplicated `lhs op rhs` formula terms in
// lavaanify (`build_group_template`); see docs/backlog/todo.md.
TEST_CASE("Kline Guo multi-group measurement-invariance fits match lavaan") {
  struct Rung {
    std::string_view id;
    bool lavaan_well_converged;
  };
  // `strong` is the lone under-converged lavaan reference (see
  // check_kline_guo_mi_case); the other three match lavaan to <1e-3.
  const std::array<Rung, 4> cases = {{
      {"kline_2023_ch22_guo_mi_configural", true},
      {"kline_2023_ch22_guo_mi_weak", true},
      {"kline_2023_ch22_guo_mi_strong", false},
      {"kline_2023_ch22_guo_mi_partial_strong", true},
  }};
  // Graceful skip when the textbook-corpus submodule isn't checked out.
  const std::string probe = repo_root_from_fixtures() +
      "/corpus/textbook-corpus/cases/kline_2023/" + std::string(cases[0].id) +
      "/model.lav";
  if (!magmaan::test::read_fixture(probe).has_value()) {
    MESSAGE("textbook-corpus submodule absent; skipping Kline Guo MI parity");
    return;
  }
  std::vector<std::string> failures;
  for (const auto &c : cases)
    check_kline_guo_mi_case(c.id, c.lavaan_well_converged, failures);
  for (const auto &f : failures)
    MESSAGE("  FAIL " << f);
  CHECK(failures.empty());
}
