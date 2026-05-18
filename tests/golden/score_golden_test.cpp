#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

const std::vector<std::string> kScoreFixtures = {
    "0001_ml_fixed_row_and_equality",
    "0002_fiml_fixed_row_and_equality",
    "0003_uls_fixed_row_and_equality",
    "0004_ordinal_dwls_fixed_row_and_equality",
    "0005_mixed_dwls_fixed_row_and_equality",
};

Eigen::MatrixXd matrix_from_json(const nlohmann::json& j) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr > 0 ? static_cast<Eigen::Index>(j[0].size()) : 0;
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r) {
    for (Eigen::Index c = 0; c < nc; ++c) {
      out(r, c) =
          j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
              .get<double>();
    }
  }
  return out;
}

Eigen::VectorXd vector_from_json(const nlohmann::json& j) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].get<double>();
  }
  return out;
}

magmaan::data::SampleStats sample_stats_from_fixture(const nlohmann::json& fit) {
  magmaan::data::SampleStats samp;
  for (const auto& b : fit["sample_cov"]) {
    samp.S.push_back(matrix_from_json(b["matrix"]));
  }
  const auto& nobs = fit["n_obs_per_block"];
  if (nobs.is_array()) {
    for (const auto& n : nobs) samp.n_obs.push_back(n.get<std::int64_t>());
  } else {
    samp.n_obs.push_back(nobs.get<std::int64_t>());
  }
  if (fit.contains("sample_mean") && !fit["sample_mean"].is_null()) {
    for (const auto& b : fit["sample_mean"]) {
      samp.mean.push_back(vector_from_json(b["vector"]));
    }
  }
  return samp;
}

magmaan::data::RawData raw_from_fixture(const nlohmann::json& exp) {
  magmaan::data::RawData raw;
  for (const auto& block : exp["raw"]) {
    const auto& Xj = block["X"];
    const auto& Mj = block["mask"];
    const Eigen::Index n = static_cast<Eigen::Index>(Xj.size());
    const Eigen::Index p = n > 0 ? static_cast<Eigen::Index>(Xj[0].size()) : 0;
    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < p; ++c) {
        const auto& x = Xj[static_cast<std::size_t>(r)]
                         [static_cast<std::size_t>(c)];
        X(r, c) = x.is_null() ? std::numeric_limits<double>::quiet_NaN()
                              : x.get<double>();
        M(r, c) = static_cast<std::uint8_t>(
            Mj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                .get<int>());
      }
    }
    raw.X.push_back(std::move(X));
    raw.mask.push_back(std::move(M));
  }
  return raw;
}

std::vector<Eigen::MatrixXd> data_blocks_from_fixture(const nlohmann::json& exp) {
  std::vector<Eigen::MatrixXd> blocks;
  for (const auto& b : exp["blocks"]) blocks.push_back(matrix_from_json(b["matrix"]));
  return blocks;
}

std::vector<std::vector<std::int32_t>>
ordered_masks_from_fixture(const nlohmann::json& exp) {
  std::vector<std::vector<std::int32_t>> ordered;
  for (const auto& b : exp["ordered_mask"]) {
    std::vector<std::int32_t> mask;
    for (const auto& z : b["mask"]) mask.push_back(z.get<std::int32_t>());
    ordered.push_back(std::move(mask));
  }
  return ordered;
}

std::string ordinal_syntax(const nlohmann::json& exp) {
  std::string src = exp["input"].get<std::string>();
  if (!src.empty() && src.back() != '\n') src.push_back('\n');

  const auto& first_block = exp["blocks"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(first_block[0].size());
  std::vector<int> max_level(static_cast<std::size_t>(p), 0);
  for (const auto& block : exp["blocks"]) {
    for (const auto& row : block["matrix"]) {
      for (Eigen::Index j = 0; j < p; ++j) {
        max_level[static_cast<std::size_t>(j)] =
            std::max(max_level[static_cast<std::size_t>(j)],
                     row[static_cast<std::size_t>(j)].get<int>());
      }
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    src += "x" + std::to_string(j + 1) + " | ";
    for (int lev = 1; lev < max_level[static_cast<std::size_t>(j)]; ++lev) {
      if (lev > 1) src += " + ";
      src += "t" + std::to_string(lev);
    }
    src += "\n";
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    src += "x" + std::to_string(j + 1) + " ~*~ 1*x" +
           std::to_string(j + 1) + "\n";
  }
  return src;
}

std::string mixed_ordinal_syntax(const nlohmann::json& exp) {
  std::string src = exp["input"].get<std::string>();
  if (!src.empty() && src.back() != '\n') src.push_back('\n');

  const auto& first_block = exp["blocks"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(first_block[0].size());
  std::vector<int> ordered(static_cast<std::size_t>(p), 0);
  for (Eigen::Index j = 0; j < p; ++j) {
    ordered[static_cast<std::size_t>(j)] =
        exp["ordered_mask"][0]["mask"][static_cast<std::size_t>(j)].get<int>();
  }
  std::vector<int> max_level(static_cast<std::size_t>(p), 0);
  for (const auto& block : exp["blocks"]) {
    for (const auto& row : block["matrix"]) {
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[static_cast<std::size_t>(j)] == 0) continue;
        max_level[static_cast<std::size_t>(j)] =
            std::max(max_level[static_cast<std::size_t>(j)],
                     row[static_cast<std::size_t>(j)].get<int>());
      }
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) continue;
    src += "x" + std::to_string(j + 1) + " | ";
    for (int lev = 1; lev < max_level[static_cast<std::size_t>(j)]; ++lev) {
      if (lev > 1) src += " + ";
      src += "t" + std::to_string(lev);
    }
    src += "\n";
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) continue;
    src += "x" + std::to_string(j + 1) + " ~*~ 1*x" +
           std::to_string(j + 1) + "\n";
  }
  return src;
}

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames names;
  magmaan::model::MatrixRep rep;
};

std::optional<Handles> build_handles(const std::string& id,
                                     const nlohmann::json& exp,
                                     std::vector<std::string>& failures) {
  std::string src = exp["input"].get<std::string>();
  const std::string kind = exp["kind"].get<std::string>();
  if (kind == "ordinal") src = ordinal_syntax(exp);
  if (kind == "mixed_ordinal") src = mixed_ordinal_syntax(exp);
  auto fp = magmaan::parse::Parser::parse(src);
  if (!fp.has_value()) {
    failures.push_back(id + ": parse - " + fp.error().detail);
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = exp.value("meanstructure", false);
  if (kind == "mixed_ordinal") opts.meanstructure = true;
  if (exp.contains("blocks") && !exp["blocks"].is_null()) {
    opts.n_groups = static_cast<std::int32_t>(exp["blocks"].size());
  }
  magmaan::spec::LatentNames names;
  auto pt = magmaan::spec::build(*fp, opts, nullptr, &names);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify - " + pt.error().detail);
    return std::nullopt;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) {
    failures.push_back(id + ": matrix_rep - " + rep.error().detail);
    return std::nullopt;
  }
  return Handles{std::move(*pt), std::move(names), std::move(*rep)};
}

std::string op_string(magmaan::parse::Op op) {
  return std::string(magmaan::parse::to_string(op));
}

const magmaan::inference::ScoreTestResult*
find_mi_row(const magmaan::inference::ScoreTestTable& table,
            const magmaan::spec::LatentNames& names,
            const nlohmann::json& want) {
  auto var_name = [&](std::int32_t v) -> std::string {
    if (v >= 0 && static_cast<std::size_t>(v) < names.var_name.size()) {
      return names.var_name[static_cast<std::size_t>(v)];
    }
    return {};
  };
  for (const auto& row : table.rows) {
    if (row.candidate.kind != magmaan::inference::ScoreCandidateKind::FixedParam) {
      continue;
    }
    const std::size_t r = row.candidate.row;
    const std::string lhs = r < names.row_lhs.size() ? names.row_lhs[r]
                                                     : var_name(row.candidate.lhs_var);
    const std::string rhs = r < names.row_rhs.size() ? names.row_rhs[r]
                                                     : var_name(row.candidate.rhs_var);
    if (lhs == want["lhs"].get<std::string>() &&
        rhs == want["rhs"].get<std::string>() &&
        op_string(row.candidate.op) == want["op"].get<std::string>() &&
        row.candidate.group == want["group"].get<int>()) {
      return &row;
    }
  }
  return nullptr;
}

magmaan::estimate::Estimates estimates_from_fixture(const nlohmann::json& fit) {
  return magmaan::estimate::Estimates{vector_from_json(fit["theta_hat"]), 0.0, 0};
}

bool compare_modindices(const std::string& id,
                        const magmaan::inference::ScoreTestTable& got,
                        const magmaan::spec::LatentNames& names,
                        const nlohmann::json& want_rows,
                        double mi_tol,
                        double epc_tol,
                        std::vector<std::string>& failures) {
  for (const auto& want : want_rows) {
    const auto* row = find_mi_row(got, names, want);
    if (row == nullptr) {
      failures.push_back(id + ": missing MI row " +
                         want["lhs"].get<std::string>() + " " +
                         want["op"].get<std::string>() + " " +
                         want["rhs"].get<std::string>());
      return false;
    }
    const double d_mi = std::abs(row->mi - want["mi"].get<double>());
    const double d_epc = std::abs(row->epc - want["epc"].get<double>());
    if (d_mi > mi_tol || d_epc > epc_tol) {
      failures.push_back(id + ": MI mismatch d_mi=" + std::to_string(d_mi) +
                         " d_epc=" + std::to_string(d_epc) + " got_mi=" +
                         std::to_string(row->mi) + " want_mi=" +
                         std::to_string(want["mi"].get<double>()));
      return false;
    }
  }
  return true;
}

bool compare_score_tests(const std::string& id,
                         const magmaan::inference::ScoreTestTable& got,
                         const nlohmann::json& want,
                         double mi_tol,
                         double p_tol,
                         std::vector<std::string>& failures) {
  const auto& rows = want["rows"];
  if (got.rows.size() != rows.size()) {
    failures.push_back(id + ": score-test row count mismatch");
    return false;
  }
  for (std::size_t i = 0; i < got.rows.size(); ++i) {
    const double d_mi = std::abs(got.rows[i].mi - rows[i]["mi"].get<double>());
    const double d_p =
        std::abs(got.rows[i].p_value - rows[i]["pvalue"].get<double>());
    if (got.rows[i].df != rows[i]["df"].get<int>() ||
        d_mi > mi_tol || d_p > p_tol) {
      failures.push_back(id + ": score-test mismatch d_mi=" +
                         std::to_string(d_mi) + " d_p=" +
                         std::to_string(d_p) + " got_mi=" +
                         std::to_string(got.rows[i].mi) + " want_mi=" +
                         std::to_string(rows[i]["mi"].get<double>()));
      return false;
    }
  }
  return true;
}

}  // namespace

TEST_CASE("score/modification-index goldens match lavaan fixed-row and equality-release targets") {
  const std::string dir = magmaan::test::fixtures_dir() + "/score";
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kScoreFixtures) {
    auto raw_json = magmaan::test::read_fixture(dir + "/" + id + ".score.json");
    REQUIRE(raw_json.has_value());
    auto exp = nlohmann::json::parse(*raw_json, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());

    auto h = build_handles(id, exp, failures);
    if (!h.has_value()) continue;

    const std::string kind = exp["kind"].get<std::string>();
    const auto& fit = exp["fit"];
    magmaan::inference::ScoreTestTable mi;
    magmaan::inference::ScoreTestTable st;
    magmaan::inference::ModificationIndexOptions mi_opts;
    mi_opts.candidates = magmaan::inference::ScoreCandidateSet::WithAbsentRows;
    bool ok = true;

    if (kind == "ml") {
      const auto samp = sample_stats_from_fixture(fit);
      const auto est = estimates_from_fixture(fit);
      auto mi_or = magmaan::inference::modification_indices(h->pt, h->rep, samp,
                                                            est, mi_opts);
      auto st_or = magmaan::inference::score_tests(h->pt, h->rep, samp, est);
      if (!mi_or.has_value() || !st_or.has_value()) {
        failures.push_back(id + ": score path failed");
        continue;
      }
      mi = std::move(*mi_or);
      st = std::move(*st_or);
      ok = compare_modindices(id, mi, h->names, fit["modindices"], 2e-2,
                              5e-3, failures) && ok;
      ok = compare_score_tests(id, st, fit["score_tests"], 2e-2, 5e-3,
                               failures) && ok;
    } else if (kind == "fiml") {
      const auto raw = raw_from_fixture(exp);
      const auto est = estimates_from_fixture(fit);
      auto mi_or = magmaan::inference::modification_indices_fiml(h->pt, h->rep, raw,
                                                           est, mi_opts);
      auto st_or = magmaan::inference::score_tests_fiml(h->pt, h->rep, raw, est);
      if (!mi_or.has_value() || !st_or.has_value()) {
        failures.push_back(id + ": FIML score path failed");
        continue;
      }
      mi = std::move(*mi_or);
      st = std::move(*st_or);
      ok = compare_modindices(id, mi, h->names, fit["modindices"], 2e-1,
                              3e-2, failures) && ok;
      ok = compare_score_tests(id, st, fit["score_tests"], 2e-1, 3e-2,
                               failures) && ok;
    } else if (kind == "ls") {
      const auto samp = sample_stats_from_fixture(fit);
      const auto est = estimates_from_fixture(fit);
      auto mi_or = magmaan::inference::modification_indices(
          h->pt, h->rep, samp, est, magmaan::estimate::gmm::Weight{}, mi_opts);
      auto st_or = magmaan::inference::score_tests(
          h->pt, h->rep, samp, est, magmaan::estimate::gmm::Weight{});
      if (!mi_or.has_value() || !st_or.has_value()) {
        failures.push_back(id + ": ULS score path failed");
        continue;
      }
      mi = std::move(*mi_or);
      st = std::move(*st_or);
      ok = compare_modindices(id, mi, h->names, fit["modindices"], 7e-2,
                              1e-2, failures) && ok;
      ok = compare_score_tests(id, st, fit["score_tests"], 5e-2, 1e-2,
                               failures) && ok;
    } else if (kind == "ordinal") {
      const auto blocks = data_blocks_from_fixture(exp);
      auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks);
      if (!stats.has_value()) {
        failures.push_back(id + ": ordinal stats - " + stats.error().detail);
        continue;
      }
      const auto est = estimates_from_fixture(fit);
      auto mi_or = magmaan::estimate::modification_indices_ordinal(
          h->pt, h->rep, *stats, est,
          magmaan::estimate::OrdinalWeightKind::DWLS, mi_opts);
      auto st_or = magmaan::estimate::score_tests_ordinal(
          h->pt, h->rep, *stats, est,
          magmaan::estimate::OrdinalWeightKind::DWLS);
      if (!mi_or.has_value() || !st_or.has_value()) {
        failures.push_back(id + ": ordinal score path failed");
        continue;
      }
      mi = std::move(*mi_or);
      st = std::move(*st_or);
      ok = compare_modindices(id, mi, h->names, fit["modindices"], 1e-1,
                              2e-2, failures) && ok;
      ok = compare_score_tests(id, st, fit["score_tests"], 1e-1, 2e-2,
                               failures) && ok;
    } else if (kind == "mixed_ordinal") {
      const auto blocks = data_blocks_from_fixture(exp);
      const auto ordered = ordered_masks_from_fixture(exp);
      auto stats = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered);
      if (!stats.has_value()) {
        failures.push_back(id + ": mixed stats - " + stats.error().detail);
        continue;
      }
      const auto est = estimates_from_fixture(fit);
      auto mi_or = magmaan::estimate::modification_indices_mixed_ordinal(
          h->pt, h->rep, *stats, est,
          magmaan::estimate::OrdinalWeightKind::DWLS, mi_opts);
      auto st_or = magmaan::estimate::score_tests_mixed_ordinal(
          h->pt, h->rep, *stats, est,
          magmaan::estimate::OrdinalWeightKind::DWLS);
      if (!mi_or.has_value() || !st_or.has_value()) {
        std::string detail;
        if (!mi_or.has_value()) detail += " MI: " + mi_or.error().detail;
        if (!st_or.has_value()) detail += " score: " + st_or.error().detail;
        failures.push_back(id + ": mixed score path failed -" + detail);
        continue;
      }
      mi = std::move(*mi_or);
      st = std::move(*st_or);
      ok = compare_modindices(id, mi, h->names, fit["modindices"], 1.1,
                              5e-2, failures) && ok;
      ok = compare_score_tests(id, st, fit["score_tests"], 3e-1, 5e-2,
                               failures) && ok;
    } else {
      failures.push_back(id + ": unknown kind " + kind);
      continue;
    }

    if (ok) ++passed;
  }

  MESSAGE("score goldens: " << passed << " / " << kScoreFixtures.size()
                            << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == static_cast<int>(kScoreFixtures.size()));
}
