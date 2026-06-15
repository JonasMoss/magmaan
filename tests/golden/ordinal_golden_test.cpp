#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/measures/effects.hpp"
#include "magmaan/measures/factor_scores.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

const std::vector<std::string> kOrdinalFixtures = {
    "0001_3cat_cfa",
    "0002_4cat_skewed_cfa",
    "0003_sparse_nonempty_pairs",
    "0004_2group_3cat_cfa",
    "0005_near_empty_5cat_cfa",
    "0006_equal_loading_3cat_cfa",
    "0007_binary_cfa",
    "0008_mixed_levels_cfa",
    "0009_sparse_binary_pair",
    "0010_near_perfect_pair",
    "0011_sixcat_threshold_heavy_cfa",
    "0012_2group_equal_loading_3cat_cfa",
    "0013_2group_threshold_invariance_3cat_cfa",
    "0014_threshold_linear_constraint_3cat_cfa",
    "0015_defined_param_3cat_cfa",
    "0016_std_lv_3cat_cfa",
};

const std::vector<std::string> kMixedOrdinalFixtures = {
    "0001_mixed_cfa",
    "0002_sparse_4cat_listwise_mixed_cfa",
};

using magmaan::test::matrix_from_json;

using magmaan::test::vector_from_json;

std::string ordinal_syntax(const nlohmann::json& exp) {
  std::string src = exp["input"].get<std::string>();
  if (!src.empty() && src.back() != '\n') src.push_back('\n');

  const auto& first_block = exp["blocks"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(first_block[0].size());

  // Fixtures with labeled or constrained thresholds carry the threshold rows
  // in the model string itself; only append auto-generated plain rows when
  // the input has none.
  const bool input_has_thresholds =
      exp.contains("input_has_thresholds") &&
      exp["input_has_thresholds"].get<bool>();
  if (!input_has_thresholds) {
    std::vector<int> max_level(static_cast<std::size_t>(p), 0);
    for (const auto& block : exp["blocks"]) {
      const auto& X = block["matrix"];
      for (const auto& row : X) {
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
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    src += "x" + std::to_string(j + 1) + " ~*~ 1*x" +
           std::to_string(j + 1) + "\n";
  }
  return src;
}

// lavaan's categorical LS test statistic is sum_g (n_g - 1) * F_g
// (lav_model_objective.R: `group.fx = 0.5 * (nobs - 1)/nobs * group.fx` for
// ULS/DWLS/WLS, combined by weighted.mean with weights n_g), while magmaan's
// documented convention (docs/design/numerical-conventions.md) is
// chisq = 2 * N * fmin = sum_g n_g * F_g. Rescale magmaan's statistic by the
// global (N - G)/N before comparing; the per-group-vs-global residual is
// O(G/N * spread of F_g) and is covered by the 5e-3 gates below. This mirrors
// the continuous GLS/WLS golden convention.
double to_lavaan_ls_chisq(double magmaan_chisq,
                          std::int64_t n_total,
                          std::size_t n_groups) {
  return magmaan_chisq *
         static_cast<double>(n_total - static_cast<std::int64_t>(n_groups)) /
         static_cast<double>(n_total);
}

// lavaan's (n_g - 1)/N estimation weights differ from magmaan's n_g/N. With
// cross-group equality constraints the two weightings produce genuinely
// (slightly) different estimators, shifting theta_hat by O(1/n_g); without
// cross-group coupling the minimizers coincide. Fixtures whose models couple
// groups through constraints get the looser documented bound.
double ordinal_theta_tol(const std::string& id) {
  if (id == "0013_2group_threshold_invariance_3cat_cfa") return 1.5e-4;
  return 1e-5;
}

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  if (a.size() == 0) return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  if (a.size() == 0) return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

bool has_fit(const nlohmann::json& exp) {
  return exp.contains("fits") && !exp["fits"].is_null();
}

Eigen::VectorXd ordinal_moments(const magmaan::data::OrdinalStats& stats,
                                std::size_t b) {
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  Eigen::VectorXd out(nth + p * (p - 1) / 2);
  out.head(nth) = stats.thresholds[b];
  Eigen::Index k = nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = stats.R[b](i, j);
    }
  }
  return out;
}

struct OrdinalHandles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::data::OrdinalStats stats;
};

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
    const auto& X = block["matrix"];
    for (const auto& row : X) {
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

struct MixedOrdinalHandles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::data::MixedOrdinalStats stats;
};

std::optional<MixedOrdinalHandles> mixed_handles_from_fixture(
    const std::string& id,
    const nlohmann::json& exp,
    std::vector<std::string>& failures) {
  std::vector<Eigen::MatrixXd> blocks;
  for (const auto& b : exp["blocks"]) blocks.push_back(matrix_from_json(b["matrix"]));
  std::vector<std::vector<std::int32_t>> ordered;
  for (const auto& b : exp["ordered_mask"]) {
    std::vector<std::int32_t> mask;
    for (const auto& z : b["mask"]) mask.push_back(z.get<std::int32_t>());
    ordered.push_back(std::move(mask));
  }
  auto stats_or = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered);
  if (!stats_or.has_value()) {
    failures.push_back(id + ": mixed_ordinal_stats_from_data — " +
                       stats_or.error().detail);
    return std::nullopt;
  }
  auto fp = magmaan::parse::Parser::parse(mixed_ordinal_syntax(exp));
  if (!fp.has_value()) {
    failures.push_back(id + ": parse");
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = static_cast<std::int32_t>(exp["blocks"].size());
  auto pt = magmaan::spec::build(*fp, opts);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify — " + pt.error().detail);
    return std::nullopt;
  }
  auto mr = magmaan::model::build_matrix_rep(*pt);
  if (!mr.has_value()) {
    failures.push_back(id + ": matrix_rep — " + mr.error().detail);
    return std::nullopt;
  }
  return MixedOrdinalHandles{std::move(*pt), std::move(*mr), std::move(*stats_or)};
}

std::optional<OrdinalHandles> handles_from_fixture(
    const std::string& id,
    const nlohmann::json& exp,
    std::vector<std::string>& failures) {
  std::vector<Eigen::MatrixXd> blocks;
  for (const auto& b : exp["blocks"]) {
    blocks.push_back(matrix_from_json(b["matrix"]));
  }
  auto stats_or = magmaan::data::ordinal_stats_from_integer_data(blocks);
  if (!stats_or.has_value()) {
    failures.push_back(id + ": ordinal_stats_from_integer_data — " +
                       stats_or.error().detail);
    return std::nullopt;
  }

  auto fp = magmaan::parse::Parser::parse(ordinal_syntax(exp));
  if (!fp.has_value()) {
    failures.push_back(id + ": parse");
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.n_groups = static_cast<std::int32_t>(exp["blocks"].size());
  if (opts.n_groups > 1) {
    opts.group_var = exp["group_var"].get<std::string>();
    for (const auto& b : exp["blocks"]) {
      opts.group_labels.push_back(b["label"].get<std::string>());
    }
  }
  auto pt = magmaan::spec::build(*fp, opts);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify — " + pt.error().detail);
    return std::nullopt;
  }
  auto mr = magmaan::model::build_matrix_rep(*pt);
  if (!mr.has_value()) {
    failures.push_back(id + ": matrix_rep — " + mr.error().detail);
    return std::nullopt;
  }
  return OrdinalHandles{std::move(*pt), std::move(*mr), std::move(*stats_or)};
}

// Build single-group handles from one data block of a multi-group ordinal
// fixture, reusing the fixture's model syntax. Used by the multi-group
// factor-score self-consistency check: for a fixture with no cross-group
// coupling, the per-group multi-group fit equals the independent single-group
// fit, so the multi-group EBM scores for block b must match scoring that block
// as a stand-alone single-group fit (which the lavaan EBM oracle gates).
std::optional<OrdinalHandles> single_group_handles_from_block(
    const std::string& id,
    const nlohmann::json& exp,
    const Eigen::MatrixXd& block,
    std::vector<std::string>& failures) {
  std::vector<Eigen::MatrixXd> blocks{block};
  auto stats_or = magmaan::data::ordinal_stats_from_integer_data(blocks);
  if (!stats_or.has_value()) {
    failures.push_back(id + " (single-group sub-fit): stats — " +
                       stats_or.error().detail);
    return std::nullopt;
  }
  auto fp = magmaan::parse::Parser::parse(ordinal_syntax(exp));
  if (!fp.has_value()) {
    failures.push_back(id + " (single-group sub-fit): parse");
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.n_groups = 1;
  auto pt = magmaan::spec::build(*fp, opts);
  if (!pt.has_value()) {
    failures.push_back(id + " (single-group sub-fit): lavaanify — " +
                       pt.error().detail);
    return std::nullopt;
  }
  auto mr = magmaan::model::build_matrix_rep(*pt);
  if (!mr.has_value()) {
    failures.push_back(id + " (single-group sub-fit): matrix_rep — " +
                       mr.error().detail);
    return std::nullopt;
  }
  return OrdinalHandles{std::move(*pt), std::move(*mr), std::move(*stats_or)};
}

}  // namespace

TEST_CASE("ordinal goldens: thresholds, polychorics, NACOV, and WLS weights vs lavaan") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ordinal";

  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kOrdinalFixtures) {
    const std::string path = dir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) {
      failures.push_back(id + ": missing fixture");
      continue;
    }
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(id + ": invalid JSON");
      continue;
    }

    std::vector<Eigen::MatrixXd> blocks;
    for (const auto& b : exp["blocks"]) {
      blocks.push_back(matrix_from_json(b["matrix"]));
    }

    auto stats_or = magmaan::data::ordinal_stats_from_integer_data(blocks);
    if (!stats_or.has_value()) {
      failures.push_back(id + ": ordinal_stats_from_integer_data — " +
                         stats_or.error().detail);
      continue;
    }
    const auto& stats = *stats_or;
    if (stats.R.size() != exp["sample_stats"].size()) {
      failures.push_back(id + ": block count mismatch");
      continue;
    }

    bool ok = true;
    for (std::size_t b = 0; b < stats.R.size(); ++b) {
      const auto& eb = exp["sample_stats"][b];
      const Eigen::VectorXd th = vector_from_json(eb["thresholds"]);
      const Eigen::MatrixXd R = matrix_from_json(eb["polychoric"]);
      const Eigen::MatrixXd NACOV = matrix_from_json(eb["NACOV"]);
      const Eigen::MatrixXd W = matrix_from_json(eb["WLS.V"]);
      const Eigen::MatrixXd WD = matrix_from_json(eb["WLS.VD"]);
      const Eigen::VectorXd moments = vector_from_json(eb["moments"]);

      const double d_th = max_abs_diff(stats.thresholds[b], th);
      const double d_R = max_abs_diff(stats.R[b], R);
      const double d_N = max_abs_diff(stats.NACOV[b], NACOV);
      const double d_WD = max_abs_diff(stats.W_dwls[b], WD);
      const double d_W = max_abs_diff(stats.W_wls[b], W);
      const double d_mom = max_abs_diff(ordinal_moments(stats, b), moments);

      if (d_th > 5e-8 || d_R > 5e-4 || d_N > 2e-2 ||
          d_WD > 2e-2 || d_W > 2e-2 || d_mom > 5e-4) {
        failures.push_back(
            id + " block " + std::to_string(b) +
            ": max diffs thresholds=" + std::to_string(d_th) +
            " R=" + std::to_string(d_R) +
            " moments=" + std::to_string(d_mom) +
            " NACOV=" + std::to_string(d_N) +
            " WLS.VD=" + std::to_string(d_WD) +
            " WLS.V=" + std::to_string(d_W));
        ok = false;
        break;
      }
    }
    if (ok) ++passed;
  }

  MESSAGE("ordinal goldens: " << passed << " / " << kOrdinalFixtures.size()
          << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == static_cast<int>(kOrdinalFixtures.size()));
}

TEST_CASE("mixed ordinal goldens: thresholds, polyserials, NACOV, and WLS weights vs lavaan") {
  const std::string dir = magmaan::test::fixtures_dir() + "/mixed_ordinal";

  int passed = 0;
  std::vector<std::string> failures;
  for (const auto& id : kMixedOrdinalFixtures) {
    const std::string path = dir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) {
      failures.push_back(id + ": missing fixture");
      continue;
    }
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(id + ": invalid JSON");
      continue;
    }
    std::vector<Eigen::MatrixXd> blocks;
    for (const auto& b : exp["blocks"]) blocks.push_back(matrix_from_json(b["matrix"]));
    std::vector<std::vector<std::int32_t>> ordered;
    for (const auto& b : exp["ordered_mask"]) {
      std::vector<std::int32_t> mask;
      for (const auto& z : b["mask"]) mask.push_back(z.get<std::int32_t>());
      ordered.push_back(std::move(mask));
    }
    auto stats_or = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered);
    if (!stats_or.has_value()) {
      failures.push_back(id + ": mixed_ordinal_stats_from_data — " +
                         stats_or.error().detail);
      continue;
    }
    const auto& stats = *stats_or;
    bool ok = true;
    for (std::size_t b = 0; b < stats.R.size(); ++b) {
      const auto& eb = exp["sample_stats"][b];
      const Eigen::VectorXd th = vector_from_json(eb["thresholds"]);
      const Eigen::VectorXd mean = vector_from_json(eb["mean"]);
      const Eigen::MatrixXd R = matrix_from_json(eb["cov"]);
      const Eigen::VectorXd moments = vector_from_json(eb["moments"]);
      const Eigen::MatrixXd NACOV = matrix_from_json(eb["NACOV"]);
      const Eigen::MatrixXd W = matrix_from_json(eb["WLS.V"]);
      const Eigen::MatrixXd WD = matrix_from_json(eb["WLS.VD"]);
      const double d_th = max_abs_diff(stats.thresholds[b], th);
      const double d_mean = max_abs_diff(stats.mean[b], mean);
      const double d_R = max_abs_diff(stats.R[b], R);
      const double d_mom = max_abs_diff(stats.moments[b], moments);
      const double d_N = max_abs_diff(stats.NACOV[b], NACOV);
      const double d_WD = max_abs_diff(stats.W_dwls[b], WD);
      const double d_W = max_abs_diff(stats.W_wls[b], W);
      if (d_th > 5e-8 || d_mean > 5e-8 || d_R > 1e-6 || d_mom > 1e-6 ||
          d_N > 1e-6 || d_WD > 1e-6 || d_W > 1e-5) {
        failures.push_back(id + " block " + std::to_string(b) +
                           ": max diffs thresholds=" + std::to_string(d_th) +
                           " mean=" + std::to_string(d_mean) +
                           " R=" + std::to_string(d_R) +
                           " moments=" + std::to_string(d_mom) +
                           " NACOV=" + std::to_string(d_N) +
                           " WLS.VD=" + std::to_string(d_WD) +
                           " WLS.V=" + std::to_string(d_W));
        ok = false;
        break;
      }
    }
    if (ok) ++passed;
  }
  MESSAGE("mixed ordinal goldens: " << passed << " / "
          << kMixedOrdinalFixtures.size() << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == static_cast<int>(kMixedOrdinalFixtures.size()));
}

TEST_CASE("ordinal fixtures: DWLS/WLS bounded fits match lavaan delta contract") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ordinal";

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kOrdinalFixtures) {
    const std::string path = dir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!has_fit(exp)) continue;

    auto h = handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    const magmaan::optim::OptimOptions opt{
        .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

    for (const auto& fit_item : exp["fits"].items()) {
      const std::string name = fit_item.key();
      if (name != "DWLS" && name != "WLS") {
        failures.push_back(id + ": unknown ordinal fit kind " + name);
        continue;
      }
      const auto kind = name == "DWLS"
          ? magmaan::estimate::OrdinalWeightKind::DWLS
          : magmaan::estimate::OrdinalWeightKind::WLS;
      ++total;
      auto est_or = magmaan::test::fit_ordinal_bounded(
          h->pt, h->rep, h->stats, magmaan::estimate::Bounds{}, kind,
          magmaan::estimate::Backend::NloptLbfgs, opt);
      if (!est_or.has_value()) {
        failures.push_back(id + " " + name + ": fit — " + est_or.error().detail);
        continue;
      }
      const double lavaan_chisq = exp["fits"][name]["chisq"].get<double>();
      const int lavaan_df = exp["fits"][name]["df"].get<int>();
      const Eigen::VectorXd lavaan_theta =
          vector_from_json(exp["fits"][name]["theta_hat"]);
      const std::int64_t n_total =
          std::accumulate(h->stats.n_obs.begin(), h->stats.n_obs.end(),
                          std::int64_t{0});
      Eigen::Index n_moments = 0;
      for (std::size_t b = 0; b < h->stats.R.size(); ++b) {
        const Eigen::Index p = h->stats.R[b].rows();
        n_moments += h->stats.thresholds[b].size() + p * (p - 1) / 2;
      }
      auto pt_for_df = h->pt;
      auto prep_for_df =
          magmaan::estimate::prepare_ordinal_delta_partable(pt_for_df, h->stats);
      if (!prep_for_df.has_value()) {
        failures.push_back(id + " " + name + ": df prep — " +
                           prep_for_df.error().detail);
        continue;
      }
      auto con_or = magmaan::estimate::build_eq_constraints(pt_for_df);
      if (!con_or.has_value()) {
        failures.push_back(id + " " + name + ": df constraints — " +
                           con_or.error().detail);
        continue;
      }
      const int df = static_cast<int>(n_moments - con_or->n_alpha);
      const double chisq = to_lavaan_ls_chisq(
          2.0 * static_cast<double>(n_total) * est_or->fmin, n_total,
          h->stats.R.size());
      const double d_chisq = std::abs(chisq - lavaan_chisq);
      const double d_theta = max_abs_diff(est_or->theta, lavaan_theta);
      const double theta_tol = ordinal_theta_tol(id);
      const double chisq_tol = 5e-3;

      if (!est_or->theta.allFinite() || !std::isfinite(est_or->fmin) ||
          est_or->fmin < 0.0) {
        failures.push_back(id + " " + name + ": non-finite fit result");
        continue;
      }
      if (est_or->theta.size() != lavaan_theta.size()) {
        failures.push_back(id + " " + name +
                           ": npar mismatch magmaan=" +
                           std::to_string(est_or->theta.size()) +
                           " lavaan=" + std::to_string(lavaan_theta.size()));
        continue;
      }
      if (df != lavaan_df || d_theta > theta_tol ||
          d_chisq > chisq_tol) {
        failures.push_back(id + " " + name +
                           ": df=" + std::to_string(df) +
                           " lavaan_df=" + std::to_string(lavaan_df) +
                           ": theta diff=" + std::to_string(d_theta) +
                           " chisq diff=" + std::to_string(d_chisq));
        continue;
      }
      if (fit_item.value().contains("robust") &&
          !fit_item.value()["robust"].is_null()) {
        auto rob_or = magmaan::estimate::robust_ordinal(
            h->pt, h->rep, h->stats, *est_or, kind);
        if (!rob_or.has_value()) {
          failures.push_back(id + " " + name + ": robust — " +
                             rob_or.error().detail);
          continue;
        }
        const auto& robust = fit_item.value()["robust"];
        const Eigen::VectorXd lavaan_se = vector_from_json(robust["se"]);
        const Eigen::VectorXd lavaan_ev = vector_from_json(robust["eigvals"]);
        const double d_se = max_abs_diff(rob_or->se, lavaan_se);
        const double d_ev = max_abs_diff(rob_or->eigvals, lavaan_ev);
        // chisq-scale robust quantities (standard, SB, mean-var, scaled-
        // shifted statistics and the additive shift) carry the same
        // N-vs-(N-G) convention gap as the plain statistic; scale factors,
        // adjusted df, and eigenvalues are ratio-scale and unaffected.
        const auto ls = [&](double v) {
          return to_lavaan_ls_chisq(v, n_total, h->stats.R.size());
        };
        const double d_standard =
            std::abs(ls(rob_or->chisq_standard) -
                     robust["chisq_standard"].get<double>());
        const double d_sb =
            std::abs(ls(rob_or->satorra_bentler.chi2_scaled) -
                     robust["satorra_bentler"]["chisq"].get<double>());
        const double d_sb_scale =
            std::abs(rob_or->satorra_bentler.scale_c -
                     robust["satorra_bentler"]["scale"].get<double>());
        const double d_mv =
            std::abs(ls(rob_or->mean_var_adjusted.chi2_adj) -
                     robust["mean_var_adjusted"]["chisq"].get<double>());
        const double d_mv_df =
            std::abs(rob_or->mean_var_adjusted.df_adj -
                     robust["mean_var_adjusted"]["df_adj"].get<double>());
        const double d_ss =
            std::abs(ls(rob_or->scaled_shifted.chi2_adj) -
                     robust["scaled_shifted"]["chisq"].get<double>());
        const double d_ss_scale =
            std::abs((1.0 / rob_or->scaled_shifted.scale_a) -
                     robust["scaled_shifted"]["scale"].get<double>());
        // The shift solves a trace system in df and UGamma eigenvalue
        // products, all N-free, so it carries no chisq-convention factor.
        const double d_ss_shift =
            std::abs(rob_or->scaled_shifted.shift_b -
                     robust["scaled_shifted"]["shift"].get<double>());
        const double robust_se_tol = 3e-4;
        const double robust_ev_tol = 2e-4;
        const double robust_scale_tol = 2e-4;
        const double robust_chisq_tol = 5e-3;
        // 0013's estimator itself differs from lavaan's by the (n_g - 1)/n_g
        // group-weighting convention (see ordinal_theta_tol); the scaled-
        // shifted solve amplifies that O(1/n) theta difference through its
        // trace products. Measured 8.5e-3 on the fixture data.
        const bool weighting_convention_fixture =
            id == "0013_2group_threshold_invariance_3cat_cfa";
        const double robust_ss_tol =
            weighting_convention_fixture ? 1.5e-2 : robust_chisq_tol;
        const double robust_ss_shift_tol =
            weighting_convention_fixture ? 1.5e-2 : 2e-4;

        if (rob_or->df != robust["df"].get<int>() ||
            rob_or->satorra_bentler.df !=
                robust["satorra_bentler"]["df"].get<int>() ||
            rob_or->scaled_shifted.df !=
                robust["scaled_shifted"]["df"].get<int>() ||
            d_se > robust_se_tol || d_ev > robust_ev_tol ||
            d_standard > robust_chisq_tol ||
            d_sb > robust_chisq_tol || d_sb_scale > robust_scale_tol ||
            d_mv > robust_chisq_tol || d_mv_df > 2e-4 ||
            d_ss > robust_ss_tol || d_ss_scale > robust_scale_tol ||
            d_ss_shift > robust_ss_shift_tol) {
          failures.push_back(
              id + " " + name +
              ": robust diffs se=" + std::to_string(d_se) +
              " eig=" + std::to_string(d_ev) +
              " standard=" + std::to_string(d_standard) +
              " SB=" + std::to_string(d_sb) +
              " SB scale=" + std::to_string(d_sb_scale) +
              " mean.var=" + std::to_string(d_mv) +
              " mean.var df=" + std::to_string(d_mv_df) +
              " scaled.shifted=" + std::to_string(d_ss) +
              " scaled.shifted scale=" + std::to_string(d_ss_scale) +
              " scaled.shifted shift=" + std::to_string(d_ss_shift));
          continue;
        }
      }
      MESSAGE(id << " " << name << ": magmaan chisq=" << chisq
                 << " lavaan chisq=" << lavaan_chisq
                 << " diff=" << d_chisq
                 << " theta diff=" << d_theta
                 << " df=" << df
                 << " npar(magmaan)=" << est_or->theta.size()
                 << " npar(lavaan)=" << exp["fits"][name]["theta_hat"].size());
      ++passed;
    }
  }

  MESSAGE("ordinal fixture fits: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}

// The threshold-profiled SNLLS path against the same lavaan contract as the
// bounded golden above. This is the lavaan anchor for the joint threshold
// design: free thresholds, cross-group threshold invariance (0013), and
// threshold-only linear equality constraints (0014) all flow through
// fit_ordinal_snlls.
TEST_CASE("ordinal fixtures: profiled SNLLS fits match lavaan delta contract") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ordinal";

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kOrdinalFixtures) {
    const std::string path = dir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!has_fit(exp)) continue;

    auto h = handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    std::vector<Eigen::MatrixXd> blocks;
    for (const auto& b : exp["blocks"]) {
      blocks.push_back(matrix_from_json(b["matrix"]));
    }

    const magmaan::optim::OptimOptions opt{
        .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

    for (const auto& fit_item : exp["fits"].items()) {
      const std::string name = fit_item.key();
      if (name != "DWLS" && name != "WLS") continue;
      const auto estimator = name == "DWLS"
          ? magmaan::data::OrdinalEstimatorKind::DWLS
          : magmaan::data::OrdinalEstimatorKind::WLS;
      ++total;

      auto plan = magmaan::data::ordinal_weight_plan(
          magmaan::data::OrdinalWorkspacePurpose::FitOnly, estimator);
      auto workspace =
          magmaan::data::ordinal_workspace_from_integer_data(blocks, plan);
      if (!workspace.has_value()) {
        failures.push_back(id + " " + name + ": workspace — " +
                           workspace.error().detail);
        continue;
      }
      auto x0 = magmaan::estimate::ordinal_start_values(
          h->pt, h->rep, workspace->moments, {});
      if (!x0.has_value()) {
        failures.push_back(id + " " + name + ": starts — " +
                           x0.error().detail);
        continue;
      }
      auto est_or = magmaan::estimate::fit_ordinal_snlls(
          h->pt, h->rep, workspace->moments, &workspace->gamma_cache, plan,
          *x0, magmaan::estimate::Backend::NloptLbfgs, opt);
      if (!est_or.has_value()) {
        // Under std.lv delta identification the latent variance is fixed to 1
        // and the delta `~*~ 1` scaling fixes the response (Theta) variances,
        // so every free structural parameter is a loading (the nonlinear β
        // block). The Golub-Pereyra separation eliminates the conditionally-
        // linear {Psi, Theta, ...} block, which is empty here, so SNLLS is
        // structurally inapplicable. The bounded full-Newton fit (gated in the
        // delta-contract golden above) carries the lavaan θ̂/threshold parity;
        // pin the expected diagnostic rather than silently skipping the arm.
        if (id == "0016_std_lv_3cat_cfa" &&
            est_or.error().detail.find("no conditionally linear") !=
                std::string::npos) {
          ++passed;
          continue;
        }
        failures.push_back(id + " " + name + ": snlls — " +
                           est_or.error().detail);
        continue;
      }

      const double lavaan_chisq = exp["fits"][name]["chisq"].get<double>();
      const Eigen::VectorXd lavaan_theta =
          vector_from_json(exp["fits"][name]["theta_hat"]);
      const std::int64_t n_total =
          std::accumulate(h->stats.n_obs.begin(), h->stats.n_obs.end(),
                          std::int64_t{0});
      const double chisq = to_lavaan_ls_chisq(
          2.0 * static_cast<double>(n_total) * est_or->fmin, n_total,
          h->stats.R.size());
      const double d_chisq = std::abs(chisq - lavaan_chisq);
      const double d_theta = max_abs_diff(est_or->theta, lavaan_theta);

      if (!est_or->theta.allFinite() || !std::isfinite(est_or->fmin) ||
          est_or->fmin < 0.0) {
        failures.push_back(id + " " + name + ": non-finite SNLLS result");
        continue;
      }
      if (est_or->theta.size() != lavaan_theta.size()) {
        failures.push_back(id + " " + name + ": npar mismatch magmaan=" +
                           std::to_string(est_or->theta.size()) + " lavaan=" +
                           std::to_string(lavaan_theta.size()));
        continue;
      }
      if (d_theta > ordinal_theta_tol(id) || d_chisq > 5e-3) {
        failures.push_back(id + " " + name +
                           ": theta diff=" + std::to_string(d_theta) +
                           " chisq=" + std::to_string(chisq) +
                           " lavaan chisq=" + std::to_string(lavaan_chisq));
        continue;
      }
      MESSAGE(id << " " << name << " SNLLS: chisq=" << chisq
                 << " lavaan=" << lavaan_chisq << " theta diff=" << d_theta);
      ++passed;
    }
  }

  MESSAGE("ordinal SNLLS fixture fits: " << passed << " / " << total
                                         << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}

TEST_CASE("ordinal theta parameterization: reparameterization invariance and "
          "lavaan theta loadings") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ordinal";

  // lavaan cfa(input, ordered=, estimator="DWLS", parameterization="theta")
  // free loadings, indicator order (x2, x3, x4); lavaan 0.6.22.
  struct ThetaOracle { const char* id; double loadings[3]; };
  const ThetaOracle oracles[] = {
      {"0001_3cat_cfa",   {0.84027, 0.72418, 0.34726}},
      {"0007_binary_cfa", {1.11134, 0.74956, 0.59699}},
  };

  std::vector<std::string> failures;
  for (const auto& orc : oracles) {
    auto raw = magmaan::test::read_fixture(dir + "/" + orc.id + ".ordinal.json");
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    auto h = handles_from_fixture(orc.id, exp, failures);
    REQUIRE(h.has_value());

    const magmaan::optim::OptimOptions opt{
        .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
    using magmaan::estimate::OrdinalParameterization;
    auto delta = magmaan::test::fit_ordinal_bounded(
        h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, opt,
        OrdinalParameterization::Delta);
    auto theta = magmaan::test::fit_ordinal_bounded(
        h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, opt,
        OrdinalParameterization::Theta);
    REQUIRE(delta.has_value());
    REQUIRE(theta.has_value());

    // Reparameterization invariance — delta and theta minimize the same
    // discrepancy. The delta χ² is lavaan-verified by the golden above, so
    // theta inherits that backing.
    CHECK(theta->fmin == doctest::Approx(delta->fmin).epsilon(1e-3));

    // After the ordinal prep the free θ entries lead with the free `=~`
    // loadings in indicator order — compare them to the lavaan theta oracle.
    auto pt_copy = h->pt;
    auto prep =
        magmaan::estimate::prepare_ordinal_delta_partable(pt_copy, h->stats);
    REQUIRE(prep.has_value());
    int li = 0;
    for (std::size_t r = 0; r < pt_copy.size(); ++r) {
      if (pt_copy.op[r] != magmaan::parse::Op::Measurement) continue;
      if (pt_copy.free[r] <= 0) continue;
      REQUIRE(li < 3);
      const double got = theta->theta(pt_copy.free[r] - 1);
      CHECK(got == doctest::Approx(orc.loadings[li]).epsilon(0.01));
      ++li;
    }
    CHECK(li == 3);
  }
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
}

TEST_CASE("mixed ordinal fixtures: DWLS/WLS bounded fits match lavaan delta contract") {
  const std::string dir = magmaan::test::fixtures_dir() + "/mixed_ordinal";

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kMixedOrdinalFixtures) {
    const std::string path = dir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());

    auto h = mixed_handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    const magmaan::optim::OptimOptions opt{
        .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};

    for (const auto& fit_item : exp["fits"].items()) {
      const std::string name = fit_item.key();
      const auto kind = name == "DWLS"
          ? magmaan::estimate::OrdinalWeightKind::DWLS
          : magmaan::estimate::OrdinalWeightKind::WLS;
      ++total;
      auto est_or = magmaan::test::fit_mixed_ordinal_bounded(
          h->pt, h->rep, h->stats, magmaan::estimate::Bounds{}, kind,
          magmaan::estimate::Backend::NloptLbfgs, opt);
      if (!est_or.has_value()) {
        failures.push_back(id + " " + name + ": fit — " + est_or.error().detail);
        continue;
      }
      const double lavaan_chisq = exp["fits"][name]["chisq"].get<double>();
      const int lavaan_df = exp["fits"][name]["df"].get<int>();
      const Eigen::VectorXd lavaan_theta =
          vector_from_json(exp["fits"][name]["theta_hat"]);
      const std::int64_t n_total =
          std::accumulate(h->stats.n_obs.begin(), h->stats.n_obs.end(),
                          std::int64_t{0});
      Eigen::Index n_moments = 0;
      for (const auto& m : h->stats.moments) n_moments += m.size();
      auto pt_for_df = h->pt;
      auto prep_for_df =
          magmaan::estimate::prepare_mixed_ordinal_delta_partable(pt_for_df, h->stats);
      if (!prep_for_df.has_value()) {
        failures.push_back(id + " " + name + ": df prep — " +
                           prep_for_df.error().detail);
        continue;
      }
      auto con_or = magmaan::estimate::build_eq_constraints(pt_for_df);
      if (!con_or.has_value()) {
        failures.push_back(id + " " + name + ": df constraints — " +
                           con_or.error().detail);
        continue;
      }
      const int df = static_cast<int>(n_moments - con_or->n_alpha);
      const double chisq = to_lavaan_ls_chisq(
          2.0 * static_cast<double>(n_total) * est_or->fmin, n_total,
          h->stats.R.size());
      const double d_chisq = std::abs(chisq - lavaan_chisq);
      const double d_theta = max_abs_diff(est_or->theta, lavaan_theta);
      if (df != lavaan_df || d_theta > 1e-5 || d_chisq > 5e-3) {
        failures.push_back(id + " " + name +
                           ": df=" + std::to_string(df) +
                           " lavaan_df=" + std::to_string(lavaan_df) +
                           " theta diff=" + std::to_string(d_theta) +
                           " chisq diff=" + std::to_string(d_chisq));
        continue;
      }
      if (fit_item.value().contains("robust") &&
          !fit_item.value()["robust"].is_null()) {
        magmaan::estimate::Estimates lavaan_est = *est_or;
        lavaan_est.theta = lavaan_theta;
        // Deliberately maps lavaan's chisq onto magmaan's 2N-convention fmin
        // so the robust standard statistic reproduces lavaan's stored value
        // exactly (gated at 1e-8 below); the scaled tests are then compared
        // free of the sum_g (n_g - 1) F_g vs sum_g n_g F_g convention gap.
        lavaan_est.fmin = lavaan_chisq / (2.0 * static_cast<double>(n_total));
        auto rob_or = magmaan::estimate::robust_mixed_ordinal(
            h->pt, h->rep, h->stats, lavaan_est, kind);
        if (!rob_or.has_value()) {
          failures.push_back(id + " " + name + ": robust — " +
                             rob_or.error().detail);
          continue;
        }
        const auto& robust = fit_item.value()["robust"];
        const Eigen::VectorXd lavaan_se = vector_from_json(robust["se"]);
        const Eigen::VectorXd lavaan_ev = vector_from_json(robust["eigvals"]);
        const double d_se = max_abs_diff(rob_or->se, lavaan_se);
        const double d_ev = max_abs_diff(rob_or->eigvals, lavaan_ev);
        const double d_standard =
            std::abs(rob_or->chisq_standard -
                     robust["chisq_standard"].get<double>());
        const double d_sb =
            std::abs(rob_or->satorra_bentler.chi2_scaled -
                     robust["satorra_bentler"]["chisq"].get<double>());
        const double d_sb_scale =
            std::abs(rob_or->satorra_bentler.scale_c -
                     robust["satorra_bentler"]["scale"].get<double>());
        const double d_mv =
            std::abs(rob_or->mean_var_adjusted.chi2_adj -
                     robust["mean_var_adjusted"]["chisq"].get<double>());
        const double d_mv_df =
            std::abs(rob_or->mean_var_adjusted.df_adj -
                     robust["mean_var_adjusted"]["df_adj"].get<double>());
        const double d_ss =
            std::abs(rob_or->scaled_shifted.chi2_adj -
                     robust["scaled_shifted"]["chisq"].get<double>());
        const double d_ss_scale =
            std::abs((1.0 / rob_or->scaled_shifted.scale_a) -
                     robust["scaled_shifted"]["scale"].get<double>());
        const double d_ss_shift =
            std::abs(rob_or->scaled_shifted.shift_b -
                     robust["scaled_shifted"]["shift"].get<double>());
        // The mixed Gamma construction mirrors lavaan's muthen1984
        // estimating-equation sandwich (stage-1 mu/var ML scores, pair-ML
        // association scores, delta-rule covariance transform), so the mixed
        // robust gates sit at the all-ordinal tightness.
        if (rob_or->df != robust["df"].get<int>() ||
            rob_or->satorra_bentler.df !=
                robust["satorra_bentler"]["df"].get<int>() ||
            rob_or->scaled_shifted.df !=
                robust["scaled_shifted"]["df"].get<int>() ||
            d_se > 3e-4 || d_ev > 2e-4 || d_standard > 1e-8 ||
            d_sb > 5e-3 || d_sb_scale > 2e-4 ||
            d_mv > 5e-3 || d_mv_df > 2e-4 ||
            d_ss > 5e-3 || d_ss_scale > 2e-4 ||
            d_ss_shift > 2e-4) {
          failures.push_back(id + " " + name +
                             ": robust diffs se=" + std::to_string(d_se) +
                             " eig=" + std::to_string(d_ev) +
                             " standard=" + std::to_string(d_standard) +
                             " SB=" + std::to_string(d_sb) +
                             " SB scale=" + std::to_string(d_sb_scale) +
                             " mean.var=" + std::to_string(d_mv) +
                             " mean.var df=" + std::to_string(d_mv_df) +
                             " scaled.shifted=" + std::to_string(d_ss) +
                             " scaled.shifted scale=" +
                             std::to_string(d_ss_scale) +
                             " scaled.shifted shift=" +
                             std::to_string(d_ss_shift));
          continue;
        }

        // End-to-end check at magmaan's own theta-hat: the chisq-scale
        // statistics convert through the (N - G)/N convention rescale, the
        // ratio/df/SE quantities compare directly.
        auto rob_own_or = magmaan::estimate::robust_mixed_ordinal(
            h->pt, h->rep, h->stats, *est_or, kind);
        if (!rob_own_or.has_value()) {
          failures.push_back(id + " " + name + ": robust (own theta) — " +
                             rob_own_or.error().detail);
          continue;
        }
        const auto ls = [&](double v) {
          return to_lavaan_ls_chisq(v, n_total, h->stats.R.size());
        };
        const double o_se = max_abs_diff(rob_own_or->se, lavaan_se);
        const double o_ev = max_abs_diff(rob_own_or->eigvals, lavaan_ev);
        const double o_standard =
            std::abs(ls(rob_own_or->chisq_standard) -
                     robust["chisq_standard"].get<double>());
        const double o_sb =
            std::abs(ls(rob_own_or->satorra_bentler.chi2_scaled) -
                     robust["satorra_bentler"]["chisq"].get<double>());
        const double o_ss =
            std::abs(ls(rob_own_or->scaled_shifted.chi2_adj) -
                     robust["scaled_shifted"]["chisq"].get<double>());
        if (o_se > 3e-4 || o_ev > 2e-4 || o_standard > 5e-3 ||
            o_sb > 5e-3 || o_ss > 5e-3) {
          failures.push_back(id + " " + name +
                             ": robust (own theta) diffs se=" +
                             std::to_string(o_se) +
                             " eig=" + std::to_string(o_ev) +
                             " standard=" + std::to_string(o_standard) +
                             " SB=" + std::to_string(o_sb) +
                             " scaled.shifted=" + std::to_string(o_ss));
          continue;
        }
      }

      auto fm_or = magmaan::estimate::fit_measures_mixed_ordinal(
          h->pt, h->rep, h->stats, *est_or, kind);
      if (!fm_or.has_value()) {
        failures.push_back(id + " " + name + ": fit measures — " +
                           fm_or.error().detail);
        continue;
      }
      const double d_cfi =
          std::abs(fm_or->indices.cfi - fit_item.value()["cfi"].get<double>());
      const double d_tli =
          std::abs(fm_or->indices.tli - fit_item.value()["tli"].get<double>());
      const double d_rmsea = std::abs(fm_or->indices.rmsea -
                                      fit_item.value()["rmsea"].get<double>());
      const double d_srmr =
          std::abs(fm_or->srmr - fit_item.value()["srmr"].get<double>());
      if (d_cfi > 5e-3 || d_tli > 5e-3 || d_rmsea > 5e-3 ||
          d_srmr > 5e-4) {
        failures.push_back(id + " " + name +
                           ": fit-measure diffs cfi=" +
                           std::to_string(d_cfi) +
                           " tli=" + std::to_string(d_tli) +
                           " rmsea=" + std::to_string(d_rmsea) +
                           " srmr=" + std::to_string(d_srmr));
        continue;
      }
      ++passed;
    }
  }

  MESSAGE("mixed ordinal fixture fits: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}

#ifdef MAGMAAN_WITH_CERES
TEST_CASE("ordinal fixtures: Ceres and NLopt L-BFGS bounded fits agree") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ordinal";

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kOrdinalFixtures) {
    const std::string path = dir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!has_fit(exp)) continue;

    auto h = handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    const magmaan::optim::OptimOptions nlopt{
        .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
    const magmaan::optim::OptimOptions ceres{
        .max_iter = 1000, .ftol = 1e-12, .gtol = 1e-8};

    for (const auto& fit_item : exp["fits"].items()) {
      const std::string name = fit_item.key();
      if (name != "DWLS" && name != "WLS") {
        failures.push_back(id + ": unknown ordinal fit kind " + name);
        continue;
      }
      const auto kind = name == "DWLS"
          ? magmaan::estimate::OrdinalWeightKind::DWLS
          : magmaan::estimate::OrdinalWeightKind::WLS;
      ++total;
      auto a = magmaan::test::fit_ordinal_bounded(
          h->pt, h->rep, h->stats, magmaan::estimate::Bounds{}, kind,
          magmaan::estimate::Backend::NloptLbfgs, nlopt);
      auto b = magmaan::test::fit_ordinal_bounded(
          h->pt, h->rep, h->stats, magmaan::estimate::Bounds{}, kind,
          magmaan::estimate::Backend::Ceres, ceres);
      if (!a.has_value() || !b.has_value()) {
        failures.push_back(id + " " + name + ": backend fit failed");
        continue;
      }
      const double d_f = std::abs(a->fmin - b->fmin);
      if (d_f > 2e-8) {
        failures.push_back(id + " " + name +
                           ": fmin diff=" + std::to_string(d_f));
        continue;
      }
      MESSAGE(id << " " << name << ": theta diff="
                 << max_abs_diff(a->theta, b->theta)
                 << " fmin diff=" << d_f);
      ++passed;
    }
  }

  MESSAGE("ordinal Ceres/NLopt L-BFGS parity: " << passed << " / " << total
          << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
#endif

// Post-hoc standardized-solution (std.lv / std.all loadings) and defined-
// parameter (`:=`) parity for ordinal and mixed-ordinal DWLS delta fits.
// These gate measures::standardize_{all,lv} and measures::compute_defined
// against lavaan, which previously lived only in the live R example
// (examples/ordinal_dwls_wls.R). Mirrors the api path used by the Rcpp
// binding: estimates/vcov live over the *prepared* (reduced) structure, the
// matrix_rep stays the original, and std.all carries ordinal_delta_unit so a
// categorical indicator's loading is standardized by the latent SD only.
//
// Oracle (per DWLS fit, in free-index order matching theta_hat):
//   fits.DWLS.standardized = {par_op, par_rhs, std_lv_est, std_all_est}
//   fits.DWLS.defined      = [{lhs, est, se}, ...]   (only the `:=` case)
// Only the `=~` loading rows are gated — the proven parity surface; thresholds
// and (co)variances are stored but not asserted.
namespace {

// Gate the `=~` loading rows of standardize_all / standardize_lv against the
// stored lavaan est.std. Returns true on full agreement.
bool check_ordinal_std(const std::string& id,
                       const magmaan::spec::LatentStructure& pt_prep,
                       const magmaan::model::MatrixRep& rep,
                       const magmaan::estimate::Estimates& est,
                       const Eigen::MatrixXd& vcov,
                       const nlohmann::json& std_oracle,
                       double tol,
                       std::vector<std::string>& failures) {
  auto sall = magmaan::measures::standardize::standardize_all(
      pt_prep, rep, est, vcov, /*ordinal_delta_unit=*/true);
  auto slv = magmaan::measures::standardize::standardize_lv(
      pt_prep, rep, est, vcov);
  if (!sall.has_value()) {
    failures.push_back(id + ": standardize_all — " + sall.error().detail);
    return false;
  }
  if (!slv.has_value()) {
    failures.push_back(id + ": standardize_lv — " + slv.error().detail);
    return false;
  }
  const auto& op = std_oracle["par_op"];
  const auto& all_e = std_oracle["std_all_est"];
  const auto& lv_e = std_oracle["std_lv_est"];
  const std::size_t n = op.size();
  if (static_cast<std::size_t>(sall->theta.size()) != n ||
      static_cast<std::size_t>(slv->theta.size()) != n) {
    failures.push_back(id + ": std n_free mismatch (magmaan=" +
                       std::to_string(sall->theta.size()) +
                       " lavaan=" + std::to_string(n) + ")");
    return false;
  }
  bool ok = true;
  char buf[256];
  for (std::size_t k = 0; k < n; ++k) {
    if (op[k].get<std::string>() != "=~") continue;  // gate loadings only
    const auto kk = static_cast<Eigen::Index>(k);
    const double d_all = std::abs(sall->theta(kk) - all_e[k].get<double>());
    const double d_lv = std::abs(slv->theta(kk) - lv_e[k].get<double>());
    if (d_all > tol || d_lv > tol) {
      std::snprintf(buf, sizeof(buf),
                    "%s std loading[%zu] std.all diff=%.2e std.lv diff=%.2e",
                    id.c_str(), k, d_all, d_lv);
      failures.push_back(buf);
      ok = false;
    }
  }
  return ok;
}

}  // namespace

TEST_CASE("ordinal/mixed standardized + := rows match lavaan") {
  const std::string odir = magmaan::test::fixtures_dir() + "/ordinal";
  const std::string mdir = magmaan::test::fixtures_dir() + "/mixed_ordinal";
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  // Loading std.all / std.lv parity tolerance matches the live R parity:
  // all-ordinal 5e-3, mixed (continuous indicators carry the σ_rr division)
  // 1e-3. The := value/SE is a delta-method transform of a vcov matched to
  // ~1e-3, gated at 5e-3.
  const double std_tol_ordinal = 5e-3;
  const double std_tol_mixed = 1e-3;
  const double defined_tol = 5e-3;

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  // --- ordinal fixtures (std.all/std.lv loadings + the := case) -----------
  for (const auto& id : kOrdinalFixtures) {
    const std::string path = odir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!has_fit(exp) || !exp["fits"].contains("DWLS")) continue;
    const auto& dwls = exp["fits"]["DWLS"];
    const bool has_std = dwls.contains("standardized");
    const bool has_def = dwls.contains("defined");
    if (!has_std && !has_def) continue;

    auto h = handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    auto est_or = magmaan::test::fit_ordinal_bounded(
        h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, opt);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    auto rob_or = magmaan::estimate::robust_ordinal(
        h->pt, h->rep, h->stats, *est_or,
        magmaan::estimate::OrdinalWeightKind::DWLS);
    if (!rob_or.has_value()) {
      failures.push_back(id + ": robust — " + rob_or.error().detail);
      continue;
    }

    // Prepared structure: the estimates / vcov index the reduced free set.
    auto pt_prep = h->pt;
    auto prep =
        magmaan::estimate::prepare_ordinal_delta_partable(pt_prep, h->stats);
    if (!prep.has_value()) {
      failures.push_back(id + ": prepare — " + prep.error().detail);
      continue;
    }

    if (has_std) {
      ++total;
      if (check_ordinal_std(id, pt_prep, h->rep, *est_or, rob_or->vcov,
                            dwls["standardized"], std_tol_ordinal, failures))
        ++passed;
    }

    if (has_def) {
      ++total;
      // compute_defined wants the original flat partable (`:=` row), the
      // prepared structure (free-index map), and the verbal LatentNames.
      auto fp = magmaan::parse::Parser::parse(ordinal_syntax(exp));
      if (!fp.has_value()) {
        failures.push_back(id + ": defined parse");
        continue;
      }
      magmaan::spec::BuildOptions bopts;
      bopts.n_groups = static_cast<std::int32_t>(exp["blocks"].size());
      magmaan::spec::LatentNames names;
      auto pt_named = magmaan::spec::build(*fp, bopts, nullptr, &names);
      if (!pt_named.has_value()) {
        failures.push_back(id + ": defined build — " + pt_named.error().detail);
        continue;
      }
      auto pt_def = *pt_named;
      auto prep_def =
          magmaan::estimate::prepare_ordinal_delta_partable(pt_def, h->stats);
      if (!prep_def.has_value()) {
        failures.push_back(id + ": defined prepare — " +
                           prep_def.error().detail);
        continue;
      }
      auto def_or = magmaan::measures::effects::compute_defined(
          *fp, pt_def, names, *est_or, rob_or->vcov);
      if (!def_or.has_value()) {
        failures.push_back(id + ": compute_defined — " + def_or.error().detail);
        continue;
      }
      bool ok = true;
      char buf[256];
      for (const auto& row : dwls["defined"]) {
        const std::string lhs = row["lhs"].get<std::string>();
        const double lav_est = row["est"].get<double>();
        const double lav_se = row["se"].get<double>();
        const auto it = std::find_if(
            def_or->entries.begin(), def_or->entries.end(),
            [&](const auto& e) { return e.name == lhs; });
        if (it == def_or->entries.end()) {
          failures.push_back(id + ": defined '" + lhs + "' missing");
          ok = false;
          continue;
        }
        const double d_est = std::abs(it->value - lav_est);
        const double d_se = std::abs(it->se - lav_se);
        if (d_est > defined_tol || d_se > defined_tol) {
          std::snprintf(buf, sizeof(buf),
                        "%s := %s est diff=%.2e se diff=%.2e", id.c_str(),
                        lhs.c_str(), d_est, d_se);
          failures.push_back(buf);
          ok = false;
        }
      }
      if (ok) ++passed;
    }
  }

  // --- mixed-ordinal fixtures (std.all/std.lv loadings) -------------------
  for (const auto& id : kMixedOrdinalFixtures) {
    const std::string path = mdir + "/" + id + ".ordinal.json";
    auto raw = magmaan::test::read_fixture(path);
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!exp.contains("fits") || !exp["fits"].contains("DWLS") ||
        !exp["fits"]["DWLS"].contains("standardized"))
      continue;

    auto h = mixed_handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    const magmaan::optim::OptimOptions mopt{
        .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};
    auto est_or = magmaan::test::fit_mixed_ordinal_bounded(
        h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, mopt);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    auto rob_or = magmaan::estimate::robust_mixed_ordinal(
        h->pt, h->rep, h->stats, *est_or,
        magmaan::estimate::OrdinalWeightKind::DWLS);
    if (!rob_or.has_value()) {
      failures.push_back(id + ": robust — " + rob_or.error().detail);
      continue;
    }
    auto pt_prep = h->pt;
    auto prep = magmaan::estimate::prepare_mixed_ordinal_delta_partable(
        pt_prep, h->stats);
    if (!prep.has_value()) {
      failures.push_back(id + ": prepare — " + prep.error().detail);
      continue;
    }
    ++total;
    if (check_ordinal_std(id, pt_prep, h->rep, *est_or, rob_or->vcov,
                          exp["fits"]["DWLS"]["standardized"], std_tol_mixed,
                          failures))
      ++passed;
  }

  MESSAGE("ordinal/mixed standardized + := : " << passed << " / " << total
          << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}

// Categorical factor-score (EBM / ML) parity against lavaan's lavPredict().
// This locks the previously live-only EBM parity (examples/ordinal_dwls_wls.R)
// into a checked-in golden, mirroring the prepared-structure path the Rcpp
// binding and api use: the estimates index the *prepared* (reduced) structure,
// the matrix_rep stays the original, and the raw integer category codes feed
// the categorical scorer directly.
//
// Oracle (per DWLS fit): fits.DWLS.fscores = {EBM:[...], (ML:[...])}, one value
// per observation in serialized block order. Scope is the proven surface:
//   - EBM, single-group, all-ordinal and mixed (the posterior mode is prior-
//     regularized and matches lavaan to ~1e-5).
//   - ML, mixed only — the continuous indicators bound the likelihood mode.
// Deliberately NOT gated (no oracle emitted):
//   - all-ordinal ML: the likelihood mode is unbounded on extreme response
//     patterns, where lavaan and magmaan legitimately diverge.
//   - multi-group: magmaan's per-group scorer diverges from lavaan for
//     non-reference groups (theta matches to 1e-5, but a non-reference group's
//     EBM drifts; tracked in docs/backlog/todo.md).
//   - EAP / posterior precision: lavaan's categorical lavPredict() rejects EAP,
//     so there is no oracle; it stays self-checked (tests/unit/api_sem_test.cpp).
TEST_CASE("ordinal/mixed factor scores (EBM/ML) match lavaan") {
  const std::string odir = magmaan::test::fixtures_dir() + "/ordinal";
  const std::string mdir = magmaan::test::fixtures_dir() + "/mixed_ordinal";
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  // Matches the live R parity gate (examples/ordinal_dwls_wls.R, 5e-4); the
  // measured spread at magmaan's own theta-hat is ~1e-5.
  const double fs_tol = 5e-4;

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  const auto gate_method =
      [&](const std::string& id, const std::string& method,
          const magmaan::measures::FactorScores& fs,
          const nlohmann::json& fscores) {
        ++total;
        const Eigen::VectorXd lav = vector_from_json(fscores[method]);
        if (fs.scores.size() != 1 || fs.scores[0].cols() != 1) {
          failures.push_back(id + " " + method + ": expected one block / one "
                             "factor column");
          return;
        }
        const Eigen::VectorXd got = fs.scores[0].col(0);
        const double d = max_abs_diff(got, lav);
        if (d > fs_tol) {
          char buf[160];
          std::snprintf(buf, sizeof(buf), "%s %s: max|diff|=%.3e (n=%lld)",
                        id.c_str(), method.c_str(), d,
                        static_cast<long long>(lav.size()));
          failures.push_back(buf);
          return;
        }
        ++passed;
      };

  // --- all-ordinal single-group fixtures (EBM) ----------------------------
  for (const auto& id : kOrdinalFixtures) {
    auto raw = magmaan::test::read_fixture(odir + "/" + id + ".ordinal.json");
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!has_fit(exp) || !exp["fits"].contains("DWLS") ||
        !exp["fits"]["DWLS"].contains("fscores"))
      continue;
    const auto& fscores = exp["fits"]["DWLS"]["fscores"];

    auto h = handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    auto est_or = magmaan::test::fit_ordinal_bounded(
        h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, opt);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    auto pt_prep = h->pt;
    auto prep =
        magmaan::estimate::prepare_ordinal_delta_partable(pt_prep, h->stats);
    if (!prep.has_value()) {
      failures.push_back(id + ": prepare — " + prep.error().detail);
      continue;
    }

    magmaan::data::RawData rd;
    for (const auto& b : exp["blocks"]) rd.X.push_back(matrix_from_json(b["matrix"]));

    auto fs = magmaan::measures::factor_scores_ordinal(
        pt_prep, h->rep, rd, h->stats, *est_or,
        magmaan::measures::FactorScoreMethod::Ebm,
        magmaan::estimate::OrdinalParameterization::Delta);
    if (!fs.has_value()) {
      ++total;
      failures.push_back(id + " EBM: factor_scores_ordinal — " +
                         fs.error().detail);
      continue;
    }
    gate_method(id, "EBM", *fs, fscores);
  }

  // --- mixed single-group fixtures (EBM + ML) -----------------------------
  for (const auto& id : kMixedOrdinalFixtures) {
    auto raw = magmaan::test::read_fixture(mdir + "/" + id + ".ordinal.json");
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    if (!exp.contains("fits") || !exp["fits"].contains("DWLS") ||
        !exp["fits"]["DWLS"].contains("fscores"))
      continue;
    const auto& fscores = exp["fits"]["DWLS"]["fscores"];

    auto h = mixed_handles_from_fixture(id, exp, failures);
    if (!h.has_value()) continue;

    const magmaan::optim::OptimOptions mopt{
        .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};
    auto est_or = magmaan::test::fit_mixed_ordinal_bounded(
        h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, mopt);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    auto pt_prep = h->pt;
    auto prep = magmaan::estimate::prepare_mixed_ordinal_delta_partable(
        pt_prep, h->stats);
    if (!prep.has_value()) {
      failures.push_back(id + ": prepare — " + prep.error().detail);
      continue;
    }

    magmaan::data::RawData rd;
    for (const auto& b : exp["blocks"]) rd.X.push_back(matrix_from_json(b["matrix"]));

    for (const auto& method : {std::string("EBM"), std::string("ML")}) {
      if (!fscores.contains(method)) continue;
      const auto fsm = method == "EBM"
          ? magmaan::measures::FactorScoreMethod::Ebm
          : magmaan::measures::FactorScoreMethod::Ml;
      auto fs = magmaan::measures::factor_scores_mixed_ordinal(
          pt_prep, h->rep, rd, h->stats, *est_or, fsm,
          magmaan::estimate::OrdinalParameterization::Delta);
      if (!fs.has_value()) {
        ++total;
        failures.push_back(id + " " + method +
                           ": factor_scores_mixed_ordinal — " +
                           fs.error().detail);
        continue;
      }
      gate_method(id, method, *fs, fscores);
    }
  }

  // --- multi-group self-consistency (transitive lavaan validation) --------
  // lavaan's multi-group categorical lavPredict() returns a non-stationary
  // point for non-reference groups (verified: at lavaan's group-2 score the
  // posterior gradient is O(1) and the posterior density is lower than at
  // magmaan's score, whose gradient is ~1e-7), so it is not a valid oracle.
  // Instead, for a fixture with no cross-group coupling the per-group fit
  // equals the independent single-group fit, so each group's multi-group EBM
  // must match scoring that group as a stand-alone single-group fit — which
  // the single-group lavaan EBM oracle above already gates. This validates the
  // multi-group scorer path against lavaan transitively, machine-precision
  // tight (measured ~3e-8).
  const std::string mg_id = "0004_2group_3cat_cfa";
  {
    auto raw = magmaan::test::read_fixture(odir + "/" + mg_id + ".ordinal.json");
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    auto h = handles_from_fixture(mg_id, exp, failures);
    if (h.has_value() && exp["blocks"].size() > 1) {
      auto est_or = magmaan::test::fit_ordinal_bounded(
          h->pt, h->rep, h->stats, magmaan::estimate::Bounds{},
          magmaan::estimate::OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptLbfgs, opt);
      auto pt_prep = h->pt;
      auto prep =
          magmaan::estimate::prepare_ordinal_delta_partable(pt_prep, h->stats);
      if (est_or.has_value() && prep.has_value()) {
        magmaan::data::RawData rd;
        for (const auto& b : exp["blocks"])
          rd.X.push_back(matrix_from_json(b["matrix"]));
        auto mg = magmaan::measures::factor_scores_ordinal(
            pt_prep, h->rep, rd, h->stats, *est_or,
            magmaan::measures::FactorScoreMethod::Ebm,
            magmaan::estimate::OrdinalParameterization::Delta);
        if (!mg.has_value()) {
          ++total;
          failures.push_back(mg_id + " multigroup EBM: " + mg.error().detail);
        } else {
          for (std::size_t b = 0; b < rd.X.size(); ++b) {
            ++total;
            auto sh = single_group_handles_from_block(mg_id, exp, rd.X[b],
                                                      failures);
            if (!sh.has_value()) continue;
            auto sest = magmaan::test::fit_ordinal_bounded(
                sh->pt, sh->rep, sh->stats, magmaan::estimate::Bounds{},
                magmaan::estimate::OrdinalWeightKind::DWLS,
                magmaan::estimate::Backend::NloptLbfgs, opt);
            auto spt = sh->pt;
            auto sprep =
                magmaan::estimate::prepare_ordinal_delta_partable(spt, sh->stats);
            if (!sest.has_value() || !sprep.has_value()) {
              failures.push_back(mg_id + " block " + std::to_string(b) +
                                 ": single-group sub-fit failed");
              continue;
            }
            magmaan::data::RawData srd;
            srd.X.push_back(rd.X[b]);
            auto sg = magmaan::measures::factor_scores_ordinal(
                spt, sh->rep, srd, sh->stats, *sest,
                magmaan::measures::FactorScoreMethod::Ebm,
                magmaan::estimate::OrdinalParameterization::Delta);
            if (!sg.has_value()) {
              failures.push_back(mg_id + " block " + std::to_string(b) +
                                 " single EBM: " + sg.error().detail);
              continue;
            }
            const Eigen::VectorXd mg_b = mg->scores[b].col(0);
            const Eigen::VectorXd sg_b = sg->scores[0].col(0);
            const double d = max_abs_diff(mg_b, sg_b);
            if (d > 1e-6) {
              char buf[160];
              std::snprintf(buf, sizeof(buf),
                            "%s block %zu: multigroup-vs-single EBM diff=%.3e",
                            mg_id.c_str(), b, d);
              failures.push_back(buf);
            } else {
              ++passed;
            }
          }
        }
      }
    }
  }

  MESSAGE("ordinal/mixed factor scores: " << passed << " / " << total
          << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
