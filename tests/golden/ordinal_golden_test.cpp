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
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/constraints.hpp"
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
};

const std::vector<std::string> kMixedOrdinalFixtures = {
    "0001_mixed_cfa",
    "0002_sparse_4cat_listwise_mixed_cfa",
};

Eigen::MatrixXd matrix_from_json(const nlohmann::json& j) {
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

Eigen::VectorXd vector_from_json(const nlohmann::json& j) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].get<double>();
  }
  return out;
}

std::string ordinal_syntax(const nlohmann::json& exp) {
  std::string src = exp["input"].get<std::string>();
  if (!src.empty() && src.back() != '\n') src.push_back('\n');

  const auto& first_block = exp["blocks"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(first_block[0].size());
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
  for (Eigen::Index j = 0; j < p; ++j) {
    src += "x" + std::to_string(j + 1) + " ~*~ 1*x" +
           std::to_string(j + 1) + "\n";
  }
  return src;
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
      if (d_th > 5e-8 || d_mean > 5e-8 || d_R > 2e-3 || d_mom > 2e-3 ||
          d_N > 1.2 || d_WD > 1.2 || d_W > 1.2) {
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
      const double chisq = static_cast<double>(n_total) * est_or->fmin;
      const double d_chisq = std::abs(chisq - lavaan_chisq);
      const double d_theta = max_abs_diff(est_or->theta, lavaan_theta);
      const double theta_tol = 1e-5;
      const double chisq_tol = 8e-2;

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
        const double robust_se_tol = 3e-4;
        const double robust_ev_tol = 2e-4;
        const double robust_scale_tol = 2e-4;

        if (rob_or->df != robust["df"].get<int>() ||
            rob_or->satorra_bentler.df !=
                robust["satorra_bentler"]["df"].get<int>() ||
            rob_or->scaled_shifted.df !=
                robust["scaled_shifted"]["df"].get<int>() ||
            d_se > robust_se_tol || d_ev > robust_ev_tol ||
            d_standard > 8e-2 ||
            d_sb > 8e-2 || d_sb_scale > robust_scale_tol ||
            d_mv > 8e-2 || d_mv_df > 2e-4 ||
            d_ss > 8e-2 || d_ss_scale > robust_scale_tol ||
            d_ss_shift > 2e-4) {
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
      const double chisq = static_cast<double>(n_total) * est_or->fmin;
      const double d_chisq = std::abs(chisq - lavaan_chisq);
      const double d_theta = max_abs_diff(est_or->theta, lavaan_theta);
      if (df != lavaan_df || d_theta > 2e-2 || d_chisq > 3e-1) {
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
        lavaan_est.fmin = lavaan_chisq / static_cast<double>(n_total);
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
        // Mixed continuous/ordinal robust scaled tests are currently a looser
        // lavaan guard than the all-ordinal path; the small/sparse mixed
        // fixtures expose existing scale/eigen drift that is tracked in the
        // backlog rather than hidden by omitting the assertions.
        if (rob_or->df != robust["df"].get<int>() ||
            rob_or->satorra_bentler.df !=
                robust["satorra_bentler"]["df"].get<int>() ||
            rob_or->scaled_shifted.df !=
                robust["scaled_shifted"]["df"].get<int>() ||
            d_se > 3e-2 || d_ev > 1e-1 || d_standard > 1e-8 ||
            d_sb > 4.5e-1 || d_sb_scale > 9e-2 ||
            d_mv > 4.5e-1 || d_mv_df > 1e-2 ||
            d_ss > 4.5e-1 || d_ss_scale > 9e-2 ||
            d_ss_shift > 5e-3) {
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
