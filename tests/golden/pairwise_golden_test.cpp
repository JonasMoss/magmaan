#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/estimate/pairwise.hpp"

namespace {

nlohmann::json read_pairwise_fixture() {
  const std::string path = magmaan::test::fixtures_dir() +
      "/pairwise/0001_pairwise_diagnostics.json";
  std::ifstream in(path);
  REQUIRE(in.good());
  nlohmann::json j;
  in >> j;
  return j;
}

Eigen::MatrixXd matrix_from_json(const nlohmann::json& j, bool null_as_nan) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr > 0 ? static_cast<Eigen::Index>(j[0].size()) : 0;
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r) {
    for (Eigen::Index c = 0; c < nc; ++c) {
      const auto& cell = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
      out(r, c) = cell.is_null() && null_as_nan
          ? std::numeric_limits<double>::quiet_NaN()
          : cell.get<double>();
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

Eigen::VectorXi int_vector_from_json(const nlohmann::json& j) {
  Eigen::VectorXi out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].get<int>();
  }
  return out;
}

std::vector<std::int32_t> int32_vector_from_json(const nlohmann::json& j) {
  std::vector<std::int32_t> out;
  out.reserve(j.size());
  for (const auto& x : j) out.push_back(x.get<std::int32_t>());
  return out;
}

std::vector<std::vector<std::int32_t>>
int32_blocks_from_json(const nlohmann::json& j) {
  std::vector<std::vector<std::int32_t>> out;
  out.reserve(j.size());
  for (const auto& block : j) out.push_back(int32_vector_from_json(block));
  return out;
}

std::string pair_kind_name(magmaan::data::MixedPairKind kind) {
  switch (kind) {
    case magmaan::data::MixedPairKind::ordinal_ordinal:
      return "ordinal_ordinal";
    case magmaan::data::MixedPairKind::continuous_ordinal:
      return "continuous_ordinal";
    case magmaan::data::MixedPairKind::continuous_continuous:
      return "continuous_continuous";
  }
  return "unknown";
}

void check_pair_counts(const magmaan::data::OrdinalPairDiagnostics& got,
                       const nlohmann::json& exp) {
  CHECK(got.label.i == exp["i"].get<std::int32_t>());
  CHECK(got.label.j == exp["j"].get<std::int32_t>());
  CHECK(got.label.n_levels_i == exp["n_levels_i"].get<std::int32_t>());
  CHECK(got.label.n_levels_j == exp["n_levels_j"].get<std::int32_t>());
  CHECK(got.n_obs == exp["n_obs"].get<std::int64_t>());
  CHECK(got.n_missing == exp["n_missing"].get<std::int64_t>());
  CHECK(got.counts.isApprox(matrix_from_json(exp["counts"], false), 0.0));
  CHECK(got.expected_counts.rows() == got.counts.rows());
  CHECK(got.expected_counts.cols() == got.counts.cols());
  CHECK(got.residual_counts.rows() == got.counts.rows());
  CHECK(got.residual_counts.cols() == got.counts.cols());
}

void check_composite_pair(const magmaan::estimate::PairwiseOrdinalCompositePair& got,
                          const nlohmann::json& exp) {
  CHECK(got.label.i == exp["i"].get<std::int32_t>());
  CHECK(got.label.j == exp["j"].get<std::int32_t>());
  CHECK(got.label.n_levels_i == exp["n_levels_i"].get<std::int32_t>());
  CHECK(got.label.n_levels_j == exp["n_levels_j"].get<std::int32_t>());
  CHECK(got.n_obs == exp["n_obs"].get<std::int64_t>());
  CHECK(got.n_missing == exp["n_missing"].get<std::int64_t>());
  CHECK(got.counts.isApprox(matrix_from_json(exp["counts"], false), 0.0));
  CHECK(got.adjusted_counts.rows() == got.counts.rows());
  CHECK(got.adjusted_counts.cols() == got.counts.cols());
  CHECK(got.expected_counts.rows() == got.counts.rows());
  CHECK(got.expected_counts.cols() == got.counts.cols());
  CHECK(got.residual_counts.rows() == got.counts.rows());
  CHECK(got.residual_counts.cols() == got.counts.cols());
  CHECK(std::isfinite(got.rho));
  CHECK(std::isfinite(got.negloglik));
}

}  // namespace

TEST_CASE("pairwise diagnostics fixtures: complete ordinal wrappers and composites") {
  const auto fixture = read_pairwise_fixture();
  const auto& exp = fixture["complete_ordinal"];
  std::vector<Eigen::MatrixXd> blocks;
  for (const auto& block : exp["blocks"]) {
    blocks.push_back(matrix_from_json(block["matrix"], false));
  }

  auto stats = magmaan::data::pairwise_ordinal_stats_from_integer_data(blocks);
  REQUIRE(stats.has_value());
  REQUIRE(stats->block_diagnostics.size() == exp["blocks"].size());

  const auto& block_exp = exp["blocks"][0];
  const auto& block_diag = stats->block_diagnostics[0];
  REQUIRE(block_diag.pair_diagnostics.size() == block_exp["pairs"].size());
  CHECK(block_diag.moment_influence.rows() == blocks[0].rows());
  CHECK(block_diag.moment_influence.cols() == block_exp["moment_dim"].get<int>());
  CHECK(block_diag.gamma.rows() == block_exp["moment_dim"].get<int>());
  CHECK(block_diag.gamma.cols() == block_exp["moment_dim"].get<int>());
  CHECK(block_diag.gamma.isApprox(stats->stats.NACOV[0], 0.0));
  CHECK(std::isfinite(block_diag.min_eigen_r));
  for (std::size_t k = 0; k < block_diag.pair_diagnostics.size(); ++k) {
    check_pair_counts(block_diag.pair_diagnostics[k], block_exp["pairs"][k]);
  }

  auto shared = magmaan::estimate::pairwise_ordinal_composite_objective(
      *stats, stats->stats.thresholds, stats->stats.R);
  auto joint = magmaan::estimate::pairwise_ordinal_joint_composite_objective(*stats);
  REQUIRE(shared.has_value());
  REQUIRE(joint.has_value());
  const auto& comp_exp = exp["composite"];
  CHECK(shared->blocks[0].pairs.size() == comp_exp["pair_count"].get<std::size_t>());
  CHECK(joint->blocks[0].pairs.size() == comp_exp["pair_count"].get<std::size_t>());
  CHECK(shared->scaling_denominator ==
        doctest::Approx(comp_exp["scaling_denominator"].get<double>()));
  CHECK(joint->scaling_denominator ==
        doctest::Approx(comp_exp["scaling_denominator"].get<double>()));
  CHECK(shared->reports_chisq == comp_exp["reports_chisq"].get<bool>());
  CHECK(joint->reports_chisq == comp_exp["reports_chisq"].get<bool>());
  CHECK(shared->df == comp_exp["df"].get<int>());
  CHECK(joint->df == comp_exp["df"].get<int>());
  for (std::size_t k = 0; k < joint->blocks[0].pairs.size(); ++k) {
    check_composite_pair(joint->blocks[0].pairs[k], block_exp["pairs"][k]);
  }
}

TEST_CASE("pairwise diagnostics fixtures: observed-pair ordinal composites") {
  const auto fixture = read_pairwise_fixture();
  const auto& exp = fixture["observed_ordinal"];
  std::vector<Eigen::MatrixXd> blocks;
  for (const auto& block : exp["blocks"]) {
    blocks.push_back(matrix_from_json(block["matrix"], true));
  }
  const auto n_levels = int32_blocks_from_json(exp["n_levels"]);

  auto observed =
      magmaan::estimate::pairwise_ordinal_observed_joint_composite_objective(
          blocks, n_levels);
  REQUIRE(observed.has_value());
  const auto& comp_exp = exp["composite"];
  CHECK(observed->blocks[0].pairs.size() == comp_exp["pair_count"].get<std::size_t>());
  CHECK(observed->scaling_denominator ==
        doctest::Approx(comp_exp["scaling_denominator"].get<double>()));
  CHECK(observed->reports_chisq == comp_exp["reports_chisq"].get<bool>());
  CHECK(observed->df == comp_exp["df"].get<int>());
  for (std::size_t k = 0; k < observed->blocks[0].pairs.size(); ++k) {
    check_composite_pair(observed->blocks[0].pairs[k], exp["blocks"][0]["pairs"][k]);
  }
}

TEST_CASE("pairwise diagnostics fixtures: mixed pair primitives") {
  const auto fixture = read_pairwise_fixture();
  const auto& exp = fixture["mixed_primitives"];
  const auto ordered = int32_vector_from_json(exp["ordered"]);
  const auto threshold_ov = int32_vector_from_json(exp["threshold_ov"]);
  const auto threshold_level = int32_vector_from_json(exp["threshold_level"]);

  auto moments = magmaan::data::mixed_moment_labels(
      ordered, threshold_ov, threshold_level);
  auto pairs = magmaan::data::mixed_pair_labels(
      ordered, static_cast<std::int32_t>(threshold_ov.size()));
  REQUIRE(moments.has_value());
  REQUIRE(pairs.has_value());
  CHECK(moments->size() == exp["moment_count"].get<std::size_t>());
  REQUIRE(pairs->size() == exp["pairs"].size());
  for (std::size_t k = 0; k < pairs->size(); ++k) {
    CHECK((*pairs)[k].i == exp["pairs"][k]["i"].get<std::int32_t>());
    CHECK((*pairs)[k].j == exp["pairs"][k]["j"].get<std::int32_t>());
    CHECK((*pairs)[k].moment_index ==
          exp["pairs"][k]["moment_index"].get<std::int32_t>());
    CHECK(pair_kind_name((*pairs)[k].kind) ==
          exp["pairs"][k]["kind"].get<std::string>());
  }

  const auto& normal_exp = exp["continuous_pair"];
  auto x_i = vector_from_json(normal_exp["x_i"]);
  auto x_j = vector_from_json(normal_exp["x_j"]);
  auto normal = magmaan::data::fit_continuous_pair_normal_ml(x_i, x_j);
  REQUIRE(normal.has_value());
  CHECK(normal->n_obs == normal_exp["n_obs"].get<std::int64_t>());
  CHECK(normal->mean_i == doctest::Approx(normal_exp["mean_i"].get<double>()));
  CHECK(normal->mean_j == doctest::Approx(normal_exp["mean_j"].get<double>()));
  CHECK(normal->var_i == doctest::Approx(normal_exp["var_i"].get<double>()));
  CHECK(normal->var_j == doctest::Approx(normal_exp["var_j"].get<double>()));
  CHECK(normal->cov == doctest::Approx(normal_exp["cov"].get<double>()));
  CHECK(std::isfinite(normal->negloglik));

  const auto& poly_exp = exp["polyserial_pair"];
  auto categories = int_vector_from_json(poly_exp["categories"]);
  auto u = vector_from_json(poly_exp["u"]);
  auto thresholds = vector_from_json(poly_exp["thresholds"]);
  auto poly = magmaan::data::fit_polyserial_pair_rho_ml(
      categories, u, thresholds);
  REQUIRE(poly.has_value());
  CHECK(std::isfinite(poly->rho));
  CHECK(std::isfinite(poly->negloglik));
  if (poly_exp["rho_sign"].get<std::string>() == "positive") {
    CHECK(poly->rho > 0.0);
  }
  auto scores = magmaan::data::polyserial_pair_scores(
      categories, u, poly->rho, thresholds);
  REQUIRE(scores.has_value());
  CHECK(scores->rho.size() == categories.size());
  CHECK(scores->thresholds.rows() == categories.size());
  CHECK(scores->thresholds.cols() == thresholds.size());
  CHECK(scores->rho.allFinite());
  CHECK(scores->thresholds.allFinite());
}
