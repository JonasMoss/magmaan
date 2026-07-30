#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

using magmaan::data::SampleStats;
using magmaan::estimate::Backend;
using magmaan::model::MatrixRep;
using magmaan::parse::Op;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;
using magmaan::test::matrix_from_json;
using magmaan::test::vector_from_json;

struct Handles {
  LatentStructure pt;
  MatrixRep rep;
  LatentNames names;
};

std::vector<int> permutation(const std::vector<std::string>& target,
                             const std::vector<std::string>& source) {
  std::vector<int> out;
  out.reserve(target.size());
  for (const auto& name : target) {
    const auto it = std::find(source.begin(), source.end(), name);
    if (it == source.end()) return {};
    out.push_back(static_cast<int>(std::distance(source.begin(), it)));
  }
  return out;
}

Eigen::MatrixXd reorder(const Eigen::MatrixXd& x,
                        const std::vector<int>& perm) {
  Eigen::MatrixXd out(x.rows(), x.cols());
  for (Eigen::Index r = 0; r < out.rows(); ++r) {
    for (Eigen::Index c = 0; c < out.cols(); ++c) {
      out(r, c) = x(perm[static_cast<std::size_t>(r)],
                    perm[static_cast<std::size_t>(c)]);
    }
  }
  return out;
}

Eigen::VectorXd reorder(const Eigen::VectorXd& x,
                        const std::vector<int>& perm) {
  Eigen::VectorXd out(x.size());
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = x(perm[static_cast<std::size_t>(i)]);
  }
  return out;
}

bool build_handles(const nlohmann::json& c, Handles& out) {
  auto flat =
      magmaan::parse::Parser::parse(c.at("model").get<std::string>());
  if (!flat.has_value()) {
    FAIL_CHECK("parse failed: " << flat.error().detail);
    return false;
  }
  magmaan::spec::BuildOptions options;
  options.fixed_x = c.at("fixed_x").get<bool>();
  options.meanstructure = c.at("meanstructure").get<bool>();
  LatentNames names;
  auto pt = magmaan::spec::build(*flat, options, nullptr, &names);
  if (!pt.has_value()) {
    FAIL_CHECK("lavaanify failed: " << pt.error().detail);
    return false;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt, &names);
  if (!rep.has_value()) {
    FAIL_CHECK("matrix representation failed: " << rep.error().detail);
    return false;
  }
  out = Handles{std::move(*pt), std::move(*rep), std::move(names)};
  return true;
}

bool build_stats(const nlohmann::json& c, const MatrixRep& rep,
                 SampleStats& out) {
  const auto fixture_names =
      c.at("observed_names").get<std::vector<std::string>>();
  if (rep.ov_names.size() != 1) {
    FAIL_CHECK("compact PSD corpus case is not single-block");
    return false;
  }
  const auto perm = permutation(rep.ov_names[0], fixture_names);
  if (perm.size() != fixture_names.size()) {
    FAIL_CHECK("fixture/model observed-variable names differ");
    return false;
  }
  out.S = {reorder(matrix_from_json(c.at("sample_cov")), perm)};
  if (!c.at("sample_mean").is_null()) {
    out.mean = {reorder(vector_from_json(c.at("sample_mean")), perm)};
  }
  out.n_obs = {c.at("n_obs").get<std::int64_t>()};
  return true;
}

magmaan::optim::OptimOptions strict_options() {
  return magmaan::optim::OptimOptions{
      .max_iter = 10000,
      .ftol = 1e-12,
      .gtol = 1e-8,
  };
}

double minimum_eigenvalue(
    const std::vector<magmaan::estimate::CovarianceBlockDiagnostics>& blocks) {
  double out = std::numeric_limits<double>::infinity();
  for (const auto& block : blocks) {
    out = std::min(out, block.min_eigenvalue);
  }
  return out;
}

bool has_negative_variance(
    const std::vector<magmaan::estimate::CovarianceBlockDiagnostics>& blocks) {
  return std::any_of(blocks.begin(), blocks.end(), [](const auto& block) {
    return !block.negative_variance_rows.empty();
  });
}

double max_abs_difference(const Eigen::VectorXd& lhs,
                          const Eigen::VectorXd& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return lhs.size() == 0 ? 0.0 : (lhs - rhs).cwiseAbs().maxCoeff();
}

double max_abs_difference(const Eigen::MatrixXd& lhs,
                          const Eigen::MatrixXd& rhs) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  return lhs.size() == 0 ? 0.0 : (lhs - rhs).cwiseAbs().maxCoeff();
}

bool covariance_row_value(const Handles& handles,
                          const Eigen::VectorXd& theta,
                          std::string_view lhs, std::string_view rhs,
                          double& out) {
  for (std::size_t row = 0; row < handles.pt.size(); ++row) {
    if (handles.names.row_lhs[row] != lhs ||
        handles.pt.op[row] != Op::Covariance ||
        handles.names.row_rhs[row] != rhs) {
      continue;
    }
    if (handles.pt.free[row] <= 0) {
      FAIL_CHECK("focus covariance row is fixed");
      return false;
    }
    out = theta(handles.pt.free[row] - 1);
    return true;
  }
  FAIL_CHECK("focus covariance row not found");
  return false;
}

void check_case(const nlohmann::json& c) {
  const std::string label =
      c.at("set").get<std::string>() + "::" + c.at("id").get<std::string>();
  INFO("PSD corpus geometry: " << label);

  Handles handles;
  if (!build_handles(c, handles)) return;
  SampleStats stats;
  if (!build_stats(c, handles.rep, stats)) return;

  const auto& expected = c.at("expected");
  const Eigen::VectorXd start = vector_from_json(c.at("ordinary_start"));
  auto ordinary = magmaan::estimate::fit_ml(
      handles.pt, handles.rep, stats, start, {}, Backend::NloptLbfgs,
      strict_options());
  if (!ordinary.has_value()) {
    FAIL_CHECK("ordinary ML failed: " << ordinary.error().detail);
    return;
  }
  auto psd = magmaan::estimate::frontier::fit_ml_psd(
      handles.pt, handles.rep, stats, ordinary->theta, Backend::NloptSlsqp,
      strict_options());
  if (!psd.has_value()) {
    FAIL_CHECK("PSD ML failed: " << psd.error().detail);
    return;
  }

  CHECK_FALSE(ordinary->diagnostics.admissibility.admissible);
  CHECK(psd->diagnostics.admissibility.admissible);
  CHECK(psd->audit.constrained);
  CHECK(psd->audit.constraint_violation_inf <= 1e-6);
  CHECK(psd->audit.stationary);

  CHECK(max_abs_difference(
            ordinary->theta,
            vector_from_json(expected.at("ordinary_theta"))) < 2e-5);
  CHECK(max_abs_difference(
            psd->theta, vector_from_json(expected.at("psd_theta"))) < 2e-5);
  CHECK(ordinary->fmin ==
        doctest::Approx(expected.at("ordinary_fmin").get<double>())
            .epsilon(1e-7));
  CHECK(psd->fmin ==
        doctest::Approx(expected.at("psd_fmin").get<double>())
            .epsilon(1e-7));

  CHECK(minimum_eigenvalue(
            ordinary->diagnostics.admissibility.psi_blocks) ==
        doctest::Approx(expected.at("ordinary_psi_min").get<double>())
            .epsilon(2e-5));
  CHECK(minimum_eigenvalue(psd->diagnostics.admissibility.psi_blocks) >=
        -1e-7);

  const double parameter_change =
      max_abs_difference(ordinary->theta, psd->theta);
  CHECK(parameter_change ==
        doctest::Approx(
            expected.at("max_abs_parameter_change").get<double>())
            .epsilon(2e-5));
  const double lr_increase =
      2.0 * static_cast<double>(stats.n_obs[0]) *
      (psd->fmin - ordinary->fmin);
  CHECK(lr_increase ==
        doctest::Approx(expected.at("lr_scale_increase").get<double>())
            .epsilon(2e-5));

  for (const auto& focus : expected.at("focus")) {
    double ordinary_value = 0.0;
    double psd_value = 0.0;
    if (!covariance_row_value(
            handles, ordinary->theta, focus.at("lhs").get<std::string>(),
            focus.at("rhs").get<std::string>(), ordinary_value) ||
        !covariance_row_value(
            handles, psd->theta, focus.at("lhs").get<std::string>(),
            focus.at("rhs").get<std::string>(), psd_value)) {
      return;
    }
    CHECK(ordinary_value ==
          doctest::Approx(focus.at("ordinary").get<double>()).epsilon(2e-5));
    CHECK(psd_value ==
          doctest::Approx(focus.at("psd").get<double>()).epsilon(2e-5));
  }

  auto evaluator =
      magmaan::model::ModelEvaluator::build(handles.pt, handles.rep);
  if (!evaluator.has_value()) {
    FAIL_CHECK("model evaluator failed: " << evaluator.error().detail);
    return;
  }
  auto ordinary_implied = evaluator->sigma(ordinary->theta);
  auto psd_implied = evaluator->sigma(psd->theta);
  if (!ordinary_implied.has_value() || !psd_implied.has_value()) {
    FAIL_CHECK("implied covariance evaluation failed");
    return;
  }
  const double implied_change = max_abs_difference(
      ordinary_implied->sigma[0], psd_implied->sigma[0]);

  const std::string geometry = c.at("geometry").get<std::string>();
  if (geometry == "same_fit_reallocation") {
    CHECK(std::abs(psd->fmin - ordinary->fmin) < 1e-10);
    CHECK(implied_change < 1e-7);
    CHECK(parameter_change > 0.03);
  } else if (geometry == "material_constrained_optimum") {
    CHECK(lr_increase > 20.0);
    CHECK(implied_change > 0.04);
  } else if (geometry == "negative_structural_disturbance") {
    CHECK(has_negative_variance(
        ordinary->diagnostics.admissibility.psi_blocks));
    CHECK(lr_increase < 0.02);
  } else if (geometry == "joint_indefinite_covariance") {
    CHECK_FALSE(has_negative_variance(
        ordinary->diagnostics.admissibility.psi_blocks));
    CHECK_FALSE(
        ordinary->diagnostics.admissibility.covariance_matrices_psd);
    CHECK(lr_increase > 0.5);
    CHECK(lr_increase < 1.5);
  } else {
    FAIL_CHECK("unknown PSD corpus geometry");
  }
}

}  // namespace

TEST_CASE("compact corpus geometries distinguish ordinary ML from PSD ML") {
  const auto fixture = magmaan::test::load_json_fixture(
      "psd_ml/corpus_geometries.json");
  REQUIRE_FALSE(fixture.is_discarded());
  REQUIRE(fixture.contains("cases"));
  REQUIRE(fixture.at("cases").size() == 4);
  for (const auto& c : fixture.at("cases")) {
    check_case(c);
  }
}
