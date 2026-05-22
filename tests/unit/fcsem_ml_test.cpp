#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "../oracle.hpp"

using magmaan::data::SampleStats;
using magmaan::model::FcSemEvaluator;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::CompositeMode;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;
using magmaan::spec::Starts;

namespace {

struct Built {
  LatentStructure pt;
  LatentNames names;
  Starts starts;
};

Built must_build(std::string_view src) {
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

Eigen::MatrixXd matrix_from_json(const nlohmann::json& j) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr == 0 ? 0 : static_cast<Eigen::Index>(j[0].size());
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r) {
    for (Eigen::Index c = 0; c < nc; ++c) {
      out(r, c) = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                      .get<double>();
    }
  }
  return out;
}

nlohmann::json load_json_fixture(const std::string& rel) {
  const std::string path = magmaan::test::fixtures_dir() + "/" + rel;
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "could not open fixture: " << path);
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  return j;
}

void set_theta(const LatentStructure& pt, const LatentNames& names,
               Eigen::VectorXd& theta, std::string_view lhs, Op op,
               std::string_view rhs, double value) {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    if (names.row_lhs[i] == lhs && pt.op[i] == op && names.row_rhs[i] == rhs) {
      theta(pt.free[i] - 1) = value;
      return;
    }
  }
  FAIL("free row not found");
}

Op op_from_json(const std::string& op) {
  if (op == "<~") return Op::Composite;
  if (op == "=~") return Op::Measurement;
  if (op == "~") return Op::Regression;
  if (op == "~~") return Op::Covariance;
  FAIL("unsupported fixture op");
  return Op::Regression;
}

std::optional<double> row_value(const LatentStructure& pt,
                                const LatentNames& names,
                                const Eigen::VectorXd& theta,
                                std::string_view lhs, Op op,
                                std::string_view rhs) {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (names.row_lhs[i] != lhs || pt.op[i] != op || names.row_rhs[i] != rhs) {
      continue;
    }
    if (pt.free[i] > 0) return theta(pt.free[i] - 1);
    if (std::isfinite(pt.fixed_value[i])) return pt.fixed_value[i];
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<Eigen::Index> row_free_index(const LatentStructure& pt,
                                           const LatentNames& names,
                                           std::string_view lhs, Op op,
                                           std::string_view rhs) {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (names.row_lhs[i] != lhs || pt.op[i] != op || names.row_rhs[i] != rhs) {
      continue;
    }
    if (pt.free[i] <= 0) return std::nullopt;
    return static_cast<Eigen::Index>(pt.free[i] - 1);
  }
  return std::nullopt;
}

std::int32_t var_id(const LatentNames& names, std::string_view name) {
  for (std::size_t i = 0; i < names.var_name.size(); ++i)
    if (names.var_name[i] == name) return static_cast<std::int32_t>(i);
  return -1;
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

std::vector<std::string> lavaan_hs_observed_order(const nlohmann::json& j) {
  if (j["sample_cov"][0].contains("names")) {
    return j["sample_cov"][0]["names"].get<std::vector<std::string>>();
  }
  const std::string input = j["input"].get<std::string>();
  if (input.find("visual") == std::string::npos) {
    return {"x1", "x2", "x3", "x4", "x5"};
  }
  if (input.find("x9 ~") != std::string::npos) {
    return {"x4", "x5", "x6", "x1", "x2", "x3", "x9"};
  }
  return {"x4", "x5", "x6", "x1", "x2", "x3"};
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
                        lavaan_hs_observed_order(j), built_observed_order(b));
}

SampleStats sample_stats_from_composite_fixture(const nlohmann::json& j,
                                                const Built& b) {
  SampleStats samp;
  samp.S = {fixture_matrix_in_built_order(b, j, "sample_cov")};
  samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  return samp;
}

Eigen::Index ov_index(const LatentStructure& pt, const LatentNames& names,
                      std::string_view name) {
  const std::int32_t id = var_id(names, name);
  REQUIRE(id >= 0);
  const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(id)];
  REQUIRE(pos >= 0);
  return static_cast<Eigen::Index>(pos);
}

double fixture_row_est(const nlohmann::json& rows, std::string_view lhs,
                       std::string_view op, std::string_view rhs) {
  for (const auto& row : rows) {
    if (row["lhs"].get<std::string>() == lhs &&
        row["op"].get<std::string>() == op &&
        row["rhs"].get<std::string>() == rhs) {
      return row["est"].get<double>();
    }
  }
  FAIL("fixture row not found");
  return std::nan("");
}

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  REQUIRE(a.rows() == b.rows());
  REQUIRE(a.cols() == b.cols());
  return (a - b).cwiseAbs().maxCoeff();
}

Eigen::VectorXd pure_hs_theta_from_fixture(const Built& b,
                                           const nlohmann::json& j,
                                           const SampleStats& samp) {
  const Eigen::MatrixXd lavaan_sigma =
      fixture_matrix_in_built_order(b, j, "implied_sigma");

  Eigen::VectorXd theta(b.pt.n_free());
  theta.setConstant(std::nan(""));

  double w2 = 0.0;
  double w3 = 0.0;
  for (const auto& row : j["weights"]) {
    const std::string rhs = row["rhs"].get<std::string>();
    const double est = row["est"].get<double>();
    if (rhs == "x2") w2 = est;
    if (rhs == "x3") w3 = est;
    if (rhs != "x1") {
      set_theta(b.pt, b.names, theta, "C", Op::Composite, rhs, est);
    }
  }

  double b4 = 0.0;
  double b5 = 0.0;
  for (const auto& row : j["rows"]) {
    const std::string lhs = row["lhs"].get<std::string>();
    const double est = row["est"].get<double>();
    if (lhs == "x4") b4 = est;
    if (lhs == "x5") b5 = est;
    set_theta(b.pt, b.names, theta, lhs, Op::Regression, "C", est);
  }

  Eigen::Vector3d w;
  w << 1.0, w2, w3;
  const double vC = (w.transpose() * samp.S[0].topLeftCorner<3, 3>() * w)(0, 0);
  set_theta(b.pt, b.names, theta, "x4", Op::Covariance, "x4",
            lavaan_sigma(3, 3) - b4 * b4 * vC);
  set_theta(b.pt, b.names, theta, "x5", Op::Covariance, "x5",
            lavaan_sigma(4, 4) - b5 * b5 * vC);
  set_theta(b.pt, b.names, theta, "x4", Op::Covariance, "x5",
            lavaan_sigma(3, 4) - b4 * b5 * vC);
  REQUIRE(theta.array().isFinite().all());
  return theta;
}

void set_free_if_present(const LatentStructure& pt, const LatentNames& names,
                         Eigen::VectorXd& theta, std::string_view lhs, Op op,
                         std::string_view rhs, double value) {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (names.row_lhs[i] == lhs && pt.op[i] == op && names.row_rhs[i] == rhs) {
      if (pt.free[i] > 0) theta(pt.free[i] - 1) = value;
      return;
    }
  }
  FAIL("row not found");
}

Eigen::VectorXd hs_theta_from_fixture(const Built& b, const nlohmann::json& j,
                                      const SampleStats& samp) {
  const std::string input = j["input"].get<std::string>();
  if (input.find("visual") == std::string::npos) {
    return pure_hs_theta_from_fixture(b, j, samp);
  }

  const Eigen::MatrixXd lavaan_sigma =
      fixture_matrix_in_built_order(b, j, "implied_sigma");
  Eigen::VectorXd theta(b.pt.n_free());
  theta.setConstant(std::nan(""));

  for (const auto& row : j["weights"]) {
    set_free_if_present(b.pt, b.names, theta, row["lhs"].get<std::string>(),
                        Op::Composite, row["rhs"].get<std::string>(),
                        row["est"].get<double>());
  }
  for (const auto& row : j["rows"]) {
    set_free_if_present(b.pt, b.names, theta, row["lhs"].get<std::string>(),
                        op_from_json(row["op"].get<std::string>()),
                        row["rhs"].get<std::string>(),
                        row["est"].get<double>());
  }

  Eigen::Vector3d w;
  w << fixture_row_est(j["weights"], "C", "<~", "x1"),
       fixture_row_est(j["weights"], "C", "<~", "x2"),
       fixture_row_est(j["weights"], "C", "<~", "x3");
  const Eigen::Matrix3d T = samp.S[0].topLeftCorner<3, 3>();
  const double vC = (w.transpose() * T * w)(0, 0);
  const Eigen::Vector3d lambdaC = (T * w) / vC;

  Eigen::Vector3d lambdaV;
  lambdaV << fixture_row_est(j["rows"], "visual", "=~", "x4"),
             fixture_row_est(j["rows"], "visual", "=~", "x5"),
             fixture_row_est(j["rows"], "visual", "=~", "x6");
  const Eigen::Index x4 = ov_index(b.pt, b.names, "x4");
  const Eigen::Index x5 = ov_index(b.pt, b.names, "x5");
  const Eigen::Index x6 = ov_index(b.pt, b.names, "x6");
  const double vVisual =
      (lavaan_sigma(x4, x5) / (lambdaV(0) * lambdaV(1)) +
       lavaan_sigma(x4, x6) / (lambdaV(0) * lambdaV(2)) +
       lavaan_sigma(x5, x6) / (lambdaV(1) * lambdaV(2))) /
      3.0;

  const char* visual_indicators[] = {"x4", "x5", "x6"};
  for (Eigen::Index k = 0; k < 3; ++k) {
    const Eigen::Index idx = ov_index(b.pt, b.names, visual_indicators[k]);
    set_theta(b.pt, b.names, theta, visual_indicators[k], Op::Covariance,
              visual_indicators[k],
              lavaan_sigma(idx, idx) - lambdaV(k) * lambdaV(k) * vVisual);
  }

  if (input.find("visual ~ C") != std::string::npos) {
    const double beta = fixture_row_est(j["rows"], "visual", "~", "C");
    set_theta(b.pt, b.names, theta, "visual", Op::Covariance, "visual",
              vVisual - beta * beta * vC);
  } else {
    set_theta(b.pt, b.names, theta, "visual", Op::Covariance, "visual",
              vVisual);
    const Eigen::Index x1 = ov_index(b.pt, b.names, "x1");
    double covCV = 0.0;
    covCV += lavaan_sigma(x1, x4) / (lambdaC(0) * lambdaV(0));
    covCV += lavaan_sigma(x1, x5) / (lambdaC(0) * lambdaV(1));
    covCV += lavaan_sigma(x1, x6) / (lambdaC(0) * lambdaV(2));
    covCV /= 3.0;
    set_theta(b.pt, b.names, theta, "C", Op::Covariance, "visual", covCV);
  }

  if (input.find("x9 ~") != std::string::npos) {
    const double bC = fixture_row_est(j["rows"], "x9", "~", "C");
    const double bV = fixture_row_est(j["rows"], "x9", "~", "visual");
    const Eigen::Index x9 = ov_index(b.pt, b.names, "x9");
    const auto covCV = row_value(b.pt, b.names, theta, "C", Op::Covariance,
                                 "visual");
    REQUIRE(covCV.has_value());
    const double explained =
        bC * bC * vC + bV * bV * vVisual + 2.0 * bC * bV * *covCV;
    set_theta(b.pt, b.names, theta, "x9", Op::Covariance, "x9",
              lavaan_sigma(x9, x9) - explained);
  }

  REQUIRE(theta.array().isFinite().all());
  return theta;
}

struct PureHsFixture {
  nlohmann::json j;
  Built built;
  SampleStats samp;
  Eigen::VectorXd theta;
};

PureHsFixture pure_hs_fixture() {
  auto j = load_json_fixture("composite/0001_pure_composite_hs.fit.json");
  Built built = must_build(j["input"].get<std::string>());
  SampleStats samp = sample_stats_from_composite_fixture(j, built);
  Eigen::VectorXd theta = pure_hs_theta_from_fixture(built, j, samp);
  return PureHsFixture{std::move(j), std::move(built), std::move(samp),
                       std::move(theta)};
}

}  // namespace

TEST_CASE("FC-SEM ML: objective matches lavaan chi-square at native estimates") {
  static constexpr const char* fixtures[] = {
      "composite/0001_pure_composite_hs.fit.json",
      "composite/0002_composite_factor_hs.fit.json",
      "composite/0003_composite_structural_hs.fit.json",
  };

  for (const char* path : fixtures) {
    SUBCASE(path) {
      auto j = load_json_fixture(path);
      Built built = must_build(j["input"].get<std::string>());
      SampleStats samp = sample_stats_from_composite_fixture(j, built);
      Eigen::VectorXd theta = hs_theta_from_fixture(built, j, samp);
      auto ev = FcSemEvaluator::build(built.pt);
      REQUIRE_MESSAGE(ev.has_value(),
                      "FcSemEvaluator::build failed: " << ev.error().detail);
      auto obj = magmaan::estimate::ml_objective(*ev, samp);
      REQUIRE_MESSAGE(obj.has_value(),
                      "ml_objective failed: " << obj.error().detail);

      Eigen::VectorXd grad(theta.size());
      const double f = obj->f(theta, grad);
      REQUIRE(std::isfinite(f));
      REQUIRE(grad.size() == theta.size());
      CHECK(grad.array().isFinite().all());

      const double expected_f =
          j["chi2"].get<double>() / static_cast<double>(j["n_obs"].get<int>());
      CHECK(std::abs(f - expected_f) < 1e-8);

      auto got = ev->sigma(samp, theta);
      REQUIRE_MESSAGE(got.has_value(), "sigma failed: " << got.error().detail);
      const Eigen::MatrixXd lavaan_sigma =
          fixture_matrix_in_built_order(built, j, "implied_sigma");
      CHECK(max_abs_diff(got->sigma[0], lavaan_sigma) < 1e-8);
    }
  }
}

TEST_CASE("FC-SEM ML: objective supplies central finite-difference gradient") {
  auto fx = pure_hs_fixture();
  fx.theta(0) += 0.03;
  fx.theta(1) -= 0.02;
  fx.theta(2) += 0.04;
  fx.theta(fx.theta.size() - 1) += 0.05;

  auto ev = FcSemEvaluator::build(fx.built.pt);
  REQUIRE_MESSAGE(ev.has_value(),
                  "FcSemEvaluator::build failed: " << ev.error().detail);
  auto obj = magmaan::estimate::ml_objective(*ev, fx.samp);
  REQUIRE_MESSAGE(obj.has_value(), "ml_objective failed: " << obj.error().detail);

  Eigen::VectorXd grad(fx.theta.size());
  const double f = obj->f(fx.theta, grad);
  REQUIRE(std::isfinite(f));
  REQUIRE(grad.array().isFinite().all());

  Eigen::VectorXd fd(fx.theta.size());
  for (Eigen::Index k = 0; k < fx.theta.size(); ++k) {
    const double h = 5e-6 * std::max(1.0, std::abs(fx.theta(k)));
    Eigen::VectorXd plus = fx.theta;
    Eigen::VectorXd minus = fx.theta;
    plus(k) += h;
    minus(k) -= h;
    Eigen::VectorXd scratch(fx.theta.size());
    const double fp = obj->f(plus, scratch);
    const double fm = obj->f(minus, scratch);
    REQUIRE(std::isfinite(fp));
    REQUIRE(std::isfinite(fm));
    fd(k) = (fp - fm) / (2.0 * h);
  }

  CHECK((grad - fd).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("FC-SEM ML: fit from native simple starts matches lavaan objective") {
  static constexpr const char* fixtures[] = {
      "composite/0001_pure_composite_hs.fit.json",
      "composite/0002_composite_factor_hs.fit.json",
      "composite/0003_composite_structural_hs.fit.json",
  };

  for (const char* path : fixtures) {
    SUBCASE(path) {
      auto j = load_json_fixture(path);
      Built built = must_build(j["input"].get<std::string>());
      SampleStats samp = sample_stats_from_composite_fixture(j, built);
      auto x0 = magmaan::estimate::simple_fcsem_start_values(built.pt, samp);
      REQUIRE_MESSAGE(x0.has_value(), "starts failed: " << x0.error().detail);
      REQUIRE(x0->size() == built.pt.n_free());

      magmaan::optim::LbfgsOptions opts;
      opts.max_iter = 4000;
      auto est = magmaan::estimate::fit_ml_fcsem(
          built.pt, samp, *x0, {}, magmaan::estimate::Backend::Lbfgs, opts);
      REQUIRE_MESSAGE(est.has_value(), "fit failed: " << est.error().detail);

      const double expected_f =
          j["chi2"].get<double>() / static_cast<double>(j["n_obs"].get<int>());
      CHECK(std::abs(est->fmin - expected_f) < 1e-6);

      auto ev = FcSemEvaluator::build(built.pt);
      REQUIRE_MESSAGE(ev.has_value(),
                      "FcSemEvaluator::build failed: " << ev.error().detail);
      auto got = ev->sigma(samp, est->theta);
      REQUIRE_MESSAGE(got.has_value(), "sigma failed: " << got.error().detail);
      const Eigen::MatrixXd lavaan_sigma =
          fixture_matrix_in_built_order(built, j, "implied_sigma");
      CHECK(max_abs_diff(got->sigma[0], lavaan_sigma) < 1e-5);

      for (const auto& row : j["weights"]) {
        const auto lhs = row["lhs"].get<std::string>();
        const auto op = op_from_json(row["op"].get<std::string>());
        const auto rhs = row["rhs"].get<std::string>();
        const auto got_value =
            row_value(built.pt, built.names, est->theta, lhs, op, rhs);
        REQUIRE_MESSAGE(got_value.has_value(), "missing fitted row " << lhs
                                  << " " << row["op"].get<std::string>()
                                  << " " << rhs);
        CHECK(std::abs(*got_value - row["est"].get<double>()) < 1e-4);
      }
      for (const auto& row : j["rows"]) {
        const auto lhs = row["lhs"].get<std::string>();
        const auto op = op_from_json(row["op"].get<std::string>());
        const auto rhs = row["rhs"].get<std::string>();
        const auto got_value =
            row_value(built.pt, built.names, est->theta, lhs, op, rhs);
        REQUIRE_MESSAGE(got_value.has_value(), "missing fitted row " << lhs
                                  << " " << row["op"].get<std::string>()
                                  << " " << rhs);
        CHECK(std::abs(*got_value - row["est"].get<double>()) < 1e-4);
      }
    }
  }
}

TEST_CASE("FC-SEM expected information: SEs match lavaan native fixtures") {
  static constexpr const char* fixtures[] = {
      "composite/0001_pure_composite_hs.fit.json",
      "composite/0002_composite_factor_hs.fit.json",
      "composite/0003_composite_structural_hs.fit.json",
  };

  for (const char* path : fixtures) {
    SUBCASE(path) {
      auto j = load_json_fixture(path);
      Built built = must_build(j["input"].get<std::string>());
      SampleStats samp = sample_stats_from_composite_fixture(j, built);
      auto x0 = magmaan::estimate::simple_fcsem_start_values(built.pt, samp);
      REQUIRE_MESSAGE(x0.has_value(), "starts failed: " << x0.error().detail);

      magmaan::optim::LbfgsOptions opts;
      opts.max_iter = 4000;
      auto est = magmaan::estimate::fit_ml_fcsem(
          built.pt, samp, *x0, {}, magmaan::estimate::Backend::Lbfgs, opts);
      REQUIRE_MESSAGE(est.has_value(), "fit failed: " << est.error().detail);

      auto info =
          magmaan::inference::information_expected_fcsem(built.pt, samp, *est);
      REQUIRE_MESSAGE(info.has_value(),
                      "information failed: " << info.error().detail);
      auto vc = magmaan::inference::vcov(*info, built.pt, est->theta);
      REQUIRE_MESSAGE(vc.has_value(), "vcov failed: " << vc.error().detail);
      const Eigen::VectorXd se = magmaan::inference::se(*vc);

      auto check_row = [&](const nlohmann::json& row) {
        const auto lhs = row["lhs"].get<std::string>();
        const auto op = op_from_json(row["op"].get<std::string>());
        const auto rhs = row["rhs"].get<std::string>();
        const auto idx = row_free_index(built.pt, built.names, lhs, op, rhs);
        if (!idx.has_value()) {
          CHECK(row["se"].get<double>() == doctest::Approx(0.0));
          return;
        }
        REQUIRE(*idx >= 0);
        REQUIRE(*idx < se.size());
        CHECK(std::abs(se(*idx) - row["se"].get<double>()) < 3e-3);
      };

      for (const auto& row : j["weights"]) check_row(row);
      for (const auto& row : j["rows"]) check_row(row);
    }
  }
}
