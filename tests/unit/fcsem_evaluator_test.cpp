#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
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

FcSemEvaluator must_eval(const LatentStructure& pt) {
  auto ev = FcSemEvaluator::build(pt);
  REQUIRE_MESSAGE(ev.has_value(),
                  "FcSemEvaluator::build failed: " << ev.error().detail);
  return std::move(*ev);
}

Eigen::MatrixXd matrix_from_json(const nlohmann::json& j) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr == 0 ? 0 : static_cast<Eigen::Index>(j[0].size());
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r)
    for (Eigen::Index c = 0; c < nc; ++c)
      out(r, c) = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                      .get<double>();
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

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  REQUIRE(a.rows() == b.rows());
  REQUIRE(a.cols() == b.cols());
  return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("FcSemEvaluator: one composite regression matches hand algebra") {
  const Built b = must_build("C <~ x1 + x2 + x3\n y ~ C");
  auto ev = must_eval(b.pt);

  SampleStats samp;
  Eigen::MatrixXd S(4, 4);
  S << 2.0, 0.3, 0.4, 0.0,
       0.3, 1.5, 0.2, 0.0,
       0.4, 0.2, 1.2, 0.0,
       0.0, 0.0, 0.0, 3.0;
  samp.S = {S};
  samp.n_obs = {100};

  Eigen::VectorXd theta(b.pt.n_free());
  theta.setConstant(std::nan(""));
  set_theta(b.pt, b.names, theta, "C", Op::Composite, "x2", 0.5);
  set_theta(b.pt, b.names, theta, "C", Op::Composite, "x3", 0.25);
  set_theta(b.pt, b.names, theta, "y", Op::Regression, "C", 0.7);
  set_theta(b.pt, b.names, theta, "y", Op::Covariance, "y", 1.1);
  REQUIRE(theta.array().isFinite().all());

  auto got = ev.sigma(samp, theta);
  REQUIRE_MESSAGE(got.has_value(), "sigma failed: " << got.error().detail);
  REQUIRE(got->sigma.size() == 1);

  Eigen::Vector3d w;
  w << 1.0, 0.5, 0.25;
  const Eigen::Matrix3d T = S.topLeftCorner<3, 3>();
  const double vC = (w.transpose() * T * w)(0, 0);
  const Eigen::Vector3d cross = 0.7 * (T * w);

  Eigen::MatrixXd expected(4, 4);
  expected.setZero();
  expected.topLeftCorner<3, 3>() = T;
  for (Eigen::Index i = 0; i < 3; ++i) {
    expected(i, 3) = cross(i);
    expected(3, i) = cross(i);
  }
  expected(3, 3) = 0.7 * 0.7 * vC + 1.1;

  CHECK(max_abs_diff(got->sigma[0], expected) < 1e-12);
}

TEST_CASE("FcSemEvaluator: HS pure composite implied covariance matches lavaan fixture") {
  const auto j = load_json_fixture("composite/0001_pure_composite_hs.fit.json");
  const Built b = must_build(j["input"].get<std::string>());
  auto ev = must_eval(b.pt);

  SampleStats samp;
  samp.S = {matrix_from_json(j["sample_cov"][0]["matrix"])};
  samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  const Eigen::MatrixXd lavaan_sigma =
      matrix_from_json(j["implied_sigma"][0]["matrix"]);

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

  auto got = ev.sigma(samp, theta);
  REQUIRE_MESSAGE(got.has_value(), "sigma failed: " << got.error().detail);
  REQUIRE(got->sigma.size() == 1);
  CHECK(max_abs_diff(got->sigma[0], lavaan_sigma) < 1e-8);
}
