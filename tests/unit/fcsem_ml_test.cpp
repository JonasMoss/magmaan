#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/start_values.hpp"
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

SampleStats sample_stats_from_composite_fixture(const nlohmann::json& j) {
  SampleStats samp;
  samp.S = {matrix_from_json(j["sample_cov"][0]["matrix"])};
  samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  return samp;
}

Eigen::VectorXd pure_hs_theta_from_fixture(const Built& b,
                                           const nlohmann::json& j,
                                           const SampleStats& samp) {
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
  SampleStats samp = sample_stats_from_composite_fixture(j);
  Eigen::VectorXd theta = pure_hs_theta_from_fixture(built, j, samp);
  return PureHsFixture{std::move(j), std::move(built), std::move(samp),
                       std::move(theta)};
}

}  // namespace

TEST_CASE("FC-SEM ML: objective matches lavaan chi-square at native estimates") {
  auto fx = pure_hs_fixture();
  auto ev = FcSemEvaluator::build(fx.built.pt);
  REQUIRE_MESSAGE(ev.has_value(),
                  "FcSemEvaluator::build failed: " << ev.error().detail);
  auto obj = magmaan::estimate::ml_objective(*ev, fx.samp);
  REQUIRE_MESSAGE(obj.has_value(), "ml_objective failed: " << obj.error().detail);

  Eigen::VectorXd grad(fx.theta.size());
  const double f = obj->f(fx.theta, grad);
  REQUIRE(std::isfinite(f));
  REQUIRE(grad.size() == fx.theta.size());
  CHECK(grad.array().isFinite().all());

  const double expected_f =
      fx.j["chi2"].get<double>() / static_cast<double>(fx.j["n_obs"].get<int>());
  CHECK(std::abs(f - expected_f) < 1e-8);
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
  auto fx = pure_hs_fixture();
  auto x0 =
      magmaan::estimate::simple_fcsem_start_values(fx.built.pt, fx.samp);
  REQUIRE_MESSAGE(x0.has_value(), "starts failed: " << x0.error().detail);
  REQUIRE(x0->size() == fx.theta.size());

  magmaan::optim::LbfgsOptions opts;
  opts.max_iter = 2000;
  auto est = magmaan::estimate::fit_ml_fcsem(fx.built.pt, fx.samp, *x0, {},
                                             magmaan::estimate::Backend::Lbfgs,
                                             opts);
  REQUIRE_MESSAGE(est.has_value(), "fit failed: " << est.error().detail);

  const double expected_f =
      fx.j["chi2"].get<double>() / static_cast<double>(fx.j["n_obs"].get<int>());
  CHECK(std::abs(est->fmin - expected_f) < 1e-6);

  auto ev = FcSemEvaluator::build(fx.built.pt);
  REQUIRE_MESSAGE(ev.has_value(),
                  "FcSemEvaluator::build failed: " << ev.error().detail);
  auto got = ev->sigma(fx.samp, est->theta);
  REQUIRE_MESSAGE(got.has_value(), "sigma failed: " << got.error().detail);
  const Eigen::MatrixXd lavaan_sigma =
      matrix_from_json(fx.j["implied_sigma"][0]["matrix"]);
  CHECK((got->sigma[0] - lavaan_sigma).cwiseAbs().maxCoeff() < 1e-5);
}
