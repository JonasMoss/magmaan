#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <string_view>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "../test_fit.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/gmm/structured_gamma_weight.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::data::RawData;
using magmaan::model::ModelEvaluator;
namespace est = magmaan::estimate;

namespace {

RawData make_factor_raw(int n, bool heavy_factor) {
  std::mt19937 rng(20260520u + static_cast<unsigned>(heavy_factor));
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi2(5.0);
  const double lam[4] = {0.8, 0.7, 0.6, 0.5};
  Eigen::MatrixXd X(n, 4);
  for (int i = 0; i < n; ++i) {
    const double factor_scale = heavy_factor ? std::sqrt(5.0 / chi2(rng)) : 1.0;
    const double f = factor_scale * z(rng);
    for (int j = 0; j < 4; ++j) {
      const double theta = 1.0 - lam[j] * lam[j];
      X(i, j) = lam[j] * f + std::sqrt(theta) * z(rng);
    }
  }
  RawData raw;
  raw.X.push_back(std::move(X));
  return raw;
}

struct Model {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

Model build_model(std::string_view src, bool meanstructure = false) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = meanstructure;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Model{std::move(*pt), std::move(*mr)};
}

double max_block_diff(const magmaan::estimate::gmm::Weight& a,
                      const magmaan::estimate::gmm::Weight& b) {
  REQUIRE(a.size() == b.size());
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].rows() == b[i].rows());
    d = std::max(d, (a[i] - b[i]).cwiseAbs().maxCoeff());
  }
  return d;
}

}  // namespace

TEST_CASE("structured_gamma_weight: returns a WLS-ready weight") {
  auto raw = make_factor_raw(1200, false);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3 + x4");
  auto fit = magmaan::test::fit_gls(m.pt, m.rep, *samp);
  REQUIRE(fit.has_value());
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());

  auto W = est::frontier::structured_gamma_weight(
      *ev, m.rep, *samp, raw, fit->theta);
  REQUIRE_MESSAGE(W.has_value(),
                  "structured_gamma_weight failed: "
                      << (W.has_value() ? "" : W.error().detail));
  const auto& weight = W.value();
  REQUIRE(weight.size() == 1);
  REQUIRE(weight[0].rows() == 10);
  REQUIRE(weight[0].cols() == 10);
  CHECK((weight[0] - weight[0].transpose()).cwiseAbs().maxCoeff() < 1e-10);
  Eigen::LLT<Eigen::MatrixXd> llt(weight[0]);
  CHECK(llt.info() == Eigen::Success);

  auto out = magmaan::test::fit_gmm(m.pt, m.rep, *samp, weight);
  REQUIRE_MESSAGE(out.has_value(),
      "fit_gmm with structured Gamma weight failed: "
          << (out.has_value() ? "" : out.error().detail));
  CHECK(std::isfinite(out->fmin));
}

TEST_CASE("structured_gamma_weight: nonnormal source changes the GLS weight") {
  auto raw = make_factor_raw(1800, true);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3 + x4");
  auto fit = magmaan::test::fit_gls(m.pt, m.rep, *samp);
  REQUIRE(fit.has_value());
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());

  auto mi4 = est::frontier::structured_gamma_weight(
      *ev, m.rep, *samp, raw, fit->theta);
  REQUIRE_MESSAGE(mi4.has_value(),
                  "structured_gamma_weight failed: "
                      << (mi4.has_value() ? "" : mi4.error().detail));
  auto gls = est::gmm::normal_theory_weight(*ev, *samp, fit->theta);
  REQUIRE(gls.has_value());

  CHECK(max_block_diff(*mi4, *gls) > 1e-3);
}

TEST_CASE("structured_gamma_weight: rejects unsupported model shapes") {
  auto raw = make_factor_raw(400, false);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto mean_model = build_model(
      "f =~ x1 + x2 + x3 + x4\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1\nx4 ~ 1",
      true);
  auto mean_ev = ModelEvaluator::build(mean_model.pt, mean_model.rep);
  REQUIRE(mean_ev.has_value());
  auto mean_x0 = est::simple_start_values(mean_model.pt, mean_model.rep, *samp, {});
  REQUIRE(mean_x0.has_value());
  CHECK_FALSE(est::frontier::structured_gamma_weight(
      *mean_ev, mean_model.rep, *samp, raw, *mean_x0).has_value());

  auto reduced = build_model("f =~ x1 + x2 + x3\nx4 ~ f");
  auto red_ev = ModelEvaluator::build(reduced.pt, reduced.rep);
  REQUIRE(red_ev.has_value());
  auto red_x0 = est::simple_start_values(reduced.pt, reduced.rep, *samp, {});
  REQUIRE(red_x0.has_value());
  CHECK_FALSE(est::frontier::structured_gamma_weight(
      *red_ev, reduced.rep, *samp, raw, *red_x0).has_value());
}
