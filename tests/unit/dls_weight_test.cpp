#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <string_view>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "../test_fit.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/dls_weight.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::data::RawData;
using magmaan::data::SampleStats;
using magmaan::estimate::dls_weight;
using magmaan::estimate::DlsWeightOptions;
using magmaan::model::ModelEvaluator;
namespace est = magmaan::estimate;

namespace {

// Deterministic 1-factor raw data: x_ij = λ_j·f_i + ε_ij.
RawData make_raw(int n) {
  std::mt19937 rng(20260517u);
  std::normal_distribution<double> z(0.0, 1.0);
  const double lam[3] = {1.0, 0.8, 1.2};
  Eigen::MatrixXd X(n, 3);
  for (int i = 0; i < n; ++i) {
    const double f = z(rng);
    for (int j = 0; j < 3; ++j) {
      X(i, j) = lam[j] * f + std::sqrt(0.5) * z(rng);
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

Model build_model(std::string_view src, bool meanstructure) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = meanstructure;
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Model{std::move(*pt), std::move(*mr)};
}

double max_block_diff(const magmaan::gmm::Weight& a,
                      const magmaan::gmm::Weight& b) {
  REQUIRE(a.size() == b.size());
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].rows() == b[i].rows());
    d = std::max(d, (a[i] - b[i]).cwiseAbs().maxCoeff());
  }
  return d;
}

}  // namespace

// ============================================================================
// Endpoint exactness: a = 0 ⇒ GLS, a = 1 ⇒ ADF/WLS.
// ============================================================================

TEST_CASE("dls_weight: a=0 reproduces the normal-theory (GLS) weight") {
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  for (bool means : {false, true}) {
    auto m = build_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1", means);
    auto ev = ModelEvaluator::build(m.pt, m.rep);
    REQUIRE(ev.has_value());
    auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
    REQUIRE(x0.has_value());

    auto nt = magmaan::gmm::normal_theory_weight(*ev, *samp, *x0);
    REQUIRE(nt.has_value());
    auto dls = dls_weight(*ev, *samp, raw, *x0, {0.0});
    REQUIRE(dls.has_value());

    CHECK(max_block_diff(*nt, *dls) < 1e-9);
  }
}

TEST_CASE("dls_weight: a=1 covariance block is the ADF (empirical) weight") {
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3", false);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  auto dls = dls_weight(*ev, *samp, raw, *x0, {1.0});
  REQUIRE(dls.has_value());

  auto g_adf = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(g_adf.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(*g_adf);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd w_adf =
      llt.solve(Eigen::MatrixXd::Identity(g_adf->rows(), g_adf->cols()));

  // No mean structure ⇒ the whole block is the covariance weight.
  CHECK((dls->at(0) - w_adf).cwiseAbs().maxCoeff() < 1e-9);
}

// ============================================================================
// Intermediate a: the covariance block inverts to the convex blend of Γ.
// ============================================================================

TEST_CASE("dls_weight: intermediate a mixes the two Gamma matrices") {
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3", false);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  const double a = 0.35;
  auto dls = dls_weight(*ev, *samp, raw, *x0, {a});
  REQUIRE(dls.has_value());

  auto g_nt = magmaan::data::gamma_nt(samp->S[0]);
  auto g_adf = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(g_nt.has_value());
  REQUIRE(g_adf.has_value());
  const Eigen::MatrixXd gamma_mix = (1.0 - a) * (*g_nt) + a * (*g_adf);

  // Inverting the returned weight block recovers the mixed Γ.
  Eigen::LLT<Eigen::MatrixXd> llt(dls->at(0));
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd gamma_back =
      llt.solve(Eigen::MatrixXd::Identity(dls->at(0).rows(),
                                          dls->at(0).cols()));
  CHECK((gamma_back - gamma_mix).cwiseAbs().maxCoeff() < 1e-8);

  // Every block is symmetric positive definite.
  for (const auto& Wb : *dls) {
    CHECK((Wb - Wb.transpose()).cwiseAbs().maxCoeff() < 1e-10);
    Eigen::LLT<Eigen::MatrixXd> wllt(Wb);
    CHECK(wllt.info() == Eigen::Success);
  }
}

// ============================================================================
// Error paths.
// ============================================================================

TEST_CASE("dls_weight: rejects an out-of-range mixing scalar") {
  auto raw = make_raw(200);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3", false);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  CHECK_FALSE(dls_weight(*ev, *samp, raw, *x0, {-0.1}).has_value());
  CHECK_FALSE(dls_weight(*ev, *samp, raw, *x0, {1.5}).has_value());
}

TEST_CASE("dls_weight: rejects raw data with missingness") {
  auto raw = make_raw(200);
  raw.mask.emplace_back(200, 3);
  raw.mask[0].setOnes();
  auto samp = magmaan::data::sample_stats_from_raw(make_raw(200));
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3", false);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  CHECK_FALSE(dls_weight(*ev, *samp, raw, *x0, {0.5}).has_value());
}

// ============================================================================
// End-to-end: the DLS weight drives a moment-quadratic fit.
// ============================================================================

TEST_CASE("dls_weight: fit_gmm converges with the DLS weight") {
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3", false);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  auto W = dls_weight(*ev, *samp, raw, *x0, {0.5});
  REQUIRE(W.has_value());

  auto out = magmaan::test::fit_gmm(m.pt, m.rep, *samp, *W);
  REQUIRE_MESSAGE(out.has_value(),
      "fit_gmm with DLS weight failed: "
          << (out.has_value() ? "" : out.error().detail));
  CHECK(std::isfinite(out->fmin));
  CHECK(out->iterations > 0);
}
