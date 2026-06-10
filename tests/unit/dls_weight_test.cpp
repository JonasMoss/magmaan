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
#include "magmaan/estimate/gmm/dls_weight.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::data::RawData;
using magmaan::data::SampleStats;
using magmaan::estimate::frontier::dls_weight;
using magmaan::estimate::frontier::DlsWeightOptions;
using magmaan::estimate::frontier::empirical_bayes_dls_mixing_scalar;
using magmaan::estimate::frontier::empirical_bayes_dls_weight;
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

RawData make_heavy_raw(int n) {
  std::mt19937 rng(20260518u);
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi2(5.0);
  const double lam[3] = {1.0, 0.8, 1.2};
  Eigen::MatrixXd X(n, 3);
  for (int i = 0; i < n; ++i) {
    const double scale = std::sqrt(5.0 / chi2(rng));
    const double f = scale * z(rng);
    for (int j = 0; j < 3; ++j) {
      X(i, j) = lam[j] * f + scale * std::sqrt(0.5) * z(rng);
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

Eigen::Index vech_index(Eigen::Index p, Eigen::Index r, Eigen::Index c) {
  if (r < c) std::swap(r, c);
  Eigen::Index idx = 0;
  for (Eigen::Index j = 0; j < c; ++j) idx += p - j;
  return idx + (r - c);
}

Eigen::MatrixXd expected_symmetric_vech_gls_weight(const Eigen::MatrixXd& Sinv) {
  const Eigen::Index p = Sinv.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(pstar, pstar);
  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1; r1 < p; ++r1) {
      const Eigen::Index k1 = vech_index(p, r1, c1);
      Eigen::MatrixXd E1 = Eigen::MatrixXd::Zero(p, p);
      E1(r1, c1) = 1.0;
      E1(c1, r1) = 1.0;
      const Eigen::MatrixXd A1 = Sinv * E1 * Sinv;
      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2; r2 < p; ++r2) {
          const Eigen::Index k2 = vech_index(p, r2, c2);
          Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(p, p);
          E2(r2, c2) = 1.0;
          E2(c2, r2) = 1.0;
          W(k1, k2) = (A1 * E2).trace();
        }
      }
    }
  }
  return W;
}

}  // namespace

// ============================================================================
// Endpoint exactness: a = 0 ⇒ GLS, a = 1 ⇒ ADF/WLS.
// ============================================================================

TEST_CASE("normal_theory_weight: mean-structure GLS uses lavaan block scaling") {
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1", true);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  auto nt = magmaan::estimate::gmm::normal_theory_weight(*ev, *samp, *x0);
  REQUIRE(nt.has_value());
  REQUIRE(nt->size() == 1);

  const Eigen::MatrixXd& S = samp->S[0];
  Eigen::LLT<Eigen::MatrixXd> s_llt(S);
  REQUIRE(s_llt.info() == Eigen::Success);
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  const Eigen::MatrixXd Sinv = s_llt.solve(Eigen::MatrixXd::Identity(p, p));
  const Eigen::MatrixXd Wcov = expected_symmetric_vech_gls_weight(Sinv);
  const Eigen::MatrixXd& W = nt->at(0);

  REQUIRE(W.rows() == p + pstar);
  CHECK((W.block(0, 0, p, p) - Sinv).cwiseAbs().maxCoeff() < 1e-12);
  CHECK(W.block(0, p, p, pstar).cwiseAbs().maxCoeff() < 1e-12);
  CHECK(W.block(p, 0, pstar, p).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((W.block(p, p, pstar, pstar) - 0.5 * Wcov).cwiseAbs().maxCoeff() <
        1e-12);
}

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

    auto nt = magmaan::estimate::gmm::normal_theory_weight(*ev, *samp, *x0);
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
// Strict gate: a rank-deficient empirical Γ̂ is refused (not silently inverted
// into a ~1e16 eigendirection that later trips the terminal stationarity audit
// — the muthen_2017_ch2_ex2_1__adf pathology). Small N makes the fourth-moment
// Γ̂ (p*=6 vech-only, p+p*=9 with means) rank-deficient while the 3×3 sample
// covariance stays PD, so the model/start machinery is still happy.
// ============================================================================

TEST_CASE("dls_weight: rejects a rank-deficient ADF Gamma with diagnostics") {
  auto raw = make_raw(5);  // n-1 = 4 < p* ⇒ empirical Γ̂ is rank deficient
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  for (bool means : {false, true}) {
    auto m = build_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1", means);
    auto ev = ModelEvaluator::build(m.pt, m.rep);
    REQUIRE(ev.has_value());
    auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
    REQUIRE(x0.has_value());

    auto dls = dls_weight(*ev, *samp, raw, *x0, {1.0});  // a = 1 ⇒ pure ADF
    REQUIRE_FALSE(dls.has_value());
    CHECK(dls.error().kind == magmaan::FitError::Kind::NumericIssue);

    const std::string& d = dls.error().detail;
    CHECK(d.find("rank deficient") != std::string::npos);
    CHECK(d.find("numerical rank=") != std::string::npos);
    CHECK(d.find("rcond=") != std::string::npos);
    CHECK(d.find("lambda_min=") != std::string::npos);
    CHECK(d.find("ADF/WLS requires an invertible fourth-moment") !=
          std::string::npos);
    // The old, wrong "too few rows" wording must be gone.
    CHECK(d.find("too few rows") == std::string::npos);
  }
}

// ============================================================================
// Intermediate a: the covariance block inverts to the convex blend of Γ.
// ============================================================================

TEST_CASE("dls_weight: a=1 with meanstructure reproduces Browne-1984 full NACOV") {
  // With meanstructure the moment vector is [m̄ ; vech(S)] and Browne-1984
  // sets W = Γ̂_full⁻¹ where Γ̂_full is the stacked empirical NACOV
  // (including the third-moment cross-block). Inverting the returned
  // weight should recover exactly that stacked NACOV.
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1", true);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  auto dls = dls_weight(*ev, *samp, raw, *x0, {1.0});
  REQUIRE(dls.has_value());

  auto g_adf_full = magmaan::data::empirical_gamma_with_means(raw.X[0]);
  REQUIRE(g_adf_full.has_value());

  Eigen::LLT<Eigen::MatrixXd> llt(dls->at(0));
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd gamma_back =
      llt.solve(Eigen::MatrixXd::Identity(dls->at(0).rows(),
                                          dls->at(0).cols()));
  CHECK((gamma_back - *g_adf_full).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("dls_weight: intermediate a with meanstructure interpolates the "
          "full stacked NACOV") {
  // Confirms the matrix-level interpolation:
  //   Γ_DLS_full = (1-a)·blockdiag(S, Γ_NT) + a·Γ̂_ADF_full
  // Inverting the returned W must recover that blend, cross-block and all.
  auto raw = make_raw(400);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3\nx1 ~ 1\nx2 ~ 1\nx3 ~ 1", true);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  const double a = 0.35;
  auto dls = dls_weight(*ev, *samp, raw, *x0, {a});
  REQUIRE(dls.has_value());

  auto g_nt = magmaan::data::gamma_nt(samp->S[0]);
  auto g_adf_full = magmaan::data::empirical_gamma_with_means(raw.X[0]);
  REQUIRE(g_nt.has_value());
  REQUIRE(g_adf_full.has_value());

  const Eigen::Index p = samp->S[0].rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::MatrixXd gamma_nt_full =
      Eigen::MatrixXd::Zero(p + pstar, p + pstar);
  gamma_nt_full.topLeftCorner(p, p)             = samp->S[0];
  gamma_nt_full.bottomRightCorner(pstar, pstar) = *g_nt;
  const Eigen::MatrixXd gamma_mix =
      (1.0 - a) * gamma_nt_full + a * (*g_adf_full);

  Eigen::LLT<Eigen::MatrixXd> llt(dls->at(0));
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd gamma_back =
      llt.solve(Eigen::MatrixXd::Identity(dls->at(0).rows(),
                                          dls->at(0).cols()));
  CHECK((gamma_back - gamma_mix).cwiseAbs().maxCoeff() < 1e-8);

  // Sanity: the cross-block of gamma_back is a·(empirical cross), so it's
  // a strictly positive fraction of the empirical cross-block.
  const double cb_back = gamma_back.topRightCorner(p, pstar).cwiseAbs().maxCoeff();
  const double cb_emp  = g_adf_full->topRightCorner(p, pstar).cwiseAbs().maxCoeff();
  CHECK(cb_back == doctest::Approx(a * cb_emp).epsilon(1e-8));
}

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
  // NLopt fills iterations = 0 by design (see src/optim/nlopt_optimizer.cpp);
  // f_evals > 0 is the backend-portable "the optimizer actually ran" check.
  CHECK(out->f_evals > 0);
}

// ============================================================================
// Empirical-Bayes scalar selection.
// ============================================================================

TEST_CASE("empirical_bayes_dls_mixing_scalar shrinks near-normal data toward GLS") {
  auto raw = make_raw(2000);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto eb = empirical_bayes_dls_mixing_scalar(*samp, raw);
  REQUIRE(eb.has_value());
  CHECK(eb->a >= 0.0);
  CHECK(eb->a <= 1.0);
  CHECK(eb->observed_delta_norm2 >= 0.0);
  CHECK(eb->estimated_noise_norm2 >= 0.0);
  CHECK(eb->a < 0.35);
}

TEST_CASE("empirical_bayes_dls_mixing_scalar moves heavy-tailed data toward ADF") {
  auto normal_raw = make_raw(2000);
  auto normal_samp = magmaan::data::sample_stats_from_raw(normal_raw);
  REQUIRE(normal_samp.has_value());
  auto normal_eb = empirical_bayes_dls_mixing_scalar(*normal_samp, normal_raw);
  REQUIRE(normal_eb.has_value());

  auto heavy_raw = make_heavy_raw(2000);
  auto heavy_samp = magmaan::data::sample_stats_from_raw(heavy_raw);
  REQUIRE(heavy_samp.has_value());
  auto heavy_eb = empirical_bayes_dls_mixing_scalar(*heavy_samp, heavy_raw);
  REQUIRE(heavy_eb.has_value());

  CHECK(heavy_eb->a > normal_eb->a);
  CHECK(heavy_eb->a > 0.50);
  CHECK(heavy_eb->signal_norm2 > 0.0);
}

TEST_CASE("empirical_bayes_dls_weight delegates to dls_weight at estimated a") {
  auto raw = make_heavy_raw(600);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto m = build_model("f =~ x1 + x2 + x3", false);
  auto ev = ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto x0 = est::simple_start_values(m.pt, m.rep, *samp, {});
  REQUIRE(x0.has_value());

  auto eb = empirical_bayes_dls_mixing_scalar(*samp, raw);
  REQUIRE(eb.has_value());
  auto direct = dls_weight(*ev, *samp, raw, *x0, {eb->a});
  REQUIRE(direct.has_value());
  auto built = empirical_bayes_dls_weight(*ev, *samp, raw, *x0);
  REQUIRE(built.has_value());

  CHECK(max_block_diff(*direct, *built) < 1e-12);
}
