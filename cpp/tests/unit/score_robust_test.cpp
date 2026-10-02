#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "../../src/inference/detail_score_flip.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/prepared_ntml.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/spec/build.hpp"

// Unit tests for the robust (generalized / SB-scaled) score & modification-index
// tests in inference::frontier. lavaan does not implement this statistic (it
// falls back to the ordinary one), so the deterministic anchors here are:
//
//   * reduction-to-NT: with the model-implied Γ_NT meat and the Expected bread,
//     the scaling factor is 1 EXACTLY (WΓ_NTW = W ⇒ meat = bread), so the robust
//     statistic equals the ordinary one bit-for-bit.
//   * the empirical gamma_hat meat equals the model-implied meat when fed Γ_NT.
//
// The non-normal behaviour (c ≠ 1) is validated by the R-assembled golden and
// the advisory simulation, not here.

using magmaan::data::SampleStats;
using magmaan::model::MatrixRep;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::LatentStructure;
namespace inf = magmaan::inference;
namespace rob = magmaan::robust;

namespace {

struct Handles {
  LatentStructure pt;
  MatrixRep rep;
};

Handles build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

Eigen::Matrix4d four_indicator_sample_cov() {
  Eigen::Vector4d lambda;
  lambda << 1.0, 0.8, 0.7, 0.9;
  Eigen::Vector4d theta;
  theta << 0.6, 0.7, 0.8, 0.5;
  Eigen::Matrix4d S =
      lambda * lambda.transpose() * 1.4 + theta.asDiagonal().toDenseMatrix();
  S(1, 0) += 0.18;
  S(0, 1) = S(1, 0);
  return S;
}

// Heavy-tailed multivariate-t sample with covariance ≈ Sigma (so Γ̂ ≠ Γ_NT and
// the robust scaling is genuinely ≠ 1). Deterministic given the RNG.
Eigen::MatrixXd multivariate_t_sample(std::mt19937& rng, Eigen::Index n,
                                      const Eigen::MatrixXd& Sigma, double df) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::Index p = Sigma.rows();
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi(df);
  const double scale = std::sqrt((df - 2.0) / df);  // ⇒ Cov ≈ Sigma
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(p);
    for (Eigen::Index j = 0; j < p; ++j) zi(j) = z(rng);
    const double w = chi(rng) / df;
    X.row(i) = (scale * (L * zi) / std::sqrt(w)).transpose();
  }
  return X;
}

inf::frontier::RobustScoreOptions robust_opts(rob::Information bread,
                                              inf::ScoreCandidateSet cands) {
  inf::frontier::RobustScoreOptions o;
  o.spec.bread = bread;
  o.base.candidates = cands;
  o.base.information = bread == rob::Information::Observed
                           ? inf::ScoreInformation::Observed
                           : inf::ScoreInformation::Expected;
  return o;
}

// Multi-group build (configural unless `src` carries cross-group labels).
Handles build_groups(std::string_view src, int n_groups) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.n_groups = n_groups;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

Handles build_groups_mean(std::string_view src, int n_groups) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = n_groups;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

// Meanstructure CFA build for the FIML robust tier (FIML estimates means).
Handles build_mean(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

// n Gaussian rows from the 4-indicator true covariance (mean 0), with MCAR
// missingness (every `period`-th row drops one rotating cell). Deterministic
// given the RNG; the correct normal model drives the robust scaling toward 1.
magmaan::data::RawData gaussian_cfa_raw(std::mt19937& rng, Eigen::Index n,
                                        Eigen::Index period) {
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  Eigen::LLT<Eigen::Matrix4d> llt(Sigma);
  const Eigen::Matrix4d L = llt.matrixL();
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::Vector4d zi;
    for (Eigen::Index j = 0; j < 4; ++j) zi(j) = z(rng);
    X.row(i) = (L * zi).transpose();
  }
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M =
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (period > 0 && i % period == 0) {
      const Eigen::Index c = i % 4;
      M(i, c) = 0;
      X(i, c) = std::numeric_limits<double>::quiet_NaN();
    }
  }
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  raw.mask.push_back(std::move(M));
  return raw;
}

magmaan::data::RawData gaussian_cfa_raw_groups(
    std::mt19937& rng, const std::vector<Eigen::Index>& ns,
    Eigen::Index period) {
  magmaan::data::RawData out;
  for (std::size_t b = 0; b < ns.size(); ++b) {
    auto block =
        gaussian_cfa_raw(rng, ns[b], period + static_cast<Eigen::Index>(b));
    out.X.push_back(std::move(block.X[0]));
    out.mask.push_back(std::move(block.mask[0]));
  }
  return out;
}

}  // namespace

TEST_CASE("frontier robust MI: model-implied Expected bread reduces to NT") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions nt_opts;
  nt_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  nt_opts.information = inf::ScoreInformation::Expected;
  auto nt = inf::modification_indices(h.pt, h.rep, samp, *est, nt_opts);
  REQUIRE(nt.has_value());

  auto rob_mi = inf::frontier::modification_indices_robust(
      h.pt, h.rep, samp, *est,
      robust_opts(rob::Information::Expected,
                  inf::ScoreCandidateSet::WithAbsentRows));
  REQUIRE(rob_mi.has_value());

  REQUIRE(rob_mi->rows.size() == nt->rows.size());
  REQUIRE(rob_mi->rows.size() > 1);
  for (std::size_t i = 0; i < rob_mi->rows.size(); ++i) {
    const auto& r = rob_mi->rows[i];
    const auto& n = nt->rows[i];
    // NT component identical to the standalone NT path.
    CHECK(std::abs(r.mi - n.mi) < 1e-9 * (1.0 + std::abs(n.mi)));
    // Expected ModelImplied ⇒ scaling factor is exactly 1, scaled == ordinary.
    CHECK(std::abs(r.scaling_factor - 1.0) < 1e-9);
    CHECK(std::abs(r.mi_scaled - r.mi) < 1e-9 * (1.0 + std::abs(r.mi)));
    CHECK(std::abs(r.v_eff - r.information) < 1e-9 * (1.0 + r.information));
  }
}

TEST_CASE("frontier robust MI: marker excluded across nearby fits and units") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  for (double units : {0.1, 1.0, 10.0}) {
    SampleStats samp;
    samp.S = {units * units * four_indicator_sample_cov()};
    samp.n_obs = {400};
    auto est = magmaan::test::fit(h.pt, h.rep, samp);
    REQUIRE(est.has_value());
    for (double displacement : {-1e-7, 0.0, 1e-7}) {
      auto nearby = *est;
      nearby.theta *= 1.0 + displacement;
      for (int n : {400, 100000000}) {
        CAPTURE(units);
        CAPTURE(displacement);
        CAPTURE(n);
        samp.n_obs = {n};
        for (auto bread : {rob::Information::Expected, rob::Information::Observed}) {
          CAPTURE(bread);
          auto result = inf::frontier::modification_indices_robust(
              h.pt, h.rep, samp, nearby,
              robust_opts(bread,
                          inf::ScoreCandidateSet::WithAbsentRows));
          REQUIRE(result.has_value());
          // All six residual covariances are identified. The sole fixed loading
          // is the marker and must never acquire a test through cancellation.
          REQUIRE(result->rows.size() == 6);
          for (const auto& row : result->rows) {
            CHECK(row.candidate.op == magmaan::parse::Op::Covariance);
            CHECK(row.candidate.lhs_var != row.candidate.rhs_var);
            CHECK(row.information > 0.0);
            if (bread == rob::Information::Expected) {
              CHECK(row.scaling_factor == doctest::Approx(1.0).epsilon(1e-8));
            } else {
              CHECK(std::isfinite(row.scaling_factor));
              CHECK(row.scaling_factor > 0.0);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("frontier robust MI: fixed loading is testable with fixed latent variance") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nf ~~ 1*f");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());
  for (auto bread : {rob::Information::Expected, rob::Information::Observed}) {
    auto result = inf::frontier::modification_indices_robust(
        h.pt, h.rep, samp, *est,
        robust_opts(bread, inf::ScoreCandidateSet::FixedRowsOnly));
    REQUIRE(result.has_value());
    bool loading_present = false;
    for (const auto& row : result->rows) {
      if (row.candidate.op == magmaan::parse::Op::Measurement) {
        loading_present = true;
        CHECK(row.information > 0.0);
        CHECK(row.mi > 0.0);
      }
    }
    CHECK(loading_present);
  }
}

TEST_CASE("frontier robust MI: gamma_hat = Gamma_NT meat equals model-implied") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  // Model-implied Σ̂ → structured Γ_NT(Σ̂).
  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto im = ev->sigma(est->theta);
  REQUIRE(im.has_value());
  auto gnt = magmaan::data::gamma_nt(im->sigma[0]);
  REQUIRE(gnt.has_value());

  const rob::InferenceSpec spec{rob::Information::Expected,
                                rob::WeightMoments::Structured,
                                rob::ScoreCovariance::Empirical};
  auto sw_mi = rob::param_space_sandwich(h.pt, h.rep, samp, *est, spec,
                                         /*reparam_constraints=*/false);
  REQUIRE(sw_mi.has_value());
  auto sw_gh = rob::param_space_sandwich(h.pt, h.rep, samp, *est, *gnt, spec,
                                         /*reparam_constraints=*/false);
  REQUIRE(sw_gh.has_value());

  // Expected bread, model-implied meat: bread == meat.
  CHECK((sw_mi->A1 - sw_mi->B1).norm() < 1e-8 * (1.0 + sw_mi->A1.norm()));
  // Feeding Γ_NT(Σ̂) as Γ̂ reproduces the model-implied meat.
  CHECK((sw_gh->B1 - sw_mi->B1).norm() < 1e-7 * (1.0 + sw_mi->B1.norm()));
}

TEST_CASE("frontier robust MI: observed bread is finite and positive") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto rob_mi = inf::frontier::modification_indices_robust(
      h.pt, h.rep, samp, *est,
      robust_opts(rob::Information::Observed,
                  inf::ScoreCandidateSet::WithAbsentRows));
  REQUIRE(rob_mi.has_value());
  REQUIRE(rob_mi->rows.size() > 1);
  for (const auto& r : rob_mi->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    CHECK(r.mi_scaled >= 0.0);
  }
}

TEST_CASE("frontier robust score test: equality release reduces to NT") {
  // An explicit `a == b` equality constraint to release.
  auto h = build("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto nt = inf::score_tests(h.pt, h.rep, samp, *est,
                             inf::ScoreInformation::Expected);
  REQUIRE(nt.has_value());
  REQUIRE(nt->rows.size() == 1);

  // score_tests_robust has no sample-stats-only overload; feed gamma_hat = Γ_NT,
  // which gives an exact scaling of 1 for the Expected bread.
  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto im = ev->sigma(est->theta);
  REQUIRE(im.has_value());
  auto gnt = magmaan::data::gamma_nt(im->sigma[0]);
  REQUIRE(gnt.has_value());
  auto rob_st = inf::frontier::score_tests_robust(
      h.pt, h.rep, samp, *gnt, *est,
      robust_opts(rob::Information::Expected,
                  inf::ScoreCandidateSet::FixedRowsOnly));
  REQUIRE(rob_st.has_value());
  REQUIRE(rob_st->rows.size() == 1);

  const auto& r = rob_st->rows[0];
  const auto& n = nt->rows[0];
  CHECK(std::abs(r.mi - n.mi) < 1e-7 * (1.0 + std::abs(n.mi)));
  CHECK(std::abs(r.scaling_factor - 1.0) < 1e-7);
  CHECK(std::abs(r.mi_scaled - n.mi) < 1e-7 * (1.0 + std::abs(n.mi)));
}

TEST_CASE("param_space_sandwich: whitened-solve A1/B1 match explicit Δ'WΔ / Δ'WΓ̂WΔ") {
  // Independent re-derivation of the bread/meat from primitives (Δ via the
  // evaluator, W = Γ_NT(Σ̂)⁻¹ by explicit inverse, Γ̂ empirical) vs the library's
  // triangular-solve path. A numeric cross-check of the novel meat machinery.
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(424242u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 2500, Sigma, 7.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());

  // Primitives: Δ = ∂σ/∂θ at θ̂, Σ̂ model-implied, W = Γ_NT(Σ̂)⁻¹, Γ̂ empirical.
  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto Delta = ev->dsigma_dtheta(est->theta);
  REQUIRE(Delta.has_value());
  auto im = ev->sigma(est->theta);
  REQUIRE(im.has_value());
  auto Gnt = magmaan::data::gamma_nt(im->sigma[0]);
  REQUIRE(Gnt.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt_gnt(*Gnt);
  REQUIRE(llt_gnt.info() == Eigen::Success);
  const Eigen::MatrixXd W =
      llt_gnt.solve(Eigen::MatrixXd::Identity(Gnt->rows(), Gnt->cols()));
  auto Ghat = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(Ghat.has_value());

  const Eigen::MatrixXd A1_ref = Delta->transpose() * W * (*Delta);
  const Eigen::MatrixXd B1_ref =
      Delta->transpose() * W * (*Ghat) * W * (*Delta);

  const rob::InferenceSpec spec{rob::Information::Expected,
                                rob::WeightMoments::Structured,
                                rob::ScoreCovariance::Empirical};
  auto sw = rob::param_space_sandwich(h.pt, h.rep, *samp, *est, *Ghat, spec,
                                      /*reparam_constraints=*/false);
  REQUIRE(sw.has_value());

  CHECK((sw->A1 - A1_ref).norm() < 1e-8 * (1.0 + A1_ref.norm()));
  CHECK((sw->B1 - B1_ref).norm() < 1e-8 * (1.0 + B1_ref.norm()));
}

TEST_CASE("param_space_sandwich: caller-supplied Zc matches raw-data overload") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(20260612u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 900, Sigma, 6.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());

  const rob::InferenceSpec spec{rob::Information::Expected,
                                rob::WeightMoments::Structured,
                                rob::ScoreCovariance::Empirical};
  auto sw_raw = rob::param_space_sandwich(h.pt, h.rep, *samp, *est, raw, spec,
                                          /*reparam_constraints=*/false);
  REQUIRE(sw_raw.has_value());
  auto Zc = rob::casewise_contributions(raw, *samp);
  REQUIRE(Zc.has_value());
  auto sw_zc = rob::param_space_sandwich(
      h.pt, h.rep, *samp, *est, *Zc, static_cast<double>(raw.X[0].rows()),
      spec, /*reparam_constraints=*/false);
  REQUIRE(sw_zc.has_value());

  CHECK((sw_zc->A1 - sw_raw->A1).norm() < 1e-12 * (1.0 + sw_raw->A1.norm()));
  CHECK((sw_zc->B1 - sw_raw->B1).norm() < 1e-12 * (1.0 + sw_raw->B1.norm()));
}

TEST_CASE("frontier robust MI: empirical raw-data path scales on non-normal data") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(20260602u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 3000, Sigma, 8.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());

  const auto opts = robust_opts(rob::Information::Expected,
                                inf::ScoreCandidateSet::WithAbsentRows);
  auto rob_raw = inf::frontier::modification_indices_robust(h.pt, h.rep, *samp,
                                                            raw, *est, opts);
  REQUIRE(rob_raw.has_value());
  REQUIRE(rob_raw->rows.size() > 1);

  // All finite, and the heavy-tailed Γ̂ genuinely rescales at least one row
  // (i.e. the empirical meat is actually used, not a silent NT fallback).
  bool any_scaled = false;
  for (const auto& r : rob_raw->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    CHECK(r.mi_scaled >= 0.0);
    if (std::abs(r.scaling_factor - 1.0) > 0.03) any_scaled = true;
  }
  CHECK(any_scaled);

  // Independent route: the raw casewise meat equals feeding the same empirical
  // Γ̂ = empirical_gamma(X) explicitly.
  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  auto rob_gh = inf::frontier::modification_indices_robust(h.pt, h.rep, *samp,
                                                           *G, *est, opts);
  REQUIRE(rob_gh.has_value());
  REQUIRE(rob_gh->rows.size() == rob_raw->rows.size());
  for (std::size_t i = 0; i < rob_raw->rows.size(); ++i) {
    CHECK(std::abs(rob_raw->rows[i].mi_scaled - rob_gh->rows[i].mi_scaled) <
          1e-7 * (1.0 + std::abs(rob_gh->rows[i].mi_scaled)));
    CHECK(std::abs(rob_raw->rows[i].scaling_factor -
                   rob_gh->rows[i].scaling_factor) < 1e-8);
  }
}

// ── Continuous-LS tier ────────────────────────────────────────────────────────
// The exact-reduction anchor for the LS sandwich: the GLS weight is Γ_NT(S)⁻¹
// (the ½ lives inside `normal_theory_weight`), so the model-implied meat under
// WeightMoments::Unstructured gives B1 = Δ'WΓ_NT(S)WΔ = Δ'WΔ = A1 and c ≡ 1.

TEST_CASE("frontier robust LS MI: GLS weight + Gamma_NT(S) meat reduces to NT") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit_gls(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto weight = magmaan::estimate::gmm::normal_theory_weight(*ev, samp,
                                                             est->theta);
  REQUIRE(weight.has_value());

  inf::ModificationIndexOptions nt_opts;
  nt_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = inf::modification_indices(h.pt, h.rep, samp, *est, *weight,
                                      nt_opts);
  REQUIRE(nt.has_value());

  inf::frontier::RobustScoreOptions opts;
  opts.spec.moments = rob::WeightMoments::Unstructured;  // Γ_NT(S) = W⁻¹
  opts.base.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto rob_mi = inf::frontier::modification_indices_robust(h.pt, h.rep, samp,
                                                           *est, *weight, opts);
  REQUIRE(rob_mi.has_value());

  REQUIRE(rob_mi->rows.size() == nt->rows.size());
  REQUIRE(rob_mi->rows.size() > 1);
  for (std::size_t i = 0; i < rob_mi->rows.size(); ++i) {
    const auto& r = rob_mi->rows[i];
    const auto& n = nt->rows[i];
    CHECK(std::abs(r.mi - n.mi) < 1e-9 * (1.0 + std::abs(n.mi)));
    CHECK(std::abs(r.scaling_factor - 1.0) < 1e-7);
    CHECK(std::abs(r.mi_scaled - r.mi) < 1e-7 * (1.0 + std::abs(r.mi)));
  }
}

TEST_CASE("frontier robust LS score test: GLS equality release reduces to NT") {
  auto h = build("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit_gls(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto weight = magmaan::estimate::gmm::normal_theory_weight(*ev, samp,
                                                             est->theta);
  REQUIRE(weight.has_value());

  auto nt = inf::score_tests(h.pt, h.rep, samp, *est, *weight);
  REQUIRE(nt.has_value());
  REQUIRE(nt->rows.size() == 1);

  inf::frontier::RobustScoreOptions opts;
  opts.spec.moments = rob::WeightMoments::Unstructured;
  auto rob_st = inf::frontier::score_tests_robust(h.pt, h.rep, samp, *est,
                                                  *weight, opts);
  REQUIRE(rob_st.has_value());
  REQUIRE(rob_st->rows.size() == 1);

  const auto& r = rob_st->rows[0];
  const auto& n = nt->rows[0];
  CHECK(std::abs(r.mi - n.mi) < 1e-9 * (1.0 + std::abs(n.mi)));
  CHECK(std::abs(r.scaling_factor - 1.0) < 1e-7);
  CHECK(std::abs(r.mi_scaled - n.mi) < 1e-7 * (1.0 + std::abs(n.mi)));
}

TEST_CASE("frontier robust LS MI: DWLS raw path scales; sandwich matches primitives") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(20260612u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 2500, Sigma, 7.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  // Diagonal-ADF (continuous DWLS) weight: W = diag(Γ̂)⁻¹, so W ≠ Γ̂⁻¹ and the
  // robust scaling is genuinely ≠ 1.
  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W_dwls(k, k) = 1.0 / (*G)(k, k);
  magmaan::estimate::gmm::Weight weight{
      magmaan::estimate::gmm::BlockWeight::dense(
          W_dwls, magmaan::FitError::Kind::NumericIssue, "W_dwls").value()};

  auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, weight);
  REQUIRE(est.has_value());

  inf::frontier::RobustScoreOptions opts;
  opts.base.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto rob_raw = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, *est, weight, opts);
  REQUIRE(rob_raw.has_value());
  REQUIRE(rob_raw->rows.size() > 1);
  bool any_scaled = false;
  for (const auto& r : rob_raw->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    CHECK(r.mi_scaled >= 0.0);
    if (std::abs(r.scaling_factor - 1.0) > 0.03) any_scaled = true;
  }
  CHECK(any_scaled);

  // Raw casewise Γ̂ equals feeding the same empirical Γ̂ block explicitly.
  std::vector<Eigen::MatrixXd> gamma_blocks{*G};
  auto rob_gh = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, gamma_blocks, *est, weight, opts);
  REQUIRE(rob_gh.has_value());
  REQUIRE(rob_gh->rows.size() == rob_raw->rows.size());
  for (std::size_t i = 0; i < rob_raw->rows.size(); ++i) {
    CHECK(std::abs(rob_raw->rows[i].scaling_factor -
                   rob_gh->rows[i].scaling_factor) < 1e-8);
  }

  // Independent re-derivation of the moment-metric sandwich from primitives.
  auto sw = magmaan::estimate::continuous_ls_param_space_sandwich(
      h.pt, h.rep, *samp, *est, weight, gamma_blocks);
  REQUIRE(sw.has_value());
  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto Delta = ev->dsigma_dtheta(est->theta);
  REQUIRE(Delta.has_value());
  const Eigen::MatrixXd A1_ref = Delta->transpose() * W_dwls * (*Delta);
  const Eigen::MatrixXd B1_ref =
      Delta->transpose() * W_dwls * (*G) * W_dwls * (*Delta);
  CHECK((sw->A1 - A1_ref).norm() < 1e-8 * (1.0 + A1_ref.norm()));
  CHECK((sw->B1 - B1_ref).norm() < 1e-8 * (1.0 + B1_ref.norm()));
}

TEST_CASE("frontier robust LS score test: raw and supplied Gamma agree") {
  auto h = build("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  std::mt19937 rng(20261001u);
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 1800,
                                         four_indicator_sample_cov(), 7.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto gamma = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(gamma.has_value());
  auto block = magmaan::estimate::gmm::BlockWeight::dense(
      gamma->inverse(), magmaan::FitError::Kind::NumericIssue, "W_adf");
  REQUIRE(block.has_value());
  magmaan::estimate::gmm::Weight weight{*block};
  auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, weight);
  REQUIRE(est.has_value());
  inf::frontier::RobustScoreOptions opts;
  auto from_raw = inf::frontier::score_tests_robust(
      h.pt, h.rep, *samp, raw, *est, weight, opts);
  auto from_gamma = inf::frontier::score_tests_robust(
      h.pt, h.rep, *samp, std::vector<Eigen::MatrixXd>{*gamma}, *est, weight, opts);
  REQUIRE(from_raw.has_value());
  REQUIRE(from_gamma.has_value());
  REQUIRE(from_raw->rows.size() == 1);
  REQUIRE(from_gamma->rows.size() == 1);
  CHECK(from_raw->rows[0].scaling_factor == doctest::Approx(1.0));
  CHECK(from_raw->rows[0].mi_scaled == doctest::Approx(from_gamma->rows[0].mi_scaled));

  for (bool estimated : {false, true}) {
    opts.estimated_weight = estimated;
    opts.spec.cov = rob::ScoreCovariance::BrowneUnbiased;
    auto mi = inf::frontier::modification_indices_robust(
        h.pt, h.rep, *samp, raw, *est, weight, opts);
    auto releases = inf::frontier::score_tests_robust(
        h.pt, h.rep, *samp, raw, *est, weight, opts);
    REQUIRE_FALSE(mi.has_value());
    REQUIRE_FALSE(releases.has_value());
    CHECK(mi.error().detail.find("Browne-unbiased") != std::string::npos);
    CHECK(releases.error().detail.find("Browne-unbiased") != std::string::npos);
  }
  opts.spec.cov = rob::ScoreCovariance::ModelImplied;
  auto mi = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, *est, weight, opts);
  auto releases = inf::frontier::score_tests_robust(
      h.pt, h.rep, *samp, raw, *est, weight, opts);
  REQUIRE_FALSE(mi.has_value());
  REQUIRE_FALSE(releases.has_value());
  CHECK(mi.error().detail.find("requires empirical") != std::string::npos);
  CHECK(releases.error().detail.find("requires empirical") != std::string::npos);
}

TEST_CASE("estimated-weight sandwich_ij: Fixed mode reduces to the fixed-weight "
          "sandwich") {
  // With no IF(Ŵ) correction (mode = Fixed) the IJ meat
  //   B1 = Σ_b (1/N)·(VΔ_b)ᵀ(VΔ_b),  V = g·W
  // must equal the fixed-weight Δ'WΓ̂WΔ from `continuous_ls_param_space_sandwich`
  // built from the SAME raw-data Γ̂ = ZᵀZ/n. This anchors the meat formula.
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(99001122u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 1800, Sigma, 8.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W_dwls(k, k) = 1.0 / (*G)(k, k);
  magmaan::estimate::gmm::Weight weight{
      magmaan::estimate::gmm::BlockWeight::dense(
          W_dwls, magmaan::FitError::Kind::NumericIssue, "W_dwls").value()};
  auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, weight);
  REQUIRE(est.has_value());

  auto sw_plain = magmaan::estimate::continuous_ls_param_space_sandwich(
      h.pt, h.rep, *samp, *est, weight, raw);
  REQUIRE(sw_plain.has_value());
  auto sw_ij = magmaan::estimate::continuous_ls_param_space_sandwich_ij(
      h.pt, h.rep, *samp, *est, weight, raw,
      magmaan::estimate::ContinuousLsIJWeightMode::Fixed);
  REQUIRE(sw_ij.has_value());

  CHECK((sw_ij->A1 - sw_plain->A1).norm() < 1e-9 * (1.0 + sw_plain->A1.norm()));
  CHECK((sw_ij->B1 - sw_plain->B1).norm() < 1e-9 * (1.0 + sw_plain->B1.norm()));
}

TEST_CASE("frontier robust LS MI: estimated-weight DWLS meat shifts the scaling") {
  // The complete (estimated-weight) sandwich adds the data-dependent-weight
  // IF(Ŵ) meat term, absent from lavaan's MI. On non-normal data with a DWLS
  // weight (W ≠ Γ̂⁻¹) it must move the per-direction scaling c away from the
  // fixed-weight value; with mode = Fixed it must coincide with it exactly.
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(20260619u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 3000, Sigma, 6.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W_dwls(k, k) = 1.0 / (*G)(k, k);
  magmaan::estimate::gmm::Weight weight{
      magmaan::estimate::gmm::BlockWeight::dense(
          W_dwls, magmaan::FitError::Kind::NumericIssue, "W_dwls").value()};
  auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, weight);
  REQUIRE(est.has_value());

  inf::frontier::RobustScoreOptions fixed_opts;
  fixed_opts.base.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto rob_fixed = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, *est, weight, fixed_opts);
  REQUIRE(rob_fixed.has_value());

  inf::frontier::RobustScoreOptions ew_opts = fixed_opts;
  ew_opts.estimated_weight = true;
  ew_opts.ij_weight_mode =
      magmaan::estimate::ContinuousLsIJWeightMode::SampleEmpiricalDwls;
  auto rob_ew = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, *est, weight, ew_opts);
  REQUIRE(rob_ew.has_value());
  REQUIRE(rob_ew->rows.size() == rob_fixed->rows.size());
  REQUIRE(rob_ew->rows.size() > 1);

  bool any_shift = false;
  for (std::size_t i = 0; i < rob_ew->rows.size(); ++i) {
    const auto& r = rob_ew->rows[i];
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    CHECK(r.mi_scaled >= 0.0);
    // unscaled mi is the same ordinary statistic; only the denominator changed
    CHECK(std::abs(r.mi - rob_fixed->rows[i].mi) < 1e-8 * (1.0 + std::abs(r.mi)));
    if (std::abs(r.scaling_factor - rob_fixed->rows[i].scaling_factor) > 0.02)
      any_shift = true;
  }
  CHECK(any_shift);

  // mode = Fixed ⇒ the estimated-weight path reproduces the fixed-weight one.
  inf::frontier::RobustScoreOptions ew_fixed = fixed_opts;
  ew_fixed.estimated_weight = true;
  ew_fixed.ij_weight_mode = magmaan::estimate::ContinuousLsIJWeightMode::Fixed;
  auto rob_ewf = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, *est, weight, ew_fixed);
  REQUIRE(rob_ewf.has_value());
  REQUIRE(rob_ewf->rows.size() == rob_fixed->rows.size());
  for (std::size_t i = 0; i < rob_fixed->rows.size(); ++i) {
    CHECK(std::abs(rob_ewf->rows[i].scaling_factor -
                   rob_fixed->rows[i].scaling_factor) < 1e-8);
  }
}

TEST_CASE("frontier robust LS MI: an estimated-weight recipe must reproduce the fitting weight") {
  // The IJ correction is the derivative of the weight its recipe rebuilds.
  // A fit made with one weight and scaled with another recipe's influence
  // describes neither estimator, so such calls are refused with a typed
  // reason. The same check guards every continuous IJ consumer.
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(20261002u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(multivariate_t_sample(rng, 1500, Sigma, 7.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  const Eigen::VectorXd theta0 = Eigen::VectorXd::Ones(h.pt.n_free());
  using Kind = magmaan::estimate::gmm::FixedWeightKind;
  using Mode = magmaan::estimate::ContinuousLsIJWeightMode;

  auto fit_with = [&](Kind kind, double a) {
    auto w = magmaan::estimate::gmm::fixed_moment_weight(
        *ev, *samp, theta0, kind, &raw,
        magmaan::estimate::gmm::FixedWeightOptions{a});
    REQUIRE(w.has_value());
    auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, *w);
    REQUIRE(est.has_value());
    return std::pair{*w, *est};
  };
  auto ew = [&](Mode mode, double a) {
    inf::frontier::RobustScoreOptions opts;
    opts.base.candidates = inf::ScoreCandidateSet::WithAbsentRows;
    opts.estimated_weight = true;
    opts.ij_weight_mode = mode;
    opts.dls_opts.a = a;
    return opts;
  };
  auto refused = [](const auto& r) {
    return !r.has_value() &&
           r.error().kind == magmaan::PostError::Kind::UnsupportedInference;
  };

  // Each recipe accepts its own fitting weight.
  for (auto [kind, mode] : {std::pair{Kind::Nt, Mode::SampleNormalTheory},
                            std::pair{Kind::Dwls, Mode::SampleEmpiricalDwls},
                            std::pair{Kind::Wls, Mode::SampleEmpiricalWls}}) {
    auto [w, est] = fit_with(kind, 0.5);
    auto mi = inf::frontier::modification_indices_robust(
        h.pt, h.rep, *samp, raw, est, w, ew(mode, 0.5));
    REQUIRE_MESSAGE(mi.has_value(), (mi.has_value() ? "" : mi.error().detail));
    CHECK(mi->rows.size() > 1);
  }

  // A DWLS fit scaled with the ADF or DLS influence is refused, in MI and in
  // the shared sandwich used by SEs.
  auto [w_dwls, est_dwls] = fit_with(Kind::Dwls, 0.5);
  CHECK(refused(inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, est_dwls, w_dwls,
      ew(Mode::SampleEmpiricalWls, 0.5))));
  CHECK(refused(inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, est_dwls, w_dwls, ew(Mode::SampleDls, 0.5))));
  CHECK(refused(magmaan::estimate::continuous_ls_param_space_sandwich_ij(
      h.pt, h.rep, *samp, est_dwls, w_dwls, raw, Mode::SampleEmpiricalWls)));

  // DLS must use the fit's own mixing scalar.
  auto [w_dls, est_dls] = fit_with(Kind::Dls, 0.3);
  CHECK(inf::frontier::modification_indices_robust(
            h.pt, h.rep, *samp, raw, est_dls, w_dls, ew(Mode::SampleDls, 0.3))
            .has_value());
  CHECK(refused(inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, est_dls, w_dls, ew(Mode::SampleDls, 0.5))));

  // A rescaled weight is a different weight for the robust scaling.
  auto w_scaled = w_dwls;
  w_scaled[0] = magmaan::estimate::gmm::BlockWeight::dense(
      2.0 * w_dwls[0].to_dense(), magmaan::FitError::Kind::NumericIssue,
      "scaled").value();
  CHECK(refused(inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, est_dwls, w_scaled,
      ew(Mode::SampleEmpiricalDwls, 0.5))));
}

TEST_CASE("estimated-weight IJ mode follows the recorded weight recipe") {
  using Kind = magmaan::estimate::gmm::FixedWeightKind;
  using Mode = magmaan::estimate::ContinuousLsIJWeightMode;
  const std::pair<Kind, Mode> expected[] = {
      {Kind::Uls, Mode::Fixed},
      {Kind::Nt, Mode::SampleNormalTheory},
      {Kind::Dwls, Mode::SampleEmpiricalDwls},
      {Kind::Wls, Mode::SampleEmpiricalWls},
      {Kind::Dls, Mode::SampleDls}};
  for (const auto& [kind, mode] : expected) {
    auto m = magmaan::estimate::continuous_ls_ij_mode_for(kind, false);
    REQUIRE(m.has_value());
    CHECK(*m == mode);
  }
  // A supplied weight has no recipe, so its influence is unknown.
  for (const auto& [kind, mode] : expected) {
    (void)mode;
    auto m = magmaan::estimate::continuous_ls_ij_mode_for(kind, true);
    REQUIRE_FALSE(m.has_value());
    CHECK(m.error().kind == magmaan::PostError::Kind::UnsupportedInference);
  }
}


// ── Ordinal tier ─────────────────────────────────────────────────────────────
// The exact-reduction anchor: the full-WLS weight is the NACOV inverse, so the
// NACOV meat collapses onto the bread (c ≡ 1) and the robust statistic equals
// the ordinary one.

namespace {

Eigen::MatrixXd ordinal_three_cat_sample(
    std::mt19937& rng,
    Eigen::Index n,
    const std::array<double, 4>& loading,
    double lo = -0.50,
    double hi = 0.45) {
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const auto idx = static_cast<std::size_t>(j);
      const double eps =
          std::sqrt(1.0 - loading[idx] * loading[idx]) * norm(rng);
      const double y = loading[idx] * eta + eps;
      X(i, j) = 1.0 + (y > lo) + (y > hi);
    }
  }
  return X;
}

Eigen::MatrixXd ordinal_three_cat_sample(std::mt19937& rng, Eigen::Index n) {
  return ordinal_three_cat_sample(rng, n, {0.88, 0.80, 0.72, 0.64});
}

Eigen::MatrixXd mixed_ordinal_sample(std::mt19937& rng,
                                     Eigen::Index n,
                                     const std::array<double, 4>& loading,
                                     double shift) {
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    const double y0 = loading[0] * eta + 0.45 * norm(rng) + shift;
    const double y1 = loading[1] * eta + 0.55 * norm(rng) - shift;
    X(i, 0) = 1.0 + (y0 > -0.60) + (y0 > 0.40);
    X(i, 1) = 1.0 + (y1 > 0.10);
    X(i, 2) = loading[2] * eta + 0.65 * norm(rng) + 0.20 + shift;
    X(i, 3) = loading[3] * eta + 0.70 * norm(rng) - 0.10 - shift;
  }
  return X;
}

constexpr const char* ordinal_cfa_syntax =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 | t1 + t2\n"
    "x2 | t1 + t2\n"
    "x3 | t1 + t2\n"
    "x4 | t1 + t2\n"
    "x1 ~*~ 1*x1\n"
    "x2 ~*~ 1*x2\n"
    "x3 ~*~ 1*x3\n"
    "x4 ~*~ 1*x4\n";

constexpr const char* ordinal_cfa_eq_syntax =
    "f =~ x1 + a*x2 + b*x3 + x4\n"
    "x1 | t1 + t2\n"
    "x2 | t1 + t2\n"
    "x3 | t1 + t2\n"
    "x4 | t1 + t2\n"
    "x1 ~*~ 1*x1\n"
    "x2 ~*~ 1*x2\n"
    "x3 ~*~ 1*x3\n"
    "x4 ~*~ 1*x4\n"
    "a == b\n";

constexpr const char* ordinal_cfa_mg_eq_syntax =
    "f =~ x1 + c(a2,b2)*x2 + c(a3,b3)*x3 + c(a4,b4)*x4\n"
    "x1 | t1 + t2\n"
    "x2 | t1 + t2\n"
    "x3 | t1 + t2\n"
    "x4 | t1 + t2\n"
    "x1 ~*~ 1*x1\n"
    "x2 ~*~ 1*x2\n"
    "x3 ~*~ 1*x3\n"
    "x4 ~*~ 1*x4\n"
    "a2 == b2\n"
    "a3 == b3\n"
    "a4 == b4\n";

constexpr const char* mixed_ordinal_mg_eq_syntax =
    "f =~ x1 + c(a2,b2)*x2 + c(a3,b3)*x3 + x4\n"
    "x1 | t1 + t2\n"
    "x2 | t1\n"
    "x1 ~*~ 1*x1\n"
    "x2 ~*~ 1*x2\n"
    "a2 == b2\n"
    "a3 == b3\n";

}  // namespace

TEST_CASE("ordinal score rank: latent units and nearby points preserve candidates") {
  std::mt19937 rng(20261002u);
  auto stats = magmaan::data::ordinal_stats_from_integer_data(
      {ordinal_three_cat_sample(rng, 700)});
  REQUIRE(stats.has_value());
  auto h = build(ordinal_cfa_syntax);
  auto prepared = magmaan::estimate::prepare_ordinal_delta_partable(h.pt, *stats);
  REQUIRE(prepared.has_value());
  inf::ModificationIndexOptions opts;
  opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  for (auto weight : {magmaan::estimate::OrdinalWeightKind::ULS,
                      magmaan::estimate::OrdinalWeightKind::DWLS,
                      magmaan::estimate::OrdinalWeightKind::WLS}) {
    auto fit = magmaan::test::fit_ordinal_bounded(h.pt, h.rep, *stats, {}, weight);
    REQUIRE(fit.has_value());
    auto baseline = magmaan::estimate::modification_indices_ordinal(
        h.pt, h.rep, *stats, *fit, weight, opts);
    REQUIRE(baseline.has_value());
    REQUIRE(baseline->rows.size() == 6);
    for (double nearby : {0.0, 1e-7}) {
      for (double units : {0.01, 1.0, 100.0}) {
        CAPTURE(units);
        CAPTURE(nearby);
        auto pt = h.pt;
        auto est = *fit;
        est.theta(0) += nearby;
        for (std::size_t r = 0; r < pt.size(); ++r) {
          double factor = 1.0;
          if (pt.op[r] == magmaan::parse::Op::Measurement) factor = units;
          if (pt.op[r] == magmaan::parse::Op::Covariance &&
              pt.lhs_var[r] == pt.rhs_var[r] &&
              pt.var_role[static_cast<std::size_t>(pt.lhs_var[r])] == magmaan::spec::VarRole::Latent) {
            factor = 1.0 / (units * units);
          }
          if (pt.free[r] > 0) est.theta(pt.free[r] - 1) *= factor;
          else if (std::isfinite(pt.fixed_value[r])) pt.fixed_value[r] *= factor;
        }
        auto ordinary = magmaan::estimate::modification_indices_ordinal(
            pt, h.rep, *stats, est, weight, opts);
        auto robust = magmaan::estimate::frontier::modification_indices_ordinal_robust(
            pt, h.rep, *stats, est, weight, opts);
        REQUIRE(ordinary.has_value());
        REQUIRE(robust.has_value());
        REQUIRE(ordinary->rows.size() == baseline->rows.size());
        REQUIRE(robust->rows.size() == ordinary->rows.size());
        for (std::size_t i = 0; i < ordinary->rows.size(); ++i) {
          CHECK(ordinary->rows[i].candidate.row == baseline->rows[i].candidate.row);
          CHECK(robust->rows[i].candidate.row == ordinary->rows[i].candidate.row);
          CHECK(ordinary->rows[i].mi == doctest::Approx(robust->rows[i].mi));
          if (nearby == 0.0) {
            CHECK(ordinary->rows[i].mi == doctest::Approx(baseline->rows[i].mi));
            CHECK(ordinary->rows[i].epc == doctest::Approx(baseline->rows[i].epc));
          }
        }
      }
    }
  }
}

TEST_CASE("frontier robust ordinal MI: WLS + NACOV meat reduces to ordinary") {
  std::mt19937 rng(20260612u);
  auto stats =
      magmaan::data::ordinal_stats_from_integer_data({ordinal_three_cat_sample(rng, 700)});
  REQUIRE(stats.has_value());
  auto h = build(ordinal_cfa_syntax);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = magmaan::estimate::modification_indices_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
      mi_opts);
  REQUIRE(nt.has_value());
  auto rob = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
      mi_opts);
  REQUIRE(rob.has_value());

  REQUIRE(rob->rows.size() == nt->rows.size());
  REQUIRE(rob->rows.size() > 1);
  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    const auto& r = rob->rows[i];
    const auto& n = nt->rows[i];
    CHECK(std::abs(r.mi - n.mi) < 1e-9 * (1.0 + std::abs(n.mi)));
    CHECK(std::abs(r.scaling_factor - 1.0) < 1e-6);
    CHECK(std::abs(r.mi_scaled - r.mi) < 1e-6 * (1.0 + std::abs(r.mi)));
  }
}

TEST_CASE("frontier robust ordinal score test: WLS equality release reduces") {
  std::mt19937 rng(20260613u);
  auto stats =
      magmaan::data::ordinal_stats_from_integer_data({ordinal_three_cat_sample(rng, 700)});
  REQUIRE(stats.has_value());
  auto h = build(ordinal_cfa_eq_syntax);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(est.has_value());

  auto nt = magmaan::estimate::score_tests_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(nt.has_value());
  REQUIRE(nt->rows.size() == 1);
  auto rob = magmaan::estimate::frontier::score_tests_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == 1);

  const auto& r = rob->rows[0];
  const auto& n = nt->rows[0];
  CHECK(std::abs(r.mi - n.mi) < 1e-9 * (1.0 + std::abs(n.mi)));
  CHECK(std::abs(r.scaling_factor - 1.0) < 1e-6);
  CHECK(std::abs(r.mi_scaled - n.mi) < 1e-6 * (1.0 + std::abs(n.mi)));
}

TEST_CASE("frontier robust ordinal MI: DWLS scales against the NACOV meat") {
  std::mt19937 rng(20260614u);
  auto stats =
      magmaan::data::ordinal_stats_from_integer_data({ordinal_three_cat_sample(rng, 700)});
  REQUIRE(stats.has_value());
  auto h = build(ordinal_cfa_syntax);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = magmaan::estimate::modification_indices_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::DWLS,
      mi_opts);
  REQUIRE(nt.has_value());
  auto rob = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::DWLS,
      mi_opts);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == nt->rows.size());
  REQUIRE(rob->rows.size() > 1);

  bool any_scaled = false;
  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    const auto& r = rob->rows[i];
    // NT component identical to the non-robust ordinal sweep.
    CHECK(std::abs(r.mi - nt->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt->rows[i].mi)));
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    if (std::abs(r.scaling_factor - 1.0) > 0.01) any_scaled = true;
  }
  CHECK(any_scaled);

  // ULS rides the identity weight against the same NACOV meat (ULSMV-style).
  auto est_uls = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE(est_uls.has_value());
  auto rob_uls = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, *stats, *est_uls, magmaan::estimate::OrdinalWeightKind::ULS,
      mi_opts);
  REQUIRE(rob_uls.has_value());
  for (const auto& r : rob_uls->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
  }

  // Missing NACOV is a hard error, not a silent NT fallback.
  auto stats_no_nacov = *stats;
  stats_no_nacov.NACOV.clear();
  auto rob_missing =
      magmaan::estimate::frontier::modification_indices_ordinal_robust(
          h.pt, h.rep, stats_no_nacov, *est,
          magmaan::estimate::OrdinalWeightKind::DWLS, mi_opts);
  CHECK_FALSE(rob_missing.has_value());
}

TEST_CASE("frontier robust ordinal MI: estimated-weight DWLS shifts the scaling") {
  // The complete (Hall-Inoue) sandwich adds the polychoric-weight IF(Ŵ) meat
  // term to the per-direction scaling — beyond lavaan's global SB scalar. The
  // weight-influence is leading-order only under MISSPECIFICATION, so the data
  // carry an x1–x2 residual association the single-factor model omits; against
  // that misfit the estimated-weight c must move off the fixed-weight value.
  std::mt19937 rng(20260619u);
  auto misspec_ordinal = [&](Eigen::Index n) {
    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::MatrixXd X(n, 4);
    for (Eigen::Index i = 0; i < n; ++i) {
      const double eta = norm(rng);
      const double nuis = norm(rng);  // extra factor on x1,x2 only (omitted)
      for (Eigen::Index j = 0; j < 4; ++j) {
        const double extra = (j < 2) ? 0.40 * nuis : 0.0;
        const double rsd = (j < 2) ? std::sqrt(0.35) : std::sqrt(0.51);
        const double y = 0.70 * eta + extra + rsd * norm(rng);
        X(i, j) = 1.0 + (y > -0.50) + (y > 0.45);
      }
    }
    return X;
  };
  auto stats =
      magmaan::data::ordinal_stats_from_integer_data({misspec_ordinal(1200)});
  REQUIRE(stats.has_value());
  auto h = build(ordinal_cfa_syntax);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto fixed = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::DWLS,
      mi_opts, magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE(fixed.has_value());
  auto ew = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::DWLS,
      mi_opts, magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/true);
  REQUIRE(ew.has_value());
  REQUIRE(ew->rows.size() == fixed->rows.size());
  REQUIRE(ew->rows.size() > 1);

  bool any_shift = false;
  for (std::size_t i = 0; i < ew->rows.size(); ++i) {
    const auto& r = ew->rows[i];
    // The ordinary mi is unchanged; only the robust denominator moved.
    CHECK(std::abs(r.mi - fixed->rows[i].mi) <
          1e-9 * (1.0 + std::abs(fixed->rows[i].mi)));
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    CHECK(r.mi_scaled >= 0.0);
    if (std::abs(r.scaling_factor - fixed->rows[i].scaling_factor) > 0.01)
      any_shift = true;
  }
  CHECK(any_shift);

  // Mixed-ordinal estimated-weight is not yet wired: it must error, not silently
  // fall back. (Exercised via the all-ordinal guard message path here through a
  // missing-influence stats object.)
  auto stats_no_infl = *stats;
  stats_no_infl.moment_influence.clear();
  auto ew_bad = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, stats_no_infl, *est,
      magmaan::estimate::OrdinalWeightKind::DWLS, mi_opts,
      magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/true);
  CHECK_FALSE(ew_bad.has_value());
}

TEST_CASE("frontier robust ordinal inference: estimated weight refuses NT, DLS and supplied weights") {
  // NT, DLS and supplied ordinal weights occupy the DWLS/WLS slots under the
  // computational labels. Their fixed-weight inference is valid; the
  // estimated-weight channel is the NACOV influence and does not apply to
  // them, so it is refused in MI, SEs and the DWLS profile family alike.
  std::mt19937 rng(20261002u);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(900, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    const double nuis = norm(rng);
    for (Eigen::Index j = 0; j < 4; ++j) {
      const double extra = (j < 2) ? 0.40 * nuis : 0.0;
      const double rsd = (j < 2) ? std::sqrt(0.35) : std::sqrt(0.51);
      const double y = 0.70 * eta + extra + rsd * norm(rng);
      X(i, j) = 1.0 + (y > -0.50) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  auto h = build(ordinal_cfa_syntax);
  using W = magmaan::estimate::OrdinalWeightKind;
  const auto delta = magmaan::estimate::OrdinalParameterization::Delta;
  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto refused = [](const auto& r) {
    return !r.has_value() &&
           r.error().kind == magmaan::PostError::Kind::UnsupportedInference;
  };
  auto mi = [&](const magmaan::data::OrdinalStats& s,
                const magmaan::estimate::Estimates& est, W weights,
                bool estimated) {
    return magmaan::estimate::frontier::modification_indices_ordinal_robust(
        h.pt, h.rep, s, est, weights, mi_opts, delta, estimated);
  };

  // The default WLS weight keeps its estimated-weight inference.
  auto est_wls = magmaan::test::fit_ordinal_bounded(h.pt, h.rep, *stats, {},
                                                    W::WLS);
  REQUIRE(est_wls.has_value());
  CHECK(mi(*stats, *est_wls, W::WLS, true).has_value());

  // NT and DLS weights: fixed-weight inference only.
  using Stage2 = magmaan::estimate::frontier::OrdinalStage2Weight;
  for (Stage2 kind : {Stage2::Nt, Stage2::Dls}) {
    auto swapped = magmaan::estimate::frontier::ordinal_stats_with_stage2_weight(
        *stats, kind, {0.4});
    REQUIRE(swapped.has_value());
    auto est = magmaan::test::fit_ordinal_bounded(h.pt, h.rep, *swapped, {},
                                                  W::WLS);
    REQUIRE(est.has_value());
    CHECK(mi(*swapped, *est, W::WLS, false).has_value());
    CHECK(refused(mi(*swapped, *est, W::WLS, true)));
    CHECK(refused(magmaan::estimate::robust_ordinal_ij(h.pt, h.rep, *swapped,
                                                       *est, W::WLS)));
  }

  // A supplied DWLS weight: same rule, including the profile family, whose
  // fixed-weight comparator stays available.
  auto supplied = *stats;
  supplied.W_dwls[0] *= 1.5;
  auto est_dwls = magmaan::test::fit_ordinal_bounded(h.pt, h.rep, supplied, {},
                                                     W::DWLS);
  REQUIRE(est_dwls.has_value());
  CHECK(mi(supplied, *est_dwls, W::DWLS, false).has_value());
  CHECK(refused(mi(supplied, *est_dwls, W::DWLS, true)));
  CHECK(refused(magmaan::estimate::ordinal_dwls_profile_rmsea(
      h.pt, h.rep, supplied, *est_dwls, delta)));
  CHECK(refused(magmaan::estimate::ordinal_rmsea_misspec_inference(
      h.pt, h.rep, supplied, *est_dwls, delta, /*estimated_weight=*/true)));
  auto fixed_rmsea = magmaan::estimate::ordinal_rmsea_misspec_inference(
      h.pt, h.rep, supplied, *est_dwls, delta, /*estimated_weight=*/false);
  CHECK_MESSAGE(fixed_rmsea.has_value(),
                (fixed_rmsea.has_value() ? "" : fixed_rmsea.error().detail));
}

TEST_CASE("frontier robust mixed ordinal: WLS reduces, DWLS finite, ULS rejected") {
  std::mt19937 rng(20260615u);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(800, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    // x1 ordinal (3 categories), x2 binary, x3/x4 continuous.
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.80 * eta + 0.60 * norm(rng) + 0.2;
    X(i, 3) = 0.70 * eta + 0.71 * norm(rng) - 0.1;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  auto fp = Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  Handles h{std::move(*pt), std::move(*rep)};

  auto est = magmaan::test::fit_mixed_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = magmaan::estimate::modification_indices_mixed_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
      mi_opts);
  REQUIRE(nt.has_value());
  auto rob =
      magmaan::estimate::frontier::modification_indices_mixed_ordinal_robust(
          h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
          mi_opts);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == nt->rows.size());
  REQUIRE(rob->rows.size() > 0);
  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    CHECK(std::abs(rob->rows[i].mi - nt->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt->rows[i].mi)));
    CHECK(std::abs(rob->rows[i].scaling_factor - 1.0) < 1e-6);
  }

  auto est_dwls = magmaan::test::fit_mixed_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(est_dwls.has_value());
  auto rob_dwls =
      magmaan::estimate::frontier::modification_indices_mixed_ordinal_robust(
          h.pt, h.rep, *stats, *est_dwls,
          magmaan::estimate::OrdinalWeightKind::DWLS, mi_opts);
  REQUIRE(rob_dwls.has_value());
  for (const auto& r : rob_dwls->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
  }

  auto rob_uls =
      magmaan::estimate::frontier::modification_indices_mixed_ordinal_robust(
          h.pt, h.rep, *stats, *est_dwls,
          magmaan::estimate::OrdinalWeightKind::ULS, mi_opts);
  CHECK_FALSE(rob_uls.has_value());
}

TEST_CASE("frontier robust ordinal MI multi-group: WLS reduces to ordinary") {
  std::mt19937 rng(20260616u);
  std::vector<Eigen::MatrixXd> blocks;
  blocks.push_back(ordinal_three_cat_sample(
      rng, 720, {0.88, 0.80, 0.72, 0.64}));
  blocks.push_back(ordinal_three_cat_sample(
      rng, 760, {0.84, 0.76, 0.68, 0.70}, -0.45, 0.55));
  auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks);
  REQUIRE(stats.has_value());

  auto h = build_groups(ordinal_cfa_syntax, 2);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = magmaan::estimate::modification_indices_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
      mi_opts);
  REQUIRE(nt.has_value());
  auto rob = magmaan::estimate::frontier::modification_indices_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
      mi_opts);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == nt->rows.size());
  REQUIRE(rob->rows.size() > 1);

  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    CHECK(std::abs(rob->rows[i].mi - nt->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt->rows[i].mi)));
    CHECK(std::abs(rob->rows[i].scaling_factor - 1.0) < 1e-6);
    CHECK(std::abs(rob->rows[i].mi_scaled - nt->rows[i].mi) <
          1e-6 * (1.0 + std::abs(nt->rows[i].mi)));
  }
}

TEST_CASE("frontier robust ordinal score test multi-group: WLS equality release reduces") {
  std::mt19937 rng(20260617u);
  std::vector<Eigen::MatrixXd> blocks;
  blocks.push_back(ordinal_three_cat_sample(
      rng, 720, {0.88, 0.80, 0.72, 0.64}));
  blocks.push_back(ordinal_three_cat_sample(
      rng, 760, {0.84, 0.76, 0.68, 0.70}, -0.45, 0.55));
  auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks);
  REQUIRE(stats.has_value());

  auto h = build_groups(ordinal_cfa_mg_eq_syntax, 2);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(est.has_value());

  auto nt = magmaan::estimate::score_tests_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(nt.has_value());
  REQUIRE(nt->rows.size() == 3);
  auto rob = magmaan::estimate::frontier::score_tests_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == nt->rows.size());

  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    CHECK(std::abs(rob->rows[i].mi - nt->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt->rows[i].mi)));
    CHECK(std::abs(rob->rows[i].scaling_factor - 1.0) < 1e-6);
    CHECK(std::abs(rob->rows[i].mi_scaled - nt->rows[i].mi) <
          1e-6 * (1.0 + std::abs(nt->rows[i].mi)));
  }
}

TEST_CASE("frontier robust mixed ordinal multi-group: WLS reductions cover MI and score") {
  std::mt19937 rng(20260618u);
  std::vector<Eigen::MatrixXd> blocks;
  blocks.push_back(mixed_ordinal_sample(
      rng, 900, {0.84, 0.74, 0.78, 0.70}, 0.00));
  blocks.push_back(mixed_ordinal_sample(
      rng, 940, {0.78, 0.68, 0.72, 0.76}, 0.12));
  const std::vector<std::vector<std::int32_t>> ordered = {
      {1, 1, 0, 0}, {1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered);
  REQUIRE(stats.has_value());

  auto h = build_groups_mean(mixed_ordinal_mg_eq_syntax, 2);
  auto est = magmaan::test::fit_mixed_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt_mi = magmaan::estimate::modification_indices_mixed_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
      mi_opts);
  REQUIRE(nt_mi.has_value());
  auto rob_mi =
      magmaan::estimate::frontier::modification_indices_mixed_ordinal_robust(
          h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS,
          mi_opts);
  REQUIRE(rob_mi.has_value());
  REQUIRE(rob_mi->rows.size() == nt_mi->rows.size());

  for (std::size_t i = 0; i < rob_mi->rows.size(); ++i) {
    CHECK(std::abs(rob_mi->rows[i].mi - nt_mi->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt_mi->rows[i].mi)));
    CHECK(std::abs(rob_mi->rows[i].scaling_factor - 1.0) < 1e-6);
    CHECK(std::abs(rob_mi->rows[i].mi_scaled - nt_mi->rows[i].mi) <
          1e-6 * (1.0 + std::abs(nt_mi->rows[i].mi)));
  }

  auto nt_st = magmaan::estimate::score_tests_mixed_ordinal(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(nt_st.has_value());
  REQUIRE(nt_st->rows.size() == 2);
  auto rob_st = magmaan::estimate::frontier::score_tests_mixed_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE(rob_st.has_value());
  REQUIRE(rob_st->rows.size() == nt_st->rows.size());

  for (std::size_t i = 0; i < rob_st->rows.size(); ++i) {
    CHECK(std::abs(rob_st->rows[i].mi - nt_st->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt_st->rows[i].mi)));
    CHECK(std::abs(rob_st->rows[i].scaling_factor - 1.0) < 1e-6);
    CHECK(std::abs(rob_st->rows[i].mi_scaled - nt_st->rows[i].mi) <
          1e-6 * (1.0 + std::abs(nt_st->rows[i].mi)));
  }
}

TEST_CASE("frontier robust ordinal score test multi-group: DWLS scales finite") {
  std::mt19937 rng(20260619u);
  std::vector<Eigen::MatrixXd> blocks;
  blocks.push_back(ordinal_three_cat_sample(
      rng, 620, {0.92, 0.82, 0.74, 0.70}));
  blocks.push_back(ordinal_three_cat_sample(
      rng, 710, {0.88, 0.70, 0.84, 0.62}, -0.55, 0.50));
  auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks);
  REQUIRE(stats.has_value());

  auto h = build_groups(ordinal_cfa_mg_eq_syntax, 2);
  auto est = magmaan::test::fit_ordinal_bounded(
      h.pt, h.rep, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(est.has_value());
  auto rob = magmaan::estimate::frontier::score_tests_ordinal_robust(
      h.pt, h.rep, *stats, *est, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == 3);

  bool any_scaled = false;
  for (const auto& r : rob->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
    CHECK(r.mi_scaled >= 0.0);
    if (std::abs(r.scaling_factor - 1.0) > 0.05) any_scaled = true;
  }
  CHECK(any_scaled);
}

// ── FIML robust tier ─────────────────────────────────────────────────────────
// The robust path shares the candidate enumeration and the NT score/information
// with the non-robust FIML MI, so the unscaled `mi` must match. Both now use
// analytic observed information. The scale of the sandwich meat
// (B1 = ¼·scoresᵀscores against A1 = (N/2)·H) is
// pinned by c → 1 on a correctly specified large-n normal model — a wrong
// constant (2× or ½×) would push c far from 1 — and exactly by golden 0009.

TEST_CASE("frontier FIML robust MI: unscaled mi matches the non-robust FIML MI") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260613u);
  const auto raw = gaussian_cfa_raw(rng, 600, 9);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 800;
  auto est = magmaan::test::fit_fiml(h.pt, h.rep, raw, opts);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = inf::modification_indices_fiml(h.pt, h.rep, raw, *est, mi_opts);
  REQUIRE(nt.has_value());
  auto rob = inf::frontier::modification_indices_fiml_robust(h.pt, h.rep, raw,
                                                             *est, mi_opts);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == nt->rows.size());
  REQUIRE(rob->rows.size() == 6);
  auto nt_other_step = inf::modification_indices_fiml(
      h.pt, h.rep, raw, *est, mi_opts, inf::FIML{}, 1e-2);
  REQUIRE(nt_other_step.has_value());
  REQUIRE(nt_other_step->rows.size() == nt->rows.size());
  auto invalid_step = inf::modification_indices_fiml(
      h.pt, h.rep, raw, *est, mi_opts, inf::FIML{}, 0.0);
  REQUIRE_FALSE(invalid_step.has_value());
  CHECK(invalid_step.error().kind == magmaan::PostError::Kind::NumericIssue);
  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    CHECK(std::abs(rob->rows[i].mi - nt->rows[i].mi) <
          1e-8 * (1.0 + std::abs(nt->rows[i].mi)));
    CHECK(nt_other_step->rows[i].mi == nt->rows[i].mi);
    CHECK(rob->rows[i].df == 1);
    CHECK(std::isfinite(rob->rows[i].scaling_factor));
    CHECK(rob->rows[i].scaling_factor > 0.0);
    CHECK(std::abs(rob->rows[i].mi_scaled -
                   rob->rows[i].mi / rob->rows[i].scaling_factor) <
          1e-9 * (1.0 + std::abs(rob->rows[i].mi)));
  }
}

TEST_CASE("frontier FIML robust MI: scaling approaches 1 on large-n normal data") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(424242u);
  const auto raw = gaussian_cfa_raw(rng, 4000, 11);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1000;
  auto est = magmaan::test::fit_fiml(h.pt, h.rep, raw, opts);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto rob = inf::frontier::modification_indices_fiml_robust(h.pt, h.rep, raw,
                                                             *est, mi_opts);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == 6);
  for (double displacement : {-1e-6, 1e-6}) {
    auto nearby = *est;
    nearby.theta *= 1.0 + displacement;
    auto candidate_tests = inf::frontier::modification_indices_fiml_robust(
        h.pt, h.rep, raw, nearby, mi_opts);
    REQUIRE(candidate_tests.has_value());
    REQUIRE(candidate_tests->rows.size() == 6);
    for (const auto& row : candidate_tests->rows) {
      CHECK(row.candidate.op != magmaan::parse::Op::Measurement);
      CHECK(row.information > 0.0);
    }
  }
  // Per-candidate c is a noisy 4th-moment-driven ratio, so anchor the *mean*
  // (variance ~1/rows lower) at 1; a wrong B1/A1 scale constant (2× or ½×)
  // would instead push every row to |c−1| ≈ 1 or ½, which max_dev catches.
  // Freeing the sole marker changes identification, not the fitted moments.
  double sum_c = 0.0;
  double max_dev = 0.0;
  std::size_t used = 0;
  for (const auto& r : rob->rows) {
    CHECK(r.candidate.op != magmaan::parse::Op::Measurement);
    ++used;
    sum_c += r.scaling_factor;
    max_dev = std::max(max_dev, std::abs(r.scaling_factor - 1.0));
  }
  REQUIRE(used > 0);
  const double mean_c = sum_c / static_cast<double>(used);
  CHECK(std::abs(mean_c - 1.0) < 0.12);
  CHECK(max_dev < 0.5);
}

TEST_CASE("frontier FIML robust score test: equality release runs and is finite") {
  auto h = build_mean("f =~ x1 + c*x2 + c*x3 + x4");
  std::mt19937 rng(99u);
  const auto raw = gaussian_cfa_raw(rng, 800, 9);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1000;
  auto est = magmaan::test::fit_fiml(h.pt, h.rep, raw, opts);
  REQUIRE(est.has_value());

  auto rob = inf::frontier::score_tests_fiml_robust(h.pt, h.rep, raw, *est);
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() > 0);
  for (const auto& r : rob->rows) {
    CHECK(r.df == 1);
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    CHECK(std::isfinite(r.mi_scaled));
  }
}

TEST_CASE("frontier FIML robust MI and score tests support multi-group raw blocks") {
  auto h = build_groups_mean("f =~ x1 + c*x2 + c*x3 + x4", 2);
  std::mt19937 rng(20260613u);
  const auto raw = gaussian_cfa_raw_groups(rng, {520, 430}, 9);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1400;
  auto est = magmaan::test::fit_fiml(h.pt, h.rep, raw, opts);
  REQUIRE_MESSAGE(est.has_value(),
      "multi-group FIML fit failed: " <<
      (est.has_value() ? "" : est.error().detail));

  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());

  inf::ModificationIndexOptions mi_opts;
  mi_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto mi_raw = inf::frontier::modification_indices_fiml_robust(
      h.pt, h.rep, raw, *est, mi_opts);
  auto mi_pack = inf::frontier::modification_indices_fiml_robust(
      h.pt, h.rep, raw, *est, *pack, mi_opts);
  REQUIRE_MESSAGE(mi_raw.has_value(),
      "multi-group FIML robust MI failed: " <<
      (mi_raw.has_value() ? "" : mi_raw.error().detail));
  REQUIRE(mi_pack.has_value());
  REQUIRE(mi_raw->rows.size() == mi_pack->rows.size());
  REQUIRE(mi_raw->rows.size() > 1);

  for (std::size_t i = 0; i < mi_raw->rows.size(); ++i) {
    const auto& a = mi_raw->rows[i];
    const auto& b = mi_pack->rows[i];
    CHECK(a.candidate.group == b.candidate.group);
    CHECK(std::isfinite(a.scaling_factor));
    CHECK(a.scaling_factor > 0.0);
    CHECK(std::isfinite(a.mi_scaled));
    CHECK(std::abs(a.scaling_factor - b.scaling_factor) < 1e-10);
    CHECK(std::abs(a.mi_scaled - b.mi_scaled) <
          1e-10 * (1.0 + std::abs(b.mi_scaled)));
  }

  auto st_raw = inf::frontier::score_tests_fiml_robust(h.pt, h.rep, raw, *est);
  auto st_pack =
      inf::frontier::score_tests_fiml_robust(h.pt, h.rep, raw, *est, *pack);
  REQUIRE_MESSAGE(st_raw.has_value(),
      "multi-group FIML robust score test failed: " <<
      (st_raw.has_value() ? "" : st_raw.error().detail));
  REQUIRE(st_pack.has_value());
  REQUIRE(st_raw->rows.size() == st_pack->rows.size());
  REQUIRE(st_raw->rows.size() > 0);
  for (std::size_t i = 0; i < st_raw->rows.size(); ++i) {
    const auto& a = st_raw->rows[i];
    const auto& b = st_pack->rows[i];
    CHECK(a.df == 1);
    CHECK(std::isfinite(a.scaling_factor));
    CHECK(a.scaling_factor > 0.0);
    CHECK(std::isfinite(a.mi_scaled));
    CHECK(std::abs(a.scaling_factor - b.scaling_factor) < 1e-10);
    CHECK(std::abs(a.mi_scaled - b.mi_scaled) <
          1e-10 * (1.0 + std::abs(b.mi_scaled)));
  }
}

// ── Multi-group (PR2) ────────────────────────────────────────────────────────
// The sandwich is assembled per block with the n_b/N weighting, so the
// reduction-to-NT anchor must still hold across groups: Expected bread +
// model-implied Γ_NT meat gives B1_b = A1_b in every block ⇒ c = 1 exactly.

TEST_CASE("frontier robust MI multi-group: model-implied Γ_NT meat reduces to NT") {
  auto h = build_groups("f =~ x1 + x2 + x3 + x4", 2);
  SampleStats samp;
  Eigen::Matrix4d S1 = four_indicator_sample_cov();
  Eigen::Matrix4d S2 = four_indicator_sample_cov();
  S2(2, 3) += 0.12;
  S2(3, 2) = S2(2, 3);  // perturb group 2 so the two fits differ
  samp.S = {S1, S2};
  samp.n_obs = {350, 450};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions nt_opts;
  nt_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  nt_opts.information = inf::ScoreInformation::Expected;
  auto nt = inf::modification_indices(h.pt, h.rep, samp, *est, nt_opts);
  REQUIRE(nt.has_value());

  auto rob = inf::frontier::modification_indices_robust(
      h.pt, h.rep, samp, *est,
      robust_opts(rob::Information::Expected,
                  inf::ScoreCandidateSet::WithAbsentRows));
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() == nt->rows.size());
  REQUIRE(rob->rows.size() > 0);
  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    CHECK(std::abs(rob->rows[i].mi - nt->rows[i].mi) <
          1e-9 * (1.0 + std::abs(nt->rows[i].mi)));
    CHECK(std::abs(rob->rows[i].scaling_factor - 1.0) < 1e-6);
  }
}

TEST_CASE("frontier robust MI multi-group: empirical raw-data path scales, finite") {
  auto h = build_groups("f =~ x1 + x2 + x3 + x4", 2);
  std::mt19937 rng(7u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(rng, 500, Sigma, 6.0),
           multivariate_t_sample(rng, 600, Sigma, 8.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  REQUIRE(samp->S.size() == 2);
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());

  auto rob = inf::frontier::modification_indices_robust(
      h.pt, h.rep, *samp, raw, *est,
      robust_opts(rob::Information::Expected,
                  inf::ScoreCandidateSet::WithAbsentRows));
  REQUIRE(rob.has_value());
  REQUIRE(rob->rows.size() > 0);
  bool any_nontrivial = false;
  for (const auto& r : rob->rows) {
    CHECK(std::isfinite(r.scaling_factor));
    CHECK(r.scaling_factor > 0.0);
    if (std::abs(r.scaling_factor - 1.0) > 0.05) any_nontrivial = true;
  }
  CHECK(any_nontrivial);  // heavy-tailed data ⇒ at least one genuine correction
}

TEST_CASE("frontier robust LS MI multi-group: GLS + Γ_NT(S) meat reduces to NT") {
  auto h = build_groups("f =~ x1 + x2 + x3 + x4", 2);
  SampleStats samp;
  Eigen::Matrix4d S1 = four_indicator_sample_cov();
  Eigen::Matrix4d S2 = four_indicator_sample_cov();
  S2(2, 3) += 0.12;
  S2(3, 2) = S2(2, 3);
  samp.S = {S1, S2};
  samp.n_obs = {350, 450};
  auto est = magmaan::test::fit_gls(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto weight =
      magmaan::estimate::gmm::normal_theory_weight(*ev, samp, est->theta);
  REQUIRE(weight.has_value());

  inf::ModificationIndexOptions nt_opts;
  nt_opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto nt = inf::modification_indices(h.pt, h.rep, samp, *est, *weight, nt_opts);
  REQUIRE(nt.has_value());

  inf::frontier::RobustScoreOptions opts;
  opts.spec.moments = rob::WeightMoments::Unstructured;  // Γ_NT(S) = W⁻¹ per block
  opts.base.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto rob_mi = inf::frontier::modification_indices_robust(h.pt, h.rep, samp,
                                                           *est, *weight, opts);
  REQUIRE(rob_mi.has_value());
  REQUIRE(rob_mi->rows.size() == nt->rows.size());
  REQUIRE(rob_mi->rows.size() > 1);
  for (std::size_t i = 0; i < rob_mi->rows.size(); ++i) {
    CHECK(std::abs(rob_mi->rows[i].scaling_factor - 1.0) < 1e-6);
    CHECK(std::abs(rob_mi->rows[i].mi - nt->rows[i].mi) <
          1e-8 * (1.0 + std::abs(nt->rows[i].mi)));
  }
}

// ── df>1 total release (PR3) ─────────────────────────────────────────────────
// The joint worker generalizes the scalar c to a df-dim subspace; at df=1 it
// must reproduce the per-row release bit-for-bit (G is a single column g, so
// c̄ = λ = gᵀB1g/gᵀA1g and T = (gᵀs)²/(gᵀIg)).

TEST_CASE("frontier robust joint: direct sandwich uses the full score meat") {
  const Eigen::Vector2d score(1.0, 2.0);
  const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
  const Eigen::Matrix2d info = 10.0 * identity;
  const Eigen::Matrix2d bread = 2.0 * identity;
  Eigen::Matrix2d meat = Eigen::Matrix2d::Zero();
  meat.diagonal() << 4.0, 16.0;
  const Eigen::MatrixXd nuisance(2, 0);

  auto out = inf::frontier::score_for_subspace_robust(
      {}, score, info, bread, meat, nuisance, identity);
  REQUIRE(out.has_value());
  REQUIRE(out->sandwich_available);
  CHECK(out->mi == doctest::Approx(0.5));
  CHECK(out->mi_sandwich == doctest::Approx(0.1));
  CHECK(out->p_sandwich == doctest::Approx(std::exp(-0.05)));
  CHECK(out->sandwich_min_eigenvalue == doctest::Approx(20.0));
  CHECK(out->sandwich_condition == doctest::Approx(4.0));
  CHECK(out->eigvals(0) == doctest::Approx(2.0));
  CHECK(out->eigvals(1) == doctest::Approx(8.0));

  auto nt = inf::frontier::score_for_subspace_robust(
      {}, score, info, info, info, nuisance, identity);
  REQUIRE(nt.has_value());
  REQUIRE(nt->sandwich_available);
  CHECK(nt->mi_sandwich == doctest::Approx(nt->mi));
  CHECK(nt->p_sandwich == doctest::Approx(nt->p_value));
}

TEST_CASE("frontier robust joint: df=1 reduces to the per-row release") {
  auto h = build("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  std::mt19937 rng(13u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(rng, 600, Sigma, 6.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());

  inf::frontier::RobustScoreOptions opts;
  opts.spec = {rob::Information::Expected, rob::WeightMoments::Structured,
               rob::ScoreCovariance::Empirical};
  auto per_row =
      inf::frontier::score_tests_robust(h.pt, h.rep, *samp, raw, *est, opts);
  REQUIRE(per_row.has_value());
  REQUIRE(per_row->rows.size() == 1);
  auto joint = inf::frontier::score_tests_robust_joint(h.pt, h.rep, *samp, raw,
                                                       *est, opts);
  REQUIRE(joint.has_value());

  const auto& row = per_row->rows[0];
  CHECK(joint->df == 1);
  CHECK(joint->eigvals.size() == 1);
  CHECK(std::abs(joint->mi - row.mi) < 1e-9 * (1.0 + std::abs(row.mi)));
  CHECK(std::abs(joint->scaling_factor - row.scaling_factor) <
        1e-9 * (1.0 + std::abs(row.scaling_factor)));
  CHECK(std::abs(joint->mi_scaled - row.mi_scaled) <
        1e-9 * (1.0 + std::abs(row.mi_scaled)));
  CHECK(std::abs(joint->eigvals(0) - joint->scaling_factor) < 1e-9);
  REQUIRE(joint->sandwich_available);
  CHECK(std::abs(joint->mi_sandwich - row.mi_scaled) <
        1e-9 * (1.0 + std::abs(row.mi_scaled)));
  CHECK(joint->p_sandwich == doctest::Approx(row.p_value));
  CHECK(joint->p_mixture > 0.0);
  CHECK(joint->p_mixture <= 1.0);
}

TEST_CASE("frontier robust joint: errors cleanly with no equality constraint") {
  auto h = build("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(21u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(rng, 400, Sigma, 7.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());
  auto joint =
      inf::frontier::score_tests_robust_joint(h.pt, h.rep, *samp, raw, *est);
  CHECK_FALSE(joint.has_value());  // no active equality constraints to release
}

TEST_CASE("frontier score flips: affine ML pair is deterministic and standardized") {
  auto h1 = build("f =~ x1 + a*x2 + b*x3 + x4");
  auto h0 = build("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  std::mt19937 rng(20260712u);
  const Eigen::Matrix4d Sigma = four_indicator_sample_cov();
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(rng, 320, Sigma, 7.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est0 = magmaan::test::fit(h0.pt, h0.rep, *samp);
  REQUIRE(est0.has_value());

  auto blocks = inf::information_expected_per_case_blocks(
      h1.pt, h1.rep, *samp, *est0);
  REQUIRE(blocks.has_value());
  REQUIRE(blocks->size() == 1);
  auto info = inf::information_expected(h1.pt, h1.rep, *samp, *est0);
  REQUIRE(info.has_value());
  CHECK((320.0 * blocks->front() - *info).norm() < 1e-10 * (1.0 + info->norm()));

  inf::frontier::ScoreFlipOptions opts;
  opts.n_flips = 127;
  opts.seed = 77;
  auto a = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, opts);
  if (!a.has_value()) MESSAGE(a.error().detail);
  REQUIRE(a.has_value());
  auto b = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, opts);
  if (!b.has_value()) MESSAGE(b.error().detail);
  REQUIRE(b.has_value());
  CHECK(a->df == 1);
  CHECK(a->n_flips == 127);
  CHECK(a->seed == 77);
  CHECK(a->p_basic == b->p_basic);
  CHECK(a->p_effective == b->p_effective);
  CHECK(a->p_standardized == b->p_standardized);
  CHECK(a->p_value == a->p_standardized);
  CHECK(a->p_basic >= 1.0 / 128.0);
  CHECK(a->p_effective >= 1.0 / 128.0);
  CHECK(a->p_standardized >= 1.0 / 128.0);
  CHECK(std::abs(a->statistic_effective - a->statistic_standardized) <
        1e-9 * (1.0 + std::abs(a->statistic_effective)));
  CHECK(a->p_effective == a->p_standardized);
  CHECK(std::isfinite(a->statistic_basic));
  CHECK(std::isfinite(a->nuisance_stationarity_norm));
  CHECK(std::isfinite(a->p_mixture));
  CHECK(a->sandwich_available);
  CHECK(std::isfinite(a->statistic_sandwich));
  CHECK(std::isfinite(a->p_sandwich));
  CHECK(a->min_variance_eigenvalue > 0.0);
  CHECK(a->max_variance_condition >= 1.0);
  CHECK(a->mean_variance_relative_shift >= 0.0);
  // The property: for the affine ML pair the flip variance does not depend on
  // the flips, so V_flip and V_identity agree. They are computed by different
  // arithmetic sequences, so they agree to rounding, not bit-exactly — asserting
  // `== 0.0` pins the arithmetic path rather than the invariant. Sweeping the
  // data seed over {1..7, 20260712} against the *pre-existing* implementation
  // gives a relative shift of 0 for four seeds and 1.8e-16 – 2.7e-16 for the
  // other four; this test's seed merely landed on a bit-exact one. The shift is
  // completely insensitive to the flip seed, confirming it is a fixed
  // arithmetic difference and not resampling noise.
  constexpr double kShiftTol = 1e-12;
  CHECK(a->mean_variance_relative_shift < kShiftTol);
  CHECK(a->max_variance_relative_shift < kShiftTol);
  // Same reason `max >= mean` is not exact here: with every resample producing
  // the identical shift, `sum / n` can round to just above that common value.
  CHECK(a->max_variance_relative_shift >=
        a->mean_variance_relative_shift - kShiftTol);
  CHECK(a->setup_seconds >= 0.0);
  CHECK(a->resampling_score_seconds >= 0.0);
  CHECK(a->resampling_standardization_seconds >= 0.0);
  CHECK(a->asymptotic_seconds >= 0.0);
  CHECK(a->total_seconds >= a->setup_seconds);
  CHECK(a->total_seconds >= a->setup_seconds + a->resampling_score_seconds +
                                a->resampling_standardization_seconds +
                                a->asymptotic_seconds);

  auto effective_opts = opts;
  effective_opts.calibration =
      inf::frontier::ScoreFlipCalibration::Effective;
  auto effective = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, effective_opts);
  if (!effective.has_value()) MESSAGE(effective.error().detail);
  REQUIRE(effective.has_value());
  CHECK(effective->statistic_effective ==
        doctest::Approx(a->statistic_effective));
  CHECK(effective->p_effective == a->p_effective);
  CHECK(effective->p_value == effective->p_effective);
  CHECK((effective->eigvals - a->eigvals).norm() < 1e-12);
  CHECK(std::isnan(effective->statistic_basic));
  CHECK(std::isnan(effective->statistic_standardized));
  CHECK(std::isnan(effective->p_basic));
  CHECK(std::isnan(effective->p_standardized));
  CHECK(effective->resampling_standardization_seconds == 0.0);
  CHECK(std::isnan(effective->mean_variance_relative_shift));

  for (const auto multiplier : {
           inf::frontier::ScoreFlipMultiplier::Mammen,
           inf::frontier::ScoreFlipMultiplier::Gaussian,
           inf::frontier::ScoreFlipMultiplier::CenteredExponential}) {
    auto multiplier_opts = effective_opts;
    multiplier_opts.multiplier = multiplier;
    auto m1 = inf::frontier::score_flip_test(
        h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, multiplier_opts);
    if (!m1.has_value()) MESSAGE(m1.error().detail);
    REQUIRE(m1.has_value());
    auto m2 = inf::frontier::score_flip_test(
        h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, multiplier_opts);
    REQUIRE(m2.has_value());
    CHECK(m1->p_effective == m2->p_effective);
    CHECK(m1->p_effective >= 1.0 / 128.0);
    CHECK(m1->p_effective <= 1.0);
    CHECK(m1->statistic_effective ==
          doctest::Approx(effective->statistic_effective));
    CHECK((m1->eigvals - effective->eigvals).norm() < 1e-12);
  }

  auto gaussian_opts = effective_opts;
  gaussian_opts.multiplier =
      inf::frontier::ScoreFlipMultiplier::Gaussian;
  gaussian_opts.n_flips = 8191;
  auto gaussian = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, gaussian_opts);
  REQUIRE(gaussian.has_value());
  CHECK(std::abs(gaussian->p_effective - gaussian->p_mixture) < 0.04);

  auto mammen_opts = effective_opts;
  mammen_opts.multiplier =
      inf::frontier::ScoreFlipMultiplier::Mammen;
  auto mammen = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, mammen_opts);
  REQUIRE(mammen.has_value());

  auto two_point_opts = mammen_opts;
  two_point_opts.multiplier =
      inf::frontier::ScoreFlipMultiplier::TwoPoint;
  two_point_opts.two_point_skewness = 1.0;
  auto two_point_mammen = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, two_point_opts);
  REQUIRE(two_point_mammen.has_value());
  CHECK(two_point_mammen->p_effective == mammen->p_effective);

  two_point_opts.two_point_skewness = 0.75;
  auto two_point = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, two_point_opts);
  REQUIRE(two_point.has_value());
  CHECK(two_point->p_effective >= 1.0 / 128.0);
  CHECK(two_point->p_effective <= 1.0);

  auto centered_opts = mammen_opts;
  centered_opts.center_multiplier_scores = true;
  auto centered = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, centered_opts);
  REQUIRE(centered.has_value());
  CHECK(centered->p_effective >= 1.0 / 128.0);
  CHECK(centered->p_effective <= 1.0);

  auto studentized_opts = mammen_opts;
  studentized_opts.multiplier_studentization =
      inf::frontier::ScoreFlipMultiplierStudentization::WeightedMeat;
  auto studentized = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, studentized_opts);
  REQUIRE(studentized.has_value());
  CHECK(studentized->statistic_multiplier_studentized ==
        doctest::Approx(studentized->statistic_sandwich));
  CHECK(studentized->p_multiplier_studentized >= 1.0 / 128.0);
  CHECK(studentized->p_multiplier_studentized <= 1.0);
  CHECK(studentized->p_value == studentized->p_multiplier_studentized);
  CHECK(studentized->mc_se_multiplier_studentized >= 0.0);
  CHECK(studentized->multiplier_studentized_min_eigenvalue > 0.0);
  CHECK(studentized->multiplier_studentized_max_condition >= 1.0);

  auto invalid_skewness_opts = two_point_opts;
  invalid_skewness_opts.two_point_skewness = -0.1;
  auto invalid_skewness = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0,
      invalid_skewness_opts);
  CHECK_FALSE(invalid_skewness.has_value());

  auto unsupported_opts = opts;
  unsupported_opts.multiplier =
      inf::frontier::ScoreFlipMultiplier::Mammen;
  auto unsupported = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, unsupported_opts);
  CHECK_FALSE(unsupported.has_value());

  auto unsupported_centering_opts = opts;
  unsupported_centering_opts.center_multiplier_scores = true;
  auto unsupported_centering = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0,
      unsupported_centering_opts);
  CHECK_FALSE(unsupported_centering.has_value());

  auto asymptotic_opts = opts;
  asymptotic_opts.n_flips = 0;
  asymptotic_opts.calibration =
      inf::frontier::ScoreFlipCalibration::AsymptoticOnly;
  auto asymptotic = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, asymptotic_opts);
  if (!asymptotic.has_value()) MESSAGE(asymptotic.error().detail);
  REQUIRE(asymptotic.has_value());
  CHECK(asymptotic->statistic_effective ==
        doctest::Approx(a->statistic_effective));
  CHECK((asymptotic->eigvals - a->eigvals).norm() < 1e-12);
  CHECK(asymptotic->n_flips == 0);
  CHECK(std::isnan(asymptotic->p_effective));
  CHECK(asymptotic->p_value == asymptotic->p_mixture);
  CHECK(asymptotic->resampling_score_seconds == 0.0);
  CHECK(asymptotic->resampling_standardization_seconds == 0.0);

  auto fixed_x_h1 = h1.pt;
  REQUIRE_FALSE(fixed_x_h1.exo.empty());
  fixed_x_h1.exo.front() = 1;
  auto fixed_x = inf::frontier::score_flip_test(
      fixed_x_h1, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, opts);
  CHECK_FALSE(fixed_x.has_value());

  auto wrong_raw = raw;
  wrong_raw.X.front()(0, 0) += 0.5;
  auto mismatch = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, wrong_raw, *est0, opts);
  CHECK_FALSE(mismatch.has_value());
}

TEST_CASE("frontier score flips: rejects invalid flip count and identical pair") {
  auto h = build("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(31u);
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(rng, 120, four_indicator_sample_cov(), 8.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());
  inf::frontier::ScoreFlipOptions opts;
  opts.n_flips = 0;
  auto bad_n = inf::frontier::score_flip_test(
      h.pt, h.rep, h.pt, h.rep, *samp, raw, *est, opts);
  CHECK_FALSE(bad_n.has_value());
  opts.n_flips = 9;
  auto same = inf::frontier::score_flip_test(
      h.pt, h.rep, h.pt, h.rep, *samp, raw, *est, opts);
  CHECK_FALSE(same.has_value());
}

TEST_CASE("frontier score flips: exact enumeration is seed-free on tiny n") {
  auto h1 = build("f =~ x1 + a*x2 + b*x3 + x4");
  auto h0 = build("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  std::mt19937 rng(20260713u);
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(rng, 12, four_indicator_sample_cov(), 9.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est0 = magmaan::test::fit(h0.pt, h0.rep, *samp);
  REQUIRE(est0.has_value());

  inf::frontier::ScoreFlipOptions opts;
  opts.exact_enumeration = true;
  opts.seed = 11;
  auto a = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, opts);
  if (!a.has_value()) MESSAGE(a.error().detail);
  REQUIRE(a.has_value());
  opts.seed = 9999;
  auto b = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw, *est0, opts);
  REQUIRE(b.has_value());

  CHECK(a->n_flips == 4095);
  CHECK(a->p_basic == b->p_basic);
  CHECK(a->p_effective == b->p_effective);
  CHECK(a->p_standardized == b->p_standardized);
  CHECK(a->mc_se_basic == 0.0);
  CHECK(a->mc_se_effective == 0.0);
  CHECK(a->mc_se_standardized == 0.0);
  CHECK(std::abs(a->p_basic * 4096.0 - std::round(a->p_basic * 4096.0)) <
        1e-12);
  CHECK(std::abs(a->p_effective * 4096.0 -
                 std::round(a->p_effective * 4096.0)) < 1e-12);
  CHECK(std::abs(a->p_standardized * 4096.0 -
                 std::round(a->p_standardized * 4096.0)) < 1e-12);

  auto too_large_raw = raw;
  too_large_raw.X = {
      multivariate_t_sample(rng, 21, four_indicator_sample_cov(), 9.0)};
  auto too_large_samp = magmaan::data::sample_stats_from_raw(too_large_raw);
  REQUIRE(too_large_samp.has_value());
  auto too_large_est = magmaan::test::fit(h0.pt, h0.rep, *too_large_samp);
  REQUIRE(too_large_est.has_value());
  auto too_large = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *too_large_samp, too_large_raw,
      *too_large_est, opts);
  CHECK_FALSE(too_large.has_value());
}

TEST_CASE("frontier FIML score flips: all-observed patterns equal complete ML") {
  auto h1 = build_mean("f =~ x1 + a*x2 + b*x3 + x4");
  auto h0 = build_mean("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  std::mt19937 rng(20260714u);
  auto raw_fiml = gaussian_cfa_raw(rng, 180, 0);
  auto raw_ml = raw_fiml;
  raw_ml.mask.clear();
  auto samp = magmaan::data::sample_stats_from_raw(raw_ml);
  REQUIRE(samp.has_value());

  magmaan::optim::OptimOptions fit_opts;
  fit_opts.max_iter = 1000;
  auto est0 = magmaan::test::fit_fiml(h0.pt, h0.rep, raw_fiml, fit_opts);
  REQUIRE(est0.has_value());
  auto pack = magmaan::estimate::fiml::fiml_pack(raw_fiml);
  REQUIRE(pack.has_value());

  inf::frontier::ScoreFlipOptions opts;
  opts.n_flips = 127;
  opts.seed = 414;
  auto complete = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw_ml, *est0, opts);
  if (!complete.has_value()) MESSAGE(complete.error().detail);
  REQUIRE(complete.has_value());
  auto fiml = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw_fiml, *pack, *est0, opts);
  if (!fiml.has_value()) MESSAGE(fiml.error().detail);
  REQUIRE(fiml.has_value());

  CHECK(fiml->df == complete->df);
  CHECK(std::abs(fiml->statistic_basic - complete->statistic_basic) < 1e-9);
  CHECK(std::abs(fiml->statistic_effective - complete->statistic_effective) <
        1e-9);
  CHECK(std::abs(fiml->statistic_standardized -
                 complete->statistic_standardized) < 1e-9);
  CHECK(fiml->p_basic == complete->p_basic);
  CHECK(fiml->p_effective == complete->p_effective);
  CHECK(fiml->p_standardized == complete->p_standardized);
  CHECK((fiml->eigvals - complete->eigvals).norm() < 1e-9);
  CHECK(fiml->mean_variance_relative_shift ==
        doctest::Approx(complete->mean_variance_relative_shift).epsilon(1e-9));

  opts.calibration = inf::frontier::ScoreFlipCalibration::Effective;
  opts.multiplier = inf::frontier::ScoreFlipMultiplier::Mammen;
  opts.center_multiplier_scores = true;
  auto complete_centered = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, *samp, raw_ml, *est0, opts);
  REQUIRE(complete_centered.has_value());
  auto fiml_centered = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw_fiml, *pack, *est0, opts);
  REQUIRE(fiml_centered.has_value());
  CHECK(fiml_centered->p_effective == complete_centered->p_effective);
}

TEST_CASE("frontier global score flip uses the curved SEM complement") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260820u);
  auto raw_masked = gaussian_cfa_raw(rng, 180, 0);
  auto raw_complete = raw_masked;
  raw_complete.mask.clear();
  magmaan::optim::OptimOptions fit_opts;
  fit_opts.max_iter = 1000;
  auto est = magmaan::test::fit_fiml(h.pt, h.rep, raw_masked, fit_opts);
  REQUIRE(est.has_value());
  auto pack_masked = magmaan::estimate::fiml::fiml_pack(raw_masked);
  auto pack_complete = magmaan::estimate::fiml::fiml_pack(raw_complete);
  REQUIRE(pack_masked.has_value());
  REQUIRE(pack_complete.has_value());

  inf::frontier::GlobalScoreFlipOptions opts;
  opts.resampling.n_flips = 127;
  opts.resampling.seed = 812;
  auto masked = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw_masked, *pack_masked, *est, opts);
  if (!masked.has_value()) MESSAGE(masked.error().detail);
  REQUIRE(masked.has_value());
  auto complete = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw_complete, *pack_complete, *est, opts);
  if (!complete.has_value()) MESSAGE(complete.error().detail);
  REQUIRE(complete.has_value());

  CHECK(masked->saturated_moment_dim == 14);
  CHECK(masked->tangent_rank == 12);
  CHECK(masked->flip.df == 2);
  CHECK(masked->flip.n_flips == 127);
  CHECK(masked->flip.p_value == masked->flip.p_effective);
  CHECK(masked->metric ==
        inf::frontier::GlobalScoreFlipOptions::Metric::ExpectedInformation);
  CHECK(masked->flip.statistic_effective ==
        doctest::Approx(complete->flip.statistic_effective).epsilon(1e-9));
  CHECK(masked->flip.p_effective == complete->flip.p_effective);
  CHECK((masked->flip.eigvals - complete->flip.eigvals).norm() < 1e-9);
  CHECK(std::isnan(masked->flip.statistic_basic));
  CHECK(std::isnan(masked->flip.statistic_standardized));
  CHECK(masked->tangent_min_singular_value > 0.0);
  CHECK(masked->tangent_condition >= 1.0);
  CHECK(std::isfinite(masked->flip.nuisance_stationarity_norm));
}

TEST_CASE("frontier global score flip supports covariance-only complete ML") {
  auto h = build("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260822u);
  magmaan::data::RawData raw;
  raw.X = {multivariate_t_sample(
      rng, 180, four_indicator_sample_cov(), 8.0)};
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(h.pt, h.rep, *samp);
  REQUIRE(est.has_value());

  inf::frontier::GlobalScoreFlipOptions opts;
  opts.resampling.n_flips = 63;
  opts.resampling.seed = 611;
  auto result = inf::frontier::global_score_flip_test(
      h.pt, h.rep, *samp, raw, *est, opts);
  if (!result.has_value()) MESSAGE(result.error().detail);
  REQUIRE(result.has_value());
  CHECK(result->saturated_moment_dim == 10);
  CHECK(result->tangent_rank == 8);
  CHECK(result->flip.df == 2);
  CHECK(result->flip.n_flips == 63);
  CHECK(std::isfinite(result->flip.p_effective));

  opts.resampling.sensitivity =
      inf::frontier::ScoreFlipSensitivity::ObservedInformation;
  auto observed = inf::frontier::global_score_flip_test(
      h.pt, h.rep, *samp, raw, *est, opts);
  if (!observed.has_value()) MESSAGE(observed.error().detail);
  REQUIRE(observed.has_value());
  CHECK(observed->flip.sensitivity ==
        inf::frontier::ScoreFlipSensitivity::ObservedInformation);
  CHECK(std::isfinite(observed->flip.statistic_effective));

  auto wrong_raw = raw;
  wrong_raw.X.front()(0, 0) += 0.5;
  auto mismatch = inf::frontier::global_score_flip_test(
      h.pt, h.rep, *samp, wrong_raw, *est, opts);
  CHECK_FALSE(mismatch.has_value());
}

TEST_CASE("frontier global FIML score flip is reproducible with missing patterns") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260821u);
  const auto raw = gaussian_cfa_raw(rng, 220, 3);
  magmaan::optim::OptimOptions fit_opts;
  fit_opts.max_iter = 1200;
  auto est = magmaan::test::fit_fiml(h.pt, h.rep, raw, fit_opts);
  REQUIRE(est.has_value());
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  REQUIRE(pack->cache.patterns.size() > 1);

  auto evaluator = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(evaluator.has_value());
  auto evaluation = evaluator->evaluate(est->theta, true, true);
  REQUIRE(evaluation.has_value());
  auto saturated_scores =
      magmaan::estimate::fiml::fiml_saturated_casewise_deviance_scores(
          raw, *pack, evaluation->moments, true);
  auto saturated_observed =
      magmaan::estimate::fiml::fiml_saturated_observed_information(
          raw, *pack, evaluation->moments, true);
  auto model_scores =
      magmaan::estimate::fiml::fiml_casewise_deviance_scores(
          h.pt, h.rep, raw, *pack, *est);
  REQUIRE(saturated_scores.has_value());
  REQUIRE(saturated_observed.has_value());
  REQUIRE(model_scores.has_value());
  Eigen::MatrixXd Delta(saturated_scores->cols(), est->theta.size());
  Delta.topRows(evaluation->J_sigma.rows()) = evaluation->J_sigma;
  Delta.bottomRows(evaluation->J_mu.rows()) = evaluation->J_mu;
  CHECK((*saturated_scores * Delta - *model_scores).norm() <
        1e-9 * (1.0 + model_scores->norm()));
  auto observed_h1 =
      magmaan::estimate::fiml::fiml_observed_h1_information(
          h.pt, h.rep, raw, *est, *pack);
  REQUIRE(observed_h1.has_value());
  const Eigen::MatrixXd projected_observed =
      Delta.transpose() * (*saturated_observed) * Delta;
  CHECK((projected_observed - *observed_h1).norm() <
        1e-9 * (1.0 + observed_h1->norm()));

  inf::frontier::GlobalScoreFlipOptions opts;
  opts.resampling.n_flips = 127;
  opts.resampling.seed = 917;
  opts.resampling.multiplier =
      inf::frontier::ScoreFlipMultiplier::Mammen;
  opts.resampling.center_multiplier_scores = true;
  auto a = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  if (!a.has_value()) MESSAGE(a.error().detail);
  REQUIRE(a.has_value());
  auto b = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  REQUIRE(b.has_value());

  CHECK(a->flip.df == 2);
  CHECK(a->flip.p_effective == b->flip.p_effective);
  CHECK(a->flip.statistic_effective ==
        doctest::Approx(b->flip.statistic_effective));
  CHECK(a->flip.p_effective >= 1.0 / 128.0);
  CHECK(a->flip.p_effective <= 1.0);
  CHECK(std::isfinite(a->flip.p_mixture));
  CHECK(a->flip.sandwich_available);
  CHECK(std::isfinite(a->flip.p_sandwich));

  opts.resampling.center_multiplier_scores = false;
  opts.resampling.sensitivity =
      inf::frontier::ScoreFlipSensitivity::ObservedInformation;
  auto corrected = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  if (!corrected.has_value()) MESSAGE(corrected.error().detail);
  REQUIRE(corrected.has_value());
  CHECK(corrected->flip.sensitivity ==
        inf::frontier::ScoreFlipSensitivity::ObservedInformation);
  CHECK(std::isfinite(corrected->flip.statistic_effective));
  CHECK(std::isfinite(corrected->flip.p_effective));
  CHECK(corrected->flip.statistic_effective !=
        doctest::Approx(a->flip.statistic_effective).epsilon(1e-12));

  opts.resampling.sensitivity =
      inf::frontier::ScoreFlipSensitivity::ObservedInformationLightShrinkage;
  auto shrunken_light = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  if (!shrunken_light.has_value()) MESSAGE(shrunken_light.error().detail);
  REQUIRE(shrunken_light.has_value());
  const double tangent_ratio =
      static_cast<double>(shrunken_light->tangent_rank) /
      static_cast<double>(raw.X.front().rows());
  CHECK(shrunken_light->sensitivity_shrinkage ==
        doctest::Approx(tangent_ratio / (1.0 + tangent_ratio)));
  CHECK(std::isfinite(shrunken_light->flip.p_sandwich));

  opts.resampling.sensitivity =
      inf::frontier::ScoreFlipSensitivity::ObservedInformationSqrtShrinkage;
  auto shrunken_sqrt = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  if (!shrunken_sqrt.has_value()) MESSAGE(shrunken_sqrt.error().detail);
  REQUIRE(shrunken_sqrt.has_value());
  CHECK(shrunken_sqrt->sensitivity_shrinkage ==
        doctest::Approx(std::sqrt(tangent_ratio) /
                        (1.0 + std::sqrt(tangent_ratio))));
  CHECK(std::isfinite(shrunken_sqrt->flip.p_sandwich));

  opts.resampling.sensitivity =
      inf::frontier::ScoreFlipSensitivity::ObservedInformation;
  opts.metric =
      inf::frontier::GlobalScoreFlipOptions::Metric::ObservedInformation;
  auto observed_metric = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  if (!observed_metric.has_value()) MESSAGE(observed_metric.error().detail);
  REQUIRE(observed_metric.has_value());
  CHECK(observed_metric->metric ==
        inf::frontier::GlobalScoreFlipOptions::Metric::ObservedInformation);
  CHECK(std::isfinite(observed_metric->flip.statistic_effective));
  CHECK(std::isfinite(observed_metric->flip.p_mixture));
  CHECK(observed_metric->flip.statistic_effective !=
        doctest::Approx(corrected->flip.statistic_effective).epsilon(1e-12));
  CHECK((observed_metric->flip.eigvals - corrected->flip.eigvals).norm() >
        1e-12);

  opts.resampling.sensitivity =
      inf::frontier::ScoreFlipSensitivity::SaturatedObservedInformation;
  opts.metric = inf::frontier::GlobalScoreFlipOptions::Metric::
      SaturatedObservedInformation;
  auto saturated_observed_score = inf::frontier::global_score_flip_test(
      h.pt, h.rep, raw, *pack, *est, opts);
  if (!saturated_observed_score.has_value()) {
    MESSAGE(saturated_observed_score.error().detail);
  }
  REQUIRE(saturated_observed_score.has_value());
  CHECK(saturated_observed_score->flip.sensitivity ==
        inf::frontier::ScoreFlipSensitivity::SaturatedObservedInformation);
  CHECK(saturated_observed_score->metric ==
        inf::frontier::GlobalScoreFlipOptions::Metric::
            SaturatedObservedInformation);
  CHECK(std::isfinite(saturated_observed_score->flip.statistic_effective));
  CHECK(std::isfinite(saturated_observed_score->flip.p_mixture));
  CHECK(saturated_observed_score->flip.min_variance_eigenvalue > 0.0);
  CHECK(saturated_observed_score->n_obs == raw.X.front().rows());
  CHECK(saturated_observed_score->projected_score.size() ==
        saturated_observed_score->flip.df);
  CHECK(saturated_observed_score->projected_metric.rows() ==
        saturated_observed_score->flip.df);
  CHECK(saturated_observed_score->projected_meat.rows() ==
        saturated_observed_score->flip.df);
  const Eigen::VectorXd metric_solution =
      saturated_observed_score->projected_metric.ldlt().solve(
          saturated_observed_score->projected_score);
  CHECK(saturated_observed_score->projected_score.dot(metric_solution) ==
        doctest::Approx(
            saturated_observed_score->flip.statistic_effective).epsilon(1e-10));
  const Eigen::VectorXd meat_solution =
      saturated_observed_score->projected_meat.ldlt().solve(
          saturated_observed_score->projected_score);
  CHECK(saturated_observed_score->projected_score.dot(meat_solution) ==
        doctest::Approx(
            saturated_observed_score->flip.statistic_sandwich).epsilon(1e-10));
  CHECK((saturated_observed_score->flip.eigvals -
         observed_metric->flip.eigvals).norm() > 1e-12);
}

TEST_CASE("frontier global ML2S score flip uses Stage-1 EM influence") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260824u);
  const auto raw = gaussian_cfa_raw(rng, 220, 3);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto h1 = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack);
  REQUIRE(h1.has_value());
  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw, *pack, *h1);
  REQUIRE(sm.has_value());
  magmaan::data::SampleStats stage2;
  stage2.S = sm->cov;
  stage2.mean = sm->mean;
  stage2.n_obs = sm->n_obs;
  auto est = magmaan::test::fit(h.pt, h.rep, stage2);
  REQUIRE(est.has_value());

  inf::frontier::GlobalScoreFlipOptions opts;
  opts.resampling.n_flips = 127;
  opts.resampling.seed = 1109;
  opts.resampling.multiplier = inf::frontier::ScoreFlipMultiplier::Mammen;
  opts.resampling.center_multiplier_scores = true;
  auto a = inf::frontier::global_score_flip_test_ml2s(
      h.pt, h.rep, raw, *pack, *h1, *sm, *est, opts);
  if (!a.has_value()) MESSAGE(a.error().detail);
  REQUIRE(a.has_value());
  auto b = inf::frontier::global_score_flip_test_ml2s(
      h.pt, h.rep, raw, *pack, *h1, *sm, *est, opts);
  REQUIRE(b.has_value());

  CHECK(a->saturated_moment_dim == 14);
  CHECK(a->tangent_rank == 12);
  CHECK(a->flip.df == 2);
  CHECK(a->flip.n_flips == 127);
  CHECK(a->flip.p_effective == b->flip.p_effective);
  CHECK(a->flip.statistic_effective ==
        doctest::Approx(b->flip.statistic_effective));
  CHECK(a->flip.p_effective >= 1.0 / 128.0);
  CHECK(a->flip.p_effective <= 1.0);
  CHECK(std::isfinite(a->flip.p_mixture));
  CHECK(a->flip.sandwich_available);
  CHECK(std::isfinite(a->flip.nuisance_stationarity_norm));
}

TEST_CASE("frontier global ML2S score flip reduces to complete-data ML") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260825u);
  auto raw = gaussian_cfa_raw(rng, 240, 0);
  raw.mask.clear();
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto h1 = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack);
  REQUIRE(h1.has_value());
  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw, *pack, *h1);
  REQUIRE(sm.has_value());
  magmaan::data::SampleStats stage2;
  stage2.S = sm->cov;
  stage2.mean = sm->mean;
  stage2.n_obs = sm->n_obs;
  auto est = magmaan::test::fit(h.pt, h.rep, stage2);
  REQUIRE(est.has_value());

  inf::frontier::GlobalScoreFlipOptions opts;
  opts.resampling.n_flips = 63;
  opts.resampling.seed = 1217;
  auto ml = inf::frontier::global_score_flip_test(
      h.pt, h.rep, stage2, raw, *est, opts);
  REQUIRE(ml.has_value());
  auto ml2s = inf::frontier::global_score_flip_test_ml2s(
      h.pt, h.rep, raw, *pack, *h1, *sm, *est, opts);
  if (!ml2s.has_value()) MESSAGE(ml2s.error().detail);
  REQUIRE(ml2s.has_value());

  CHECK(ml2s->flip.df == ml->flip.df);
  CHECK(ml2s->flip.statistic_effective ==
        doctest::Approx(ml->flip.statistic_effective).epsilon(1e-8));
  CHECK(ml2s->flip.p_effective == ml->flip.p_effective);
  // The observed gate and multiplier rank reduce exactly. The asymptotic
  // spectrum is only first-order equivalent because ML2S propagates centered
  // sample-moment influence rather than retaining raw likelihood-score rows.
  CHECK((ml2s->flip.eigvals - ml->flip.eigvals).norm() < 0.05);
}

TEST_CASE("frontier FIML score flips: missing-pattern correction is reproducible") {
  auto h1 = build_mean("f =~ x1 + a*x2 + b*x3 + x4");
  auto h0 = build_mean("f =~ x1 + a*x2 + b*x3 + x4\na == b");
  std::mt19937 rng(1717u);
  const auto raw = gaussian_cfa_raw(rng, 240, 3);
  magmaan::optim::OptimOptions fit_opts;
  fit_opts.max_iter = 1200;
  auto est0 = magmaan::test::fit_fiml(h0.pt, h0.rep, raw, fit_opts);
  REQUIRE(est0.has_value());
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  REQUIRE(pack->cache.patterns.size() > 1);

  inf::frontier::ScoreFlipOptions opts;
  opts.n_flips = 127;
  opts.seed = 991;
  auto a = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw, *pack, *est0, opts);
  if (!a.has_value()) MESSAGE(a.error().detail);
  REQUIRE(a.has_value());
  auto b = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw, *pack, *est0, opts);
  REQUIRE(b.has_value());

  CHECK(a->df == 1);
  CHECK(a->p_basic == b->p_basic);
  CHECK(a->p_effective == b->p_effective);
  CHECK(a->p_standardized == b->p_standardized);
  CHECK(std::abs(a->statistic_effective - a->statistic_standardized) < 1e-9);
  CHECK(a->mean_variance_relative_shift > 0.0);
  CHECK(a->max_variance_relative_shift >= a->mean_variance_relative_shift);
  CHECK(a->min_variance_eigenvalue > 0.0);
  CHECK(a->max_variance_condition >= 1.0);
  CHECK(std::isfinite(a->p_mean_scaled));
  CHECK(std::isfinite(a->p_mixture));
  CHECK(a->sandwich_available);
  CHECK(std::isfinite(a->p_sandwich));

  opts.calibration = inf::frontier::ScoreFlipCalibration::Effective;
  opts.multiplier = inf::frontier::ScoreFlipMultiplier::Mammen;
  opts.sensitivity =
      inf::frontier::ScoreFlipSensitivity::ObservedInformation;
  auto corrected_a = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw, *pack, *est0, opts);
  if (!corrected_a.has_value()) MESSAGE(corrected_a.error().detail);
  REQUIRE(corrected_a.has_value());
  auto corrected_b = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw, *pack, *est0, opts);
  REQUIRE(corrected_b.has_value());
  CHECK(corrected_a->sensitivity ==
        inf::frontier::ScoreFlipSensitivity::ObservedInformation);
  CHECK(corrected_a->p_effective == corrected_b->p_effective);
  CHECK(corrected_a->statistic_effective ==
        doctest::Approx(corrected_b->statistic_effective));
  CHECK(std::isfinite(corrected_a->p_mixture));

  opts.calibration = inf::frontier::ScoreFlipCalibration::All;
  auto unsupported = inf::frontier::score_flip_test(
      h1.pt, h1.rep, h0.pt, h0.rep, raw, *pack, *est0, opts);
  CHECK_FALSE(unsupported.has_value());
}

TEST_CASE("frontier score flips: grouped variance matches dense case oracle") {
  using inf::frontier::detail::ScoreFlipGroupGeometry;
  using inf::frontier::detail::score_flip_variance;
  const std::vector<std::int64_t> n_obs{2, 3};
  std::vector<Eigen::Matrix4d> J(2);
  Eigen::Matrix4d L0;
  L0 << 1.2, 0.0, 0.0, 0.0,
        0.3, 0.9, 0.0, 0.0,
       -0.2, 0.1, 1.1, 0.0,
        0.4, -0.1, 0.2, 0.8;
  Eigen::Matrix4d L1;
  L1 << 0.8, 0.0, 0.0, 0.0,
       -0.1, 1.3, 0.0, 0.0,
        0.5, 0.2, 0.7, 0.0,
       -0.2, 0.3, 0.1, 1.0;
  J[0] = L0 * L0.transpose();
  J[1] = L1 * L1.transpose();

  Eigen::Matrix<double, 4, 2> K;
  K << 1.0, 0.0,
       0.0, 1.0,
       0.2, -0.3,
       0.1, 0.4;
  Eigen::Matrix<double, 4, 2> D;
  D << 0.2, -0.1,
       0.3, 0.4,
       1.0, 0.2,
      -0.2, 1.0;
  const Eigen::Matrix4d I = 2.0 * J[0] + 3.0 * J[1];
  const Eigen::Matrix2d Ainv = (K.transpose() * I * K).inverse();
  const Eigen::Matrix<double, 4, 2> G =
      D - K * Ainv * K.transpose() * I * D;
  std::vector<ScoreFlipGroupGeometry> geometry;
  for (const auto& Jb : J) {
    geometry.push_back({G.transpose() * Jb * G,
                        G.transpose() * Jb * K,
                        K.transpose() * Jb * K});
  }

  for (std::uint64_t mask = 0; mask < 32; ++mask) {
    std::vector<double> signs(5);
    for (std::size_t i = 0; i < signs.size(); ++i) {
      signs[i] = ((mask >> i) & 1ULL) == 0ULL ? -1.0 : 1.0;
    }
    const std::vector<std::int64_t> sign_sum{
        static_cast<std::int64_t>(signs[0] + signs[1]),
        static_cast<std::int64_t>(signs[2] + signs[3] + signs[4])};
    const Eigen::Matrix2d grouped =
        score_flip_variance(geometry, Ainv, n_obs, sign_sum);

    Eigen::Matrix2d cross = Eigen::Matrix2d::Zero();
    for (std::size_t i = 0; i < signs.size(); ++i) {
      const std::size_t b = i < 2 ? 0 : 1;
      cross.noalias() += signs[i] * G.transpose() * J[b] * K;
    }
    const Eigen::Matrix2d R = cross * Ainv;
    Eigen::Matrix2d dense = Eigen::Matrix2d::Zero();
    for (std::size_t i = 0; i < signs.size(); ++i) {
      const std::size_t b = i < 2 ? 0 : 1;
      const Eigen::Matrix<double, 4, 2> adjusted =
          G - signs[i] * K * R.transpose();
      dense.noalias() += adjusted.transpose() * J[b] * adjusted;
    }
    CHECK((grouped - dense).norm() < 1e-12 * (1.0 + dense.norm()));
  }
}

TEST_CASE("score primitives: efficient rank uses relative information across units") {
  // The candidate duplicates the nuisance except for delta units of genuinely
  // new information. Scaling its parameter coordinate must not change rank.
  for (double units : {1e-8, 1.0, 1e8}) {
    for (double delta : {0.0, 1e-14, 1e-6}) {
      CAPTURE(units);
      CAPTURE(delta);
      Eigen::Matrix2d information;
      information << 1.0, units, units, units * units * (1.0 + delta);
      Eigen::Vector2d score(0.0, units * delta);
      Eigen::Matrix<double, 2, 1> nuisance;
      nuisance << 1.0, 0.0;
      const Eigen::Vector2d direction(0.0, 1.0);
      auto result = inf::frontier::score_for_direction_robust(
          {}, score, information, information, 2.0 * information,
          nuisance, direction);
      auto ordinary = inf::score_for_direction(
          {}, score, information, nuisance, direction);
      if (delta < 1e-10) {
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().kind == magmaan::PostError::Kind::InfoMatrixSingular);
        REQUIRE_FALSE(ordinary.has_value());
        CHECK(ordinary.error().kind == magmaan::PostError::Kind::InfoMatrixSingular);
      } else {
        REQUIRE(result.has_value());
        REQUIRE(ordinary.has_value());
        CHECK(result->mi == doctest::Approx(delta).epsilon(1e-8));
        CHECK(result->scaling_factor == doctest::Approx(2.0));
        CHECK(result->epc * units == doctest::Approx(1.0).epsilon(1e-8));
        CHECK(ordinary->mi == doctest::Approx(result->mi));
        CHECK(ordinary->epc == doctest::Approx(result->epc));
      }
    }
  }
}

TEST_CASE("score primitives: supplied geometry, basis invariance and PSD meat") {
  using namespace inf::frontier;
  ScoreComponents c;
  c.score = Eigen::Vector3d(2, 3, 4);
  c.metric = Eigen::Matrix3d::Identity();
  c.sensitivity = c.metric;
  c.nuisance = Eigen::MatrixXd::Zero(3, 1);
  c.nuisance(0,0) = 1;
  c.directions = Eigen::MatrixXd::Zero(3,2);
  c.directions(1,0) = 1; c.directions(2,1) = 1;
  c.rows = Eigen::MatrixXd::Zero(4,3);
  c.rows.col(1) << 1, 2, -1, 1;
  // Deliberately rank-deficient meat; the observed score is independent of rows.
  c.influence_rows = true;
  auto p = project_scores(c, true);
  REQUIRE(p.has_value());
  CHECK(p->statistic == doctest::Approx(25.0));
  auto eigen = score_spectrum(*p);
  REQUIRE(eigen.has_value());
  CHECK((*eigen)(0) == doctest::Approx(0));
  CHECK((*eigen)(1) == doctest::Approx(7));
  CHECK_FALSE(score_sandwich(*p).has_value());
  auto scale = score_mean_scale(*p);
  REQUIRE(scale.has_value());
  CHECK(*scale == doctest::Approx(3.5));
  Eigen::Matrix2d change;
  change << 2, 1, 0, 3;
  c.directions = (c.directions * change).eval();
  auto rotated = project_scores(c);
  REQUIRE(rotated.has_value());
  CHECK(rotated->statistic == doctest::Approx(p->statistic));
  auto re = score_spectrum(*rotated);
  REQUIRE(re.has_value());
  CHECK((*re - *eigen).norm() < 1e-10);
  auto centered = project_scores(c, true, true);
  REQUIRE(centered.has_value());
  CHECK(centered->statistic == doctest::Approx(p->statistic));
  CHECK(centered->rows.colwise().sum().norm() < 1e-10);
  auto draws = resample_scores(*p, 127, 42);
  REQUIRE(draws.has_value());
  CHECK(draws->p_value >= 1.0/128.0);
  CHECK_FALSE(resample_scores(*rotated, 127, 42).has_value());
  c.metric(0,0) = -1;
  c.metric(1,1) = -1;
  CHECK_FALSE(project_scores(c).has_value());
}

TEST_CASE("score primitives: global FIML preparation and retained resampling agree") {
  auto h = build_mean("f =~ x1 + x2 + x3 + x4");
  std::mt19937 rng(20260918u);
  auto raw = gaussian_cfa_raw(rng, 180, 3);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto est = magmaan::test::fit_fiml(h.pt,h.rep,raw);
  REQUIRE(est.has_value());
  inf::frontier::GlobalScoreFlipOptions options;
  options.resampling.n_flips = 31;
  options.resampling.seed = 19;
  auto c = inf::frontier::global_score_components(h.pt,h.rep,raw,*pack,*est);
  REQUIRE(c.has_value());
  CHECK((c->score - c->rows.colwise().sum().transpose()).norm() < 1e-10);
  auto p = inf::frontier::project_scores(*c,true);
  REQUIRE(p.has_value());
  auto legacy = inf::frontier::global_score_flip_test(h.pt,h.rep,raw,*pack,*est,options);
  REQUIRE(legacy.has_value());
  auto spectrum = inf::frontier::score_spectrum(*p);
  REQUIRE(spectrum.has_value());
  CHECK(p->statistic == doctest::Approx(legacy->flip.statistic_effective).epsilon(1e-9));
  CHECK((*spectrum - legacy->flip.eigvals).norm() < 1e-9);
  auto flip = inf::frontier::resample_scores(*p,31,19);
  REQUIRE(flip.has_value());
  CHECK(flip->p_value == legacy->flip.p_effective);
}

TEST_CASE("prepared NTML shares contributions and geometry across score LR and covariance") {
  using namespace rob::frontier;
  for (bool means : {false,true}) {
    auto h = means ? build_groups_mean("f =~ x1 + a*x2 + b*x3 + x4",2)
                   : build_groups("f =~ x1 + a*x2 + b*x3 + x4",2);
    std::mt19937 rng(382);
    magmaan::data::RawData raw;
    raw.X.push_back(multivariate_t_sample(rng,140,four_indicator_sample_cov(),7));
    raw.X.push_back(multivariate_t_sample(rng,210,four_indicator_sample_cov(),7));
    auto data = prepare_ntml_data(raw,means,ContributionStorage::Casewise);
    REQUIRE(data.has_value());
    auto est = magmaan::test::fit(h.pt,h.rep,(*data)->sample);
    REQUIRE(est.has_value());
    auto fit = prepare_ntml_fit(*data,h.pt,h.rep,*est);
    REQUIRE(fit.has_value());
    auto score = ntml_quadratic(**fit,true), lr = ntml_quadratic(**fit,false);
    REQUIRE(score.has_value()); REQUIRE(lr.has_value());
    auto components = inf::frontier::global_score_components(h.pt,h.rep,(*data)->sample,raw,*est);
    REQUIRE(components.has_value());
    auto projected = inf::frontier::project_scores(*components);
    REQUIRE(projected.has_value());
    CHECK((**score).statistic == doctest::Approx(projected->statistic).epsilon(1e-8));
    auto se = ntml_spectrum(**score); auto oldse = inf::frontier::score_spectrum(*projected);
    REQUIRE(se.has_value()); REQUIRE(oldse.has_value());
    CHECK((**se-*oldse).norm() < 1e-7);
    auto u = rob::build_u_factor(h.pt,h.rep,(*data)->sample,*est);
    REQUIRE(u.has_value());
    Eigen::Vector2d denom(140,210);
    auto z = rob::casewise_contributions(raw,(*data)->sample,means);
    REQUIRE(z.has_value());
    auto M = rob::reduced_gamma_sample(*u,*z,denom);
    REQUIRE(M.has_value());
    auto expected = rob::ugamma_eigenvalues(*M); auto got = ntml_spectrum(**lr);
    REQUIRE(expected.has_value()); REQUIRE(got.has_value());
    CHECK((**got-*expected).norm()<1e-9);
    auto info = ntml_information(**fit);
    auto oldinfo = inf::information_expected(h.pt,h.rep,(*data)->sample,*est);
    REQUIRE(info.has_value()); REQUIRE(oldinfo.has_value());
    CHECK((**info-*oldinfo).norm()<1e-7);
    REQUIRE(ntml_covariance(**fit).has_value()); REQUIRE(ntml_covariance(**fit,true).has_value());
    const auto passes=(*data)->projection_passes;
    CHECK(ntml_quadratic(**fit,true).value()==*score);
    REQUIRE(ntml_covariance(**fit,true).has_value()); REQUIRE(ntml_spectrum(**lr).has_value());
    CHECK((*data)->projection_passes==passes);
    CHECK((*data)->contribution_builds==1);
    CHECK((*fit)->geometry_builds==1); CHECK((*fit)->u_builds==1);
    CHECK((**lr).spectrum_builds==1);
    auto tiled=prepare_ntml_data(raw,means,ContributionStorage::Tiled);
    REQUIRE(tiled.has_value());
    auto tf=prepare_ntml_fit(*tiled,h.pt,h.rep,*est); REQUIRE(tf.has_value());
    auto tq=ntml_quadratic(**tf,false); REQUIRE(tq.has_value());
    auto te=ntml_spectrum(**tq); REQUIRE(te.has_value());
    CHECK((**te-**got).norm()<1e-9);
    CHECK((*tiled)->contribution_builds==0);
  }
}

TEST_CASE("prepared NTML spectrum uses the smaller row space and caches it") {
  rob::frontier::NTMLQuadratic q;
  q.df=5; q.rows=Eigen::MatrixXd::Random(2,5); q.statistic=1;
  auto e=rob::frontier::ntml_spectrum(q); REQUIRE(e.has_value());
  auto dense=rob::ugamma_eigenvalues(q.rows.transpose()*q.rows); REQUIRE(dense.has_value());
  CHECK((**e-*dense).norm()<1e-12); CHECK(q.row_space);
  REQUIRE(rob::frontier::ntml_spectrum(q).has_value()); CHECK(q.spectrum_builds==1);
}

// ── Two-stage (ML2S) MI and equality-release tests ──────────────────────────

namespace {

// Heavy-tailed four-indicator data, optionally with MCAR cells, split into
// blocks of the given sizes.
magmaan::data::RawData t_cfa_raw(std::mt19937& rng,
                                 const std::vector<Eigen::Index>& sizes,
                                 double df, Eigen::Index period) {
  magmaan::data::RawData raw;
  for (Eigen::Index n : sizes) {
    Eigen::MatrixXd X =
        multivariate_t_sample(rng, n, four_indicator_sample_cov(), df);
    if (period > 0) {
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M =
          Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(n, 4);
      for (Eigen::Index i = 0; i < n; i += period) {
        M(i, i % 4) = 0;
        X(i, i % 4) = std::numeric_limits<double>::quiet_NaN();
      }
      raw.mask.push_back(M);
    }
    raw.X.push_back(std::move(X));
  }
  return raw;
}

magmaan::data::SampleStats stage1_moments(
    const magmaan::estimate::fiml::SaturatedMoments& sm) {
  magmaan::data::SampleStats s;
  s.S = sm.cov;
  s.mean = sm.mean;
  s.n_obs = sm.n_obs;
  return s;
}

void check_same_scores(const inf::ScoreTestTable& a,
                       const inf::ScoreTestTable& b, double tol) {
  REQUIRE(a.rows.size() == b.rows.size());
  REQUIRE(!a.rows.empty());
  for (std::size_t i = 0; i < a.rows.size(); ++i) {
    CHECK(a.rows[i].candidate.row == b.rows[i].candidate.row);
    CHECK(a.rows[i].mi == doctest::Approx(b.rows[i].mi).epsilon(tol));
    CHECK(a.rows[i].scaling_factor ==
          doctest::Approx(b.rows[i].scaling_factor).epsilon(tol));
    CHECK(a.rows[i].mi_scaled ==
          doctest::Approx(b.rows[i].mi_scaled).epsilon(tol));
  }
}

}  // namespace

TEST_CASE("frontier ML2S MI and releases reduce to complete-data robust tests") {
  // Without missing data the Stage-1 EM moments are the sample moments and
  // the Stage-1 covariance n·ACOV is the empirical Gamma, so every two-stage
  // score test must equal the complete-data robust test with the empirical
  // Gamma and the same Stage-2 discrepancy and weight, including the
  // estimated-weight meat. Two groups check the per-group n_b/N weighting.
  using TW = magmaan::estimate::fiml::TwoStageWeight;
  const char* syntax = "f =~ x1 + a*x2 + a*x3 + x4\nx1 ~~ 0*x2";
  for (int groups : {1, 2}) {
    CAPTURE(groups);
    auto h = groups == 1 ? build_mean(syntax) : build_groups_mean(syntax, 2);
    std::mt19937 rng(20261002u + static_cast<unsigned>(groups));
    const std::vector<Eigen::Index> sizes =
        groups == 1 ? std::vector<Eigen::Index>{800}
                    : std::vector<Eigen::Index>{500, 350};
    auto raw = t_cfa_raw(rng, sizes, 7.0, 0);
    auto pack = magmaan::estimate::fiml::fiml_pack(raw);
    REQUIRE(pack.has_value());
    auto h1 = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack);
    REQUIRE(h1.has_value());
    auto sm = magmaan::estimate::fiml::saturated_em_moments(raw, *pack, *h1);
    REQUIRE(sm.has_value());
    const auto stage2 = stage1_moments(*sm);

    inf::frontier::Ml2sScoreOptions o;
    o.base.candidates = inf::ScoreCandidateSet::WithAbsentRows;
    inf::frontier::RobustScoreOptions r;
    r.base = o.base;
    r.spec = rob::InferenceSpec{rob::Information::Expected,
                                rob::WeightMoments::Structured,
                                rob::ScoreCovariance::Empirical};

    // NT: the ML discrepancy with the structured normal-theory bread.
    auto est_ml = magmaan::test::fit(h.pt, h.rep, stage2);
    REQUIRE(est_ml.has_value());
    auto mi_ml2s = inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm,
                                                            *est_ml, o);
    REQUIRE_MESSAGE(mi_ml2s.has_value(),
                    (mi_ml2s.has_value() ? "" : mi_ml2s.error().detail));
    auto mi_ml = inf::frontier::modification_indices_robust(
        h.pt, h.rep, stage2, raw, *est_ml, r);
    REQUIRE(mi_ml.has_value());
    check_same_scores(*mi_ml2s, *mi_ml, 1e-7);
    auto rel_ml2s = inf::frontier::score_tests_ml2s(h.pt, h.rep, *sm, *est_ml, o);
    REQUIRE(rel_ml2s.has_value());
    auto rel_ml = inf::frontier::score_tests_robust(h.pt, h.rep, stage2, raw,
                                                    *est_ml, r);
    REQUIRE(rel_ml.has_value());
    check_same_scores(*rel_ml2s, *rel_ml, 1e-7);

    // Moment-quadratic Stage-2 weights, fixed and estimated.
    using Mode = magmaan::estimate::ContinuousLsIJWeightMode;
    const std::tuple<TW, Mode, double> weights[] = {
        {TW::Uls, Mode::Fixed, 0.5},
        {TW::Dwls, Mode::SampleEmpiricalDwls, 0.5},
        {TW::Adf, Mode::SampleEmpiricalWls, 0.5},
        {TW::Dls, Mode::SampleDls, 0.4}};
    for (const auto& [kind, mode, a] : weights) {
      CAPTURE(static_cast<int>(kind));
      auto w = magmaan::estimate::fiml::two_stage_stage2_weight_structured(
          *sm, kind, {a});
      REQUIRE(w.has_value());
      auto est = magmaan::test::fit_gmm(h.pt, h.rep, stage2, *w);
      REQUIRE(est.has_value());
      for (bool estimated : {false, true}) {
        CAPTURE(estimated);
        inf::frontier::Ml2sScoreOptions ok = o;
        ok.weight = kind;
        ok.dls.a = a;
        ok.robust = {estimated, &raw, &*pack, &*h1};
        inf::frontier::RobustScoreOptions rk = r;
        rk.estimated_weight = estimated;
        rk.ij_weight_mode = mode;
        rk.dls_opts.a = a;
        auto a_mi = inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm,
                                                             *est, ok);
        REQUIRE_MESSAGE(a_mi.has_value(),
                        (a_mi.has_value() ? "" : a_mi.error().detail));
        auto b_mi = inf::frontier::modification_indices_robust(
            h.pt, h.rep, stage2, raw, *est, *w, rk);
        REQUIRE_MESSAGE(b_mi.has_value(),
                        (b_mi.has_value() ? "" : b_mi.error().detail));
        check_same_scores(*a_mi, *b_mi, 1e-7);
        auto a_rel = inf::frontier::score_tests_ml2s(h.pt, h.rep, *sm, *est, ok);
        REQUIRE(a_rel.has_value());
        auto b_rel = inf::frontier::score_tests_robust(h.pt, h.rep, stage2, raw,
                                                       *est, *w, rk);
        REQUIRE(b_rel.has_value());
        check_same_scores(*a_rel, *b_rel, 1e-7);
      }
    }
  }
}

TEST_CASE("frontier ML2S MI: naive statistic plus Stage-1 scaling under missing data") {
  // The unscaled statistic is the Stage-2 discrepancy's test on the EM
  // moments, exactly the naive comparator; the Stage-1 covariance changes
  // only the scaling. The estimated-weight meat moves it further for DWLS.
  using TW = magmaan::estimate::fiml::TwoStageWeight;
  auto h = build_mean("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  std::mt19937 rng(20261003u);
  auto raw = t_cfa_raw(rng, {900}, 6.0, 3);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto h1 = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack);
  REQUIRE(h1.has_value());
  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw, *pack, *h1);
  REQUIRE(sm.has_value());
  const auto stage2 = stage1_moments(*sm);
  inf::ModificationIndexOptions base;
  base.candidates = inf::ScoreCandidateSet::WithAbsentRows;

  auto est_ml = magmaan::test::fit(h.pt, h.rep, stage2);
  REQUIRE(est_ml.has_value());
  inf::frontier::Ml2sScoreOptions o;
  o.base = base;
  auto nt = inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm, *est_ml, o);
  REQUIRE(nt.has_value());
  auto naive = inf::modification_indices(h.pt, h.rep, stage2, *est_ml, base);
  REQUIRE(naive.has_value());
  REQUIRE(nt->rows.size() == naive->rows.size());
  for (std::size_t i = 0; i < nt->rows.size(); ++i) {
    CHECK(nt->rows[i].mi ==
          doctest::Approx(naive->rows[i].mi).epsilon(1e-9));
    CHECK(std::isfinite(nt->rows[i].scaling_factor));
    CHECK(nt->rows[i].scaling_factor > 0.0);
  }

  auto w = magmaan::estimate::fiml::two_stage_stage2_weight_structured(
      *sm, TW::Dwls);
  REQUIRE(w.has_value());
  auto est = magmaan::test::fit_gmm(h.pt, h.rep, stage2, *w);
  REQUIRE(est.has_value());
  inf::frontier::Ml2sScoreOptions fixed = o;
  fixed.weight = TW::Dwls;
  auto f = inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm, *est,
                                                    fixed);
  REQUIRE(f.has_value());
  auto naive_ls = inf::modification_indices(h.pt, h.rep, stage2, *est, *w, base);
  REQUIRE(naive_ls.has_value());
  inf::frontier::Ml2sScoreOptions ew = fixed;
  ew.robust = {true, &raw, &*pack, &*h1};
  auto e = inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm, *est, ew);
  REQUIRE_MESSAGE(e.has_value(), (e.has_value() ? "" : e.error().detail));
  REQUIRE(f->rows.size() == naive_ls->rows.size());
  REQUIRE(e->rows.size() == f->rows.size());
  bool any_shift = false;
  for (std::size_t i = 0; i < f->rows.size(); ++i) {
    CHECK(f->rows[i].mi ==
          doctest::Approx(naive_ls->rows[i].mi).epsilon(1e-9));
    CHECK(e->rows[i].mi == doctest::Approx(f->rows[i].mi).epsilon(1e-12));
    CHECK(std::isfinite(e->rows[i].scaling_factor));
    if (std::abs(e->rows[i].scaling_factor - f->rows[i].scaling_factor) > 1e-3)
      any_shift = true;
  }
  CHECK(any_shift);

  // Unsupported requests are explicit.
  inf::frontier::Ml2sScoreOptions observed = o;
  observed.base.information = inf::ScoreInformation::Observed;
  auto obs = inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm,
                                                      *est_ml, observed);
  REQUIRE_FALSE(obs.has_value());
  CHECK(obs.error().kind == magmaan::PostError::Kind::UnsupportedInference);
  inf::frontier::Ml2sScoreOptions no_raw = fixed;
  no_raw.robust.estimated_weight = true;
  CHECK_FALSE(inf::frontier::modification_indices_ml2s(h.pt, h.rep, *sm, *est,
                                                       no_raw).has_value());
}
