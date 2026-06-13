#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

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
  magmaan::estimate::gmm::Weight weight{W_dwls};

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
// with the non-robust FIML MI, so the unscaled `mi` must match (the only
// difference is analytic vs central-difference info, so to the FD floor). The
// scale of the sandwich meat (B1 = ¼·scoresᵀscores against A1 = (N/2)·H) is
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
  REQUIRE(rob->rows.size() > 0);
  for (std::size_t i = 0; i < rob->rows.size(); ++i) {
    CHECK(std::abs(rob->rows[i].mi - nt->rows[i].mi) <
          1e-4 * (1.0 + std::abs(nt->rows[i].mi)));
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
  REQUIRE(rob->rows.size() > 0);
  // Per-candidate c is a noisy 4th-moment-driven ratio, so anchor the *mean*
  // (variance ~1/rows lower) at 1; a wrong B1/A1 scale constant (2× or ½×)
  // would instead push every row to |c−1| ≈ 1 or ½, which max_dev catches.
  double sum_c = 0.0;
  double max_dev = 0.0;
  for (const auto& r : rob->rows) {
    sum_c += r.scaling_factor;
    max_dev = std::max(max_dev, std::abs(r.scaling_factor - 1.0));
  }
  const double mean_c = sum_c / static_cast<double>(rob->rows.size());
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

  // Single-group guard (v1): the FIML robust tier rejects multi-block raw.
  magmaan::data::RawData two_block = raw;
  two_block.X.push_back(raw.X[0]);
  two_block.mask.push_back(raw.mask[0]);
  auto rob_mg = inf::frontier::score_tests_fiml_robust(h.pt, h.rep, two_block,
                                                       *est);
  CHECK_FALSE(rob_mg.has_value());
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
