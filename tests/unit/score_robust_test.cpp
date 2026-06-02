#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <random>
#include <string_view>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/robust.hpp"
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
