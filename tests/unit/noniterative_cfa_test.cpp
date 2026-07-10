#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/frontier/noniterative_cfa.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/frontier/noniterative_inference.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::data::SampleStats;
using magmaan::data::RawData;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::LatentStructure;
namespace ef = magmaan::estimate::frontier;
namespace rf = magmaan::robust::frontier;

// No-exceptions doctest: REQUIRE does not abort, so guard derefs. Bail out of
// the void test body on an error-state std::expected.
#define REQUIRE_OK(value)                                                     \
  do {                                                                        \
    INFO("error: " << ((value).has_value() ? "" : (value).error().detail));  \
    REQUIRE((value).has_value());                                             \
    if (!(value).has_value()) return;                                         \
  } while (false)

namespace {

struct Built {
  LatentStructure pt;
  MatrixRep rep;
};

Built build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr)};
}

// Exact 2-factor population: f1 ← x1,x2,x3 (marker x1), f2 ← x4,x5,x6 (marker x4).
Eigen::MatrixXd two_factor_cov() {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L(0, 0) = 1.0; L(1, 0) = 0.8; L(2, 0) = 1.2;
  L(3, 1) = 1.0; L(4, 1) = 0.7; L(5, 1) = 1.3;
  Eigen::MatrixXd Psi(2, 2);
  Psi << 1.0, 0.3, 0.3, 1.0;
  Eigen::MatrixXd S = L * Psi * L.transpose();
  for (Eigen::Index i = 0; i < 6; ++i) S(i, i) += 0.5;
  return S;
}

Eigen::MatrixXd three_factor_equicorrelated_cov() {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(9, 3);
  for (Eigen::Index f = 0; f < 3; ++f) {
    L(3 * f, f) = 1.0;
    L(3 * f + 1, f) = 0.8;
    L(3 * f + 2, f) = 1.2;
  }
  Eigen::MatrixXd Phi = Eigen::MatrixXd::Constant(3, 3, 0.6);
  Phi.diagonal().setOnes();
  Eigen::MatrixXd S = L * Phi * L.transpose();
  S.diagonal().array() += 0.5;
  return S;
}

RawData raw_from_cov(const Eigen::MatrixXd& S) {
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::Index p = S.rows();
  const Eigen::Index n = 2 * p;
  const double scale = std::sqrt(static_cast<double>(n) / 2.0);
  const Eigen::MatrixXd L = llt.matrixL();
  Eigen::MatrixXd X(n, p);
  X.topRows(p) = scale * L.transpose();
  X.bottomRows(p) = -scale * L.transpose();
  RawData raw;
  raw.X = {std::move(X)};
  return raw;
}

constexpr const char* kTwoFactor =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ x4 + x5 + x6\n"
    "f1 ~~ f2\n";

constexpr const char* kThreeFactor =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ x4 + x5 + x6\n"
    "f3 =~ x7 + x8 + x9\n"
    "f1 ~~ f2 + f3\n"
    "f2 ~~ f3\n";

// The θ₀ that the exact population implies, keyed by matrix cell.
double true_param(const magmaan::model::ParamLocation& loc) {
  using magmaan::model::MatId;
  switch (loc.mat) {
    case MatId::Lambda:
      switch (loc.row) {
        case 1: return 0.8;
        case 2: return 1.2;
        case 4: return 0.7;
        case 5: return 1.3;
        default: return 1.0;
      }
    case MatId::Psi:
      return (loc.row == loc.col) ? 1.0 : 0.3;
    case MatId::Theta:
      return 0.5;
    default:
      return 0.0;
  }
}

magmaan::fit_expected<Eigen::MatrixXd>
finite_difference_jacobian(const LatentStructure& pt, const MatrixRep& rep,
                           const ModelEvaluator& ev, const SampleStats& samp,
                           ef::NonIterativeEstimator which,
                           ef::CompositeWeight composite,
                           ef::AdmissibilityConfig admissibility = {},
                           ef::ScoreConditioningConfig score_conditioning = {}) {
  if (samp.S.empty()) {
    return std::unexpected(magmaan::FitError{
        magmaan::FitError::Kind::NumericIssue, "test: empty sample stats"});
  }
  const Eigen::MatrixXd S0 = samp.S[0];
  const Eigen::Index p = S0.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::MatrixXd J(static_cast<Eigen::Index>(ev.n_free()), pstar);
  SampleStats work = samp;
  Eigen::Index col = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      const double h = 2e-6 * std::max(std::abs(S0(r, c)), 1.0);
      work.S[0] = S0;
      work.S[0](r, c) += h;
      if (r != c) work.S[0](c, r) += h;
      auto tp = ef::noniterative_cfa_theta(
          pt, rep, ev, work, which, composite, admissibility,
          score_conditioning);
      if (!tp.has_value()) return std::unexpected(tp.error());

      work.S[0] = S0;
      work.S[0](r, c) -= h;
      if (r != c) work.S[0](c, r) -= h;
      auto tm = ef::noniterative_cfa_theta(
          pt, rep, ev, work, which, composite, admissibility,
          score_conditioning);
      if (!tm.has_value()) return std::unexpected(tm.error());
      J.col(col) = (*tp - *tm) / (2.0 * h);
      ++col;
    }
  }
  return J;
}

}  // namespace

TEST_CASE("noniterative Guttman map recovers full theta on exact population") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  for (auto which : {ef::NonIterativeEstimator::GuttmanLavaan,
                     ef::NonIterativeEstimator::GuttmanAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp, which);
    REQUIRE_OK(th);

    const auto locs = ev->param_locations();
    REQUIRE(static_cast<Eigen::Index>(locs.size()) == th->size());
    for (std::size_t k = 0; k < locs.size(); ++k)
      CHECK((*th)(static_cast<Eigen::Index>(k)) ==
            doctest::Approx(true_param(locs[k])).epsilon(1e-5));
  }
}

TEST_CASE("restricted aligned map clamps an improper communality draw") {
  Built b = build("f =~ x1 + x2 + x3\n");
  SampleStats samp;
  Eigen::MatrixXd S(3, 3);
  S << 1.0, 0.30, 0.45,
       0.30, 1.0, -0.10,
       0.45, -0.10, 1.0;
  samp.S = {S};
  samp.n_obs = {400};

  ef::AdmissibilityConfig hard;
  hard.policy = ef::AdmissibilityPolicy::Hard;
  hard.margin = 0.01;
  auto hard_fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CommunalityMethod::ExtendedTriadLeastSquares,
      ef::CompositeWeight::Standardized, hard);
  REQUIRE_OK(hard_fit);
  REQUIRE(hard_fit->n_h2_clamped.size() == 1);
  CHECK(hard_fit->n_h2_clamped[0] > 0);

  ef::AdmissibilityConfig soft = hard;
  soft.policy = ef::AdmissibilityPolicy::Soft;
  soft.beta0 = 1.0;
  soft.rate_exp = 0.5;
  auto soft_fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CommunalityMethod::ExtendedTriadLeastSquares,
      ef::CompositeWeight::Standardized, soft);
  REQUIRE_OK(soft_fit);
  CHECK(soft_fit->n_h2_clamped[0] > 0);

  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  auto analytic = ef::estimator_map_jacobian_restricted_block(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned, 0,
      2e-6, ef::CommunalityMethod::ExtendedTriadLeastSquares,
      ef::CompositeWeight::Standardized, soft);
  REQUIRE_OK(analytic);
  Eigen::MatrixXd fd(analytic->rows(), analytic->cols());
  Eigen::Index col = 0;
  for (Eigen::Index c = 0; c < S.rows(); ++c) {
    for (Eigen::Index r = c; r < S.rows(); ++r) {
      const double h = 2e-6 * std::max(std::abs(S(r, c)), 1.0);
      SampleStats plus = samp;
      SampleStats minus = samp;
      plus.S[0](r, c) += h;
      minus.S[0](r, c) -= h;
      if (r != c) {
        plus.S[0](c, r) += h;
        minus.S[0](c, r) -= h;
      }
      auto fp = ef::fit_noniterative_cfa_restricted(
          b.pt, b.rep, plus, ef::NonIterativeEstimator::GuttmanAligned,
          ef::CommunalityMethod::ExtendedTriadLeastSquares,
          ef::CompositeWeight::Standardized, soft);
      auto fm = ef::fit_noniterative_cfa_restricted(
          b.pt, b.rep, minus, ef::NonIterativeEstimator::GuttmanAligned,
          ef::CommunalityMethod::ExtendedTriadLeastSquares,
          ef::CompositeWeight::Standardized, soft);
      REQUIRE_OK(fp);
      REQUIRE_OK(fm);
      fd.col(col++) = (fp->theta - fm->theta) / (2.0 * h);
    }
  }
  CHECK((*analytic - fd).cwiseAbs().maxCoeff() < 5e-5);
}

TEST_CASE("standardized composite weights are exact and inferable") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  for (auto composite : {ef::CompositeWeight::Unit,
                         ef::CompositeWeight::Standardized,
                         ef::CompositeWeight::Adaptive}) {
    INFO("composite ordinal: " << static_cast<int>(composite));
    auto th = ef::noniterative_cfa_theta(
        b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
        composite);
    REQUIRE_OK(th);

    const auto locs = ev->param_locations();
    for (std::size_t k = 0; k < locs.size(); ++k)
      CHECK((*th)(static_cast<Eigen::Index>(k)) ==
            doctest::Approx(true_param(locs[k])).epsilon(1e-5));

    auto D = ev->dsigma_dtheta(*th);
    REQUIRE_OK(D);
    auto J = ef::estimator_map_jacobian(
        b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
        1e-6, composite);
    REQUIRE_OK(J);
    const Eigen::MatrixXd JD = (*J) * (*D);
    CHECK((JD - Eigen::MatrixXd::Identity(th->size(), th->size()))
              .cwiseAbs()
              .maxCoeff() < 1e-4);

    auto inf = rf::noniterative_inference_nt(
        b.pt, b.rep, samp, *th, ef::NonIterativeEstimator::GuttmanAligned,
        rf::Discrepancy::ULS, composite);
    REQUIRE_OK(inf);
    CHECK(inf->Omega.allFinite());
  }
}

TEST_CASE("standardized composite weights change the off-model map") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 4) += 0.15;
  S(4, 0) += 0.15;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {500};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  auto unit = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Unit);
  auto std = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized);
  REQUIRE_OK(unit);
  REQUIRE_OK(std);
  CHECK((*unit - *std).cwiseAbs().maxCoeff() > 1e-6);
}

TEST_CASE("aligned score conditioning preserves raw compatibility and resolves auto") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 4) += 0.12;
  S(4, 0) += 0.12;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  ef::ScoreConditioningConfig raw;
  auto std_default = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized);
  auto std_raw = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, raw);
  auto aligned_auto = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned);
  REQUIRE_OK(std_default);
  REQUIRE_OK(std_raw);
  REQUIRE_OK(aligned_auto);
  CHECK((std_default->array() == std_raw->array()).all());
  CHECK((std_default->array() == aligned_auto->array()).all());

  raw.floor0 = -123.0;
  raw.rate_exp = -4.0;
  auto adaptive_compat = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Adaptive, {}, raw);
  auto adaptive_default = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Adaptive);
  REQUIRE_OK(adaptive_compat);
  REQUIRE_OK(adaptive_default);
  CHECK((adaptive_compat->array() == adaptive_default->array()).all());
}

TEST_CASE("hard and soft score conditioning attain the normalized floor") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  ef::ScoreConditioningConfig inactive;
  inactive.policy = ef::ScoreConditioningPolicy::Hard;
  inactive.floor0 = 1.0;  // delta_400 = 0.05 < lambda_min(R)
  auto raw = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized);
  auto hard_inactive = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, inactive);
  REQUIRE_OK(raw);
  REQUIRE_OK(hard_inactive);
  CHECK((raw->theta.array() == hard_inactive->theta.array()).all());
  REQUIRE(hard_inactive->score_conditioning_diagnostics.size() == 1);
  CHECK(hard_inactive->score_conditioning_diagnostics[0].shrinkage == 0.0);

  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  ef::ScoreConditioningConfig boundary = inactive;
  boundary.floor0 =
      20.0 * raw->score_conditioning_diagnostics[0]
                   .raw_normalized_min_eigenvalue;
  auto boundary_analytic = ef::estimator_map_jacobian_analytic(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, boundary);
  CHECK_FALSE(boundary_analytic.has_value());
  auto boundary_fallback = ef::estimator_map_jacobian(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      2e-6, ef::CompositeWeight::Standardized, {}, boundary);
  REQUIRE_OK(boundary_fallback);

  ef::ScoreConditioningConfig active = inactive;
  active.floor0 = 16.0;  // delta_400 = 0.8 > lambda_min(R) = 0.7
  for (auto policy : {ef::ScoreConditioningPolicy::Hard,
                      ef::ScoreConditioningPolicy::Soft}) {
    active.policy = policy;
    INFO("score policy ordinal: " << static_cast<int>(policy));
    auto fit = ef::fit_noniterative_cfa(
        b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
        ef::CompositeWeight::Standardized, {}, active);
    REQUIRE_OK(fit);
    REQUIRE(fit->score_conditioning_diagnostics.size() == 1);
    const auto& d = fit->score_conditioning_diagnostics[0];
    CHECK(d.target_floor == doctest::Approx(0.8));
    CHECK(d.raw_normalized_min_eigenvalue < d.target_floor);
    CHECK(d.repaired_normalized_min_eigenvalue >= d.target_floor - 1e-10);
    CHECK(d.shrinkage > 0.0);
    CHECK(d.hard_violation);
    CHECK(d.min_score_variance > 0.0);
    CHECK(d.min_abs_marker > 0.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fit->Phi[0]);
    REQUIRE(es.info() == Eigen::Success);
    CHECK(es.eigenvalues().minCoeff() > 0.0);
  }
}

TEST_CASE("score conditioning is scale equivariant and validates its scope") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  SampleStats scaled = samp;
  scaled.S[0] *= 7.0;

  ef::ScoreConditioningConfig hard;
  hard.policy = ef::ScoreConditioningPolicy::Hard;
  hard.floor0 = 16.0;
  auto fit = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Unit, {}, hard);
  auto fit_scaled = ef::fit_noniterative_cfa(
      b.pt, b.rep, scaled, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Unit, {}, hard);
  REQUIRE_OK(fit);
  REQUIRE_OK(fit_scaled);
  CHECK((fit_scaled->Lambda[0] - fit->Lambda[0]).cwiseAbs().maxCoeff() < 1e-11);
  CHECK((fit_scaled->Phi[0] - 7.0 * fit->Phi[0]).cwiseAbs().maxCoeff() < 1e-10);
  CHECK((fit_scaled->psi[0] - 7.0 * fit->psi[0]).cwiseAbs().maxCoeff() < 1e-10);
  CHECK(fit_scaled->score_conditioning_diagnostics[0]
            .raw_normalized_min_eigenvalue ==
        doctest::Approx(fit->score_conditioning_diagnostics[0]
                            .raw_normalized_min_eigenvalue));
  CHECK(fit_scaled->score_conditioning_diagnostics[0].shrinkage ==
        doctest::Approx(fit->score_conditioning_diagnostics[0].shrinkage));

  ef::ScoreConditioningConfig invalid = hard;
  invalid.floor0 = 0.0;
  CHECK_FALSE(ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Unit, {}, invalid).has_value());
  invalid.floor0 = 20.0;  // delta_400 = 1
  CHECK_FALSE(ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Unit, {}, invalid).has_value());
  CHECK_FALSE(ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanLavaan,
      ef::CompositeWeight::Unit, {}, hard).has_value());
  CHECK_FALSE(ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Adaptive, {}, hard).has_value());
}

TEST_CASE("fixed-diagonal H conditioning repairs the aligned proxy for point fits") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  auto raw = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized);
  REQUIRE_OK(raw);

  ef::HConditioningConfig hard;
  hard.policy = ef::HConditioningPolicy::Hard;
  hard.floor0 = 16.0;
  auto repaired = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, {}, hard);
  REQUIRE_OK(repaired);
  REQUIRE(repaired->h_conditioning_diagnostics.size() == 1);
  const auto& d = repaired->h_conditioning_diagnostics[0];
  CHECK(d.target_floor == doctest::Approx(0.8));
  CHECK(d.min_h_variance > 0.0);
  CHECK(d.repaired_normalized_min_eigenvalue >= d.target_floor - 1e-10);
  CHECK(d.shrinkage >= 0.0);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(repaired->Phi[0]);
  REQUIRE(es.info() == Eigen::Success);
  CHECK(es.eigenvalues().minCoeff() > 0.0);

  auto raw_explicit = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, {}, {});
  REQUIRE_OK(raw_explicit);
  CHECK((raw->theta.array() == raw_explicit->theta.array()).all());
  CHECK_FALSE(ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanLavaan,
      ef::CompositeWeight::Unit, {}, {}, hard).has_value());
  CHECK_FALSE(ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Adaptive, {}, {}, hard).has_value());
}

TEST_CASE("conditioned analytic Jacobians match central differences and inference") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 0) += 0.10;
  S(0, 4) += 0.04;
  S(4, 0) += 0.04;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  for (const auto config : std::array{
           ef::ScoreConditioningConfig{ef::ScoreConditioningPolicy::Hard, 1.0, 0.5},
           ef::ScoreConditioningConfig{ef::ScoreConditioningPolicy::Hard, 16.0, 0.5},
           ef::ScoreConditioningConfig{ef::ScoreConditioningPolicy::Soft, 16.0, 0.5}}) {
    INFO("score policy ordinal: " << static_cast<int>(config.policy));
    auto Ja = ef::estimator_map_jacobian_analytic(
        b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
        ef::CompositeWeight::Standardized, {}, config);
    auto Jfd = finite_difference_jacobian(
        b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
        ef::CompositeWeight::Standardized, {}, config);
    REQUIRE_OK(Ja);
    REQUIRE_OK(Jfd);
    CHECK((*Ja - *Jfd).cwiseAbs().maxCoeff() < 8e-5);

    auto fit = ef::fit_noniterative_cfa(
        b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
        ef::CompositeWeight::Standardized, {}, config);
    REQUIRE_OK(fit);
    auto se = rf::noniterative_se_nt(
        b.pt, b.rep, samp, fit->theta,
        ef::NonIterativeEstimator::GuttmanAligned,
        ef::CompositeWeight::Standardized, {}, config);
    REQUIRE_OK(se);
    CHECK(se->Omega.allFinite());
  }
}

TEST_CASE("hard score ties fall back while soft repeated eigenvalues stay analytic") {
  Built b = build(kThreeFactor);
  SampleStats samp;
  samp.S = {three_factor_equicorrelated_cov()};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  ef::ScoreConditioningConfig hard{
      ef::ScoreConditioningPolicy::Hard, 16.0, 0.5};
  auto hard_fit = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, hard);
  REQUIRE_OK(hard_fit);
  CHECK(hard_fit->score_conditioning_diagnostics[0]
            .repaired_normalized_min_eigenvalue >= 0.8 - 1e-10);

  auto hard_analytic = ef::estimator_map_jacobian_analytic(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, hard);
  CHECK_FALSE(hard_analytic.has_value());
  auto hard_fallback = ef::estimator_map_jacobian(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      2e-6, ef::CompositeWeight::Standardized, {}, hard);
  REQUIRE_OK(hard_fallback);

  ef::ScoreConditioningConfig soft = hard;
  soft.policy = ef::ScoreConditioningPolicy::Soft;
  auto soft_analytic = ef::estimator_map_jacobian_analytic(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, soft);
  auto soft_fd = finite_difference_jacobian(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned,
      ef::CompositeWeight::Standardized, {}, soft);
  REQUIRE_OK(soft_analytic);
  REQUIRE_OK(soft_fd);
  CHECK((*soft_analytic - *soft_fd).cwiseAbs().maxCoeff() < 1e-4);
}

TEST_CASE("analytic configural Jacobian matches central finite differences") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 0) += 0.20;
  S(3, 3) += 0.15;
  S(0, 4) += 0.12;
  S(4, 0) += 0.12;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {500};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  for (auto which : {ef::NonIterativeEstimator::GuttmanLavaan,
                     ef::NonIterativeEstimator::GuttmanAligned}) {
    for (auto composite : {ef::CompositeWeight::Unit,
                           ef::CompositeWeight::Standardized,
                           ef::CompositeWeight::Adaptive}) {
      INFO("estimator ordinal: " << static_cast<int>(which));
      INFO("composite ordinal: " << static_cast<int>(composite));
      auto Ja = ef::estimator_map_jacobian_analytic(
          b.pt, b.rep, *ev, samp, which, composite);
      auto Jfd = finite_difference_jacobian(
          b.pt, b.rep, *ev, samp, which, composite);
      REQUIRE_OK(Ja);
      REQUIRE_OK(Jfd);
      CHECK((*Ja - *Jfd).cwiseAbs().maxCoeff() < 2e-5);
    }
  }
}

TEST_CASE("noniterative map Jacobian satisfies Fisher consistency J*Delta = I") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  for (auto which : {ef::NonIterativeEstimator::GuttmanLavaan,
                     ef::NonIterativeEstimator::GuttmanAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp, which);
    REQUIRE_OK(th);

    auto D = ev->dsigma_dtheta(*th);   // Δ, p* × q
    REQUIRE_OK(D);
    auto J = ef::estimator_map_jacobian(b.pt, b.rep, *ev, samp, which);  // q × p*
    REQUIRE_OK(J);

    const Eigen::Index q = th->size();
    const Eigen::MatrixXd JD = (*J) * (*D);
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(q, q);
    CHECK((JD - I).cwiseAbs().maxCoeff() < 1e-4);
  }
}

TEST_CASE("noniterative GOF is ~0 at exact fit with correct df and finite SEs") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  for (auto which : {ef::NonIterativeEstimator::GuttmanLavaan,
                     ef::NonIterativeEstimator::GuttmanAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp, which);
    REQUIRE_OK(th);

    for (auto disc : {rf::Discrepancy::ULS, rf::Discrepancy::NTML}) {
      auto inf = rf::noniterative_inference_nt(b.pt, b.rep, samp, *th, which, disc);
      REQUIRE_OK(inf);
      CHECK(inf->T_gof < 1e-6);            // residual is ~0 at exact fit
      CHECK(inf->df == 8);                 // p*=21, q=13
      CHECK(inf->gof_eigenvalues.size() == 8);
      CHECK(inf->se.size() == th->size());
      for (Eigen::Index i = 0; i < inf->se.size(); ++i) {
        CHECK(std::isfinite(inf->se(i)));
        CHECK(inf->se(i) >= 0.0);
      }
    }
  }
}

TEST_CASE("noniterative SE-only path matches the full inference covariance") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 4) += 0.12;
  S(4, 0) += 0.12;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {500};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  for (auto which : {ef::NonIterativeEstimator::GuttmanLavaan,
                     ef::NonIterativeEstimator::GuttmanAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp, which);
    REQUIRE_OK(th);
    auto inf = rf::noniterative_inference_nt(
        b.pt, b.rep, samp, *th, which, rf::Discrepancy::NTML);
    auto se = rf::noniterative_se_nt(b.pt, b.rep, samp, *th, which);
    REQUIRE_OK(inf);
    REQUIRE_OK(se);

    CHECK((se->theta_hat - *th).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((se->Omega - inf->Omega).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((se->se - inf->se).cwiseAbs().maxCoeff() < 1e-12);
  }
}

TEST_CASE("noniterative empirical SE-only path matches dense Gamma inference") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 4) += 0.10;
  S(4, 0) += 0.10;
  RawData raw = raw_from_cov(S);
  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE_OK(samp_or);
  SampleStats samp = std::move(*samp_or);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);

  auto th = ef::noniterative_cfa_theta(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanAligned);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_empirical(
      b.pt, b.rep, samp, raw, *th, ef::NonIterativeEstimator::GuttmanAligned,
      rf::Discrepancy::ULS);
  auto se = rf::noniterative_se_empirical(
      b.pt, b.rep, samp, raw, *th, ef::NonIterativeEstimator::GuttmanAligned);
  REQUIRE_OK(inf);
  REQUIRE_OK(se);

  CHECK((se->Omega - inf->Omega).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((se->se - inf->se).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("NTML GOF statistic equals the RLS chi-square (model-implied weight)") {
  // Perturb S so the CFA cannot fit it exactly; then T_NTML must equal
  // N·½tr((Σ̂⁻¹(S−Σ̂))²) = rls_chi2. This fails if V uses the sample GLS weight.
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 4) += 0.15; S(4, 0) += 0.15;   // a cross-factor residual the model omits
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {500};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);

  auto inf = rf::noniterative_inference_nt(b.pt, b.rep, samp, *th,
                                           ef::NonIterativeEstimator::GuttmanLavaan,
                                           rf::Discrepancy::NTML);
  REQUIRE_OK(inf);
  CHECK(inf->T_gof > 0.0);
  CHECK(std::isfinite(inf->rls_check));
  CHECK(inf->T_gof == doctest::Approx(inf->rls_check).epsilon(1e-8));
}

TEST_CASE("ULS GOF statistic equals N*tr((S-Sigma)^2) (D'D weight, vech aligned)") {
  Built b = build(kTwoFactor);
  Eigen::MatrixXd S = two_factor_cov();
  S(0, 4) += 0.15; S(4, 0) += 0.15;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {500};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto sig = ev->sigma(*th);
  REQUIRE_OK(sig);
  const Eigen::MatrixXd D = S - sig->sigma[0];
  const double expected = 500.0 * D.squaredNorm();   // N·tr(D²) for symmetric D

  auto inf = rf::noniterative_inference_nt(b.pt, b.rep, samp, *th,
                                           ef::NonIterativeEstimator::GuttmanLavaan,
                                           rf::Discrepancy::ULS);
  REQUIRE_OK(inf);
  CHECK(inf->T_gof == doctest::Approx(expected).epsilon(1e-9));
}

TEST_CASE("noniterative Wald test: exact at the truth, positive off it, df correct") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE_OK(ev);
  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_nt(b.pt, b.rep, samp, *th,
                                           ef::NonIterativeEstimator::GuttmanLavaan,
                                           rf::Discrepancy::NTML);
  REQUIRE_OK(inf);

  // Restriction on the x2 loading (Lambda row 1), whose truth is 0.8.
  const auto locs = ev->param_locations();
  Eigen::Index kidx = -1;
  for (std::size_t k = 0; k < locs.size(); ++k)
    if (locs[k].mat == magmaan::model::MatId::Lambda && locs[k].row == 1) kidx = static_cast<Eigen::Index>(k);
  REQUIRE(kidx >= 0);

  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, th->size());
  R(0, kidx) = 1.0;
  Eigen::VectorXd qtrue(1);
  qtrue(0) = 0.8;
  auto w_true = rf::noniterative_wald(*th, *inf, R, qtrue);
  REQUIRE_OK(w_true);
  CHECK(w_true->df == 1);
  CHECK(w_true->chi2 < 1e-4);

  Eigen::VectorXd qoff(1);
  qoff(0) = 1.3;
  auto w_off = rf::noniterative_wald(*th, *inf, R, qoff);
  REQUIRE_OK(w_off);
  CHECK(w_off->chi2 > 0.0);
  CHECK(std::isfinite(w_off->chi2));
}

TEST_CASE("noniterative residual modification indices expose raw, residualized, and drop diagnostics") {
  Built h0 = build(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f1 ~~ 0*f2\n");
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev0 = ModelEvaluator::build(h0.pt, h0.rep);
  REQUIRE_OK(ev0);
  auto t0 = ef::noniterative_cfa_theta(h0.pt, h0.rep, *ev0, samp);
  REQUIRE_OK(t0);
  auto inf0 = rf::noniterative_inference_nt(
      h0.pt, h0.rep, samp, *t0, ef::NonIterativeEstimator::GuttmanLavaan,
      rf::Discrepancy::ULS);
  REQUIRE_OK(inf0);

  magmaan::inference::ModificationIndexOptions fixed_opts;
  auto fixed = rf::noniterative_modification_indices(
      h0.pt, h0.rep, samp, *t0, *inf0, fixed_opts);
  REQUIRE_OK(fixed);
  REQUIRE(fixed->rows.size() == 1);
  const auto& cov = fixed->rows.front();
  CHECK(cov.candidate.op == magmaan::parse::Op::Covariance);
  CHECK(cov.score_raw == doctest::Approx(cov.score_resid).epsilon(1e-10));
  CHECK(cov.mi_raw > 0.0);
  CHECK(cov.mi_resid > 0.0);
  CHECK(cov.drop_resid > 0.0);
  CHECK(std::isfinite(cov.epc_resid));
  CHECK(cov.residualized_norm > 0.0);

  Built cfg = build(kTwoFactor);
  auto ev = ModelEvaluator::build(cfg.pt, cfg.rep);
  REQUIRE_OK(ev);
  auto th = ef::noniterative_cfa_theta(cfg.pt, cfg.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_nt(
      cfg.pt, cfg.rep, samp, *th, ef::NonIterativeEstimator::GuttmanLavaan,
      rf::Discrepancy::ULS);
  REQUIRE_OK(inf);

  magmaan::inference::ModificationIndexOptions absent_opts;
  absent_opts.candidates =
      magmaan::inference::ScoreCandidateSet::WithAbsentRows;
  absent_opts.include_loadings = true;
  absent_opts.include_covariances = false;
  auto absent = rf::noniterative_modification_indices(
      cfg.pt, cfg.rep, samp, *th, *inf, absent_opts);
  REQUIRE_OK(absent);
  REQUIRE(absent->rows.size() > 0);
  bool saw_loading = false;
  for (const auto& row : absent->rows) {
    if (row.candidate.op != magmaan::parse::Op::Measurement) continue;
    saw_loading = true;
    CHECK(std::isfinite(row.mi_raw));
    CHECK(std::isfinite(row.mi_resid));
    CHECK(std::isfinite(row.drop_raw));
    CHECK(std::isfinite(row.drop_resid));
    CHECK(std::isfinite(row.epc_raw));
    CHECK(std::isfinite(row.epc_resid));
    CHECK(row.signature_norm > 0.0);
    CHECK(row.residualized_norm >= 0.0);
  }
  CHECK(saw_loading);
}

TEST_CASE("noniterative difference test flags a false zero-covariance restriction") {
  // Population has f1~~f2 = 0.3. H0 fixes it to 0 (misspecified); H1 frees it.
  Built h1 = build(kTwoFactor);
  Built h0 = build(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f1 ~~ 0*f2\n");
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};
  auto ev1 = ModelEvaluator::build(h1.pt, h1.rep);
  auto ev0 = ModelEvaluator::build(h0.pt, h0.rep);
  REQUIRE_OK(ev1);
  REQUIRE_OK(ev0);
  auto t1 = ef::noniterative_cfa_theta(h1.pt, h1.rep, *ev1, samp);
  auto t0 = ef::noniterative_cfa_theta(h0.pt, h0.rep, *ev0, samp);
  REQUIRE_OK(t1);
  REQUIRE_OK(t0);

  auto inf1 = rf::noniterative_inference_nt(h1.pt, h1.rep, samp, *t1,
                                            ef::NonIterativeEstimator::GuttmanLavaan,
                                            rf::Discrepancy::ULS);
  auto inf0 = rf::noniterative_inference_nt(h0.pt, h0.rep, samp, *t0,
                                            ef::NonIterativeEstimator::GuttmanLavaan,
                                            rf::Discrepancy::ULS);
  REQUIRE_OK(inf1);
  REQUIRE_OK(inf0);

  auto diff = rf::noniterative_difference_test(*inf0, *inf1, 1);
  REQUIRE_OK(diff);
  CHECK(diff->df_d == 1);
  CHECK(diff->eigenvalues.size() == 1);
  CHECK(diff->T_d > 0.0);   // H1 fits, H0 does not
}
