#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/frontier/noniterative_cfa.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/frontier/noniterative_inference.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatrixRep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::GroupEqual;
using magmaan::spec::LatentStructure;
namespace ef = magmaan::estimate::frontier;
namespace rf = magmaan::robust::frontier;

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

Built build_mg(std::string_view src, std::int32_t ngroups,
               std::vector<GroupEqual> ge = {}, bool means = false) {
  BuildOptions opts;
  opts.n_groups = ngroups;
  opts.group_equal = std::move(ge);
  opts.meanstructure = means;
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr)};
}

Eigen::VectorXd mk_mean(const std::array<double, 6>& v) {
  Eigen::VectorXd m(6);
  for (Eigen::Index i = 0; i < 6; ++i) m(i) = v[static_cast<std::size_t>(i)];
  return m;
}

// The 6×2 loading matrix (markers x1,x4) used by both tf_cov and the scalar
// mean-shift construction.
Eigen::MatrixXd tf_loadings(const std::array<double, 6>& lam) {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L(0, 0) = lam[0]; L(1, 0) = lam[1]; L(2, 0) = lam[2];
  L(3, 1) = lam[3]; L(4, 1) = lam[4]; L(5, 1) = lam[5];
  return L;
}

// Exact 2-factor covariance: markers x1,x4; loadings `lam` (incl. markers),
// factor variance `pd`, factor covariance `po`, common residual `th`.
Eigen::MatrixXd tf_cov(const std::array<double, 6>& lam, double pd, double po,
                       double th) {
  Eigen::MatrixXd L = tf_loadings(lam);
  Eigen::MatrixXd Psi(2, 2);
  Psi << pd, po, po, pd;
  Eigen::MatrixXd S = L * Psi * L.transpose();
  for (Eigen::Index i = 0; i < 6; ++i) S(i, i) += th;
  return S;
}

Eigen::MatrixXd of_cov(const std::array<double, 4>& lam, double psi,
                       const std::array<double, 4>& theta) {
  Eigen::VectorXd l(4);
  for (Eigen::Index i = 0; i < 4; ++i) l(i) = lam[static_cast<std::size_t>(i)];
  Eigen::MatrixXd S = psi * (l * l.transpose());
  for (Eigen::Index i = 0; i < 4; ++i) S(i, i) += theta[static_cast<std::size_t>(i)];
  return S;
}

constexpr const char* kTwoFactor =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ x4 + x5 + x6\n"
    "f1 ~~ f2\n";

constexpr const char* kTwoFactorMarkerX2 =
    "f1 =~ NA*x1 + 1*x2 + x3\n"
    "f2 =~ NA*x4 + 1*x5 + x6\n"
    "f1 ~~ f2\n";

constexpr const char* kOneFactorResidualTie =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 ~~ eq*x1\n"
    "x2 ~~ eq*x2\n";

constexpr const char* kOneFactorResidualTieX1X3 =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 ~~ eq*x1\n"
    "x3 ~~ eq*x3\n";

constexpr const char* kOneFactorResidualTieX1X3MarkerX2 =
    "f =~ NA*x1 + 1*x2 + x3 + x4\n"
    "x1 ~~ eq*x1\n"
    "x3 ~~ eq*x3\n";

const std::array<double, 6> kLam = {1.0, 0.8, 1.2, 1.0, 0.7, 1.3};
const std::array<double, 4> kLam4 = {1.0, 0.8, 1.2, 0.7};

double max_sigma_diff(const ModelEvaluator& a_ev, const Eigen::VectorXd& a_theta,
                      const ModelEvaluator& b_ev, const Eigen::VectorXd& b_theta) {
  auto ai = a_ev.sigma(a_theta);
  auto bi = b_ev.sigma(b_theta);
  REQUIRE(ai.has_value());
  REQUIRE(bi.has_value());
  REQUIRE(ai->sigma.size() == bi->sigma.size());
  double out = 0.0;
  for (std::size_t b = 0; b < ai->sigma.size(); ++b)
    out = std::max(out, (ai->sigma[b] - bi->sigma[b]).cwiseAbs().maxCoeff());
  return out;
}

}  // namespace

TEST_CASE("grouped map: two-group configural recovers per-group truth") {
  auto b = build_mg(kTwoFactor, 2);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());

  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);

  // Group-1 and group-2 free loadings should recover the (identical) truth; the
  // factor variances differ across groups (1.0 vs 1.4).
  const auto locs = ev->param_locations();
  using magmaan::model::MatId;
  int checked_loadings = 0, checked_var = 0;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& loc = locs[k];
    if (loc.mat == MatId::Lambda) {
      const double truth = kLam[static_cast<std::size_t>(loc.row)];
      CHECK((*th)(static_cast<Eigen::Index>(k)) == doctest::Approx(truth).epsilon(1e-9));
      ++checked_loadings;
    } else if (loc.mat == MatId::Psi && loc.row == loc.col) {
      const double truth = (loc.block == 0) ? 1.0 : 1.4;
      CHECK((*th)(static_cast<Eigen::Index>(k)) == doctest::Approx(truth).epsilon(1e-9));
      ++checked_var;
    }
  }
  CHECK(checked_loadings == 8);  // 4 free loadings x 2 groups
  CHECK(checked_var == 4);       // 2 factor variances x 2 groups
}

TEST_CASE("grouped inference: Omega is block-diagonal across groups") {
  auto b = build_mg(kTwoFactor, 2);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);

  // Off-block-diagonal entries of Omega (group-1 param x group-2 param) are 0.
  double max_offblock = 0.0;
  const auto& bop = inf->block_of_param;
  for (Eigen::Index i = 0; i < inf->Omega.rows(); ++i)
    for (Eigen::Index j = 0; j < inf->Omega.cols(); ++j)
      if (bop[static_cast<std::size_t>(i)] != bop[static_cast<std::size_t>(j)])
        max_offblock = std::max(max_offblock, std::abs(inf->Omega(i, j)));
  CHECK(max_offblock < 1e-12);
  CHECK(inf->df == 2 * (21 - 13));  // per group: p*=21, q=13 free -> df 8 each
}

TEST_CASE("metric invariance: W ~ 0 when loadings are equal, projection is a no-op") {
  auto b = build_mg(kTwoFactor, 2, {GroupEqual::Loadings});
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};  // equal loadings
  samp.n_obs = {500, 600};

  for (auto which : {ef::NonIterativeEstimator::Guttman,
                     ef::NonIterativeEstimator::GuttmanGlsAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp, which);
    REQUIRE_OK(th);
    auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                     which, rf::Discrepancy::NTML);
    REQUIRE_OK(inf);
    auto con = rf::noniterative_constrained_fit(b.pt, *inf);
    REQUIRE_OK(con);

    CHECK(con->k == 4);  // 4 free loadings equated across 2 groups
    CHECK(con->W < 1e-8);
    CHECK((con->theta_tilde - con->theta_hat).cwiseAbs().maxCoeff() < 1e-8);
    CHECK(con->p_wald == doctest::Approx(1.0).epsilon(1e-6));
  }
}

TEST_CASE("metric invariance: W > 0 when loadings differ, projection equates them") {
  auto b = build_mg(kTwoFactor, 2, {GroupEqual::Loadings});
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  const std::array<double, 6> lam2 = {1.0, 1.05, 1.2, 1.0, 0.7, 1.3};  // x2 loading differs
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(lam2, 1.4, 0.2, 0.7)};
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);
  auto con = rf::noniterative_constrained_fit(b.pt, *inf);
  REQUIRE_OK(con);

  CHECK(con->W > 1.0);  // a real loading difference at N=500/600

  // The projected estimate satisfies the constraints R theta_tilde = c.
  auto eqc = magmaan::estimate::build_eq_constraints(b.pt);
  REQUIRE_OK(eqc);
  const double con_resid = (eqc->A_eq * con->theta_tilde - eqc->b_eq).cwiseAbs().maxCoeff();
  CHECK(con_resid < 1e-9);

  // Wald = min-distance duality: W == (theta_hat - theta_tilde)' Omega^-1 (...).
  const Eigen::VectorXd d = con->theta_hat - con->theta_tilde;
  const Eigen::MatrixXd Omega_pinv =
      con->Omega.completeOrthogonalDecomposition().pseudoInverse();
  const double W_quad = d.dot(Omega_pinv * d);
  CHECK(W_quad == doctest::Approx(con->W).epsilon(1e-6));

  // Omega_tilde loses exactly k dimensions.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(con->Omega_tilde);
  const double tol = 1e-9 * svd.singularValues()(0);
  int rank = 0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i)
    if (svd.singularValues()(i) > tol) ++rank;
  CHECK(rank == con->Omega_tilde.rows() - con->k);
}

TEST_CASE("estimator-side metric map recovers equal loading shape on exact population") {
  auto b = build_mg(kTwoFactor, 2, {GroupEqual::Loadings});
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
  samp.n_obs = {500, 600};

  for (auto which : {ef::NonIterativeEstimator::Guttman,
                     ef::NonIterativeEstimator::GuttmanGlsAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto fit = ef::fit_noniterative_cfa_metric(b.pt, b.rep, samp, which);
    REQUIRE_OK(fit);

    const auto locs = ev->param_locations();
    for (std::size_t k = 0; k < locs.size(); ++k) {
      const auto& loc = locs[k];
      if (loc.mat == magmaan::model::MatId::Lambda) {
        CHECK(fit->theta(static_cast<Eigen::Index>(k)) ==
              doctest::Approx(kLam[static_cast<std::size_t>(loc.row)]).epsilon(1e-8));
      }
    }
  }
}

TEST_CASE("estimator-side metric map is marker-chart invariant off the surface") {
  auto bx1 = build_mg(kTwoFactor, 2, {GroupEqual::Loadings});
  auto bx2 = build_mg(kTwoFactorMarkerX2, 2, {GroupEqual::Loadings});
  auto ev1 = ModelEvaluator::build(bx1.pt, bx1.rep);
  auto ev2 = ModelEvaluator::build(bx2.pt, bx2.rep);
  REQUIRE(ev1.has_value());
  REQUIRE(ev2.has_value());

  const std::array<double, 6> lam2 = {1.0, 0.95, 1.2, 1.0, 0.7, 1.05};
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(lam2, 1.4, 0.2, 0.7)};
  samp.n_obs = {500, 600};

  for (auto which : {ef::NonIterativeEstimator::Guttman,
                     ef::NonIterativeEstimator::GuttmanGlsAligned}) {
    INFO("estimator ordinal: " << static_cast<int>(which));
    auto fit1 = ef::fit_noniterative_cfa_metric(bx1.pt, bx1.rep, samp, which);
    auto fit2 = ef::fit_noniterative_cfa_metric(bx2.pt, bx2.rep, samp, which);
    REQUIRE_OK(fit1);
    REQUIRE_OK(fit2);
    CHECK(max_sigma_diff(*ev1, fit1->theta, *ev2, fit2->theta) < 1e-8);
  }
}

TEST_CASE("residual-restricted map with no constraints equals configural aligned fit") {
  auto b = build_mg(kTwoFactor, 1);
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5)};
  samp.n_obs = {500};

  auto cfg = ef::fit_noniterative_cfa(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  auto res = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  REQUIRE_OK(cfg);
  REQUIRE_OK(res);
  CHECK((cfg->theta - res->theta).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("residual-restricted map enforces residual variance equality") {
  auto b = build_mg(kOneFactorResidualTie, 1);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {of_cov(kLam4, 1.2, {0.40, 0.75, 0.55, 0.65})};
  samp.n_obs = {500};

  auto fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  REQUIRE_OK(fit);

  using magmaan::model::MatId;
  const auto locs = ev->param_locations();
  double x1 = std::numeric_limits<double>::quiet_NaN();
  double x2 = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& loc = locs[k];
    if (loc.mat == MatId::Theta && loc.row == loc.col && loc.row == 0)
      x1 = fit->theta(static_cast<Eigen::Index>(k));
    if (loc.mat == MatId::Theta && loc.row == loc.col && loc.row == 1)
      x2 = fit->theta(static_cast<Eigen::Index>(k));
  }
  REQUIRE(std::isfinite(x1));
  REQUIRE(std::isfinite(x2));
  CHECK(x1 == doctest::Approx(x2).epsilon(1e-10));

  auto eqc = magmaan::estimate::build_eq_constraints(b.pt);
  REQUIRE_OK(eqc);
  CHECK((eqc->A_eq * fit->theta - eqc->b_eq).cwiseAbs().maxCoeff() < 1e-9);
}

TEST_CASE("residual-restricted Jacobian is tangent to equality rows") {
  auto b = build_mg(kOneFactorResidualTie, 1);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {of_cov(kLam4, 1.2, {0.40, 0.75, 0.55, 0.65})};
  samp.n_obs = {500};

  auto J = ef::estimator_map_jacobian_restricted(
      b.pt, b.rep, *ev, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  REQUIRE_OK(J);
  auto eqc = magmaan::estimate::build_eq_constraints(b.pt);
  REQUIRE_OK(eqc);
  REQUIRE(eqc->A_eq.rows() > 0);
  CHECK((eqc->A_eq * (*J)).cwiseAbs().maxCoeff() < 1e-6);

  auto fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  REQUIRE_OK(fit);
  auto inf = rf::noniterative_inference_grouped_restricted_nt(
      b.pt, b.rep, samp, fit->theta,
      ef::NonIterativeEstimator::GuttmanGlsAligned, rf::Discrepancy::ULS);
  REQUIRE_OK(inf);
  CHECK(inf->df == 3);
  CHECK(inf->Omega.allFinite());
}

TEST_CASE("residual-restricted grouped inference carries cross-block covariance") {
  auto b = build_mg(kOneFactorResidualTie, 2, {GroupEqual::Residuals});
  SampleStats samp;
  samp.S = {of_cov(kLam4, 1.2, {0.40, 0.75, 0.55, 0.65}),
            of_cov(kLam4, 0.9, {0.70, 0.50, 0.60, 0.80})};
  samp.n_obs = {500, 700};

  auto fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  REQUIRE_OK(fit);
  auto inf = rf::noniterative_inference_grouped_restricted_nt(
      b.pt, b.rep, samp, fit->theta,
      ef::NonIterativeEstimator::GuttmanGlsAligned, rf::Discrepancy::ULS);
  REQUIRE_OK(inf);

  double max_offblock = 0.0;
  const auto& bop = inf->block_of_param;
  for (Eigen::Index i = 0; i < inf->Omega.rows(); ++i)
    for (Eigen::Index j = 0; j < inf->Omega.cols(); ++j)
      if (bop[static_cast<std::size_t>(i)] != bop[static_cast<std::size_t>(j)])
        max_offblock = std::max(max_offblock, std::abs(inf->Omega(i, j)));
  CHECK(max_offblock > 1e-12);
}

TEST_CASE("residual-restricted map rejects mixed residual-loading rows") {
  auto b = build_mg(
      "f =~ x1 + l2*x2 + x3\n"
      "x2 ~~ e2*x2\n"
      "l2 == e2\n",
      1);
  SampleStats samp;
  samp.S = {of_cov(std::array<double, 4>{1.0, 0.8, 1.1, 0.0}, 1.0,
                   std::array<double, 4>{0.5, 0.6, 0.7, 1.0})
                .topLeftCorner(3, 3)
                .eval()};
  samp.n_obs = {500};

  auto fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  CHECK_FALSE(fit.has_value());
  CHECK(fit.error().detail.find("mixes") != std::string::npos);
}

TEST_CASE("residual-restricted map rejects factor-variance rows") {
  auto b = build_mg(
      "f =~ x1 + x2 + x3\n"
      "f ~~ v*f\n"
      "v == 1\n",
      1);
  SampleStats samp;
  samp.S = {of_cov(std::array<double, 4>{1.0, 0.8, 1.1, 0.0}, 1.0,
                   std::array<double, 4>{0.5, 0.6, 0.7, 1.0})
                .topLeftCorner(3, 3)
                .eval()};
  samp.n_obs = {500};

  auto fit = ef::fit_noniterative_cfa_restricted(
      b.pt, b.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  CHECK_FALSE(fit.has_value());
  CHECK(fit.error().detail.find("factor") != std::string::npos);
}

TEST_CASE("residual-restricted map is marker-chart invariant off the surface") {
  auto bx1 = build_mg(kOneFactorResidualTieX1X3, 1);
  auto bx2 = build_mg(kOneFactorResidualTieX1X3MarkerX2, 1);
  auto ev1 = ModelEvaluator::build(bx1.pt, bx1.rep);
  auto ev2 = ModelEvaluator::build(bx2.pt, bx2.rep);
  REQUIRE(ev1.has_value());
  REQUIRE(ev2.has_value());

  SampleStats samp;
  samp.S = {of_cov(kLam4, 1.2, {0.40, 0.75, 0.55, 0.65})};
  samp.n_obs = {500};

  auto fit1 = ef::fit_noniterative_cfa_restricted(
      bx1.pt, bx1.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  auto fit2 = ef::fit_noniterative_cfa_restricted(
      bx2.pt, bx2.rep, samp, ef::NonIterativeEstimator::GuttmanGlsAligned);
  REQUIRE_OK(fit1);
  REQUIRE_OK(fit2);
  CHECK(max_sigma_diff(*ev1, fit1->theta, *ev2, fit2->theta) < 1e-8);
}

TEST_CASE("mean structure: configural intercepts recover nu_g = m_g exactly") {
  auto b = build_mg(kTwoFactor, 2, {}, /*means=*/true);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
  samp.mean = {mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
               mk_mean({1.5, 2.5, 3.5, 4.5, 5.5, 6.5})};
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);

  using magmaan::model::MatId;
  const auto locs = ev->param_locations();
  int checked_nu = 0, checked_load = 0;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& loc = locs[k];
    if (loc.mat == MatId::Nu) {
      const double truth = samp.mean[static_cast<std::size_t>(loc.block)](loc.row);
      CHECK((*th)(static_cast<Eigen::Index>(k)) == doctest::Approx(truth).epsilon(1e-12));
      ++checked_nu;
    } else if (loc.mat == MatId::Lambda) {
      CHECK((*th)(static_cast<Eigen::Index>(k)) ==
            doctest::Approx(kLam[static_cast<std::size_t>(loc.row)]).epsilon(1e-9));
      ++checked_load;
    }
  }
  CHECK(checked_nu == 12);    // 6 intercepts x 2 groups
  CHECK(checked_load == 8);   // covariance map unchanged by the mean structure
}

TEST_CASE("mean structure: intercept SE is the analytic saturated-mean SE") {
  auto b = build_mg(kTwoFactor, 2, {}, /*means=*/true);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
  samp.mean = {mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
               mk_mean({1.5, 2.5, 3.5, 4.5, 5.5, 6.5})};
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);

  // On the exact population Σ̂ = S, so SE(ν_i) = √(S_b(i,i) / N_b). The mean–vech
  // cross-block of Γ_NT is zero, so the ν rows of Ω have no covariance leakage.
  using magmaan::model::MatId;
  const auto locs = ev->param_locations();
  int checked = 0;
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& loc = locs[k];
    if (loc.mat != MatId::Nu) continue;
    const auto bb = static_cast<std::size_t>(loc.block);
    const double N_b = static_cast<double>(samp.n_obs[bb]);
    const double want = std::sqrt(samp.S[bb](loc.row, loc.row) / N_b);
    CHECK(inf->se(static_cast<Eigen::Index>(k)) == doctest::Approx(want).epsilon(1e-6));
    // No ν→cov cross-covariance under normal theory.
    for (std::size_t l = 0; l < locs.size(); ++l) {
      if (locs[l].mat == MatId::Nu) continue;
      CHECK(std::abs(inf->Omega(static_cast<Eigen::Index>(k),
                                static_cast<Eigen::Index>(l))) < 1e-10);
    }
    ++checked;
  }
  CHECK(checked == 12);
  // The covariance-only GOF df is unchanged by the mean augmentation.
  CHECK(inf->df == 2 * (21 - 13));
}

TEST_CASE("mean structure: intercept-equality projection (alpha = 0)") {
  // Equal loadings AND equal intercepts across groups. group.equal ties both;
  // latent means stay fixed at 0, so this is the exact-χ² intercept-equality
  // intermediate (not lavaan's free-α scalar, which is Phase C).
  auto b = build_mg(kTwoFactor, 2, {GroupEqual::Loadings, GroupEqual::Intercepts},
                    /*means=*/true);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());

  SUBCASE("W ~ 0 when means are equal across groups") {
    SampleStats samp;
    samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
    const auto mu = mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    samp.mean = {mu, mu};  // equal means -> equal saturated intercepts
    samp.n_obs = {500, 600};

    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
    REQUIRE_OK(th);
    auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                     ef::NonIterativeEstimator::Guttman,
                                                     rf::Discrepancy::NTML);
    REQUIRE_OK(inf);
    auto con = rf::noniterative_constrained_fit(b.pt, *inf);
    REQUIRE_OK(con);
    CHECK(con->k == 10);  // 4 loadings + 6 intercepts equated across 2 groups
    CHECK(con->W < 1e-6);
  }

  SUBCASE("W > 0 when a group mean differs; projection equates the intercepts") {
    SampleStats samp;
    samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
    samp.mean = {mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 mk_mean({1.8, 2.0, 3.0, 4.0, 5.0, 6.0})};  // x1 intercept differs
    samp.n_obs = {500, 600};

    auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
    REQUIRE_OK(th);
    auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                     ef::NonIterativeEstimator::Guttman,
                                                     rf::Discrepancy::NTML);
    REQUIRE_OK(inf);
    auto con = rf::noniterative_constrained_fit(b.pt, *inf);
    REQUIRE_OK(con);
    CHECK(con->W > 1.0);
    auto eqc = magmaan::estimate::build_eq_constraints(b.pt);
    REQUIRE_OK(eqc);
    const double resid = (eqc->A_eq * con->theta_tilde - eqc->b_eq).cwiseAbs().maxCoeff();
    CHECK(resid < 1e-9);
  }
}

TEST_CASE("mean structure: single-block path errors, directing to the grouped path") {
  auto b = build_mg(kTwoFactor, 1, {}, /*means=*/true);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5)};
  samp.mean = {mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0})};
  samp.n_obs = {500};
  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_nt(b.pt, b.rep, samp, *th,
                                           ef::NonIterativeEstimator::Guttman,
                                           rf::Discrepancy::NTML);
  CHECK_FALSE(inf.has_value());  // single-block SEs would omit the intercept block
}

TEST_CASE("scalar invariance: recovers latent means and W ~ 0 under true scalar") {
  // Equal loadings, equal intercepts ν; group-2 differs only in its latent mean
  // α_2, so m_2 = ν + Λ α_2 lies in ν + col(Λ) and the mean residual d_2 = 0.
  auto b = build_mg(kTwoFactor, 2, {}, /*means=*/true);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());

  const Eigen::MatrixXd L = tf_loadings(kLam);
  Eigen::VectorXd alpha2(2);
  alpha2 << 0.5, -0.3;
  const Eigen::VectorXd nu = mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.2, 0.25, 0.6)};
  samp.mean = {nu, nu + L * alpha2};  // m_1 = ν (α_1=0), m_2 = ν + Λ α_2
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);
  auto sc = rf::noniterative_scalar_invariance(b.pt, b.rep, *inf);
  REQUIRE_OK(sc);

  CHECK(sc->df == 1 * (6 - 2));               // (G-1)(p - #factors) = 4
  CHECK(sc->rank == sc->df);
  CHECK(sc->W < 1e-8);                        // d_2 = 0 exactly on the population
  CHECK(sc->groups.size() == 1);
  CHECK(sc->alpha.size() == 1);
  // Recovered α_2 = (Λ'Λ)⁻¹Λ'(m_2 − m_1) = α_2 exactly.
  CHECK((sc->alpha[0] - alpha2).cwiseAbs().maxCoeff() < 1e-9);
  // Common intercept ν = m_1 (reference).
  CHECK((sc->nu - nu).cwiseAbs().maxCoeff() < 1e-12);
  CHECK(sc->alpha_se[0].minCoeff() > 0.0);    // finite delta-method SEs
  CHECK(sc->p_value == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("scalar invariance: W > 0 when an intercept deviates off col(Lambda)") {
  auto b = build_mg(kTwoFactor, 2, {}, /*means=*/true);
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());

  const Eigen::MatrixXd L = tf_loadings(kLam);
  Eigen::VectorXd alpha2(2);
  alpha2 << 0.5, -0.3;
  const Eigen::VectorXd nu = mk_mean({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  // eps ⟂ col(Λ): dot with each loading column is 0, so d_2 = (I−P) eps = eps.
  Eigen::VectorXd eps(6);
  eps << 0.6, 0.0, -0.5, 0.0, 0.0, 0.0;  // 1*0.6 + 1.2*(-0.5) = 0 against col f1
  REQUIRE(std::abs((L.transpose() * eps).cwiseAbs().maxCoeff()) < 1e-12);

  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.2, 0.25, 0.6)};
  samp.mean = {nu, nu + L * alpha2 + eps};  // scalar violated in the ⟂ direction
  samp.n_obs = {500, 600};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);
  auto sc = rf::noniterative_scalar_invariance(b.pt, b.rep, *inf);
  REQUIRE_OK(sc);

  CHECK(sc->W > 20.0);  // a clear violation at N = 500/600
  CHECK((sc->d_stacked - eps).cwiseAbs().maxCoeff() < 1e-9);  // d_2 = eps
  CHECK(sc->p_value < 1e-3);
}

TEST_CASE("scalar invariance: errors on a covariance-only model") {
  auto b = build_mg(kTwoFactor, 2);  // no mean structure
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5), tf_cov(kLam, 1.4, 0.2, 0.7)};
  samp.n_obs = {500, 600};
  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);
  auto sc = rf::noniterative_scalar_invariance(b.pt, b.rep, *inf);
  CHECK_FALSE(sc.has_value());
}

TEST_CASE("grouped inference reduces to single-block inference at G=1") {
  auto b = build_mg(kTwoFactor, 1);  // single group
  auto ev = ModelEvaluator::build(b.pt, b.rep);
  REQUIRE(ev.has_value());
  SampleStats samp;
  samp.S = {tf_cov(kLam, 1.0, 0.3, 0.5)};
  samp.n_obs = {400};

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto single = rf::noniterative_inference_nt(b.pt, b.rep, samp, *th,
                                              ef::NonIterativeEstimator::Guttman,
                                              rf::Discrepancy::NTML);
  REQUIRE_OK(single);
  auto grouped = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                       ef::NonIterativeEstimator::Guttman,
                                                       rf::Discrepancy::NTML);
  REQUIRE_OK(grouped);

  CHECK((single->Omega - grouped->Omega).cwiseAbs().maxCoeff() < 1e-10);
  CHECK(single->T_gof == doctest::Approx(grouped->T_gof).epsilon(1e-9));
  CHECK(single->df == grouped->df);
  CHECK(single->p_mixture == doctest::Approx(grouped->p_mixture).epsilon(1e-6));
}
