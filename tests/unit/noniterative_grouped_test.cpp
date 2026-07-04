#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
               std::vector<GroupEqual> ge = {}) {
  BuildOptions opts;
  opts.n_groups = ngroups;
  opts.group_equal = std::move(ge);
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr)};
}

// Exact 2-factor covariance: markers x1,x4; loadings `lam` (incl. markers),
// factor variance `pd`, factor covariance `po`, common residual `th`.
Eigen::MatrixXd tf_cov(const std::array<double, 6>& lam, double pd, double po,
                       double th) {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L(0, 0) = lam[0]; L(1, 0) = lam[1]; L(2, 0) = lam[2];
  L(3, 1) = lam[3]; L(4, 1) = lam[4]; L(5, 1) = lam[5];
  Eigen::MatrixXd Psi(2, 2);
  Psi << pd, po, po, pd;
  Eigen::MatrixXd S = L * Psi * L.transpose();
  for (Eigen::Index i = 0; i < 6; ++i) S(i, i) += th;
  return S;
}

constexpr const char* kTwoFactor =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ x4 + x5 + x6\n"
    "f1 ~~ f2\n";

const std::array<double, 6> kLam = {1.0, 0.8, 1.2, 1.0, 0.7, 1.3};

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

  auto th = ef::noniterative_cfa_theta(b.pt, b.rep, *ev, samp);
  REQUIRE_OK(th);
  auto inf = rf::noniterative_inference_grouped_nt(b.pt, b.rep, samp, *th,
                                                   ef::NonIterativeEstimator::Guttman,
                                                   rf::Discrepancy::NTML);
  REQUIRE_OK(inf);
  auto con = rf::noniterative_constrained_fit(b.pt, *inf);
  REQUIRE_OK(con);

  CHECK(con->k == 4);  // 4 free loadings equated across 2 groups
  CHECK(con->W < 1e-8);
  CHECK((con->theta_tilde - con->theta_hat).cwiseAbs().maxCoeff() < 1e-8);
  CHECK(con->p_wald == doctest::Approx(1.0).epsilon(1e-6));
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
