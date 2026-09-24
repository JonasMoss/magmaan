#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string_view>

#include "magmaan/estimate/frontier/newton_adapters.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {
namespace nf = magmaan::estimate::frontier;
using namespace magmaan;
struct Model {
  spec::LatentStructure pt;
  model::MatrixRep rep;
};
Model model_for(std::string_view syntax, bool means = false) {
  auto parsed = parse::Parser::parse(syntax);
  REQUIRE(parsed.has_value());
  spec::BuildOptions opts;
  opts.fixed_x = false;
  opts.meanstructure = means;
  auto pt = spec::build(*parsed, opts);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}
data::SampleStats sample3() {
  data::SampleStats s;
  Eigen::Matrix3d S;
  S << 1.2, .4, .3, .4, 1.1, .2, .3, .2, 1.3;
  s.S = {S}; s.n_obs = {400};
  return s;
}
void check_artifacts(const nf::NewtonAudit& a) {
  INFO(a.derivatives.detail);
  REQUIRE(a.derivatives.status == estimate::NewtonAccuracyStatus::Available);
  REQUIRE(a.geometry.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(a.derivatives.hessian.allFinite());
  CHECK(a.derivatives.hessian.isApprox(a.derivatives.hessian.transpose(), 1e-10));
  const Eigen::MatrixXd B = a.geometry.equality_basis * a.geometry.tangent_basis;
  CHECK(a.geometry.reduced_hessian.isApprox(B.transpose() *
      (a.derivatives.hessian + a.geometry.curvature_correction) * B, 1e-10));
  CHECK((a.geometry.reduced_gradient - B.transpose() * a.derivatives.gradient).norm() < 1e-10);
}
} // namespace

TEST_CASE("Newton adapters: numerical curvature retains scale and rejects unreliable probes") {
  optim::ScalarProblem p;
  p.n_param = 2;
  Eigen::Matrix2d H; H << 4, 1, 1, 3;
  p.f = [H](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    g = H * x; return 0.5 * x.dot(g);
  };
  const Eigen::Vector2d x(.2, -.1);
  auto d = nf::evaluate_newton_objective(p, x, 40, 40);
  REQUIRE(d.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(d.hessian.isApprox(40 * H, 1e-10));
  CHECK(d.gradient.isApprox(40 * H * x, 1e-10));
  CHECK(d.objective == doctest::Approx(.5 * x.dot(H * x)));
  CHECK(d.native_to_total == 40);
  CHECK(d.difference_steps.size() == 2);
  auto total = nf::evaluate_newton_objective(p, x, 40, 1);
  CHECK(total.hessian.isApprox(H, 1e-10));
  p.n_param = 1;
  p.f = [](const Eigen::VectorXd& t, Eigen::VectorXd& g) {
    g = Eigen::VectorXd::Constant(1, 4 * std::pow(t(0), 3));
    return std::pow(t(0), 4);
  };
  nf::NewtonDifferenceOptions opts;
  opts.relative_step = .2; opts.max_relative_error = 1e-8;
  auto bad = nf::evaluate_newton_objective(p, Eigen::VectorXd::Ones(1), 1, 1,
      nf::NewtonObjectiveKind::Supplied, opts);
  CHECK(bad.status == estimate::NewtonAccuracyStatus::Unavailable);
  CHECK(bad.hessian.rows() == 1);
  CHECK(bad.hessian_relative_error > opts.max_relative_error);
  opts.parameter_scales = Eigen::VectorXd::Ones(2);
  CHECK(nf::evaluate_newton_objective(p, Eigen::VectorXd::Ones(1), 1, 1,
      nf::NewtonObjectiveKind::Supplied, opts).status == estimate::NewtonAccuracyStatus::Unavailable);
}

TEST_CASE("Newton adapters: ULS GLS WLS and expanded SNLLS use the original objective") {
  auto m = model_for("x1 ~~ x1 + x2 + x3\nx2 ~~ x2 + x3\nx3 ~~ x3");
  auto s = sample3();
  auto start = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(start.has_value());
  auto fitted = estimate::fit_gmm(m.pt, m.rep, s, *start, {});
  REQUIRE(fitted.has_value());
  const auto& theta = fitted->theta;
  auto uls = nf::audit_newton_uls(m.pt, m.rep, s, theta);
  REQUIRE(uls.has_value()); check_artifacts(*uls);
  CHECK(uls->diagnostics.passed);
  CHECK(uls->derivatives.objective == doctest::Approx(fitted->fmin).epsilon(1e-8));
  auto ev = model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto weight = estimate::gmm::normal_theory_weight(*ev, s, theta);
  REQUIRE(weight.has_value());
  auto gls = nf::audit_newton_gls(m.pt, m.rep, s, theta);
  auto wls = nf::audit_newton_wls(m.pt, m.rep, s, theta, *weight);
  REQUIRE(gls.has_value()); REQUIRE(wls.has_value());
  check_artifacts(*gls); check_artifacts(*wls);
  CHECK(gls->derivatives.hessian.isApprox(wls->derivatives.hessian));
  auto snlls = nf::audit_newton_snlls(m.pt, m.rep, s, theta, *weight);
  REQUIRE(snlls.has_value());
  CHECK(snlls->derivatives.hessian.isApprox(wls->derivatives.hessian));
  nf::NewtonAdapterOptions gn;
  gn.gauss_newton = true;
  auto approx = nf::audit_newton_gmm(m.pt, m.rep, s, theta, *weight, gn);
  REQUIRE(approx.has_value()); check_artifacts(*approx);
  CHECK(approx->derivatives.curvature_kind == nf::NewtonCurvatureKind::GaussNewton);
  // Saturated covariance model is linear, so GN equals the exact Hessian.
  CHECK(approx->derivatives.hessian.isApprox(wls->derivatives.hessian, 1e-8));
  CHECK_FALSE(nf::audit_newton_wls(m.pt, m.rep, s, theta, {}).has_value());
  gn.bounds.lower = theta;
  gn.bounds.upper = Eigen::VectorXd::Constant(theta.size(), std::numeric_limits<double>::infinity());
  auto bounded = nf::audit_newton_gmm(m.pt, m.rep, s, theta, {}, gn);
  REQUIRE(bounded.has_value());
  CHECK(bounded->diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(bounded->diagnostics.box_constrained);
  CHECK(bounded->derivatives.status == estimate::NewtonAccuracyStatus::Available);
}

TEST_CASE("Newton adapters: FIML retains analytic observed information and penalty curvature") {
  auto m = model_for("x1 ~~ x1 + x2\nx2 ~~ x2", true);
  data::RawData raw;
  Eigen::MatrixXd X(12, 2);
  X << -1, -.4, -.6, .5, -.2, -.7, .4, -.1, .8, .7, 1.1, .2,
       -.8, .9, -.4, -.8, .1, .4, .6, -.5, .9, 1.2, 1.3, -.2;
  raw.X = {X};
  raw.mask = {Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(12, 2)};
  raw.mask[0](2, 1) = 0; raw.X[0](2, 1) = std::numeric_limits<double>::quiet_NaN();
  auto pack = estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto start = estimate::simple_start_values(m.pt, m.rep, pack->start_stats, {});
  REQUIRE(start.has_value());
  auto a = nf::audit_newton_fiml(m.pt, m.rep, raw, *pack, *start);
  REQUIRE(a.has_value()); check_artifacts(*a);
  estimate::Estimates at; at.theta = *start;
  auto info = estimate::fiml::fiml_observed_information(m.pt, m.rep, raw, at, *pack);
  REQUIRE(info.has_value());
  CHECK(a->derivatives.hessian.isApprox(*info));
  CHECK(a->derivatives.curvature_kind == nf::NewtonCurvatureKind::AnalyticObserved);
  nf::MultiInfoPenaltyOptions zero; zero.weight = 0;
  auto penalized = nf::audit_newton_penalized_fiml(m.pt, m.rep, raw, *pack, *start, zero);
  REQUIRE(penalized.has_value()); check_artifacts(*penalized);
  CHECK(penalized->derivatives.hessian.isApprox(a->derivatives.hessian, 1e-6));
  CHECK(penalized->derivatives.gradient.isApprox(a->derivatives.gradient, 1e-9));
  CHECK(penalized->derivatives.objective == doctest::Approx(a->derivatives.objective));
}

TEST_CASE("Newton adapters: penalized ML differentiates the actual penalized objective") {
  auto m = model_for("f =~ x1 + x2 + x3");
  auto s = sample3();
  auto start = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(start.has_value());
  auto base = nf::audit_newton_ml(m.pt, m.rep, s, *start);
  nf::MultiInfoPenaltyOptions zero; zero.weight = 0;
  auto z = nf::audit_newton_penalized_ml(m.pt, m.rep, s, *start, zero);
  REQUIRE(z.has_value()); check_artifacts(*z);
  CHECK(z->derivatives.hessian.isApprox(base.derivatives.hessian, 1e-5));
  nf::MultiInfoPenaltyOptions positive; positive.weight = 2;
  auto a = nf::audit_newton_penalized_ml(m.pt, m.rep, s, *start, positive);
  REQUIRE(a.has_value()); check_artifacts(*a);
  CHECK((a->derivatives.hessian - z->derivatives.hessian).norm() > 1e-4);
  auto ev = model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto layout = nf::multiinfo_penalty_layout(*ev, *start);
  REQUIRE(layout.has_value());
  auto penalty = nf::multiinfo_penalty(*layout, *ev, *start, true);
  REQUIRE(penalty.has_value());
  CHECK((a->derivatives.gradient - z->derivatives.gradient + 2 * penalty->gradient).norm() < 1e-8);
  CHECK(a->derivatives.objective == doctest::Approx(base.derivatives.objective - 2 * penalty->value / 400));
}

TEST_CASE("Newton adapters: ordinal LS and CatML distinguish thresholds and curvature") {
  auto m = model_for("f =~ x1 + x2 + x3\nx1 | t11 + t12\nx2 | t21 + t22\nx3 | t31 + t32");
  data::OrdinalStats s;
  Eigen::Matrix3d R; R << 1, .56, .48, .56, 1, .336, .48, .336, 1;
  s.R = {R}; s.n_obs = {400}; s.n_levels = {{3, 3, 3}};
  Eigen::VectorXd t(6); t << -.5, .5, -.4, .6, -.6, .4;
  s.thresholds = {t}; s.threshold_ov = {{0, 0, 1, 1, 2, 2}};
  s.threshold_level = {{1, 2, 1, 2, 1, 2}};
  s.NACOV = {Eigen::MatrixXd::Identity(9, 9)};
  s.W_dwls = s.NACOV; s.W_wls = s.NACOV;
  s.ov_names = {{"x1", "x2", "x3"}};
  auto start = estimate::ordinal_start_values(m.pt, m.rep, s, {});
  REQUIRE(start.has_value());
  auto ls = nf::audit_newton_ordinal(m.pt, m.rep, s, *start);
  REQUIRE(ls.has_value()); check_artifacts(*ls);
  CHECK(ls->derivatives.fixed_coordinates.empty());
  auto fit = nf::fit_catml(m.pt, m.rep, s, *start);
  REQUIRE(fit.has_value());
  auto cat = nf::audit_newton_catml(m.pt, m.rep, s, fit->theta);
  REQUIRE(cat.has_value()); check_artifacts(*cat);
  CHECK(cat->derivatives.objective == doctest::Approx(fit->fmin).epsilon(1e-8));
  CHECK(cat->derivatives.fixed_coordinates.size() == 6);
  CHECK(cat->diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  const auto& K = cat->geometry.equality_basis;
  for (auto k : cat->derivatives.fixed_coordinates) CHECK(K.row(k).norm() < 1e-12);
}

TEST_CASE("Newton adapters: two-level curvature uses total likelihood normalization") {
  auto m = model_for("level: 1\nx ~~ x\nlevel: 2\nx ~~ x\nx ~ 1", true);
  data::ClusterSampleStats s;
  data::ClusterGroupStats g;
  g.n_within = 40; g.n_clusters = 10; g.p_within = g.p_between = 1;
  g.grand_mean = Eigen::VectorXd::Zero(1);
  g.within_scatter = Eigen::MatrixXd::Constant(1, 1, 30);
  data::ClusterSizePattern pattern;
  pattern.cluster_size = 4; pattern.n_clusters = 10;
  pattern.sum_cluster_mean = Eigen::VectorXd::Zero(1);
  pattern.sum_cluster_mean_cp = Eigen::MatrixXd::Constant(1, 1, 7.5);
  g.size_patterns = {pattern}; s.groups = {g};
  auto start = estimate::twolevel::twolevel_start_values(m.pt, m.rep, s);
  REQUIRE(start.has_value());
  auto a = nf::audit_newton_twolevel(m.pt, m.rep, s, *start);
  REQUIRE(a.has_value()); check_artifacts(*a);
  auto ev = model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto info = estimate::twolevel::twolevel_information(*ev, s, *start, false);
  REQUIRE(info.has_value());
  CHECK(a->derivatives.hessian.isApprox(*info, 1e-6));
  CHECK(a->derivatives.native_to_total == 1);
  CHECK(a->derivatives.n_obs == 40);
  auto problem = estimate::twolevel::twolevel_ml_objective(*ev, s);
  REQUIRE(problem.has_value());
  Eigen::VectorXd gradient;
  CHECK(a->derivatives.objective * 40 == doctest::Approx(problem->f(*start, gradient)));
  CHECK((a->derivatives.gradient - gradient).norm() < 1e-10);
}

TEST_CASE("Newton adapters: mixed ordinal keeps the complete moment objective") {
  auto m = model_for("f =~ x1 + x2 + x3 + x4\nx1 | t11 + t12\nx2 | t21 + t22\nx1 ~*~ 1*x1\nx2 ~*~ 1*x2", true);
  Eigen::MatrixXd X(240, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double t = static_cast<double>(i + 1);
    const double eta = std::sin(.17 * t) + std::cos(.31 * t);
    const double y1 = .7 * eta + std::sin(1.31 * t);
    const double y2 = .8 * eta + std::cos(1.71 * t);
    X(i, 0) = 1.0 + (y1 > -.5) + (y1 > .5);
    X(i, 1) = 1.0 + (y2 > -.6) + (y2 > .4);
    X(i, 2) = .6 * eta + std::sin(2.11 * t);
    X(i, 3) = .7 * eta + std::cos(2.71 * t);
  }
  auto s = data::mixed_ordinal_stats_from_data({X}, {{1, 1, 0, 0}});
  REQUIRE(s.has_value());
  auto start = estimate::mixed_ordinal_start_values(m.pt, m.rep, *s, {});
  REQUIRE(start.has_value());
  for (auto parameterization : {estimate::OrdinalParameterization::Delta,
                               estimate::OrdinalParameterization::Theta}) {
    auto a = nf::audit_newton_mixed_ordinal(m.pt, m.rep, *s, *start,
        estimate::OrdinalWeightKind::DWLS, parameterization);
    REQUIRE(a.has_value()); check_artifacts(*a);
    estimate::Estimates at; at.theta = *start;
    auto original = nf::mixed_ordinal_ls_objective(m.pt, m.rep, *s, at,
        estimate::OrdinalWeightKind::DWLS, parameterization);
    REQUIRE(original.has_value());
    auto prob = optim::scalarize(original->problem);
    Eigen::VectorXd gradient;
    CHECK(a->derivatives.objective == doctest::Approx(prob.f(*start, gradient)));
    CHECK((a->derivatives.gradient - 240 * gradient).norm() < 1e-8);
  }
}

TEST_CASE("Newton adapters: nonlinear constraints do not silently receive an unconstrained certificate") {
  auto m = model_for("f =~ x1 + a*x2 + b*x3\na == b^2");
  auto s = sample3();
  auto start = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(start.has_value());
  auto a = nf::audit_newton_uls(m.pt, m.rep, s, *start);
  REQUIRE(a.has_value());
  CHECK(a->derivatives.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(a->diagnostics.status == estimate::NewtonAccuracyStatus::Unsupported);
  CHECK_FALSE(a->diagnostics.passed);
}

TEST_CASE("Newton adapters: a profiled LS fit is audited in full coordinates") {
  auto m = model_for("f =~ x1 + x2 + x3");
  auto s = sample3();
  auto start = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(start.has_value());
  auto fit = estimate::fit_snlls(m.pt, m.rep, s, *start);
  REQUIRE(fit.has_value());
  auto a = nf::audit_newton_snlls(m.pt, m.rep, s, fit->theta);
  REQUIRE(a.has_value()); check_artifacts(*a);
  CHECK(a->derivatives.hessian.rows() == m.pt.n_free());
  CHECK(a->derivatives.objective == doctest::Approx(fit->fmin).epsilon(1e-8));
  CHECK((a->derivatives.gradient - 400 * a->derivatives.whitened_jacobian.transpose() *
      a->derivatives.whitened_residual).norm() < 1e-10);
  CHECK(a->diagnostics.passed);
}

TEST_CASE("Newton adapters: fitted weight is frozen at the audit point") {
  auto m = model_for("x1 ~~ x1 + x2 + x3\nx2 ~~ x2 + x3\nx3 ~~ x3");
  auto s = sample3();
  auto theta = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(theta.has_value());
  auto ev = model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto w = estimate::gmm::expected_information_weight(*ev, s, *theta);
  REQUIRE(w.has_value());
  auto a = nf::audit_newton_gmm_fitted_weight(m.pt, m.rep, s, *theta);
  auto direct = nf::audit_newton_gmm(m.pt, m.rep, s, *theta, *w);
  REQUIRE(a.has_value()); REQUIRE(direct.has_value());
  check_artifacts(*a);
  CHECK(a->derivatives.hessian.isApprox(direct->derivatives.hessian, 1e-12));
  // The saturated model is linear: the frozen-weight Hessian is exactly J'J
  // even away from the optimum. Differentiating the weight would violate this.
  CHECK(a->derivatives.hessian.isApprox(400 *
      a->derivatives.whitened_jacobian.transpose() * a->derivatives.whitened_jacobian, 1e-8));
  CHECK(a->derivatives.gradient.norm() > 1);
  CHECK_FALSE(nf::audit_newton_gmm_fitted_weight(m.pt, m.rep, s, *theta,
      static_cast<nf::GmmFittedWeightKind>(99)).has_value());
}

TEST_CASE("Newton adapters: ML2S reuses moments for all five Stage-2 policies") {
  auto m = model_for("x1 ~~ x1 + x2\nx2 ~~ x2", true);
  data::SampleStats s;
  Eigen::Matrix2d S; S << 1.2, .3, .3, 1.1;
  s.S = {S}; s.mean = {Eigen::Vector2d(.2, -.1)}; s.n_obs = {200};
  estimate::fiml::SaturatedMoments sm;
  sm.cov = s.S; sm.mean = s.mean; sm.n_obs = s.n_obs;
  sm.acov = Eigen::MatrixXd::Identity(5, 5) / 200;
  auto theta = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(theta.has_value());
  using Kind = estimate::fiml::TwoStageWeight;
  for (auto kind : {Kind::Nt, Kind::Uls, Kind::Dwls, Kind::Adf, Kind::Dls}) {
    CAPTURE(static_cast<int>(kind));
    auto a = nf::audit_newton_ml2s(m.pt, m.rep, sm, *theta, kind);
    REQUIRE(a.has_value()); check_artifacts(*a);
    if (kind == Kind::Nt) {
      auto direct = nf::audit_newton_ml(m.pt, m.rep, s, *theta);
      CHECK(a->derivatives.hessian.isApprox(direct.derivatives.hessian));
      CHECK(a->derivatives.objective == doctest::Approx(direct.derivatives.objective));
      CHECK(a->derivatives.curvature_kind == nf::NewtonCurvatureKind::AnalyticObserved);
    } else {
      auto w = estimate::fiml::two_stage_stage2_weight_structured(sm, kind);
      REQUIRE(w.has_value());
      auto direct = nf::audit_newton_gmm(m.pt, m.rep, s, *theta, *w);
      REQUIRE(direct.has_value());
      CHECK(a->derivatives.hessian.isApprox(direct->derivatives.hessian));
      CHECK(a->derivatives.objective == doctest::Approx(direct->derivatives.objective));
      CHECK(a->derivatives.hessian.isApprox(200 *
          a->derivatives.whitened_jacobian.transpose() * a->derivatives.whitened_jacobian, 1e-8));
    }
  }
  nf::NewtonAdapterOptions gn; gn.gauss_newton = true;
  CHECK_FALSE(nf::audit_newton_ml2s(m.pt, m.rep, sm, *theta, Kind::Nt, {}, gn).has_value());
  estimate::fiml::TwoStageDlsOptions bad; bad.a = 2;
  CHECK_FALSE(nf::audit_newton_ml2s(m.pt, m.rep, sm, *theta, Kind::Dls, bad).has_value());
  sm.mean.clear();
  CHECK_FALSE(nf::audit_newton_ml2s(m.pt, m.rep, sm, *theta).has_value());
}

TEST_CASE("Newton adapters: unequal groups preserve total LS curvature") {
  auto parsed = parse::Parser::parse("x1 ~~ x1 + x2\nx2 ~~ x2");
  REQUIRE(parsed.has_value());
  spec::BuildOptions opts; opts.fixed_x = false; opts.n_groups = 2;
  auto pt = spec::build(*parsed, opts);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  data::SampleStats s;
  Eigen::Matrix2d S; S << 1.2, .3, .3, 1.1;
  s.S = {S, 2 * S}; s.n_obs = {100, 300};
  auto theta = estimate::simple_start_values(*pt, *rep, s, {});
  REQUIRE(theta.has_value());
  auto a = nf::audit_newton_uls(*pt, *rep, s, *theta);
  REQUIRE(a.has_value()); check_artifacts(*a);
  const Eigen::MatrixXd H = 400 * a->derivatives.whitened_jacobian.transpose() *
      a->derivatives.whitened_jacobian;
  CHECK(a->derivatives.hessian.isApprox(H, 1e-8));
  CHECK(a->derivatives.gradient.isApprox(400 *
      a->derivatives.whitened_jacobian.transpose() * a->derivatives.whitened_residual, 1e-8));
  // Each free covariance appears once in vech, with its own group's n.
  for (std::size_t r = 0; r < pt->op.size(); ++r)
    if (pt->free[r] > 0)
      CHECK(H(pt->free[r] - 1, pt->free[r] - 1) ==
          doctest::Approx(s.n_obs[static_cast<std::size_t>(pt->group[r] - 1)]));
}
