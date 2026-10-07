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
  auto fit = nf::fit_ml(m.pt, m.rep, s, *start);
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
    const auto& d = a->derivatives;
    REQUIRE(d.metric_factor.cols() == start->size());
    CHECK((d.metric_factor - std::sqrt(240.0) * d.whitened_jacobian).norm() < 1e-10);
    CHECK((d.metric_factor.transpose() * d.metric_score_residual - d.gradient).norm() < 1e-8);
    CHECK((d.hessian - d.metric - d.ls_curvature_correction).norm() < 1e-10);
    auto fd = nf::evaluate_newton_objective(prob, *start, 240, 240);
    CHECK((d.hessian - fd.hessian).norm() < 1e-6 * (1 + fd.hessian.norm()));
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

namespace {
Model model_with(std::string_view syntax, bool means, bool std_lv, int groups = 1) {
  auto parsed = parse::Parser::parse(syntax);
  REQUIRE(parsed.has_value());
  spec::BuildOptions opts;
  opts.fixed_x = false;
  opts.meanstructure = means;
  opts.std_lv = std_lv;
  opts.n_groups = groups;
  auto pt = spec::build(*parsed, opts);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}
// A positive-definite sample with means, reproducible across platforms.
data::SampleStats pd_sample(Eigen::Index p, int groups, double shift) {
  data::SampleStats s;
  for (int g = 0; g < groups; ++g) {
    Eigen::MatrixXd A(p, p);
    for (Eigen::Index i = 0; i < p; ++i)
      for (Eigen::Index j = 0; j < p; ++j)
        A(i, j) = std::sin(1.3 * static_cast<double>(i + 1) + 0.7 * static_cast<double>(j + 1) + shift * (g + 1));
    Eigen::MatrixXd S = A * A.transpose() / static_cast<double>(p) +
        Eigen::MatrixXd::Identity(p, p);
    Eigen::VectorXd m(p);
    for (Eigen::Index i = 0; i < p; ++i) m(i) = 0.1 * std::cos(static_cast<double>(i) + g);
    s.S.push_back(S);
    s.mean.push_back(m);
    s.n_obs.push_back(150 + 100 * g);
  }
  return s;
}
// Exact Hessian against central differences of the analytic gradient, at a
// point away from the optimum so the residual term matters.
void check_moment_quadratic_hessian(const Model& m, const data::SampleStats& s,
                                    const estimate::gmm::Weight& w) {
  auto theta = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(theta.has_value());
  Eigen::VectorXd x = *theta;
  for (Eigen::Index k = 0; k < x.size(); ++k) x(k) += 0.05 * std::sin(3.0 * static_cast<double>(k) + 1.0);
  auto ev = model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto exact = nf::evaluate_newton_moment_quadratic(*ev, s, x, w);
  INFO(exact.detail);
  REQUIRE(exact.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(exact.curvature_kind == nf::NewtonCurvatureKind::AnalyticObserved);
  CHECK(exact.metric_kind == nf::NewtonMetricKind::Sandwich);
  CHECK((exact.metric_factor.transpose() * exact.metric_factor - exact.metric).norm() <=
      1e-12 * (1 + exact.metric.norm()));
  CHECK((exact.metric_factor.transpose() * exact.metric_score_residual - exact.gradient).norm() <=
      1e-12 * (1 + exact.gradient.norm()));
  auto problem = estimate::gmm::residuals(*ev, s, x, w);
  REQUIRE(problem.has_value());
  double n = 0; for (auto nb : s.n_obs) n += static_cast<double>(nb);
  auto fd = nf::evaluate_newton_objective(optim::scalarize(*problem), x, n, n);
  REQUIRE(fd.status == estimate::NewtonAccuracyStatus::Available);
  CHECK((exact.gradient - fd.gradient).norm() <= 1e-10 * (1 + fd.gradient.norm()));
  CHECK((exact.hessian - fd.hessian).norm() <= 1e-6 * (1 + fd.hessian.norm()));
  // The Gauss-Newton part alone differs here: the residual term is not zero.
  const Eigen::MatrixXd gn = n * exact.whitened_jacobian.transpose() * exact.whitened_jacobian;
  CHECK((exact.hessian - gn - exact.ls_curvature_correction).norm() <=
      1e-12 * (1 + exact.hessian.norm()));
  CHECK(exact.ls_curvature_correction.isApprox(exact.ls_curvature_correction.transpose(), 1e-12));
  CHECK((exact.hessian - gn).norm() > 1e-4 * gn.norm());
}
} // namespace

TEST_CASE("Newton adapters: the analytic moment-quadratic Hessian matches gradient differences") {
  const auto cfa = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6", true, false);
  const auto sem = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf3 =~ x7 + x8 + x9\n"
                              "f2 ~ f1\nf3 ~ f1 + f2", false, false);
  // Nonrecursive: f1 and f2 regress on each other, identified by f3 and x7.
  const auto nonrec = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\n"
                                 "f1 ~ f2 + x7\nf2 ~ f1 + x8", false, false);
  const auto groups = model_with("f =~ x1 + x2 + x3 + x4", true, true, 2);
  const auto s6 = pd_sample(6, 1, 0.3);
  const auto s9 = pd_sample(9, 1, 0.5);
  const auto s8 = pd_sample(8, 1, 0.9);
  const auto s4 = pd_sample(4, 2, 0.2);
  for (const auto* c : {&cfa, &sem, &nonrec, &groups}) {
    const auto& s = c == &cfa ? s6 : c == &sem ? s9 : c == &nonrec ? s8 : s4;
    auto ev = model::ModelEvaluator::build(c->pt, c->rep);
    REQUIRE(ev.has_value());
    auto theta = estimate::simple_start_values(c->pt, c->rep, s, {});
    REQUIRE(theta.has_value());
    auto gls = estimate::gmm::normal_theory_weight(*ev, s, *theta);
    REQUIRE(gls.has_value());
    estimate::gmm::Weight dwls;
    for (const auto& b : *gls) dwls.push_back(estimate::gmm::BlockWeight::diagonal(b.to_dense().diagonal()));
    check_moment_quadratic_hessian(*c, s, {});
    check_moment_quadratic_hessian(*c, s, *gls);
    check_moment_quadratic_hessian(*c, s, dwls);
  }
}

TEST_CASE("Newton adapters: the least-squares metric is the normal-theory gradient variance") {
  const auto m = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6", true, false);
  const auto s = pd_sample(6, 1, 0.3);
  auto ev = model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto theta = estimate::simple_start_values(m.pt, m.rep, s, {});
  REQUIRE(theta.has_value());
  auto gls = estimate::gmm::normal_theory_weight(*ev, s, *theta);
  REQUIRE(gls.has_value());
  // GLS: W = Gamma_NT^{-1}, so the gradient variance is N J~'J~.
  auto d = nf::evaluate_newton_moment_quadratic(*ev, s, *theta, *gls);
  REQUIRE(d.status == estimate::NewtonAccuracyStatus::Available);
  const Eigen::MatrixXd gn = 150.0 * d.whitened_jacobian.transpose() * d.whitened_jacobian;
  CHECK(d.metric.isApprox(gn, 1e-8));
  // ULS at its optimum: the distance is sqrt(G' Omega^{-1} G), free of units.
  auto fit = estimate::fit_gmm(m.pt, m.rep, s, *theta, {});
  REQUIRE(fit.has_value());
  auto a = nf::audit_newton_uls(m.pt, m.rep, s, fit->theta);
  REQUIRE(a.has_value()); check_artifacts(*a);
  CHECK(a->diagnostics.metric == estimate::NewtonMetricKind::Sandwich);
  const Eigen::VectorXd& G = a->geometry.reduced_gradient;
  const double expect = std::sqrt(G.dot(a->geometry.reduced_metric.ldlt().solve(G)));
  CHECK(a->diagnostics.distance == doctest::Approx(expect).epsilon(1e-6));
  CHECK(a->diagnostics.passed);
  // Units: multiplying every variable by 10 leaves d unchanged under the
  // sandwich metric and multiplies it by 100 under the ULS Hessian metric.
  const auto m2 = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6", false, false);
  data::SampleStats s2 = s;
  s2.mean.clear();
  auto fit2 = estimate::fit_gmm(m2.pt, m2.rep, s2, *estimate::simple_start_values(m2.pt, m2.rep, s2, {}), {});
  REQUIRE(fit2.has_value());
  Eigen::VectorXd x = fit2->theta;
  for (Eigen::Index k = 0; k < x.size(); ++k) x(k) += 1e-3 * std::cos(static_cast<double>(k));
  data::SampleStats scaled = s2;
  scaled.S[0] *= 100.0;
  Eigen::VectorXd xs = x;
  for (std::size_t r = 0; r < m2.pt.size(); ++r)
    if (m2.pt.free[r] > 0 && m2.pt.op[r] == parse::Op::Covariance) xs(m2.pt.free[r] - 1) *= 100.0;
  auto base = nf::audit_newton_uls(m2.pt, m2.rep, s2, x);
  auto moved = nf::audit_newton_uls(m2.pt, m2.rep, scaled, xs);
  REQUIRE(base.has_value()); REQUIRE(moved.has_value());
  REQUIRE(base->diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  REQUIRE(moved->diagnostics.status == estimate::NewtonAccuracyStatus::Available);
  CHECK(moved->diagnostics.distance == doctest::Approx(base->diagnostics.distance).epsilon(1e-8));
  nf::NewtonAdapterOptions hess;
  hess.gauss_newton = true;  // objective-Hessian metric
  auto hb = nf::audit_newton_uls(m2.pt, m2.rep, s2, x, hess);
  auto hm = nf::audit_newton_uls(m2.pt, m2.rep, scaled, xs, hess);
  REQUIRE(hb.has_value()); REQUIRE(hm.has_value());
  CHECK(hm->diagnostics.distance == doctest::Approx(100 * hb->diagnostics.distance).epsilon(1e-8));
}

TEST_CASE("Newton adapters: the exact LS Hessian rejects a saddle that Gauss-Newton cannot see") {
  // One factor in std.lv: zero loadings with residual variances at diag(S)
  // make the ULS gradient vanish, but the objective decreases along any
  // loading direction aligned with the sample covariances.
  const auto m = model_with("f =~ x1 + x2 + x3", false, true);
  auto s = sample3();
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(m.pt.n_free());
  for (std::size_t r = 0; r < m.pt.size(); ++r) {
    const auto& cell = m.rep.cell_for_row[r];
    if (m.pt.free[r] <= 0 || !cell.used || cell.mat != model::MatId::Theta) continue;
    theta(m.pt.free[r] - 1) = s.S[0](cell.row, cell.row);
  }
  auto a = nf::audit_newton_uls(m.pt, m.rep, s, theta);
  REQUIRE(a.has_value());
  CHECK(a->derivatives.gradient.norm() < 1e-12);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> exact(a->derivatives.hessian);
  CHECK(exact.eigenvalues().minCoeff() < -1.0);
  CHECK(a->diagnostics.status == estimate::NewtonAccuracyStatus::NonpositiveCurvature);
  CHECK_FALSE(a->diagnostics.passed);
  nf::NewtonAdapterOptions gn;
  gn.gauss_newton = true;
  auto approx = nf::audit_newton_uls(m.pt, m.rep, s, theta, gn);
  REQUIRE(approx.has_value());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> gauss(approx->derivatives.hessian);
  CHECK(gauss.eigenvalues().minCoeff() > -1e-10);
}

namespace {
// Integer ordinal data (three categories per item) from one smooth factor.
Eigen::MatrixXd ordinal_items(Eigen::Index n, int items, double shift) {
  Eigen::MatrixXd X(n, items);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double t = static_cast<double>(i + 1) + shift;
    const double eta = std::sin(.17 * t) + std::cos(.31 * t);
    for (int j = 0; j < items; ++j) {
      const double y = (.6 + .05 * j) * eta + std::sin((1.3 + .41 * j) * t);
      X(i, j) = 1.0 + (y > -.5 + .1 * j) + (y > .5 - .05 * j);
    }
  }
  return X;
}
void check_ordinal_hessian(const Model& m, const data::OrdinalStats& s,
                           estimate::OrdinalWeightKind w,
                           estimate::OrdinalParameterization param) {
  auto start = estimate::ordinal_start_values(m.pt, m.rep, s, {});
  const std::string why = start.has_value() ? std::string() : start.error().detail;
  INFO(why);
  REQUIRE(start.has_value());
  Eigen::VectorXd x = *start;
  for (Eigen::Index k = 0; k < x.size(); ++k) x(k) += 0.03 * std::sin(2.0 * static_cast<double>(k) + 0.5);
  auto parts = nf::ordinal_ls_newton_parts(m.pt, m.rep, s, x, w, param);
  const std::string detail = parts.has_value() ? std::string() : parts.error().detail;
  INFO(detail);
  REQUIRE(parts.has_value());
  estimate::Estimates at; at.theta = x;
  auto original = nf::ordinal_ls_objective(m.pt, m.rep, s, at, w, param);
  REQUIRE(original.has_value());
  double n = 0; for (auto nb : s.n_obs) n += static_cast<double>(nb);
  auto fd = nf::evaluate_newton_objective(optim::scalarize(original->problem), x, n, n);
  REQUIRE(fd.status == estimate::NewtonAccuracyStatus::Available);
  CHECK((parts->hessian - fd.hessian).norm() <= 1e-6 * (1 + fd.hessian.norm()));
  CHECK(parts->gradient_variance.allFinite());
  const auto& A = parts->metric_factor;
  const auto& b = parts->metric_score_residual;
  auto J = original->problem.J(x);
  auto r = original->problem.r(x);
  REQUIRE(J.has_value()); REQUIRE(r.has_value());
  CHECK((A - std::sqrt(n) * (*J)).norm() < 1e-10 * (1 + A.norm()));
  CHECK((b - std::sqrt(n) * (*r)).norm() < 1e-10 * (1 + b.norm()));
  CHECK((parts->gradient_variance - A.transpose() * A).norm() < 1e-10);
  CHECK((parts->hessian - A.transpose() * A - parts->curvature_correction).norm()
        < 1e-10 * (1 + parts->hessian.norm()));
  CHECK((A.transpose() * b - fd.gradient).norm() < 1e-9 * (1 + fd.gradient.norm()));
}
} // namespace

TEST_CASE("Newton adapters: the analytic ordinal Hessian matches gradient differences") {
  auto s = data::ordinal_stats_from_integer_data({ordinal_items(400, 4, 0.0)}, true);
  REQUIRE(s.has_value());
  const auto delta = model_with("f =~ x1 + x2 + x3 + x4\nx1 | t1 + t2\nx2 | t1 + t2\n"
                                "x3 | t1 + t2\nx4 | t1 + t2", false, false);
  // Two groups with equal thresholds free the second group's latent mean and
  // response scales, which exercises the mean Jacobian of the implied
  // thresholds and the standardization terms.
  auto s2 = data::ordinal_stats_from_integer_data(
      {ordinal_items(400, 4, 0.0), ordinal_items(300, 4, 7.0)}, true);
  REQUIRE(s2.has_value());
  auto parsed = parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\nx1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n");
  REQUIRE(parsed.has_value());
  spec::BuildOptions two;
  two.n_groups = 2;
  two.meanstructure = true;
  auto pt2 = spec::build(*parsed, two);
  REQUIRE(pt2.has_value());
  pt2->group_equal = {spec::GroupEqual::Thresholds};
  auto rep2 = model::build_matrix_rep(*pt2);
  REQUIRE(rep2.has_value());
  const Model mean{std::move(*pt2), std::move(*rep2)};
  for (auto w : {estimate::OrdinalWeightKind::ULS, estimate::OrdinalWeightKind::DWLS,
                 estimate::OrdinalWeightKind::WLS}) {
    for (auto param : {estimate::OrdinalParameterization::Delta,
                       estimate::OrdinalParameterization::Theta}) {
      CAPTURE(static_cast<int>(w)); CAPTURE(static_cast<int>(param));
      check_ordinal_hessian(delta, *s, w, param);
    }
  }
  auto moments2 = data::ordinal_moments_from_stats(*s2);
  auto start2 = estimate::ordinal_start_values(mean.pt, mean.rep, moments2, {});
  const std::string why2 = start2.has_value() ? std::string() : start2.error().detail;
  INFO("two-group start: " << why2);
  REQUIRE(start2.has_value());
  for (auto w : {estimate::OrdinalWeightKind::ULS, estimate::OrdinalWeightKind::DWLS,
                 estimate::OrdinalWeightKind::WLS}) {
    CAPTURE(static_cast<int>(w));
    Eigen::VectorXd x = *start2;
    for (Eigen::Index k = 0; k < x.size(); ++k) x(k) += 0.03 * std::sin(2.0 * static_cast<double>(k) + 0.5);
    auto parts = nf::ordinal_ls_newton_parts(mean.pt, mean.rep, *s2, x, w);
    const std::string detail = parts.has_value() ? std::string() : parts.error().detail;
    INFO(detail);
    REQUIRE(parts.has_value());
    // The prepared model frees the second group's latent mean and scales.
    int free_means = 0;
    for (std::size_t r = 0; r < parts->pt.size(); ++r)
      free_means += parts->pt.op[r] == parse::Op::Intercept && parts->pt.free[r] > 0;
    CHECK(free_means > 0);
    estimate::Estimates at; at.theta = x;
    auto original = nf::ordinal_ls_objective(mean.pt, mean.rep, *s2, at, w);
    REQUIRE(original.has_value());
    auto fd = nf::evaluate_newton_objective(optim::scalarize(original->problem), x, 700, 700);
    REQUIRE(fd.status == estimate::NewtonAccuracyStatus::Available);
    CHECK((parts->hessian - fd.hessian).norm() <= 1e-6 * (1 + fd.hessian.norm()));
  }
  // The audit uses analytic curvature and the fitting-weight working metric.
  auto start = estimate::ordinal_start_values(delta.pt, delta.rep, *s, {});
  REQUIRE(start.has_value());
  auto a = nf::audit_newton_ordinal(delta.pt, delta.rep, *s, *start);
  REQUIRE(a.has_value()); check_artifacts(*a);
  CHECK(a->derivatives.curvature_kind == nf::NewtonCurvatureKind::AnalyticObserved);
  CHECK(a->diagnostics.metric == estimate::NewtonMetricKind::Sandwich);
  CHECK(a->derivatives.metric_factor.rows() == a->derivatives.whitened_jacobian.rows());
  CHECK(a->derivatives.ls_curvature_correction.rows() == start->size());
}

TEST_CASE("Newton adapters: the analytic multi-information penalty Hessian matches gradient differences") {
  const auto cfa = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nx1 ~~ x4", false, false);
  const auto sem = model_with("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf3 =~ x7 + x8 + x9\n"
                              "f2 ~ f1\nf3 ~ f1 + f2", false, false);
  const auto groups = model_with("f =~ x1 + x2 + x3 + x4", false, true, 2);
  const auto s6 = pd_sample(6, 1, 0.3);
  const auto s9 = pd_sample(9, 1, 0.5);
  const auto s4 = pd_sample(4, 2, 0.2);
  for (const auto* c : {&cfa, &sem, &groups}) {
    const auto& s = c == &cfa ? s6 : c == &sem ? s9 : s4;
    auto ev = model::ModelEvaluator::build(c->pt, c->rep);
    REQUIRE(ev.has_value());
    auto theta = estimate::simple_start_values(c->pt, c->rep, s, {});
    REQUIRE(theta.has_value());
    Eigen::VectorXd x = *theta;
    for (Eigen::Index k = 0; k < x.size(); ++k) x(k) += 0.05 * std::sin(1.7 * static_cast<double>(k) + 0.3);
    for (auto target : {nf::PenaltyTarget::Joint, nf::PenaltyTarget::Determinacy}) {
      CAPTURE(static_cast<int>(target));
      auto layout = nf::multiinfo_penalty_layout(*ev, x, target);
      REQUIRE(layout.has_value());
      optim::ScalarProblem p;
      p.n_param = x.size();
      p.f = [&](const Eigen::VectorXd& t, Eigen::VectorXd& g) {
        auto v = nf::multiinfo_penalty(*layout, *ev, t, true);
        if (!v) return std::numeric_limits<double>::infinity();
        g = v->gradient;
        return v->value;
      };
      auto fd = nf::evaluate_newton_objective(p, x, 1, 1);
      REQUIRE(fd.status == estimate::NewtonAccuracyStatus::Available);
      auto H = nf::multiinfo_penalty_hessian(*layout, *ev, x);
      REQUIRE(H.has_value());
      CHECK((*H - fd.hessian).norm() <= 1e-6 * (1 + fd.hessian.norm()));
    }
  }
}
