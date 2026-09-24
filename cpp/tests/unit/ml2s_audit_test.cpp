#include <doctest/doctest.h>
#include <cmath>
#include <limits>
#include "magmaan/estimate/frontier/ml2s_audit.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using namespace magmaan;
namespace af = estimate::frontier;
namespace {
data::RawData observations(bool missing = false, bool groups = false) {
  Eigen::MatrixXd X(12, 2);
  X << -1, -.4, -.6, .5, -.2, -.7, .4, -.1, .8, .7, 1.1, .2,
       -.8, .9, -.4, -.8, .1, .4, .6, -.5, .9, 1.2, 1.3, -.2;
  data::RawData raw; raw.X = {X};
  if (groups) raw.X.push_back(2 * X.topRows(10));
  if (missing) {
    for (auto& block : raw.X) {
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask =
          Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(block.rows(), 2);
      mask(1, 0) = 0; mask(4, 1) = 0;
      block(1, 0) = std::numeric_limits<double>::quiet_NaN();
      block(4, 1) = std::numeric_limits<double>::quiet_NaN();
      raw.mask.push_back(mask);
    }
  }
  return raw;
}
struct Model { spec::LatentStructure pt; model::MatrixRep rep; };
Model saturated() {
  auto parsed = parse::Parser::parse("x ~~ x + y\ny ~~ y"); REQUIRE(parsed.has_value());
  spec::BuildOptions opts; opts.meanstructure = true; opts.fixed_x = false;
  auto pt = spec::build(*parsed, opts); REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  return {std::move(*pt), std::move(*rep)};
}
estimate::Estimates stage2_fit(const Model& m, const af::Ml2sStage2Input& input) {
  data::SampleStats s; s.S = input.moments.cov; s.mean = input.moments.mean; s.n_obs = input.moments.n_obs;
  auto start = estimate::simple_start_values(m.pt, m.rep, s, {}); REQUIRE(start.has_value());
  if (input.kind == estimate::fiml::TwoStageWeight::Nt) {
    auto fit = estimate::fit_ml(m.pt, m.rep, s, *start); REQUIRE(fit.has_value()); return *fit;
  }
  auto w = estimate::fiml::two_stage_stage2_weight_structured(input.moments, input.kind, input.dls);
  REQUIRE(w.has_value());
  auto fit = estimate::fit_gmm(m.pt, m.rep, s, *start, *w); REQUIRE(fit.has_value()); return *fit;
}
}

TEST_CASE("ML2S audit: direct solution and independent raw derivatives") {
  auto raw = observations(); auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack); REQUIRE(h1.has_value());
  REQUIRE(h1->solver_recorded); REQUIRE(h1->solver_blocks.size() == 1);
  CHECK(h1->solver_blocks[0].stop == estimate::fiml::H1StopReason::Direct);
  CHECK(h1->solver_blocks[0].iterations == 0);
  auto sm = estimate::fiml::saturated_em_moments(raw, *pack, *h1); REQUIRE(sm.has_value());
  CHECK(sm->solver_recorded);
  auto fresh = af::audit_saturated_endpoint(raw, *pack, *h1); REQUIRE(fresh.has_value());
  auto reused = af::audit_saturated_endpoint(raw, *pack, *h1, {}, &*sm); REQUIRE(reused.has_value());
  CHECK(fresh->derivatives.hessian.isApprox(reused->derivatives.hessian, 1e-12));
  CHECK((fresh->derivatives.gradient - reused->derivatives.gradient).norm() < 1e-12);
  CHECK(af::assess_convergence(*fresh, af::newton_convergence_policy()).status == estimate::FitCheck::Passed);
  // Independent finite-difference check of total objective normalization.
  auto displaced = *h1; displaced.sigma[0](0, 0) *= 1.1;
  auto audit = af::audit_saturated_endpoint(raw, *pack, displaced); REQUIRE(audit.has_value());
  model::ImpliedMoments lo, hi; lo.mu = hi.mu = displaced.mu; lo.sigma = hi.sigma = displaced.sigma;
  const double step = 1e-5; lo.sigma[0](0, 0) -= step; hi.sigma[0](0, 0) += step;
  auto fl = estimate::fiml::FIML{}.value(raw, pack->cache, lo);
  auto fh = estimate::fiml::FIML{}.value(raw, pack->cache, hi);
  REQUIRE(fl.has_value()); REQUIRE(fh.has_value());
  CHECK(audit->derivatives.gradient[0] == doctest::Approx(12 * .5 * (*fh - *fl) / (2 * step)).epsilon(1e-6));
  CHECK_FALSE(af::audit_saturated_endpoint(raw, *pack, displaced, {}, &*sm).has_value());
}

TEST_CASE("ML2S audit: multi-group missing endpoint uses the correct derivative order") {
  auto raw = observations(true, true); auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack); REQUIRE(h1.has_value());
  auto sm = estimate::fiml::saturated_em_moments(raw, *pack, *h1); REQUIRE(sm.has_value());
  auto fresh = af::audit_saturated_endpoint(raw, *pack, *h1); REQUIRE(fresh.has_value());
  auto cached = af::audit_saturated_endpoint(raw, *pack, *h1, {}, &*sm); REQUIRE(cached.has_value());
  CHECK(fresh->derivatives.n_obs == 22);
  CHECK(fresh->derivatives.hessian.isApprox(cached->derivatives.hessian, 1e-12));
  CHECK((fresh->derivatives.gradient - cached->derivatives.gradient).norm() < 1e-10);
  CHECK(af::assess_convergence(*fresh, af::newton_convergence_policy()).status == estimate::FitCheck::Passed);
  CHECK(fresh->evidence.objective.consistent);
  for (const auto& b : h1->solver_blocks) CHECK(b.stop == estimate::fiml::H1StopReason::ParameterTolerance);
}

TEST_CASE("ML2S audit: raw negative curvature survives information repair") {
  auto raw = observations(); auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack); REQUIRE(h1.has_value());
  h1->sigma[0] *= 10;
  auto sm = estimate::fiml::saturated_em_moments(raw, *pack, *h1); REQUIRE(sm.has_value());
  REQUIRE(sm->information_repaired); CHECK(sm->information_ridge > 0);
  CHECK(sm->information_min_eigen < 0);
  CHECK((sm->H - sm->raw_H).isApprox(sm->information_ridge * Eigen::MatrixXd::Identity(5, 5), 1e-8));
  auto a = af::audit_saturated_endpoint(raw, *pack, *h1, {}, &*sm); REQUIRE(a.has_value());
  CHECK(a->solution.status == estimate::NewtonAccuracyStatus::NonpositiveCurvature);
  CHECK(af::assess_convergence(*a, af::newton_convergence_policy()).newton.status == estimate::FitCheck::Failed);
}

TEST_CASE("ML2S audit: early EM stop retains its final evidence") {
  auto raw = observations(true); auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  estimate::fiml::FIMLH1Options opts; opts.max_iter = 1; opts.parameter_tol = 1e-14;
  opts.error_on_nonconvergence = false; opts.covariance_floor = 2;
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack, opts); REQUIRE(h1.has_value());
  REQUIRE(h1->solver_blocks.size() == 1);
  CHECK(h1->solver_blocks[0].stop == estimate::fiml::H1StopReason::IterationLimit);
  CHECK(h1->solver_blocks[0].iterations == 1);
  CHECK(h1->solver_blocks[0].parameter_change > opts.parameter_tol);
  CHECK(h1->solver_blocks[0].covariance_repairs > 0);
  CHECK(h1->solver_options.covariance_floor == 2);
  auto a = af::audit_saturated_endpoint(raw, *pack, *h1); REQUIRE(a.has_value());
  CHECK(a->evidence.objective.consistent);
  CHECK(af::assess_convergence(*a).status == estimate::FitCheck::Failed);
}

TEST_CASE("ML2S audit: compose all Stage-2 policies with explicit handoff evidence") {
  auto m = saturated(); auto raw = observations(true);
  auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack); REQUIRE(h1.has_value());
  auto sm = estimate::fiml::saturated_em_moments(raw, *pack, *h1); REQUIRE(sm.has_value());
  using K = estimate::fiml::TwoStageWeight;
  for (auto kind : {K::Nt, K::Uls, K::Dwls, K::Adf, K::Dls}) {
    CAPTURE(static_cast<int>(kind));
    af::Ml2sStage2Input input{*sm, kind, {}};
    auto fit = stage2_fit(m, input);
    auto report = af::audit_ml2s(m.pt, m.rep, raw, *pack, *h1, fit.theta, kind, {}, {}, input, fit.fmin, &*sm);
    REQUIRE(report.has_value());
    af::Ml2sConvergencePolicy p;
    p.stage1 = af::newton_convergence_policy(); p.stage2 = af::newton_convergence_policy();
    p.stage1.require_objective_consistency = true; p.stage2.require_objective_consistency = true;
    p.require_solver_stop = true;
    auto a = af::assess_convergence(*report, p);
    CHECK(a.stage1.status == estimate::FitCheck::Passed);
    CHECK(a.stage2.status == estimate::FitCheck::Passed);
    CHECK(a.handoff.status == estimate::FitCheck::Passed);
    CHECK(a.status == estimate::FitCheck::Passed);
  }
  af::Ml2sStage2Input input{*sm, K::Nt, {}}; auto fit = stage2_fit(m, input);
  auto unverified = af::audit_ml2s(m.pt, m.rep, raw, *pack, *h1, fit.theta);
  REQUIRE(unverified.has_value());
  CHECK(af::assess_convergence(*unverified).status == estimate::FitCheck::Unchecked);
  af::Ml2sConvergencePolicy p; p.require_handoff = false;
  CHECK(af::assess_convergence(*unverified, p).status == estimate::FitCheck::Passed);
  unverified->stage1.endpoint.solver_blocks[0].stop = estimate::fiml::H1StopReason::IterationLimit;
  CHECK(af::assess_convergence(*unverified, p).status == estimate::FitCheck::Passed);
  p.require_solver_stop = true;
  CHECK(af::assess_convergence(*unverified, p).status == estimate::FitCheck::Failed);
  unverified->stage1.endpoint.solver_recorded = false;
  CHECK(af::assess_convergence(*unverified, p).status == estimate::FitCheck::Unchecked);
  input.moments.cov[0](0, 0) += .1;
  auto mismatch = af::audit_ml2s(m.pt, m.rep, raw, *pack, *h1, fit.theta, K::Nt, {}, {}, input);
  REQUIRE(mismatch.has_value());
  CHECK(af::assess_convergence(*mismatch).status == estimate::FitCheck::Failed);
}

TEST_CASE("ML2S audit: transformed handoff keeps the original endpoint separate") {
  auto m = saturated(); auto raw = observations(); auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack); REQUIRE(h1.has_value());
  auto sm = estimate::fiml::saturated_em_moments(raw, *pack, *h1); REQUIRE(sm.has_value());
  af::Ml2sAuditOptions opts; opts.transformation.enabled = true; opts.transformation.intensity = .5;
  auto transformed = estimate::fiml::regularize_saturated_stage1(*sm, opts.transformation); REQUIRE(transformed.has_value());
  CHECK(transformed->moments.raw_H.size() == 0);
  af::Ml2sStage2Input input{transformed->moments, estimate::fiml::TwoStageWeight::Nt, {}};
  auto fit = stage2_fit(m, input);
  auto r = af::audit_ml2s(m.pt, m.rep, raw, *pack, *h1, fit.theta, input.kind, {}, opts, input, fit.fmin);
  REQUIRE(r.has_value());
  CHECK(r->source.cov[0].isApprox(h1->sigma[0]));
  CHECK_FALSE(r->source.cov[0].isApprox(r->stage2_input.moments.cov[0]));
  CHECK(r->transformation_blocks[0].applied);
  CHECK(af::assess_convergence(*r).status == estimate::FitCheck::Passed);
}

TEST_CASE("ML2S audit: first-order requests and diagnostic curvature provenance") {
  auto raw = observations(); auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack.has_value());
  auto h1 = estimate::fiml::fiml_h1_moments(raw, *pack); REQUIRE(h1.has_value());
  auto fd = estimate::fiml::diagnostic::saturated_em_moments_fd(raw); REQUIRE(fd.has_value());
  CHECK_FALSE(fd->raw_hessian_analytic);
  // An unlabelled/diagnostic Hessian must not masquerade as analytic evidence.
  fd->raw_H.setZero();
  auto a = af::audit_saturated_endpoint(raw, *pack, *h1, {}, &*fd); REQUIRE(a.has_value());
  CHECK(a->derivatives.hessian.norm() > 1);
  CHECK(af::assess_convergence(*a, af::newton_convergence_policy()).status == estimate::FitCheck::Passed);
  af::SaturatedAuditOptions opts; opts.newton = false;
  auto first = af::audit_saturated_endpoint(raw, *pack, *h1, opts); REQUIRE(first.has_value());
  CHECK(first->derivatives.hessian.size() == 0);
  CHECK(af::assess_convergence(*first).status == estimate::FitCheck::Passed);
  CHECK(af::assess_convergence(*first, af::newton_convergence_policy()).status == estimate::FitCheck::Unchecked);
}
