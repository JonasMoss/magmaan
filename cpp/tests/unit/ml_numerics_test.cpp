#include <doctest/doctest.h>
#include "magmaan/estimate/ml_numerics.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include <cmath>
#include <limits>

using namespace magmaan;
namespace {
struct Case { spec::LatentStructure pt; model::MatrixRep rep; data::SampleStats sample; };
Case example(const char* syntax = "f =~ x1 + x2 + x3 + x4") {
  auto parsed = parse::Parser::parse(syntax); REQUIRE(parsed);
  spec::BuildOptions options; options.fixed_x = false;
  auto pt = spec::build(*parsed, options); REQUIRE(pt);
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep);
  Eigen::Vector4d l(1.,.8,1.2,.7);
  data::SampleStats samp;
  samp.S = {l*l.transpose()+.5*Eigen::Matrix4d::Identity()}; samp.n_obs={500};
  return {*pt,*rep,samp};
}
}
TEST_CASE("ML numerical policy transports std.lv starts without changing moments") {
  auto c=example();
  auto start=estimate::scaled_fabin_start_values(c.pt,c.rep,c.sample); REQUIRE(start);
  CHECK(start->branch==estimate::MlStartBranch::TransportedStdLv);
  auto parsed=parse::Parser::parse("f =~ x1 + x2 + x3 + x4"); REQUIRE(parsed);
  spec::BuildOptions options; options.fixed_x=false; options.std_lv=true;
  auto pt=spec::build(*parsed,options); REQUIRE(pt);
  auto rep=model::build_matrix_rep(*pt); REQUIRE(rep);
  auto standard=estimate::fabin_start_values(*pt,*rep,c.sample); REQUIRE(standard);
  auto se=model::ModelEvaluator::build(*pt,*rep); REQUIRE(se);
  auto me=model::ModelEvaluator::build(c.pt,c.rep); REQUIRE(me);
  auto a=se->evaluate(*standard,false,false), b=me->evaluate(start->theta,false,false);
  REQUIRE(a); REQUIRE(b);
  CHECK((a->moments.sigma[0]-b->moments.sigma[0]).norm()<1e-12);
  auto scaled=c.sample; scaled.S[0]*=100;
  auto other=estimate::scaled_fabin_start_values(c.pt,c.rep,scaled); REQUIRE(other);
  auto v=me->evaluate(other->theta,false,false); REQUIRE(v);
  CHECK((v->moments.sigma[0]-100*b->moments.sigma[0]).norm()<1e-9);
  spec::Starts hints; hints.hint.resize(static_cast<std::size_t>(c.pt.n_free()),std::numeric_limits<double>::quiet_NaN());
  hints.hint[0]=.314159;
  auto explicit_hint=estimate::scaled_fabin_start_values(c.pt,c.rep,c.sample,hints); REQUIRE(explicit_hint);
  CHECK(explicit_hint->theta(0)==.314159);
}
TEST_CASE("ML numerical policy falls back for fixed values and equalities") {
  for (const char* syntax : {"f =~ x1 + 2*x2 + x3 + x4", "f =~ x1 + a*x2 + a*x3 + x4", "f =~ x1 + a*x2 + b*x3 + x4\na == b*b"}) {
    auto c=example(syntax);
    auto start=estimate::scaled_fabin_start_values(c.pt,c.rep,c.sample); REQUIRE(start);
    auto native=estimate::fabin_start_values(c.pt,c.rep,c.sample); REQUIRE(native);
    CHECK(start->branch==estimate::MlStartBranch::NativeFabin);
    if (start->fallback_reason == estimate::StartTransportIssue::EqualityConstraints &&
        c.pt.nl_constraints.empty()) {
      for (std::size_t i = 0; i < c.pt.size(); ++i) {
        const auto cell = c.rep.cell_for_row[i];
        if (!cell.used || c.pt.free[i] <= 0) continue;
        const auto k = c.pt.free[i] - 1;
        const double expected = cell.mat == model::MatId::Psi && cell.row == cell.col
            ? .05 * c.sample.S[0](0, 0) : (*native)(k);
        CHECK(start->theta(k) == doctest::Approx(expected).epsilon(1e-14));
      }
    } else {
      CHECK((start->theta-*native).norm()==0);
    }
  }
}
TEST_CASE("ML numerical policy preserves fitted objective under scaling and bounds") {
  for (const char* syntax : {"f =~ x1 + x2 + x3 + x4", "f =~ x1 + a*x2 + a*x3 + x4"}) {
    auto c=example(syntax);
    for(double units : {.1,1.,10.}) {
      auto sample=c.sample; sample.S[0]*=units*units;
      auto start=estimate::scaled_fabin_start_values(c.pt,c.rep,sample); REQUIRE(start);
      auto opts=estimate::ml_optim_options();
      estimate::Bounds bounds;
      bounds.lower=Eigen::VectorXd::Constant(c.pt.n_free(),-.5*units*units);
      bounds.upper=Eigen::VectorXd::Constant(c.pt.n_free(),1000.);
      auto scaled=estimate::fit_ml(c.pt,c.rep,sample,start->theta,bounds,estimate::Backend::NloptSlsqp,opts); REQUIRE(scaled);
      CHECK(scaled->coordinate_scaling!=optim::CoordinateScaling::None);
      CHECK((scaled->theta.array()>=bounds.lower.array()-1e-8).all());
      opts.coordinate_scaling=optim::CoordinateScaling::None;
      auto reference=estimate::fit_ml(c.pt,c.rep,sample,scaled->theta,bounds,estimate::Backend::NloptSlsqp,opts); REQUIRE(reference);
      CHECK(reference->coordinate_scaling==optim::CoordinateScaling::None);
      CHECK(std::abs(scaled->fmin-reference->fmin)<1e-8);
      CHECK(scaled->diagnostics.lin_eq_residual_inf<1e-8);
    }
  }
}
TEST_CASE("ML numerical policy defaults and coordinate defaults") {
  auto opts=estimate::ml_optim_options();
  CHECK(*opts.nlopt.ftol_rel==1e-12); CHECK(*opts.nlopt.xtol_rel==1e-10); CHECK(*opts.nlopt.max_eval==5000);
  CHECK(opts.coordinate_scaling==optim::CoordinateScaling::Information); CHECK(optim::OptimOptions{}.coordinate_scaling==optim::CoordinateScaling::Information);
  CHECK(estimate::frontier::ml_psd_options().diagonal_preconditioning);
  CHECK(*estimate::frontier::ml_psd_optim_options().nlopt.constraint_tol==1e-8);
  auto c=example(); c.sample.S[0](0,0)=0;
  auto con=estimate::build_eq_constraints(c.pt); REQUIRE(con);
  CHECK_FALSE(estimate::ml_coordinate_scale(c.pt,c.rep,*con,c.sample));
}

TEST_CASE("ML numerical policy scales constrained adapters") {
  for (const char* syntax : {"f =~ x1 + a*x2 + b*x3 + x4\na == b*b",
                             "f =~ x1 + a*x2 + b*x3 + x4\na + b == 2"}) {
    auto c=example(syntax);
    auto start=estimate::scaled_fabin_start_values(c.pt,c.rep,c.sample); REQUIRE(start);
    estimate::Bounds bounds;
    bounds.lower=Eigen::VectorXd::Constant(c.pt.n_free(),-10.);
    bounds.upper=Eigen::VectorXd::Constant(c.pt.n_free(),10.);
    auto fit=estimate::fit_ml(c.pt,c.rep,c.sample,start->theta,bounds,
                             estimate::Backend::NloptSlsqp);
    REQUIRE(fit);
    CHECK(fit->coordinate_scaling!=optim::CoordinateScaling::None);
    CHECK(fit->diagnostics.lin_eq_satisfied);
    CHECK(fit->diagnostics.nl_eq_satisfied);
  }
}

TEST_CASE("Start transport is independent of constructor and rejects unusable scales") {
  auto c = example();
  auto plan = estimate::prepare_std_lv_transport(c.pt, c.rep); REQUIRE(plan);
  for (auto method : {estimate::StartMethod::Simple, estimate::StartMethod::Fabin2,
                      estimate::StartMethod::Fabin3}) {
    auto source = estimate::construct_start_values(plan->source, plan->source_rep, c.sample, method);
    REQUIRE(source);
    auto transported = estimate::transport_start_values(*plan, *source); REQUIRE(transported);
    auto se = model::ModelEvaluator::build(plan->source, plan->source_rep); REQUIRE(se);
    auto te = model::ModelEvaluator::build(c.pt, c.rep); REQUIRE(te);
    auto a = se->evaluate(*source, false, false);
    auto b = te->evaluate(*transported, false, false); REQUIRE(a); REQUIRE(b);
    CHECK((a->moments.sigma[0] - b->moments.sigma[0]).norm() < 1e-12);
    auto pipeline = estimate::start_values(c.pt, c.rep, c.sample,
        {method, estimate::StartTransport::RequireStdLv}); REQUIRE(pipeline);
    CHECK((pipeline->theta - *transported).norm() == 0);
    CHECK(pipeline->method == method);
  }
  auto source = estimate::construct_start_values(plan->source, plan->source_rep,
      c.sample, estimate::StartMethod::Fabin3); REQUIRE(source);
  const int marker_row = plan->markers[0][0];
  (*source)(plan->source.free[static_cast<std::size_t>(marker_row)] - 1) = 0;
  auto zero = estimate::transport_start_values(*plan, *source);
  REQUIRE_FALSE(zero); CHECK(zero.error() == estimate::StartTransportIssue::InvalidScale);
  source->setConstant(std::numeric_limits<double>::quiet_NaN());
  auto nan = estimate::transport_start_values(*plan, *source);
  REQUIRE_FALSE(nan); CHECK(nan.error() == estimate::StartTransportIssue::InvalidVector);
  auto short_vector = estimate::transport_start_values(*plan, Eigen::VectorXd::Zero(1));
  REQUIRE_FALSE(short_vector); CHECK(short_vector.error() == estimate::StartTransportIssue::InvalidVector);
}

TEST_CASE("Start policy reports fallback and supports requiring or disabling transport") {
  for (const char* syntax : {"f =~ x1 + 2*x2 + x3 + x4",
                             "f =~ x1 + a*x2 + a*x3 + x4"}) {
    auto c = example(syntax);
    auto automatic = estimate::scaled_fabin_start_values(c.pt, c.rep, c.sample); REQUIRE(automatic);
    CHECK(automatic->fallback_reason != estimate::StartTransportIssue::None);
    CHECK_FALSE(estimate::start_values(c.pt, c.rep, c.sample,
        {estimate::StartMethod::Fabin3, estimate::StartTransport::RequireStdLv}));
    auto native = estimate::start_values(c.pt, c.rep, c.sample,
        {estimate::StartMethod::Fabin3, estimate::StartTransport::Native}); REQUIRE(native);
    CHECK(native->fallback_reason == estimate::StartTransportIssue::None);
    if (automatic->fallback_reason == estimate::StartTransportIssue::FixedValues)
      CHECK((native->theta - automatic->theta).norm() == 0);
    else {
      for (std::size_t i = 0; i < c.pt.size(); ++i) {
        const auto cell = c.rep.cell_for_row[i];
        if (!cell.used || cell.mat != model::MatId::Psi || cell.row != cell.col ||
            c.pt.free[i] <= 0) continue;
        const auto k = c.pt.free[i] - 1;
        CHECK(native->theta(k) == doctest::Approx(.05));
        CHECK(automatic->theta(k) == doctest::Approx(.05 * c.sample.S[0](0, 0)));
      }
    }
  }
}

TEST_CASE("Transport preserves structural covariance including non-PSD starts") {
  auto c = example("f =~ x1 + x2\ng =~ x3 + x4\ng ~ f");
  auto plan = estimate::prepare_std_lv_transport(c.pt, c.rep); REQUIRE(plan);
  auto source = estimate::construct_start_values(plan->source, plan->source_rep,
      c.sample, estimate::StartMethod::Simple); REQUIRE(source);
  for (std::size_t i = 0; i < plan->source.size(); ++i) {
    const auto cell = plan->source_rep.cell_for_row[i];
    const int free = plan->source.free[i];
    if (!cell.used || free <= 0) continue;
    if (cell.mat == model::MatId::Beta) (*source)(free-1) = .3;
    if (cell.mat == model::MatId::Theta && cell.row == cell.col) (*source)(free-1) = -.01;
  }
  auto transported = estimate::transport_start_values(*plan, *source); REQUIRE(transported);
  auto se = model::ModelEvaluator::build(plan->source, plan->source_rep); REQUIRE(se);
  auto te = model::ModelEvaluator::build(c.pt, c.rep); REQUIRE(te);
  auto a = se->evaluate(*source, false, false);
  auto b = te->evaluate(*transported, false, false); REQUIRE(a); REQUIRE(b);
  CHECK((a->moments.sigma[0] - b->moments.sigma[0]).norm() < 1e-12);
}

TEST_CASE("Start transport preserves implied means and signed marker scales") {
  auto parsed = parse::Parser::parse("f =~ x1 + x2 + x3 + x4"); REQUIRE(parsed);
  spec::BuildOptions options;
  options.fixed_x = false;
  options.meanstructure = true;
  options.int_lv_free = true;
  auto pt = spec::build(*parsed, options); REQUIRE(pt);
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep);
  auto plan = estimate::prepare_std_lv_transport(*pt, *rep); REQUIRE(plan);
  auto sample = example().sample;
  sample.mean = {Eigen::Vector4d(1., 2., 3., 4.)};
  auto source = estimate::construct_start_values(plan->source, plan->source_rep,
      sample, estimate::StartMethod::Simple); REQUIRE(source);
  for (std::size_t i = 0; i < plan->source.size(); ++i) {
    const auto cell = plan->source_rep.cell_for_row[i];
    const int free = plan->source.free[i];
    if (!cell.used || free <= 0) continue;
    if (cell.mat == model::MatId::Lambda) (*source)(free-1) *= -1;
    if (cell.mat == model::MatId::Alpha) (*source)(free-1) = 2;
  }
  auto transported = estimate::transport_start_values(*plan, *source); REQUIRE(transported);
  auto se = model::ModelEvaluator::build(plan->source, plan->source_rep); REQUIRE(se);
  auto te = model::ModelEvaluator::build(*pt, *rep); REQUIRE(te);
  auto a = se->evaluate(*source, false, false);
  auto b = te->evaluate(*transported, false, false); REQUIRE(a); REQUIRE(b);
  CHECK((a->moments.sigma[0] - b->moments.sigma[0]).norm() < 1e-12);
  CHECK((a->moments.mu[0] - b->moments.mu[0]).norm() < 1e-12);
}

TEST_CASE("Start policy preserves marker-only constructors when transport is requested") {
  auto c = example();
  using M = estimate::StartMethod;
  using T = estimate::StartTransport;
  for (auto method : {M::Guttman, M::Bentler1982, M::JamesStein}) {
    auto native = estimate::construct_start_values(c.pt, c.rep, c.sample, method);
    auto automatic = estimate::start_values(c.pt, c.rep, c.sample, {method, T::AutoStdLv});
    REQUIRE(native); REQUIRE(automatic);
    CHECK(automatic->theta.isApprox(*native));
    CHECK(automatic->branch == estimate::StartBranch::Native);
    CHECK(automatic->fallback_reason == estimate::StartTransportIssue::ConstructorRequiresMarker);
    CHECK(automatic->requested_transport == T::AutoStdLv);
    CHECK_FALSE(estimate::start_values(c.pt, c.rep, c.sample, {method, T::RequireStdLv}));
  }
  auto invalid = Eigen::VectorXd::Zero(c.pt.n_free()).eval();
  CHECK(estimate::explicit_start_values(c.pt, invalid));
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_FALSE(estimate::explicit_start_values(c.pt, invalid));
  CHECK_FALSE(estimate::explicit_start_values(c.pt, Eigen::VectorXd::Zero(1)));
}

TEST_CASE("ML and GLS route nonlinear equalities to SLSQP and record it") {
  auto c = example("f =~ x1 + a*x2 + b*x3 + x4\na == b*b");
  auto start = estimate::ml_start_values(c.pt, c.rep, c.sample); REQUIRE(start);
  for (auto backend : {estimate::Backend::NloptLbfgs, estimate::Backend::Port}) {
    auto ml = estimate::fit_ml(c.pt, c.rep, c.sample, start->theta, {}, backend); REQUIRE(ml);
    if (!ml) continue;
    REQUIRE(ml->substituted_backend.has_value());
    if (!ml->substituted_backend) continue;
    CHECK(*ml->substituted_backend == estimate::Backend::NloptSlsqp);
    CHECK(ml->diagnostics.nl_eq_satisfied);
    auto gls = estimate::fit_gls(c.pt, c.rep, c.sample, start->theta, {}, backend); REQUIRE(gls);
    if (!gls) continue;
    REQUIRE(gls->substituted_backend.has_value());
    if (!gls->substituted_backend) continue;
    CHECK(*gls->substituted_backend == estimate::Backend::NloptSlsqp);
  }
  auto chosen = estimate::fit_ml(c.pt, c.rep, c.sample, start->theta, {},
                                 estimate::Backend::NloptSlsqp); REQUIRE(chosen);
  CHECK_FALSE(chosen->substituted_backend.has_value());
  auto routed = estimate::fit_ml(c.pt, c.rep, c.sample, start->theta); REQUIRE(routed);
  CHECK(std::abs(chosen->fmin - routed->fmin) < 1e-10);
  auto plain = example("f =~ x1 + x2 + x3 + x4");
  auto p0 = estimate::ml_start_values(plain.pt, plain.rep, plain.sample); REQUIRE(p0);
  auto unconstrained = estimate::fit_ml(plain.pt, plain.rep, plain.sample, p0->theta);
  REQUIRE(unconstrained);
  CHECK_FALSE(unconstrained->substituted_backend.has_value());
}

TEST_CASE("Equality fallback starts and barrier fits follow common data units") {
  auto c = example("f =~ x1 + a*x2 + a*x3 + x4");
  c.sample.S[0] = Eigen::Matrix4d::Constant(.49);
  c.sample.S[0].diagonal().setOnes();
  auto baseline = estimate::scaled_fabin_start_values(c.pt, c.rep, c.sample);
  REQUIRE(baseline);
  auto evaluator = model::ModelEvaluator::build(c.pt, c.rep); REQUIRE(evaluator);
  auto moments = evaluator->evaluate(baseline->theta, false, false); REQUIRE(moments);
  for (double scale : {.01, 1., 100.}) {
    auto sample = c.sample;
    sample.S[0] *= scale * scale;
    auto start = estimate::scaled_fabin_start_values(c.pt, c.rep, sample); REQUIRE(start);
    CHECK(start->fallback_reason == estimate::StartTransportIssue::EqualityConstraints);
    auto scaled_moments = evaluator->evaluate(start->theta, false, false); REQUIRE(scaled_moments);
    CHECK((scaled_moments->moments.sigma[0] / (scale * scale) - moments->moments.sigma[0]).norm() < 1e-10);
    auto fit = estimate::frontier::fit_ml_multiinfo(c.pt, c.rep, sample, start->theta,
        {}, {}, estimate::Backend::Port);
    REQUIRE(fit);
    CHECK((fit->estimates.theta - start->theta).norm() > 1e-8);
    CHECK(estimate::common_fit_verdict(fit->estimates.diagnostics).status == estimate::FitCheck::Passed);
    spec::Starts hints;
    hints.hint.assign(static_cast<std::size_t>(c.pt.n_free()), std::numeric_limits<double>::quiet_NaN());
    hints.hint[0] = .123;
    auto hinted = estimate::start_values(c.pt, c.rep, sample, {}, hints); REQUIRE(hinted);
    CHECK(hinted->theta(0) == doctest::Approx(.123));
  }
}

TEST_CASE("Equality-aware std.lv transport classifies simple scale ties") {
  using Issue = estimate::StartTransportIssue;
  for (const char* syntax : {
      "f =~ x1 + a*x2 + a*x3 + x4",
      "f =~ x1 + a*x2 + b*x3 + x4\na == 2*b",
      "f =~ x1 + a*x2\ng =~ x3 + a*x4",
      "f =~ x1 + x2\ng =~ x3 + x4\nx1 ~~ r*x1\nx3 ~~ r*x3",
      "f =~ x1 + a*x2\ng =~ x3 + a*x4\nf ~~ c*g"}) {
    auto c = example(syntax);
    auto plan = estimate::prepare_std_lv_transport(c.pt, c.rep); REQUIRE(plan);
    auto old_con = estimate::build_eq_constraints(c.pt); REQUIRE(old_con);
    auto new_con = estimate::build_eq_constraints(plan->source); REQUIRE(new_con);
    CHECK(old_con->n_alpha == new_con->n_alpha);
  }
  for (auto [syntax, issue] : {
      std::pair{"f =~ x1 + a*x2 + b*x3 + x4\na == b*b", Issue::NonlinearConstraints},
      std::pair{"f =~ x1 + a*x2 + x3 + x4\nx1 ~~ a*x1", Issue::MixedScaleForms},
      std::pair{"f =~ x1 + a*x2\ng =~ x3 + x4\ng ~ a*f", Issue::RegressionEqualities},
      std::pair{"f =~ x1 + a*x2 + b*x3 + x4\na + b == 2", Issue::GeneralLinearEqualities}}) {
    auto c = example(syntax);
    auto plan = estimate::prepare_std_lv_transport(c.pt, c.rep);
    REQUIRE_FALSE(plan); CHECK(plan.error() == issue);
  }
}

TEST_CASE("Equality-aware std.lv transport preserves tied markers and the model tangent") {
  auto c = example("f =~ x1 + a*x2\ng =~ x3 + a*x4");
  auto plan = estimate::prepare_std_lv_transport(c.pt, c.rep); REQUIRE(plan);
  CHECK(plan->scale_markers[0][0] == plan->scale_markers[0][1]);
  auto con = estimate::build_eq_constraints(plan->source); REQUIRE(con);
  auto source = estimate::construct_start_values(plan->source, plan->source_rep,
      c.sample, estimate::StartMethod::Simple); REQUIRE(source);
  auto se = model::ModelEvaluator::build(plan->source, plan->source_rep); REQUIRE(se);
  auto te = model::ModelEvaluator::build(c.pt, c.rep); REQUIRE(te);
  // Twenty interior chart points vary both the common gauge and the unanchored
  // variance. Projecting onto the equality manifold includes the marker ties.
  for (int trial = 0; trial < 20; ++trial) {
    for (std::size_t i = 0; i < plan->source.size(); ++i) {
      const int free = plan->source.free[i]; if (free <= 0) continue;
      const auto cell = plan->source_rep.cell_for_row[i];
      (*source)(free - 1) = cell.mat == model::MatId::Lambda ? .7 + .01 * trial
          : cell.row == cell.col ? .5 + .02 * trial : .05;
    }
    *source = con->expand(con->contract(*source));
    auto target = estimate::transport_start_values(*plan, *source); REQUIRE(target);
    auto target_con = estimate::build_eq_constraints(c.pt); REQUIRE(target_con);
    CHECK((target_con->expand(target_con->contract(*target)) - *target).norm() < 1e-12);
    auto before = se->evaluate(*source, false, false); REQUIRE(before);
    auto after = te->evaluate(*target, false, false); REQUIRE(after);
    CHECK((before->moments.sigma[0] - after->moments.sigma[0]).norm() < 1e-12);
    auto matrices = te->assembled(*target); REQUIRE(matrices);
    const double scale = std::sqrt(matrices->blocks[0].Psi(0, 0));
    const auto& m = matrices->blocks[0];
    Eigen::VectorXd round_trip = Eigen::VectorXd::Zero(source->size());
    for (std::size_t i = 0; i < plan->source.size(); ++i) {
      const int free = plan->source.free[i]; if (free <= 0) continue;
      const auto cell = plan->source_rep.cell_for_row[i];
      round_trip(free - 1) = cell.mat == model::MatId::Lambda ? m.Lambda(cell.row, cell.col) * scale
          : cell.mat == model::MatId::Psi ? m.Psi(cell.row, cell.col) / (scale * scale)
          : m.Theta(cell.row, cell.col);
    }
    CHECK((round_trip - *source).norm() < 1e-12);
  }
}

TEST_CASE("Equality-aware std.lv transport ties group scales for loadings and latent means") {
  for (auto family : {spec::GroupEqual::Loadings, spec::GroupEqual::Means, spec::GroupEqual::LvVariances}) {
    auto parsed = parse::Parser::parse("f =~ x1 + x2 + x3 + x4"); REQUIRE(parsed);
    spec::BuildOptions options;
    options.fixed_x = false; options.n_groups = 2;
    options.meanstructure = true; options.int_lv_free = true;
    options.group_equal = {family};
    if (family == spec::GroupEqual::LvVariances)
      options.group_equal.push_back(spec::GroupEqual::Loadings);
    auto pt = spec::build(*parsed, options); REQUIRE(pt);
    auto rep = model::build_matrix_rep(*pt); REQUIRE(rep);
    auto plan = estimate::prepare_std_lv_transport(*pt, *rep); REQUIRE(plan);
    CHECK(plan->scale_markers[0][0] == plan->scale_markers[1][0]);
    auto old_con = estimate::build_eq_constraints(*pt); REQUIRE(old_con);
    auto new_con = estimate::build_eq_constraints(plan->source); REQUIRE(new_con);
    CHECK(old_con->n_alpha == new_con->n_alpha);
    Eigen::VectorXd point = Eigen::VectorXd::Constant(plan->source.n_free(), .7);
    point = new_con->expand(new_con->contract(point));
    auto transported = estimate::transport_start_values(*plan, point); REQUIRE(transported);
    auto se = model::ModelEvaluator::build(plan->source, plan->source_rep); REQUIRE(se);
    auto te = model::ModelEvaluator::build(*pt, *rep); REQUIRE(te);
    auto before = se->evaluate(point, false, false); REQUIRE(before);
    auto after = te->evaluate(*transported, false, false); REQUIRE(after);
    for (std::size_t b = 0; b < 2; ++b) {
      CHECK((before->moments.sigma[b] - after->moments.sigma[b]).norm() < 1e-12);
      CHECK((before->moments.mu[b] - after->moments.mu[b]).norm() < 1e-12);
    }
    CHECK((old_con->expand(old_con->contract(*transported)) - *transported).norm() < 1e-12);
    auto matrices = te->assembled(*transported); REQUIRE(matrices);
    Eigen::VectorXd round_trip = Eigen::VectorXd::Zero(point.size());
    for (std::size_t i = 0; i < plan->source.size(); ++i) {
      const int free = plan->source.free[i]; if (free <= 0) continue;
      const auto cell = plan->source_rep.cell_for_row[i];
      const auto marker = rep->cell_for_row[static_cast<std::size_t>(plan->scale_markers[static_cast<std::size_t>(cell.block)][0])];
      const double scale = std::sqrt(matrices->blocks[static_cast<std::size_t>(marker.block)].Psi(marker.col, marker.col));
      const auto& m = matrices->blocks[static_cast<std::size_t>(cell.block)];
      switch (cell.mat) {
        case model::MatId::Lambda: round_trip(free-1) = m.Lambda(cell.row,cell.col)*scale; break;
        case model::MatId::Psi: round_trip(free-1) = m.Psi(cell.row,cell.col)/(scale*scale); break;
        case model::MatId::Theta: round_trip(free-1) = m.Theta(cell.row,cell.col); break;
        case model::MatId::Nu: round_trip(free-1) = m.Nu(cell.row); break;
        case model::MatId::Alpha: round_trip(free-1) = m.Alpha(cell.row)/scale; break;
        case model::MatId::Beta: round_trip(free-1) = m.Beta(cell.row,cell.col); break;
      }
    }
    CHECK((round_trip - point).norm() < 1e-12);
  }
}

TEST_CASE("Equality-aware std.lv covariance forms must agree after simple ties") {
  auto parsed = parse::Parser::parse("f =~ x1 + x2\ng =~ x3 + x4"); REQUIRE(parsed);
  spec::BuildOptions options; options.fixed_x = false; options.n_groups = 2;
  options.group_equal = {spec::GroupEqual::Loadings, spec::GroupEqual::LvCovariances};
  auto pt = spec::build(*parsed, options); REQUIRE(pt);
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep);
  auto plan = estimate::prepare_std_lv_transport(*pt, *rep); REQUIRE(plan);
  CHECK(plan->scale_markers[0][0] == plan->scale_markers[1][0]);
  CHECK(plan->scale_markers[0][1] == plan->scale_markers[1][1]);
  auto old_con = estimate::build_eq_constraints(*pt); REQUIRE(old_con);
  auto new_con = estimate::build_eq_constraints(plan->source); REQUIRE(new_con);
  CHECK(old_con->n_alpha == new_con->n_alpha);
  options.group_equal = {spec::GroupEqual::LvCovariances};
  pt = spec::build(*parsed, options); REQUIRE(pt);
  rep = model::build_matrix_rep(*pt); REQUIRE(rep);
  auto unsupported = estimate::prepare_std_lv_transport(*pt, *rep);
  REQUIRE_FALSE(unsupported);
  CHECK(unsupported.error() == estimate::StartTransportIssue::MixedScaleForms);
}
