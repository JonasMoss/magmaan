#include <doctest/doctest.h>
#include "magmaan/estimate/ml_numerics.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
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
  auto start=estimate::ml_start_values(c.pt,c.rep,c.sample); REQUIRE(start);
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
  auto other=estimate::ml_start_values(c.pt,c.rep,scaled); REQUIRE(other);
  auto v=me->evaluate(other->theta,false,false); REQUIRE(v);
  CHECK((v->moments.sigma[0]-100*b->moments.sigma[0]).norm()<1e-9);
  spec::Starts hints; hints.hint.resize(static_cast<std::size_t>(c.pt.n_free()),std::numeric_limits<double>::quiet_NaN());
  hints.hint[0]=.314159;
  auto explicit_hint=estimate::ml_start_values(c.pt,c.rep,c.sample,hints); REQUIRE(explicit_hint);
  CHECK(explicit_hint->theta(0)==.314159);
}
TEST_CASE("ML numerical policy falls back for fixed values and equalities") {
  for (const char* syntax : {"f =~ x1 + 2*x2 + x3 + x4", "f =~ x1 + a*x2 + a*x3 + x4", "f =~ x1 + a*x2 + b*x3 + x4\na == b*b"}) {
    auto c=example(syntax);
    auto start=estimate::ml_start_values(c.pt,c.rep,c.sample); REQUIRE(start);
    auto native=estimate::fabin_start_values(c.pt,c.rep,c.sample); REQUIRE(native);
    CHECK(start->branch==estimate::MlStartBranch::NativeFabin);
    CHECK((start->theta-*native).norm()==0);
  }
}
TEST_CASE("ML numerical policy preserves fitted objective under scaling and bounds") {
  for (const char* syntax : {"f =~ x1 + x2 + x3 + x4", "f =~ x1 + a*x2 + a*x3 + x4"}) {
    auto c=example(syntax);
    for(double units : {.1,1.,10.}) {
      auto sample=c.sample; sample.S[0]*=units*units;
      auto start=estimate::ml_start_values(c.pt,c.rep,sample); REQUIRE(start);
      auto opts=estimate::ml_optim_options();
      estimate::Bounds bounds;
      bounds.lower=Eigen::VectorXd::Constant(c.pt.n_free(),-.5*units*units);
      bounds.upper=Eigen::VectorXd::Constant(c.pt.n_free(),1000.);
      auto scaled=estimate::fit_ml(c.pt,c.rep,sample,start->theta,bounds,estimate::Backend::NloptSlsqp,opts); REQUIRE(scaled);
      CHECK(scaled->ml_sample_scaling_applied);
      CHECK((scaled->theta.array()>=bounds.lower.array()-1e-8).all());
      opts.ml_sample_scaling=false;
      auto reference=estimate::fit_ml(c.pt,c.rep,sample,scaled->theta,bounds,estimate::Backend::NloptSlsqp,opts); REQUIRE(reference);
      CHECK_FALSE(reference->ml_sample_scaling_applied);
      CHECK(std::abs(scaled->fmin-reference->fmin)<1e-8);
      CHECK(scaled->diagnostics.lin_eq_residual_inf<1e-8);
    }
  }
}
TEST_CASE("ML numerical policy defaults and opt-out are separate from generic options") {
  auto opts=estimate::ml_optim_options();
  CHECK(*opts.nlopt.ftol_rel==1e-12); CHECK(*opts.nlopt.xtol_rel==1e-10); CHECK(*opts.nlopt.max_eval==5000);
  CHECK(opts.ml_sample_scaling); CHECK_FALSE(optim::OptimOptions{}.ml_sample_scaling);
  CHECK(estimate::frontier::ml_psd_options().diagonal_preconditioning);
  CHECK(*estimate::frontier::ml_psd_optim_options().nlopt.constraint_tol==1e-8);
  auto c=example(); c.sample.S[0](0,0)=0;
  auto con=estimate::build_eq_constraints(c.pt); REQUIRE(con);
  CHECK_FALSE(estimate::ml_coordinate_scale(c.pt,c.rep,*con,c.sample));
}

TEST_CASE("ML numerical policy retains constrained adapters") {
  for (const char* syntax : {"f =~ x1 + a*x2 + b*x3 + x4\na == b*b",
                             "f =~ x1 + a*x2 + b*x3 + x4\na + b == 2"}) {
    auto c=example(syntax);
    auto start=estimate::ml_start_values(c.pt,c.rep,c.sample); REQUIRE(start);
    estimate::Bounds bounds;
    bounds.lower=Eigen::VectorXd::Constant(c.pt.n_free(),-10.);
    bounds.upper=Eigen::VectorXd::Constant(c.pt.n_free(),10.);
    auto fit=estimate::fit_ml(c.pt,c.rep,c.sample,start->theta,bounds,
                             estimate::Backend::NloptSlsqp);
    REQUIRE(fit);
    CHECK_FALSE(fit->ml_sample_scaling_applied);
    CHECK(fit->diagnostics.lin_eq_satisfied);
    CHECK(fit->diagnostics.nl_eq_satisfied);
  }
}
