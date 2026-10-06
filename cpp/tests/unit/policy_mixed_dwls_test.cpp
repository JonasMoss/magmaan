#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include <random>
#include <numeric>
#include "magmaan/api/policy.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace {
using namespace magmaan;
using estimate::OrdinalParameterization;
using estimate::OrdinalWeightKind;

// One factor with an omitted residual association: estimated weights matter
// at the population projection, rather than only through sampling residuals.
Eigen::MatrixXd mixed_block(unsigned seed, Eigen::Index n) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z;
  Eigen::MatrixXd x(n, 6);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double f = z(rng), nuisance = z(rng);
    for (int j = 0; j < 6; ++j) {
      const double loading = 0.8 - 0.04*j;
      const double shared = j < 2 ? 0.35 : 0.0;
      const double y = loading*f + shared*nuisance +
          std::sqrt(1-loading*loading-shared*shared)*z(rng);
      x(i,j) = j < 3 ? 1.0+(y > -0.5)+(y > 0.5) : y+0.1*j;
    }
  }
  return x;
}
struct MixedModel { spec::LatentStructure pt; model::MatrixRep rep; };
MixedModel mixed_model(int groups, bool restricted = false, bool marker_x2 = false) {
  std::string syntax =
      "f =~ x1 + x2 + x3 + x4 + x5 + x6\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\n";
  if (marker_x2) syntax.replace(0, syntax.find("\n"), "f =~ NA*x1 + 1*x2 + x3 + x4 + x5 + x6");
  if (restricted) syntax += "x4 ~~ 0.6*x4\n";
  auto parsed = parse::Parser::parse(syntax);
  REQUIRE(parsed.has_value());
  spec::BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = groups;
  auto pt = spec::build(*parsed, opts);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {*pt,*rep};
}
optim::OptimOptions tight() {
  optim::OptimOptions opts;
  opts.max_iter=3000; opts.ftol=1e-16; opts.gtol=1e-13;
  return opts;
}

}

TEST_CASE("Mixed DWLS policy: exact sampling global law and IJ nested law") {
  for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
    for (int groups : {1,2}) {
      CAPTURE(groups); CAPTURE(static_cast<int>(parameterization));
      std::vector<Eigen::MatrixXd> blocks;
      std::vector<std::vector<std::int32_t>> ordered(static_cast<std::size_t>(groups), {1,1,1,0,0,0});
      for (int g=0; g<groups; ++g) blocks.push_back(mixed_block(7100u+static_cast<unsigned>(g),400));
      auto stats=data::mixed_ordinal_stats_from_data(blocks,ordered,false);
      REQUIRE(stats);
      const auto m=mixed_model(groups), m0=mixed_model(groups,true);
      auto fit=test::fit_mixed_ordinal_bounded(m.pt,m.rep,*stats,{},OrdinalWeightKind::DWLS,
          estimate::Backend::NloptLbfgs,tight(),parameterization);
      auto fit0=test::fit_mixed_ordinal_bounded(m0.pt,m0.rep,*stats,{},OrdinalWeightKind::DWLS,
          estimate::Backend::NloptLbfgs,tight(),parameterization);
      REQUIRE(fit); REQUIRE(fit0);
      api::MixedDwlsPolicyFit cache(m.pt,m.rep,*stats,*fit,parameterization);
      const auto out=api::policy_inference_dwls(cache,{});
      REQUIRE_MESSAGE(out.covariance_reason==api::InferenceReason::Available,out.covariance_detail);
      REQUIRE_MESSAGE(out.score.reason==api::InferenceReason::Available,out.score.detail);
      auto ij=estimate::robust_mixed_ordinal_ij(m.pt,m.rep,*stats,*fit,OrdinalWeightKind::DWLS,parameterization);
      REQUIRE(ij);
      CHECK((out.covariance-ij->vcov).norm()<1e-12);
      CHECK(out.lr.reason==api::InferenceReason::Inapplicable);
      CHECK(out.score.reference=="all");
      auto exact=*stats;
      auto profile_stats=*stats;
      for (std::size_t b=0; b<blocks.size(); ++b) {
        auto rows=data::mixed_moment_sampling_influence(blocks[b],ordered[b],stats->n_levels[b],
            stats->thresholds[b],stats->mean[b],stats->R[b]);
        REQUIRE(rows);
        exact.NACOV[b]=rows->transpose()*(*rows)/static_cast<double>(rows->rows());
        profile_stats.moment_influence[b]=*rows;
      }
      auto global=estimate::robust_mixed_ordinal(m.pt,m.rep,exact,*fit,OrdinalWeightKind::DWLS,parameterization);
      auto opg=estimate::robust_mixed_ordinal(m.pt,m.rep,*stats,*fit,OrdinalWeightKind::DWLS,parameterization);
      REQUIRE(global); REQUIRE(opg);
      CHECK((out.score.eigenvalues-global->eigvals).norm()<1e-10);
      CHECK((out.score.eigenvalues-opg->eigvals).norm()>1e-3);
      const double N=std::accumulate(stats->n_obs.begin(),stats->n_obs.end(),0.0);
      CHECK(out.score.statistic==doctest::Approx(2*N*fit->fmin));
      api::MixedDwlsPolicyFit cache0(m0.pt,m0.rep,*stats,*fit0,parameterization);
      const auto nested=api::policy_nested_dwls(cache0,{},cache,{});
      REQUIRE_MESSAGE(nested.lr.reason==api::InferenceReason::Available,nested.lr.detail);
      CHECK(nested.score.reason==api::InferenceReason::UnsupportedModel);
      CHECK(nested.lr.label=="fit_function_difference");
      CHECK(nested.lr.reference == "all");
      CHECK(nested.lr.p_all == doctest::Approx(magmaan::robust::frontier::fmg_test(
          nested.lr.statistic, nested.lr.df, nested.lr.eigenvalues,
          {magmaan::robust::frontier::FmgMethod::All, 0.0, true}).p_value).epsilon(1e-12));

      auto p1=m.pt,p0=m0.pt;
      REQUIRE(estimate::prepare_mixed_ordinal_partable(p1,*stats,parameterization));
      REQUIRE(estimate::prepare_mixed_ordinal_partable(p0,*stats,parameterization));
      auto c1=estimate::build_eq_constraints(p1),c0=estimate::build_eq_constraints(p0);
      REQUIRE(c1); REQUIRE(c0);
      auto embedding=robust::embed_nested_null(p1,m.rep,p0,m0.rep,fit0->theta,*c1,*c0);
      REQUIRE(embedding);
      const auto common_null=robust::embedded_null_structure(p1,embedding->null_constraints);
      // The legacy lab profile consumes its explicit moment_influence channel.
      // Supply the same exact rows as the policy, preserving NACOV weights.
      auto common=estimate::mixed_ordinal_dwls_profile_lrt(p1,m.rep,profile_stats,*fit,
          common_null,m.rep,*fit,parameterization);
      REQUIRE_MESSAGE(common.has_value(),(common.has_value()? "" : common.error().detail));
      const Eigen::VectorXd eig=common->eigvals.tail(nested.lr.df);
      CHECK((nested.lr.eigenvalues-eig).norm()<=1e-10*eig.norm());
      // Independent saturated-moment projection: N Cov(weighted residuals)
      // has blocks W^(1/2) Gamma_exact W^(1/2); the fitted Jacobian supplies U.
      auto objective=estimate::frontier::mixed_ordinal_ls_objective(p1,m.rep,*stats,*fit,
          OrdinalWeightKind::DWLS,parameterization);
      REQUIRE(objective);
      auto jacobian=objective->problem.J(fit->theta);
      REQUIRE(jacobian);
      const Eigen::MatrixXd D=(*jacobian)*c1->K();
      const Eigen::MatrixXd P=Eigen::MatrixXd::Identity(D.rows(),D.rows())-
          D*(D.transpose()*D).ldlt().solve(D.transpose());
      Eigen::MatrixXd Omega=Eigen::MatrixXd::Zero(D.rows(),D.rows());
      Eigen::Index offset=0;
      for (std::size_t b=0; b<blocks.size(); ++b) {
        const Eigen::VectorXd root=stats->W_dwls[b].diagonal().array().sqrt();
        const Eigen::Index mb=root.size();
        Omega.block(offset,offset,mb,mb)=root.asDiagonal()*exact.NACOV[b]*root.asDiagonal();
        offset+=mb;
      }
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> global_eigen(P*Omega*P);
      REQUIRE(global_eigen.info()==Eigen::Success);
      CHECK((out.score.eigenvalues-global_eigen.eigenvalues().tail(out.score.df)).norm()<1e-10);
      auto tangent=api::frontier::moment_nested_tangent(m0.pt,m0.rep,*fit0,m.pt,m.rep,*fit,*stats,parameterization);
      REQUIRE_MESSAGE(tangent.has_value(),(tangent.has_value()? "" : tangent.error().detail));
      auto parts=estimate::frontier::mixed_ordinal_ls_newton_parts_prepared(p1,m.rep,*stats,
          fit->theta,OrdinalWeightKind::DWLS,parameterization);
      REQUIRE(parts);
      const auto& K=c1->K();
      const Eigen::MatrixXd H=K.transpose()*parts->hessian*K/N;
      const Eigen::MatrixXd L=(K.transpose()*K).ldlt().solve(K.transpose());
      const Eigen::MatrixXd V=N*L*ij->vcov*L.transpose();
      auto direct=robust::compute_satorra2000_from_sandwich(H,H*V*H.transpose(),embedding->restriction.A);
      REQUIRE(direct);
      CHECK((nested.lr.eigenvalues-direct->eigenvalues).norm()<=1e-10*eig.norm());
      auto moment=robust::compute_satorra2000_from_sandwich(H,H*V*H.transpose(),tangent->A);
      REQUIRE(moment);
      CHECK((nested.lr.eigenvalues-moment->eigenvalues).norm()<=1e-10*eig.norm());
      // Moving the marker preserves implied moments but changes parameter
      // slots; the mixed policy must use its moment-nesting fallback.
      const auto reparameterized=mixed_model(groups,true,true);
      auto fit_reparameterized=test::fit_mixed_ordinal_bounded(reparameterized.pt,
          reparameterized.rep,*stats,{},OrdinalWeightKind::DWLS,
          estimate::Backend::NloptLbfgs,tight(),parameterization);
      REQUIRE(fit_reparameterized);
      const auto moment_nested=api::policy_nested_dwls(reparameterized.pt,
          reparameterized.rep,*fit_reparameterized,{},m.pt,m.rep,*fit,{},*stats,parameterization);
      REQUIRE_MESSAGE(moment_nested.lr.reason==api::InferenceReason::Available,moment_nested.lr.detail);
      CHECK(moment_nested.lr.reference == "all");
      CHECK(moment_nested.lr.p_all == doctest::Approx(magmaan::robust::frontier::fmg_test(
          moment_nested.lr.statistic, moment_nested.lr.df, moment_nested.lr.eigenvalues,
          {magmaan::robust::frontier::FmgMethod::All, 0.0, true}).p_value).epsilon(1e-12));
      CHECK(moment_nested.lr.df==nested.lr.df);
      CHECK(moment_nested.lr.statistic==doctest::Approx(nested.lr.statistic).epsilon(1e-7));
      CHECK((moment_nested.lr.eigenvalues-nested.lr.eigenvalues).norm()<1e-10);
      CHECK(api::policy_ingredient_builds(cache)==2);
      api::policy_inference_dwls(cache,{});
      api::policy_nested_dwls(cache0,{},cache,{});
      CHECK(api::policy_ingredient_builds(cache)==2);
      auto failed=api::PolicyFitState{}; failed.converged=false;
      CHECK(api::policy_inference_dwls(cache,failed).score.reason==api::InferenceReason::NotConverged);
      CHECK(api::policy_nested_dwls(cache0,failed,cache,{}).lr.reason==api::InferenceReason::NotConverged);
      CHECK(api::policy_nested_dwls(cache,{},cache,{}).lr.reason==api::InferenceReason::EquivalentModels);
      auto supplied=*stats; supplied.W_dwls[0]*=2.0;
      CHECK(api::policy_inference_dwls(m.pt,m.rep,supplied,*fit,parameterization,{}).score.reason==api::InferenceReason::UnsupportedModel);
      auto missing=*stats; missing.raw_data.clear();
      CHECK(api::policy_inference_dwls(m.pt,m.rep,missing,*fit,parameterization,{}).score.reason==api::InferenceReason::UnsupportedModel);
    }
  }
}

TEST_CASE("Mixed DWLS policy fit measures: exact primitive points") {
  auto stats = data::mixed_ordinal_stats_from_data({mixed_block(101u,400)},{{1,1,1,0,0,0}},false);
  REQUIRE(stats);
  const auto m = mixed_model(1);
  auto est = test::fit_mixed_ordinal_bounded(m.pt,m.rep,*stats,{},OrdinalWeightKind::DWLS,
      estimate::Backend::NloptLbfgs,tight(),OrdinalParameterization::Delta);
  REQUIRE(est);
  auto policy = api::policy_fit_measures(m.pt,m.rep,*stats,*est,{});
  const auto exact = estimate::OrdinalFirstStage::Exact;
  auto rm = estimate::mixed_ordinal_rmsea_misspec_inference(m.pt,m.rep,*stats,*est,
      OrdinalParameterization::Delta,true,0.9,1e-10,exact);
  auto ct = estimate::mixed_ordinal_cfi_tli_misspec_inference(m.pt,m.rep,*stats,*est,
      OrdinalParameterization::Delta,true,0.9,1e-10,exact);
  auto cr = estimate::mixed_ordinal_crmr_misspec_inference(m.pt,m.rep,*stats,*est,
      OrdinalParameterization::Delta,true,false,0.9,1e-10,exact);
  auto sr = estimate::mixed_ordinal_crmr_misspec_inference(m.pt,m.rep,*stats,*est,
      OrdinalParameterization::Delta,true,true,0.9,1e-10,exact);
  REQUIRE(rm); REQUIRE(ct); REQUIRE(cr); REQUIRE(sr);
  CHECK(policy.user.trace == rm->bias_trace);
  CHECK(policy.baseline.trace == ct->gendf_baseline);
  for (const auto& i : policy.indices) {
    REQUIRE_MESSAGE(i.reason == api::InferenceReason::Available,i.detail);
    if (i.index == "rmsea") CHECK(i.estimate == rm->point);
    if (i.index == "cfi") CHECK(i.estimate == ct->cfi);
    if (i.index == "crmr") CHECK(i.estimate == cr->point_bias_corrected);
    if (i.index == "srmr") CHECK(i.estimate == sr->point_bias_corrected);
  }
}
