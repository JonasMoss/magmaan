#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "magmaan/api/conventions.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"

namespace {
using namespace magmaan;
parse::Op delta_fixture_op(const std::string& value) {
  for (auto op : {parse::Op::Measurement,parse::Op::Regression,parse::Op::Covariance,
      parse::Op::Threshold,parse::Op::ResponseScale,parse::Op::Intercept,parse::Op::EqConstraint})
    if (parse::to_string(op)==value) return op;
  return parse::Op::DefineParam;
}
compat::lavaan::LavaanParTable delta_fixture_partable(const nlohmann::json& rows) {
  compat::lavaan::LavaanParTable out;
  for (const auto& r : rows) {
    out.id.push_back(r["id"]); out.user.push_back(r["user"]);
    out.lhs.push_back(r["lhs"]); out.op.push_back(delta_fixture_op(r["op"])); out.rhs.push_back(r["rhs"]);
    out.block.push_back(r["block"]); out.group.push_back(r["group"]); out.free.push_back(r["free"]);
    out.exo.push_back(r["exo"]); out.label.push_back(r["label"]); out.plabel.push_back(r["plabel"]);
    out.ustart.push_back(r["ustart"].is_null() ? std::numeric_limits<double>::quiet_NaN() : r["ustart"].get<double>());
  }
  return out;
}
bool delta_close(double actual, double reference) {
  return std::abs(actual-reference)<=1e-5*(1+std::max(std::abs(actual),std::abs(reference)));
}
}

TEST_CASE("DELTA live scales match frozen lavaan rows estimates SEs and tests") {
  using namespace magmaan;
  auto raw=test::read_fixture(test::fixtures_dir()+"/ordinal/delta_scales.json");
  REQUIRE(raw);
  auto fixture=nlohmann::json::parse(*raw,nullptr,false); REQUIRE_FALSE(fixture.is_discarded());
  for (const auto& item : fixture["cases"].items()) {
    CAPTURE(item.key()); const auto& c=item.value();
    auto table=delta_fixture_partable(c["partable"]);
    auto triple=compat::lavaan::from_lavaan_partable(table);
    if (!c["group_equal"].is_null()) triple.structure.group_equal={spec::GroupEqual::Thresholds,spec::GroupEqual::Loadings};
    data::OrdinalStats stats;
    stats.n_obs=c["n_obs"].get<std::vector<std::int64_t>>();
    for (std::size_t b=0;b<c["R"].size();++b) {
      stats.R.push_back(test::matrix_from_json(c["R"][b]));
      stats.thresholds.push_back(test::vector_from_json(c["thresholds"][b]));
      stats.NACOV.push_back(test::matrix_from_json(c["nacov"][b]));
      stats.W_dwls.push_back(test::vector_from_json(c["weight"][b]).asDiagonal());
      stats.threshold_ov.push_back({0,0,1,1,2,2,3,3,4,4,5,5});
      stats.threshold_level.push_back({1,2,1,2,1,2,1,2,1,2,1,2});
      stats.n_levels.push_back({3,3,3,3,3,3});
    }
    auto& pt=triple.structure;
    REQUIRE(estimate::prepare_ordinal_partable(pt,stats,estimate::OrdinalParameterization::Delta,
        &triple.starts,&triple.names.row_user));
    auto rep=model::build_matrix_rep(pt,&triple.names); REQUIRE(rep);
    auto start=estimate::ordinal_start_values(pt,*rep,stats,triple.starts,&triple.names.row_user); REQUIRE(start);
    optim::OptimOptions options; options.max_iter=4000; options.ftol=1e-13; options.gtol=1e-8;
    auto est=estimate::fit_ordinal_bounded(pt,*rep,stats,{},estimate::OrdinalWeightKind::DWLS,*start,
        estimate::Backend::NloptLbfgs,options,estimate::OrdinalParameterization::Delta,&triple.names.row_user);
    REQUIRE(est); REQUIRE(api::policy_fit_state(*est).converged);
    if (item.key()=="equal_scales") {
      auto probe=*est;
      probe.theta.array() += 0.03;
      auto objective=estimate::frontier::ordinal_ls_objective(pt,*rep,stats,probe,estimate::OrdinalWeightKind::DWLS);
      auto parts=estimate::frontier::ordinal_ls_newton_parts(pt,*rep,stats,probe.theta,estimate::OrdinalWeightKind::DWLS);
      REQUIRE(objective); REQUIRE(parts);
      const double total=static_cast<double>(stats.n_obs[0]);
      Eigen::MatrixXd numeric(probe.theta.size(),probe.theta.size());
      for (Eigen::Index k=0;k<probe.theta.size();++k) {
        const double step=1e-5;
        auto plus=probe.theta, minus=probe.theta;
        plus(k)+=step; minus(k)-=step;
        auto ep=objective->problem.eval(plus), em=objective->problem.eval(minus);
        REQUIRE(ep); REQUIRE(em);
        numeric.col(k)=total*(ep->jacobian.transpose()*ep->residual-em->jacobian.transpose()*em->residual)/(2*step);
      }
      CHECK((parts->hessian-numeric).norm()/(1+numeric.norm())<1e-6);
    }
    auto moments=data::ordinal_moments_from_stats(stats);
    const auto plan=data::ordinal_weight_plan(data::OrdinalWorkspacePurpose::FitOnly,
        data::OrdinalEstimatorKind::ULS,data::OrdinalMomentParameterization::Delta);
    auto compact=estimate::fit_ordinal_bounded(pt,*rep,moments,nullptr,{},plan,*start,
        estimate::Backend::NloptLbfgs,options);
    auto gp=estimate::fit_ordinal_snlls(pt,*rep,moments,nullptr,plan,*start,
        estimate::Backend::NloptLbfgs,options);
    REQUIRE(compact); REQUIRE(gp);
    REQUIRE(api::policy_fit_state(*compact).converged);
    REQUIRE(api::policy_fit_state(*gp).converged);
    for (const auto* fitted : {&*compact,&*gp}) {
      auto projected=estimate::ordinal_parameter_values(pt,*rep,fitted->theta,estimate::OrdinalParameterization::Delta);
      REQUIRE(projected);
      for (std::size_t i=0;i<pt.size();++i) if(pt.group[i]>0)
        CHECK(delta_close((*projected)(static_cast<Eigen::Index>(i)),c["uls_est"][i]));
    }
    auto values=estimate::ordinal_parameter_values(pt,*rep,est->theta,estimate::OrdinalParameterization::Delta); REQUIRE(values);
    auto wlsmv=api::lavaan_inference_ordinal(pt,*rep,stats,*est,estimate::OrdinalWeightKind::DWLS,
        estimate::OrdinalParameterization::Delta,api::LavaanConvention::WLSMV,api::policy_fit_state(*est));
    REQUIRE(wlsmv.covariance_reason==api::InferenceReason::Available);
    REQUIRE(wlsmv.test.reason==api::InferenceReason::Available);
    CHECK(wlsmv.test.df==c["test"]["df"].get<int>());
    CHECK(delta_close(wlsmv.test.unscaled_statistic,c["test"]["chisq"]));
    CHECK(delta_close(wlsmv.test.statistic,c["test"]["chisq.scaled"]));
    CHECK(delta_close(wlsmv.test.scale,c["test"]["chisq.scaling.factor"]));
    CHECK(delta_close(wlsmv.test.shift,c["shift"]));
    auto dwls=api::lavaan_inference_ordinal(pt,*rep,stats,*est,estimate::OrdinalWeightKind::DWLS,
        estimate::OrdinalParameterization::Delta,api::LavaanConvention::DWLS,api::policy_fit_state(*est));
    REQUIRE(dwls.covariance_reason==api::InferenceReason::Available);
    for (std::size_t i=0;i<pt.size();++i) {
      if (pt.group[i]<=0) continue;
      CAPTURE(i); CAPTURE(table.lhs[i]); CAPTURE(table.rhs[i]);
      CHECK((pt.free[i]>0)==(table.free[i]>0));
      CHECK(triple.names.row_label[i]==table.label[i]);
      CHECK(delta_close((*values)(static_cast<Eigen::Index>(i)),c["partable"][i]["est"]));
      if (pt.free[i]>0) {
        CHECK(delta_close(std::sqrt(wlsmv.covariance(pt.free[i]-1,pt.free[i]-1)),c["partable"][i]["se"]));
        CHECK(delta_close(std::sqrt(dwls.covariance(pt.free[i]-1,pt.free[i]-1)),c["dwls_se"][i]));
      }
    }
  }
}
