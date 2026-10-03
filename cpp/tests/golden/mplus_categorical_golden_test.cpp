#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "magmaan/api/conventions.hpp"
#include "magmaan/compat/mplus/model.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"

namespace {
bool categorical_close(double actual,double reference) {
  return std::abs(actual-reference)<=1e-5*(1+std::max(std::abs(actual),std::abs(reference)));
}
}
TEST_CASE("Mplus categorical lowering matches independent frozen lavaan models") {
  using namespace magmaan;
  auto raw=test::read_fixture(test::fixtures_dir()+"/mplus/golden_categorical.json");REQUIRE(raw);
  auto fixture=nlohmann::json::parse(*raw,nullptr,false);REQUIRE_FALSE(fixture.is_discarded());
  for(const auto& item:fixture["cases"].items()) {
    CAPTURE(item.key());const auto& c=item.value();
    auto counts=c["category_counts"].get<std::vector<std::vector<std::int32_t>>>();
    auto triple=compat::mplus::prepare_ordinal_model(c["input"].get<std::string>(),counts);REQUIRE(triple);
    auto par=c["parameterization"]=="theta" ? estimate::OrdinalParameterization::Theta : estimate::OrdinalParameterization::Delta;
    auto& pt=triple->structure;
    data::OrdinalStats stats;stats.n_obs=c["n_obs"].get<std::vector<std::int64_t>>();
    for(std::size_t b=0;b<counts.size();++b) {
      stats.R.push_back(test::matrix_from_json(c["R"][b]));
      stats.thresholds.push_back(test::vector_from_json(c["thresholds"][b]));
      stats.NACOV.push_back(test::matrix_from_json(c["nacov"][b]));
      stats.W_dwls.push_back(test::vector_from_json(c["weight"][b]).asDiagonal());
      std::vector<std::int32_t> ov,level;
      for(std::int32_t j=0;j<6;++j) for(std::int32_t t=1;t<counts[b][static_cast<std::size_t>(j)];++t) {ov.push_back(j);level.push_back(t);}
      stats.threshold_ov.push_back(ov);stats.threshold_level.push_back(level);stats.n_levels.push_back(counts[b]);
    }
    auto rep=model::build_matrix_rep(pt,&triple->names);REQUIRE(rep);
    auto start=estimate::ordinal_start_values(pt,*rep,stats,triple->starts,&triple->names.row_user,par);REQUIRE(start);
    optim::OptimOptions options;options.max_iter=4000;options.ftol=1e-13;options.gtol=1e-8;
    auto est=estimate::fit_ordinal_bounded(pt,*rep,stats,{},estimate::OrdinalWeightKind::DWLS,*start,estimate::Backend::NloptLbfgs,options,par,&triple->names.row_user);REQUIRE(est);REQUIRE(api::policy_fit_state(*est).converged);
    auto values=estimate::ordinal_parameter_values(pt,*rep,est->theta,par);REQUIRE(values);
    auto table=compat::lavaan::to_lavaan_partable(pt,triple->names,triple->starts);
    auto inference=api::lavaan_inference_ordinal(pt,*rep,stats,*est,estimate::OrdinalWeightKind::DWLS,par,api::LavaanConvention::WLSMV,api::policy_fit_state(*est));
    REQUIRE(inference.covariance_reason==api::InferenceReason::Available);REQUIRE(inference.test.reason==api::InferenceReason::Available);
    CHECK(inference.test.df==c["test"]["df"].get<int>());
    CHECK(categorical_close(inference.test.unscaled_statistic,c["test"]["chisq"]));
    CHECK(categorical_close(inference.test.statistic,c["test"]["chisq.scaled"]));
    CHECK(categorical_close(inference.test.scale,c["test"]["chisq.scaling.factor"]));
    CHECK(categorical_close(inference.test.shift,c["shift"]));
    std::size_t matched=0;
    for(std::size_t i=0;i<table.size();++i) {
      if(table.group[i]<=0) continue;
      const auto op=std::string(parse::to_string(table.op[i]));
      auto row=std::find_if(c["partable"].begin(),c["partable"].end(),[&](const auto& r){return r["lhs"]==table.lhs[i] && r["op"]==op && r["rhs"]==table.rhs[i] && r["group"]==table.group[i];});
      // lavaan lists the lower triangle; frontend preserves source ordering.
      if(row==c["partable"].end() && table.op[i]==parse::Op::Covariance) row=std::find_if(c["partable"].begin(),c["partable"].end(),[&](const auto& r){return r["lhs"]==table.rhs[i] && r["op"]==op && r["rhs"]==table.lhs[i] && r["group"]==table.group[i];});
      CAPTURE(i);CAPTURE(table.lhs[i]);CAPTURE(op);REQUIRE(row!=c["partable"].end());++matched;
      CHECK((table.free[i]>0)==((*row)["free"].get<int>()>0));
      CHECK(categorical_close((*values)(static_cast<Eigen::Index>(i)),(*row)["est"]));
      if(table.free[i]>0) CHECK(categorical_close(std::sqrt(inference.covariance(table.free[i]-1,table.free[i]-1)),(*row)["se"]));
    }
    const auto expected=std::count_if(c["partable"].begin(),c["partable"].end(),[](const auto& r){return r["group"].template get<int>()>0;});
    CHECK(matched==static_cast<std::size_t>(expected));
  }
}
