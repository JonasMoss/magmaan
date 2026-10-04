#include <doctest/doctest.h>
#include <Eigen/LU>
#include <numeric>
#include "../oracle.hpp"
#include "magmaan/api/conventions.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/compat/mplus/model.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"

namespace {
using namespace magmaan;
parse::Op nonlinear_fixture_op(const std::string& s) {
  for (auto op : {parse::Op::Measurement, parse::Op::Regression, parse::Op::Covariance,
      parse::Op::Threshold, parse::Op::ResponseScale, parse::Op::Intercept, parse::Op::EqConstraint})
    if (parse::to_string(op) == s) return op;
  return parse::Op::DefineParam;
}
compat::lavaan::LavaanParTable nonlinear_fixture_table(const nlohmann::json& rows) {
  compat::lavaan::LavaanParTable out;
  for (const auto& r : rows) {
    out.id.push_back(r["id"]); out.user.push_back(r["user"]);
    out.lhs.push_back(r["lhs"]); out.op.push_back(nonlinear_fixture_op(r["op"])); out.rhs.push_back(r["rhs"]);
    out.block.push_back(r["block"]); out.group.push_back(r["group"]); out.free.push_back(r["free"]);
    out.exo.push_back(r["exo"]); out.label.push_back(r["label"]); out.plabel.push_back(r["plabel"]);
    out.ustart.push_back(r["ustart"].is_null() ? std::numeric_limits<double>::quiet_NaN() : r["ustart"].get<double>());
  }
  return out;
}
data::OrdinalStats nonlinear_fixture_stats(const nlohmann::json& c) {
  data::OrdinalStats stats;
  stats.n_obs = c["n_obs"].get<std::vector<std::int64_t>>();
  for (std::size_t b=0; b<c["R"].size(); ++b) {
    stats.R.push_back(test::matrix_from_json(c["R"][b]));
    stats.thresholds.push_back(test::vector_from_json(c["thresholds"][b]));
    stats.NACOV.push_back(test::matrix_from_json(c["nacov"][b]));
    stats.W_dwls.push_back(test::vector_from_json(c["weight"][b]).asDiagonal());
    stats.W_wls.push_back(stats.NACOV.back().inverse());
    auto counts = c["category_counts"][b].get<std::vector<std::int32_t>>();
    std::vector<std::int32_t> ov, levels;
    for (std::int32_t j=0; j<static_cast<std::int32_t>(counts.size()); ++j)
      for (std::int32_t t=1; t<counts[static_cast<std::size_t>(j)]; ++t) { ov.push_back(j); levels.push_back(t); }
    stats.threshold_ov.push_back(ov); stats.threshold_level.push_back(levels); stats.n_levels.push_back(counts);
  }
  return stats;
}
bool nonlinear_close(double a, double b) {
  return std::abs(a-b) <= 1e-5*(1+std::max(std::abs(a),std::abs(b)));
}
}

TEST_CASE("Ordinal nonlinear equalities match frozen lavaan estimates covariance tests and df") {
  using namespace magmaan;
  auto raw=test::read_fixture(test::fixtures_dir()+"/ordinal/nonlinear_equalities.json"); REQUIRE(raw);
  auto fixture=nlohmann::json::parse(*raw,nullptr,false); REQUIRE_FALSE(fixture.is_discarded());
  for (const auto& item : fixture["cases"].items()) {
    CAPTURE(item.key()); const auto& c=item.value();
    const auto par=c["parameterization"]=="theta" ? estimate::OrdinalParameterization::Theta : estimate::OrdinalParameterization::Delta;
    auto stats=nonlinear_fixture_stats(c);
    for (const auto& fit : c["fits"].items()) {
      CAPTURE(fit.key()); const auto& reference=fit.value();
      auto table=nonlinear_fixture_table(reference["partable"]);
      auto triple=compat::lavaan::from_lavaan_partable(table);
      auto& pt=triple.structure;
      REQUIRE(estimate::prepare_ordinal_partable(pt,stats,par,&triple.starts,&triple.names.row_user));
      auto rep=model::build_matrix_rep(pt,&triple.names); REQUIRE(rep);
      auto start=estimate::ordinal_start_values(pt,*rep,stats,triple.starts,&triple.names.row_user,par); REQUIRE(start);
      auto kind=fit.key().starts_with("ULS") ? estimate::OrdinalWeightKind::ULS : fit.key()=="WLS" ? estimate::OrdinalWeightKind::WLS : estimate::OrdinalWeightKind::DWLS;
      optim::OptimOptions options; options.max_iter=4000; options.ftol=1e-13; options.gtol=1e-8;
      auto est=estimate::fit_ordinal_bounded(pt,*rep,stats,{},kind,*start,estimate::Backend::NloptLbfgs,options,par,&triple.names.row_user);
      REQUIRE_MESSAGE(est.has_value(), (est ? "" : est.error().detail));
      CHECK(estimate::build_nl_constraints(pt).h(est->theta).lpNorm<Eigen::Infinity>() < 1e-8);
      auto values=estimate::ordinal_parameter_values(pt,*rep,est->theta,par); REQUIRE(values);
      auto convention=fit.key()=="WLSMV" ? api::LavaanConvention::WLSMV : fit.key()=="DWLS" ? api::LavaanConvention::DWLS : fit.key()=="ULSMV" ? api::LavaanConvention::ULSMV : fit.key()=="ULS" ? api::LavaanConvention::ULS : api::LavaanConvention::WLS;
      auto inference=api::lavaan_inference_ordinal(pt,*rep,stats,*est,kind,par,convention,api::policy_fit_state(*est));
      REQUIRE_MESSAGE(inference.covariance_reason==api::InferenceReason::Available,inference.covariance_detail);
      REQUIRE_MESSAGE(inference.test.reason==api::InferenceReason::Available,inference.test.detail);
      CHECK(inference.test.df==reference["test"]["df"].get<int>());
      CHECK(nonlinear_close(inference.test.unscaled_statistic,reference["test"]["chisq"]));
      auto df=estimate::ordinal_df_stat(pt,stats,est->theta); REQUIRE(df); CHECK(*df==inference.test.df);
      if (fit.key()=="WLSMV" || fit.key()=="ULSMV") {
        CHECK(nonlinear_close(inference.test.statistic,reference["test"]["chisq.scaled"]));
        CHECK(nonlinear_close(inference.test.scale,reference["test"]["chisq.scaling.factor"]));
        CHECK(nonlinear_close(inference.test.shift,reference["shift"]));
      }
      for (std::size_t i=0; i<pt.size(); ++i) {
        if (pt.group[i]<=0) continue;
        CAPTURE(i); CAPTURE(table.lhs[i]); CAPTURE(table.rhs[i]);
        CHECK(nonlinear_close((*values)(static_cast<Eigen::Index>(i)),reference["partable"][i]["est"]));
        if (pt.free[i]>0) CHECK(nonlinear_close(std::sqrt(inference.covariance(pt.free[i]-1,pt.free[i]-1)),reference["partable"][i]["se"]));
      }
      auto tangent=estimate::build_eq_tangent(pt,est->theta); REQUIRE(tangent);
      CHECK((estimate::build_nl_constraints(pt).jacobian(est->theta)*tangent->K()).norm()<1e-9);
      auto observed=estimate::robust_ordinal(pt,*rep,stats,*est,kind,par,robust::Information::Observed,&triple.names.row_user);
      REQUIRE_FALSE(observed); CHECK(observed.error().kind==PostError::Kind::UnsupportedInference);
      auto policy=api::policy_inference_dwls(pt,*rep,stats,*est,par,api::policy_fit_state(*est),&triple.names.row_user);
      CHECK(policy.covariance_reason==api::InferenceReason::UnsupportedModel);
      if (fit.key()=="WLSMV") {
        auto scores=estimate::score_tests_ordinal(pt,*rep,stats,*est,kind,par,&triple.names.row_user); REQUIRE(scores);
        REQUIRE(scores->rows.size()==reference["score"].size());
        // Core scores use N*F/2. Lavaan divides the score (not its expected
        // information) by N/(N-G) for these equal-size ordinal groups.
        const double n=std::accumulate(stats.n_obs.begin(),stats.n_obs.end(),0.0);
        const double divisor=(n-static_cast<double>(stats.n_obs.size()))/n;
        std::vector<double> actual, expected;
        for (const auto& row : scores->rows) actual.push_back(row.mi*divisor*divisor);
        for (const auto& value : reference["score"]) expected.push_back(value.get<double>());
        std::sort(actual.begin(),actual.end()); std::sort(expected.begin(),expected.end());
        for (std::size_t j=0;j<actual.size();++j) CHECK(nonlinear_close(actual[j],expected[j]));
        if (reference.contains("null_partable")) {
          auto unrestricted=compat::lavaan::from_lavaan_partable(nonlinear_fixture_table(reference["null_partable"]));
          REQUIRE(estimate::prepare_ordinal_partable(unrestricted.structure,stats,par,&unrestricted.starts,&unrestricted.names.row_user));
          auto r1=model::build_matrix_rep(unrestricted.structure,&unrestricted.names); REQUIRE(r1);
          auto x1=estimate::ordinal_start_values(unrestricted.structure,*r1,stats,unrestricted.starts,&unrestricted.names.row_user,par); REQUIRE(x1);
          auto e1=estimate::fit_ordinal_bounded(unrestricted.structure,*r1,stats,{},kind,*x1,estimate::Backend::NloptLbfgs,options,par,&unrestricted.names.row_user); REQUIRE(e1);
          auto nested=api::lavaan_nested_ordinal(pt,*rep,*est,api::policy_fit_state(*est),
              unrestricted.structure,*r1,*e1,api::policy_fit_state(*e1),stats,kind,par,api::LavaanConvention::WLSMV,
              &triple.names.row_user,&unrestricted.names.row_user);
          REQUIRE_MESSAGE(nested.reason==api::InferenceReason::Available,nested.detail);
          CHECK(nested.df==reference["nested"]["Df diff"].get<int>());
          CAPTURE(nested.statistic); CAPTURE(reference["nested"]["Chisq diff"]);
          CHECK(nonlinear_close(nested.statistic,reference["nested"]["Chisq diff"]));
        }
        auto fm=estimate::fit_measures_ordinal(pt,*rep,stats,*est,kind,par); REQUIRE(fm);
        if (item.key()=="square_delta") {
          estimate::FittingOptions fitting; fitting.preset="lavaan-0.7.2";
          auto configured=estimate::fit_ordinal_configured(pt,*rep,stats,fitting,triple.starts,{}, {},kind,par,&triple.names.row_user);
          REQUIRE_MESSAGE(configured.has_value(), (configured ? "" : configured.error().detail));
          CHECK((configured->theta-est->theta).norm()<1e-5);
          REQUIRE(configured->fitting);
          CHECK(configured->fitting->setup.optimizer=="nlopt-slsqp");
          CHECK(configured->fitting->setup.modified_preset);
        }
        auto moments=data::ordinal_moments_from_stats(stats);
        auto plan=data::ordinal_weight_plan(data::OrdinalWorkspacePurpose::FitOnly,data::OrdinalEstimatorKind::DWLS,
            par==estimate::OrdinalParameterization::Theta ? data::OrdinalMomentParameterization::Theta : data::OrdinalMomentParameterization::Delta);
        data::OrdinalGammaCache cache;
        cache.blocks.resize(stats.NACOV.size());
        for (std::size_t b=0; b<stats.NACOV.size(); ++b) { cache.blocks[b].gamma=stats.NACOV[b]; cache.blocks[b].has_full=true; }
        auto compact=estimate::fit_ordinal_bounded(pt,*rep,moments,&cache,{},plan,*start,estimate::Backend::NloptLbfgs,options);
        REQUIRE_MESSAGE(compact.has_value(), (compact ? "" : compact.error().detail));
        CHECK((compact->theta-est->theta).norm()<1e-5);
      }
    }
    if (c.contains("mplus")) {
      auto m=compat::mplus::prepare_ordinal_model(c["mplus"].get<std::string>(),c["category_counts"].get<std::vector<std::vector<std::int32_t>>>());
      REQUIRE_MESSAGE(m.has_value(), (m ? "" : m.error().detail));

      auto rep=model::build_matrix_rep(m->structure,&m->names); REQUIRE(rep);
      auto start=estimate::ordinal_start_values(m->structure,*rep,stats,m->starts,&m->names.row_user,par); REQUIRE(start);
      auto est=estimate::fit_ordinal_bounded(m->structure,*rep,stats,{},estimate::OrdinalWeightKind::DWLS,*start,estimate::Backend::NloptSlsqp,{},par,&m->names.row_user); REQUIRE(est);
      auto tangent=estimate::build_eq_tangent(m->structure,est->theta); REQUIRE(tangent);
      CHECK(tangent->n_alpha==c["demo"]["npar"].get<int>());
      auto inf=api::lavaan_inference_ordinal(m->structure,*rep,stats,*est,estimate::OrdinalWeightKind::DWLS,par,api::LavaanConvention::WLSMV,api::policy_fit_state(*est));
      REQUIRE_MESSAGE(inf.test.reason==api::InferenceReason::Available,inf.test.detail);
      CHECK(inf.test.df==c["demo"]["df"].get<int>());
      CHECK(nonlinear_close(inf.test.statistic,c["fits"]["WLSMV"]["test"]["chisq.scaled"]));
      auto values=estimate::ordinal_parameter_values(m->structure,*rep,est->theta,par); REQUIRE(values);
      auto table=compat::lavaan::to_lavaan_partable(m->structure,m->names,m->starts);
      const auto& rows=c["fits"]["WLSMV"]["partable"];
      for (std::size_t i=0;i<table.size();++i) {
        if (table.group[i]<=0) continue;
        auto row=std::find_if(rows.begin(),rows.end(),[&](const auto& r) {
          return r["lhs"]==table.lhs[i] && r["op"]==parse::to_string(table.op[i]) &&
                 r["rhs"]==table.rhs[i] && r["group"]==table.group[i];
        });
        REQUIRE(row!=rows.end());
        CHECK((table.free[i]>0)==((*row)["free"].get<int>()>0));
        CHECK(nonlinear_close((*values)(static_cast<Eigen::Index>(i)),(*row)["est"]));
        if (table.free[i]>0) CHECK(nonlinear_close(std::sqrt(inf.covariance(table.free[i]-1,table.free[i]-1)),(*row)["se"]));
      }

    }
  }
}
