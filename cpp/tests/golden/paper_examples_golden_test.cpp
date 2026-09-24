#include <doctest/doctest.h>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "../oracle.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace {
using namespace magmaan;
using test::matrix_from_json;
using test::vector_from_json;
const std::vector<std::string> cases = {
  "boivin_hierarchical", "boivin_two_factor", "boivin_crossloadings",
  "boivin_equal_crossloadings", "boivin_equal_primary_crossloading",
  "pregnancy_mediation", "kievit_ulcs", "kievit_milcs", "kievit_blcs",
  "kievit_bdcs", "kievit_mg_ulcs", "kievit_milcs_missing",
  "jiwani_dass_1f_cfa", "jiwani_dass_3f_cfa"};
parse::Op op_from_string(const std::string &value) {
  for (unsigned i=0;i<=static_cast<unsigned>(parse::Op::Composite);++i) {
    auto op=static_cast<parse::Op>(i);
    if (parse::to_string(op)==value) return op;
  }
  FAIL("Unknown partable operator: " << value);
  return parse::Op::Covariance;
}
compat::lavaan::ParsedLavaanParTable model_from_json(const nlohmann::json &j) {
  compat::lavaan::LavaanParTable pt;
  for (const auto &r:j["partable"]) {
    pt.id.push_back(r["id"]);
    pt.user.push_back(r["user"]);
    pt.lhs.push_back(r["lhs"]);
    pt.rhs.push_back(r["rhs"]);
    pt.op.push_back(op_from_string(r["op"]));
    pt.block.push_back(r["block"]);
    pt.group.push_back(r["group"]);
    pt.free.push_back(r["free"]);
    pt.exo.push_back(r["exo"]);
    pt.ustart.push_back(r["ustart"].is_null()?std::numeric_limits<double>::quiet_NaN():r["ustart"].get<double>());
    pt.label.push_back(r["label"]);
    pt.plabel.push_back(r["plabel"]);
  }
  return compat::lavaan::from_lavaan_partable(pt);
}
data::OrdinalStats ordinal_from_json(const nlohmann::json &j) {
  data::OrdinalStats s;
  for (const auto &b:j["blocks"]) {
    s.R.push_back(matrix_from_json(b["R"]));
    s.thresholds.push_back(vector_from_json(b["thresholds"]));
    s.threshold_ov.push_back(b["threshold_ov"].get<std::vector<std::int32_t>>());
    s.threshold_level.push_back(b["threshold_level"].get<std::vector<std::int32_t>>());
    s.n_levels.push_back(b["n_levels"].get<std::vector<std::int32_t>>());
    s.NACOV.push_back(matrix_from_json(b["NACOV"]));
    s.W_dwls.push_back(matrix_from_json(b["W"]));
    s.n_obs.push_back(b["n"]);
    s.ov_names.push_back(j["observed_variables"].get<std::vector<std::string>>());
  }
  return s;
}
estimate::fiml::FIMLPack pack_from_json(const nlohmann::json &j) {
  estimate::fiml::FIMLPack pack;
  Eigen::Index so=0,mo=0;
  for (std::size_t b=0;b<j["blocks"].size();++b) {
    const auto &block=j["blocks"][b];
    auto cov=matrix_from_json(block["cov"]);
    const auto p=cov.rows();
    pack.start_stats.S.push_back(cov);
    pack.start_stats.mean.push_back(vector_from_json(block["mean"]));
    pack.start_stats.n_obs.push_back(block["n"]);
    pack.cache.n_total+=block["n"].get<std::int64_t>();
    pack.cache.block_p.push_back(p);
    pack.cache.sigma_offsets.push_back(so);
    so+=p*(p+1)/2;
    pack.cache.mu_offsets.push_back(mo);
    mo+=p;
    for (const auto &pat:block["patterns"]) {
      estimate::fiml::FIMLPattern row;
      row.block=b;
      row.n_obs=pat["n"];
      row.observed=pat["observed"].get<std::vector<Eigen::Index>>();
      row.mean=vector_from_json(pat["mean"]);
      row.cov=matrix_from_json(pat["cov"]);
      pack.cache.patterns.push_back(std::move(row));
    }
  }
  return pack;
}
void compare_mi(const inference::ScoreTestTable &table,const spec::LatentNames &names,
const nlohmann::json &refs, double gradient_scale) {
  for (const auto &ref:refs) {
    CAPTURE(ref.dump());
    const inference::ScoreTestResult *found=nullptr;
    for (const auto &row:table.rows) {
      const auto &c=row.candidate;
      if (c.kind!=inference::ScoreCandidateKind::FixedParam) continue;
      auto lhs=c.row<names.row_lhs.size()?names.row_lhs[c.row]:names.var_name[static_cast<std::size_t>(c.lhs_var)];
      auto rhs=c.row<names.row_rhs.size()?names.row_rhs[c.row]:names.var_name[static_cast<std::size_t>(c.rhs_var)];
      if (lhs==ref["lhs"].get<std::string>() && rhs==ref["rhs"].get<std::string>() && parse::to_string(c.op)==ref["op"].get<std::string>() && c.group==ref["group"].get<int>()) {found=&row;break;}
    }
    REQUIRE(found!=nullptr);
    CHECK(std::abs(found->mi*gradient_scale*gradient_scale-(ref["mi"].get<double>())) < 0.02);
    CHECK(std::abs(found->epc*gradient_scale-(ref["epc"].get<double>())) < 0.005);
  }
}
}

TEST_CASE("Paper examples aggregate fits and ordinal modification indices match lavaan") {
  REQUIRE(cases.size() == 14);
  for (const auto &id:cases) {
    SUBCASE(id.c_str()) {
      CAPTURE(id);
      auto raw=test::read_fixture(test::fixtures_dir()+"/paper_corpus/examples/"+id+".json");
      REQUIRE(raw.has_value());
      auto j=nlohmann::json::parse(*raw,nullptr,false);
      REQUIRE_FALSE(j.is_discarded());
      REQUIRE(j["_meta"]["aggregate_only"].get<bool>());
      auto h=model_from_json(j);
      auto& pt=h.structure;
      REQUIRE(pt.ov_order.size()==j["observed_variables"].size());
      for (std::size_t i=0;i<pt.ov_order.size();++i)
      REQUIRE(h.names.var_name[static_cast<std::size_t>(pt.ov_order[i])]==j["observed_variables"][i].get<std::string>());
      auto rep=model::build_matrix_rep(pt);
      REQUIRE_MESSAGE(rep.has_value(),(rep.has_value()?"":rep.error().detail));
      auto theta=vector_from_json(j["theta"]),start=vector_from_json(j["start"]);
      REQUIRE(theta.size()==pt.n_free());
      optim::OptimOptions opts{.max_iter=7000,.ftol=1e-13,.gtol=1e-8};
      estimate::Estimates est;
      if (j["kind"]=="ordinal") {

        auto stats=ordinal_from_json(j);
        // Delta preparation orders loadings, then thresholds, then other
        // parameters. Translate the imported lavaan vectors by partable row.
        const auto old_free=pt.free;
        auto prep=estimate::prepare_ordinal_delta_partable(pt,stats);
        REQUIRE(prep.has_value());
        Eigen::VectorXd reordered_theta(pt.n_free()),reordered_start(pt.n_free());
        for (std::size_t r=0;r<pt.size();++r) if (pt.free[r]>0) {
          REQUIRE(old_free[r]>0);
          reordered_theta(pt.free[r]-1)=theta(old_free[r]-1);
          reordered_start(pt.free[r]-1)=start(old_free[r]-1);
        }
        theta=std::move(reordered_theta);start=std::move(reordered_start);
        rep=model::build_matrix_rep(pt);
        REQUIRE(rep.has_value());
        auto fit=estimate::fit_ordinal_bounded(pt,*rep,stats,{},estimate::OrdinalWeightKind::DWLS,start,estimate::Backend::NloptLbfgs,opts);
        REQUIRE_MESSAGE(fit.has_value(),(fit.has_value()?"":fit.error().detail));
        est=*fit;
        const double n=static_cast<double>(stats.n_obs[0]);
        CHECK(std::abs(2*(n-1)*est.fmin-(j["fit"]["chisq"].get<double>())) < 0.005);
        // Separate score parity at the oracle point from optimizer convergence.
        estimate::Estimates oracle{theta,0.0,0};
        inference::ModificationIndexOptions mo;
        mo.candidates=inference::ScoreCandidateSet::WithAbsentRows;
        auto mi=estimate::modification_indices_ordinal(pt,*rep,stats,oracle,estimate::OrdinalWeightKind::DWLS,mo);
        REQUIRE_MESSAGE(mi.has_value(),(mi.has_value()?"":mi.error().detail));
        // lavaan categorical gradients carry (N-1)/N whereas expected
        // information is Delta' W Delta. Scores therefore have the square
        // of that factor; EPCs carry one factor. Fits use the same minimizer.
        compare_mi(*mi,h.names,j["modindices"],(n-1)/n);
        if (!j["score_tests"].empty()) {
          auto scores=estimate::score_tests_ordinal(pt,*rep,stats,oracle,estimate::OrdinalWeightKind::DWLS);
          REQUIRE_MESSAGE(scores.has_value(),(scores.has_value()?"":scores.error().detail));
          REQUIRE(scores->rows.size()==j["score_tests"].size());
          for (std::size_t i=0;i<scores->rows.size();++i)
          CHECK(std::abs(scores->rows[i].mi*(n-1)*(n-1)/(n*n)-(j["score_tests"][i]["X2"].get<double>())) < 0.02);
        }
      } else {
        auto pack=pack_from_json(j);
        // FIML point estimation uses the supplied pattern sufficient statistics;
        // no casewise rows are needed (robust casewise-score SEs are not tested).
        auto fit=estimate::fiml::fit_fiml(pt,*rep,data::RawData{},start,pack,estimate::Backend::NloptLbfgsSlsqpFallback,opts);
        REQUIRE_MESSAGE(fit.has_value(),(fit.has_value()?"":fit.error().detail));
        est=*fit;
        double constant=0;
        for (const auto &p:pack.cache.patterns) constant+=static_cast<double>(p.n_obs)*static_cast<double>(p.observed.size())*std::log(2*std::numbers::pi);
        const double logl=-static_cast<double>(pack.cache.n_total)*est.fmin-0.5*constant;
        CHECK(std::abs(logl-(j["fit"]["logl"].get<double>())) < 0.005);
      }
      auto constraints=estimate::build_eq_constraints(pt);
      REQUIRE(constraints.has_value());
      Eigen::Index moment_count=0;
      for (const auto &block:j["blocks"]) {
        const auto p=static_cast<Eigen::Index>(j["observed_variables"].size());
        moment_count += j["kind"]=="ordinal"
        ? static_cast<Eigen::Index>(block["thresholds"].size())+p*(p-1)/2
        : p*(p+1)/2+p;
      }
      CHECK(moment_count-constraints->n_alpha==j["fit"]["df"].get<int>());
      REQUIRE(est.theta.size()==theta.size());
      CHECK((est.theta-theta).cwiseAbs().maxCoeff()<0.002);
      auto evaluator=model::ModelEvaluator::build(pt,*rep);
      REQUIRE(evaluator.has_value());
      auto moments=evaluator->sigma(est.theta);
      REQUIRE(moments.has_value());
      for (std::size_t b=0;b<j["implied"].size();++b) {
        auto expected=matrix_from_json(j["implied"][b]["cov"]);
        Eigen::MatrixXd got=moments->sigma[b];
        // Delta ordinal models fix the latent-response diagonal to one;
        // the generic evaluator returns the unadjusted residual diagonal.
        if (j["kind"]=="ordinal") got.diagonal().setOnes();
        CHECK((got-expected).cwiseAbs().maxCoeff()<0.002);
        if (j["kind"]=="fiml") {
          auto mean=vector_from_json(j["implied"][b]["mean"]);
          CHECK((moments->mu[b]-mean).cwiseAbs().maxCoeff()<0.002);
        }
      }
      CHECK(j["admissible"].get<bool>() == (id != "jiwani_dass_3f_cfa"));
    }
  }

}
