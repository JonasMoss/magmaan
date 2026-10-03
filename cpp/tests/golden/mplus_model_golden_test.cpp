#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/compat/mplus/model.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/mplus_parser.hpp"

namespace {
std::string key(std::string lhs, std::string op, std::string rhs) {
  if (op == "~~" && rhs < lhs) std::swap(lhs, rhs);
  return lhs + "|" + op + "|" + rhs;
}
Eigen::MatrixXd matrix(const nlohmann::json& data) {
  Eigen::MatrixXd result(static_cast<Eigen::Index>(data.size()),
                         static_cast<Eigen::Index>(data[0].size()));
  for (Eigen::Index r = 0; r < result.rows(); ++r)
    for (Eigen::Index c = 0; c < result.cols(); ++c)
      result(r,c) = data[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>();
  return result;
}
}

TEST_CASE("Mplus MODEL: model rows, starts, ML estimates, implied covariance and inference match independent lavaan fixtures") {
  auto raw = magmaan::test::read_fixture(magmaan::test::fixtures_dir() + "/mplus/golden.json");
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());
  for (const auto& test : fixture["cases"]) {
    CAPTURE(test["id"]);
    auto parsed = magmaan::parse::MplusParser::parse(test["mplus_input"].get<std::string>());
    REQUIRE_MESSAGE(parsed.has_value(), (parsed ? "" : parsed.error().detail));
    auto* flat = &parsed->flat;

    magmaan::spec::LatentNames names;
    magmaan::spec::Starts starts;
    auto structure = magmaan::spec::build(*flat, magmaan::compat::mplus::build_options(parsed->input), &starts, &names);
    REQUIRE(structure.has_value());
    auto pt = magmaan::compat::lavaan::to_lavaan_partable(*structure, names, starts);
    std::map<std::string, nlohmann::json> expected;
    for (const auto& row : test["lavaan"]["rows"])
      expected.emplace(key(row["lhs"],row["op"],row["rhs"]),row);
    std::set<std::string> flat_keys, built_keys;
    for (const auto& row : flat->rows) flat_keys.insert(key(std::string(row.lhs),std::string(magmaan::parse::to_string(row.op)),std::string(row.rhs)));
    std::map<int,std::string> group_labels;
    std::map<std::string,int> label_groups;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] == magmaan::parse::Op::EqConstraint) continue;
      const auto k = key(pt.lhs[i],std::string(magmaan::parse::to_string(pt.op[i])),pt.rhs[i]);
      if (pt.exo[i] == 0) built_keys.insert(k);
    }
    CHECK(flat_keys == built_keys);
    std::set<std::string> expected_keys, all_built_keys;
    for (const auto& row : test["lavaan"]["rows"]) if (row["op"] != "==") expected_keys.insert(key(row["lhs"],row["op"],row["rhs"]));
    for (std::size_t i = 0; i < pt.size(); ++i) if (pt.op[i] != magmaan::parse::Op::EqConstraint) all_built_keys.insert(key(pt.lhs[i],std::string(magmaan::parse::to_string(pt.op[i])),pt.rhs[i]));
    CHECK(all_built_keys == expected_keys);
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] == magmaan::parse::Op::EqConstraint) continue;
      const auto k = key(pt.lhs[i],std::string(magmaan::parse::to_string(pt.op[i])),pt.rhs[i]);
      CAPTURE(k);
      REQUIRE(expected.contains(k));
      const auto& row = expected.at(k);
      CHECK((pt.free[i] > 0) == (row["free"].get<int>() > 0));
      if (pt.free[i] > 0) {
        const auto group=structure->eq_groups[static_cast<std::size_t>(pt.free[i]-1)];
        const auto label=row["label"].get<std::string>().empty()?k:row["label"].get<std::string>();
        if(group_labels.contains(group)) CHECK(group_labels.at(group)==label); else group_labels[group]=label;
        if(label_groups.contains(label)) CHECK(label_groups.at(label)==group); else label_groups[label]=group;
      }
      if (row["ustart"].is_null()) CHECK(std::isnan(pt.ustart[i]));
      else CHECK(pt.ustart[i] == doctest::Approx(row["ustart"].get<double>()).epsilon(1e-12));
    }
    auto rep = magmaan::model::build_matrix_rep(*structure, &names);
    REQUIRE(rep.has_value());

    magmaan::data::SampleStats sample;
    const auto variables = test["variables"].get<std::vector<std::string>>();
    const auto p = static_cast<Eigen::Index>(variables.size());
    Eigen::MatrixXd cov(p,p); Eigen::VectorXd mean(p);
    const auto original = matrix(test["sample_cov"]);
    for (Eigen::Index i = 0; i < p; ++i) {
      const auto& name = rep->ov_names[0][static_cast<std::size_t>(i)];
      const auto a = static_cast<std::size_t>(std::find(variables.begin(),variables.end(),name)-variables.begin());
      REQUIRE(a < variables.size()); mean[i] = test["sample_mean"][a].get<double>();
      for (Eigen::Index j = 0; j < p; ++j) {
        const auto b = static_cast<std::size_t>(std::find(variables.begin(),variables.end(),rep->ov_names[0][static_cast<std::size_t>(j)])-variables.begin());
        cov(i,j) = original(static_cast<Eigen::Index>(a),static_cast<Eigen::Index>(b));
      }
    }
    sample.S = {cov}; sample.n_obs = {test["n"].get<int>()};
    if (!parsed->input.nomeanstructure) sample.mean = {mean};
    REQUIRE(magmaan::estimate::resolve_fixed_x_from_sample(*structure,*rep,sample));
    auto start = magmaan::estimate::simple_start_values(*structure,*rep,sample,starts);
    REQUIRE(start.has_value());
    auto fit = magmaan::estimate::fit_ml(*structure,*rep,sample,*start);
    REQUIRE_MESSAGE(fit.has_value(), (fit ? "" : fit.error().detail));
    auto info = magmaan::inference::information_expected(*structure,*rep,sample,*fit);
    REQUIRE(info.has_value());
    auto vcov = magmaan::inference::vcov(*info,*structure);
    REQUIRE(vcov.has_value());
    auto se = magmaan::inference::se(*vcov);
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] == magmaan::parse::Op::EqConstraint) continue;
      const auto k = key(pt.lhs[i],std::string(magmaan::parse::to_string(pt.op[i])),pt.rhs[i]);
      CAPTURE(k);
      const auto& row = expected.at(k);
      if (pt.free[i] > 0) {
        const auto index = pt.free[i] - 1;
        CHECK(std::abs(fit->theta[index]-row["estimate"].get<double>()) < 1e-5);
        CHECK(std::abs(se[index]-row["se"].get<double>()) < 1e-5);
      }
    }
    auto evaluator = magmaan::model::ModelEvaluator::build(*structure,*rep);
    REQUIRE(evaluator.has_value());
    auto implied = evaluator->sigma(fit->theta);
    REQUIRE(implied.has_value());
    const auto implied_variables = test["lavaan"]["implied_variables"].get<std::vector<std::string>>();
    const auto expected_cov = matrix(test["lavaan"]["implied_cov"]);
    for (Eigen::Index i = 0; i < p; ++i) {
      const auto a = static_cast<std::size_t>(std::find(implied_variables.begin(),implied_variables.end(),rep->ov_names[0][static_cast<std::size_t>(i)])-implied_variables.begin());
      for (Eigen::Index j = 0; j < p; ++j) {
        const auto b = static_cast<std::size_t>(std::find(implied_variables.begin(),implied_variables.end(),rep->ov_names[0][static_cast<std::size_t>(j)])-implied_variables.begin());
        CHECK(std::abs(implied->sigma[0](i,j)-expected_cov(static_cast<Eigen::Index>(a),static_cast<Eigen::Index>(b))) < 1e-5);
      }
    }
    if (!sample.mean.empty()) {

      for (Eigen::Index i = 0; i < p; ++i) {
        const auto a = static_cast<std::size_t>(std::find(implied_variables.begin(),implied_variables.end(),rep->ov_names[0][static_cast<std::size_t>(i)])-implied_variables.begin());
        CHECK(std::abs(implied->mu[0][i]-test["lavaan"]["implied_mean"][a].get<double>()) < 1e-5);
      }
    }
    std::set<int> groups(structure->eq_groups.begin(),structure->eq_groups.end());
    CHECK(groups.size() == test["lavaan"]["npar"].get<std::size_t>());
    auto df = magmaan::inference::df_stat(*structure,sample);
    REQUIRE(df.has_value());
    CHECK(*df == test["lavaan"]["df"].get<int>());
    CHECK(std::abs(magmaan::inference::chi2_stat(sample,*fit)-test["lavaan"]["chisq"].get<double>()) < 1e-5);
  }
}

TEST_CASE("Mplus MODEL: multigroup rows TECH1 and numerical lavaan goldens") {
  using namespace magmaan;
  auto raw=test::read_fixture(test::fixtures_dir()+"/mplus/golden.json");REQUIRE(raw);
  auto fixture=nlohmann::json::parse(*raw,nullptr,false);REQUIRE_FALSE(fixture.is_discarded());
  for(const auto& t:fixture["multigroup_cases"]) {
    CAPTURE(t["id"]);
    auto parsed=parse::MplusParser::parse(t["mplus_input"].get<std::string>());REQUIRE_MESSAGE(parsed,(parsed?"":parsed.error().detail));
    spec::LatentNames names;spec::Starts starts;
    auto st=spec::build(parsed->flat,compat::mplus::build_options(parsed->input),&starts,&names);REQUIRE(st);
    auto pt=compat::lavaan::to_lavaan_partable(*st,names,starts);
    CHECK(pt.group_labels==t["codes"].get<std::vector<std::string>>());
    std::map<std::string,nlohmann::json> expected;
    for(const auto& row:t["rows"]) if(row["op"]!="==") expected[std::to_string(row["group"].get<int>())+":"+key(row["lhs"],row["op"],row["rhs"])]=row;
    std::map<int,int> our_to_lavaan,lavaan_to_our,our_to_demo,demo_to_our;
    std::set<std::string> actual;
    auto upper=[](std::string s){for(auto& c:s) if(c>='a' && c<='z') c=static_cast<char>(c-'a'+'A');return s;};
    for(std::size_t i=0;i<pt.size();++i) if(pt.op[i]!=parse::Op::EqConstraint) {
      const auto k=std::to_string(pt.group[i])+":"+key(pt.lhs[i],std::string(parse::to_string(pt.op[i])),pt.rhs[i]);
      CAPTURE(k);actual.insert(k);REQUIRE(expected.contains(k));const auto& r=expected.at(k);
      CHECK((pt.free[i]>0)==(r["free"].get<int>()>0));
      if(pt.free[i]>0) {
        const int eq=st->eq_groups[static_cast<std::size_t>(pt.free[i]-1)], ref=r["free"].get<int>();
        // lavaan has separate free indices tied by equality rows: use labels
        // for the independent equality partition instead of its raw indices.
        const std::string label=r["label"].get<std::string>();
        int canonical=ref;
        if(!label.empty()) for(const auto& other:t["rows"]) if(other["label"]==label && other["free"].get<int>()>0) canonical=std::min(canonical,other["free"].get<int>());
        if(our_to_lavaan.contains(eq)) CHECK(our_to_lavaan.at(eq)==canonical);else our_to_lavaan[eq]=canonical;
        if(lavaan_to_our.contains(canonical)) CHECK(lavaan_to_our.at(canonical)==eq);else lavaan_to_our[canonical]=eq;
      } else if(!r["ustart"].is_null()) CHECK(pt.ustart[i]==doctest::Approx(r["ustart"].get<double>()));
      if(pt.exo[i]) continue;
      std::string row=upper(pt.lhs[i]),col=upper(pt.rhs[i]);std::vector<std::string> mats;
      if(pt.op[i]==parse::Op::Measurement) {mats={"LAMBDA","BETA"};std::swap(row,col);}
      else if(pt.op[i]==parse::Op::Regression) mats={"BETA"};
      else if(pt.op[i]==parse::Op::Covariance) mats={"THETA","PSI"};
      else if(pt.op[i]==parse::Op::Intercept) {mats={"NU","ALPHA"};col=row;row="vector";}
      std::optional<int> number;
      const auto label=upper(parsed->input.groups[static_cast<std::size_t>(pt.group[i]-1)].label);
      for(const auto& cell:t["demo"]["tech1"]) if(cell["group"]==label && std::find(mats.begin(),mats.end(),cell["matrix"].get<std::string>())!=mats.end() &&
        ((cell["row"]==row && cell["col"]==col) || (pt.op[i]==parse::Op::Covariance && cell["row"]==col && cell["col"]==row))) {
        const int n=cell["number"].get<int>();if(!number || n>0) number=n;
      }
      REQUIRE(number);if(!number) continue;CHECK((pt.free[i]>0)==(*number>0));
      if(pt.free[i]>0 && *number>0) {
        const int eq=st->eq_groups[static_cast<std::size_t>(pt.free[i]-1)];
        if(our_to_demo.contains(eq)) CHECK(our_to_demo.at(eq)==*number);else our_to_demo[eq]=*number;
        if(demo_to_our.contains(*number)) CHECK(demo_to_our.at(*number)==eq);else demo_to_our[*number]=eq;
      }
    }
    CHECK(actual.size()==expected.size());
    auto rep=model::build_matrix_rep(*st,&names);REQUIRE(rep);
    data::SampleStats sample;const auto vars=t["variables"].get<std::vector<std::string>>();
    for(std::size_t g=0;g<parsed->input.groups.size();++g) {
      const auto p=static_cast<Eigen::Index>(rep->ov_names[g].size());Eigen::MatrixXd cov(p,p);Eigen::VectorXd mean(p);
      const auto original=matrix(t["sample"][g]["cov"]);
      for(Eigen::Index a=0;a<p;++a) {
        const auto ai=static_cast<std::size_t>(std::find(vars.begin(),vars.end(),rep->ov_names[g][static_cast<std::size_t>(a)])-vars.begin());REQUIRE(ai<vars.size());mean[a]=t["sample"][g]["mean"][ai].get<double>();
        for(Eigen::Index b=0;b<p;++b) {
          const auto bi=static_cast<std::size_t>(std::find(vars.begin(),vars.end(),rep->ov_names[g][static_cast<std::size_t>(b)])-vars.begin());REQUIRE(bi<vars.size());cov(a,b)=original(static_cast<Eigen::Index>(ai),static_cast<Eigen::Index>(bi));
        }
      }
      sample.S.push_back(cov);sample.mean.push_back(mean);sample.n_obs.push_back(t["sample"][g]["n"].get<int>());
    }
    REQUIRE(estimate::resolve_fixed_x_from_sample(*st,*rep,sample));
    auto start=estimate::simple_start_values(*st,*rep,sample,starts);REQUIRE(start);
    auto fit=estimate::fit_ml(*st,*rep,sample,*start);REQUIRE_MESSAGE(fit,(fit?"":fit.error().detail));
    auto info=inference::information_expected(*st,*rep,sample,*fit);REQUIRE(info);auto vc=inference::vcov(*info,*st);REQUIRE(vc);auto se=inference::se(*vc);
    for(std::size_t i=0;i<pt.size();++i) if(pt.op[i]!=parse::Op::EqConstraint && pt.free[i]>0) {
      const auto k=std::to_string(pt.group[i])+":"+key(pt.lhs[i],std::string(parse::to_string(pt.op[i])),pt.rhs[i]);const auto& r=expected.at(k);const auto index=pt.free[i]-1;
      CAPTURE(k);CHECK(std::abs(fit->theta[index]-r["est"].get<double>())<1e-5);CHECK(std::abs(se[index]-r["se"].get<double>())<1e-5);
    }
    auto evaluator=model::ModelEvaluator::build(*st,*rep);REQUIRE(evaluator);auto implied=evaluator->sigma(fit->theta);REQUIRE(implied);
    for(std::size_t g=0;g<sample.S.size();++g) {
      const auto iv=t["implied"][g]["variables"].get<std::vector<std::string>>();const auto ec=matrix(t["implied"][g]["cov"]);
      for(std::size_t a=0;a<rep->ov_names[g].size();++a) {
        const auto ai=static_cast<std::size_t>(std::find(iv.begin(),iv.end(),rep->ov_names[g][a])-iv.begin());REQUIRE(ai<iv.size());
        CHECK(std::abs(implied->mu[g][static_cast<Eigen::Index>(a)]-t["implied"][g]["mean"][ai].get<double>())<1e-5);
        for(std::size_t b=0;b<rep->ov_names[g].size();++b) {const auto bi=static_cast<std::size_t>(std::find(iv.begin(),iv.end(),rep->ov_names[g][b])-iv.begin());REQUIRE(bi<iv.size());CHECK(std::abs(implied->sigma[g](static_cast<Eigen::Index>(a),static_cast<Eigen::Index>(b))-ec(static_cast<Eigen::Index>(ai),static_cast<Eigen::Index>(bi)))<1e-5);}
      }
    }
    std::set<int> eq(st->eq_groups.begin(),st->eq_groups.end());CHECK(eq.size()==t["npar"].get<std::size_t>());
    auto df=inference::df_stat(*st,sample);REQUIRE(df);CHECK(*df==t["df"].get<int>());CHECK(std::abs(inference::chi2_stat(sample,*fit)-t["chisq"].get<double>())<1e-5);
  }
}
