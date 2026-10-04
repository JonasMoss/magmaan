#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "magmaan/api/sem.hpp"
#include "magmaan/inference/inference.hpp"
#include <map>
#include "magmaan/compat/lavaan/partable_view.hpp"

namespace {
std::string growth_key(std::string lhs,std::string op,std::string rhs) {
  if(op=="~~" && rhs<lhs) std::swap(lhs,rhs);
  return lhs+"|"+op+"|"+rhs;
}
}
TEST_CASE("Mplus growth and constraints preserve independently frozen estimates, inference and restrictions") {
  using namespace magmaan;
  auto raw=test::read_fixture(test::fixtures_dir()+"/mplus/golden_growth.json");REQUIRE(raw);
  auto fixture=nlohmann::json::parse(*raw,nullptr,false);REQUIRE_FALSE(fixture.is_discarded());
  for(const auto& item:fixture["cases"].items()) {
    CAPTURE(item.key());const auto& c=item.value();
    auto m=api::model_from_mplus(c["input"].get<std::string>());REQUIRE_MESSAGE(m,(m ? "" : m.error().detail));
    const auto& rep=m->model.matrix_rep();const auto& st=m->model.structure();
    const auto vars=c["variables"].get<std::vector<std::string>>();
    data::SampleStats sample;
    for(std::size_t g=0;g<c["group_n"].size();++g) {
      sample.n_obs.push_back(c["group_n"][g].get<std::int64_t>());
      const auto cov=test::matrix_from_json(c["group_sample_cov"][g]);const auto mean=test::vector_from_json(c["group_sample_mean"][g]);
      Eigen::MatrixXd S(cov.rows(),cov.cols());Eigen::VectorXd mu(mean.size());
      for(Eigen::Index i=0;i<S.rows();++i) {
        auto a=std::find(vars.begin(),vars.end(),rep.ov_names[g][static_cast<std::size_t>(i)])-vars.begin();REQUIRE(a<static_cast<std::ptrdiff_t>(vars.size()));mu[i]=mean[a];
        for(Eigen::Index j=0;j<S.cols();++j) {auto b=std::find(vars.begin(),vars.end(),rep.ov_names[g][static_cast<std::size_t>(j)])-vars.begin();S(i,j)=cov(a,b);}
      }
      sample.S.push_back(S);sample.mean.push_back(mu);
    }
    auto fit=api::fit(m->model,api::Data::from_sample_stats(sample),api::ml());REQUIRE_MESSAGE(fit,(fit ? "" : fit.error().detail));
    auto se=api::standard_errors(*fit,api::expected_information());REQUIRE_MESSAGE(se,(se ? "" : se.error().detail));
    auto table=compat::lavaan::to_lavaan_partable(st,m->model.names(),m->model.starts());
    std::map<std::string,nlohmann::json> expected;
    for(const auto& row:c["rows"]) if(row["op"]!="==" && row["op"]!=":=") expected[std::to_string(row["group"].get<int>())+":"+growth_key(row["lhs"],row["op"],row["rhs"])]=row;
    for(std::size_t i=0;i<table.size();++i) {
      if(table.op[i]==parse::Op::EqConstraint || table.op[i]==parse::Op::DefineParam) continue;
      const auto k=std::to_string(table.group[i])+":"+growth_key(table.lhs[i],std::string(parse::to_string(table.op[i])),table.rhs[i]);CAPTURE(k);REQUIRE(expected.contains(k));
      const auto& row=expected.at(k);CHECK((table.free[i]>0)==(row["free"].get<int>()>0));
      if(table.free[i]>0) {
        CHECK(std::abs(fit->estimates().theta[table.free[i]-1]-row["est"].get<double>())<1e-5);
        CHECK(std::abs(se->se[table.free[i]-1]-row["se"].get<double>())<1e-5);
      }
    }
    auto defs=api::compute_defined(*fit,se->vcov);REQUIRE_MESSAGE(defs,(defs ? "" : defs.error().detail));
    std::size_t defined_count=0;
    for(const auto& row:c["rows"]) if(row["op"]==":=") {
      ++defined_count;auto hit=std::find_if(defs->entries.begin(),defs->entries.end(),[&](const auto& d){return d.name==row["lhs"].get<std::string>();});REQUIRE(hit!=defs->entries.end());
      CHECK(std::abs(hit->value-row["est"].get<double>())<1e-5);CHECK(std::abs(hit->se-row["se"].get<double>())<1e-5);
    }
    CHECK(defs->entries.size()==defined_count);
    auto fm=api::fit_measures(*fit);REQUIRE_MESSAGE(fm,(fm ? "" : fm.error().detail));
    REQUIRE(fm->complete_data_extras);CHECK(fm->complete_data_extras->npar==c["npar"].get<int>());
    auto tst=api::test(*fit,api::standard_chi_square());REQUIRE(tst);
    CHECK(tst->df==c["df"].get<int>());
    CHECK(std::abs(inference::chi2_stat(sample,fit->estimates())-c["chisq"].get<double>())<1e-5);
    // P-CN1 meaning gate: equations constrain the fit, not only the display.
    if(item.key().starts_with("nonlinear")) {
      std::map<std::string,double> value;
      for(std::size_t i=0;i<st.size();++i) if(st.free[i]>0) value[m->model.names().row_label[i]]=fit->estimates().theta[st.free[i]-1];
      CHECK(std::abs(value["p1"]-value["p2"]*value["p2"]-value["p3"]*value["p3"])<1e-8);
    }
  }
}
