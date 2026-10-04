#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include "../oracle.hpp"
#include "magmaan/compat/mplus/model.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/expr_format.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include <Eigen/SVD>

namespace {
using namespace magmaan;
std::string source(std::string model, std::string analysis = "", std::string vars = "y1 y2 y3 y4 x1 x2") {
  return "DATA: FILE=x;\nVARIABLE: NAMES="+vars+";\nANALYSIS: "+analysis+"\nMODEL:\n"+model+"\n";
}
std::string key(std::string lhs, parse::Op op, std::string rhs) {
  if (op == parse::Op::Covariance && rhs < lhs) std::swap(lhs,rhs);
  return lhs+std::string(parse::to_string(op))+rhs;
}
std::map<std::string,std::string> rows(const std::string& text) {
  auto m = parse::MplusParser::parse(text); REQUIRE_MESSAGE(m, (m ? "" : m.error().detail));
  if (!m) return {};
  auto moved = std::move(*m);
  spec::LatentNames names; spec::Starts starts;
  auto s = spec::build(moved.flat,compat::mplus::build_options(moved.input),&starts,&names);
  REQUIRE_MESSAGE(s,(s ? "" : s.error().detail)); if (!s) return {};
  REQUIRE(model::build_matrix_rep(*s,&names));
  auto pt = compat::lavaan::to_lavaan_partable(*s,names,starts);
  std::map<std::string,std::string> result;
  for (std::size_t i=0;i<pt.size();++i) {
    if (pt.op[i] == parse::Op::EqConstraint || pt.exo[i]) continue;
    result[key(pt.lhs[i],pt.op[i],pt.rhs[i])] = pt.free[i] ? (pt.label[i].empty() ? "free" : pt.label[i]) : std::to_string(pt.ustart[i]);
  }
  return result;
}
void reject(std::string model, std::string rule, std::string vars="y1 y2 y3 y4 x1 x2") {
  auto m = parse::MplusParser::parse(source(model,"",vars));
  REQUIRE_FALSE(m); if (m) return;
  CHECK(m.error().span.line > 0);
  for (const auto* part : {"found '", "Mplus", "magmaan", "instead"})
    CHECK_MESSAGE(m.error().detail.find(part) != std::string::npos, m.error().detail);
  CHECK_MESSAGE(m.error().detail.find("["+rule+"]") != std::string::npos,m.error().detail);
}
}
TEST_CASE("Mplus MODEL: MS02 LB01 LB07 LB08 explicit marker and last mention") {
  auto r=rows(source("f BY y1 y2; f BY y3 y4;"));
  CHECK(r.at("f=~y1")=="1.000000"); CHECK(r.at("f=~y3")=="free");
  r=rows(source("f BY y1-y3; f BY y1;")); CHECK(r.at("f=~y1")=="free");
  r=rows(source("f BY y1 y2* y3;")); CHECK(r.at("f=~y1")=="1.000000");
  r=rows(source("f BY y1* y2 y3; f@1;")); CHECK(r.at("f=~y1")=="free"); CHECK(r.at("f~~f")=="1.000000");
  reject("f BY y1-y3; f@;","LB01");
}
TEST_CASE("Mplus MODEL: DF01-DF13 roles and explicit defaults") {
  auto r=rows(source("f BY y1-y3; f ON x1; y4 ON x1;"));
  CHECK(r.at("f~~y4")=="free"); CHECK(r.at("f~1")=="0.000000"); CHECK(r.at("y1~1")=="free");
  CHECK_FALSE(r.contains("x1~~x1")); CHECK_FALSE(r.contains("f~~x1")); CHECK_FALSE(r.contains("y1~~y2"));
  r=rows(source("y1 y2 ON x1; y3 ON y1;")); CHECK(r.at("y2~~y3")=="free"); CHECK_FALSE(r.contains("y1~~y2")); CHECK_FALSE(r.contains("y1~~y3")); CHECK(r.at("y4~~y4")=="free"); CHECK(r.at("y4~1")=="free");
  r=rows(source("f1 BY y1-y2; f2 BY y3-y4;")); CHECK(r.at("f1~~f2")=="free");
  r=rows(source("f1 BY y1-y2; f2 BY y3-y4; g BY f1 f2; f1 f2 ON x1;")); CHECK_FALSE(r.contains("f1~~f2"));
  r=rows(source("f1 BY y1-y2; f2 BY y3-y4; y1 WITH y3;","MODEL=NOCOVARIANCES;")); CHECK_FALSE(r.contains("f1~~f2")); CHECK(r.at("y1~~y3")=="free");
  r=rows(source("f BY y1-y3;","MODEL=NOMEANSTRUCTURE; INFORMATION=EXPECTED;")); CHECK_FALSE(r.contains("f~1")); CHECK_FALSE(r.contains("y1~1"));
  for (auto model : {"y1 ON x1; x1;","y1 ON x1; [x1];","y1 ON x1; x1 WITH x2;"}) reject(model,"MS08");
}
TEST_CASE("Mplus MODEL: NM03 NM04 NM05 MS03 MS05 MS06 ranges and pairing") {
  auto r=rows(source("f BY y1-y3;","","y1 x1 y2 y3")); CHECK(r.contains("f=~x1"));
  r=rows(source("y1 y2 PON x1 x2; y1 y2 PWITH y3 y4;")); CHECK(r.contains("y1~x1")); CHECK_FALSE(r.contains("y1~x2")); CHECK(r.contains("y1~~y3")); CHECK_FALSE(r.contains("y1~~y4"));
  r=rows(source("y1-y3 WITH y1-y3;")); CHECK(r.at("y1~~y2")=="free"); CHECK(r.at("y2~~y3")=="free");
  reject("f BY y3-y1;","NM03"); reject("f BY y1-y3; g BY f-y3;","NM04"); reject("g BY f; f BY y1-y3;","MS03"); reject("y1 y2 PON x1;","MS05"); reject("unknown ON x1;","NM03");
}
TEST_CASE("Mplus MODEL: LB02-LB06 labels equality sets starts and ownership") {
  auto r=rows(source("y1-y3 ON x1-x2 (p1-p6);")); CHECK(r.at("y1~x2")=="p2"); CHECK(r.at("y2~x1")=="p3");
  r=rows(source("f BY y1-y4 (1-4);")); CHECK(r.at("f=~y1")=="1.000000"); CHECK(r.at("f=~y2")==".eq2.");
  r=rows(source("f BY y1\ny2 (B)\ny3 (b);")); CHECK(r.at("f=~y2")=="b"); CHECK(r.at("f=~y3")=="b");
  auto m=parse::MplusParser::parse(source("y1 ON x1*0.5 (b);\ny2 ON x1 (b);")); REQUIRE(m);
  spec::Starts starts; spec::LatentNames names; auto s=spec::build(m->flat,compat::mplus::build_options(m->input),&starts,&names); REQUIRE(s);
  auto pt=compat::lavaan::to_lavaan_partable(*s,names,starts);
  for(std::size_t i=0;i<pt.size();++i) if(pt.lhs[i]=="y1" && pt.op[i]==parse::Op::Regression) { CHECK(pt.label[i]=="b"); CHECK(pt.ustart[i]==doctest::Approx(.5)); }
  reject("f BY y1-y4 (a2-a4);","LB04"); reject("y1-y3 ON x1-x2 (p1-p5);","LB05"); reject("f BY y1-y3 (1) y4;","LB03"); reject("f BY y1@1 (l1) y2-y3;","LB03");
}
TEST_CASE("Mplus MODEL: out-of-family constructs have classified rejections") {
  reject("{y1};","CT04"); reject("[y1$1];","CT02"); reject("i | y1;","GR07"); reject("%OVERALL% y1 ON x1;","MS10"); reject("y1#1;","MS10"); reject("f BY y1-y3 (*rot);","MS09"); reject("f BY y1~0 y2;","MS09");
}
TEST_CASE("Mplus MODEL: local corpus sweep") {
  const auto* path=std::getenv("MAGMAAN_MPLUS_CORPUS"); if(!path) return;
  std::map<std::string,int> tally; int accepted=0,total=0;
  for(const auto& entry:std::filesystem::recursive_directory_iterator(path)) {
    if(!entry.is_regular_file() || entry.path().extension()!=".inp") continue;
    std::ifstream file(entry.path()); std::stringstream buffer; buffer<<file.rdbuf(); ++total;
    auto m=parse::MplusParser::parse(buffer.str()); if(m) {++accepted;continue;}
    const auto& detail=m.error().detail; std::set<std::string> rules;
    for(std::size_t i=0;(i=detail.find('[',i))!=std::string::npos;++i) {auto end=detail.find(']',i); if(end==std::string::npos) break;auto rule=detail.substr(i+1,end-i-1);
      if(rule.size()>=3 && rule[0]>='A' && rule[0]<='Z' && rule[1]>='A' && rule[1]<='Z' && std::all_of(rule.begin()+2,rule.end(),[](char c){return c>='0' && c<='9';})) rules.insert(rule);}
    CHECK_MESSAGE(!rules.empty(),(entry.path().string()+": "+detail)); for(const auto& rule:rules) ++tally[rule];
  }
  std::cout<<"Mplus corpus: total="<<total<<" accepted="<<accepted<<" rejected="<<total-accepted<<'\n';
  for(const auto& [rule,count]:tally) std::cout<<rule<<' '<<count<<'\n';
}

TEST_CASE("Mplus MODEL: Demo TECH1 parameter counts cells and equality partitions") {
  auto raw=test::read_fixture(test::fixtures_dir()+"/mplus/probes.json"); REQUIRE(raw);
  auto probes=nlohmann::json::parse(*raw,nullptr,false); REQUIRE_FALSE(probes.is_discarded());
  auto categorical_raw=test::read_fixture(test::fixtures_dir()+"/mplus/probes_categorical.json"); REQUIRE(categorical_raw);
  auto categorical=nlohmann::json::parse(*categorical_raw,nullptr,false); REQUIRE_FALSE(categorical.is_discarded());
  probes.update(categorical);
  int checked=0;
  for(auto probe=probes.begin();probe!=probes.end();++probe) for(auto variant=probe.value()["variants"].begin();variant!=probe.value()["variants"].end();++variant) {
    INFO(probe.key(),"/",variant.key()); const auto& v=variant.value();
    const auto title=v["title"].is_null()?std::string{}:v["title"].get<std::string>();
    auto body=v["model"].get<std::string>();
    if(probe.key()=="P-LB2" || probe.key()=="P-LB4" || probe.key()=="P-LB6") { const auto pos=body.find("MODEL CONSTRAINT:"); if(pos!=std::string::npos) body.resize(pos); }
    auto text="TITLE: "+title+"\nDATA: FILE=x;\n"+v["data"].get<std::string>()+"\nVARIABLE: "+v["variable"].get<std::string>()+"\nANALYSIS: "+v["analysis"].get<std::string>()+"\nMODEL: "+body+"\n";
    auto m=parse::MplusParser::parse(text);
    if((probe.key()=="P-NM3" && (variant.key()=="use_order" || variant.key()=="mixed_range")) || probe.key()=="P-LB5" || (probe.key()=="P-LB6" && variant.key()=="fixed_label")) { CHECK_FALSE(m); continue; }
    if(!m) {
      // Deliberate boundaries: LX02 truncation, LX05 multiline comments,
      // NM02 unusual generators, MS08 mixed x conditioning, MS11 ignored
      // NOMEANSTRUCTURE, bare @ LB01, later commands CL/CT/GR/MS10/MS09.
      CHECK(m.error().detail.find('[')!=std::string::npos); continue;
    }
    if(v["status"]!="accepted" || v["free_parameters"].empty()) continue;
    if (!m->input.categorical.empty()) {
      std::vector<std::int32_t> counts(m->input.categorical.size(),2);
      for(const auto& matrix:v["tech1"]) if(matrix["name"]=="TAU")
        for(std::size_t j=0;j<counts.size();++j) for(const auto& col:matrix["columns"]) {
          auto name=m->input.categorical[j];for(auto& c:name) if(c>='a' && c<='z') c=static_cast<char>(c-'a'+'A');
          const auto column=col.get<std::string>();
          if(column.starts_with(name+"$")) counts[j]=std::max(counts[j],static_cast<std::int32_t>(std::stoi(column.substr(name.size()+1))+1));
        }
      m=parse::MplusParser::parse_ordinal(text,counts);REQUIRE(m);
    }

    spec::LatentNames names; auto s=spec::build(m->flat,compat::mplus::build_options(m->input),nullptr,&names); REQUIRE(s);
    auto affine=estimate::build_eq_constraints(*s,true);REQUIRE(affine);
    auto dimension=affine->n_alpha;
    if(!s->nl_constraints.empty()) {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(estimate::build_nl_constraints(*s).jacobian(Eigen::VectorXd::Constant(s->n_free(),.7)));
      svd.setThreshold(1e-9);dimension-=static_cast<std::int32_t>(svd.rank());
    }
    CHECK(dimension==v["free_parameters"][0].get<int>()); ++checked;
    auto pt=compat::lavaan::to_lavaan_partable(*s,names,{});
    std::map<int,int> our_to_demo,demo_to_our;
    auto upper=[](std::string name){for(auto& c:name) if(c>='a' && c<='z') c=static_cast<char>(c-'a'+'A'); return name;};
    for(std::size_t i=0;i<pt.size();++i) {
      if(pt.op[i]==parse::Op::EqConstraint || pt.exo[i]) continue;
      auto lhs=upper(pt.lhs[i]),rhs=upper(pt.rhs[i]);
      // TECH1 alone truncates to eight characters (NM01a); ambiguous names
      // cannot identify a cell, so the count remains the gate for that probe.
      if(lhs.size()>8 || rhs.size()>8) continue;
      std::vector<std::string> matrices; std::string row=lhs,col=rhs;
      if(pt.op[i]==parse::Op::Measurement) {matrices={"LAMBDA","BETA"};row=rhs;col=lhs;}
      else if(pt.op[i]==parse::Op::Regression) matrices={"BETA","GAMMA"};
      else if(pt.op[i]==parse::Op::Covariance) {
        if(lhs==rhs && m->input.parameterization=="DELTA" &&
            std::any_of(m->input.categorical.begin(),m->input.categorical.end(),[&](auto n){return upper(n)==lhs;})) continue; // derived residual, not a TECH1 parameter
        matrices={"THETA","PSI"};
      }
      else if(pt.op[i]==parse::Op::Intercept) {matrices={"NU","ALPHA"};row="vector";col=lhs;}
      else if(pt.op[i]==parse::Op::Threshold) {matrices={"TAU"};row="vector";col=lhs+"$"+rhs.substr(1);}
      else if(pt.op[i]==parse::Op::ResponseScale) {
        if(m->input.parameterization=="THETA") continue; // derived response scale
        matrices={"DELTA"};row="vector";col=lhs;
      }
      else continue;
      INFO(lhs,std::string(parse::to_string(pt.op[i])),rhs);
      std::optional<int> number;
      for(const auto& matrix:v["tech1"]) {
        if (!m->input.groups.empty() && matrix["group"].get<std::string>()!=upper(m->input.groups[static_cast<std::size_t>(pt.group[i]-1)].label)) continue;
        if(std::find(matrices.begin(),matrices.end(),matrix["name"].get<std::string>())==matrices.end()) continue;
        const auto& cells=matrix["rows"];
        if(cells.contains(row) && cells[row].contains(col)) { const int n=cells[row][col].get<int>(); if(!number || n>0) number=n; }
        else if(pt.op[i]==parse::Op::Covariance && cells.contains(col) && cells[col].contains(row)) {const int n=cells[col][row].get<int>();if(!number || n>0) number=n;}
      }
      if(!number && pt.op[i]==parse::Op::ResponseScale && m->input.groups.empty()) {
        CHECK(pt.free[i]==0);CHECK(pt.ustart[i]==1);continue; // Demo omits fixed single-group DELTA scales
      }
      REQUIRE_MESSAGE(number, (lhs+std::string(parse::to_string(pt.op[i]))+rhs));
      CHECK((pt.free[i]>0)==(*number>0));
      if(pt.free[i]>0 && *number>0) {
        const int group=s->eq_groups[static_cast<std::size_t>(pt.free[i]-1)];
        if(our_to_demo.contains(group)) CHECK(our_to_demo.at(group)==*number); else our_to_demo[group]=*number;
        if(demo_to_our.contains(*number)) CHECK(demo_to_our.at(*number)==group); else demo_to_our[*number]=group;
      }
    }
  }
  CHECK(checked>=15);
}

TEST_CASE("Mplus MODEL: quoted data plan text and explicit means do not alter MODEL lexing") {
  auto m=parse::MplusParser::parse("DATA: FILE='!*literal.dat';\nVARIABLE: NAMES=y1 y2 y3;\nMODEL: f BY y1-y3;\n"); REQUIRE(m); CHECK(m->flat.rows.size()==11);
  auto r=rows(source("f BY y1-y3; [f*0.2] (m); [y1-y3@0];")); CHECK(r.at("f~1")=="m"); CHECK(r.at("y2~1")=="0.000000");
  reject("f BY y1 y2 y3 (a1-a3);","LB04");
  r=rows(source("y1 y2 PON\nx1\nx2;")); CHECK(r.contains("y1~x1")); CHECK(r.contains("y2~x2"));
}

TEST_CASE("Mplus MODEL: LB05 preserves groups per left-hand variable") {
  auto r=rows(source("y1 y2 ON x1 x2 (p1-p2 q);"));
  CHECK(r.at("y1~x1")=="p1"); CHECK(r.at("y1~x2")=="p2");
  CHECK(r.at("y2~x1")=="q"); CHECK(r.at("y2~x2")=="q");
  reject("y1 y2 ON x1 x2 (p1-p3 q);","LB05");
}

TEST_CASE("Mplus MODEL: canonical observed and first BY spelling and BY lists") {
  auto m = parse::MplusParser::parse(source("Factor BY y1 Y2 y3; factor ON x;", "", "Y1 y2 Y3 X"));
  REQUIRE(m);
  CHECK(m->flat.rows[0].lhs == "Factor");
  CHECK(m->flat.rows[0].rhs == "Y1");
  reject("f1-f3 BY y1-y3;", "MS09");
  reject("f1 f2 BY y1-y3;", "MS09");
}

TEST_CASE("Mplus MODEL: MG04 MG06 MG07 MG08 ordered groups and overrides") {
  const std::string text="DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3 y4 g;\nGROUPING=g(2=b 1=a);\nMODEL: f BY y1-y4;\nMODEL b: f BY y2*0.8;\n[y3]; y1 WITH y2;\nMODEL a: [f];\nMODEL b: f BY y2@0.9;\nMODEL b: f BY y2;\n";
  auto m=parse::MplusParser::parse(text);REQUIRE_MESSAGE(m,(m ? "" : m.error().detail));
  spec::LatentNames names;spec::Starts starts;
  auto st=spec::build(m->flat,compat::mplus::build_options(m->input),&starts,&names);REQUIRE(st);
  auto pt=compat::lavaan::to_lavaan_partable(*st,names,starts);
  CHECK(pt.group_labels==std::vector<std::string>{"1","2"});
  std::map<std::string,std::string> got;
  for(std::size_t i=0;i<pt.size();++i) if(pt.op[i]!=parse::Op::EqConstraint && !pt.exo[i])
    got[std::to_string(pt.group[i])+":"+key(pt.lhs[i],pt.op[i],pt.rhs[i])]=pt.free[i] ? (pt.label[i].empty()?"free":pt.label[i]) : std::to_string(pt.ustart[i]);
  const std::map<std::string,std::string> expected={
    {"1:f=~y1","1.000000"},{"2:f=~y1","1.000000"},
    {"1:f=~y2",".mg1."},{"2:f=~y2","free"},
    {"1:f=~y3",".mg2."},{"2:f=~y3",".mg2."},
    {"1:f=~y4",".mg3."},{"2:f=~y4",".mg3."},
    {"1:y1~1",".mg4."},{"2:y1~1",".mg4."},
    {"1:y2~1",".mg5."},{"2:y2~1",".mg5."},
    {"1:y3~1",".mg6."},{"2:y3~1","free"},
    {"1:y4~1",".mg7."},{"2:y4~1",".mg7."},
    {"1:y1~~y1","free"},{"2:y1~~y1","free"},
    {"1:y2~~y2","free"},{"2:y2~~y2","free"},
    {"1:y3~~y3","free"},{"2:y3~~y3","free"},
    {"1:y4~~y4","free"},{"2:y4~~y4","free"},
    {"1:f~~f","free"},{"2:f~~f","free"},
    {"1:f~1","free"},{"2:f~1","free"},
    {"1:y1~~y2","0.000000"},{"2:y1~~y2","free"}};
  CHECK(got==expected);CHECK(model::build_matrix_rep(*st,&names));
}
TEST_CASE("Mplus MODEL: IV05 marker and variance identification counts") {
  for(const auto& method:{"CONFIGURAL","METRIC","SCALAR"}) for(bool variance:{false,true}) {
    auto text=std::string("DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3 y4 y5 y6 g;\nGROUPING=g(1=a 2=b);\nANALYSIS: MODEL=")+method+";\nMODEL: "+(variance?"f1 BY y1* y2-y3; f2 BY y4* y5-y6; f1@1 f2@1;":"f1 BY y1-y3; f2 BY y4-y6;")+"\n";
    auto m=parse::MplusParser::parse(text);REQUIRE_MESSAGE(m,(m?"":m.error().detail));
    spec::LatentNames names;spec::Starts starts;auto st=spec::build(m->flat,compat::mplus::build_options(m->input),&starts,&names);REQUIRE(st);
    std::set<int> eq;for(auto group:st->eq_groups) eq.insert(group);
    CHECK(eq.size()==(std::string(method)=="CONFIGURAL"?38:std::string(method)=="METRIC"?34:30));
  }
}

TEST_CASE("Mplus MODEL: MG06 group-specific variable roles are explicit boundaries") {
  for(const auto& section:{"y4 ON x1;","f BY y4;"}) {
    auto m=parse::MplusParser::parse(std::string("DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3 y4 x1 g;\nGROUPING=g(1=a 2=b);\nMODEL: f BY y1-y3;\nMODEL b: ")+section+"\n");
    REQUIRE_FALSE(m);if(m) continue;CHECK(m.error().detail.find("[MG06]")!=std::string::npos);CHECK(m.error().detail.find("instead")!=std::string::npos);
  }
}

TEST_CASE("Mplus MODEL: MS08 mixed conditioning is rejected in group sections") {
  for(const auto& section:{"[x1];","x1;","y1 WITH x1;"}) {
    auto m=parse::MplusParser::parse(std::string("DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3 x1 g;\nGROUPING=g(1=a 2=b);\nMODEL: f BY y1-y3; f ON x1;\nMODEL b: ")+section+"\n");
    REQUIRE_FALSE(m);if(m) continue;CHECK(m.error().detail.find("[MS08]")!=std::string::npos);CHECK(m.error().detail.find("remove")!=std::string::npos);
  }
}

TEST_CASE("Mplus categorical probes: 9.1 shortcut evidence") {
  auto raw=test::read_fixture(test::fixtures_dir()+"/mplus/probes_categorical.json"); REQUIRE(raw);
  const auto probes=nlohmann::json::parse(*raw,nullptr,false); REQUIRE_FALSE(probes.is_discarded());
  const auto& variants=probes.at("P-IV2").at("variants");
  REQUIRE(variants.size()==12);
  for (const auto& [name, v]:variants.items()) {
    INFO(name);
    if (name.ends_with("_metric")) {
      CHECK(v.at("status")=="error");
      CHECK_FALSE(v.at("diagnostics").empty());
      continue;
    }
    CHECK(v.at("status")=="accepted");
    REQUIRE_FALSE(v.at("tech1").empty());
    if (name.ends_with("_configural")) {
      REQUIRE(v.at("free_parameters").size()==1);
      CHECK(v.at("free_parameters")[0]==(name.find("_binary_")!=std::string::npos ? 26 : 38));
      CHECK(v.at("chi_square")[0].at("df")==16);
    } else if (name.find("_ordinal_")!=std::string::npos) {
      REQUIRE(v.at("free_parameters").size()==1);
      CHECK(v.at("free_parameters")[0]==30);
      CHECK(v.at("chi_square")[0].at("df")==24);
    } else {
      // Binary scalar input meaning is available from TECH1 even though this
      // fixed sample does not converge. It is not numerical golden evidence.
      CHECK(v.at("free_parameters").empty());
      CHECK_FALSE(v.at("estimation_messages").empty());
    }
  }
}

TEST_CASE("Mplus categorical: CT01-CT07 materialization and restrictions") {
  const auto text=[](std::string model,std::string parameterization="DELTA") {
    return "DATA: FILE=x;\nVARIABLE: NAMES=u1-u4; CATEGORICAL=u1-u4;\nANALYSIS: PARAMETERIZATION="+parameterization+";\nMODEL: f BY u1-u4;\n"+model;
  };
  auto completed=compat::mplus::prepare_ordinal_model(text("[u1$1-u1$2]; [u2$1-u4$1] (a);"),{{3,3,3,3}});
  REQUIRE(completed);
  auto pt=compat::lavaan::to_lavaan_partable(completed->structure,completed->names,completed->starts);
  int thresholds=0;
  for(std::size_t i=0;i<pt.size();++i) {
    if(pt.op[i]==parse::Op::Threshold) ++thresholds;
    if(pt.op[i]==parse::Op::Intercept && pt.lhs[i][0]=='u') CHECK(pt.ustart[i]==0);
  }
  CHECK(thresholds==8);
  CHECK_FALSE(compat::mplus::prepare_ordinal_model(text("[u1$3];"),{{3,3,3,3}}));
  CHECK_FALSE(compat::mplus::prepare_ordinal_model(text(""),{{11,3,3,3}}));
  for(const auto& [model,par,rule]:std::vector<std::tuple<std::string,std::string,std::string>>{
      {"[u1];","DELTA","CT02"},{"u1;","DELTA","CT04"},
      {"{u1};","THETA","CT04"},{"u2 ON u1;","DELTA","CT05"}}) {
    auto actual=parse::MplusParser::parse(text(model,par));REQUIRE_FALSE(actual);
    CHECK(actual.error().detail.find("["+rule+"]")!=std::string::npos);
  }
  REQUIRE(parse::MplusParser::parse(text("u1-u2 (s);","THETA")));
  REQUIRE(parse::MplusParser::parse(text("{u1@0.8};")));
  REQUIRE(parse::MplusParser::parse(text("{u1-u2} (s);")));
}

TEST_CASE("Mplus growth: GR01-GR07 polynomial defaults, scores and overrides") {
  auto r=rows(source("i s | y1@0 y2@1 y3@2 y4@3;","","y1 y2 y3 y4"));
  CHECK(r.at("i=~y1")=="1.000000"); CHECK(r.at("s=~y4")=="3.000000");
  CHECK(r.at("i~1")=="free"); CHECK(r.at("s~1")=="free");
  CHECK(r.at("y1~1")=="0.000000"); CHECK(r.at("i~~s")=="free");
  r=rows(source("i s q | y1@0 y2@1 y3@2 y4@3;","","y1 y2 y3 y4"));
  CHECK(r.at("q=~y4")=="9.000000");
  r=rows(source("i s | y1@0 y2@1 y3 y4;","","y1 y2 y3 y4"));
  CHECK(r.at("s=~y3")=="gr_s_y3");
  for(auto m:{"[y1@2]; i s | y1@0 y2@1 y3@2 y4@3;", "i s | y1@0 y2@1 y3@2 y4@3; [y1@2];"}) {
    r=rows(source(m,"","y1 y2 y3 y4"));CHECK(r.at("y1~1")=="2.000000");
  }
  r=rows(source("i s1 | y1@0 y2@1 y3@2 y4@2; i s2 | y1@0 y2@0 y3@0 y4@1;","","y1 y2 y3 y4"));
  CHECK(r.at("i=~y1")=="1.000000"); CHECK(r.at("s2=~y4")=="1.000000");
  reject("i s q | y1@0 y2@1 y3 y4;","GR02");
  reject("s | y1 ON x1;","GR07");
}
TEST_CASE("Mplus constraints: CN01-CN04 declarations, equations, loops and lifetime") {
  auto m=parse::MplusParser::parse(source("y1 ON x1 (p1);\ny2 ON x1 (p2);\nMODEL CONSTRAINT:\nNEW(c*.6 r); p2=p1+c; r=PHI(p1)+SQRT(p2**2)+LOG10(10);"));
  REQUIRE_MESSAGE(m,(m ? "" : m.error().detail));
  auto moved=std::move(*m);spec::LatentNames names;spec::Starts starts;
  auto st=spec::build(moved.flat,compat::mplus::build_options(moved.input),&starts,&names);
  REQUIRE_MESSAGE(st,(st ? "" : st.error().detail));
  CHECK(st->lin_constraint_d.size()==1); CHECK(moved.flat.constraints.size()==2);
  CHECK(std::count(st->op.begin(),st->op.end(),parse::Op::AuxiliaryParam)==1);
  for(std::size_t i=0;i<st->size();++i) if(st->op[i]==parse::Op::AuxiliaryParam) {
    CHECK(names.row_label[i]=="c");CHECK(st->lhs_var[i]==-1);CHECK(starts.hint[static_cast<std::size_t>(st->free[i]-1)]==doctest::Approx(.6));
  }
  auto projected=compat::lavaan::to_lavaan_partable(*st,names,starts);
  auto rebuilt=compat::lavaan::from_lavaan_partable(projected);
  CHECK(rebuilt.structure.n_free()==st->n_free());
  CHECK(rebuilt.structure.lin_constraint_d==st->lin_constraint_d);
  auto rep=model::build_matrix_rep(*st,&names);REQUIRE(rep);
  auto ev=model::ModelEvaluator::build(*st,*rep);REQUIRE(ev);CHECK(ev->n_free()==static_cast<std::size_t>(st->n_free()));
  m=parse::MplusParser::parse(source("y1 ON x1 (p1);\ny2 ON x1 (p2);\nMODEL CONSTRAINT:\nNEW(r1-r2); DO (1,2) r#=p#**2; LOOP(t,0,1,.1); PLOT(r1);"));
  REQUIRE_MESSAGE(m,(m ? "" : m.error().detail));CHECK(m->flat.constraints.size()==2);
  m=parse::MplusParser::parse(source("y1 ON x1 (p1);\nMODEL CONSTRAINT:\nNEW(r11 r12 r21 r22); DO (1,2) DO (1,2) r#$=p1+#*$;"));
  REQUIRE_MESSAGE(m,(m ? "" : m.error().detail));CHECK(m->flat.constraints.size()==4);
  m=parse::MplusParser::parse(source("y1 ON x1 (p1);\nMODEL CONSTRAINT:\nNEW(r); r=p1**2; LOOP(z,4,6,.1); PLOT(indirect,direct); indirect=r*z; direct=p1+z;"));
  REQUIRE_MESSAGE(m,(m ? "" : m.error().detail));CHECK(m->flat.constraints.size()==1);
  reject("y1 ON x1 (a);\nMODEL CONSTRAINT: a>0;","CN01");
}
TEST_CASE("Mplus indirect: CN05 total, specific, VIA and absent reverse paths") {
  auto m=parse::MplusParser::parse(source("y1 ON x1 (a);\ny2 ON y1 (b);\ny3 ON y2 (c);\ny3 ON y1 (d);\nMODEL INDIRECT: y3 IND x1; y3 IND y2 y1 x1; y3 VIA y2 x1; y3 IND y1 y2 x1;"));
  REQUIRE_MESSAGE(m,(m ? "" : m.error().detail)); REQUIRE(m->flat.constraints.size()==4);
  CHECK(parse::expr_to_canonical(m->flat.constraints[0].rhs)=="a*b*c+a*d");
  CHECK(parse::expr_to_canonical(m->flat.constraints[1].rhs)=="1*a*b*c");
  CHECK(parse::expr_to_canonical(m->flat.constraints[2].rhs)=="a*b*c");
  CHECK(parse::expr_to_canonical(m->flat.constraints[3].rhs)=="0");
}

TEST_CASE("Mplus rejection contracts cover malformed constraint growth and threshold classes") {
  for (const auto& [model, rule, instruction] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
         {"y1 ON x1 (a);\nMODEL CONSTRAINT: NEW (r*x);", "CN02", "numeric starts"},
         {"y1 ON x1 (a);\nMODEL CONSTRAINT: DO (3,1) r#=a;", "CN03", "reduce loop limits"},
         {"y1 ON x1;\nMODEL INDIRECT: y1 MOD x1;", "CN05", "y VIA m x"},
         {"y1 s | y1@0 y2@1;", "GR01", "distinct growth factors"},
         {"i s q | y1@0 y2@1 y3;", "GR02", "fix at least one time score"}}) {
    const auto result = parse::MplusParser::parse(source(model));
    REQUIRE_FALSE(result);
    if (result) continue;
    CHECK(result.error().span.line > 0);
    CHECK(result.error().detail.find("["+rule+"]") != std::string::npos);
    for (const auto* part : {"found '", "Mplus", "magmaan", "instead"})
      CHECK_MESSAGE(result.error().detail.find(part) != std::string::npos, result.error().detail);
    CHECK_MESSAGE(result.error().detail.find(instruction) != std::string::npos, result.error().detail);
  }
  for (const auto& [model, rule, instruction] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
         {"f BY u1-u3; [u1$0];", "CT01", "two through ten categories"},
         {"f BY u1-u3; [u1$2-u1$1];", "CT03", "ascending order"},
         {"u2 ON u1;", "CT07", "model each CATEGORICAL variable as an outcome"}}) {
    const auto result = parse::MplusParser::parse(
      "DATA: FILE=x;\nVARIABLE: NAMES=u1-u3; CATEGORICAL=u1-u3;\nMODEL: "+model);
    REQUIRE_FALSE(result);
    if (result) continue;
    CHECK(result.error().span.line > 0);
    CHECK(result.error().detail.find("["+rule+"]") != std::string::npos);
    for (const auto* part : {"found '", "Mplus", "magmaan", "instead"})
      CHECK_MESSAGE(result.error().detail.find(part) != std::string::npos, result.error().detail);
    CHECK_MESSAGE(result.error().detail.find(instruction) != std::string::npos, result.error().detail);
  }
}
