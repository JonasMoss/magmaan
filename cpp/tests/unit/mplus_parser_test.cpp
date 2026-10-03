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
TEST_CASE("Mplus MODEL: later constructs have classified rejections") {
  reject("{y1};","CT04"); reject("[y1$1];","CT02"); reject("i s | y1@0 y2@1;","GR01"); reject("%OVERALL% y1 ON x1;","MS10"); reject("y1#1;","MS10"); reject("f BY y1-y3 (*rot);","MS09"); reject("f BY y1~0 y2;","MS09");
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
    spec::LatentNames names; auto s=spec::build(m->flat,compat::mplus::build_options(m->input),nullptr,&names); REQUIRE(s);
    std::set<int> distinct(s->eq_groups.begin(),s->eq_groups.end());
    CHECK(distinct.size()==v["free_parameters"][0].get<std::size_t>()); ++checked;
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
      else if(pt.op[i]==parse::Op::Regression) matrices={"BETA"};
      else if(pt.op[i]==parse::Op::Covariance) matrices={"THETA","PSI"};
      else if(pt.op[i]==parse::Op::Intercept) {matrices={"NU","ALPHA"};row="vector";col=lhs;}
      else continue;
      std::optional<int> number;
      for(const auto& matrix:v["tech1"]) {
        if(std::find(matrices.begin(),matrices.end(),matrix["name"].get<std::string>())==matrices.end()) continue;
        const auto& cells=matrix["rows"];
        if(cells.contains(row) && cells[row].contains(col)) { const int n=cells[row][col].get<int>(); if(!number || n>0) number=n; }
        else if(pt.op[i]==parse::Op::Covariance && cells.contains(col) && cells[col].contains(row)) {const int n=cells[col][row].get<int>();if(!number || n>0) number=n;}
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
