#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "magmaan/parse/mplus_parser.hpp"

namespace {
using magmaan::parse::MplusParser;
using magmaan::parse::MplusClass;

std::string input(const std::string& variable = "NAMES = y1 y2 y3;",
                  const std::string& analysis = "", const std::string& data = "FILE = probe.dat;") {
  // Keep unrelated option-classification cases inside the physical line limit.
  const auto records = [](const std::string& options) {
    std::string result;
    char quote = 0;
    for (char c : options) {
      result += c;
      if (quote) { if (c == quote) quote = 0; }
      else if (c == '\'' || c == '"') quote = c;
      else if (c == ';') result += '\n';
    }
    return result;
  };
  return "DATA: " + records(data) + "\nVARIABLE: " + variable + "\nANALYSIS: " + records(analysis) + "\nMODEL: f BY y1-y3;\n";
}
void rejection(const std::string& source, const std::string& rule, const std::string& message = "") {
  const auto result = MplusParser::read(source);
  REQUIRE_FALSE(result.has_value());
  if (result) return;  // No-exception doctest assertions do not unwind helpers.
  CHECK(result.error().kind == magmaan::ParseError::Kind::RejectedConstruct);
  CHECK_MESSAGE(result.error().detail.find("[" + rule + "]") != std::string::npos, result.error().detail);
  CHECK_MESSAGE(result.error().detail.find(message) != std::string::npos, result.error().detail);
}
std::vector<std::string> split(const std::string& text) {
  std::istringstream stream(text);
  std::vector<std::string> result;
  for (std::string word; stream >> word;) result.push_back(word);
  return result;
}
}

TEST_CASE("Mplus input: LX01 LX04 LX06 LX07 command boundaries and owned spans") {
  const std::string source = "title: test anal: x model: x\nmore title: words\n"
      "  vari: names ARE Y1, y2 Y3; usev IS y3 Y1;\n"
      "data: file = 'path with ! and ;.dat';\nanal: esti = ML;\n"
      "model: f BY y1-y3;\noutput: arbitrary unknown;\n";
  auto r = MplusParser::read(source);
  REQUIRE_MESSAGE(r.has_value(), (r ? "" : r.error().detail));
  CHECK(r->source == source);
  CHECK(r->names == std::vector<std::string>{"Y1", "y2", "Y3"});
  CHECK(r->analysis == std::vector<std::string>{"Y3", "Y1"});
  CHECK(r->estimator == "ML");
  CHECK(r->source.substr(r->model_body.begin, r->model_body.end - r->model_body.begin) == " f BY y1-y3;\n");
  CHECK(r->model_body.line == 6);
  CHECK(r->model_body.col == 7);
  REQUIRE(r->notes.size() == 4);
  CHECK(r->notes[0].rule == "CL01");
  CHECK(r->notes[0].span.begin == 0);
  CHECK(r->notes[1].klass == MplusClass::DataDescription);
  CHECK(r->notes[2].rule == "CL18");
  CHECK(r->notes[3].rule == "CL30");
  rejection("VARIABLE: NAMES=y1;", "LX01", "required command: DATA");
  rejection("DATA: FILE=x;", "LX01", "required command: VARIABLE");
  rejection(input() + "DATA: FILE=x;\n", "LX01", "repeated command");
  rejection("DATA: FILE=x; VARIABLE: NAMES=y1;\n", "LX01");
  rejection(input("NAMES=y1 y2; USEV=y1"), "LX01", "semicolon");
  rejection("garbage\n" + input(), "LX01", "before first command");
  auto absent = MplusParser::read("DATA: FILE=x;\nVARIABLE: NAMES=y1;\n");
  REQUIRE(absent);
  CHECK(absent->model_body.begin == absent->model_body.end);
  CHECK(absent->analysis == absent->names);
}

TEST_CASE("Mplus input: LX02 LX05 physical lines and comments") {
  const auto base = input();
  auto exact = MplusParser::read(base + "!" + std::string(89, 'x') + "\n");
  CHECK(exact.has_value());
  CHECK(MplusParser::read(base + "!" + std::string(200, 'x') + "\n"));
  CHECK(MplusParser::read("TITLE: " + std::string(200, 'x') + "\n" + base));
  rejection(base + "MODEL: " + std::string(90, ' ') + "y1;\n", "LX02", "column 90");
  CHECK(MplusParser::read("!* block\ncomment *!\n" + base).has_value());
  CHECK(MplusParser::read(input("NAMES=y1 !* inline *! y2 y3; ! ignored" )).has_value());
  rejection(input("NAMES=y1; !* unsafe\ncomment *!"), "LX05", "opening line");
  rejection(base + "!* unclosed", "LX05", "unterminated");
  CHECK(MplusParser::read("! line\n\t!* block *!\n" + base).has_value());
  CHECK(MplusParser::read(input("NAMES=y1,y2,y3;", "", "FILE=x;\r")).has_value());
}

TEST_CASE("Mplus input: LX03 unique option prefixes and exact setting stems") {
  CHECK(MplusParser::read(input("NAME=y1 y2 y3; USEVAR=y1-y3;", "TYPE=GEN; ESTI=ML; MODEL=NOCOV;")).has_value());
  rejection(input("NAMES=y1 y2 y3; USE=y1;"), "LX03", "option");
  rejection(input("NAMES=y1 y2 y3;", "TYPE=GENE;"), "LX03", "setting");
  rejection(input("NAMES=y1 y2 y3;", "MODEL=NOCOVAR;"), "LX03");
  auto execution = MplusParser::read(input("NAMES=y1 y2 y3;", "STCO=1;"));
  REQUIRE(execution);
  CHECK(execution->notes.back().rule == "CL25");
  CHECK(execution->notes.back().message == "STCONVERGENCE is recognized but not imported");
  rejection(input("NAMES=y1 y2 y3;", "STCX=1;"), "LX03", "unknown");
  rejection(input() + "UNKNOWN: x;\n", "LX03", "command");
  rejection(input() + "OUT: x;\n", "LX03");
  for (const auto& command : {"OUTPUT", "SAVEDATA", "PLOT"}) {
    const auto r = MplusParser::read(input() + command + ": anything(@-;\n");
    REQUIRE(r);
    CHECK(r->notes.back().rule == "CL30");
  }
}

TEST_CASE("Mplus input: NM01 NM02 NAMES expansion and full-name identity") {
  auto r = MplusParser::read(input("NAMES = y08-y11 itema-itemd abcdefgh1 abcdefgh2;"));
  REQUIRE(r);
  CHECK(r->names == std::vector<std::string>{"y08", "y09", "y10", "y11", "itema", "itemb", "itemc", "itemd", "abcdefgh1", "abcdefgh2"});
  auto ordinary = MplusParser::read(input("NAMES=y1-y12;"));
  REQUIRE(ordinary);
  CHECK(ordinary->names.front() == "y1");
  CHECK(ordinary->names.back() == "y12");
  CHECK(ordinary->names.size() == 12);
  rejection(input("NAMES=y1 Y1;"), "NM02", "duplicate");
  rejection(input("NAMES=y3-y1;"), "NM02");
  rejection(input("NAMES=itemd-itema;"), "NM02");
  rejection(input("NAMES=a1b-a3b;"), "NM02", "shape");
  rejection(input("NAMES=y1-z3;"), "NM02", "shape");
  rejection(input("NAMES=y0-y10001;"), "NM02", "10001");
  rejection(input("NAMES=y1-y99999999999999999999999;"), "NM02");
  rejection(input("NAMES=1y;"), "NM02");
  CHECK(MplusParser::read(input("NAMES=y0-y10000;" )).has_value());
  auto multiple = MplusParser::read(input("NAMES=y0-y10000 z;"));
  REQUIRE(multiple);
  CHECK(multiple->names.size() == 10002);  // Limit is per expansion, not per file.
}

TEST_CASE("Mplus input: CL08 NM03 USEVARIABLES selects NAMES positions") {
  auto r = MplusParser::read(input("NAMES=Y1 x1 y2 y3; USEVARIABLES=y1-y3;"));
  REQUIRE(r);
  CHECK(r->analysis == std::vector<std::string>{"Y1", "x1", "y2", "y3"});
  r = MplusParser::read(input("USEV=y3 Y1 y2 x1; NAMES=Y1 x1 y2 y3;"));
  REQUIRE(r);
  CHECK(r->analysis == std::vector<std::string>{"y3", "Y1", "y2", "x1"});
  rejection(input("NAMES=y1 y2 y3; USEV=ALL y1;"), "NM03", "duplicate");
  rejection(input("NAMES=y1 y2 y3; USEV=y3-y1;"), "NM03", "backward");
  rejection(input("NAMES=y1 y2 y3; USEV=unknown;"), "NM03", "unknown");
  rejection(input("NAMES=y1 y2 y3; USEV=y1 ALL;"), "NM03");
  rejection(input("NAMES=y1 y2 y3; USEV=;"), "NM03", "empty");
}

TEST_CASE("Mplus input: CL17 CL19 CL20 CL21 CL18 CL22 CL23 MS11 settings") {
  auto r = MplusParser::read(input("NAMES=y1 y2 y3;", "TYPE=GEN MISSING MEANSTRUCTURE H1; MODEL=NOMEAN NOCOV; INFORMATION=EXP; ESTIMATOR=MLR;"));
  REQUIRE(r);
  CHECK(r->type_settings == std::vector<std::string>{"GENERAL", "MISSING", "MEANSTRUCTURE", "H1"});
  CHECK(r->nomeanstructure);
  CHECK(r->nocovariances);
  CHECK(r->information == "EXPECTED");
  CHECK(r->estimator == "MLR");
  CHECK(std::count_if(r->notes.begin(), r->notes.end(), [](const auto& n) { return n.rule == "CL17"; }) == 3);
  for (auto s : split("BASIC BAS RANDOM RAND COMPLEX COM MIXTURE MIX TWOLEVEL TWO THREELEVEL THREE CROSSCLASSIFIED CROSS EFA"))
    rejection(input("NAMES=y1;", "TYPE=" + s + ";"), "CL17");
  for (auto s : split("CONFIGURAL CONFIG METRIC SCALAR")) rejection(input("NAMES=y1;", "MODEL=" + s + ";"), "IV01", "GROUPING");
  rejection(input("NAMES=y1;", "MODEL=ALL;"), "CL21");
  for (auto s : {"BAYES", "MUML"}) rejection(input("NAMES=y1;", std::string("ESTIMATOR=") + s + ";"), "CL18");
  for (auto s : split("ML MLM MLMV MLR MLF WLS WLSM WLSMV ULS ULSMV GLS")) {
    auto fit = MplusParser::read(input("NAMES=y1;", "ESTIMATOR=" + s + ";"));
    REQUIRE(fit);
    CHECK(fit->estimator == s);
  }
  for (auto s : split("DELTA THETA LOGIT LOGLIN PROB RESCOV")) rejection(input("NAMES=y1;", "PARAMETERIZATION=" + s + ";"), "CL22");
  for (auto s : split("SKEW TDIST SKEWT")) rejection(input("NAMES=y1;", "DISTRIBUTION=" + s + ";"), "CL23");
  rejection(input("NAMES=y1;", "MATRIX=CORR;"), "CL23");
  CHECK(MplusParser::read(input("NAMES=y1;", "DISTRIBUTION=NORM; MATRIX=COVA;")).has_value());
  rejection(input("NAMES=y1;", "PARAMETERIZATION=DELT;"), "LX03");
  for (auto setting : {"OBSERVED", "OBS", "COMBINATION", "COMB"}) {
    auto info = MplusParser::read(input("NAMES=y1;", std::string("INFORMATION=") + setting + ";"));
    REQUIRE(info);
    CHECK(info->information == (std::string(setting).starts_with("OBS") ? "OBSERVED" : "COMBINATION"));
  }
  for (auto s : {"", "INFORMATION=OBS;", "INFORMATION=COMB;"})
    rejection(input("NAMES=y1;", std::string("MODEL=NOMEAN;") + s), "MS11", "Mplus ignores NOMEANSTRUCTURE");
  rejection(input("NAMES=y1;", "ESTIMATOR=M;"), "LX03");
  rejection(input("NAMES=y1;", "INFORMATION=EXPE;"), "LX03");
}

TEST_CASE("Mplus input: CL02 CL03 CL04 CL05 DATA option classification") {
  auto r = MplusParser::read(input("NAMES=y1;", "", "FILE=probe.dat; FORMAT=3F2.1; TYPE=IND; NOBSERVATIONS=20; LISTWISE=ON; VARIANCES=CHECK;"));
  REQUIRE(r);
  REQUIRE(r->notes.size() == 6);
  for (std::size_t i = 0; i < 5; ++i) CHECK(r->notes[i].klass == MplusClass::DataDescription);
  CHECK(r->notes[5].klass == MplusClass::Reported);
  CHECK(r->notes[5].rule == "CL05");
  for (auto s : split("COVARIANCE COVA CORRELATION CORR FULLCOV FULLCORR MEANS STDEVIATIONS STD"))
    if(s=="MEANS" || s=="STDEVIATIONS" || s=="STD") rejection(input("NAMES=y1;", "", "TYPE=" + s + ";"), "CL03");
    else CHECK(MplusParser::read(input("NAMES=y1;", "", "TYPE=" + s + "; NOBSERVATIONS=20;")));
  for (auto s : split("MONTECARLO MONTE IMPUTATION IMP")) rejection(input("NAMES=y1;", "", "TYPE=" + s + ";"), "CL04");
  rejection(input("NAMES=y1;", "", "SWMATRIX=x;"), "CL04");
  CHECK(MplusParser::read(input("NAMES=y1;", "", "FILE (g1)=x;")));
  CHECK(MplusParser::read(input("NAMES=y1;", "", "FILE='file(with parens).dat';")).has_value());
  rejection(input("NAMES=y1;", "", "NGROUPS=2;"), "MG02");
}

TEST_CASE("Mplus input: CL07 CL09 CL10 CL11 CL12 CL13 CL14 CL15 VARIABLE classification") {
  auto r = MplusParser::read(input("NAMES=y1 y2; MISSING=ALL(-99); IDVARIABLE=y1; AUXILIARY=y2;"));
  REQUIRE(r);
  REQUIRE(r->notes.size() == 4);
  CHECK(r->notes[1].rule == "CL09");
  CHECK(r->notes[1].klass == MplusClass::DataDescription);
  CHECK(r->notes[2].rule == "CL12");
  CHECK(r->notes[3].rule == "CL12");
  rejection(input("NAMES=y1; CATEGORICAL=y1;"), "CL10", "increment 3");
  CHECK(MplusParser::read(input("NAMES=y1 g; GROUPING=g(1=a 2=b);")));
  rejection(input("NAMES=y1; AUXILIARY=y1(m);"), "CL15");
  for (const auto& name : split("USEOBSERVATIONS SUBPOPULATION")) rejection(input("NAMES=y1; " + name + "=x;"), "CL13");
  for (const auto& name : split("CENSORED NOMINAL COUNT DSURVIVAL TSCORES SURVIVAL TIMECENSORED LAGGED TINTERVAL")) rejection(input("NAMES=y1; " + name + "=x;"), "CL14");
  for (const auto& name : split("FREQWEIGHT CONSTRAINT PATTERN STRATIFICATION CLUSTER WEIGHT WTSCALE BWEIGHT B2WEIGHT B3WEIGHT BWTSCALE REPWEIGHTS FINITE CLASSES KNOWNCLASS TRAINING WITHIN BETWEEN")) rejection(input("NAMES=y1; " + name + "=x;"), "CL15");
}

TEST_CASE("Mplus input: CL24 CL25 ANALYSIS execution option inventory") {
  for (const auto& name : split("ROTATION ROWSTANDARDIZATION PARALLEL REPSE MULTIPLIER BASEHAZARD RSTARTS RITERATIONS RCONVERGENCE ASTARTS AITERATIONS ACONVERGENCE SIMPLICITY TOLERANCE METRIC"))
    rejection(input("NAMES=y1;", name + "=x;"), "CL24");
  rejection(input("NAMES=y1;", "LINK=x;"), "CL23");
  rejection(input("NAMES=y1;", "ALIGNMENT=x;"), "CL21");
  for (const auto& name : split("CHOLESKY ALGORITHM INTEGRATION MCSEED ADAPTIVE BOOTSTRAP LRTBOOTSTRAP STARTS STITERATIONS STCONVERGENCE STSCALE STSEED OPTSEED K-1STARTS LRTSTARTS H1STARTS DIFFTEST COVERAGE ADDFREQUENCY ITERATIONS SDITERATIONS H1ITERATIONS MITERATIONS MCITERATIONS MUITERATIONS CONVERGENCE H1CONVERGENCE LOGCRITERION RLOGCRITERION MCONVERGENCE MCCONVERGENCE MUCONVERGENCE MIXC MIXU LOGHIGH LOGLOW UCELLSIZE VARIANCE POINT CHAINS BSEED STVALUES PREDICTOR BCONVERGENCE BITERATIONS FBITERATIONS THIN MDITERATIONS KOLMOGOROV PRIOR INTERACTIVE PROCESSORS NESTED")) {
    const auto r = MplusParser::read(input("NAMES=y1;", name + "=1;"));
    REQUIRE_MESSAGE(r.has_value(), name, (r ? "" : r.error().detail));
    CHECK(r->notes.back().rule == "CL25");
    CHECK(r->notes.back().klass == MplusClass::Reported);
  }
}

TEST_CASE("Mplus input: CL06 CL16 CL26 CL27 CL28 CL29 CL31 CL32 opaque commands") {
  for (const auto& q : split("IMPUTATION WIDETOLONG LONGTOWIDE TWOPART MISSING SURVIVAL COHORT"))
    rejection(input() + "DATA " + q + ": anything;\n", "CL06");
  rejection(input() + "DEFINE: y1=y2;\n", "CL16");
  rejection(input() + "MONTECARLO: anything;\n", "CL29");
  for (const auto& q : {"CONSTRAINT", "INDIRECT"}) rejection(input() + "MODEL " + q + ": anything;\n", "CL27", "increment 4");
  for (const auto& q : {"POPULATION", "COVERAGE", "MISSING", "POPULATION-g1"}) rejection(input() + "MODEL " + q + ": anything;\n", "CL29");
  rejection(input() + "MODEL PRIORS: anything;\n", "CL32");
  rejection(input() + "MODEL TEST: 0=a;\nMODEL TEST: 0=b;\n", "LX01", "repeated command");
  rejection(input() + "MODEL g1: y1;\n", "CL31");
  CHECK(MplusParser::read(input("NAMES=y1 g; GROUPING=g(1=g1 2=g2);") + "MODEL g1: y1;\n"));
  auto r = MplusParser::read(input() + "MODEL TEST: 0=a-b;\n");
  REQUIRE(r);
  CHECK(r->notes.back().rule == "CL28");
  CHECK(r->notes.back().klass == MplusClass::Reported);
  // The input reader deliberately leaves every MODEL statement to the next card.
  CHECK(MplusParser::read(input() + "! model syntax was not parsed\n").has_value());
}

TEST_CASE("Mplus input: aggregate rejection order and arbitrary input stability") {
  auto r = MplusParser::read("DATA: SWMATRIX=x;\nVARIABLE: NAMES=y1; WEIGHT=w;\nDEFINE: y1=2;\nANALYSIS: TYPE=BASIC;\n");
  REQUIRE_FALSE(r);
  CHECK(r.error().span.line == 1);
  CHECK(r.error().span.col == 7);
  const auto& d = r.error().detail;
  CHECK(d.find("[CL04]") < d.find("[CL15]"));
  CHECK(d.find("[CL15]") < d.find("[CL16]"));
  CHECK(d.find("[CL16]") < d.find("[CL17]"));
  std::mt19937 random(511);
  for (int n = 0; n < 1000; ++n) {
    std::string source;
    for (unsigned i = 0, count = random() % 500; i < count; ++i) source += static_cast<char>(random() % 256);
    auto result = MplusParser::read(source);
    if (result) CHECK(result->source == source);
    else CHECK(result.error().span.end <= source.size());
  }
}

TEST_CASE("Mplus input: Mplus 9.1 Demo input-reader agreement gate") {
  std::ifstream file(std::string(MAGMAAN_FIXTURES_DIR) + "/mplus/probes.json");
  REQUIRE(file.good());
  const auto probes = nlohmann::json::parse(file, nullptr, false);
  REQUIRE_FALSE(probes.is_discarded());
  std::size_t count = 0, gated = 0;
  for (const auto& [id, probe] : probes.items()) for (const auto& [variant, v] : probe.at("variants").items()) {
    const auto title = v.at("title").is_null() ? "TITLE: " + id + " " + variant + ";" : v.at("title").get<std::string>();
    const auto source = title + "\nDATA: FILE = probe.dat; " + v.at("data").get<std::string>() +
        "\nVARIABLE:\n" + v.at("variable").get<std::string>() + "\nANALYSIS:\n" +
        v.at("analysis").get<std::string>() + "\nMODEL:\n" + v.at("model").get<std::string>() + "\nOUTPUT: TECH1;\n";
    const auto result = MplusParser::read(source);
    ++count;
    INFO(id, "/", variant, (result ? " accepted" : result.error().detail));
    const bool gate = id == "P-LX2" || id == "P-LX3" || id == "P-LX4" || id == "P-NM1" || id == "P-NM2" ||
        (id == "P-NM3" && variant == "names_order") || id == "P-DF5" || id == "P-DF6";
    if (!gate) continue;  // MODEL meaning is gated by task-51.2, not this reader.
    ++gated;
    std::string deviation;
    if (id == "P-LX4" && variant == "columns_91") deviation = "LX02";
    if (id == "P-NM1" && variant == "mixed_range") deviation = "NM02";
    if (id == "P-DF5" && variant == "single") deviation = "MS11";
    if (id == "P-DF5" && variant == "groups") deviation = "MS11";
    if (!deviation.empty()) {
      REQUIRE_FALSE(result);
      CHECK(result.error().detail.find("[" + deviation + "]") != std::string::npos);
    } else CHECK(result.has_value() == (v.at("status") == "accepted"));
  }
  CHECK(count == 116);
  CHECK(gated > 15);
}

TEST_CASE("Mplus input: reader refinements preserve actionable boundaries") {
  rejection(input("NAMES=y1 y2 y3; USEV=y1 y1;"), "NM03", "duplicate USEVARIABLES name 'y1'");
  rejection(input("NAMES=y1;", "TYPE=EFA 1 3;"), "CL17", "EFA");
  auto efa=MplusParser::read(input("NAMES=y1;", "TYPE=EFA 1 3;")); REQUIRE_FALSE(efa);
  CHECK(efa.error().detail.find("unknown setting")==std::string::npos);
  rejection(input("NAMES=y1;", "TYPE=EFAUNKNOWN;"), "LX03", "accepted words/stems");
  auto summary=MplusParser::read(input("NAMES=y1;", "", "FILE=x; TYPE=CORRELATION MEANS STDEVIATIONS;")); REQUIRE_FALSE(summary);
  CHECK(summary.error().detail.find("expected one TYPE")==std::string::npos);
  CHECK(MplusParser::read(input("NAMES=y1;", "", "FILE(g1)=x;")+"MODEL g1: y1;\n"));
  rejection(input("NAMES=y1; CLASSES=c(2);")+"MODEL c1: y1;\n", "MS10", "mixture class");
  std::string many="NAMES=";
  for(int n=0;n<10;++n) many+="v"+std::string(1,static_cast<char>('a'+n))+"0-v"+std::string(1,static_cast<char>('a'+n))+"10000\n";
  many+=';'; rejection(input(many),"NM02","100000");
}

TEST_CASE("Mplus input: MG03 explicit integer codes and cumulative sections") {
  auto m=MplusParser::read(input("NAMES=Y1 G; GROUPING IS g (2.0 = B, -1 = A);")+"MODEL a: [y1];\nMODEL A: y1@1;\n");
  REQUIRE(m); CHECK(m->grouping_variable=="G"); REQUIRE(m->groups.size()==2);
  CHECK(m->groups[0].code=="-1");CHECK(m->groups[1].code=="2");
  CHECK(m->groups[0].label=="A");CHECK(m->group_sections.size()==2);
  for (const auto& value:{"g (2)","g (1 2)","g (1-2)","g (1=a 1.0=b)","g (1=a 2=A)","g h (1=a 2=b)","g (1=a 2.5=b)"})
    rejection(input(std::string("NAMES=y1 g h; GROUPING=")+value+";"),"MG03","instead");
  rejection(input("NAMES=y1 g; GROUPING=g(1=a 2=b);")+"MODEL a b: y1;\n","MG06","one");
  rejection(input("NAMES=y1 g; GROUPING=g(1=a 2=b);")+"MODEL c: y1;\n","MG06","declared");
  rejection(input("NAMES=y1 g; GROUPING=g(1=a 2=b);","MODEL=CONFIGURAL METRIC;"),"IV04","exactly one");
}

TEST_CASE("Mplus data plans: fixed formats and missing flags") {
  auto parsed=MplusParser::read(input("NAMES = y1 y2 y3;\nMISSING = ALL (-9, -7--5);", "", "FILE = 'some path.dat'; FORMAT = (2(F2.1,1X),T10,3.0,/);") );
  REQUIRE(parsed.has_value());
  CHECK(parsed->data_plan.files[0].path=="some path.dat");
  CHECK(parsed->data_plan.format.size()==7);
  CHECK(parsed->data_plan.missing[0].values==std::vector<double>{-9,-7,-6,-5});
  CHECK(parsed->data_plan.missing[0].variables==std::vector<std::string>{"y1","y2","y3"});
  for(const auto& format:{"F0.1","2(F2.1","T0,F2.1","F2.","0F2.1","(X)","F2.4"})
    rejection(input("NAMES=y1 y2 y3;","",std::string("FORMAT=")+format+";"),"DA01");
  rejection(input("NAMES=y1 y2 y3;\nMISSING=BLANK;"),"DA03");
  rejection(input("NAMES=y1 y2 y3;\nMISSING=ALL (oops);"),"DA03");
  rejection(input("NAMES=y1 y2 y3;\nMISSING=z (99);"),"DA03");
  auto symbol=MplusParser::read(input("NAMES=y1 y2 y3;\nMISSING=*;"));
  REQUIRE(symbol); CHECK(symbol->data_plan.missing_symbol=="*");
}

TEST_CASE("Mplus data plans: summary and separate-file groups") {
  auto parsed=MplusParser::read(input("NAMES=y1 y2 y3;", "", "FILE(a)=a.dat; FILE(b)=b.dat; NOBSERVATIONS=20 30;"));
  REQUIRE(parsed); CHECK(parsed->groups.size()==2); CHECK(parsed->groups[0].label=="a");
  CHECK(parsed->groups[0].code.empty()); CHECK(parsed->grouping_variable==".mplus_group");
  auto summary=MplusParser::parse(input("NAMES=y1 y2 y3;", "", "TYPE=COVA; NGROUPS=2; NOBSERVATIONS=20 30;"));
  REQUIRE(summary); CHECK(summary->input.groups[1].label=="g2"); CHECK(summary->input.nomeanstructure);
  for(const auto& row:summary->flat.rows) CHECK(row.op!=magmaan::parse::Op::Intercept);
  for(const auto& type:{"COVA","CORR","FULLCOV","FULLCORR"}) {
    auto result=MplusParser::read(input("NAMES=y1 y2 y3;","",std::string("TYPE=")+type+" MEANS; NOBSERVATIONS=20;"));
    REQUIRE(result); CHECK(result->data_plan.means);
  }
  rejection(input("NAMES=y1 y2 y3;","","TYPE=COVA;"),"DA02");
  rejection(input("NAMES=y1 y2 y3;","","TYPE=COVA STD; NOBSERVATIONS=20;"),"CL03");
  rejection(input("NAMES=y1 y2 y3;","","TYPE=IND MEANS;"),"CL03");
  rejection(input("NAMES=y1 y2 y3;","","NGROUPS=2;"),"MG02");
  rejection(input("NAMES=y1 y2 y3;","","TYPE=CORR; NGROUPS=2; NOBSERVATIONS=20;"),"DA02");
  // An unqualified explicit bracket is independently rejected by the MODEL parser.
  auto source=input("NAMES=y1 y2 y3;","","TYPE=COVA; NOBSERVATIONS=20;");
  source.insert(source.find("f BY"),"[y1]; ");
  auto invalid=MplusParser::parse(source); REQUIRE_FALSE(invalid); CHECK(invalid.error().detail.find("DA02")!=std::string::npos);
}

TEST_CASE("Mplus data Demo derived agreement evidence") {
  std::ifstream file(std::string(MAGMAAN_FIXTURES_DIR)+"/mplus/data_summary.json");
  REQUIRE(file.good());
  const auto evidence=nlohmann::json::parse(file,nullptr,false);
  REQUIRE_FALSE(evidence.is_discarded());
  CHECK(evidence.at("covariance_divisor")=="N");
  CHECK(evidence.at("comparisons").size()==28);
  for(const auto& row:evidence.at("comparisons")) {
    CHECK(row.at("n").get<int>()==5);
    CHECK(row.at("max_covariance_error").get<double>()<=0.00050001);
    if(!row.at("max_mean_error").is_null()) CHECK(row.at("max_mean_error").get<double>()<=0.00050001);
  }
}
