#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <utility>
#include <variant>

#include "magmaan/compat/eqs/model.hpp"
#include "magmaan/api/sem.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/parse/eqs_parser.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {
using magmaan::parse::EqsParser;
using magmaan::compat::lavaan::LavaanParTable;

LavaanParTable partable(const magmaan::parse::FlatPartable& flat) {
  magmaan::spec::LatentNames names;
  magmaan::spec::Starts starts;
  auto result = magmaan::spec::build(flat, magmaan::compat::eqs::build_options(), &starts, &names);
  REQUIRE_MESSAGE(result.has_value(), (result ? "" : result.error().detail));
  return magmaan::compat::lavaan::to_lavaan_partable(*result, names, starts);
}
std::size_t row(const LavaanParTable& pt, const std::string& lhs,
                const std::string& op, const std::string& rhs) {
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (magmaan::parse::to_string(pt.op[i]) == op &&
        ((pt.lhs[i] == lhs && pt.rhs[i] == rhs) ||
         (op == "~~" && pt.lhs[i] == rhs && pt.rhs[i] == lhs))) return i;
  FAIL(("missing row " + lhs + op + rhs));
  return 0;
}
}

TEST_CASE("EQS: fixed unit paths differ from free paths starting at one") {
  auto flat = EqsParser::parse(
      "/EQU V1=1F1+E7; V2=1*F1+E8; V3=*F1+E9; /VAR F1=1*; E7-E9=.5*;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  auto a = row(pt, "F1", "=~", "V1");
  auto b = row(pt, "F1", "=~", "V2");
  auto c = row(pt, "F1", "=~", "V3");
  CHECK(pt.free[a] == 0);
  CHECK(pt.ustart[a] == 1.0);
  CHECK(pt.free[b] > 0);
  CHECK(pt.ustart[b] == 1.0);
  CHECK(pt.free[c] > 0);
  CHECK(std::isnan(pt.ustart[c]));
  CHECK(pt.free[row(pt, "V1", "~~", "V1")] > 0);
  CHECK(pt.ustart[row(pt, "V1", "~~", "V1")] == .5);
}

TEST_CASE("EQS: unit factor variance leaves every loading free") {
  auto flat = EqsParser::parse(
      "/EQUATIONS V1=*F1+E1; V2=.7*F1+E2; V3=*F1+E3; /VARIANCES F1=1;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  for (const auto* v : {"V1", "V2", "V3"}) CHECK(pt.free[row(pt, "F1", "=~", v)] > 0);
  CHECK(pt.free[row(pt, "F1", "~~", "F1")] == 0);
  CHECK(pt.ustart[row(pt, "F1", "~~", "F1")] == 1.0);
}

TEST_CASE("EQS: staged C++ model preserves source and identification") {
  const std::string source = "/EQU V1=*F1+E1; V2=.7*F1+E2; V3=.8*F1+E3; /VAR F1=1;";
  auto model = magmaan::api::model_from_eqs(source, {"x1", "x2", "x3"});
  REQUIRE(model.has_value());
  CHECK(model->source() == source);
  CHECK_FALSE(model->options().build.auto_fix_first);
  CHECK_FALSE(model->options().build.fixed_x);
  CHECK(model->matrix_rep().ov_names[0] == std::vector<std::string>{"x1", "x2", "x3"});
}

TEST_CASE("EQS: original source survives moves, case folding and mapped names") {
  const std::string source = "! model\n/equ v1=.7*f1+e7;\nv2=F1+e8; /var f1=1;";
  auto parsed = EqsParser::parse(source, {"first_item", "second.item"});
  REQUIRE(parsed.has_value());
  auto flat = std::move(*parsed);
  CHECK(flat.source() == source);
  CHECK(flat.rows[0].lhs == "F1");
  CHECK(flat.rows[0].rhs == "first_item");
  CHECK(flat.rows[0].span.line == 2);
  CHECK(source.substr(flat.rows[0].span.begin,
                      flat.rows[0].span.end - flat.rows[0].span.begin) == "f1");
  CHECK(flat.rows[1].rhs == "second.item");
  auto pt = partable(flat);
  CHECK(pt.free[row(pt, "F1", "=~", "second.item")] == 0);
}

TEST_CASE("EQS: omitted covariances stay zero and errors map by owning equation") {
  auto flat = EqsParser::parse(
      "/EQU V1=F1+E8; V2=*F1+E7; V3=F2+E6; V4=*F2+E5; /COV E8,E5=.2*;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  CHECK(pt.ustart[row(pt, "V1", "~~", "V4")] == .2);
  bool factor_covariance = false;
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (pt.op[i] == magmaan::parse::Op::Covariance && pt.lhs[i] != pt.rhs[i] &&
        (pt.lhs[i] == "F1" || pt.lhs[i] == "F2")) factor_covariance = true;
  CHECK_FALSE(factor_covariance);
}

TEST_CASE("EQS: structural disturbances become latent residual variances") {
  auto flat = EqsParser::parse(
      "/EQU V1=F1+E1; V2=*F1+E2; V3=F2+E3; V4=*F2+E4;"
      "F2=.4*F1-.2*V5+D8; /VAR F1=1; D8=.7*; V5=2; /COV F1,V5=.1*;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  CHECK(pt.ustart[row(pt, "F2", "~", "F1")] == .4);
  CHECK(pt.ustart[row(pt, "F2", "~", "V5")] == -.2);
  CHECK(pt.ustart[row(pt, "F2", "~~", "F2")] == .7);
  CHECK(pt.ustart[row(pt, "V5", "~~", "V5")] == 2.0);
}

TEST_CASE("EQS: ranges, overrides, scientific numbers and fixed error coefficients") {
  auto flat = EqsParser::parse(
      "/EQU V1=.7*F1+1E1; V2=.8*F1+1.0E2; /VAR F1=1; E1 TO E2=1e-1*; E2=-.2;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  CHECK(pt.ustart[row(pt, "V1", "~~", "V1")] == .1);
  CHECK(pt.free[row(pt, "V2", "~~", "V2")] == 0);
  CHECK(pt.ustart[row(pt, "V2", "~~", "V2")] == -.2);
}

TEST_CASE("EQS: explicit single-indicator residuals are preserved") {
  auto flat = EqsParser::parse("/EQU V1=F1+E1; /VAR F1=1; E1=.4*;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  CHECK(pt.free[row(pt, "V1", "~~", "V1")] > 0);
}

TEST_CASE("EQS: observed structural paths preserve the equation covariance") {
  auto model = magmaan::api::model_from_eqs(
      "/EQU V1=.2V2+E1; /VAR V2=1; E1=.4;");
  REQUIRE_MESSAGE(model.has_value(), (model ? "" : model.error().detail));
  if (!model) return;
  auto evaluator = magmaan::model::ModelEvaluator::build(model->structure(), model->matrix_rep());
  REQUIRE(evaluator.has_value());
  auto implied = evaluator->sigma(Eigen::VectorXd{});
  REQUIRE(implied.has_value());
  // V1 = .2 V2 + E1, with independent primitives.
  CHECK(implied->sigma[0](0,0) == doctest::Approx(.2*.2 + .4));
  CHECK(implied->sigma[0](0,1) == doctest::Approx(.2));
}

TEST_CASE("EQS: covariance lists expand independently and later specifications override") {
  auto flat = EqsParser::parse("/COV V1-V3=.2*; V2,V1=.3;");
  REQUIRE(flat.has_value());
  auto pt = partable(*flat);
  CHECK(pt.size() == 6);
  CHECK(pt.free[row(pt, "V1", "~~", "V2")] == 0);
  CHECK(pt.ustart[row(pt, "V1", "~~", "V2")] == .3);
  CHECK(pt.free[row(pt, "V1", "~~", "V3")] > 0);
  CHECK(pt.free[row(pt, "V2", "~~", "V3")] > 0);
}

TEST_CASE("EQS: lowered textual projection preserves fixed values and starts") {
  auto flat = EqsParser::parse("/EQU V1=.7*F1+E1; V2=F1+E2; /VAR F1=1;");
  REQUIRE(flat.has_value());
  auto reparsed = magmaan::parse::Parser::parse(magmaan::compat::eqs::to_lavaan_syntax(*flat));
  REQUIRE(reparsed.has_value());
  auto a = partable(*flat), b = partable(*reparsed);
  CHECK(a.lhs == b.lhs);
  CHECK(a.rhs == b.rhs);
  CHECK(a.op == b.op);
  CHECK(a.free == b.free);
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::isnan(a.ustart[i])) CHECK(std::isnan(b.ustart[i]));
    else CHECK(a.ustart[i] == b.ustart[i]);
}

TEST_CASE("EQS: unsupported and malformed inputs fail with source locations") {
  for (const auto* source : {
      "", "/END", "/EQU", "/MODEL (V1,V2) ON F1;", "/SPEC VAR=3;",
      "/EQU V1=F1+E1", "/EQU V1=F1+E1; V1=F1+E2;",
      "/EQU E1=*F1;", "/EQU V1=*F1;", "/EQU V1=F1+.5E1;",
      "/EQU V1=F1+*E1;", "/EQU V1=F1+D1;",
      "/EQU V1=F1+E1; V2=F1+E1;", "/EQU V1=F1+E1; /VAR V1=1;",
      "/EQU V1=F1+E1; /COV F1,E1=*;", "/EQU V1=F1+E1; /VAR E2=1;",
      "/EQU V1=V999+E1;", "/EQU V0=F1+E1;", "/EQU V1=.F1+E1;",
      "/EQU V1=1e+999*F1+E1;", "/EQU V1=F1+E1; /VAR E2-E1=1;",
      "/EQU V1=F1+E1; /COV E1,E1=*;", "/EQU V1=V1+E1;",
      "/EQU V1=.7F1+.2V2+E1; V2=.8F1+E2;",
      "/EQU V1=F1+E1; F1=.2V1+D1;",
      "/VAR F1=1;", "/VAR V1=1; /END /VAR V2=1;"}) {
    CAPTURE(source);
    auto parsed = EqsParser::parse(source);
    REQUIRE_FALSE(parsed.has_value());
    CHECK(parsed.error().span.line >= 1);
    CHECK(parsed.error().span.end >= parsed.error().span.begin);
    CHECK_FALSE(parsed.error().detail.empty());
  }
  CHECK_FALSE(EqsParser::parse("/VAR V2=1;", {"only_one"}).has_value());
  CHECK_FALSE(EqsParser::parse("/VAR V1,V2=1;", {"same", "same"}).has_value());
  CHECK_FALSE(EqsParser::parse("/EQU V1=F1+E1;", {"F1"}).has_value());
  CHECK_FALSE(EqsParser::parse("/VAR V1=1;", {"not a name"}).has_value());
  CHECK_FALSE(EqsParser::parse("/VAR V1=1;", {"."}).has_value());
}
