#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "../test_fit.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/spec/build.hpp"

namespace {
nlohmann::json reporting_fixture() {
  auto raw = magmaan::test::read_fixture(
      magmaan::test::fixtures_dir() + "/parameter_reporting.json");
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  return j;
}
const nlohmann::json& oracle_row(const nlohmann::json& rows,
                               const magmaan::spec::LatentNames& names,
                               const magmaan::spec::LatentStructure& pt,
                               std::size_t i) {
  for (const auto& r : rows) {
    if (r["lhs"] == names.row_lhs[i] && r["rhs"] == names.row_rhs[i] &&
        r["op"] == magmaan::parse::to_string(pt.op[i]) && r["group"] == pt.group[i])
      return r;
  }
  FAIL("missing oracle row " << names.row_lhs[i]);
  return rows[0];
}
}

TEST_CASE("parameter reporting: effect coding matches lavaan means and constraints") {
  using namespace magmaan;
  const auto fixture = reporting_fixture();
  for (const auto& item : fixture["effect_coding"].items()) {
    CAPTURE(item.key());
    const auto& c = item.value();
    spec::BuildOptions opts;
    opts.effect_coding = true;
    opts.meanstructure = true;
    opts.n_groups = c["n_groups"].get<int>();
    if (item.key() == "scalar")
      opts.group_equal = {spec::GroupEqual::Loadings, spec::GroupEqual::Intercepts};
    auto flat = parse::Parser::parse(c["input"].get<std::string>());
    REQUIRE(flat.has_value());
    spec::LatentNames names;
    auto pt = spec::build(*flat, opts, nullptr, &names);
    REQUIRE(pt.has_value());
    auto rep = model::build_matrix_rep(*pt, &names);
    REQUIRE(rep.has_value());
    data::SampleStats sample;
    const auto ns = c["nobs"];
    for (std::size_t b = 0; b < c["sample"].size(); ++b) {
      sample.S.push_back(test::matrix_from_json(c["sample"][b]["cov"]));
      sample.mean.push_back(test::vector_from_json(c["sample"][b]["mean"]));
      sample.n_obs.push_back(ns.is_array() ? ns[b].get<int>() : ns.get<int>());
    }
    auto est = test::fit(*pt, *rep, sample, {}, estimate::Backend::NloptSlsqp,
                        {.max_iter=4000, .ftol=1e-13, .gtol=1e-8});
    REQUIRE_MESSAGE(est.has_value(), (est ? "" : est.error().detail));
    for (std::size_t i = 0; i < pt->size(); ++i) {
      if (pt->group[i] == 0) continue;
      const auto& r = oracle_row(c["rows"], names, *pt, i);
      CHECK((pt->free[i] > 0) == (r["free"].get<int>() > 0));
      double value = pt->free[i] > 0 ? est->theta(pt->free[i]-1) : pt->fixed_value[i];
      CHECK(value == doctest::Approx(r["est"].get<double>()).epsilon(2e-5));
    }
    auto evaluator = model::ModelEvaluator::build(*pt, *rep);
    REQUIRE(evaluator.has_value());
    auto implied = evaluator->sigma(est->theta);
    REQUIRE(implied.has_value());
    for (std::size_t b = 0; b < implied->sigma.size(); ++b) {
      CHECK((implied->sigma[b] - test::matrix_from_json(c["implied"][b]["cov"])).cwiseAbs().maxCoeff() < 2e-5);
      CHECK((implied->mu[b] - test::vector_from_json(c["implied"][b]["mean"])).cwiseAbs().maxCoeff() < 2e-5);
      double loading_sum = 0.0, intercept_sum = 0.0;
      for (std::size_t i = 0; i < pt->size(); ++i) {
        if (pt->group[i] != static_cast<int>(b + 1)) continue;
        const double value = pt->free[i] > 0 ? est->theta(pt->free[i]-1) : pt->fixed_value[i];
        if (pt->op[i] == parse::Op::Measurement) loading_sum += value;
        if (pt->op[i] == parse::Op::Intercept && names.row_lhs[i] != "f") intercept_sum += value;
      }
      if (item.key() != "fixed_loading") CHECK(loading_sum == doctest::Approx(4.0));
      if (item.key() != "fixed_intercept") CHECK(std::abs(intercept_sum) < 1e-8);
    }
    if (item.key() == "single" || item.key() == "groups" || item.key() == "scalar") {
      opts.effect_coding = false;
      auto marker_pt = spec::build(*flat, opts);
      REQUIRE(marker_pt.has_value());
      auto marker_rep = model::build_matrix_rep(*marker_pt);
      REQUIRE(marker_rep.has_value());
      auto marker_est = test::fit(*marker_pt, *marker_rep, sample, {},
          estimate::Backend::NloptSlsqp, {.max_iter=4000, .ftol=1e-13, .gtol=1e-8});
      REQUIRE(marker_est.has_value());
      auto marker_ev = model::ModelEvaluator::build(*marker_pt, *marker_rep);
      REQUIRE(marker_ev.has_value());
      auto marker = marker_ev->sigma(marker_est->theta);
      REQUIRE(marker.has_value());
      for (std::size_t b = 0; b < implied->sigma.size(); ++b) {
        CHECK((implied->sigma[b] - marker->sigma[b]).cwiseAbs().maxCoeff() < 2e-5);
        CHECK((implied->mu[b] - marker->mu[b]).cwiseAbs().maxCoeff() < 2e-5);
      }
    }
    auto con = estimate::build_eq_constraints(*pt);
    REQUIRE(con.has_value());
    CHECK(14 * opts.n_groups - con->n_alpha == c["df"].get<int>());
  }
}

TEST_CASE("parameter reporting: ordinal residuals and response scales match lavaan") {
  using namespace magmaan;
  const auto fixture = reporting_fixture();
  for (const auto& item : fixture["ordinal"].items()) {
    CAPTURE(item.key());
    const auto& c = item.value();
    spec::BuildOptions opts;
    opts.meanstructure = true;
    opts.n_groups = c["n_groups"].get<int>();
    auto flat = parse::Parser::parse(c["input"].get<std::string>());
    REQUIRE(flat.has_value());
    spec::LatentNames names;
    auto pt = spec::build(*flat, opts, nullptr, &names);
    REQUIRE(pt.has_value());
    // Oracle free coordinates with the same fixed-unit setup used by ordinal
    // preparation. This gates reconstruction independently of optimizer error.
    for (std::size_t i = 0; i < pt->size(); ++i) {
      const auto& r = oracle_row(c["rows"], names, *pt, i);
      pt->free[i] = r["free"].get<int>();
      if (pt->free[i] == 0) pt->fixed_value[i] = r["ustart"].is_null() ? 0.0 : r["ustart"].get<double>();
      if (pt->op[i] == parse::Op::Covariance && pt->lhs_var[i] == pt->rhs_var[i] && pt->free[i] == 0)
        pt->fixed_value[i] = 1.0;
    }
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(pt->n_free());
    for (std::size_t i = 0; i < pt->size(); ++i)
      if (pt->free[i] > 0) theta(pt->free[i]-1) = oracle_row(c["rows"], names, *pt, i)["est"].get<double>();
    auto rep = model::build_matrix_rep(*pt, &names);
    REQUIRE(rep.has_value());
    const auto param = c["parameterization"] == "delta" ? estimate::OrdinalParameterization::Delta : estimate::OrdinalParameterization::Theta;
    const auto fixed_before = pt->fixed_value;
    const auto free_before = pt->free;
    const auto theta_before = theta;
    auto values = estimate::ordinal_parameter_values(*pt, *rep, theta, param);
    REQUIRE(values.has_value());
    CHECK(pt->free == free_before);
    CHECK(theta == theta_before);
    auto ev = model::ModelEvaluator::build(*pt, *rep);
    REQUIRE(ev.has_value());
    auto matrices = ev->assembled(theta);
    REQUIRE(matrices.has_value());
    auto moments = ev->sigma(theta);
    REQUIRE(moments.has_value());
    for (std::size_t i = 0; i < pt->size(); ++i) {
      CHECK((*values)(static_cast<Eigen::Index>(i)) == doctest::Approx(oracle_row(c["rows"], names, *pt, i)["est"].get<double>()).epsilon(1e-9));
      if (pt->free[i] == 0) CHECK(pt->fixed_value[i] == fixed_before[i]);
      if (param == estimate::OrdinalParameterization::Theta &&
          pt->op[i] == parse::Op::ResponseScale && pt->free[i] == 0) {
        const auto block = static_cast<std::size_t>(pt->block_of(i) - 1);
        const auto ov = pt->ov_pos[static_cast<std::size_t>(pt->lhs_var[i])];
        const double scale = (*values)(static_cast<Eigen::Index>(i));
        CHECK(scale * scale * moments->sigma[block](ov, ov) == doctest::Approx(1.0));
      }
      const auto& cell = rep->cell_for_row[i];
      if (param == estimate::OrdinalParameterization::Delta && pt->free[i] == 0 &&
          cell.used && cell.mat == model::MatId::Theta && cell.row == cell.col) {
        const auto& m = matrices->blocks[static_cast<std::size_t>(cell.block)];
        const double explained = (m.Lambda.row(cell.row) * m.Mid * m.Lambda.row(cell.row).transpose())(0,0);
        CHECK((*values)(static_cast<Eigen::Index>(i)) + explained == doctest::Approx(1.0));
      }
    }
  }
}

TEST_CASE("parameter reporting: theta scales use response variance and retain free coordinates") {
  using namespace magmaan;
  auto flat = parse::Parser::parse("y ~~ v*y\ny | t1\ny ~*~ 1*y");
  REQUIRE(flat.has_value());
  auto pt = spec::build(*flat);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(pt->n_free());
  std::size_t scale_row = pt->size();
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] == parse::Op::Covariance) theta(pt->free[i] - 1) = 4.0;
    if (pt->op[i] == parse::Op::ResponseScale) scale_row = i;
  }
  REQUIRE(scale_row < pt->size());
  auto values = estimate::ordinal_parameter_values(
      *pt, *rep, theta, estimate::OrdinalParameterization::Theta);
  REQUIRE(values.has_value());
  CHECK((*values)(static_cast<Eigen::Index>(scale_row)) == doctest::Approx(0.5));
  for (double variance : {0.0, -1.0}) {
    for (std::size_t i = 0; i < pt->size(); ++i)
      if (pt->op[i] == parse::Op::Covariance) theta(pt->free[i] - 1) = variance;
    CHECK_FALSE(estimate::ordinal_parameter_values(
        *pt, *rep, theta, estimate::OrdinalParameterization::Theta).has_value());
  }
  for (std::size_t i = 0; i < pt->size(); ++i)
    if (pt->op[i] == parse::Op::Covariance) theta(pt->free[i] - 1) = 4.0;
  pt->free[scale_row] = pt->n_free() + 1;
  theta.conservativeResize(pt->n_free());
  theta(theta.size() - 1) = 0.7;
  rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  values = estimate::ordinal_parameter_values(
      *pt, *rep, theta, estimate::OrdinalParameterization::Theta);
  REQUIRE(values.has_value());
  CHECK((*values)(static_cast<Eigen::Index>(scale_row)) == doctest::Approx(0.7));
}

TEST_CASE("parameter reporting: effect coding retains lavaan group mean identification") {
  using namespace magmaan;
  const auto fixture = reporting_fixture();
  for (const auto& item : fixture["effect_identification"].items()) {
    CAPTURE(item.key());
    const auto& c = item.value();
    spec::BuildOptions opts;
    opts.meanstructure = true;
    opts.effect_coding = true;
    opts.n_groups = 2;
    opts.group_equal = {spec::GroupEqual::Means};
    if (item.key() == "scalar_means")
      opts.group_equal = {spec::GroupEqual::Loadings, spec::GroupEqual::Intercepts,
                          spec::GroupEqual::Means};
    auto flat = parse::Parser::parse(c["input"].get<std::string>());
    REQUIRE(flat.has_value());
    spec::LatentNames names;
    spec::Starts starts;
    auto pt = spec::build(*flat, opts, &starts, &names);
    REQUIRE(pt.has_value());
    for (std::size_t i = 0; i < pt->size(); ++i) {
      if (pt->group[i] == 0) continue;
      const auto& r = oracle_row(c["rows"], names, *pt, i);
      CHECK((pt->free[i] > 0) == (r["free"].get<int>() > 0));
      if (pt->op[i] == parse::Op::Intercept && names.row_lhs[i] == "f") {
        const double start = pt->free[i] > 0 ? starts.hint[static_cast<std::size_t>(pt->free[i]-1)] : pt->fixed_value[i];
        CHECK(start == doctest::Approx(r["ustart"].get<double>()));
      }
    }
  }
}
