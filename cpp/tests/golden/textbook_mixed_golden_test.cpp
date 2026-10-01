#include <doctest/doctest.h>
#include "../oracle.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#include <limits>

namespace {
using namespace magmaan;

compat::lavaan::ParsedLavaanParTable mixed_fixture_model(const nlohmann::json& j) {
  compat::lavaan::LavaanParTable table;
  for (const auto& r : j["partable"]) {
    parse::Op op = parse::Op::Covariance;
    bool found = false;
    for (unsigned k = 0; k <= static_cast<unsigned>(parse::Op::Composite); ++k) {
      if (parse::to_string(static_cast<parse::Op>(k)) == r["op"].get<std::string>()) {
        op = static_cast<parse::Op>(k);
        found = true;
        break;
      }
    }
    REQUIRE(found);
    table.id.push_back(r["id"]); table.user.push_back(r["user"]);
    table.lhs.push_back(r["lhs"]); table.rhs.push_back(r["rhs"]);
    table.op.push_back(op); table.block.push_back(r["block"]);
    table.group.push_back(r["group"]); table.free.push_back(r["free"]);
    table.exo.push_back(r["exo"]);
    table.ustart.push_back(r["ustart"].is_null() ?
        std::numeric_limits<double>::quiet_NaN() : r["ustart"].get<double>());
    table.label.push_back(r["label"]); table.plabel.push_back(r["plabel"]);
  }
  return compat::lavaan::from_lavaan_partable(table);
}
}

TEST_CASE("Textbook mixed DWLS: matched setup reproduces lavaan objectives and complete tables") {
  using namespace magmaan;
  for (const std::string id : {"newsom_2015_ex5_3a", "newsom_2015_ex5_3b",
                              "newsom_2015_ex5_7a", "newsom_2024_ex5_8b"}) {
    CAPTURE(id);
    auto raw = test::read_fixture(test::fixtures_dir() + "/textbook_mixed/" + id + ".json");
    REQUIRE(raw.has_value());
    auto j = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(j.is_discarded());
    auto model = mixed_fixture_model(j);
    auto& pt = model.structure;
    // sem() enables terminal-outcome covariances. The old raw-data audit
    // accidentally compared the narrower default builder setup with sem().
    auto flat = parse::Parser::parse(j["input"].get<std::string>());
    REQUIRE(flat.has_value());
    spec::BuildOptions opts;
    opts.auto_cov_y = j["options"]["auto_cov_y"];
    opts.fixed_x = j["options"]["fixed_x"];
    opts.meanstructure = j["options"]["meanstructure"];
    spec::LatentNames built_names;
    auto built = spec::build(*flat, opts, nullptr, &built_names);
    REQUIRE(built.has_value());
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] != parse::Op::Covariance) continue;
      bool matched = false;
      for (std::size_t k = 0; k < built->size(); ++k) {
        if (built->op[k] != pt.op[i]) continue;
        const auto& lhs = built_names.row_lhs[k];
        const auto& rhs = built_names.row_rhs[k];
        if ((lhs == model.names.row_lhs[i] && rhs == model.names.row_rhs[i]) ||
            (rhs == model.names.row_lhs[i] && lhs == model.names.row_rhs[i])) {
          matched = true;
          break;
        }
      }
      CHECK(matched);
    }
    data::MixedOrdinalStats stats;
    stats.R.push_back(test::matrix_from_json(j["R"]));
    stats.mean.push_back(test::vector_from_json(j["mean"]));
    stats.ordered.push_back(j["ordered_mask"].get<std::vector<std::int32_t>>());
    stats.thresholds.push_back(test::vector_from_json(j["thresholds"]));
    stats.threshold_ov.push_back(j["threshold_ov"].get<std::vector<std::int32_t>>());
    stats.threshold_level.push_back(j["threshold_level"].get<std::vector<std::int32_t>>());
    stats.n_levels.emplace_back(stats.ordered[0].size(), 0);
    for (std::size_t h = 0; h < stats.threshold_ov[0].size(); ++h) {
      const auto ov = static_cast<std::size_t>(stats.threshold_ov[0][h]);
      stats.n_levels[0][ov] = stats.threshold_level[0][h] + 1;
    }
    stats.moments.push_back(test::vector_from_json(j["moments"]));
    // These fixtures gate DWLS only: its objective consumes the NACOV diagonal.
    stats.NACOV.emplace_back(test::vector_from_json(j["NACOV_diag"]).asDiagonal());
    stats.W_dwls.emplace_back(test::vector_from_json(j["W_diag"]).asDiagonal());
    stats.n_obs.push_back(j["n"]);
    stats.ov_names.push_back(j["ov_names"].get<std::vector<std::string>>());
    const auto param = j["options"]["parameterization"] == "theta" ?
        estimate::OrdinalParameterization::Theta : estimate::OrdinalParameterization::Delta;
    auto prep = estimate::prepare_mixed_ordinal_partable(
        pt, stats, param, &model.starts, &model.names.row_user);
    REQUIRE_MESSAGE(prep.has_value(), (prep ? "" : prep.error().detail));
    auto rep = model::build_matrix_rep(pt, &model.names);
    REQUIRE(rep.has_value());
    CHECK(rep->form == model::RepForm::Reduced);
    REQUIRE(rep->ov_names == stats.ov_names);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(pt.n_free());
    for (std::size_t i = 0; i < pt.size(); ++i)
      if (pt.free[i] > 0) theta(pt.free[i] - 1) = j["partable"][i]["est"].get<double>();
    estimate::Estimates at;
    at.theta = theta;
    auto objective = estimate::frontier::mixed_ordinal_ls_objective(
        pt, *rep, stats, at, estimate::OrdinalWeightKind::DWLS, param);
    REQUIRE_MESSAGE(objective.has_value(), (objective ? "" : objective.error().detail));
    auto residual = objective->problem.r(theta);
    REQUIRE(residual.has_value());
    const double criterion = residual->squaredNorm();
    const double n = static_cast<double>(stats.n_obs[0]);
    const double fmin = 0.5 * criterion * (n - 1.0) / n;
    CHECK(fmin == doctest::Approx(j["fit"]["fmin"].get<double>()).epsilon(1e-8));
    const Eigen::VectorXd fitted_moments = stats.moments[0].array() +
        residual->array() / stats.W_dwls[0].diagonal().array().sqrt();
    CHECK((fitted_moments - test::vector_from_json(j["model_moments"]))
              .cwiseAbs().maxCoeff() < 1e-8);
    auto values = estimate::ordinal_parameter_values(pt, *rep, theta, param);
    REQUIRE(values.has_value());
    for (std::size_t i = 0; i < pt.size(); ++i) {
      CAPTURE(i);
      CHECK((*values)(static_cast<Eigen::Index>(i)) ==
            doctest::Approx(j["partable"][i]["est"].get<double>()).epsilon(1e-8));
    }
    // Refit away from the oracle endpoint to gate the optimizer and constraints.
    auto start = theta * 0.95;
    auto fit = estimate::fit_mixed_ordinal_bounded(pt, *rep, stats, {},
        estimate::OrdinalWeightKind::DWLS, start, estimate::Backend::NloptLbfgs,
        {.max_iter = 3000, .ftol = 1e-12, .gtol = 1e-7}, param,
        &model.names.row_user);
    REQUIRE_MESSAGE(fit.has_value(), (fit ? "" : fit.error().detail));
    CHECK(estimate::fit_verdict(*fit).status == estimate::FitCheck::Passed);
    CHECK(fit->fmin * (n - 1.0) / n ==
          doctest::Approx(j["fit"]["fmin"].get<double>()).epsilon(1e-7));
    auto fitted_values = estimate::ordinal_parameter_values(pt, *rep, fit->theta, param);
    REQUIRE(fitted_values.has_value());
    for (std::size_t i = 0; i < pt.size(); ++i) {
      CAPTURE(i);
      CHECK((*fitted_values)(static_cast<Eigen::Index>(i)) ==
            doctest::Approx(j["partable"][i]["est"].get<double>()).epsilon(1e-4));
    }
  }
}
