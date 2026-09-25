// Textbook categorical (WLSMV) models vs lavaan.
//
// Fixtures: cpp/tests/fixtures/textbook_ordinal/, written by
// cpp/tests/tools/regen_textbook_ordinal_fixtures.R from the textbook corpus.
// Each case is an all-ordinal, covariate-free model that the corpus verified
// against its book's own output (Mplus .out or the author's lavaan call). The
// fixtures carry derived statistics only (thresholds, polychorics, NACOV, the
// DWLS weight); the model enters through lavaan's parameter table.
//
// Checked per case: DWLS point estimates from lavaan's starting values, the
// fit-function value and df, and the implied latent-response correlations.
// WLSMV differs from DWLS only in its test statistic, which is not checked
// here. A solution that differs from lavaan's only by reflected loadings
// (same objective, same implied moments, same absolute estimates) counts as
// a match.
//
// kKnownGaps lists cases magmaan's ordinal estimators do not yet reproduce
// (project/backlog/todo.md, "ordinal partable semantics"); they are run and
// reported but do not fail the test. A gap case that starts passing is
// reported so it can move to kCases proper.

#include <doctest/doctest.h>
#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../oracle.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace {

using namespace magmaan;
using test::matrix_from_json;
using test::vector_from_json;

const std::vector<std::string> kCases = {
    "mplus_users_guide_v8_ch5_ex5_2",  "mplus_users_guide_v8_ch5_ex5_10",
    "mplus_users_guide_v8_ch5_ex5_19", "mplus_users_guide_v8_ch5_ex5_22",
    "mplus_users_guide_v8_ch6_ex6_4",  "mplus_users_guide_v8_ch6_ex6_5",
    "mplus_users_guide_v8_ch6_ex6_15", "newsom_2015_ex2_4a",
    "newsom_2015_ex2_8a",              "newsom_2015_ex3_3a",
    "newsom_2015_ex3_3b",              "newsom_2015_ex3_3c",
    "newsom_2015_ex7_2b",              "newsom_2015_ex9_2",
    "newsom_2024_ex1_3c",              "newsom_2024_ex6_4a",
    "newsom_2024_ex6_4b",              "newsom_2024_ex6_4c",
    "newsom_2024_ex6_4d",              "newsom_2024_ex7_2a",
    "newsom_2024_ex9_2"};

struct KnownGap {
  std::string id;
  std::string reason;
};
const std::vector<KnownGap> kKnownGaps = {
    {"mplus_users_guide_v8_ch6_ex6_4",
     "delta scale factors forced to 1 and latent means not in the threshold "
     "structure (categorical growth)"},
    {"mplus_users_guide_v8_ch6_ex6_5",
     "theta residual variances forced to 1 (free at later occasions)"},
    {"mplus_users_guide_v8_ch6_ex6_15",
     "delta scale factors forced to 1 and latent means not in the threshold "
     "structure (multiple-indicator categorical growth)"},
    {"newsom_2015_ex3_3a",
     "theta residual variances forced to 1 where the model fixes them at 0 "
     "(an equivalent fit in a rescaled parameterization)"},
    {"newsom_2015_ex9_2",
     "delta scale factors forced to 1 and latent means not in the threshold "
     "structure (latent change model)"},
    {"newsom_2024_ex1_3c",
     "optimizer: from lavaan's starts L-BFGS stops on a flat ridge of this "
     "saturated theta model (fmin 5.8e-9); from lavaan's estimates magmaan "
     "stays at lavaan's solution"},
    {"newsom_2024_ex7_2a",
     "latent means not in the threshold structure (categorical growth)"},
    {"newsom_2024_ex9_2",
     "delta scale factors forced to 1 and latent means not in the threshold "
     "structure (latent change model)"}};

const KnownGap* known_gap(const std::string& id) {
  for (const auto& g : kKnownGaps)
    if (g.id == id) return &g;
  return nullptr;
}

parse::Op op_from_string(const std::string& value) {
  for (unsigned i = 0; i <= static_cast<unsigned>(parse::Op::Composite); ++i) {
    auto op = static_cast<parse::Op>(i);
    if (parse::to_string(op) == value) return op;
  }
  FAIL("Unknown partable operator: " << value);
  return parse::Op::Covariance;
}

compat::lavaan::ParsedLavaanParTable model_from_json(const nlohmann::json& j) {
  compat::lavaan::LavaanParTable pt;
  for (const auto& r : j["partable"]) {
    pt.id.push_back(r["id"]);
    pt.user.push_back(r["user"]);
    pt.lhs.push_back(r["lhs"]);
    pt.rhs.push_back(r["rhs"]);
    pt.op.push_back(op_from_string(r["op"]));
    pt.block.push_back(r["block"]);
    pt.group.push_back(r["group"]);
    pt.free.push_back(r["free"]);
    pt.exo.push_back(r["exo"]);
    pt.ustart.push_back(r["ustart"].is_null()
                            ? std::numeric_limits<double>::quiet_NaN()
                            : r["ustart"].get<double>());
    pt.label.push_back(r["label"]);
    pt.plabel.push_back(r["plabel"]);
  }
  return compat::lavaan::from_lavaan_partable(pt);
}

data::OrdinalStats ordinal_from_json(const nlohmann::json& j) {
  data::OrdinalStats s;
  for (const auto& b : j["blocks"]) {
    s.R.push_back(matrix_from_json(b["R"]));
    s.thresholds.push_back(vector_from_json(b["thresholds"]));
    s.threshold_ov.push_back(b["threshold_ov"].get<std::vector<std::int32_t>>());
    s.threshold_level.push_back(
        b["threshold_level"].get<std::vector<std::int32_t>>());
    s.n_levels.push_back(b["n_levels"].get<std::vector<std::int32_t>>());
    s.NACOV.push_back(matrix_from_json(b["NACOV"]));
    s.W_dwls.push_back(matrix_from_json(b["W"]));
    s.n_obs.push_back(b["n"]);
    s.ov_names.push_back(j["observed_variables"].get<std::vector<std::string>>());
  }
  return s;
}

}  // namespace

TEST_CASE("Textbook categorical (WLSMV) models match lavaan's DWLS estimates") {
  int passed = 0;
  std::vector<std::string> failures;
  for (const auto& id : kCases) {
    CAPTURE(id);
    auto raw = test::read_fixture(test::fixtures_dir() + "/textbook_ordinal/" +
                                  id + ".json");
    REQUIRE_MESSAGE(raw.has_value(), "missing fixture " << id);
    auto j = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(j.is_discarded());
    REQUIRE(j["_meta"]["aggregate_only"].get<bool>());

    auto fail = [&](const std::string& why) { failures.push_back(id + ": " + why); };

    auto h = model_from_json(j);
    auto& pt = h.structure;
    auto stats = ordinal_from_json(j);
    const auto param = j["parameterization"].get<std::string>() == "theta"
                           ? estimate::OrdinalParameterization::Theta
                           : estimate::OrdinalParameterization::Delta;

    Eigen::VectorXd theta = vector_from_json(j["theta"]);
    Eigen::VectorXd start = vector_from_json(j["start"]);
    if (theta.size() != pt.n_free()) {
      fail("free-parameter count " + std::to_string(pt.n_free()) + " vs lavaan " +
           std::to_string(theta.size()));
      continue;
    }
    // Ordinal preparation reorders free parameters; carry lavaan's vectors
    // across by partable row.
    const auto old_free = pt.free;
    auto prep = estimate::prepare_ordinal_partable(pt, stats, param);
    if (!prep.has_value()) {
      fail("prepare_ordinal_partable: " + prep.error().detail);
      continue;
    }
    if (theta.size() != pt.n_free()) {
      if (known_gap(id))
        MESSAGE(id << ": known gap (" << known_gap(id)->reason
                   << "): preparation changed the free-parameter count");
      else
        fail("preparation changed the free-parameter count");
      continue;
    }
    Eigen::VectorXd th2(pt.n_free()), st2(pt.n_free());
    for (std::size_t r = 0; r < pt.size(); ++r) {
      if (pt.free[r] > 0) {
        th2(pt.free[r] - 1) = theta(old_free[r] - 1);
        st2(pt.free[r] - 1) = start(old_free[r] - 1);
      }
    }
    theta = std::move(th2);
    start = std::move(st2);

    auto rep = model::build_matrix_rep(pt);
    if (!rep.has_value()) {
      fail("build_matrix_rep: " + rep.error().detail);
      continue;
    }
    optim::OptimOptions opts{.max_iter = 7000, .ftol = 1e-13, .gtol = 1e-8};
    auto fit = estimate::fit_ordinal_bounded(
        pt, *rep, stats, {}, estimate::OrdinalWeightKind::DWLS, start,
        estimate::Backend::NloptLbfgs, opts, param);
    if (!fit.has_value()) {
      fail("fit_ordinal_bounded: " + fit.error().detail);
      continue;
    }

    std::ostringstream why;
    // lavaan's fmin is half the DWLS fit function with the N-1 moment
    // convention; magmaan's fmin is half the fit function with N.
    double n_total = 0.0;
    for (auto n : stats.n_obs) n_total += static_cast<double>(n);
    const double lav_fmin = j["fit"]["fmin"].get<double>();
    const double lav_chisq = j["fit"]["chisq"].get<double>();
    const double d_theta = (fit->theta - theta).cwiseAbs().maxCoeff();
    const double d_abs_theta =
        (fit->theta.cwiseAbs() - theta.cwiseAbs()).cwiseAbs().maxCoeff();
    const bool reflected = d_theta > 1e-3 && d_abs_theta <= 1e-3;
    if (d_theta > 1e-3 && !reflected) {
      why << " max|dtheta|=" << d_theta << " [";
      const auto& rows = j["partable"];
      for (std::size_t r = 0; r < pt.size() && r < rows.size(); ++r) {
        if (pt.free[r] <= 0) continue;
        const auto k = static_cast<Eigen::Index>(pt.free[r] - 1);
        if (std::abs(fit->theta(k) - theta(k)) > 1e-3)
          why << " " << rows[r]["lhs"].get<std::string>() << rows[r]["op"].get<std::string>()
              << rows[r]["rhs"].get<std::string>() << "@g" << rows[r]["group"].get<int>()
              << " " << fit->theta(k) << "/" << theta(k);
      }
      why << " ]";
    }

    auto constraints = estimate::build_eq_constraints(pt);
    REQUIRE(constraints.has_value());
    Eigen::Index moments = 0;
    const auto p = static_cast<Eigen::Index>(j["observed_variables"].size());
    for (const auto& b : j["blocks"])
      moments += static_cast<Eigen::Index>(b["thresholds"].size()) + p * (p - 1) / 2;
    const int df = j["fit"]["df"].get<int>();
    if (moments - constraints->n_alpha != df)
      why << " df " << (moments - constraints->n_alpha) << " vs " << df;

    auto evaluator = model::ModelEvaluator::build(pt, *rep);
    REQUIRE(evaluator.has_value());
    auto implied = evaluator->sigma(fit->theta);
    REQUIRE(implied.has_value());
    double d_cor = 0.0;
    for (std::size_t b = 0; b < j["implied"].size(); ++b) {
      Eigen::MatrixXd expected = matrix_from_json(j["implied"][b]["cov"]);
      Eigen::MatrixXd got = implied->sigma[b];
      // Compare latent-response correlations: delta fixes the diagonal at one,
      // theta implies it from the residual variances.
      Eigen::VectorXd sg = got.diagonal().cwiseSqrt().cwiseInverse();
      Eigen::VectorXd se = expected.diagonal().cwiseSqrt().cwiseInverse();
      if (param == estimate::OrdinalParameterization::Delta) got.diagonal().setOnes();
      else got = sg.asDiagonal() * got * sg.asDiagonal();
      expected = se.asDiagonal() * expected * se.asDiagonal();
      d_cor = std::max(d_cor, (got - expected).cwiseAbs().maxCoeff());
    }
    if (d_cor > 1e-3) why << " implied correlations max diff " << d_cor;

    // fmin conventions: magmaan divides by N, lavaan by N - 1 per block.
    double lav_fmin_n = 0.0;
    for (auto n : stats.n_obs)
      lav_fmin_n += static_cast<double>(n) / static_cast<double>(n - 1) *
                    static_cast<double>(n) / n_total;
    lav_fmin_n *= lav_fmin;
    if (std::abs(fit->fmin - lav_fmin_n) > 1e-6 + 1e-4 * lav_fmin_n)
      why << " fmin " << fit->fmin << " vs lavaan " << lav_fmin_n;

    MESSAGE(id << ": fmin " << fit->fmin << " (lavaan " << lav_fmin_n
               << " on magmaan's N convention, chisq " << lav_chisq
               << ") max|dtheta| " << d_theta
               << std::string(reflected ? " (reflected loadings)" : "")
               << " implied " << d_cor);
    // Separate optimizer trouble from model semantics: refit from lavaan's
    // estimates. If magmaan stays there at lavaan's objective, its model
    // agrees and only the path from lavaan's starts differs.
    if (!why.str().empty()) {
      auto refit = estimate::fit_ordinal_bounded(
          pt, *rep, stats, {}, estimate::OrdinalWeightKind::DWLS, theta,
          estimate::Backend::NloptLbfgs, opts, param);
      if (refit.has_value())
        why << " | from lavaan's theta: fmin " << refit->fmin << ", max|dtheta| "
            << (refit->theta - theta).cwiseAbs().maxCoeff();
    }
    const auto* gap = known_gap(id);
    if (gap) {
      if (why.str().empty())
        MESSAGE(id << ": known gap now passes; move it out of kKnownGaps");
      else
        MESSAGE(id << ": known gap (" << gap->reason << "):" << why.str());
      continue;
    }
    if (why.str().empty()) ++passed;
    else fail(why.str());
  }
  const int expected = static_cast<int>(kCases.size() - kKnownGaps.size());
  MESSAGE("textbook ordinal: " << passed << " / " << expected
          << " pass (" << kKnownGaps.size() << " known gaps reported above)");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == expected);
}
