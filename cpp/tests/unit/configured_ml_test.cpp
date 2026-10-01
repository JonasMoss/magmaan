#include <doctest/doctest.h>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>
#include <nlohmann/json.hpp>

#include "magmaan/api/policy.hpp"
#include "magmaan/estimate/configured_ml.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {
struct Fixture {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::data::SampleStats sample;
};
Fixture fixture(std::string_view syntax, const Eigen::MatrixXd& covariance) {
  auto parsed = magmaan::parse::Parser::parse(syntax);
  REQUIRE(parsed);
  auto pt = magmaan::spec::build(*parsed);
  REQUIRE(pt);
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep);
  magmaan::data::SampleStats sample;
  sample.S = {covariance}; sample.n_obs = {200};
  return {std::move(*pt), std::move(*rep), std::move(sample)};
}
}

TEST_CASE("configured ML resolves independent components and pins versions") {
  using namespace magmaan::estimate;
  FittingOptions options;
  options.preset = "lavaan-0.7.2";
  auto full = resolve_fitting_options(options);
  REQUIRE(full);
  CHECK(full->optimizer == "lavaan-0.7.2");
  CHECK_FALSE(full->modified_preset);
  options.convergence = "newton";
  auto hybrid = resolve_fitting_options(options);
  REQUIRE(hybrid);
  CHECK(hybrid->modified_preset);
  CHECK(hybrid->starts == "lavaan-0.7.2");
  CHECK(hybrid->convergence == "newton");
  options.preset = "lavaan";
  CHECK_FALSE(resolve_fitting_options(options));
  options.preset = "lavaan-0.7.3";
  CHECK_FALSE(resolve_fitting_options(options));
  options = {};
  options.convergence = "lavaan-0.7.2";
  CHECK_FALSE(resolve_fitting_options(options));
  options.optimizer = "port";
  CHECK(resolve_fitting_options(options));
  for (const char* alias : {"default", "magmaan"}) {
    FittingOptions aliased;
    aliased.convergence = alias;
    auto resolved = resolve_fitting_options(aliased);
    REQUIRE(resolved);
    CHECK(resolved->convergence == "newton");
  }
}

TEST_CASE("lavaan ML starts reject constrained models whichever component asks") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d s = lambda * lambda.transpose();
  s.diagonal().array() += .7;
  auto f = fixture("f =~ x1 + a*x2 + a*x3 + x4", s);
  CHECK_FALSE(lavaan_ml_start_values(f.pt, f.rep, f.sample));
  FittingOptions starts_only;
  starts_only.starts = "lavaan-0.7.2";
  CHECK_FALSE(fit_ml_configured(f.pt, f.rep, f.sample, starts_only));
  FittingOptions native;
  native.starts = "fabin3";
  CHECK(fit_ml_configured(f.pt, f.rep, f.sample, native));
}

TEST_CASE("lavaan single-indicator starts follow the oracle's exogenous latents") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d s = lambda * lambda.transpose();
  s.diagonal().array() += .7;
  using magmaan::parse::Op;
  // The latent with exactly one measurement row, and that indicator.
  auto single_indicator = [](const Fixture& f) {
    for (std::size_t i = 0; i < f.pt.size(); ++i) {
      if (f.pt.op[i] != Op::Measurement) continue;
      int count = 0;
      for (std::size_t j = 0; j < f.pt.size(); ++j)
        count += f.pt.op[j] == Op::Measurement && f.pt.lhs_var[j] == f.pt.lhs_var[i];
      if (count == 1) return std::pair{f.pt.lhs_var[i], f.pt.rhs_var[i]};
    }
    return std::pair{-1, -1};
  };
  auto variance_slot = [](const Fixture& f, int v) {
    for (std::size_t i = 0; i < f.pt.size(); ++i)
      if (f.pt.free[i] > 0 && f.pt.op[i] == Op::Covariance && f.pt.lhs_var[i] == v && f.pt.rhs_var[i] == v)
        return static_cast<Eigen::Index>(f.pt.free[i] - 1);
    return Eigen::Index{-1};
  };
  // A first-order factor that indicates another factor is endogenous, so it
  // keeps the 0.05 latent default (lavaan 0.7.2: f1 ~~ f1 starts at 0.05).
  auto higher = fixture("f1 =~ x1\nx1 ~~ 0.3*x1\ng =~ f1 + x2 + x3 + x4", s);
  auto start = lavaan_ml_start_values(higher.pt, higher.rep, higher.sample);
  REQUIRE(start);
  const auto f1 = variance_slot(higher, single_indicator(higher).first);
  REQUIRE(f1 >= 0);
  CHECK((*start)(f1) == doctest::Approx(0.05));
  // A free indicator residual uses its start hint (lavaan: 1.7 - 0.4 = 1.3);
  // without a hint the oracle's missing user value counts as 1.
  auto single = fixture("f1 =~ x1\nx1 ~~ x1\nf2 =~ x2 + x3 + x4", s);
  const auto [latent, indicator] = single_indicator(single);
  const auto latent_slot = variance_slot(single, latent);
  const auto residual_slot = variance_slot(single, indicator);
  REQUIRE(latent_slot >= 0);
  REQUIRE(residual_slot >= 0);
  magmaan::spec::Starts hints;
  hints.hint.assign(static_cast<std::size_t>(single.pt.n_free()), std::numeric_limits<double>::quiet_NaN());
  hints.hint[static_cast<std::size_t>(residual_slot)] = 0.4;
  auto hinted = lavaan_ml_start_values(single.pt, single.rep, single.sample, hints);
  REQUIRE(hinted);
  CHECK((*hinted)(latent_slot) == doctest::Approx(1.3));
  auto unhinted = lavaan_ml_start_values(single.pt, single.rep, single.sample);
  REQUIRE(unhinted);
  CHECK((*unhinted)(latent_slot) == doctest::Approx(0.7));
}

TEST_CASE("lavaan ML starts use OLS, predictor moments and final user hints") {
  Eigen::Matrix3d s;
  s << 4, 1, 2, 1, 9, 3, 2, 3, 16;
  auto f = fixture("y ~ x1 + x2", s);
  // Canonical ordering is y,x1,x2, so build moments in that order.
  f.sample.mean = {Eigen::Vector3d(5, 2, 3)};
  const auto y = f.pt.ov_pos[static_cast<std::size_t>(f.pt.lhs_var[0])];
  CHECK(y == 0);
  auto start = magmaan::estimate::lavaan_ml_start_values(f.pt, f.rep, f.sample);
  REQUIRE(start);
  // OLS predictors have Sxx=[[9,3],[3,16]], Sxy=[1,2].
  Eigen::Vector2d beta(10.0 / 135.0, 15.0 / 135.0);
  for (std::size_t i = 0; i < f.pt.size(); ++i) {
    if (f.pt.free[i] <= 0) continue;
    if (f.pt.op[i] == magmaan::parse::Op::Regression)
      CHECK((*start)(f.pt.free[i] - 1) == doctest::Approx(beta(f.pt.ov_pos[static_cast<std::size_t>(f.pt.rhs_var[i])] - 1)));
  }
  magmaan::spec::Starts hints;
  hints.hint.resize(static_cast<std::size_t>(f.pt.n_free()), std::numeric_limits<double>::quiet_NaN());
  hints.hint[0] = 0.4321;
  auto hinted = magmaan::estimate::lavaan_ml_start_values(f.pt, f.rep, f.sample, hints);
  REQUIRE(hinted);
  CHECK((*hinted)(0) == 0.4321);
  auto simple = magmaan::estimate::lavaan_ml_start_values(f.pt, f.rep, f.sample, {}, true);
  REQUIRE(simple);
  for (std::size_t i = 0; i < f.pt.size(); ++i) if (f.pt.free[i] > 0) {
    const double expected = f.pt.op[i] == magmaan::parse::Op::Covariance && f.pt.lhs_var[i] == f.pt.rhs_var[i] ? 1.0 : 0.0;
    CHECK((*simple)(f.pt.free[i] - 1) == expected);
  }
}

#ifdef MAGMAAN_WITH_PORT
TEST_CASE("configured ML matches frozen lavaan 0.7.2 starts and fits") {
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) + "/fitting/lavaan_0_7_2.json");
  REQUIRE(in.good());
  auto root = nlohmann::json::parse(in, nullptr, false);
  REQUIRE_FALSE(root.is_discarded());
  for (const auto& c : root["cases"]) {
    auto parsed = magmaan::parse::Parser::parse(c["model"].get<std::string>());
    REQUIRE(parsed);
    magmaan::spec::BuildOptions opts;
    opts.std_lv = c["std_lv"].get<bool>(); opts.fixed_x = false;
    magmaan::spec::LatentNames names;
    auto pt = magmaan::spec::build(*parsed, opts, nullptr, &names);
    REQUIRE(pt);
    auto rep = magmaan::model::build_matrix_rep(*pt, &names);
    REQUIRE(rep);
    Eigen::MatrixXd s(4, 4);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
      s(i, j) = c["sample_cov"][static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get<double>();
    magmaan::data::SampleStats sample;
    sample.S = {s}; sample.n_obs = {c["n"].get<int>()};
    magmaan::estimate::FittingOptions fitting;
    fitting.preset = "lavaan-0.7.2";
    Eigen::VectorXd explicit_start;
    if (c.value("invalid_start", false)) explicit_start = Eigen::VectorXd::Zero(pt->n_free());
    auto est = magmaan::estimate::fit_ml_configured(*pt, *rep, sample, fitting, {}, explicit_start);
    REQUIRE(est);
    REQUIRE(est->fitting);
    // Ill-conditioned retries reach the oracle's starts, coordinates and
    // verdict, but their endpoints depend on floating-point paths.
    const bool endpoint = c.value("endpoint_parity", true);
    for (std::size_t i = 0; i < pt->size(); ++i) if (pt->free[i] > 0) {
      bool found = false;
      for (const auto& row : c["parameters"]) {
        if (row["lhs"] != names.row_lhs[i] || row["rhs"] != names.row_rhs[i] ||
            row["op"].get<std::string>() != magmaan::parse::to_string(pt->op[i])) continue;
        found = true;
        CHECK(est->fitting->attempts.front().start(pt->free[i] - 1) == doctest::Approx(row["start"].get<double>()).epsilon(1e-9));
        if (endpoint) CHECK(est->theta(pt->free[i] - 1) == doctest::Approx(row["est"].get<double>()).epsilon(1e-5));
      }
      CHECK(found);
    }
    CHECK((magmaan::estimate::fit_verdict(*est).status == magmaan::estimate::FitCheck::Passed) == c["converged"].get<bool>());
    if (c["fmin"].is_null()) {
      CHECK(std::isnan(est->fmin));
      REQUIRE(est->fitting->attempts.size() == 4);
      const auto& last = est->fitting->attempts.back();
      for (Eigen::Index i = 0; i < last.optimizer_start.size(); ++i)
        CHECK(last.optimizer_start(i) == doctest::Approx(c["optimizer_x"][static_cast<std::size_t>(i)].get<double>()).epsilon(1e-12));
    } else if (endpoint) CHECK(est->fmin == doctest::Approx(c["fmin"].get<double>()).epsilon(1e-9));
    // The selected attempt ran in lavaan's coordinates, and the shared
    // acceptance gradient reproduces the search's own measurement there.
    const auto& selected = est->fitting->attempts[est->fitting->selected_attempt];
    const auto& parscale = c["parscale"];
    if (!parscale.empty()) {
      REQUIRE(static_cast<Eigen::Index>(parscale.size()) == selected.parameter_scale.size());
      for (Eigen::Index i = 0; i < selected.parameter_scale.size(); ++i)
        CHECK(selected.parameter_scale(i) == doctest::Approx(parscale[static_cast<std::size_t>(i)].get<double>()).epsilon(1e-12));
    }
    if (selected.gradient_max >= 0) {
      auto g = magmaan::estimate::lavaan_acceptance_gradient(*pt, *rep, sample, est->theta, {}, selected.parameter_scale);
      REQUIRE(g);
      CHECK(*g == doctest::Approx(selected.gradient_max).epsilon(1e-10));
    }
  }
}

TEST_CASE("lavaan acceptance on the native PORT search is judged in lavaan's units") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d s = lambda * lambda.transpose();
  s.diagonal().array() += .7;
  // Rescaled variables separate magmaan's normalized search coordinates from
  // the parameter units lavaan's rule uses.
  const Eigen::Vector4d d(100, 1, 0.01, 1);
  s = d.asDiagonal() * s * d.asDiagonal();
  auto f = fixture("f =~ x1+x2+x3+x4", s);
  FittingOptions options;
  options.optimizer = "port";
  options.convergence = "lavaan-0.7.2";
  auto est = fit_ml_configured(f.pt, f.rep, f.sample, options);
  REQUIRE(est);
  REQUIRE(est->fitting);
  REQUIRE(est->fitting->attempts.size() == 1);
  const auto& attempt = est->fitting->attempts.front();
  auto at_estimate = lavaan_acceptance_gradient(f.pt, f.rep, f.sample, est->theta);
  REQUIRE(at_estimate);
  CHECK(attempt.gradient_max == *at_estimate);
  // Independently, the helper is the parameter-unit gradient of the reported
  // unnormalized objective: central differences of fmin, with steps relative
  // to each parameter, at a non-stationary point (the start is exact here).
  const Eigen::VectorXd theta = 1.1 * attempt.start;
  auto off = lavaan_acceptance_gradient(f.pt, f.rep, f.sample, theta);
  REQUIRE(off);
  double fd_max = 0.0;
  for (Eigen::Index j = 0; j < theta.size(); ++j) {
    const double h = 1e-6 * std::abs(theta(j));
    REQUIRE(h > 0);
    Eigen::VectorXd up = theta, down = theta;
    up(j) += h; down(j) -= h;
    auto fu = evaluate_at(f.pt, f.rep, f.sample, up, Estimator::ML);
    auto fd = evaluate_at(f.pt, f.rep, f.sample, down, Estimator::ML);
    REQUIRE(fu);
    REQUIRE(fd);
    fd_max = std::max(fd_max, std::abs((fu->fmin - fd->fmin) / (2 * h)));
  }
  CHECK(*off > 1e-3);
  CHECK(*off == doctest::Approx(fd_max).epsilon(1e-5));
}

TEST_CASE("policy state reports when lavaan's rule and magmaan's check disagree") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d s = lambda * lambda.transpose();
  s.diagonal().array() += .7;
  FittingOptions preset;
  preset.preset = "lavaan-0.7.2";
  auto plain = fixture("f =~ x1+x2+x3+x4", s);
  auto agree = fit_ml_configured(plain.pt, plain.rep, plain.sample, preset);
  REQUIRE(agree);
  auto state = magmaan::api::policy_fit_state(*agree);
  CHECK(state.converged);
  REQUIRE(state.native_converged);
  CHECK(*state.native_converged);
  CHECK_FALSE(magmaan::api::verdict_disagreement(state));
  // The oracle's moments with x1 scaled by 1000 and x3 by 1/1000: lavaan's
  // rule accepts a standardized-retry endpoint at fmin 0.276 on exact-fit
  // moments, and magmaan's check rejects it. The endpoint is path-sensitive,
  // so the test reads the oracle's own matrix.
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) + "/fitting/lavaan_0_7_2.json");
  REQUIRE(in.good());
  auto root = nlohmann::json::parse(in, nullptr, false);
  REQUIRE_FALSE(root.is_discarded());
  Eigen::Matrix4d scaled_s = Eigen::Matrix4d::Zero();
  bool found = false;
  for (const auto& c : root["cases"]) {
    if (c.value("rescale", 0.0) != 1000.0) continue;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
      scaled_s(i, j) = c["sample_cov"][static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get<double>();
    found = true;
  }
  REQUIRE(found);
  auto scaled = fixture("f =~ x1+x2+x3+x4", scaled_s);
  auto accepted = fit_ml_configured(scaled.pt, scaled.rep, scaled.sample, preset);
  REQUIRE(accepted);
  CHECK(accepted->fmin > 0.2);
  state = magmaan::api::policy_fit_state(*accepted);
  CHECK(state.converged);
  REQUIRE(state.native_converged);
  CHECK_FALSE(*state.native_converged);
  CHECK(magmaan::api::verdict_disagreement(state));
  // The reverse direction: the rule rejects a point magmaan's check accepts.
  agree->selected_verdict->status = FitCheck::Failed;
  state = magmaan::api::policy_fit_state(*agree);
  CHECK_FALSE(state.converged);
  CHECK(magmaan::api::verdict_disagreement(state));
  // Under magmaan's own rule there is no second verdict to disagree with.
  FittingOptions native;
  native.convergence = "newton";
  auto own = fit_ml_configured(plain.pt, plain.rep, plain.sample, native);
  REQUIRE(own);
  CHECK_FALSE(magmaan::api::policy_fit_state(*own).native_converged);
}

TEST_CASE("lavaan ML retains original-covariance rejection across retries") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d s = lambda * lambda.transpose();
  s.diagonal().array() += .7;
  auto f = fixture("f =~ x1+x2+x3+x4", s);
  FittingOptions options;
  options.preset = "lavaan-0.7.2";
  auto est = fit_ml_configured(f.pt, f.rep, f.sample, options, {}, Eigen::VectorXd::Zero(f.pt.n_free()));
  REQUIRE(est);
  REQUIRE(est->fitting);
  REQUIRE(est->fitting->attempts.size() == 4);
  CHECK(est->fitting->selected_attempt == 3);
  CHECK_FALSE(est->fitting->attempts[0].accepted);
  CHECK(est->fitting->attempts[2].simple_start);
  CHECK(fit_verdict(*est).status == FitCheck::Failed);
  CHECK(est->theta.isApprox(est->fitting->attempts[3].optimizer_start, 1e-12));
  CHECK(est->diagnostics.newton_accuracy.checked);
  CHECK(std::isnan(est->fmin));
  CHECK_FALSE(magmaan::api::policy_fit_state(*est).converged);
  est->selected_verdict->status = FitCheck::Passed;
  CHECK(magmaan::api::policy_fit_state(*est).converged);
}
#endif
