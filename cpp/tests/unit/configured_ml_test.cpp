#include <doctest/doctest.h>
#include <fstream>
#include <nlohmann/json.hpp>

#include "magmaan/api/policy.hpp"
#include "magmaan/estimate/configured_ml.hpp"
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
    for (std::size_t i = 0; i < pt->size(); ++i) if (pt->free[i] > 0) {
      bool found = false;
      for (const auto& row : c["parameters"]) {
        if (row["lhs"] != names.row_lhs[i] || row["rhs"] != names.row_rhs[i] ||
            row["op"].get<std::string>() != magmaan::parse::to_string(pt->op[i])) continue;
        found = true;
        CHECK(est->fitting->attempts.front().start(pt->free[i] - 1) == doctest::Approx(row["start"].get<double>()).epsilon(1e-9));
        CHECK(est->theta(pt->free[i] - 1) == doctest::Approx(row["est"].get<double>()).epsilon(1e-5));
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
    } else CHECK(est->fmin == doctest::Approx(c["fmin"].get<double>()).epsilon(1e-9));
  }
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
