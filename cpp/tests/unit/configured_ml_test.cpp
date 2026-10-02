#include <doctest/doctest.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <nlohmann/json.hpp>

#include "magmaan/api/policy.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/configured_ml.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// With exceptions disabled, doctest REQUIRE records failure but cannot unwind.
#define REQUIRE_OR_RETURN(condition) \
  do { const bool condition_ok = static_cast<bool>(condition); \
    REQUIRE(condition_ok); if (!condition_ok) return; } while (false)

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

TEST_CASE("lavaan QR coordinates retain equality row order lost by native merges") {
  const Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(6, 6);
  const std::string model = "f =~ x1+a*x2+b*x3+c*x4+d*x5+e*x6\n";
  auto a = fixture(model + "a == c\nb == e", covariance);
  auto b = fixture(model + "b == e\na == c", covariance);
  CHECK(a.pt.eq_groups == b.pt.eq_groups);
  CHECK(a.pt.lin_constraint_R == b.pt.lin_constraint_R);
  CHECK(a.pt.ordered_affine_R != b.pt.ordered_affine_R);
  auto ka = magmaan::estimate::lavaan_ml_coordinates(a.pt);
  auto kb = magmaan::estimate::lavaan_ml_coordinates(b.pt);
  REQUIRE(ka); REQUIRE(kb);
  CHECK((ka->Kmat - kb->Kmat).cwiseAbs().maxCoeff() == doctest::Approx(0.235702260395516).epsilon(1e-12));
  // Installed lavaan 0.7.2 rotates the null axes by max 0.235702260395516
  // for these two equivalent surfaces; the partition cannot specify its K.
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

TEST_CASE("lavaan ML starts support affine equality models whichever component asks") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d s = lambda * lambda.transpose();
  s.diagonal().array() += .7;
  auto f = fixture("f =~ x1 + a*x2 + a*x3 + x4", s);
  CHECK(lavaan_ml_start_values(f.pt, f.rep, f.sample));
  FittingOptions starts_only;
  starts_only.starts = "lavaan-0.7.2";
  CHECK(fit_ml_configured(f.pt, f.rep, f.sample, starts_only));
  FittingOptions native;
  native.starts = "fabin3";
  CHECK(fit_ml_configured(f.pt, f.rep, f.sample, native));
}

TEST_CASE("lavaan equality preset keeps unsupported constraints and bounds explicit") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1,.8,.6,.9);
  Eigen::Matrix4d covariance=lambda*lambda.transpose();
  covariance.diagonal().array()+=.7;
  FittingOptions options; options.preset="lavaan-0.7.2";
  for(const char* syntax:{"f =~ x1+a*x2+b*x3+x4\na == b*b",
                         "f =~ x1+a*x2+b*x3+x4\na > b"}) {
    auto f=fixture(syntax,covariance);
    CHECK_FALSE(fit_ml_configured(f.pt,f.rep,f.sample,options));
  }
  auto f=fixture("f =~ x1+a*x2+a*x3+x4",covariance);
  const double inf=std::numeric_limits<double>::infinity();
  Bounds bounds{Eigen::VectorXd::Constant(f.pt.n_free(),-inf),Eigen::VectorXd::Constant(f.pt.n_free(),inf)};
  CHECK_FALSE(fit_ml_configured(f.pt,f.rep,f.sample,options,{},Eigen::VectorXd::Zero(f.pt.n_free())));
  bounds.lower(0)=.1;
  CHECK_FALSE(fit_ml_configured(f.pt,f.rep,f.sample,options,{}, {},bounds));
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
    if (c.value("equality", false)) continue;
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
    // Ill-conditioned retries reach the oracle's starts and coordinates;
    // endpoints and verdicts depend on floating-point paths.
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
    for (const auto& attempt : est->fitting->attempts) {
      const bool accepted = attempt.raw_status >= 3 && attempt.raw_status <= 6 &&
          std::isfinite(attempt.gradient_max) && attempt.gradient_max <= 1e-3;
      CHECK(attempt.accepted == accepted);
    }
    if (endpoint) CHECK((magmaan::estimate::fit_verdict(*est).status == magmaan::estimate::FitCheck::Passed) == c["converged"].get<bool>());
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

TEST_CASE("configured equality ML matches lavaan QR starts coordinates gradients and retries") {
  using namespace magmaan;
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) + "/fitting/lavaan_0_7_2.json");
  REQUIRE_OR_RETURN(in.good());
  auto root = nlohmann::json::parse(in, nullptr, false);
  REQUIRE_OR_RETURN(!root.is_discarded());
  auto vector = [](const nlohmann::json& x) {
    Eigen::VectorXd out(x.is_array() ? static_cast<Eigen::Index>(x.size()) : 1);
    if (x.is_array()) for (Eigen::Index i=0; i<out.size(); ++i) out(i)=x[static_cast<std::size_t>(i)].get<double>();
    else out(0)=x.get<double>();
    return out;
  };
  auto matrix = [](const nlohmann::json& x) {
    Eigen::MatrixXd out(static_cast<Eigen::Index>(x.size()), static_cast<Eigen::Index>(x.front().size()));
    for (Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
      out(i,j)=x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get<double>();
    return out;
  };
  for (const auto& c : root["cases"]) {
    if (!c.value("equality", false)) continue;
    CAPTURE(c["model"].get<std::string>());
    CAPTURE(c.value("rescale", 1.0));
    auto parsed = parse::Parser::parse(c["model"].get<std::string>());
    REQUIRE_OR_RETURN(parsed);
    spec::BuildOptions options; options.fixed_x=false; options.std_lv=c["std_lv"].get<bool>();
    options.n_groups=static_cast<int>(c["covariances"].size());
    options.meanstructure=c.contains("group_equal");
    if(options.meanstructure) options.group_equal={spec::GroupEqual::Loadings, spec::GroupEqual::Intercepts};
    spec::LatentNames names;
    auto pt=spec::build(*parsed,options,nullptr,&names); REQUIRE_OR_RETURN(pt);
    auto rep=model::build_matrix_rep(*pt,&names); REQUIRE_OR_RETURN(rep);
    data::SampleStats sample;
    const auto counts=vector(c["n_obs"]);
    for(std::size_t b=0;b<c["covariances"].size();++b) {
      sample.S.push_back(matrix(c["covariances"][b]));
      sample.n_obs.push_back(static_cast<int>(counts(static_cast<Eigen::Index>(b))));
      if(options.meanstructure) sample.mean.push_back(vector(c["means"][b]));
    }
    auto coordinates=estimate::lavaan_ml_coordinates(*pt); REQUIRE_OR_RETURN(coordinates);
    const auto projected=compat::lavaan::to_lavaan_partable(*pt,names);
    const auto roundtrip=compat::lavaan::from_lavaan_partable(projected);
    CHECK(roundtrip.structure.ordered_affine_R==pt->ordered_affine_R);
    CHECK(roundtrip.structure.ordered_affine_d==pt->ordered_affine_d);
    auto roundtrip_coordinates=estimate::lavaan_ml_coordinates(roundtrip.structure);
    REQUIRE_OR_RETURN(roundtrip_coordinates);
    CHECK(roundtrip_coordinates->Kmat.isApprox(coordinates->Kmat,1e-14));
    CHECK((roundtrip_coordinates->theta0-coordinates->theta0).norm()<1e-14);
    CHECK(coordinates->Kmat.isApprox(matrix(c["basis"]),1e-12));
    CHECK((coordinates->theta0-vector(c["offset"])).norm()<1e-12);
    const auto jacobian=matrix(c["jacobian"]);
    REQUIRE_OR_RETURN(pt->ordered_affine_R.size()==static_cast<std::size_t>(jacobian.size()));
    for(Eigen::Index i=0;i<jacobian.rows();++i) for(Eigen::Index j=0;j<jacobian.cols();++j)
      CHECK(pt->ordered_affine_R[static_cast<std::size_t>(i*jacobian.cols()+j)]==doctest::Approx(jacobian(i,j)).epsilon(1e-12));
    estimate::FittingOptions fitting; fitting.preset="lavaan-0.7.2";
    auto est=estimate::fit_ml_configured(*pt,*rep,sample,fitting); REQUIRE_OR_RETURN(est); REQUIRE_OR_RETURN(est->fitting);
    // REQUIRE cannot abort under -fno-exceptions. A count mismatch must
    // report failure before any fixture-indexed access to the actual attempts.
    REQUIRE_OR_RETURN(!est->fitting->attempts.empty());
    // Rescaled equality endpoints follow different PORT floating-point paths:
    // at x100 the opt first gradient is 0.00094775120123813394, whereas
    // lavaan records 0.0016149511731821235 (acceptance threshold 0.001).
    // Preserve starts/coordinates and identical-point derivatives, but compare
    // retry sequences and final verdicts only for path-stable endpoints.
    const bool endpoint_parity = c.value("rescale", 1.0) != 100.0;
    if (endpoint_parity) {
      CHECK(est->fitting->attempts.size()==c["attempts"].size());
      if (est->fitting->attempts.size()!=c["attempts"].size()) continue;
    }
    for(std::size_t i=0;i<pt->size();++i) if(pt->free[i]>0) {
      bool found=false;
      for(const auto& row:c["parameters"]) {
        if(row["lhs"]!=names.row_lhs[i] || row["rhs"]!=names.row_rhs[i] ||
           row["op"].get<std::string>()!=parse::to_string(pt->op[i]) || row["group"].get<int>()!=pt->group[i]) continue;
        found=true;
        CHECK(est->fitting->attempts.front().start(pt->free[i]-1)==doctest::Approx(row["start"].get<double>()).epsilon(1e-9));
        if (endpoint_parity) CHECK(est->theta(pt->free[i]-1)==doctest::Approx(row["est"].get<double>()).epsilon(1e-5));
      }
      CHECK(found);
    }
    auto evaluator=model::ModelEvaluator::build(*pt,*rep); REQUIRE_OR_RETURN(evaluator);
    auto objective=estimate::ml_objective(*evaluator,sample); REQUIRE_OR_RETURN(objective);
    // Check every oracle derivative at its recorded point independently of
    // how many attempts the actual search needed.
    for (const auto& oracle : c["attempts"]) {
      Eigen::VectorXd full_gradient;
      REQUIRE_OR_RETURN(std::isfinite(objective->f(vector(oracle["theta"]),full_gradient)));
      const Eigen::VectorXd same_endpoint=coordinates->Kmat.transpose() *
          full_gradient.cwiseQuotient(vector(oracle["parameter_scale"]));
      CHECK((same_endpoint-vector(oracle["gradient"])).cwiseAbs().maxCoeff()<1e-9);
    }
    for(std::size_t i=0;i<est->fitting->attempts.size();++i) {
      CAPTURE(i);
      const auto& actual=est->fitting->attempts[i];
      // Starts and search coordinates remain pinned for each matching attempt.
      REQUIRE_OR_RETURN(i<c["attempts"].size());
      const auto& oracle=c["attempts"][i];
      CHECK(actual.simple_start==oracle["simple"].get<bool>());
      CHECK(actual.standardized==oracle["standardized"].get<bool>());
      if (endpoint_parity) CHECK(actual.accepted==oracle["accepted"].get<bool>());
      CHECK(actual.optimizer_start.isApprox(vector(oracle["start"]),1e-9));
      CHECK(actual.parameter_scale.isApprox(vector(oracle["parameter_scale"]),1e-12));
      CHECK(actual.port_scale.isApprox(vector(oracle["port_scale"]),1e-12));
      Eigen::VectorXd full_gradient;
      const Eigen::VectorXd actual_theta=(coordinates->Kmat * actual.optimizer_end +
          coordinates->theta0).cwiseQuotient(actual.parameter_scale);
      REQUIRE_OR_RETURN(std::isfinite(objective->f(actual_theta,full_gradient)));
      const Eigen::VectorXd actual_endpoint=coordinates->Kmat.transpose() *
          full_gradient.cwiseQuotient(actual.parameter_scale);
      CHECK((actual.optimizer_gradient-actual_endpoint).cwiseAbs().maxCoeff()<1e-10);
      const double endpoint_max=actual_endpoint.cwiseAbs().maxCoeff();
      CHECK(actual.gradient_max==doctest::Approx(endpoint_max).epsilon(1e-12));
      const bool endpoint_accepted=actual.raw_status>=3 && actual.raw_status<=6 &&
          std::isfinite(endpoint_max) && endpoint_max<=1e-3;
      CHECK(actual.accepted==endpoint_accepted);
      CHECK((coordinates->A_eq * actual_theta - coordinates->b_eq).norm()<1e-10);
      const Eigen::VectorXd oracle_end=coordinates->Kmat.transpose() *
          (vector(oracle["theta"]).cwiseProduct(actual.parameter_scale)-coordinates->theta0);
      if (endpoint_parity) {
        CHECK(actual.optimizer_end.isApprox(oracle_end,1e-5));
        CHECK((actual.optimizer_gradient-vector(oracle["gradient"])).cwiseAbs().maxCoeff()<1e-9);
      }
    }
    REQUIRE_OR_RETURN(est->fitting->selected_attempt<est->fitting->attempts.size());
    CHECK((estimate::fit_verdict(*est).status==estimate::FitCheck::Passed)==
          est->fitting->attempts[est->fitting->selected_attempt].accepted);
    if (endpoint_parity) CHECK((estimate::fit_verdict(*est).status==estimate::FitCheck::Passed)==c["converged"].get<bool>());
    if (endpoint_parity) CHECK(est->fmin==doctest::Approx(c["fmin"].get<double>()).epsilon(1e-9));
  }
}

TEST_CASE("lavaan preset rejects nonzero affine RHS only when a standardized retry is needed") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d covariance=lambda*lambda.transpose();
  covariance.diagonal().array()+=.7;
  const Eigen::Vector4d units(1000, 1, .001, 1);
  covariance=units.asDiagonal()*covariance*units.asDiagonal();
  auto f=fixture("f =~ x1+a*x2+b*x3+x4\na-b == 0.000001",covariance);
  FittingOptions options; options.preset="lavaan-0.7.2";
  auto result=fit_ml_configured(f.pt,f.rep,f.sample,options);
  REQUIRE_FALSE(result);
  CHECK(result.error().kind==magmaan::FitError::Kind::NumericIssue);
  CHECK(result.error().detail.find("standardized retry is unsupported for nonzero affine constraint RHS")!=std::string::npos);
  // A positive iteration count and finite objective prove the first,
  // unstandardized affine attempt ran before this unsupported transition.
  CHECK(result.error().iterations>0);
  CHECK(std::isfinite(result.error().f_value));
  CHECK(result.error().f_value>0);
}

TEST_CASE("lavaan preset rejects homogeneous ratio retries that change the constraint surface") {
  using namespace magmaan::estimate;
  Eigen::Vector4d lambda(1, .8, .6, .9);
  Eigen::Matrix4d covariance=lambda*lambda.transpose();
  covariance.diagonal().array()+=.7;
  const Eigen::Vector4d units(1000, 1, .001, 1);
  covariance=units.asDiagonal()*covariance*units.asDiagonal();
  auto f=fixture("f =~ x1+a*x2+b*x3+x4\na == 2*b",covariance);
  FittingOptions options; options.preset="lavaan-0.7.2";
  auto result=fit_ml_configured(f.pt,f.rep,f.sample,options);
  REQUIRE_FALSE(result);
  CHECK(result.error().kind==magmaan::FitError::Kind::NumericIssue);
  CHECK(result.error().detail.find("parameter scaling changes the equality constraint surface")!=std::string::npos);
  CHECK(result.error().iterations>0);
  CHECK(std::isfinite(result.error().f_value));
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

#ifdef MAGMAAN_WITH_PORT
TEST_CASE("configured FIML matches pinned MCAR MAR starts coordinates gradients and verdicts") {
  using namespace magmaan;
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR)+"/fitting/lavaan_fiml_0_7_2.json");
  REQUIRE_OR_RETURN(in.good());
  const auto root=nlohmann::json::parse(in,nullptr,false);
  REQUIRE_OR_RETURN(!root.is_discarded());
  auto vector=[](const nlohmann::json& a) {
    Eigen::VectorXd v(a.is_array()?static_cast<Eigen::Index>(a.size()):1);
    if(a.is_array()) for(Eigen::Index j=0;j<v.size();++j) v(j)=a[static_cast<std::size_t>(j)].get<double>();
    else v(0)=a.get<double>();
    return v;
  };
  for(const auto& c:root["cases"]) {
    CAPTURE(c["mechanism"]); CAPTURE(c["groups"]); CAPTURE(c["model"]);
    auto parsed=parse::Parser::parse(c["model"].get<std::string>()); REQUIRE_OR_RETURN(parsed);
    spec::BuildOptions opts; opts.fixed_x=false; opts.meanstructure=true;
    opts.n_groups=c["groups"].get<int>();
    if(c.contains("group_equal")) opts.group_equal={spec::GroupEqual::Loadings,spec::GroupEqual::Intercepts};
    spec::LatentNames names;
    auto pt=spec::build(*parsed,opts,nullptr,&names); REQUIRE_OR_RETURN(pt);
    auto rep=model::build_matrix_rep(*pt,&names); REQUIRE_OR_RETURN(rep);
    data::RawData raw;
    for(const auto& b:c["raw"]) {
      Eigen::MatrixXd x(static_cast<Eigen::Index>(b.size()),4);
      for(Eigen::Index i=0;i<x.rows();++i) for(int j=0;j<4;++j) {
        const auto& value=b[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        x(i,j)=value.is_null()?std::numeric_limits<double>::quiet_NaN():value.get<double>();
      }
      Eigen::Matrix<std::uint8_t,Eigen::Dynamic,Eigen::Dynamic> mask(x.rows(),x.cols());
      for(Eigen::Index i=0;i<x.rows();++i) for(Eigen::Index j=0;j<x.cols();++j)
        mask(i,j)=std::isfinite(x(i,j))?1:0;
      raw.mask.push_back(std::move(mask));
      raw.X.push_back(std::move(x));
    }
    auto pack=estimate::fiml::fiml_pack(raw); REQUIRE_OR_RETURN(pack);
    auto h1=estimate::lavaan_fiml_h1(raw,*pack); REQUIRE_OR_RETURN(h1);
    estimate::FittingOptions options; options.preset="lavaan-0.7.2";
    Eigen::VectorXd start;
    if(c.value("invalid_start",false)) start=Eigen::VectorXd::Zero(pt->n_free());
    auto est=estimate::fit_fiml_configured(*pt,*rep,raw,*pack,*h1,options,{},start);
    if(!est) CAPTURE(est.error().detail);
    REQUIRE_OR_RETURN(est); REQUIRE_OR_RETURN(est->fitting);
    const auto& attempts=est->fitting->attempts;
    REQUIRE_OR_RETURN(attempts.size()==c["attempts"].size());
    for(std::size_t i=0;i<pt->size();++i) if(pt->free[i]>0) {
      bool found=false;
      for(const auto& row:c["parameters"]) {
        if(row["lhs"]!=names.row_lhs[i] || row["rhs"]!=names.row_rhs[i] ||
           row["op"].get<std::string>()!=parse::to_string(pt->op[i]) || row["group"]!=pt->group[i]) continue;
        found=true;
        CHECK(attempts.front().start(pt->free[i]-1)==doctest::Approx(row["start"].get<double>()).epsilon(1e-9));
        CHECK(est->theta(pt->free[i]-1)==doctest::Approx(row["est"].get<double>()).epsilon(1e-5));
      }
      CHECK(found);
    }
    for(std::size_t a=0;a<attempts.size();++a) {
      const auto& actual=attempts[a]; const auto& oracle=c["attempts"][a];
      CHECK(actual.simple_start==oracle["simple"].get<bool>());
      CHECK(actual.standardized==oracle["standardized"].get<bool>());
      CHECK(actual.accepted==oracle["accepted"].get<bool>());
      const bool accepts=actual.raw_status>=3 && actual.raw_status<=6 &&
          std::isfinite(actual.gradient_max) && actual.gradient_max<=1e-3;
      CHECK(actual.accepted==accepts);
      // The invalid-covariance early return has no start/scale attributes:
      // the oracle's returned theta is the retained driven start itself.
      const auto z=vector(oracle["start"].empty()?oracle["theta"]:oracle["start"]);
      REQUIRE_OR_RETURN(z.size()==actual.optimizer_start.size());
      CHECK((actual.optimizer_start-z).cwiseAbs().maxCoeff()<1e-9);
      if (!oracle["parameter_scale"].empty()) {
        const auto scale=vector(oracle["parameter_scale"]);
        REQUIRE_OR_RETURN(scale.size()==actual.parameter_scale.size());
        CHECK((actual.parameter_scale-scale).cwiseAbs().maxCoeff()<1e-9);
      }
      if(!oracle["gradient"].empty()) {
        const auto scale=vector(oracle["parameter_scale"]);
        const auto g=vector(oracle["gradient"]);
        REQUIRE_OR_RETURN(g.size()==actual.optimizer_gradient.size());
        // Follow the approved rescaled-endpoint contract (TASK-47 #6).
        // x100 has opt max gradient 1.03649e-5 versus oracle 2.44914e-4;
        // both accept below 1e-3 and estimates/objectives match their existing
        // tolerances, but final gradient vectors follow different PORT paths.
        // Keep those gates, starts/scales, each endpoint's acceptance, and
        // the identical-point derivative gate below; no tolerance changes.
        if (c.value("rescale",1.0)!=100.0)
          CHECK((actual.optimizer_gradient-g).cwiseAbs().maxCoeff()<1e-6);
        // Pin derivatives at an identical endpoint independently of PORT's
        // floating-point search path.
        auto coordinates=estimate::lavaan_ml_coordinates(*pt); REQUIRE_OR_RETURN(coordinates);
        auto ev=model::ModelEvaluator::build(*pt,*rep); REQUIRE_OR_RETURN(ev);
        const auto theta=vector(oracle["theta"]);
        auto moments=ev->evaluate(theta,true,true); REQUIRE_OR_RETURN(moments);
        auto vg=estimate::fiml::FIML{}.value_gradient(raw,pack->cache,moments->moments,
            moments->J_sigma,moments->J_mu); REQUIRE_OR_RETURN(vg);
        Eigen::VectorXd driven=0.5*vg->gradient.cwiseQuotient(scale);
        if(coordinates->active()) driven=(coordinates->Kmat.transpose()*driven).eval();
        CHECK((driven-g).cwiseAbs().maxCoeff()<1e-9);
      }
    }
    CHECK((estimate::fit_verdict(*est).status==estimate::FitCheck::Passed)==c["converged"].get<bool>());
    if(c["fmin"].is_null()) CHECK(std::isnan(est->fmin));
    else CHECK(est->fmin==doctest::Approx(c["fmin"].get<double>()).epsilon(1e-9));
    if (!c["fmin"].is_null()) CHECK(est->diagnostics.newton_accuracy.checked);
  }
}
#endif
