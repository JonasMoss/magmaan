#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "../test_fit.hpp"
#include "../../src/estimate/detail_identification_probe.hpp"
#include "magmaan/api/sem.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/frontier/convergence_policy.hpp"
#include "magmaan/estimate/frontier/identification.hpp"
#include "magmaan/estimate/frontier/ml_psd_fallback.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// Structural identification (board TASK-33.3): the data-free rank check on
// the route's moment map, and its fold into the fit verdict. Local Newton
// accuracy, Hessian positivity or a condition number never decide it.

namespace {

using magmaan::estimate::FitCheck;
using magmaan::estimate::IdentificationMap;
using magmaan::estimate::IdentificationReason;
using magmaan::estimate::IdentificationReport;
using magmaan::estimate::IdentificationStatus;
namespace fr = magmaan::estimate::frontier;

struct Model {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames names;
  magmaan::model::MatrixRep rep;
};

Model model(const std::string& syntax, magmaan::spec::BuildOptions options = {}) {
  auto flat = magmaan::parse::Parser::parse(syntax);
  REQUIRE(flat.has_value());
  Model m;
  auto pt = magmaan::spec::build(*flat, options, nullptr, &m.names);
  REQUIRE_MESSAGE(pt.has_value(), (pt.has_value() ? "" : pt.error().detail));
  m.pt = std::move(*pt);
  auto rep = magmaan::model::build_matrix_rep(m.pt);
  REQUIRE(rep.has_value());
  m.rep = std::move(*rep);
  return m;
}

magmaan::spec::BuildOptions free_marker() {
  magmaan::spec::BuildOptions o;
  o.auto_fix_first = false;
  return o;
}

IdentificationReport check(const Model& m) {
  auto r = fr::check_structural_identification(m.pt, m.rep);
  REQUIRE(r.has_value());
  return *r;
}

// One-factor population covariance with the given loadings and unit
// uniquenesses minus the communality (unit variances when loadings < 1).
Eigen::MatrixXd one_factor_cov(const std::vector<double>& loading) {
  const auto p = static_cast<Eigen::Index>(loading.size());
  Eigen::VectorXd l(p);
  for (Eigen::Index j = 0; j < p; ++j) l(j) = loading[static_cast<std::size_t>(j)];
  Eigen::MatrixXd S = l * l.transpose();
  S.diagonal().setOnes();
  return S;
}

magmaan::data::SampleStats sample(const Eigen::MatrixXd& S, std::int64_t n = 300) {
  magmaan::data::SampleStats s;
  s.S.push_back(S);
  s.n_obs.push_back(n);
  return s;
}

Eigen::MatrixXd one_factor_data(Eigen::Index n, const std::vector<double>& loading,
                                unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  const auto p = static_cast<Eigen::Index>(loading.size());
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < p; ++j) {
      const double l = loading[static_cast<std::size_t>(j)];
      X(i, j) = 1.0 + 0.5 * static_cast<double>(j) + l * eta +
                std::sqrt(1.0 - l * l) * norm(rng);
    }
  }
  return X;
}

double smallest_relative_singular_value(const Eigen::MatrixXd& J) {
  Eigen::MatrixXd scaled = J;
  for (Eigen::Index j = 0; j < scaled.cols(); ++j) {
    const double n = scaled.col(j).norm();
    if (n > 0) scaled.col(j) /= n;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(scaled);
  const auto& v = svd.singularValues();
  return J.rows() < J.cols() ? 0.0 : v(v.size() - 1) / v(0);
}

std::string effects_coded_eight_factors(bool intercept_constraints) {
  // The structure of Little (2013) Table 10.3: eight correlated factors with
  // three indicators each, effects coding of every loading block and, with a
  // mean structure, of every intercept block, latent means free.
  std::string s;
  for (int f = 1; f <= 8; ++f) {
    const int a = 3 * f - 2, b = 3 * f - 1, c = 3 * f;
    s += "F" + std::to_string(f) + " =~ NA*l" + std::to_string(a) + "*y" +
         std::to_string(a) + " + l" + std::to_string(b) + "*y" + std::to_string(b) +
         " + l" + std::to_string(c) + "*y" + std::to_string(c) + "\n";
    for (int k : {a, b, c})
      s += "y" + std::to_string(k) + " ~ t" + std::to_string(k) + "*1\n";
    s += "F" + std::to_string(f) + " ~ 1\n";
    s += "l" + std::to_string(a) + " == 3 - l" + std::to_string(b) + " - l" +
         std::to_string(c) + "\n";
    if (intercept_constraints)
      s += "t" + std::to_string(a) + " == 0 - t" + std::to_string(b) + " - t" +
           std::to_string(c) + "\n";
  }
  return s;
}

}  // namespace

TEST_CASE("Identification: the free-marker three-indicator ridge fails the counting rule") {
  const Model m = model("f =~ x1 + x2 + x3", free_marker());
  const auto r = check(m);
  CHECK(r.status == IdentificationStatus::Unidentified);
  CHECK(r.reason == IdentificationReason::CountingRule);
  CHECK(r.map == IdentificationMap::Covariance);
  CHECK(r.n_parameters == 7);
  CHECK(r.n_moments == 6);
  CHECK_FALSE(r.counting_rule);
  CHECK(r.rank == 6);
  REQUIRE(r.null_directions.cols() == 1);
  CHECK(r.null_directions.rows() == 7);
}

TEST_CASE("Identification: the null direction is the loading/variance rescaling") {
  // Four indicators pass the counting rule (13 parameters, 14 moments with
  // means); only the rank sees the ridge Lambda -> c Lambda, Psi -> Psi / c^2.
  auto options = free_marker();
  options.meanstructure = true;
  const Model m = model("f =~ x1 + x2 + x3 + x4", options);
  auto ev = magmaan::model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto con = magmaan::estimate::build_eq_constraints(m.pt);
  REQUIRE(con.has_value());
  // A fitted-looking point: loadings, factor variance, residuals, intercepts.
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(m.pt.n_free());
  std::vector<int> loading_index, psi_index;
  for (std::size_t i = 0; i < m.pt.size(); ++i) {
    if (m.pt.free[i] <= 0) continue;
    const auto k = m.pt.free[i] - 1;
    if (m.pt.op[i] == magmaan::parse::Op::Measurement) {
      theta(k) = 0.6 + 0.1 * static_cast<double>(loading_index.size());
      loading_index.push_back(k);
    } else if (m.pt.op[i] == magmaan::parse::Op::Covariance) {
      theta(k) = 0.7;
      if (m.pt.lhs_var[i] == m.pt.rhs_var[i] && m.pt.is_user_latent[static_cast<std::size_t>(m.pt.lhs_var[i])])
        psi_index.push_back(k);
    } else {
      theta(k) = 0.25;
    }
  }
  REQUIRE(loading_index.size() == 4);
  REQUIRE(psi_index.size() == 1);
  const auto r = fr::check_structural_identification(m.pt, *ev, *con, false, &theta);
  CHECK(r.status == IdentificationStatus::Unidentified);
  CHECK(r.reason == IdentificationReason::Rank);
  CHECK(r.counting_rule);
  CHECK(r.n_parameters == 13);
  CHECK(r.n_moments == 14);
  CHECK(r.rank == 12);
  REQUIRE(r.null_directions.cols() == 1);
  CHECK(r.directions_at_estimate);
  // The exact scale direction at theta: d lambda = lambda, d psi = -2 psi.
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(theta.size());
  for (int k : loading_index) expected(k) = theta(k);
  expected(psi_index[0]) = -2.0 * theta(psi_index[0]);
  expected.normalize();
  const Eigen::VectorXd d = r.null_directions.col(0);
  CHECK(std::abs(std::abs(d.dot(expected)) - 1.0) < 1e-10);
  // Null values sit at roundoff; the generic rank clears the gap by far.
  REQUIRE(r.smallest_singular_values.size() >= 2);
  CHECK(r.smallest_singular_values(0) < 1e-13);
  CHECK(r.smallest_singular_values(1) > 1e-4);
  CHECK(r.n_points == 3);
  for (Eigen::Index k = 0; k < r.min_relative_singular_values.size(); ++k)
    CHECK(r.min_relative_singular_values(k) < r.null_tolerance);

  const auto labels = fr::free_parameter_labels(m.pt, m.names);
  const auto text = fr::describe_null_directions(r, labels);
  REQUIRE(text.size() == 1);
  CHECK(text[0].find("f=~x1") != std::string::npos);
  CHECK(text[0].find("f~~f") != std::string::npos);
  CHECK(text[0].find("x1~1") == std::string::npos);
  CHECK(text[0].find("x1~~x1") == std::string::npos);
}

TEST_CASE("Identification: marker, std.lv, effects coding and tau-equivalence are identified") {
  auto mean = magmaan::spec::BuildOptions{};
  mean.meanstructure = true;
  auto stdlv = magmaan::spec::BuildOptions{};
  stdlv.std_lv = true;
  auto effects = magmaan::spec::BuildOptions{};
  effects.effect_coding = true;
  effects.meanstructure = true;
  const std::vector<std::pair<std::string, magmaan::spec::BuildOptions>> cases = {
      {"f =~ x1 + x2 + x3", {}},
      {"f =~ x1 + x2 + x3", stdlv},
      {"f =~ x1 + x2 + x3 + x4", mean},
      {"f =~ x1 + x2 + x3 + x4", stdlv},
      {"f =~ x1 + x2 + x3 + x4", effects},
      {"f =~ a*x1 + a*x2 + a*x3", stdlv},
      {"f =~ 1*x1 + 1*x2 + 1*x3", {}},
      // Two-indicator factors are generically identified through their
      // covariance with another factor.
      {"f1 =~ x1 + x2\nf2 =~ x3 + x4", {}},
      {"f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf3 =~ x7 + x8 + x9\n"
       "f3 ~ f1 + f2", {}},
      {"y ~ x1 + x2", {}},
  };
  for (const auto& [syntax, options] : cases) {
    CAPTURE(syntax);
    const auto r = check(model(syntax, options));
    CHECK(r.status == IdentificationStatus::Identified);
    CHECK(r.reason == IdentificationReason::Rank);
    CHECK(r.rank == r.n_parameters);
    CHECK(r.n_points == 1);
    REQUIRE(r.min_relative_singular_values.size() == 1);
    CHECK(r.min_relative_singular_values(0) >= r.identified_tolerance);
    CHECK(r.null_directions.size() == 0);
  }
}

TEST_CASE("Identification: equal loadings with a free factor variance keep the ridge") {
  const auto r = check(model("f =~ a*x1 + a*x2 + a*x3", free_marker()));
  CHECK(r.status == IdentificationStatus::Unidentified);
  CHECK(r.reason == IdentificationReason::Rank);
  CHECK(r.n_parameters == 5);
  CHECK(r.rank == 4);
  // A lone two-indicator factor fails the counting rule.
  const auto two = check(model("f =~ x1 + x2"));
  CHECK(two.status == IdentificationStatus::Unidentified);
  CHECK(two.reason == IdentificationReason::CountingRule);
}

TEST_CASE("Identification: Little Table 10.3 effects coding of loadings and intercepts") {
  auto options = magmaan::spec::BuildOptions{};
  options.meanstructure = true;
  const auto pinned = check(model(effects_coded_eight_factors(true), options));
  CHECK(pinned.status == IdentificationStatus::Identified);
  CHECK(pinned.map == IdentificationMap::CovarianceMean);
  // Without the intercept sums each latent mean trades against its three
  // intercepts: one null direction per factor.
  const auto loose = check(model(effects_coded_eight_factors(false), options));
  CHECK(loose.status == IdentificationStatus::Unidentified);
  CHECK(loose.reason == IdentificationReason::Rank);
  CHECK(loose.n_parameters - loose.rank == 8);
  CHECK(loose.null_directions.cols() == 8);
}

TEST_CASE("Identification: multi-group invariance models") {
  auto options = magmaan::spec::BuildOptions{};
  options.meanstructure = true;
  options.n_groups = 2;
  const std::string syntax = "f =~ x1 + x2 + x3 + x4";
  CHECK(check(model(syntax, options)).status == IdentificationStatus::Identified);
  options.group_equal = {magmaan::spec::GroupEqual::Loadings,
                         magmaan::spec::GroupEqual::Intercepts};
  const Model scalar = model(syntax, options);
  CHECK(check(scalar).status == IdentificationStatus::Identified);

  // Free latent means in both groups with equal intercepts: one mean is a
  // free location for the intercepts.
  const Model both = model("f =~ x1 + x2 + x3 + x4\nf ~ c(m1, m2)*1", options);
  const auto r = check(both);
  CHECK(r.status == IdentificationStatus::Unidentified);
  CHECK(r.n_parameters - r.rank == 1);

  // The scalar model fits with an accepted verdict.
  magmaan::data::SampleStats s;
  const Eigen::MatrixXd S1 = one_factor_cov({0.8, 0.7, 0.6, 0.5});
  const Eigen::MatrixXd S2 = one_factor_cov({0.8, 0.7, 0.6, 0.5}) * 1.2;
  s.S = {S1, S2};
  Eigen::VectorXd m1(4), m2(4);
  m1 << 1.0, 2.0, 3.0, 4.0;
  m2 << 1.2, 2.2, 3.1, 4.2;
  s.mean = {m1, m2};
  s.n_obs = {300, 250};
  auto fit = magmaan::test::fit(scalar.pt, scalar.rep, s);
  REQUIRE(fit.has_value());
  CHECK(fit->diagnostics.identification.status == IdentificationStatus::Identified);
  CHECK(magmaan::estimate::fit_verdict(*fit).identification == FitCheck::Passed);
  CHECK(magmaan::estimate::fit_verdict(*fit).status == FitCheck::Passed);
}

TEST_CASE("Identification: nonlinear constraints are unchecked") {
  const auto r = check(model("f =~ x1 + b*x2 + c*x3\nb == c^2"));
  CHECK(r.status == IdentificationStatus::Unchecked);
  CHECK(r.reason == IdentificationReason::NonlinearConstraints);
}

TEST_CASE("Identification: the report is deterministic and independent of the data scale") {
  const Model m = model("f =~ x1 + x2 + x3 + x4");
  const auto a = check(m);
  const auto b = check(m);
  CHECK(a.status == b.status);
  CHECK(a.min_relative_singular_values == b.min_relative_singular_values);
  CHECK(a.smallest_singular_values == b.smallest_singular_values);

  // Rescale the indicators by 10 and 0.1: the fit verdicts stay accepted and
  // the identification reports are identical, since the draws never look at
  // the data.
  const Eigen::MatrixXd S = one_factor_cov({0.8, 0.7, 0.6, 0.5});
  Eigen::VectorXd units(4);
  units << 10.0, 0.1, 1.0, 10.0;
  const Eigen::MatrixXd scaled = units.asDiagonal() * S * units.asDiagonal();
  auto base = magmaan::test::fit(m.pt, m.rep, sample(S));
  auto rescaled = magmaan::test::fit(m.pt, m.rep, sample(scaled));
  REQUIRE(base.has_value());
  REQUIRE(rescaled.has_value());
  const auto& ra = base->diagnostics.identification;
  const auto& rb = rescaled->diagnostics.identification;
  CHECK(ra.status == IdentificationStatus::Identified);
  CHECK(rb.status == IdentificationStatus::Identified);
  CHECK(ra.min_relative_singular_values == rb.min_relative_singular_values);
  CHECK(magmaan::estimate::fit_verdict(*base).status == FitCheck::Passed);
  CHECK(magmaan::estimate::fit_verdict(*rescaled).status == FitCheck::Passed);
}

TEST_CASE("Identification: an empirically underidentified estimate is not refused") {
  // Two two-indicator factors are generically identified, but at a point
  // with zero factor covariance each block has four parameters for three
  // moments. That is near-singularity at the estimate, not structural.
  const Model m = model("f1 =~ x1 + x2\nf2 =~ x3 + x4");
  auto ev = magmaan::model::ModelEvaluator::build(m.pt, m.rep);
  REQUIRE(ev.has_value());
  auto con = magmaan::estimate::build_eq_constraints(m.pt);
  REQUIRE(con.has_value());
  Eigen::VectorXd theta = Eigen::VectorXd::Constant(m.pt.n_free(), 0.6);
  for (std::size_t i = 0; i < m.pt.size(); ++i) {
    if (m.pt.free[i] > 0 && m.pt.op[i] == magmaan::parse::Op::Covariance &&
        m.pt.lhs_var[i] != m.pt.rhs_var[i])
      theta(m.pt.free[i] - 1) = 0.0;
  }
  auto J = ev->dsigma_dtheta(theta);
  REQUIRE(J.has_value());
  CHECK(smallest_relative_singular_value(*J) < 1e-12);
  const auto r = fr::check_structural_identification(m.pt, *ev, *con, false, &theta);
  CHECK(r.status == IdentificationStatus::Identified);

  // A fit to a sample with zero cross-block covariance lands on that set.
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(4, 4);
  S(0, 1) = S(1, 0) = 0.5;
  S(2, 3) = S(3, 2) = 0.4;
  auto fit = magmaan::test::fit(m.pt, m.rep, sample(S));
  REQUIRE(fit.has_value());
  CHECK(fit->diagnostics.identification.status == IdentificationStatus::Identified);
  // Whatever the numerical checks conclude there, identification is not the
  // reason.
  CHECK(magmaan::estimate::fit_verdict(*fit).identification == FitCheck::Passed);
}

TEST_CASE("Identification: an unidentified model fails every verdict") {
  IdentificationReport unidentified;
  unidentified.status = IdentificationStatus::Unidentified;
  IdentificationReport unchecked;
  magmaan::estimate::FitVerdict passed;
  passed.status = passed.objective = passed.stationarity = FitCheck::Passed;
  const auto failed = magmaan::estimate::with_identification(passed, unidentified);
  CHECK(failed.status == FitCheck::Failed);
  CHECK(failed.identification == FitCheck::Failed);
  CHECK(failed.stationarity == FitCheck::Passed);
  const auto same = magmaan::estimate::with_identification(passed, unchecked);
  CHECK(same.status == FitCheck::Passed);
  CHECK(same.identification == FitCheck::Unchecked);

  // A selected compatibility verdict cannot override the failure.
  magmaan::estimate::Estimates est;
  est.selected_verdict = passed;
  est.diagnostics.identification = unidentified;
  CHECK(magmaan::estimate::fit_verdict(est).status == FitCheck::Failed);
  est.diagnostics.identification = unchecked;
  CHECK(magmaan::estimate::fit_verdict(est).status == FitCheck::Passed);

  // Explicit policies fail on it too, with the reason recorded.
  magmaan::estimate::FitDiagnostics d;
  d.identification = unidentified;
  const auto explicit_assessment = fr::assess_convergence(d, fr::newton_convergence_policy());
  CHECK(explicit_assessment.status == FitCheck::Failed);
  CHECK(explicit_assessment.identification.status == FitCheck::Failed);
  CHECK(explicit_assessment.identification.reason.find("unidentified") != std::string::npos);
  const auto compatibility = fr::assess_convergence(d, fr::compatibility_convergence_policy());
  CHECK(compatibility.status == FitCheck::Failed);
  CHECK(compatibility.compatibility_verdict.identification == FitCheck::Failed);
}

TEST_CASE("Identification: the PSD fallback cannot accept the free-marker ridge") {
  // The task-28 witness: a PSD fit on the scale ridge used to pass the local
  // Newton check (condition 8e10 under the 1e12 cap).
  const Model m = model("f =~ x1 + x2 + x3", free_marker());
  Eigen::MatrixXd S(3, 3);
  S << 1, .5, .4, .5, 1, .32, .4, .32, 1;
  const auto witness = sample(S);
  auto start = magmaan::estimate::simple_start_values(m.pt, m.rep, witness, {});
  REQUIRE(start.has_value());
  fr::MlPsdFallbackOptions options;
  options.ordinary.nlopt.max_eval = 1;
  auto policy = fr::fit_ml_psd_fallback(m.pt, m.rep, witness, *start, options);
  REQUIRE(policy.ordinary.has_value());
  CHECK(policy.ordinary->diagnostics.identification.status ==
        IdentificationStatus::Unidentified);
  REQUIRE(policy.psd.has_value());
  REQUIRE(policy.psd->has_value());
  const auto& psd = **policy.psd;
  CHECK(psd.diagnostics.identification.status == IdentificationStatus::Unidentified);
  CHECK(psd.diagnostics.identification.reason == IdentificationReason::CountingRule);
  CHECK(magmaan::estimate::fit_verdict(psd).status == FitCheck::Failed);
  CHECK(magmaan::estimate::fit_verdict(psd).identification == FitCheck::Failed);
  CHECK(policy.accepted_fit() == nullptr);
}

TEST_CASE("Identification: direct FIML witness and its marker control") {
  const Eigen::MatrixXd X = one_factor_data(400, {0.8, 0.7, 0.6, 0.5}, 20261007u);
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  auto options = free_marker();
  options.meanstructure = true;
  options.fixed_x = false;
  const Model ridge = model("f =~ x1 + x2 + x3 + x4", options);
  auto fit = magmaan::test::fit_fiml(ridge.pt, ridge.rep, raw);
  REQUIRE(fit.has_value());
  const auto& r = fit->diagnostics.identification;
  CHECK(r.status == IdentificationStatus::Unidentified);
  CHECK(r.reason == IdentificationReason::Rank);
  CHECK(r.counting_rule);
  CHECK(r.map == IdentificationMap::CovarianceMean);
  CHECK(r.n_parameters - r.rank == 1);
  CHECK(magmaan::estimate::fit_verdict(*fit).status == FitCheck::Failed);

  auto marker_options = magmaan::spec::BuildOptions{};
  marker_options.meanstructure = true;
  marker_options.fixed_x = false;
  const Model marker = model("f =~ x1 + x2 + x3 + x4", marker_options);
  auto control = magmaan::test::fit_fiml(marker.pt, marker.rep, raw);
  REQUIRE(control.has_value());
  CHECK(control->diagnostics.identification.status == IdentificationStatus::Identified);
  CHECK(magmaan::estimate::fit_verdict(*control).status == FitCheck::Passed);
}

TEST_CASE("Identification: ordinal routes rank the threshold/correlation map") {
  std::mt19937 rng(20260518);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  const double loading[4] = {0.9, 0.8, 0.7, 0.6};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.6) + (y > 0.5);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  using magmaan::estimate::OrdinalParameterization;
  using magmaan::estimate::OrdinalWeightKind;
  for (const auto parameterization :
       {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
    for (const bool identified : {true, false}) {
      CAPTURE(identified);
      CAPTURE(parameterization == OrdinalParameterization::Theta);
      const std::string thresholds =
          "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
          "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
      const Model m = model((identified ? "f =~ x1 + x2 + x3 + x4\n"
                                        : "f =~ NA*x1 + x2 + x3 + x4\n") +
                            thresholds);
      auto fit = magmaan::test::fit_ordinal_bounded(
          m.pt, m.rep, *stats, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptLbfgs, {}, parameterization);
      REQUIRE_MESSAGE(fit.has_value(), (fit.has_value() ? "" : fit.error().detail));
      const auto& r = fit->diagnostics.identification;
      CHECK(r.map == IdentificationMap::Ordinal);
      if (identified) {
        CHECK(r.status == IdentificationStatus::Identified);
        CHECK(magmaan::estimate::fit_verdict(*fit).identification == FitCheck::Passed);
      } else {
        CHECK(r.status == IdentificationStatus::Unidentified);
        CHECK(r.n_parameters - r.rank == 1);
        CHECK(magmaan::estimate::fit_verdict(*fit).status == FitCheck::Failed);
      }
    }
  }
}

#ifdef MAGMAAN_ENABLE_TEST_PROBES
namespace {
std::size_t identification_checks = 0;
void count_identification(const IdentificationReport&, double) {
  ++identification_checks;
}
struct IdentificationObserver {
  IdentificationObserver() {
    identification_checks = 0;
    magmaan::estimate::identification_test::set_observer(count_identification);
  }
  ~IdentificationObserver() {
    magmaan::estimate::identification_test::set_observer(nullptr);
  }
};
}
TEST_CASE("Identification: immutable API models cache one data-free report across fits") {
  IdentificationObserver observer;
  magmaan::api::ModelOptions options;
  options.build.auto_fix_first = false;
  auto m = magmaan::api::model_from_lavaan("f =~ x1 + x2 + x3 + x4", options);
  REQUIRE(m.has_value());
  REQUIRE(m->identification() != nullptr);
  CHECK(identification_checks == 1);
  CHECK(m->identification()->status == IdentificationStatus::Unidentified);
  CHECK_FALSE(m->identification()->directions_at_estimate);
  const auto shared = m->matrix_rep().identification;
  const auto copied = *m;
  CHECK(copied.matrix_rep().identification == shared);
  for (const double factor : {1.0, 2.0}) {
    const auto ss = sample(factor * one_factor_cov({.8, .7, .6, .5}));
    auto fit = magmaan::test::fit(m->structure(), m->matrix_rep(), ss);
    REQUIRE(fit.has_value());
    CHECK(fit->diagnostics.identification.status == IdentificationStatus::Unidentified);
    CHECK(fit->diagnostics.identification.null_directions.isApprox(shared->null_directions));
    CHECK(identification_checks == 1);
  }
  auto ev = magmaan::model::ModelEvaluator::build(m->structure(), m->matrix_rep());
  auto con = magmaan::estimate::build_eq_constraints(m->structure());
  REQUIRE(ev.has_value());
  REQUIRE(con.has_value());
  const auto extra = fr::check_structural_identification(m->structure(), *ev, *con, true);
  CHECK(extra.status == IdentificationStatus::Unchecked);
  CHECK(extra.reason == IdentificationReason::NonlinearConstraints);
}
TEST_CASE("Identification: prepared ordinal structures reuse one schema-only rank check") {
  auto X = one_factor_data(400, {.8, .7, .6, .5}, 20261008);
  for (Eigen::Index i = 0; i < X.rows(); ++i)
    for (Eigen::Index j = 0; j < X.cols(); ++j)
      X(i, j) = 1.0 + (X(i, j) > 0.8 + 0.5 * static_cast<double>(j)) +
          (X(i, j) > 1.4 + 0.5 * static_cast<double>(j));
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto m = model("f =~ x1 + x2 + x3 + x4\nx1 | t1 + t2\nx2 | t1 + t2\n"
                 "x3 | t1 + t2\nx4 | t1 + t2");
  using magmaan::estimate::OrdinalParameterization;
  const auto parameterization = OrdinalParameterization::Delta;
  auto prepared = magmaan::estimate::prepare_ordinal_partable(m.pt, *stats, parameterization);
  REQUIRE(prepared.has_value());
  IdentificationObserver observer;
  auto report = fr::check_structural_identification(m.pt, m.rep, *stats, parameterization);
  REQUIRE(report.has_value());
  REQUIRE(report->status == IdentificationStatus::Identified);
  CHECK(identification_checks == 1);
  m.rep.identification = std::make_shared<const IdentificationReport>(*report);
  m.rep.identification_n_free = m.pt.n_free();
  auto repeated = fr::check_structural_identification(m.pt, m.rep, *stats, parameterization);
  REQUIRE(repeated.has_value());
  CHECK(identification_checks == 1);
  for (int i = 0; i < 2; ++i) {
    auto fit = magmaan::test::fit_ordinal_bounded(m.pt, m.rep, *stats, {},
        magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::Backend::NloptLbfgs, {}, parameterization);
    REQUIRE(fit.has_value());
    CHECK(fit->diagnostics.identification.status == IdentificationStatus::Identified);
    CHECK(identification_checks == 1);
  }
  auto fixed = model("f =~ x1 + x2 + x3 + x4\nx1 | 0*t1 + t2\nx2 | t1 + t2\n"
                     "x3 | t1 + t2\nx4 | t1 + t2");
  // Four declared categories need three thresholds, while this explicit
  // fixed-threshold specification supplies only two (the R regression case).
  auto four = one_factor_data(400, {.8, .7, .6, .5}, 20261008);
  for (Eigen::Index i = 0; i < four.rows(); ++i)
    for (Eigen::Index j = 0; j < four.cols(); ++j) {
      const double y = four(i, j);
      four(i, j) = 1.0 + (y > 0.0) + (y > 0.5) + (y > 1.0);
    }
  auto four_stats = magmaan::data::ordinal_stats_from_integer_data({four});
  REQUIRE(four_stats.has_value());
  auto ok = magmaan::estimate::prepare_ordinal_partable(
      fixed.pt, *four_stats, parameterization, nullptr, &fixed.names.row_user);
  REQUIRE(ok.has_value());
  auto fixed_rep = magmaan::model::build_matrix_rep(fixed.pt, &fixed.names);
  REQUIRE(fixed_rep.has_value());
  auto unavailable = fr::check_structural_identification(fixed.pt, *fixed_rep, *four_stats, parameterization);
  REQUIRE(unavailable.has_value());
  CHECK(unavailable->status == IdentificationStatus::Unchecked);
  CHECK(unavailable->reason == IdentificationReason::UnsupportedModel);
  CHECK(identification_checks == 2);
}
#endif
