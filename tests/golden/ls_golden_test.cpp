#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

const std::vector<std::string> kLsFixtures = {
    "0001_three_factor_hs",
    "0002_multigroup_3f_school",
    "0003_labeled_equality",
    "0004_two_factor_meanstructure",
    "0005_obs_exo_cfa_fixedx",
    "0006_three_factor_meanstructure_hs",
    "0007_general_linear_hs",
};

Eigen::MatrixXd matrix_from_json(const nlohmann::json& j) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr > 0 ? static_cast<Eigen::Index>(j[0].size()) : 0;
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r) {
    for (Eigen::Index c = 0; c < nc; ++c) {
      out(r, c) = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                      .get<double>();
    }
  }
  return out;
}

Eigen::VectorXd vector_from_json(const nlohmann::json& j) {
  if (j.is_number()) {
    Eigen::VectorXd out(1);
    out(0) = j.get<double>();
    return out;
  }
  Eigen::VectorXd out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].get<double>();
  }
  return out;
}

std::vector<Eigen::MatrixXd> matrices_from_blocks(const nlohmann::json& blocks) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(blocks.size());
  for (const auto& b : blocks) out.push_back(matrix_from_json(b["matrix"]));
  return out;
}

// The moment-quadratic weight for a continuous LS estimator: empty ⇒ ULS;
// `WLS.V` from the fixture ⇒ WLS; the normal-theory weight ⇒ GLS.
magmaan::estimate::gmm::Weight ls_estimator_weight(
    const std::string& estimator, const nlohmann::json& fit,
    const magmaan::spec::LatentStructure& pt,
    const magmaan::model::MatrixRep& rep,
    const magmaan::data::SampleStats& samp,
    const magmaan::estimate::Estimates& est) {
  if (estimator == "ULS") return {};
  if (estimator == "WLS") {
    return magmaan::estimate::gmm::Weight(matrices_from_blocks(fit["WLS.V"]));
  }
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  REQUIRE(ev.has_value());
  auto w = magmaan::estimate::gmm::normal_theory_weight(*ev, samp, est.theta);
  REQUIRE(w.has_value());
  return *w;
}

struct LsHandles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

std::optional<LsHandles> handles_from_fixture(const std::string& id,
                                              const nlohmann::json& exp,
                                              std::vector<std::string>& failures) {
  auto fp = magmaan::parse::Parser::parse(exp["input"].get<std::string>());
  if (!fp.has_value()) {
    failures.push_back(id + ": parse — " + fp.error().detail);
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.n_groups = exp["n_groups"].get<std::int32_t>();
  opts.meanstructure = exp["meanstructure"].get<bool>();
  auto pt = magmaan::spec::build(*fp, opts);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify — " + pt.error().detail);
    return std::nullopt;
  }
  auto mr = magmaan::model::build_matrix_rep(*pt);
  if (!mr.has_value()) {
    failures.push_back(id + ": matrix_rep — " + mr.error().detail);
    return std::nullopt;
  }
  return LsHandles{std::move(*pt), std::move(*mr)};
}

magmaan::data::SampleStats sample_stats_from_fit(const nlohmann::json& fit) {
  magmaan::data::SampleStats samp;
  for (const auto& b : fit["sample_cov"]) {
    samp.S.push_back(matrix_from_json(b["matrix"]));
  }
  const auto& nobs = fit["n_obs_per_block"];
  if (nobs.is_array()) {
    for (const auto& n : nobs) samp.n_obs.push_back(n.get<std::int64_t>());
  } else {
    samp.n_obs.push_back(nobs.get<std::int64_t>());
  }
  if (fit.contains("sample_mean") && !fit["sample_mean"].is_null()) {
    for (const auto& b : fit["sample_mean"]) {
      samp.mean.push_back(vector_from_json(b["vector"]));
    }
  }
  return samp;
}

double max_theta_diff(const Eigen::VectorXd& theta, const nlohmann::json& exp) {
  if (static_cast<std::size_t>(theta.size()) != exp.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double out = 0.0;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    const double d = std::abs(theta(k) -
        exp[static_cast<std::size_t>(k)].get<double>());
    if (d > out) out = d;
  }
  return out;
}

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double out = 0.0;
  for (Eigen::Index k = 0; k < a.size(); ++k) {
    out = std::max(out, std::abs(a(k) - b(k)));
  }
  return out;
}

double total_n(const magmaan::data::SampleStats& samp) noexcept {
  double out = 0.0;
  for (auto n : samp.n_obs) out += static_cast<double>(n);
  return out;
}

magmaan::fit_expected<magmaan::estimate::Estimates>
fit_ls_estimator(const std::string& estimator,
                 const nlohmann::json& fit,
                 const LsHandles& handles,
                 const magmaan::data::SampleStats& samp,
                 magmaan::estimate::Backend backend,
                 magmaan::optim::OptimOptions opt) {
  if (estimator == "ULS") {
    return magmaan::test::fit_gmm(handles.pt, handles.rep, samp, {},
                                  magmaan::estimate::Bounds{}, backend, opt);
  }
  if (estimator == "GLS") {
    return magmaan::test::fit_gls(handles.pt, handles.rep, samp,
                                  magmaan::estimate::Bounds{}, backend, opt);
  }
  return magmaan::test::fit_gmm(
      handles.pt, handles.rep, samp,
      magmaan::estimate::gmm::Weight(matrices_from_blocks(fit["WLS.V"])),
      magmaan::estimate::Bounds{}, backend, opt);
}

magmaan::fit_expected<magmaan::estimate::Estimates>
fit_snlls_estimator(const std::string& estimator,
                    const nlohmann::json& fit,
                    const LsHandles& handles,
                    const magmaan::data::SampleStats& samp,
                    magmaan::estimate::Backend backend,
                    magmaan::optim::OptimOptions opt) {
  auto x0 = magmaan::estimate::simple_start_values(
      handles.pt, handles.rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  if (estimator == "GLS") {
    return magmaan::estimate::fit_snlls_gls(
        handles.pt, handles.rep, samp, *x0, backend, opt);
  }
  magmaan::estimate::gmm::Weight weight;
  if (estimator == "WLS") {
    weight = magmaan::estimate::gmm::Weight(
        matrices_from_blocks(fit["WLS.V"]));
  }
  return magmaan::estimate::fit_snlls(
      handles.pt, handles.rep, samp, *x0, std::move(weight), backend, opt);
}

bool check_estimate(const std::string& id,
                    const std::string& estimator,
                    const nlohmann::json& fit,
                    const magmaan::data::SampleStats& samp,
                    const magmaan::estimate::Estimates& est,
                    const magmaan::spec::LatentStructure& pt,
                    const magmaan::model::MatrixRep& rep,
                    std::vector<std::string>& failures) {
  const double d_theta = max_theta_diff(est.theta, fit["theta_hat"]);
  // `fmin` is on lavaan's objective scale for every estimator — the GLS
  // moment weight carries no extra factor (χ² = 2N·fmin uniformly). Kept as
  // an objective-scale regression check.
  const double d_fmin = std::abs(est.fmin - fit["fmin"].get<double>());

  const double theta_tol =
      id == "0002_multigroup_3f_school" ? 2e-4 :
      id == "0004_two_factor_meanstructure" ? 1e-4 :
      5e-5;
  const double fmin_tol = 2e-3;

  if (d_theta > theta_tol || d_fmin > fmin_tol) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%s/%s: max|theta-lavaan|=%.3e, |fmin-lavaan|=%.3e "
                  "(ours %.9g, lavaan %.9g, iters=%d)",
                  id.c_str(), estimator.c_str(), d_theta, d_fmin,
                  est.fmin,
                  fit["fmin"].get<double>(), est.iterations);
    failures.push_back(buf);
    return false;
  }

  auto df_or = magmaan::inference::df_stat(pt, samp);
  if (!df_or.has_value()) {
    failures.push_back(id + "/" + estimator + ": df_stat — " +
                       df_or.error().detail);
    return false;
  }
  if (*df_or != fit["df"].get<int>()) {
    failures.push_back(id + "/" + estimator + ": df mismatch");
    return false;
  }
  // lavaan npar is the post-equality independent parameter count; magmaan's
  // equivalent is EqConstraints::n_alpha (equal to the θ size when there are
  // no equality constraints).
  auto con_or = magmaan::estimate::build_eq_constraints(pt);
  const int magmaan_npar = con_or.has_value()
      ? static_cast<int>(con_or->n_alpha)
      : static_cast<int>(est.theta.size());
  if (magmaan_npar != fit["npar"].get<int>()) {
    failures.push_back(id + "/" + estimator + ": npar mismatch (" +
                       std::to_string(magmaan_npar) + " vs lavaan " +
                       std::to_string(fit["npar"].get<int>()) + ")");
    return false;
  }

  double chisq = std::numeric_limits<double>::quiet_NaN();
  {
    auto chisq_or = magmaan::estimate::continuous_ls_chisq(
        samp, pt, rep, est,
        ls_estimator_weight(estimator, fit, pt, rep, samp, est));
    if (!chisq_or.has_value()) {
      failures.push_back(id + "/" + estimator + ": continuous_ls_chisq — " +
                         chisq_or.error().detail);
      return false;
    }
    chisq = *chisq_or;
  }

  // magmaan keeps the N·F statistic multiplier; lavaan reports (N−G)·F
  // (Wishart/unbiased). They relate by EXACTLY (N−G)/N — ULS already carries
  // the (N−G) factor (browne_residual_nt's n_used) and matches lavaan directly,
  // while GLS/WLS need the rescale. Pin the exact relation tightly rather than
  // hiding the multiplier gap behind a slack tolerance; the residual is fixture
  // storage precision, not estimator slack. (G = covariance-block count.)
  const double n_total = total_n(samp);
  const double n_groups = static_cast<double>(samp.S.size());
  const double conv =
      (estimator == "ULS") ? 1.0 : (n_total - n_groups) / n_total;
  const double lavaan_pred = chisq * conv;
  const double d_chisq = std::abs(lavaan_pred - fit["chisq"].get<double>());
  const double chisq_tol = 5e-3;
  if (d_chisq > chisq_tol) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "%s/%s: |chisq·(N−G)/N − lavaan|=%.3e "
                  "(magmaan N·F %.9g, lavaan %.9g)",
                  id.c_str(), estimator.c_str(), d_chisq, chisq,
                  fit["chisq"].get<double>());
    failures.push_back(buf);
    return false;
  }
  return true;
}

bool check_uls_robust(const std::string& id,
                      const nlohmann::json& fit,
                      const magmaan::data::SampleStats& samp,
                      const magmaan::estimate::Estimates& est,
                      const magmaan::spec::LatentStructure& pt,
                      const magmaan::model::MatrixRep& rep,
                      std::vector<std::string>& failures) {
  if (!fit.contains("robust") || fit["robust"].is_null()) return true;
  const auto& robust = fit["robust"];
  auto rob_or = magmaan::estimate::robust_continuous_ls(
      pt, rep, samp, est, magmaan::estimate::gmm::Weight{},
      matrices_from_blocks(robust["gamma"]));
  if (!rob_or.has_value()) {
    failures.push_back(id + "/ULS robust: robust_continuous_ls — " +
                       rob_or.error().detail);
    return false;
  }
  const Eigen::VectorXd lavaan_se = vector_from_json(robust["se"]);
  const Eigen::VectorXd lavaan_ev = vector_from_json(robust["eigvals"]);
  const double d_se = max_abs_diff(rob_or->se, lavaan_se);
  const double d_ev = max_abs_diff(rob_or->eigvals, lavaan_ev);
  const double d_standard = std::abs(rob_or->chisq_standard -
                                     robust["chisq_standard"].get<double>());
  const double d_sb = std::abs(rob_or->satorra_bentler.chi2_scaled -
                               robust["satorra_bentler"]["chisq"].get<double>());
  const double d_sb_scale = std::abs(rob_or->satorra_bentler.scale_c -
                                     robust["satorra_bentler"]["scale"].get<double>());
  const double d_mv = std::abs(rob_or->mean_var_adjusted.chi2_adj -
                               robust["mean_var_adjusted"]["chisq"].get<double>());
  const double d_mv_df = std::abs(rob_or->mean_var_adjusted.df_adj -
                                  robust["mean_var_adjusted"]["df_adj"].get<double>());
  const double d_ss = std::abs(rob_or->scaled_shifted.chi2_adj -
                               robust["scaled_shifted"]["chisq"].get<double>());
  const double d_ss_scale = std::abs(
      rob_or->scaled_shifted.scale_a -
      1.0 / robust["scaled_shifted"]["scale"].get<double>());
  const double d_ss_shift = std::abs(rob_or->scaled_shifted.shift_b -
                                     robust["scaled_shifted"]["shift"].get<double>());

  if (rob_or->df != robust["df"].get<int>() ||
      rob_or->satorra_bentler.df != robust["satorra_bentler"]["df"].get<int>() ||
      rob_or->scaled_shifted.df != robust["scaled_shifted"]["df"].get<int>() ||
      d_se > 2e-3 || d_ev > 1e-5 || d_standard > 7e-1 ||
      d_sb > 9e-1 || d_sb_scale > 5e-4 || d_mv > 5e-1 ||
      d_mv_df > 5e-3 || d_ss > 7e-1 || d_ss_scale > 5e-4 ||
      d_ss_shift > 2e-2) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s/ULS robust: diffs se=%.3e eig=%.3e standard=%.3e "
                  "sb=%.3e sb_scale=%.3e mv=%.3e mv_df=%.3e "
                  "ss=%.3e ss_scale=%.3e ss_shift=%.3e",
                  id.c_str(), d_se, d_ev, d_standard, d_sb, d_sb_scale,
                  d_mv, d_mv_df, d_ss, d_ss_scale, d_ss_shift);
    failures.push_back(buf);
    return false;
  }
  return true;
}

}  // namespace

TEST_CASE("continuous LS fixtures: ULS/GLS/WLS bounded fits match lavaan") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ls";
  const magmaan::optim::OptimOptions opt{
      .max_iter = 10000, .ftol = 1e-14, .gtol = 1e-8};

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kLsFixtures) {
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) {
      failures.push_back(id + ": missing fixture");
      continue;
    }
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(id + ": invalid JSON");
      continue;
    }
    auto handles = handles_from_fixture(id, exp, failures);
    if (!handles.has_value()) continue;

    for (const std::string estimator : {"ULS", "GLS", "WLS"}) {
      ++total;
      const auto& fit = exp["fits"][estimator];
      auto samp = sample_stats_from_fit(fit);

      auto est_or = fit_ls_estimator(
          estimator, fit, *handles, samp, magmaan::estimate::Backend::NloptLbfgs,
          opt);

      if (!est_or.has_value()) {
        failures.push_back(id + "/" + estimator + ": fit — " +
                           est_or.error().detail);
        continue;
      }
      if (check_estimate(id, estimator, fit, samp, *est_or, handles->pt,
                         handles->rep, failures)) {
        ++passed;
      }
    }
  }

  MESSAGE("continuous LS fixtures: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}

TEST_CASE("continuous mean-structure LS fixtures: SNLLS agrees with full LS") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ls";
  const magmaan::optim::OptimOptions opt{
      .max_iter = 10000, .ftol = 1e-14, .gtol = 1e-8};

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : {"0004_two_factor_meanstructure",
                         "0006_three_factor_meanstructure_hs"}) {
    auto raw = magmaan::test::read_fixture(dir + "/" + id + ".fit.json");
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    auto handles = handles_from_fixture(id, exp, failures);
    if (!handles.has_value()) continue;

    for (const std::string estimator : {"ULS", "GLS", "WLS"}) {
      ++total;
      const auto& fit = exp["fits"][estimator];
      auto samp = sample_stats_from_fit(fit);
      auto full = fit_ls_estimator(estimator, fit, *handles, samp,
                                   magmaan::estimate::Backend::NloptLbfgs, opt);
      auto prof = fit_snlls_estimator(estimator, fit, *handles, samp,
                                      magmaan::estimate::Backend::NloptLbfgs, opt);
      if (!full.has_value() || !prof.has_value()) {
        failures.push_back(std::string(id) + "/" + estimator +
                           ": full/SNLLS fit failed");
        continue;
      }
      const double d_f = std::abs(full->fmin - prof->fmin);
      if (d_f > 1e-7) {
        failures.push_back(std::string(id) + "/" + estimator +
                           ": |full fmin - SNLLS fmin|=" +
                           std::to_string(d_f));
        continue;
      }
      ++passed;
    }
  }

  MESSAGE("continuous mean-structure SNLLS parity: " << passed << " / "
                                                     << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}

#ifdef MAGMAAN_WITH_CERES
TEST_CASE("continuous mean-structure LS fixtures: Ceres and NLopt L-BFGS agree") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ls";
  const magmaan::optim::OptimOptions nlopt{
      .max_iter = 10000, .ftol = 1e-14, .gtol = 1e-8};
  const magmaan::optim::OptimOptions ceres{
      .max_iter = 1000, .ftol = 1e-12, .gtol = 1e-8};

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : {"0004_two_factor_meanstructure",
                         "0006_three_factor_meanstructure_hs"}) {
    auto raw = magmaan::test::read_fixture(dir + "/" + id + ".fit.json");
    REQUIRE(raw.has_value());
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    REQUIRE_FALSE(exp.is_discarded());
    auto handles = handles_from_fixture(id, exp, failures);
    if (!handles.has_value()) continue;

    for (const std::string estimator : {"ULS", "GLS", "WLS"}) {
      ++total;
      const auto& fit = exp["fits"][estimator];
      auto samp = sample_stats_from_fit(fit);
      auto a = fit_ls_estimator(estimator, fit, *handles, samp,
                                magmaan::estimate::Backend::NloptLbfgs, nlopt);
      auto b = fit_ls_estimator(estimator, fit, *handles, samp,
                                magmaan::estimate::Backend::Ceres, ceres);
      if (!a.has_value() || !b.has_value()) {
        failures.push_back(std::string(id) + "/" + estimator +
                           ": NLopt L-BFGS/Ceres fit failed");
        continue;
      }
      const double d_f = std::abs(a->fmin - b->fmin);
      if (d_f > 1e-7) {
        failures.push_back(std::string(id) + "/" + estimator +
                           ": |NLopt L-BFGS fmin - Ceres fmin|=" +
                           std::to_string(d_f));
        continue;
      }
      ++passed;
    }
  }

  MESSAGE("continuous mean-structure Ceres parity: " << passed << " / "
                                                     << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
#endif

TEST_CASE("continuous LS robust ULS fixtures match lavaan robust.sem") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ls";
  const magmaan::optim::OptimOptions opt{
      .max_iter = 10000, .ftol = 1e-14, .gtol = 1e-8};

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;

  for (const auto& id : kLsFixtures) {
    // The fixed.x LS robust target follows lavaan's conditional exogenous
    // bookkeeping, which is still tracked separately from this adapter slice.
    if (id == "0005_obs_exo_cfa_fixedx") continue;
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) {
      failures.push_back(id + ": missing fixture");
      continue;
    }
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(id + ": invalid JSON");
      continue;
    }
    auto handles = handles_from_fixture(id, exp, failures);
    if (!handles.has_value()) continue;

    const auto& fit = exp["fits"]["ULS"];
    if (!fit.contains("robust") || fit["robust"].is_null()) continue;
    ++total;
    auto samp = sample_stats_from_fit(fit);
    auto est_or = magmaan::test::fit_gmm(
        handles->pt, handles->rep, samp, {}, magmaan::estimate::Bounds{},
        magmaan::estimate::Backend::NloptLbfgs, opt);
    if (!est_or.has_value()) {
      failures.push_back(id + "/ULS robust: fit — " +
                         est_or.error().detail);
      continue;
    }
    if (check_uls_robust(id, fit, samp, *est_or, handles->pt, handles->rep,
                         failures)) {
      ++passed;
    }
  }

  MESSAGE("continuous LS robust ULS fixtures: " << passed << " / " << total
                                                << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
