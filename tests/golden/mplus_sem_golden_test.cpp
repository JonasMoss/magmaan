#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/spec/build.hpp"

namespace {

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
  Eigen::VectorXd out(static_cast<Eigen::Index>(j.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[static_cast<std::size_t>(i)].get<double>();
  }
  return out;
}

std::vector<Eigen::MatrixXd> matrices_from_blocks(const nlohmann::json& j) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(j.size());
  for (const auto& b : j) out.push_back(matrix_from_json(b["matrix"]));
  return out;
}

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  if (a.size() == 0) return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  if (a.size() == 0) return 0.0;
  return (a - b).cwiseAbs().maxCoeff();
}

bool close(double a, double b, double tol) {
  return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

magmaan::data::SampleStats sample_stats_from_fit(const nlohmann::json& fit) {
  magmaan::data::SampleStats samp;
  samp.S = {matrix_from_json(fit["sample_cov"])};
  if (fit.contains("sample_mean") && !fit["sample_mean"].is_null()) {
    samp.mean = {vector_from_json(fit["sample_mean"])};
  }
  samp.n_obs = {fit["n_obs"].get<std::int64_t>()};
  return samp;
}

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

std::optional<Handles> handles_from_case(const nlohmann::json& c,
                                         std::vector<std::string>& failures) {
  const std::string id = c["id"].get<std::string>();
  auto fp = magmaan::parse::Parser::parse(c["model"].get<std::string>());
  if (!fp.has_value()) {
    failures.push_back(id + ": parse");
    return std::nullopt;
  }
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = c.value("meanstructure", false);
  opts.fixed_x = c.value("fixed_x", true);
  if (c.value("lavaan_function", std::string{}) == "growth") {
    opts.meanstructure = true;
    opts.int_ov_free = false;
    opts.int_lv_free = true;
    opts.fixed_x = false;
  }
  auto pt = magmaan::spec::build(*fp, opts);
  if (!pt.has_value()) {
    failures.push_back(id + ": lavaanify - " + pt.error().detail);
    return std::nullopt;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) {
    failures.push_back(id + ": matrix_rep - " + rep.error().detail);
    return std::nullopt;
  }
  return Handles{std::move(*pt), std::move(*rep)};
}

magmaan::optim::LbfgsOptions mplus_opts() {
  return magmaan::optim::LbfgsOptions{
      .max_iter = 7000,
      .ftol = 1e-13,
      .gtol = 1e-8,
      .history = 10,
  };
}

void check_implied(const std::string& label,
                   const magmaan::spec::LatentStructure& pt,
                   const magmaan::model::MatrixRep& rep,
                   const magmaan::estimate::Estimates& est,
                   const nlohmann::json& fit,
                   std::vector<std::string>& failures) {
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    failures.push_back(label + ": ModelEvaluator - " + ev.error().detail);
    return;
  }
  auto im = ev->sigma(est.theta);
  if (!im.has_value()) {
    failures.push_back(label + ": implied moments - " + im.error().detail);
    return;
  }
  const double d_sigma = max_abs_diff(im->sigma[0], matrix_from_json(fit["sigma"]));
  if (d_sigma > 5e-4) {
    failures.push_back(label + ": max|Sigma - lavaan| = " +
                       std::to_string(d_sigma));
  }
  if (!im->mu.empty() && fit.contains("mu") && !fit["mu"].is_null()) {
    const double d_mu = max_abs_diff(im->mu[0], vector_from_json(fit["mu"]));
    if (d_mu > 5e-4) {
      failures.push_back(label + ": max|mu - lavaan| = " +
                         std::to_string(d_mu));
    }
  }
}

void check_common_fit(const std::string& label,
                      const magmaan::spec::LatentStructure& pt,
                      const magmaan::model::MatrixRep& rep,
                      const magmaan::estimate::Estimates& est,
                      const magmaan::data::SampleStats& samp,
                      const nlohmann::json& fit,
                      double theta_tol,
                      std::vector<std::string>& failures) {
  if (!est.theta.allFinite() || !std::isfinite(est.fmin)) {
    failures.push_back(label + ": non-finite fit result");
    return;
  }
  const Eigen::VectorXd lavaan_theta = vector_from_json(fit["theta_hat"]);
  const double d_theta = max_abs_diff(est.theta, lavaan_theta);
  if (d_theta > theta_tol) {
    failures.push_back(label + ": max|theta - lavaan| = " +
                       std::to_string(d_theta));
  }
  auto df_or = magmaan::inference::df_stat(pt, samp);
  if (!df_or.has_value()) {
    failures.push_back(label + ": df_stat - " + df_or.error().detail);
  } else if (*df_or != fit["df"].get<int>()) {
    failures.push_back(label + ": df = " + std::to_string(*df_or) +
                       ", lavaan = " + std::to_string(fit["df"].get<int>()));
  }
  check_implied(label, pt, rep, est, fit, failures);
}

}  // namespace

TEST_CASE("Mplus SEM corpus fixtures are well formed") {
  const std::string dir = magmaan::test::fixtures_dir() + "/mplus_sem";
  for (const std::string file : {"manifest.json", "continuous_reference.json",
                                "ordinal_reference.json",
                                "mixed_reference.json"}) {
    auto raw = magmaan::test::read_fixture(dir + "/" + file);
    REQUIRE_MESSAGE(raw.has_value(), "missing " << file);
    auto j = nlohmann::json::parse(*raw, nullptr, false);
    CHECK_MESSAGE(!j.is_discarded(), file << " is invalid JSON");
    CHECK_MESSAGE(j.contains("_meta"), file << " lacks _meta");
  }
}

TEST_CASE("Mplus SEM continuous goldens match lavaan") {
  const std::string path =
      magmaan::test::fixtures_dir() + "/mplus_sem/continuous_reference.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("cases"));

  int total = 0;
  int passed = 0;
  std::vector<std::string> failures;
  const auto opt = mplus_opts();

  for (const auto& c : j["cases"]) {
    const std::string id = c["id"].get<std::string>();
    CAPTURE(id);
    auto handles = handles_from_case(c, failures);
    if (!handles.has_value()) continue;

    for (const auto& item : c["fits"].items()) {
      const std::string estimator = item.key();
      const auto& fit = item.value();
      ++total;
      const std::string label = id + "/" + estimator;
      auto samp = sample_stats_from_fit(fit);

      if (estimator == "ML") {
        auto est_or = magmaan::test::fit(handles->pt, handles->rep, samp,
                                         magmaan::estimate::Bounds{},
                                         magmaan::estimate::Backend::Lbfgs, opt);
        if (!est_or.has_value()) {
          failures.push_back(label + ": fit - " + est_or.error().detail);
          continue;
        }
        check_common_fit(label, handles->pt, handles->rep, *est_or, samp, fit,
                         1e-3, failures);
        const double lavaan_f = 2.0 * fit["fmin"].get<double>();
        if (!close(est_or->fmin, lavaan_f, 2e-4)) {
          failures.push_back(label + ": fmin = " + std::to_string(est_or->fmin) +
                             ", lavaan 2*fmin = " + std::to_string(lavaan_f));
          continue;
        }
        const double chisq =
            static_cast<double>(samp.n_obs[0]) * est_or->fmin;
        if (!close(chisq, fit["chisq"].get<double>(), 5e-3)) {
          failures.push_back(label + ": chisq = " + std::to_string(chisq) +
                             ", lavaan = " +
                             std::to_string(fit["chisq"].get<double>()));
          continue;
        }
      } else {
        magmaan::estimate::gmm::Weight weight;
        magmaan::fit_expected<magmaan::estimate::Estimates> est_or =
            std::unexpected(magmaan::FitError{
                magmaan::FitError::Kind::NumericIssue, "not run", 0, 0.0});
        if (estimator == "ULS") {
          est_or = magmaan::test::fit_gmm(
              handles->pt, handles->rep, samp, weight,
              magmaan::estimate::Bounds{}, magmaan::estimate::Backend::Lbfgs,
              opt);
        } else if (estimator == "GLS") {
          est_or = magmaan::test::fit_gls(
              handles->pt, handles->rep, samp, magmaan::estimate::Bounds{},
              magmaan::estimate::Backend::Lbfgs, opt);
        } else if (estimator == "WLS") {
          weight = magmaan::estimate::gmm::Weight(matrices_from_blocks(fit["WLS.V"]));
          est_or = magmaan::test::fit_gmm(
              handles->pt, handles->rep, samp, weight,
              magmaan::estimate::Bounds{}, magmaan::estimate::Backend::Lbfgs,
              opt);
        } else {
          failures.push_back(label + ": unknown estimator");
          continue;
        }
        if (!est_or.has_value()) {
          failures.push_back(label + ": fit - " + est_or.error().detail);
          continue;
        }
        check_common_fit(label, handles->pt, handles->rep, *est_or, samp, fit,
                         3e-3, failures);

        if (estimator == "GLS") {
          auto ev = magmaan::model::ModelEvaluator::build(handles->pt, handles->rep);
          if (!ev.has_value()) {
            failures.push_back(label + ": ModelEvaluator - " + ev.error().detail);
            continue;
          }
          auto w = magmaan::estimate::gmm::normal_theory_weight(*ev, samp,
                                                                est_or->theta);
          if (!w.has_value()) {
            failures.push_back(label + ": normal_theory_weight - " +
                               w.error().detail);
            continue;
          }
          weight = std::move(*w);
        }
        auto chisq_or = magmaan::estimate::continuous_ls_chisq(
            samp, handles->pt, handles->rep, *est_or, weight);
        if (!chisq_or.has_value()) {
          failures.push_back(label + ": continuous_ls_chisq - " +
                             chisq_or.error().detail);
          continue;
        }
        if (!close(*chisq_or, fit["chisq"].get<double>(), 5e-2)) {
          failures.push_back(label + ": chisq = " +
                             std::to_string(*chisq_or) +
                             ", lavaan = " +
                             std::to_string(fit["chisq"].get<double>()));
          continue;
        }
      }
      ++passed;
    }
  }

  MESSAGE("Mplus SEM continuous fits: " << passed << " / " << total << " pass");
  for (const auto& f : failures) MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
