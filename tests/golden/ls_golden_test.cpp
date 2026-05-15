#include <doctest/doctest.h>

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
#include "magmaan/gls/gls.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/nt/infer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

const std::vector<std::string> kLsFixtures = {
    "0001_three_factor_hs",
    "0002_multigroup_3f_school",
    "0003_labeled_equality",
    "0004_two_factor_meanstructure",
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
  magmaan::spec::LavaanifyOptions opts;
  opts.n_groups = exp["n_groups"].get<std::int32_t>();
  opts.meanstructure = exp["meanstructure"].get<bool>();
  auto pt = magmaan::spec::lavaanify(*fp, opts);
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

double total_n(const magmaan::data::SampleStats& samp) noexcept {
  double out = 0.0;
  for (auto n : samp.n_obs) out += static_cast<double>(n);
  return out;
}

bool check_estimate(const std::string& id,
                    const std::string& estimator,
                    const nlohmann::json& fit,
                    const magmaan::data::SampleStats& samp,
                    const magmaan::fit::Estimates& est,
                    const magmaan::spec::LatentStructure& pt,
                    const magmaan::model::MatrixRep& rep,
                    std::vector<std::string>& failures) {
  const double d_theta = max_theta_diff(est.theta, fit["theta_hat"]);
  // GLS currently uses the explicit 0.5 * tr(S^-1 D S^-1 D) convention in
  // `GLS::value()`, while lavaan's reported `fmin` is half of that value.
  // Keep `fmin` as an objective-scale regression check; lavaan's public LS
  // chi-square reporting is asserted separately below.
  const double fmin_on_lavaan_scale =
      estimator == "GLS" ? 0.5 * est.fmin : est.fmin;
  const double d_fmin = std::abs(fmin_on_lavaan_scale -
                                 fit["fmin"].get<double>());

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
                  fmin_on_lavaan_scale,
                  fit["fmin"].get<double>(), est.iterations);
    failures.push_back(buf);
    return false;
  }

  auto df_or = magmaan::nt::infer::df_stat(pt, samp);
  if (!df_or.has_value()) {
    failures.push_back(id + "/" + estimator + ": df_stat — " +
                       df_or.error().detail);
    return false;
  }
  if (*df_or != fit["df"].get<int>()) {
    failures.push_back(id + "/" + estimator + ": df mismatch");
    return false;
  }
  if (est.theta.size() != fit["npar"].get<int>()) {
    failures.push_back(id + "/" + estimator + ": npar mismatch");
    return false;
  }

  double chisq = std::numeric_limits<double>::quiet_NaN();
  if (estimator == "ULS") {
    auto br_or = magmaan::nt::infer::browne_residual_nt(pt, rep, samp, est);
    if (!br_or.has_value()) {
      failures.push_back(id + "/" + estimator + ": browne_residual_nt — " +
                         br_or.error().detail);
      return false;
    }
    chisq = *br_or;
  } else {
    chisq = 2.0 * total_n(samp) * fmin_on_lavaan_scale;
  }

  const double d_chisq = std::abs(chisq - fit["chisq"].get<double>());
  const double chisq_tol =
      id == "0002_multigroup_3f_school" ? 2e-1 :
      id == "0004_two_factor_meanstructure" ? 7e-2 :
      5e-2;
  if (d_chisq > chisq_tol) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "%s/%s: |chisq-lavaan|=%.3e (ours %.9g, lavaan %.9g)",
                  id.c_str(), estimator.c_str(), d_chisq, chisq,
                  fit["chisq"].get<double>());
    failures.push_back(buf);
    return false;
  }
  return true;
}

}  // namespace

TEST_CASE("continuous LS fixtures: ULS/GLS/WLS bounded fits match lavaan") {
  const std::string dir = magmaan::test::fixtures_dir() + "/ls";
  magmaan::optim::LbfgsBOptimizer opt(magmaan::optim::LbfgsBOptions{
      .max_iter = 10000, .ftol = 1e-14, .gtol = 1e-8});

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

      magmaan::fit_expected<magmaan::fit::Estimates> est_or =
          std::unexpected(magmaan::FitError{
              magmaan::FitError::Kind::NumericIssue, "not run", 0, 0.0});
      if (estimator == "ULS") {
        est_or = magmaan::estimate::fit_bounded(
            handles->pt, handles->rep, samp, magmaan::estimate::Bounds{},
            magmaan::gls::ULS{}, opt);
      } else if (estimator == "GLS") {
        est_or = magmaan::estimate::fit_bounded(
            handles->pt, handles->rep, samp, magmaan::estimate::Bounds{},
            magmaan::gls::GLS{}, opt);
      } else {
        est_or = magmaan::estimate::fit_bounded(
            handles->pt, handles->rep, samp, magmaan::estimate::Bounds{},
            magmaan::gls::WLS(matrices_from_blocks(fit["WLS.V"])), opt);
      }

      if (!est_or.has_value()) {
        failures.push_back(id + "/" + estimator + ": fit_bounded — " +
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
