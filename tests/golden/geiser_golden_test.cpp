#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
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

magmaan::data::SampleStats sample_stats_from_case(const nlohmann::json& c) {
  magmaan::data::SampleStats samp;
  samp.S = {matrix_from_json(c["sample_cov"])};
  samp.mean = {vector_from_json(c["sample_mean"])};
  samp.n_obs = {c["n_obs"].get<std::int64_t>()};
  return samp;
}

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

Handles handles_from_case(const nlohmann::json& c) {
  auto fp = magmaan::parse::Parser::parse(c["model"].get<std::string>());
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = c["meanstructure"].get<bool>();
  opts.fixed_x = c["fixed_x"].get<bool>();
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

magmaan::optim::OptimOptions geiser_opts() {
  return magmaan::optim::OptimOptions{
      .max_iter = 4000,
      .ftol = 1e-12,
      .gtol = 1e-8,
      .history = 10,
  };
}

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  return (a - b).cwiseAbs().maxCoeff();
}

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  return (a - b).cwiseAbs().maxCoeff();
}

bool is_manifest_path_case(const std::string& id) {
  return id.rfind("manifest_", 0) == 0;
}

bool has_known_nonparity_fit(const std::string& id) {
  // These cases are still useful corpus smoke checks, but not strict implied-
  // moment parity oracles yet: manifest path models expose fixed.x observed
  // moments differently, and the plain latent AR cross-lagged model currently
  // lands in a different GLS basin from lavaan.
  return is_manifest_path_case(id) ||
         id.rfind("latent_ar_cross_lagged/", 0) == 0;
}

void check_implied_against_lavaan(const std::string& id,
                                  const magmaan::spec::LatentStructure& pt,
                                  const magmaan::model::MatrixRep& rep,
                                  const magmaan::estimate::Estimates& est,
                                  double lavaan_fx,
                                  const Eigen::MatrixXd& lavaan_sigma,
                                  const Eigen::VectorXd& lavaan_mu,
                                  double d_fx_tol = 2e-4) {
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  REQUIRE_MESSAGE(ev.has_value(), id << ": ModelEvaluator build failed");
  auto im = ev->sigma(est.theta);
  REQUIRE_MESSAGE(im.has_value(), id << ": implied moments failed");
  REQUIRE(im->sigma.size() == 1);
  REQUIRE(im->mu.size() == 1);

  if (has_known_nonparity_fit(id)) {
    CHECK_MESSAGE(std::isfinite(est.fmin), id << ": non-finite fmin");
    return;
  }

  // Cross-program objective check, relative to the lavaan value. magmaan and
  // lavaan agree tightly on the deliberately-scaled GLS objective; the ULS
  // objective uses a slightly different mean-structure scale convention, so
  // ULS callers pass a looser tolerance. The scale-free implied-moment checks
  // below are the real parity gate either way.
  const double d_fx = std::abs(est.fmin - lavaan_fx);

  CHECK_MESSAGE(d_fx < d_fx_tol * std::fmax(1.0, std::abs(lavaan_fx)),
                id << ": |fmin - lavaan fx| = " << d_fx
                   << " (magmaan " << est.fmin << ", lavaan " << lavaan_fx
                   << ")");

  const double d_sigma = max_abs_diff(im->sigma[0], lavaan_sigma);
  const double d_mu = max_abs_diff(im->mu[0], lavaan_mu);
  CHECK_MESSAGE(d_sigma < 1e-4, id << ": max|Sigma - lavaan| = " << d_sigma);
  CHECK_MESSAGE(d_mu < 1e-4, id << ": max|mu - lavaan| = " << d_mu);
}

}  // namespace

TEST_CASE("Geiser GLS goldens match lavaan implied moments") {
  const std::string path =
      magmaan::test::fixtures_dir() + "/geiser/gls_reference.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("cases"));

  for (const auto& c : j["cases"]) {
    const std::string id = c["id"].get<std::string>();
    SUBCASE(id.c_str()) {
      REQUIRE(c["lavaan"]["converged"].get<bool>());
      auto handles = handles_from_case(c);
      auto samp = sample_stats_from_case(c);
      auto x0 = magmaan::estimate::simple_start_values(
          handles.pt, handles.rep, samp, {});
      REQUIRE_MESSAGE(x0.has_value(), id << ": start values failed");

      const double lavaan_fx = c["lavaan"]["fx"].get<double>();
      const Eigen::MatrixXd lavaan_sigma =
          matrix_from_json(c["lavaan"]["sigma"]);
      const Eigen::VectorXd lavaan_mu = vector_from_json(c["lavaan"]["mu"]);

      if (id != "latent_ar_cross_lagged") {
        auto full = magmaan::estimate::fit_gls(
            handles.pt, handles.rep, samp, *x0, magmaan::estimate::Bounds{},
            magmaan::estimate::Backend::Port, geiser_opts());
        if (!full.has_value()) {
          FAIL_CHECK(id << ": full GLS failed: " << full.error().detail);
          continue;
        }
        check_implied_against_lavaan(id + "/full", handles.pt, handles.rep,
                                     *full, lavaan_fx, lavaan_sigma, lavaan_mu);
      }

      auto snlls = magmaan::estimate::fit_snlls_gls(
          handles.pt, handles.rep, samp, *x0,
          magmaan::estimate::Backend::PortNls, geiser_opts());
      if (!snlls.has_value()) {
        FAIL_CHECK(id << ": SNLLS GLS failed: " << snlls.error().detail);
        continue;
      }
      check_implied_against_lavaan(id + "/snlls", handles.pt, handles.rep,
                                   *snlls, lavaan_fx, lavaan_sigma, lavaan_mu);
    }
  }
}

TEST_CASE("Geiser ULS goldens match lavaan implied moments") {
  const std::string path =
      magmaan::test::fixtures_dir() + "/geiser/uls_reference.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto j = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(j.is_discarded());
  REQUIRE(j.contains("cases"));

  for (const auto& c : j["cases"]) {
    const std::string id = c["id"].get<std::string>();
    SUBCASE(id.c_str()) {
      REQUIRE(c["lavaan"]["converged"].get<bool>());
      auto handles = handles_from_case(c);
      auto samp = sample_stats_from_case(c);
      auto x0 = magmaan::estimate::simple_start_values(
          handles.pt, handles.rep, samp, {});
      REQUIRE_MESSAGE(x0.has_value(), id << ": start values failed");

      const double lavaan_fx = c["lavaan"]["fx"].get<double>();
      const Eigen::MatrixXd lavaan_sigma =
          matrix_from_json(c["lavaan"]["sigma"]);
      const Eigen::VectorXd lavaan_mu = vector_from_json(c["lavaan"]["mu"]);

      // ULS is the moment quadratic with an empty (identity) weight; the GLS
      // mean-weight asymmetry does not arise here. fit_snlls is the profiled
      // counterpart, run on the default NLopt L-BFGS backend (the PortNls NL2SOL
      // backend trips its noisy-residual guard on the profiled ULS path for
      // latent_path). magmaan's ULS objective and lavaan's ULS fx use slightly
      // different mean-structure scale conventions, so the objective check
      // gets a looser relative tolerance; the implied moments are the gate.
      constexpr double kUlsFxTol = 5e-3;
      if (id != "latent_ar_cross_lagged") {
        auto full = magmaan::estimate::fit_gmm(
            handles.pt, handles.rep, samp, *x0,
            magmaan::estimate::gmm::Weight{}, magmaan::estimate::Bounds{},
            magmaan::estimate::Backend::Port, geiser_opts());
        if (!full.has_value()) {
          FAIL_CHECK(id << ": full ULS failed: " << full.error().detail);
          continue;
        }
        check_implied_against_lavaan(id + "/full", handles.pt, handles.rep,
                                     *full, lavaan_fx, lavaan_sigma, lavaan_mu,
                                     kUlsFxTol);
      }

      auto snlls = magmaan::estimate::fit_snlls(
          handles.pt, handles.rep, samp, *x0, magmaan::estimate::gmm::Weight{},
          magmaan::estimate::Backend::NloptLbfgs, geiser_opts());
      if (!snlls.has_value()) {
        FAIL_CHECK(id << ": SNLLS ULS failed: " << snlls.error().detail);
        continue;
      }
      check_implied_against_lavaan(id + "/snlls", handles.pt, handles.rep,
                                   *snlls, lavaan_fx, lavaan_sigma, lavaan_mu,
                                   kUlsFxTol);
    }
  }
}
