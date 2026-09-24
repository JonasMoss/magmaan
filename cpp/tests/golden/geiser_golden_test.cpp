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
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include <vector>

namespace {

using magmaan::test::matrix_from_json;

using magmaan::test::vector_from_json;

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
  magmaan::spec::LatentNames names;
};

Handles handles_from_case(const nlohmann::json& c) {
  auto fp = magmaan::parse::Parser::parse(c["model"].get<std::string>());
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = c["meanstructure"].get<bool>();
  opts.fixed_x = c["fixed_x"].get<bool>();
  magmaan::spec::LatentNames names;
  auto pt = magmaan::spec::build(*fp, opts, nullptr, &names);
  REQUIRE(pt.has_value());
  // Pass `names` so `rep.ov_names` carries the real observed-variable order;
  // the fixtures supply summary stats and lavaan moments in their own data
  // column order, which we reconcile to magmaan's `ov_order` by name below.
  auto rep = magmaan::model::build_matrix_rep(*pt, &names);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep), std::move(names)};
}

// magmaan orders observed variables as `[ov.y, ov.x]` (classify), which need
// not match the fixture's data column order. `SampleStats` is name-free and
// positionally aligned to `ov_order`, so summary stats (and the lavaan moments
// we compare against) must be permuted into magmaan's observed order by name.
// For models whose data order already equals magmaan's `ov_order` this is the
// identity, so it leaves the previously-passing Geiser cases untouched.
std::vector<int> perm_to_magmaan(const std::vector<std::string>& magmaan_order,
                                 const std::vector<std::string>& fixture_order) {
  std::vector<int> perm(magmaan_order.size(), -1);
  for (std::size_t k = 0; k < magmaan_order.size(); ++k) {
    for (std::size_t f = 0; f < fixture_order.size(); ++f) {
      if (magmaan_order[k] == fixture_order[f]) { perm[k] = static_cast<int>(f); break; }
    }
  }
  return perm;
}

Eigen::MatrixXd reorder_sym(const Eigen::MatrixXd& A, const std::vector<int>& perm) {
  Eigen::MatrixXd B(A.rows(), A.cols());
  for (Eigen::Index i = 0; i < A.rows(); ++i)
    for (Eigen::Index j = 0; j < A.cols(); ++j)
      B(i, j) = A(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(j)]);
  return B;
}

Eigen::VectorXd reorder_vec(const Eigen::VectorXd& v, const std::vector<int>& perm) {
  Eigen::VectorXd w(v.size());
  for (Eigen::Index i = 0; i < v.size(); ++i)
    w(i) = v(perm[static_cast<std::size_t>(i)]);
  return w;
}

magmaan::optim::OptimOptions geiser_opts() {
  return magmaan::optim::OptimOptions{
      .max_iter = 4000,
      .ftol = 1e-12,
      .gtol = 1e-8,
      .history = 10,
  };
}

// A second moment-estimator (GLS/ULS) starting point. The cross-lagged panel
// models in this corpus have shallow moment-objective surfaces where
// `simple_start_values` can settle in a non-lavaan basin (the latent AR
// cross-lagged family). A plain ML fit is well-conditioned for these and lands
// in lavaan's basin; using its θ̂ as a second warm start, then keeping whichever
// of the two fits reaches the lower objective (`best_start`), rescues those
// cases without disturbing the well-behaved ones, where the simple start
// already reaches lavaan's basin (and a warm-started full fit can otherwise
// stop short of the GLS optimum). On ML failure we fall back to the simple
// start, so this is never worse than the prior simple-start recipe.
Eigen::VectorXd warm_start(Handles& h, const magmaan::data::SampleStats& samp,
                           const Eigen::VectorXd& x0) {
  auto ml = magmaan::estimate::fit_ml(h.pt, h.rep, samp, x0, {},
                                      magmaan::estimate::Backend::NloptLbfgs,
                                      geiser_opts());
  return ml.has_value() ? ml->theta : x0;
}

// Run `fit` from two starts and keep the lower-objective (better-basin) result.
template <class Fit>
magmaan::fit_expected<magmaan::estimate::Estimates>
best_start(Fit&& fit, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  auto ra = fit(a);
  auto rb = fit(b);
  if (!ra.has_value()) return rb;
  if (!rb.has_value()) return ra;
  return rb->fmin < ra->fmin ? std::move(rb) : std::move(ra);
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

// `lavaan_sigma` / `lavaan_mu` arrive already permuted into magmaan's observed
// order (see `perm_to_magmaan` at the call sites), so they line up positionally
// with magmaan's implied moments.
void check_implied_against_lavaan(const std::string& id,
                                  const magmaan::spec::LatentStructure& pt,
                                  const magmaan::model::MatrixRep& rep,
                                  const magmaan::estimate::Estimates& est,
                                  double lavaan_fx,
                                  const Eigen::MatrixXd& lavaan_sigma,
                                  const Eigen::VectorXd& lavaan_mu,
                                  double d_fx_abs = 2e-4,
                                  double d_fx_rel = 2.5e-3) {
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  REQUIRE_MESSAGE(ev.has_value(), id << ": ModelEvaluator build failed");
  auto im = ev->sigma(est.theta);
  REQUIRE_MESSAGE(im.has_value(), id << ": implied moments failed");
  REQUIRE(im->sigma.size() == 1);
  REQUIRE(im->mu.size() == 1);

  // Cross-program objective check. The implied-moment checks below are the
  // authoritative parity gate; this scalar comparison is a secondary sanity
  // bound that must tolerate the known ~1/N GLS/ULS mean-structure scale
  // convention difference between magmaan and lavaan. That difference is
  // relative, so the gate is `max(absolute floor, relative · |fx|)`: for the
  // GLS callers (`2e-4`, `2.5e-3`) it stays at the `2e-4` floor for every case
  // except the high-objective latent AR cross-lagged model (|fx| ≈ 0.27, where
  // Σ and μ still match lavaan to ~1e-7 so the residual gap is pure
  // convention); for the looser ULS callers (`5e-3`, `5e-3`) it reproduces the
  // prior `5e-3 · max(1, |fx|)` behavior exactly.
  const double d_fx = std::abs(est.fmin - lavaan_fx);
  const double fx_gate = std::fmax(d_fx_abs, d_fx_rel * std::abs(lavaan_fx));
  CHECK_MESSAGE(d_fx < fx_gate,
                id << ": |fmin - lavaan fx| = " << d_fx
                   << " (magmaan " << est.fmin << ", lavaan " << lavaan_fx
                   << ", gate " << fx_gate << ")");

  const double d_sigma = max_abs_diff(im->sigma[0], lavaan_sigma);
  const double d_mu = max_abs_diff(im->mu[0], lavaan_mu);
  CHECK_MESSAGE(d_sigma < 1e-4, id << ": max|Sigma - lavaan| = " << d_sigma);
  CHECK_MESSAGE(d_mu < 1e-4, id << ": max|mu - lavaan| = " << d_mu);
}

// Fill the fixed.x exogenous rows (exo=1, free=0, fixed_value=NaN) with the
// sample moments. The fit_* functions take `pt` by value and resolve their own
// copy, so the implied-moment check — which rebuilds the evaluator from the
// caller's `pt` — needs the resolution applied here too, or the exogenous block
// of Σ stays NaN-filled. Mirrors fit_implied_golden.
void resolve_handles(Handles& h, const magmaan::data::SampleStats& samp,
                     const std::string& id) {
  REQUIRE_MESSAGE(magmaan::estimate::resolve_fixed_x_from_sample(h.pt, h.rep,
                                                                 samp)
                      .has_value(),
                  id << ": resolve_fixed_x failed");
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
      // Reconcile the fixture's data column order with magmaan's `ov_order`
      // (which groups endogenous before exogenous): `SampleStats` is name-free
      // and positionally aligned to `ov_order`.
      const auto perm = perm_to_magmaan(
          handles.rep.ov_names[0],
          c["ov_names"].get<std::vector<std::string>>());
      samp.S[0] = reorder_sym(samp.S[0], perm);
      samp.mean[0] = reorder_vec(samp.mean[0], perm);
      resolve_handles(handles, samp, id);
      auto x0 = magmaan::estimate::simple_start_values(
          handles.pt, handles.rep, samp, {});
      REQUIRE_MESSAGE(x0.has_value(), id << ": start values failed");
      const Eigen::VectorXd ws = warm_start(handles, samp, *x0);

      const double lavaan_fx = c["lavaan"]["fx"].get<double>();
      const Eigen::MatrixXd lavaan_sigma =
          reorder_sym(matrix_from_json(c["lavaan"]["sigma"]), perm);
      const Eigen::VectorXd lavaan_mu =
          reorder_vec(vector_from_json(c["lavaan"]["mu"]), perm);

      auto full = best_start(
          [&](const Eigen::VectorXd& s) {
            return magmaan::estimate::fit_gls(
                handles.pt, handles.rep, samp, s, magmaan::estimate::Bounds{},
                magmaan::estimate::Backend::Port, geiser_opts());
          },
          *x0, ws);
      if (!full.has_value()) {
        FAIL_CHECK(id << ": full GLS failed: " << full.error().detail);
        continue;
      }
      check_implied_against_lavaan(id + "/full", handles.pt, handles.rep,
                                   *full, lavaan_fx, lavaan_sigma, lavaan_mu);

      auto snlls = best_start(
          [&](const Eigen::VectorXd& s) {
            return magmaan::estimate::fit_snlls_gls(
                handles.pt, handles.rep, samp, s,
                magmaan::estimate::Backend::PortNls, geiser_opts());
          },
          *x0, ws);
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
      // Reconcile the fixture's data column order with magmaan's `ov_order`
      // (which groups endogenous before exogenous): `SampleStats` is name-free
      // and positionally aligned to `ov_order`.
      const auto perm = perm_to_magmaan(
          handles.rep.ov_names[0],
          c["ov_names"].get<std::vector<std::string>>());
      samp.S[0] = reorder_sym(samp.S[0], perm);
      samp.mean[0] = reorder_vec(samp.mean[0], perm);
      resolve_handles(handles, samp, id);
      auto x0 = magmaan::estimate::simple_start_values(
          handles.pt, handles.rep, samp, {});
      REQUIRE_MESSAGE(x0.has_value(), id << ": start values failed");
      const Eigen::VectorXd ws = warm_start(handles, samp, *x0);

      const double lavaan_fx = c["lavaan"]["fx"].get<double>();
      const Eigen::MatrixXd lavaan_sigma =
          reorder_sym(matrix_from_json(c["lavaan"]["sigma"]), perm);
      const Eigen::VectorXd lavaan_mu =
          reorder_vec(vector_from_json(c["lavaan"]["mu"]), perm);

      // ULS is the moment quadratic with an empty (identity) weight; the GLS
      // mean-weight asymmetry does not arise here. fit_snlls is the profiled
      // counterpart, run on the default NLopt L-BFGS backend (the PortNls
      // NL2SOL backend trips its noisy-residual guard on the profiled ULS path
      // for latent_path). magmaan's ULS objective and lavaan's ULS fx use
      // slightly different mean-structure scale conventions, so the objective
      // check gets a looser relative tolerance; the implied moments are the
      // gate.
      constexpr double kUlsFxAbs = 5e-3;
      constexpr double kUlsFxRel = 5e-3;
      auto full = best_start(
          [&](const Eigen::VectorXd& s) {
            return magmaan::estimate::fit_gmm(
                handles.pt, handles.rep, samp, s,
                magmaan::estimate::gmm::Weight{}, magmaan::estimate::Bounds{},
                magmaan::estimate::Backend::Port, geiser_opts());
          },
          *x0, ws);
      if (!full.has_value()) {
        FAIL_CHECK(id << ": full ULS failed: " << full.error().detail);
        continue;
      }
      check_implied_against_lavaan(id + "/full", handles.pt, handles.rep,
                                   *full, lavaan_fx, lavaan_sigma, lavaan_mu,
                                   kUlsFxAbs, kUlsFxRel);

      auto snlls = best_start(
          [&](const Eigen::VectorXd& s) {
            return magmaan::estimate::fit_snlls(
                handles.pt, handles.rep, samp, s,
                magmaan::estimate::gmm::Weight{},
                magmaan::estimate::Backend::NloptLbfgs, geiser_opts());
          },
          *x0, ws);
      if (!snlls.has_value()) {
        FAIL_CHECK(id << ": SNLLS ULS failed: " << snlls.error().detail);
        continue;
      }
      check_implied_against_lavaan(id + "/snlls", handles.pt, handles.rep,
                                   *snlls, lavaan_fx, lavaan_sigma, lavaan_mu,
                                   kUlsFxAbs, kUlsFxRel);
    }
  }
}
