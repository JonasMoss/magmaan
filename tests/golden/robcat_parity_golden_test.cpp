// robcat-parity golden tests — robust polychoric correlation sanity suite.
//
// robcat (Welz, Mair & Alfons, 2026; vendored in external/robcat) is the
// canonical robust polychoric estimator. This suite pins magmaan's bivariate
// ordinal estimators against it so that any divergence is caught and gets
// investigated.
//
// === Estimator equivalence (verified from the sources) =====================
//
// robcat polycor(c=C) minimises  Σ_k p_k · ρ_fun(f̂_k / p_k)
//   (external/robcat/src/polycor.cpp:rho_fun_fast, R/polycor.R:polycor), with
//   the internal reparametrisation c1 = 0, c2 = C + 1:
//       ρ_fun(t) = t·log(t)                  for t ≤ C+1
//       ρ_fun(t) = t·(log(C+1)+1) − (C+1)    for t > C+1
//
// magmaan WmaHardCap (src/data/h_score.cpp) has phi(t) identical with the hard
// cap k = C+1, and the joint fitter minimises Σ p·phi(t). Hence
//
//       robcat polycor(c=C)  ≡  magmaan WmaHardCap, k = C + 1
//       robcat polycor_mle   ≡  magmaan ML  (= WmaHardCap, k = ∞)
//
// Point estimates (rho + thresholds) are hard-gated: residual disagreement is
// only optimiser tolerance + bivariate-normal CDF numerics, so a gap beyond
// tolerance is a genuine divergence.
//
// === Standard errors ========================================================
//
// Robust C-estimator SEs are hard-gated against robcat's sandwich for c > 0:
//
//   The C-estimation sandwich meat is W·Λ·Wᵀ with column W_k = h'(t_k)·s_k —
//   the estimating function linearised through the h-score *derivative* h'(t)
//   (robcat: polycor_variance / get_MW). magmaan stores the expanded,
//   centered h'(t)-weighted rows in
//   OrdinalPairHWeightedInfluence::estimating_functions, so
//   score_gamma = rowsᵀ rows / n matches W·Λ·Wᵀ before the analytic bread is
//   applied.
//
// The c = 0 corner is still fit and checked for finite positive SEs, but its
// magnitude is not hard-gated: at k = 1 the WMA hard cap is a degenerate
// boundary case where the usual sandwich-normal approximation is not the
// inferential target we want to bless.
//
// Fixtures: tests/fixtures/robcat/<id>.json, generated once by
// tools/regen_robcat_fixtures.R (needs R + the pinned robcat version). No R
// runs here — the JSON oracle is frozen and committed.
//
//   ./build/fast/tests/magmaan_test_robcat -s

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/h_score.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"

namespace {

namespace data = magmaan::data;

// One fixture per case; each must have tests/fixtures/robcat/<id>.json.
const std::vector<std::string> kRobcatCases = {
    "clean_3x3_moderate", "clean_5x5_strong",  "clean_4x4_negative",
    "clean_2x2",          "skewed_5x5",        "contam_5x5_corner",
    "contam_4x4_uniform", "mild_3x3_lowcorr",
};

// robcat fixtures store a length-1 threshold list as a bare scalar
// (jsonlite auto_unbox); accept number-or-array.
Eigen::VectorXd json_to_vec(const nlohmann::json& v) {
  if (v.is_array()) {
    Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
    for (Eigen::Index k = 0; k < out.size(); ++k)
      out(k) = v[static_cast<std::size_t>(k)].get<double>();
    return out;
  }
  Eigen::VectorXd out(1);
  out(0) = v.get<double>();
  return out;
}

Eigen::MatrixXd json_to_mat(const nlohmann::json& v) {
  const auto r = static_cast<Eigen::Index>(v.size());
  const auto c =
      r > 0 ? static_cast<Eigen::Index>(v[0].size()) : Eigen::Index{0};
  Eigen::MatrixXd M(r, c);
  for (Eigen::Index i = 0; i < r; ++i)
    for (Eigen::Index j = 0; j < c; ++j)
      M(i, j) = v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    .get<double>();
  return M;
}

std::string fmt(double x) {
  std::ostringstream s;
  s.precision(6);
  s << x;
  return s.str();
}

}  // namespace

TEST_CASE("robcat-parity — robust polychoric vs robcat (canonical)") {
  using data::PolychoricHScoreKind;
  using data::PolychoricHScoreOptions;

  std::vector<std::string> failures;  // hard gate

  for (const std::string& id : kRobcatCases) {
    const std::string path =
        magmaan::test::fixtures_dir() + "/robcat/" + id + ".json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) {
      failures.push_back(id +
                         ": missing fixture — run tools/regen_robcat_fixtures.R");
      continue;
    }
    auto fx = nlohmann::json::parse(*raw, nullptr, /*allow_exceptions=*/false);
    if (fx.is_discarded() || !fx.contains("counts") || !fx.contains("robcat")) {
      failures.push_back(id + ": malformed fixture JSON");
      continue;
    }

    const Eigen::MatrixXd counts = json_to_mat(fx["counts"]);
    const double n_obs = counts.sum();
    const bool contaminated = !fx["dgp"]["contamination"].is_null();

    // Point-estimate tolerance: clean tables land on a sharp interior optimum
    // both fitters reach; contaminated tables admit a touch more optimiser
    // drift. A real algorithmic divergence is ≥1e-2 — well outside either.
    const double tol_pt = contaminated ? 6e-3 : 3e-3;

    double worst_pt = 0.0;  // per-case worst margins, for the summary line
    double worst_se = 0.0;

    auto check_scalar = [&](const std::string& tag, double got, double exp) {
      const double d = std::abs(got - exp);
      worst_pt = std::max(worst_pt, d);
      if (!(d <= tol_pt))
        failures.push_back(id + " [" + tag + "]: magmaan=" + fmt(got) +
                           " robcat=" + fmt(exp) + " |Δ|=" + fmt(d) +
                           " tol=" + fmt(tol_pt));
    };
    auto check_vec = [&](const std::string& tag, const Eigen::VectorXd& got,
                         const Eigen::VectorXd& exp) {
      if (got.size() != exp.size()) {
        failures.push_back(id + " [" + tag + "]: length " +
                           std::to_string(got.size()) + " ≠ robcat " +
                           std::to_string(exp.size()));
        return;
      }
      for (Eigen::Index k = 0; k < got.size(); ++k)
        check_scalar(tag + "[" + std::to_string(k) + "]", got(k), exp(k));
    };

    // Fit the joint (thresholds + rho) h-weighted model on the contingency
    // table. lavaan_adjust_2x2 = false: robcat applies no 2×2 continuity
    // adjustment (and the flag is a no-op for ≥3-category tables).
    auto do_fit = [&](const std::string& tag, const PolychoricHScoreOptions& hs)
        -> std::optional<data::OrdinalPairJointHWeightedResult> {
      data::OrdinalPairJointHWeightedOptions opts;
      opts.lavaan_adjust_2x2 = false;
      opts.h_score = hs;
      auto fit = data::fit_ordinal_pair_joint_h_weighted(counts, opts);
      if (!fit.has_value()) {
        failures.push_back(id + " [" + tag +
                           "]: magmaan fit failed — " + fit.error().detail);
        return std::nullopt;
      }
      if (!fit->converged || fit->hit_lower || fit->hit_upper)
        failures.push_back(id + " [" + tag + "]: fit not clean (converged=" +
                           std::to_string(fit->converged) + " hit_lower=" +
                           std::to_string(fit->hit_lower) + " hit_upper=" +
                           std::to_string(fit->hit_upper) + ")");
      return std::move(*fit);
    };

    // Sandwich SEs from the casewise influence at the fitted parameters,
    // permuted from magmaan order [thr_i, thr_j, rho] into robcat order
    // [rho, a.., b..]. gamma = influence'influence/n is the asymptotic
    // covariance of √n·θ̂, so se(θ̂) = sqrt(diag(gamma)/n).
    auto check_se = [&](const std::string& tag,
                        const data::OrdinalPairJointHWeightedResult& fit,
                        const PolychoricHScoreOptions& hs,
                        const nlohmann::json& robj,
                        bool gate_magnitude) {
      data::OrdinalPairHWeightedInfluenceOptions iopts;
      iopts.h_score = hs;
      auto infl = data::ordinal_pair_h_weighted_influence(
          counts, fit.thresholds_i, fit.thresholds_j, fit.rho, iopts);
      if (!infl.has_value()) {
        failures.push_back(id + " [" + tag +
                           ".se]: influence failed — " + infl.error().detail);
        return;
      }
      const Eigen::Index nti = fit.thresholds_i.size();
      const Eigen::Index ntj = fit.thresholds_j.size();
      const Eigen::Index npar = nti + ntj + 1;
      Eigen::VectorXd se_mag(npar);
      for (Eigen::Index k = 0; k < npar; ++k)
        se_mag(k) = std::sqrt(std::max(0.0, infl->gamma(k, k) / n_obs));
      Eigen::VectorXd se(npar);
      se(0) = se_mag(npar - 1);                            // rho
      se.segment(1, nti) = se_mag.head(nti);               // a-thresholds
      se.segment(1 + nti, ntj) = se_mag.segment(nti, ntj);  // b-thresholds

      // Hard gate: SEs must be finite and strictly positive.
      for (Eigen::Index k = 0; k < npar; ++k) {
        if (!(std::isfinite(se(k)) && se(k) > 0.0))
          failures.push_back(id + " [" + tag + ".se[" + std::to_string(k) +
                             "]]: non-finite/non-positive SE " + fmt(se(k)));
      }

      const Eigen::VectorXd se_rob = json_to_vec(robj["stderr"]);
      if (se.size() != se_rob.size()) {
        failures.push_back(id + " [" + tag + ".se]: length " +
                           std::to_string(se.size()) + " ≠ robcat " +
                           std::to_string(se_rob.size()));
        return;
      }
      double max_d = 0.0, min_ratio = std::numeric_limits<double>::infinity(),
             max_ratio = 0.0;
      for (Eigen::Index k = 0; k < se.size(); ++k) {
        max_d = std::max(max_d, std::abs(se(k) - se_rob(k)));
        if (se_rob(k) > 0.0) {
          const double r = se(k) / se_rob(k);
          min_ratio = std::min(min_ratio, r);
          max_ratio = std::max(max_ratio, r);
        }
      }
      worst_se = std::max(worst_se, max_d);
      const double se_tol =
          std::max(3e-3, 5e-2 * se_rob.cwiseAbs().maxCoeff());
      if (gate_magnitude && max_d > se_tol)
        failures.push_back(id + " [" + tag +
                           ".se]: magmaan/robcat ratio in [" +
                           fmt(min_ratio) + ", " + fmt(max_ratio) +
                           "]  max|Δ|=" + fmt(max_d) +
                           " tol=" + fmt(se_tol));
    };

    auto compare = [&](const std::string& tag, const PolychoricHScoreOptions& hs,
                       const nlohmann::json& robj, bool check_se_too,
                       bool gate_se_magnitude = true) {
      auto fit = do_fit(tag, hs);
      if (!fit) return;
      check_scalar(tag + ".rho", fit->rho, robj["rho"].get<double>());
      check_vec(tag + ".thr_x", fit->thresholds_i,
                json_to_vec(robj["thresholds_x"]));
      check_vec(tag + ".thr_y", fit->thresholds_j,
                json_to_vec(robj["thresholds_y"]));
      if (check_se_too) check_se(tag, *fit, hs, robj, gate_se_magnitude);
    };

    // MLE: robcat polycor_mle ≡ magmaan ML. Point estimates gated; SEs are
    // not compared — robcat's polycor_mle vcov is Fisher-information based,
    // not the sandwich.
    compare("mle", PolychoricHScoreOptions{}, fx["robcat"]["mle"],
            /*check_se_too=*/false);

    // Robust C-estimator sweep: robcat polycor(c) ≡ magmaan WmaHardCap, k=c+1.
    // Point estimates and robust sandwich SEs are gated.
    for (const auto& rj : fx["robcat"]["robust"]) {
      const double c = rj["c"].get<double>();
      PolychoricHScoreOptions hs;
      hs.kind = PolychoricHScoreKind::WmaHardCap;
      hs.k = c + 1.0;
      compare("c=" + fmt(c), hs, rj, /*check_se_too=*/true,
              /*gate_se_magnitude=*/c > 0.0);
    }

    // doctest MESSAGE expands to `mb * <arg>`, so pass one pre-built string.
    std::string summary = std::string(id) + ": worst |Δ| point=" +
                          fmt(worst_pt) + " se=" + fmt(worst_se);
    if (contaminated) summary += "  [contaminated]";
    MESSAGE(summary);
  }

  for (const std::string& f : failures) MESSAGE(f);
  CHECK(failures.empty());
}
