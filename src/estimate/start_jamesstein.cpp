#include "magmaan/estimate/start_values.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/cfa_utils.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;

// jamesstein_start_values — James-Stein-type shrinkage start values for CFA
// factor loadings (Burghgraeve, De Neve & Rosseel 2021), used as a start-value
// producer. Port of lavaan's `lav_cfa_jamesstein` (the non-aggregated "JS"
// variant).
//
// For the non-aggregated estimator each non-marker loading is the OLS slope of
// the indicator on a James-Stein conditional expectation E(η|y_marker). Since
// E(η|y_marker) is an affine function of the marker column, that slope reduces
// exactly to a covariance ratio,
//
//     λ̂_j = Cov(y_j, y_marker) / (R · Var(y_marker)),
//     R   = 1 − (N−3)·θ_marker / ((N−1)·Var(y_marker)),
//
// with θ_marker the marker's Spearman communality residual. So plain JS needs
// only the sample covariance and N — no raw data. (The aggregated "JSA"
// variant, which optimises per-row aggregation weights, would need raw data;
// it is out of scope here.)
//
// Equals `simple_start_values` except for free loadings; a block outside the
// method's domain (a structural part, crossloadings, a markerless factor, a
// factor with < 3 indicators, an unreliable marker) keeps the simple baseline.
// User hints still win. CFA-only.
fit_expected<Eigen::VectorXd>
jamesstein_start_values(const spec::LatentStructure& pt,
                        const model::MatrixRep& rep,
                        const SampleStats& samp,
                        const spec::Starts& starts) {
  auto base = simple_start_values(pt, rep, samp, starts);
  if (!base.has_value()) return base;
  Eigen::VectorXd x0 = std::move(*base);
  if (x0.size() == 0) return x0;
  if (rep.form != model::RepForm::PureCFA) return x0;  // CFA-only method

  const std::vector<CfaBlockLayout> layouts = cfa_block_layouts(pt, rep);
  for (std::size_t b = 0; b < layouts.size(); ++b) {
    const CfaBlockLayout& L = layouts[b];
    const Eigen::Index nfac = L.n_factor();
    if (nfac == 0 || L.crossloadings || !L.all_have_marker) continue;
    if (b >= samp.S.size()) continue;
    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::Index nvar = L.n_observed;
    if (S.rows() != nvar || S.cols() != nvar) continue;
    const std::int64_t N = b < samp.n_obs.size() ? samp.n_obs[b] : 0;
    if (N < 4) continue;  // need N − 3 > 0

    std::vector<std::vector<Eigen::Index>> rows_of(static_cast<std::size_t>(nfac));
    for (const auto& load : L.loads) {
      rows_of[static_cast<std::size_t>(load.factor)].push_back(load.ov_row);
    }
    bool ok = true;
    for (const auto& r : rows_of) {
      if (r.size() < 3) ok = false;  // Spearman communality needs >= 3
    }
    if (!ok) continue;

    const double nn = static_cast<double>(N);
    for (Eigen::Index f = 0; f < nfac; ++f) {
      const auto& rr = rows_of[static_cast<std::size_t>(f)];
      const std::int16_t marker = L.marker_ov[static_cast<std::size_t>(f)];
      if (marker < 0 || marker >= nvar) continue;

      // Spearman communalities for this factor; pick out the marker's.
      Eigen::MatrixXd Sf(static_cast<Eigen::Index>(rr.size()),
                         static_cast<Eigen::Index>(rr.size()));
      Eigen::Index marker_local = -1;
      for (std::size_t a = 0; a < rr.size(); ++a) {
        if (rr[a] == marker) marker_local = static_cast<Eigen::Index>(a);
        for (std::size_t d = 0; d < rr.size(); ++d) {
          Sf(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(d)) =
              S(rr[a], rr[d]);
        }
      }
      if (marker_local < 0) continue;
      const double theta_marker = theta_spearman(Sf)(marker_local);

      const double var_marker = S(marker, marker);
      if (var_marker <= 0.0) continue;
      const double R = 1.0 - (nn - 3.0) * theta_marker / ((nn - 1.0) * var_marker);
      if (R < 1e-3) continue;  // unreliable marker ⇒ keep the baseline
      const double denom = R * var_marker;

      for (const auto& load : L.loads) {
        if (load.factor != f || load.free_idx < 0) continue;
        if (load.ov_row == marker) continue;
        const std::size_t fi = static_cast<std::size_t>(load.free_idx);
        if (fi < starts.hint.size() && std::isfinite(starts.hint[fi])) continue;
        const double slope = S(load.ov_row, marker) / denom;
        if (std::isfinite(slope)) x0(load.free_idx) = slope;
      }
    }
  }

  return x0;
}

}  // namespace magmaan::estimate
