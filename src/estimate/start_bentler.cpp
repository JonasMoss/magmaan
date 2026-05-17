#include "magmaan/estimate/start_values.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/cfa_utils.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;

// bentler1982_start_values — Bentler's (1982) non-iterative CFA estimator,
// used as a start-value producer (ULS variant). Port of `lav_cfa_bentler1982`.
// Equals `simple_start_values` for every parameter except free loadings; a
// block it cannot handle (a structural part, crossloadings, a factor without a
// marker, no non-marker indicators, a singular intermediate) keeps its
// simple-baseline loadings. User hints still win.
fit_expected<Eigen::VectorXd>
bentler1982_start_values(const spec::LatentStructure& pt,
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
    const Eigen::Index p = nvar - nfac;  // non-marker indicators
    if (p < 1) continue;

    // Partition the observed rows into markers (mk, one per factor) and the
    // remaining non-marker indicators (nm).
    std::vector<Eigen::Index> mk(static_cast<std::size_t>(nfac));
    std::vector<char> is_marker(static_cast<std::size_t>(nvar), 0);
    std::vector<std::int16_t> factor_of(static_cast<std::size_t>(nvar), -1);
    bool mk_ok = true;
    for (Eigen::Index f = 0; f < nfac; ++f) {
      const std::int16_t m = L.marker_ov[static_cast<std::size_t>(f)];
      if (m < 0 || m >= nvar) {
        mk_ok = false;
        break;
      }
      mk[static_cast<std::size_t>(f)] = m;
      is_marker[static_cast<std::size_t>(m)] = 1;
    }
    if (!mk_ok) continue;
    for (const auto& load : L.loads) {
      factor_of[static_cast<std::size_t>(load.ov_row)] = load.factor;
    }
    std::vector<Eigen::Index> nm;
    nm.reserve(static_cast<std::size_t>(p));
    for (Eigen::Index i = 0; i < nvar; ++i) {
      if (!is_marker[static_cast<std::size_t>(i)]) nm.push_back(i);
    }
    if (static_cast<Eigen::Index>(nm.size()) != p) continue;

    // Sub-blocks of S.
    Eigen::MatrixXd S_yx(p, nfac), S_yy(p, p);
    Eigen::MatrixXd B_y = Eigen::MatrixXd::Zero(p, nfac);
    for (Eigen::Index a = 0; a < p; ++a) {
      const Eigen::Index ra = nm[static_cast<std::size_t>(a)];
      for (Eigen::Index f = 0; f < nfac; ++f) {
        S_yx(a, f) = S(ra, mk[static_cast<std::size_t>(f)]);
      }
      for (Eigen::Index d = 0; d < p; ++d) {
        S_yy(a, d) = S(ra, nm[static_cast<std::size_t>(d)]);
      }
      const std::int16_t f = factor_of[static_cast<std::size_t>(ra)];
      if (f >= 0 && f < nfac) B_y(a, f) = 1.0;
    }
    Eigen::VectorXd S_xx_diag(nfac);
    for (Eigen::Index f = 0; f < nfac; ++f) {
      S_xx_diag(f) = S(mk[static_cast<std::size_t>(f)],
                       mk[static_cast<std::size_t>(f)]);
    }

    // ULS: G is the projection onto the column space of S_yx.
    const Eigen::MatrixXd xy = S_yx.transpose() * S_yx;  // nfac × nfac
    Eigen::LDLT<Eigen::MatrixXd> xy_ldlt(xy);
    if (xy_ldlt.info() != Eigen::Success) continue;
    const Eigen::MatrixXd G = S_yx * xy_ldlt.solve(S_yx.transpose());  // p × p

    // Residual variances of the non-marker indicators.
    const Eigen::MatrixXd tmp1 =
        Eigen::MatrixXd::Identity(p, p) - G.cwiseProduct(G);
    const Eigen::MatrixXd tmp2 = S_yy - G * S_yy * G;
    Eigen::VectorXd theta_f =
        tmp1.colPivHouseholderQr().solve(tmp2.diagonal());
    for (Eigen::Index a = 0; a < p; ++a) {
      theta_f(a) = std::min(std::max(theta_f(a), 0.0), S_yy(a, a));
    }
    const Eigen::MatrixXd Theta_yhat = theta_f.asDiagonal();

    // SminTheta with the optional PD "lambda" correction (Bentler/lavaan).
    Eigen::MatrixXd SminTheta = S_yy - Theta_yhat;
    const std::int64_t nobs = b < samp.n_obs.size() ? samp.n_obs[b] : 0;
    if (nobs > 2) {
      const std::optional<double> root = smallest_gen_root(S_yy, Theta_yhat);
      if (root.has_value()) {
        const double inv = 1.0 / static_cast<double>(nobs - 1);
        if (*root < 1.0 + inv) {
          SminTheta = S_yy - (*root - inv) * Theta_yhat;
        }
      }
    }

    // PSI (factor covariances).
    const Eigen::MatrixXd tmp2b = S_yx.transpose() * SminTheta * S_yx;
    Eigen::LDLT<Eigen::MatrixXd> tmp2b_ldlt(tmp2b);
    if (tmp2b_ldlt.info() != Eigen::Success) continue;
    Eigen::MatrixXd PSI = xy * tmp2b_ldlt.solve(xy);
    for (Eigen::Index f = 0; f < nfac; ++f) {
      const double lo = 0.1 * S_xx_diag(f);
      if (PSI(f, f) < lo) PSI(f, f) = lo;
      if (PSI(f, f) > S_xx_diag(f)) PSI(f, f) = S_xx_diag(f);
    }
    PSI = force_pd(PSI, 1e-4);

    // Non-marker loadings: Λ_y = (PSI⁻¹ · S_yxᵀ)ᵀ, masked by the pattern.
    Eigen::LDLT<Eigen::MatrixXd> psi_ldlt(PSI);
    if (psi_ldlt.info() != Eigen::Success) continue;
    const Eigen::MatrixXd LAMBDA_y =
        psi_ldlt.solve(S_yx.transpose()).transpose().cwiseProduct(B_y);

    for (const auto& load : L.loads) {
      if (load.free_idx < 0) continue;  // markers / fixed loadings
      if (is_marker[static_cast<std::size_t>(load.ov_row)]) continue;
      const std::size_t fi = static_cast<std::size_t>(load.free_idx);
      if (fi < starts.hint.size() && std::isfinite(starts.hint[fi])) continue;
      // Locate this indicator's row in the non-marker ordering.
      Eigen::Index a = -1;
      for (Eigen::Index t = 0; t < p; ++t) {
        if (nm[static_cast<std::size_t>(t)] == load.ov_row) {
          a = t;
          break;
        }
      }
      if (a < 0) continue;
      const double v = LAMBDA_y(a, load.factor);
      if (std::isfinite(v)) x0(load.free_idx) = v;
    }
  }

  return x0;
}

}  // namespace magmaan::estimate
