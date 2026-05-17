#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;

// Compute starting values for each free parameter, sized n_free.
//
// This is the "simple" start-value scheme — it corresponds to lavaan's
// `start = "simple"`, NOT lavaan's default (the default runs FABIN3 for factor
// loadings; see fabin_start_values once it lands). It is one of several
// free-standing start-value *producers*: the caller picks one and passes the
// result as the `x0` argument of `fit*`. The fitter no longer computes starts.
//
// Strategy:
//   - A user-supplied start hint (`starts.hint[k]`, from `start(v)*x` / `v?x`) wins.
//   - Free Λ entries (loadings):       ±0.7 (signed by the marker covariance)
//   - Free Θ diagonal (residual var):  0.5 * S_diag from sample
//   - Free Θ off-diagonal (resid cov): 0.0
//   - Free Ψ diagonal (latent var):    0.05
//   - Free Ψ off-diagonal (latent cov):0.0
//
// `starts` is the optional hint vector produced by `lavaanify` (an empty /
// shorter-than-n_free `hint` simply means "no hints").
fit_expected<Eigen::VectorXd>
simple_start_values(const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    const SampleStats& samp,
                    const spec::Starts& starts = {});

// FABIN variant: FABIN3 weights the instruments by S33⁻¹ (Hägglund 1982);
// FABIN2 omits that weighting. FABIN3 falls back to FABIN2 per indicator when
// the instrument submatrix is singular.
enum class FabinVariant : std::uint8_t { Fabin2, Fabin3 };

// FABIN start-value producer — this is lavaan's *default* loadings scheme.
//
// Equals `simple_start_values` for every parameter except free factor
// loadings, which are replaced by per-factor FABIN estimates (Hägglund 1982),
// run on each factor's indicator submatrix of `samp.S[block]`. A factor that
// FABIN cannot handle (a higher-order factor, fewer than two indicators, a
// non-finite estimate) keeps its simple-baseline loading. User start hints
// still win over the FABIN estimate.
fit_expected<Eigen::VectorXd>
fabin_start_values(const spec::LatentStructure& pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const spec::Starts& starts = {},
                   FabinVariant variant = FabinVariant::Fabin3);

// Guttman/MGM start-value producer — Guttman's (1952) multiple-group method
// for CFA factor loadings. Like `fabin_start_values`, equals the simple scheme
// except for free loadings; a block outside the method's domain (a structural
// part, crossloadings, a markerless factor, < 3 indicators) keeps the simple
// baseline. CFA-only.
fit_expected<Eigen::VectorXd>
guttman_start_values(const spec::LatentStructure& pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const spec::Starts& starts = {});

// Bentler-1982 start-value producer — Bentler's (1982) non-iterative CFA
// estimator (ULS variant). Same contract as `guttman_start_values`; a block
// with no non-marker indicators or a singular intermediate keeps the simple
// baseline. CFA-only.
fit_expected<Eigen::VectorXd>
bentler1982_start_values(const spec::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         const SampleStats& samp,
                         const spec::Starts& starts = {});

// James-Stein start-value producer — James-Stein-type shrinkage loadings
// (Burghgraeve, De Neve & Rosseel 2021), the non-aggregated "JS" variant.
// Plain JS reduces to a covariance ratio, so it needs only `samp` (no raw
// data). Same contract as `guttman_start_values`. CFA-only.
fit_expected<Eigen::VectorXd>
jamesstein_start_values(const spec::LatentStructure& pt,
                        const model::MatrixRep& rep,
                        const SampleStats& samp,
                        const spec::Starts& starts = {});

}  // namespace magmaan::estimate
