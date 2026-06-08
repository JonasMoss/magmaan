#pragma once

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate::fiml {

using data::RawData;
using data::SampleStats;
using estimate::Estimates;
using estimate::EqConstraints;
using estimate::build_eq_constraints;
using estimate::resolve_fixed_x_from_sample;
using estimate::simple_start_values;
using measures::BaselineFit;

struct FIMLPattern {
  std::size_t block = 0;
  std::vector<Eigen::Index> observed;
  std::int64_t n_obs = 0;
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

struct FIMLCache {
  std::vector<FIMLPattern> patterns;
  std::vector<Eigen::Index> sigma_offsets;
  std::vector<Eigen::Index> mu_offsets;
  std::vector<Eigen::Index> block_p;
  std::int64_t n_total = 0;
};

struct FIMLValueGradient {
  double value = 0.0;
  Eigen::VectorXd gradient;
};

struct FIMLExtras {
  double       logl              = 0.0;
  double       unrestricted_logl = 0.0;
  double       chi2              = 0.0;
  double       aic               = 0.0;
  double       bic               = 0.0;
  double       bic2              = 0.0;
  // Bentler-type SRMR of the model-implied moments against the FIML
  // saturated (H1, EM) moments — the missing-data analogue of the
  // complete-data `fit_extras().srmr`.
  double       srmr              = 0.0;
  int          npar              = 0;
  std::int64_t ntotal            = 0;
};

// Saturated (H1) ML/EM moments under missingness, packaged as a
// methods-developer surface for the Savalei-Bentler (2009) two-stage approach.
// Per block, η_b = (μ_b, vech(Σ_b)) with column-major lower-triangle vech (the
// same layout `fiml_saturated_scores_block` and `robust::casewise_contributions`
// use with `include_means = true`). The block-stacked η has
// `Σ_b (p_b + p_b·(p_b+1)/2)` entries; `H` and `J` are block-diagonal across
// blocks because multi-group observations are independent.
//
// Convention: `H` and `J` are reported in *log-likelihood* (not deviance) units.
//   H_b = -∂² logL_b / ∂η_b ∂η_bᵀ  (summed over rows of block b)
//   J_b = Σ_r (∂logL_r / ∂η_b)(∂logL_r / ∂η_b)ᵀ
//   acov = H⁻¹ J H⁻¹ — the sandwich asymptotic covariance of η̂_S; in finite
//   samples it carries 1/n_total units, matching the standard
//   `√n (η̂_S - η_S) → N(0, n·acov)` interpretation.
struct SaturatedMoments {
  std::vector<Eigen::VectorXd> mean;   // μ̂_S per block (size p_b)
  std::vector<Eigen::MatrixXd> cov;    // Σ̂_S per block (p_b × p_b)
  std::vector<std::int64_t>    n_obs;  // rows per block
  Eigen::MatrixXd              H;      // block-diagonal saturated information
  Eigen::MatrixXd              J;      // block-diagonal saturated score covariance
  Eigen::MatrixXd              acov;   // sandwich ACOV(η̂_S) = H⁻¹ J H⁻¹
};

struct FIMLRobustMLR {
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  Eigen::VectorXd eigvals;
  double       chisq_scaled      = std::numeric_limits<double>::quiet_NaN();
  double       scaling_factor    = std::numeric_limits<double>::quiet_NaN();
  double       trace_ugamma      = std::numeric_limits<double>::quiet_NaN();
  double       trace_ugamma_h1   = std::numeric_limits<double>::quiet_NaN();
  double       trace_ugamma_h0   = std::numeric_limits<double>::quiet_NaN();
  int          df                = 0;
  std::int64_t ntotal            = 0;
};

// The full missing-data UΓ spectrum for FMG goodness-of-fit tests under FIML.
// Unlike `FIMLRobustMLR` (which carries only the first cumulant via a trace
// difference and the q parameter-space eigenvalues of H⁻¹·meat), this returns
// the df nonzero eigenvalues of U·Γ_mis in the *saturated-moment* space — the
// reference law T → Σ λ_j χ²_1 that every FMG eigenvalue-tail transform consumes.
// Built first-principles from the saturated information (V = H), the
// saturated-moment ACOV (Γ_mis = H⁻¹JH⁻¹), and the model Jacobian
// Δ = ∂η_model/∂θ: U = V − VΔ(ΔᵀVΔ)⁻¹ΔᵀV. No lavaan-UGamma-rescaling hack.
struct FIMLUGammaSpectrum {
  Eigen::VectorXd eigvals;          // df nonzero eigenvalues of U·Γ_mis, ascending
  double          chi2_lrt = 0.0;   // FIML LRT (the FMG base statistic under FIML)
  int             df       = 0;
  double          trace_xcheck = 0.0;  // Σ eigvals; == fiml_robust_mlr().trace_ugamma
};

struct FIMLEtaJacobian {
  // Δ = ∂[μ_1; vech(Σ_1); μ_2; vech(Σ_2); ...] / ∂θ in the same saturated
  // eta layout used by `saturated_em_moments`.
  Eigen::MatrixXd Delta_theta;
};

// Full-information ML over raw continuous data with arbitrary observed-value
// patterns. The optimized scalar is the per-observation observed-pattern
// normal-theory deviance without saturated/H1 constants:
//
//   sum_patterns (n_r / N) * [log|Σ_oo| + tr(Σ_oo^-1 (S_r + dd'))]
//
// where d = xbar_r - μ_o and S_r is the N-divisor covariance inside the
// pattern. This is the right objective for point estimation; saturated/H1 and
// baseline likelihood accounting are handled by the post-fit helpers below.
struct FIML {
  static constexpr std::string_view name = "FIML";

  fit_expected<FIMLCache>
  prepare(const RawData& raw) const;

  fit_expected<double>
  value(const RawData& raw, const FIMLCache& cache,
        const model::ImpliedMoments& moments) const;

  fit_expected<Eigen::VectorXd>
  gradient(const RawData& raw, const FIMLCache& cache,
           const model::ImpliedMoments& moments,
           const Eigen::MatrixXd& J_sigma,
           const Eigen::MatrixXd& J_mu) const;

  fit_expected<FIMLValueGradient>
  value_gradient(const RawData& raw, const FIMLCache& cache,
                 const model::ImpliedMoments& moments,
                 const Eigen::MatrixXd& J_sigma,
                 const Eigen::MatrixXd& J_mu) const;
};

// Start-value and fixed.x helper for FIML. Means use all observed values in
// each column; covariances use pairwise observed rows with an N divisor.
fit_expected<SampleStats>
fiml_start_sample_stats(const RawData& raw);

// First public FIML fixed.x policy: observed exogenous variables may be fixed
// from complete raw data, but missing values in those variables are not yet
// supported because lavaan's conditional fixed.x likelihood accounting is a
// separate contract from the joint observed-data FIML objective.
fit_expected<void>
validate_fiml_fixed_x_missing_policy(const spec::LatentStructure& pt,
                                     const RawData& raw);

// Post-fit likelihood accounting for continuous raw-data FIML. The optimizer
// minimizes only the observed-pattern deviance without constants; this helper
// adds the normal constants and fits the saturated/H1 observed-data normal
// model so logl, unrestricted.logl, chi-square, and information criteria use
// lavaan-compatible missing-data likelihood accounting.
post_expected<FIMLExtras>
fiml_extras(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const RawData& raw,
            const Estimates& est,
            FIML discrepancy = {});

// Observed FIML information matrix — the npar × npar `−∂²logl/∂θ²` for a
// continuous raw-data FIML fit, computed as `(N/2)·H` where `H` is the
// finite-difference Hessian of the per-observation-averaged deviance. Its
// inverse (via `inference::vcov`, which folds in any equality constraints) is
// the *non-robust* missing-data standard error — the `se = "standard"`
// counterpart to `fiml_robust_mlr`'s sandwich SEs.
post_expected<Eigen::MatrixXd>
fiml_observed_information(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          FIML discrepancy = {},
                          double h_step = 1e-4);

// Robust missing-data reporting for lavaan's continuous FIML MLR corner
// (`missing = "fiml", estimator = "MLR"`). The sandwich meat is built from
// observed-pattern casewise deviance gradients, and the bread is a
// finite-difference observed FIML Hessian.
post_expected<FIMLRobustMLR>
fiml_robust_mlr(spec::LatentStructure pt,
                const model::MatrixRep& rep,
                const RawData& raw,
                const Estimates& est,
                int df,
                double chisq,
                FIML discrepancy = {},
                double h_step = 1e-4);

// Saturated (H1) ML/EM moments — runs the EM iteration that already drives
// FIML's H1 likelihood accounting, then aggregates per-block scores and
// finite-difference Hessians into a block-diagonal `(H, J, H⁻¹JH⁻¹)`.
// Multi-group safe; takes raw data only (no `LatentStructure` is needed because
// the saturated model has no structural restrictions). See `SaturatedMoments`
// for the η layout and scaling conventions.
post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw, double h_step = 1e-4);

post_expected<FIMLEtaJacobian>
fiml_eta_jacobian(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const RawData& raw,
                  const Estimates& est,
                  FIML discrepancy = {});

// First-principles FIML UΓ spectrum for FMG goodness-of-fit tests. Multi-group
// safe (H/J/acov are block-diagonal across groups; Δ stacks per group). `df` and
// `chi2_lrt` are passed in (the caller already has them from `infer_df_stat` /
// `fiml_extras`) to avoid recomputation, mirroring `fiml_robust_mlr`. Returns the
// df nonzero eigenvalues of U·Γ_mis; `trace_xcheck = Σ eigvals` must equal
// `fiml_robust_mlr(...).trace_ugamma` (validated in tests).
post_expected<FIMLUGammaSpectrum>
fiml_ugamma_spectrum(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     int df,
                     double chi2_lrt,
                     FIML discrepancy = {},
                     double h_step = 1e-4);

// FIML independence/baseline chi-square for raw continuous data with missing
// values. Unlike complete-data `baseline_chi2(SampleStats)`, this evaluates the
// diagonal normal model directly over observed-value patterns and compares it
// to the FIML saturated/H1 likelihood.
post_expected<BaselineFit>
fiml_baseline_chi2(const RawData& raw,
                   FIML discrepancy = {});

post_expected<BaselineFit>
fiml_baseline_chi2(const spec::LatentStructure& pt,
                   const RawData& raw,
                   FIML discrepancy = {});

// Full-information ML fit over raw continuous data. `backend` selects the
// scalar optimizer (NLopt L-BFGS by default, optional IPOPT when enabled);
// equality constraints are folded in via the θ = θ₀ + K·α reparameterization.
fit_expected<Estimates>
fit_fiml(spec::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         const Eigen::VectorXd& x0,      // start values, size pt.n_free()
         FIML discrepancy = {},
         Backend backend = Backend::NloptLbfgs,
         optim::OptimOptions opts = {});

}  // namespace magmaan::estimate::fiml

namespace magmaan::estimate {

using estimate::fiml::FIMLExtras;
using estimate::fiml::SaturatedMoments;
using estimate::fiml::fit_fiml;
using estimate::fiml::fiml_extras;
using estimate::fiml::saturated_em_moments;

}  // namespace magmaan::estimate
