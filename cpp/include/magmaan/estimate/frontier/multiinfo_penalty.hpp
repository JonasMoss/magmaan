#pragma once

// Complete-data multi-information penalty (frontier; not a lavaan feature).
//
// For each block let v = [η; y] stack the extended latent vector (user latents
// plus the phantom ov.y / ov.x slots) and the observed vector. In RAM form
// v = A v + u with A = [[B, 0], [Λ, 0]], Cov(u) = S = diag(Ψ, Θ),
// E = (I − A)⁻¹ and complete-data covariance C = E S Eᵀ. K is the set of
// variables whose residual-variance cell S_ii is free or fixed at a nonzero
// value (fixed.x covariates stay in K; phantom observed copies with Θ_ii ≡ 0
// and zero-variance latents drop out). The penalty is
//
//   P(θ) = Σ_b [ log det C_KK − Σ_{i∈K} log C_ii ] = Σ_b log det Corr(v_K) ≤ 0,
//
// i.e. minus twice the Gaussian total correlation of v_K. It is invariant to
// rescaling any observed or latent variable, hence to marker vs std.lv
// identification. For recursive B, E_KK is unit triangular, so
//
//   P_b = log det S_KK − Σ log C_ii
//       = Σ_{i∈K} log(1 − R²_i) + log det Corr(S_KK),   R²_i = 1 − S_ii / C_ii,
//
// and P → −∞ exactly when S_KK approaches singularity: the penalized estimate
// lies in the interior of the covariance (PSD) domain. The penalized ML
// estimate maximizes ℓ(θ) + λ P(θ) with λ = η − 1 (the log of an LKJ(η)
// density on each Corr(v_K)); on the optimizer's ½F scale, with ℓ = −N·fmin,
// the minimized objective is fmin(θ) − (λ/N)·P(θ). λ is O(1), so the estimate
// differs from ML by O_p(1/N) at interior points.

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ml_numerics.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::frontier {

// `weight`, when finite, is the raw λ and overrides `eta − 1`. The default
// λ = 0.25 (a near-flat LKJ(1.25)) keeps the barrier while leaving χ² and Wald
// calibration at the ordinary/PSD-ML level in experiment 85's N = 50..400 grid;
// λ = 1 (η = 2) over-shrinks high-R² equations and correlations near 1 there.
struct MultiInfoPenaltyOptions {
  double eta = 1.25;
  double weight = std::numeric_limits<double>::quiet_NaN();
};

// Per-block structure, resolved once from the evaluator at a layout point.
// Complete-data indices: [0, m) are extended latents, [m, m + p) observed.
struct MultiInfoPenaltyBlock {
  std::int16_t m = 0;
  std::int16_t p = 0;
  std::vector<std::int32_t> keep;  // K, ascending
  bool recursive = true;           // B has an acyclic nonzero pattern
};

struct MultiInfoPenaltyLayout {
  std::vector<MultiInfoPenaltyBlock> blocks;
  std::vector<model::ParamLocation> locations;  // one per free θ entry

  bool recursive() const noexcept {
    for (const auto& b : blocks) {
      if (!b.recursive) return false;
    }
    return true;
  }
};

// Fixed cells are read at `theta`; any admissible θ gives the same layout.
// Fails when a variable outside K carries a covariance (free, or fixed nonzero)
// with another variable: such an S cannot be positive semidefinite.
fit_expected<MultiInfoPenaltyLayout>
multiinfo_penalty_layout(const model::ModelEvaluator& ev,
                         const Eigen::VectorXd& theta);

struct MultiInfoPenaltyValue {
  double value = 0.0;
  Eigen::VectorXd gradient;  // ∂P/∂θ over free θ; empty unless requested
};

// P(θ) and optionally ∂P/∂θ. Fails NumericIssue when some C_KK is not
// positive definite or some C_ii is not positive; callers treat that as an
// infeasible point (+∞ objective), never clamp.
fit_expected<MultiInfoPenaltyValue>
multiinfo_penalty(const MultiInfoPenaltyLayout& layout,
                  const model::ModelEvaluator& ev,
                  const Eigen::VectorXd& theta, bool with_gradient);

// One term per complete-data variable in K.
struct MultiInfoPenaltyTerm {
  std::int16_t block = 0;
  bool latent = false;
  std::int16_t index = 0;             // row of Ψ (latent) or Θ (observed)
  double log_one_minus_r2 = 0.0;      // log(S_ii / C_ii)
};

// For recursive blocks, block_value = Σ terms + residual_log_det_corr exactly.
// For nonrecursive blocks the remainder is 2 log|det E_KK|.
struct MultiInfoPenaltyReport {
  double value = 0.0;
  bool recursive = true;
  std::vector<double> block_value;
  std::vector<double> residual_log_det_corr;  // log det Corr(S_KK); NaN if S_KK not PD
  std::vector<MultiInfoPenaltyTerm> terms;
};

fit_expected<MultiInfoPenaltyReport>
multiinfo_penalty_report(const MultiInfoPenaltyLayout& layout,
                         const model::ModelEvaluator& ev,
                         const Eigen::VectorXd& theta);

// λ actually applied: `weight` when finite, else `eta − 1`. Fails for a
// negative or non-finite result.
fit_expected<double>
multiinfo_penalty_weight(const MultiInfoPenaltyOptions& options);

// Wrap a θ-space ½F objective as f(θ) − (λ/N)·P(θ). The returned problem
// borrows `base`, `layout`, and `ev`.
optim::ScalarProblem
multiinfo_penalized_problem(const optim::ScalarProblem& base,
                            const MultiInfoPenaltyLayout& layout,
                            const model::ModelEvaluator& ev,
                            double weight, double n_total);

// `estimates.theta` is the penalized estimate θ̃; `estimates.fmin` is the
// UNPENALIZED ½F at θ̃, so the usual χ² = 2N·fmin and fit measures read the
// ordinary criterion; the penalty only moves the point estimate.
// Optimizer telemetry, audits, and stationarity diagnostics describe the
// penalized objective actually minimized.
struct PenalizedFit {
  Estimates estimates;
  double weight = 0.0;           // λ
  double n_total = 0.0;          // N used in the ½F scaling
  double penalized_fmin = 0.0;   // fmin − (λ/N)·P at θ̃
  MultiInfoPenaltyReport penalty;
  bool start_repaired = false;   // x0 had to be moved into the barrier domain
};

// Complete-data normal-theory ML plus the multi-information penalty. Runs
// unbounded by default (the barrier replaces variance bounds); `bounds` is kept
// for comparisons. Single-level models only.
fit_expected<PenalizedFit>
fit_ml_multiinfo(spec::LatentStructure pt, const model::MatrixRep& rep,
                 const data::SampleStats& samp, const Eigen::VectorXd& x0,
                 MultiInfoPenaltyOptions options = {}, Bounds bounds = {},
                 Backend backend = Backend::NloptLbfgs,
                 optim::OptimOptions opts = ml_optim_options());

}  // namespace magmaan::estimate::frontier

namespace magmaan::estimate::fiml::frontier {

// Casewise FIML plus the multi-information penalty, N = number of cases.
fit_expected<estimate::frontier::PenalizedFit>
fit_fiml_multiinfo(spec::LatentStructure pt, const model::MatrixRep& rep,
                   const data::RawData& raw, const Eigen::VectorXd& x0,
                   const FIMLPack& pack,
                   estimate::frontier::MultiInfoPenaltyOptions options = {},
                   Bounds bounds = {},
                   Backend backend = Backend::NloptLbfgs,
                   optim::OptimOptions opts = {});

}  // namespace magmaan::estimate::fiml::frontier
