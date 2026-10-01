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
//
// Determinacy target (latent-determinacy barrier). Let L be the genuine
// latents of a block: latents in K that no error-free indicator reproduces
// exactly (phantom ov.y / ov.x slots and single indicators with Θ ≡ 0 are
// observed data, not latent). With V = Var(η_L | y) = C_LL − C_Ly Σ⁻¹ C_yL,
//
//   Q_b = D_L^{-1/2} V D_L^{-1/2},   D_L = diag(C_LL),
//   P_b = log det Q_b = log det C_JJ − log det Σ − Σ_{j∈L} log C_jj   (J = y ∪ L)
//       = log det Corr(C_LL) + log det Var(y | η_L) − log det Σ
//       = −2 [TC(η_L) + I(η_L; y)] ≤ 0.
//
// P is the joint penalty with the observed margin conditioned out: it is zero
// for models without genuine latents, depends on the observed block only
// through Σ (which the likelihood already keeps positive definite), and tends
// to −∞ exactly where Var(η_L | y) loses rank with Σ positive definite, i.e.
// at the improper faces the likelihood does not guard. With Σ ≻ 0, C_JJ ≻ 0
// holds exactly when every residual (co)variance matrix is positive definite,
// so the Cholesky factorization of C_JJ is the domain check. The value is
// invariant to rescaling any latent or observed variable. Vanishing latents
// (C_jj → 0) are not faces of this barrier.

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/ml_numerics.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::frontier {

// `weight`, when finite, is the raw λ and overrides `eta − 1`. The default
// λ = 0.25 (a near-flat LKJ(1.25)) keeps the barrier while leaving χ² and Wald
// calibration at the ordinary/PSD-ML level in experiment research/47's N = 50..400 grid;
// λ = 1 (η = 2) over-shrinks high-R² equations and correlations near 1 there.
// Which barrier the penalty reads (see the header comment).
//   Joint:       log det Corr(v_K), the multi-information barrier.
//   Determinacy: log det Q, the joint barrier conditioned on the observed block.
enum class PenaltyTarget : std::uint8_t { Joint, Determinacy };

struct MultiInfoPenaltyOptions {
  double eta = 1.25;
  double weight = std::numeric_limits<double>::quiet_NaN();
  PenaltyTarget target = PenaltyTarget::Joint;
};

// Per-block structure, resolved once from the evaluator at a layout point.
// Complete-data indices: [0, m) are extended latents, [m, m + p) observed.
struct MultiInfoPenaltyBlock {
  std::int16_t m = 0;
  std::int16_t p = 0;
  // Joint: K, ascending. Determinacy: L (genuine latents, ascending, all < m);
  // the conditioning block is always the p observed variables.
  std::vector<std::int32_t> keep;
  bool recursive = true;           // B has an acyclic nonzero pattern
};

struct MultiInfoPenaltyLayout {
  PenaltyTarget target = PenaltyTarget::Joint;
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
// with another variable: such an S cannot be positive semidefinite. The
// Determinacy target also fails when an error-free indicator loads on more
// than one latent, since Var(η | y) is then singular for every θ.
fit_expected<MultiInfoPenaltyLayout>
multiinfo_penalty_layout(const model::ModelEvaluator& ev,
                         const Eigen::VectorXd& theta,
                         PenaltyTarget target = PenaltyTarget::Joint);

// ∇²P(θ), analytic, over free θ. P is a signed sum of log determinants of
// principal submatrices of the complete-data covariance C = E S Eᵀ minus log
// diagonal entries, and S and the RAM A are linear in θ, so every ∂C/∂θ_a is
// a symmetric rank-two matrix and the Hessian needs no finite differences.
// Fails like multiinfo_penalty at infeasible points.
fit_expected<Eigen::MatrixXd>
multiinfo_penalty_hessian(const MultiInfoPenaltyLayout& layout,
                          const model::ModelEvaluator& ev,
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

// Joint: one term per complete-data variable in K, log(1 − R²_i) =
// log(S_ii / C_ii). Determinacy: one term per genuine latent, log(1 − ρ²_j) =
// log(V_jj / C_jj), with ρ²_j the factor-score determinacy of η_j.
struct MultiInfoPenaltyTerm {
  std::int16_t block = 0;
  bool latent = false;
  std::int16_t index = 0;             // row of Ψ (latent) or Θ (observed)
  double log_one_minus_r2 = 0.0;
};

// Joint: for recursive blocks, block_value = Σ terms + residual_log_det_corr
// exactly, with residual log det Corr(S_KK); for nonrecursive blocks the
// remainder is 2 log|det E_KK|. Determinacy: block_value = Σ terms +
// residual_log_det_corr exactly for every B, with residual log det Corr(V),
// and block_value = −2 (total_correlation + mutual_information).
struct MultiInfoPenaltyReport {
  PenaltyTarget target = PenaltyTarget::Joint;
  double value = 0.0;
  bool recursive = true;
  std::vector<double> block_value;
  std::vector<double> residual_log_det_corr;  // NaN if the matrix is not PD
  std::vector<MultiInfoPenaltyTerm> terms;
  std::vector<double> total_correlation;      // TC(η_L); Determinacy only, else NaN
  std::vector<double> mutual_information;     // I(η_L; y); Determinacy only, else NaN
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
  MultiInfoPenaltyReport penalty; // value may be NaN outside the barrier when weight=0
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

// Fixed moment-quadratic weight plus the same model penalty. Empty weight is
// ULS; GLS callers build normal_theory_weight once before entering this route.
// The penalty makes this a scalar problem, including for LS discrepancies.
fit_expected<PenalizedFit>
fit_gmm_multiinfo(spec::LatentStructure pt, const model::MatrixRep& rep,
                  const data::SampleStats& samp, const Eigen::VectorXd& x0,
                  gmm::Weight weight = {}, MultiInfoPenaltyOptions options = {},
                  Backend backend = Backend::NloptLbfgs,
                  optim::OptimOptions opts = ml_optim_options());

// All-ordinal association ML (ml=true) or existing ordinal LS. ML conditions
// on saturated Stage-1 thresholds; LS retains its threshold estimands. Input
// moments and weights are unchanged. Mixed/polyserial data are not supported.
fit_expected<PenalizedFit>
fit_ordinal_multiinfo(spec::LatentStructure pt, const model::MatrixRep& rep,
                      const data::OrdinalStats& stats, const Eigen::VectorXd& x0,
                      bool ml, OrdinalWeightKind weights = OrdinalWeightKind::DWLS,
                      OrdinalParameterization parameterization = OrdinalParameterization::Delta,
                      MultiInfoPenaltyOptions options = {},
                      Backend backend = Backend::NloptLbfgs,
                      optim::OptimOptions opts = {},
                      const std::vector<std::int8_t>* row_user = nullptr);

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
