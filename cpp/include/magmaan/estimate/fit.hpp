#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/frontier/dls_weight.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/pairwise_cov.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/estimate/ml_numerics.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using optim::OptimOptions;

// Estimation results — pure data, NO back-pointer to the LatentStructure.
// (See plan: separation of concerns. Caller composes pt + Estimates.)
enum class Backend;  // defined below; Estimates records a substituted backend

struct Estimates {
  Eigen::VectorXd theta;     // size = pt.n_free()
  double          fmin       = 0.0;
  int             iterations = 0;
  // Optimizer evaluation counts: `f_evals` = times the optimizer requested an
  // objective value, `g_evals` = times it requested a gradient. PORT counts
  // these separately; NLopt's gradient algorithms request both jointly, so
  // there `f_evals == g_evals`.
  int             f_evals = 0;
  int             g_evals = 0;
  // Refined optimizer termination status and the (projected) gradient
  // infinity-norm at the solution. The defaults describe an exact/closed-form
  // solve where no outer optimizer ran; `grad_inf_norm < 0` means not computed.
  optim::OptimStatus optimizer_status = optim::OptimStatus::Converged;
  double             grad_inf_norm    = -1.0;
  // Terminal audit at the optimizer's returned iterate (driven coordinates).
  // Filled by the wrapping `compose_*` paths; default-constructed for closed-
  // form or hard-failure paths that never invoke an optimizer. The `= {}`
  // keeps existing `Estimates{...}` aggregate inits passing under
  // `-Wmissing-field-initializers`. See `optim::TerminalAudit`.
  optim::TerminalAudit  audit        = {};
  // Fit finalization diagnostics on full θ (implied-Σ PD per group, linear /
  // nonlinear equality residuals, bounds active set). See `FitDiagnostics`.
  FitDiagnostics        diagnostics  = {};
  // SNLLS profile shape: `n_nonlinear` is the β-block size the outer
  // optimizer sees, `n_linear` is the α-block size that Golub–Pereyra
  // profiled out. The Fisher-SNLLS path also uses these fields for its local
  // Schur-complement β/α split. The default `-1` flags "no separable split
  // applies" (full-θ paths, FCSEM, FIML).
  std::int32_t          n_nonlinear  = -1;
  std::int32_t          n_linear     = -1;
  // SNLLS inner-solve telemetry: number of `profile_at()` cache misses
  // that took the fast Cholesky-on-normal-equations path vs the
  // rank-revealing QR fallback. Populated alongside `n_nonlinear` /
  // `n_linear`; the default `-1` flags "no separable split applies".
  // See `project/design/snlls-fast-alpha-solve.md` for the gate semantics.
  std::int32_t          n_alpha_solve_fast     = -1;
  std::int32_t          n_alpha_solve_fallback = -1;
  // Search coordinates the optimizer actually used (None when unscaled).
  optim::CoordinateScaling coordinate_scaling = optim::CoordinateScaling::None;
  // Set when the requested backend could not take the model's nonlinear
  // equality constraints and this backend ran instead.
  std::optional<Backend> substituted_backend = {};
  // theta is in caller units; diagnostics/terminal audit describe the
  // normalized fitting model when this flag is true.
  bool sample_normalized = false;

};

// Consumers use this common verdict; optimizer_status explains termination.
inline FitVerdict fit_verdict(const Estimates& estimates) {
  return common_fit_verdict(estimates.diagnostics);
}

// Optimizer backend selector for the convenience composers below.
//   Ceres        — Levenberg–Marquardt; LS path only, needs MAGMAAN_WITH_CERES.
//   CeresBfgs    — Ceres line-search dense BFGS; unbounded LS path only.
//   NloptSlsqp   — NLopt SLSQP (Kraft 1988); sequential quadratic programming
//                  with native box bounds and nonlinear equality constraints
//                  on scalar ML/LS/FIML paths.
//   NloptBobyqa  — NLopt BOBYQA (Powell 2009); derivative-free quadratic-model
//                  trust region. Requires *finite* bounds.
//   NloptTnewton — NLopt LD_TNEWTON_PRECOND_RESTART (Nash 1985); preconditioned
//                  truncated Newton with CG inner solve and restart. Distinct
//                  curvature scheme from L-BFGS.
//   NloptVar2    — NLopt LD_VAR2 (Shanno-Phua 1980); *full* (dense) BFGS
//                  variable-metric. The non-limited-memory counterpart to
//                  L-BFGS; can outperform L-BFGS at SEM-sized n.
//   NloptLbfgs   — NLopt's own L-BFGS; scalar line-search backend.
//   NloptLbfgsSlsqpFallback — policy backend: L-BFGS first, retry SLSQP when
//                  L-BFGS fails or returns a non-clean status.
//   Ipopt        — IPOPT interior-point backend with limited-memory Hessian
//                  approximation. Needs MAGMAAN_WITH_IPOPT. General scalar
//                  backend and an optional nonlinear-equality constraint
//                  backend alongside NLopt SLSQP.
//   Port         — PORT drmngb model-Hessian trust region with bounds (the
//                  algorithm behind R's `nlminb`; TOMS 611). Needs
//                  MAGMAAN_WITH_PORT (default ON). Replaces the now-retired
//                  CppNumericalSolvers-backed TrustRegion entry.
//   PortNls      — PORT drn2gb NL2SOL adaptive trust region with bounds (the
//                  algorithm behind R's `nls`; TOMS 573 Dennis-Gay-Welsch).
//                  LS path only — sees the multi-residual structure directly
//                  rather than the scalarised ½‖r‖² collapse. Needs
//                  MAGMAAN_WITH_PORT.
enum class Backend {
  Ceres,
  CeresBfgs,
  NloptSlsqp,
  NloptBobyqa,
  NloptTnewton,
  NloptVar2,
  NloptLbfgs,
  NloptLbfgsSlsqpFallback,
  Ipopt,
  Port,
  PortNls,
};

// Which second-stage weight's data influence the continuous-LS IJ blocks carry.
// `Fixed` means caller-fixed weight (no correction; ULS and pre-supplied weights);
// the sample modes rebuild the weight from raw data and add IF(W-hat) rows.
enum class ContinuousLsIJWeightMode {
  Fixed,
  SampleNormalTheory,   // GLS: W from the sample S
  SampleEmpiricalWls,   // WLS/ADF: dense Browne empirical Gamma-hat inverse
  SampleEmpiricalDwls,  // DWLS: diagonal Browne empirical Gamma-hat inverse
  SampleDls,            // DLS: mixed (1-a)Gamma_NT + a Gamma_ADF
};

namespace frontier {

// Programmatic nonlinear equality constraints, evaluated in the full free
// parameter vector θ. This is the frontier counterpart to the partable
// nonlinear-`==` rows: callers can append constraints such as g(θ) = g0 where
// g is a matrix functional that the lavaan-style expression DSL cannot spell.
struct ExtraNonlinearEqConstraints {
  optim::ConstraintFn    h;             // residuals, length n_constraint
  optim::ConstraintJacFn jacobian;      // n_constraint × n_free
  Eigen::Index           n_constraint = 0;

  bool active() const noexcept { return n_constraint > 0; }
};

// Controls for the frontier covariance-honest parameterization. The
// optimizer drives internal Cholesky factors for every covariance-connected
// component of Θ and Ψ and links them exactly to the ordinary partable
// parameters.
// `start_eigen_floor` is used only to construct a valid objective start; it is
// not a lower bound at the solution, where PSD boundary points remain allowed.
struct PsdFitOptions {
  double start_eigen_floor = 1e-6;
  double feasibility_tol = 1e-6;
  // Complete-data ML only. Freeze blockwise diagonal expected-information
  // scaling at the lifted start; final audits remain in original coordinates.
  bool diagonal_preconditioning = false;
};

inline OptimOptions ml_psd_optim_options() {
  auto out = ml_optim_options();
  out.nlopt.constraint_tol = 1e-8;
  return out;
}

inline PsdFitOptions ml_psd_options() {
  PsdFitOptions out;
  out.diagonal_preconditioning = true;
  return out;
}

// Scalar function used by profile-LR helpers. `value(theta)` returns g(θ).
// `gradient(theta)` may be left empty; the implementation then uses a central
// finite-difference fallback in θ-space.
struct ScalarFunctional {
  std::function<double(const Eigen::VectorXd&)> value;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradient;
};

struct ScalarProfileLrtResult {
  Estimates constrained;
  double unrestricted_value = 0.0;
  double constrained_value = 0.0;
  double target = 0.0;
  double constraint_residual = 0.0;
  double fmin_unrestricted = 0.0;  // optimizer scale, 0.5 * F
  double fmin_constrained = 0.0;
  double T = 0.0;                  // 2N * (fmin_constrained - fmin_unrestricted)
  double p_value = 1.0;            // ordinary χ²_1 reference; robust/Bartlett layers sit above
  double scaling_factor = std::numeric_limits<double>::quiet_NaN();
  double T_scaled = std::numeric_limits<double>::quiet_NaN();
  double p_value_scaled = std::numeric_limits<double>::quiet_NaN();
  double misspec_scaling_factor = std::numeric_limits<double>::quiet_NaN();
  double T_misspec_scaled = std::numeric_limits<double>::quiet_NaN();
  double p_value_misspec_scaled = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd misspec_eigvals;
  double p_value_misspec_mixture = std::numeric_limits<double>::quiet_NaN();
  double misspec_mixture_cutoff = std::numeric_limits<double>::quiet_NaN();
  double n_obs = 0.0;
  int df = 1;
};

enum class ScalarProfileReference {
  Ordinary,
  RobustScaled,
  MisspecScaled,
  MisspecMixture,
};

struct GmmProfileRobustOptions {
  bool estimated_weight = false;
  ContinuousLsIJWeightMode ij_weight_mode = ContinuousLsIJWeightMode::Fixed;
  frontier::DlsWeightOptions dls_opts{};
};

struct ScalarProfileCiOptions {
  double confidence_level = 0.95;
  double cutoff = std::numeric_limits<double>::quiet_NaN();
  double lower_bound = std::numeric_limits<double>::quiet_NaN();
  double upper_bound = std::numeric_limits<double>::quiet_NaN();
  double initial_step = std::numeric_limits<double>::quiet_NaN();
  double target_tol = 1e-5;
  double statistic_tol = 1e-6;
  int max_iter = 60;
  int max_expand = 40;
  ScalarProfileReference reference = ScalarProfileReference::Ordinary;
};

struct ScalarProfileCiResult {
  ScalarProfileLrtResult lower_profile;
  ScalarProfileLrtResult upper_profile;
  double estimate = 0.0;
  double lower = 0.0;
  double upper = 0.0;
  double confidence_level = 0.95;
  double cutoff = 0.0;
  double lower_cutoff = 0.0;
  double upper_cutoff = 0.0;
  int lower_evals = 0;
  int upper_evals = 0;
  bool lower_at_bound = false;
  bool upper_at_bound = false;
};

}  // namespace frontier

// ============================================================================
// Convenience composers — the template-free core entry points.
// ============================================================================
//
// Each packages the canonical pipeline: resolve fixed.x → build the model
// evaluator → build the objective → fold equality constraints → optimize →
// expand to full θ. `x0` (size pt.n_free()) is the caller-supplied start
// vector; an empty `bounds` means unbounded. `backend` selects the optimizer
// (see `Backend` above). `opts` tunes the optimizer; Ceres reads max_iter /
// ftol / gtol from it when the LS path dispatches there.

// Normal-theory maximum likelihood. `backend` selects the optimizer; the
// default NLopt L-BFGS, the NLopt SLSQP cross-check, or the PORT (= nlminb)
// trust-region cross-check. `Backend::Ceres` is rejected here because Ceres
// applies to the least-squares path only. A model with nonlinear equality
// constraints runs on NLopt SLSQP when `backend` cannot take them (fit_gls
// likewise), and the fit records the substitution in `substituted_backend`.
// Single-level complete-data models with linear equalities (including groups)
// use sample-normalized model/data coordinates unless opts.normalize_sample is
// false. x0, bounds and returned theta are always in caller coordinates;
// optimizer tolerances and diagnostics refer to the internal representation.
fit_expected<Estimates>
fit_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
       const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds = {},
       Backend backend = Backend::NloptLbfgs, OptimOptions opts = ml_optim_options());

namespace frontier {

// Normal-theory ML with caller-supplied nonlinear equality constraints appended
// to any partable-derived nonlinear constraints. The optimizer layer is the
// same constrained scalar path used by ordinary nonlinear `==` rows.
fit_expected<Estimates>
fit_ml_constrained(spec::LatentStructure pt, const model::MatrixRep& rep,
                   const SampleStats& samp, const Eigen::VectorXd& x0,
                   ExtraNonlinearEqConstraints extra, Bounds bounds = {},
                   Backend backend = Backend::NloptSlsqp,
                   OptimOptions opts = {});

// Complete-data normal-theory ML over the LISREL-honest parameter space:
// every primitive covariance block Θ_b and Ψ_b is positive semidefinite, while
// the implied Σ_b remains positive definite through the ordinary ML domain.
// The returned θ has the original partable dimension and semantics; Cholesky
// variables are internal. Partable linear/nonlinear equality constraints are
// honored. Explicit box bounds are intentionally not part of this first
// frontier slice because they live in θ-space, not the lifted coordinates.
fit_expected<Estimates>
fit_ml_psd(spec::LatentStructure pt, const model::MatrixRep& rep,
           const SampleStats& samp, const Eigen::VectorXd& x0,
           Backend backend = Backend::NloptSlsqp,
           OptimOptions opts = ml_psd_optim_options(),
           PsdFitOptions psd_opts = ml_psd_options());

// Fixed-weight moment-quadratic estimation over the same LISREL-honest
// covariance domain. Empty `weight` gives ULS; a caller-supplied fixed weight
// gives WLS/ADF/DWLS on the continuous [mean; vech(covariance)] moment stack.
// The returned parameter vector retains the ordinary partable coordinates.
fit_expected<Estimates>
fit_gmm_psd(spec::LatentStructure pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Eigen::VectorXd& x0,
            gmm::Weight weight = {},
            Backend backend = Backend::NloptSlsqp,
            OptimOptions opts = {}, PsdFitOptions psd_opts = {});

// Normal-theory GLS specialization: constructs the ordinary sample-based,
// fixed GLS weight once, then solves the covariance-honest GMM problem.
fit_expected<Estimates>
fit_gls_psd(spec::LatentStructure pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Eigen::VectorXd& x0,
            Backend backend = Backend::NloptSlsqp,
            OptimOptions opts = {}, PsdFitOptions psd_opts = {});

// Profile-LR test of H0: g(θ) = target against the already-fitted unrestricted
// ML estimate. This is intentionally only the ordinary χ²_1 reference; robust
// Satorra scaling and small-sample/Bartlett factors are separate policies.
fit_expected<ScalarProfileLrtResult>
profile_lrt_scalar_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
                      const SampleStats& samp,
                      const Estimates& unrestricted,
                      ScalarFunctional functional, double target,
                      Bounds bounds = {},
                      Backend backend = Backend::NloptSlsqp,
                      OptimOptions opts = {},
                      double constraint_tol = 1e-6,
                      const data::RawData* robust_raw = nullptr,
                      ScalarProfileReference reference =
                          ScalarProfileReference::RobustScaled);

fit_expected<ScalarProfileLrtResult>
profile_lrt_parameter_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
                         const SampleStats& samp,
                         const Estimates& unrestricted,
                         Eigen::Index parameter, double target,
                         Bounds bounds = {},
                         Backend backend = Backend::NloptSlsqp,
                         OptimOptions opts = {},
                         double constraint_tol = 1e-6,
                         const data::RawData* robust_raw = nullptr,
                         ScalarProfileReference reference =
                             ScalarProfileReference::RobustScaled);

// Moment-quadratic analogues for a caller-fixed GMM/LS weight. These are the
// first functional profile-LRT building blocks outside ML; robust Satorra /
// sandwich scaling and Bartlett factors remain explicit policy layers above.
fit_expected<Estimates>
fit_gmm_constrained(spec::LatentStructure pt, const model::MatrixRep& rep,
                    const SampleStats& samp, const Eigen::VectorXd& x0,
                    gmm::Weight weight, ExtraNonlinearEqConstraints extra,
                    Bounds bounds = {},
                    Backend backend = Backend::NloptSlsqp,
                    OptimOptions opts = {});

fit_expected<ScalarProfileLrtResult>
profile_lrt_scalar_gmm(spec::LatentStructure pt, const model::MatrixRep& rep,
                       const SampleStats& samp,
                       const Estimates& unrestricted,
                       gmm::Weight weight,
                       ScalarFunctional functional, double target,
                       Bounds bounds = {},
                       Backend backend = Backend::NloptSlsqp,
                       OptimOptions opts = {},
                       double constraint_tol = 1e-6,
                       const data::RawData* robust_raw = nullptr,
                       GmmProfileRobustOptions robust_options = {},
                       ScalarProfileReference reference =
                           ScalarProfileReference::RobustScaled);

fit_expected<ScalarProfileLrtResult>
profile_lrt_parameter_gmm(spec::LatentStructure pt, const model::MatrixRep& rep,
                          const SampleStats& samp,
                          const Estimates& unrestricted,
                          gmm::Weight weight,
                          Eigen::Index parameter, double target,
                          Bounds bounds = {},
                          Backend backend = Backend::NloptSlsqp,
                          OptimOptions opts = {},
                          double constraint_tol = 1e-6,
                          const data::RawData* robust_raw = nullptr,
                          GmmProfileRobustOptions robust_options = {},
                          ScalarProfileReference reference =
                              ScalarProfileReference::RobustScaled);

fit_expected<ScalarProfileCiResult>
profile_lrt_ci_parameter_ml(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const Estimates& unrestricted,
                            Eigen::Index parameter,
                            ScalarProfileCiOptions ci_options = {},
                            Bounds bounds = {},
                            Backend backend = Backend::NloptSlsqp,
                            OptimOptions opts = {},
                            double constraint_tol = 1e-6,
                            const data::RawData* robust_raw = nullptr);

fit_expected<ScalarProfileCiResult>
profile_lrt_ci_parameter_gmm(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const SampleStats& samp,
                             const Estimates& unrestricted,
                             gmm::Weight weight,
                             Eigen::Index parameter,
                             ScalarProfileCiOptions ci_options = {},
                             Bounds bounds = {},
                             Backend backend = Backend::NloptSlsqp,
                             OptimOptions opts = {},
                             double constraint_tol = 1e-6,
                             const data::RawData* robust_raw = nullptr,
                             GmmProfileRobustOptions robust_options = {});

}  // namespace frontier

// Normal-theory ML via local Fisher scoring. At each iterate this computes the
// analytic ML gradient and the expected-information Hessian approximation on
// the F_ML scale, solves H_E(θ_k) d = -∇F_ML(θ_k), then accepts the step with
// Armijo backtracking on the true ML objective. Unlike `fit_ml_irls`, this is
// a local Fisher step rather than a frozen nonlinear GLS reoptimization.
// `opts.max_iter`, `opts.ftol`, and `opts.gtol` tune the outer scoring loop.
fit_expected<Estimates>
fit_ml_fisher(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Bounds bounds = {}, OptimOptions opts = {});

// Fisher-ML with a separable block solve. The scoring equation is still the
// same expected-information ML system as `fit_ml_fisher`, but the local
// quadratic is partitioned into nonlinear β (Λ, Β) and conditionally-linear α
// (Θ, Ψ, ν, α) coordinates and solved through the Schur complement. This is a
// local Fisher block elimination, not Golub-Pereyra objective profiling.
fit_expected<Estimates>
fit_ml_fisher_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
                    const SampleStats& samp, const Eigen::VectorXd& x0,
                    Bounds bounds = {}, OptimOptions opts = {});

// Outer-loop controls for `fit_ml_irls`. `max_outer` caps the number of
// reweight steps; `ftol` and `gtol` decide convergence on, respectively, the
// relative change in F_ML and the (projected) F_ML gradient infinity norm.
// `armijo_c` is the sufficient-decrease constant for the backtracking line
// search that guards each accepted step; `armijo_max_backtracks` caps the
// halvings before the loop bails as `LineSearchSalvaged`.
struct IrlsOptions {
  int    max_outer             = 50;
  double ftol                  = 1e-10;
  double gtol                  = 1e-7;
  double armijo_c              = 1e-4;
  int    armijo_max_backtracks = 20;
};

// Normal-theory ML via iteratively reweighted GLS (Fisher scoring). Same ML
// objective as `fit_ml`, different algorithm: at iterate θ_k the GLS weight
// W(θ_k) = ½ D'(Σ(θ_k)⁻¹ ⊗ Σ(θ_k)⁻¹) D (the expected Fisher information block)
// is built from the *current* implied covariance, then an inner GLS solve via
// `backend` proposes θ_trial; backtracking on F_ML accepts a damped step. With
// mean structures, the frozen inner covariance target is adjusted by the
// current mean residual d_k d_k' so the inner score matches the ML score up to
// a constant factor. Linear equality constraints are handled in reduced
// coordinates; nonlinear equality constraints are rejected. `backend` names
// the *inner* LS solver — `PortNls` is the natural default (NL2SOL sees the
// residual structure directly); any LS-shape backend works. `opts` tunes the
// inner solve; `irls_opts` tunes the outer loop.
fit_expected<Estimates>
fit_ml_irls(spec::LatentStructure pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Eigen::VectorXd& x0,
            Bounds bounds = {}, Backend backend = Backend::PortNls,
            OptimOptions opts = {}, IrlsOptions irls_opts = {});

// SNLLS-flavoured IRLS-ML: outer Fisher reweight on Σ(θ), inner step solves
// the GLS subproblem with Golub–Pereyra variable projection (β = Λ, B
// optimized; α = Θ, Ψ, ν closed-form). Same convergence story and outer-loop
// safeguards as `fit_ml_irls`; rejects box bounds, nonlinear constraints, and
// the non-separable models `gmm::gp_compatible` rejects. Spirit matches
// Kreiberg's Matlab SNLRLS reference (`external/kreiberg/snlrls/`); our
// analytic dΣ/dθ Jacobian and Armijo line search are local additions.
fit_expected<Estimates>
fit_ml_irls_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
                  const SampleStats& samp, const Eigen::VectorXd& x0,
                  Bounds bounds = {}, Backend backend = Backend::PortNls,
                  OptimOptions opts = {}, IrlsOptions irls_opts = {});

// Native FC-SEM maximum likelihood. This is intentionally parallel to
// `fit_ml`, but uses `model::FcSemEvaluator` instead of MatrixRep/LISREL.
// Callers provide starts explicitly, typically from
// `simple_fcsem_start_values`.
fit_expected<Estimates>
fit_ml_fcsem(spec::LatentStructure pt, const SampleStats& samp,
             const Eigen::VectorXd& x0, Bounds bounds = {},
             Backend backend = Backend::NloptLbfgs, OptimOptions opts = {});

// Moment quadratic. Empty `weight` ⇒ ULS (identity); a caller-supplied weight
// ⇒ WLS / DWLS.
fit_expected<Estimates>
fit_gmm(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        gmm::Weight weight = {}, Bounds bounds = {},
        Backend backend = Backend::NloptLbfgs, OptimOptions opts = {});

// Moment quadratic with the normal-theory (GLS) weight, built once from S.
//
// Pairwise / incomplete-data note: handing `samp.S = Ŝ^pw` (Van Praag
// pairwise covariance from `data::pairwise_sample_stats`) here uses the
// ordinary sample-based GLS metric. Pairwise moments are estimator-independent
// inputs under MCAR. A missingness-adjusted fixed metric can instead be built
// from `data::gamma_nt_pairwise` and supplied to `fit_gmm`; post-fit inference
// must account for the pairwise moments' sampling law.
fit_expected<Estimates>
fit_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        Bounds bounds = {}, Backend backend = Backend::NloptLbfgs,
        OptimOptions opts = {});

// Golub–Pereyra profiled fit: eliminates the conditionally-linear parameters
// (Θ, Ψ, ν, α) analytically, optimizes only the nonlinear block (Λ, Β). Fails
// (NumericIssue) when the model is not separable — see `gmm::gp_compatible`.
// Empty `weight` ⇒ ULS; a caller-supplied weight ⇒ WLS / DWLS.
fit_expected<Estimates>
fit_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
          const SampleStats& samp, const Eigen::VectorXd& x0,
          gmm::Weight weight = {}, Backend backend = Backend::NloptLbfgs,
          OptimOptions opts = {});

// Golub–Pereyra profiled fit with the normal-theory (GLS) weight, built once
// from S — the profiled counterpart of `fit_gls`.
fit_expected<Estimates>
fit_snlls_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Backend backend = Backend::NloptLbfgs, OptimOptions opts = {});

}  // namespace magmaan::estimate
