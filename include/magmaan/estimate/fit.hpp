#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/pairwise_cov.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using optim::OptimOptions;

// Estimation results — pure data, NO back-pointer to the LatentStructure.
// (See plan: separation of concerns. Caller composes pt + Estimates.)
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
  // profiled out. Populated only by `fit_snlls` / `fit_snlls_gls` /
  // `fit_snlls_wls`; the default `-1` flags "no separable split applies"
  // (full-θ paths, FCSEM, FIML). Surfaced through the SNLLS R wrapper for
  // the corpus speed tables.
  std::int32_t          n_nonlinear  = -1;
  std::int32_t          n_linear     = -1;
};

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
//   NloptLbfgs   — NLopt's own L-BFGS; current default scalar backend.
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
  Ipopt,
  Port,
  PortNls,
};

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
// applies to the least-squares path only.
fit_expected<Estimates>
fit_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
       const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds = {},
       Backend backend = Backend::NloptLbfgs, OptimOptions opts = {});

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
// pairwise covariance from `data::pairwise_sample_stats`) here is the
// Σ-only-weight pairwise GLS convention (Savalei-Bentler 2005,
// Gold-Bentler-Kim 2003): the weight is `Γ_NT(Ŝ^pw)⁻¹`, ignoring that the
// asymptotic ACOV of `√n vech(Ŝ^pw − Σ)` is actually `Γ_NT^pw = Ω ∘ Γ_NT`
// with `Ω_{a,b} = π_{a,b}/(π_a π_b)`. Consistent but asymptotically
// suboptimal under MAR; trace-identity form survives because W is Σ-only.
// For the asymptotically efficient alternative — Γ_NT^pw-weighted, quadratic
// form, slightly more expensive per iteration — see `fit_gls_pairwise`
// below. Both reduce to the same fit on complete data.
fit_expected<Estimates>
fit_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        Bounds bounds = {}, Backend backend = Backend::NloptLbfgs,
        OptimOptions opts = {});

// Pairwise GLS with the Γ_NT^pw-weighted quadratic. Breaks the trace
// identity — the Hadamard product `Ω ∘` doesn't commute with the
// `2·Σ·H·Σ` operator — but uses the asymptotically efficient weight under
// MAR. Materializes `Γ_NT^pw = E_R[diag(A/π̂)·Γ_NT(Ŝ^pw)·diag(A/π̂)]` once
// (pattern-grouped over the K distinct missingness patterns,
// O(K·p*²) + O(p*³)), inverts to W^pw, hands to `fit_gmm`. Per-iteration
// cost goes from O(p³) trace to O(p⁴) quadratic — modest constant; setup
// stays outside the optimizer loop. The Σ-only counterpart is to run
// `fit_gls` above with `samp.S = pw.S`. Cov-block uses Γ_NT^pw weight;
// μ-block (when the model has a mean structure) keeps the Σ-only mean
// weight `Ŝ^pw⁻¹` until the pairwise μ ACOV gets its own treatment.
// Complete-data degeneracy: `fit_gls` and `fit_gls_pairwise` produce the
// same fit when there's no missingness.
fit_expected<Estimates>
fit_gls_pairwise(spec::LatentStructure pt, const model::MatrixRep& rep,
                 const data::RawData& raw,
                 const data::PairwiseSampleStats& pw,
                 const Eigen::VectorXd& x0,
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
