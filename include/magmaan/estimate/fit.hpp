#pragma once

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using optim::LbfgsOptions;

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
};

// Optimizer backend selector for the convenience composers below.
//   Lbfgs        — L-BFGS / L-BFGS-B (the default).
//   Ceres        — Levenberg–Marquardt; LS path only, needs MAGMAAN_WITH_CERES.
//   CeresBfgs    — Ceres line-search dense BFGS; unbounded LS path only.
//   NloptSlsqp   — NLopt SLSQP (Kraft 1988); sequential quadratic programming
//                  with native box bounds. Needs MAGMAAN_WITH_NLOPT.
//   NloptBobyqa  — NLopt BOBYQA (Powell 2009); derivative-free quadratic-model
//                  trust region. Requires *finite* bounds. Needs
//                  MAGMAAN_WITH_NLOPT.
//   NloptTnewton — NLopt LD_TNEWTON_PRECOND_RESTART (Nash 1985); preconditioned
//                  truncated Newton with CG inner solve and restart. Distinct
//                  curvature scheme from L-BFGS. Needs MAGMAAN_WITH_NLOPT.
//   NloptVar2    — NLopt LD_VAR2 (Shanno-Phua 1980); *full* (dense) BFGS
//                  variable-metric. The non-limited-memory counterpart to
//                  L-BFGS; can outperform L-BFGS at SEM-sized n. Needs
//                  MAGMAAN_WITH_NLOPT.
//   NloptLbfgs   — NLopt's own L-BFGS; sanity-check against LBFGS++ as the
//                  default `Lbfgs` backend.
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
  Lbfgs,
  Ceres,
  CeresBfgs,
  NloptSlsqp,
  NloptBobyqa,
  NloptTnewton,
  NloptVar2,
  NloptLbfgs,
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
// (see `Backend` above): `Backend::Ceres` / `Backend::Nlopt*` / `Backend::Port*`
// need their build flags, and `Backend::Ceres` / `Backend::PortNls` apply to
// the least-squares (gmm) path only. `opts` tunes the optimizer (the Ceres
// path reads max_iter / ftol / gtol from it).

// Normal-theory maximum likelihood. `backend` selects the optimizer; the
// default L-BFGS, the NLopt SLSQP cross-check, or the PORT (= nlminb)
// trust-region cross-check. `Backend::Ceres` is rejected here — Ceres
// applies to the least-squares path only.
fit_expected<Estimates>
fit_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
       const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds = {},
       Backend backend = Backend::Lbfgs, LbfgsOptions opts = {});

// Native FC-SEM maximum likelihood. This is intentionally parallel to
// `fit_ml`, but uses `model::FcSemEvaluator` instead of MatrixRep/LISREL.
// Callers provide starts explicitly, typically from
// `simple_fcsem_start_values`.
fit_expected<Estimates>
fit_ml_fcsem(spec::LatentStructure pt, const SampleStats& samp,
             const Eigen::VectorXd& x0, Bounds bounds = {},
             Backend backend = Backend::Lbfgs, LbfgsOptions opts = {});

// Moment quadratic. Empty `weight` ⇒ ULS (identity); a caller-supplied weight
// ⇒ WLS / DWLS.
fit_expected<Estimates>
fit_gmm(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        gmm::Weight weight = {}, Bounds bounds = {},
        Backend backend = Backend::Lbfgs, LbfgsOptions opts = {});

// Moment quadratic with the normal-theory (GLS) weight, built once from S.
fit_expected<Estimates>
fit_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        Bounds bounds = {}, Backend backend = Backend::Lbfgs,
        LbfgsOptions opts = {});

// Golub–Pereyra profiled fit: eliminates the conditionally-linear parameters
// (Θ, Ψ, ν, α) analytically, optimizes only the nonlinear block (Λ, Β). Fails
// (NumericIssue) when the model is not separable — see `gmm::gp_compatible`.
// Empty `weight` ⇒ ULS; a caller-supplied weight ⇒ WLS / DWLS.
fit_expected<Estimates>
fit_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
          const SampleStats& samp, const Eigen::VectorXd& x0,
          gmm::Weight weight = {}, Backend backend = Backend::Lbfgs,
          LbfgsOptions opts = {});

// Golub–Pereyra profiled fit with the normal-theory (GLS) weight, built once
// from S — the profiled counterpart of `fit_gls`.
fit_expected<Estimates>
fit_snlls_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Backend backend = Backend::Lbfgs, LbfgsOptions opts = {});

}  // namespace magmaan::estimate
