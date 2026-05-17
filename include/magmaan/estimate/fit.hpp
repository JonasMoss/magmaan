#pragma once

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
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
};

// Optimizer backend selector for the convenience composers below.
//   Lbfgs       — L-BFGS / L-BFGS-B (the default).
//   Ceres       — Levenberg–Marquardt; least-squares path only, needs MAGMAAN_WITH_CERES.
//   Nlopt       — NLopt SLSQP cross-check; needs MAGMAAN_WITH_NLOPT.
//   TrustRegion — CppNumericalSolvers Newton trust-region cross-check; unbounded only.
enum class Backend { Lbfgs, Ceres, Nlopt, TrustRegion };

// ============================================================================
// Convenience composers — the template-free core entry points.
// ============================================================================
//
// Each packages the canonical pipeline: resolve fixed.x → build the model
// evaluator → build the objective → fold equality constraints → optimize →
// expand to full θ. `x0` (size pt.n_free()) is the caller-supplied start
// vector; an empty `bounds` means unbounded. `backend` selects the optimizer
// (see `Backend` above): `Backend::Ceres` / `Backend::Nlopt` need their build
// flags, `Backend::Ceres` applies to the least-squares (gmm) path only, and
// `Backend::TrustRegion` requires empty bounds. `opts` tunes the optimizer
// (the Ceres path reads max_iter / ftol / gtol from it).

// Normal-theory maximum likelihood. `backend` selects the optimizer; the
// default L-BFGS, the NLopt SLSQP cross-check, or the unbounded trust-region
// cross-check. `Backend::Ceres` is rejected here — Ceres applies to the
// least-squares path only.
fit_expected<Estimates>
fit_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
       const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds = {},
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
