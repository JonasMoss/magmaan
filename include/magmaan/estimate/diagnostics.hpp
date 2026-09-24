#pragma once

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"  // Bounds, ActiveBoundDiagnostics

// Fit finalization audit (Layer 2).
//
// L1 (`optim::audit_terminal_iterate`) diagnoses first-order stationarity in
// the *driven* coordinate system the optimizer minimized over. L2 lives on the
// other side of `prob.expand(out->x)`: it operates on full θ and answers a
// different question — "is this fit usable for downstream inference?" — by
// recording the authoritative common numerical verdict and the diagnostics
// that SE/χ²/robust-correction code needs to decide
// whether its formulae apply. L2 never blocks a fit; it records.
//
// Concretely:
//   - Implied-Σ positive-definiteness per group. ML/GLS need Σ⁻¹; ULS does
//     not. Consumers decide what they need.
//   - Linear equality constraint residual. The K-reparameterization is exact
//     in the pure-merge case (residual to roundoff) and tiny under
//     general-linear constraints. A non-tiny residual means the expansion
//     itself is broken — a hard correctness signal worth surfacing.
//   - Nonlinear equality constraint residual at θ̂. IPOPT drives
//     `h(θ̂) → 0`; recording the achieved infinity-norm tells the user
//     whether the nonlinear-constraint solve actually converged.
//   - Active-bound set on the FULL expanded θ — this is the Heywood-case
//     detector: a variance parameter at its 0 bound has a one-sided
//     derivative, and the standard info-matrix SE for it is not valid.
//
// Forward-declarations keep the header light; the cpp pulls in the model and
// constraint headers it actually needs.

namespace magmaan::model {
class ModelEvaluator;
}
namespace magmaan::spec {
struct LatentStructure;
}
namespace magmaan::estimate {
struct EqConstraints;
struct NonlinearEqConstraints;
}

namespace magmaan::estimate {

// Admissibility of one assembled covariance matrix. `block` and every row
// index are 0-based in C++; the R boundary converts them to 1-based indices.
// `covariance_rows` contains every source partable row that writes this matrix
// block, so a failed joint PSD check remains traceable even when no individual
// variance or correlation explains the failure.
struct CovarianceBlockDiagnostics {
  std::int32_t             block = -1;
  double                   min_eigenvalue = 0.0;
  bool                     finite = true;
  bool                     psd = true;
  bool                     positive_definite = true;
  std::vector<std::int32_t> covariance_rows;
  std::vector<std::int32_t> negative_variance_rows;
  std::vector<std::int32_t> invalid_correlation_rows;
};

// Estimator-neutral post-fit covariance admissibility. The primitive
// reduced-LISREL covariance matrices are Ψ (latent innovations/disturbances)
// and Θ (observed residuals). They are admissible when PSD; the implied
// observed Σ must additionally be PD for the ordinary Gaussian likelihood.
//
// Singular-but-PSD Ψ/Θ blocks remain admissible covariance matrices. Their
// `positive_definite` field is false so consumers can identify boundary cases
// without conflating them with an indefinite matrix.
struct AdmissibilityDiagnostics {
  bool checked = false;
  bool covariance_matrices_psd = false;
  bool implied_sigma_pd = false;
  bool admissible = false;
  std::vector<CovarianceBlockDiagnostics> theta_blocks;
  std::vector<CovarianceBlockDiagnostics> psi_blocks;
};

// First-order stationarity in the ordinary full-model representation. Unlike
// `optim::TerminalAudit`, this diagnostic never sees optimizer-driven or
// Cholesky coordinates. It measures the objective differential against the
// tangent/normal geometry of the original linear equalities, nonlinear-
// equality tangent space, box bounds, and primitive PSD covariance cones.
//
// The scalar residual still needs a metric. `model_frobenius` is the product
// Frobenius metric induced by the LISREL matrices: symmetric off-diagonal
// covariance entries receive weight 2, ordinary matrix/vector entries weight
// 1, and repeated/shared coordinates accumulate their contributions. This is
// invariant to a change of parameter basis when the metric is transformed
// with the model map, and to orthogonal changes of basis within covariance
// blocks. It is deliberately not advertised as invariant to arbitrary changes
// of measurement units.
struct GeometricStationarityDiagnostics {
  bool checked = false;
  bool gradient_finite = false;
  bool feasible = false;
  bool covariance_feasible = false;

  // Equality/bound normal cone only. This is the common-full-model companion
  // to the driven/lavaan-compatible audit and excludes PSD normal multipliers.
  // The stationary verdict thresholds the metric-dual L2 residual; the
  // coordinatewise infinity norm is retained only as a familiar readout.
  bool ambient_stationary = false;
  double ambient_residual_inf = -1.0;
  double ambient_residual_l2 = -1.0;

  // Equality/bound normals plus the true PSD-cone normals in the null spaces
  // of singular primitive covariance blocks.
  bool cone_stationary = false;
  double cone_residual_inf = -1.0;
  double cone_residual_l2 = -1.0;

  double raw_gradient_inf = -1.0;
  double stationarity_tol = 1e-3;
  double covariance_eigen_tol = 1e-8;
  std::int32_t covariance_active_blocks = 0;
  std::int32_t covariance_nullity = 0;

  bool ambient_projection_converged = false;
  bool cone_projection_converged = false;
  std::int32_t ambient_projection_iterations = 0;
  std::int32_t cone_projection_iterations = 0;
};

// Local accuracy of a returned complete-data ML fit (Layer 2). With G the
// total score and I the total observed information, both reduced by the
// linear-equality basis, the Newton distance is d = sqrt(G' I^{-1} G): under
// the local quadratic approximation, the largest predicted Newton correction
// of any linear contrast in its information-based standard errors. The
// accepted budget is d <= .01 (docs/research/interior-newton-audit.md). It is
// a local accuracy approximation, not a bound on the distance to an optimum.
// Computed by `frontier::newton_accuracy_ml`; complete-data ML fit paths
// attach it (`checked`), and `common_fit_verdict` uses it at regular interior
// points. The conditioning and solve guards are numerical safeguards, not
// identification tests.
enum class NewtonAccuracyStatus : std::uint8_t {
  Available,             // d computed
  Unavailable,           // objective, gradient, constraints or information failed
  Unsupported,           // nonlinear equality constraints (needs the Lagrangian)
  NonpositiveCurvature,  // reduced observed information not positive definite
  IllConditioned,        // equilibrated condition number above max_condition
  SolveUnreliable,       // relative linear-solve residual above max_solve_residual
};

std::string_view to_string(NewtonAccuracyStatus s) noexcept;

struct NewtonAccuracyOptions {
  double budget = 0.01;
  double max_condition = 1e12;
  double max_solve_residual = 1e-10;
  // A primitive covariance block counts as interior when its smallest
  // eigenvalue exceeds this multiple of its largest absolute eigenvalue.
  double interior_eigen_tol = 1e-8;
};

struct NewtonAccuracyDiagnostics {
  // False on fit paths that do not compute the diagnostic (non-ML
  // discrepancies, FIML, ordinal, two-level, penalized fits).
  bool checked = false;
  NewtonAccuracyStatus status = NewtonAccuracyStatus::Unavailable;
  double distance = std::numeric_limits<double>::quiet_NaN();
  // d^2 / 2: predicted remaining decrease of the total negative log likelihood.
  double predicted_gain = std::numeric_limits<double>::quiet_NaN();
  // Largest absolute predicted Newton correction in reduced coordinates.
  double max_step = std::numeric_limits<double>::quiet_NaN();
  double condition = std::numeric_limits<double>::quiet_NaN();
  double solve_residual = std::numeric_limits<double>::quiet_NaN();
  double budget = 0.01;
  bool passed = false;  // status == Available && distance <= budget
  bool covariance_interior = false;
  std::int32_t n_reduced = 0;
  // Covariance-domain version (`newton_accuracy_ml_psd`): the Newton step is
  // restricted to the face of the PSD cone that the estimate lies on. Null
  // directions of the primitive blocks whose multiplier is positive stay
  // null (U' dC U = 0), the face's curvature 2 tr(M dC C^+ dC) is added to
  // the observed information, and d is computed on the tangent space. Null
  // directions with a zero or negative multiplier stay free. Without null
  // directions this is the interior check.
  bool psd_domain = false;
  std::int32_t null_directions = 0;         // numerical nullity, structural zeros excluded
  std::int32_t constrained_directions = 0;  // null directions held on the face
  // Smallest multiplier eigenvalue over null directions, on the total
  // negative-log-likelihood scale per unit of covariance. NaN without null
  // directions.
  double min_multiplier = std::numeric_limits<double>::quiet_NaN();
};

// The domain is declared by the fit entry point, never selected by which
// residual happens to pass. Backend status is not an input to this verdict.
enum class FitCheck { Unchecked, Passed, Failed };
enum class StationarityDomain { Ambient, Psd };
// Which check decided `FitVerdict::stationarity`. `Newton` is the accuracy
// check at a regular interior point of the fitting domain; `FirstOrder` is the
// metric-dual residual (ambient, or cone at a PSD boundary).
enum class StationarityCriterion { FirstOrder, Newton };

struct ObjectiveDiagnostics {
  bool checked = false;
  bool finite = false;
  bool consistent = false;
  double recomputed = std::numeric_limits<double>::quiet_NaN();
  double reported = std::numeric_limits<double>::quiet_NaN();
  double consistency_tolerance = std::numeric_limits<double>::quiet_NaN();
  // Multiply the native reported/recomputed objective AND gradient by this
  // factor to reach the common per-observation half-discrepancy scale.
  double multiplier = 1.0;
};

struct FitVerdict {
  FitCheck status = FitCheck::Unchecked;
  FitCheck objective = FitCheck::Unchecked;
  FitCheck stationarity = FitCheck::Unchecked;
  StationarityDomain domain = StationarityDomain::Ambient;
  StationarityCriterion criterion = StationarityCriterion::FirstOrder;
};

struct FitDiagnostics {
  // Implied Σ Cholesky per group; `sigma_pd_all` is the && over the vector.
  // ML/GLS need PD; ULS only needs finite. Empty when the evaluator could
  // not compute Σ at all (e.g. non-finite θ propagated through model
  // evaluation) — `sigma_pd_all` then defaults to false.
  std::vector<bool>      sigma_pd_per_block;
  bool                   sigma_pd_all = false;

  // Linear equality constraint residual at θ̂ = max |(A_eq · θ - b_eq)_k|.
  // 0 when the model has no linear equality constraints (`EqConstraints::active() == false`).
  double                 lin_eq_residual_inf = 0.0;
  bool                   lin_eq_satisfied    = true;

  // Nonlinear equality constraint residual at θ̂ = h(θ̂). Empty when there
  // are no nonlinear constraints. `_inf` is the infinity-norm; `_satisfied`
  // is `_inf <= nl_eq_residual_tol`.
  Eigen::VectorXd        nl_eq_residual;
  double                 nl_eq_residual_inf  = 0.0;
  bool                   nl_eq_satisfied     = true;

  // Which expanded-θ coordinates sit on a finite bound (the Heywood-case
  // diagnostic). Empty when `bounds.empty()` or no parameter is active.
  ActiveBoundDiagnostics active_bounds_full;

  // Covariance-domain audit on the assembled reduced-LISREL matrices.
  AdmissibilityDiagnostics admissibility;

  // Common-coordinate / PSD-cone stationarity diagnostic. It remains
  // unchecked on fit paths that have not supplied an ordinary full-θ
  // objective gradient; `fit$audit` is never overwritten.
  GeometricStationarityDiagnostics geometric_stationarity;

  // Interior accuracy check for complete-data ML fits; unchecked elsewhere.
  NewtonAccuracyDiagnostics newton_accuracy;

  // SNLLS-only: did the gp expand take the affine fallback `θ₀ + K_β·β`
  // because `profiled(β)` returned an error? v1 leaves this `false` —
  // wiring it requires a flag on `GpProblem`. Recorded here so consumers
  // (and a follow-up PR) have a stable schema slot.
  bool                   snlls_profile_fallback = false;

  ObjectiveDiagnostics objective = {};
  StationarityDomain stationarity_domain = StationarityDomain::Ambient;
};

struct DiagnosticsOptions {
  // Active-bound coordinate tolerance — matches the existing
  // `estimate::active_bounds` default (a parameter is "active" if it sits
  // within `1e-6` of a finite bound).
  double active_bound_tol    = 1e-6;
  // The K reparameterization is exact under pure-merge constraints and at
  // worst orthonormal-roundoff under general-linear ones. A residual
  // exceeding 1e-8 is a correctness signal that the expansion machinery,
  // not the optimizer, is misbehaving.
  double lin_eq_residual_tol = 1e-8;
  // Nonlinear equality feasibility target.
  double nl_eq_residual_tol  = 1e-6;
  // Scale-relative tolerances for declaring a symmetric covariance matrix
  // PSD/PD, a variance negative, or a defined correlation outside [-1, 1].
  double covariance_eigen_tol = 1e-8;
  double variance_tol         = 1e-8;
  double correlation_tol      = 1e-8;
};

struct GeometricStationarityOptions {
  double stationarity_tol = 1e-3;
  double covariance_eigen_tol = 1e-8;
  double equality_tol = 1e-6;
  double active_bound_tol = 1e-8;
  double projection_tol = 1e-11;
  std::int32_t projection_max_iter = 20000;
};

// Authoritative numerical verdict. Admissibility remains separate for ambient
// fits; PSD feasibility participates in the PSD stationarity check. Missing
// checks are never replaced by optimizer status or driven-coordinate audits.
//
// Stationarity is decided by the Newton accuracy check when the fit carries
// one and the point is a regular interior point of its fitting domain: every
// ambient fit (improper estimates included, since they are interior to the
// ambient domain) and every PSD fit with no singular primitive covariance
// block. The fit passes when d <= budget; a nonpositive-curvature,
// ill-conditioned or unreliable solve fails it, because no accuracy statement
// is then available. The first-order check decides at PSD-boundary points,
// at active box bounds, under nonlinear equality constraints (Newton
// unsupported), and on fit paths without the Newton diagnostic.
FitVerdict common_fit_verdict(const FitDiagnostics& diagnostics);

// Record the original half-discrepancy and its full-coordinate gradient. The
// caller evaluates these at the returned theta before invoking this function.
void audit_full_model_fit(
    FitDiagnostics& diagnostics,
    const Eigen::VectorXd& theta_full,
    const Eigen::VectorXd& gradient_full,
    double reported_value, double recomputed_value,
    const spec::LatentStructure& pt, const model::ModelEvaluator& ev,
    const EqConstraints& con, const NonlinearEqConstraints& nl,
    const Bounds& bounds,
    StationarityDomain domain = StationarityDomain::Ambient,
    GeometricStationarityOptions opts = {},
    double objective_multiplier = 1.0);

// Audit a terminal full-θ objective gradient against the original model
// geometry. This is public so research code can recompute the geometric audit
// with alternative gradients or tolerances without rerunning an optimizer.
GeometricStationarityDiagnostics
audit_geometric_stationarity(
    const Eigen::VectorXd& theta_full,
    const Eigen::VectorXd& gradient_full,
    const spec::LatentStructure& pt,
    const model::ModelEvaluator& ev,
    const EqConstraints& con,
    const NonlinearEqConstraints& nl,
    const Bounds& bounds,
    GeometricStationarityOptions opts = {});

// Build a FitDiagnostics from an expanded θ and the prelude bits the fit
// path already carried. Never errors — every check that cannot be performed
// (no bounds, no constraints, no nonlinear block, evaluator error) records a
// benign default that downstream consumers can read uniformly.
//
// `snlls_profile_fallback_flag` is wired only from the SNLLS expand site
// (v1: always `false` — see `FitDiagnostics::snlls_profile_fallback`).
FitDiagnostics
finalize_fit_diagnostics(const Eigen::VectorXd&        theta_full,
                         const model::ModelEvaluator&  ev,
                         const EqConstraints&          con,
                         const NonlinearEqConstraints& nl,
                         const Bounds&                 bounds,
                         bool                          snlls_profile_fallback_flag = false,
                         DiagnosticsOptions            opts = {});

// Row-aware overload used by fit composers. The numeric audit is identical to
// the compatibility overload above, but source rows are populated from `pt`.
FitDiagnostics
finalize_fit_diagnostics(const Eigen::VectorXd&        theta_full,
                         const spec::LatentStructure&  pt,
                         const model::ModelEvaluator&  ev,
                         const EqConstraints&          con,
                         const NonlinearEqConstraints& nl,
                         const Bounds&                 bounds,
                         bool                          snlls_profile_fallback_flag = false,
                         DiagnosticsOptions            opts = {});

}  // namespace magmaan::estimate
