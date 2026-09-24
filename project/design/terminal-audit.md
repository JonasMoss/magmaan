# Terminal Audit (L1) and Fit Diagnostics (L2)

## Authoritative fit verdict (2026-09-12)

The full-model audit is the authority for numerical acceptance throughout
magmaan. Backend termination explains how the search stopped; it is neither
proof of convergence nor an additional veto on a passing common verdict.
The driven-coordinate L1 record remains available for debugging and
lavaan-compatible inspection. The backend adapter policies described below
are implementation details, not the fit-level acceptance contract.

`estimate::fit_verdict(estimates)` is the C++ entry point. Its status and
component checks are `Passed`, `Failed`, or `Unchecked`. A failed component
makes the overall verdict failed; all required components must pass to make
it passed. An incomplete audit with no demonstrated failure is unchecked.
No unchecked component falls back to a backend status or L1 boolean. A
normal-cone projection that did not converge leaves stationarity unchecked;
it is not evidence that the fitted point is nonstationary.

The required checks are:

1. Recompute the original objective at the returned full parameter vector,
   on the common per-observation half-discrepancy scale. Require finite parameters/objective
   and agreement with the reported objective within
   `1e-6 * (1 + abs(reported))`; expose the normalized values and threshold.
   Record `objective_multiplier`, applied to both value and gradient. It is
   one for ordinary ML/LS/FIML and `1/N` for two-level ML, whose native `fmin`
   is a total deviance. This conversion does not change fitting or inference.
2. Recompute the full-model objective gradient, including eliminated linear
   and threshold coordinates. Check the declared constraints and the existing
   model-Frobenius metric-dual L2 stationarity residual at tolerance `1e-3`.
   This change does not recalibrate tolerances or assert global optimality.
3. For complete-data ML (2026-09-24), stationarity at a regular interior
   point of the fitting domain is decided by the Newton accuracy check
   instead: `d = sqrt(G' I^{-1} G) <= .01` from the total score and observed
   information after linear-equality reduction, with the conditioning and
   solve guards of `newton_accuracy_ml`. Every ambient fit is interior to its
   domain, improper estimates included. A PSD fit is interior when no
   primitive covariance block is singular (`covariance_nullity == 0`).
   Nonpositive curvature, ill conditioning or an unreliable solve fail the
   check, because no accuracy statement is available there. PSD ML fits use
   the covariance-domain version at every point, boundaries included
   (2026-09-24): for each singular connected component of a primitive block,
   with null basis U and multipliers from least squares on the face
   constraints, null directions with positive multipliers are held
   (U_c' dC U_c = 0), the reduced Hessian adds the rank-constrained set's
   curvature 2 tr(M dC C^+ dC) (Shapiro's sigma term; the blocks are linear
   in theta), and d is computed on the tangent space. Directions with zero
   or negative multipliers stay free. Fixed-zero variances hold their rows.
   Active box bounds and nonlinear equalities keep the first-order check of
   item 2, and so does every path without the Newton diagnostic (LS, FIML,
   ordinal, two-level, penalized). The interior first-order
   residual is retained as telemetry, not as an extra veto. The verdict's
   `criterion` records which check decided.

The fit entry point declares `Ambient` or `Psd` before the verdict is selected.
Ordinary fits use equality/bound normals; explicitly PSD-constrained fits
also use primitive covariance-cone normals and require PSD feasibility.
Never select the domain according to which residual passes. Both residuals
remain visible and coincide in the positive-definite interior.

Covariance admissibility is a separate diagnostic for ordinary fits: an
unconstrained optimum may be numerically converged but inadmissible. Numerical
convergence also does not establish identification, a local minimum, or the
validity of a particular standard-error or test formula. Consumers may require
those properties in addition, with separately reported reasons.

R exposes `fit$verdict` and `fit$diagnostics$verdict`; `fit$converged` is the
logical projection: TRUE for passed, FALSE for failed, NA for unchecked.
`optimizer_status` and `fit$audit` remain diagnostic. Consumers should use
`isTRUE(fit$converged)` when they require a certified numerical fit and must
not turn missing/NA verdicts into success. `evaluate_at()` still exposes
`audit_options` for its legacy L1 record; these do not alter the common
verdict's metric or default tolerances. Closed-form estimators that are not
objective minimizers report numerical convergence as not applicable rather
than inventing a stationary objective.

Failed or unchecked fits remain inspectable return values. A future adapter
migration must also preserve evaluable terminal candidates across soft backend
exits so the common audit can assess them; invalid inputs and absent/nonfinite
candidates remain errors. Different stopping algorithms are allowed, but
candidate retention and final assessment must follow one policy.

### Standardization rollout

Implemented first: one verdict function independent of optimizer status,
original-objective verification at the existing continuous ML/LS, FIML,
ML2S, Fisher/IRLS, two-level ML and ordinal finalization seams, explicit PSD
domain selection, continuous closed-form SNLLS coverage, full-coordinate audits for profiled ordinal paths,
and Estimates-based R result serialization. Uncovered specialized or extra-
callback-constraint paths report unchecked; they are not silently certified.
The existing frozen SNLLS handoff remains evidence of its pinned revision and
retains its original screening policy, not a template for new studies.

Remaining migration work belongs in `project/backlog/todo.md`: backend candidate
retention, complete specialized/extra-constraint coverage, and migration of
research consumers that still read L1/backend flags. This section supersedes
older statements below that treat the full-model audit as merely additive or
an optimizer status as the authoritative fit verdict.

## Context

magmaan's Newsom corpus speed survey turned up cases where the L-BFGS Full
GLS fit reaches the *objective value* of the optimum (`ex5_4`:
`f = 0.00301331`, matching SNLLS to 6 digits and lavaan to 5) but is then
discarded as a `LineSearchFailed`. NLopt's `nlopt_optimize` returns
`NLOPT_FAILURE` because its line search cannot find a measurable decrease
step at the floating-point noise floor of a near-perfect-fit GLS objective;
`cpp/src/optim/nlopt_optimizer.cpp` mapped that straight to
`FitError::LineSearchFailed` and discarded the iterate **without checking
whether it was a stationary point**. The gradient-norm computation at the
bottom of the function sat on the success branch only, so the wrapper never
examined the geometry of the point it just rejected.

This is structurally the wrong shape. The principle is well-established in
optimization engineering (Nocedal & Wright; Ceres' explicit projected-gradient
KKT termination; NLopt's own acknowledgement that `ROUNDOFF_LIMITED` may
still leave a useful minimum; lme4's convergence guidance treating optimizer
status as diagnostic, not verdict): **the optimizer's return code is a hint;
first-order stationarity at the returned point is the fact**.

This change introduces a terminal audit as two cooperating layers, both as
plain structs carried on the fit record, with R-side surfacing as nested
`fit$audit` and `fit$diagnostics` sub-lists.

## Principle

**Optimizers propose; the audit disposes.** Classification of the returned
iterate is by *geometry* (first-order stationarity via a projected gradient
or equality-constrained KKT residual), not by the backend's return code. The
two layers separate concerns that the old code conflated:

- **L1 — Optimizer Terminal Audit** lives in `cpp/src/optim/`. Operates in
  **driven coordinates** — the reduced/profiled/lifted space the optimizer
  actually minimized over. Answers "is this point primal-feasible and
  first-order stationary for the problem the backend solved?"
- **L2 — Fit Finalization Audit** lives in `cpp/src/estimate/`. Operates on
  **expanded full θ**. It records admissibility and feasibility, and—when the
  fit path supplies the ordinary full-`theta` objective gradient—audits the
  same terminal differential against the model's covariance-cone geometry.

Two coordinate systems, two questions, one fit record. L1's stationarity
verdict is what `fit$audit$stationary` reports; L2 records what
SE/χ²/robust-correction code needs and exposes the cross-method stationarity
comparison at `fit$diagnostics$geometric_stationarity`. The latter is
additive: it never overwrites the lavaan-compatible L1 result.

## L1: optimizer terminal audit

**Files:**
- `cpp/include/magmaan/optim/terminal_audit.hpp` — declarations of the
  `audit_terminal_iterate` and
  `audit_equality_constrained_terminal_iterate` free functions.
- `cpp/include/magmaan/optim/problem.hpp` — `TerminalAudit` and
  `TerminalAuditOptions` structs (placed alongside `OptimResult` because they
  are part of the optimizer-output contract).
- `cpp/src/optim/terminal_audit.cpp` — implementation.

### Signature

```cpp
TerminalAudit audit_terminal_iterate(
    const ObjectiveFn&     f,
    const Eigen::VectorXd& x,
    double                 reported_f,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    TerminalAuditOptions   opts = {});

TerminalAudit audit_equality_constrained_terminal_iterate(
    const ConstrainedScalarProblem& prob,
    const Eigen::VectorXd&          x,
    double                          reported_f,
    const Eigen::VectorXd&          lower,
    const Eigen::VectorXd&          upper,
    TerminalAuditOptions            opts = {});
```

The audit recomputes `f(x, grad)` at the returned `x`, builds a projected
gradient infinity-norm against the (possibly ±∞) bounds, applies the configured
absolute or relative stationarity test, records an active-set readout in
driven coordinates, and produces an *advisory* `OptimStatus`.

For equality-constrained scalar problems, the companion audit additionally
recomputes \(h(x)\) and \(J_h(x)\), solves the rank-revealing least-squares
multiplier problem, and tests

```text
|| project_box(grad f(x) + J_h(x)' lambda) ||_inf
```

together with primal equality feasibility. Rank-deficient constraint
Jacobians are allowed and their numerical rank is recorded. With active box
bounds, multipliers are fitted from interior coordinates before the one-sided
bound KKT signs are applied; when multipliers are non-unique this is
conservative. This constrained audit is essential for nonlinear `==` models
and for PSD-ML's Cholesky lift, whose covariance links are nonlinear
equalities even though the resulting primitive covariance matrices are PSD by
construction.

### Options

```cpp
struct TerminalAuditOptions {
  enum class StationarityMode { Absolute, Relative };
  StationarityMode stationarity_mode = StationarityMode::Absolute;
  double absolute_tol      = 1e-3;
  double stationarity_tol  = 1e-6;
  double active_bound_tol  = 1e-12;
  double f_consistency_rel = 1e-6;
  double constraint_tol    = 1e-6;
};
```

- `stationarity_mode` selects the shape of the stationarity test.

  - **Absolute (v1 default)**: `‖Pg‖_∞ ≤ absolute_tol`. The default
    `absolute_tol = 1e-3` was motivated by lavaan's `optim.dx.tol`.
    This is a historical choice, not an equivalence claim: optimizer
    coordinates, parameter scaling, bound treatment, and which termination
    paths receive the check can differ. In particular, the inspected lavaan
    0.7-2 implementation excludes selected exact-bound coordinates rather
    than implementing our full covariance-cone audit. The common half-
    discrepancy objective scale is necessary but insufficient for numerical
    comparability. The current full-model L2 criterion is different again.
  - **Relative**: `‖Pg‖_∞ ≤ stationarity_tol · (1 + |f|)`. The shape an
    earlier revision shipped as the default. The argument for Relative is
    that `|f|` spans many orders of magnitude across the SEM corpus, so an
    absolute threshold may under- or over-reject at the extremes. The
    argument against is that there is no calibration study justifying a
    specific relative tolerance against downstream inference (SE, χ²,
    LRT), so the choice is opinionated. Relative remains available as a
    research-grade alternative, and the test file
    `cpp/tests/unit/terminal_audit_test.cpp` keeps explicit Relative-mode
    coverage so the path doesn't bit-rot.

  This is the first genuinely hard design call in magmaan and the choice
  is genuinely unstable. The mode flag encodes the decision in one place
  so future experiments are one option flip away; the "Tolerance
  calibration" section below sketches the empirical work that would let
  us revisit the default with data instead of a defensive choice.

- `absolute_tol` is consulted only in Absolute mode. `stationarity_tol` is
  consulted only in Relative mode. They are not interchangeable.
- `active_bound_tol` is a coordinate-distance threshold for the
  projected-gradient construction — masking a gradient component, not a
  diagnostic readout.
- `f_consistency_rel` catches the rare case where a backend leaves the
  *last-tried* iterate in `x` rather than the *best-found* one. Relative
  because the same floating-point noise that motivates the audit also
  applies here.
- `constraint_tol` is the maximum equality residual accepted by the
  constrained audit. It certifies the returned point; it does not control the
  optimizer's search tolerance.

### Promotion policy (v1)

The audit is **observation only**. The wrapper still owns the final
`OptimStatus` it returns. The v1 policy:

| Backend reports | Audit says stationary | Wrapper returns |
|---|---|---|
| clean success | yes | `OptimStatus::Converged` (unchanged) |
| clean success | no | `OptimStatus::Converged` (unchanged — v1 does NOT downgrade) |
| soft failure | yes | **PROMOTE** to `OptimStatus::LineSearchSalvaged` |
| soft failure | no | original `FitError` (unchanged) |
| hard failure | (audit skipped) | original `FitError` |

Hard failures (no usable `x` or `f`) bypass the audit. A future PR may add
symmetric downgrade — if a backend declares success at a non-stationary
point, surface that — but this is a semantics-change separate from v1.

### Wired backends

| Backend | File | Soft-failure sites that now audit |
|---|---|---|
| `NloptOptimizer` (NLopt C API) | `cpp/src/optim/nlopt_optimizer.cpp` | `MAXEVAL/MAXTIME`; `FORCED_STOP`; generic `FAILURE`; `ROUNDOFF_LIMITED` (kept as salvaged regardless, audit fills `grad_inf_norm`); SLSQP equality problems use the KKT audit |
| `PortOptimizer` (PORT `drmngb`) | `cpp/src/optim/port_optimizer.cpp` | IV(1)=8 noisy; IV(1)=9 false convergence; IV(1)=10 budget; IV(1)≥11 other; IV(1)=7 singular keeps `SingularConvergence` |
| `IpoptOptimizer` | `cpp/src/optim/ipopt_optimizer.cpp` | unconstrained problems use projected gradients; equality-constrained problems use the same KKT audit and may salvage stationary budget/small-step stops |

Every success path also calls the audit so `OptimResult::grad_inf_norm` has
a single source of truth (the bespoke projected-gradient loops are gone).
The audit costs one extra `f(x, grad)` evaluation per fit — exactly
what the success paths already did inline.

**Not wired in v1:** `PortNlsOptimizer` and `ceres_lm` (residual-driven,
no scalar `ObjectiveFn` to audit); `CeresBfgsOptimizer` (no surveyed
failures motivate it).

### Semantic shift: PORT IV(1)=8

PORT's "noisy gradient detected" status (`IV(1)=8`) was previously a hard
`FitError::NumericIssue`. After v1, IV(1)=8 plus a stationary audit becomes
`OptimStatus::LineSearchSalvaged` (success); IV(1)=8 plus non-stationary
keeps `NumericIssue`. The "noisy" message remains informative about *why*
the audit failed when it does.

## L2: fit finalization audit

**Files:**
- `cpp/include/magmaan/estimate/diagnostics.hpp` — `FitDiagnostics`,
  `DiagnosticsOptions`, `finalize_fit_diagnostics` free function.
- `cpp/src/estimate/diagnostics.cpp` — implementation; reuses the PD-check
  pattern from `cpp/src/measures/fit_measures.cpp:130-135`,
  `estimate::active_bounds`, and `model::ModelEvaluator::sigma`.

```cpp
struct FitDiagnostics {
  std::vector<bool>      sigma_pd_per_block;
  bool                   sigma_pd_all = false;
  double                 lin_eq_residual_inf = 0.0;
  bool                   lin_eq_satisfied    = true;
  Eigen::VectorXd        nl_eq_residual;          // empty when no NL constraints
  double                 nl_eq_residual_inf  = 0.0;
  bool                   nl_eq_satisfied     = true;
  ActiveBoundDiagnostics active_bounds_full;      // indexes full θ (distinct from L1 active_set)
  bool                   snlls_profile_fallback = false;  // see v1 non-goals
};
```

`finalize_fit_diagnostics(theta_full, ev, con, nl, bounds, ...)` is called
from every public `fit_*` entry point that holds a `ModelEvaluator` (i.e.
the MatrixRep paths: `fit_ml`, `fit_gls`, `fit_gmm`, `fit_snlls`,
`fit_snlls_gls`). The FCSEM path uses a different evaluator type and is
excluded — its `fit$diagnostics` is the default-constructed schema slot.

### What L2 *records* vs *gates*

L2 never blocks a fit. It records what downstream consumers need:

- `sigma_pd_per_block` / `sigma_pd_all` — ML/GLS need `Σ⁻¹` so a non-PD
  block invalidates SEs; ULS only needs finite, so a consumer reading this
  decides per-formula. (A blanket gate would wrongly reject ULS fits whose
  Σ has a singular block that ULS does not actually require to invert.)
- `lin_eq_residual_inf` — the K-reparameterization enforces `A_eq·θ = b_eq`
  by construction; a residual exceeding `lin_eq_residual_tol` signals the
  expansion itself is misbehaving (a correctness signal, not an optimizer
  signal).
- `nl_eq_residual` / `_inf` — IPOPT drives `h(θ̂) → 0`; recording the achieved
  infinity-norm tells the user whether the nonlinear constraints were actually
  satisfied (separate from the optimizer's terminal status).
- `active_bounds_full` — the Heywood-case detector: a variance at its 0
  bound has a one-sided derivative, and the standard info-matrix SE for it
  is not valid. Downstream `magmaan_se()` etc. can flag or fall back.

### Common-coordinate covariance-cone stationarity

The lifted Cholesky audit and an ordinary partable-coordinate audit are not
the same first-order question at a singular covariance boundary. For the
scalar analogue (v=ell^2), an inward objective gradient at (v=0) becomes
zero after differentiating through (ell=0). Consequently, equality KKT in
the lifted chart can declare the terminal point stationary even though a
feasible first-order direction exists in the original covariance variable.

`audit_geometric_stationarity()` avoids that chart singularity. It receives
the analytic objective gradient in ordinary full-`theta` coordinates and
computes two normal-cone projection residuals:

1. **Ambient residual:** linear-equality normals, nonlinear-equality tangent
   normals, and active box-bound normals.
2. **Cone residual:** the same normals plus
   $-D_b^*(U_{0b} H_b U_{0b}^{\mathsf T})$, $H_b \succeq 0$, for every
   singular primitive covariance block. Here $U_{0b}$ spans the numerical
   null space of $\Theta_b$ or $\Psi_b$, and $D_b^*$ maps a covariance
   differential back to full model coordinates.

The cone residual is therefore the distance of the objective differential
from the KKT normal cone of the original PSD-constrained model. At a
positive-definite point every covariance null space is empty, so it reduces
exactly to the ambient equality/bound residual.

A scalar residual requires a metric. The declared default is
`model_frobenius`: the product Frobenius metric induced by the assembled
LISREL matrices. Symmetric off-diagonal covariance entries have weight two,
and shared coordinates accumulate every matrix occurrence. This is invariant
to orthogonal changes of basis inside covariance blocks and avoids privileging
the Cholesky chart. It is not invariant to arbitrary changes of measurement
units; such invariance is neither claimed nor numerically possible without a
separate scale convention.

The stationarity verdict thresholds the metric-dual L2 distance, which is
unchanged by isometric changes of model coordinates. The coordinatewise
infinity residual is also retained as a familiar diagnostic, but it does not
drive the verdict because it depends on the selected basis. Both residuals use
the same half-discrepancy objective scale and numerical `1e-3` default as the
L1 audit; this common number does not make the L2 and infinity-norm criteria
identical. Feasibility uses the original
equalities, bounds, and PSD blocks. The multiplier projection is observation
only and never changes fit return semantics. Complete-data ML/GMM/LS, FIML,
PSD FIML, ML2S through its ML/GMM Stage 2, ordinal/mixed-ordinal LS, and CatML
fit paths supply full-model gradients. Profiled SNLLS and extra callback
constraints remain unchecked until their eliminated/extra constraint normals
are available in the common representation.

## R schema

Surfacing happens in `r-package/src/fit.cpp` via two helpers
(`audit_to_r`, `diagnostics_to_r`) called from `fit_result()` — the single
shared assembler for ML/GLS/ULS/WLS/SNLLS — and mirrored in
`fcsem_fit_result()`. Active-bound indices convert 0-based → 1-based at the
R boundary so they index `theta` / `partable` rows directly.

```
fit$audit$
  stationary       (logical)   geometric + primal-feasibility verdict
  grad_inf_norm    (numeric)   projected objective/Lagrangian gradient norm
  raw_grad_inf_norm (numeric)  unprojected objective-gradient norm
  grad_scaled_inf  (numeric)   scale-aware version of grad_inf_norm
  stationarity_rhs (numeric)   configured absolute/relative comparison RHS
  f_recomputed     (numeric)   f at x, recomputed by the audit
  f_consistent     (logical)   |f_recomputed - reported| ≤ rel·(1+|reported|)
  f_finite         (logical)
  constrained      (logical)   equality-KKT audit ran
  constraint_violation_inf (numeric) max primal equality residual
  constraint_jacobian_rank (integer) numerical rank of J_h(x)
  active_set       (integer)   in DRIVEN coords: {-1, 0, +1}
  advisory_status  (character) "converged" / "line_search_salvaged" / ...

fit$diagnostics$
  sigma_pd_per_block      (logical)   one per group
  sigma_pd_all            (logical)
  lin_eq_residual_inf     (numeric)
  lin_eq_satisfied        (logical)
  nl_eq_residual          (numeric)   empty when no NL constraints
  nl_eq_residual_inf      (numeric)
  nl_eq_satisfied         (logical)
  active_bounds_lower     (integer)   1-based θ indices (Heywood detector)
  active_bounds_upper     (integer)
  geometric_stationarity  (list)
    checked               (logical)
    metric                (character) "model_frobenius"
    ambient_stationary    (logical)   equality/bound geometry only
    ambient_residual_inf  (numeric)
    ambient_residual_l2   (numeric)   metric-dual norm; drives verdict
    cone_stationary       (logical)   adds primitive PSD normal cones
    cone_residual_inf     (numeric)
    cone_residual_l2      (numeric)   metric-dual norm; drives verdict
    covariance_nullity    (integer)   summed active null-space dimension
  snlls_profile_fallback  (logical)
```

**Back-compat:** `fit$converged` stays strict (true only for
`OptimStatus::Converged`). Existing R consumers
(`r-package/R/model_data.R:1300` print method; paper harness
`harness-benchmark.R`, `harness-sim-benchmark.R`) keep working bit-for-bit.
Compatibility code reads `fit$audit$stationary` for the driven-coordinate
lavaan-style verdict and `fit$optimizer_status` for the refined status
string. Cross-method studies read
`fit$diagnostics$geometric_stationarity$cone_stationary`.

**Two coordinate systems, two active-set readouts.** `fit$audit$active_set`
is in the driven (reduced/profiled) coordinates the optimizer minimized
over; `fit$diagnostics$active_bounds_lower/upper` indexes the expanded full
θ. They will not match in general — this is the L1/L2 split made visible.

## What the corpus survey revealed

The intended PR success criterion was "Newsom `Full-fail` drops 3 → 1." The
empirical result is *no change in Full-fail count* across all 290 corpus
models — and that turns out to be the more informative outcome.

For `ex5_4` / `ex5_4c` the audit confirms: the iterate sits at the
*objective value* of the optimum (`f = 0.003`, matching lavaan to 5
digits) but the projected gradient there is **0.0015 / 0.0073** — not
machine zero, not noise-floor tiny. The objective surface is near-flat in
some directions of the constraint-reduced α space; the optimizer's line
search stopped because no further `f` decrease was measurable, and the
non-zero gradient remains.

This is exactly the distinction the audit was built to make. The previous
ad-hoc salvage at `max(1e-3, 1e3·gtol)` wouldn't have caught these either
(`0.0015 > 1e-3` borderline, `0.0073 ≫ 1e-3`), so v1
introduces no behavioral regression on previously-salvaged iterates while
adding principled stationarity verification everywhere. The honest
classification of `ex5_4` / `ex5_4c` as **non-stationary** points to the
right next investigation (evaluator accuracy, re-parameterization), rather
than masking the problem with a looser tolerance.

`ex12_3` is a separate mechanism — NLopt L-BFGS gets stuck early at
`f = 982` (true optimum `3.06`); PORT and SNLLS both converge it. The audit
correctly reports non-stationary at `f = 982` (large gradient there). Not
fixed by tolerance tuning.

## v1 non-goals (explicit out-of-scope)

1. **Symmetric downgrade.** If a backend reports `Converged` but the audit
   says non-stationary, v1 keeps the reported status. The
   `fit$audit$stationary` field records the driven-coordinate verdict for
   consumers. A separate semantics-change PR can flip this once a survey
   pass confirms no surprise.
2. **`allFit`-style cross-backend agreement** (Layer 3).
3. **LS-backend audit wiring** (`PortNlsOptimizer`, `ceres_lm`):
   residual-driven, no scalar `ObjectiveFn`.
4. **Per-estimator tolerance tuning.** Every backend uses the same
   `TerminalAuditOptions` defaults. The v1 Absolute / 1e-3 default is a
   defensive cross-package match (lavaan), not a calibrated choice; per-
   estimator tuning, or a switch to Relative with a calibrated tolerance,
   may emerge from the "Tolerance calibration" study below if ML / GLS /
   ULS / DWLS / FIML show systematically different gradient noise floors.

## Tolerance calibration (high-priority follow-up)

The full-model model-Frobenius dual L2 cutoff of `1e-3` is a convention,
not a calibrated accuracy guarantee. The older L1 infinity-norm default was
motivated by a lavaan gradient check; reusing that number for the L2 audit
neither reproduces lavaan's criterion nor establishes comparable accuracy
across models. In particular, model-Frobenius geometry removes dependence on
the optimizer's Cholesky coordinates but does not remove measurement units,
objective scaling, dimension, or curvature from the numerical problem.

A scalar example is sufficient: for the half Gaussian discrepancy
`F(v) = (log(v/s) + s/v - 1)/2`, changing units by `y_new = c*y` sends
`v_new = c^2*v` and the gradient to `g_new = g/c^2`, while leaving the
statistical fit and discrepancy unchanged. A fixed absolute gradient cutoff
can therefore change its verdict solely because the measurement unit changes.
Feasibility tolerances and active-eigenvalue decisions need analogous review.

The useful optimization practice is to separate feasibility, stationarity,
and solver termination and declare their scales. For example,
[Ipopt's termination documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_Termination)
distinguishes scaled overall optimality from separate unscaled feasibility,
dual, and complementarity tolerances. This is guidance on the structure of an
audit, not a reason to borrow Ipopt's numerical defaults for a different norm.
[lme4's convergence guidance](https://lme4.github.io/lme4/reference/convergence.html)
recommends checking tighter tolerances, scaling, derivatives, restarts, and
agreement across optimizers; it also discusses the limitations of Hessian-based
gradient scaling. No universal scalar threshold follows from either source.

Calibration work should proceed as follows:

1. Keep the original full-model domain and objective normalization explicit.
   Record continuous stationarity residuals, feasibility violations, projection
   status, and solver status. Unchecked projection is not failure or success.
2. Test equivalent parameter representations and changes of units on fixed
   fitted points, separately from refitting. Evaluate a declared dimensionless
   scaling convention. An information/curvature metric is a candidate on
   regular identifiable interiors, not an automatic solution at singular or
   PSD-boundary points.
3. Compare candidate residuals against high-accuracy reference fits from
   tightened optimization and independent backends. These are reference fits,
   not known global optima. Inspect objective gaps, standardized parameter
   changes, and implied means/covariances; tiny objective changes alone can
   hide parameter changes in flat directions.
4. Report sensitivity across several tolerances without changing the
   prespecified primary cutoff after seeing which method benefits. Distinguish
   solver return, approximate stationarity at the stated accuracy, admissibility,
   and downstream stability. First-order stationarity does not prove a minimum.
5. Calibrate a practical point-estimation accuracy target on one collection
   and validate on held-out models, units, sample sizes, and boundary ranks.
   Extend to inferential stability only in a separate inference study; the
   current fitting-only experiments do not validate SEs or tests.

The September 2026 SLSQP control diagnostic illustrates why this matters:
stricter objective stopping recovered eight failed stationarity checks in a
30-sample subset, with objective changes at most `1.6e-9`. That is numerical
polishing, not evidence of finding a different likelihood maximum. Keep the
current residual and threshold visible pending calibration; do not silently
relax acceptance or replace the audit by a solver's success flag.

### Accuracy interpretation and boundary follow-up (2026-09-21)

A normal-cone stationarity residual is naturally a **backward error**: at a
feasible point in a closed convex set, it measures the smallest linear
perturbation of the objective gradient that makes that point stationary,
in the chosen metric. It is not a bound on parameter error. In a flat
quadratic, arbitrarily small gradients coexist with appreciable parameter
error. Accordingly, call the current result approximate first-order
stationarity at a declared tolerance, not an accuracy or minimum certificate.

A candidate revised assessment should distinguish:

- normalized feasibility, gradient/normal residual, projection reliability,
  and primal/dual complementarity;
- a dimensionless model metric with an explicit reference-scale convention;
- local accuracy diagnostics and conditioning, when available;
- optimizer termination and any separate recovery policy.

Near a PSD boundary, selecting an approximate nullspace by an eigenvalue
cutoff implicitly approximates complementarity. It does not bound
`trace(Z P)` independently of the dual multiplier magnitude. Record and
calibrate complementarity as well as primal/dual feasibility. In the scalar
variance problem `f(t) = (log(1+t) + .5/(1+t))/2`, `t >= 0`, the optimum is
zero; at `t=1e-10` its strict interior gradient is about `.25`. Thus the
exact normal residual can remain large while the parameter error vanishes.
A metric projected-gradient mapping is a candidate that avoids this hard
active-set discontinuity for convex restrictions, but requires a declared
step scale and an accurately solved joint projection. It is not yet an
implemented replacement, and nonlinear constraints need separate treatment.

In identifiable regular reduced coordinates, a positive-definite Hessian
`H` yields a local predicted step `-H^{-1}g` and predicted objective gap
`g' H^{-1} g / 2`. With our per-observation ML objective,
`N * g' H^{-1} g` approximates both squared displacement in information
units and the remaining twice-log-likelihood improvement. This is a local
approximation, not a global error bound. Inspect the standardized step as
well as the decrement; do not erase unidentified directions with an
unqualified pseudoinverse. Boundary and arbitrary-discrepancy cases do not
inherit the regular ML statistical interpretation.

For calibration, compare likelihood gaps, maximum standardized parameter
and moment changes, and block RMS/Frobenius changes against qualified
same-domain references. The maximum protects against dilution of a single
unstable component; aggregate block errors describe overall movement.
Illustrative budgets `2N*delta_f <= 1e-3` and standardized changes below
`1e-4` are exploratory reporting targets, not approved defaults. Reference
fits require tighter polishing and independent optimizer agreement; if
these fail, label numerical accuracy unresolved. Self-comparison to the
lowest objective returned by a failing portfolio is not accuracy evidence.
Freeze targets and controls before held-out model/unit/rank validation.

Sources: [Dennis and Schnabel, chapter 7](https://doi.org/10.1137/1.9781611971200.ch7),
[Boyd and Vandenberghe, section 9.5](https://web.stanford.edu/~boyd/cvxbook/),
[Bates and Watts (1981)](https://doi.org/10.2307/1268035), and
[Ceres' projected-gradient stopping rule](https://ceres-solver.readthedocs.io/latest/nnls_solving.html).
The inspected current lme4 `checkConv` additionally takes componentwise
minima of raw and scaled gradients and skips remaining gradient checks on
singular fits; its procedure is not a drop-in PSD-boundary audit.

### Regular-interior candidate profile (2026-09-21)

The advisory C++ prototype in `cpp/tests/checks/interior_newton/` uses the
canonical observed information and original ML gradient after linear-equality
reduction. Accuracy budget: Newton distance
`d=sqrt(N*g' H^{-1}g) <= .01`, equivalently total-negative-log-likelihood
EDM <=5e-5. This is a local accuracy approximation. It became part of the
authoritative verdict for complete-data ML on 2026-09-24 (item 3 above). Ineligible boundary/improper points and unreliable curvature remain
separate outcomes. The prototype's numerical conditioning/solve guards are
provisional and do not establish identification.

The paired study covers seven Gaussian model/settings, three sample sizes,
three repetitions, three measurement units, and nine L-BFGS control profiles,
with stock and extended backtracking (3,402 fits). A candidate control profile
is `ftol_rel=1e-12`, `xtol_rel=1e-10`, internal `tolg` default and automatic
memory. Legacy `gtol` names the step test and legacy `history` is not
forwarded to NLopt. Explicit `OptimOptions::nlopt` controls now expose
`xtol_rel`, `tolg` and `vector_storage` independently; see
[optimizer controls](../reference/optimizer-controls.md). Stock line-search
domain failure is a separate unresolved issue.
No production defaults changed. Full findings and limits are in
[the study note](../validation/interior-newton-audit.md).

### Remaining historical implementation items

5. **`fit$converged` boolean semantics:** unchanged.
6. **`snlls_profile_fallback` plumbing:** the flag exists on
   `FitDiagnostics` and surfaces to R, but the v1 SNLLS expand site leaves
   it `false`. Wiring requires a small flag on `GpProblem` — follow-up.
   Documented as a known v1 gap.
7. **No retry / fallback / warm-restart inside the audit.** The audit
   observes; it never re-runs the optimizer. That's Layer 3.

## References

- Implementation roadmap: [`project/architecture/roadmap.md`](../architecture/roadmap.md) — see the "Optimizer backends" section
  for the existing `OptimStatus` / `grad_norm` surfacing that the audit
  now sits behind.
- Open backlog: [`project/backlog/newsom-corpus-failures.md`](../backlog/newsom-corpus-failures.md) — section 1 amended
  after the audit's empirical findings on `ex5_4` / `ex5_4c`.
- Primary bug site (now fixed): `cpp/src/optim/nlopt_optimizer.cpp` — the
  `NLOPT_FAILURE` path that previously discarded any returned iterate
  without examining its geometry.
- Prior art:
  - Nocedal & Wright, *Numerical Optimization* — line-search convergence
    analyzed via gradient-norm / first-order stationarity, not "the line
    search returned success."
  - Ceres Solver: projected-gradient KKT termination — `‖x − Π(x − g)‖∞ ≤
    gradient_tolerance` (equivalent to the per-component active-bound
    masking used here, for box bounds).
  - NLopt manual: `ROUNDOFF_LIMITED` may still leave a useful minimum;
    the API returns `xopt`, `fmin`, and status separately to support
    auditing the returned point.
  - lme4 convergence guidance: optimizer warnings as diagnostics
    cross-checked with gradients, Hessians, and alternate optimizers;
    `allFit` as the practical cross-check pattern.
