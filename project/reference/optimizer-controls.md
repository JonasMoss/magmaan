# Optimizer search controls

Search termination and the independent terminal audit are separate. Neither a
small step nor a backend success flag guarantees stationarity. The generic optimizer defaults remain available; complete-data ML uses the
validated profile below. The proposed interior Newton budget is not deployed
as a production acceptance rule.

`OptimOptions` retains its four legacy fields for source compatibility.
Explicit backend fields override the corresponding legacy setting. In R, the
same fields are nested lists inside `control`; the common parser passes these
through estimator wrappers. Each backend reads its own block, allowing a
fallback policy to carry settings for both solvers. Other backend blocks are
inactive. Unknown or duplicate names inside a backend block are errors in R.
An option unsupported by the selected NLopt algorithm is an error.

```cpp
magmaan::optim::OptimOptions options;
options.nlopt.ftol_rel = 1e-12;
options.nlopt.xtol_rel = 1e-10;
options.nlopt.vector_storage = 0; // automatic, not memory 10
```

```r
control <- list(nlopt = list(ftol_rel = 1e-12, xtol_rel = 1e-10))
# Pass control to the selected fitting entry point.
```

## Complete-data ML and GLS start: the layered moment start (2026-09-26)

Decided in `experiments/decisions/01-optimizer-defaults`, lane ml-gls (third
run, author's decision), with the evidence in its report.

- **Start.** The layered moment start (`estimate::layered_start_values`, now in
  core, R `start = "layered"`) is the default of the following:
  - C++ `ml_start_values`;
  - `api::ml()` (`ml_starts()`, also `layered_starts()`) and `api::gls()`;
  - R `fit_ml`, `fit_gls`, `fit_model(estimator = "ML" | "GLS")` and
    `fit_start_values`.
- **The former defaults stay selectable.** `scaled-fabin` (C++
  `scaled_fabin_start_values`, api `scaled_fabin_starts()`) was the former ML
  default. `fabin3` and `simple` give lavaan-compatible starts.
- **Optimizer: unchanged, NLopt L-BFGS.** PORT certified more fits, but also
  about eight times as many runaway fits (points far along a divergent path
  that the Newton check accepts). It waits for a runaway check in the verdict.
- **Unchanged routes.** Two-level and ordinal fits keep their own start
  constructors, and PSD ML keeps transported FABIN3. So does the explicit
  ordinary-first PSD recovery (`frontier_fit_ml_psd_fallback`), whose
  ordinary stage is pinned to FABIN3 with L-BFGS, apart from the ML default.
  A second pre-registered PSD run tested the layered start for both and
  rejected it (decisions/01, lane psd-ml, `2026-09-26-second`).

## Nonlinear equality constraints (2026-09-26)

`fit_ml` and `fit_gls` run NLopt SLSQP when the model has nonlinear equality
constraints and the selected backend cannot take them (L-BFGS, PORT and the
other unconstrained backends), instead of returning an error. The fit records
the backend that ran in `Estimates::substituted_backend` (R
`fit$optimizer_substituted`). An explicit SLSQP, L-BFGS-with-SLSQP-fallback or
IPOPT request is kept; IPOPT still errors in builds without it.

## Complete-data ML defaults (2026-09-22)

Bare C++ `fit_ml`, `frontier::fit_ml_psd`, the staged `api::ml()` factory,
and their R wrappers now use the validated ML numerical profile:

- NLopt relative objective tolerance 1e-12, relative step tolerance 1e-10,
  and maximum 5000 objective evaluations. Luksan gradient tolerance and
  memory remain backend defaults. PSD ML uses constraint tolerance 1e-8.
- (Superseded 2026-09-26 by the layered start, above.) High-level ML starts
  used `ml_start_values`: std.lv FABIN starts transported to the original
  marker chart when safe, native FABIN otherwise; this is now
  `scaled_fabin_start_values`. Fixed values,
  linear/nonlinear equalities, unsupported marker layouts and failed transport
  cause native fallback. Finite user hints override transported values.
  Low-level fitters continue to use the caller's explicit start vector.
  R `ml_start_policy` reports transported, fallback, or explicitly selected starts.
- Ordinary ML searches in unit-equivariant optimizer coordinates for every
  scalar backend (see "Optimizer coordinates" below).
- PSD ML enables its existing lifted expected-information diagonal scaling.
  It does not add a retry or change the PSD feasible set.

Explicit C++ `OptimOptions` replace the default argument: `{}` retains the
generic legacy tolerances; information coordinates are the default in both.
To customize the new policy,
start with `ml_optim_options()` or `frontier::ml_psd_optim_options()` and edit
its backend fields. An explicit `api::OptimizerSpec` likewise replaces that
factory's optimizer options; use the ML profile when constructing it. Selecting
a backend without controls, e.g. `ml().optimizer(nlopt_slsqp())`, inherits the
ML profile. Passing `nlopt_slsqp(OptimOptions{})` explicitly opts out.
For PSD scaling, the default argument is `ml_psd_options()`; an explicit
`PsdFitOptions{}` retains its former unscaled behavior.

In R, `control=list(start="fabin3", coordinate_scaling="none",
normalize_sample=FALSE)` selects native starts and raw optimizer coordinates. `control$start` also accepts `scaled-fabin`,
`simple` and the existing named start methods. Use `preconditioning="none"`
for unscaled PSD ML. Legacy `max_iter`, `ftol` and `gtol` explicitly override
profile defaults; a supplied nested NLopt field overrides the corresponding
legacy field. Other estimators retain their existing start/control defaults.

These search changes do not implement the research Newton audit, change the
production fit verdict, or make PSD fitting the default estimator. The
validation and remaining boundary/fallback limitations are documented in
[the numerical study](../validation/interior-newton-audit.md).

## Complete-data ML and PSD sample normalization (2026-09-27)

**Settled default; this implementation/adoption slice is closed (2026-09-27).**
Keep sample normalization enabled for ordinary complete-data ML and direct PSD
ML. The option was already enabled during the development comparisons; closure
confirms that behavior rather than changing the estimator or introducing a
new runtime switch. `fit_model(..., estimator="ML", psd=TRUE)` continues to use
direct PSD. Ordinary ML remains unrestricted ML, not implicitly PSD-constrained.

| Scope | Normalized fitting support |
|---|---|
| Single-level continuous, complete-data ML and PSD ML | Supported, on by default |
| Multiple groups, unequal group sizes, cross-group equalities | Supported |
| Means/intercepts, fixed and structural cells, marker/std.lv identification | Supported |
| Equal labels and general affine linear equalities | Supported; constants and coefficients are transported |
| Original-unit starts/hints; ML box bounds with linear equalities | Supported |
| Arbitrary linear `<`/`>` constraints | Unsupported by the constraint interface; not implied by support for linear equalities |
| FIML, multilevel, ordinal/other objectives, barrier normalization | Outside this completed slice |

Validation includes multigroup fitting under mixed variable/group scales,
means and cross-group loading equalities; affine-equality and boundary audits;
explicit-start/bound transport; and R integration. The broader fitting studies
use saved single-group development problems, so they do not establish general
multigroup success rates. Closing normalization does not close requested-chart
rejection, caller-unit admissibility robustness, or difficult-solution search.
Independent ordinary-first fallback remains a separately tested policy; the
existing opt-in fallback API still reuses usable ordinary estimates.

`OptimOptions::normalize_sample` defaults to `true` for the ordinary ML and
PSD ML fitting entry points. Single-level complete-data models, including
multiple groups and cross-group linear equalities, are transformed internally
by their sample standard deviations and identification-aware latent units.
Fixed/structural cells, affine constraints, explicit starts and ML box bounds
are transported with the model. Nonlinear equality and multilevel models retain
their existing paths; FIML, barrier and other objective families do not use this
new fitting transformation.

The staged C++ API and R ML/PSD wrappers construct automatic starts in the
normalized model. Low-level callers supplying `x0` continue to supply it in
original units; `normalized_ml_start_values()` exposes the matching constructor.
User start hints also remain in original units. Ordinary-then-PSD fallback
returns its ordinary estimates to original units before transporting its warm
start into the PSD stage. There is no automatic marker change or multistart.

`Estimates::sample_normalized` (R: `fit$sample_normalized`) records whether the
fit used this transformation. Estimates, partables, implied moments and
post-fit information calculations retain the caller's units; optimizer audit,
stationarity/admissibility diagnostics and stopping tolerances describe the
internal fitting representation. Standalone complete-data ML and PSD Newton
accuracy audits normalize independently, preserving their thresholds and
returning retained derivatives in caller coordinates.

Set `control$normalize_sample=FALSE` (C++: `opts.normalize_sample=false`) to
reproduce the original-unit fitting path. This switch is separate from
`coordinate_scaling`: the latter still controls optimizer preconditioning
inside whichever model representation is used. PSD covariance floors and
feasibility tolerances therefore apply in normalized units when enabled. ML
boxes combined with weighted linear equalities are imposed together in full
parameter coordinates; if needed the backend switches to SLSQP and records
`substituted_backend`, rather than dropping the bounds in an affine reduction.
Normalization can change starts, search paths and the selected local minimum;
it is not a guarantee of a better objective. The user requested enabling this
shared fitting machinery; the earlier exploratory pilot is not a new held-out
comparison establishing universal improvement.

## Optimizer coordinates (2026-09-25)

Quasi-Newton searches are not invariant to the coordinates they run in.
L-BFGS starts from an identity metric and takes its first step along the raw
gradient; PORT starts from the Hessian guess $D^2$ for its scale vector $D$,
bounds its trust region in $D$-scaled steps, and declares X-convergence when
the largest $D$-scaled step is small relative to the largest $D$-scaled
coordinate. In raw coordinates a single large coordinate (a variance in the
thousands, a mean far from zero) dominates that test and ends the search
early. lavaan passes `nlminb` the scale $1/\lvert x_0\rvert$ for start values above one
in magnitude, which depends on the start and switches at one, so its search is
not invariant to the data's units either.

`estimate/coordinates.hpp` defines the shared coordinate layer. The optimizer
drives $z = (\alpha - c)/s$ on the equality-reduced parameter $\alpha$; objective,
constraints, bounds and the reported terminal audit keep their model meaning,
and the audit is recomputed in $\alpha$ after the search. `OptimOptions::
coordinate_scaling` selects $s$:

- `SampleUnits`: the unit each parameter carries. Observed variables take
  their sample standard deviations. A latent variable takes its unit from a
  fixed nonzero loading (marker, including the structural unit loadings of
  phantom latents), else a fixed positive variance (std.lv, phantoms), else a
  fixed nonzero regression on a latent with a unit (higher-order markers), else
  an equality tying one of its parameters to a parameter with a unit
  (cross-group loadings), else the mean standard deviation of its indicators
  (effect coding). Loadings take indicator/latent, regressions
  outcome/predictor, (co)variances the product, intercepts and means their
  variable's unit. Equality-reduced coordinates take $1/\lVert K_j \oslash u\rVert$.
- `Information` (the default): the sample units refined by the expected
  information at the start, $h_j$, the larger of its values weighted by the
  sample covariance and by the start's implied covariance (where positive
  definite). A coordinate shrinks to $1/\sqrt{h_j}$ where that is finer than its
  unit, down to $10^{-3}$ of it, and never grows beyond the unit: low
  information at a start signals a degeneracy (a collapsed variance leaves its
  loadings information-free), not a safe large step. The start-weighted term
  keeps a start near a singular implied covariance from receiving a first
  step its own curvature does not allow.
- `None`: raw coordinates.

With `center_locations` (default on), coordinates that act only on means and
intercepts are centered at their start values, so a large location does not
dominate relative step tests. Both scales transform like the parameters when
observed variables change units, so a scaled search from a transported start
retraces the same path (`tests/unit/coordinates_test.cpp`).

Routes. The layer covers complete-data ML (including the multi-information
penalty and the extra-constraint entry), GLS in its scalar and residual forms,
the moment least-squares family (ULS, DWLS, WLS), pairwise GLS, and FIML, for
every scalar backend: L-BFGS, SLSQP, the L-BFGS/SLSQP fallback, PORT, TNEWTON,
VAR2, BOBYQA and IPOPT, with and without nonlinear equality constraints and
general affine constraints. Ceres LM and NL2SOL see the residual structure
and scale themselves, so they keep raw coordinates. PSD ML keeps its own
lifted information scaling and the sphere route its sample units. Ordinal,
mixed ordinal, CatML, two-level, SNLLS outer, IRLS inner, fitted-weight, RBM
and pairwise-likelihood fits remain in raw coordinates; the backlog tracks
them. `Estimates::coordinate_scaling` (R: `fit$coordinate_scaling`) reports
the coordinates used; `magmaan_core$estimate_coordinate_map()` returns the
units, information diagonal, center and scale for a model and start.

`OptimOptions::coordinate_scaling` defaults to `Information` in `OptimOptions{}`
and in `ml_optim_options()`, so every route with the layer searches in
information coordinates unless a caller selects `SampleUnits` or `None`
(2026-09-26, engineering/active/17-corpus-optimizer-recovery: on the 608 ML/GLS corpus pairs and on a panel with
each observed variable multiplied by a power of ten between $10^{-2}$ and
$10^{2}$, it is the best or tied rule for L-BFGS and PORT from the layered start
and for L-BFGS from the current start; PORT from the current start loses a few
pairs to path sensitivity from far starts). In R, `control$coordinate_scaling`
is `"none"`, `"sample_units"` or `"information"`, and `control$center_locations`
a logical. The legacy `ml_sample_scaling = TRUE/FALSE` maps to
`"sample_units"`/`"none"`.

## Composable starting values

Start construction and optimizer-coordinate scaling are separate operations.
The public `estimate/start_pipeline.hpp` interface provides:

- `construct_start_values(..., StartMethod, hints)` for native simple, FABIN2,
  FABIN3, Guttman, Bentler-1982 and James–Stein constructors.
- `prepare_std_lv_transport(target, rep)` to check eligibility and prepare the
  auxiliary unit-latent-variance model. It does not use sample statistics.
- `transport_start_values(plan, source_theta)` to transport any start producer's
  vector into the original marker model. It preserves implied moments before
  target-coordinate user hints are applied; it does not enforce PSD.
- `start_values(..., StartPolicy, hints)` to compose these steps. `Native`
  skips transport; `AutoStdLv` falls back to the same constructor in the target
  model; `RequireStdLv` returns an error instead of falling back.
  `ml_start_values` is the auto-transported FABIN3 shortcut.

For example, after checking each returned `expected`, the explicit C++ steps are:

```cpp
auto plan = estimate::prepare_std_lv_transport(pt, rep);
auto source = estimate::simple_start_values(plan->source, plan->source_rep, stats);
auto target = estimate::transport_start_values(*plan, *source);
```

The equivalent shortcut is `start_values(pt, rep, stats,
{StartMethod::Simple, StartTransport::RequireStdLv})`.
The prepared source model and its representation can be passed directly to
any existing start producer. Transport currently supports the conservative
std.lv-to-marker route only; effect-coded, spherical and constrained layouts
are not silently converted. Unsupported equalities, fixed values and marker
layouts have distinct reasons from an invalid source vector, unusable marker
scale, nonfinite transported values or fixed-value mismatch. A failed native
constructor returns an error. Existing within-constructor fallbacks remain
unchanged (for example FABIN3 uses FABIN2 for a singular instrument matrix
and retains simple loadings outside its supported indicator layouts). Finite target-coordinate hints take precedence;
changing those hints can change the implied moments after transport.

Ordinary coordinate scaling is the coordinate layer above (`ml_coordinate_scale`
returns its sample units). PSD information scaling remains part of its lifted
optimizer. Neither operation is implemented by multiplying a start vector alone.

Continuous-data and FIML R fit entry points accept the same controls:

```r
control = list(start = "fabin2", start_transport = "auto")
# Or supply a finite vector in target free-parameter order:
control = list(start = x0)
```

`start_transport` is `"native"`, `"auto"`, or `"required"`. An explicit vector
must have exactly `n_free` finite entries; it supersedes partable start hints
and cannot request nonnative transport. Omission and `start="default"` preserve
each entry point's existing default: ordinary complete-data ML and GLS use the
layered start (since 2026-09-26); the PSD and multi-information ML wrappers use
auto-transported FABIN3; ULS, WLS, SNLLS, FIML and prepared wrappers retain
native FABIN3. Explicit method names retain
native construction unless transport is selected separately. `scaled-fabin`
remains an alias for FABIN3 with automatic transport.

Each participating fit returns `fit$start`, containing `theta`, `method`,
`requested_transport`, applied `transport`, and `fallback_reason`. `theta` is
the vector supplied to the fitter, before any bound projection, PSD repair or
SNLLS profiling. `method` names the requested constructor (or `explicit`);
within-constructor substitutions are not yet reported per factor/block.
Existing `ml_start_policy` and `ml_start_fallback_reason` fields remain for
compatibility. An ordinary-to-PSD retry reports its actual warm-start input
separately from the original construction.

Marker-only Guttman, Bentler-1982 and James–Stein constructors cannot run in
the auxiliary std.lv model. Automatic transport therefore runs the requested
native constructor and reports `constructor-requires-marker`; required
transport fails. It no longer reports a successful transported method after
silently constructing simple starts.

The standalone `magmaan_core$estimate_start_values()` helper defaults to
auto-transported FABIN3 and accepts separate `start` and `transport` arguments.
It returns a numeric vector with `start_method`, `start_transport` and
`start_fallback_reason` attributes. Its former implicit simple default was
changed by the original pipeline commit; request `start="simple"` explicitly
when that is required.

The friendly C++ API accepts
`estimator.starts(api::start_policy({StartMethod::Fabin2, StartTransport::AutoStdLv}))`.
Continuous and FIML `api::Fit::starts()` retains the owning `StartValues`,
including the supplied vector and requested transport. Existing convenience
start selectors route through the same pipeline and keep their defaults.
Low-level C++ fitters still take explicit vectors and do not construct starts.
Ordinal, two-level and native FCSEM adapters remain separate; the new generic
policy is rejected where unsupported. Spherical initialization and per-stage
ML2S policy selection remain separate follow-ups.

Validation: constructor/transport fallback regressions, continuous/FIML C++
policy integration, and `r-package/examples/start_policy.R` exercise direct,
PSD, penalized, least-squares, SNLLS and prepared routes. The existing ML
numerical-default example checks compatibility behavior.

## NLopt: L-BFGS, SLSQP, VAR2, TNEWTON and BOBYQA

| Explicit field | Meaning | When absent |
|---|---|---|
| `ftol_rel` | Relative objective-change stopping | legacy `ftol` = 1e-10 |
| `ftol_abs` | Absolute objective-change stopping | NLopt default (disabled) |
| `xtol_rel` | Relative parameter-step stopping | legacy `gtol` = 1e-7 |
| `xtol_abs` | Same absolute step threshold for each coordinate | NLopt default (disabled) |
| `max_eval` | Function-evaluation budget, positive integer | legacy `max_iter` = 1000 |
| `tolg` | Luksan internal gradient stopping: L-BFGS, VAR2, TNEWTON | backend default (1e-8 in inspected NLopt 2.10.1) |
| `vector_storage` | Luksan storage control: L-BFGS, VAR2, TNEWTON | automatic backend choice |
| `constraint_tol` | Equality feasibility tolerance in constrained SLSQP | max(legacy `gtol`, 1e-12) |

Zero disables the f/x tolerances; `tolg=0` selects the internal default,
not a disabled gradient criterion. `vector_storage=0` selects automatic
storage. The stopping tests are alternatives, so tightening only one need
not improve accuracy. Memory semantics depend on the Luksan algorithm; it
is not a universal BFGS history length. Legacy `history` remains ignored
by NLopt to preserve old calls; use explicit `vector_storage` instead.
In the L-BFGS-to-SLSQP fallback policy, `tolg` and `vector_storage` apply
only to L-BFGS; common f/x/evaluation controls apply to both stages.
Finite, evaluable NLopt budget and line-search exits now return a candidate
instead of a fit error. `optimizer_status` records the stop; `converged` still
comes from the independent fit verdict. `audit$raw_backend_status` retains the
NLopt code, and `audit$nlopt_controls` records the resolved adapter controls
(zero `tolg`/storage entries still mean backend defaults). Re-auditing after
coordinate transport preserves this evidence. The explicit L-BFGS/SLSQP
fallback keeps the original start for each stage, counts evaluations from
both stages and retains the lower objective if both candidates fail. It does
not replace a previously usable first-stage return with a failed second stage.
IRLS can use a retained inner candidate under its existing true-ML Armijo
safeguard; the inner stop does not certify outer convergence. Detailed outer
iteration histories and stopping calibration remain unbanked.
Invalid controls and endpoints that cannot be evaluated still return errors.
The ten-reduction L-BFGS backtracking limit is not exposed by NLopt and is
still a separate recovery task. Constraint tolerance is independent of
explicit step tolerance and does not change the terminal audit threshold.

## PORT scalar and PORT-NLS

| Explicit field | PORT setting | When absent |
|---|---|---|
| `rel_f_tol` | V(RFCTOL), relative function convergence | positive legacy `ftol`; otherwise PORT default |
| `abs_f_tol` | V(AFCTOL), absolute objective **size**, not change | PORT default |
| `x_tol` | V(XCTOL), scaled relative step convergence | PORT default |
| `false_conv_tol` | V(XFTOL), false-convergence step threshold | PORT default |
| `max_eval` | IV(MXFCAL), function-evaluation budget | 10 × the iteration budget |
| `max_iter` | IV(MXITER), iteration budget | legacy `max_iter` |

Without `port.max_iter`, the legacy `max_iter` remains the iteration budget.
In R, a legacy `max_iter` or `ftol` in `control` replaces any explicit PORT
iteration/evaluation budget or relative function tolerance of the profile, as
it replaces the NLopt ones. Legacy `gtol` and `history` are unused. Explicit nonnegative tolerances are passed verbatim, including zero;
PORT's native validity checks and stopping rules apply. Do not interpret
zero as a universal disable switch. PORT distinguishes x, relative-function,
absolute-function, singular and false-convergence exits; these are not a
shared gradient test. See vendored `dv7dfl.c`, `dparck.c` and `da7sst.c`
for defaults, permitted ranges and the coupled stopping conditions.

## IPOPT and Ceres

IPOPT's block exposes native `tol` (overall scaled optimality tolerance),
`acceptable_tol`, `acceptable_iter` (zero disables acceptable termination),
and `limited_memory_max_history`. Absent overrides, the adapter retains
legacy `gtol`, max(`ftol`, `gtol`), IPOPT's acceptable-iteration default,
and legacy `history`, respectively. These are not objective/step tolerances.
IPOPT validates native ranges.

Ceres's block exposes `function_tolerance`, `gradient_tolerance` and
`parameter_tolerance` in the ordinary and ordinal LS estimator bridges.
Their absence preserves legacy `ftol`, legacy `gtol`, and CeresOptions'
`ptol=1e-8`. Direct C++ Ceres entry points retain their existing separate
`CeresOptions` (`ftol`, `gtol`, `ptol`, `max_iter`, `verbose`). These are
Ceres stopping criteria, not NLopt equivalents.

## Scope and remaining work

Shared optimizer-backed estimator paths receive the backend blocks without
changing objectives or estimator defaults. Specialized scoring/EM/IRLS outer
loops retain their own controls; a backend block tunes the inner optimizer
where one is used, not an outer loop. NLopt now retains raw codes and resolved
adapter controls through ordinary coordinate transport; uniform reporting for
other backends, complete fallback histories and outer-loop stopping remain
backlog work.
The outer R `control` list also carries estimator-specific options and is
not globally restricted by the shared optimizer parser.

Validation for this change: 39 NLopt/PORT tests (147 assertions), four IPOPT
checks, and the R `optimizer_controls.R` and `common_verdict.R` examples passed.
The optimized core and R package built successfully. Ceres was unavailable
in the validation environment, so its optional bridges were not run.
The repository-wide layering check still flags pre-existing pinned-paper
references; no finding named a file changed by this cleanup.
