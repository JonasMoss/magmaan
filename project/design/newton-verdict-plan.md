# The Newton check in every default verdict: plan (2026-09-25)

Status: plan, decided in principle on 2026-09-25, nothing implemented. When a
phase lands, fold its contract into [terminal-audit.md](terminal-audit.md) and
shorten this file. Part of the
[convergence-engineering vision](convergence-engineering.md), whose judge must
be the same for every estimator.

## Decision

Every iterative estimator's default fit verdict uses the Newton check, with an
analytic Hessian of the fitted objective. Today only complete-data ML does
(including Fisher scoring and IRLS). Every other family decides convergence
with the first-order check and so cannot reject a stationary point that is not
a minimum. engineering/17 shows the effect: nine Little models that ML rejects
for nonpositive curvature are accepted under GLS at a worse objective.

## What exists (survey 2026-09-25)

**Default inference does not need the exact Hessian.** Default standard errors
for the least-squares family use the Gauss-Newton matrix `J'WJ`
(`ls_information`), which leaves out the residual term and is positive
semidefinite by construction. It can never report negative curvature, so it
cannot serve the curvature half of the check.

**The exact Hessians exist anyway,** built for misspecification-robust sandwich
standard errors and case influence:

| Family | Exact analytic Hessian | Where | State |
|---|---|---|---|
| complete-data ML | `inference::information_observed_analytic` | inference.cpp | public, in the verdict |
| FIML | `fiml_observed_hessian_analytic`, public `fiml_observed_information` | fiml.cpp | public; the fit path does not attach it |
| continuous LS: ULS, GLS, WLS, DWLS, fixed-weight GMM | `continuous_ls_observed_bread_analytic` | robust/weighted_inference.cpp | anonymous namespace, `K'HK` reduced, per observation |
| SNLLS (ordinary, GLS) | the continuous LS Hessian at the expanded `theta` | as above | the profile Hessian is not needed |
| ML2S stage 2 | normal theory: the ML Hessian; other weights: finite differences (`ml2s_observed_bread`) | fiml.cpp | the continuous LS Hessian replaces the finite differences directly |
| ordinal and mixed LS | `ordinal_observed_bread_analytic`, `mixed_observed_bread_analytic` | ordinal.cpp | anonymous namespace, reduced; delta and theta covered |
| two-level ML | none; observed information is finite differences (h = 1e-5) | twolevel_objective.cpp | to build |
| CatML | none | fit.cpp, ordinal.cpp | to build |
| barrier penalties | analytic gradient only | multiinfo_penalty.cpp | to build |
| FCSEM | none; numerical covariance Jacobian | fcsem_evaluator | out of scope |
| implicit RBM | would need third derivatives | rbm.cpp | out of scope |

All of them share `detail_second_order.hpp`, the contraction
`tr(G d2Sigma/dtheta_a dtheta_b)` and `d2mu` for any symmetric weight `G`.

**The Newton machinery is already generic.** `audit_newton_derivatives`
(newton_accuracy.cpp) takes any total gradient and Hessian through
`NewtonDerivatives`, and does the equality reduction, PSD-face curvature
(Shapiro's term), equilibration, the curvature and conditioning guards, and
the solve. Post-fit adapters exist for every family (`newton_adapters.hpp`,
2026-09-24), but apart from ML, FIML and normal-theory ML2S they
differentiate the gradient numerically. They do not use the analytic Hessians
above, which are private to the robust and ordinal code and reduced to the
equality basis.

**Gaps found on the way:**
- No metric other than the objective's own Hessian is supported.
- `NewtonAccuracyDiagnostics` records neither the curvature source nor the
  metric.
- The ordinal and mixed constrained fits attach no diagnostics, so their
  verdict is unchecked.
- `fit_ml_psd_sphere` is ML but attaches no Newton check.
- The all-ordinal bread evaluates without mean Jacobians (`evaluate(theta,
  true, false)`), which may drop `mu` terms for theta-parameterization or
  released-scale blocks. Suspected, to confirm with a finite-difference test.
- `detail_second_order.hpp` has no test for nonrecursive `B`.
- R's serializer omits `box_constrained`, and `frontier_newton_accuracy()`
  refuses non-ML fits.
- Once `checked` is true, an `Unavailable` curvature fails the verdict. The
  design doc does not say so, and with finite differences a failed probe
  would become a non-convergence.

## Design

### Curvature and metric are separate

The Newton step comes from the fitted objective's own Hessian `H`, which must
be positive definite on the reduced (and, at a PSD face, tangent) space. That
is the curvature check. The step is measured in standard-error units:

    delta = H^{-1} g,     d^2 = N delta' V^{-1} delta,     pass if d <= .01

with `V` the estimator's asymptotic covariance, all in the reduced
coordinates.

| Objective | `V^{-1}` | Resulting `d^2` |
|---|---|---|
| ML, FIML, two-level ML, normal-theory ML2S, Fisher, IRLS | `H` (observed information) | `N g' H^{-1} g`, today's check |
| least squares with weight `W` | `H Omega^{-1} H`, `Omega = J' W Gamma W J` | `N g' Omega^{-1} g` |
| barrier penalties | the ML observed information | `N delta' H_ML delta` |

For GLS, and WLS with `W = Gamma^{-1}`, `Omega = J'WJ`, which matches the
information, so the ML budget carries over. For ULS and DWLS the sandwich
makes `d` free of units. `Gamma` is the one the estimator's default standard
errors use: normal theory for continuous ULS and GLS, the estimated asymptotic
covariance for DWLS, WLS and ordinal fits. The convergence metric then follows
the inference the user sees, and adds no new choice.

### Only analytic curvature decides a default verdict

Finite differences stay available in explicit audits. A family without an
analytic Hessian keeps the first-order check until it gets one, and its
verdict records that criterion. `Unavailable` from an analytic Hessian means
the objective, gradient or Hessian failed at the returned point, which stays a
failure.

### Provenance

`NewtonAccuracyDiagnostics` gains the curvature source (analytic, finite
differences, Gauss-Newton) and the metric (Hessian, information, sandwich).
The verdict's `criterion` keeps saying which check decided. R serializes all
fields.

## Phases

Each phase runs its dry run before wiring, lands with the tests listed, and
records every fit whose `converged` changes.

### 0. Core

- Extend `audit_newton_derivatives` with a metric argument, and pass through
  the gradient the geometric audit already computed.
- Move the continuous LS, ordinal and mixed analytic Hessians into a shared
  detail header, with full-`theta` variants on the total scale. The existing
  reduced callers keep their results.
- Switch the adapters to the analytic Hessians where they exist.
- Finite-difference cross-checks of every analytic Hessian at random interior
  points, covering means, several groups, equality constraints, thresholds,
  delta and theta, missing-data patterns and a nonrecursive `B`. Confirm or
  clear the ordinal mean-Jacobian suspicion.
- Attach the ML check to `fit_ml_psd_sphere`.

### Dry run, before each family is wired

Using the post-fit adapters, compute `d`, the curvature status and the
metric for every fit of that family on the test suite and the textbook corpus
(the engineering/17 set), under current controls. Classify the would-be flips:

- **curvature:** nonpositive curvature, a genuine non-minimum. These flips
  are the point of the change.
- **budget:** positive definite, but `d > .01`. This is a stopping-control
  problem. complete-data ML got its tightened profile (`ftol_rel` 1e-12,
  `xtol_rel` 1e-10, 5000 evaluations) together with the check for this
  reason. The other families still run the legacy controls.
- **unavailable or ill-conditioned.**

The dry run predicts the flip rate and decides, per family, whether to adopt
the ML control profile before switching.

### 1. Likelihood families, analytic Hessians already public

FIML (`fit_fiml_impl` finalization, which also serves pattern NTML and the
constrained FIML fit without callback constraints), FIML-PSD, the FIML sphere
fit. Metric: the Hessian itself.

### 2. Continuous least squares

GLS (scalar and residual paths), ULS, WLS, DWLS, fixed-weight GMM,
fitted-weight GMM (at the frozen final weight), GMM-PSD, SNLLS and GLS-SNLLS
(at the expanded `theta`), ML2S stage 2 for non-normal-theory weights
(replacing its finite-difference bread), the least-squares sphere fit, and
`evaluate_at` for LS estimators. Metric: the sandwich.

### 3. Ordinal and mixed

Bounded, SNLLS, full-threshold, theta free-threshold and PSD variants. First
give the constrained ordinal and mixed fits their geometric diagnostics.
Metric: the sandwich with the estimated asymptotic covariance.

### 4. The missing Hessians

- Barrier penalties (`log det Corr`, `log det Q`): second derivatives of
  `log det` of the standardized complete-data blocks, on top of the existing
  gradient. Then `fit_ml_multiinfo` and `fit_fiml_multiinfo` join, with the ML
  information as metric.
- Two-level ML: analytic observed Hessian from the same second-order
  contraction on both levels. It also replaces the finite differences behind
  two-level observed standard errors.
- CatML: second derivatives of the covariance-to-correlation map. The
  ordinal standardization terms are the model.

### Tests expected to change (survey 2026-09-25)

- C++: `snlls_test.cpp` (saturated ULS SNLLS, the ULS/GLS-SNLLS/WLS loop),
  `ordinal_test.cpp` (ordinal SNLLS DWLS and WLS), and 16 non-ML calls of
  `check_psd_terminal` in `psd_ml_test.cpp`.
- R: two-level (4), multiinfo (6), pattern NTML (3), admissibility (FIML,
  ordinal, CatML, mixed and LS PSD fits), SNLLS, FIML scalar invariance, FIML
  PSD.
- Goldens assert no magmaan verdicts.

A test that flips is inspected, not loosened. A flip on a correct fit means
the controls or the Hessian are wrong. A flip on a saddle means the old
assertion was wrong.

## Scope tiers

- **Build now:** phases 0 to 3, the barrier Hessian from phase 4 (it is the
  default route for `papers/sem-barrier`), provenance, R serialization, the
  stale comments (diagnostics.hpp, fit.cpp, evaluate.hpp,
  R/newton_accuracy.R) and the design-doc update.
- **Deferred, methods known:** the two-level and CatML Hessians; the Hessian
  of the Lagrangian for nonlinear and callback equality constraints (these
  report `Unsupported` and keep the first-order check); boxes interacting
  with singular PSD faces; a public R Newton helper for every family.
- **Open questions:**
  - Whether the `.01` budget, argued for ML, is right when `V` itself is
    estimated (DWLS and ordinal fits with an estimated asymptotic
    covariance). Report sensitivity at `.003` and `.03` as the ML study did.
  - A final safeguarded Newton correction when `H` is positive definite and
    `d` is just above budget. It costs one solve with a Hessian that is
    already computed. It changes the stopping rule, so it is an optimizer
    policy for the defaults register, decided on the dry-run data.
  - FCSEM and implicit RBM, which have no second derivatives.
