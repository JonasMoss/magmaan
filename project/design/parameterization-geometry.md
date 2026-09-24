# Parameterization geometry: what a latent scaling convention can and cannot buy

**Status:** findings recorded 2026-09-19. Measured by
`experiments/82-latent-metric-geometry`, which supersedes
`experiments/_archive/02-latent-metric-identification`.

**Scope correction (2026-09-19):** the original conclusions below are exploratory
measurements, not a proof that no materially better chart exists. A PE/IN ratio
below one does not establish numerical optimality. The structural fit gaps also
need a common-domain audit: marker fits can admit negative disturbance variances
that a fixed-positive-disturbance chart cannot represent. Experiment 83 now tests
six recursive structures with matched starts and a common strictly PD component
domain. Its R prototype implements total-variance elimination and analytic
derivatives; it makes no native-library speed claim. The earlier statement that
identification bounds total variance away from zero is not a general theorem.

Why this exists: "std.lv works best" circulates as folklore, the repo had partial
and partly contradictory evidence for it, and there was no framework for deciding
whether a *better* convention exists. This records the framework, the measured
numbers, and two claims that turned out to be wrong.

## An identification convention is a gauge choice

The latent scaling indeterminacy is a group orbit `c -> (c*lambda, phi/c^2)`. Every
convention is a hypersurface cutting those orbits, so the split is sharp:

- **Invariant**: fitted Sigma, chi-square, df, residuals, fit indices, the LRT.
- **Not invariant**: standard errors, Wald tests, the shape of theta-hat's sampling
  distribution, conditioning, optimizer work, and where the admissibility
  boundaries sit.

Exp 82 gates the first list on every run. Across-chart spread comes out at 1.4e-09
for chi-square and 1.1e-08 for the LRT p-value, so any across-chart difference in
the second list is attributable to the chart alone.

## Why marker degrades: transversality

Conditioning degrades as the gauge-fixing slice becomes *tangent* to the gauge
orbit, because then moving within the slice nearly moves along the unidentified
direction. The three conventions differ only in what anchors that transversality:

| convention | slice | transversality proportional to |
|---|---|---|
| marker (ULI) | `lambda_1 = 1` | **`lambda_1`** |
| std_lv (UVI) | `phi = 1` | **`phi`** |
| effect coding | `sum(lambda) = p` | **`sum(lambda)`** |

Marker anchors the gauge to one arbitrarily chosen loading, which is legitimately
allowed to be near zero. std_lv anchors it to a factor variance, which identification
keeps away from zero. Effect coding anchors it to a sum, which needs cancellation to
fail. So marker is the only convention that can degenerate on ordinary data, and it
does so exactly when the marker indicator is weak.

Measured, one-factor `p = 12`, only `lambda_1` varying:

| lambda_1 | pcond marker | pcond std_lv | pcond effect |
|---:|---:|---:|---:|
| 0.3 | **34593.9** | 6.40 | 21.99 |
| 0.5 | 502.7 | 6.56 | 17.82 |
| 0.7 | 28.4 | 6.77 | 12.45 |
| 0.9 | 25.1 | 28.04 | 12.44 |

## Is there a better convention? Bates and Watts make it decidable

Bates and Watts (1980), *Relative Curvature Measures of Nonlinearity*, JRSS-B 42,
1-25, split the second-derivative array of the model map into a component tangent to
the model manifold, **parameter-effects (PE) curvature**, and a component normal to
it, **intrinsic (IN) curvature**. PE depends on the chart and can be driven to zero
at a point. IN is a property of the manifold and is invariant under any
reparameterization. Their empirical finding across real datasets is that PE usually
dominates IN by an order of magnitude, meaning most apparent nonlinearity is an
artifact of coordinates. The symptoms of large PE are banana-shaped likelihood
contours, so Wald regions fail while likelihood regions do not, plus skewed
estimates and slower start-dependent convergence.

`PE/IN` is therefore the headroom. Measured, `p = 12` (IN agrees across charts to
2.4e-08, which is the correctness gate on this computation):

| lambda_1 | PE/IN marker | PE/IN std_lv | PE/IN effect |
|---:|---:|---:|---:|
| 0.3 | **4.103** | 0.572 | 0.791 |
| 0.5 | 1.599 | 0.572 | 0.790 |
| 0.7 | 1.006 | 0.572 | 0.790 |
| 0.9 | 0.824 | 0.573 | 0.792 |

**Answer: no, not materially.** std_lv's PE sits below IN at every marker strength
and is flat to three digits, so the removable curvature has already been removed and
what remains is the irreducible manifold curvature. On the `p = 6` grid the same
comparison reads marker 8.25 against std_lv 0.98 at `lambda_1 = 0.3`. Taking total
curvature as the quadrature sum there, marker to std_lv is an 83 percent reduction,
and std_lv to the theoretical optimum with PE driven to zero would be a further 29
percent, at the cost of a data-dependent chart with no interpretation.

A globally curvature-free chart exists if and only if the Fisher-Rao metric is flat,
which SEM's generally is not. That is the hard floor, and it is gauge-invariant.

## std_lv and effect coding trade places

std_lv wins on PE curvature. Effect coding wins on linear conditioning at moderate to
strong `lambda_1`. This matters less than it looks, because **full Newton and Fisher
scoring are affine-invariant**, so linear conditioning is absorbed for free, whereas
no optimizer absorbs PE curvature. Quasi-Newton sits in between, since `H_0 = I` is
basis-dependent but the metric is learned. First-order methods are fully exposed.

The practical consequence for magmaan is that the numerics lever is the optimizer's
metric, not the chart. Exp 02's per-backend split already pointed this way, with
nlopt-lbfgs at 0.789 against port at 0.895.

## Cost, measured properly

Wall time is measurable at these sizes. The earlier claim here that microsecond fits
are untimeable was wrong, and so was exp 02's method, but for a sharper reason than
"noise". `system.time()` quantises to about 1 ms on Linux, so ten timings of the same
225 us fit return min 0.000, median 0.001, max 0.007. exp 02 timed each fit once, so
its 0.839 and 0.996 were computed from a quantised timer reading sub-millisecond
operations. Batching removes it: calibrate a batch to at least 50 ms, take the median
over five batches, and relative IQR lands at **1.7 percent**.

Median per-call fit time, `n = 400`, all `lambda1` pooled:

| p | backend | marker | std_lv | std_lv/marker |
|---:|---|---:|---:|---:|
| 6 | nlopt-lbfgs | 160 us | 111 us | 0.696 |
| 12 | nlopt-lbfgs | 429 us | 213 us | 0.496 |
| 24 | nlopt-lbfgs | 1596 us | 681 us | **0.427** |
| 6 | port | 188 us | 144 us | 0.767 |
| 12 | port | 459 us | 326 us | 0.709 |
| 24 | port | 1725 us | 1193 us | 0.691 |

So the ratio exp 02 put at 0.839 is 0.43, and it improves with p rather than washing
out. The backend split confirms the conditioning mechanism: nlopt-lbfgs gains far more
than port, because a quasi-Newton method starting from `H_0 = I` is exposed to the
chart's conditioning while a more metric-aware method is not. `f_evals` ratios follow
the same pattern, 0.27 for lbfgs against 0.61 for port at `p = 24`.

## Back-conversion is cheap, and exp 02's wash was an artifact

Two halves, and exp 02 measured only the first.

- **Exactness.** Back-converting the fitted theta into marker coordinates and pushing
  it through the marker map reproduces the native marker fit's implied Sigma to
  **0 to 1e-15**. This is the disqualifying check and it passes.
- **Cost.** Point-estimate conversion is an O(p) loading rescale, flat near 1.8 us,
  falling to **0.07 percent** of the fit by `p = 48`. The vcov needs the delta-method
  sandwich `J V J'`, dense O(p^3), which a user asking for the marker chart needs
  because they want marker standard errors and not just marker point estimates. That
  is the term exp 02 never counted.

| p | fit | bc point | bc vcov | point % | vcov % |
|---:|---:|---:|---:|---:|---:|
| 6 | 104 us | 1.62 us | 2.84 us | 1.56 | 2.74 |
| 12 | 191 us | 1.59 us | 5.04 us | 0.83 | 2.63 |
| 24 | 514 us | 1.82 us | 16.1 us | 0.35 | 3.14 |
| 48 | 2485 us | 1.81 us | 83.0 us | 0.07 | 3.34 |

The vcov share **plateaus near three percent rather than shrinking**, because the fit
is superlinear too. That corrects a claim made earlier in this file's history that the
overhead simply falls with p. Net: about three percent overhead against a 30 to 57
percent saving on the fit, so the internal-chart substitution pays and exp 02's
roughly 18 percent figure was a measurement artifact.

## Convergence at small n: the failures are strictly nested

The geometry tables are population-level and say nothing about actual failure. Non-convergence
rate at `p = 12` under nlopt-lbfgs, 25 replications per cell:

| n | marker | std_lv | effect |
|---:|---:|---:|---:|
| 50 | **0.0204** | 0 | 0 |
| 100 | 0.0100 | 0 | 0 |
| 75 / 150 / 400 | 0 | 0 | 0 |

The rates are small, but every chart sees the same dataset per cell, so the per-draw
comparison is available and is much stronger than a rate comparison. Over 2000 matched
draws:

- marker fails and std_lv succeeds: **7**
- std_lv fails and marker succeeds: **0**
- both fail: 0

A dominance relation, not a rate difference. There is no draw in this design where the
marker chart succeeds and std_lv does not.

**Caveat that matters:** improper solutions occur in **1 of 6000 draws** across the
whole arm, which is far too few to compare charts on, so this design does not stress
admissibility and says nothing about the exp 03 Heywood finding below. Loadings of 0.7
with `psi = 0.51` are not extreme enough. A harder population is needed before the two
results can be put on one grid.

## Structural models reverse the verdict

Everything above is CFA, where every latent is exogenous and the `Psi` diagonal is the
latent's **total** variance. In a structural model that stops being true, and the
consequence is not a caveat but a reversal.

`std_lv` works mechanically. On `f3 ~ f1 + f2` over three measured factors, marker and
`std_lv` give identical `fmin` (0.1417035245), chi-square (85.3055) and df (24), so it
is still a reparameterization. Second-order factors (`g =~ f1 + f2 + f3`) likewise
agree to all digits. No implementation gap.

The problem is *what* it fixes. For an endogenous latent, `f ~~ f` is the **residual**
variance, not the total variance, so `std_lv` pins the residual to 1 and the latent is
not standardised at all. Its implied total variance reaches **10.3 at R² = 0.90 and
50.3 at R² = 0.98**. And unlike a total variance, a residual variance is **not bounded
away from zero**: it goes to zero as R² goes to one. So the gauge loses its grip, and
`std_lv` acquires precisely marker's failure mode with a different trigger. Marker
degenerates when the marker indicator is weak; `std_lv` degenerates when the
endogenous latent is well explained.

Conditioning of the expected information, one latent regressed on another:

| R² | residual var | marker | std_lv | ratio |
|---:|---:|---:|---:|---:|
| 0.01 | 0.990 | 11.05 | 4.75 | 0.43 |
| 0.25 | 0.750 | 11.49 | 9.93 | 0.86 |
| 0.49 | 0.510 | 12.83 | 23.7 | **1.85** |
| 0.81 | 0.190 | 23.23 | 373 | 16.1 |
| 0.90 | 0.098 | 30.91 | 3627 | 117 |
| 0.98 | 0.020 | 41.43 | **1.53e6** | **36887** |

**The crossover is near R² = 0.4**, which is an ordinary value in published structural
models, not a corner case. Marker's conditioning barely moves across the whole sweep.

At R² = 0.98 this stops being about speed. Roughly **10 percent of `std_lv` fits land
at a worse optimum than the marker fit on the same data**, and both charts parameterise
the same manifold, so that is unambiguously an optimizer failure. On one such draw:

| engine / chart | converged | fmin |
|---|---|---:|
| marker, all three magmaan backends | TRUE | 0.0166472682 |
| lavaan, marker | TRUE | 0.0166472682 |
| `std_lv`, nlopt-lbfgs | **TRUE** | **0.0171836769** |
| `std_lv`, port | FALSE | 0.0171376063 |
| `std_lv`, nlopt-slsqp | error | -- |
| lavaan, `std.lv` | failed | "a solution has NOT been found", negative lv variances |

So it is not a magmaan weakness: lavaan fails on the same cell, louder. But note the
third row. magmaan's `nlopt-lbfgs` reports success at a point 3.2 percent above the
optimum in `fmin` (chi-square 13.75 against 13.32), and the terminal audit agrees:
`stationary = TRUE`, `grad_inf_norm = 2.51e-05` against a `stationarity_rhs` of 1e-3,
`verdict$stationarity = "passed"`.

That is **not a broken check**. With a condition number near 1e6 the objective really
is flat there, so the point genuinely is stationary to that tolerance while sitting far
from the minimum in function value. A gradient-norm criterion cannot distinguish the
two. Which is the sharpest available argument for taking conditioning seriously: bad
conditioning does not merely cost iterations, it silently corrupts the answer and
defeats the stationarity audit that exists to catch exactly this.

**Practical rule.** `std_lv` is safe for exogenous latents and risky for endogenous
ones with high R². The natural repair is a chart that fixes the latent's **total**
variance rather than its residual, which would be well-conditioned in both roles. That
is not what lavaan's `std.lv` does, and it cannot be expressed by fixing a single
parameter, because the total variance involves the structural coefficients. It is a
*nonlinear* gauge condition, so magmaan's existing nonlinear equality-constraint
machinery could express it. Speculative, and it would need its own conditioning study
before anyone believed it.

## Growth models: not a reparameterization at all

The user-remembered difficulty. With every loading user-fixed there is nothing to
absorb the rescaling, so fixing the latent variances adds real restrictions instead of
renaming coordinates. On `Demo.growth`, both magmaan and lavaan go **npar 9 to 7, df 5
to 7, chi-square 8.0687 to 106.8532**. magmaan reproduces lavaan exactly, so this is
faithful rather than a bug, and it is pinned by the "std.lv on all-fixed loadings adds
constraints, not coordinates" case in `cpp/tests/unit/lavaanify_test.cpp`.

The general condition: `std_lv` is a change of coordinates exactly when the latent's
scale is otherwise free, meaning at least one loading on it is free to absorb the
rescaling. Growth models violate that by construction.

## Two things that were wrong

**"The Fisher-orthogonal slice is the optimal gauge."** Nonsense. Moving along a gauge
orbit does not change Sigma, so the orbit direction lies in the *null space* of the
Fisher information: `I v_orbit = 0`, hence `u' I v_orbit = 0` for every `u`. Every
direction is Fisher-orthogonal to the orbit and the criterion is vacuous. The
defensible statement is the flat-metric one above.

**"std_lv removes a class of improper solution because phi is pinned."** Refuted by
`experiments/_archive/03-heywood-box-constraints`: *"std.lv without bounds, 0/3
std.lv regular-start cases admissible in both engines. The latent metric alone does
not remove the Heywood behavior."* Under marker the damage lands in latent variances
(-2.34, -0.21); under std_lv it relocates to observed variances (-0.0039, -53.7,
-148.6). The winning recipe there is marker plus nonnegative variance bounds, 2/3.
std_lv helps the optimizer, not admissibility, so exp 82's improperness detector
reads every estimated variance rather than the latent one.

## Standing verdict

**Split by whether the latent is exogenous. There is no single answer.**

*Exogenous latents (so: all of CFA).* `std_lv` wins conditioning, curvature, optimizer
work, wall-clock fit time (0.43 of marker at `p = 24`), and small-n convergence
(strictly dominant, 7-0 across matched draws). It loses on admissibility, where it
relocates Heywood cases rather than removing them. It does **not** tie on end-to-end
speed, which was exp 02's conclusion and is superseded: the back-conversion costs about
three percent against a 30 to 57 percent saving.

*Endogenous latents.* The ranking reverses above **R² ≈ 0.4**, because `std_lv` pins
the residual variance rather than the total variance and a residual variance is not
bounded away from zero. By R² = 0.98 it produces wrong fits, one in ten of them
reporting success. So `std_lv` is not a safe blanket internal default for structural
models, and anything that adopted it as one would need to condition on the latent's
role and its explained variance.

*Growth or any all-fixed-loadings block.* Not a reparameterization at all. Do not treat
it as an internal substitution under any circumstances.

The reported parameterization should still stay whatever the user asked for, because
the case for `std_lv` is about internal numerics and not interpretation, and the
back-conversion is exact to 1e-15 so nothing is lost by honouring the request.

## "standardized" in the closed-form work is a different axis

Worth stating because the names collide. In `papers/guttman-inference` and
`papers/closed-form-omega`, `standardized` is the **composite-weight** choice
`A_std = diag(S)^{-1/2} Z`, a within-factor reweighting that changes the composite
direction, not `std.lv`. The extraction metric there is UVI by construction with
marker applied afterwards. The supporting evidence is thinner than the recipe's
prominence suggests: exp 58's paper-grade run pins `aligned_composite = standardized`
as a fixed setting with no composite-versus-composite arm, exp 55's map probe is
`reps=8`, exp 59 is `reps=5`, and `Remark 2` of the paper says outright that the
choice is a declaration. Exp 58 does carry an `equal`/`unequal` indicator-scale factor,
which is where the standardized composite should earn its keep, but no scale-sliced
result has been written up.

## Where parameterization bites harder than it does under ML

For ML a convention is a coordinate change and the fit is invariant. For the
closed-form estimators it is part of the estimator's *definition*:

- `experiments/_archive/56-noniterative-constraint-charts` shows the marker chart is
  irrelevant to the closed-form fit, moving the implied covariance by 8.9e-16
  configural and 1.3e-15 metric-constrained while coordinates move by 1.32 and 0.90.
- But `guttman_estimator_criterion.tex` records that *"marker-style scalings that
  depend on the loading matrix can make the same constraint linear in one chart and
  nonlinear in another"*, which decides whether the restricted fit stays closed-form
  at all. Different `V_Y` also define different off-model targets.
- Under `std.lv` ordinal delta the SNLLS arm is structurally N/A, because the
  conditionally-linear block empties out once the latent variance is fixed.

There is also one documented case where a convention is a *different model* rather
than a reparameterization: `growth(std_lv = TRUE)` takes lavaan from npar 9 to 7,
df 5 to 7, and chi-square 8.07 to 106.85, silently. That is the all-fixed-loadings
exception, pinned by the `"std.lv on all-fixed loadings adds constraints, not
coordinates"` case in `cpp/tests/unit/lavaanify_test.cpp`.

## Open

- **The start-value confound.** In exp 82's cost arm std_lv needs roughly a third of
  marker's function evaluations, and the gap persists at `lambda_1 = 0.9` (66 against
  19 at `p = 12`) where marker is competitive on both curvature and conditioning.
  Geometry does not explain that, so magmaan's per-chart start heuristics are the
  remaining suspect. This is the most actionable item here.
- **The p-scaling of the internal-chart substitution.** The back-convert falls from
  5.7 percent of the fit at `p = 6` to 3.6 percent at `p = 12`, against exp 02's
  roughly 18 percent, so the substitution looks better than exp 02 concluded. Needs
  the `p = 24` arm and a real `--full` run.
- **Chart-free inference is the principled escape.** The LRT is already invariant to
  eight digits, so every gain from chart-hunting is bounded by the intrinsic-curvature
  floor. Profile-likelihood test-inversion intervals sidestep gauge dependence
  entirely, which sharpens the motivation for the deferred `funLR` project beyond
  small-sample coverage.
