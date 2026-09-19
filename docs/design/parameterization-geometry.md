# Parameterization geometry: what a latent scaling convention can and cannot buy

**Status:** findings recorded 2026-09-19. Measured by
`experiments/82-latent-metric-geometry`, which supersedes
`experiments/_archive/02-latent-metric-identification`.

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

std_lv wins conditioning, curvature, and optimizer work. It loses on admissibility,
where it relocates rather than removes. It ties on end-to-end speed. It is not a
clean win, and the reported parameterization should stay whatever the user asked for,
since the case for std_lv is about internal numerics and not interpretation.

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

- `experiments/56-noniterative-constraint-charts` shows the marker chart is
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
coordinates"` case in `tests/unit/lavaanify_test.cpp`.

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
