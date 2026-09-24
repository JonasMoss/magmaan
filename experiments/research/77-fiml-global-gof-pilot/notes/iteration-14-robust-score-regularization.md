# Iteration 14: can the direct robust score avoid spectral correction?

## Question

Can the genuinely sandwich-studentized projected score

\[
T_{\mathrm{sand}}=u'(G'\widehat B G)^{-1}u
\]

provide a useful chi-square goodness-of-fit test without SB, pEBA, or another
approximation to a weighted chi-square distribution? If its finite-sample
problem is matrix instability, can an untuned regularization repair it?

## Diagnostics

`global_score_flip_test()` now returns the summed projected score, its
quadratic metric, and the raw projected OPG meat. The experiment uses these to
reproduce the existing direct sandwich statistic and compare:

1. raw OPG with a chi-square reference;
2. globally centered OPG with a chi-square reference;
3. the exact one-sample Hotelling transformation of the centered quadratic;
4. centered meat shrunk toward a trace-matched expected-information metric,
   with either
   \(\rho=d/(n+d)\) or
   \(\rho=\sqrt{d/n}/(1+\sqrt{d/n})\).

Both shrinkage weights depend only on dimension and sample size and vanish for
fixed dimension. They were fixed before examining the new simulations. No
rejection-rate tuning was used.

The initial results localized a second instability in the nuisance projection.
Two parallel diagnostics therefore shrink the observed H0 sensitivity toward
expected Fisher information, using the same light and square-root rules with
the nuisance-tangent dimension in place of test df. These also vanish under
fixed-dimensional asymptotics. They are exposed as
`sensitivity = "observed-shrink-light"` and
`"observed-shrink-sqrt"` on the frontier global-score function.

## Exact nonnormal-MAR pseudo-null

The exact three-df block-independence pseudo-null from iteration 13 was rerun
with 1,000 replications in 12 cells: \(n=200,500,2000\), crossed with MAR
slopes 0, 1.5, 3, and 5. All 84,000 score constructions were usable.

For observed-H0 sensitivity and the expected metric, rejection-rate summaries
over the 12 cells were:

| Reference | Mean size | Mean \(|\mathrm{size}-.05|\) | Min | Max |
|---|---:|---:|---:|---:|
| Score SB | .054 | .0090 | .040 | .077 |
| Score pEBA(4) | .053 | .0089 | .040 | .074 |
| Direct sandwich, raw OPG | .052 | .0193 | .025 | .114 |
| Direct sandwich, centered Hotelling | .053 | .0189 | .025 | .114 |
| Direct sandwich, light meat shrinkage | .056 | .0183 | .028 | .123 |
| Direct sandwich, square-root meat shrinkage | .054 | .0174 | .024 | .109 |

Square-root meat shrinkage helps slightly, but the result remains about twice
as variable around .05 as SB or pEBA(4). Centering alone is liberal; the
Hotelling reference mostly reverses that change and does not materially
improve on the raw OPG statistic.

Expected-H0 sensitivity remains invalid under strong nonnormal MAR: direct
sandwich rejection is zero in every \(\beta=3\) and \(\beta=5\) cell. The
empirical meat does not repair a nuisance projection built from the wrong
population sensitivity.

Shrinking observed sensitivity also provides no general improvement. Across
the 12 cells, direct-sandwich mean absolute size error was .0193 without
shrinkage, .0185 with light shrinkage, and .0211 with square-root shrinkage.

## Representative latent-model screen

A second null screen used \(n=200\), 1,000 replications per cell, and three
pilot SEMs spanning 6, 10, and 15 indicators and 9, 34, and 87 df. Normal,
severe Vale--Maurelli, and severe independent-generator data were crossed with
complete sampling and 30% MCAR, giving 18 cells and 18,000 fits. Every fit
converged and all but one observed-sensitivity score construction was usable.

Here direct studentization fails badly:

| Reference | Mean size | Mean \(|\mathrm{size}-.05|\) | Min | Max |
|---|---:|---:|---:|---:|
| Score pEBA(4) | .027 | .033 | .000 | .095 |
| Score SB | .034 | .034 | .000 | .110 |
| Direct sandwich, square-root meat shrinkage | .354 | .304 | .100 | .727 |
| Direct sandwich, raw OPG | .536 | .486 | .102 | .997 |
| Direct sandwich, light meat shrinkage | .548 | .498 | .123 | .932 |
| Direct sandwich, centered Hotelling | .586 | .536 | .109 | .999 |

The poor FMG rates at 87 df also warn that \(n=200\) is a harsh setting for
the largest model, but the direct sandwich failure is qualitatively different:
it is grossly liberal rather than conservative and is already large at 9 and
34 df.

## Sensitivity, not only meat

A separate 500-replication normal-complete comparison held the meat estimator
fixed and changed only nuisance sensitivity:

| df | Expected | Observed | Light-shrunken observed | Sqrt-shrunken observed |
|---:|---:|---:|---:|---:|
| 9 | .042 | .096 | .096 | .090 |
| 34 | .056 | .246 | .240 | .230 |
| 87 | .028 | .394 | .392 | .370 |

Under normality, expected Fisher is the correct population sensitivity and is
far more stable. The MAR-consistent observed Hessian approaches the same limit
but is much too noisy in finite samples, especially after inversion in a
moderate-dimensional nuisance tangent. The two predeclared vanishing
shrinkages are much too weak to repair this. Making them strong enough at
\(n=200\) would amount to tuning toward the expected-Fisher answer and would
reintroduce the nonnormal-MAR inconsistency that motivated observed
sensitivity.

## Decision

The direct robust score is not a replacement for score SB or pEBA(4) in the
planned paper. Its asymptotic pivot is attractive, but practical SEM sample
sizes expose two interacting plug-in problems: inversion of the empirical
efficient-score meat and instability of the observed-sensitivity nuisance
projection. Hotelling calibration and simple, predeclared vanishing shrinkage
do not solve them.

Keep the projected geometry and shrinkage options as research diagnostics, but
do not add these variants to the publication battery or tune a fixed
shrinkage constant on the pilot grid. The cleaner paper story remains the
weighted-chi-square score statistic, with SB as the simple correction and
pEBA(4) as the current performance candidate.

## Reproduction

```sh
Rscript experiments/research/77-fiml-global-gof-pilot/investigate_pseudonull_information.R \
  --reps 1000 --n 200,500,2000 --beta 0,1.5,3,5 --cores 4

Rscript experiments/research/77-fiml-global-gof-pilot/run_sem_models.R \
  --reps 1000 --n 200 --flips 1 --cores 4 \
  --models one_factor_6,two_factor_fmg_10,three_factor_15 \
  --estimators FIML --distributions normal,vm2,ig2 \
  --missingness complete,mcar_30 --regions identified_null
```

The result CSVs are ignored local artifacts. The design, numerical summaries,
and decision are tracked here.
