# Permutation Measurement Invariance

Experiment 61 evaluates fully recomputed, studentized Wald permutation tests
for multi-group CFA measurement invariance. It is a paper experiment, not a
single-method benchmark: every permutation relabels cases, rebuilds the
estimation input, refits the unrestricted model, and recomputes the statistic.

## Target Design

The next experiment revision is a continuous-data main study. It follows the
two-group measurement-invariance design spine used by Brace and Savalei (2017)
and the repository's FIML-FMG work, rather than crossing arbitrary CFA sizes.

| Module | Data generation | CFA size | Invariance steps | Purpose |
| --- | --- | --- | --- | --- |
| Continuous main study | Normal; Pearson IG moderate/severe nonnormality | Two correlated factors; `p = 8,16` indicators | Metric, scalar, strict | Main size and power comparison |
| Categorical companion | Ordinal-probit projection, analyzed as integer codes by robust continuous ML | Two correlated factors; `p = 8,16` indicators | Metric, scalar, strict | Categorical size and power comparison |

The main grid uses total sample sizes `N = 220, 440, 1760` and group-size ratios
`1:1` and `1:3`. The null has equal loadings, intercepts, and residual variances
but unequal group factor variances and latent means, so it preserves the
non-exchangeable nuisance structure relevant to the permutation claim. The
exact-exchangeability control is retained at the reference `p = 8`, `N = 440`
cell to separate a resampling implementation failure from failure under the
heterogeneous invariant null.

The nonnormal conditions are the calibrated Pearson independent generator at
the moderate `(skew, excess kurtosis) = (2, 7)` and severe `(3, 21)` settings
used by the repository's FIML-FMG studies. They replace the old multivariate-t
condition: t is symmetric and is a weak stressor for several robust difference
tests, while the IG generator preserves each group's target covariance and
varies the marginal mechanism in a literature-recognized way.

The categorical companion draws latent-response data from magmaan's grouped
ordinal-correlation simulator and projects each indicator to ordered categories.
It then analyzes the numeric category scores with robust continuous-data ML,
not polychorics, thresholds, or DWLS. This is the deliberately misspecified but
common applied workflow studied by Rhemtulla, Brosseau-Liard, and Savalei
(2012). Category count and threshold asymmetry are explicit factors in this
arm, because continuous ML is most defensible with at least five approximately
symmetric categories.

Each metric, scalar, and strict alternative is calibrated separately to 50%
studentized-permutation power in the normal, balanced, `N = 440` reference
cell. The calibrated population departure is then held fixed over sample size,
allocation, indicator count, and generator conditions. This is the repository's
existing power-calibration convention; it is preferable to a fixed raw shift
and serves the same comparability role as the RMSEA-anchored alternatives in
Brace and Savalei's design.

## Target Test Definitions

Metric invariance tests equality of non-marker loadings from a configural CFA.
Scalar and strict tests use a reference-group mean chart: in the unrestricted
scalar chart, each factor has its group-1 mean fixed, its group-2 mean free,
and one marker intercept tied across groups. This is observationally equivalent
to the conventional metric mean model, but makes the remaining intercept
restrictions explicit and nested. Strict invariance then tests equality of the
residual variances from the scalar chart.

The primary comparison in each continuous cell will be ordinary LRT, robust nested
LRT, model-based Wald, sandwich Wald, studentized permutation Wald, and the
raw-contrast permutation negative control.

## Target Report

The report will present a design manifest, size tables, power tables, fit failure
rates, and the fraction of usable permutations. It will show the three ordered
invariance steps separately and will not hide invalid permutation draws.
