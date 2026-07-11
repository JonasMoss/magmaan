# Permutation Measurement Invariance

Experiment 61 evaluates fully recomputed, studentized Wald permutation tests
for multi-group CFA measurement invariance. It is a paper experiment, not a
single-method benchmark: every permutation relabels cases, rebuilds the
estimation input, refits the unrestricted model, and recomputes the statistic.

## Target Design

The next experiment revision is deliberately fractional. It separates the
primary continuous claim from the ordinal validity gate rather than crossing
every dimension in one uninformative large grid.

| Module | Data generation | CFA size | Invariance steps | Purpose |
| --- | --- | --- | --- | --- |
| Continuous primary | Normal; Pearson independent generator | 1, 3, 5 factors | Metric, scalar, strict | Main size and power comparison |
| Ordinal feasibility | Grouped ordinal-correlation simulator, 4 categories | 1, 3 factors | Threshold/loading invariance | Test whether raw-label permutation is empirically viable |

The target continuous designs use balanced and unbalanced group sizes plus heterogeneous
group factor/residual scales. The exact-exchangeability control is retained to
separate a resampling implementation failure from failure under the
heterogeneous invariant null. The independent-generator condition replaces the
old multivariate-t condition: it preserves the target covariance while varying
the marginal mechanism through magmaan's calibrated Pearson IG generator.

The continuous alternative is calibrated to comparable sandwich-Wald signal in
the normal `n = (250, 250)` reference cells. Raw and implied standardized
loading departures are reported; equal standardized departures were rejected
because they generated radically unequal power for a raw-loading test.

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

The ordinal module will use theta-DWLS and rebuild thresholds, polychorics, NACOV, weights,
and fits after every label permutation. This arm is an empirical feasibility
gate only: Chung--Romano asymptotic pivotality has not yet been established for
the pooled ordinal moment estimator. It does not enter a main paper claim
unless its heterogeneous-null controls pass.

## Target Report

The report will present a design manifest, size tables, power tables, fit failure
rates, and the fraction of usable permutations. It does not collapse ordinal
and continuous results into one table or hide invalid permutation draws.
