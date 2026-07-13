# Residual-score flip GOF probe

This experiment asks whether the sign-flip idea extends from affine nested
score tests to single-model goodness of fit. It is deliberately a probe, not a
new supported magmaan test and not a replication of Foldnes--Moss--Gronneberg.

The first slice is single-group, complete-data, covariance-only ML. For each
case it forms the model-centred saturated covariance contribution and projects
it through magmaan's expected-information residual basis `B`. That removes the
fitted model tangent before signs are applied. The all-plus quadratic is
exactly the structured-weight RLS GOF statistic; the runner gates this identity
on every successful replication.

Two randomization statistics are compared:

- **effective residual flip:** Euclidean quadratic in the projected residual
  contributions, using the model-implied normal-theory metric;
- **covariance-standardized residual flip:** the same projected contributions
  whitened by their empirical cross-product before the flip rank is computed.

The second operation is on the efficient/projected contributions themselves.
It is not experiment 62/64's flip-specific nuisance re-estimation correction:
the GOF residual basis has already removed the nuisance tangent, and its
empirical cross-product is invariant to signs. No ridge is chosen. Consequently
the standardized arm is reported unavailable when residual df reaches sample
rank, rather than becoming an ad hoc regularized test.

## Design

- one-factor CFA with `p = 5` (df 5) or `p = 20` (df 170);
- `n = 100` or 400;
- normal or the severe polynomial-logistic construction targeting marginal
  skewness 3 and excess kurtosis 21;
- correct specification or one omitted residual covariance, `theta[1,2]=.25`;
- effective/standardized flips plus standard, SB, MV, SS, pEBA4, and exact-ALL
  FMG transforms with both ML and RLS base statistics;
- cellwise rejection, null p-value histograms, paired decisions, empirical-meat
  conditioning, nominal/matched-null power, and phase timing.

The loading vector and severe marginal targets are the familiar published-DGP
ingredients used in experiments 63/64. The GOF misspecification is intentionally
new and simple; it is not attributed to that paper.

## Commands

```sh
Rscript experiments/65-residual-flip-gof-probe/validate_probe.R
Rscript experiments/65-residual-flip-gof-probe/run_experiment.R --smoke
Rscript experiments/65-residual-flip-gof-probe/run_experiment.R --probe
quarto render experiments/65-residual-flip-gof-probe/report.qmd
```

The 16-cell smoke uses two replications and 19 random flips. The small probe
uses 200 replications and 199 flips. Polynomial-logistic calibration is cached
across the two sample sizes inside each `(p, distribution, truth)` DGP task, and
is timed separately from fitting and inference.

## Interpretation gate

Carry the method into a core frontier API only if the probe gives useful null
calibration without an unacceptable high-df power loss. Treat standardization
as worthwhile only where its paired calibration gain exceeds its loss of
availability/conditioning. A later core design must separately settle grouped
weights, means, missingness, and whether empirical-meat shrinkage is principled
enough to rescue `df >= n`; none is silently decided here.
