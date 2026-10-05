# Small-N observed nested ML score diagnosis

The replay points to compression by the fitted quadratic geometry, rather than
uniform inflation of the meat. In the normal cells, the observed statistic has
mean 5.14/5.33 and variance 7.09/8.05 (correct/mild), while its fitted reference
mixture has mean 5.79/5.83 and variance 12.01/12.28. Under skewness the means are
10.08/9.57 versus 10.61/10.97, but the variances are 20.07/22.53 versus
55.78/64.87. Thus the variance shortfall is much larger than the mean shortfall.
This is a diagnostic of the conservative tail, not a fresh size estimate.

The estimated raw meat trace divided by the Monte Carlo covariance trace of
sqrt(total N) times the average projected score is 1.032/1.009 (normal) and
0.902/1.000 (skewed). These numbers do not support overall meat inflation as
the dominant explanation. Mean score vectors, covariance and mean-meat ordered
eigenvalues are retained in the summary CSV (the mean-vector norm is reported;
full vectors and matrices stay in ignored raw batches). Sorted eigenvalues
compare marginal spectra, not matched eigendirections.

Freezing observed sensitivity at the population value raises the normal
statistic variances to 8.17/10.15 and skewed variances to 30.92/38.76. This
isolates a contribution from noisy nuisance projection, but leaves substantial
compression. The expected-sensitivity comparator gives 10.71/12.26 and
46.56/53.57. Relative to that comparator, the observed statistic variance is
34% lower in normal cells and 57–58% lower in skewed cells. The realized
observed sensitivity differs from population sensitivity by mean relative
Frobenius norm 0.366/0.369 (normal) and 0.744/0.716 (skewed).

An additional arm freezes both sensitivity and expected metric at population
values. Its statistic variance rises to 11.46/16.66 (normal) and 86.52/87.67
(skewed), implicating sample-dependent metric/studentization as well as
projection. This arm is a diagnostic, not a proposed usable test: the fitted
meat still fails to reproduce its reference-law shape, especially under
skewness (reference variances 187.1/233.9). Freezing just sensitivity changes
the meat/covariance trace ratio to 1.082/1.058 and 1.086/1.209. There is no
single scalar meat bias that explains all arms.

The dominant observed-versus-expected shortfall in this replay is therefore
**sample-dependent projection and quadratic metric compression of the statistic's
variance**, with additional plug-in reference-shape error under skewness.
The interventions identify contributions; they do not establish a unique
higher-order expansion or prove which mechanism alone determines 5% rejection.

Two corrections warrant a later registered study:

1. A nuisance-refitting bootstrap or multiplier procedure that recomputes
   observed sensitivity and the expected metric in each resample, approximating
   the joint studentized statistic rather than holding fitted geometry fixed.
   Its population approximation must preserve misspecification; a Gaussian
   bootstrap from the fitted SEM would not do that.
2. A higher-order moment correction for the joint projected-score/metric law,
   estimating the statistic's mean and variance including geometry-estimation
   terms. A uniform HC-style meat reduction is poorly motivated by these trace
   ratios and could aggravate the skewed correct-model cell.

Both require fresh seeds, size and size-adjusted power, a larger-df family and
strong misspecification before adoption. No library default changes follow.

The replay uses the first 200 production seeds for each of cells 1, 5, 9, 13:
N=100 per group, true null, correct/mild larger model, normal/skewed data.
It reuses this study's generator and cell inventory; seed base 726100041 and
seed = base + 10000 * cell ID + replicate. All 800 draws converged and both
sample-sensitivity arms matched the policy statistic within 1e-7. Population
sensitivity and metric use ML fits to exact population covariance moments with
1000 observations per group; ML geometry depends only on those moments, so
this is not a noisy large-N simulation. Population and sample tangent matrices
are checked to agree before projection. Expected sensitivity is a comparator
only. The quadratic metric stays expected in the observed arm. Casewise meat
is raw and uncentered. Here n denotes total N=200; projected score vectors are
on a common six-dimensional parameter-direction basis across draws.

Reference moments use sum(lambda) and 2 sum(lambda²); the reported variance
also includes variance across fitted reference means. This accounts for the
mixture across replicate-specific laws. With only 200 draws per cell,
Monte Carlo variance estimates, especially skewed ones, remain imprecise.
These are reused production seeds, exploratory mechanism evidence, not held-out
confirmation or an estimate of a correction's performance.

Frozen summary, failure count and provenance live in
`../results/score-diagnostic/frozen-2026-10-05/`. They are retained as tiny
mechanism evidence; raw score vectors, meat matrices, metrics and spectra stay
ignored. The parent report links this note. A clean checkout needs installed
magmaan, magmaanlab and lavaan; it does not need any raw production files.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
R_LIBS="$HOME/.cache/magmaan-rlib/lane-e" nice -n 10 Rscript \
  experiments/decisions/04-nested-ml-geometry/diagnostics/replay_score.R \
  --reps 200 --run-id fresh-replay
```

Use `--reps 2 --run-id fresh-smoke` for the smoke path and `--help` for usage.
The two-worker runner stops starting batches after 260 seconds; the requested
frozen replay completed in 35.1 seconds; the initial and additional diagnostic
runs together also stayed below five minutes. Exact runtime, binary fingerprint,
package versions and command are in metadata.csv. Any failures are saved and
cause a nonzero exit.
