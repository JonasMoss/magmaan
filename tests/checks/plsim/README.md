# PLSIM Calibration Checks

Advisory local checks for the piecewise-linear simulation calibration paths.
They are intentionally outside the default CTest suite because timings depend
on the build type and machine.

The driver compares:

- Hermite-series covariance calibration;
- bivariate Gauss-Hermite quadrature calibration;
- conditional-normal rectangle-moment calibration;
- Hermite-initialized quadrature refinement.
- Hermite-initialized rectangle refinement.

For each condition it reports calibration time, strategy-implied target
residual, and the same calibrated intermediate correlations re-evaluated by the
quadrature and rectangle evaluators. The rectangle path evaluates the
Foldnes-Gronneberg segment-rectangle decomposition with conditional-normal
one-dimensional adaptive integration, avoiding the kink-smoothing behavior of
full-plane Gauss-Hermite quadrature.

Run locally from this directory:

```sh
just quick
```

For a longer run:

```sh
just all
```

The columns are:

- `own_err`: maximum absolute residual under the strategy's own covariance
  evaluator;
- `quad_err`: maximum absolute residual after re-evaluating the calibrated
  intermediate correlations with the quadrature reference;
- `rect_err`: maximum absolute residual after re-evaluating the calibrated
  intermediate correlations with the rectangle-moment evaluator;
- `rho_delta`: maximum latent-correlation difference from pure quadrature;
- `samp_corr` / `samp_shape`: stochastic sample diagnostics from one generated
  sample, useful only as a rough smoke check.

On the initial quick smoke (`reps=3`, `sample_n=20000`), Hermite-only was much
faster and agreed with the rectangle evaluator at the requested tolerance.
Pure quadrature and Hermite-initialized quadrature agreed with each other, but
were about `1e-3` away from the rectangle/Hermite paths in the smoke cases.

Useful sweep knobs:

```sh
build/plsim_calibration_bench --reps=1 --sample-n=20000 --hermite-order=60
build/plsim_calibration_bench --reps=1 --sample-n=20000 --hermite-order=120
```

Increase `--quadrature-points=` when checking whether the quadrature reference,
rather than Hermite truncation, is the limiting term.

Initial sweeps showed that increasing Hermite order from 12 to 96 did not
materially change the Hermite-only quadrature-reference residuals for the smoke
cases. Increasing quadrature nodes did move those residuals, sometimes
non-monotonically, which is consistent with Gauss-Hermite quadrature being a
rough deterministic reference for kinked PL transforms. Prefer the rectangle
path when judging calibration accuracy.

The stress cases include high positive correlation, negative correlation with
heavy skewed tails, AR(1)-style high shape, and mixed-sign correlation patterns.
The high-positive and AR high-shape cases are useful feasibility checks: they
currently fail because either no pairwise intermediate root exists or the
assembled intermediate matrix is not positive definite. The negative-correlation
tail case is the clearest Hermite-order stress: with order 6, Hermite can be
several `1e-3` away from Rectangle; by order 12 this dropped below `5e-4`, and
by the default order 30 it was around `5e-5` in the local quick run.

## Cross-check against covsim (`plsim_vs_covsim.R`)

`plsim_calibration_bench.cpp` is self-consistency only (magmaan's three
integrators agree). `plsim_vs_covsim.R` is the external check: it compares the R
bindings against covsim's `rPLSIM`, the authoritative Foldnes-Gronneberg
implementation vendored at `external/r_source/covsim/R/`. Like
`tests/checks/ig/pearson_draw_bench.R`, it sources covsim from `external/` and
skips gracefully when covsim (or `tmvtnorm`/`MASS`/`Matrix`/`lavaan`) is absent.
It is advisory, not in CTest.

Run it after installing the R bindings (`just opt && just r-dev` from the repo
root):

```sh
just vs-covsim                 # n=100000, num_segments=4, all five methods
just vs-covsim --n=200000      # extra args pass through to the script
```

The **headline** is a covariance cross-check that avoids a subtle confound: the
piecewise-linear marginal for a given (skewness, excess kurtosis) is *not
unique*, so covsim and magmaan generally fit different slopes and therefore
different intermediate correlations even when both are correct. Comparing the raw
intermediate matrices is thus not apples-to-apples. Instead, for each method the
check feeds magmaan's **own** fitted marginals and solved intermediate
correlation into covsim's exact truncated-moment integrator `get_cov()` and
reports `|get_cov - target|`. That isolates magmaan's covariance integral plus
root-find and measures it against covsim's integrator, free of the marginal
ambiguity. `rectangle` (the conditional-normal integrator closest to covsim's
`tmvtnorm`-based `get_cov`) is the pass criterion; the other four methods are
reported for context and are expected to show the documented Hermite-truncation
gap (~`1e-3`).

Per case the script also reports: realized marginal moments (covsim's
`get_pl_moments` on magmaan's fit, vs target); magmaan's self-reported achieved
correlation; the exact `gamma` breakpoints (both `qnorm((1:k)/k)[1:(k-1)]`); the
raw `z.corr` difference (flagged informational, per the non-uniqueness above);
and large-`N` sample moments for both engines. A final discrimination probe
perturbs the solved intermediate correlation by `0.05` and confirms the headline
residual jumps, proving the check has teeth rather than passing vacuously.

Notes and constraints:

- The grid uses moderate shapes (the workhorse case is covsim's own docstring
  example, skew `1` / excess kurtosis `4`) on positive-definite correlation
  targets, so covsim fits at `num_segments=4`; extreme or non-PD cases are out of
  scope here (covsim repairs a non-PD intermediate via `nearPD`, magmaan via
  Cholesky jitter, so those would not be comparable). The headline still works
  even if covsim's `rPLSIM` fails to fit a shape, since it uses magmaan's own
  marginals.
- The magmaan R binding defaults to `num_segments=12`; the check pins
  `num_segments=4` to match covsim's default. Pass `--num-segments=` to vary it.
- RNG is not shared (R's Mersenne vs magmaan's `std::mt19937_64`), so sample
  moments are paired by target, not by draw; use large `N`.
- A future `sim_plsim_covariance(marg_i, marg_j, rho, method)` export would let
  the covariance integral be isolated without routing through covsim's `get_cov`;
  deferred for now.
