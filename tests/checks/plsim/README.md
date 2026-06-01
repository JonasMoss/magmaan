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
