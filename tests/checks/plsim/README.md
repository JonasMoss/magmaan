# PLSIM Calibration Checks

Advisory local checks for the piecewise-linear simulation calibration paths.
They are intentionally outside the default CTest suite because timings depend
on the build type and machine.

The driver compares:

- Hermite-series covariance calibration;
- bivariate Gauss-Hermite quadrature calibration;
- Hermite-initialized quadrature refinement.

For each condition it reports calibration time, strategy-implied target
residual, and the same calibrated intermediate correlations re-evaluated by the
quadrature reference. The quadrature path is the current deterministic local
reference until a specialized Foldnes-Gronneberg rectangle-moment kernel lands.

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
- `rho_delta`: maximum latent-correlation difference from pure quadrature;
- `samp_corr` / `samp_shape`: stochastic sample diagnostics from one generated
  sample, useful only as a rough smoke check.

On the initial quick smoke (`reps=3`, `sample_n=20000`), Hermite-only was much
faster but had quadrature-reference residuals around `1e-3`; pure quadrature
and Hermite-initialized quadrature agreed at the requested calibration tolerance.

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
rough deterministic reference for kinked PL transforms. The next precision step
is therefore the exact bivariate normal rectangle-moment evaluator, not simply
more Hermite coefficients.
