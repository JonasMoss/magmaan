# FIML FMG trace convergence check

Advisory check for the FIML FMG trace identity under the null model. This is
not part of `ctest`: it is stochastic, fits several FIML models, and is meant
to answer a methods/debugging question rather than guard a fixed fixture.

The script generates data from a one-factor CFA model, imposes MCAR missingness
independent of the generated values, fits the same model with FIML, and
compares:

```text
sum(eig(U Gamma_mis))
```

from `infer_fiml_fmg_spectrum()` against:

```text
fiml_robust_mlr()$trace_ugamma
```

Under the null and a compatible asymptotic convention, the gap should shrink
as `n` grows. Finite samples will not be monotone replicate-by-replicate,
because the H0 MLR trace still uses the fitted restricted-model observed
Hessian while FMG uses the saturated eta-space tangent projection.

The script also reports rejection rates for FIML FMG `std`, `sb`, `all`, and
`pall` p-values at `--alpha` (default 0.05). These are stochastic smoke checks:
the small recipes should be read directionally, while `just nominal` gives a
somewhat more useful PALL calibration run.

Two null distributions are supported:

- `--dist=normal`: ordinary normal-theory H0 data.
- `--dist=t --t-df=5`: heavy-tailed H0 data with covariance rescaled to the
  same CFA population covariance. The trace gap should still shrink, but the
  FMG eigenvalues need not be close to one.

## Run

```sh
just quick
just heavy
just nominal
just all
just all-heavy
```

The script assumes the R package has already been installed, for example with
`just r-install-fast` from the repository root.
