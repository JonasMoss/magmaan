# FIML H1 edge check

Advisory check for lavaan-like saturated-H1 behavior under high-dimensional
missing continuous data. This is intentionally outside `ctest`: it requires an
installed R package, optionally compares against installed lavaan, and exercises
slow/fragile p30 MCAR edge cases.

The script uses a deterministic five-factor CFA population with 30 indicators,
draws normal data, imposes MCAR missingness, and reports:

- lavaan convergence, H1 covariance minimum eigenvalue, H1 information minimum
  eigenvalue, and warnings;
- magmaan saturated EM Stage-1 warnings plus covariance/information eigenvalues.

The magmaan side calls `magmaan_core$estimate_saturated_em_moments()` directly on
raw matrices with `NA`s, so this check stays scoped to H1 Stage-1 behavior rather
than the parser/model-fit path.

The default `quick` recipe includes the known stress cases that motivated the
H1 robustness policy: lavaan returns the last H1 EM iterate with warnings rather
than aborting when the H1 EM cap or tiny covariance eigenvalues occur.

## Run

```sh
just quick
just harsh
```

Install the R package first, for example with `just r-install-fast` from the
repository root. Lavaan output is skipped when lavaan is not installed.
