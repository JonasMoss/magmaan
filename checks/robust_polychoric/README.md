# Robust Polychoric Sanity Checks

This directory holds local simulation checks for the experimental robust
all-ordinal polychoric track. It is intentionally separate from the package
build, root `justfile`, and CI tests.

The current driver checks three surfaces:

- pair-local h-weighted polychoric estimates, comparing the empirical
  covariance of `sqrt(N) * theta_hat` to the average sandwich Gamma returned by
  `data::ordinal_pair_h_weighted_influence()`;
- an all-ordinal Holzinger-Swineford-style correlation-matrix moment vector,
  comparing empirical `sqrt(N)` moment covariance to `OrdinalStats::NACOV`;
- the same HS-style ordinal CFA fit, comparing empirical `sqrt(N)` parameter
  covariance to `N * robust_ordinal(...).vcov`.

Robust polyserial/mixed ordinal checks are deliberately not included yet:
`docs/todo.md` still tracks mixed continuous/ordinal robust estimators as
waiting behind the all-ordinal robust Gamma milestone.

Run locally from this directory:

```sh
just all
```

For a quicker smoke run:

```sh
just quick
```

The Monte Carlo output is advisory. Sampling error, optimizer failures on
individual replicates, and the deliberately contaminated data-generating
process are reported rather than treated as test failures.
