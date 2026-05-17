# Robust Polychoric Sanity Checks

This directory holds local simulation checks for the experimental robust
all-ordinal polychoric track. It is intentionally separate from the package
build, root `justfile`, and CI tests.

The current driver checks four surfaces:

- pair-local h-weighted polychoric estimates, comparing the empirical
  covariance of `sqrt(N) * theta_hat` to the average sandwich Gamma returned by
  `data::ordinal_pair_h_weighted_influence()`;
- an all-ordinal Holzinger-Swineford-style correlation-matrix moment vector,
  comparing empirical `sqrt(N)` moment covariance to `OrdinalStats::NACOV`;
- a pair-level full polyserial DPD comparison between ML and
  `data::fit_polyserial_pair_joint_dpd()`, which jointly moves the continuous
  mean/scale, ordinal thresholds, and rho;
- an HS-style ordinal CFA fit, comparing empirical `sqrt(N)` parameter
  covariance to `N * robust_ordinal(...).vcov`.

The all-ordinal checks use the experimental h-weighted WMA hard-cap path. Full
polyserial DPD is intentionally pair-local here; SEM-level mixed DPD integration
needs an explicit design for pair-local thresholds in the shared
`MixedOrdinalStats` moment/Gamma layout.

Terminology: DPD means density power divergence. robcat parity applies only to
the h-score/WMA C-estimators, not to DPD.

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
