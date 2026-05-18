# DLS Empirical-Bayes Checks

This directory holds local Monte Carlo checks for the empirical-Bayes DLS
mixing-scalar builder. It is intentionally separate from the package build,
root `justfile`, and CI tests.

The driver simulates one-factor complete continuous data and reports the
estimated `a_EB` for three conditions:

- multivariate normal data, where the scalar should usually shrink toward GLS;
- elliptical heavy-tailed data, where stable fourth-moment departures should
  move the scalar toward ADF/WLS;
- skewed data, as a second non-normal stress case.

Run locally from this directory:

```sh
just all
```

For a quick smoke run:

```sh
just quick
```

The output is advisory. Sampling error and individual replicate failures are
reported rather than treated as test failures.

Reference: Du, H., & Wu, H. (2024). Estimating the Weight Matrix in
Distributionally Weighted Least Squares Estimation: An Empirical Bayesian
Solution. *Structural Equation Modeling: A Multidisciplinary Journal*, 31(6),
952-964. https://doi.org/10.1080/10705511.2024.2337661
