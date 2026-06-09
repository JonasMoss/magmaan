# FIML nested FMG smoke checks

Advisory simulation checks for the FIML nested/model-pair restriction-map
route. These are not part of `ctest`: they are stochastic and fit two FIML
models per replicate.

The script generates data under a one-factor CFA model where a nested loading
equality restriction is true:

```text
H1: f =~ x1 + x2 + x3 + x4 + x5
H0: f =~ x1 + a*x2 + a*x3 + a*x4 + x5
```

It then imposes MCAR missingness, fits both models with FIML, and calls
`nestedTest(..., method = "restriction_map")`. The reported rejection rates are
stochastic smoke checks for the nested p-values:

- unscaled chi-square;
- mean-scaled/Satorra-Bentler;
- mean+variance adjusted;
- scaled+shifted;
- exact mixture through Imhof.

`--dist=t --t-df=5` keeps the model restriction true while making the data
heavy-tailed, so normal-theory and robust nested behavior can be compared. This
is a smoke check, not a calibration claim: early runs show the robust
restriction-map p-values are much less anti-conservative than the unscaled
normal-theory difference, but the nested heavy-tail case needs a larger
simulation grid before treating it as nominal.

## Run

```sh
just quick
just heavy
just nominal
```

The script assumes the R package has already been installed, for example with
`just r-install-fast` from the repository root.
