# magmaan <img src="docs/logo/logo_compact.png" align="right" height="170" /></a>

> "This world ever was, and is, and shall be, an ever-living Fire."
> - Heraclitus

`magmaan` is a C++23 toolkit for methods developers working on linear SEM. Its
near-term contract is lavaan-equivalent behavior for complete-data
normal-theory estimators, with explicit extension points for discrepancies,
optimizers, inference methods, fit measures, and R boundary experiments.

**Status:** active prototype
**Language:** C++23, built with `-fno-exceptions -fno-rtti`
**Scope:** complete-data linear SEM under normal-theory ML plus ULS/GLS/WLS work
**Philosophy:** lavaan is the oracle; failures are values; APIs stay explicit and composable

## Roadmap

The live roadmap is [docs/roadmap.md](docs/roadmap.md). In short:

1. Turn ULS/GLS/WLS support into lavaan-parity fixture coverage.
2. Close the remaining observed-information and robust multi-block gaps.
3. Finish R/API parity polish around defined parameters, inference shortcuts,
   and semantic partable comparisons.
4. Complete the namespace cleanup from transitional `fit`/`partable` headers to
   target public domains.

## Build

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev

cmake --preset opt
cmake --build --preset opt
ctest --preset opt
```

`just` wraps the usual loops: `just build`, `just test`, `just opt`,
`just test-opt`, `just r-check`, and `just check`.

## Scope Boundary

Currently out of scope: FIML/missing data, ordinal/DWLS/polychoric, Bayesian,
multilevel, latent interactions/mixtures, EFA, and end-user `cfa(model, data)`
ergonomics.
