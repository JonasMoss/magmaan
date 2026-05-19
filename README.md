# magmaan <img src="docs/assets/logo/logo_compact.png" align="right" height="170" /></a>

> "This world ever was, and is, and shall be, an ever-living Fire."
> - Heraclitus

`magmaan` is the subterranean cousin of `lavaan`: a C++23 toolkit for methods developers working on linear SEM, with a focus on modularity, extensibility, and full control over moving parts.

**Status:** active prototype
**Language:** C++23, built with `-fno-exceptions -fno-rtti`
**Scope:** complete-data linear SEM under normal-theory ML plus ULS/GLS/WLS work
**Philosophy:** lavaan is the oracle; failures are values; APIs stay explicit and composable


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

**Currently out of scope:**
* Multilevel SEM
* Latent interactions/mixtures
* Non-linear constraints
* Non-interior solutions / non-standard asymptotics

**Probably always out of scope:**
* Bayesian models
* EFA
* IRT
