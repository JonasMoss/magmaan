# magmaan <img src="docs/assets/logo/logo_compact.png" align="right" height="170" />

> "This world ever was, and is, and shall be, an ever-living Fire."
> - Heraclitus

`magmaan` is the subterranean cousin of `lavaan`: a C++23 toolkit for methods developers working on linear SEM, with a focus on modularity, extensibility, and full control over moving parts.

**Status:** alpha (v0.0.1) — no API-stability promise yet; the lavaan-parity core should stabilize first
**Language:** C++23, built with `-fno-exceptions -fno-rtti`
**Scope:** linear SEM by ML and FIML, plus ULS/GLS/(D)WLS least squares — continuous and ordinal/categorical data, with Satorra-Bentler-family robust tests
**Philosophy:** lavaan is the oracle; failures are values; APIs stay explicit and composable


## Build

Needs CMake ≥ 3.28 and a C++23 compiler. The bundled presets assume `clang++`,
`ccache`, and `mold` are on `PATH`.

```sh
cmake --preset fast
cmake --build --preset fast
ctest --preset fast

cmake --preset opt
cmake --build --preset opt
ctest --preset opt
```

`fast` is the everyday Debug loop; `opt` is the optimized build; `dev` adds
AddressSanitizer + UBSan. `just` wraps the usual loops: `just build`,
`just test`, `just opt`, `just test-opt`, `just r-check`, and `just check`.

## Scope Boundary

**Currently out of scope:**
* Multilevel SEM
* Latent interactions/mixtures
* Inequality constraints (and active-bound inference)
* Non-interior solutions / non-standard asymptotics

**Probably always out of scope:**
* Bayesian models
* EFA
* IRT

## License

magmaan is released under the MIT License (see [`LICENSE`](LICENSE)). The
vendored PORT optimizer routines under `third_party/port/` are BSD-3-Clause;
their upstream license files live alongside the sources.
