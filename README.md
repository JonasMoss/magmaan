# magmaan <img src="docs/assets/logo/logo_compact.png" align="right" height="170" />

`magmaan` is the subterranean cousin of `lavaan`. It's C++23 toolkit for methods developers and simulation ethusiasts working on linear SEM, with a focus on modularity, extensibility, and full control over moving parts.

* **Status:** alpha (v0.0.1). No API-stability promise yet, the lavaan-parity core should stabilize first.

* **Language:** C++23, built with `-fno-exceptions -fno-rtti`.

* **Scope:** Estimation and inference for linear SEM + convenience functions.

* **Philosophy:** `lavaan` is the oracle, failures are values, APIs stay explicit and composable.

`magmaan` is heavily tasted against `lavaan`, and the two libraries agree n nb    on a massive corpus of models. See ((todo: add lavan parity document in experiments)) for details. 


## Speed

`magmaan` fits SEM models orders of magnitue faster than `lavaan`.

| Model                                        | Estimator | lavaan ms | magmaan ms | Speedup |
| -------------------------------------------- | --------- | --------: | ---------: | ------: |
| Latent state-trait (Geiser 2013)             | ML        |     103.7 |       1.28 |    103× |
| Latent state strict invariance (Geiser 2013) | ML        |      34.6 |       0.63 |     80× |
| Three-factor CFA (Geiser 2013)               | ML        |      32.6 |       0.73 |     73× |
| HS 3-factor CFA                              | ULS       |      31.2 |       0.78 |     40× |
| HS 3-factor CFA (masked)                     | FIML      |      59.4 |       1.53 |     39× |
| Bollen democracy SEM                         | ML        |      33.8 |       1.49 |     23× |

(TODO: Link to more detail benchmark.)


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

