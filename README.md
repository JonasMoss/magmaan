# magmaan <img src="docs/assets/logo/logo_compact.png" align="right" height="85" />

`magmaan` is the subterranean cousin of `lavaan`. It's C++23 toolkit for methods developers and simulation ethusiasts working on linear SEM, with a focus on modularity, extensibility, and full control over moving parts.

* **Status:** alpha (v0.0.1). No API-stability promise yet, the lavaan-parity core should stabilize first.

* **Language:** C++23, built with `-fno-exceptions -fno-rtti`.

* **Scope:** Estimation and inference for linear SEM + convenience functions.

* **Philosophy:** `lavaan` is the oracle, failures are values, APIs stay explicit and composable.

`magmaan` is heavily tested against `lavaan`, and the two libraries agree on a
large corpus of models. See the
[lavaan audit parity report](experiments/00-lavaan-parity/report.md) for the
current experiment summary. The
[experiments index](experiments/README.md) maps every parity audit, literature
replication, and benchmark.


## Speed

`magmaan` fits SEM models orders of magnitude faster than `lavaan` on many
fit-heavy workloads.

| Model                                        | Estimator | lavaan ms | magmaan ms | Speedup |
| -------------------------------------------- | --------- | --------: | ---------: | ------: |
| Latent state-trait (Geiser 2013)             | ML        |     103.7 |       1.28 |    103× |
| Latent state strict invariance (Geiser 2013) | ML        |      34.6 |       0.63 |     80× |
| Three-factor CFA (Geiser 2013)               | ML        |      32.6 |       0.73 |     73× |
| HS 3-factor CFA                              | ULS       |      23.9 |       0.70 |     34× |
| HS 3-factor CFA (masked)                     | FIML      |      40.7 |       1.70 |     24× |
| Bollen democracy SEM                         | ML        |      26.7 |       1.90 |     14× |

Small whole-pipeline smoke rows, including raw-data statistic construction and
selected post-fit reporting:

| Workflow                         | Estimator   | Data path                | lavaan ms | magmaan ms | Speedup |
| -------------------------------- | ----------- | ------------------------ | --------: | ---------: | ------: |
| HS 3-factor CFA robust report    | ML + UGamma | raw continuous           |      32.0 |        2.0 |     16× |
| Ordinal CFA robust report        | DWLS        | all ordinal              |      46.0 |       18.0 |    2.6× |
| Mixed ordinal/continuous CFA fit | DWLS        | 2 ordinal + 2 continuous |      48.5 |       12.5 |    3.9× |

See the
[magmaan vs lavaan speed benchmark report](experiments/05-lavaan-speed-bench/report.md)
for methodology, caveats, and the full benchmark slices.


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


## License

magmaan is released under the [MIT License](LICENSE).

Vendored third-party sources under `third_party/` keep their own licenses (see
each subdirectory's `LICENSE-*` and `README.md`). A handful of test fixtures
embed well-known public SEM teaching datasets (Holzinger-Swineford,
PoliticalDemocracy, bfi, ...) reproduced from their original distributions;
their provenance and terms are documented in
[`tests/fixtures/DATASETS.md`](tests/fixtures/DATASETS.md). Real-data corpora
curated from copyrighted textbooks are a private, optional dependency and are
**not** part of this repository (see [`corpus/README.md`](corpus/README.md)).
