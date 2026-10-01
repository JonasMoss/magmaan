# magmaan <img src="project/assets/logo/logo_compact.png" align="right" height="85" />

`magmaan` is the subterranean cousin of `lavaan`: a C++23 core for linear SEM,
checked component by component against `lavaan`, exposed to two different
audiences over the same statistics.

* **Ordinary users** (R package [`magmaan`](r-magmaan/)) write one call. It
  estimates the model and computes inference under a single policy that
  magmaan chooses and justifies — few options, no compatibility switches
  (no MLR, no WLSMV, no information/SE/test conventions to pick between).
* **Methods developers** (R package [`magmaanlab`](r-package/), or the C++
  core directly) get the full composable surface: every estimator path, every
  information and Gamma convention, lavaan-compatibility routes, and the
  frontier methods the research depends on.

Neither package is a second SEM implementation; both sit over the same
`cpp/` core. See
[project/design/r-interface-vision.md](project/design/r-interface-vision.md)
for the design.

* **Status:** alpha (v0.0.1). No API-stability promise yet; the lavaan-parity core should stabilize first.

* **Language:** C++23 core, built with `-fno-exceptions -fno-rtti`. Failures are values (`std::expected`), not exceptions.

* **Scope:** Linear SEM and inference for population approximation parameters;
  see the [statistical scope](project/scope.md) for sampling assumptions and
  banked extensions, from the lavaan-parity core to frontier methods.

* **Philosophy:** `lavaan` is the oracle for the parity core, failures are values, APIs stay explicit and composable, no virtual functions on the hot path.

`magmaan` is heavily tested against `lavaan`, and the two libraries agree on a
large corpus of models. See the
[lavaan audit parity report](experiments/showcases/01-lavaan-parity/report.md) for the
current experiment summary. The
[experiments index](experiments/README.md) maps every parity audit, literature
replication, and benchmark.


## Speed

Estimate-only fits are commonly one to two orders of magnitude faster than
`lavaan` on the same model: magmaan skips R's dispatch/model-building
overhead and runs a lean C++ core. Whole-pipeline workloads (raw-data
statistic construction, robust reporting, ordinal polychoric fits) see
smaller but still real speedups.

Numbers live in one place, not two: see the
[magmaan vs lavaan speed benchmark report](experiments/showcases/02-lavaan-speed-bench/report.md)
for current measurements, methodology, and caveats. It's regenerated from
`experiments/showcases/02-lavaan-speed-bench/run_experiment.R`, not copied
into this README, so it can't go stale here — check its own timestamp (and
rerun it) after a change to estimator/optimizer defaults.


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

`fast` is the everyday Debug loop; `opt` is the local Release/native-CPU
build; `dev` adds AddressSanitizer + UBSan; `ceres`/`ipopt` add optional
optimizer backends. `just` wraps the usual loops: `just build` / `just test`
(the `fast` loop), `just opt`, `just test-opt`, `just r-dev` (fast R-bindings
dev loop, linking the prebuilt C++ core), `just r-install` (portable,
self-contained R build, as a release install would do), `just r-check`
(R build + example smoke tests + `magmaan` package tests), and `just check`
(everything: layering, tracked-file, vendor-drift, C++ and R checks). Bare
`just` lists every recipe.


## License

magmaan is released under the [MIT License](LICENSE).

Vendored third-party sources under `cpp/third_party/` (PORT, BSD-3-Clause;
QUADPACK, public domain) keep their own terms, recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and each subdirectory's
`LICENSE-*`/`README.md`.

A handful of test fixtures
embed well-known public SEM teaching datasets (Holzinger-Swineford,
PoliticalDemocracy, bfi, ...) reproduced from their original distributions;
their provenance and terms are documented in
[`cpp/tests/fixtures/DATASETS.md`](cpp/tests/fixtures/DATASETS.md). Real-data corpora
curated from copyrighted textbooks are a private, optional dependency and are
**not** part of this repository (see [`project/reference/textbook-corpus.md`](project/reference/textbook-corpus.md)).

## Navigate

- [cpp/](cpp/): C++ source, headers, tests, fixtures, dependencies, and CMake.
- [r-package/](r-package/): `magmaanlab`, the compiled methods-developer R
  package (Rcpp bindings over the C++ core); build helpers are in `tools/`.
- [r-magmaan/](r-magmaan/): `magmaan`, the pure-R ordinary-user package
  (imports `magmaanlab`, no compiled code).
- [project/](project/): architecture, backlog, grammar, design, and validation.
- [experiments/](experiments/): showcases, replications, research, and engineering investigations.
- [benchmarks/](benchmarks/): shared benchmark harness.
- [external/](external/): optional source collections and reference material.
- `papers/` and `private/`: independent local repositories, excluded from magmaan.

Run `just configure` once to set up the `fast`/`dev`/`opt` build trees, then
`just build`, `just test`, `just check`, etc. from here, the repo root. CMake
presets live in `cpp/`; direct `cmake`/`ctest` preset commands should run from
that directory.
