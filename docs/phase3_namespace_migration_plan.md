# Phase 3: Convert Public Usage to New Namespaces

## Summary

Phase 3 is the public namespace migration. Phase 1 added alias headers, and
Phase 2 moved `src/` files. Phase 3 should make repo code use the new public
API paths and namespaces directly, while keeping old `magmaan::fit` and
`magmaan::partable` headers/names as compatibility aliases for one transition
window.

Success means tests and R glue compile against `spec`, `lavaan`, `estimate`,
`optim`, `nt`, `gls`, and `data`, with old includes still tested as deprecated
compatibility.

## Key Changes

- Make new namespace headers the primary public definitions:
  - `spec`: `LatentStructure`, `LatentNames`, `Starts`, `LavaanifyOptions`,
    `lavaanify`, `compute_eq_groups`, `LinearForm`, linear-constraint helpers.
  - `lavaan`: `LavaanParTable`, `ParsedLavaanParTable`, lavaan partable
    projection functions.
  - `data`: `SampleStats`, `RawData`, sample-stat and gamma helpers.
  - `estimate`: `Estimates`, `fit`, `fit_bounded`, starts, bounds, fixed.x,
    equality constraints.
  - `optim`: optimizer concepts and optimizer classes.
  - `nt::*`: ML, inference, robust tests, measures, standardization, effects.
  - `gls`: `ULS`.

- Keep old headers as compatibility shims:
  - `include/magmaan/fit/*.hpp` should include the new header and alias old
    names into `magmaan::fit`.
  - `include/magmaan/partable/*.hpp` should include the new header and alias
    old names into `magmaan::partable`.
  - Do not remove old includes in this phase.

- Update implementation namespaces to match file layout:
  - Change definitions in moved `src/*` files to the new namespaces.
  - Use qualified references where cross-domain dependencies remain, e.g.
    `estimate` consuming `data::SampleStats`, `spec::LatentStructure`,
    `optim::LbfgsOptimizer`, `nt::ml::ML`.
  - Keep shared private helpers under an internal detail namespace that does
    not imply the old `fit` domain, e.g. `magmaan::detail` or
    `magmaan::nt::detail` depending on usage.

- Convert tests and R glue to the new API:
  - Golden and unit tests should include new headers and call new namespaces.
  - R C++ bindings should use new C++ namespaces internally, while keeping
    current R export names unchanged.
  - Leave one focused compatibility test that includes representative old
    headers and asserts old aliases still map to the new types/functions.

- Update docs after code compiles:
  - Mark Phase 2 source movement complete in `docs/roadmap.md`.
  - Mark Phase 3 namespace conversion complete once tests/R glue use the new
    names.
  - Update stale source references such as `src/fit/...` in roadmap prose to
    new paths where they are not deliberately historical.

## Test Plan

- Build and run C++ dev suite:
  - `cmake --build --preset dev`
  - `ASAN_OPTIONS=detect_leaks=0 ctest --preset dev`
- Add/keep focused compile coverage for old compatibility aliases.
- Run R-facing smoke checks if available in the local workflow:
  - `just r-install`
  - existing R examples that exercise fit, robust, nested/Satorra, and
    multigroup paths.
- Confirm no repo code outside compatibility headers/tests still includes
  `magmaan/fit/*` or `magmaan/partable/*`.

## Assumptions

- This phase is still an API organization refactor only: no numerical behavior
  changes.
- Old C++ namespace aliases remain for one transition window; removal is
  Phase 4.
- R export names stay unchanged for now.
- Compatibility aliases may be implemented with `using` declarations; no
  duplicate implementations.

