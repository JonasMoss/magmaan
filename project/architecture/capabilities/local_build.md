### Local build workflow

- The normal C++ edit loop is the `fast` preset and `just test-fast` / `just
  test`: Debug, no sanitizers, Ceres off, with ccache and mold enabled.
- The `dev` preset is the sanitizer validation loop: Debug plus ASan/UBSan.
  Use `just test-dev` before handing back risky core changes, and prefer
  targeted `ctest --test-dir cpp/build/dev --output-on-failure -R ...` while narrowing failures.
- The C++ doctest suite is split into labeled executables (`smoke`, `spec`,
  `estimate`, `inference`, `ordinal`, `api`, `parity`, and `robcat`) behind
  the aggregate `magmaan_tests` target. This keeps
  `cmake --build --preset <p> --target magmaan_tests` working while allowing
  narrower relinks and `ctest -L`.
- R bindings are intentionally outside the default C++ loop. The fast dev loop
  is `just r-dev` (glue-only compile, prebuilt-core link); `just r-install` is
  the portable self-contained build (vendored core, system NLopt);
  `just r-install-ceres` / `just r-install-ipopt` are the explicit backend
  dev paths (now aliases over `just r-dev`).
- R development installs preserve the mirror's dependency files, configuration
  stamp and dev Makevars, so no-op and R-only edits reuse glue objects and skip linking.
  Glue uses the CMake core's compiler (Clang in local presets), ccache when
  available and `-O1 -g0` by default; the core remains optimized.
  `MAGMAAN_R_CXXFLAGS` allows debugging/performance overrides, and
  compiler/flag changes invalidate objects. Compiler-generated header dependencies
  rebuild affected glue to protect ABI compatibility; implementation-only core
  edits only relink R's shared library. Pure-R package edits need only
  `just r-magmaan-test` once matching bindings are installed.
  The development install links the selected preset's core archive rather than
  compiling a second vendored core with R's compiler. GCC/portable compilation
  remains a release-validation and CI path.
  During R work, `just test-opt` shares that core with `just r-dev`; building
  the `fast` tree as well is unnecessary unless a separate Debug check is wanted.

#### Build-loop timings

Fast-loop snapshot refreshed on 2026-05-19 (commit `97dd197`), on a 13th-gen
i7-1355U (12 threads), clang 19.1.7, with ccache and mold enabled.
Wall-clock; approximate orientation, not a benchmark. Rows marked
"not remeasured" are older orientation values retained until the corresponding
loop is refreshed.

R development timings refreshed on 2026-10-02 with Clang 21 and R 4.6.1.
The existing `opt` Ninja dependency database was corrupt and repeatedly
invalidated core objects; regenerating it restored no-op C++ builds.

| Loop step (`just` recipe)                   | Time   | Notes |
|---------------------------------------------|--------|-------|
| no-op `fast` build (`just fast`)            | 0.03 s | nothing changed |
| touched core TU, `fast`                     | 1.2 s  | not remeasured; mtime only, ccache hit, relink `libmagmaan` + test exes |
| edited core TU, `fast`                      | 7.5 s  | not remeasured; content change, cold clang compile of one core TU |
| touched test TU, `fast`                     | 0.5 s  | not remeasured; mtime only, ccache hit, relink one test exe |
| edited test TU, `fast`                      | 5.6 s  | not remeasured; content change, cold compile of one test TU |
| C++ suite (`just test-fast`)                | 81.5 s | 458 tests; no-op build + `ctest` |
| C++ suite minus parity (`just test-quick`)  | 37.4 s | 454 tests; excludes the 4 parity tests |
| sanitizer suite (`just test-dev`)           | 287 s  | not remeasured; old ASan/UBSan orientation value |
| R dev install, cold glue (`just r-dev`)     | 33.3 s | warm core objects; six glue TUs with Clang, `-O1 -g0`, ccache |
| R dev install, unchanged (`just r-dev`)     | 4.1 s  | no compilation or linking; objects, dependencies, library and config stamp retained |

The everyday inner loop (edit a core file, `just test-quick`) is comfortably
under a minute on the measured fast tree; the sanitizer suite and the Ceres R
path are minutes-scale and run deliberately rather than on every change.
