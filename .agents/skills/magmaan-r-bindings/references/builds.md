# Binding builds and generated code

Run root `just` recipes from magmaan. Inspect `justfile` and package tools when
options or generation paths matter.

## Fast loop

`just r-dev` builds the non-sanitized `opt` core, then compiles Rcpp glue in
`r-package/build-rdev/` using `r-package/tools/r-makevars-dev` and links
`libmagmaan.a`. A changed header must reach the rebuilt glue; the mirror is
disposable and the shipped Makevars is not edited. Optional backends use
`just r-dev ceres 1 0` or `just r-dev ipopt 0 1`.

The mirror persists between installs, including objects, dependency files,
the configuration stamp and dev Makevars. Unchanged and R-only edits skip
compilation and linking.
Glue uses the CMake core's compiler through ccache when available, with `-O1 -g0`;
the default core remains `opt`. Override `MAGMAAN_R_CXXFLAGS` in the environment
for debugging or performance work. Compiler/flag changes invalidate the glue
objects through the configuration stamp. For edits only in `r-magmaan/R/`, run
`just r-magmaan-test` against the installed bindings without rebuilding them.
Daily development compiles the core once in the selected preset and links that
archive into R. The portable install uses R's configured compiler independently;
keep that second compiler/build path for release validation and CI.
Use `just test-opt` for C++ checks during R work to share the `opt` core rather
than compiling another copy in `fast`.
Compiler-generated header dependencies rebuild affected glue when canonical or
adapter headers change; implementation-only core edits only relink R's library.

If Ninja repeatedly rebuilds unchanged core objects and warns about a premature
end of file, inspect `ninja -C cpp/build/opt -d explain magmaan`. A corrupt
`.ninja_deps` can keep invalidating objects. With no build running, move that
database aside and rebuild to regenerate it; retain objects and ccache.

Use relevant examples/tests for a focused check. `just r-check` performs the
fast install, all binding examples (including live lavaan comparisons), and
ordinary-package installation/tests. `just r-magmaan-test` handles the pure-R
package after a matching compiled dependency is installed.

## Generated exports

The checked-in export files identify `Rcpp::compileAttributes()` as their
generator. From the root, regenerate with
`Rscript -e 'Rcpp::compileAttributes("r-package")'` after annotated signatures
change. `just r-dev` installs the existing exports; it does not regenerate them.
Keep `R/RcppExports.R` and `src/RcppExports.cpp` synchronized and inspect their
diffs. Registration/namespace composition and `R/zzz_core.R` may need separate
deliberate changes. Re-exporting or renaming a wrapper does not supply a missing
C++ primitive.

## Portable packaging

`just vendor` runs `r-package/tools/vendor-cpp.sh`, copying canonical
`cpp/src/`, `cpp/include/magmaan/` and PORT/QUADPACK into
`r-package/src/{core,magmaan,third_party}/`, with generated banners. Ceres/IPOPT
adapters are omitted from the portable package. `just r-install` builds this
vendored package with no CMake or prebuilt library. NLopt uses system pkg-config
or the `nloptr` package fallback. Validate this path when changing packaging,
compiler/dependency flags or vendor behavior.

`just vendor-check` depends on `vendor`: it refreshes the tree, then fails if the
generated paths have Git changes. It does not distinguish legitimate uncommitted
vendor changes from drift. Inspect/check the actual generated diffs before
committing and run it on the completed committed state when needed. Do not use
it as a read-only audit in a dirty checkout.

For an R-load undefined symbol, compare core/glue signatures and build modes;
run `just r-clean` and reinstall to eliminate stale ABI artifacts. Do not silence
the symptom with an unrelated alias. Cluster instructions are in
`r-package/tools/saga/README.md`.
