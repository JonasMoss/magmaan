# magmaanlab bindings

This is the compiled methods-development package. Export thin R wrappers around
C++ entry points, keeping the C++ argument structure and convention choices
visible. `fit_model()` is estimate-only; inference is composed in explicit
post-fit calls. R helpers may compose wrappers, validate R-shaped inputs and
preserve names/groups; do not implement parallel SEM logic here.

The pure-R ordinary-user package lives in `../r-magmaan/` and imports this one.
No exported name may have different meanings in the two packages. Read the
[interface vision](../project/design/r-interface-vision.md) before changing
their boundary. Ordinary-user inference policy is composed in C++.

Partables are projections of the model triple. Under identical options they
must match the corresponding lavaan rows and estimates within documented
tolerances. Fix mismatches in the model triple, projection or fit reconstruction,
not R formatting. Fixed-zero intercept/mean rows must appear or disappear as
lavaan would. Keep unsupported rows/operations explicit.

## Source and build ownership

Use the repository's `magmaan-r-bindings` skill for changing glue, exports or
vendored code. Hand-written glue lives at `src/` top level. Generated
`src/{core,magmaan,third_party}/` mirrors canonical C++ sources plus PORT and
QUADPACK; never edit it directly. `just vendor` refreshes it.

`just r-dev` is the fast loop: build `opt`, compile glue in disposable
`build-rdev/`, link `libmagmaan.a`. The mirror persists across installs: preserve
its objects, dependency files, configuration stamp and dev Makevars so unchanged
installs do not compile or link. Glue uses the core's compiler, ccache when available and
`-O1 -g0`; override with `MAGMAAN_R_CXXFLAGS` for debugging/performance work.
Compiler-generated dependencies rebuild affected glue after header changes;
implementation-only core edits only relink the shared library.
For C++ checks during R work, use `just test-opt` to reuse the same core rather
than also building `fast`. `just r-install` is the portable build:
compile the vendored core without CMake or a prebuilt library. NLopt comes from
system pkg-config or the `nloptr` package fallback. The shipped package must
work independently of this checkout. For cluster installation see
[tools/saga/README.md](tools/saga/README.md).

`just r-check` runs the fast install, R examples against lavaan, and ordinary
package tests. CI runs portable source-tarball `R CMD check --no-manual` for both
packages; live lavaan comparisons are permitted there. Select relevant
checks for a focused change and validate the portable build when changing its
packaging/toolchain path. On an R-load undefined symbol, run `just r-clean` and
reinstall. `just vendor-check` first refreshes generated files, then checks Git
status; it is a mutating check, not a read-only comparison.

Portable `configure` resolves NLopt and enumerates vendored objects into ignored
`src/Makevars` from `src/Makevars.in`. The dev install supplies its own Makevars
and bypasses configure when `MAGMAAN_ROOT` is set. `USE_C17` selects R's legacy
C compiler mode for f2c callbacks. The portable R build's PORT I/O guard raises
an R error on internal STOP instead of terminating the process.
