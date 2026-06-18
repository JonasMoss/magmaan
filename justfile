# magmaan dev tasks. `just` runs from the repo root regardless of cwd; bare
# `just` lists the recipes.

coverage_profiles := "build/coverage/profiles"
coverage_profdata := "build/coverage/coverage.profdata"
coverage_html := "build/coverage/html"
coverage_ignore := "(/_deps/|/third_party/|/tests/|/usr/)"

# Locate the LLVM coverage tools. Debian/Ubuntu ship them versioned
# (`llvm-cov-21`) under the compiler's own bin dir and do NOT put bare
# `llvm-cov`/`llvm-profdata` on PATH, so derive the bin dir from the active
# clang++ — this also guarantees the tool version matches the instrumentation.
# Falls back to bare names on PATH (Homebrew, source builds, manual symlinks).
llvm_bindir := ```
    dir=$(clang++ -print-resource-dir 2>/dev/null)
    dir=${dir%/lib/clang/*}/bin
    if [ -x "$dir/llvm-cov" ] && [ -x "$dir/llvm-profdata" ]; then echo "$dir"; fi
  ```
llvm_cov := if llvm_bindir != "" { llvm_bindir / "llvm-cov" } else { "llvm-cov" }
llvm_profdata := if llvm_bindir != "" { llvm_bindir / "llvm-profdata" } else { "llvm-profdata" }

default:
    @just --list

# Configure the first-class local build trees (once, or after a preset change).
configure:
    cmake --preset fast
    cmake --preset dev
    cmake --preset opt

# Build the fast local C++ tree (Debug, no sanitizers).
fast:
    cmake --build --preset fast

# Build + run the fast local C++ test suite.
test-fast: fast
    ctest --preset fast

# Build + run one fast-suite area (smoke|spec|estimate|inference|ordinal|api|sim|parity|robcat); optional 2nd arg filters test names by regex.
test-area area regex="":
    cmake --build --preset fast --target magmaan_test_{{area}}
    ctest --preset fast -L {{area}} {{ if regex == "" { "" } else { "-R '" + regex + "'" } }}

# Build + run the fast suite minus the heavy real-data parity tests.
test-quick: fast
    ctest --preset fast -LE parity

# Build the sanitizer validation tree (Debug + AddressSanitizer + UBSan).
dev:
    cmake --build --preset dev

# Build + run the sanitizer C++ test suite.
test-dev: dev
    ctest --preset dev

# Build + run the LLVM source-coverage suite and print a terminal report.
coverage:
    #!/usr/bin/env bash
    set -euo pipefail
    cmake --preset coverage
    cmake --build --preset coverage --target magmaan_tests
    rm -rf {{coverage_profiles}} {{coverage_profdata}}
    mkdir -p {{coverage_profiles}}
    LLVM_PROFILE_FILE="$PWD/{{coverage_profiles}}/%p-%m.profraw" ctest --preset coverage
    {{llvm_profdata}} merge -sparse {{coverage_profiles}}/*.profraw -o {{coverage_profdata}}
    objects=(
        build/coverage/tests/magmaan_test_smoke
        build/coverage/tests/magmaan_test_spec
        build/coverage/tests/magmaan_test_estimate
        build/coverage/tests/magmaan_test_inference
        build/coverage/tests/magmaan_test_ordinal
        build/coverage/tests/magmaan_test_api
        build/coverage/tests/magmaan_test_sim
        build/coverage/tests/magmaan_test_parity
        build/coverage/tests/magmaan_test_robcat
    )
    object_args=()
    for obj in "${objects[@]:1}"; do
        object_args+=(--object "$obj")
    done
    {{llvm_cov}} report "${objects[0]}" "${object_args[@]}" \
        --instr-profile={{coverage_profdata}} \
        --ignore-filename-regex='{{coverage_ignore}}'

# Build + run coverage, then write a browsable HTML report.
coverage-html: coverage
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf {{coverage_html}}
    objects=(
        build/coverage/tests/magmaan_test_smoke
        build/coverage/tests/magmaan_test_spec
        build/coverage/tests/magmaan_test_estimate
        build/coverage/tests/magmaan_test_inference
        build/coverage/tests/magmaan_test_ordinal
        build/coverage/tests/magmaan_test_api
        build/coverage/tests/magmaan_test_sim
        build/coverage/tests/magmaan_test_parity
        build/coverage/tests/magmaan_test_robcat
    )
    object_args=()
    for obj in "${objects[@]:1}"; do
        object_args+=(--object "$obj")
    done
    {{llvm_cov}} show "${objects[0]}" "${object_args[@]}" \
        --instr-profile={{coverage_profdata}} \
        --format=html \
        --output-dir={{coverage_html}} \
        --ignore-filename-regex='{{coverage_ignore}}'
    echo "Coverage HTML: {{coverage_html}}/index.html"

# Build the local optimized tree (Release + native CPU tuning).
opt:
    cmake --build --preset opt

# Build + run the optimized C++ test suite.
test-opt: opt
    ctest --preset opt

# Build the optional IPOPT optimizer tree (requires system IPOPT).
ipopt:
    cmake --build --preset ipopt

# Build + run the optional IPOPT optimizer test suite.
test-ipopt: ipopt
    ctest --preset ipopt

# Back-compatible alias for the normal local C++ build.
build: fast

# Back-compatible alias for the normal local C++ test suite.
test: test-fast

# Build + run the fast C++ test suite and write JUnit XML.
test-report: fast
    ctest --preset fast --output-junit "$PWD/build/fast/test-results.xml"

# Build + run the quick fast-suite and write JUnit XML.
test-quick-report: fast
    ctest --preset fast -LE parity --output-junit "$PWD/build/fast/test-quick-results.xml"

# Local maintainer health check: quick report plus source-coverage summary.
health: test-quick-report coverage
    @echo "Health reports: build/fast/test-quick-results.xml and build/coverage/"

# === R bindings: fast dev loop vs portable ship ============================
# `just r-dev` is the FAST daily loop; `just r-install` is the PORTABLE,
# self-contained build (compiles the vendored C++ core, links a system NLopt)
# that install_github / a tarball / an HPC scp would do. After any change under
# src/ or include/, the vendored copies must be refreshed with `just vendor`.

# Vendor the C++ core + PORT + QUADPACK into the self-contained r-package/src/.
vendor:
    dev/vendor-cpp.sh

# Fail if the vendored copies drift from canonical (a src/ or include/ change
# without re-vendoring, or a hand-edited vendored copy). For CI / pre-commit.
vendor-check: vendor
    #!/usr/bin/env bash
    set -euo pipefail
    # Only the vendored trees, not the hand-written Makevars/glue at src/ top level.
    paths="r-package/src/core r-package/src/magmaan r-package/src/third_party"
    if [ -n "$(git status --porcelain -- $paths)" ]; then
        echo "vendored r-package/src/{core,magmaan,third_party} out of sync — run 'just vendor' and commit:"
        git status --porcelain -- $paths
        exit 1
    fi

# PORTABLE install: the self-contained vendored package (system NLopt), exactly
# as install_github / a release builds it. Slower than r-dev; use it to validate
# the shippable package or to install where there is no CMake build. NLopt comes
# from pkg-config; override with NLOPT_CFLAGS / NLOPT_LIBS (or R_MAKEVARS_USER on
# a cluster). See dev/saga/README.md.
r-install: vendor
    MAKEFLAGS="-j$(nproc)" R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# FAST dev install (the daily loop). Compiles only the Rcpp glue and links the
# prebuilt opt libmagmaan.a, via a throwaway build-rdev/ mirror with the dev-only
# dev/r-makevars-dev swapped in — so the committed self-contained
# r-package/src/Makevars is never touched. Optional backends:
# `just r-dev ceres 1 0` or `just r-dev ipopt 0 1`.
r-dev preset="opt" ceres="0" ipopt="0":
    #!/usr/bin/env bash
    set -euo pipefail
    cmake --preset {{preset}}
    cmake --build --preset {{preset}} --target magmaan --parallel "$(nproc)"
    root="$(pwd)"
    rsync -a --delete \
        --exclude='*.o' --exclude='*.so' \
        --exclude='/src/core' --exclude='/src/magmaan' --exclude='/src/third_party' \
        --exclude='/examples' --exclude='/.RData' --exclude='/.Rhistory' --exclude='/.Rproj.user' \
        r-package/ build-rdev/
    cp dev/r-makevars-dev build-rdev/src/Makevars
    MAGMAAN_ROOT="$root" MAGMAAN_PRESET={{preset}} \
        MAGMAAN_WITH_CERES_R={{ceres}} MAGMAAN_WITH_IPOPT_R={{ipopt}} \
        MAKEFLAGS="-j$(nproc)" \
        R CMD INSTALL --no-byte-compile --no-docs --no-help build-rdev

# Back-compat aliases for the old fast/ceres/ipopt installs (now via r-dev).
r-install-fast: (r-dev "fast")
r-install-ceres: (r-dev "ceres" "1" "0")
r-install-ipopt: (r-dev "ipopt" "0" "1")

# Run every r-package/examples/*.R script (the R-side smoke tests vs lavaan).
r-examples:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in r-package/examples/*.R; do
        echo "=== $f ==="
        Rscript "$f"
    done

# Fast dev install + the example smoke tests.
r-check: r-dev r-examples

# Force-clean the in-tree R build artifacts + the dev mirror.
r-clean:
    rm -rf build-rdev
    rm -f r-package/src/*.o r-package/src/*.so r-package/src/.magmaan-build-config

# Regenerate the lavaan oracle fixtures (needs R + the pinned lavaan version).
regen-oracle:
    Rscript tests/tools/regen_oracle.R

# Regenerate the robcat parity fixtures (needs R + the pinned robcat version).
regen-robcat:
    Rscript tests/tools/regen_robcat_fixtures.R

# Enforce the dependency-layering rule (leaves are sinks). Fast: no build, no R.
check-layering:
    bash tests/tools/check_layering.sh

# Dependency layering + vendor-drift + C++ tests + R smoke — everything. The
# cheap structural lints (layering, vendor sync) run first so they fail fast
# before the slow build.
check: check-layering vendor-check test r-check

# FIML-FMG paper endpoint sync. Dry-run unless the nested helper receives
# `--apply`, e.g. `just fmg-sync export-overleaf --apply`.
fmg-sync *ARGS:
    cd papers/fiml-fmg && scripts/sync_external.sh {{ARGS}}

notes_dir := "docs/research/notes"

# Build one research note (named, or the most recently edited) and open the PDF.
note-build name="":
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{notes_dir}}"
    if [ -n "{{name}}" ]; then
        tex="$dir/$(basename "{{name}}" .tex).tex"
    else
        tex="$(ls -t "$dir"/*.tex 2>/dev/null | head -1 || true)"
    fi
    [ -n "${tex:-}" ] && [ -f "$tex" ] || { echo "no .tex note found in $dir"; exit 1; }
    base="$(basename "$tex" .tex)"
    cd "$dir"
    pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null
    rm -f "$base".{aux,log,out,toc}
    if command -v xdg-open >/dev/null 2>&1; then opener=xdg-open; else opener=open; fi
    if command -v setsid >/dev/null 2>&1; then
        setsid "$opener" "$base.pdf" >/dev/null 2>&1 < /dev/null &
    else
        "$opener" "$base.pdf" >/dev/null 2>&1 &
    fi
    echo "built and opened $dir/$base.pdf"

# Rebuild every research note (no viewer); report any that fail to compile.
note-build-all:
    #!/usr/bin/env bash
    set -uo pipefail
    dir="{{notes_dir}}"
    fail=0
    shopt -s nullglob
    for tex in "$dir"/*.tex; do
        base="$(basename "$tex" .tex)"
        if ( cd "$dir" && pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null 2>&1 \
                       && pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null 2>&1 ); then
            ( cd "$dir" && rm -f "$base".{aux,log,out,toc} )
            echo "ok   $base"
        else
            echo "FAIL $base"; fail=1
        fi
    done
    exit $fail
