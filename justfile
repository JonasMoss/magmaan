# magmaan dev tasks. `just` runs from the repo root regardless of cwd; bare
# `just` lists the recipes.

coverage_profiles := "cpp/build/coverage/profiles"
coverage_profdata := "cpp/build/coverage/coverage.profdata"
coverage_html := "cpp/build/coverage/html"
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

# Cap parallel compile jobs by available RAM, not just core count. magmaan's
# C++23/Eigen translation units can each hold 1-3 GB resident, so an unbounded
# build (Ninja defaults to ncpu+2) can exhaust RAM+swap and trip the kernel OOM
# killer, taking browser/terminal sessions down with it. Reserve ~10 GB for the
# rest of the desktop, budget ~2.5 GB per job, then clamp to the core count.
# Override per invocation, e.g. `just jobs=12 opt` (all cores) or `just jobs=4 build`.
jobs := ```
    cores=$(nproc)
    mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    by_mem=$(( (mem_gb - 10) * 2 / 5 ))
    [ "$by_mem" -lt 1 ] && by_mem=1
    if [ "$by_mem" -lt "$cores" ]; then echo "$by_mem"; else echo "$cores"; fi
  ```

default:
    @just --list

# Configure the first-class local build trees (once, or after a preset change).
configure:
    cmake -S cpp --preset fast
    cmake -S cpp --preset dev
    cmake -S cpp --preset opt

# Build the fast local C++ tree (Debug, no sanitizers).
fast:
    cmake --build cpp/build/fast --parallel {{jobs}}

# Build + run the fast local C++ tests excluding slow gates (test-opt runs all).
test-fast: fast
    ctest --test-dir cpp/build/fast --output-on-failure -LE slow

# Build + run one fast-suite area (smoke|spec|estimate|inference|ordinal|api|sim|parity|robcat); optional 2nd arg filters test names by regex.
test-area area regex="":
    cmake --build cpp/build/fast --target magmaan_test_{{area}} --parallel {{jobs}}
    ctest --test-dir cpp/build/fast --output-on-failure -L {{area}} {{ if regex == "" { "" } else { "-R '" + regex + "'" } }}

# Build + run the fast suite excluding parity and slow gates.
test-quick: fast
    ctest --test-dir cpp/build/fast --output-on-failure -LE "parity|slow"

# Build the sanitizer validation tree (Debug + AddressSanitizer + UBSan).
dev:
    cmake --build cpp/build/dev --parallel {{jobs}}

# Build + run the sanitizer C++ test suite.
test-dev: dev
    ctest --test-dir cpp/build/dev --output-on-failure

# Build + run the LLVM source-coverage suite and print a terminal report.
coverage:
    #!/usr/bin/env bash
    set -euo pipefail
    cmake -S cpp --preset coverage
    cmake --build cpp/build/coverage --target magmaan_tests --parallel {{jobs}}
    rm -rf {{coverage_profiles}} {{coverage_profdata}}
    mkdir -p {{coverage_profiles}}
    LLVM_PROFILE_FILE="$PWD/{{coverage_profiles}}/%p-%m.profraw" ctest --test-dir cpp/build/coverage --output-on-failure
    {{llvm_profdata}} merge -sparse {{coverage_profiles}}/*.profraw -o {{coverage_profdata}}
    objects=(
        cpp/build/coverage/tests/magmaan_test_smoke
        cpp/build/coverage/tests/magmaan_test_spec
        cpp/build/coverage/tests/magmaan_test_estimate
        cpp/build/coverage/tests/magmaan_test_inference
        cpp/build/coverage/tests/magmaan_test_ordinal
        cpp/build/coverage/tests/magmaan_test_api
        cpp/build/coverage/tests/magmaan_test_sim
        cpp/build/coverage/tests/magmaan_test_parity
        cpp/build/coverage/tests/magmaan_test_robcat
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
        cpp/build/coverage/tests/magmaan_test_smoke
        cpp/build/coverage/tests/magmaan_test_spec
        cpp/build/coverage/tests/magmaan_test_estimate
        cpp/build/coverage/tests/magmaan_test_inference
        cpp/build/coverage/tests/magmaan_test_ordinal
        cpp/build/coverage/tests/magmaan_test_api
        cpp/build/coverage/tests/magmaan_test_sim
        cpp/build/coverage/tests/magmaan_test_parity
        cpp/build/coverage/tests/magmaan_test_robcat
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
    cmake --build cpp/build/opt --parallel {{jobs}}

# Build + run the optimized C++ test suite.
test-opt: opt
    ctest --test-dir cpp/build/opt --output-on-failure

# Build the optional IPOPT optimizer tree (requires system IPOPT).
ipopt:
    cmake --build cpp/build/ipopt --parallel {{jobs}}

# Build + run the optional IPOPT optimizer test suite.
test-ipopt: ipopt
    ctest --test-dir cpp/build/ipopt --output-on-failure

# Back-compatible alias for the normal local C++ build.
build: fast

# Back-compatible alias for the normal local C++ test suite.
test: test-fast

# Build + run fast tests excluding slow gates and write JUnit XML.
test-report: fast
    ctest --test-dir cpp/build/fast --output-on-failure -LE slow --output-junit "$PWD/build/fast/test-results.xml"

# Build + run the quick fast-suite and write JUnit XML.
test-quick-report: fast
    ctest --test-dir cpp/build/fast --output-on-failure -LE "parity|slow" --output-junit "$PWD/build/fast/test-quick-results.xml"

# Local maintainer health check: quick report plus source-coverage summary.
health: test-quick-report coverage
    @echo "Health reports: cpp/build/fast/test-quick-results.xml and cpp/build/coverage/"

# === R bindings: fast dev loop vs portable ship ============================
# `just r-dev` is the FAST daily loop; `just r-install` is the PORTABLE,
# self-contained build (compiles the vendored C++ core, links a system NLopt)
# that install_github / a tarball / an HPC scp would do. After any change under
# cpp/src/ or cpp/include/, the vendored copies must be refreshed with `just vendor`.

# Vendor the C++ core + PORT + QUADPACK into the self-contained r-package/src/.
vendor:
    r-package/tools/vendor-cpp.sh

# Fail if the vendored copies drift from canonical (a cpp/src/ or cpp/include/ change
# without re-vendoring, or a hand-edited vendored copy). For CI / pre-commit.
vendor-check: vendor
    #!/usr/bin/env bash
    set -euo pipefail
    # Only the vendored trees, not the hand-written Makevars/glue at cpp/src/ top level.
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
# a cluster). See r-package/tools/saga/README.md.
r-install: vendor
    MAKEFLAGS="-j{{jobs}}" R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# FAST dev install (the daily loop). Compiles only the Rcpp glue and links the
# prebuilt opt libmagmaan.a, via a persistent r-package/build-rdev/ mirror with the dev-only
# r-package/tools/r-makevars-dev swapped in — preserving the portable
# src/Makevars.in template. Optional backends:
# `just r-dev ceres 1 0` or `just r-dev ipopt 0 1`.
r-dev preset="opt" ceres="0" ipopt="0":
    #!/usr/bin/env bash
    set -euo pipefail
    cmake -S cpp --preset {{preset}}
    cmake --build cpp/build/{{preset}} --target magmaan --parallel {{jobs}}
    root="$(pwd)"
    compiler="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' cpp/build/{{preset}}/CMakeCache.txt)"
    test -n "$compiler"
    rsync -a --delete \
        --exclude='*.o' --exclude='*.so' --exclude='*.d' \
        --exclude='/src/.magmaan-build-config' --exclude='/src/Makevars' \
        --exclude='/src/core' --exclude='/src/magmaan' --exclude='/src/third_party' \
        --exclude='/build-rdev' --exclude='/tools' --exclude='/examples' --exclude='/.RData' --exclude='/.Rhistory' --exclude='/.Rproj.user' \
        r-package/ r-package/build-rdev/
    if ! cmp -s r-package/tools/r-makevars-dev r-package/build-rdev/src/Makevars; then
        cp r-package/tools/r-makevars-dev r-package/build-rdev/src/Makevars
    fi
    MAGMAAN_ROOT="$root" MAGMAAN_PRESET={{preset}} MAGMAAN_R_CXX="$compiler" \
        MAGMAAN_WITH_CERES_R={{ceres}} MAGMAAN_WITH_IPOPT_R={{ipopt}} \
        MAKEFLAGS="-j{{jobs}}" \
        R CMD INSTALL --no-byte-compile --no-docs --no-help r-package/build-rdev

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

# Install the pure-R ordinary-user package `magmaan` from r-magmaan/. It imports
# magmaanlab, so install that first (`just r-dev` or `just r-install`).
r-magmaan:
    R CMD INSTALL --no-byte-compile --no-docs --no-help r-magmaan

# Install and test the ordinary-user package against the installed magmaanlab.
r-magmaan-test: r-magmaan
    Rscript -e 'testthat::test_dir("r-magmaan/tests/testthat", package = "magmaan", load_package = "installed", stop_on_failure = TRUE)'

# Fast dev install + the example smoke tests + the ordinary-user package tests.
r-check: r-dev r-examples r-magmaan-test

# Portable source-tarball checks; outputs and the dependency install stay outside the repo.
r-cmd-check output_dir="$HOME/.cache/magmaan-logs/r-cmd-check" library="$HOME/.cache/magmaan-rlib/check":
    #!/usr/bin/env bash
    set -euo pipefail
    root="$PWD"
    mkdir -p "{{output_dir}}" "{{library}}"
    output_dir="$(cd "{{output_dir}}" && pwd)"
    library="$(cd "{{library}}" && pwd)"
    export R_LIBS="$library${R_LIBS:+:$R_LIBS}"
    export MAKEFLAGS="-j{{jobs}}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    cd "$output_dir"
    for package_dir in r-package r-magmaan; do
        package="$(sed -n 's/^Package: //p' "$root/$package_dir/DESCRIPTION")"
        version="$(sed -n 's/^Version: //p' "$root/$package_dir/DESCRIPTION")"
        nice -n 10 R CMD build "$root/$package_dir"
        nice -n 10 R CMD check --no-manual "$package"_"$version".tar.gz
        if grep -Eq '[0-9]+ (ERROR|WARNING)' "$package.Rcheck/00check.log"; then
            exit 1
        fi
        if [ "$package" = magmaanlab ]; then
            nice -n 10 R CMD INSTALL --library="$library" "$package.Rcheck/00_pkg_src/$package"
        fi
    done

# Force-clean the in-tree R build artifacts + the dev mirror.
r-clean:
    rm -rf r-package/build-rdev
    rm -f r-package/src/*.o r-package/src/*.so r-package/src/.magmaan-build-config

# Regenerate the lavaan oracle fixtures (needs R + the pinned lavaan version).
regen-oracle:
    Rscript cpp/tests/tools/regen_oracle.R

# Regenerate the robcat parity fixtures (needs R + the pinned robcat version).
regen-robcat:
    Rscript cpp/tests/tools/regen_robcat_fixtures.R

# Regenerate the semfindr case-influence fixtures (needs R + the pinned semfindr).
regen-semfindr:
    Rscript cpp/tests/tools/regen_semfindr_fixtures.R

# Enforce the dependency-layering rule (leaves are sinks). Fast: no build, no R.
check-layering:
    bash cpp/tests/tools/check_layering.sh

# Enforce what may be tracked: only magmaan content, no third-party binaries,
# no session state, nothing over 1 MB. Checks the index, so run it before
# committing new files.
check-tracked:
    bash cpp/tests/tools/check_tracked_files.sh

# Dependency layering + tracked content + vendor-drift + C++ tests + R smoke —
# everything. The cheap structural lints (layering, tracked content, vendor
# sync) run first so they fail fast before the slow build.
check: check-layering check-tracked vendor-check test r-check
