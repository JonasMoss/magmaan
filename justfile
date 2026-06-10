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

# (Re)install the exploratory R bindings against the optimized non-Ceres core.
r-install:
    cmake --build --preset opt --target magmaan
    MAGMAAN_PRESET=opt MAGMAAN_WITH_CERES_R=0 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# Fast R install for interactive wrapper work; not a numeric-performance check.
r-install-fast:
    cmake --build --preset fast --target magmaan
    MAGMAAN_PRESET=fast MAGMAAN_WITH_CERES_R=0 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# Optional optimizer R install: Ceres, with NLopt and PORT enabled by default.
r-install-ceres:
    cmake --build --preset ceres --target magmaan
    MAGMAAN_PRESET=ceres MAGMAAN_WITH_CERES_R=1 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# Optional optimizer R install: IPOPT, with NLopt and PORT enabled by default.
r-install-ipopt:
    cmake --build --preset ipopt --target magmaan
    MAGMAAN_PRESET=ipopt MAGMAAN_WITH_IPOPT_R=1 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# Run every r-package/examples/*.R script (the R-side smoke tests vs lavaan).
r-examples:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in r-package/examples/*.R; do
        echo "=== $f ==="
        Rscript "$f"
    done

# Reinstall the R bindings and run the example smoke tests.
r-check: r-install r-examples

# Force-clean the in-tree R build artifacts (the Makevars dep makes this rarely needed).
r-clean:
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

# Dependency layering + C++ tests + R smoke — everything. Layering runs first so
# a cheap structural lint fails fast before the slow build.
check: check-layering test r-check
