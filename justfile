# magmaan dev tasks. `just` runs from the repo root regardless of cwd; bare
# `just` lists the recipes.

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

# Build + run one fast-suite area (smoke|spec|estimate|inference|ordinal|parity); optional 2nd arg filters test names by regex.
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

# Build the local optimized tree (Release + native CPU tuning).
opt:
    cmake --build --preset opt

# Build + run the optimized C++ test suite.
test-opt: opt
    ctest --preset opt

# Back-compatible alias for the normal local C++ build.
build: fast

# Back-compatible alias for the normal local C++ test suite.
test: test-fast

# (Re)install the exploratory R bindings against the optimized non-Ceres core.
r-install:
    cmake --build --preset opt --target magmaan
    MAGMAAN_PRESET=opt MAGMAAN_WITH_CERES_R=0 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# Fast R install for interactive wrapper work; not a numeric-performance check.
r-install-fast:
    cmake --build --preset fast --target magmaan
    MAGMAAN_PRESET=fast MAGMAAN_WITH_CERES_R=0 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

# Ceres-enabled R install, only when exploring the Ceres backend from R.
r-install-ceres:
    cmake --build --preset ceres --target magmaan
    MAGMAAN_PRESET=ceres MAGMAAN_WITH_CERES_R=1 R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

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
    Rscript tools/regen_oracle.R

# C++ tests + R smoke — everything.
check: test r-check
