# magmaan dev tasks. `just` runs from the repo root regardless of cwd; bare
# `just` lists the recipes.

default:
    @just --list

# Configure the first-class local build trees (once, or after a preset change).
configure:
    cmake --preset dev
    cmake --preset opt

# Build the local dev tree (Debug + AddressSanitizer + UBSan).
dev:
    cmake --build --preset dev

# Build + run the dev C++ test suite.
test-dev: dev
    ctest --preset dev

# Build the local optimized tree (Release + native CPU tuning).
opt:
    cmake --build --preset opt

# Build + run the optimized C++ test suite.
test-opt: opt
    ctest --preset opt

# Back-compatible alias for the local dev build.
build: dev

# Back-compatible alias for the local dev test suite.
test: test-dev

# (Re)install the exploratory R bindings (rebuilds libmagmaan.a; Makevars handles stale .o).
r-install:
    cmake --build --preset opt --target magmaan
    R CMD INSTALL --no-byte-compile --no-docs --no-help r-package

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
    rm -f r-package/src/*.o r-package/src/*.so

# Regenerate the lavaan oracle fixtures (needs R + the pinned lavaan version).
regen-oracle:
    Rscript tools/regen_oracle.R

# C++ tests + R smoke — everything.
check: test r-check
