# latva dev tasks. `just` runs from the repo root regardless of cwd; bare
# `just` lists the recipes.

default:
    @just --list

# Configure the CMake build trees (once, or after a preset change).
configure:
    cmake --preset default
    cmake --preset asan

# Build the asan tree (canonical CI build: AddressSanitizer + UBSan).
build:
    cmake --build --preset asan

# Build + run the full C++ test suite.
test: build
    ctest --preset asan

# (Re)install the exploratory R bindings (rebuilds liblatva.a; Makevars handles stale .o).
r-install:
    cmake --build --preset default --target latva
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
