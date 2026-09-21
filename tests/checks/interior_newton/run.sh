#!/usr/bin/env bash
set -euo pipefail
: "${1:?supply a fresh output directory}"
test ! -e "$1"
mkdir -p "$1"
run_dir=$(realpath "$1")
cmake --build --preset opt --target magmaan > "$run_dir/build.log" 2>&1
extra_objects=()
if [[ "${2:-10}" == 60 ]]; then
  sed 's/mred = 10;/mred = 60;/' build/opt/_deps/nlopt-src/src/algs/luksan/plis.c > "$run_dir/plis-mred60.c"
  cc -O3 -Ibuild/opt/_deps/nlopt-src/src/algs/luksan \
    -Ibuild/opt/_deps/nlopt-src/src/util -Ibuild/opt/_deps/nlopt-src/src/api \
    -Ibuild/opt/_deps/nlopt-build -c "$run_dir/plis-mred60.c" -o "$run_dir/plis-mred60.o"
  extra_objects+=("$run_dir/plis-mred60.o")
elif [[ "${2:-10}" != 10 ]]; then
  exit 2
fi
printf '%s\n' "${2:-10}" > "$run_dir/backtracking-limit.txt"
check_source=tests/checks/interior_newton/check.cpp
if [[ "${3:-}" == advanced ]]; then check_source=tests/checks/interior_newton/advanced.cpp; fi
clang++ -std=c++23 -O3 -DNDEBUG -march=native -fno-exceptions -fno-rtti \
 -DEIGEN_NO_EXCEPTIONS=1 -DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_DONT_PARALLELIZE=1 \
 -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_RUNTIME_NO_MALLOC=1 \
 -Ibuild/opt/_deps/nlohmann_json-src/single_include -Iinclude -isystem build/fast/_deps/eigen3-src -Ibuild/opt/_deps/nlopt-src/src/api \
 "$check_source" "${extra_objects[@]}" build/opt/libmagmaan.a \
 build/opt/_deps/nlopt-build/libnlopt.a build/opt/libquadpack.a build/opt/libport.a \
 -lm -o "$run_dir/check"
git rev-parse HEAD > "$run_dir/source-parent.txt"
sha256sum "$check_source" tests/checks/interior_newton/check.cpp tests/checks/interior_newton/run.sh \
 build/opt/libmagmaan.a build/opt/_deps/nlopt-build/libnlopt.a > "$run_dir/hashes.txt"
mode_args=()
if [[ "${3:-}" == refine || "${3:-}" == options || "${3:-}" == validate ]]; then mode_args+=("$3"); elif [[ -n "${3:-}" && "${3:-}" != advanced ]]; then exit 2; fi
"$run_dir/check" "$run_dir/raw.csv" "${mode_args[@]}" > "$run_dir/stdout.log" 2> "$run_dir/progress.log"
