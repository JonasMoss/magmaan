#!/usr/bin/env bash
# Usage: run_cells.sh STUDY RLIB RUN_DIR WORKERS MODE CELL_IDS...
# Production requires prior compute authorization. No remote connection here.
set -euo pipefail
STUDY=$(realpath "$1"); RLIB=$(realpath "$2"); RUN=$(realpath -m "$3"); WORKERS=$4; MODE=$5; shift 5
[[ "$WORKERS" =~ ^[12]$ ]] || { echo 'Workers must be 1 or 2' >&2; exit 1; }
[[ "$MODE" =~ ^(smoke|pilot|production)$ ]] || exit 1
export STUDY RUN MODE R_LIBS="$RLIB" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RUN/cells" "$RUN/attempts" "$RUN/logs"
run_one() {
  local id=$1 out attempt
  out=$(printf '%s/cells/cell_%03d' "$RUN" "$id")
  # Existing completed cells are validated by combine, not silently overwritten.
  [ -f "$out/COMPLETE" ] && return 0
  attempt=$(printf '%s/attempts/cell_%03d_%s' "$RUN" "$id" "$(date +%s%N)")
  nice -n 10 Rscript "$STUDY/run_experiment.R" "--$MODE" --cell "$id" --workers 1 --out-dir "$attempt" > "$RUN/logs/cell_$id.log" 2>&1 || return 1
  [ -f "$attempt/COMPLETE" ] || return 1
  if [ -d "$out" ]; then echo "Incomplete target already exists: $out" >&2; return 1; fi
  mv "$attempt" "$out"; echo "Cell $id complete"
}
export -f run_one
printf '%s\n' "$@" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$1"' _ {}
