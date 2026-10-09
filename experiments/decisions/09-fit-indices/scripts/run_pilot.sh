#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export R_LIBS="${R_LIBS:-$HOME/.cache/magmaan-rlib/lane-c}"
exec nice -n 10 timeout 20m Rscript "$(dirname "$0")/../run_experiment.R" --pilot --workers 2 --reps 5 --run-id "${1:-pilot-task145}"
