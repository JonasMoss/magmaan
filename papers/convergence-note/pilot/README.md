# Convergence Pilot

This directory is a scratch-but-tracked pilot area for convergence/start-value
experiments. It is intentionally separate from the manuscript files under
`papers/convergence-note/`.

The current pilot is boring on purpose: it runs the root R package over a small
grid of published hard-case simulation factories and strict textbook-corpus
cases, records CSV output, then builds compact CSV/TeX summaries.

## Run

From the repository root:

```sh
Rscript papers/convergence-note/pilot/scripts/run_convergence_sims.R
Rscript papers/convergence-note/pilot/scripts/run_textbook_subset.R
Rscript papers/convergence-note/pilot/scripts/build_tables.R
```

Or from `papers/convergence-note/pilot/`:

```sh
make
```

The scripts require an installed `magmaan` R package built from this checkout.
If `library(magmaan)` fails or an exported helper is missing, reinstall the root
package first.

Optional environment variables:

- `PILOT_REPS`: simulation replications per design, default `10`.
- `PILOT_DESIGNS`: comma-separated `convergence_sim_catalog()$design` subset.
- `PILOT_OPTIMIZERS`: comma-separated optimizer names, default
  `lbfgs,nlopt-lbfgs,nlopt-var2,port`.
- `PILOT_TEXTBOOK_N`: number of strict textbook cases to run, default `24`.
- `PILOT_MAX_ITER`: optimizer iteration cap, default `5000`.

Outputs:

- `results/convergence_sims_raw.csv`
- `results/textbook_subset_raw.csv`
- `tables/*.csv`
- `report/pilot-report.tex`
