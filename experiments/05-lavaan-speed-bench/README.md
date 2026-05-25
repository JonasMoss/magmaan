# Lavaan Speed Bench

A small, README-facing speed comparison: magmaan vs lavaan on a handful of
classic lavaan-tutorial models, at the estimate-only single-thread workload.

This experiment is the *publishing layer* — it does no harness work of its own.
Timings are produced by `benchmarks/r/run_benchmark.R` (the same harness that
validates magmaan against the lavaan oracle to 1e-3); headline numbers from
five published Geiser-2013 latent-variable models are lifted verbatim from
`papers/snlls-constrained/tables/snlls_geiser_corpus_lavaan.csv`.

The rendered `report.html` is committed so the top-level `README.md` can link
to it directly.

## Cases

The zoo run covers six active cases from `benchmarks/cases.yml` that use
in-package lavaan datasets (no external downloads):

- `hs_3factor_cfa` — Holzinger-Swineford 3-factor CFA (ML).
- `hs_3factor_cfa_fiml_masked` — same model with controlled missingness, FIML.
- `hs_3factor_cfa_uls` — same model, ULS.
- `hs_3factor_cfa_gls` — same model, GLS.
- `bollen_democracy_sem` — Bollen's industrialization/democracy SEM (ML).
- `demo_growth_linear` — `Demo.growth` linear latent growth curve (ML, mean
  structure; magmaan uses `model_type = "growth"` to match lavaan's
  `growth()` defaults).

The headline (Geiser 2013) numbers cover five additional real published models
from the SNLLS paper, kept for context.

## Run

Install the magmaan R package first:

```sh
just r-install
```

Run the experiment (single-threaded BLAS, 30 iterations per case):

```sh
Rscript experiments/05-lavaan-speed-bench/run_experiment.R
```

Render the report:

```sh
(cd experiments/05-lavaan-speed-bench && quarto render report.qmd)
```

## Outputs

- `results/zoo.csv` — one row per case from this experiment: model, estimator,
  `n_obs`, `n_vars`, lavaan median ms, magmaan median ms, speedup, max-abs
  estimate diff against the lavaan oracle.
- `results/geiser.csv` — verbatim copy of the SNLLS paper's Geiser-corpus
  table for the headline numbers.
- `report.html` — rendered report (committed).

## Methodology

- Single-threaded BLAS / OpenMP (env vars set in the runner; `bench::mark`'s
  `iterations = 30L`).
- "Estimate-only" workload: magmaan via `magmaan::magmaan()`; lavaan with
  `se = "none"`, `test = "none"` (same as `benchmarks/r/run_benchmark.R`).
- Correctness gate: magmaan's free-parameter estimates must agree with the
  lavaan oracle to `1e-3` (max absolute) before timing is reported.
- Times are medians over the bench iterations.
