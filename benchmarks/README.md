# magmaan SEM Zoo Benchmarks

This directory is the staging area for repeatable benchmark cases. The harness
is R-first because the public comparison target is lavaan and the exploratory R
package is the user-facing path for now.

The repository tracks case metadata, model syntax, source notes, and small
reference summaries. Downloaded raw data, prepared CSV files, and raw timing
results stay in ignored local cache directories until redistribution terms are
explicitly clear.

## Layout

- `cases.yml` is the human-readable case manifest.
- `cases/<case_id>/model.lav` stores lavaan-style model syntax.
- `cases/<case_id>/source.yml` records source, status, license notes, and
  feature requirements.
- `r/` contains the preparation and reference-generation scripts.
- `data/` is an ignored local cache for raw and prepared data.
- `results/` is an ignored local cache for timing outputs.

## First Commands

```sh
Rscript benchmarks/r/check_sources.R
Rscript benchmarks/r/fetch_data.R
Rscript benchmarks/r/prepare_case.R all
Rscript benchmarks/r/make_lavaan_reference.R all
Rscript benchmarks/r/run_benchmark.R all
```

`prepare_case.R` and `make_lavaan_reference.R` accept explicit case ids or
`all` (every case marked `supported_now` in `r/cases.R`). `run_benchmark.R`
fits each active case with the magmaan R package and with lavaan, validates
magmaan against the lavaan oracle (estimate drift gate plus reported SE and
chi-square agreement), times both at the estimate-only workload level, and
writes a result summary into the ignored `results/` cache.

External cases such as Stata Press and Mplus examples use `fetch_data.R` to
download raw data into the ignored cache. Cases whose terms are unclear should
stay as metadata and manual-download notes until they are audited.

## Outstanding

- License audit: the Stata Press and Mplus example datasets are fetched into
  the local `data/` cache for local benchmarking only. Their redistribution
  terms have not been audited -- do this before vendoring any raw or cleaned
  data into the repository.
- `stata_higher_order` and `stata_correlated_uniqueness` ship as Stata
  summary-statistics (`ssd`) files, not raw observations; activating them needs
  covariance-matrix extraction plus model syntax pinned from the Stata SEM
  Reference Manual.
- `stata_growth` has real raw data cached but its `model.lav` is still a
  placeholder; the latent-curve syntax must be pinned from source.
- magmaan-side smoke gaps surfaced by `run_benchmark.R`: ML L-BFGS fails the
  line search on `bollen_democracy_sem` and does not converge on `bfi_5factor`;
  magmaan has no `growth()` equivalent, so `demo_growth_linear` does not match
  the lavaan growth parameterization.
