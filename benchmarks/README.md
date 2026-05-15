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
Rscript benchmarks/r/prepare_case.R hs_3factor_cfa bollen_democracy_sem demo_growth_linear
Rscript benchmarks/r/make_lavaan_reference.R hs_3factor_cfa bollen_democracy_sem demo_growth_linear
```

External cases such as Stata Press and Mplus examples use fetch scripts into
the ignored cache. Cases whose terms are unclear should stay as metadata and
manual-download notes until they are audited.
