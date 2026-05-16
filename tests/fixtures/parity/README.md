# Parity fixtures

Self-contained lavaan-parity fixtures for the real-data benchmark cases. The
C++ golden test `tests/golden/lavaan_parity_golden_test.cpp` consumes these and
gates magmaan against lavaan with **no R at run time**.

## Why this layer exists

The test suite has three comparison surfaces, kept deliberately distinct:

- **corpus golden tests** (`tests/fixtures/{lexer,flat,ptable,fit,...}`,
  oracle `tools/regen_oracle.R`) — *breadth*: 26 small synthetic models that
  exercise every syntax/operator feature. Cheap and exhaustive over surface
  area, but statistically weak.
- **parity golden tests** (this directory, oracle
  `tools/regen_parity_fixtures.R`) — *depth*: real datasets, real N, real
  conditioning, latent SEM/growth, real SEs and fit measures, and the
  raw-data ingestion path.
- **`benchmarks/`** — advisory, fits *live lavaan* every run, R-dependent,
  gates timing not CI correctness.

The parity layer is the formal bridge: `benchmarks/` proves correctness
against live lavaan during development; these fixtures *freeze* that into the
gated C++ suite. CI never runs R; regeneration is a manual developer step.

## Regenerating

```sh
Rscript benchmarks/r/prepare_case.R all      # once, populates benchmarks/data/
Rscript tools/regen_parity_fixtures.R        # all cases (or pass case ids)
```

## Layout

```
parity/<case_id>/
  reference.json   lavaan oracle outputs (always committed)
  data.json        committed raw data    (redistributable datasets only)
```

## `reference.json` schema

| field | meaning |
|---|---|
| `case_id`, `model` | case id and lavaan model syntax |
| `lavaan_version`, `lavaan_function`, `estimator` | oracle provenance (`cfa`/`sem`/`growth`, `ML`) |
| `meanstructure`, `auto_cov_y`, `n_groups` | lavaanify options needed to rebuild the model |
| `n_obs`, `ov_names`, `n_free` | sample size, observed-variable order, free-parameter count |
| `magmaan_aligned` | `true` when magmaan's free set matches lavaan's one-to-one |
| `sample_cov` | `{names, values}` — N-divisor sample covariance (`lavInspect "sampstat"`) |
| `sample_mean` | `{names, values}` for meanstructure cases, else `null` |
| `lavaan_fmin`, `chisq`, `df`, `pvalue` | lavaan fit statistics (magmaan `F == 2 · lavaan_fmin`) |
| `fit_measures` | `{npar, logl, unrestricted_logl, aic, bic, bic2, cfi, tli, rmsea, rmsea_ci_*, rmsea_pvalue, srmr}` |
| `param_lhs/op/rhs` | per-free-parameter labels, **magmaan free order** (aligned cases only) |
| `theta_hat`, `se`, `theta_start_lavaan` | lavaan estimates / SEs / start values, reordered into **magmaan free order** (aligned cases only) |

When `magmaan_aligned` is `false`, the per-parameter arrays are omitted: magmaan
parameterizes the model differently than lavaan, so there is no clean index
mapping. Such a case is a tracked known gap — the parity test still fits it but
only soft-checks it. `demo_growth_linear` is currently in this state (magmaan
has no `growth()` equivalent; see `docs/todo.md`).

## `data.json` schema

Raw observations, reusing the `raw: [{X, [mask]}]` grid schema shared with the
FIML golden fixtures (`tests/golden/fiml_golden_test.cpp`). One block per group;
`mask` is omitted for complete data, so `data::sample_stats_from_raw()` treats
`X` as fully observed. Rows are the listwise-complete cases in `ov_names` order
— exactly what lavaan fit under `missing = "listwise"`.

## Licensing carve-out

`data.json` is committed only for datasets cleared for redistribution. A case
without `data.json` is still gated on ML estimates via `sample_cov`; only the
raw-data ingestion sub-check is skipped.

| case | dataset | `data.json` | basis |
|---|---|---|---|
| `hs_3factor_cfa` | `lavaan::HolzingerSwineford1939` | yes | lavaan, GPL-3 |
| `bollen_democracy_sem` | `lavaan::PoliticalDemocracy` | yes | lavaan, GPL-3 |
| `demo_growth_linear` | `lavaan::Demo.growth` | yes | lavaan, GPL-3 |
| `bfi_5factor` | `psychTools::bfi` | yes | psychTools, GPL-2 \| GPL-3 |
| `mplus_ex5_1` | Mplus User's Guide ex5.1 | no | redistribution not cleared |
