# Parity fixtures

Self-contained lavaan-parity fixtures for the real-data benchmark cases. The
C++ golden test `tests/golden/lavaan_parity_golden_test.cpp` consumes these and
gates magmaan against lavaan with **no R at run time**.

## Why this layer exists

The test suite has three comparison surfaces, kept deliberately distinct:

- **corpus golden tests** (`tests/fixtures/{lexer,flat,ptable,fit,ls,ordinal,
  ...}`, oracle `tools/regen_oracle.R`) — *breadth*: small synthetic models
  that exercise every syntax/operator feature. Cheap and exhaustive over
  surface area, but statistically weak.
- **parity golden tests** (this directory, oracle
  `tools/regen_parity_fixtures.R`) — *depth*: real datasets, real N, real
  conditioning, real SEs and fit measures, and the raw-data ingestion path.
- **`benchmarks/`** — advisory, fits *live lavaan* every run, R-dependent,
  gates timing not CI correctness.

The parity layer is the formal bridge: `benchmarks/` proves correctness
against live lavaan during development; these fixtures *freeze* that into the
gated C++ suite. CI never runs R; regeneration is a manual developer step.

## Estimator families

Each parity case belongs to one of four families, one C++ `TEST_CASE` each:

| family | cases | what it exercises |
|---|---|---|
| `ML` | `hs_3factor_cfa`, `bollen_democracy_sem`, `demo_growth_linear`, `bfi_5factor`, `mplus_ex5_1` | complete-data normal-theory ML |
| `FIML` | `bfi_fiml` | raw-data full-information ML over genuine missingness |
| `LS` | `hs_3factor_ls`, `hs_3factor_ls_mg_configural` | continuous ULS / GLS / WLS — single- and multi-group configural (HS by school) |
| `ordinal` | `bfi_ordinal_dwls` | ordinal DWLS / WLS on integer Likert data |

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

Common to every family: `case_id`, `model` (or `input` for ordinal),
`family`, `lavaan_version`, `n_groups`, `magmaan_aligned`.

`magmaan_aligned` is `true` when magmaan's free set matches lavaan's one-to-one.
When `false`, per-parameter arrays are omitted and the case is a tracked known
gap — the parity test fits it but only soft-checks it. `demo_growth_linear` is
currently in this state (magmaan has no `growth()` equivalent; see
`docs/todo.md`).

### ML family

| field | meaning |
|---|---|
| `lavaan_function`, `estimator`, `meanstructure`, `auto_cov_y`, `n_obs`, `ov_names`, `n_free` | oracle provenance and lavaanify options |
| `sample_cov` | `{names, values}` — N-divisor sample covariance |
| `sample_mean` | `{names, values}` for meanstructure cases, else `null` |
| `lavaan_fmin`, `chisq`, `df`, `pvalue` | lavaan fit statistics (magmaan `F == 2 · lavaan_fmin`) |
| `fit_measures` | `{npar, logl, unrestricted_logl, aic, bic, bic2, cfi, tli, rmsea, rmsea_ci_*, rmsea_pvalue, srmr}` |
| `param_lhs/op/rhs`, `theta_hat`, `se`, `theta_start_lavaan` | per-free-parameter labels / estimates / SEs / starts, in **magmaan free order** (aligned cases only) |

### FIML family

`theta_hat`, `param_lhs/op/rhs` in magmaan free order; `df`, `chisq`, `logl`,
`unrestricted_logl`, `baseline_chisq`, `baseline_df`, `aic`, `bic`, `bic2`,
`npar`, `cfi`, `tli`, `rmsea`, `rmsea_ci_*`, `rmsea_pvalue`. A `robust` block
holds `se_robust_huberwhite` (magmaan free order) plus `mlr_chisq_scaled`,
`mlr_scaling_factor`, `mlr_trace_ugamma{,_h1,_h0}` from a second lavaan
`estimator = "MLR"` fit.

### LS family

`estimators` lists the estimators present. `fits` maps each estimator name to
`{theta_hat (magmaan order), fmin, chisq, df, npar, WLS.V, sample_cov,
sample_mean, n_obs}`. The `ULS` entry additionally carries a `robust` block
(`se`, `gamma`, `eigvals`, `chisq_standard`, Satorra-Bentler-family
statistics). `WLS.V` is lavaan's weight matrix — the C++ test feeds it into
`gls::WLS` so magmaan fits the same weighted moments lavaan did.

Multi-group LS cases set `n_groups > 1` and add `group_var` / `group_labels`;
`sample_cov`, `WLS.V`, and the robust `gamma` each become one `{block, matrix}`
entry per group, and every fit's `n_obs_per_block` carries the per-group
counts.

### ordinal family

`input` is the bare model; the C++ test appends the threshold/`~*~` rows.
`ordered` lists the ordinal variables. `fits` maps `DWLS`/`WLS` to
`{theta_hat, free_rows, chisq, df, cfi, tli, rmsea, srmr}`; the `DWLS` entry
additionally carries a `robust` block.

## What this layer does *not* gate

magmaan's post-fit API exposes no *standard-SE* path for FIML or LS, and no
LS/ordinal fit-measure (cfi/tli/rmsea/srmr) path. The parity test gates every
quantity lavaan exposes that magmaan can also produce:

| family | gated | not gated |
|---|---|---|
| `FIML` | theta, logl, unrestricted_logl, chi2, df, AIC/BIC/BIC2, npar, baseline, CFI/TLI/RMSEA, robust MLR SEs + scaled test | standard `missing="ml"` SE, SRMR |
| `LS` | theta, chi2, df, npar, robust ULS SEs + scaled tests | CFI/TLI/RMSEA/SRMR, GLS/WLS SEs |
| `ordinal` | theta, chi2, df, robust SEs + scaled tests | CFI/TLI/RMSEA/SRMR |

ULS standard chi-square is lavaan's Browne residual NT statistic, which
magmaan reproduces exactly; GLS/WLS chi-square follows the `2·N·fmin`
convention and is gated loosely against lavaan's reported chi-square (the same
estimator-convention divergence the `ls/` corpus goldens accept).

## `data.json` schema

Raw observations under the `raw: [{X, [mask]}]` grid schema shared with the
FIML golden fixtures. One block per group.

- **ML / LS** — `raw: [{X}, …]`, no mask: listwise-complete rows in `ov_names`
  order, one block per group (single-group cases have a single block), exactly
  what lavaan fit under `missing = "listwise"`.
- **FIML** — `raw: [{X, mask}]`: every row with at least one observed value,
  NAs preserved (`null` in `X`, `0` in `mask`). The genuine missing-data path.
- **ordinal** — `{ordered, blocks: [{block, label, matrix}]}`: integer Likert
  data, one block per group.

## Licensing carve-out

`data.json` is committed only for datasets cleared for redistribution. A case
without `data.json` is still gated on estimates via `sample_cov`; only the
raw-data ingestion sub-check is skipped.

| case | dataset | `data.json` | basis |
|---|---|---|---|
| `hs_3factor_cfa` | `lavaan::HolzingerSwineford1939` | yes | lavaan, GPL-3 |
| `bollen_democracy_sem` | `lavaan::PoliticalDemocracy` | yes | lavaan, GPL-3 |
| `demo_growth_linear` | `lavaan::Demo.growth` | yes | lavaan, GPL-3 |
| `bfi_5factor` | `psychTools::bfi` | yes | psychTools, GPL-2 \| GPL-3 |
| `mplus_ex5_1` | Mplus User's Guide ex5.1 | no | redistribution not cleared |
| `bfi_fiml` | `psychTools::bfi` | yes | psychTools, GPL-2 \| GPL-3 |
| `hs_3factor_ls` | `lavaan::HolzingerSwineford1939` | yes | lavaan, GPL-3 |
| `hs_3factor_ls_mg_configural` | `lavaan::HolzingerSwineford1939` | yes | lavaan, GPL-3 |
| `bfi_ordinal_dwls` | `psychTools::bfi` | yes | psychTools, GPL-2 \| GPL-3 |
