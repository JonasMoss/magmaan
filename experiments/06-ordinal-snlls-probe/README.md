# Ordinal SNLLS Probe

This is a deliberately small experiment around the question: what can the
current magmaan R surface do for ordinal least-squares timing, and where does
native ordinal SNLLS start requiring source work?

The runner uses checked-in ordinal fixtures:

- `tests/fixtures/ordinal/0001_3cat_cfa.ordinal.json`
- `tests/fixtures/ordinal/0008_mixed_levels_cfa.ordinal.json`
- `tests/fixtures/parity/bfi_ordinal_dwls/{data,reference}.json`

For each case it times:

- construction only: thresholds, polychoric correlations, `NACOV`, `W_dwls`,
  and `W_wls` through `data_ordinal_stats_from_df()`;
- fit only: reuse the already constructed ordinal statistics;
- whole path: construct statistics and fit in the same timed call.

Estimator conditions are DWLS, WLS, and an explicitly labeled ULS proxy. The
proxy replaces the `W_dwls` matrices with identities and calls the current
ordinal DWLS fit entry; this is useful as an identity-weight benchmark, but it
is not a native public ordinal ULS estimator.

The script also records native ULS and native SNLLS attempts. In the current
codebase those are expected to report unsupported for true ordinal data:
ordinal fitting exposes DWLS/WLS only, and SNLLS only accepts continuous
sample-statistic objects.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/06-ordinal-snlls-probe/run_experiment.R --reps 2
```

Default run:

```sh
Rscript experiments/06-ordinal-snlls-probe/run_experiment.R
```

Render:

```sh
(cd experiments/06-ordinal-snlls-probe && quarto render report.qmd)
```

## Outputs

Generated files live under `results/` and are ignored by git:

- `metadata.csv`: package/session metadata.
- `construction.csv`: one row per case with ordinal-statistic construction
  time and moment dimensions.
- `fits.csv`: fit-only and whole-path timings for DWLS, WLS, ULS proxy, and
  the polychoric-correlation ULS-SNLLS proxy where it runs.
- `attempts.csv`: direct native ULS/SNLLS attempts on the ordinal objects.

The construction split is intentionally coarse. The current R builder creates
`NACOV`, `W_dwls`, and `W_wls` together, so this experiment can separate
construction from fitting but cannot isolate DWLS-weight construction from
full-WLS-weight construction without a new data-construction API.
