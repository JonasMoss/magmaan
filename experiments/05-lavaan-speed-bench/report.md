# magmaan vs lavaan: speed bench


How fast is magmaan compared to lavaan on the same models? Three
complementary slices below: a *headline* slice on five real published
latent-variable models from Geiser (2013), a small *zoo* slice on
classic lavaan tutorial models that the benchmark harness already
validates against the lavaan oracle, and a whole-pipeline slice that
includes raw-data statistic construction and selected post-fit
reporting.

The headline and zoo slices time **estimate-only** workloads
(`se = "none"`, `test = "none"` in lavaan) at **single-threaded** BLAS /
OpenMP. The pipeline slice times broader calls: UGamma robust reporting,
all-ordinal DWLS with polychorics, and mixed ordinal/continuous DWLS
with polyserial moments.

## Headline: Geiser (2013) corpus

Five real published latent-variable models from Geiser’s textbook, drawn
from the SNLLS paper’s `snlls_geiser_corpus_lavaan.csv` table (committed
under `papers/snlls-constrained/tables/`). magmaan’s `full` and `snlls`
columns are the same fit reported at two backend choices; both are well
under a millisecond at this scale.

| Model | Lavaan ms | magmaan ms (full ML) | magmaan ms (SNLLS) | Speedup |
|:---|---:|---:|---:|---:|
| Latent regression (Geiser 2013) | 33.258 | 0.361 | 0.775 | 42.9× |
| Three-factor CFA (Geiser 2013) | 32.566 | 0.728 | 0.448 | 72.6× |
| First-order linear latent growth curve (Geiser 2013) | 15.948 | 0.301 | 0.240 | 66.6× |
| Latent state-trait model (Geiser 2013) | 103.743 | 1.284 | 1.004 | 103.3× |
| Latent state strict invariance (Geiser 2013) | 34.585 | 0.633 | 0.432 | 80.0× |

## Zoo: classic lavaan tutorial models

Re-run live in this experiment off the `benchmarks/` harness, on
lavaan’s in-package datasets:

- **Holzinger-Swineford 3-factor CFA** (`HolzingerSwineford1939`, n=301,
  9 indicators) at three estimators: ML, ULS, GLS.
- **Bollen industrialization/democracy SEM** (`PoliticalDemocracy`,
  n=75, 11 observed variables, latent regressions + residual
  covariances).
- **`Demo.growth` linear latent growth curve** (n=400, 4 waves + 2
  time-varying covariates + 4 controls, with mean structure).

Speedup = `lavaan_ms_median / magmaan_ms_median`.

| Model | Estimator | n | Obs vars | Params | Lavaan ms | magmaan ms | Speedup | Max \|Δθ\| |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Holzinger-Swineford 3-factor CFA | ML | 301 | 9 | 21 | 16.900 | 0.700 | 24.1× | 9.5e-07 |
| Holzinger-Swineford 3-factor CFA (masked) | FIML | 301 | 9 | 30 | 40.700 | 1.700 | 23.9× | 1.8e-06 |
| Holzinger-Swineford 3-factor CFA | ULS | 301 | 9 | 21 | 23.850 | 0.700 | 34.1× | 3.8e-03 |
| Holzinger-Swineford 3-factor CFA | GLS | 301 | 9 | 21 | 22.400 | 0.900 | 24.9× | 3.5e-03 |
| Bollen democracy SEM | ML | 75 | 11 | 31 | 26.650 | 1.900 | 14.0× | 2.0e-05 |
| Demo.growth linear LGC | ML | 400 | 10 | 17 | 26.550 | 1.400 | 19.0× | 6.7e-07 |

## Whole-pipeline smoke

These rows are deliberately small. They ask whether the README headline
still has a representative path beyond continuous estimate-only fitting:
raw data are converted to the required sample statistics, the model is
fit, and the named post-fit report is materialized where magmaan exposes
one.

| Workflow | Estimator | Data Path | n | Obs vars | Lavaan ms | magmaan ms | Speedup | Check |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| HS 3-factor CFA robust report | ML + UGamma | raw continuous | 301 | 9 | 32.000 | 2.000 | 16.0× | 1.8e-05 |
| Ordinal CFA robust report | DWLS | all ordinal | 360 | 4 | 46.000 | 18.000 | 2.6× | 0.0087 |
| Mixed ordinal/continuous CFA | DWLS | 2 ordinal + 2 continuous | 360 | 4 | 48.500 | 12.500 | 3.9× | 0 |

The `Check` column is the row-specific parity smoke: scaled
chi-square/scale factor drift for the ML UGamma row, unscaled chi-square
drift for the ordinal DWLS row, and free-parameter-count drift for the
mixed row.

## Methodology

- **Estimate-only**: magmaan via `magmaan::magmaan()`; lavaan with
  `se = "none"`, `test = "none"`. The post-fit machinery (standard
  errors, test statistics, fit indices) is excluded — both libraries are
  timed on the *core fit loop* in the headline and zoo slices only.
- **Whole-pipeline**: the smoke rows include raw-data statistic
  construction and the named reporting path. The ML row includes UGamma
  spectra, Satorra-Bentler scaling, and robust SEs; the ordinal row
  includes polychorics, NACOV/weights, DWLS, and robust ordinal
  reporting; the mixed row includes threshold/polyserial/polychoric
  moment construction and DWLS fit.
- **Single-threaded**: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` are set to `1` before R loads BLAS. The
  Geiser-corpus numbers were produced with the same protocol; see
  `papers/snlls-constrained`.
- **Iterations**: 30 per case. Times are medians of repeated
  elapsed-time measurements.
- **Correctness gate**: each zoo case’s magmaan estimates must agree
  with the lavaan oracle to `1e-3` (max absolute over free parameters)
  before timing is reported. The pipeline rows use the row-specific
  smoke check shown in the table.

## Reproduce

``` sh
Rscript experiments/05-lavaan-speed-bench/run_experiment.R
(cd experiments/05-lavaan-speed-bench && quarto render report.qmd)
```

## See also

- [`docs/validation/lavaan_tutorial_parity.md`](../../docs/validation/lavaan_tutorial_parity.md)
  — central lavaan-tutorial parity audit.
- [`benchmarks/README.md`](../../benchmarks/README.md) — the live
  magmaan-vs-lavaan harness this experiment is built on.
- The deeper Geiser-corpus study (estimator timing, convergence) lives
  in the separate SNLLS paper repository; its frozen
  `snlls_geiser_corpus_lavaan.csv` is copied here as
  [`results/geiser.csv`](results/geiser.csv).
