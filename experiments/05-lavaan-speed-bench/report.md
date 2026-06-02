# magmaan vs lavaan: speed bench


How fast is magmaan compared to lavaan on the same models? Two
complementary slices below: a *headline* slice on five real published
latent-variable models from Geiser (2013), and a small *zoo* slice on
classic lavaan tutorial models that the benchmark harness already
validates against the lavaan oracle.

Both slices time **estimate-only** workloads (`se = "none"`,
`test = "none"` in lavaan) at **single-threaded** BLAS / OpenMP.
Magmaan’s free-parameter estimates agree with the lavaan oracle to
better than `1e-3` (max absolute) on every case shown.

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
| Holzinger-Swineford 3-factor CFA | ML | 301 | 9 | 21 | 20.780 | 0.782 | 26.6× | 9.5e-07 |
| Holzinger-Swineford 3-factor CFA (masked) | FIML | 301 | 9 | 30 | 59.401 | 1.528 | 38.9× | 1.8e-06 |
| Holzinger-Swineford 3-factor CFA | ULS | 301 | 9 | 21 | 31.248 | 0.778 | 40.2× | 3.8e-03 |
| Holzinger-Swineford 3-factor CFA | GLS | 301 | 9 | 21 | 34.688 | 0.892 | 38.9× | 3.5e-03 |
| Bollen democracy SEM | ML | 75 | 11 | 31 | 33.778 | 1.493 | 22.6× | 2.0e-05 |
| Demo.growth linear LGC | ML | 400 | 10 | 17 | 38.564 | 3.432 | 11.2× | 6.7e-07 |

## Methodology

- **Estimate-only**: magmaan via `magmaan::magmaan()`; lavaan with
  `se = "none"`, `test = "none"`. The post-fit machinery (standard
  errors, test statistics, fit indices) is excluded — both libraries are
  timed on the *core fit loop*.
- **Single-threaded**: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` are set to `1` before R loads BLAS. The
  Geiser-corpus numbers were produced with the same protocol; see
  `papers/snlls-constrained`.
- **Iterations**: 30 per case via `bench::mark`. Times are medians.
- **Correctness gate**: each zoo case’s magmaan estimates must agree
  with the lavaan oracle to `1e-3` (max absolute over free parameters)
  before timing is reported. The harness that enforces this is
  `benchmarks/r/run_benchmark.R`.

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
