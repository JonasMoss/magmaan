# magmaan vs lavaan: speed bench


How much faster is magmaan than lavaan on the same fitted model?

Fitting the identical one-factor CFA, magmaan is about 18× faster on
plain ML, 33× faster under 10% FIML missingness, and 28× faster on
ordinal DWLS — geometric means across three population sizes
(n=200-1000, p=6-12), with every fit checked to match lavaan’s estimates
before it is timed.

## Evidence

| Fit                         | Speedup |
|:----------------------------|--------:|
| Continuous ML               |     18× |
| FIML (10% MCAR)             |     33× |
| Ordinal DWLS (4 categories) |     28× |

Per-size detail (all three population sizes, both engines’ median times)
is in [`results/cases.csv`](results/cases.csv).

## Caveats

- Both engines are timed on the fit call only: parsing model syntax and
  building sample statistics happens once beforehand, outside the timer,
  for both sides. This is a repeated-fit speedup, not
  raw-syntax-to-answer speed.
- Estimate-only: no standard errors, test statistics, or fit indices are
  requested from either engine here.
- Small, synthetic, single-factor populations at three sizes. Not a
  general performance claim across model families or larger problems.
- Timed with a shared batched timer (seven batches, engine order rotated
  across batches, single BLAS/OpenMP thread), not a single
  `system.time()` call.
- A fuller, staged-attribution speed study (raw-to-report timing broken
  into parsing/fitting/inference) is tracked separately and does not yet
  have public numbers; see
  [project/validation/benchmark_plan.md](../../../project/validation/benchmark_plan.md).

## Reproduce

``` sh
Rscript experiments/showcases/02-lavaan-speed-bench/run_experiment.R
(cd experiments/showcases/02-lavaan-speed-bench && quarto render report.qmd)
```

## See also

- [`project/validation/lavaan_tutorial_parity.md`](../../../project/validation/lavaan_tutorial_parity.md)
  — central lavaan-tutorial parity audit.
- [`benchmarks/README.md`](../../../benchmarks/README.md) — the shared
  benchmark harness this experiment builds on.
