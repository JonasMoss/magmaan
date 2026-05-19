# lavaan tutorial parity — runnable examples

One script per in-scope section of the [lavaan tutorial](https://lavaan.ugent.be/tutorial/).
Each reproduces that section's example end-to-end on magmaan's R surface
(`magmaan()` for estimation, the explicit `magmaan_core` primitives for
post-fit inference) and cross-checks the numbers against live `lavaan`. They
are *informal* verification — runnable, self-checking, eyeball-friendly — not
the formal C++ golden/parity suite.

Run one: `Rscript r-package/examples/tutorial/05_cfa.R`
Run all: `Rscript r-package/examples/tutorial/run_all.R`

| Script | Tutorial section |
|---|---|
| `04_syntax1.R`    | Model syntax 1 — the `=~ ~ ~~ ~1` operators |
| `05_cfa.R`        | A CFA example |
| `06_sem.R`        | A SEM example |
| `07_syntax2.R`    | Model syntax 2 — fixing, `start()`, labels, `equal()`, `orthogonal=`, nonlinear `==` |
| `08_means.R`      | Meanstructures |
| `09_groups.R`     | Multiple groups |
| `10_growth.R`     | Growth curves |
| `11_categorical.R`| Categorical data |
| `12_cov_input.R`  | Covariance-matrix input |
| `13_estimators.R` | Estimators and more |
| `14_mediation.R`  | Mediation |
| `15_modindices.R` | Modification indices |
| `16_extract.R`    | Extracting information |

Out of scope (per the project's scope): tutorial sections *Multilevel SEM* and
*ESEM/EFA*. See `docs/validation/lavaan_tutorial_parity.md` for the full audit,
and `docs/backlog/todo.md` §4 for the deferred backlog. Where a script notes a gap (e.g.
modification indices have no R binding yet), that gap is tracked there.
