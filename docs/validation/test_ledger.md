# Test Ledger

This ledger maps magmaan's current validation surface by subsystem. It is a
maintainer aid, not a second backlog: use it to find the tests, fixtures, and
reports that protect an area before editing it, then keep remaining-work details
in [`docs/backlog/todo.md`](../backlog/todo.md) or the simulation-specific
backlog.

## How to Use This Ledger

- Start with the CTest label that owns the area: `spec`, `estimate`,
  `inference`, `ordinal`, `api`, `sim`, `parity`, or `robcat`.
- Prefer the narrow loop while developing, for example
  `just test-area estimate FIML` or `just test-area inference score`.
- Run `just test-quick` before handing back ordinary C++ changes; add
  `just test-area parity` or `just r-check` when the edited behavior crosses
  real-data parity or R boundary code.
- Treat advisory checks under `tests/checks/` and experiments as evidence for
  research claims, not as default CI gates.

## Regression Notes

When a fixed bug gets a guard test, preserve the bug shape and the protecting
test so future edits know why the assertion exists. Use this compact format:

```text
Regression: <short symptom and root cause>.
Guard: <test, fixture, example, or report that fails if it comes back>.
Scope: <optional remaining gap or intentionally uncovered cases>.
```

Put the note near the focused test when the guard is local and easy to find.
Put it in this ledger when the bug crosses subsystems, depends on an external
oracle, or needs a maintainer to run a non-obvious report. Keep notes to the
reason the test exists; unresolved work still belongs in the backlog.

## Validation Areas

| Area | Oracle | Protection | Important files/tests | Known gaps |
|---|---|---|---|---|
| Parser and lexer | `docs/grammar/grammar.ebnf`, checked parser fixtures | Unit plus golden tests under `spec` | `tests/unit/lexer_test.cpp`, `tests/unit/parser_test.cpp`, `tests/golden/lexer_golden_test.cpp`, `tests/golden/parser_golden_test.cpp` | Grammar-coverage walk remains manual; grammar changes must edit EBNF first. |
| Lavaanify, spec, and partable projection | lavaan `parTable()` fixtures and corpus exports | Unit, golden, and corpus parity under `spec` and `parity` | `tests/unit/lavaanify_test.cpp`, `tests/golden/lavaanify_golden_test.cpp`, `tests/golden/textbook_corpus_golden_test.cpp` | Little/Newsom and Mplus corpus promotion remains ongoing. |
| Matrix representation and model evaluation | lavaan implied moments plus algebraic invariants | Unit plus golden tests under `spec` | `tests/unit/matrix_rep_test.cpp`, `tests/unit/model_evaluator_test.cpp`, `tests/golden/matrix_rep_golden_test.cpp`, `tests/golden/fit_implied_golden_test.cpp` | Some observed fixed.x path-model implied-moment comparisons remain documented parity exceptions. |
| Complete-data ML and LS estimation | lavaan JSON fixtures and real-data parity fixtures | Unit, golden, and parity tests under `estimate` and `parity` | `tests/unit/ml_test.cpp`, `tests/unit/ls_path_test.cpp`, `tests/golden/ls_golden_test.cpp`, `tests/golden/lavaan_parity_golden_test.cpp` | Tolerance audit is still open; Geiser GLS exceptions need tighter documentation or fixes. |
| FIML and missing data | lavaan FIML fixtures, saturated EM invariants, FIML FMG diagnostics | Unit, golden, R examples, and advisory checks | `tests/unit/fiml_test.cpp`, `tests/golden/fiml_golden_test.cpp`, `r-package/examples/fiml.R`, `tests/checks/fiml_fmg_trace/` | High-level `magmaan(estimator = "FIML")` mean-structure defaults and multi-group starts need care. |
| Ordinal and mixed moments | lavaan ordinal fixtures, robcat fixtures, internal moment invariants | Unit, golden, robcat, R examples, and experiments | `tests/unit/ordinal_test.cpp`, `tests/golden/ordinal_golden_test.cpp`, `tests/golden/robcat_parity_golden_test.cpp`, `r-package/examples/ordinal_dwls_wls.R` | Mixed robust scaled-test parity is still loose; lazy mixed WLS and mixed theta SNLLS remain open. |
| Inference, standardization, and fit measures | lavaan SE, score, standardized, and fit-measure fixtures | Unit plus golden tests under `inference` | `tests/unit/inference_test.cpp`, `tests/unit/score_test.cpp`, `tests/unit/standardized_test.cpp`, `tests/golden/inference_golden_test.cpp`, `tests/golden/standardized_golden_test.cpp` | Ordinal defined-parameter validity and additional ordinal-SEM standardized goldens remain follow-ups. |
| Robust tests, FMG, and nested restrictions | lavaan/semTests parity, R-internals fixtures, weighted-chi-square oracles | Unit, golden, R examples, and advisory checks | `tests/unit/robust_test.cpp`, `tests/unit/fmg_test.cpp`, `tests/golden/score_robust_golden_test.cpp`, `r-package/examples/fmg.R` | Robust MI is still complete-data ML single-group only; self-contained FMG p-value transform goldens are wanted. |
| Optimizers and terminal audits | Recomputed objectives, projected gradients, cross-backend agreement | Unit tests and benchmark/report tracks under `estimate` | `tests/unit/terminal_audit_test.cpp`, `tests/unit/optimizer_crosscheck_test.cpp`, `tests/unit/fit_diagnostics_test.cpp`, `docs/design/terminal-audit.md` | Ultimate verifier and stationarity tolerance calibration remain research work. |
| Simulation | Distribution goldens, deterministic calibration fixtures, stochastic smokes | Unit tests under `sim` plus advisory checks | `tests/unit/norta_test.cpp`, `tests/unit/plsim_test.cpp`, `tests/unit/vale_maurelli_test.cpp`, `tests/checks/plsim/` | Model-implied simulation, ordinal/mixed observed-correlation calibration, and persistent caches remain open. |
| R boundary and examples | lavaan parity through examples and R-shaped wrapper checks | `just r-check` examples plus C++ API tests | `tests/unit/api_sem_test.cpp`, `r-package/examples/*.R`, `r-package/examples/tutorial/run_all.R` | Examples are smoke tests, not exhaustive wrapper coverage; R reconstruction is sensitive around means and groups. |
| Composite frontier | lavaan native composite fixtures and FC-SEM evaluator invariants | Unit, golden, and R frontier example tests | `tests/unit/fcsem_evaluator_test.cpp`, `tests/unit/fcsem_ml_test.cpp`, `tests/golden/composite_golden_test.cpp`, `r-package/examples/fcsem_frontier.R` | Post-fit native W-matrix parity validation is still skipped while fixture ownership is settled. |
| Corpus parity | lavaan-generated real-data and textbook fixtures | Heavy `parity` target and corpus-specific goldens | `tests/golden/geiser_golden_test.cpp`, `tests/golden/mplus_sem_golden_test.cpp`, `tests/golden/paper_corpus_golden_test.cpp`, `tests/golden/textbook_corpus_golden_test.cpp` | Corpus breadth is intentionally staged; some cases document alternate optima or unsupported syntax. |

## High-Risk Map

### FIML and Missing Data

Protected by:

- `tests/unit/fiml_test.cpp`
- `tests/golden/fiml_golden_test.cpp`
- `r-package/examples/fiml.R`
- `tests/checks/fiml_fmg_trace/` and `tests/checks/fiml_fmg_nested/`

Known weak spots: high-level mean-structure defaults, multi-group starts, and
nonlinear equality tangent-space support remain backlog items.

### Ordinal and Mixed Moments

Protected by:

- `tests/unit/ordinal_test.cpp`
- `tests/golden/ordinal_golden_test.cpp`
- `tests/golden/pairwise_golden_test.cpp`
- `tests/golden/robcat_parity_golden_test.cpp`
- `r-package/examples/ordinal_dwls_wls.R`
- `experiments/10-ordinal-inference-cache-probe` and
  `experiments/11-ordinal-snlls-speed`

Known weak spots: mixed robust scaled-test parity has loose guards, and lazy
mixed WLS construction plus mixed theta SNLLS are still open.

### Robust U-Gamma and FMG Reductions

Protected by:

- `tests/unit/robust_test.cpp`
- `tests/unit/fmg_test.cpp`
- `tests/unit/weighted_chisq_test.cpp`
- `tests/golden/score_robust_golden_test.cpp`
- `r-package/examples/fmg.R`
- `tests/checks/robust_score/`

Known weak spots: robust MI has deferred estimator tiers and no df-greater-than
one joint-release path yet; FMG p-value transforms still need a self-contained
C++ golden independent of R examples.

### Optimizer Terminal Audit

Protected by:

- `tests/unit/terminal_audit_test.cpp`
- `tests/unit/fit_diagnostics_test.cpp`
- `tests/unit/optimizer_crosscheck_test.cpp`
- `tests/golden/lavaan_parity_golden_test.cpp`
- `docs/design/terminal-audit.md`

Known weak spots: the absolute stationarity tolerance is provisional, and SNLLS
still needs a post-hoc full-theta audit path for apples-to-apples diagnostics.

### Parser and Lavaanify

Protected by:

- `docs/grammar/grammar.ebnf`
- `tests/unit/parser_test.cpp`
- `tests/unit/lavaanify_test.cpp`
- `tests/golden/parser_golden_test.cpp`
- `tests/golden/lavaanify_golden_test.cpp`
- `tests/golden/textbook_corpus_golden_test.cpp`

Known weak spots: grammar coverage is not yet mechanically reported, and
external corpus promotion is still staged case by case.

### R Boundary Reconstruction

Protected by:

- `tests/unit/api_sem_test.cpp`
- `r-package/examples/high_level_magmaan.R`
- `r-package/examples/model_spec_df_to_data.R`
- `r-package/examples/lavaan_partable_comparison.R`
- `r-package/examples/fit_measures.R`
- `r-package/examples/fmg.R`

Known weak spots: R examples catch workflow regressions but are not exhaustive;
mean-structure, group, ordinal, and post-fit reconstruction paths remain the
places to validate deliberately after R glue edits.
