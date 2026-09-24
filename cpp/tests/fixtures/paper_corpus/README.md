# Paper-corpus fixtures

The independent, ignored `external/paper-corpus` repository owns acquisition,
extraction, and lavaan oracle generation. CI reads these frozen JSON files and
needs neither R nor the downloaded datasets.

The original `zxqvn_reference.json` checks complete-data ML. The `examples/`
batch contains 14 additional fits from four sources, exported with lavaan 0.7.2:

- [Boivin CFA tutorial](https://osf.io/8yp5m/): five WLSMV specifications,
  13 ordinal indicators, 492 complete observations after the original listwise
  deletion. Includes hierarchical and two-factor CFA, cross-loadings, and
  equality-constrained cross-loadings.
- [Pregnancy intentions](https://osf.io/d6ptc/): eight-variable FIML mediation,
  298 input rows, 204 incomplete. The original model is saturated.
- [Kievit latent-change tutorial](https://osf.io/4bpmq/): five models on deposited
  simulated, complete datasets (500 rows), plus an explicitly labelled MILCS
  variant with 157 incomplete rows. The variant removes all T2 indicators on
  every fifth row and T1X2 on every seventh row; it is not a published analysis.
- [ACEs/IPV/mental health](https://osf.io/xe4g7/): two DASS models on complete
  item data (312 rows). The hierarchical model converges to a negative latent
  variance in lavaan; it is retained as an inadmissible diagnostic case.

## Inputs and coverage

No participant-level rows or third-party analysis code are included. Ordinal
inputs are thresholds, polychorics, their NACOV, and the DWLS weight. Continuous
inputs are observed-pattern counts, means, N-divisor covariances, and pairwise
start moments. The latter are sufficient for normal-theory FIML estimation,
but not for robust casewise-score inference.

Models enter through the lavaan partable compatibility boundary. This tests
model reconstruction and estimation; it does not independently gate the syntax
parser or the categorical raw-data preprocessing. Point estimation starts from
lavaan's *initial* values, not its fitted coefficients. Ordinal MI/EPC and
univariate equality releases are checked separately at the oracle solution.

Adapter details are explicit:

- Delta ordinal preparation reorders free parameters; the test translates
  starts and reference coefficients by partable row.
- Auto-generated equality rows from legacy `equal(...)` syntax are exported as
  explicit equalities so their distinct source labels do not lose constraints.
- Lavaan's single-group categorical gradient carries `(N-1)/N`, while expected
  information is `Delta' W Delta`. Thus MI/score statistics have the square
  of this factor and EPCs one factor relative to magmaan's N-weight convention.
  The exporter verifies both gradient and information identities against lavaan.
- The delta latent-response covariance has unit diagonal; the generic evaluator
  supplies its off-diagonal entries.

## Regeneration

After extraction, run:

```sh
Rscript external/paper-corpus/scripts/export_examples.R
Rscript cpp/tests/tools/regen_paper_corpus_fixtures.R
```

The bridge checks the corpus identity and copies the expected 14 files. Source
URLs and hashes, data kind, and original/constructed-case notes accompany each
fixture. Do not replace oracle values with magmaan output.
