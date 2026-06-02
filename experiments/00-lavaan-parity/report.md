# lavaan audit parity


<style>
.scope-box {
  border-left: 4px solid #475569;
  padding: 0.7rem 1rem;
  background: #f8fafc;
  margin: 1rem 0;
}
.failure-box {
  border-left: 4px solid #b91c1c;
  padding: 0.7rem 1rem;
  background: #fff1f2;
  margin: 1rem 0;
}
.pass-box {
  border-left: 4px solid #15803d;
  padding: 0.7rem 1rem;
  background: #f0fdf4;
  margin: 1rem 0;
}
.caveat-box {
  border-left: 4px solid #92400e;
  padding: 0.7rem 1rem;
  background: #fffbeb;
  margin: 1rem 0;
}
.audit-table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5rem 0 1.4rem 0;
  font-size: 0.92rem;
}
.audit-table th,
.audit-table td {
  border: 1px solid #d7dde5;
  padding: 0.35rem 0.45rem;
  vertical-align: top;
}
.audit-table th {
  background: #e2e8f0;
}
.audit-table tr.status-fail td {
  background: #fff1f2;
}
.audit-table tr.status-error td {
  background: #fffbeb;
}
.audit-table tr.status-pass td {
  background: #f8fafc;
}
.empty-table {
  color: #64748b;
  font-style: italic;
}
</style>

## Scope

<div class="scope-box">

This experiment presents the paper-side lavaan audit-parity validation
in a single inspectable place. The full sweep fits each SNLLS
textbook-corpus <code>(case, weight)</code> cell in <code>lavaan</code>,
aligns lavaan’s returned free-parameter vector to magmaan’s model,
evaluates the same point with <code>magmaan_core\$evaluate_at()</code>,
and compares lavaan’s <code>lavInspect(fit, “converged”)</code>
self-report with magmaan’s terminal audit stationarity verdict. This is
a convergence-verdict validation, not a new estimator, not a fixture
generator, and not the benchmark speed surface.

</div>

The tables below read the latest normalized outputs written by
`run_experiment.R`. Plain `run_experiment.R` copies the legacy
paper-side combined sweep, audit-parity-v7; `--from-oracle` instead
reuses a lavaan-only cache and reruns just magmaan’s audit. The targeted
refit is not a full sweep: it reruns only the 24 v7 disagreement cells
and projects the current failure surface under the assumption that the
703 v7 agreement cells did not regress.

<div class="caveat-box">

<strong>ADF conditioning caveat.</strong> The conditioned ADF columns
are diagnostic telemetry only. They apply a rank-revealing spectral trim
to the empirical Browne NACOV and ask whether the KKT audit is
stationary on the retained numerical subspace. They do not replace the
primary full-weight ADF objective, do not change magmaan’s optimizer
behavior, and should not be read as a silent fallback estimator. A
conditioned pass with a strict-audit failure means “the full inverse is
numerically unstable here,” not “the ADF objective was changed and the
mismatch disappeared.”

</div>

## Oracle Cache

<table class="audit-table">

<thead>

<tr>

<th>

Field
</th>

<th>

Value
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

Audit Mode
</td>

<td>

cached lavaan oracle
</td>

</tr>

<tr>

<td>

Audit Source
</td>

<td>

lavaan-0.6.21-rds-20260525-170200
</td>

</tr>

<tr>

<td>

Oracle Run
</td>

<td>

lavaan-0.6.21-rds-20260525-170200
</td>

</tr>

<tr>

<td>

lavaan Version
</td>

<td>

0.6.21
</td>

</tr>

<tr>

<td>

Refit Source
</td>

<td>

refit_disagreements_v8.csv
</td>

</tr>

</tbody>

</table>

## Current Failures

<div class="failure-box">

<strong>Current/projected failures are shown first.</strong> When the v8
refit table is present, this block is based on the 24-cell targeted
refit; otherwise it falls back to the v7 disagreement table.

</div>

<table class="audit-table">

<thead>

<tr>

<th>

Case
</th>

<th>

Classification
</th>

<th>

Weight
</th>

<th>

Audit Grad Inf
</th>

<th>

magmaan fmin
</th>

<th>

lavaan fmin bare
</th>

<th>

fmin Ratio
</th>

<th>

ADF Γ Rank
</th>

<th>

ADF Warning
</th>

<th>

Conditioned fmin
</th>

<th>

Conditioned Grad Inf
</th>

<th>

Conditioned Stationary
</th>

</tr>

</thead>

<tbody>

<tr class="status-fail">

<td>

muthen_2017_ch2_ex2_1\_\_adf
</td>

<td>

ADF conditioning side finding
</td>

<td>

ADF
</td>

<td>

3.86
</td>

<td>

7.5e-16
</td>

<td>

8.15e-34
</td>

<td>

9.21e+17
</td>

<td>

8/9
</td>

<td>

TRUE
</td>

<td>

8.89e-31
</td>

<td>

8.84e-15
</td>

<td>

TRUE
</td>

</tr>

</tbody>

</table>

## Block A: Full-Sweep Headline

<table class="audit-table">

<thead>

<tr>

<th>

Audit Run
</th>

<th>

Cells
</th>

<th>

Cells With Both Fits
</th>

<th>

Errors
</th>

<th>

Agree
</th>

<th>

lavaan Yes / Audit No
</th>

<th>

lavaan No / Audit Yes
</th>

<th>

Agreement Rate
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

oracle-lavaan-0.6.21-rds-20260525-170200
</td>

<td>

756
</td>

<td>

704
</td>

<td>

52
</td>

<td>

703
</td>

<td>

1
</td>

<td>

0
</td>

<td>

99.9%
</td>

</tr>

</tbody>

</table>

<table class="audit-table">

<thead>

<tr>

<th>

metric
</th>

<th>

value
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

v7 disagreement cells refit
</td>

<td>

24
</td>

</tr>

<tr>

<td>

v8 agreements among v7 disagreements
</td>

<td>

23
</td>

</tr>

<tr>

<td>

v8 remaining disagreements
</td>

<td>

1
</td>

</tr>

<tr>

<td>

v8 errors
</td>

<td>

0
</td>

</tr>

</tbody>

</table>

## Block B: Disagreements From The Last Full Sweep

These are the v7 off-diagonal cells before the targeted v8 refit. The
classification column separates the LCS / phantom-latent objective gap,
the single-indicator `auto.fix.single` parity gap, the Muthen ADF
conditioning side finding, and ordinary gradient-verdict splits.

<table class="audit-table">

<thead>

<tr>

<th>

Case
</th>

<th>

Classification
</th>

<th>

Book
</th>

<th>

Family
</th>

<th>

Weight
</th>

<th>

lavaan Converged
</th>

<th>

Audit Stationary
</th>

<th>

Audit Grad Inf
</th>

<th>

magmaan fmin
</th>

<th>

lavaan fmin bare
</th>

<th>

fmin Ratio
</th>

</tr>

</thead>

<tbody>

<tr class="status-fail">

<td>

muthen_2017_ch2_ex2_1\_\_adf
</td>

<td>

ADF conditioning side finding
</td>

<td>

muthen_2017
</td>

<td>

sem
</td>

<td>

ADF
</td>

<td>

TRUE
</td>

<td>

FALSE
</td>

<td>

3.86
</td>

<td>

7.5e-16
</td>

<td>

8.15e-34
</td>

<td>

9.21e+17
</td>

</tr>

</tbody>

</table>

## Block C: Breakdown By Book And Weight

<table class="audit-table">

<thead>

<tr>

<th>

Block
</th>

<th>

Book
</th>

<th>

Cells
</th>

<th>

Agree
</th>

<th>

lavaan Yes / Audit No
</th>

<th>

lavaan No / Audit Yes
</th>

<th>

Agreement Rate
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

book
</td>

<td>

brown_2015
</td>

<td>

43
</td>

<td>

43
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

<tr>

<td>

book
</td>

<td>

geiser_2013
</td>

<td>

70
</td>

<td>

70
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

<tr>

<td>

book
</td>

<td>

kline_2023
</td>

<td>

74
</td>

<td>

74
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

<tr>

<td>

book
</td>

<td>

little_2013
</td>

<td>

95
</td>

<td>

95
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

<tr>

<td>

book
</td>

<td>

mplus_users_guide_v8
</td>

<td>

42
</td>

<td>

42
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

<tr>

<td>

book
</td>

<td>

muthen_2017
</td>

<td>

19
</td>

<td>

18
</td>

<td>

1
</td>

<td>

0
</td>

<td>

94.7%
</td>

</tr>

<tr>

<td>

book
</td>

<td>

newsom_2015
</td>

<td>

361
</td>

<td>

361
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

</tbody>

</table>

<table class="audit-table">

<thead>

<tr>

<th>

Block
</th>

<th>

Weight
</th>

<th>

Cells
</th>

<th>

Agree
</th>

<th>

lavaan Yes / Audit No
</th>

<th>

lavaan No / Audit Yes
</th>

<th>

Agreement Rate
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

weight
</td>

<td>

ADF
</td>

<td>

166
</td>

<td>

165
</td>

<td>

1
</td>

<td>

0
</td>

<td>

99.4%
</td>

</tr>

<tr>

<td>

weight
</td>

<td>

GLS
</td>

<td>

269
</td>

<td>

269
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

<tr>

<td>

weight
</td>

<td>

ULS
</td>

<td>

269
</td>

<td>

269
</td>

<td>

0
</td>

<td>

0
</td>

<td>

100.0%
</td>

</tr>

</tbody>

</table>

## Block D: LCS Objective-Gap Probe

The LCS probe compares `Σhat(theta_hat)`, `muhat(theta_hat)`, sample
covariances, and sample means element by element on three target cells.
It diagnoses the old large fmin-ratio outliers as an implied-moment
assembly problem in LCS / phantom-latent patterns.

<table class="audit-table">

<thead>

<tr>

<th>

case
</th>

<th>

n
</th>

<th>

p
</th>

<th>

magmaan_fmin
</th>

<th>

lavaan_fx
</th>

<th>

ratio
</th>

<th>

audit_grad_inf
</th>

<th>

worst_element
</th>

<th>

worst_magmaan
</th>

<th>

worst_lavaan
</th>

<th>

worst_diff
</th>

<th>

worst_rel_diff
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

newsom_2015_ex9_3\_\_uls
</td>

<td>

5335
</td>

<td>

6
</td>

<td>

1.18e+03
</td>

<td>

1.18e+03
</td>

<td>

1
</td>

<td>

0.00011
</td>

<td>

sigma\[bmi6,bmi5\]
</td>

<td>

47.6
</td>

<td>

47.6
</td>

<td>

-6.39e-14
</td>

<td>

-1.34e-15
</td>

</tr>

<tr>

<td>

newsom_2015_ex3_4d\_\_gls
</td>

<td>

574
</td>

<td>

3
</td>

<td>

0.0526
</td>

<td>

0.0525
</td>

<td>

1
</td>

<td>

9.72e-09
</td>

<td>

S\[w1posaff,w1posaff\]
</td>

<td>

0.399
</td>

<td>

0.399
</td>

<td>

1.05e-15
</td>

<td>

2.65e-15
</td>

</tr>

<tr>

<td>

newsom_2015_ex9_5b\_\_gls
</td>

<td>

5335
</td>

<td>

11
</td>

<td>

0.0146
</td>

<td>

0.0146
</td>

<td>

1
</td>

<td>

4.83e-07
</td>

<td>

sigma\[bmi1,bmi1\]
</td>

<td>

21.7
</td>

<td>

23.1
</td>

<td>

-1.37
</td>

<td>

-0.0612
</td>

</tr>

</tbody>

</table>

## Block E: Driver Inventory

<table class="audit-table">

<thead>

<tr>

<th>

driver
</th>

<th>

path
</th>

<th>

role
</th>

<th>

expected_runtime
</th>

</tr>

</thead>

<tbody>

<tr>

<td>

lavaan oracle collector
</td>

<td>

experiments/00-lavaan-parity/scripts/collect_lavaan_oracle.R
</td>

<td>

slow lavaan-only theta_hat/objective/convergence cache
</td>

<td>

slow, about lavaan side of full sweep
</td>

</tr>

<tr>

<td>

cached-oracle magmaan audit
</td>

<td>

experiments/00-lavaan-parity/run_experiment.R
</td>

<td>

fresh magmaan evaluate_at audit against cached lavaan theta_hat
</td>

<td>

fast relative to lavaan collection
</td>

</tr>

<tr>

<td>

legacy full audit sweep
</td>

<td>

papers/snlls-continuous/scripts/run_lavaan_audit_parity.R
</td>

<td>

756-cell combined lavaan self-report vs magmaan terminal-audit sweep
</td>

<td>

about 1 h
</td>

</tr>

<tr>

<td>

LCS moment diagnosis
</td>

<td>

papers/snlls-continuous/dev/scripts/diagnose_lcs_disagreement.R
</td>

<td>

side-by-side implied/sample moments on three LCS target cells
</td>

<td>

about 30 s
</td>

</tr>

<tr>

<td>

v7 disagreement refit
</td>

<td>

experiments/00-lavaan-parity/scripts/refit_disagreements.R
</td>

<td>

targeted refit of the 24 audit-parity-v7 disagreement cells
</td>

<td>

about 3 s
</td>

</tr>

<tr>

<td>

ADF conditioning probe
</td>

<td>

experiments/00-lavaan-parity/scripts/diagnose_adf_conditioning.R
</td>

<td>

eigendecomposition of the saturated Muthen ADF conditioning case
</td>

<td>

about 1 s
</td>

</tr>

<tr>

<td>

paper supplement
</td>

<td>

papers/snlls-continuous/supplement/lavaan-audit-parity.qmd
</td>

<td>

original Quarto supplement consumed by the paper
</td>

<td>

render only
</td>

</tr>

<tr>

<td>

LCS diagnosis writeup
</td>

<td>

papers/snlls-continuous/dev/audits/lcs-objective-gap.md
</td>

<td>

diagnostic narrative for the LCS / phantom-latent objective gap
</td>

<td>

read only
</td>

</tr>

</tbody>

</table>

## Reproduce

Normalize existing outputs and render this report:

``` sh
Rscript experiments/00-lavaan-parity/run_experiment.R
(cd experiments/00-lavaan-parity && quarto render report.qmd)
```

Collect a new lavaan-only oracle cache using the newest CRAN lavaan,
then audit current magmaan against that cached oracle:

``` sh
Rscript experiments/00-lavaan-parity/run_experiment.R --collect-lavaan --install-latest-lavaan
Rscript experiments/00-lavaan-parity/run_experiment.R --from-oracle
```

Run the legacy combined paper sweep only when needed:

``` sh
Rscript experiments/00-lavaan-parity/run_experiment.R --run-full
```

Run the cheap targeted probes:

``` sh
Rscript experiments/00-lavaan-parity/scripts/refit_disagreements.R
Rscript experiments/00-lavaan-parity/scripts/diagnose_adf_conditioning.R
Rscript experiments/00-lavaan-parity/run_experiment.R --run-lcs
```
