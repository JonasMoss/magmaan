# The Effect of Measurement Error on Hypothesis Testing in Small Sample Structural Equation Modeling: A Comparison of Various Estimation Approaches

**Cite.** Bogaert, J., Loh, W. W., Schuberth, F., & Rosseel, Y. (2025). *Structural Equation Modeling: A Multidisciplinary Journal*, 32(2):215-236. DOI: 10.1080/10705511.2024.2398759.
**PDF.** `external/refs/The Effect of Measurement Error on Hypothesis Testing in Small Sample Structural Equation Modeling  A Comparison of Various Estimation Approaches.pdf`
**Read.** 2026-06-25  ·  **Verdict.** → speculative

## TL;DR
A three-study, ~1300-condition simulation horse race (N from 20 to 1000) on
Type I error and power for a structural coefficient under random measurement
error. The result is the Brunner-Austin regression warning carried into SEM:
estimators that ignore measurement error (UFSR, PLS) keep nominal size with a
single latent predictor, but inflate Type I error up to ~0.89 once two
*correlated* latent predictors are measured with error, and the inflation grows
with N, ρ, and lower reliability rather than washing out.

## Contribution
- Empirical, not a new estimator. Compares eight approaches split into two
  camps. **Unbiased / corrected:** SEM-ML (joint ML, the gold standard) and
  **LSAM** (local structural-after-measurement, Rosseel & Loh 2024; ML mapping
  matrix, equivalent to Croon/Wall-Amemiya bias-corrected factor-score
  regression). **Uncorrected:** UFSR (raw factor-score regression), PLS-A,
  PLS-B; and the partly-corrected SSCN.
- Promotes **LSAM-SSC**, the small-sample correction for LSAM (Bogaert et al.
  2023): Fuller's α-modification shrinks the bias correction to trade variance
  for a little bias. Three α choices benchmarked: SSCP1 (α=P+1), SSCP5 (α=P+5),
  SSCN (α=(N-1)/2). SSCN sits closest to uncorrected and itself inflates size.
- Three population models: one predictor; two correlated predictors; a
  three-wave cross-lagged panel. Reliability ∈ {0.5, 0.8}, ρ ∈ {0,.2,.4,.6}.
- Findings: with one predictor everyone is fine. With correlated predictors the
  uncorrected camp inflates Type I badly and worse with N; PLS-B is the outlier
  (conservative, low power). Power is poor across the board, below 80% even at
  N=200. SEM-ML test statistics go bimodal at N≤30. Recommendation: use SEM-ML
  or LSAM (unbiased camp), especially as N grows; at N≤50 no estimator is ideal
  and a conservative choice may be preferable.

## Relevance to magmaan
Two layers. (1) **Core parity, already covered.** SEM-ML is exactly magmaan's
core surface (complete-data normal-theory ML) and the paper's gold standard;
nothing to build there, this is confirmation that the core estimator is the one
to beat in small-N inference. (2) **A `estimate::frontier` SAM estimator,
unbuilt.** LSAM and its SSC variants are the paper's whole "unbiased
alternative" case, and magmaan has no two-step SAM path.

The structural fit is a clean match: `model::MatrixRep` already carries the
measurement/structural split (Λ, Θ vs B, Ψ) that SAM's two steps need. What
magmaan would add is (a) a measurement-only fit, ideally block-by-block for the
*local* flavor, (b) the Croon/Wall-Amemiya bias correction of the factor
covariance Var(η), and (c) a structural-step solver consuming the corrected
Var(η). lavaan's `sam()` is the oracle, so the LSAM point estimates are
gateable to parity.

Caveats that gate a build, all pushing this to speculative not todo: the
**SSC α-modification is research-tier** (a shrinkage knob with its own
bias-variance validation, not a parity target), and **LSAM-SSC has no
closed-form SE** (the paper falls back to a weighted average of UFSR and LSAM
SEs, plus a t-reference with N-1 df) which is exactly the
[[feedback-shortcut-variants]] "pragmatic approximation carries hidden
validation" pattern. The paper's own limitations section flags **non-normal and
missing data as unexamined** for SAM, which is the territory magmaan's
FIML-FMG / ML2S / misspec-robust-SE work already occupies, so a SAM frontier
would be the natural locus for a "SAM under non-normality/missingness" study
rather than a plain lavaan-parity reimplementation.

## Verdict
→ speculative — recorded as the `estimate::frontier` SAM entry in
[backlog/speculative.md](../../backlog/speculative.md) ("Structural-after-measurement
(SAM / LSAM) two-step estimator"). Build trigger: a paper row or methods
workflow that needs the unbiased two-step alternative on a magmaan fit, most
naturally a small-N inference or SAM-under-misspecification study. Until then,
SEM-ML is the core estimator and lavaan's `sam()` covers the practical case.
