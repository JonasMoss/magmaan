# An Evaluation of Non-Iterative Estimators in the Structural after Measurement (SAM) Approach to Structural Equation Modeling (SEM)

**Cite.** Dhaene, S., & Rosseel, Y. (2023). *Structural Equation Modeling: A Multidisciplinary Journal*, 30(6):926-940. DOI: 10.1080/10705511.2023.2220135. Open access.
**PDF.** `external/refs/Dhaene and Rosseel 2023 - An evaluation of non-iterative estimators in t ... fter measurement (SAM) approach to structural equation modeling (SEM).pdf`
**Read.** 2026-06-25  ·  **Verdict.** → speculative (folds into existing SAM entry)

## TL;DR
Sibling of the SAM/LSAM work: in the SAM two-step recipe, step 1 (the
measurement model) can be fit by old closed-form non-iterative factor-analytic
estimators instead of iterative ML. The verdict is that the step-1 estimator
choice barely matters; SAM's small-sample win comes from decoupling + bounds +
compartmentalization, not from the estimator.

## Contribution
- Empirical horse race of step-1 measurement estimators inside *local* SAM (ML
  mapping matrix): Multiple-Group (Guttman/Thurstone), FABIN2 (Hägglund),
  Bentler-1982 ULS, James-Stein, plus bounded-ML and bounded-ULS as iterative
  baselines, against full joint SEM (ML, bounded-ML, ULS, MIIV-2SLS). Two
  5-factor sims, 3 indicators/factor, n ∈ {50,100,200,1000}, R² ∈ {.10,.40};
  Study 1 varies reliability/normality, Study 2 varies structural misspec +
  number of measurement blocks.
- Result 1: SAM beats vanilla SEM in small-to-moderate N: no convergence
  failures, no Heywood/inadmissible solutions, smaller MSE (joint ML failed
  ~11% in the worst cell; local SAM never).
- Result 2: differences between iterative and non-iterative step-1 estimators
  are negligible; "may call into question the added value of advanced iterative
  algorithms over closed-form expressions." Among non-iterative ones,
  Multiple-Group and FABIN2 edge out Bentler-ULS and James-Stein. Closed-form is
  far cheaper at scale (48 indicators/factor: ML ~12 s vs Multiple-Group ~0.3 s).
- Theory aside: factor-score regression with Croon's correction is exactly local
  SAM with the ML mapping matrix (their eq. 10 = Bartlett's factor-score matrix),
  so FSR's properties carry over for free.

## Relevance to magmaan
No new item: this sharpens the existing SAM/LSAM speculative entry rather than
opening one. The actionable nuance is a design note for *if* SAM gets built:
don't port the closed-form measurement-estimator zoo (FABIN2 / Guttman /
James-Stein / Bentler), because step-1 estimator choice is near-irrelevant to
accuracy. Reuse magmaan's existing ML measurement fit (or one cheap closed
form); spend the effort on decoupling + Croon correction + bounds. The FSR =
local-SAM-with-ML-mapping identity also ties into the existing `measures`
factor-score path. Parity oracle for any SAM build is lavaan `sam()`; the
non-iterative measurement estimators live in MIIVsem / lavaan's internals.

## Verdict
→ speculative — folded into the SAM entry in
[backlog/speculative.md](../../backlog/speculative.md)
("Structural-after-measurement (SAM / LSAM) two-step estimator") as a step-1
design note, alongside the [Bogaert 2025 eval](2025-bogaert-measurement-error-small-sample.md).
Companion still queued: Dhaene & Rosseel 2024 (non-iterative estimators in CFA),
the structural-part-free prequel, PDF in `external/refs/`.
