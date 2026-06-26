# Paper evals

Short, opinionated reads of recent SEM / psychometrics papers, scored for
**relevance to magmaan**. The job of an eval is to answer one question fast:
*does this change what magmaan should build, validate, or cite — and if so,
where does it land?* These are reading notes, not literature reviews.

The PDFs themselves live in the ignored `external/refs/`; only the evals are
tracked. So a `external/refs/...pdf` link resolves on the author's machine but
not in a fresh clone — that's expected.

## The graduation flow

An eval is the scratch; the **verdict** is the product. When an eval finds
something actionable, the conclusion graduates to a tracked home and the eval
records where:

- background only → stays here, verdict `background`.
- a may-never-build capability → [`../../backlog/speculative.md`](../../backlog/speculative.md).
- committed work → [`../../backlog/todo.md`](../../backlog/todo.md) (or `simulation.md`).
- a paper-worthy idea → a `papers/<name>/` note or a new experiment.

Example: the Schuhbeck 2025 MI-effect-sizes eval graduated to a
`measures::frontier` entry in `speculative.md`; the eval links to it.

## Adding an eval

One file per paper, named `YYYY-firstauthor-topic.md` (`YYYY` = publication
year). Add a row to the index below. Copy the template:

```markdown
# <Title>

**Cite.** <Authors> (<year>). <Venue>, <vol(issue):pages>. DOI: <doi>.
**PDF.** `external/refs/<filename>.pdf`
**Read.** <YYYY-MM-DD>  ·  **Verdict.** <ignore | background | → speculative | → todo | → paper>

## TL;DR
<1-2 sentences: what it does and the single reason it matters (or doesn't).>

## Contribution
- <terse bullets: the actual new thing, not the abstract's framing>

## Relevance to magmaan
<Does it map to core parity, a frontier method, a paper, an experiment, or
nothing? What inputs would magmaan need vs. already produce? Caveats that gate a
build.>

## Verdict
<ignore | background | → speculative | → todo | → paper> — <one line>.
<If it graduated, link the landing spot: [speculative.md#anchor](...) etc.>
```

Keep it terse (the house paper-prose rules apply: no em-dashes, no tutorial
voice). An eval that says "background, nothing to build" in three lines is a
complete eval.

## Index

| Read | Paper | Topic | Verdict |
|------|-------|-------|---------|
| 2026-06-25 | [Zhang & Yang 2022](2022-zhang-mi-detection-methods.md) | FR vs B-H vs alignment for indicator-level MI detection (continuous MGCFA) | background |
| 2026-06-25 | [Cox, Kelcey & Bai 2023](2023-cox-multilevel-croon-latent-interactions.md) | Croon's bias correction for multilevel SEM with latent interactions | background |
| 2026-06-25 | [Schuhbeck, Sterner & Goretzko 2025](2025-schuhbeck-mi-effect-sizes.md) | MI effect sizes (dMACS/EDM), O(q) closed forms | → speculative |
| 2026-06-25 | [Bogaert, Loh, Schuberth & Rosseel 2025](2025-bogaert-measurement-error-small-sample.md) | Measurement error + small-N hypothesis testing; SEM-ML/LSAM vs UFSR/PLS | → speculative |
| 2026-06-25 | [Zhang & Wu 2024](2024-zhang-structural-model-fit.md) | Structural-model fit: corrected χ²/RMSEA/CFI/SRMR + CIs via two-step | → speculative |
| 2026-06-25 | [Dhaene & Rosseel 2023](2023-dhaene-noniterative-sam.md) | Non-iterative step-1 estimators in SAM; estimator choice barely matters | → speculative (folds into SAM) |
| 2026-06-25 | [De Jonckere & Rosseel 2023](2023-dejonckere-model-based-shrinkage-target.md) | Model-based covariance-shrinkage target for small-N nonconvergence | → speculative (penalized-ML alternative) |
| 2026-06-26 | [Bell, Chalmers & Flora 2024](2024-bell-omega-misspecification.md) | Bias of model-based omega (omega_u / omega_H) under structural misspecification | → speculative (model-based omega) |
