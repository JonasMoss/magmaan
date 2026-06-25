# Detecting Measurement Noninvariance with Continuous Indicators Using Three Different Statistical Methods under the Framework of Latent Variable Modeling

**Cite.** Zhang, M., & Yang, L. (2022). *Structural Equation Modeling: A Multidisciplinary Journal*, 29(4):550-568. DOI: 10.1080/10705511.2021.2021533.
**PDF.** `external/refs/Detecting Measurement Noninvariance with Continuous Indicators Using Three Different Statistical Methods under the Framework of Latent Variable Modeli.pdf`
**Read.** 2026-06-25  ·  **Verdict.** background

## TL;DR
Simulation horse-race of three indicator-level MI-detection procedures (free
baseline, Benjamini-Hochberg FDR, alignment) for two-group continuous MGCFA. No
new statistic or estimator; a "which method when" recommendation paper. B-H is
most powerful, alignment controls Type I best, free baseline is worst because a
mis-chosen reference indicator poisons it.

## Contribution
- Benchmarks three existing approaches for locating *which* indicator's
  loading/intercept is noninvariant, all under the LVM (MGCFA) framework:
  - **FR (free baseline)**: fix one reference indicator (loading=1, intercept
    equal), free the rest, test each remaining parameter by a single-df nested
    Δχ² against the baseline, Bonferroni-adjust.
  - **B-H**: start from the fully-constrained baseline, release one parameter at
    a time, get a single-df Δχ² per release, control the false discovery rate
    over the resulting p-values via Benjamini-Hochberg (Raykov et al. 2013).
  - **AM (alignment)**: Asparouhov-Muthén (2014) configural model plus
    component-loss optimization of latent means/variances, then pairwise
    parameter comparisons. No reference indicator and no nested comparison.
- Manipulates N (200/500/1000 per group), location (loading vs intercept),
  degree (0.05-0.45 loadings, 0.10-0.90 intercepts), and percentage of
  noninvariant parameters (proportion, same-indicator, indicator count 3-10).
  Data generated in Mplus, 200 reps per cell.
- Findings: B-H has the highest power and benefits from larger N / larger
  noninvariance degree; AM holds Type I near the base level across the board but
  pays in power and degrades once contamination exceeds ~25%; FR is dominated
  almost everywhere because the fixed RI assumption is the weak link. Δχ² is
  reaffirmed as N-sensitive. Recommendation matrix in Table 8.

## Relevance to magmaan
Background, mostly related-work framing for the FIML-FMG MI difference-test line
(exps 21/25, `papers/fiml-fmg/`, [[fiml-fmg-invariance-litmap]]), not a build.

- **FR and B-H are already covered.** Both are wrappers over single-df nested
  LR tests, which is exactly magmaan's MI difference-test territory: magmaan
  produces the constrained/released χ² inputs, and B-H is a downstream
  `p.adjust(method = "BH")` over per-parameter p-values, not core C++ work. The
  "Δχ² sensitive to N" and FDR-multiplicity points are precisely the framing the
  fiml-fmg lit-map already tracks (multiplicity stance, Thissen-Steinberg-Kuang
  2002). This paper is a clean *power + mean-structure* citation for that paper,
  which is what its Psych Methods reviewers asked for ([[peba-nested-reviews]]),
  and its Type-I-control emphasis echoes the exp-23 "case rests on Type I"
  finding ([[exp23-fiml-fmg-vs-mlr]]).
- **Alignment (AM) is the one genuine capability gap.** magmaan has no configural
  alignment; it is not a lavaan-core feature (Mplus / `sirt` own it). It maps to
  the existing multi-group machinery plus a new component-loss optimization over
  latent mean/variance on the `optim` backends. But this comparison paper is not
  its driver (Asparouhov-Muthén 2014 is), and the LRT/robust-chisq MI line the
  paper would inform does not need it. Not graduating it on this paper's
  strength; promote only if an alignment consumer (e.g. many-group cross-national
  MI where pairwise nested LRT is impractical) actually appears.
- Caveats: continuous indicators only, two groups, uniform noninvariance
  direction (the authors flag mixed-direction and categorical as unstudied), so
  the recommendations do not transfer to magmaan's ordinal `group.equal` path
  unchanged.

## Verdict
background — the LRT-based methods (FR, B-H) are already magmaan's nested-test +
downstream-FDR territory; the paper's value is as related-work / power /
mean-structure framing for the fiml-fmg MI difference-test paper. Alignment is
the lone unbuilt method but this paper does not motivate building it.
