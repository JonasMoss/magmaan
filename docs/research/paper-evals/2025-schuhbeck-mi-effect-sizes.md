# Quantifying Measurement Non-Invariance Beyond Simple Structure: The Closed Formulas of Universal Effect Size Measures for MI

**Cite.** Schuhbeck, T. M. B., Sterner, P., & Goretzko, D. (2025). *Structural Equation Modeling: A Multidisciplinary Journal*, 33(1):81-88. DOI: 10.1080/10705511.2025.2570447. Open access.
**PDF.** `external/refs/Schuhbeck et al. 2025 - Quantifying measurement non-invariance beyond si ... ucture - The closed formulas of universal effect size measures for MI.pdf`
**Read.** 2026-06-25  ·  **Verdict.** → speculative

## TL;DR
Generalizes the dMACS family of measurement-invariance *effect sizes* (quantify
how much MI is violated, not just whether) from simple-structure models to
arbitrary common-factor models, and replaces the exponential-in-q numerical
integration with O(q) closed forms.

## Contribution
- Unifies all existing MI effect sizes as one class, **Expected Difference
  Measures (EDMs)**: dMACS / dMACS_Signed (Nye & Drasgow 2011), the Glass-Δ
  analogue UDI/SDI and weighted WUDI/WSDI (Gunn et al. 2020), and the Cohen-f
  analogue fMACS (Lai et al. 2025). Every one is built from three
  normal-expectation terms: signed, absolute, and squared model-implied
  item-score difference (eqs. 3a-3c).
- Drops the simple-structure restriction: works for cross-loadings and
  correlated/multidimensional factors by rewriting the defining integral as a
  multivariate expectation over the latent vector. This is what makes the
  measures usable with EFA-based MI methods (EFA trees, mixture MGFA) and
  non-ICM CFA.
- **Closed formulas** via two facts: (i) a linear factor model's between-group
  difference is itself normal (affine transform of η ∼ N(κ, Σ)); (ii) E[X],
  E[|X|] (folded normal), E[X²] of a normal have exact solutions (Obs. 1-2,
  eqs. 4-6). Complexity O(q), linear in the number of factors, vs. the
  predecessors' exponential numerical quadrature.
- R package (lavaan-compatible): `github.com/TiziSchuh/ExpectedDifferenceMeasures`.

## Relevance to magmaan
Adjacent, not core. magmaan ports lavaan for point estimates / SEs / fit stats;
this is a downstream *effect-size reporting* layer on a fitted multi-group
model. The connection is the multi-group invariance machinery magmaan already
has (FIML/ML2S MI difference-test papers, exps 21/25, ordinal `group.equal`):
those answer the *testing* question, this would answer the *how much* question.

Inputs are exactly what a fitted magmaan multi-group model already produces:
group-specific τ_g, Λ_g, and latent (κ_g, Σ_g). The formulas are fully
specified (eqs. 4-6), so a `measures::frontier` implementation is
transcription-plus-validation, not research. Two caveats gate it: the
rotational indeterminacy the authors flag for EFA solutions (dMACS depends on
the rotation, needs its own convention), and the research-tier benchmark
cutoffs (Nye et al. 2019) that should not ship as core interpretive thresholds.

## Verdict
→ speculative — recorded as the `measures::frontier` MI-effect-size entry in
[backlog/speculative.md](../../backlog/speculative.md) ("MI effect sizes
(dMACS / EDM family) for fitted multi-group models"). Build trigger: a paper
row needing MI effect sizes on a magmaan fit, or an EFA-MI consumer that needs
the O(q) form. Until then, the authors' R package covers the practical case on
a lavaan refit.
