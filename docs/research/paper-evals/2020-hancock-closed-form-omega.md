# A Closed-Form Alternative for Estimating ω Reliability under Unidimensionality

**Cite.** Hancock, G. R., & An, J. (2020). Measurement: Interdisciplinary Research and Perspectives, 18(1):1-14. DOI: 10.1080/15366367.2019.1656049.
**PDF.** `external/refs/A Closed-Form Alternative for Estimating   Reliability under Unidimensionality.pdf`
**Read.** 2026-06-26  ·  **Verdict.** background

## TL;DR
The peer-reviewed prior art for closed-form McDonald's omega under unidimensionality: loadings from triple-covariance ratios, no CFA fit. It is the oracle and citation for magmaan's already-landed `measures::frontier::reliability` Spearman-Guttman covariance omega, not new build work.

## Contribution
- Single-factor omega without a CFA: loadings from the triple-covariance identity `λ_i² = (s_ij s_ik)/s_jk`, aggregated by Spearman's (1927) ratio-of-sums `λ̂_i = [Σ_{j<k} s_ij s_ik / Σ_{j<k} s_jk]^{1/2}`, then `ω̂ = (Σ λ̂_i)² / s_Y²`. The denominator is the observed total-score variance, so no model enters there.
- 256-cell sim (P in {3,6,9,12}, loading magnitude/variability, n in {250..5000}, continuous + 3 ordinal types) vs CFA-omega: bias, precision (SE vs CFA-SE), and failure rates highly comparable; the closed form only struggles at P=3 with low/variable loadings and small n.
- Explicitly single factor; hierarchical/bifactor and local dependence are out of scope.

## Relevance to magmaan
Directly the prior art for the covariance-only **Spearman-Guttman omega** that landed in `measures::frontier::reliability` (2026-06-26; `magmaan_core$measures_reliability_cov`, `experiments/41-reliability-lambda6`) and that the `docs/research/notes/guttman_cfa_asymptotics.tex` note derives as a population functional `ω_G(Σ)`. Verified this session: on an exact one-factor population Hancock-An's omega equals the note's centroid `ω_G` to machine precision; off-model they differ (e.g. 0.712 vs 0.749 on a two-factor covariance read as one factor) because Hancock-An pool the triple ratios (ratio-of-sums) while the note averages them (average-of-ratios). Pooling is more stable when some `s_jk` are near zero, so the landed module should use, and cite, the ratio-of-sums aggregation, and Hancock-An's sim is the parity target for exp 41.

Open lane Hancock-An do not cover (and what the note generalizes to): the grouping-matrix / weighted form `ω_G(σ;w) = w'CX(X'CX)⁻¹X'C w / w'Sw` gives closed-form omega-total for multi-factor, multi-group, and weighted composites, and a second-stage centroid on the Guttman factor correlation `Φ_G` (a Schmid-Leiman general factor, k>=3) gives a closed-form omega-hierarchical; a non-proportional bifactor solution is outside the simple-structure `X`. These extensions sit alongside the existing model-based-omega and maximal-reliability speculative entries; this paper itself warrants no new build.

## Verdict
background — prior-art oracle and citation for the landed `measures::frontier::reliability` Spearman-Guttman omega; single-factor closed-form omega is solved. Pointer added to the [reliability todo entry](../../backlog/todo.md). The multi-factor / omega-hierarchical X-generalization is the open lane, captured in the note and memory `closed-form-omega-priorart`, not warranted as new build by this paper.
