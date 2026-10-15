# Lane psd-factor: criteria (draft 2026-10-15, written before any run)

**Status.** Approved by the author on 2026-10-15 ("sure lets go"). The production run waits for simbox and a go from the author; the smoke run may start now.

**Decision.** Whether the default covariance factor of every PSD route
(complete-data ML, FIML, ML2S, the least-squares family and ordinal/mixed
DWLS) becomes the full square factor `PsdFactor::Square` instead of the lower
triangular `PsdFactor::Triangular`.

**Why.** With a square factor the local minima of the lifted problem are
exactly those of the PSD problem (Levin, Kileel and Boumal 2025; Ding and
Wright 2025). The triangular factor loses this at singular blocks of two or
more variables, as the covariance-honest paper's appendix example shows. This
is a non-inferiority study: the square factor is adopted for its guarantee
unless it loses in practice.

**Disclosure.** Development evidence, never gating: TASK-33.15 fitted 2,000
draws of the weak-marker three-factor CFA (p = 18, marker loadings .5, other
loadings .3, N = 100) with both factors under the library's SLSQP route.
Triangular and square were usable in 1,919 and 1,923 draws; both reached the
best known objective in 1,927; the square factor was lower in 17 draws and the
triangular in 15; median times were 34 and 36 ms. R probes with bounded nlminb
gave the same picture.

## Configurations

Both arms use the same route and differ only in the factor: transported FABIN3
start, SLSQP, diagonal preconditioning where the route has it, the library's
controls and tolerances.

| Arm | Factor |
|---|---|
| `triangular` | library default, no options (baseline) |
| `square` | `psd_factor = "square"` |

Both arms go through the magmaanlab PSD wrappers. The report also gives, per
family, the number of draws where the `triangular` arm differs from
`magmaan(covariance = "psd")` by more than $10^{-4}$ in $T$.

## Problems

Test families, ML:

- **F1.** The test populations of `R/families.R` at their two sample sizes,
  100 replications each, as in the ml-gls lane. Ernst stays a control.
- **F2. Near-collinear factors.** A three-factor CFA, six indicators per
  factor, loadings .7, marker identification. Factor correlations
  $(\rho_{12}, \rho_{13}, \rho_{23})$ are (.95, .5, .5), (.99, .5, .5) and
  (1, .5, .5), the last one singular. Also (.99, .99, .99), which is close to
  rank one, with two near-zero eigenvalues. N = 50 and 100, 100 replications
  each.
- **F3. Singular residual block.** A one-factor model with eight indicators,
  loadings .7, and a correlated-residual component on x1 to x3 with population
  block $vv^\top + ww^\top$, where $v = (.6, .6, .3)$ and $w = (0, .3, .6)$, a
  rank-two block. The other residual variances are .51. N = 100 and 200, 100
  replications each.
- **F4. Collapsed-column starts.** Every F2 draw, refitted from the library
  start with factor 1's variance pinned to $10^{-6}$ and its covariances to
  0, the position where the triangular factor's first column is dead. Both arms
  start there.

Test families for other estimators, on F2 and F3: FIML with 15% MCAR, ULS,
and ordinal DWLS (five categories cut at fixed population thresholds),
100 replications per N.

Controls, reported only: Ernst, and the weak-marker three-factor CFA of the
disclosure at N = 100.

**Draws.** Seed base 2026101501, never used before; seeds are cell-stable.
The smoke run uses two replications per cell at seed base 2026101502 and does
not count.

## Judge

- A fit succeeds when the library's `fit$converged` holds.
- The best known objective is the lowest successful objective among the arms
  for that draw, reached within $10^{-6}(1 + \lvert f \rvert)$.
- Paired outcomes per draw: a win for an arm when it succeeds and the other
  does not, or when both succeed and only it reaches the best known objective.

## Rules

They apply per test family and estimator, never pooled.

1. **Adopt** `Square` as the default of every PSD route if, in every test
   family and estimator, `square` has no more losses than wins in successes
   and in best known, and its total seconds are at most 1.25 times those of
   `triangular`. Ties favour `square`, because of its guarantee.
2. **Otherwise** nothing changes. The report lists every loss and attributes
   it to a mechanism (start, rotation drift, link feasibility, budget). Then
   the author decides.
3. **Regression test.** Whatever the outcome, the draw with the largest
   objective gap between the arms in F2 to F4 becomes a C++ regression test
   (follow-up card).

## Outputs

Per-run summary CSVs and `metadata.csv` under `results/psd-factor/<run-id>/`;
raw per-fit rows stay ignored. The report adds this lane to the default
register. If adopted: switch the C++ default, update the capability area file
and the roadmap, and give the covariance-honest paper's §2.2 the square-factor
explanation (loadings on independent sources, identified up to rotation like
exploratory loadings).
