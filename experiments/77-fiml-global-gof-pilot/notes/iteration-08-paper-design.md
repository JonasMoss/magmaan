# Iteration 08: sourceable paper design

Date: 2026-08-21

## Purpose

The pilot models were chosen to exercise the implementation. A publication
panel needs a different standard: the five models should cover recognizably
different applied SEM uses, remain small enough for large FIML simulations, and
have auditable provenance. A paper that merely cites literature near arbitrary
population parameters would not meet that standard.

The shortlist below is provisional. Model selection and parameter provenance
must be frozen before the source-based populations replace the pilot code and
before power is retuned.

## Recommended five-model panel

| Model | Size | Expected df | Applied role | Proposed population provenance | Status |
|---|---:|---:|---|---|---|
| One-factor CFA | 6 indicators | 9 | A small unidimensional measurement model | Exact Wolf et al. six-indicator condition: standardized loadings `.65`, factor variance 1, indicator residual variances `.5775` | Strong source match |
| Correlated three-factor CFA | 15 indicators | 87 | A larger multidimensional measurement model with heterogeneous quality | Deterministic reduction of Savalei--Falk Model 2: first five listed indicators from each seven-indicator factor, preserving their reported standardized loadings and factor correlations | Source-based adaptation; must freeze reduction |
| Orthogonal bifactor | 12 indicators, 3 specifics | 42 | A common but numerically less forgiving multidimensional measurement model | Ximénez et al.: general loadings `.60`, three four-indicator group factors, and a `.50` group-loading condition | Strong source match; zero-cross-loading null is an adaptation |
| Latent mediation SEM | 9 indicators, 3 factors | 24 | A structural model with direct and indirect latent paths | Exact Wolf et al. medium condition: three indicators/factor, loadings `.65`, and all three structural paths `.40` | Strong source match |
| Linear latent growth | 5 waves | 10 | A longitudinal mean-and-covariance model | Five-wave linear form grounded in Graham et al.; parameterization follows the standard intercept loadings `1` and slope loadings `0:4` | Model form sourced; numeric population still unresolved |

This panel deliberately removes the pilot's two-factor CFA. A one-factor CFA,
a correlated three-factor CFA, and a bifactor model already span simple,
multifactor, and general-plus-specific measurement structures. The freed slot
adds an actual structural SEM.

### 1. Six-indicator one-factor CFA

Wolf et al. studied one-factor models with four, six, or eight indicators and
standardized loadings of `.50`, `.65`, or `.80`. Selecting their six-indicator,
`.65` condition gives a compact, exactly published DGP:

- factor variance `1` and mean `0`;
- six equal standardized loadings `.65`;
- six residual variances `1 - .65^2 = .5775`; and
- zero indicator intercepts.

This should replace the pilot's heterogeneous, unstandardized loadings. The
model is intentionally easy; heterogeneity is supplied by the larger CFA.

Source: Wolf, Harrington, Clark, and Miller (2013),
<https://doi.org/10.1177/0013164413495237>.

### 2. Fifteen-indicator correlated three-factor CFA

Savalei and Falk's Model 2 has seven indicators per factor and uses
standardized loadings estimated from real empathy data. Its factor correlations
are `.446`, `-.124`, and `-.085`. The full 21-indicator model is larger than the
desired FIML budget, so the proposed reduction takes the first five indicators
listed for each factor in their Appendix A, without selecting on pilot results:

| Factor | Source indicators | Standardized loadings |
|---|---|---|
| F1 | V1--V5 | `.743, .252, .604, .540, .201` |
| F2 | V8--V12 | `.536, .528, .692, .694, .509` |
| F3 | V15--V19 | `.426, .530, .637, .755, .564` |

Observed variances remain one, so each residual variance is `1 - lambda^2`.
The final implementation must preserve the source ordering of the three factor
correlations and archive the resulting covariance matrix and df. This is a
transparent 15-variable adaptation, not an exact reproduction of their Model 2.
The proposed factor-correlation matrix is positive definite (smallest
eigenvalue `.552`), and the resulting adapted observed covariance has smallest
eigenvalue `.496`; this is a population sanity check, not yet a fit gate.

Source: Savalei and Falk (2014),
<https://doi.org/10.1080/10705511.2014.882692>.

### 3. Twelve-indicator orthogonal bifactor model

Ximénez, Revuelta, and Castañeda used a 12-item bifactor population with a
general loading of `.60` on every item and three group factors with four items
each. Their group-loading conditions ranged from `.15` to `.60`; `.50` gives a
nontrivial but estimable publication baseline. With orthogonal unit-variance
factors and no cross-loadings, the standardized residual variance is
`1 - .60^2 - .50^2 = .39`.

Their generating models also varied three cross-loadings from `.05` to `.40`.
For our null-calibration cells, those cross-loadings must be set to zero because
the fitted simple bifactor model omits them. This is an explicit adaptation.
One or more of the published cross-loading magnitudes is then a particularly
well-motivated power alternative.

Source: Ximénez, Revuelta, and Castañeda (2022),
<https://doi.org/10.3389/fpsyg.2022.923877>.

### 4. Nine-indicator latent mediation SEM

Wolf et al.'s latent mediation model has three factors, three indicators per
factor, indicator loadings `.65`, and equal direct paths. Their medium condition
sets all three paths to `.40`, yielding about 45% explained variance in the
latent outcome. In structural notation,

`M ~ .40 X` and `Y ~ .40 X + .40 M`.

The exogenous factor has variance one; latent and indicator residual variances
are chosen as in the published standardized model so all latent and observed
variances are one. The latent structural portion is saturated, but the
single-loading measurement blocks remain overidentified. It therefore adds a
real structural path model without becoming large.

Source: Wolf et al. (2013),
<https://doi.org/10.1177/0013164413495237>.

The 11-variable Bollen political-democracy model is the named alternative if a
classic substantive SEM is preferred. Its exact path specification is already
tracked in `benchmarks/cases/bollen_democracy_sem/` and comes from the lavaan
SEM tutorial, <https://lavaan.ugent.be/tutorial/sem.html>. It should replace,
not supplement, the mediation model so the headline panel remains five models.

### 5. Five-wave linear latent-growth model

The desired compact form is

`y_t = i + t s + e_t`, for `t = 0, 1, 2, 3, 4`,

with freely estimated intercept and slope means, their two variances and
covariance, and wave-specific residual variances. Graham, Taylor, and Cumsille
used a univariate linear latent-growth model with five measurement occasions;
the lavaan growth tutorial provides the same standard SEM parameterization.

Unlike the first four candidates, the pilot's numeric growth means and
variances are not yet tied to a published population. They must not silently
survive into the paper. The final choice should be either an exact published
five-wave simulation population or a reproducible population fit from a named
public growth data example, with the fitted parameter table archived.

Sources: Graham, Taylor, and Cumsille (2001),
<https://doi.org/10.1037/10409-011>, and the lavaan growth tutorial,
<https://lavaan.ugent.be/tutorial/growth.html>.

## Missingness sources and the adverse MAR arm

Savalei and Falk provide the best primary template because they cross four
mechanisms with both few and many missing-pattern designs:

- MCAR;
- MAR-L: deletion when the always-observed conditioning variable is above
  zero;
- MAR-L2: deletion when it is below zero; and
- MAR-NL: deletion when its absolute value exceeds `.54`.

They target 15% or 30% missingness on selected variable blocks and explicitly
construct at most four versus 64 population patterns. Their exact block table
cannot be copied mechanically to every model, but its organizing principle can
be adapted before results are seen.

The adverse arm should be a diagnostic for a deeper issue, not simply a more
extreme logistic slope. Yuan and Bentler show in a bivariate design how
Gaussian likelihood under nonnormal MAR can target biased mean/covariance
quantities when the second variable is deleted according to the always-observed
first variable. Savalei explains why complete-data robustness conditions do not
generally carry over across missingness patterns. Our version should combine an
always-observed skewed tail, strong association with variables made missing,
and a threshold/tail rule, then compute the Gaussian-FIML pseudo-true target
before simulation. Such cells are estimand stress, not ordinary null-size
cells.

Sources: Savalei (2008),
<https://doi.org/10.1080/10705510701758091>, and Yuan and Bentler (2000),
<https://doi.org/10.1111/0081-1750.00078>.

## Empirical examples

The desired empirical pattern is MLR rejecting while projected-score pEBA(4)
does not. Finding a dataset with that pattern is not enough. The candidate
corpus, public-data constraints, model source, and screening rule must be
declared first, and the full screen must be reported. A naturally incomplete
dataset is preferable. Imposing a source-based MAR mechanism on a complete
public dataset remains useful, but it is a semi-empirical illustration and
should be labeled as such.

For each selected example, save enough diagnostics to explain the disagreement
rather than merely exhibit two p-values: correction moments/eigenvalues,
pattern counts, influential score contributions, convergence, and comparison
with score SB and the viable LR correction.

## Immediate decision

Do not yet edit `R/sem_models.R`. First decide whether the Savalei--Falk
15-indicator reduction and the Wolf mediation model are the final choices, and
resolve the growth population. Then replace all five populations in one
coherent change and rerun only a small source-model gate before retuning power.
