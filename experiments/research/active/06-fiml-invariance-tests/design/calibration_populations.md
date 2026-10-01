# Historical invariance design

Preserved from former research/06. Only the six-indicator population is
implemented; larger published models and scalar-nesting/power work remain
unfinished. This historical draft does not establish a new null or freeze a
prospective publication design. The parent report owns the current status.

## Models and the invariance ladder

Multi-group CFA **with mean structure** is the canonical and durable testbed for
measurement invariance (Meredith 1993; Vandenberg & Lance 2000; Millsap 2011).
Mean structure is non-negotiable here for two reasons: (1) the scalar rung *is*
about intercepts, and a loadings-only design earns an automatic "what about
scalar?"; (2) MAR bias concentrates in the means/intercepts and FIML's whole
selling point over listwise is unbiased means under MAR, so the mean structure
is where the missing-data contrast actually bites.

**Population.** Following the Brace & Savalei structural precedent: a two-group,
two-factor CFA, factor correlation $\phi = 0.5$, with model size

$$p \in \{8,\ 16,\ 30\}\quad(\text{2 factors} \times \{4, 8, 15\}\ \text{indicators}),$$

where $p = 30$ is the high-dimensional cell that is **pEBA's home turf** (FMG
2026 ran to 40 observed variables; the SB corrections degrade fastest exactly
there). A one-factor six-indicator reduction (identical to exp showcases/05, research/04) is the
minimal smoke model.

**Ladder and difference tests.** configural $\to$ metric (loadings) $\to$
scalar (intercepts) $\to$ strict (residuals). Three difference tests:

| step | constrains | "is invariant at the …" |
|---|---|---|
| configural vs metric | loadings | weak / metric level |
| metric vs scalar | + intercepts | strong / scalar level |
| scalar vs strict | + residual variances | strict level |

**Type-I vs power, from one factor.** As in Brace & Savalei, the
*noninvariance-location* factor is the power machinery and the fully-invariant
population is the Type-I cell:

- **null** (fully invariant population) $\Rightarrow$ Type-I rate of each step.
- **targeted violation at a rung** $\Rightarrow$ power of that step (a planted
  loading / intercept / residual difference in group 2).

**What we add over Brace & Savalei.** They used a *single* effect size
(one $\Delta\text{RMSEA}$ per rung) and reported difference-test power without
size adjustment. We add:

1. an **effect-size gradient** (several $\Delta\text{RMSEA}$ levels including 0)
   so we get power *curves*, not points;
2. **partial invariance** as a factor (number of noninvariant indicators: 1 vs
   several), which both pre-empts the partial-invariance nag and is realistic;
3. **size-adjusted power** throughout (empirical-$\alpha$ calibration against the
   paired null cell), because tests with different Type-I are otherwise not
   comparable, exactly the conclusion exp research/04 reached.

## Estimators and missingness

**Estimators (expensive axis, refit per cell):**

- **FIML** -- `magmaanlab::magmaan_core$fit_fiml` (lavaan `missing = "ml"` analogue).
- **ML2S** -- `magmaanlab::magmaan_core$fit_ml2s`, two-stage ML (saturated EM stage
  1, structured stage 2; lavaan `missing = "two.stage"`). Cheaper per ladder
  than FIML (stage-1 saturated fit + its ACOV are shared across all four rungs)
  and Savalei & Falk's winner under incomplete non-normal data.

Both estimators feed the **same** test battery: `fmg_tests()` auto-detects ML2S
and reads the two-stage spectrum, so each fit costs one battery call.

**Missingness (expensive axis):** complete, MCAR, MAR, each at rate
$\in \{.15, .30\}$, via the Savalei-Bentler (2005) generators in
`experiments/_support` (`sb2005_mcar`, `sb2005_mar`). MAR is driven by the
always-observed marker indicators (the MAR cause). Two regimes, made explicit:

- **MCAR** -- normal-theory FIML/ML2S is consistent for any distribution, so
  this is a **clean** test of the reference-law correction. The headline.
- **MAR + non-normality** -- the Gaussian likelihood imputes with linear
  conditional means, so the statistic is non-central even under a true model and
  **no** reference-law correction can fully restore level. We store the base
  statistic and the UGamma trace so the report can attribute MAR over-rejection
  to **estimator bias**, not the test (the diagnostic exp research/04 introduced). MAR is
  the honest stress test, not the headline.

## Non-normality

Reuse the experiment-showcases/03 / FMG non-normal families rather than reinventing them:
`norm`, `vm1/2` (Vale-Maurelli), `ig1/2` (independent generator), `pl1/2`
(piecewise-linear, Foldnes & Gronneberg 2022), where `*1` is skew 2 / kurtosis 7
and `*2` is skew 3 / kurtosis 21. VM is a Gaussian copula, which is fine here
(continuous indicators); IG/PL bite harder than the copula when we want them to.

## The cost asymmetry, and the grid that exploits it

The key structural fact: **once the UGamma spectrum is formed, every p-value is
nearly free.** So the design splits into a free axis and an expensive factorial.

**Free axis (harvested per replication, NOT grid cells).** Every statistic is
computed from the one eigen-spectrum of each fit, for both `h1_information =`
`"saturated"` and `"structured"`:

```
GOF / per-rung:  naive, MLR (YB_mplus), YB_exact/SB, SS, SF,
                 EBA2/4/6, pEBA2/4/6, pall, pOLS, all
difference:      naive, SB, mean-var-adjusted, mixture, + pEBA-on-difference
```

"Lots of p-values" costs *columns*, not cells.

**Expensive factorial (each level needs a refit + a new spectrum):**

| axis | levels (starting point) | rationale |
|---|---|---|
| estimator | FIML, ML2S | Savalei-Falk; ML2S cheaper per ladder |
| missingness | complete, MCAR-.15/.30, MAR-.15/.30 | the new contribution |
| non-normality | norm, vm2, ig2, pl2 (+vm1 …) | reuse exp-showcases/03 families |
| total $N$ | 100, 200, 500, 1000 | push small; SB dies there |
| group ratio | 1:1, 1:3 | unevenness breaks SB-1 (B&S) |
| model size $p$ | 8, 16, 30 | $p=30$ is pEBA's home turf |
| invariance condition | null + {weak, strong, strict} $\times$ {small, med, large $\Delta\text{RMSEA}$} | Type-I (null) + power curves |

Full crossing $\approx 2 \times 5 \times 4 \times 4 \times 2 \times 3 \times 10
\approx 9{,}600$ cells. At the 5000-rep house standard (and **no Monte Carlo
error reported**, per house convention) compute is *not* the binding constraint
on a 100-CPU SLURM allocation; **presentability is**.

## Output schema (decide before the run)

One **long-format** row per `(cell, estimator, rung, outcome, method,
h1_information)`; per-replicate spectra to an `.rds` so any later p-value
transform is recomputable without refitting.

`results/fits.csv` columns:

```
# design keys
p, n_total, ratio, dist, mech, rate, realized_rate, invariance_cond,
delta_rmsea, n_noninvariant, estimator, rung, outcome (gof|nested),
# result
method, h1_information, p_value, base_stat, df, trace, scaling_factor, truth (h0|power)
```

Plus `spectra.rds` (per-rep eigenvalues; sufficient statistic),
`summary_rejection.csv` (rejection rate by design $\times$ method),
`power_adjusted.csv` (size-adjusted power vs paired null), `cells.csv`,
`metadata.csv`.
