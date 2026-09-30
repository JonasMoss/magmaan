# Iteration 01: projected residual flips for GOF

## Construction

Let `s` be the sample covariance vector, `sigma(theta_hat)` the fitted model
covariance vector, and `z_ci` the casewise covariance contribution centred at
`s`. For the expected-information U-factor, magmaan stores a residual basis
`B` satisfying

```text
B' Gamma_NT(sigma_hat) B = I
B' Delta(theta_hat) = 0.
```

The casewise residual estimating functions are

```text
psi_i = B' { z_ci + s - sigma(theta_hat) }.
```

Their sum is `n B' (s - sigma_hat)`. Therefore

```text
T_effective = || sum_i psi_i ||^2 / n
```

is the structured normal-theory residual quadratic. The implementation checks
against magmaan's independent RLS GOF route on every fitted replication. This
is the key bridge: GOF is treated as testing the df-dimensional complement of
the fitted model tangent inside the saturated moment space.

For signs `g_i in {-1,+1}`, the effective reference statistic replaces the
observed sum with `sum_i g_i psi_i`. The identity sign vector is included in
the Monte Carlo rank. Exactness still requires finite-sample sign symmetry;
under the weaker score-flip conditions the claim is asymptotic, not magical.

## What “standardized” means here

The empirical projected meat is

```text
V_hat = sum_i psi_i psi_i'.
```

The standardized statistic is

```text
T_standardized(g) = (sum_i g_i psi_i)' V_hat^-1 (sum_i g_i psi_i).
```

This is literally a whitening operation on the already effective residual
scores. Because sign flips do not change outer products, `V_hat` is fixed over
the randomization distribution. This differs from the nested-score method,
where flipping destroys sample nuisance orthogonality and the conditional
variance is recomputed from sign sums. Here the nuisance/model tangent has
already been removed by `B`; the extra standardization addresses empirical
anisotropy of the residual contributions.

That distinction prevents a misleading reuse of the word “standardized.” The
GOF arm is closer to direct sandwich studentization after efficient-score
projection than to the De Santis et al. flip-specific nuisance correction.

## Predictions

1. Under normality with `n` comfortably above df, the model metric should be
   close enough to the empirical metric that standardization rarely changes a
   decision.
2. Under severe nonnormality, empirical residual eigenvalues should spread and
   standardization may improve calibration if the covariance estimate remains
   stable.
3. The benefit cannot be monotone in dimension. At `df >= n`, the empirical
   meat is singular; near that boundary it can be extremely ill-conditioned.
   A ridge would introduce precisely the tuning decision we are trying not to
   hide.
4. Effective flipping may therefore be the more genuinely non-ad-hoc default:
   it uses the fitted model's normal-theory geometry but obtains the tail by
   randomization. Standardization is a conditional diagnostic unless a stable
   empirical metric is available.
5. A single omitted residual covariance should lose power as the residual
   space grows from df 5 to 170. Nominal power must be paired with achieved-null
   calibration; a raw rejection-rate advantage is not enough.

## Scope boundary

This experiment supports one complete covariance block. Means require adding
model-centred first-moment contributions; grouped models require checking the
sample-size weights in the pooled residual basis; FIML needs patternwise
saturated influence contributions. None is a mechanical toggle. If this probe
is promising, the next implementation step is a core
`inference::frontier`/`api::frontier` result with those boundaries explicit.

## First probe outcome (2026-07-13)

The local run used 16 cells, 100 replications per cell, 199 random flips, and
four DGP workers. All 1,600 fits, residual flips, and 12-test FMG batteries
succeeded in 36.4 wall-clock seconds. The projected all-plus statistic matched
the independent RLS statistic within `1.2e-7` over the whole run.

The result is a qualified go for the construction and a no for immediate broad
promotion:

- at df 5, effective null rejection was .01/.03 for normal n=100/400 and
  .08/.03 for PL;
- at df 170, effective null rejection was .12/.13 for normal n=100/400 and
  .33/.08 for PL;
- empirical standardization was structurally unavailable at df 170, n=100;
  at n=400 it changed many decisions but worsened rejection to .23 under
  normality and .11 under PL;
- at df 170, n=400, median projected-meat condition numbers were about 37 under
  normality and 42,000 under PL;
- the high-rank comparator field also split badly: scalar ML/SB references were
  liberal, MV/SS/ALL could be extremely conservative under PL, and pEBA4-RLS
  occupied a middle ground rather than solving every cell.

At p=20,n=400, median per-replication time was about 55 ms: 5 ms for effective
flip resampling, 7 ms for the empirical eigendecomposition, 2 ms for applying
standardization, and 33 ms for all 12 FMG tests. The p=20 PL calibrations cost
roughly 19--23 seconds once per null/power DGP and were shared across sample
sizes. Standardization is thus computationally cheap in this formulation; its
rank/conditioning behavior, not timing, fails the first gate.

The next design should concentrate on low/moderate residual df with a larger
null budget and an explicit n/df ladder. Do not add an empirical ridge merely
to fill the high-rank cells: that would reintroduce a tuning decision and needs
its own calibration/power comparison.
