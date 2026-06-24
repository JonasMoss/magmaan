# Robust mixed moments: recipe taxonomy and the mixed-model design question

Status: **design note, deferred.** Captures a 2026-06-24 design discussion so we
can resume without re-deriving it. No code decision was taken; the mixed-robust
missing-data wiring ("flavor 1") is parked pending the fork below.

## The question that started it

We wanted to wire a *robust polyserial* estimator for missing data (pairwise
overlap), with WMA hard cap as the intended lead. Recon surfaced a category
error: **"WMA polyserial" does not exist.** WMA is a *polychoric*
(ordinal-ordinal) mechanism. So the real question became: is WMA a special case
of a general recipe that *does* reach polyserials, and if so, what is it?

## WMA is a minimum-disparity (RAF) estimator

From `h-polychorics.tex`: the h-weighted polychoric solves

    Psi_h(theta) = sum_j  p_j(theta) * h(t_j) * s_j(theta) = 0,
    t_j = fhat_j / p_j   (cell overcount ratio = 1 + Pearson residual delta_j),
    s_j = grad log p_j.

This is Lindsay's residual-adjustment-function / phi-divergence framework on the
contingency-table cells. Members differ only in the **shape of the weight h**:

- ML:            h(t) = t            (linear, unbounded -> not robust)
- WMA hard cap:  h(t) = min(t, k)    (one-sided cap; bounds over-counted cells)
- smooth/exp cap: smooth bounded h

`eval_huber_residual_clip` is the *same idea in psi-form*: HardHuber clips |r| at
k, PseudoHuber smooths, Tukey redescends to 0 — bounded-influence psi-functions
on a residual r. So **WMA and Huber/Tukey are siblings**: both bound the
influence of a Pearson-type residual; they differ in (i) the RAF/psi shape
(WMA = one-sided cap on the ratio; Huber = two-sided clip; Tukey = redescending)
and (ii) *which residual* is bounded.

## The general recipe ports; the residual is what's table-specific

General recipe: **bound the influence of a Pearson-type discrepancy in the
estimating equation**, `sum w(z;theta) s(z;theta) = 0` with bounded weight `w`.
What is table-specific is only the residual the bound is applied to. WMA's
residual is a *cell overcount* `fhat/p`, which needs a contingency table. A
polyserial pair has a continuous margin -> no cells. Two escapes, both already
implemented:

| recipe              | residual it bounds                          | needs bins? | shape          | covers                 |
|---------------------|---------------------------------------------|-------------|----------------|------------------------|
| WMA hard cap        | cell overcount fhat/p (ord-ord)             | yes         | one-sided cap  | polychoric only        |
| Huber/Tukey-residual| per-case **conditional** Pearson residual   | no          | two-sided clip / redescend | polychoric AND polyserial |
| DPD                 | downweight by likelihood^alpha (density power) | no       | smooth power   | polychoric AND polyserial |

So **Huber/Tukey-residual is the genuine spiritual port of WMA** to the
continuous-conditional case (same "clip the Pearson residual" logic, per-case
instead of per-cell). **DPD** is the density-divergence cousin — a different
umbrella (phi-divergence vs density-power divergence; they overlap only at
KL/ML), and the one member needing neither binning nor kernel smoothing, which
is why it is the implemented polyserial robustifier. Binning u to force WMA onto
a polyserial was considered and rejected (perturbs the estimate, breaks lavaan
parity).

## The mixed-model fork (the actual open decision)

For a mixed model the robust estimator must cover ord-ord (polychoric), cont-ord
(polyserial), cont-cont (Pearson), and means/vars. The options:

1. **Uniform DPD.** DPD applies to *every* pair type as one density divergence
   (all-ordinal DPD exists via `SharedRobustOrdinalKind::Dpd`; polyserial DPD
   exists). This is the one **non-mixing**, aesthetically uniform recipe — but
   our limited pilot had WMA hard cap doing better, and the user finds DPD less
   promising empirically.
2. **Intrinsic mix.** WMA (or Huber/Tukey) on the cell residual for polychorics
   + Huber/Tukey on the conditional residual for polyserials. One *philosophy*
   (bound the Pearson-residual influence) but two residual definitions ->
   feels like mixing estimators.
3. **All-ordinal-only robust paper.** WMA is clean, uniform, and the pilot
   winner *when every pair is a polychoric*. Keep the WMA robust paper
   all-ordinal (where there is no mixing question at all) and treat the
   mixed/polyserial robust extension as genuinely separate future work.
   ("Hard cap is more than enough for its own paper anyway.")

**Aesthetic stance (user, 2026-06-24):** mixing estimators across pair types is
"probably not very beautiful." That pushes away from option 2 and toward either
a single uniform recipe (option 1, DPD — at a possible performance cost) or
scoping the robust paper to the case where the question doesn't arise (option 3,
all-ordinal). The beauty-vs-performance tension (uniform DPD vs better-but-
non-portable WMA) is the crux to resolve when we resume.

WMA also has a non-aesthetic edge worth recording: its objective phi is
closed-form (`t*(log k + 1) - k`), whereas smooth/exp caps carry Gauss-Legendre
quadrature in `phi_from_h`; DPD/Huber are different mechanisms again.

## Missing data is orthogonal to all of the above

The missing-data layer is independent of the robust-recipe choice: it is the
overlap reweighting `Gamma^pw = Pi_div (x) Gamma`, `(Pi_div)_ab =
pi_ab/(pi_a pi_b)`, from `ordinal_pd_gamma.tex` ("pd" = pairwise deletion, not
density power). It is already implemented for the ML observed builders
(`mixed_ordinal_stats_from_observed_data`, `ordinal_stats_from_observed_integer_data`),
which carry support-aware moment-influence and (for mixed) the
`gamma_*_influence` rows. So once a robust recipe is chosen, the missing-data
wiring is the same mirror we did for the ML path; the recipe choice is the only
blocker.

## Evidence update (2026-06-24)

The copula distributional-misspecification stress test now has results:
[robust_ordinal_copula_results.md](robust_ordinal_copula_results.md) (full
8-estimator family, faithful Welz §8.2 replica, 5000 reps). They bear directly
on this fork: the **Huber/Tukey residual-clip recipe** (called "the genuine
spiritual port of WMA" above, option 2) is the **worst performer** on the
copula stress and Tukey's SEs are unusable; **DPD** (option 1, uniform) is the
only cross-copula-stable robust recipe with calibrated SEs. This pushes a
*mixed* extension toward DPD and away from intrinsic-mix Huber/Tukey. Caveat:
that test is distributional, not partial-contamination (WMA's headline case);
cross-check against the contamination designs before treating it as final.

## Resume checklist

- Decide the fork: uniform-DPD vs intrinsic-mix vs all-ordinal-only-paper.
  Copula evidence (above) now favours uniform-DPD for the mixed case.
- If a mixed robust builder is built, the natural shape is a unified
  `mixed_*_shared_robust_*` taking the recipe as an option (no such builder
  exists today; DPD and Huber are separate functions with separate result
  structs, and there is no mixed WMA path at all).
- Whatever recipe wins, flavor 1 = give it the observed/overlap missing-data
  treatment of `mixed_ordinal_stats_from_observed_data` (NaN support, per-pair
  overlap, support-aware Gamma influence rows), then flavor 2 = the hybrid
  FIML-normal continuous block.
