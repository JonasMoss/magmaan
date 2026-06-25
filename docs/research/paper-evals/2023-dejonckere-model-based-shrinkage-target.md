# A model-based shrinkage target to avoid nonconvergence in small sample SEM

**Cite.** De Jonckere, J., & Rosseel, Y. (2023). Manuscript / Ghent University dissertation chapter (no journal header on the PDF; distinct from the 2022 *SEM* bounded-estimation paper). DOI: n/a. Code: osf.io/8qp3f.
**PDF.** `external/refs/De Jonckere and Rosseel 2023 - A model-based shrinkage target to avoid non-convergence in small sample SEM.pdf`
**Read.** 2026-06-25  ·  **Verdict.** → speculative

## TL;DR
Covariance-shrinkage preprocessing to cure small-N nonconvergence: blend the
sample covariance with a structured target, `S_a = (1-λ)S + λT`, then run vanilla
ML on `S_a`. The one novelty over generic shrinkage is a *model-based* target
built from the SEM itself. The general device is already in magmaan
(`data::frontier` shrinkage + `fit_ml_ridge_continuation`); the model-based
target is a weak addition (it biases the GOF test toward the fitted model), and
the principled thing the paper points away from, parameter-space penalized ML, is
the genuinely unbuilt idea.

## Contribution
- **The device (not new).** Ledoit-Wolf covariance shrinkage as pre-estimation
  conditioning: `S_a = (1-λ)S + λT`, fed to an unchanged ML estimator. Prior SEM
  work shrank only the diagonal (ridge SEM, `S + αI`); the pitch is to also bend
  the off-diagonals.
- **The novelty.** A model-based target `T_SEM = D(Λ*Ψ*Λ*' + Θ*)D`: express the
  model as an *idealized standardized CFA* with all loadings fixed at 0.7,
  indicator reliability 0.8, diagonal Θ*, and one of four choices for the
  latent-correlation block Ψ* (user-supplied correlations / a single common
  correlation / zero / leave the inter-factor block of S untouched), then rescale
  to S's diagonal. So the target injects the model's own structure into the data
  before estimation.
- **λ selection.** Study 1: the smallest λ on a 51-point grid that yields
  convergence (the goal is convergence, not best fit, so CV is rejected as
  wasteful at small N). Study 2: the Ledoit-Wolf δ/N loss-minimizer (eq. 3).
- **Evidence.** Two sims (P ∈ {6,14,18}, β ∈ {0,.1,.25,.3,1,2}, N ∈ {10..100},
  1000 reps, normal complete data, correct models). Model-based targets (MB1/MB2)
  give zero/near-zero nonconvergence and the lowest bias and MSE at small N, small
  β, small P; the edge vanishes by N=100, and the constant-correlation target with
  the Ledoit-Wolf λ matches it. Honestly reported: MSE is sometimes *worse* than
  ML, and the model-based target's real claim is "needs less adjustment (smaller
  λ)," not uniform dominance.

## Relevance to magmaan
**The method is already in magmaan; only the model-based target is unbuilt.**
`data::frontier` covariance shrinkage is exactly `S_a = (1-λ)S + λT` for both
continuous `SampleStats` and mixed `MixedOrdinalStats` (it propagates the moment
transform through `NACOV` and rebuilds DWLS/WLS weights).
`estimate::frontier::fit_ml_ridge_continuation()` is the convergence-targeted,
warm-started version that continues `S_alpha = (1-α)S + α T(S)` down to α=0, and
`experiments/04-near-singular-ml-continuation` already grids diagonal /
scaled-identity / raw-identity targets over λ sequences. So this paper validates a
path magmaan has, with simpler targets. The only addition it offers is the
idealized-CFA `T_SEM`, which would slot into that existing target menu.

**But the model-based target is a weak addition, for the reason we want on
record.** Shrinking S toward a *model-shaped* target before testing that model
biases the goodness-of-fit downward; using idealized loadings (0.7) rather than
*fitted* ones softens the circularity but does not remove it, and the χ² computed
on `S_a` no longer references the true sampling law of S. The paper's selling
point ("keeps every downstream estimator, SE, and fit index intact") is partly
illusory: it does not remove the non-standardness, it *moves* it from θ-space into
S-space and hides it inside output that looks like ordinary ML. If `T_SEM` is
added at all it belongs in experiment 04's target grid with this GOF-bias caveat
documented, not in a core path.

**The actionable idea is the alternative the paper points away from:
parameter-space penalized ML, which magmaan does not have.** Same goal (cure
nonconvergence / inject structure), cleaner statistics, because you penalize θ and
leave S (hence the reference law of the fit test) untouched:

- **Variance log-barrier**, `P_ρ(θ) = -ρ Σ_i log ψ_i` (equivalently `ρ Σ_i
  ψ_i⁻¹`). This is an inverse-gamma prior on the residual variances in MAP terms,
  and the smooth sibling of De Jonckere & Rosseel (2022) bounded estimation, which
  hard-clips the same variances. It is the *on-target* convergence fix, since the
  small-N disease is overwhelmingly Heywood cases (ψ_i → 0⁻), and it never touches
  the fit statistic.
- **Structured Gaussian penalty**, `P_ρ(θ) = ρ(θ-θ⁰)'A(θ-θ⁰)` toward the same
  idealized standardized model. This is the *honest* analog of the model-based
  target: it injects identical structure transparently, in parameter space,
  without contaminating S, so the GOF stays valid. (Their `S_a` is approximately
  this penalty toward θ⁰ but measured in the ML/KL-divergence geometry, with the
  unwanted side effect of bending the fit statistic.)
- **L1 / regsem** is named for completeness but solves a different problem
  (sparsity/selection), not nonconvergence.

Penalization is not free inference either, but the cost is *writable and
localized*: a penalized sandwich `Cov(θ̂) = (H + P_ρ'')⁻¹ V (H + P_ρ'')⁻¹` (penalty
curvature enters the bread) plus an effective-df correction for the fit test. The
ρ selector and the non-standard inference are research-tier, the same
[[feedback-shortcut-variants]] hidden-validation pattern as bounded estimation's
own bias-variance knob.

## Verdict
→ speculative — the paper's own covariance trick is **background** (already
covered by `data::frontier` shrinkage + `fit_ml_ridge_continuation` + experiment
04). The discussion it provokes graduates: the parameter-space penalized-ML
alternative is recorded in
[backlog/speculative.md](../../backlog/speculative.md) ("Penalized ML for
small-sample convergence and structure"). Build trigger: a small-N paper row, or a
case where covariance conditioning is insufficient / an honest (non-circular)
structural-shrinkage estimator with valid GOF is wanted, most naturally a
moment-space-vs-parameter-space study extending experiment 04.
