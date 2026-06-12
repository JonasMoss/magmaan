# Speculative backlog

Items we may never need but want to keep findable. Each entry names the gap,
the cheaper alternative that already covers the practical case, and the
specific condition under which we'd actually build the item. Unlike
[`todo.md`](todo.md), nothing here is committed work — promote to `todo.md`
only when a concrete downstream consumer (paper row, user request, parity
failure) appears.

## Estimation / inference

### `spectral_truncate` weight policy for degenerate ADF/WLS Γ̂

An optional non-default pseudo-inverse weight policy for degenerate saturated
continuous ADF/WLS Γ̂: spectral truncation on the retained subspace, returning
dropped rank, retained weighted residual, and conditioned projected-gradient
norm, as a parity-restoring alternative to the current rank-deficiency
refusal.

**Alternative already available.** The explicit refusal
(`detail::symmetric_inverse_pd_gated`, returning dim/rank/rcond/λmin) plus the
advisory `conditioned_adf_weight()` telemetry in `experiments/00-lavaan-parity`.
Convention documented in
[docs/design/numerical-conventions.md](../design/numerical-conventions.md).

**Build if.** A real parity or methods case needs lavaan-matching estimates on
a degenerate Γ̂ instead of an explicit refusal. Research-tier: a pseudo-inverse
estimator's sampling behavior is its own validation problem, not a free
parity patch.

### `statistic = "N" | "N-1"` selector for the GLS/WLS test multiplier

A user-facing selector (or report-both mode) for the GLS/WLS chi-square
multiplier convention.

**Alternative already available.** The convention is fixed and documented in
[docs/design/numerical-conventions.md](../design/numerical-conventions.md);
parity tests pin `magmaan·(N−G)/N == lavaan` at 5e-3.

**Build if.** A concrete methods workflow needs the other convention or a
side-by-side report.

### Reduced-Gamma ordinal robust-inference products

Robust ordinal/mixed reporting that consumes reduced Gamma products
(`OrdinalGammaCacheBlock` influence factor with Γ = IF'IF/n,
`OrdinalGammaMaterialization::Reduced` plan rules, and a
`robust_ordinal`/`robust_mixed_ordinal` path through `robust::build_u_factor`
plus `reduced_gamma_sample_from_gamma`) instead of materializing the dense
m×m Gamma per block for ULSMV/DWLS scaled tests.

**Alternative already available.** The cache-aware robust paths reuse an
attached full Gamma (fit-plus-inference plans materialize it once), the lazy
workspaces defer the WLS inverse, and the influence factor is already
computed inside the stats builders — only the cache plumbing and the reduced
sandwich route are missing. `Materialization::Reduced` exists in the enum,
unconsumed.

**Build if.** An ordinal/mixed model is large enough that the m×m Gamma
materialization dominates robust reporting in a benchmark or paper grid
(moment_dim ≳ 1000; experiment 13 measured full Gamma at ~100-200× the
diagonal's memory), or the ordinal-snlls paper's fit-plus-inference rows
need the reduced route to make their cost story complete.

### Ordinal EBM factor scores

A factor-score path for ordinal / mixed-ordinal fits that matches lavaan's
`lavPredict()`: empirical-Bayes-modal (and ML) scoring by numerical integration
over the latent-response distribution implied by the thresholds and the fitted
Λ/Θ/Ψ, rather than the continuous regression/Bartlett predictor.

**Alternative already available.** `api::factor_scores` is exposed for
continuous fits; ordinal fits are explicitly guarded (`require_not_ordinal`,
asserted in `r-package/examples/ordinal_dwls_wls.R`). The neighbouring
parameterization-agnostic transforms — `standardize_lv`/`standardize_all` and
`compute_defined` — are exposed for ordinal fits because they are exact
functions of the fitted parameters and the caller vcov; factor scores are not,
they are a distinct estimator.

**Build if.** A paper row or methods workflow needs per-observation ordinal
factor scores and wants lavaan-`lavPredict()` parity. This is a real estimator
(quadrature over y\* per missingness/response pattern), not a guard flip —
treat as research-tier, separate from the standardization/defined-parameter
exposure that motivated removing their guards.

## Optimizers

### Exact Hessians for IPOPT

The IPOPT adapter uses limited-memory Hessian approximation and supplies
objective gradients plus nonlinear-constraint Jacobians only; exact objective /
Lagrangian Hessians are unimplemented.

**Alternative already available.** IPOPT's internal L-BFGS approximation, and
the other nine backends for unconstrained/bounded problems.

**Build if.** The optimizer comparison studies show exact objective /
Lagrangian Hessians materially help ML, GLS/WLS/ULS, or nonlinear-constraint
fits.

## Build, packaging, and docs

### Documentation system (two-surface manual)

The design lives in
[docs/design/documentation_proposal.md](../design/documentation_proposal.md):
two surfaces, a C++ compositional methods manual and a staged API manual. The
first concrete step per the proposal is settling audience, vocabulary, and the
API-status/evidence catalog.

**Alternative already available.** The roadmap, design notes, and AGENTS.md
serve the current methods-developer audience.

**Build if.** We commit to public docs.

### Dependency-license manifest for binary artifacts

A full dependency-license manifest (Eigen, optional Ceres, required NLopt,
optional IPOPT, vendored PORT and QUADPACK, test-only nlohmann_json) with
versions and redistribution obligations.

**Alternative already available.** The repo's MIT `LICENSE` also notes the
vendored BSD-3 PORT routines, sufficient for a source release — Eigen, Ceres,
NLopt, and nlohmann_json are fetched at build time, not redistributed.

**Build if.** We ship a binary or packaged artifact.

### Ceres preset in regular validation

Promote the `ceres` preset into regular validation where relevant, without
making the default build pay the Ceres dependency cost.

**Alternative already available.** The `ceres` preset builds and tests on
demand; PORT and NLopt backends are exercised by the default presets.

**Build if.** Calibrated coverage shows Ceres-specific paths are the
undervalidated gap, or a Ceres-backend regression slips through local testing.

### Opt-in precompiled headers for Eigen-heavy builds

**Alternative already available.** Current build-loop timings (tracked in the
roadmap's timings table) are acceptable.

**Build if.** PCH measurably improves changed-TU rebuilds without worsening
no-op or full rebuild ergonomics.

## Experiment extensions (paper-gated)

Completed or archived experiments whose extensions are needed only if a paper's
evidence base demands them. Each experiment's landed coverage is recorded in
the experiment folder and the roadmap.

### Ordinal SNLLS speed pilot: literature-grade grid

**Fired 2026-06 and absorbed into `papers/ordinal-snlls/`.** The native
benchmark gained two-group invariance cells and the naive corr-block-WLS
comparison row, and the paper's `run_speed_grid.R` drives the full `q ≤ 12`
sweep; results land under the paper's `results/`. No remaining trigger.

### Ordinal threshold-constraint experiment: multi-group examples

**Fired 2026-06 and absorbed.** Multi-group threshold invariance is now a
core capability (joint cross-block profiling), gated by unit parity tests,
the lavaan oracle fixture 0013, and the benchmark's `invariance_2group`
cells consumed by `papers/ordinal-snlls/`. No remaining trigger.

### Ordinal construction-boundary experiment: broader blocks

Extend `experiments/_archive/13-ordinal-construction-boundary` beyond
all-ordinal synthetic blocks up to `p = 16`, `c = 5`.

**Alternative already available.** The lazy opt pilot times fit-only ULS/DWLS
raw workspace construction against eager legacy stats construction, projection
to `OrdinalMoments`, diagonal/full Gamma cache copies, DWLS weight
construction, and WLS reinversion.

**Build if.** A paper needs broader construction evidence.

### Robust polychoric threshold parameterization / PD-repair revisit

Revisit the threshold parameterization and positive-definiteness repair policy
of the landed h-score / WMA robust polychoric path.

**Alternative already available.** The all-ordinal h-weighted moment path is
landed (see roadmap); design rationale lives in
`docs/research/notes/h-polychorics.tex` and `robust_ordinal_gamma.tex`. The
remaining committed work is the h-weighted polyserial item in
[todo.md](todo.md).

**Build if.** The robust-ordinal paper track demands it.

### Rhemtulla 2012 replication: nonnormal y* and asymmetric thresholds

`experiments/15-rhemtulla-2012` v1 covers only the symmetric-threshold,
underlying-normal conditions (categories 2–7 × N). Deferred paper conditions:
(1) nonnormal underlying `y*` (skew 2, kurtosis 7 in the paper's convention) —
the C++ cubic Fleishman / Vale-Maurelli primitive covers it but the wiring
into the experiment is missing; this is the condition where cat-LS's own
underlying-normality assumption breaks; (2) the moderate/extreme
asymmetric-threshold conditions, whose exact threshold tables are in the
paper's unavailable supplement (would need a documented rule validated against
their Table 1 skew/kurtosis).

**Alternative already available.** The v1 symmetric/normal replication of the
cat-LS-vs-continuous-ML horse race.

**Build if.** The replication needs to exercise cat-LS bias under
nonnormality.

## Pairwise covariance / missing-data side

### Pairwise μ ACOV for `fit_gls_pairwise` mean-structure models

The μ-block of the GLS weight currently uses the Σ-only `Ŝ_pw⁻¹`
convention. The asymptotically efficient choice under MAR is
`(Γ_NT^μ-pw)⁻¹` with `Γ_NT^μ-pw_{j,k} = σ_{jk} · π_{jk}/(π_j π_k)` — the
same Hadamard structure as the σ-side, applied to the p × p mean ACOV.
Same pattern-grouped trick: enumerate distinct missingness patterns,
average masked-and-rescaled `Σ̂_pw` per pattern.

**Alternative already available.** The current Σ-only μ-block is
consistent under MAR; only asymptotic efficiency on the mean side is
sacrificed.

**Build if.** A pairwise-GLS fit with `meanstructure = TRUE` shows up as
a downstream consumer AND the means side carries non-trivial information
about θ AND the missingness is heavy enough that the μ-block weight
matters. Most CFA / SEM work fits `meanstructure = FALSE` so this is a
double-edge case.

### Pairwise Browne-unbiased

The paper (`papers/ugamma-fast/ugamma-fast.tex`, eq. 19) flags that the
literal finite-sample-unbiased pairwise Γ_u^pw has entrywise-dependent
overlap-set coefficients and does not collapse cleanly to the projected
form. Two paths exist, neither is core machinery:

- **Literal Γ_u^pw**: entrywise overlap-set bookkeeping, p*² entries
  each with its own (a_{ab}, b_{ab}) coefficients and overlap-restricted
  Γ_NT / rank-1 pieces. Worst-case O(p*² · n + p*² · df) per block.
  Expensive and intricate. This is the canonical estimator.

- **Pragmatic fallback**: global Browne coefficients with
  Hadamard-adjusted NT and rank-1 pieces. Cheap. Asymptotic behaviour
  of the substitution is not established literature.

**Alternative already available.** Complete-data Browne unbiased is in
`robust::reduced_gamma_unbiased`. Pairwise data with empirical meat
(Ψ̂'Ψ̂/n) is in `pairwise_casewise_contributions`. The model-implied NT
pairwise meat is the inference-side reducer above.

**Build if.** A downstream project needs a finite-sample-unbiased
pairwise meat AND the empirical alternative is demonstrably inadequate.
Path: literal first (so it's a credible reference); then pragmatic as
opt-in with a bias study comparing the two — see
[[feedback-shortcut-variants]] memo: pragmatic-only is research-tier
work, not a free addition.

## FMG / U-Gamma unbiased-spectrum performance

Both items below were investigated and deferred: the biased and `_ug`
(Du-Bentler / Browne unbiased Gamma) FMG tests share the expensive setup (one
`UFactor`, one tiled casewise-projection / reduced-Gamma accumulation), but a
`_ug` request still adds a second `df × df` matrix formation and values-only
eigensolve. Both shortcuts relied on the wrong `Γ_NT(Σ̂) = I` expected-bread
identity that the unbiased NT-term fix retired (see the FMG regression note in
[`../validation/test_ledger.md`](../validation/test_ledger.md)).

### Row-space eigensolve for the unbiased spectrum

For empirical spectra the biased path eigensolves in row space when
`df > N_total` — the nonzero eigenvalues of `(ZcB)'(ZcB)` equal those of
`(ZcB)(ZcB)'`, avoiding a much larger `df × df` symmetric eigendecomposition for
large-`p` models. The unbiased path cannot reuse it: its low-rank form
`M_u = -bI + (a/N)Y'Y + dww'` was correct only under the identity NT term. With
the correct sample NT term the `-bI` becomes `-b·B'Γ_NT(S)B` (non-diagonal), so
the unbiased path always takes the full reduced route.

**Alternative already available.** The full reduced route
(`reduced_gamma_sample_tiled`, `reduced_gamma_nt_sample`,
`reduced_gamma_unbiased`, `ugamma_eigenvalues`) computes the correct unbiased
spectrum; only the row-space shortcut is missing for it.

**Build if.** A tiny-`N` / high-`df` FMG case (`df ≫ N`) that requests a `_ug`
test makes the full `df × df` unbiased eigensolve the bottleneck. The fix folds
the sample NT term into the row-space form rather than applying a scalar `-b`
shift.

### Rank-one secular update for the unbiased spectrum

A rank-one secular update could avoid the second full eigendecomposition. The
old form `M_unbiased = a·M_sample − b·I + d·vv'` was a diagonal-plus-rank-one
perturbation of `M_sample` (secular-solvable); with the correct sample NT term
`−b·I` becomes `−b·B'Γ_NT(S)B`, no longer diagonal, so the secular form does not
apply directly. Even a perfect secular solve needs eigenvectors of `M_sample` to
rotate `v`, and Eigen's `ComputeEigenvectors` path was ~4–6× slower than
`EigenvaluesOnly` in synthetic `df = 300/739/1500` checks, so two values-only
solves still win at the `p ≈ 40` target.

**Alternative already available.** Two independent values-only eigensolves
(biased, then unbiased) via `ugamma_eigenvalues`; only the unbiased one runs when
a `_ug` test is requested.

**Build if.** A tridiagonal-level update or a LAPACK-backed eigensolver makes the
eigenvector rotation cheap enough to beat two values-only solves.
