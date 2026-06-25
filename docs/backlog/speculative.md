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

### Structural-after-measurement (SAM / LSAM) two-step estimator

A `estimate::frontier` two-step estimator: fit the measurement model (Λ, Θ),
form a Croon/Wall-Amemiya bias-corrected factor covariance Var(η), then solve
the structural model (B, Ψ) against it. The *local* flavor (LSAM, Rosseel & Loh
2024) fits the measurement model block-by-block so structural misspecification
does not leak back into the measurement step; the *global* flavor fits all
measurement parameters jointly. Bogaert, Loh, Schuberth & Rosseel (2025, *SEM*
32:2, 215-236; PDF in `external/refs/`, eval in
[paper-evals](../research/paper-evals/2025-bogaert-measurement-error-small-sample.md))
show LSAM is the principled small-sample alternative to joint SEM-ML: it keeps
two-step robustness where SEM-ML test statistics go bimodal (N≤30), while
correcting the attenuation that inflates Type I error for uncorrected
factor-score regression (UFSR) and PLS once two correlated latent predictors
are measured with error. The structural fit is a clean match to magmaan's
existing measurement/structural matrix split (`model::MatrixRep`: Λ, Θ vs B, Ψ).

**Alternative already available.** SEM-ML (joint, complete-data normal-theory)
is magmaan's core estimator and the paper's gold standard; for the two-step
alternative, lavaan's `sam()` covers the practical case on a refit and is the
parity oracle for LSAM point estimates.

**Build if.** A paper row or methods workflow needs the unbiased two-step
alternative on a magmaan fit, most naturally a small-N inference study or a
SAM-under-misspecification / non-normality / missingness study (the paper's own
limitations section flags non-normal and missing data as unexamined for SAM,
which is exactly the FIML-FMG / ML2S / misspec-robust-SE territory). Sequencing:
LSAM point estimates to lavaan `sam()` parity first (a credible reference), then
the structural-step SEs. The small-sample SSC variant (Fuller α-modification,
Bogaert et al. 2023) and its SE convention (a weighted average of UFSR/LSAM SEs;
no closed form) are research-tier with their own bias-variance validation, not a
free parity addition (see [[feedback-shortcut-variants]]); defer them past plain
LSAM.

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

### Multi-factor EAP and non-diagonal residual ordinal factor scores

The landed ordinal/mixed factor-score path covers diagonal residual `Theta`:
EBM and ML by per-pattern Newton over the latent-response likelihood/posterior,
and one-factor EAP by QUADPACK integration. Two heavier extensions remain
separate: multi-factor EAP (adaptive/product Gauss-Hermite centered at the EBM
mode) and models with correlated ordinal residuals, where a response pattern's
probability is a multivariate orthant integral rather than a product of
univariate interval probabilities.

**Alternative already available.** Use the diagonal-Theta categorical scorer:
`api::factor_scores` / R `factor_scores()` expose EBM, ML, and one-factor EAP for
all-ordinal and mixed complete data, while continuous fits keep the
regression/Bartlett path.

**Build if.** A methods workflow needs posterior means for two-or-more-factor
categorical models, or a paper/model requires residual covariance among ordinal
indicators and wants lavaan-`lavPredict()` parity instead of an explicit
unsupported-shape error.

### Mplus/lavaan WLSMV invariance extensions

The implemented ordinal-invariance surface covers the standard lavaan-theta
route and the explicit Mplus-style delta ladder, but three edges stay deferred:
mixed continuous/ordinal pairwise missing data, broader live-Mplus goldens, and
a nicer diagnostic for lavaan's delta `group.equal` `~*~` singularity.

**Alternative already available.** `mplus_wlsmv_invariance()` covers
all-ordinal pairwise/listwise and mixed listwise/complete data; the all-ordinal
pairwise scalar probe is Mplus-Demo-gated, the mixed listwise case is a
deterministic regression, and lavaan-theta `group.equal` parity remains the
standard lavaan oracle path. For lavaan-style delta, the supported guidance is
explicit: do not gate the degenerate `group.equal` release; use theta for
lavaan parity or the explicit Mplus-style delta helper when released-delta
semantics are the target.

**Build if.**

- A mixed continuous/ordinal pairwise missing-data consumer appears: implement
  mixed pairwise moment/NACOV construction first, then unlock
  `missing = "pairwise"` in the helper and add Mplus Demo-scale goldens.
- A broader Mplus parity claim is needed: add a maintainer-only Mplus fixture
  generator and checked JSON/CSV outputs, keeping CI Mplus-free.
- A user-facing compatibility problem recurs around lavaan delta invariance:
  add a focused diagnostic/warning that explains why the released `~*~` rows are
  unidentified and points users to theta or the Mplus-style helper.

### Robust mixed/polyserial moments under missing data

A *robust* (h-weighted / WMA, DPD, Huber) mixed stage-1 for MISSING data, the
missing-data sibling of the landed complete-data robust ordinal moments: the
three ML-mirroring flavors (pairwise/observed-overlap, hybrid FIML-normal, and
the matching robust casewise influence / `NACOV` so DWLS/WLS weights and the
estimated-weight γ channel carry through under missingness).

**Alternative already available.** Complete-data robust ordinal moments are
landed (the robust-ordinal paper track ships them, all-ordinal), and the *ML*
missing-data mixed stack — `mixed_ordinal_stats_from_observed_data`
(MCAR/pairwise-overlap) and `mixed_ordinal_stats_hybrid_fiml_from_observed_data`
— already exists, so the missing-data layer (`Gamma^pw` overlap reweighting) is
ready to mirror once a recipe is chosen. The blocker is not effort but a recipe
choice: "WMA polyserial" is a category error (WMA is a polychoric cell-overcount
cap; the polyserial robust members are DPD and Huber/Tukey-residual), so a mixed
robust estimator must go uniform-DPD, intrinsic-mix (which the user finds
inelegant), or keep the robust paper all-ordinal where the question doesn't
arise. The copula stress results favour uniform-DPD and kill the intrinsic-mix
Huber/Tukey port. Full synthesis + resume checklist:
[docs/research/notes/robust_mixed_recipe_taxonomy.md](../research/notes/robust_mixed_recipe_taxonomy.md);
copula evidence in
[docs/research/notes/robust_ordinal_copula_results.md](../research/notes/robust_ordinal_copula_results.md).

**Build if.** A concrete mixed-robust-missing-data consumer (paper row, user
request) appears AND the recipe fork is resolved. If it fires, build the chosen
recipe as a unified `mixed_*_shared_robust_*` taking the recipe as an option,
then flavor 1 (observed/overlap missing-data treatment) then flavor 2 (hybrid
FIML-normal continuous block).

## Measures / reporting

### MI effect sizes (dMACS / EDM family) for fitted multi-group models

A `measures::frontier` post-fit surface that quantifies *how much* measurement
invariance is violated, rather than only testing whether it holds: the dMACS /
dMACS_Signed pair (Nye & Drasgow 2011), Glass-Δ analogue UDI/SDI and the
weighted WUDI/WSDI (Gunn et al. 2020), and the Cohen-f analogue fMACS (Lai et
al. 2025). Schuhbeck, Sterner & Goretzko (2025, *SEM* 33:1, 81-88; PDF in
`external/refs/`) unify all of these as the class of *Expected Difference
Measures* (EDMs), built from three normal-expectation terms (signed / absolute /
squared model-implied item-score difference), and give **closed formulas with
O(q) complexity** for general common-factor models (cross-loadings, correlated
factors), replacing the exponential-in-q numerical integration that limited the
predecessors to simple structure. Inputs are exactly what a fitted magmaan
multi-group model already produces: group-specific intercepts τ_g, loadings
Λ_g, and the latent mean/covariance (κ_g, Σ_g). Reference R package:
`github.com/TiziSchuh/ExpectedDifferenceMeasures` (lavaan-compatible).

**Alternative already available.** magmaan's multi-group invariance machinery
covers the *testing* question (configural/metric/scalar fits, nested
difference tests, the FIML/ML2S MI difference-test papers in exps 21/25, ordinal
`group.equal`). Effect-size reporting can be done by hand from the exposed τ/Λ
estimates, or in the authors' R package directly on a lavaan refit.

**Build if.** A paper row or methods workflow needs MI effect sizes reported on
a magmaan multi-group fit (most naturally alongside the invariance-test papers),
*or* an EFA-based MI method (EFA trees / mixture MGFA) that calls dMACS many
times needs the O(q) closed form rather than a per-call numerical integral. The
closed formulas are fully specified in the paper (eqs. 4-6), so this is a
transcription-plus-validation task, not a research problem — except the
rotational-indeterminacy caveat the authors flag for EFA solutions, which would
need its own convention. Research-tier benchmark cutoffs (Nye et al. 2019) carry
their own validation burden; do not ship interpretive thresholds as core.

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

### Overlap-corrected Γ for WLSMV / ordinal missing data (PD repair)

The continuous `fit_gls_pairwise` path already repairs the saturated Γ for
missingness via the pattern-overlap Hadamard rescaling
(`σ_{jk}·π_{jk}/(π_j π_k)`, `pairwise_casewise_contributions`); the ordinal
counterpart is unbuilt. Default WLSMV + pairwise deletion (Mplus
`estimator = WLSMV` with missing data) estimates thresholds + polychorics from
each available-case margin but forms the weight matrix and the
mean-and-variance-adjusted Δχ² *as if the polychoric matrix came from complete
data* — the non-uniform per-margin N and the missing-data uncertainty are never
propagated. Chen, Wu, Garnier-Villarreal, Kite & Jia (2020, MBR 55:1; PDF in
`papers/fiml-fmg/dev/refs/`) confirm the resulting Δχ² Type-I inflates to ~0.55
(50% missing, N = 1000) and are explicit that the statistic is mis-constructed
rather than this being an implementation defect. The fix is the ordinal sibling
of Savalei & Bentler (2005) "statistically justified pairwise ML": restrict each
ordinal influence-factor cross-product (`OrdinalGammaCacheBlock` IF,
Γ = IF'IF/n) to its pairwise overlap set and rescale by the same π proportions,
giving a calibrated WLSMV missing-data weight + scaled test. Folds into the FMG
spectrum: an overlap-weighted ordinal Γ yields an ordinal-FMG / pEBA difference
test under missingness, extending the Paper 2 ordinal-fmg track
(`papers/fiml-fmg/`) into the incomplete-data cell.

**Alternative already available.** Continuous FIML / ML2S + FMG covers the
continuous missing-data inference case (the current fiml-fmg paper); the ordinal
*complete-data* WLSMV / robust path is landed (roadmap); the continuous
pairwise-overlap Γ correction (this section's other entries) is the template.
Only the ordinal-margin routing is missing, and magmaan has no ordinal *missing*
data path today, so this is an unbuilt capability, not a regression.

**Build if.** A concrete ordinal-missing-data paper row appears, or the ordinal
`fmg_test` path lands (the Paper 2 gate) and there is appetite to push it into
the incomplete-data cell. Sequencing: the exact overlap-set ordinal Γ is the
credible reference first; a pragmatic global-π shortcut is its own validation
problem, not a free addition ([[feedback-shortcut-variants]]). Distinct from the
complete-data "Robust polychoric threshold parameterization / PD-repair revisit"
above — this is the missing-data *weight matrix*, not threshold parameterization.

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
