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

### Penalized ML for small-sample convergence and structure (parameter-space)

A `estimate::frontier` penalized-ML surface that conditions in *parameter* space
rather than moment space, the principled counterpart to the existing covariance
shrinkage. Two penalties: (a) a residual-variance log-barrier
`P_ρ(θ) = -ρ Σ_i log ψ_i` (equivalently `ρ Σ_i ψ_i⁻¹`; an inverse-gamma prior in
MAP terms, the smooth sibling of De Jonckere & Rosseel 2022 bounded estimation)
to cure small-N nonconvergence by repelling residual variances from zero, the
on-target fix since the disease is Heywood cases (ψ_i → 0⁻); and (b) a structured
Gaussian penalty `P_ρ(θ) = ρ(θ-θ⁰)'A(θ-θ⁰)` toward an idealized standardized model
(all loadings 0.7, reliability 0.8), the non-circular parameter-space analog of
the model-based *covariance* target of De Jonckere & Rosseel (2023, PDF in
`external/refs/`, eval in
[paper-evals](../research/paper-evals/2023-dejonckere-model-based-shrinkage-target.md)).
Unlike that covariance target, penalizing θ leaves S, hence the reference law of
the χ² fit test, uncontaminated, so the GOF stays valid. Closes with a
penalized-sandwich SE `Cov(θ̂) = (H + P_ρ'')⁻¹ V (H + P_ρ'')⁻¹` and an effective-df
correction for the fit test.

**Alternative already available.** Covariance-space conditioning already covers
the convergence case: `data::frontier` covariance shrinkage (`S_a = (1-λ)S + λT`),
`estimate::frontier::fit_ml_ridge_continuation()` (warm-started continuation to
α=0), and the `experiments/04-near-singular-ml-continuation` target/λ grid. Hard
parameter bounds are De Jonckere & Rosseel (2022) bounded estimation (the hard
sibling of the log-barrier); Bayesian priors on θ are the MAP version. lavaan
bounded estimation and `regsem` cover the practical case on a refit.

**Build if.** A small-N paper row or methods workflow needs *parameter-space*
regularization specifically, i.e. a case where covariance conditioning is
insufficient, or where an honest (non-circular) structural-shrinkage estimator
with a valid GOF test is required, most naturally a moment-space-vs-parameter-space
convergence/bias/MSE study extending experiment 04. Sequencing: the variance
log-barrier first (smallest delta over bounded estimation, reuses its bound
computation), then the structured Gaussian penalty, then the penalized-sandwich
SE + effective-df. The ρ selector and the non-standard inference are research-tier
([[feedback-shortcut-variants]]). Do not ship the De Jonckere & Rosseel (2023)
model-based *covariance* target as core: it biases the GOF toward the fitted
model (shrinking S toward a model-shaped target); if added at all it belongs in
experiment 04's target menu with that caveat documented.

### Regularized H1 references for two-stage and nested SEM

A separate methods/paper track for regularizing the saturated-H1 references
consumed by two-stage estimators and missing-data nested-test reference laws. One
motivating case is ML2S under missing data: Stage 1 fits the saturated FIML
mean/covariance model, but hard small-N / high-p cells can produce a near-singular
saturated covariance that makes the Stage-2 ML surface and the robust/nested-test
spectra numerically pathological. A second motivating case is direct FIML
Satorra-2000 nested testing: the saturated-H1 information/acov reference can be
near-singular at moderate p under missingness, causing `U*Gamma` to carry giant
spurious eigenvalues and over-scale the difference test. The categorical analogy
is direct: ordinal SEM estimates thresholds plus a polychoric/polyserial
association matrix in Stage 1, then fits the substantive model to those summaries
in Stage 2; PD repair/shrinkage of that association matrix is a familiar
preprocessing/repair layer, not the estimator's default scientific definition.

The proposed estimator family has two related surfaces. For input moments,
transform the Stage-1 summaries, `S_reg = (1 - λ) S_H1 + λ T`, before the Stage-2
fit. Use non-model-shaped targets first (`diag(S)` for covariance-scale summaries,
`I` for correlation-scale summaries), choose the smallest λ needed to hit a
prespecified eigenvalue or condition-number floor, and ledger λ plus raw/repaired
eigen diagnostics. The continuous ML2S covariance-input version of this surface
now exists as opt-in `stage1_regularization`; the broader project remains for
calibration, categorical analogues, and reference-law regularization. For
reference spectra, regularize either the H1 information
before inversion (eigenfloor/truncated pseudo-inverse) or the final `Gamma`
toward a well-conditioned target; do not treat an unqualified additive
`Gamma + λI` as automatically benign, because it can inflate the trace that drives
Satorra-Bentler scaling. For inference, any transformation of the Stage-1
estimator must be propagated through the Stage-1 asymptotic covariance by the
delta method (`Γ_reg = J Γ J'`), mirroring the landed
`data::frontier::shrink_mixed_ordinal_stats()` behavior that rebuilds `NACOV`,
DWLS, and WLS weights after shrinking mixed ordinal moments. Fit-only or
reference-only shrinkage is a diagnostic/frontier reference law unless its
regularization vanishes asymptotically or is otherwise calibrated.

**Alternative already available.** Standard ML2S/lavaan-parity remains the core
path for the FIML-FMG paper. Existing stabilizers are narrower and do not change
the estimand: fail-closed PD guards, hard iteration/walltime caps, parameter
bounds, SLSQP fallback for line-search failures, and the H1 EM moment-tolerance
fix. Existing `data::frontier` covariance shrinkage and complete-data
`fit_ml_ridge_continuation()` cover ordinary complete-data ML and mixed ordinal
moment experiments. The new ML2S-H1 regularized-input estimator is available as
frontier infrastructure, but it is not a calibrated default and does not cover
the direct-FIML H/Gamma nested-reference problem.

**Build if.** A dedicated regularized-reference project wants the bias/variance,
coverage, and reference-law consequences of H1 conditioning, especially for
missing-data ML2S, FIML Satorra-2000 nested spectra, and categorical-style
summary-input estimators in near-singular small-N / high-p regimes. This should
not be folded into `papers/fiml-fmg`: it changes the estimator/reference law and
would dilute that paper's reference-law question. Sequencing: (1) point-estimate
and nested-spectrum probes on frozen hard cells using minimal eigenfloor/condition
cap regularization; (2) diagnostics and calibration for reference-only
regularization, building on the landed Gamma propagation for transformed ML2S
input moments; (3) simulation comparing raw ML2S,
regularized-input ML2S, regularized FIML nested references, categorical ULS/DWLS
analogues, and parameter-space regularization under correct specification and
misspecification.

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

### ESEM (exploratory structural equation modeling): free loading blocks + rotation

An EFA measurement block embedded in an SEM (Asparouhov & Muthén 2009): a set of
`m` factors with all `p x m` loadings free, identified up to an `m x m` rotation,
estimated as the unrotated MLE under `m^2` identification constraints (factor
variances/covariances fixed via `Psi_block = I` plus the `m(m-1)/2` echelon zeros
in the upper triangle of the block's `Lambda`), then *rotated* to a criterion
(geomin, quartimin/oblimin, Crawford-Ferguson family, target, varimax) by the
Bernaards-Jennrich gradient projection algorithm, with delta-method standard
errors for the rotated solution. ESEM does not violate the charter (it is linear,
complete-data normal-theory ML), it was simply deferred: EFA/ESEM is listed out
of scope in [docs/architecture/roadmap.md](../architecture/roadmap.md) (line ~7),
[docs/validation/lavaan_tutorial_parity.md](../validation/lavaan_tutorial_parity.md)
(section 18), and the tutorial README. Adopting it would flip those three
statements; the natural home is `frontier`-tier (`estimate::frontier::efa` or a
`measures/rotation/` module), so it carries no lavaan-stable promise even though
v1 would still be gated against lavaan.

Most of the substrate already exists. The *unrotated* block needs almost no new
numeric code: "free every loading, no marker" is the `effect_coding` pattern
(`src/spec/build.cpp:1160`) plus the `auto_fix_first` skip; `Psi_block = I` is
`apply_std_lv` (`build.cpp:502`); the echelon zeros are ordinary fixed cells; and
`model::ModelEvaluator` is sparsity-agnostic (it reads a write-table), so a dense
`Lambda` column block estimates with zero evaluator changes. Estimated EFA factors
are then ordinary latents in the Reduced form (`src/model/matrix_rep.cpp:33`), so
EFA factors regressing on/with other factors already works structurally. Rotated
SEs have a near-exact template in `src/measures/standardized.cpp`: build
`J = Identity(n_free)`, overwrite the transformed rows (mixing analytic rows and the
finite-difference `fill_fd_row` helper), then `SE = sqrt(diag(J vcov J^T))`, with the
unrotated covariance (including the constrained `Z(Z^T I Z)^-1 Z^T` null-space form)
coming straight from `inference::vcov`.

Two pieces are genuinely new. (1) The rotation module itself: criteria + the
orthogonal/oblique GPA + random restarts (Bernaards & Jennrich 2005, the
GPArotation reference). It is self-contained, off the hot path, and unit-testable
against GPArotation/lavaan in isolation. (2) The rotation Jacobian for SEs: the
Asparouhov-Muthén delta method propagates the unrotated covariance through
`d(rotated)/d(unrotated)`, whose implicit part comes from the implicit-function
theorem on the rotation criterion's stationarity conditions (Jennrich 2007); a v1
can ship a full finite-difference Jacobian through the rotate-from-unrotated map
(slow but correct and parity-checkable) and swap in the analytic derivative once
validated. The remaining cost is parity finickiness: matching lavaan's rotated
output requires replicating its column reordering, sign reflections, and Kaiser
normalization (`lav_matrix_rotate` / `lav_efa`). Other small new bits: grammar for
the `efa("block")` modifier (currently rejected as `ModifierEvalFailed`,
[docs/grammar/grammar.md](../grammar/grammar.md) line ~78; edit the EBNF first); an
"EFA block" grouping on `LatentStructure` (template: `composite_blocks`); and an
EFA start (the FABIN/Guttman producers in `estimate/start_values.hpp` assume marker
CFA, so a PCA/principal-axis start is needed).

Tiering: **v1 (lavaan parity)** = single-group ML, one EFA block alongside
CFA/structural rows, oblique geomin + quartimin + target + varimax, GPA, rotated
point estimates and delta-method SEs gated against lavaan `efa()` / `rotation=`
(shipped since 0.6-13, so a real oracle exists). **Literature-scope** = multiple
EFA blocks; multigroup ESEM with rotation (and rotation-based invariance); full CF
family + promax + Jennrich-Bentler bifactor rotations; FIML; robust/MLR rotated
SEs. **Research-tier** = rotated SEs under misspecification (sandwich / the exp-35
robust bread), ESEM on the frontier robust/DPD estimators, ordinal/polychoric
ESEM. Two design forks to settle before any code: (i) `frontier`-module vs
supported track (recommend `frontier`, v1 still lavaan-gated); (ii) post-fit
rotation of the unrotated MLE vs imposing rotation as a constraint during
estimation (recommend post-fit, which is the classic CEFA-then-rotate pipeline and
how lavaan structures it; the constrained-optimizer path exists but is the harder
road).

**Alternative already available.** None inside magmaan today; for the practical
case, lavaan's `efa()` / `rotation=` on a refit is the full implementation and the
parity oracle. magmaan's confirmatory surface (marker / `std.lv` / `effect.coding`
identification, structural regressions among latents) covers every model where the
loading sparsity pattern is known a priori, which is the entire current scope.

**Build if.** A paper row or methods workflow needs an exploratory loading block
on a magmaan fit rather than a lavaan refit. The most likely trigger is a research
question that wants rotation *plus* something magmaan already owns and lavaan does
not combine with ESEM: rotated SEs under misspecification, ESEM under the frontier
robust/DPD estimators, or ordinal/polychoric ESEM. The adjacent EFA hook already
on this list (the dMACS / EDM rotational-indeterminacy caveat under
[MI effect sizes](#mi-effect-sizes-dmacs--edm-family-for-fitted-multi-group-models))
would also be unblocked by a real rotation convention. v1 is a transcription +
parity task (the algorithms are fully specified in Asparouhov-Muthén 2009 and
Bernaards-Jennrich 2005); the analytic rotation-Jacobian SE is the one piece with
genuine derivation work.

### Small-sample distribution-free intervals for covariance functionals (Kauermann-Carroll)

A small-N coverage correction for the distribution-free Wald interval of any
smooth covariance functional `g(S)`: Cronbach's alpha, the omega family
(omega_total, omega_h, H, H_general), reliability differences, correlations, and
standardized SEM parameters. Each has an ADF/influence SE `se^2 = var(v)/N` with
`v_k = <grad g(S), d_k d_k' - S>` the per-case influence values; the plain Wald
(z) interval undercovers at small N because, per Kauermann & Carroll (2001, JASA;
PDF in `external/refs/corrections/`), the variance estimate is itself noisy with
undercoverage `= c_p * var(se^2)/se^4`, `c_p = phi(z_p)(z_p^3 + z_p)/8` (a
Cornish-Fisher term), and `var(se^2)` is governed by the KURTOSIS of the
influence values. The correction is a t reference on effective df
`f = 2N/(kappa_hat - 1)`, `kappa_hat = mean(e^4)/mean(e^2)^2` (`e = v - mean(v)`),
the Satterthwaite realization of their quantile adjustment, from the same
influence values; no fit, no bootstrap. It is design-free, so it applies to every
covariance functional, unlike the leverage-driven Bell-McCaffrey / Imbens-Kolesar
df (inert for a plain functional). Reference set and the keystone derivation in
[`external/refs/corrections/README.md`](../../external/refs/corrections/README.md)
(KC 2001, Bell-McCaffrey 2002, Imbens-Kolesar 2016, Yuan-Bentler 1997/1998,
Satterthwaite 1946).

**Proof of concept landed.** `experiments/44-alpha-kc-coverage` runs the 2x2
{raw, transform} x {Wald z, KC eff-df t} on Cronbach's alpha (congeneric p=6,
logit transform) and the Pearson correlation swept over rho (Fisher-z transform),
under normal and contaminated-normal data. The rho sweep separates **two
orthogonal small-N defects**: (1) the two-sided coverage deficit = variance of the
variance, roughly FLAT in rho, fixed by the KC effective-df t (the influence-SE
reproduces the textbook `(1-r^2)^2/N`); (2) the left/right miss imbalance =
skewness, ~zero at rho=0 and GROWING with rho (raw Wald misses below/above ~97/6
per 1000 at rho=0.9, N=30, vs balanced ~58/56 at rho=0), fixed by the transform.
The transform alone balances the sides but does not lift the total; the KC t alone
lifts the total but stays imbalanced; only `transform + KC t` is both near-nominal
and balanced, for alpha and the correlation alike. The two corrections compose for
free: the transform rescales every influence value by one constant, leaving the
kurtosis (hence the KC df) unchanged, so order does not matter. Honest residual at
N<=30 under heavy tails: `kappa_hat` is a fourth moment estimated from few cases
and the ADF SE is downward-biased there (the HC2/Bell-McCaffrey meat-debiasing's
job, not a df or transform job).

**Alternative already available.** The bootstrap (Kelley & Pornprasertmanit 2016)
is the field's small-N omega-CI recommendation and, for a closed form, nearly
free; plain logit-Wald is what `papers/closed-form-omega` ships, and that interval
was deliberately kept simple to keep the paper focused. The maximal-reliability
entry below already lands a heavier *per-coefficient* small-N CI stack (logit +
robust sandwich + second-order functional bias correction + Bartlett-corrected
profile-LR, `experiments/43`); the KC route is the lighter, family-wide
alternative.

**Build if.** A "small-sample psychometric inference" paper or methods workflow
wants one calibrated, fit-free, bootstrap-free interval recipe across several
psychometric covariance functionals (alpha, the omega/H family, reliability
differences, correlations), or a closed-form-omega referee pushes on small-N
coverage hard enough to want more than logit-Wald. Sequencing: port the
experiment-44 correction to the omega family on the closed-form-omega /
reliability harness first (same influence values), then add the HC2 meat-debiasing
(Bell-McCaffrey) for the SE downward bias and a second-order functional bias
correction for the point bias to close the N<=30 residual. The `kappa_hat`
stabilization at tiny N (trimmed / shrunken influence kurtosis) is research-tier
with its own validation ([[feedback-shortcut-variants]]); the Bartlett-corrected
profile-LR is the heavier alternative when df-widening underperforms (now scoped as
its own family-wide project, *Small-sample-corrected profile-LR confidence intervals
for the reliability family*, in the Measures section below).

Design cautions (author, 2026-06-30, after the experiment-44 2x2):

- *Mix-and-match risk.* transform + KC effective-df + HC2 avar-debias is a stack
  of corrections that interact (the KC df and the HC2 meat-debias both bear on the
  variance estimate and can over-correct together; the bias correction adds yet
  another estimated higher moment). Treat the stack as a deliberate study with its
  own calibration check, not a free pile-up; add arms one at a time and watch for
  over-coverage, especially at small N.
- *Do not target exactly 0.95 at tiny N.* Distribution-free inference cannot
  recover fourth-moment information the sample does not contain, so the honest goal
  at N<=30 (heavy tails) is calibration *improvement* and left/right *balance*, not
  nominal coverage. experiment 44's `transform + KC t` already delivers that
  (near-nominal and balanced on normal data to N=20; the residual is heavy-tails x
  tiny-N, the avar-debias's territory).

### Studentized permutation measurement-invariance test

Permutation testing of metric invariance was scoped as a possible raison d'etre for
the non-iterative (Guttman) CFA estimators ([[noniterative-cfa-inference]]): a
permutation reference needs `B` refits, the closed-form map refits far faster than
ML, so a `B`-fold speedup looked like it was sitting there. **Probed 2026-07-09:
the frozen-bread Wald shortcut was rejected, and the full studentized benchmark
was corrected to use the SE-only path.** The probe is tracked at
`docs/research/sims/r/guttman_permutation_invariance_probe.R`; both tables below
reproduce from it.

The statistic is a Wald test of the metric restriction in the marker metric,
`W(labels) = (A theta)'(A Omega A')^-1 (A theta)`, which both estimators compute
through the same `inference_wald_test_theta` primitive with the same `A` and the
same permutations. Plain label permutation is exact only under *full*
exchangeability, which metric invariance does not deliver: the groups may still
differ in `Phi`, `alpha`, `Psi`. The pooled data is nevertheless a common-factor
model with the same `Lambda` (the between-mean term lives in `Lambda`'s column
space), so `A theta` stays centred at 0 under permutation and the heterogeneity
leaks in only through the *variance*. That is a Behrens-Fisher problem, and by
Chung & Romano (2013, Ann. Statist.) a permutation test with an asymptotically
*pivotal* (studentized) statistic stays asymptotically valid when exchangeability
fails, and exact when it holds. So the studentizer is not a convenience, it is the
precondition.

Three studentizer arms, each computing `Omega` the same way for the observed and
permuted labellings (the symmetry a permutation test requires; freezing `Omega` at
the *observed* fit self-studentizes only the observed statistic and is spuriously
conservative, which is a trap worth remembering):

| arm | `Omega(labels)` | Jacobians |
| --- | --- | --- |
| `full` | `J(S(labels)) Gamma(X(labels)) J(S(labels))'/N` | `B+1` |
| `pbread` | `J(S_pool) Gamma(X(labels)) J(S_pool)'/N` | 1 |
| `pboth` | `J(S_pool) Gamma(X_pool) J(S_pool)'/N` | 1, constant |

Level at `alpha=.05`, 2-group 2-factor 6-indicator CFA, metric-invariant loadings,
600 reps x 199 permutations:

| cell | map `full` | `pbread` | `pboth` | ML `full` |
| --- | --- | --- | --- | --- |
| `n=50`, `phi=(1,1)` | 0.064 | 0.042 | 0.042 | 0.059 |
| `n=50`, `phi=(1,3)` | 0.053 | **0.217** | **0.232** | 0.067 |
| `n=100`, `phi=(1,3)` | 0.042 | **0.240** | **0.250** | 0.043 |
| `n=300`, `phi=(1,3)` | 0.038 | **0.183** | **0.190** | 0.037 |

The frozen arms are exact under exchangeability and reject at 18-25% under
factor-variance heterogeneity, and that does *not* shrink with `N`. Updating the
meat buys essentially nothing over freezing everything (0.217 vs 0.232). **The
pivotality lives entirely in the bread `J(S(labels))`**. The first cost pass
accidentally timed the full `noniterative_cfa_grouped_inference()` GOF bundle
while only using `$vcov`; the current probe uses `noniterative_cfa_se()` and
therefore times the needed studentizer rather than the residual spectrum. Cost of
one *valid* permutation (refit + restudentize), `n=300` per group:

| `p` | map point | map full | ML point | ML full |
| --- | --- | --- | --- | --- |
| 6 | 0.065 | 0.213 | 0.229 | 0.303 |
| 15 | 0.197 | 1.375 | 1.430 | 2.008 |
| 25 | 0.363 | **4.062** | 4.328 | **8.719** |

The map is ~12x faster per point fit and ~2x faster per valid studentized
permutation at `p=25` on this slice. Statistically the map still buys little:
level and power track ML to within noise (power 0.265 vs 0.282 in the original
level/power run). The corrected lesson is narrower: the valid Wald permutation
statistic needs a fresh bread, but it must be benchmarked through the SE-only
path, not through residual GOF.

**Alternative already available.** ML with per-permutation expected information is
already a cheap and correct engine for this test, and `semTools::permuteMeasEq`
(Jorgensen, Kite, Chen & Short 2018) is the incumbent. Note their `Delta chi^2` is
LRT-based and therefore *implicitly* studentized (the nuisance `Phi` is re-estimated
under both models), so the failure documented above is specific to Wald statistics
carrying a frozen covariance, not to permutation MI testing as such. Confirm that
before treating the level table as a criticism of the incumbent.

**Build if.** Two live leads, neither of which is closed-form-specific.
(1) *A small-sample MI paper, estimator-agnostic.* At `n=50` the asymptotic
chi-square Wald rejects at 0.008 where the studentized permutation Wald is nominal
(0.053) and carries roughly twice the power (0.166 vs 0.093). That is a real
finding about MI testing at small `N` and it holds for ML. Gate it on first
checking `permuteMeasEq`'s `Delta chi^2` under the same `phi` heterogeneity: if the
LRT is already calibrated, the contribution shrinks to "do not permute a
frozen-covariance Wald," which is a note, not a paper.
(2) *The redirect.* The map's profile pays off in procedures that resample the
**point estimate** and need no per-replicate studentizer: bootstrap percentile CIs,
jackknife, cross-validation, Monte Carlo. That is exactly the lane below, which
this probe materially strengthens. Do not benchmark permutation MI through the
GOF-producing grouped-inference path when the statistic needs only `Omega`.

### Null-bootstrap GOF calibration for non-iterative CFA

Absolute GOF is a better resampling target than permutation MI for the
non-iterative CFA estimators ([[noniterative-cfa-inference]]). The analytic
robust/mixture GOF path is expensive because it builds the estimator-map
Jacobian, residual projector `M = I - Delta J`, fourth-moment matrix, and
weighted-chi-square spectrum. A null bootstrap needs only the raw discrepancy
statistic on each draw:
`T = N (vech(S) - vech(Sigma_hat))' V (vech(S) - vech(Sigma_hat))`.
So a bootstrap replicate is `S_b -> tau(S_b) -> T_b`, with no `J`, no `Omega`,
and no spectrum.

**Probe (2026-07-09).** Tracked at
`docs/research/sims/r/guttman_bollen_stine_gof_probe.R`. The script compares
naive central chi-square, analytic NT mixture, analytic empirical-Gamma mixture,
Gaussian parametric bootstrap, and a Bollen-Stine covariance-null row bootstrap.
The Bollen-Stine arm recentres the data, maps its ML covariance to the fitted
`Sigma_hat`, and row-resamples the transformed cases, preserving non-normal shape
while imposing the covariance null.

Cost on a one-factor block at `n=300`:

| `p` | point | `T` only | analytic empirical GOF |
| --- | --- | --- | --- |
| 6 | 0.079 | 0.114 | 0.440 |
| 9 | 0.527 | 0.615 | 6.367 |
| 12 | 6.609 | 6.922 | 89.000 |
| 15 | 43.750 | 43.375 | 663.500 |

The bootstrap draw cost is essentially the point-map cost, whereas analytic
empirical-Gamma GOF is 4x-15x slower in this grid. The large single-block
point-map cost at `p >= 12` is itself a reminder that block size matters; this
is still a much better fit to the estimator profile than per-replicate
studentized MI.

Calibration/power pilot (`n=100`, `p=9`, 200 datasets x 99 bootstrap draws,
ULS statistic):

| cell | chisq | NT mix | empirical mix | param boot | Bollen-Stine |
| --- | --- | --- | --- | --- | --- |
| normal null | 0.000 | 0.060 | 0.025 | 0.050 | 0.035 |
| t5 null | 0.000 | 0.075 | 0.005 | 0.070 | 0.040 |
| chisq3 null | 0.000 | 0.075 | 0.010 | 0.055 | 0.075 |
| t5 residual-covariance alternative | 0.005 | 0.255 | 0.110 | 0.220 | 0.195 |

This is a cautious positive, not a victory lap. Bollen-Stine did not obviously
collapse: it was closest to nominal across normal/heavy-tail/skew, but it gave up
power relative to NT mixture and Gaussian parametric bootstrap in the alternative
row, and the skew null was high in this small pilot. The empirical-Gamma mixture
was conservative here. A larger grid with `B >= 399`, more alternatives, and an
ML Bollen-Stine comparator is needed before treating this as an application.

**Build if.** The next step is still R-only: expand the probe to a serious grid
and compare against ML / lavaan Bollen-Stine where feasible. Promote only if
Bollen-Stine keeps level under non-normality while retaining enough power to be
useful. If it does, add a core `noniterative_gof_stat` / R
`noniterative_cfa_gof_bootstrap()` surface so production code does not have to
call the full inference bundle for bootstrap draws.

### Efficient leave-one-out / infinitesimal jackknife for closed-form (non-iterative) CFA

Near-free leave-one-out cross-validation for the non-iterative (Guttman) CFA
estimators ([[noniterative-cfa-inference]]), exploiting that the point estimate is
a closed-form covariance map `theta-hat = tau(S)` whose Jacobian `J = d tau / d
vech(S)` is *already computed* for the delta-method SEs. For an iterative ML fit,
LOO is `N` re-optimizations; here it is either `N` closed-form map evaluations
(exact) or a single precomputed reduction (first order), so the "so much least
squares" structure buys a genuine cost collapse.

The permutation-MI probe above is direct empirical support for this lane's premise:
the map is ~12x faster than ML per point fit at `p=25` and the gap widens with `p`,
so any resampling scheme that needs *only* the point map (no per-replicate `Omega`)
inherits that factor. Resampling schemes that need a per-replicate studentizer
invert it.

The one primitive is a rank-1 *scatter* downdate. Work with the cross-product
`W = sum_j u_j u_j'`, `u_j = x_j - xbar`; removing case `i` *with* recentering is
exactly `W_{-i} = W - (N/(N-1)) u_i u_i'`, so `S_{-i} = W_{-i}/(N-1)` and the only
per-case temporary is the residual vector `u_i` (apply the perturbation
matrix-free, never form it). Two tiers:

- **Exact sweep.** `theta-hat_{-i} = tau(S_{-i})` re-runs the closed-form stages
  on the downdated scatter, reusing each stage's factorization (Sherman-Morrison
  RHS update on the GLS regression nodes, perturbed-eigenpair on the loading
  blocks). O(p^3)-ish per case, no optimization.
- **First order (RidgeCV-shaped).** Freeze all stage designs and the whole
  pipeline collapses to the operator already in hand: each parameter's leave-one-out
  shift is a *quadratic form in the case residual*,
  `theta-hat_{-i,r} ~= theta-hat_r - (1/(N-1)) ( u_i' G_r u_i - tr(G_r S) )`,
  where `G_r` is the r-th row of `J` reshaped to a symmetric `p x p` matrix (via
  the duplication-matrix inner-product bookkeeping). Precompute `{G_r}` and
  `{tr(G_r S)}` once (pure reorganization of `J`), then every case is one quadratic
  form per parameter, O(p^2) each, zero refits. `u_i' G_r u_i` is the direct
  analogue of RidgeCV's leverage `x_i'(X'X)^{-1}x_i`. This *is* the infinitesimal
  jackknife / empirical influence function, i.e. the closed-form specialization of
  the existing case-influence one-step ([[case-influence-semfindr]]).

Why there is no single `e_i/(1-h_ii)` collapse: the estimator's internal
regressions are over *moments*, not over cases. A row of any stage's normal
equations is a covariance entry, so leaving out a case is not a leave-one-row-out
of any regression, it is a rank-1 perturbation of the data feeding *every*
regression at once. Hence a forward rank-1 sweep, not a hat matrix. The
downstream deliverables are all reductions over the same `N` LOO vectors:
(a) a **jackknife SE** `V_jack = ((N-1)/N) sum (theta-hat_{-i}-thetabar)(.)'`, a
third distribution-free leg alongside delta-method-NT and delta-method-empirical-Gamma
(exp-52 showed NT-Gamma SEs are asymptotically wrong on non-normal data and
empirical-Gamma is calibrated -- does jackknife track empirical-Gamma at N=50?);
(b) an **estimator-native LOO-CV discrepancy** `(1/N) sum_i F(S, Sigma(theta-hat_{-i}))`
(ULS or NTML) for model comparison where AIC/BIC are documented-invalid for this
estimator (the fit-indices note flags logl/AIC/BIC as ML-referenced quantities
evaluated at a non-ML estimate) -- the empirical realization of Browne-Cudeck
ECVI, which is the *analytic* approximation to exactly this; (c) a **CV-debiased
omega** `(1/N) sum_i omega(Sigma(theta-hat_{-i}))` correcting the in-sample
optimism of reliability, tying into [[closed-form-omega-engine-modal]]; (d) exact
cheap **case influence / leverage** at scale.

**Alternative already available.** The first-order object already exists
conceptually as the case-influence one-step ([[case-influence-semfindr]], the
`continuous_ls_casewise_influence_ij` / semfindr-parity lane); the delta-method-NT
and empirical-Gamma sandwich SEs cover the SE question; ECVI on a refit and the
(invalid-here) information criteria are the model-selection incumbents; logit-Wald
and the bootstrap (Kelley-Pornprasertmanit) cover reliability CIs. Nothing wires
the *closed-form* LOO specifically, and nothing exploits the exact-sweep tier or
the precomputed-`G_r` quadratic-form collapse.

**Build if.** A concrete consumer for the non-iterative estimator appears that
one of the incumbents does not cover: a **model-selection** need where the
documented-invalid AIC/BIC leave a real hole (LOO-CV discrepancy is the
likelihood-free answer, and the closed form is what makes it affordable -- the
strength/weakness symmetry is the hook), a **jackknife-vs-empirical-Gamma SE**
calibration study on the existing exp-52 harness, or **CV-debiased reliability**
graduating the closed-form-omega lane. Cheapest probe (~40 lines, pure R over the
landed post-fit surface): `loo_influence(fit)` = the first-order quadratic forms,
validated against a brute-force exact refit sweep on Holzinger-Swineford; if they
track, jackknife SE / CV discrepancy / debiased omega are all reductions over the
`N` vectors. Honest risks: (i) ECVI already occupies the analytic-CV chair, so the
differentiator must be exact + cheap + likelihood-free, not "CV for SEM"; (ii) the
first-order approximation is expected to degrade at high leverage / near-Heywood,
the same boundary regime that broke the bifactor `rho*` correction ([[profile-lr-reliability-ci-project]]),
which is exactly what the exact-sweep tier is for as a check. LOO-CV for a
covariance-structure target (prediction object = a case's covariance contribution,
not a scalar `y`) is a less familiar construction and needs its estimand pinned
before any coverage claim.

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

### Model-based omega (omega_u / omega_H / omega_ho) reliability

A `measures::frontier` post-fit surface for the CFA-parameter forms of coefficient
omega: omega-unidimensional `omega_u = (1' Lambda)^2 / sigma_X^2` off a one-factor
fit, omega-hierarchical `omega_H` off a bifactor fit (general-factor loadings in
the numerator), and omega-higher-order `omega_ho` off a higher-order fit
(`lambda_jk * gamma_k` products), with sigma_X^2 either model-implied (`1' Sigma-hat 1`)
or observed (`1' S 1`), plus delta-method SEs reusing the gradient-times-Gamma
path. Bell, Chalmers & Flora (2024, *EPM* 84:1, 5-39; PDF in `external/refs/`,
eval in
[paper-evals](../research/paper-evals/2024-bell-omega-misspecification.md)) is the
recent oracle for the bias-under-misspecification story: omega_u is strongly
positively biased when error correlations are ignored or the population is
multidimensional, omega_H stays nearly unbiased even when the bifactor model is
itself wrong, the model-implied-vs-observed denominator choice barely matters, and
fit indices only weakly track omega bias. This is the CFA-parameter object, distinct
from the S-based coefficients (alpha, Guttman lambda6, Spearman-Guttman omega) in the
in-flight `measures::frontier::reliability` module, and the same object as the exp-20
omega-alpha thread in [todo.md](todo.md) (omega from a one-factor ML fit, alpha = omega
of a ULS tau-equivalent fit) and the roadmap `infer_gamma_nt` omega.

**Alternative already available.** magmaan already produces every input from a fitted
CFA (`Lambda`, model-implied `Sigma-hat`), and the continuous C++ frontier surface now
does this directly as `omega_from_fit` with robust delta SEs. The S-based glb-family
coefficients and the exp-20 omega-alpha difference test cover the adjacent reliability
questions.

**Progress (2026-07-02).** The first ordinal proving slice landed in C++:
`ordinal_observed_score_covariance` / `omega_ordinal_observed` compute omega on the
observed integer category-score covariance induced by thresholds and a latent-response
correlation matrix, and `estimate::frontier::ordinal_observed_omega` applies that to a
single-group all-ordinal DWLS/WLS/ULS fit with a complete IJ-sandwich delta SE. This
addresses the "can the ordinal/DWLS stack report an observed-score reliability metric?"
infrastructure question, but not the literal Green-Yang/Flora target. Experiment 48
(`ordinal-omega-target-audit`) compares this covariance omega to the direct one-factor
ordinal true-score target `Var(sum E[Y_j | eta]) / Var(sum Y_j)`: tau-equivalent
equal-threshold cells match, equal-threshold congeneric cells are nearly identical, and
heterogeneous thresholds reached a 0.0158 absolute population-target gap. It does not
settle the Bell-style misspecification bias study, omega_ho/maximal-reliability ordinal
extensions, multi-group target definition, direct Green-Yang/Flora coefficient
implementation, or small-sample/profile-LR corrections. R exposure landed immediately
after as `magmaan_core$measures_reliability_ordinal_observed_omega`, with
`experiments/46-ordinal-observed-omega-dwls` as the initial smoke/calibration probe.
The no-integration latent-response/polychoric sibling is also exposed in R as
`magmaan_core$measures_reliability_ordinal_polychoric_omega`: it applies
`omega_multidim` to `stats$R[[group]]` and reuses the ordinal `NACOV` correlation
block for a robust delta SE. This is the simple ordinal-omega coefficient people
actually report, not the observed-score Green-Yang/Flora target. Experiment 49
(`ordinal-polychoric-omega-coverage`) stress-tests that helper in correctly
specified ordinal-probit data: with 200 reps per cell over `N in {50,100,250}`,
balanced and threshold-extreme cuts had 95% coverage 0.915-0.965 among
successful draws for the latent-response omega target 0.829. The only
construction failures were 6/200 threshold-extreme `N=50` draws with an empty
item category.
Experiment 50 (`ordinal-polychoric-omega-stress`) extends that to skewed,
heavy-tailed, and locally dependent ordinal DGPs, targeting the large-sample
pseudo-true polychoric omega. Mild/local-dependence cells remain near nominal,
but skew/heavy-tail cells bend: skew-extreme gives 0.815/0.903/0.935 and
heavy-tail extreme gives 0.778/0.808/0.845 for `N=50/100/250`, with 42/200
heavy-tail `N=50` construction failures. The no-integration delta path is not
the small-N stress solution.

**Build if.** A paper row or methods workflow needs model-based omega reported on a
magmaan fit, most naturally the exp-20 omega-alpha thread graduating to core, or a
misspecification-bias study replicating Bell. The genuinely novel cell is the ordinal
one beyond that first slice: Bell uses continuous normal data only and explicitly
leaves polychoric-CFA omega under misspecification (scaled into the observed total-score
metric, Green-Yang 2009b / Flora 2020) and omega_ho finite-sample behavior unexamined;
magmaan's ordinal/DWLS/sim stack makes that an author-flagged gap. Note this is
*structural-form* misspecification biasing a point estimate, orthogonal to the
distributional/weight misspecification of the [[misspec-robust-se-weight-influence]] /
`papers/estimated-weight-se` SE track; do not fold the two together. Research-tier
interpretive cutoffs are not core. The *maximal* (optimally-weighted) counterpart of
omega_H is the separate maximal-reliability entry below, which omega_H lower-bounds
(ρ_gen ≥ omega_H).

### Maximal reliability / corrected coefficient H for bifactor models

A `measures::frontier` post-fit surface for the *model-based maximal reliability* of
a fitted bifactor (or one-factor) model: the optimally-weighted counterpart of the
unit-weighted model-based omega above. Three coefficients, each a closed form on
quantities magmaan already produces (model-implied `Sigma-hat` and the loading
matrix): the optimal-linear-composite reliability for the general factor `rho*_gen =
1/(1 + 1/(lambda_G' (Sigma - lambda_G lambda_G')^-1 lambda_G))` and for group factor
j (all items, eqs. 14-17 of the paper), the optimal-linear-*sub*-composite
reliability `rho*_gj` (only that factor's items, eqs. 19-20 - the *correct*
generalization of coefficient H), the corresponding OLC/OLSC weight vectors, and
factor determinacy `FD = sqrt(rho*)`. Ordering: `rho*_grp,j >= rho*_gj >= omega_HS,j`
and `rho_gen >= omega_H`, so this entry sits directly above the model-based omega
entry. Li & Savalei (2026, *MBR* 61:3, 469-490; PDF in `external/refs/`, eval in
[paper-evals](../research/paper-evals/2026-li-bifactor-maximal-reliability.md)) is
the reference: they show coefficient H as applied to bifactor models (Rodriguez et
al. 2016, standardized bifactor loadings into the one-factor H formula) is the
reliability of *no* composite and must be replaced by these, and they prove the
unification that Thurstone regression factor scores *are* OLCs (so `rho*` = the
regression-score reliability) and `FD = sqrt(rho*)`.

**Alternative already available.** magmaan already produces every input, and
`measures::factor_scores` already computes the regression-score weights
`(A Psi A') Lambda' Sigma-hat^-1` whose rows are the OLC weights, so `rho*` and `FD`
are a few lines over machinery in hand. The clean parity oracle is lavaan itself:
`lavInspect(fit, "fs.reliability")` = per-factor maximal reliability and
`"fs.determinacy"` = FD (the paper cross-checks against exactly these). Distinct from
the *S-based* `measures::frontier::reliability` module (alpha / Guttman lambda6 /
Spearman-Guttman omega off the sample covariance): this is a model-based (theta-hat)
object, same class as model-based omega.

**Build if.** A bifactor/hierarchical reliability experiment or paper needs maximal
reliability as a first-class output, OR the continuous factor-score path graduates to
expose `fs.reliability` / `fs.determinacy` (at which point `rho*` and `FD` are nearly
free and gate against lavInspect), OR the ordinal extension of the omega/reliability
cluster is built (the paper, like Bell, is continuous-normal only; categorical
maximal reliability is the unexamined cell). Heed the paper's own discouragement:
group-factor composites are unreliable at realistic sizes (~200 indicators for an
OLC, ~1000 for an OLSC, to reach 0.8) and carry pervasive negative weights, so the
practical deliverable is a *model diagnostic*, not a scoring tool; ship the
coefficient, not interpretive cutoffs.

**Inference is the open part, and `experiments/43-li-savalei-2026-maximal-reliability-ci`
now scopes it.** The SEs/CIs the paper flags as future work are not free: over a
1000-rep sweep of the correct orthogonal bifactor by ML, the textbook delta-method
Wald CI is badly miscalibrated at realistic N (general-factor coverage falls to
~0.40 at p=18, N=50), and two independent fixes are each needed: the sandwich `V`
(`vcov(fit, regime="robust")`) for variance under-estimation (esp. under
non-normality) and a logit-scale interval for the `[0,1]` boundary and the wild
width at low `rho*`. Together they are the best *simple* interval (near-nominal by
N~200 at p=9, N~500 at p=18), but the binding residual error is the positive bias
of `rho*` itself (up to ~0.20 for the group OLSC at p=18, N=50; Aguirre-Urreta 2019
carried into the bifactor coefficients), which no symmetric interval removes. The
concrete fix the experiment lands on is an after-the-fact second-order bias
correction of the *functional* (not the parameters): `rho-tilde = rho-hat - 1/2
tr(g'' V)`, taken on the logit scale and mapped back so the inverse logit bounds it.
That version is stable down to N=50 (the raw-scale correction over-shoots, to ~-1.2
for the group OLSC at p9/N=50) and essentially unbiased by N=100, and it lifts the
group-factor small-N coverage. The residual gap is the general factor at N<=100,
whose sampling distribution piles against the upper bound at 1 (boundary, not bias);
an analytic skew-aware (Cornish-Fisher) interval does *not* close it, because the
leading-order skewness it needs is itself over-estimated several-fold at N=50 (the
same breakdown, one cumulant up), and a BCa bootstrap would fail the same way. What
*does* close it is to abandon estimator-centred intervals entirely and **invert a
likelihood-ratio test**: the CI is `{rho0 : N(F_con(rho0)-F_unc) <= q}`, the
constrained fit done with a nonlinear maximal-reliability constraint (nloptr SLSQP in
the experiment). It is transformation-invariant, range-respecting, and asymmetric by
construction; the plain chi^2_1 version under-covers at small N only because the LR
is mean-inflated (E[LR] ~ 3-4 at N=50), and a **Bartlett correction** (rescale by
E[LR], estimable analytically or by parametric bootstrap) restores nominal coverage
at every N including 50, for both coefficients. So the recommended built surface is:
the logit-scale-corrected point estimate + robust delta SE for routine use, and a
Bartlett-corrected profile-LR interval for the small-N / boundary regime; the point
coefficients still gate against lavInspect for free. Open: the LR is normal-theory
only (non-normal data needs a Satorra-Bentler weight on it, not Bartlett; that robust
LR is Falk's 2018 R-LCI and already exists in `semlbci`, so what is *actually* open is
small-sample-correcting it, now scoped family-wide in the entry below), interval
*width* (vs the coverage shown) needs profile root-finding, and everything is
conditional on an admissible (non-Heywood) fit.

### Small-sample-corrected profile-LR confidence intervals for the reliability family

**Name: funLR (Functional profile-LR CI).** The engine is generic over any scalar
functional `g(theta)`; reliability is the flagship application, not the scope. Use
"funLR" as the short handle for this lane in commits, notes, and discussion.
`experiments/45-profile-lr-reliability-ci` is its home.

The likelihood-ratio (test-inversion) sibling of the Kauermann-Carroll Wald lane
above: rather than widen a Wald interval's df, build the interval by inverting a
(robust) chi-square difference test, `{g0 : T(g0) <= q}` with `T(g0) = N(F_con(g0) -
F_unc)` the LR for the nonlinear constraint `g(theta) = g0`. One engine serves the
whole reliability family (omega, omega_total, omega_h, H, H_general, model-based
maximal reliability, and reliability *differences* via a 2-parameter region); only the
constraint `g` changes. Test inversion is transformation-invariant, range-respecting
(the constrained fit cannot leave `[0,1]`), and asymmetric by construction, so it
sidesteps the logit-vs-Fisher scale choice and the boundary pile-up that defeat every
estimator-centred interval. `experiments/43` already lands this for bifactor maximal
reliability (the entry above); this item is the generalization to the family as a
standalone project, deliberately NOT folded into `papers/closed-form-omega` (which
ships plain logit-Wald to stay focused). It is the fit-based counterpart to the
fit-free KC Wald route, and `experiments/44` is the Wald-side precedent in the same
spirit.

**Prior-art map (gated 2026-06-30).** The construction is fully owned; the
small-sample correction is the unoccupied cell.

| piece | owner | status |
| --- | --- | --- |
| profile-LR CI for a function `g(theta)` | Pek & Wu 2015; Wu & Neale 2012; Cheung 2009 | done; `semlbci` (Cheung & Pesigan 2023) |
| robust (Satorra-2000-scaled) LR CI | Falk 2018 | done; in `semlbci` |
| which scaled difference test is best for it | Falk & Chen 2026 | done (use Satorra 2000) |
| single attainable-bound LR correction | Pritikin et al. 2017; Wu & Neale 2012 | done (one-sided, normal, mixture-chi^2) |
| logit-scale omega interval | Raykov & Marcoulides 2011 | done (delta method) |
| **small-sample / Bartlett correction of the (robust) LR CI** | -- | **OPEN** |
| **two-sided `[0,1]` bounded-reliability LR CI** | -- | **OPEN** |
| **reliability application of any of it** | -- | **OPEN** |

The SEM Bartlett literature (Yuan, Tian & Yanagihara 2015) corrects the *overall*
model `T_ML`, not a df-1 profile-LR CI for a function. The bounded-parameter LR work
(Pritikin, Wu-Neale) is one-sided, normal-theory, variance-component flavour, never
two-sided `[0,1]` and never small-sample-corrected; Falk & Chen 2026 explicitly
exclude the boundary case ("we do not consider this case ... nor that part of
[Wu-Neale's] algorithm").

**The correction is the contribution, not a footnote.** Under correct normal theory
the df-1 profile-LR is already mean-inflated at the N reliability is reported at
(`E[LR] ~ 3-4` at N=50 in exp-43), so the plain `chi^2_1` interval under-covers; a
Bartlett-type rescaling (an analytic Lawley 1956 factor, or a parametric-bootstrap
`E[LR]`) is what restores nominal coverage. Without it the method fails in the regime
that matters; with it, exp-43 shows nominal coverage at every N including 50, where
robust-SE, logit, second-order bias correction, Cornish-Fisher, and the percentile
bootstrap all fall short. The motivating contrast: Falk & Chen 2026 already found the
*uncorrected* robust LR comparable to the percentile bootstrap, so a
small-sample-corrected, range-aware version should pass bootstrap precisely in the
small-N bounded regime where bootstrap is expensive and (per exp-43) frequently
non-convergent.

**Build if.** A "small-sample reliability inference" paper or workflow wants one
calibrated interval recipe across the reliability family with a likelihood (fit-based)
construction, rather than the fit-free KC Wald route above; or a downstream consumer
needs CIs for a *bounded* reliability where Wald df-widening underperforms (high
reliability near 1; reliability differences near 0). Sequencing: (1) generalize the
exp-43 constrained-fit + Bartlett engine from the bifactor maximal-reliability
constraint to a generic `g(theta)` over a `ModelEvaluator` (omega, omega_h, H first);
(2) make the Bartlett factor operational without the oracle (analytic Lawley term or
parametric-bootstrap `E[LR]`); (3) layer the robust (Satorra-2000) scaling for
non-normal data, i.e. small-sample-correct Falk's R-LCI; (4) the 2-parameter
confidence-region route (Pek-Wu 2015 sec 3) for reliability *differences* (ties to the
exp-20 alpha-omega difference thread). Magmaan's core does the constrained fits
cleanly, itself worth something given Falk & Chen document `semlbci` / forked-`lavaan`
as finicky and optimizer-dependent. The robust + small-sample combination is
research-tier with its own calibration ([[feedback-shortcut-variants]]); add arms one
at a time per the KC entry's design cautions.

**Design (2026-07-01, co-designed; target venue JASA).** The reliability family is
the motivating application; the method is the headline: small-sample-corrected
profile-LR confidence intervals for a smooth *bounded* functional of a *possibly
misspecified* moment-structure model, valid under normal-theory, non-normal, and
misspecified regimes. Concrete decisions:

- *Constraint mechanism = C++ closure, NOT the partable `:=`/`==` DSL.* Magmaan
  already imposes nonlinear equality constraints end to end (`optim::ConstrainedScalarProblem`
  carries `std::function` `h` / `J_h` closures; `estimate::run_scalar_constrained`
  dispatches to NLopt SLSQP or IPOPT; `compose_scalar_ml` already merges the linear
  K-reparameterization with the nonlinear block). But the partable constraint DSL is
  scalar-arithmetic forward-mode AD only (`Add/Sub/Mul/Div/Pow/Exp/Log` over named
  params; `expr_eval.hpp`), so it cannot express matrix-inverse functionals (`H`,
  `H_general`, `rho*`), and `g0` is a moving bisection target (re-lavaanifying per step
  is absurd). So `g(theta)=g0` is a programmatic constraint injected by the CI
  algorithm as a closure, not a user model row.
- *Only new core surface = one thin additive `estimate::frontier` entry point*
  (landed 2026-07-02 for complete-data ML),
  `fit_ml_constrained(pt, rep, samp, x0, extra_nl_closure, bounds, backend)`, that
  appends the caller-supplied nonlinear-equality closure to the partable-derived `nl`
  block before the existing dispatch. The optimizer layer is untouched (SLSQP default,
  IPOPT fallback because SLSQP-based constrained fits are finicky per Falk & Chen). A
  boundary is rejected by the profile helper when the achieved scalar residual has
  `|g(theta_con)-g0| > tol` (exp-43's NULL-if-infeasible guard, promoted). Because it
  is a closure, the whole family can still prototype in the exp-45 R style (`nloptr`
  + a generic `g`-closure); the C++ surface is productization, not the small-sample
  contribution.
- *Functional interface = one value-and-gradient callable* returning
  `{double value; VectorXd grad_theta}` given `(MatrixRep, want_grad)`; analytic
  gradient is the practice default (`dg/d{Lambda,Psi,Phi} . d{blocks}/dtheta`, or
  `dg/dSigma . dsigma/dtheta` reusing the `inference` model Jacobian), empty grad
  triggers an FD fallback. The engine (a) gates the gradient by `want_grad` so
  root-only evaluations skip it, (b) memoizes on `x` so SLSQP's separate `h`/`J_h`
  calls factor the shared inverses (`Sigma^-1`, `(Sigma-gg')^-1`) once, and (c) applies
  the `K'` reduction to driven coordinates itself, so the functional stays in theta
  space and never sees user constraints.
- *Three robustness tiers = one engine with a swappable reference-scaling policy*,
  mapped 1:1 to magmaan's information/covariance regimes. (1) NT: `T ~ chi^2_1` +
  small-sample Bartlett factor (parametric bootstrap or analytic Lawley); the
  load-bearing correction. (2) Robust (non-normal, correct model): `T ~ c*chi^2_1` with
  Satorra-2000 `c` from the ADF `Gamma-hat` (reuses `robust::`); this is
  small-sample-corrected Falk R-LCI. (3) Misspec-robust (research-grade, the JASA
  "completeness" tier): a sandwich/observed-bread reference targeting the pseudo-true
  `g(theta*)` under a wrong model, tying [[misspec-robust-se-weight-influence]] and Lai
  2018 / the estimated-weight-SE thread. For a scalar focal `g` the misspec reference
  is again a scalar scaling `c_misspec` (one eigenvalue of the sandwich pencil),
  structurally like Satorra's `c` with the misspec meat; the genuinely hard part is
  small-sample-correcting *that* (a robust bootstrap under a pseudo-true DGP), which is
  why it is the completeness tier and not the MVP. The Bartlett/small-sample correction
  composes on top of each tier.

**Progress (2026-07-01): NT generic engine validated.** `experiments/45-profile-lr-reliability-ci`
lands the generic-`g` profile-LR engine (a functional closure + `nloptr` constrained
fit, zero core change, exactly the R-prototype path above) and checks it against
`semlbci` (Wu-Neale). Wherever `semlbci` converges the engine reproduces its
likelihood-based bounds to `<=1e-7` for omega_total and `H` (one-factor) and
omega_total and omega_h (orthogonal three-group bifactor). It also carries a
first-principles validity gate (`T(bound) = qchisq(.95,1)`): every engine bound passes
across all cells, while `semlbci`'s default search fails 7 bifactor upper bounds (its
constrained refit sticks at a suboptimal optimum and its post-check re-certifies the
stuck value); the engine's constrained fit is strictly better and its bound is the one
that solves the equation.

**Progress (2026-07-01): coverage + Bartlett, and the estimator problem.** Same
experiment, `scripts/coverage.R`. Coverage of the population value is exactly
`{T(g_pop) <= threshold}`, so only `T(g_pop)` is needed per replication (normal data,
correct model). Findings: (a) for `omega` the plain profile-LR is ALREADY near-exact
(`E[T] ~ 1`, coverage 0.93-0.97), so the correction is a near-no-op; (b) for MAXIMAL
RELIABILITY it collapses: bifactor general-factor `rho*` at N=50 gives `E[T] ~ 4.2` and
coverage `0.61` (the exp-43 inflation, confirmed). The Bartlett rescaling of the
threshold by `c = E[T]` is necessary and, at the ORACLE `c`, sufficient (0.61 -> 0.98).
The crux: estimating `c` feasibly is hard for the near-ceiling functional. A
parametric bootstrap from the fitted model underestimates because a small-N fitted DGP
is "cleaner" than the truth; bootstrapping from the NULL model `theta_tilde(g0)` (the
constrained MLE imposing `g=g0`) is better (0.61 -> 0.89, vs 0.77 for the unconstrained
DGP) but still recovers only ~76% of `c` (stable across N -> looks like a fixable
multiplicative bias). Open feasible-estimator routes: iterated/double bootstrap, or the
analytic Lawley factor. Design refinements this session: the three robustness tiers are
a swappable REFERENCE SCALING = a variance ratio, `c_robust = Var_robust(g)/Var_expected(g)`
and `c_misspec = Var_robust(g)/Var_observed(g)` (sandwich/ADF numerator; expected vs
OBSERVED-Hessian denominator) — VALIDATED: engine `se_model` reproduces lavaan's delta SE
to 5 decimals, and under normal+correct both `c -> 1`, so robust/misspec reduce to NT
(expected-vs-observed info) exactly as expected. The small-sample factor is separate and
multiplies on top. All three bootstraps kept as selectable options (`--boot unc,null,bs`):
the two Gaussian ones (NT) and a distribution-free Bollen-Stine null bootstrap (rotate
data to the null Sigma, case-resample) whose bootstrapped `E[T]` folds the Satorra
scaling and the Bartlett inflation into one number under non-normality; Bollen-Stine
reproduces the Gaussian null bootstrap on normal data (validated). Non-normal stress and
the feasible-`c` fix are the next runs. Still deferred: analytic functional gradients.

**Progress (2026-07-01): analytic Lawley factor investigated -> insufficient for the one
case that needs it (the decisive result).** `scripts/lawley.R` decomposes the mean
inflation `E[T]` with a LOCAL-GAUSSIAN comparator that is exactly the analytic curvature
(Lawley-type) factor evaluated by simulation: draw `theta_hat ~ N(theta_pop, A^-1/N)` and
set `W = N * min_{g(theta)=g_pop}(theta-theta_hat)' A (theta-theta_hat)` — a quadratic
likelihood + asymptotic-normal `theta_hat` but the EXACT nonlinear constraint, so it
carries the functional's constraint-surface curvature to all orders and nothing else. The
profile-LR is reparameterization-invariant, so `E[T] != 1` is a property of the constraint
SURFACE `{g=g_pop}` inside the model manifold, not of `g`'s scaling. Validated: a linear
control functional gives local-Gauss `E[W] = 1` exactly. Findings (bifactor, 1500 reps /
2500 draws): (a) the mild members (`omega`, `omega_h`, and 1-factor `H`) are pure smooth
`O(1/N)`: their whole inflation is the generic model term (linear control `E[T] ~ 1.12` at
N=50, halving to `~1.045` at N=100), they are already near-nominal, and need no correction;
(b) bifactor `rho*` is BOUNDARY-DOMINATED at every N: curvature recovers only 19% (N=50) ->
25% (N=100) of the inflation (oracle `E[T]` 5.04 -> 2.92; curvature 1.78 -> 1.47), barely
improving with N. The near-Heywood quartile (smallest fitted uniqueness) sits at
`E[T|near] = 7.4 / 5.4` vs the interior quartile `E[T|far] = 2.9 / 2.1`
(`cor(T, min psi_hat) ~ -0.3`); even the INTERIOR level (2.9) is far below the overall
mean (5.0), and the correct Bartlett constant is set by the boundary mass. The
model contrast is the clincher: SAME functional form (maximal reliability) at the SAME
`g_pop ~ 0.8-0.9` gives 1-factor `H` a near-nominal `E[T] = 1.15` but bifactor `rho*`
`5.04` — the 40x is the bifactor's competing factors making fitted uniquenesses swing
near-singular at small N, which detonates `lambda' E^-1 lambda`. **Verdict:** a smooth
`O(1/N)` analytic factor (curvature-only OR a full Lawley with the model 3rd/4th-cumulant
term, whose smooth ceiling the interior-quartile `E[T]` proxies at 2.9 << 5.0) is
structurally capped near the interior level and cannot hit the boundary-driven overall
mean; only a method that refits the real (non-quadratic) likelihood and reproduces the
near-boundary mass can — which is exactly why the null-model bootstrap wins on `rho*` and
recovers more than curvature. So the correction for the hard member must be ESTIMATED
(bootstrapped), not COMPUTED. This RE-FRAMES the open feasible-`c` problem from
"analytic vs bootstrap" to "make the bootstrap boundary-aware" (iterated/double bootstrap,
or a DGP that does not clean up the near-Heywood mass). Caveat: local-Gauss isolates
curvature and omits the model-cumulant term, so it lower-bounds analytic-achievable; the
interior-quartile proxy shows the shortfall is fundamental regardless. Files:
`scripts/lawley.R`, `results/lawley_bf.csv`, `results/lawley_high.csv`.

**Progress (2026-07-01): double bootstrap + the CONSTANT-vs-per-sample finding (the
correction must be a calibrated constant).** First, ENABLER: added closed-form analytic
z-space gradients to the reliability functionals (`omega_total`, `omega_h`, `H`/`rho*`),
validated vs central FD to ~1e-11 (`for H`: `dq/dlambda_t=2w`, `dq/dL_{.,j}=-2w(L[,j]'w)`,
`dq/dpsi_i=-w_i^2`, `w=E^-1 lambda`); the engine already skips FD when a functional supplies
its grad, so every constrained fit now uses an analytic constraint Jacobian -> 11x speedup
(double-boot smoke 434s -> 38s), which is what makes nested bootstraps affordable. Then the
DOUBLE BOOTSTRAP (`scripts/double_boot.R`, motivated by the stable ~76% one-level recovery =
multiplicative-bias fingerprint): it correctly fixes the MEAN of `c` (60% -> 72% additive ->
82% multiplicative, exactly as `rho`-constant theory predicts), BUT coverage does NOT improve
(0.85 plug-in -> 0.83/0.84 double) because coverage never depended on the mean. Decisive
diagnostic: (a) `cor(T_real, c) = -0.19/-0.15` -> the per-sample bootstrap factor is
ANTI-CORRELATED with need (smallest `c` exactly on the near-boundary samples where `T` is
biggest); (b) the double bootstrap INFLATES the per-sample CV (0.41 -> 0.72); (c) a CONSTANT
factor beats a per-sample one at every mean level (`c_plugin`: per-sample 0.85 vs constant
0.898; `c_mult`: per-sample 0.84 vs constant 0.934; oracle constant `c=5.04` -> 0.976); (d)
bootstrap size is NOT the lever: `cov_bart_null = 0.803` IDENTICAL at `B in {40,120,400}`, so
the per-sample ceiling (~0.80-0.85) is structural (theta_tilde variation + anti-correlation),
not Monte-Carlo. VERDICT: the small-sample factor for the boundary-dominated members must be a
STABLE MODEL-LEVEL CONSTANT (calibrated by simulation over the model class), NOT a smooth
analytic formula (misses the boundary) and NOT a per-dataset resampling estimate (anti-
correlated, high-variance). This is a cleaner and more publishable message than "double-boot
fixes it": the per-sample-bootstrap paradigm itself is the wrong frame for near-boundary
coefficients. OPEN: a low-variance calibrated-constant recipe usable from one dataset (the
point-estimate near-Heywood bias is the deeper residual); the double-boot machinery + analytic
grads are kept as validated infrastructure. Report gained the finding "Why the correction must
be a calibrated constant" (constant-vs-per-sample coverage table). Files: `scripts/double_boot.R`,
`results/double_boot_bf_n50.csv`, faster `R/functionals.R`.

**Progress (2026-07-02): first C++ productization slice.** The formerly build-if-only
core hook now exists for complete-data ML:
`estimate::frontier::fit_ml_constrained` accepts extra programmatic nonlinear equality
closures in θ-space, appends them to any partable nonlinear `==` rows, and reuses the
existing SLSQP/IPOPT constrained scalar optimizer. `profile_lrt_scalar_ml` wraps that
as a df-1 ordinary profile-LR test for `g(θ)=g0`; `profile_lrt_parameter_ml` is the
unit-gradient parameter special case. This is not a solution to the hard funLR problem:
there is no LS/ordinal constrained-fit counterpart yet, no robust/Satorra scalar
reference scaling, no Bartlett/calibrated-constant policy, no CI inversion wrapper, and
the initial landing had no R exposure. Unit tests gate constraint satisfaction and the
`2N·Δfmin` statistic.

**Progress (2026-07-03): fixed-weight GMM scalar/profile-LRT seed.** The continuous
moment-quadratic sibling now exists:
`estimate::frontier::fit_gmm_constrained` scalarizes a caller-fixed GMM/LS objective,
appends the same programmatic equality closure, and uses SLSQP/IPOPT. The scalar and
parameter helpers (`profile_lrt_scalar_gmm`, `profile_lrt_parameter_gmm`) report the
ordinary df-1 `2N·Δfmin` statistic; R exposes the parameter slice as
`magmaan_core$frontier_profile_lrt_parameter_gmm()`. This removes the cheap continuous
LS infrastructure gap, but it is still not the ordinal/fitted-functional solution:
robust/Satorra scaling, Bartlett/calibrated constants, functional callbacks, and an
ordinal DWLS functional constraint path remain open.

**Progress (2026-07-03): fitted-weight parameter profile + CI inversion seed.** The
continuous moment-quadratic parameter lane now has a fitted-weight optimizer policy:
`fit_gmm_fitted_weight` / `fit_gmm_fitted_weight_constrained` refresh the
expected-information weight `W(theta)` in an outer fixed-point loop, and
`profile_lrt_parameter_gmm_fitted_weight` compares the unrestricted and constrained
fits under that same policy. The first test-inversion layer is also in core:
`profile_lrt_ci_parameter_ml`, `profile_lrt_ci_parameter_gmm`, and
`profile_lrt_ci_parameter_gmm_fitted_weight` bracket and bisect the ordinary df-1
profile statistic against the chi-square cutoff, returning endpoint diagnostics and
the endpoint constrained fits. R exposes these parameter-only wrappers through
`magmaan_core$frontier_profile_lrt_parameter_gmm_fitted_weight` and
`magmaan_core$frontier_profile_lrt_ci_parameter_*`. This is still infrastructure, not
the funLR scientific gate: robust/misspec policy for weight-derivative regimes,
Bartlett/calibrated constants, functional R callbacks, and ordinal DWLS
fitted-functional constraints are still open.

**Progress (2026-07-03): continuous parameter robust scaling seed.** Complete-data ML
and caller-fixed continuous GMM parameter profile LRT/CI now accept complete raw data
for empirical 1-df sandwich scaling. The constrained result keeps ordinary `T` and
`p_value`, adds `scaling_factor`, `T_scaled`, and `p_value_scaled`, and the CI inverter
can root on `ScalarProfileReference::RobustScaled`. ML reuses
`robust::param_space_sandwich`; fixed-weight GMM reuses
`continuous_ls_param_space_sandwich` with the caller-supplied weight. R exposes the
option as `raw_data=` plus `robust=TRUE` for the ML/GMM parameter LRT and CI helpers,
and the fixed-weight profile example smokes the scaled LRT/CI path.

**Progress (2026-07-03): fitted-weight profile-metric robust scaling seed.** The
fitted-weight GMM parameter profile LRT/CI now accepts the same raw-data robust path.
At each constrained endpoint it rebuilds the final expected-information weight
`W(theta)` and computes the empirical 1-df sandwich scale for that profile metric,
then roots CIs on `T_scaled` when requested. This closes the cheap asymptotic-robust
parameter proving slice across ML, fixed-weight GMM, and fitted-weight GMM, but not
the hard funLR gate: complete weight-derivative / misspec references, functional
callbacks, ordinal fitted-functionals, and Bartlett/calibrated constants remain open.
Experiment 47 now carries the paired empirical probe: multivariate `t(5)` data,
ULS/GMM loading CI inversion, fixed versus fitted profile weights, and ordinary
versus robust-scaled references over `N in {100,200,500}`.

**Progress (2026-07-03): fixed-profile estimated-weight robust scaling.**
Caller-fixed continuous GMM parameter profile LRT/CI can now request
estimated-weight robust scaling with complete raw data. The new
`GmmProfileRobustOptions` switch leaves ordinary `T` unchanged, but routes the
1-df scale through `continuous_ls_param_space_sandwich_ij`: GLS uses the
sample-normal-theory IJ weight mode, WLS uses the empirical-WLS IJ weight mode,
and ULS remains the fixed-weight reference. R exposes this as
`estimated_weight=TRUE` on the fixed GMM parameter LRT/CI wrappers, requiring
`robust=TRUE` and `raw_data`. Experiment 47 now keeps the ULS fixed/fitted
comparison and adds GLS/WLS fixed-profile rows for `ordinary`, `robust_fixed`,
and `robust_estimated_weight`. This advances the continuous non-normal stress
slice, but does not solve the funLR gate: fitted-weight derivative/misspec
references, scalar functionals, ordinal fitted-functionals, and
Bartlett/calibrated constants remain open.

**Progress (2026-07-03): model-misspec profile references.** The scalar profile
LRT/CI surface now has explicit reference modes rather than overloading
`robust`: ordinary chi-square, Satorra-style robust scaled, misspec scaled, and
misspec mixture. Results expose both ordinary and reference-specific fields,
including observed-bread misspec scale, the singleton mixture eigenvalue, and
endpoint-specific mixture cutoffs for CI inversion. The implementation covers
ML, caller-fixed continuous GMM, fitted-weight continuous GMM, direct FIML,
ML2S-NT, ordinal and mixed-ordinal parameter profiles, and the no-integration
ordinal polychoric-omega functional. Continuous fixed GLS/WLS can route misspec
meat through the IJ estimated-weight correction; ordinal and mixed ordinal use
the complete IJ meat, including fitted DWLS/WLS weight influence. FIML uses
observed-pattern score meat, ML2S-NT uses the saturated EM moment sandwich, and
both switch to observed bread for misspec references. The fitted-weight
continuous path is explicitly profile-metric only: it rebuilds endpoint
`W(theta)` and treats it as fixed, so derivative-of-weight misspec corrections
remain a research extension. R exposes the choice as `reference=` and examples
smoke the GMM, FIML/ML2S/mixed ordinal, and ordinal omega paths. This closes the
model-misspec asymptotic reference slice, but still not the hard funLR gate:
model-level small-sample/Bartlett/calibrated constants and richer functional
targets remain open.

**Progress (2026-07-02): R-facing parameter probe.** The complete-data ML parameter
special case is exposed as `magmaan_core$frontier_profile_lrt_parameter_ml()`,
with experiment 47 (`experiments/47-ml-parameter-profile-lrt`) as the small-N
interior-parameter calibration check. At 200 reps over `N in {30,50,100,500}`,
the ordinary `chi^2_1` reference is stable for the free loading baseline
(`type1_05` 0.050-0.055, zero constrained-solve failures). This remains only
the proving slice for the C++ seed; bounded functionals, ordinal constrained
fits, robust/misspec scaling, and small-sample constants remain open.

**Open next steps (as of 2026-07-03), roughly in priority order:**

1. **Calibrated-constant recipe from one dataset** (the "what do we ship" question, now the
   crux). The factor must be a stable model-level constant; the per-dataset bootstrap is
   anti-correlated with need. Candidates: (a) a big bootstrap from a *regularized* / de-biased
   theta_hat that breaks the T-vs-c anti-correlation; (b) a precomputed calibration surface
   `c(model structure, N, functional)` (a Bartlett table) applied at the fitted structure;
   (c) shrinkage of the per-sample factor toward a model-class prior. Metric = coverage, not
   mean-c recovery (coverage is the robust target; means are tail-noisy).
2. **Non-normal stress run** (the queued breadth item; makes the robust/misspec tier do real
   work). Under non-normal data `c_robust`/`c_misspec` finally diverge from 1 and from each
   other, and the "constant factor" becomes a Satorra-2000-scaled constant; tests whether the
   constant-vs-per-sample verdict survives leaving NT. Reuse `scripts/coverage.R` with a
   non-normal generator; exercise `--boot unc,null,bs --scalings`.
3. **Point-estimate near-Heywood bias** = the deeper residual under-coverage source (even the
   oracle constant leaves a small gap at N<=30 in the KC lane; the near-boundary point bias of
   `rho*` mirrors it). Ties [[small-sample-df-coverage-lane]].
4. **Reliability-difference two-parameter confidence regions** (Pek-Wu sec 3; ties
   [[exp20-deng-chan-alpha-omega]]).
5. **Extend the C++ seed beyond the current exposed functionals:** richer
   functional R callbacks; observed-score / Green-Yang-like ordinal omega
   targets; fitted-weight derivative-of-weight misspec corrections if an
   experiment needs them; then expose additional ordinal/fitted-functional
   surfaces only once an experiment consumes them.
6. **Done / infrastructure (do not redo):** NT generic-g engine + semlbci validation; coverage
   + Bartlett characterization; analytic Lawley ruled out (boundary); double bootstrap ruled out
   (anti-correlation); analytic functional gradients (11x speedup, validated ~1e-11);
   complete-data ML `fit_ml_constrained` / scalar and parameter profile-LR seed;
   fixed-weight and fitted-weight GMM scalar/parameter profile seeds; parameter
   CI inversion; complete-data ML, fixed-weight GMM, fitted-weight GMM, and
   fixed-profile estimated-weight GLS/WLS parameter robust scaling; model-misspec
   scaled/mixture references for ML, continuous GMM, ordinal parameters, and
   ordinal polychoric omega; R parameter/omega wrappers + experiment-47
   interior-parameter calibration probe.

**Progress (2026-07-04): ordinal finite-sample calibration lane scoped.** The
general Bartlett problem is now split from a specific categorical-limited-information
taxonomy. Working note `docs/research/notes/ordinal_profile_lrt_finite_sample_calibration.tex`
defines the ordinal DWLS/WLS problem as a decomposition into LR inflation,
reference-scale error, sparse-threshold/NACOV instability, omega point bias, and
boundary/feasibility mass. Experiment 53
(`experiments/53-ordinal-profile-lrt-calibration`) is the first taxonomy run:
all-ordinal one-factor probit data, polychoric/latent-response omega target,
DWLS ordinary / robust-scaled / misspec-mixture profile tests, and the fit-free
ordinal polychoric omega Wald interval as the baseline. The WLS robust-scaled
comparison is deferred so the main grid can spend its budget on DWLS
replications. The smoke is intentionally diagnostic only; the decision rule for
the full grid is: scalar constants advance only if the reference-scaled
statistic is the failure mode, otherwise correction effort moves to ordinal
summary regularization or target-bias diagnostics.

First full DWLS-only grid (1000 reps/cell, `N = 50,100,250,500`) says ordinary
DWLS is the wrong reference for this functional (coverage about 0.60-0.69;
mean unscaled statistic about 3.5-6.4), while endpoint robust and
misspecification-mixture references are near nominal in all but the sparse
threshold-extreme `N=50` cell. That cell still has profile failures and huge
fitted-weight scale instability, so the next useful probe is sparse ordinal
summary / weight regularization rather than a universal Bartlett factor.

Misspecified extension (same grid, local residual correlations in the
latent-response DGP but the same fitted one-factor DWLS model) targets the
large-sample pseudo-true fitted profile omega. The conclusion survives:
ordinary DWLS covers about 0.59-0.69, robust-scaled covers about 0.94-0.97, and
misspecification-mixture covers about 0.94-0.97. The local-dependence target is
about 0.0023-0.0025 above the fit-free sample-polychoric omega target, so the
report now separates profile-target coverage from the Wald/sample-polychoric
diagnostic. The remaining weak point is still sparse threshold-extreme `N=50`,
not model misspecification per se.

CI inversion probe (200 reps/cell, `--ci`, robust-scaled + misspec-mixture):
robust-scaled inversion is stable enough to treat as the v1 interval lane
(3168 attempts, 38 failures, zero target-test/CI disagreements, coverage
0.918-0.975). Misspecification-mixture inversion is not stable yet (3154
attempts, 1201 failures, mostly non-positive endpoint robust profile
quadratics; one target-test/CI disagreement among successful rows). Keep
misspec-mixture as a pointwise reference-law diagnostic until endpoint
quadratic/reference construction is hardened for CI roots.

Reference set (PDFs collected in `papers/closed-form-omega/extern/`, several mirrored
in `external/refs/`): Pek & Wu 2015 (`10.1007/s11336-015-9461-1`), Wu & Neale 2012
(`10.1007/s10519-012-9560-z`), Cheung 2009 (`10.1080/10705510902751291`), Cheung &
Pesigan 2023 / `semlbci` (`10.1080/10705511.2023.2183860`), Falk 2018
(`10.1080/10705511.2017.1367254`), Falk & Chen 2026 (`10.1080/10705511.2025.2555612`),
Pritikin, Rappaport & Neale 2017 (`10.1080/10705511.2016.1275969`), Satorra 2000
(`10.1007/978-1-4615-4603-0_17`), Lawley 1956 (`10.1093/biomet/43.3-4.295`), Yuan,
Tian & Yanagihara 2015 (`10.1007/s11336-013-9386-5`), Lai 2018
(`10.1080/10705511.2018.1505522`, robust SE under misspecification, adjacent).

### Structural-model fit indices, tests, and CIs (two-step / SAM)

A `measures::frontier` post-fit surface that evaluates the fit of *just the
structural model* of a full SEM, isolated from measurement misfit, via the
Hancock-Mueller two-step (a special case of joint-measurement local SAM): Step 1
fits the measurement model with a saturated latent covariance to get Φ̂, Step 2
fits the hypothesized structural model treating the latents as observed. The
deliverables are a structural chi-square test, structural RMSEA/CFI/SRMR, and
CIs for all three. Zhang & Wu (2024, *SEM* 31:5, 863-881; PDF in
`external/refs/`, eval in
[paper-evals](../research/paper-evals/2024-zhang-structural-model-fit.md)) show
the naive structural chi-square has Type-I error above 80% in small
low-reliability samples because Φ̂ is not Wishart (the statistic is a mixture of
1-df chi-squares, not χ²_df), and supply a Satorra-Bentler-type mean correction
T_HS,C = T_HS/ĉ_HS (eq. 24), corrected indices (eqs. 25-27), and the first
structural fit-index CIs (Welch/delta, Table 2). Their recommended versions V2
and V5 use the expected information of the saturated structural model in Step 2.

**Downstream of the SAM entry.** Requires the SAM two-step scaffolding above
(Step-1 saturated-latent fit producing Φ̂ and its asymptotic covariance Γ̂_SS,
then a structural Step-2 fit). Builds on machinery magmaan already has:
`inference` observed/expected/first-order information, the `robust`
Satorra-Bentler scaling family, and `measures` full-SEM RMSEA/CFI/SRMR +
baseline.

**Alternative already available.** lavaan's original two-step HS indices
(RMSEA_HS, CFI_HS, naive-BC SRMR_HS) and their CIs cover the *uncorrected* case;
the authors' OSF code (osf.io/dcj2z) covers the *corrected* versions on a lavaan
refit. magmaan's full-SEM GOF surface evaluates the whole model, not the
structural part in isolation.

**Build if.** The SAM two-step lands *and* a methods workflow or paper needs
isolated structural GOF with CIs (the small-N inference or
SAM-under-misspecification studies are the natural home). Notes: the structural
scaling ĉ_HS needs Γ̂_SS = the W_SS⁻¹ block, which the authors flag as
unspecified future work (not pure transcription); gate the corrected versions
transitively or first-principles since only the original HS versions have a
lavaan oracle; structural CFI needs an uncorrelated-latent structural baseline,
where magmaan's [[fixedx-baseline-df-gap]] would bite. Over-the-top extension:
the structural chi-square is a chi-square mixture and ĉ_HS matches only its first
moment, so the FMG / pEBA spectrum machinery in `papers/fiml-fmg/` gives a
sharper "structural-FMG" GOF test if a paper wants more than the mean correction.

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
