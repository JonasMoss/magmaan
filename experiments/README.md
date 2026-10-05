# Experiments

Experiments are grouped by purpose. Research and engineering also separate
**active work**, **banked ideas**, and **retained evidence**. Numbers are stable
across activity moves; cite the slug and its current path. Archived folders
have no numeric prefix. Conventions are in [AGENTS.md](AGENTS.md).

For new work, follow [Where new work belongs](AGENTS.md#where-new-work-belongs):
search for an existing question first, choose the purpose and activity, allocate
a new category number above those already assigned, and register the study before
its first run. Add runs or method lanes to an existing leaf when they serve the
same question; speculative ideas start in the trigger register.

| Purpose | Studies | Activity |
|---------|--------:|----------|
| [Decisions](#decisions) | 4 | Maintained registers and held-out evidence for library defaults. |
| [Showcases](#showcases) | 8 | Capabilities, parity and performance demonstrations. |
| [Replications](#replications-and-reference-studies) | 8 | Published results and reference reconstructions. |
| [Research](#research) | 24 | 8 active, 11 banked, 5 retained evidence studies. |
| [Engineering](#engineering-checks) | 8 | 4 active, 3 banked, 1 completed paper-evidence study. |
| [Archive](#archived) | 15 | Settled investigations and predecessors, with sources/results retained. |

## Decisions

Pre-registered studies whose results set library defaults; see
[AGENTS.md](AGENTS.md#decisions-set-library-defaults) for the standard.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [optimizer-defaults](decisions/01-optimizer-defaults/report.qmd) | benchmark | active | Which start, optimizer and PSD route should each estimation route use by default, judged by the library verdict on held-out simulated problems? |
| 02 | [barrier-defaults](decisions/02-barrier-defaults/report.qmd) | benchmark | active | Which start and optimizer should the complete-data ML barrier fitter use by default, judged by the library verdict on held-out simulated problems? |
| 03 | [score-centering](decisions/03-score-centering/report.qmd) | probe | banked | Where do uncentered, globally centered or within-group likelihood-score meats give justified parameter covariance and analytic score calibration for ML and prospective FIML? |
| 04 | [nested-ml-geometry](decisions/04-nested-ml-geometry/report.qmd) | probe | active | Does observed nested ML geometry calibrate at least as well as expected geometry, especially under larger-model misspecification? Saturated-mean confirmation frozen; structured-mean production (2026-10-05) frozen with no flag: the exact-row LR meat is never worse than the pre-66 meat. |
| 05 | [dwls-policy-calibration](decisions/05-dwls-policy-calibration/report.qmd) | probe | active | How well calibrated are the adopted DWLS policy components? TASK-79 adds fresh-draw exact/OPG all-ordinal reconfirmation with paired intervals and a local full-grid pricing pilot (production awaits registration); TASK-84 owns latent non-normality. TASK-80 adds a registered mixed continuous/ordinal lane, with a full-grid pricing pilot and production pending merger review. All-ordinal evidence: Production (146 cells, Modal), a 13-reference exploration and two fresh-seed confirmations: IJ covariance calibrated; global test uses the exact spectrum (All) tail; nested test uses the r-term parameter-space law after the profile law's cancellation artifact was diagnosed. Reopen for heavier-tailed latent responses or threshold-invariance restriction maps. |
| 06 | [ordinal-threshold-invariance](decisions/06-ordinal-threshold-invariance/report.qmd) | probe | active | Does the moment-nested DWLS threshold-invariance test retain nominal size? Production 2026-10-05 frozen: SB/PEBA4 over-reject with seven categories, All calibrated everywhere; All confirmed on fresh draws; next: the user's reference choice (TASK-81). |

**Score-centering bank:** retain raw for tested regular ML and as the prospective
FIML comparator; 72,000 confirming datasets establish no qualifying global-centering
benefit and a separate fixed-allocation covariance identity. Criteria and evidence
remain in the decision study. No queued comparison; reopen for a named component
with a demonstrated covariance defect or centering-specific benefit under the
[likelihood-score trigger](../project/backlog/speculative.md#likelihood-score-centering-alternatives).
FIML calibration and interval validation remain active in the primary backlog.

## Showcases

Demonstrate magmaan capabilities, agreement with lavaan/other references, or a concrete performance comparison.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [lavaan-parity](showcases/01-lavaan-parity/report.md) | parity | active | Does magmaan's ML inference match lavaan across the textbook corpus? |
| 02 | [lavaan-speed-bench](showcases/02-lavaan-speed-bench/report.md) | benchmark | active | How much faster is magmaan than lavaan on the same prepared-input CFA fit? |
| 03 | [foldnes-moss-gronneberg-peba](showcases/03-foldnes-moss-gronneberg-peba/report.qmd) | replication | complete | Do the penalized EBA goodness-of-fit tests hold nominal Type I under nonnormality? |
| 04 | [foldnes-moss-gronneberg-2026](showcases/04-foldnes-moss-gronneberg-2026/report.qmd) | replication | active | Can magmaan reproduce the full FMG goodness-of-fit machinery versus lavaan and semTests? Local lavaan-parity output is absent; recover it before rendering. |
| 06 | [speed-attribution](showcases/06-speed-attribution/report.qmd) | benchmark | active | How much of matched ML + GOF/Wald runtime is preparation, fitting, or inference, and which comparisons pass numerical agreement before a public speed claim? |
| 07 | [scalar-intervals](showcases/07-scalar-intervals/report.qmd) | probe | complete | How do Wald, score and profile-LR scalar intervals compare under normal and nonnormal sampling, accounting for failures and cost? |
| 08 | [reliability-targets](showcases/08-reliability-targets/report.qmd) | benchmark | complete | Which population quantity do covariance coefficients and fitted omega target, and how does misspecification change that target? |
| 09 | [robust-test-calibration](showcases/09-robust-test-calibration/report.qmd) | benchmark | complete | How do robust score and LR tests calibrate across models and DGPs, and what do spectral and correctness controls explain? |

## Replications and reference studies

Reproduce published results or reconstruct the reference method needed to interpret them.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [maydeu-olivares-2017](replications/01-maydeu-olivares-2017/report.qmd) | replication | complete | Do the SE methods and χ² adjustments hold under nonnormality (two-factor CFA)? Retained local output has ten attempts per cell; substantive calibration is unconfirmed. |
| 02 | [rhemtulla-2012](replications/02-rhemtulla-2012/report.qmd) | replication | complete | When can ordinal variables be treated as continuous (cat-LS vs continuous ML)? Retained local output has ten attempts per cell; substantive calibration is unconfirmed. |
| 03 | [li-2021-mixed](replications/03-li-2021-mixed/report.qmd) | replication | complete | DWLS or MLR for a mix of continuous and categorical indicators? Retained local output has ten attempts per cell; substantive calibration is unconfirmed. |
| 04 | [li-2016-ordinal](replications/04-li-2016-ordinal/report.qmd) | replication | complete | DWLS/ULS or continuous ML for an all-ordinal five-factor SEM? Retained local output is ten-rep smoke without parity; recover substantive evidence before quoting rankings. |
| 05 | [chen-2020-wlsmv-pd](replications/05-chen-2020-wlsmv-pd/report.qmd) | replication | active | Can the Chen et al. (2020) WLSMV_PD Type-I inflation cell be reproduced with current lavaan WLSMV pairwise deletion? |
| 06 | [jamil-rosseel-2026-rbm-sem](replications/06-jamil-rosseel-2026-rbm-sem/report.qmd) | replication | active | Can we reproduce the SEM reduced-bias paper's two-factor and growth-curve RBM examples from the authors' OSF outputs before magmaan-owned reruns? |
| 07 | [foldnes-moss-gronneberg-2026-study2](replications/07-foldnes-moss-gronneberg-2026-study2/report.qmd) | replication | complete | Can native magmaan express the 189-cell Study 2 weak-invariance Type-I comparison, including the unbiased-Gamma variants, with semTests retained only as a parity sentinel? Translation endpoint retained; local run artifacts are absent. |
| 08 | [Savalei-Falk-2014 test conventions](replications/08-savalei-falk-2014-test-conventions/report.qmd) | replication | active | Which written conventions and numerical fingerprints explain the robust FIML/ML2S tests, and what same-data matrix/EQS evidence is still needed for exact identity? |

## Research

Investigate statistical behavior, new methods, estimands and inferential validity.

### Active research (7)

Concrete remaining checks and protected ongoing paper pipelines.

| # | Experiment | Kind | Lifecycle | Question | Finding / next check or reopening trigger |
|--:|------------|------|-----------|----------|------------------------------------------|
| 06 | [fiml-invariance-tests](research/active/06-fiml-invariance-tests/report.qmd) | paper-sim | active | How do difference and direct-score tests calibrate nested measurement-invariance restrictions under incomplete nonnormal data? | Six-indicator 5,000-rep null study plus separate score screen/confirmation, pilot power and oracle lanes. Finish valid scalar nesting, larger-model nulls and size-matched power; multiplier expansion is banked. |
| 21 | [ml-parameter-profile-lrt](research/active/21-ml-parameter-profile-lrt/report.qmd) | probe | active | Do ML and fixed-weight continuous-GMM scalar profiles calibrate, including nonnormal complete-IJ coverage? | Retain ML and nonnormal fixed-weight GLS/WLS/ULS profiles with complete IJ reference; assess coverage separately from the removed fitted-weight arm. |
| 22 | [ordinal-functional-inference](research/active/22-ordinal-functional-inference/report.qmd) | probe | active | How do delta and fitted-profile intervals behave for their distinct ordinal functional targets? | Retain 200-rep delta coverage/stress and separate profile grids. Audit sparse-summary and endpoint failures, connectedness and independent pseudo-target precision; pointwise acceptance is not CI availability. |
| 43 | [psd-ml-repair-risk](research/active/43-psd-ml-repair-risk/report.qmd) | benchmark | active | What does PSD-ML repair when ordinary NTML leaves the primitive covariance domain, and what is its near-boundary estimation risk? | Likelihood cost and estimation risk differ across improper-solution types; extend the pilot boundary path without treating repair cost as a test. |
| 47 | [multiinfo-penalty-improper](research/active/47-multiinfo-penalty-improper/report.qmd) | benchmark | active | Do the joint and latent-determinacy (Q) barriers remove improper solutions without costing PSD-constrained ML's accuracy, also with the truth on a face, which fit test goes with a barrier estimate, and does the barrier's local limit hold? | Paper pipeline: barrier properness trades risk near a face; complete boundary/path and fit-test checks rather than selecting a weight from pooled RMSE. |
| 48 | [multiinfo-jeffreys-posterior](research/active/48-multiinfo-jeffreys-posterior/report.qmd) | benchmark | active | Does the multi-information barrier approximate the Jeffreys-prior posterior over admissible solutions, for which weight, and is that posterior a better estimator? | Paper pipeline: posterior accuracy depends on functional, prior and ESS; recover substantive outputs and finish joint/Q/path comparisons (local main run: smoke; posterior rankings remain exploratory). |
| 49 | [spectral-tail-calibration](research/active/49-spectral-tail-calibration/report.qmd) | benchmark | active | Can spectral shrinkage and debiased moment corrections improve score and LRT calibration under non-normality? | IG oracle spectra can calibrate while estimated moments fail; investigate heterogeneous spectral estimation bias and failure conditioning. |

### Banked research ideas (12)

No queued run. Reopen for the named consumer or validation trigger.

| # | Experiment | Kind | Lifecycle | Question | Finding / next check or reopening trigger |
|--:|------------|------|-----------|----------|------------------------------------------|
| 03 | [deng-chan-2017-alpha-omega](research/banked/03-deng-chan-2017-alpha-omega/report.qmd) | paper-sim | banked | What follows from the nonregular alpha–omega contrast at tau-equivalence? | Keep second-order, nonnormal delta and ULS derivations; eight-rep main smoke cannot establish size. Reopen for a named nonregular-functional consumer/methods paper; existing structural equal-loading tests are cheaper. |
| 09 | [rmsea-like-catml-dwls](research/banked/09-rmsea-like-catml-dwls/report.qmd) | probe | banked | Does lavaan's categorical robust RMSEA behave as a consistent CATML-at-DWLS criterion-at-estimator statistic under ordinal misspecification? | CATML-at-DWLS targets a hybrid criterion; reopen for a named RMSEA consumer and a large-N pseudo-target check (local run: smoke). |
| 11 | [chen-ordinal-fmg-pvalues](research/banked/11-chen-ordinal-fmg-pvalues/report.qmd) | probe | banked | Is the WLSMV pairwise-missing scalar Type-I inflation a defect of the test/p-value family or of the missing-data mechanism (MCAR vs MAR)? | Reported MAR inflation points to biased polychorics; reopen for stronger missingness/bias diagnostics and recover the substantive run (local output: 3 reps). |
| 18 | [bounded-functional-intervals](research/banked/18-bounded-functional-intervals/report.qmd) | probe | banked | Which delta, transformed and profile constructions give useful intervals for bounded fitted functionals? | Separate bifactor and profile lanes preserve selected-fit and oracle diagnostics. Reopen for a named consumer with independent operational correction and actual endpoint validation. |
| 19 | [covariance-functional-intervals](research/banked/19-covariance-functional-intervals/report.qmd) | benchmark | banked | Can effective-df references and transforms address distinct small-sample interval errors for covariance functionals? | Correlation/alpha are demonstrations; 300-rep limited grid does not establish distribution-free coverage. Reopen for a named interval consumer and held-out heavy-tail/boundary confirmation. |
| 31 | [permutation-measurement-invariance](research/banked/31-permutation-measurement-invariance/report.qmd) | probe | banked | Can fully recomputed pivotal score or Wald label permutations calibrate continuous metric-invariance tests under heterogeneous group populations, and can a Hotelling reference change the permutation-score test? | Score permutation improves the unequal-allocation pivot; reopen for a named ordinal-invariance consumer after estimator-specific influence/pivotality validation. The continuous result does not establish ordinal permutation validity. |
| 32 | [standardized-flip-score](research/banked/32-standardized-flip-score/report.qmd) | probe | banked | Do nuisance-effective and flip-specifically standardized score flips calibrate continuous metric-invariance tests at small N? | Centered Mammen is promising; reopen for fresh confirmation beyond complete-data affine rank-eight ML or a concrete meat-shrinkage consumer. |
| 35 | [residual-flip-gof-probe](research/banked/35-residual-flip-gof-probe/report.qmd) | probe | banked | Can efficient residual-score flipping calibrate single-model SEM goodness of fit without an ad hoc regularizer? | Residual-score flip algebra works, but high rank causes failure/miscalibration; reopen for a low/moderate-rank consumer or a separately justified shrinkage method. |
| 40 | [finite-sample-gof](research/banked/40-finite-sample-gof/report.qmd) | probe | banked | Can Hotelling or projected Satterthwaite references improve finite-sample robust-score calibration? | Separate residual-rank/DGP lanes; reopen for a named GOF consumer with independently validated nuisance influence and moments. |
| 44 | [fiml-global-tests](research/banked/44-fiml-global-tests/report.qmd) | probe | banked | Which score and LR/D global tests calibrate FIML and ML2S, and which sensitivity/metric geometry survives missing-data pseudo-nulls? | Adopted observed-H0 sensitivity/expected metric for the FIML policy (2026-10-02): consistent under MAR, where expected-H0 fails asymptotically, but conservative as df grows (0.0–0.7% pEBA4 at df 87, N=200). Reopen for a finite-sample correction of the observed reference or a consumer needing FIML score power at high df/N. Older /04 and /05 designs are banked lanes with ten-rep local outputs. |
| 46 | [latent-metric-geometry](research/banked/46-latent-metric-geometry/report.qmd) | benchmark | banked | Which latent scaling convention should a SEM library use internally, is there a better one than the named three, and does the answer survive the move from CFA to structural models? | CFA chart conditioning does not settle structural scaling; reopen for matched starts and genuinely near-Heywood structural populations (local run: smoke). |
| 53 | [rbm-estimation-risk](research/banked/53-rbm-estimation-risk/report.qmd) | probe | banked | When do post-hoc and integrated reduced-bias corrections improve error, and when do tails or fit failures erase that gain? | 5,000-rep N=30 evidence exposes correction tails; reopen for a concrete RBM admissibility/estimation-risk consumer or nonnormal/FIML extension. |

### Retained research evidence (5)

Completed reference answers and frozen studies, with their scope limits.

| # | Experiment | Kind | Lifecycle | Question | Finding / next check or reopening trigger |
|--:|------------|------|-----------|----------|------------------------------------------|
| 10 | [mplus-demo-wlsmv-difftest](research/evidence/10-mplus-demo-wlsmv-difftest/report.qmd) | probe | complete | Does Mplus Demo WLSMV DIFFTEST for a demo-sized ordinal pairwise-missing invariance model match lavaan/magmaan Satorra-2000 statistics? | Demo DIFFTEST agrees to rounding after matching scalar constraints; limited demo-sized oracle evidence, with local Mplus needed to rerun. |
| 12 | [misspec-robust-se](research/evidence/12-misspec-robust-se/report.qmd) | probe | complete | Does the observed-Hessian ("robust" regime) bread recover the true sampling SD of ordinal DWLS estimates under structural misspecification, while coinciding with the conventional SE under the null? | Full estimated-weight IJ, rather than bread alone, repairs the misspecified DWLS example; evidence remains local to the stated models/targets. |
| 13 | [ordinal-dwls-profile-lrt](research/evidence/13-ordinal-dwls-profile-lrt/report.md) | paper-sim | complete | Does the standard scaled difference test for nested all-ordinal DWLS models stay calibrated when the larger model is misspecified, and does the estimated-weight profile law restore calibration? | On the exact pseudo-null, fixed-weight rejection rises to 10.5% while estimated-weight stays near 4.2%; retain the native C++/just calibration pipeline and Markdown report (explicit legacy layout exception). |
| 39 | [fiml-information-choice](research/evidence/39-fiml-information-choice/report.qmd) | benchmark | complete | How do FIML information choices calibrate parameter SEs around each estimator’s limit? | Completed 48-cell, 200,000-rep run and oracle sentinels retained independently of global/nested tests. Reopen for an information-contract change, parity defect or named new target regime. |
| 42 | [psd-ml-small-n-convergence](research/evidence/42-psd-ml-small-n-convergence/report.qmd) | benchmark | complete | On the De Jonckere–Rosseel / Ernst small-N SEM design, does covariance-honest NTML converge more reliably than ordinary NTML? | Completed small-N PSD convergence evidence, including equality-KKT telemetry; preserve the paper pipeline and attribution controls. |

## Engineering checks

### Active implementation choices

| # | Study | Kind | Current question and next check |
|--:|-------|------|---------------------------------|
| 13 | [PSD estimator stress](engineering/active/13-psd-estimator-stress/report.qmd) | benchmark | Where do covariance-honest estimators fail under conditioning and boundary stress? Extend the missing-data/categorical smoke anchors under the current verdict; retain all failures and cost. |
| 15 | [Sphere reference fits](engineering/active/15-sphere-reference-fits/report.qmd) | probe | Which failures are missed finite minima, accuracy failures or requested-chart poles? Preserve the witnesses; validate chart rejection and independent PSD fallback separately. |
| 17 | [Corpus optimizer recovery](engineering/active/17-corpus-optimizer-recovery/report.qmd) | probe | Starts and information coordinates have landed. Isolate the remaining domain/line-search failures and keep known worse local minima distinct from rejected endpoints. |
| 19 | [Newton verdict migration](engineering/active/19-newton-verdict-migration/report.qmd) | probe | The Newton verdict has landed. Resolve the exposed categorical bad endpoints and stopping-control near misses, with every changed verdict explained. |

### Banked ideas — no run queued

| # | Study | Kind | Evidence and reopening trigger |
|--:|-------|------|-------------------------------|
| 03 | [IRLS Ernst convergence](engineering/banked/03-irls-ernst-convergence/report.qmd) | benchmark | Local results have one replicate per sample-size cell. Reopen for a concrete Fisher-scoring/SNLLS use case with a cost-normalized comparison under the current verdict. |
| 07 | [Ordinal observed-score omega](engineering/banked/07-ordinal-observed-omega/report.qmd) | probe | A 16-cell population audit distinguishes covariance omega from direct true-score reliability; 200-replicate checks cover the covariance target only. Reopen for direct-target inference or a named coverage study. |
| 09 | [SAM efficiency and stability](engineering/banked/09-sam-efficiency-stability/report.qmd) | benchmark | Two replicates per regime do not rank SAM against ML. Reopen for a specified two-step SEM use case requiring efficiency or SE calibration evidence. |

### Completed paper evidence

| # | Study | Kind | Lifecycle | Retention reason |
|--:|-------|------|-----------|------------------|
| 12 | [PSD ML basin audit](engineering/evidence/12-psd-ml-basin-audit/report.qmd) | benchmark | complete | Prespecified random-core basin frequencies, enriched diagnostics and tight confirmation for the covariance-honest paper. Preserve the original design and recover absent local artifacts. |

## Archived

Retired investigations retain their sources and existing local results. They
carry no queued work. Reopen only for a named failure, consumer or changed decision.

| Experiment | Question |
|------------|----------|
| [latent-metric-identification](_archive/latent-metric-identification/report.qmd) | Does `std.lv` beat marker-variable parameterization once spec-rebuild and back-conversion costs are counted? |
| [heywood-box-constraints](_archive/heywood-box-constraints/report.qmd) | Do variance box constraints turn inadmissible Heywood ML solutions into admissible boundary optima? |
| [ordinal-snlls-probe](_archive/ordinal-snlls-probe/report.qmd) | Do the cache-aware and SNLLS ordinal paths reproduce the materialized DWLS/WLS fits? |
| [ordinal-inference-cache-probe](_archive/ordinal-inference-cache-probe/report.qmd) | Does carrying a full cache through fitting help robust ordinal reporting requested right after a bounded fit? |
| [ordinal-snlls-speed](_archive/ordinal-snlls-speed/report.qmd) | Does ordinal SNLLS show the same speed pattern as continuous SNLLS once thresholds and ordinal weights are in the objective? |
| [ordinal-threshold-constraints](_archive/ordinal-threshold-constraints/report.qmd) | Which ordinal fitting paths can handle equality constraints on thresholds? |
| [ordinal-construction-boundary](_archive/ordinal-construction-boundary/report.qmd) | What does ordinal statistic construction (lazy vs eager) cost before fitting begins? |
| [sem-total-variance](_archive/sem-total-variance/report.qmd) | Engineering decision: retain unscaled native PSD ML; diagonal scaling stays opt-in and boundary-specific restarts remain experimental. |
| [complete-data-estimator-speed](_archive/complete-data-estimator-speed/report.qmd) | Historical NT/ULS/GLS corpus timing snapshot. Future timing work belongs in the shared benchmark machinery. |
| [near-singular-ml-continuation](_archive/near-singular-ml-continuation/report.qmd) | Tested covariance continuation paths added cost without improving convergence. |
| [psd-ml-timing](_archive/psd-ml-timing/report.qmd) | Historical direct-PSD and audit-first costs, including backend checks. Current route decisions live in decisions/01. |
| [sphere-chart-sanity](_archive/sphere-chart-sanity/report.qmd) | Deterministic sphere correctness and translation checks; core properties are now regression-gated. |
| [complete-ml-global-test-geometry](_archive/complete-ml-global-test-geometry/report.qmd) | Supports expected information for complete-data ML global tests. Reopen for a named misspecification, power or model-family counterexample. |
| [barrier-optimizer](_archive/barrier-optimizer/report.qmd) | Exploratory evidence for PORT; the maintained default decision is decisions/02. |
| [fmg-score-flip-illustration](_archive/fmg-score-flip-illustration/report.qmd) | Former research/33 completed published-design illustration: high restriction rank alone did not justify expensive standardization; fixed-rank geometry is parked in the speculative backlog. |

`_support/` (path, metadata, and I/O helpers; no SEM logic) is the only shared
sibling an experiment may consume. Use category-qualified slugs in references. Numeric prefixes are local ordering,
not permanent study identifiers.

## Evidence availability

The activity labels describe maintained work, not a guarantee that every raw
run is present in a checkout. Current local artifact gaps are the PSD basin
audit, FMG Study 1's lavaan-parity output and the FMG Study 2 run tree. Their
reports stop with reproduction instructions; recover original artifacts before
making quantitative claims. Most raw runs are deliberately ignored. The native
ordinal-DWLS evidence study uses C++/just and a retained Markdown report;
its local `AGENTS.md` documents that legacy exception.

<details>
<summary>Cleanup record: 30 September–1 October 2026</summary>

## Engineering cleanup, 30 September 2026

The flat engineering collection was reviewed against current decisions, tests
and locally available evidence. The active workspace now contains only
unresolved implementation choices. The tables below state the next check or
reopening trigger; a banked study does not imply a queued simulation.

- **Merged:** ordinal observed-score omega sampling (former engineering/07) and
  its target audit (former engineering/08) now share one study, one report and
  one runner. Their result trees and metadata remain separate and unchanged.
- **Moved to research:** the former small RBM check (engineering/06) is
  [research/53](research/banked/53-rbm-estimation-risk/report.qmd). Its 5,000-replication
  N=30 run exposes post-hoc correction tails; the report now defaults to it.
- **Archived:** estimator timing (01), covariance continuation (02), PSD timing
  (11), sphere correctness (14), global-test geometry (16), and barrier optimizer
  selection (18). The corresponding findings remain available at their new paths.
- **Deleted:** the robust-score/modification-index demonstration (04), whose
  scaling and reduction checks are maintained in
  [score_robust_test.cpp](../cpp/tests/unit/score_robust_test.cpp),
  [score_robust_golden_test.cpp](../cpp/tests/golden/score_robust_golden_test.cpp)
  and the [R example](../r-package/examples/estimated_weight_modindices.R). Its
  twelve-replication demonstration supplied no additional calibration evidence.
- **Preserved independently:** the PSD basin audit supports the covariance-honest
  paper. Its frequency sample must stay distinct from the sphere development
  cases. Its result CSVs were absent locally at review; recover the original
  artifacts or reproduce the prespecified run before rebuilding its report.

Frozen result metadata, pre-registered criteria and external cloud identifiers
retain their historical numbers. Navigation changes do not rescore those runs.
Paper-supporting pipelines remain protected regardless of their kind label.

## Research cleanup, 30 September 2026

The research collection was reviewed against reports, runner scope, local run
metadata and the maintained backlogs. Activity folders express whether a study
has a concrete remaining task, an explicit reopening trigger, or a retained
answer. They do not change the statistical design or library defaults.

- **Merged:** Gaussian coverage (former research/22) and nonnormal/local-dependence
  stress (former research/23) now form [polychoric omega](research/active/22-ordinal-functional-inference/report.qmd).
  Those arms are now delta lanes in the later consolidation; historical outputs stay in
  separate subdirectories. Their generating and pseudo-true targets remain distinct.
- **First-pass archives:** the flip illustration (33) and FIML pilot (36)
  were moved under `_archive/`. The FIML consolidation below brings the pilot
  into the invariance study; the flip illustration remains archived. The second
  pass removed redundant ordinal probes and extracted the score/LR diagnostics.
- **Banked:** exploratory permutations, flips, covariance-functional corrections
  and geometry ideas now have a named trigger in the table below. A promising
  result does not automatically queue another simulation.
- **Protected:** paper pipelines remain independent; frozen normal/robust interval
  and robust-calibration summaries stay unchanged. Merging removed duplicate
  navigation and narrative while keeping each design and its original evidence.

The active work has four main threads: parameter/ordinal inference, incomplete-data
inference, PSD/barrier estimation, and heterogeneous spectral calibration. The
banked candidates with the clearest payoff are centered-Mammen resampling,
score permutation for unequal groups, KC plus transformed covariance-functional
intervals, and feasible near-boundary reliability corrections. Each still needs
the independent check named below before it can support a new policy.

**Evidence limits:** several older reports describe larger historical runs while
only smoke outputs are present locally. The tables flag those cases; folder
placement is not a claim that all promised grids were executed. Research/06 has
5,000-replication six-indicator null artifacts;
its larger-model and power design remains unfinished. Do not overwrite frozen
artifacts or relax resume fingerprints to reuse them after navigation changes.

## Research consolidation, 1 October 2026

- **Scalar intervals:** former research/50 and /51 form
  [showcase/07](showcases/07-scalar-intervals/report.qmd), with separate normal
  and robust lanes, original prespecified designs and unchanged frozen sources/results.
- **Reliability targets:** former research/16 and /17 form
  [showcase/08](showcases/08-reliability-targets/report.qmd). Keep the deterministic
  coefficient targets and Bell-style population examples; the old 30-rep sampling
  pilot stays as local provenance, not coverage evidence. Formula/gradient/delta
  correctness is maintained in [reliability tests](../cpp/tests/unit/reliability_test.cpp).
- **Robust calibration:** former research/52, /38 and selected /45 diagnostics
  form [showcase/09](showcases/09-robust-test-calibration/report.qmd): a frozen
  null-size battery, homogeneous spectral controls and correctness/timing checks.
  The overlapping /45 comparison runner is removed; all its original local
  outputs remain under `results/diagnostics/`. Heterogeneous-tail research/49
  remains active and independent.
- **Finite-sample GOF:** research/40 and /41 form one
  [banked study](research/banked/40-finite-sample-gof/report.qmd). Projected-shape
  and Hotelling lanes retain their different ranks, innovation moments and outputs.
  Reopen for a concrete consumer plus independent moment/nuisance-influence validation.
- **Deleted:** former research/07 and /08 ordinal probes, whose local smoke
  outputs do not support calibration comparisons. Overlap influence and shared
  ULS/DWLS/WLS/NT/DLS stage-two construction are maintained in
  [ordinal unit tests](../cpp/tests/unit/ordinal_test.cpp) and
  [pairwise ordinal golden checks](../cpp/tests/golden/ordinal_golden_test.cpp).
- **Retired without local results:** research/34's unrun flip-standardization
  pipeline. Its hypothesis, cheaper alternatives, reopening trigger and legacy
  remote identifiers now live in the
  [speculative backlog](../project/backlog/speculative.md#fixed-rank-flip-standardization).

Consolidation is navigation and evidence maintenance; it does not select a new
library default. Results from different targets/DGPs are never pooled.

## Pairwise retirement and FIML consolidation, 1 October 2026

- **Deleted at the user's request:** dedicated pairwise efficiency/speed studies
  (research/01 and /02), the mixed FIML/pairwise efficiency probe (research/14),
  and the pairwise-composite nested probe (engineering/05). Their small local
  runs did not establish estimator rankings or calibration. Existing APIs remain
  covered by [pairwise GLS tests](../cpp/tests/unit/pairwise_gls_test.cpp),
  [pairwise inference tests](../cpp/tests/unit/pairwise_inference_test.cpp),
  [casewise influence tests](../cpp/tests/unit/pairwise_casewise_test.cpp),
  [pairwise golden checks](../cpp/tests/golden/pairwise_golden_test.cpp) and
  the composite/Godambe/hybrid cases in [ordinal tests](../cpp/tests/unit/ordinal_test.cpp).
  Future pairwise work requires a named consumer; see the
  [speculative backlog](../project/backlog/speculative.md#pairwise-and-spectrum-performance).
  Ordinal missing-data reference/replication studies keep their distinct targets.
- **Global FIML/ML2S tests:** former research/44, /04 and /05 form
  [research/44](research/banked/44-fiml-global-tests/report.qmd). One report and
  runner cover representative SEM calibration, affine/pseudo-null diagnostics
  and the older banked paper designs. Their populations and outputs stay separate.
- **Nested invariance tests:** former research/06 and /37, showcase/05 and the
  archived FIML score-flip pilot form
  [research/06](research/active/06-fiml-invariance-tests/report.qmd). It retains
  the 5,000-rep difference-test run, independent score confirmation, unique pilot
  power arm and oracle controls. Completed flip expansion is banked within this
  active study; scalar nesting, larger-model calibration and matched power remain.
- **Kept independent:** [research/39](research/evidence/39-fiml-information-choice/report.qmd)
  studies parameter SEs around an estimator's limit, rather than exact global fit
  or nested restrictions. Its substantive overnight run is present locally.

All 15,445 moved result files were checked byte-for-byte by hash. Original raw
results, seeds, metadata and remote IDs are retained; compact source-hashed CSVs
make the two reports render without the large raw trees. Reports separate
nonnormal-MAR generating-null stress from justified FIML pseudo-nulls and count
numerical availability separately. The consolidation neither pools the runs
nor changes a library default. Completed iteration diaries were folded into
the reports; substantive prospective designs and technical appendices remain.

## Functional and reference reorganization, 1 October 2026

- **Banked:** alpha–omega equality (03), with the nonregularity derivations retained
  and eight-rep main smoke described honestly. The redundant corrective draft is
  folded into the report/design. Covariance-functional intervals (19) keeps its
  number with a broader name; alpha and correlation are demonstrations.
- **Merged and banked:** maximal-reliability intervals (18) and reliability
  profile-LR (20) form [bounded functional intervals](research/banked/18-bounded-functional-intervals/report.qmd).
  Bifactor/profile lanes retain distinct coefficients, populations and exclusions.
  Oracle acceptance diagnostics do not establish feasible endpoint coverage.
- **Merged and active:** polychoric delta (22, including former stress 23) and
  fitted-profile calibration (25) form [ordinal functional inference](research/active/22-ordinal-functional-inference/report.qmd).
  The fit-free and fitted pseudo-targets remain distinct; endpoint and sparse-category
  failures, rather than a generic full-grid expansion, define the next check.
- **Completed evidence:** [FIML information (39)](research/evidence/39-fiml-information-choice/report.qmd)
  retains the 48-cell × 200,000-rep parameter-uncertainty study and oracle sentinels.
- **Merged and active:** Savalei–Falk definitions (08) and matrix choices (09) form
  [one reference study](replications/08-savalei-falk-2014-test-conventions/report.qmd).
  Written formulas and numerical similarity stay separate; EQS exact identity is
  unresolved and the 192-cell reference design has no queued run.

All **4,900 original result files** were verified unchanged by hash after the
moves. One report and entry point serve each merged study; lane-specific results,
metadata, seed schedules and remote IDs remain intact. Compact source-hashed CSVs
support portable reports. Resume guards remain strict after source-path changes.
Research now has 24 studies: 8 active, 11 banked and 5 retained evidence.

</details>
