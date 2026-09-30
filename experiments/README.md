# Experiments

Experiments are grouped by purpose. Research and engineering also separate
**active work**, **banked ideas**, and **retained evidence**. Numbers are stable
across activity moves; cite the slug and its current path. Archived folders
have no numeric prefix. Conventions are in [AGENTS.md](AGENTS.md).

| Purpose | Studies | Activity |
|---------|--------:|----------|
| [Decisions](#decisions) | 2 | Maintained registers and held-out evidence for library defaults. |
| [Showcases](#showcases) | 6 | Capabilities, parity and performance demonstrations. |
| [Replications](#replications-and-reference-studies) | 9 | Published results and reference reconstructions. |
| [Research](#research) | 41 | 16 active, 16 banked, 9 retained evidence studies. |
| [Engineering](#engineering-checks) | 9 | 4 active, 4 banked, 1 completed paper-evidence study. |
| [Archive](#archived) | 18 | Settled investigations and predecessors, with sources/results retained. |

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
  stress (former research/23) now form [polychoric omega](research/active/22-ordinal-polychoric-omega/report.qmd).
  One runner selects `--lane coverage|stress`; historical outputs stay in
  separate subdirectories. Their generating and pseudo-true targets remain distinct.
- **Archived:** the ordinal overlap-Gamma construction check (07), the completed
  flip illustration (33), the FIML flip pilot (36), and the initial score-versus-LR
  comparison (45). The wider stress/battery studies retain the current questions;
  the earlier sources and results remain reproducible under `_archive/`.
  The trace-only SB diagnostic in the last archive remains useful maintainer evidence.
- **Banked:** exploratory permutations, flips, covariance-functional corrections
  and geometry ideas now have a named trigger in the table below. A promising
  result does not automatically queue another simulation.
- **Protected:** paper pipelines remain independent; frozen normal/robust interval
  and robust-calibration summaries stay unchanged. No unique research evidence
  was deleted. Merging removed the second omega folder and duplicate narrative.

The active work has four main threads: functional/profile intervals, incomplete-data
inference, PSD/barrier estimation, and heterogeneous spectral calibration. The
banked candidates with the clearest payoff are centered-Mammen resampling,
score permutation for unequal groups, KC plus transformed covariance-functional
intervals, and feasible near-boundary reliability corrections. Each still needs
the independent check named below before it can support a new policy.

**Evidence limits:** several older reports describe larger historical runs while
only smoke outputs are present locally. The tables flag those cases; folder
placement is not a claim that all promised grids were executed. Research/06 has
5,000-replication six-indicator null artifacts despite its draft-style report;
its larger-model and power design remains unfinished. Do not overwrite frozen
artifacts or relax resume fingerprints to reuse them after navigation changes.

## Decisions

Pre-registered studies whose results set library defaults; see
[AGENTS.md](AGENTS.md#decisions-set-library-defaults) for the standard.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [optimizer-defaults](decisions/01-optimizer-defaults/report.qmd) | benchmark | active | Which start, optimizer and PSD route should each estimation route use by default, judged by the library verdict on held-out simulated problems? |
| 02 | [barrier-defaults](decisions/02-barrier-defaults/report.qmd) | benchmark | active | Which start and optimizer should the complete-data ML barrier fitter use by default, judged by the library verdict on held-out simulated problems? |

## Showcases

Demonstrate magmaan capabilities, agreement with lavaan/other references, or a concrete performance comparison.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [lavaan-parity](showcases/01-lavaan-parity/report.md) | parity | active | Does magmaan's ML inference match lavaan across the textbook corpus? |
| 02 | [lavaan-speed-bench](showcases/02-lavaan-speed-bench/report.md) | benchmark | active | How much faster is magmaan than lavaan on the same prepared-input CFA fit? |
| 03 | [foldnes-moss-gronneberg-peba](showcases/03-foldnes-moss-gronneberg-peba/report.qmd) | replication | complete | Do the penalized EBA goodness-of-fit tests hold nominal Type I under nonnormality? |
| 04 | [foldnes-moss-gronneberg-2026](showcases/04-foldnes-moss-gronneberg-2026/report.qmd) | replication | active | Can magmaan reproduce the full FMG goodness-of-fit machinery versus lavaan and semTests? |
| 05 | [fiml-measurement-invariance-fmg](showcases/05-fiml-measurement-invariance-fmg/report.qmd) | paper-sim | active | Is the FIML FMG robust-test family feature-complete for measurement invariance? |
| 06 | [speed-attribution](showcases/06-speed-attribution/report.qmd) | benchmark | active | How much of matched ML + GOF/Wald runtime is preparation, fitting, or inference, and which comparisons pass numerical agreement before a public speed claim? |

## Replications and reference studies

Reproduce published results or reconstruct the reference method needed to interpret them.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [maydeu-olivares-2017](replications/01-maydeu-olivares-2017/report.qmd) | replication | complete | Do the SE methods and χ² adjustments hold under nonnormality (two-factor CFA)? |
| 02 | [rhemtulla-2012](replications/02-rhemtulla-2012/report.qmd) | replication | complete | When can ordinal variables be treated as continuous (cat-LS vs continuous ML)? |
| 03 | [li-2021-mixed](replications/03-li-2021-mixed/report.qmd) | replication | complete | DWLS or MLR for a mix of continuous and categorical indicators? |
| 04 | [li-2016-ordinal](replications/04-li-2016-ordinal/report.qmd) | replication | complete | DWLS/ULS or continuous ML for an all-ordinal five-factor SEM? |
| 05 | [chen-2020-wlsmv-pd](replications/05-chen-2020-wlsmv-pd/report.qmd) | replication | active | Can the Chen et al. (2020) WLSMV_PD Type-I inflation cell be reproduced with current lavaan WLSMV pairwise deletion? |
| 06 | [jamil-rosseel-2026-rbm-sem](replications/06-jamil-rosseel-2026-rbm-sem/report.qmd) | replication | active | Can we reproduce the SEM reduced-bias paper's two-factor and growth-curve RBM examples from the authors' OSF outputs before magmaan-owned reruns? |
| 07 | [foldnes-moss-gronneberg-2026-study2](replications/07-foldnes-moss-gronneberg-2026-study2/report.qmd) | replication | complete | Can native magmaan express the 189-cell Study 2 weak-invariance Type-I comparison, including the unbiased-Gamma variants, with semTests retained only as a parity sentinel? |
| 08 | [Savalei-Falk-2014 test map](replications/08-savalei-falk-2014-test-map/report.qmd) | replication | active | Which exact null/reference models and observed-information corrections define Savalei and Falk's robust FIML and two-stage tests, and how should magmaan verify them? |
| 09 | [Savalei-Falk-2014 matrix choices](replications/09-savalei-falk-2014-ml2s-information/report.qmd) | replication | complete | Which Yuan-Bentler-style matrix choice gives rejection behavior most similar to Savalei and Falk, without claiming an exact EQS replication? |

## Research

Investigate statistical behavior, new methods, estimands and inferential validity.

### Active research (16)

Concrete remaining checks and protected ongoing paper pipelines.

| # | Experiment | Kind | Lifecycle | Question | Finding / next check or reopening trigger |
|--:|------------|------|-----------|----------|------------------------------------------|
| 01 | [pairwise-gls-efficiency](research/active/01-pairwise-gls-efficiency/report.qmd) | paper-sim | active | Which of five missing-data estimators is most efficient? | Local outputs are a 3-rep smoke; recover/full-run the prespecified efficiency grid before quantitative use. |
| 02 | [pairwise-fit-speed](research/active/02-pairwise-fit-speed/report.qmd) | paper-sim | active | Which pairwise missing-data estimator is fastest? | Retain the paper timing pipeline; extend beyond two models and distinguish preparation/inference from fit cost. |
| 03 | [deng-chan-2017-alpha-omega](research/active/03-deng-chan-2017-alpha-omega/report.qmd) | paper-sim | active | Is the Deng-Chan Wald test of coefficient α = ω valid? | First-order Wald is singular at tau-equivalence; finish the second-order reference and recover substantive artifacts (local main metadata is smoke). |
| 04 | [fiml-fmg-vs-mlr](research/active/04-fiml-fmg-vs-mlr/report.qmd) | paper-sim | active | Under non-normality + MCAR, do the FMG/pEBA FIML goodness-of-fit and nested tests beat the dominant MLR (Yuan-Bentler) default? | Local main outputs are a 10-rep slice; recover the full comparison and strengthen MAR beyond sign-based selection. |
| 05 | [fiml-twostage-fmg-chisq](research/active/05-fiml-twostage-fmg-chisq/report.qmd) | paper-sim | active | Do the FMG full-spectrum goodness-of-fit chi-squares calibrate FIML and two-stage ML (ML2S) better than the Savalei low-moment corrections under non-normal incomplete data? | Two-stage naive tests can inflate even under normality; recover full grids and report size uncertainty separately by estimator/mechanism. |
| 06 | [fiml-invariance-fmg-power](research/active/06-fiml-invariance-fmg-power/report.qmd) | paper-sim | active | Under non-normal incomplete data (FIML and ML2S), do the FMG/pEBA eigenvalue p-values calibrate the measurement-invariance difference test better than the MLR/Satorra-Bentler default, and with what power? (extends Brace & Savalei 2017) | Six-indicator null runs have 5,000 reps; complete larger published model sizes, scalar nesting and size-matched power. |
| 20 | [profile-lr-reliability-ci](research/active/20-profile-lr-reliability-ci/report.qmd) | probe | active | **funLR (Functional profile-LR CI).** Can a generic profile-LR (test-inversion) engine reproduce semlbci across the reliability family (omega_total, omega_h, H, maximal reliability), and can the small-sample under-coverage of near-boundary coefficients be repaired by a Bartlett factor (analytic, bootstrap, or calibrated constant)? | Feasible near-boundary Bartlett correction remains weaker than the oracle; reduce correction noise and evaluate actual interval inversion. |
| 21 | [ml-parameter-profile-lrt](research/active/21-ml-parameter-profile-lrt/report.qmd) | probe | active | Do ML and fixed-weight continuous-GMM scalar profiles calibrate, including nonnormal complete-IJ coverage? | Retain ML and nonnormal fixed-weight GLS/WLS/ULS profiles with complete IJ reference; assess coverage separately from the removed fitted-weight arm. |
| 22 | [ordinal-polychoric-omega](research/active/22-ordinal-polychoric-omega/report.qmd) | probe | active | Does fit-free polychoric-omega delta coverage hold for Gaussian targets and nonnormal/local-dependence pseudo-targets? | Run both merged arms beyond smoke size; quantify sparse-category failures and pseudo-target precision without pooling the two targets. |
| 25 | [ordinal-profile-lrt-calibration](research/active/25-ordinal-profile-lrt-calibration/report.qmd) | probe | active | What causes small-sample miscoverage of ordinal polychoric-omega profile-LRT intervals: LR inflation, scaling, sparse summaries, bias, or misspecification? | Separate profile-statistic inflation, sparse-summary failure and inversion geometry; finish the full grid before choosing a correction. |
| 39 | [fiml-information-choice](research/active/39-fiml-information-choice/report.qmd) | benchmark | active | For continuous FIML, how do expected Fisher, observed-H1, and full observed-Hessian information compare for model-based and same-meat sandwich SE calibration under complete, MCAR, MAR, nonnormal, and misspecified cells? | Complete/recover the 48-cell information-choice study and oracle sentinels; judge correct and misspecified targets separately. |
| 43 | [psd-ml-repair-risk](research/active/43-psd-ml-repair-risk/report.qmd) | benchmark | active | What does PSD-ML repair when ordinary NTML leaves the primitive covariance domain, and what is its near-boundary estimation risk? | Likelihood cost and estimation risk differ across improper-solution types; extend the pilot boundary path without treating repair cost as a test. |
| 44 | [fiml-global-gof-pilot](research/active/44-fiml-global-gof-pilot/report.qmd) | probe | active | Across finite-moment stress, MAR, and explicit contract violations, do effective multiplier or spectrum-aware scores calibrate global FIML GOF better than Yuan–Bentler MLR? | Carry observed-H0 sensitivity/expected metric into the source-based latent-model gate; freeze source adaptations, missingness maps and screening before the publication grid. Keep completed multiplier diagnostics; do not expand that arm. |
| 47 | [multiinfo-penalty-improper](research/active/47-multiinfo-penalty-improper/report.qmd) | benchmark | active | Do the joint and latent-determinacy (Q) barriers remove improper solutions without costing PSD-constrained ML's accuracy, also with the truth on a face, which fit test goes with a barrier estimate, and does the barrier's local limit hold? | Paper pipeline: barrier properness trades risk near a face; complete boundary/path and fit-test checks rather than selecting a weight from pooled RMSE. |
| 48 | [multiinfo-jeffreys-posterior](research/active/48-multiinfo-jeffreys-posterior/report.qmd) | benchmark | active | Does the multi-information barrier approximate the Jeffreys-prior posterior over admissible solutions, for which weight, and is that posterior a better estimator? | Paper pipeline: posterior accuracy depends on functional, prior and ESS; recover substantive outputs and finish joint/Q/path comparisons (local main run: smoke). |
| 49 | [spectral-tail-calibration](research/active/49-spectral-tail-calibration/report.qmd) | benchmark | active | Can spectral shrinkage and debiased moment corrections improve score and LRT calibration under non-normality? | IG oracle spectra can calibrate while estimated moments fail; investigate heterogeneous spectral estimation bias and failure conditioning. |

### Banked research ideas (16)

No queued run. Reopen for the named consumer or validation trigger.

| # | Experiment | Kind | Lifecycle | Question | Finding / next check or reopening trigger |
|--:|------------|------|-----------|----------|------------------------------------------|
| 08 | [ordinal-stage2-pairwise](research/banked/08-ordinal-stage2-pairwise/report.qmd) | benchmark | banked | When pairwise ordinal Gamma is reused, how do ULS/DWLS/WLS/NT/DLS stage-two estimators compare on p-values, SE diagnostics, and runtime? | Stage-two estimators reuse the same moments; reopen for a named ordinal estimator decision and a full grid (local run: 2 reps). |
| 09 | [rmsea-like-catml-dwls](research/banked/09-rmsea-like-catml-dwls/report.qmd) | probe | banked | Does lavaan's categorical robust RMSEA behave as a consistent CATML-at-DWLS criterion-at-estimator statistic under ordinal misspecification? | CATML-at-DWLS targets a hybrid criterion; reopen for a named RMSEA consumer and a large-N pseudo-target check (local run: smoke). |
| 11 | [chen-ordinal-fmg-pvalues](research/banked/11-chen-ordinal-fmg-pvalues/report.qmd) | probe | banked | Is the WLSMV pairwise-missing scalar Type-I inflation a defect of the test/p-value family or of the missing-data mechanism (MCAR vs MAR)? | Reported MAR inflation points to biased polychorics; reopen for stronger missingness/bias diagnostics and recover the substantive run (local output: 3 reps). |
| 14 | [mixed-fiml-pairwise-efficiency](research/banked/14-mixed-fiml-pairwise-efficiency/report.qmd) | benchmark | banked | Under item missingness in mixed continuous/ordinal SEM, does a continuous-FIML first stage (pairwise x FIML) buy efficiency and reduce MAR bias over fully pairwise statistics, and on which parameter block? | Only the continuous block changes under the FIML hybrid; reopen for a mixed-data estimator consumer and a full run (local output: 4 reps). |
| 16 | [reliability-lambda6](research/banked/16-reliability-lambda6/report.qmd) | benchmark | banked | How do alpha, Guttman's lambda6, covariance omega, and fitted NT omega compare for total-score reliability under correct and misspecified one-factor summaries? | Functionals have different population targets; reopen for a family-wide reliability interval consumer and nonnormal/misspecified coverage (local output: 30-rep smoke). |
| 18 | [li-savalei-2026-maximal-reliability-ci](research/banked/18-li-savalei-2026-maximal-reliability-ci/report.qmd) | replication | banked | Can you put a trustworthy confidence interval around bifactor maximal reliability, the inference Li & Savalei (2026) leave open? | Oracle Bartlett repairs near-ceiling coverage but is not a feasible interval; reopen when the functional-profile project needs bifactor boundary validation. |
| 19 | [alpha-kc-coverage](research/banked/19-alpha-kc-coverage/report.qmd) | benchmark | banked | A distribution-free interval for a covariance functional (alpha, correlation) has two small-N defects: does the Kauermann-Carroll effective-df t fix the two-sided coverage deficit (variance-of-variance, flat in rho) while a variance-stabilizing transform (Fisher z / logit) fixes the skew-driven left/right imbalance (growing in rho), and do the two compose? | KC t and transforms address different defects; reopen for one practical interval recipe across covariance functionals, with heavy-tail confirmation. |
| 31 | [permutation-measurement-invariance](research/banked/31-permutation-measurement-invariance/report.qmd) | probe | banked | Can fully recomputed pivotal score or Wald label permutations calibrate continuous metric-invariance tests under heterogeneous group populations, and can a Hotelling reference change the permutation-score test? | Score permutation improves the unequal-allocation pivot; reopen for a named ordinal-invariance consumer after estimator-specific influence/pivotality validation. The continuous result does not establish ordinal permutation validity. |
| 32 | [standardized-flip-score](research/banked/32-standardized-flip-score/report.qmd) | probe | banked | Do nuisance-effective and flip-specifically standardized score flips calibrate continuous metric-invariance tests at small N? | Centered Mammen is promising; reopen for fresh confirmation beyond complete-data affine rank-eight ML or a concrete meat-shrinkage consumer. |
| 34 | [flip-calibration-frontier](research/banked/34-flip-calibration-frontier/report.qmd) | probe | banked | At fixed test rank, when do nuisance dimension, allocation, information geometry, and sample size make flip standardization worthwhile? | Fixed-rank geometry/allocation is a distinct question; reopen only for targeted corners where standardization changes calibration enough to pay its cost. |
| 35 | [residual-flip-gof-probe](research/banked/35-residual-flip-gof-probe/report.qmd) | probe | banked | Can efficient residual-score flipping calibrate single-model SEM goodness of fit without an ad hoc regularizer? | Residual-score flip algebra works, but high rank causes failure/miscalibration; reopen for a low/moderate-rank consumer or a separately justified shrinkage method. |
| 37 | [fiml-flip-stress](research/banked/37-fiml-flip-stress/report.qmd) | probe | banked | Across generator family, sample size, missingness mechanism, and restriction rank, where do direct-FIML effective and standardized score flips remain calibrated? | The broad and hard-cell FIML runs replace the pilot; reopen for a concrete incomplete-data flip consumer or a new rank/missingness failure regime. |
| 40 | [projected-score-satterthwaite](research/banked/40-projected-score-satterthwaite/report.qmd) | probe | banked | Can a projected Satterthwaite effective denominator improve calibration of an already-pivotal robust SEM goodness-of-fit score, and is the empirical rule's apparent success genuine self-normalization or moment-estimation bias? | Projected denominator correction and same-sample bias need separation; reopen for independent shape estimation at a named small-N GOF target. |
| 41 | [hotelling-robust-gof-grid](research/banked/41-hotelling-robust-gof-grid/report.qmd) | probe | banked | Does the ordinary-Hotelling reference improve a centered pivotal SEM goodness-of-fit score across residual rank and sample-to-rank ratio, relative to its chi-square limit and SB/MV/pEBA4 corrections of the working RLS score? | Hotelling is a finite-sample improvement, not an exact SEM law; reopen when sample-to-rank correction is a concrete inference requirement. |
| 46 | [latent-metric-geometry](research/banked/46-latent-metric-geometry/report.qmd) | benchmark | banked | Which latent scaling convention should a SEM library use internally, is there a better one than the named three, and does the answer survive the move from CFA to structural models? | CFA chart conditioning does not settle structural scaling; reopen for matched starts and genuinely near-Heywood structural populations (local run: smoke). |
| 53 | [rbm-estimation-risk](research/banked/53-rbm-estimation-risk/report.qmd) | probe | banked | When do post-hoc and integrated reduced-bias corrections improve error, and when do tails or fit failures erase that gain? | 5,000-rep N=30 evidence exposes correction tails; reopen for a concrete RBM admissibility/estimation-risk consumer or nonnormal/FIML extension. |

### Retained research evidence (9)

Completed reference answers and frozen studies, with their scope limits.

| # | Experiment | Kind | Lifecycle | Question | Finding / next check or reopening trigger |
|--:|------------|------|-----------|----------|------------------------------------------|
| 10 | [mplus-demo-wlsmv-difftest](research/evidence/10-mplus-demo-wlsmv-difftest/report.qmd) | probe | complete | Does Mplus Demo WLSMV DIFFTEST for a demo-sized ordinal pairwise-missing invariance model match lavaan/magmaan Satorra-2000 statistics? | Demo DIFFTEST agrees to rounding after matching scalar constraints; limited demo-sized oracle evidence, with local Mplus needed to rerun. |
| 12 | [misspec-robust-se](research/evidence/12-misspec-robust-se/report.qmd) | probe | complete | Does the observed-Hessian ("robust" regime) bread recover the true sampling SD of ordinal DWLS estimates under structural misspecification, while coinciding with the conventional SE under the null? | Full estimated-weight IJ, rather than bread alone, repairs the misspecified DWLS example; evidence remains local to the stated models/targets. |
| 13 | [ordinal-dwls-profile-lrt](research/evidence/13-ordinal-dwls-profile-lrt/report.md) | paper-sim | complete | Does the standard scaled difference test for nested all-ordinal DWLS models stay calibrated when the larger model is misspecified, and does the estimated-weight profile law restore calibration? | On the exact pseudo-null, fixed-weight rejection rises to 10.5% while estimated-weight stays near 4.2%; retain the C++ calibration pipeline. |
| 17 | [bell-omega-bias](research/evidence/17-bell-omega-bias/report.qmd) | benchmark | complete | At Bell-style population moments, how biased is ordinary one-factor omega under misspecification for ML, ULS, GLS, and the Spearman-Guttman covariance form? | Population-moment evidence: model sensitivity is descriptive, not an identified bias estimator; retain Bell cells and higher-order extension. |
| 38 | [score-test-calibration-drivers](research/evidence/38-score-test-calibration-drivers/report.qmd) | probe | complete | What governs the calibration of the mean-scaled (Satorra–Bentler) non-normal score test — restriction df, structural information geometry, or eigenvalue-spectrum dispersion — and how do the spectrum-aware and sign-flip variants compare? | Homogeneous-tail df/geometry controls explain SB behavior; heterogeneous kurtosis is outside this answer. Retain the mechanism evidence. |
| 42 | [psd-ml-small-n-convergence](research/evidence/42-psd-ml-small-n-convergence/report.qmd) | benchmark | complete | On the De Jonckere–Rosseel / Ernst small-N SEM design, does covariance-honest NTML converge more reliably than ordinary NTML? | Completed small-N PSD convergence evidence, including equality-KKT telemetry; preserve the paper pipeline and attribution controls. |
| 50 | [normal-parameter-intervals](research/evidence/50-normal-parameter-intervals/report.qmd) | probe | complete | Do normal-theory Wald, score and profile-LR parameter intervals calibrate, and is candidate-specific bootstrap Bartlett inversion feasible? | Frozen normal-interval and Bartlett feasibility evidence; no reliable practical correction gain in the pilot. Preserve fingerprints and precision checks. |
| 51 | [robust-parameter-intervals](research/evidence/51-robust-parameter-intervals/report.qmd) | probe | complete | How do robust Wald, score and profile-LR scalar intervals compare in coverage, width, failures and cost under non-normality? | Frozen 3,000-dataset scalar-interval evidence; expected score gains are limited by target, validity and cost. No general interval-policy conclusion. |
| 52 | [robust-calibration-battery](research/evidence/52-robust-calibration-battery/report.qmd) | benchmark | complete | Across textbook SEMs, sample sizes and non-normal data including discretized 5-point items analysed by ML, which of SB, MV, MV-UG, corrected MV and pEBA4 applied to the score, LR and RLS statistics keeps nominal size for global and nested tests? | Frozen 5,000-rep/cell null battery; score corrections improve several families, but IG remains unresolved. No power claim. |

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
| 05 | [Pairwise composite nested tests](engineering/banked/05-pairwise-composite-nested/report.qmd) | probe | Six replicates per truth condition establish execution only. Reopen when a consumer needs this estimator's nested-test calibration or stable Godambe calculation. |
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

| [ordinal-pd-gamma](_archive/ordinal-pd-gamma/report.qmd) | Former research/07 construction probe: overlap-Gamma mechanics are maintained in core; wider estimator/reference studies carry the current questions. Local artifacts are smoke evidence. |
| [fmg-score-flip-illustration](_archive/fmg-score-flip-illustration/report.qmd) | Former research/33 completed published-design illustration: high restriction rank alone did not justify expensive standardization; fixed-rank geometry remains banked in research/34. |
| [fiml-score-flip-probe](_archive/fiml-score-flip-probe/report.qmd) | Former research/36 pilot, superseded by research/37's broad screen and independent hard-cell confirmation. Retains the original power arm and construction evidence. |
| [score-vs-lrt](_archive/score-vs-lrt/report.qmd) | Former research/45 paired initial comparison; research/52 carries the broader null battery. Primitive timing and the trace-only SB diagnostic remain available here. |

`_support/` (path, metadata, and I/O helpers; no SEM logic) is the only shared
sibling an experiment may consume. Use category-qualified slugs in references. Numeric prefixes are local ordering,
not permanent study identifiers.
