# Experiments

Experiments are grouped by purpose. Numbering starts at **01 within each
category**; cite the category and slug because numbers are only navigation.
Archived folders have **no numeric prefix**. Reports and existing results are
preserved; this reorganization does not rerun or reassess the studies.

| Purpose | What belongs here | Count |
|---------|-------------------|------:|
| [Decisions](#decisions) | Pre-registered studies whose results set library defaults, with the register of those defaults. | 2 |
| [Showcases](#showcases) | Demonstrate magmaan capabilities, agreement with lavaan/other references, or a concrete performance comparison. | 8 |
| [Replications and reference studies](#replications-and-reference-studies) | Reproduce published results or reconstruct the reference method needed to interpret them. | 9 |
| [Research](#research) | Investigate statistical behavior, new methods, estimands, or inferential validity. | 49 |
| [Engineering checks](#engineering-checks) | Inform active implementation and default choices; inherited studies await a focused review. | 19 |
| [Archived](#archived) | Already retired engineering investigations; kept for provenance. | 9 |

The existing **kind** (`parity`, `replication`, `paper-sim`, `benchmark`, `probe`)
and **lifecycle** (`active`, `complete`, `archived`) remain separate metadata.
Lifecycle labels are carried forward, not certified by this skim. In particular,
`complete` does not imply disposable, and `paper-sim` remains protected regardless
of its navigation category. Conventions and shared harness rules are in
[AGENTS.md](AGENTS.md).

## Direction for the next cleanup

**Engineering should stay live and focused:** each retained study should inform
a current choice, such as the default estimator, optimizer, starts, scaling,
or convergence rule. State the current default, credible alternatives, decision
criteria, latest evidence, and what would trigger reconsideration. Once a decision
is settled, capture the contract in maintained tests/docs and archive the study;
keep a recurring benchmark only if it still informs a live decision.

This is the direction for the next cleanup, not a claim that every inherited
engineering entry already meets it. `engineering/03-irls-ernst-convergence` remains a completed
engineering check pending that review. Research may stay exploratory and less
uniform; no additional studies are archived in this renaming pass. Paper-supporting
pipelines and frozen evidence remain protected.

Classification follows the main question, not the presence of a paper title.
The Deng–Chan critique, Bell omega target investigation, and Li–Savalei interval
extension remain research. Replications include reference reconstructions and
explicitly labeled analogues; placement does not certify exact replication.
Showcase reports still carry their own accuracy and scope caveats.

Historical output metadata and cloud app/volume identifiers can retain the old
global numbers. They identify existing evidence or storage, not current folder
positions; do not rewrite frozen results or rename remote storage for navigation.

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
| 02 | [lavaan-speed-bench](showcases/02-lavaan-speed-bench/report.md) | benchmark | active | How fast is magmaan versus lavaan on the Geiser textbook corpus? |
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

Investigate statistical behavior, new methods, estimands, or inferential validity.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [pairwise-gls-efficiency](research/01-pairwise-gls-efficiency/report.qmd) | paper-sim | active | Which of five missing-data estimators is most efficient? |
| 02 | [pairwise-fit-speed](research/02-pairwise-fit-speed/report.qmd) | paper-sim | active | Which pairwise missing-data estimator is fastest? |
| 03 | [deng-chan-2017-alpha-omega](research/03-deng-chan-2017-alpha-omega/report.qmd) | paper-sim | active | Is the Deng-Chan Wald test of coefficient α = ω valid? |
| 04 | [fiml-fmg-vs-mlr](research/04-fiml-fmg-vs-mlr/report.qmd) | paper-sim | active | Under non-normality + MCAR, do the FMG/pEBA FIML goodness-of-fit and nested tests beat the dominant MLR (Yuan-Bentler) default? |
| 05 | [fiml-twostage-fmg-chisq](research/05-fiml-twostage-fmg-chisq/report.qmd) | paper-sim | active | Do the FMG full-spectrum goodness-of-fit chi-squares calibrate FIML and two-stage ML (ML2S) better than the Savalei low-moment corrections under non-normal incomplete data? |
| 06 | [fiml-invariance-fmg-power](research/06-fiml-invariance-fmg-power/report.qmd) | paper-sim | active | Under non-normal incomplete data (FIML and ML2S), do the FMG/pEBA eigenvalue p-values calibrate the measurement-invariance difference test better than the MLR/Satorra-Bentler default, and with what power? (extends Brace & Savalei 2017) |
| 07 | [ordinal-pd-gamma](research/07-ordinal-pd-gamma/report.qmd) | probe | active | Does overlap-weighting the ordinal pairwise-deletion Gamma fix the nominal-N WLSMV missing-data scaling problem? |
| 08 | [ordinal-stage2-pairwise](research/08-ordinal-stage2-pairwise/report.qmd) | benchmark | active | When pairwise ordinal Gamma is reused, how do ULS/DWLS/WLS/NT/DLS stage-two estimators compare on p-values, SE diagnostics, and runtime? |
| 09 | [rmsea-like-catml-dwls](research/09-rmsea-like-catml-dwls/report.qmd) | probe | active | Does lavaan's categorical robust RMSEA behave as a consistent CATML-at-DWLS criterion-at-estimator statistic under ordinal misspecification? |
| 10 | [mplus-demo-wlsmv-difftest](research/10-mplus-demo-wlsmv-difftest/report.qmd) | probe | active | Does Mplus Demo WLSMV DIFFTEST for a demo-sized ordinal pairwise-missing invariance model match lavaan/magmaan Satorra-2000 statistics? |
| 11 | [chen-ordinal-fmg-pvalues](research/11-chen-ordinal-fmg-pvalues/report.qmd) | probe | active | Is the WLSMV pairwise-missing scalar Type-I inflation a defect of the test/p-value family or of the missing-data mechanism (MCAR vs MAR)? |
| 12 | [misspec-robust-se](research/12-misspec-robust-se/report.qmd) | probe | active | Does the observed-Hessian ("robust" regime) bread recover the true sampling SD of ordinal DWLS estimates under structural misspecification, while coinciding with the conventional SE under the null? |
| 13 | [ordinal-dwls-profile-lrt](research/13-ordinal-dwls-profile-lrt/report.md) | paper-sim | active | Does the standard scaled difference test for nested all-ordinal DWLS models stay calibrated when the larger model is misspecified, and does the estimated-weight profile law restore calibration? |
| 14 | [mixed-fiml-pairwise-efficiency](research/14-mixed-fiml-pairwise-efficiency/report.qmd) | benchmark | active | Under item missingness in mixed continuous/ordinal SEM, does a continuous-FIML first stage (pairwise x FIML) buy efficiency and reduce MAR bias over fully pairwise statistics, and on which parameter block? |
| 15 | [guttman-cfa-asymptotics](research/15-guttman-cfa-asymptotics/report.qmd) | benchmark | active | How do Guttman's closed-form CFA estimates compare with NT/ULS/GLS/WLS in asymptotic variance and fixed-misspecification risk? |
| 16 | [reliability-lambda6](research/16-reliability-lambda6/report.qmd) | benchmark | active | How do alpha, Guttman's lambda6, covariance omega, and fitted NT omega compare for total-score reliability under correct and misspecified one-factor summaries? |
| 17 | [bell-omega-bias](research/17-bell-omega-bias/report.qmd) | benchmark | active | At Bell-style population moments, how biased is ordinary one-factor omega under misspecification for ML, ULS, GLS, and the Spearman-Guttman covariance form? |
| 18 | [li-savalei-2026-maximal-reliability-ci](research/18-li-savalei-2026-maximal-reliability-ci/report.qmd) | replication | active | Can you put a trustworthy confidence interval around bifactor maximal reliability, the inference Li & Savalei (2026) leave open? |
| 19 | [alpha-kc-coverage](research/19-alpha-kc-coverage/report.qmd) | benchmark | active | A distribution-free interval for a covariance functional (alpha, correlation) has two small-N defects: does the Kauermann-Carroll effective-df t fix the two-sided coverage deficit (variance-of-variance, flat in rho) while a variance-stabilizing transform (Fisher z / logit) fixes the skew-driven left/right imbalance (growing in rho), and do the two compose? |
| 20 | [profile-lr-reliability-ci](research/20-profile-lr-reliability-ci/report.qmd) | probe | active | **funLR (Functional profile-LR CI).** Can a generic profile-LR (test-inversion) engine reproduce semlbci across the reliability family (omega_total, omega_h, H, maximal reliability), and can the small-sample under-coverage of near-boundary coefficients be repaired by a Bartlett factor (analytic, bootstrap, or calibrated constant)? |
| 21 | [ml-parameter-profile-lrt](research/21-ml-parameter-profile-lrt/report.qmd) | probe | active | Does the new C++ complete-data ML scalar/profile-LR seed calibrate as ordinary `chi^2_1` for an interior free parameter at small N, before robust scaling and bounded functionals are layered on? |
| 22 | [ordinal-polychoric-omega-coverage](research/22-ordinal-polychoric-omega-coverage/report.qmd) | probe | active | Does the robust delta interval for no-integration ordinal polychoric omega cover its latent-response target across small-N balanced and threshold-extreme ordinal cells? |
| 23 | [ordinal-polychoric-omega-stress](research/23-ordinal-polychoric-omega-stress/report.qmd) | probe | active | Under nonnormal or locally dependent ordinal data, does the robust delta interval for no-integration ordinal polychoric omega cover the pseudo-true polychoric target? |
| 24 | [noniterative-cfa-tests](research/24-noniterative-cfa-tests/report.qmd) | benchmark | active | Are delta-method SEs, residual goodness-of-fit, and difference tests for the closed-form Guttman CFA estimator calibrated in finite samples, and at what efficiency cost vs ML, across normal / independent-generator / ordinal-as-continuous data? |
| 25 | [ordinal-profile-lrt-calibration](research/25-ordinal-profile-lrt-calibration/report.qmd) | probe | active | What causes small-sample miscoverage of ordinal polychoric-omega profile-LRT intervals: LR inflation, scaling, sparse summaries, bias, or misspecification? |
| 26 | [noniterative-invariance](research/26-noniterative-invariance/report.qmd) | benchmark | active | Are the closed-form Guttman metric (projection Wald) and scalar (reference-group mean map) measurement-invariance tests calibrated and powered vs the ML likelihood-ratio test, and does the empirical fourth-moment matrix restore the level under non-normal data? |
| 27 | [guttman-communality-estimators](research/27-guttman-communality-estimators/report.qmd) | benchmark | active | Which closed-form communality rule should the Guttman CFA paper recommend once RS, ILM, plug-in triad GMM, magmaan NT ML, and an ordinal Pearson-code stress arm are compared? |
| 28 | [guttman-standardized-hs](research/28-guttman-standardized-hs/report.qmd) | benchmark | active | On a Holzinger-Swineford-shaped CFA, how do standardized augmented Guttman ILS estimates, intervals, and tests compare with NTML and ULS? |
| 29 | [guttman-rmse-coverage](research/29-guttman-rmse-coverage/report.qmd) | paper-sim | active | When does the chosen raw Guttman recipe (extended_triad_ls + standardized) retain useful RMSE, coverage, and admissibility, configural and correctly restricted, relative to legacy Guttman, ML, and ULS? |
| 30 | [guttman-vs-ntml](research/30-guttman-vs-ntml/report.qmd) | benchmark | active | Does the chosen closed-form recipe (extended_triad_ls + standardized, aligned configural) match NTML on parameter RMSE and empirical-SE coverage, including the high-rho/weak-loading regime, at a modest Sigma-fit cost while being faster and never failing? |
| 31 | [permutation-measurement-invariance](research/31-permutation-measurement-invariance/report.qmd) | probe | active | Can fully recomputed pivotal score or Wald label permutations calibrate continuous metric-invariance tests under heterogeneous group populations, and can a Hotelling reference change the permutation-score test? |
| 32 | [standardized-flip-score](research/32-standardized-flip-score/report.qmd) | probe | active | Do nuisance-effective and flip-specifically standardized score flips calibrate continuous metric-invariance tests at small N? |
| 33 | [fmg-score-flip-illustration](research/33-fmg-score-flip-illustration/report.qmd) | probe | complete | In a focused published weak-invariance design, how do score flips compare with the basic FMG nested-test variants? |
| 34 | [flip-calibration-frontier](research/34-flip-calibration-frontier/report.qmd) | probe | active | At fixed test rank, when do nuisance dimension, allocation, information geometry, and sample size make flip standardization worthwhile? |
| 35 | [residual-flip-gof-probe](research/35-residual-flip-gof-probe/report.qmd) | probe | active | Can efficient residual-score flipping calibrate single-model SEM goodness of fit without an ad hoc regularizer? |
| 36 | [fiml-score-flip-probe](research/36-fiml-score-flip-probe/report.qmd) | probe | active | Can nuisance-effective and pattern-standardized score flips rescue small-sample nested FIML tests in the published FIML--FMG design? |
| 37 | [fiml-flip-stress](research/37-fiml-flip-stress/report.qmd) | probe | active | Across generator family, sample size, missingness mechanism, and restriction rank, where do direct-FIML effective and standardized score flips remain calibrated? |
| 38 | [score-test-calibration-drivers](research/38-score-test-calibration-drivers/report.qmd) | probe | active | What governs the calibration of the mean-scaled (Satorra–Bentler) non-normal score test — restriction df, structural information geometry, or eigenvalue-spectrum dispersion — and how do the spectrum-aware and sign-flip variants compare? |
| 39 | [fiml-information-choice](research/39-fiml-information-choice/report.qmd) | benchmark | active | For continuous FIML, how do expected Fisher, observed-H1, and full observed-Hessian information compare for model-based and same-meat sandwich SE calibration under complete, MCAR, MAR, nonnormal, and misspecified cells? |
| 40 | [projected-score-satterthwaite](research/40-projected-score-satterthwaite/report.qmd) | probe | active | Can a projected Satterthwaite effective denominator improve calibration of an already-pivotal robust SEM goodness-of-fit score, and is the empirical rule's apparent success genuine self-normalization or moment-estimation bias? |
| 41 | [hotelling-robust-gof-grid](research/41-hotelling-robust-gof-grid/report.qmd) | probe | active | Does the ordinary-Hotelling reference improve a centered pivotal SEM goodness-of-fit score across residual rank and sample-to-rank ratio, relative to its chi-square limit and SB/MV/pEBA4 corrections of the working RLS score? |
| 42 | [psd-ml-small-n-convergence](research/42-psd-ml-small-n-convergence/report.qmd) | benchmark | complete | On the De Jonckere–Rosseel / Ernst small-N SEM design, does covariance-honest NTML converge more reliably than ordinary NTML? |
| 43 | [psd-ml-repair-risk](research/43-psd-ml-repair-risk/report.qmd) | benchmark | active | What does PSD-ML repair when ordinary NTML leaves the primitive covariance domain, and what is its near-boundary estimation risk? |
| 44 | [fiml-global-gof-pilot](research/44-fiml-global-gof-pilot/report.qmd) | probe | active | Across finite-moment stress, MAR, and explicit contract violations, do effective multiplier or spectrum-aware scores calibrate global FIML GOF better than Yuan–Bentler MLR? |
| 45 | [score-vs-lrt](research/45-score-vs-lrt/report.qmd) | benchmark | active | Do score-based SB and pEBA tests calibrate better than LR-based tests, and how long does the paired simulation take? |
| 46 | [latent-metric-geometry](research/46-latent-metric-geometry/report.qmd) | benchmark | active | Which latent scaling convention should a SEM library use internally, is there a better one than the named three, and does the answer survive the move from CFA to structural models? |
| 47 | [multiinfo-penalty-improper](research/47-multiinfo-penalty-improper/report.qmd) | benchmark | active | Do the joint and latent-determinacy (Q) barriers remove improper solutions without costing PSD-constrained ML's accuracy, also with the truth on a face, which fit test goes with a barrier estimate, and does the barrier's local limit hold? |
| 48 | [multiinfo-jeffreys-posterior](research/48-multiinfo-jeffreys-posterior/report.qmd) | benchmark | active | Does the multi-information barrier approximate the Jeffreys-prior posterior over admissible solutions, for which weight, and is that posterior a better estimator? |
| 49 | [spectral-tail-calibration](research/49-spectral-tail-calibration/report.qmd) | benchmark | active | Can spectral shrinkage and debiased moment corrections improve score and LRT calibration under non-normality? |

## Engineering checks

Inform active implementation and default choices; inherited studies await a focused review.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [complete-data-estimator-speed](engineering/01-complete-data-estimator-speed/report.qmd) | benchmark | active | How do the NT/ULS/GLS estimators compare on wall-time across the corpus? |
| 02 | [near-singular-ml-continuation](engineering/02-near-singular-ml-continuation/report.qmd) | benchmark | active | Does shrinkage-blended covariance continuation help ML converge on near-singular problems? |
| 03 | [irls-ernst-convergence](engineering/03-irls-ernst-convergence/report.qmd) | benchmark | complete | Does Fisher-scoring IRLS improve ML convergence on the Ernst small-sample design? |
| 04 | [robust-score-modification-indices](engineering/04-robust-score-modification-indices/report.qmd) | probe | active | Do robust modification indices / score tests change the omitted-path call (ordinal DWLS, continuous GLS), and reduce to naive where theory says c=1? |
| 05 | [pairwise-composite-nested](engineering/05-pairwise-composite-nested/report.qmd) | probe | active | Does the frontier pairwise/composite ordinal estimator produce usable nested LR inference under a small ordinal MCAR loading-equality setup? |
| 06 | [rbm-bias-small](engineering/06-rbm-bias-small/report.qmd) | probe | active | In a small magmaan-owned CFA run, do standard, explicit post-hoc RBM, and implicit integrated RBM differ in finite-sample bias for GLS, FIML, and ordinal DWLS? |
| 07 | [ordinal-observed-omega-dwls](engineering/07-ordinal-observed-omega-dwls/report.qmd) | probe | active | Does observed-category-score omega from an all-ordinal DWLS fit run end-to-end with complete-sandwich delta SEs, and where do balanced vs threshold-extreme smoke cells first bend? |
| 08 | [ordinal-omega-target-audit](engineering/08-ordinal-omega-target-audit/report.qmd) | probe | active | Does the current ordinal observed-score covariance omega equal the direct one-factor ordinal true-score target, or is it only on the same observed-score metric? |
| 09 | [sam-efficiency-stability](engineering/09-sam-efficiency-stability/report.qmd) | benchmark | active | Under normal, native independent-generator, and pseudo-continuous ordinal stress data, how do local SAM and joint ML compare on SE calibration, failures, and runtime at small N? |
| 10 | [guttman-admissibility-clamp](engineering/10-guttman-admissibility-clamp/report.qmd) | benchmark | active | Which hard or soft finite-sample communality clamp best repairs inadmissible aligned Guttman draws without degrading loading RMSE or empirical-SE coverage relative to Raw and NTML? |
| 11 | [psd-ml-timing](engineering/11-psd-ml-timing/report.qmd) | benchmark | active | What does covariance-honest complete-data NTML cost when run directly or only after an ordinary fit fails its covariance audit? |
| 12 | [psd-ml-basin-audit](engineering/12-psd-ml-basin-audit/report.qmd) | benchmark | active | When PSD-ML returns an admissible KKT-stationary solution, how often does a multistart portfolio find a materially better basin? |
| 13 | [psd-estimator-stress](engineering/13-psd-estimator-stress/report.qmd) | benchmark | active | Across the supported single-level estimator families, where do covariance-honest point fits remain correct, admissible, stable across starts, and computationally practical? |
| 14 | [sphere-chart-sanity](engineering/14-sphere-chart-sanity/report.qmd) | probe | active | Does the sphere chart reproduce the standard fit across identification conventions, invariance, constraint syntax and estimators, recover known populations in every identification that holds them, and flag the ones that cannot? |
| 15 | [sphere-local-convergence](engineering/15-sphere-local-convergence/report.qmd) | probe | active | How often does ML reach a certified local optimum on the ordinary and sphere routes under marker and std.lv identification, and are the failures optimizer failures or missing estimates? |
| 16 | [complete-ml-global-test-geometry](engineering/16-complete-ml-global-test-geometry/report.qmd) | benchmark | active | For complete-data ML, which information choices in the global score and LR tests (expected or observed sensitivity, score metric, LR spectrum) hold the nominal size with SB and PEBA4 under normal and severe non-normal data? |
| 17 | [corpus-optimizer-recovery](engineering/17-corpus-optimizer-recovery/report.qmd) | probe | active | On the corrected continuous corpus, which current optimizer policies recover accepted good objectives, do the historical GLS failures still reproduce, and does a layered moment start fix the start-induced failures, and which optimizer coordinates make PORT and L-BFGS indifferent to the data's units? |
| 18 | [barrier-optimizer](engineering/18-barrier-optimizer/report.qmd) | probe | active | Which optimizer should fit the latent-determinacy barrier by default? |
| 19 | [newton-verdict-migration](engineering/19-newton-verdict-migration/report.qmd) | probe | active | What does the Newton check change when it decides the FIML and least-squares verdicts, and is every changed verdict right? |

## Archived

Retired engineering investigations, retained with their sources and local results.
The marker-chart sanity and total-variance studies joined the archive on 2026-09-24.

| Experiment | Question |
|------------|----------|
| [latent-metric-identification](_archive/latent-metric-identification/report.qmd) | Does `std.lv` beat marker-variable parameterization once spec-rebuild and back-conversion costs are counted? |
| [heywood-box-constraints](_archive/heywood-box-constraints/report.qmd) | Do variance box constraints turn inadmissible Heywood ML solutions into admissible boundary optima? |
| [ordinal-snlls-probe](_archive/ordinal-snlls-probe/report.qmd) | Do the cache-aware and SNLLS ordinal paths reproduce the materialized DWLS/WLS fits? |
| [ordinal-inference-cache-probe](_archive/ordinal-inference-cache-probe/report.qmd) | Does carrying a full cache through fitting help robust ordinal reporting requested right after a bounded fit? |
| [ordinal-snlls-speed](_archive/ordinal-snlls-speed/report.qmd) | Does ordinal SNLLS show the same speed pattern as continuous SNLLS once thresholds and ordinal weights are in the objective? |
| [ordinal-threshold-constraints](_archive/ordinal-threshold-constraints/report.qmd) | Which ordinal fitting paths can handle equality constraints on thresholds? |
| [ordinal-construction-boundary](_archive/ordinal-construction-boundary/report.qmd) | What does ordinal statistic construction (lazy vs eager) cost before fitting begins? |
| [noniterative-constraint-charts](_archive/noniterative-constraint-charts/report.qmd) | Does the estimator-side metric Guttman map stay invariant to arbitrary marker-chart choices, on and off the constraint surface? |
| [sem-total-variance](_archive/sem-total-variance/report.qmd) | Engineering decision: retain unscaled native PSD ML; diagonal scaling stays opt-in and boundary-specific restarts remain experimental. |

`_support/` (path, metadata, and I/O helpers; no SEM logic) is the only shared
sibling an experiment may consume. Use category-qualified slugs in references. Numeric prefixes are local ordering,
not permanent study identifiers.
