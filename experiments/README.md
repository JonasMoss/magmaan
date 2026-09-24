# Experiments

A first-pass classification from the report questions, openings, and caveats
(2026-09-24). These are **navigation categories**, not a quality ranking or a
claim that a report is ready for publication. The category folders now reflect this classification; experiments 56 and 83
are archived. Open the
linked report for the evidence; this pass did not rerun or render the studies.

| Purpose | What belongs here | Count |
|---------|-------------------|------:|
| [Showcases](#showcases) | Demonstrate magmaan capabilities, agreement with lavaan/other references, or a concrete performance comparison. | 8 |
| [Replications and reference studies](#replications-and-reference-studies) | Reproduce published results or reconstruct the reference method needed to interpret them. | 9 |
| [Research](#research) | Investigate statistical behavior, new methods, estimands, or inferential validity. | 48 |
| [Engineering checks](#engineering-checks) | Check implementation behavior, performance, robustness, or a library design/default decision. | 15 |
| [Archived](#archived) | Already retired engineering investigations; kept for provenance. | 9 |

The existing **kind** (`parity`, `replication`, `paper-sim`, `benchmark`, `probe`)
and **lifecycle** (`active`, `complete`, `archived`) remain separate metadata.
Lifecycle labels are carried forward, not certified by this skim. In particular,
`complete` does not imply disposable, and `paper-sim` remains protected regardless
of its navigation category. Conventions and shared harness rules are in
[AGENTS.md](AGENTS.md).

Some deliberate borderline decisions:

- **17/18** go with showcases because the reports prominently demonstrate the
  implemented FMG machinery and agreement with reference packages; they also
  reproduce published designs. **69** is primarily the Study 2 replication.
- **20, 42, 43** go with research: named papers motivate a critique, an estimand
  investigation, or new confidence intervals, rather than a straight replication.
- **38** reproduces summaries from authors' saved results; **79** maps reference
  tests; **80** evaluates analogues and explicitly disclaims exact EQS replication.
  Keeping them together does not certify them as exact replications.
- **63** is an illustration of a research comparison, not a settled capability
  showcase. **84** is a showcase candidate with an explicit numerical screening
  caveat; its category is not permission to publish an unrestricted speed claim.
- **51, 60, 75, 88** could sit with research, but their report framing primarily
  supports implementation or default decisions. **82** stays with research because
  the geometric explanation is a substantial part of the question.

## Possible archive candidates — later

**56 and 83 are archived** under `_archive/` (2026-09-24): the bounded
marker-chart check and the closed engineering-default decision are retained
with their existing results for provenance. **14** remains a completed
**Engineering check**; review it during the later cleanup before deciding
whether its convergence study should also be archived.

Do not archive **46/48** merely because they are small probes: they address
implementation and estimand questions whose current resolution needs checking.
Keep published replications, reusable performance studies, and paper-supporting
pipelines available even when complete. Existing archived studies remain where
they are. Experiment **53**, previously absent from this index, is included below.

## Showcases

Demonstrate magmaan capabilities, agreement with lavaan/other references, or a concrete performance comparison.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 00 | [lavaan-parity](showcases/00-lavaan-parity/report.md) | parity | active | Does magmaan's ML inference match lavaan across the textbook corpus? |
| 05 | [lavaan-speed-bench](showcases/05-lavaan-speed-bench/report.md) | benchmark | active | How fast is magmaan versus lavaan on the Geiser textbook corpus? |
| 17 | [foldnes-moss-gronneberg-peba](showcases/17-foldnes-moss-gronneberg-peba/report.qmd) | replication | complete | Do the penalized EBA goodness-of-fit tests hold nominal Type I under nonnormality? |
| 18 | [foldnes-moss-gronneberg-2026](showcases/18-foldnes-moss-gronneberg-2026/report.qmd) | replication | active | Can magmaan reproduce the full FMG goodness-of-fit machinery versus lavaan and semTests? |
| 21 | [fiml-measurement-invariance-fmg](showcases/21-fiml-measurement-invariance-fmg/report.qmd) | paper-sim | active | Is the FIML FMG robust-test family feature-complete for measurement invariance? |
| 31 | [etzel-2024-kas](showcases/31-etzel-2024-kas/report.qmd) | probe | active | Which part of Etzel's OSF Mplus models is inside magmaan's SEM scope? |
| 32 | [schlechter-2024-asci-cfa](showcases/32-schlechter-2024-asci-cfa/report.qmd) | parity | complete | Do Schlechter et al.'s final ASCI ordinal CFA models match lavaan on paper data? |
| 84 | [speed-attribution](showcases/84-speed-attribution/report.qmd) | benchmark | active | How much of matched ML + GOF/Wald runtime is preparation, fitting, or inference, and which comparisons pass numerical agreement before a public speed claim? |

## Replications and reference studies

Reproduce published results or reconstruct the reference method needed to interpret them.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 07 | [maydeu-olivares-2017](replications/07-maydeu-olivares-2017/report.qmd) | replication | complete | Do the SE methods and χ² adjustments hold under nonnormality (two-factor CFA)? |
| 15 | [rhemtulla-2012](replications/15-rhemtulla-2012/report.qmd) | replication | complete | When can ordinal variables be treated as continuous (cat-LS vs continuous ML)? |
| 16 | [li-2021-mixed](replications/16-li-2021-mixed/report.qmd) | replication | complete | DWLS or MLR for a mix of continuous and categorical indicators? |
| 19 | [li-2016-ordinal](replications/19-li-2016-ordinal/report.qmd) | replication | complete | DWLS/ULS or continuous ML for an all-ordinal five-factor SEM? |
| 29 | [chen-2020-wlsmv-pd](replications/29-chen-2020-wlsmv-pd/report.qmd) | replication | active | Can the Chen et al. (2020) WLSMV_PD Type-I inflation cell be reproduced with current lavaan WLSMV pairwise deletion? |
| 38 | [jamil-rosseel-2026-rbm-sem](replications/38-jamil-rosseel-2026-rbm-sem/report.qmd) | replication | active | Can we reproduce the SEM reduced-bias paper's two-factor and growth-curve RBM examples from the authors' OSF outputs before magmaan-owned reruns? |
| 69 | [foldnes-moss-gronneberg-2026-study2](replications/69-foldnes-moss-gronneberg-2026-study2/report.qmd) | replication | complete | Can native magmaan express the 189-cell Study 2 weak-invariance Type-I comparison, including the unbiased-Gamma variants, with semTests retained only as a parity sentinel? |
| 79 | [Savalei-Falk-2014 test map](replications/79-savalei-falk-2014-test-map/report.qmd) | replication | active | Which exact null/reference models and observed-information corrections define Savalei and Falk's robust FIML and two-stage tests, and how should magmaan verify them? |
| 80 | [Savalei-Falk-2014 matrix choices](replications/80-savalei-falk-2014-ml2s-information/report.qmd) | replication | complete | Which Yuan-Bentler-style matrix choice gives rejection behavior most similar to Savalei and Falk, without claiming an exact EQS replication? |

## Research

Investigate statistical behavior, new methods, estimands, or inferential validity.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 08 | [pairwise-gls-efficiency](research/08-pairwise-gls-efficiency/report.qmd) | paper-sim | active | Which of five missing-data estimators is most efficient? |
| 09 | [pairwise-fit-speed](research/09-pairwise-fit-speed/report.qmd) | paper-sim | active | Which pairwise missing-data estimator is fastest? |
| 20 | [deng-chan-2017-alpha-omega](research/20-deng-chan-2017-alpha-omega/report.qmd) | paper-sim | active | Is the Deng-Chan Wald test of coefficient α = ω valid? |
| 23 | [fiml-fmg-vs-mlr](research/23-fiml-fmg-vs-mlr/report.qmd) | paper-sim | active | Under non-normality + MCAR, do the FMG/pEBA FIML goodness-of-fit and nested tests beat the dominant MLR (Yuan-Bentler) default? |
| 24 | [fiml-twostage-fmg-chisq](research/24-fiml-twostage-fmg-chisq/report.qmd) | paper-sim | active | Do the FMG full-spectrum goodness-of-fit chi-squares calibrate FIML and two-stage ML (ML2S) better than the Savalei low-moment corrections under non-normal incomplete data? |
| 25 | [fiml-invariance-fmg-power](research/25-fiml-invariance-fmg-power/report.qmd) | paper-sim | active | Under non-normal incomplete data (FIML and ML2S), do the FMG/pEBA eigenvalue p-values calibrate the measurement-invariance difference test better than the MLR/Satorra-Bentler default, and with what power? (extends Brace & Savalei 2017) |
| 26 | [ordinal-pd-gamma](research/26-ordinal-pd-gamma/report.qmd) | probe | active | Does overlap-weighting the ordinal pairwise-deletion Gamma fix the nominal-N WLSMV missing-data scaling problem? |
| 28 | [ordinal-stage2-pairwise](research/28-ordinal-stage2-pairwise/report.qmd) | benchmark | active | When pairwise ordinal Gamma is reused, how do ULS/DWLS/WLS/NT/DLS stage-two estimators compare on p-values, SE diagnostics, and runtime? |
| 30 | [rmsea-like-catml-dwls](research/30-rmsea-like-catml-dwls/report.qmd) | probe | active | Does lavaan's categorical robust RMSEA behave as a consistent CATML-at-DWLS criterion-at-estimator statistic under ordinal misspecification? |
| 33 | [mplus-demo-wlsmv-difftest](research/33-mplus-demo-wlsmv-difftest/report.qmd) | probe | active | Does Mplus Demo WLSMV DIFFTEST for a demo-sized ordinal pairwise-missing invariance model match lavaan/magmaan Satorra-2000 statistics? |
| 34 | [chen-ordinal-fmg-pvalues](research/34-chen-ordinal-fmg-pvalues/report.qmd) | probe | active | Is the WLSMV pairwise-missing scalar Type-I inflation a defect of the test/p-value family or of the missing-data mechanism (MCAR vs MAR)? |
| 35 | [misspec-robust-se](research/35-misspec-robust-se/report.qmd) | probe | active | Does the observed-Hessian ("robust" regime) bread recover the true sampling SD of ordinal DWLS estimates under structural misspecification, while coinciding with the conventional SE under the null? |
| 36 | [ordinal-dwls-profile-lrt](research/36-ordinal-dwls-profile-lrt/report.md) | paper-sim | active | Does the standard scaled difference test for nested all-ordinal DWLS models stay calibrated when the larger model is misspecified, and does the estimated-weight profile law restore calibration? |
| 37 | [mixed-fiml-pairwise-efficiency](research/37-mixed-fiml-pairwise-efficiency/report.qmd) | benchmark | active | Under item missingness in mixed continuous/ordinal SEM, does a continuous-FIML first stage (pairwise x FIML) buy efficiency and reduce MAR bias over fully pairwise statistics, and on which parameter block? |
| 40 | [guttman-cfa-asymptotics](research/40-guttman-cfa-asymptotics/report.qmd) | benchmark | active | How do Guttman's closed-form CFA estimates compare with NT/ULS/GLS/WLS in asymptotic variance and fixed-misspecification risk? |
| 41 | [reliability-lambda6](research/41-reliability-lambda6/report.qmd) | benchmark | active | How do alpha, Guttman's lambda6, covariance omega, and fitted NT omega compare for total-score reliability under correct and misspecified one-factor summaries? |
| 42 | [bell-omega-bias](research/42-bell-omega-bias/report.qmd) | benchmark | active | At Bell-style population moments, how biased is ordinary one-factor omega under misspecification for ML, ULS, GLS, and the Spearman-Guttman covariance form? |
| 43 | [li-savalei-2026-maximal-reliability-ci](research/43-li-savalei-2026-maximal-reliability-ci/report.qmd) | replication | active | Can you put a trustworthy confidence interval around bifactor maximal reliability, the inference Li & Savalei (2026) leave open? |
| 44 | [alpha-kc-coverage](research/44-alpha-kc-coverage/report.qmd) | benchmark | active | A distribution-free interval for a covariance functional (alpha, correlation) has two small-N defects: does the Kauermann-Carroll effective-df t fix the two-sided coverage deficit (variance-of-variance, flat in rho) while a variance-stabilizing transform (Fisher z / logit) fixes the skew-driven left/right imbalance (growing in rho), and do the two compose? |
| 45 | [profile-lr-reliability-ci](research/45-profile-lr-reliability-ci/report.qmd) | probe | active | **funLR (Functional profile-LR CI).** Can a generic profile-LR (test-inversion) engine reproduce semlbci across the reliability family (omega_total, omega_h, H, maximal reliability), and can the small-sample under-coverage of near-boundary coefficients be repaired by a Bartlett factor (analytic, bootstrap, or calibrated constant)? |
| 47 | [ml-parameter-profile-lrt](research/47-ml-parameter-profile-lrt/report.qmd) | probe | active | Does the new C++ complete-data ML scalar/profile-LR seed calibrate as ordinary `chi^2_1` for an interior free parameter at small N, before robust scaling and bounded functionals are layered on? |
| 49 | [ordinal-polychoric-omega-coverage](research/49-ordinal-polychoric-omega-coverage/report.qmd) | probe | active | Does the robust delta interval for no-integration ordinal polychoric omega cover its latent-response target across small-N balanced and threshold-extreme ordinal cells? |
| 50 | [ordinal-polychoric-omega-stress](research/50-ordinal-polychoric-omega-stress/report.qmd) | probe | active | Under nonnormal or locally dependent ordinal data, does the robust delta interval for no-integration ordinal polychoric omega cover the pseudo-true polychoric target? |
| 52 | [noniterative-cfa-tests](research/52-noniterative-cfa-tests/report.qmd) | benchmark | active | Are delta-method SEs, residual goodness-of-fit, and difference tests for the closed-form Guttman CFA estimator calibrated in finite samples, and at what efficiency cost vs ML, across normal / independent-generator / ordinal-as-continuous data? |
| 53 | [ordinal-profile-lrt-calibration](research/53-ordinal-profile-lrt-calibration/report.qmd) | probe | active | What causes small-sample miscoverage of ordinal polychoric-omega profile-LRT intervals: LR inflation, scaling, sparse summaries, bias, or misspecification? |
| 54 | [noniterative-invariance](research/54-noniterative-invariance/report.qmd) | benchmark | active | Are the closed-form Guttman metric (projection Wald) and scalar (reference-group mean map) measurement-invariance tests calibrated and powered vs the ML likelihood-ratio test, and does the empirical fourth-moment matrix restore the level under non-normal data? |
| 55 | [guttman-communality-estimators](research/55-guttman-communality-estimators/report.qmd) | benchmark | active | Which closed-form communality rule should the Guttman CFA paper recommend once RS, ILM, plug-in triad GMM, magmaan NT ML, and an ordinal Pearson-code stress arm are compared? |
| 57 | [guttman-standardized-hs](research/57-guttman-standardized-hs/report.qmd) | benchmark | active | On a Holzinger-Swineford-shaped CFA, how do standardized augmented Guttman ILS estimates, intervals, and tests compare with NTML and ULS? |
| 58 | [guttman-rmse-coverage](research/58-guttman-rmse-coverage/report.qmd) | paper-sim | active | When does the chosen raw Guttman recipe (extended_triad_ls + standardized) retain useful RMSE, coverage, and admissibility, configural and correctly restricted, relative to legacy Guttman, ML, and ULS? |
| 59 | [guttman-vs-ntml](research/59-guttman-vs-ntml/report.qmd) | benchmark | active | Does the chosen closed-form recipe (extended_triad_ls + standardized, aligned configural) match NTML on parameter RMSE and empirical-SE coverage, including the high-rho/weak-loading regime, at a modest Sigma-fit cost while being faster and never failing? |
| 61 | [permutation-measurement-invariance](research/61-permutation-measurement-invariance/report.qmd) | probe | active | Can fully recomputed pivotal score or Wald label permutations calibrate continuous metric-invariance tests under heterogeneous group populations, and can a Hotelling reference change the permutation-score test? |
| 62 | [standardized-flip-score](research/62-standardized-flip-score/report.qmd) | probe | active | Do nuisance-effective and flip-specifically standardized score flips calibrate continuous metric-invariance tests at small N? |
| 63 | [fmg-score-flip-illustration](research/63-fmg-score-flip-illustration/report.qmd) | probe | complete | In a focused published weak-invariance design, how do score flips compare with the basic FMG nested-test variants? |
| 64 | [flip-calibration-frontier](research/64-flip-calibration-frontier/report.qmd) | probe | active | At fixed test rank, when do nuisance dimension, allocation, information geometry, and sample size make flip standardization worthwhile? |
| 65 | [residual-flip-gof-probe](research/65-residual-flip-gof-probe/report.qmd) | probe | active | Can efficient residual-score flipping calibrate single-model SEM goodness of fit without an ad hoc regularizer? |
| 66 | [fiml-score-flip-probe](research/66-fiml-score-flip-probe/report.qmd) | probe | active | Can nuisance-effective and pattern-standardized score flips rescue small-sample nested FIML tests in the published FIML--FMG design? |
| 67 | [fiml-flip-stress](research/67-fiml-flip-stress/report.qmd) | probe | active | Across generator family, sample size, missingness mechanism, and restriction rank, where do direct-FIML effective and standardized score flips remain calibrated? |
| 68 | [score-test-calibration-drivers](research/68-score-test-calibration-drivers/report.qmd) | probe | active | What governs the calibration of the mean-scaled (Satorra–Bentler) non-normal score test — restriction df, structural information geometry, or eigenvalue-spectrum dispersion — and how do the spectrum-aware and sign-flip variants compare? |
| 70 | [fiml-information-choice](research/70-fiml-information-choice/report.qmd) | benchmark | active | For continuous FIML, how do expected Fisher, observed-H1, and full observed-Hessian information compare for model-based and same-meat sandwich SE calibration under complete, MCAR, MAR, nonnormal, and misspecified cells? |
| 71 | [projected-score-satterthwaite](research/71-projected-score-satterthwaite/report.qmd) | probe | active | Can a projected Satterthwaite effective denominator improve calibration of an already-pivotal robust SEM goodness-of-fit score, and is the empirical rule's apparent success genuine self-normalization or moment-estimation bias? |
| 72 | [hotelling-robust-gof-grid](research/72-hotelling-robust-gof-grid/report.qmd) | probe | active | Does the ordinary-Hotelling reference improve a centered pivotal SEM goodness-of-fit score across residual rank and sample-to-rank ratio, relative to its chi-square limit and SB/MV/pEBA4 corrections of the working RLS score? |
| 74 | [psd-ml-small-n-convergence](research/74-psd-ml-small-n-convergence/report.qmd) | benchmark | complete | On the De Jonckere–Rosseel / Ernst small-N SEM design, does covariance-honest NTML converge more reliably than ordinary NTML? |
| 76 | [psd-ml-repair-risk](research/76-psd-ml-repair-risk/report.qmd) | benchmark | active | What does PSD-ML repair when ordinary NTML leaves the primitive covariance domain, and what is its near-boundary estimation risk? |
| 77 | [fiml-global-gof-pilot](research/77-fiml-global-gof-pilot/report.qmd) | probe | active | Across finite-moment stress, MAR, and explicit contract violations, do effective multiplier or spectrum-aware scores calibrate global FIML GOF better than Yuan–Bentler MLR? |
| 81 | [score-vs-lrt](research/81-score-vs-lrt/report.qmd) | benchmark | active | Do score-based SB and pEBA tests calibrate better than LR-based tests, and how long does the paired simulation take? |
| 82 | [latent-metric-geometry](research/82-latent-metric-geometry/report.qmd) | benchmark | active | Which latent scaling convention should a SEM library use internally, is there a better one than the named three, and does the answer survive the move from CFA to structural models? |
| 85 | [multiinfo-penalty-improper](research/85-multiinfo-penalty-improper/report.qmd) | benchmark | active | Does the complete-data multi-information penalty remove improper and boundary solutions without costing ML's accuracy and calibration, and how large should its weight be? |
| 86 | [multiinfo-jeffreys-posterior](research/86-multiinfo-jeffreys-posterior/report.qmd) | benchmark | active | Does the multi-information barrier approximate the Jeffreys-prior posterior over admissible solutions, for which weight, and is that posterior a better estimator? |

## Engineering checks

Check implementation behavior, performance, robustness, or a library design/default decision.

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 01 | [complete-data-estimator-speed](engineering/01-complete-data-estimator-speed/report.qmd) | benchmark | active | How do the NT/ULS/GLS estimators compare on wall-time across the corpus? |
| 04 | [near-singular-ml-continuation](engineering/04-near-singular-ml-continuation/report.qmd) | benchmark | active | Does shrinkage-blended covariance continuation help ML converge on near-singular problems? |
| 14 | [irls-ernst-convergence](engineering/14-irls-ernst-convergence/report.qmd) | benchmark | complete | Does Fisher-scoring IRLS improve ML convergence on the Ernst small-sample design? |
| 22 | [robust-score-modification-indices](engineering/22-robust-score-modification-indices/report.qmd) | probe | active | Do robust modification indices / score tests change the omitted-path call (ordinal DWLS, continuous GLS), and reduce to naive where theory says c=1? |
| 27 | [pairwise-composite-nested](engineering/27-pairwise-composite-nested/report.qmd) | probe | active | Does the frontier pairwise/composite ordinal estimator produce usable nested LR inference under a small ordinal MCAR loading-equality setup? |
| 39 | [rbm-bias-small](engineering/39-rbm-bias-small/report.qmd) | probe | active | In a small magmaan-owned CFA run, do standard, explicit post-hoc RBM, and implicit integrated RBM differ in finite-sample bias for GLS, FIML, and ordinal DWLS? |
| 46 | [ordinal-observed-omega-dwls](engineering/46-ordinal-observed-omega-dwls/report.qmd) | probe | active | Does observed-category-score omega from an all-ordinal DWLS fit run end-to-end with complete-sandwich delta SEs, and where do balanced vs threshold-extreme smoke cells first bend? |
| 48 | [ordinal-omega-target-audit](engineering/48-ordinal-omega-target-audit/report.qmd) | probe | active | Does the current ordinal observed-score covariance omega equal the direct one-factor ordinal true-score target, or is it only on the same observed-score metric? |
| 51 | [sam-efficiency-stability](engineering/51-sam-efficiency-stability/report.qmd) | benchmark | active | Under normal, native independent-generator, and pseudo-continuous ordinal stress data, how do local SAM and joint ML compare on SE calibration, failures, and runtime at small N? |
| 60 | [guttman-admissibility-clamp](engineering/60-guttman-admissibility-clamp/report.qmd) | benchmark | active | Which hard or soft finite-sample communality clamp best repairs inadmissible aligned Guttman draws without degrading loading RMSE or empirical-SE coverage relative to Raw and NTML? |
| 73 | [psd-ml-timing](engineering/73-psd-ml-timing/report.qmd) | benchmark | active | What does covariance-honest complete-data NTML cost when run directly or only after an ordinary fit fails its covariance audit? |
| 75 | [psd-ml-basin-audit](engineering/75-psd-ml-basin-audit/report.qmd) | benchmark | active | When PSD-ML returns an admissible KKT-stationary solution, how often does a multistart portfolio find a materially better basin? |
| 78 | [psd-estimator-stress](engineering/78-psd-estimator-stress/report.qmd) | benchmark | active | Across the supported single-level estimator families, where do covariance-honest point fits remain correct, admissible, stable across starts, and computationally practical? |
| 87 | [sphere-chart-sanity](engineering/87-sphere-chart-sanity/report.qmd) | probe | active | Does the sphere chart reproduce the standard fit across identification conventions, invariance, constraint syntax and estimators, recover known populations in every identification that holds them, and flag the ones that cannot? |
| 88 | [sphere-local-convergence](engineering/88-sphere-local-convergence/report.qmd) | probe | active | How often does ML reach a certified local optimum on the ordinary and sphere routes under marker and std.lv identification, and are the failures optimizer failures or missing estimates? |

## Archived

Retired engineering investigations. Experiments 56 and 83 were archived on
2026-09-24; their sources and local results were preserved.

| # | Experiment | Question |
|--:|------------|----------|
| 02 | [latent-metric-identification](_archive/02-latent-metric-identification/report.qmd) | Does `std.lv` beat marker-variable parameterization once spec-rebuild and back-conversion costs are counted? |
| 03 | [heywood-box-constraints](_archive/03-heywood-box-constraints/report.qmd) | Do variance box constraints turn inadmissible Heywood ML solutions into admissible boundary optima? |
| 06 | [ordinal-snlls-probe](_archive/06-ordinal-snlls-probe/report.qmd) | Do the cache-aware and SNLLS ordinal paths reproduce the materialized DWLS/WLS fits? |
| 10 | [ordinal-inference-cache-probe](_archive/10-ordinal-inference-cache-probe/report.qmd) | Does carrying a full cache through fitting help robust ordinal reporting requested right after a bounded fit? |
| 11 | [ordinal-snlls-speed](_archive/11-ordinal-snlls-speed/report.qmd) | Does ordinal SNLLS show the same speed pattern as continuous SNLLS once thresholds and ordinal weights are in the objective? |
| 12 | [ordinal-threshold-constraints](_archive/12-ordinal-threshold-constraints/report.qmd) | Which ordinal fitting paths can handle equality constraints on thresholds? |
| 13 | [ordinal-construction-boundary](_archive/13-ordinal-construction-boundary/report.qmd) | What does ordinal statistic construction (lazy vs eager) cost before fitting begins? |
| 56 | [noniterative-constraint-charts](_archive/56-noniterative-constraint-charts/report.qmd) | Does the estimator-side metric Guttman map stay invariant to arbitrary marker-chart choices, on and off the constraint surface? |
| 83 | [sem-total-variance](_archive/83-sem-total-variance/report.qmd) | Engineering decision: retain unscaled native PSD ML; diagonal scaling stays opt-in and boundary-specific restarts remain experimental. |

`_support/` (path, metadata, and I/O helpers; no SEM logic) is the only shared
sibling an experiment may consume. Numbers are permanent IDs; classifications
do not rename or renumber experiments.
