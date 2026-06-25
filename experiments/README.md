# Experiments

Each folder answers **one** research or engineering question and is a standalone,
reproducible endpoint. This file is the map; open an experiment's `report.qmd` (or
`report.md`) for the full story. Conventions, the canonical folder shape, and the
shared `_support` harness live in [AGENTS.md](AGENTS.md).

**Kind**
- `parity` — audits magmaan output against lavaan.
- `replication` — reproduces a published simulation study (named author-year).
- `paper-sim` — a simulation pipeline a paper's results depend on; do not delete.
- `benchmark` — speed or statistical-efficiency comparison.
- `probe` — a one-off engineering diagnostic.

**Lifecycle**
- `active` — still rerun, extended, or load-bearing.
- `complete` — finished, kept flat for reference value.
- `archived` — inert; answer baked into the core library; moved to
  [`_archive/`](_archive/).

## Active

| # | Experiment | Kind | Lifecycle | Question |
|--:|------------|------|-----------|----------|
| 00 | [lavaan-parity](00-lavaan-parity/report.md) | parity | active | Does magmaan's ML inference match lavaan across the textbook corpus? |
| 01 | [complete-data-estimator-speed](01-complete-data-estimator-speed/report.qmd) | benchmark | active | How do the NT/ULS/GLS estimators compare on wall-time across the corpus? |
| 04 | [near-singular-ml-continuation](04-near-singular-ml-continuation/report.qmd) | benchmark | active | Does shrinkage-blended covariance continuation help ML converge on near-singular problems? |
| 05 | [lavaan-speed-bench](05-lavaan-speed-bench/report.md) | benchmark | active | How fast is magmaan versus lavaan on the Geiser textbook corpus? |
| 07 | [maydeu-olivares-2017](07-maydeu-olivares-2017/report.qmd) | replication | complete | Do the SE methods and χ² adjustments hold under nonnormality (two-factor CFA)? |
| 08 | [pairwise-gls-efficiency](08-pairwise-gls-efficiency/report.qmd) | paper-sim | active | Which of five missing-data estimators is most efficient? |
| 09 | [pairwise-fit-speed](09-pairwise-fit-speed/report.qmd) | paper-sim | active | Which pairwise missing-data estimator is fastest? |
| 14 | [irls-ernst-convergence](14-irls-ernst-convergence/report.qmd) | benchmark | complete | Does Fisher-scoring IRLS improve ML convergence on the Ernst small-sample design? |
| 15 | [rhemtulla-2012](15-rhemtulla-2012/report.qmd) | replication | complete | When can ordinal variables be treated as continuous (cat-LS vs continuous ML)? |
| 16 | [li-2021-mixed](16-li-2021-mixed/report.qmd) | replication | complete | DWLS or MLR for a mix of continuous and categorical indicators? |
| 17 | [foldnes-moss-gronneberg-peba](17-foldnes-moss-gronneberg-peba/report.qmd) | replication | complete | Do the penalized EBA goodness-of-fit tests hold nominal Type I under nonnormality? |
| 18 | [foldnes-moss-gronneberg-2026](18-foldnes-moss-gronneberg-2026/report.qmd) | replication | active | Can magmaan reproduce the full FMG goodness-of-fit machinery versus lavaan and semTests? |
| 19 | [li-2016-ordinal](19-li-2016-ordinal/report.qmd) | replication | complete | DWLS/ULS or continuous ML for an all-ordinal five-factor SEM? |
| 20 | [deng-chan-2017-alpha-omega](20-deng-chan-2017-alpha-omega/report.qmd) | paper-sim | active | Is the Deng-Chan Wald test of coefficient α = ω valid? |
| 21 | [fiml-measurement-invariance-fmg](21-fiml-measurement-invariance-fmg/report.qmd) | paper-sim | active | Is the FIML FMG robust-test family feature-complete for measurement invariance? |
| 22 | [robust-score-modification-indices](22-robust-score-modification-indices/report.qmd) | probe | active | Do robust modification indices / score tests change the omitted-path call (ordinal DWLS, continuous GLS), and reduce to naive where theory says c=1? |
| 23 | [fiml-fmg-vs-mlr](23-fiml-fmg-vs-mlr/report.qmd) | paper-sim | active | Under non-normality + MCAR, do the FMG/pEBA FIML goodness-of-fit and nested tests beat the dominant MLR (Yuan-Bentler) default? |
| 24 | [fiml-twostage-fmg-chisq](24-fiml-twostage-fmg-chisq/report.qmd) | paper-sim | active | Do the FMG full-spectrum goodness-of-fit chi-squares calibrate FIML and two-stage ML (ML2S) better than the Savalei low-moment corrections under non-normal incomplete data? |
| 25 | [fiml-invariance-fmg-power](25-fiml-invariance-fmg-power/report.qmd) | paper-sim | active | Under non-normal incomplete data (FIML and ML2S), do the FMG/pEBA eigenvalue p-values calibrate the measurement-invariance difference test better than the MLR/Satorra-Bentler default, and with what power? (extends Brace & Savalei 2017) |
| 26 | [ordinal-pd-gamma](26-ordinal-pd-gamma/report.qmd) | probe | active | Does overlap-weighting the ordinal pairwise-deletion Gamma fix the nominal-N WLSMV missing-data scaling problem? |
| 27 | [pairwise-composite-nested](27-pairwise-composite-nested/report.qmd) | probe | active | Does the frontier pairwise/composite ordinal estimator produce usable nested LR inference under a small ordinal MCAR loading-equality setup? |
| 28 | [ordinal-stage2-pairwise](28-ordinal-stage2-pairwise/report.qmd) | benchmark | active | When pairwise ordinal Gamma is reused, how do ULS/DWLS/WLS/NT/DLS stage-two estimators compare on p-values, SE diagnostics, and runtime? |
| 29 | [chen-2020-wlsmv-pd](29-chen-2020-wlsmv-pd/report.qmd) | replication | active | Can the Chen et al. (2020) WLSMV_PD Type-I inflation cell be reproduced with current lavaan WLSMV pairwise deletion? |
| 30 | [rmsea-like-catml-dwls](30-rmsea-like-catml-dwls/report.qmd) | probe | active | Does lavaan's categorical robust RMSEA behave as a consistent CATML-at-DWLS criterion-at-estimator statistic under ordinal misspecification? |
| 31 | [etzel-2024-kas](31-etzel-2024-kas/report.qmd) | probe | active | Which part of Etzel's OSF Mplus models is inside magmaan's SEM scope? |
| 32 | [schlechter-2024-asci-cfa](32-schlechter-2024-asci-cfa/report.qmd) | parity | complete | Do Schlechter et al.'s final ASCI ordinal CFA models match lavaan on paper data? |
| 33 | [mplus-demo-wlsmv-difftest](33-mplus-demo-wlsmv-difftest/report.qmd) | probe | active | Does Mplus Demo WLSMV DIFFTEST for a demo-sized ordinal pairwise-missing invariance model match lavaan/magmaan Satorra-2000 statistics? |
| 34 | [chen-ordinal-fmg-pvalues](34-chen-ordinal-fmg-pvalues/report.qmd) | probe | active | Is the WLSMV pairwise-missing scalar Type-I inflation a defect of the test/p-value family or of the missing-data mechanism (MCAR vs MAR)? |
| 35 | [misspec-robust-se](35-misspec-robust-se/report.qmd) | probe | active | Does the observed-Hessian ("robust" regime) bread recover the true sampling SD of ordinal DWLS estimates under structural misspecification, while coinciding with the conventional SE under the null? |
| 36 | [ordinal-dwls-profile-lrt](36-ordinal-dwls-profile-lrt/report.md) | paper-sim | active | Does the standard scaled difference test for nested all-ordinal DWLS models stay calibrated when the larger model is misspecified, and does the estimated-weight profile law restore calibration? |
| 37 | [mixed-fiml-pairwise-efficiency](37-mixed-fiml-pairwise-efficiency/report.qmd) | benchmark | active | Under item missingness in mixed continuous/ordinal SEM, does a continuous-FIML first stage (pairwise x FIML) buy efficiency and reduce MAR bias over fully pairwise statistics, and on which parameter block? |
| 38 | [jamil-rosseel-2026-rbm-sem](38-jamil-rosseel-2026-rbm-sem/report.qmd) | replication | active | Can we reproduce the SEM reduced-bias paper's two-factor and growth-curve RBM examples from the authors' OSF outputs before magmaan-owned reruns? |

## Archived

Frozen engineering probes whose question is answered and whose result is now baked
into the core library. Kept for provenance; not rerun. See [`_archive/`](_archive/).

| # | Experiment | Question |
|--:|------------|----------|
| 02 | [latent-metric-identification](_archive/02-latent-metric-identification/report.qmd) | Does `std.lv` beat marker-variable parameterization once spec-rebuild and back-conversion costs are counted? |
| 03 | [heywood-box-constraints](_archive/03-heywood-box-constraints/report.qmd) | Do variance box constraints turn inadmissible Heywood ML solutions into admissible boundary optima? |
| 06 | [ordinal-snlls-probe](_archive/06-ordinal-snlls-probe/report.qmd) | Do the cache-aware and SNLLS ordinal paths reproduce the materialized DWLS/WLS fits? |
| 10 | [ordinal-inference-cache-probe](_archive/10-ordinal-inference-cache-probe/report.qmd) | Does carrying a full cache through fitting help robust ordinal reporting requested right after a bounded fit? |
| 11 | [ordinal-snlls-speed](_archive/11-ordinal-snlls-speed/report.qmd) | Does ordinal SNLLS show the same speed pattern as continuous SNLLS once thresholds and ordinal weights are in the objective? |
| 12 | [ordinal-threshold-constraints](_archive/12-ordinal-threshold-constraints/report.qmd) | Which ordinal fitting paths can handle equality constraints on thresholds? |
| 13 | [ordinal-construction-boundary](_archive/13-ordinal-construction-boundary/report.qmd) | What does ordinal statistic construction (lazy vs eager) cost before fitting begins? |

---

`_support/` (path / metadata / IO helpers, no SEM logic) is the only shared sibling
an experiment may consume. Numbers are permanent IDs: an archived experiment keeps
its number, and new experiments take the next free one regardless of gaps.
