# Tier-1 representative basis (sem-score models)

Copied 2026-10-09 from the private `papers/sem-score-tests` repository
(commit ce775a3, `code/studies/calibration/populations/`) so experiments
never read a paper repository (experiments/AGENTS.md). Plain data only:
each file holds the corpus source, `h1` (correct model, used here as the
fitted model), the paper's restricted `h0`, and population values per group.
Do not edit in place; replace with a new dated copy and update this table.
See project/design/simulation-study-principles.md (tier 1).

| File | SHA-256 | Type | Corpus source |
|---|---|---|---|
| cfa_bifactor15.json | b3d112bd6bd6b1db | bifactor CFA, general + 2 specific factors, one correlated residual | brown_2015_tab8_6_bifactor_perceived_control |
| cfa_corr24.json | 96dca29d35970e6e | CFA, 8 correlated factors x 3 items, means, effects coding | little_2013_ch10_tab3_correlation |
| cfa_long18.json | 74b0bd1c6d21c07e | longitudinal CFA, means, effects coding, lagged residual covariances | little_2013_ch5_tab5_5_configfi_2by3 |
| cfa_long30.json | 34e0ee4b9f566085 | longitudinal CFA, 2 constructs x 5 waves x 3 items, means, effects coding | little_2013_ch8_tab8_bull_config |
| cfa_mtmm9.json | 550ae996baf0f861 | CFA, MTMM 3 traits x 3 methods, correlated uniquenesses | brown_2015_tab6_3_mtmm_correlated_uniqueness |
| cfa_ptsd8.json | 2fdd0b5a5ffcda79 | CFA, 2 factors x 4 items, one correlated residual | brown_2015_tab8_8_reliability_ptsd |
| cfa_second12.json | b9425925f50408f6 | second-order CFA, 4 first-order factors, 2 second-order factors | brown_2015_tab8_2_higher_order_coping |
| growth6.json | f0ef3dfa7ca07ac9 | linear growth of CES-D, 6 waves, structured means | newsom_2015_ex7_6c |
| mg_path5.json | 80c6b4064ee5411f | two-group path model, means, cross-group equalities | kline_2023_ch12_lynam_indirect |
| mgcfa_mdd9.json | 615d8b1f46482639 | two-group one-factor CFA, metric invariance, latent mean difference | brown_2015_tab7_14_invariance_mdd_scalar |
| mimic7.json | e0e491ec57009b16 | MIMIC, 2 factors x 3 items, one covariate with a direct effect | brown_2015_tab7_17_mimic_phobia |
| sem_worland11.json | 01c75fbbaeaeb74a | latent SEM, two structural equations | kline_2023_ch15_worland_sr_step2a |
