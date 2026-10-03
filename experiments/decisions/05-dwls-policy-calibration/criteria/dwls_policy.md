# DWLS policy calibration criteria

Registered before smoke or pilot. Calibration evidence only; the recipes are fixed by the misspecification-robust requirement in project/scope.md. Covariance is the estimated-weight IJ sandwich; the global test is n F with SB/PEBA4 on the robust_ordinal spectrum; the nested test uses the estimated-weight profile law with SB/PEBA4. Comparators never change the recipe.

Design (write into the criteria verbatim). Generator: thresholded multivariate normal (Gaussian copula); 2,000 replicates per null/coverage cell, 1,000 per power cell.
- Global size: one-factor 6 indicators (df 9), two-factor 12 (df 53), three-factor 18 (df 132) x categories {2, 5} x thresholds {symmetric, skewed (binary p = .15; 5-category moderate skew)} x N {200, 500, 1000} = 36 cells; plus two groups (configural): two-factor 12 and three-factor 18 x {2, 5} x N {400, 1000} x {delta, theta} = 16 cells. Arms: policy SB, policy PEBA4; comparators lavaan-style scaled-shifted and mean-variance adjusted statistics.
- Nested: two groups with identical populations; metric vs configural and thresholds+loadings vs loadings (theta); larger model correct or misspecified (an omitted cross-loading of .3 in both groups, so invariance holds at the pseudo-true value): two-factor 12 and three-factor 18 x {2, 5} x {correct, misspecified} x N {400, 1000} x 2 nestings = 32 cells. Arms: policy profile law SB/PEBA4; comparator fixed-weight Satorra-2000 SB/PEBA4 (at the H1 point). Record df and spectrum size.
- Coverage: IJ (policy) vs expected-bread and observed-bread sandwiches (comparators) for a loading, a factor correlation, a threshold and a structural path (add a regression between factors in a two-factor model); misspecification {0, .2, .4} (omitted cross-loading); N {150, 300, 1000}; categories {2, 5}; plus a theta two-group subset: about 40 cells. Pseudo-true values from a population DWLS fit with the population weight from one very large draw (state N and seed).
- Power: global with an omitted cross-loading; nested with one non-invariant loading and one non-invariant threshold (one cell family each, size-adjusted).
Reporting rule (registered): calibration evidence only. Report per-cell rejection with Wilson intervals, coverage, failures and comparator arms. Flag a cell for finite-sample follow-up when a policy arm rejects outside [3%, 7%] at N >= 500, or IJ coverage falls below 93% at N >= 300; flags never switch the recipe.


## Operational definitions

N denotes total observations, divided equally between groups. Loadings cycle through .65, .70, .75, .80, .70, .75; factor correlations are .3. Residual variances standardize every latent response. Cross-loadings affect the first indicator of factor 2 on factor 1. Symmetric five-category probabilities are (.2,.2,.2,.2,.2); skewed probabilities are (.45,.25,.15,.10,.05). Binary skew uses upper-category probability .15. Marker-variable identification uses the actual ordinary model routes.

Coverage has 36 single-group cells: covariance CFA and regression SEM crossed with 3 misspecification levels, 3 sample sizes and 2 category counts. Six theta two-group covariance-CFA cells add N {300,1000}, categories {2,5} at misspecification .2, plus N=300 categories {2,5} at .4: 42 cells. Targets are a non-marker loading, factor correlation (delta method from covariance/variances), first threshold, and structural path in the SEM. Population targets use one fixed 100,000-row draw per group, seed base 817120001, fitted with its population DWLS weight; Monte Carlo approximation is recorded and is a limitation.

Global power adds cross-loading .3 to the 12 single-group symmetric two/three-factor cells at N {200,500,1000}, categories {2,5}. Nested power adds a group-b loading increase .15 or first-threshold shift .3, in the corresponding metric or threshold nesting, for two/three factors, categories {2,5}, N {400,1000}: 16 cells. Each power cell uses its matched null cell and arm-specific empirical 95th percentile of the scaled statistic to estimate size-adjusted power; pilot estimates are engineering checks only.

Smoke, pilot and production use distinct deterministic seed bases 817130001, 817140001 and 817150001. Smoke uses 2 and pilot 20 draws per cell. Production uses the counts above, requires a separate user compute decision, and is never launched by task-17.1. Batch size 20; at most four workers, one math thread each. Library fit$converged is the acceptance verdict; errors, nonconvergence and unavailable arms stay in denominators and failure summaries.

Per-replicate policy gaps compare explicit library primitive covariance/statistic/SB/PEBA4 to policy_inference()/policy_nested(), tolerance 1e-7. Record restriction df separately from spectrum size. Wilson intervals use successful-arm denominators alongside attempted/failed counts. Report coverage per target and cell, never pooled. Follow-up flags never switch the recipes. Freeze pilot summary CSVs and metadata only; raw rows remain ignored.

## Amendment 2026-10-03 (after availability preflight, before pilot)

Board task-17.1 comment 6 corrects the second theta nesting: H1 uses group_equal = "thresholds"; H0 uses c("thresholds", "loadings"). This is the Wu–Estabrook metric step, with loading restrictions. Thresholds-plus-loadings versus loadings-only releases group-2 response scales/intercepts and is not nested by parameter restriction; the recorded not_nested verdict was correct. All other registered design choices remain as written. The threshold-shift power family is retained as requested, but its interpretation must respect this corrected pair.

## Amendment 2026-10-03 (after corrected preflight, before smoke or pilot)

Board task-17.1 comment 10 drops the threshold-shift nested power family.
Both models in the corrected Wu–Estabrook metric step impose equal thresholds,
so a threshold perturbation misspecifies both and is not threshold-invariance
power. Keep the non-invariant-loading family for the loading step (metric
versus configural); eight nested power cells remain, for 146 cells overall.
A genuine threshold-invariance test requires a separately validated restriction
map and is outside this study. All other design choices remain unchanged.

## Amendment 2026-10-03 (after production, before confirmation)

Production (`results/dwls-policy/production-2026-10-03`) found the policy
global SB and PEBA4 references over-rejecting at df >= 53 (SB median 11%,
up to 20.5%; still 8-12% at N = 1000), while the scaled-shifted adjustment of
the same robust-ordinal spectrum stayed within 3.5-7.8%. Size-adjusted power
was identical across arms. The statistic (n F), its spectrum and the IJ
covariance stay; only the reference-law approximation is reconsidered, which
keeps the misspecification-robust ingredients.

Production evaluated four of the implemented reference laws and did not save
spectra. Before choosing, an exploratory rerun of the 64 global cells
(52 null, 12 power) uses the production seed base 817150001, so its draws equal
production, saves every replicate's spectrum and evaluates the implemented
family: SB, scaled-shifted, mean-variance, scaled F, All, penalized All,
EBA and pEBA with 2, 4 and 6 blocks, and pOLS. The user chooses the global
reference from that table. This selection is post hoc.

The user chose **All** (2026-10-03): the exact weighted chi-square tail with
every positive sample eigenvalue of the robust-ordinal spectrum, as computed by
the exploration's `all` arm (truncate_negative = TRUE). Exploration rejection
was 2.9-7.0% (median 4.8%), with no null cell outside [3%, 7%] at N >= 500.

The chosen reference is then confirmed on fresh draws: the same 64 cells with
seed base 817160001 and the registered replicate counts (2,000 null, 1,000
power). Confirmation criterion, fixed now: policy rejection within [3%, 7%] in
every null cell with N >= 500, the registered flag rule. Cells outside it are
reported as flags; the reference is not tuned on the confirmation draws.

The nested estimated-weight profile law is not changed by this amendment.
Its production calibration failed in both directions (SB 0.1-2.2%, PEBA4
4-18% and worsening with N) with 80-370 spectrum terms for 10-15 restrictions;
it is under diagnosis (board task-17.3) and any replacement needs its own
registered confirmation.

## Amendment 2026-10-03 (after the nested diagnosis, before nested confirmation)

The production-seed diagnosis (`diagnostics/nested_profile_law.md`, task-17.3)
found the separate-point profile law's excess terms to be a plug-in
cancellation artifact: each model's profile is built at its own fitted point,
so residual-dependent blocks cancel only up to O_p(N^-1/2). The first-order law
of T = N (F_H0 - F_H1) under the nested null at the larger model's pseudo-true
point has exactly r = df_diff terms,
lambda = eig{(A H^-1 A')^-1 A H^-1 B H^-1 A'},
with H the observed DWLS Hessian at the larger fit, B the estimated-weight IJ
meat (H^-1 B H^-1 / N is the IJ policy covariance) and A the exact restriction
map. It is misspecification-robust under the scope requirement and replaces the
separate-point profile law as the policy's nested reference.

Confirmation on fresh draws, fixed now: nested cells 53-84 (null) and 139-146
(power), seed base 817160001, 2,000 null and 1,000 power replicates. The policy
reports SB and PEBA4 on this spectrum, as for ML and FIML. Comparators: the
separate-point profile law (SB, PEBA4), fixed-weight Satorra-2000 at H1 (SB,
PEBA4) and the implemented reference family on the new spectrum; spectra are
saved. Criterion: policy SB and PEBA4 rejection within [3%, 7%] in every null
cell with N >= 500 (the registered flag rule); N = 400 cells are reported
alongside. Cells outside are flags; nothing is tuned on the confirmation draws.
