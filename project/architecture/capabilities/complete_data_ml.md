### Complete-data ML and inference

- `policy_score_contracts_test.cpp` gates two-group complete-data covariance
  and nested observed-score reduction from FIML, equivalent label/`==`/
  `group.equal` restrictions at a shared evaluation point (relative 1e-8),
  and finite or typed policy inference on an actual Heywood PSD-boundary fit.
  These gates preserve raw score second moments and existing policy defaults.
- Normal-theory ML fitting defaults to NLopt L-BFGS. Heuristic start values
  sign each free loading by its indicator's covariance with the factor's
  marker, and terminal audit records projected-gradient stationarity at the
  returned iterate so soft optimizer failures can be classified by geometry.
- Frontier profile-LR scalar infrastructure for complete-data ML and
  moment-quadratic fits:
  `estimate::frontier::fit_ml_constrained()` appends caller-supplied nonlinear
  equality closures `h(θ)` / `J_h(θ)` to any partable nonlinear `==` rows and
  runs the existing SLSQP/IPOPT constrained scalar optimizer. The first helper
  layer, `profile_lrt_scalar_ml()` / `profile_lrt_parameter_ml()`, refits
  `g(θ)=g0` and reports the ordinary df-1 LR statistic
  `2N(fmin_constrained - fmin_unrestricted)`. The moment-quadratic sibling,
  `fit_gmm_constrained()` plus `profile_lrt_scalar_gmm()` /
  `profile_lrt_parameter_gmm()`, applies the same programmatic constraint path
  to a caller-fixed ULS/GLS/WLS/DWLS/DLS weight. The first CI-inversion seed,
  `profile_lrt_ci_parameter_{ml,gmm}()`, bisects the df-1
  profile statistic and returns root diagnostics. The complete-data ML and
  caller-fixed GMM parameter helpers also accept raw data for opt-in profile
  reference tiers: `RobustScaled` keeps ordinary `T`/`p_value` and adds the
  Satorra-style `scaling_factor`, `T_scaled`, and `p_value_scaled`, while
  `MisspecScaled` / `MisspecMixture` add the observed-bread misspecification
  fields (`misspec_scaling_factor`, `T_misspec_scaled`, mixture eigenvalues,
  mixture p-value, and endpoint cutoffs for CI inversion). ML routes the
  misspecification reference through observed Hessian bread plus empirical
  score meat; caller-fixed GMM routes it through the continuous-LS observed
  bread plus fixed-weight or IJ estimated-weight meat, including the supported
  sample-normal-theory, empirical-WLS, empirical-DWLS, and fixed-`a` DLS IJ
  weight families. R exposes the parameter special cases as
  `frontier_profile_lrt_parameter_*` and
  `frontier_profile_lrt_ci_parameter_*`, with `reference=` selecting the
  target. The all-ordinal frontier companion,
  `fit_ordinal_constrained()` plus `profile_lrt_scalar_ordinal()` /
  `profile_lrt_parameter_ordinal()`, applies the same programmatic equality
  pattern to the full ordinal ULS/DWLS/WLS threshold + polychoric moment stack;
  R exposes the parameter helper as
  `frontier_profile_lrt_parameter_ordinal()`, and
  `profile_lrt_ci_parameter_ordinal()` / the matching R helper invert that
  ordinary df-1 statistic by bisection. The first no-integration ordinal
  functional slice, `profile_lrt_ordinal_polychoric_omega()` /
  `profile_lrt_ci_ordinal_polychoric_omega()` and the matching R helpers,
  profiles model-implied latent-response/polychoric omega through the same
  fixed-weight path. All ordinal parameter and polychoric-omega profile helpers
  support the same ordinary, robust-scaled, misspec-scaled, and misspec-mixture
  references. The robust/misspec scaling is recomputed at each constrained
  profile point from the complete ordinal IJ sandwich used by robust SE/MI,
  including fitted DWLS/WLS weight influence; misspec references combine that
  meat matrix with the analytic observed ordinal bread. Bartlett /
  small-sample correction, observed-score / Green-Yang-like ordinal omega
  targets, and functional R callbacks remain research layers above this seed
  surface.
  The same parameter-profile reference family also spans the missing-data and
  mixed-categorical estimators: `estimate::fiml::frontier::fit_fiml_constrained`
  plus `profile_lrt_parameter_fiml()` / `profile_lrt_ci_parameter_fiml()` refit
  direct FIML against `theta_k = theta0` using the retained `FIMLPack`;
  `profile_lrt_parameter_ml2s()` / `profile_lrt_ci_parameter_ml2s()` cover
  ML2S Stage-2 `nt`, `uls`, `dwls`, `adf`/`wls`, and fixed-`a` `dls`
  weights (`*_ml2s_nt` remains the NT compatibility wrapper); and
  `profile_lrt_parameter_mixed_ordinal()` /
  `profile_lrt_ci_parameter_mixed_ordinal()` apply the same df-1 inversion to
  mixed continuous/ordinal ULS/DWLS/WLS fits. These helpers support ordinary,
  robust-scaled, misspec-scaled, and misspec-mixture references. Direct FIML
  builds the scalar profile scale from observed-pattern score meat and observed
  FIML bread; ML2S uses the Stage-1 saturated-moment sandwich and switches to
  analytic observed Stage-2 bread for misspec references, with estimated-weight
  IJ meat for data-dependent non-NT Stage-2 weights when raw/FIML internals are
  available; mixed ordinal uses the same
  IJ meat as robust mixed-ordinal SE/MI, including DWLS/WLS fitted-weight
  influence, and analytic observed mixed bread for misspec references. R exposes
  these as
  `frontier_profile_lrt_{parameter,ci}_fiml`,
  `frontier_profile_lrt_{parameter,ci}_ml2s` /
  `frontier_profile_lrt_{parameter,ci}_ml2s_nt`, and
  `frontier_profile_lrt_{parameter,ci}_mixed_ordinal`, with `reference=`
  selecting the endpoint statistic used by CI inversion.

- Frontier non-iterative CFA inference (2026-07) turns closed-form CFA
  estimators into delta-method-inferable ones. `estimate::frontier::
  noniterative_cfa_theta` maps `sigma -> theta` and returns a complete
  `(Lambda, Phi, psi)`. The legacy `guttman` selector preserves the existing
  lavaan-like Spearman/incidence Guttman map; `guttman_aligned` is the
  promoted research selector, using the blockwise triad-GMM H diagonal and the
  aligned score reconstruction from the communality experiment. The Guttman
  regression also has an explicit composite-weight axis:
  `auto` resolves to `unit` for `guttman_lavaan` and to `standardized` for
  `guttman_aligned`; `unit` uses incidence weights,
  `standardized` uses `diag(S)^-1/2 Z` and is rebuilt from the live covariance
  in every map evaluation. Aligned maps read loadings from the own-composite regression
  `K_if = (HB)_if / Q_ff` (the least-squares simple-structure fit; 2026-09-29),
  not the multiple regression `HB Q^-1` that `guttman_lavaan` keeps, and report
  residual variances as the communality split `diag(S) - diag(H)`.
  `estimator_map_jacobian` is its `J = dtheta/dvech(S)`: configural Guttman
  maps use an analytic regular-interior derivative (including the
  correlation-standardization, triad-GMM communality, fixed-rank
  Moore-Penrose-weight, composite-weight, inverse, and marker-scaling chains)
  with central differences retained as a boundary/rank-change fallback and as
  the generality seam for future FABIN2/Bentler/JS/MIIV maps. The promoted
  `guttman_aligned` configural path batches the block-GMM communality
  Jacobian and downstream score-regression derivative over all covariance
  coordinates; the GMM communality Jacobian now works on block-active
  derivative columns, applies the pseudo-inverse derivative through its
  required `dW e` action instead of materializing dense `dW` columns, and uses a
  guarded full-column-rank row-weight pseudo-inverse fast path. The aligned map
  has an explicit sample-size-resolved communality
  admissibility policy: `raw` is the exact historical no-op default, `hard`
  clips correlation-scale h2 to `[margin, 1-margin]`, and `soft` uses a smooth
  box with `beta = beta0 * n^rate`. Value, directional, batched, and
  constrained-KKT Jacobian paths compose the same clamp derivative. C++ and R
  fit/inference surfaces carry the policy and per-block activation counts;
  `guttman_lavaan` remains untouched. The aligned `unit` and `standardized`
  paths also have opt-in score-covariance conditioning for
  `Q = B' H B`. With `D = diag(Q)` and `R = D^-1/2 Q D^-1/2`, the hard map
  adds the exact diagonal intensity needed to attain
  `delta_n = floor0 n^-rate`, while the soft map uses a smooth spectral
  minimum and softplus activation; both return `(Q + tD)/(1+t)`, preserve the
  score variances, and use the repaired matrix consistently in the score
  inverse and factor covariance. The batched analytic Jacobian caches one
  eigensystem per block; hard activation/tie boundaries fall back to central
  differences and soft repeated eigenvalues use the invariant spectral
  projector. Fit objects retain the conditioning configuration and per-block
  raw/repaired eigenvalues, normalized eigenvalues, intensity, floor violation,
  score variance, and marker diagnostics, so post-fit inference reconstructs
  the identical map. Conditioning remains `raw` by default, is rejected for
  legacy `guttman_lavaan` . The retired engineering/10
  screen calibrated hard and smooth score repairs jointly with the
  communality-clamp finalists; its 24-cell/300-rep run (2026-07-10) produced no
  survivor (findings kept in the guttman-inference paper's notes since
  2026-09-29):
  raw-H arms frequently hit the existing improper-communality-split guard, and
  even clamped-H arms could have non-positive score variances. Fixed-diagonal
  Q repair intentionally cannot cure that latter failure because it preserves
  diag(Q). A separate, opt-in point-estimation feasibility branch now repairs
  the normalized H proxy itself toward the identity while preserving diag(H),
  before score construction. It is restricted to aligned unit/standardized
  maps, records raw/repaired H spectra and intensity, and has no post-fit
  inference claim until its derivative and calibration work are complete. Its
  18-cell/100-rep stress feasibility screen is also a no-go: it restored 100%
  point-fit availability and PD Phi, but every tested H repair worsened loading
  RMSE, often sharply, because the required normalized-H shrinkage was large
  (median intensity roughly 6--12 with substantial upper tails). The soft and
  hard variants saturated to the same repair in this failure region. Keep this
  as documented research evidence, not a production or inference candidate.
  Restricted
  estimator-side maps differentiate the regular constrained communality KKT
  system and the loading projection, with central differences retained for
  rank-changing pseudo-inverse or singular projection boundary cases. The
  restricted path batches the constrained h2 KKT right-hand sides for
  `triad_ls`, `extended_triad_ls`, and `triad_wls`; the GMM branch uses the same
  active-column `dW e` derivative as configural. The grouped restricted path
  embeds the active block RHS in the stacked communality system, reuses one
  joint KKT factorization across all covariance columns, feeds every block's
  constrained `dH` diagonal into the batched score-regression derivative, and
  applies the loading-projection derivative in action form, avoiding per-column
  inverse-derivative matrices. When no residual-communality rows are active,
  restricted maps bypass the stacked KKT system and evaluate the selected H
  diagonal directly; this keeps the `extended_triad_ls` +
  `standardized` configural proxy on the fast path. The correlation-standardizing
  Jacobian scatters each `dR_ab` row through its three nonzero covariance
  coordinates rather than adding dense `p*` rows. `triad_wls_joint` remains
  analytic but direction-wise.
  `robust::frontier::noniterative_se*` is the SE-only primitive:
  normal-theory paths contract `J Gamma J'/N`, while empirical paths stream
  casewise moment rows in parameter space rather than materializing dense
  moment-space Gamma. `robust::frontier::noniterative_inference*` is the full
  GOF bundle: it builds the residual projector `M = I - Delta J` and reuses the
  existing weighted-chi2 spectrum reducer for ULS or model-implied NTML tests,
  then includes the same delta-method SEs. `noniterative_wald` and
  `noniterative_difference_test` are the nested tests. R surface:
  `magmaan_core$noniterative_cfa_{fit,se,inference,wald,difference,
  pseudo_lrt}_impl`.
  The R fits are now first-class `magmaan_fit` objects (classed
  `magmaan_noniterative_fit`), so the ordinary post-fit measures apply
  unchanged: `vcov()` (regime `model` -> NT Gamma, `robust` -> empirical Gamma),
  `standardized()`, `residuals()` / `lav_residuals()` (SRMR), `factor_scores()`,
  `composite_weights()`, `compute_defined()`, the general `parameter_table()`
  (est/se/z/p/CI), and `fit_measures()`. The last routes to
  `fit_measures_noniterative()`: the residual GOF supplies the user statistic
  under the NT (`ntml`) or ULS discrepancy, referred to a same-discrepancy
  independence baseline, giving naive and robust/mixture-scaled CFI/TLI/RMSEA
  plus SRMR (the ULS baseline and the baseline scaling `c_0` are closed-form; NT
  `c_0 = 1` exactly). Multi-group non-iterative fits route through the grouped
  residual inference path, empirical baseline scaling is fixture-checked against
  raw cross-product variances, and likelihood information criteria
  (`logl`/AIC/BIC/BIC2) are reported as `NA` because the closed-form map is not
  an ML estimator. The ML score / LRT machinery
  (`modification_indices`, `score_tests`, their robust/LRT variants,
  `case_rerun`, `nestedTest`) is guarded off with a message pointing to the
  residual-based diagnostics and nested tests, because a closed-form map is not
  a gradient-zero minimizer. The explicit
  `noniterative_cfa_modification_indices()` surface reports three diagnostics
  for each fixed-zero or absent one-parameter candidate: a raw residual score,
  a `V`-tangent-residualized residual score, and local one-step
  discrepancy-drop/EPC values. It is single-group covariance-only in v1 and
  intentionally does not reuse the ML score-test name. Glue in
  `r-package/R/noniterative_postfit.R` and `r-package/R/noniterative.R`;
  derivations in the
  paper notes `noniterative_fit_indices` and
  `noniterative_modification_indices`; validated by
  `r-package/examples/noniterative_postfit.R`.
  Theory in
  the guttman-inference paper's derivation notes
  (`papers/guttman-inference/dev/notes/noniterative_cfa_tests`); validated on the
  legacy map by the retired research/24 study, whose runner and findings moved
  to the guttman-inference paper on 2026-09-29 (empirical Gamma calibrated across
  normal / independent-component / ordinal-as-continuous generators and across a
  0.3-0.7 reliability sweep, NT Gamma asymptotically miscalibrated on non-normal
  data, GOF power near 1; Guttman's efficiency gap vs ML is small at high
  reliability and grows on the structural parameters as reliability drops and,
  under a heterogeneous loading pattern, lands on the weak indicator's
  factor-mates via the triad-based communality step; the closed form is more
  robust than ML to uniformly weak signal but less robust to a single weak
  indicator).
  The H-diagonal communality rules from the follow-up Guttman work are now a
  separate frontier primitive:
  `estimate::frontier::estimate_h_communalities()` / R `guttman_h()` compute
  AR, RS, triad least squares, anchor triad least squares
  (identity-weighted one-same-block anchor triads), blockwise triad-GMM, and
  full selected-triad-GMM diagonals for a fixed simple-structure indicator
  block vector. They return `h2`, `diag(H)`, and
  `H = S` with only the diagonal replaced. The ordinary `guttman_aligned`
  point-estimator lane consumes the blockwise triad-GMM rule. The
  residual-restricted Guttman map can also select any least-squares-form
  H-diagonal rule (`triad_ls`, `extended_triad_ls`, `triad_wls`, `triad_wls_joint`) and
  any composite weight (`unit`, `standardized`), reusing both
  choices in its restricted analytic-first Jacobian and grouped inference;
  AR/RS remain
  low-level H-estimation diagnostics because they are not constraint-compatible
  LS systems. The retired research/27 smoke study (findings in the
  guttman-inference paper's notes) ranked extended triad LS first.
- Frontier multi-group / constrained / mean-structure non-iterative CFA
  (2026-07) extends the closed-form estimator to measurement invariance. The map
  fits each group's Guttman block independently and stacks them, so
  `robust::frontier::noniterative_inference_grouped*` carry a block-diagonal
  delta-method `Omega`, a joint residual GOF, and `block_of_param`. Mean
  structure is supported for free intercepts with latent means fixed at 0
  (`nu_g = m_g` saturated); by Proposition 2 of the note the mean part is inert
  for fit, so only `Omega` gains the intercept block (via `gamma_nt_with_means`).
  The estimator-side metric map `fit_noniterative_cfa_metric` estimates a common
  standardized loading shape across groups by a Sigma/H-level rank-one
  reconstruction and then converts to the partable's marker chart. The
  estimator-side restricted map `fit_noniterative_cfa_restricted` imposes
  separable loading/residual linear constraints inside the Guttman
  reconstruction: residual rows enter the selected LS-form communality/H step
  (default `triad_wls`). Loading rows confined to one factor, and fixed
  non-marker loadings, are homogeneous linear restrictions on that factor's
  own-composite coefficients and are imposed exactly by a partable-only
  Euclidean projector before the marker rescaling (tau-equivalence averages,
  independent of the marker); rows spanning factors or groups fall back to a
  second-stage marker-chart projection over all loading rows. Unsupported
  mixed/factor/mean rows error. Grouped restricted
  inference uses the restricted map's full stacked analytic-first Jacobian
  with the same communality and composite choices, so cross-block constraints
  propagate into `Omega`, GOF, and pseudo-LRTs. General linear equality testing
  still has the
  `Omega`-metric minimum-distance projection `noniterative_constrained_fit`,
  whose statistic is an exact chi2_k (Wald = min-distance duality). True
  (free-latent-mean) scalar
  invariance is `noniterative_scalar_invariance`, the reference-group mean map
  `alpha_g = (Lr'Lr)^-1 Lr'(m_g - m_r)` with a linearized pseudo-inverse Wald on
  the mean residual orthogonal to the loadings, df `(G-1)(p-#factors)`. R surface:
  `fit_noniterative_cfa_{metric,restricted}` plus
  `magmaan_core$noniterative_cfa_{se,grouped_inference,pseudo_lrt,constrained,
  scalar}_impl`.
  Theory in the guttman-inference paper's constrained-CFA note
  (`constrained_noniterative_cfa`, removed from the paper's notes on 2026-09-29
  when invariance left its scope; recoverable from that repo's history); validated on the
  legacy map by the retired research/26 study, 300 reps, findings in the
  guttman-inference paper's notes (metric Wald tracks the ML LRT on
  normal data with matched power; on non-normal data the NT-Gamma metric Wald
  over-rejects and the empirical Gamma restores the level, mirroring the ML
  NT-vs-robust split; the scalar Wald is exactly nominal on normal data, far more
  robust to non-normality, and delivers true scalar in one closed-form step where
  the ML nested test cannot). The retired constraint-charts check (archived
  study, deleted 2026-09-29) showed for the estimator-side metric map that both the configural and
  metric-constrained implied covariances are marker-chart invariant at roundoff,
  including deliberately off-surface metric-violation cells; the raw theta
  coordinates differ, as they should.
- Frontier empirical reduced-bias estimation (2026-06) implements the
  Kosmidis-Lunardon trace adjustment for raw-data normal-theory SEM and the
  moment-quadratic family. The C++ surface covers
  `rbm_{explicit,implicit}_{ml,fiml,continuous_ls,ordinal,mixed_ordinal,two_stage}`;
  the R research surface is `magmaan_core$frontier_rbm(fit, raw_data, weight,
  stage2_weight, dls_a, method = "explicit" | "implicit")`. Complete-data ML
  uses literal per-row normal score rows, FIML uses observed-pattern score rows,
  continuous ULS/GLS/WLS uses the complete-data moment IJ rows, ordinal and
  mixed ordinal use the estimated polychoric/polyserial IJ rows, and ML2S uses
  the same saturated-moment IJ stack as weighted inference. The trace,
  correction, and implicit penalty are computed in the linear-constraint-reduced
  alpha space (`K' J K`, `K' E K`); full and reduced bread/meat matrices are
  reported for diagnostics. Nonlinear equality constraints are still rejected.
- Frontier structural-after-measurement (SAM / LSAM) estimation is available as
  `estimate::frontier::fit_sam()` and from R as
  `magmaan_core$frontier_sam()` / `sam()`. The C++ core fits local or global
  measurement blocks, builds the Croon/Wall-Amemiya latent covariance with
  ML/GLS/ULS mapping and Fuller lambda correction, then fits the promoted
  structural submodel. Lavaan `sam()` parity is covered for point estimates and
  classic `se = "standard"` / `se = "twostep"` standard errors from
  `SampleStats`. The raw-data overload adds lavaan-parity
  `se = "twostep.robust"` for the landed scope: single-group, local,
  covariance-only, complete continuous data with `alpha_correction = 0`; it is
  lavaan's nonnormality-robust Yuan-Chan correction, not the frontier
  misspecification / estimated-weight sandwich still tracked in the backlog.
- Experimental complete-data ML IRLS paths fit the same ML objective through
  outer Fisher reweighting and inner GLS solves:
  `estimate::fit_ml_irls()` uses the full parameter block, while
  `estimate::fit_ml_irls_snlls()` uses Golub-Pereyra profiling for separable
  models. Mean-structure IRLS adjusts each frozen inner covariance target by
  the current mean residual outer product so the inner score matches the ML
  score up to scale. Linear equality constraints are handled in reduced
  coordinates; nonlinear constraints are rejected, and the SNLLS variant also
  rejects box bounds.
- Experimental local Fisher scoring for complete-data ML is exposed as
  `estimate::fit_ml_fisher()` and `magmaan_core$fit_ml_fisher()`. It computes
  the analytic ML gradient and expected-information Hessian approximation on
  the true `F_ML` scale, solves a damped local Fisher equation, and accepts
  steps by Armijo backtracking on the ML objective. This is separate from the
  IRLS paths above, which reoptimize a frozen GLS subproblem at each outer
  iterate. A companion `estimate::fit_ml_fisher_snlls()` /
  `magmaan_core$fit_ml_fisher_snlls()` path solves the same local Fisher
  equation through a Schur complement over the SNLLS β/α split; this is local
  block elimination, not Golub-Pereyra objective profiling.
- Objective-scale convention (unified 2026-06-09): `est.fmin = ½·F` for EVERY
  estimator (ML/FIML, ULS/GLS/WLS, ordinal, mixed) — the optimiser's minimum,
  half the discrepancy, and the quantity whose Hessian is the Fisher
  information. The goodness-of-fit statistic is `T = 2N·fmin = N·F` uniformly
  (`inference::chi2_stat`), matching lavaan's stored `fmin` element-for-element.
  The `½` lives only in the optimiser adapters; the math kernels stay full-`F`
  so the information/SE and score paths are untouched. Deliberate exceptions
  (ULS-standard Browne, FIML-standard LRT, the test-side `(N−G)/N` lavaan
  offset) are documented at `chi2_stat` and in
  [project/design/numerical-conventions.md](../../design/numerical-conventions.md).
- Expected information, finite-difference observed information, and analytic
  observed information for covariance and mean-structure models.
- Vcov/SE, Wald/z tests, chi-square/df helpers, LR/Satorra-2000 and
  Satorra-Bentler 2001/2010 compatibility nested tests, robust U-Gamma
  machinery, Satorra-Bentler-family statistics, robust SEs,
  FMG eigenvalue p-value tests (explicit method/options API, no parser),
  Browne residual NT/ADF, fixed-parameter modification indices,
  equality-release score tests, fit measures including RMSEA close-fit
  p-values and lavaan's saturated-user-model `TLI = 1` convention,
  lavaan-style robust/scaled fit-measure formula helpers for the core
  `chisq.scaled`/baseline/CFI/TLI/RMSEA family, the FIML corrected robust
  fit-measure reduction (`estimate::fiml::fiml_corrected_fit_measures`) that
  builds missing-data `XX3`/`df3`/`c.hat3` and baseline counterparts with
  lavaan's FIML-C(V3) trace correction, using the missing-data saturated H1
  information plus the complete-data H1 information at the EM moments, and ML2S
  `robust.two.stage` robust/scaled global indices
  (`estimate::fiml::two_stage_fit_measures`),
  structural-aware standardization, C++ defined-parameter evaluation, and the
  first frontier reliability covariance functionals
  (`measures::frontier::reliability`: alpha, Guttman's lambda6, and
  Spearman-Guttman covariance omega with delta-method covariance-scale SEs;
  exposed in R through `magmaan_core$measures_reliability_cov`). Extended with
  closed-form **multidimensional** omega: `omega_multidim` for
  `OmegaTarget::Total` (the weighted composite `w'CX(X'CX)^-1 X'C w / w'Sw`) and
  `OmegaTarget::Hierarchical` (two-stage centroid Schmid-Leiman general factor,
  k>=3), with Spearman ratio-of-sums communalities, a finite-difference gradient,
  and the full-Gamma `omega_multidim_delta` SE; exposed through
  `magmaan_core$measures_reliability_omega_multidim`. Consumed by
  `papers/closed-form-omega`. Model-based CFA omega is also present:
  `omega_from_fit` computes continuous single-group `omega_total` /
  `omega_hierarchical` from fitted LISREL matrices and reuses ML/GLS/ULS robust
  parameter vcovs for delta SEs; the all-ordinal frontier slice adds
  `ordinal_observed_score_covariance` / `omega_ordinal_observed` plus
  `estimate::frontier::ordinal_observed_omega` for a fitted single-group
  DWLS/WLS/ULS model. That ordinal path maps fitted thresholds and
  latent-response correlations to the observed integer category-score covariance
  (0,1,...,K-1), then contracts the finite-difference scalar gradient with the
  complete IJ parameter sandwich from `robust_ordinal_ij`. It intentionally has
  no Bartlett/profile-LR/small-sample scaling policy; those remain research
  layers above the delta surface. The R surface is
  `magmaan_core$measures_reliability_ordinal_observed_omega`. The simpler
  no-integration ordinal reliability target,
  `magmaan_core$measures_reliability_ordinal_polychoric_omega`, computes
  closed-form omega directly on the all-ordinal polychoric correlation matrix:
  it extracts the correlation block of the ordinal `NACOV`, pads it into the
  off-diagonal slots of a full vech-correlation Gamma with fixed unit diagonals,
  and reuses `omega_multidim_delta` for the robust delta SE.
- `inference::frontier` robust (generalized / Satorra-Bentler-scaled)
  modification indices and equality-release score tests: each candidate carries
  the ordinary `mi` and a `mi_scaled = mi / c` with the per-direction scaling
  `c = gᵀB1g / gᵀA1g`, where A1/B1 are the parameter-space sandwich bread/meat
  surfaced by `robust::param_space_sandwich` (the same Δ'WΔ / Δ'WΓ̂WΔ that
  `robust_se` uses) and g is the efficient-score direction. Goes beyond lavaan,
  which falls back to the ordinary statistic when `se != "standard"`. Covers
  complete-data ML, both breads (`Information::Expected` ≈ robust.sem/MLM;
  `Information::Observed` ≈ robust.huber.white/MLR), single or multi-group (the
  per-block `n_b/N`-weighted sandwich pools across groups); reduces to the
  ordinary statistic exactly under the model-implied Γ_NT meat (Expected bread).
  Friendly entries under `api::frontier::{modification_indices,score_tests}_robust`.
  The fixed-parameter robust MI sweep batches eligible candidate rows into one
  augmented null-point evaluation, so score/info and the parameter-space sandwich
  are built once per sweep rather than once per candidate; duplicate matrix-cell
  edge cases fall back to the conservative one-candidate path. The sandwich
  helper also accepts precomputed casewise contributions (`Zc`) for callers that
  reuse the same raw-data meat.
  One-dimensional NT and robust candidate tests judge efficient-information
  rank relative to the marginal and nuisance-removed information terms
  (`I_eff > 1e-10 max(|I_dd|, |I_da I_aa^-1 I_ad|)`), rather than an absolute
  information floor. The robust bread uses the cancellation bound
  `g' A1 g > 1e-12 |g|' |A1| |g|`. These homogeneous checks exclude
  identification-only releases without rejecting identified directions merely
  because their parameter units make information small. Observed-information
  workers first check candidate rank in expected-information tangent geometry:
  residual-gradient curvature on a nonlinear identification orbit at nearby
  parameter values must not create a hypothesis. This gate covers fixed and
  equality releases without changing the information used for their statistics.
  Gates cover the batched ML sweep across nearby fits, observation counts and
  units; the original FIML
  robust-MI normal-data witness now requires marker exclusion, and R MI tables
  are compared with live lavaan for complete/missing data across indicator units.
  Ordinary FIML MI/equality releases also use the existing analytic observed
  information, matching the robust path: a fixed finite-difference step could
  otherwise manufacture curvature in redundant directions. Their `h_step`
  argument remains validated for compatibility and no longer tunes information.
  Validated four ways (lavaan implements no robust score test to diff against):
  exact reduction-to-NT, independent A1/B1 re-assembly, an R-internals oracle
  built from lavaan's delta/wls.v/gamma/ceq.JAC (`regen_robust_score.R`,
  convention-free θ-space scaling), and an advisory calibration + Wald/LRT-trinity
  simulation (`cpp/tests/checks/robust_score/`).
- **RLS is Browne's statistic with a model-based Γ (2026-09-19):**
  `inference::rls_chi2()` claimed lavaan `browne.residual.nt.model` parity but
  computed `Σ_b n_b·½·tr((Σ̂_b⁻¹(S_b−Σ̂_b))²)` from moments alone. Reading
  `lav_test_browne.R`, lavaan's statistic is
  `N·(r'Γ⁻¹r − b'A⁻¹b)` with `b = Δ'Γ⁻¹r`, `A = Δ'Γ⁻¹Δ`, over a residual `r`
  that **includes the mean block** whenever the model has a mean structure, and
  `browne.residual.nt` vs `.model` differ *only* in whether Γ is built at `S` or
  at `Σ̂`. The trace form equals that projected quadratic only when `r` is
  already Γ-orthogonal to the model tangent space — true at the ML optimum for
  covariance-only or **saturated-mean** models, false once means are genuinely
  restricted, where it was wrong by 24-50%.
  - Fix: `browne_residual_nt` gained an `inference::GammaAt {Sample, Model}`
    parameter (the only behavioural difference), and `rls_chi2` is now
    `browne_residual_nt(…, GammaAt::Model)`. Mean structures are handled by the
    shared residual vector, so no separate mean-aware entry point exists:
    `frontier::rls_mean_cov_chi2` is retired.
  - The moments-only overload is **removed**, not fixed: the projection cannot
    be recovered without the Jacobian. What it actually computed — the
    unprojected `N·r'Γ(Σ̂)⁻¹r`, mean block included — survives under an honest
    name as `frontier::nt_moment_quadratic`, which is a legitimate primitive
    (the closed-form CFA `rls_check` and `noniterative_cfa_test` want exactly
    the trace form) but is **not** a lavaan test statistic and not χ²(df).
  - Why it went unnoticed: every fixture carrying an `rls_chi2` oracle value had
    saturated or absent means, where the two formulas agree exactly. The only
    restricted-mean fixture, `0023_scalar_invariance_3f_hs`, had *null* oracle
    values because `regen_oracle.R` used `tryCatch(…, warning = function(w)
    NULL)`, and that fixture emits the benign "a single label per parameter in a
    multiple group setting implies imposing equality constraints" warning — which
    is precisely what the fixture intends. One informational warning silently
    nulled 7 statistics for 4 of 19 fixtures (0013, 0015, 0018, 0023). Replaced
    by a shared `fit_or_error()` helper that retries under `suppressWarnings`
    and treats only errors as errors; the restored 0023 now gates the
    restricted-mean case, and RLS there is 187.339 vs the old 115.586.
- **Matched ML speed pilot (2026-09-20):** `benchmarks/r/timing.R` now supplies
  shared adaptive batching, balanced arm order, and raw batch records for R
  workloads. The experiment collection contains a two-case continuous-ML
  pilot with directly timed raw/prepared/post-fit boundaries and separate
  covariance, SB, pEBA-4, and adapter checks. The HS native pEBA-4 comparison
  remains rejected on relative tail agreement even at identical statistic and
  spectrum. Follow-up isolates lavaan's default absolute integration tolerance:
  a tighter tail integral and an independent high-precision Erlang-chain
  reference agree with magmaan. The timing rejection remains until an
  accuracy-matched reference route is timed; there is no statistical-convention
  discrepancy. Controlled cross-engine evaluator/backend attribution and public
  promotion remain open under `project/validation/benchmark_plan.md`.
- **Expected information via the whitened Jacobian (2026-09-18):**
  `inference::information_expected_per_case_blocks` and
  `expected_info_covariance_only` no longer materialize
  `T[k][b] = Σ_b⁻¹·unvech(J[:,k])` for every free parameter × block. Both terms
  of the expected information are the same bilinear form — the normal-theory
  inner product `⟨X,Y⟩_A = ½·tr(A⁻¹XA⁻¹Y)` — which the NormalTheory
  `estimate::gmm::BlockWeight` already carries in factored form, so with
  `Y_b = Fᵀ[dμ/dθ ; dvech(Σ)/dθ]_b` the per-case block is exactly `Y_bᵀ Y_b`
  with no residual scaling. One whitened Jacobian per block, built and
  discarded in turn, plus one BLAS-3 syrk, replacing an
  `n_free × n_blocks` array of p×p matrices and an
  `n_free²·n_blocks·p²` elementwise reduction. Multi-group models benefit most,
  since a parameter usually touches one group and the rest of that array was
  explicitly-stored zeros: at p=48 with 4 groups, 14.4x faster and 27.0 MB →
  3.45 MB (`benchmarks/expected_info_bench.cpp`, paired/rotated via `benchmarks/timing/timing.hpp`). Pinned against an independent
  explicit-trace reference in `cpp/tests/unit/expected_info_whitened_test.cpp`.
  This is the surviving half of the retired "share the p×p factor" backlog item;
  the CPU-sharing half was retired (ceiling 0.006% at p=48; see
  the continuous-weight capability below).
- **Persistent continuous NTML inference (2026-09-16):** the existing
  U-factor shared phase is now the owning `robust::NTMLGeometry`, exposed by
  `prepare_ntml_geometry` and consumed by expected/observed U-factor tails.
  `robust::frontier::{NTMLData,NTMLFit,NTMLHypothesis,NTMLQuadratic}` retain
  complete-data sample/pattern setup, casewise contributions, fitted moments,
  derivatives, weight factorizations, expected information/covariance and
  test-specific reductions. These mutable native caches are session-local and
  not thread-safe; prepared inputs must remain immutable.

  R `prepare_inference_data` provides a shared dataset to `prepare_inference`.
  `prepare_hypothesis` owns an exact nested pair. Parameter-key embedding
  expresses dropped or fixed H0 paths as affine restrictions in H1's slots;
  existing same-slot affine pairs retain their calculation path.
  `inference_quadratic` produces global score/ML GOF or nested
  score/exact H1-anchored Satorra–2000 LR without invoking a test wrapper;
  `inference_rows` returns its casewise rows (a score statistic is the squared
  norm of their column sums, and their crossproduct is the spectrum's reduced
  matrix), so moment-based calibrations can be composed outside the core.
  `inference_covariance` shares expected information and empirical contributions
  with Wald consumers. Existing compatible GOF (including unbiased spectra),
  expected-information, expected sandwich-SE and exact empirical streaming LR
  wrappers reuse these native snapshots; edited extracted fit lists invalidate
  their handles. Other inference conventions retain the existing paths.

  Casewise storage expands centered moment contributions once across models;
  automatic storage switches to tiled projection above a 64-MiB contribution
  budget. Global score and GOF share a projection. Score and nested LR apply the fitted-mean
  linear/constant likelihood correction without reconstructing fourth-moment
  contributions. Nested LR evaluates these rows at the alternative fit for
  both observed and expected geometry, including cached expected reductions.
  A two-group restricted, misspecified-mean gate matches direct complete-data
  FIML and finite-difference likelihood scores in casewise and tiled storage.
  Global LR remains a centered sample-moment GOF reduction: its null asserts
  correct specification, under which the mean shift vanishes. Global score and
  expected/observed score sandwiches already use exact likelihood rows.
  The default lab `robust_nested_lrt()` delegates to policy and inherits this
  fix; explicit Satorra-2000 streaming/materialized/dense drivers retain
  their centered empirical-Gamma convention pending the lab-centering study.
  Decisions/04 frees group intercepts and restricts only loadings, so its
  saturated-mean design is unaffected.
  The large-N tiled path accumulates both reduced matrices in one pass instead
  of retaining N-by-df rows. The spectrum uses row space when N < df; SB-only
  calibration uses a trace and spectra are cached on demand. Distinct tiled
  hypotheses may require distinct raw-data passes. Explicit unbiased GOF
  retains the existing casewise correction, which may materialize contributions.
  Neither full empirical Gamma nor full U is a prerequisite of sharing.

  Validation: the optimized inference suite passes 325 cases / 39,332 assertions;
  focused reuse tests cover unequal group sizes, mean/covariance layouts,
  tiled/casewise parity, smaller row space and construction counts. R checks
  additionally cover global and nested score/LR parity, unbiased GOF, sandwich
  covariance/Wald, restricted means, repeated calls and ownership/invalidation.
  `inference_reuse` exposes construction counters and
  `benchmarks/inference_reuse.R` provides a bounded timing comparison.
  FIML/DWLS ordinary-policy evaluation-point ingredients now have owning
  snapshots (TASK-23; see [FIML](fiml.md#ordinary-policy) and
  [ordinal](ordinal_and_mixed.md)). Generic FIML/ML2S score geometry and
  additional bread/nesting conventions remain separate; the first-class score
  interfaces below remain available.

- **Reusable score primitives (2026-09-16):**
  `inference::frontier::{global_score_components,global_score_components_ml2s,
  nested_score_components}` now own construction independently of tests.
  `ScoreGeometryOptions` selects sensitivity and metric without any resampling
  option. `ScoreComponents` retains the observed total score, casewise rows,
  nuisance/test directions and geometry. NT-ML2S marks its rows as influence
  contributions and preserves the separately computed observed Stage-2 score.
  `project_scores` constructs an owning `ProjectedScore` with a retained metric
  Cholesky factor, quadratic and meat; rows are optionally retained for explicit
  multiplier resampling. `score_spectrum`, trace-only `score_mean_scale`,
  `score_sandwich` and `resample_scores` consume that object separately. PSD
  meat remains valid for mixture inference when sandwich inversion is unavailable.
  Legacy nested/global score-flip wrappers now consume the same construction
  and projection; their historical exact-mixture/sandwich diagnostics remain
  compatibility behavior, not prerequisites of the primitive path.
  The friendly `api::frontier::score_components` adapters accept global ML/FIML
  fits or an H1 model plus H0 fit for nested hypotheses.

  R exposes `prepare_inference`, `scores`, `score_components`,
  `score_components_from_matrices`, `project_scores`, `score_quadratic`,
  `score_spectrum`, `calibrate_quadratic`, `score_sandwich` and
  `resample_scores` as thin wrappers over native construction/calculation.
  Preparation snapshots the fitted model, parameters, raw data and FIML pack
  once. Locked environments own native objects; restored process-local handles
  require re-preparation. Bare likelihood scores use summed log-likelihood
  units; globally centered covariance rows are an explicit projection option,
  distinct from legacy within-pattern centering. `inference_information` and
  `parameter_covariance` expose reusable ML/FIML matrices; `wald_test` accepts
  an existing covariance. Tagged matrices reject a different snapshot.
  Existing `fmg_tests`, `fmg_nested` and `robust_nested_lrt` also accept these
  snapshots and reuse their native fit context. Extracted fit lists invalidate
  that cache when their structural/sample objects change. LR/GOF-specific
  geometry still follows its own method contracts; `quadratic_reference`
  retains its computed statistic/spectrum for repeated downstream calibration.

  Validation: the optimized inference suite passes 323 cases / 39,248
  assertions; score-focused API checks pass 4 cases / 90 assertions. The R
  `examples/scores.R` checks ML/FIML/ML2S reductions, nested scores without an
  H1 fit, reference reuse, PSD meat, supplied matrices, Wald composition and
  snapshot ownership/invalidation. A bounded 15-call benchmark at N=400 and
  p=20 measures the new component/projection/spectrum/SB+pEBA pipeline at
  about 8 ms (ML), 17 ms (FIML), and 21 ms (ML2S), compared with 15/23/29 ms
  through the legacy score-flip wrapper; preparation is separately timed.
  Recalibration from an existing spectrum takes about 0.5 ms. These laptop
  timings are advisory; see `benchmarks/score_primitives.R`. They do not
  replace the historical talk simulation runtime or imply identical
  finite-sample score/LR spectra.

- `inference::frontier::score_flip_test` adds Monte Carlo Rademacher calibration
  for affine nested complete-data ML and direct-FIML models. It derives the
  tested directions from the exact H1/H0 restriction map, evaluates individual
  Gaussian observed-data likelihood-score contributions at the H0 fit, and
  reports three references:
  basic score flips, Hemerik-Goeman-Finos nuisance-effective flips, and the
  De Santis-Goeman-Hemerik-Davenport-Finos flip-specifically standardized
  quadratic statistic. The effective direction
  `G = D - K(K'IK)^-1 K'ID` is orthogonal to H0's nuisance tangent `K`; each
  transformed statistic additionally recomputes the conditional variance
  induced by estimating that nuisance vector. Complete data use per-group
  expected-information blocks; FIML uses conditional Fisher-information blocks
  per `(group, observed-pattern)` stratum. Both compress the correction without
  storing an `n x n` hat matrix, and the all-observed FIML path is unit-gated to
  the complete-data statistics, p-values, and mixture spectrum. The observed
  identity is included in the Monte Carlo rank,
  `p_value` aliases the standardized p-value, and deterministic seeds, Monte
  Carlo SEs, asymptotic score comparators, nuisance-stationarity, and variance-
  conditioning diagnostics are returned. The result also measures mean/max
  flip-specific covariance displacement and high-resolution setup,
  basic/effective resampling, standardization, asymptotic-comparator, and total
  timings. `ScoreFlipOptions::calibration` can now request asymptotic-only,
  effective-only, effective-plus-standardized, or the full historical battery;
  skipped references do not draw signs or build their covariance geometry and
  are returned as unavailable. `api::frontier::score_flip_test` and the R
  `score_flip_test()` wrapper expose both the original fit-pair route and a
  model-plus-H0 route: the latter uses H1's tangent without fitting H1. The R
  `nested_score_test()` convenience selects the zero-resampling route and
  reports score pEBA4 as its primary p-value alongside SB, exact-mixture, and
  direct-sandwich diagnostics. `ScoreFlipOptions::sensitivity` additionally
  exposes an opt-in observed-information correction. It replaces the expected
  metric in the nuisance projection
  `G_A = D - K(K'AK)^-1K'AD` with the realized likelihood Hessian while
  retaining expected information as a stable common quadratic metric. This is
  the pseudo-true/MAR path when
  information equality fails; the historical expected-information
  construction remains the default comparator. Observed sensitivity is
  effective/asymptotic-only and deliberately forbids within-pattern centering,
  whose conditional means need not vanish under MAR. Direct FIML reuses the H0
  fit's raw data and cached missingness pack. The current contract rejects
  fixed-X rows, nonlinear/inequality constraints, boundary null fits, and
  estimators other than complete ML or direct FIML.
  The decisions/04 task-73 production-seed mechanism replay finds sample-
  dependent projection/metric variance compression at N=100 per group:
  200 draws in each correct/mild normal/skewed null cell, no default change.
  Uniform meat inflation is not supported; registered finite-sample correction
  and larger-df/strong-misspecification confirmation remain open.
  The C++ frontier option additionally has a deterministic verification-only
  exact-enumeration path capped at n=20. Its unit gate enumerates all 4,096
  sign vectors at n=12, is seed-invariant, and returns zero Monte Carlo error;
  a separate independent dense per-case nuisance-adjustment oracle matches the
  production group-sufficient covariance formula over every sign vector in a
  two-group n=5 construction. The R surface remains Monte Carlo-only.
- `inference::frontier::global_score_flip_test` is the curved-model global-GOF
  extension for continuous ML/FIML, with
  `inference::frontier::global_score_flip_test_ml2s` providing the
  normal-theory two-stage counterpart. The ML/FIML path evaluates direct
  casewise saturated likelihood scores at the fitted model moments, builds the
  conditional Fisher metric for each observed-data pattern, and projects the
  scores off the local SEM moment tangent. Its opt-in observed-sensitivity
  variant instead uses the analytic realized saturated-moment Hessian at the
  restricted fitted moments for the tangent projection; a projection identity is unit-gated against the
  existing structural observed-H1 information. A separate opt-in global-score
  metric can now use that same observed H0 information for the score quadratic
  and generalized-eigenvalue bread, while the established default retains the
  conditional Fisher metric. A second observed-information diagnostic uses
  the realized saturated H1 Hessian for either the nuisance projection or the
  quadratic/spectrum metric. At a pseudo-null it has the same population target
  as observed H0 information, while evaluation at the saturated optimum keeps
  its finite-sample curvature positive definite. The R surface names this
  choice `observed-h1`; its score rows and observed statistic remain evaluated
  at H0, so it is an H1-geometry score test rather than an LR construction.
  The global result also exposes the summed df-dimensional projected score,
  quadratic metric, and raw OPG meat, allowing experiment-side checks of the
  direct sandwich without reconstructing the saturated tangent. Two explicitly
  diagnostic observed-H0 sensitivity variants shrink the realized Hessian
  toward expected Fisher information with weights based only on
  `tangent_rank / n` (light and square-root rules); both weights vanish under
  fixed-dimensional asymptotics and the realized weight is returned.
  The full-observed H0 variant fails closed when its
  tested-complement information is not positive definite and is unavailable
  for ML2S. The tested dimension is therefore
  the saturated mean/covariance dimension minus the numerical tangent rank; no saturated H1
  fit and no refit per multiplier draw is required. Complete data and an
  all-observed FIML mask are unit-gated to the same statistic and multiplier
  ranks, while a separate missing-pattern gate exercises stratum centering.
  The first production contract is effective/asymptotic calibration only,
  random-X, affine equality constraints, no active bounds, and a required mean
  structure when observations are missing. `api::frontier` and the R
  `global_score_flip_test()` wrapper expose the result together with saturated
  dimension, tangent-rank, singular-value, conditioning, stationarity, and
  runtime diagnostics. ML2S instead evaluates the observed Stage-2 saturated
  score at the fitted model and propagates the Stage-1 saturated-EM casewise
  moment influence through the same fixed fitted-model NT metric and tangent
  projection. It deliberately supports only the unregularized, fixed-NT
  Stage-2 estimator: estimated/non-NT weights would require their additional
  weight-influence channel. Complete-data ML2S is unit-gated to the ordinary ML
  observed statistic and multiplier rank/p-value, while its asymptotic
  spectrum is only first-order equivalent because it uses centered moment
  influence rather than raw likelihood-score rows. The R
  `global_score_flip_test()` wrapper dispatches `estimator = "ML2S"` to this
  route and rejects regularized or non-NT two-stage fits.
  Experiment research/44 (now `fiml-global-tests`, global lane)'s 20,000-fit representative-SEM null gate found mean cell
  rejection .058 across 40 normal/VM/IG complete/MCAR cells (range
  .018--.122; 33/40 in [.025,.075]). Twenty-five finite calls lost numerical
  tangent rank and were strongly rejection-prone; downstream experiments must
  therefore report the prespecified-rank denominator separately and fail closed
  on non-nominal geometry rather than treating a finite p-value alone as full
  test success.
  A 300-fit, 30-cell `n=120` triage of the full-observed metric across all five
  pilot models, normal/VM/IG generators, and complete/MAR data found only
  226/300 usable calls; all 74 failures were non-positive-definite projected
  observed information. Among usable calls, observed-metric score pEBA4
  rejected 58%, versus 2.7% for the established observed-sensitivity/Fisher-
  metric score pEBA4 on all 300 fits. Ten replications per cell are not a size
  study, but the magnitude and conditioning failures rule this construction
  out as a default; retain it only as a diagnostic comparator.
  Experiment research/32's 300-replication
  probe found the basic test extremely conservative, effective flips close to
  nominal, and standardization a small improvement concentrated at n=30; the
  hard t5 / threefold factor-variance cell rejected at 0.054, 0.077, and 0.050
  for n=30/50/100, versus 0.190, 0.163, and 0.107 for the mean-scaled
  Satorra-2000 difference test. The 2026-07-13 expansion held a 62-slot
  two-factor nuisance model fixed while crossing 1/4/8 restrictions, total
  N=60/100/200, 1:1/1:3 allocation, three group-information geometries, and
  normal/t5/skew distributions (162 cells, 200 attempts each). Across 32,243
  successful fits, standardization changed 70 decisions, all from effective
  rejection to acceptance. At N=60, effective/standardized rejection was
  0.060/0.059 (df=1), 0.069/0.064 (df=4), and 0.083/0.077 (df=8); by N=200 the
  differences were 0.0006--0.0017. A warm serial 499-flip call cost 2.3/5.4/8.7
  ms total at df=1/4/8, of which standardization cost 0.7/2.7/4.2 ms. This is
  frontier evidence, not a support claim. The follow-up power pass added 540
  sparse/dense alternative cells (150 attempts each) and the complete-data
  nested LR/FMG battery: unscaled, SB, mean/variance-adjusted, scaled-shifted,
  exact mixture, FMG SB/MV/SS/SF, EBA2/4/6, pEBA2/4/6, PALL, pOLS, and ALL.
  FMG SB/MV/SS matched their nested-test routes to machine precision. Raw
  nominal power was not rankable because overall null rejection ranged from
  0.063 for standardized flips through 0.110 for MV/ALL to 0.138 for SB.
  After method- and design-cell-specific empirical-null calibration,
  standardized/effective flips averaged about 0.186 power and MV, pEBA4, and
  ALL about 0.184; the apparent nominal FMG/SB advantage disappeared. Sparse
  violations were markedly easier than equal-Euclidean-norm dense violations,
  and adding null restriction directions reduced power at df=8. Under
  16-worker contention, median latency was 13 ms for two fits, 22 ms for the
  499-flip battery, 3 ms for the nested LR battery, and 8 ms for all 13 FMG
  transforms. The null and power sweeps took 147 and 396 wall-clock seconds.
  A later audit separated the score base from its spectrum transform and added
  the direct sandwich-studentized joint statistic
  `u'(G'B1G)^-1u ~ chi-square(df)` to `JointScoreTestResult` and
  `ScoreFlipTestResult`, including availability, minimum-meat-eigenvalue, and
  condition diagnostics. It reduces to the ordinary joint score under normal
  theory and to the existing robust scalar release at df=1. Experiment research/32 shows
  why it is a comparator rather than a small-sample default: main-grid rejection
  was 0.036 overall and 0.024 at df=8, with median df=8 meat condition numbers
  of 36/22/14 at N=60/100/200. Under the severe PL/VM copulas it rejected
  0.029/0.067 while mean cellwise condition numbers reached 251/96. Score-SB's
  near-nominal aggregate is therefore interpreted as trace-only spectral
  shrinkage and partial error cancellation, not correctness of the scaled
  chi-square law; pEBA4 was the most stable compromise between scalar collapse,
  noisy plug-in ALL, and unregularized meat inversion. A separate 32-cell
  dimension replay found direct-sandwich rejection between 0.038 and 0.061;
  with adequate sample ratios it was 0.050/0.046/0.043/0.058 at
  p=10/20/30/40 while median condition numbers improved from 45 to 22 as N grew.
  The derivation and iteration decisions live in experiment research/32's
  `notes/score-sb-audit.tex`.
  Experiment research/33 then replaced that broad synthetic grid with a focused,
  published weak-invariance design: exact Foldnes-Grønneberg-Moss Study 2/3
  loadings at p=5/20, G=2/8, n=400 per group; normal plus severe VM/IG/PL;
  null plus published power alternatives (32 cells, 200 replications, 199
  flips, no t5 or unbiased-Gamma variants). The covariance-equivalent marker
  parameterization keeps the paper's `(G-1)(p-1)` weak-invariance pair affine.
  All 6,400 fits/flips/nested batteries succeeded. Effective/standardized null
  rejection was 0.0553/0.0547 and differed on only 2 of 3,200 decisions; raw
  power was 0.2422/0.2419 and matched-null power 0.2131/0.2125. Thus restriction
  rank alone did not make standardization useful at n/group=400: median
  flip-covariance displacement was 0.00093 and its correlation with |delta p|
  only 0.09. Cost nevertheless exploded with rank. At df=4/19/28/133, median
  standardization time was 0.8/23/87/4,080 ms; the df=133 flip call took 7.35 s,
  of which score resampling was 1.94 s. The eight-worker run took 36.7 minutes
  with zero failures (severe PL calibration at p=20,G=8 added about six minutes
  per cell). In this homogeneous, adequately sized design the effective flip is
  the practical choice; standardization needs small-n/leverage or visible
  variance-displacement evidence, not merely many restrictions.
  The compact comparator battery reinforced the earlier caution about aggregate
  size: direct-sandwich score, score-SB, score-pEBA4, and nested ALL/MV rejected
  0.049/0.054/0.045 and 0.050/0.052 overall, but each moved across the VM/IG/PL
  families. Their matched-null powers lay between roughly 0.22 and 0.24, versus
  0.21 for the flips, at only 200 null draws per cell; this is illustrative
  evidence rather than a method ranking.
  The former research/34 fixed-rank frontier had no local substantive results
  and its unused harness has been retired. Its p=5 versus p=20 nuisance-dimension
  question at G=8, df=28 is now a consumer-gated hypothesis in
  `project/backlog/speculative.md#fixed-rank-flip-standardization`, with cheaper
  effective-flip/score alternatives and historical remote identifiers. It is
  not a queued simulation; maintained flip methods and dense-oracle gates remain.
  Experiment research/35 is the first concrete residual/RLS-flip GOF derivation, kept
  leaf-local pending calibration. For one complete covariance block it forms
  model-centred saturated covariance contributions and projects them through
  the expected-information residual U-factor; the all-plus quadratic equals
  the structured RLS GOF statistic (independently gated to `1.2e-7`). Random
  signs calibrate that df-dimensional residual sum. An optional empirical-meat
  whitening acts on the already projected residual scores and is deliberately
  reported unavailable when its covariance is rank deficient; it is not the
  nested test's flip-specific nuisance correction and adds no hidden ridge.
  The 16-cell normal/PL null-power probe (p=5/20, n=100/400, one omitted
  residual covariance, 100 replications, 199 flips) completed 1,600 batteries
  without failure in 36.4 seconds. Effective null rejection was plausible but
  noisy at df=5 (.01--.08 across four cells), and poor at df=170: .12/.13 under
  normal n=100/400 and .33/.08 under PL. Empirical standardization was
  impossible at df=170,n=100 and worsened n=400 rejection to .23/.11 while the
  median projected-meat condition number reached about 37/42,000 under
  normal/PL. Thus the algebraic GOF bridge survives, but broad core promotion
  does not: the next gate is a larger low/moderate-rank n/df calibration, not a
  regularized high-rank default.
  Former research/36 (now research/06's pilot lane) extends direct-FIML nested scores and probes
  the published FIML--FMG two-group, six-indicator configural-to-metric design
  (`df=5`) at group-1 n=50/100/200, 0/15/30% MCAR, normal/severe PL data, and
  null/loading-power truths (36 cells, 100 replications, 199 signs). Basic,
  nuisance-effective, and standardized flip null rejection averaged
  0.009/0.064/0.062; effective versus standardized differed on only 3 null
  decisions. The important split was normal versus PL (effective 0.038/0.090),
  not missingness. Pattern-specific covariance displacement increased with
  missingness, but standardization rose from about 0.8 ms complete to 5--7 ms
  at n1=100 with missingness and supplied no material calibration gain. Score
  pEBA4 rejected 0.038/0.060 under normal/PL, whereas nested-LR pEBA4 rejected
  0.035/0.185. Across 3,600 attempts there were 27 fit failures and 29 further
  nested-battery conditioning failures, but zero flip failures conditional on a
  successful fit; the four-worker run took 41.9 seconds. The subsequent screen
  below supersedes the proposed larger null gate. Its unique power control is
  retained, with no independent replication claim from overlapping seed bases.
  Former research/37 (now research/06's score-flips lane) completed a 240-cell atlas: normal and
  severe VM/IG/PL data, group-1 n=50/100/200/400 (group 2 at 70%), complete,
  15/30% MCAR, paper-style 30% MAR, stronger logistic 30% MAR, and loading-
  equality ranks 1/3/5. Its 500-attempt, 199-sign screen reuses one configural
  FIML fit across ranks and produced 120,000 rank-specific test attempts.
  Equal-cell rejection was 0.0128/0.0693/0.0678 for basic/effective/
  standardized flips, 0.0526/0.0496 for score SB/pEBA4, and 0.1135/0.1102/
  0.1004 for nested LR SB/pEBA4/exact mixture. Normal effective/standardized
  rejection was 0.0558/0.0549, while VM and PL were about 0.075/0.073 and
  0.076/0.074; small n and higher rank, not missingness percentage alone,
  marked the liberal frontier. Score pEBA4 put 97.9% of cells in 0.025--0.075.
  An independent predefined 11-cell confirmation (2,000 attempts, 999 signs)
  put effective/standardized flips at 0.0808/0.0778 and score pEBA4 at 0.0473;
  VM/PL n1=50, 30%-MCAR flips remained around 0.11--0.12. Standardization
  changed 0.145% of screen and 0.301% of confirmation decisions. Its median
  phase cost rose from 3.8 ms at 199 signs to 94.8 ms at 999 signs, versus
  12.1 ms for the effective sign sums in the latter run. Flip versus nested
  numerical availability was 99.49% versus 98.85% in the screen and 98.17%
  versus 96.10% in the hard-cell run. The 12-worker runs took 14.3 and 10.8
  minutes. This makes the effective flip a useful non-ad-hoc diagnostic, not a
  uniformly calibrated small-n default under severe nonnormality; at rank <=5,
  pattern standardization is not the missing correction. Score SB's 0.0526
  aggregate is retained only as empirical error cancellation: at ranks 3/5 its
  p-values differ from pEBA4 by about 0.0048/0.0063 while the score-spectrum
  coefficient of variation averages 0.57/0.72. These historical aggregates include
  nonnormal MAR, whose generating-model restrictions are not independently proved
  FIML pseudo-nulls. The consolidated report separates those stress cells from
  complete/MCAR and normal-MAR calibration. Further flip expansion is banked
  pending a named incomplete-data consumer or new rank/missingness regime.
- The same scaling in the moment metric for the LS estimator tiers (2026-06).
  Continuous ULS/GLS/WLS/DWLS: `inference::frontier`
  `{modification_indices,score_tests}_robust` overloads taking the
  `estimate::gmm::Weight`, with A1 = Σ_b (n_b/N)·Δ'WΔ and
  B1 = Σ_b (n_b/N)·Δ'WΓ̂WΔ built by
  `estimate::continuous_ls_param_space_sandwich` (expected bread only; Γ̂
  empirical from raw, caller-supplied per-block, or model-implied Γ_NT per
  `WeightMoments` — the GLS weight with the Γ_NT(S) meat collapses to c ≡ 1
  exactly). All-ordinal and mixed DWLS/WLS (plus all-ordinal ULS):
  `estimate::frontier`
  `{modification_indices,score_tests}_{ordinal,mixed_ordinal}_robust` over the
  [thresholds ; associations] moment metric, reusing the `robust_ordinal`
  block assembly (W = estimation weight, Γ̂ = `stats.NACOV`) through the shared
  `estimate::weighted_param_space_sandwich`; full WLS (W = NACOV⁻¹) reduces to
  the ordinary statistic exactly, and `mi`/`mi_scaled` keep the lavaan-matched
  row-type moment-scale convention (c carries no moment-scale factor). The
  per-direction worker is shared as
  `inference::frontier::score_for_direction_robust` (c is not W-scale
  invariant: A1/B1 must be on the same weight scale as score/info). The
  continuous ML/LS and ordinal/mixed-ordinal tiers are single- or multi-group
  (see the multi-group bullets). api/R wrappers deferred until a concrete
  consumer appears.
  Oracles:
  continuous DWLS (`se = "robust.sem"`, so lavaan's wls.v/gamma are the
  ADF NACOV — gaps ~1e-9) and all-ordinal WLSMV (polychoric NACOV — c gap
  ~6e-10) release-score fixtures 0007/0008, plus the two-group ordinal WLSMV
  release-score fixture 0012, in `regen_robust_score.R`; exact WLS/GLS
  reductions and a primitives re-assembly live in
  `cpp/tests/unit/score_robust_test.cpp`.
- Ordinary ordinal MI/release rank completion (2026-10-01):
  `inference::score_for_direction` exposes the existing ordinary ML/LS score
  projection, and ordinal fixed/absent candidates and equality releases reuse
  it. The duplicate ordinal worker and its absolute efficient-information floor
  are removed; ordinary and robust tests now use the same relative rank gate.
  Candidate enumeration, threshold/association moment scales and the fitting
  discrepancy are preserved. Gates in `score_robust_test.cpp` and
  `test_ordinal_score_rank.R` cover ULS/DWLS/WLS, latent units and nearby points,
  delta/theta, unequal groups, identified fixed loadings versus identification
  markers, fixed thresholds, equality releases, fixed-zero/absent rows, and
  fitting-weight scaling with invariant robust statistics. Weighted provenance
  and adapters remain in the MI completion matrix; mixed completion stays in
  0.3.0.
- Continuous-LS robust MI/release covariance dispatch (2026-10-01): the R
  wrappers honor `cov="empirical"` versus `cov="model_implied"` independently of
  the WLS estimator label, preserving explicit fitting W. Empirical covariance
  needs complete fitting observations; normal-theory covariance uses the selected
  structured/unstructured moments. A full empirical WLS weight reduces to the
  ordinary statistic only with the matching empirical Gamma. The C++ workers
  reject unimplemented Browne-unbiased covariance and non-empirical
  estimated-weight covariance, including the old R model-implied-to-empirical
  substitution. Caller-Gamma overloads remain core-only; broader weight
  provenance/adapters stay in the MI completion matrix. Gates cover raw versus
  supplied Gamma, means/unequal groups, diagonal/full weights, scaling transport,
  empirical versus NT controls, estimated-weight mode and explicit unsupported
  errors in `score_robust_test.cpp` and `test_wls_robust_covariance.R`.
- Estimated-weight recipe guard (2026-10-02, C++): every continuous IJ
  consumer (robust SEs, MI/release, profile tests, RBM, residuals) rebuilds
  the weight from its recipe, and a non-empty caller weight must equal that
  rebuild (relative Frobenius gap at most 1e-6 per block, scale included) or
  the call fails with the new `PostError::Kind::UnsupportedInference`. An
  empty weight lets a recipe-named entry point define it.
  `estimate::continuous_ls_ij_mode_for(FixedWeightKind, supplied)` maps a
  fit's recorded recipe to its IJ mode and refuses supplied weights, whose
  influence is unknown. The ordinal IJ (SEs and MI) and the estimated-weight
  DWLS profile, RMSEA, CFI/TLI and CRMR paths require `W_dwls` =
  diag(NACOV)^-1 or `W_wls` = NACOV^-1, compared against the stats builder's
  own inverse; NT, DLS and supplied ordinal weights keep fixed-weight
  inference and are refused for the estimated-weight channel. Previously the
  R glue chose the IJ mode from the estimator label, so continuous DWLS, DLS
  and supplied-W fits (labelled WLS) received the ADF influence silently, DLS
  used a=0.5, and ordinal NT/DLS/supplied fits received the NACOV influence.
  Gates in `score_robust_test.cpp` cover each recipe's acceptance of its own
  weight, cross-recipe, wrong-a and rescaled-weight refusals in MI and the
  shared sandwich, the recipe-to-mode map, and ordinal NT/DLS/supplied
  refusals across MI, `robust_ordinal_ij` and the profile family with the
  fixed-weight RMSEA comparator retained. The R glue reads the recorded
  recipe (2026-10-02): `fit$moment_weight` ("custom" for a supplied W),
  `fit$stage2_dls_a` and `fit$W` resolve through `continuous_ls_ij_mode_for`
  in MI/release, the GMM profile test and CI, RBM, estimated-weight residuals
  and case influence; estimated-weight requests for supplied-W fits fail
  with `UnsupportedInference`. Lab switches default to
  `estimated_weight = TRUE` for misspecification-robust inference, including
  RBM (continuous LS, ordinal/mixed and ML2S); explicit FALSE retains the
  fixed-weight comparator.
  WLS-computed fits use `fit$W`; an explicit `weight` must equal it or a
  common positive multiple (same minimizer, kept for the weight-scale
  transport checks). Fits without the record fall back to the label's recipe
  under the C++ guard. The `ij_weight`/`dls_a` profile overrides are removed.
  ML2S RBM and case influence read `stage2_weight`/`stage2_dls_a` from the fit
  and refuse a disagreeing argument (previously `frontier_rbm()` defaulted to
  NT and case influence dropped the DLS a). The ordinal LS score workers
  (ordinary and robust) refuse association-ML estimates with
  `UnsupportedInference`, closing the gap in `api::modification_indices`/
  `score_tests`; R already refused them. testthat
  (`test_weight_recipe_inference.R`) uses DLS(a = 1) = ADF exactly and
  DLS(a = 0) = GLS to optimizer precision across MI, releases, residuals,
  case influence, profile tests and RBM, plus supplied-W, ordinal NT/DLS/
  supplied and association-ML refusals.
- Two-stage (ML2S) MI and equality-release score tests (2026-10-02, C++):
  `inference::frontier::{modification_indices,score_tests}_ml2s` take the
  Stage-1 saturated moments and the matching Stage-2 fit. `mi` is the naive
  Stage-2 statistic on the EM moments (the ML MI for NT, the moment-quadratic
  MI with the Stage-2 weight otherwise); `mi_scaled` uses the Stage-1
  covariance n·ACOV as meat, so missing-data uncertainty enters the scaling.
  NT pairs the ML score with the structured normal-theory bread and the
  Stage-1 covariance as (n_b/N)-weighted caller meat (covariance rows only for
  models without means); ULS/DWLS/ADF/DLS use the new full-θ
  `estimate::fiml::frontier::ml2s_param_space_sandwich`, which adds the
  Stage-2 weight's data influence through the ML2S IJ blocks when
  `estimated_weight` is set. Expected information only; non-NT weights need a
  mean structure. Gates: on complete data every weight, fixed and estimated,
  equals the complete-data ML/LS robust tests with the empirical Gamma to
  1e-7, in one and two groups, for MI and releases; under MCAR the unscaled
  statistic equals the naive comparator and the estimated-weight DWLS meat
  moves the scaling. R (2026-10-02): `modification_indices{,_robust}()` and
  `score_tests{,_robust}()` dispatch ML2S fits through `$stage1`,
  `stage2_weight`, `stage2_dls_a` and the retained raw data; the ordinary
  wrappers return the naive statistic (now also for weighted Stage-2 fits and
  release tests), and every ML2S table carries `mi_type = "naive_stage2"`.
  `data`, `weight` and non-default bread/moments/cov are refused; observed
  information is a typed `UnsupportedInference`. On complete data the R
  tables equal the complete-data ML/LS robust tables to about 1e-14 for every
  weight, fixed and estimated, MI and releases. lavaan's `modindices()` on a
  `missing = "two.stage"` fit is not a fixture for the naive NT row: lavaan
  defaults two-stage fits to `h1.information = "unstructured"`, and its MI
  equals the score test on the EM moments with the unstructured expected
  information (4e-13 complete, 3e-5 with missing data), 31% away from the
  structured row on HolzingerSwineford1939.
- Caller-Gamma lab adapters: robust MI and equality releases accept validated
  per-group NACOV for complete ML, continuous LS and ordinal/mixed LS; the
  adapter applies ML's n/N block weights and preserves the fitting W. Supplied
  Gamma requires explicit fixed-weight inference, rejects casewise weight
  influence, and is unavailable for FIML/ML2S scores. Continuous-LS covariance
  and ML/LS profile-LRT wrappers expose their existing supplied-Gamma overloads.
  `test_caller_gamma.R` gates raw/supplied agreement, means and unequal groups,
  ordinal fitting-weight preservation and malformed/provenance refusals.
- MI/release component matrix (2026-10-02):
  [the estimator/weight inventory](../../validation/capabilities.md#mi-and-equality-release-score-components) names
  C++ and R gates or actual refusals for 23 recipes and 92 cells. It adds
  unequal-group and MAR ML2S reductions, continuous LS means/shared-label
  constraints, and complete ordinal recipe/release gates. GLS/WLS fixtures
  retain raw lavaan MI/EPC and explicitly transport its `(N-1)/N` score
  convention, with independent one-factor analytic Schur reconstruction;
  primary raw golden tolerances are tightened to measured floors. R still
  refuses model-implied robust ML release shortcuts and mixed ULS fitting;
  explicitly supplied Gamma_NT reaches the ML release core. Mixed robust
  unavailable choices report `NumericIssue`; ordinal NT/DLS/supplied estimated
  weights and association-ML report `UnsupportedInference`. The mixed ordinary
  MI factor-two oracle discrepancy remains TASK-33.4, with limited validation
  recorded visibly. No implementation, sampling-law or default changed.
- FIML (missing-data) robust MI and equality-release score tests, the MLR corner
  (2026-06): `inference::frontier::{modification_indices,score_tests}_fiml_robust`
  build the bread A1 = (N/2)·H (the analytic observed FIML information) and the
  meat B1 = ¼·scoresᵀscores from the casewise observed-pattern deviance
  gradients, reusing the FIML-FMG machinery via the shared
  `estimate::fiml::fiml_score_meat_bread` (also feeds `fiml_robust_mlr`). Because
  the FIML score is `-½·Σ_i scores_i`, the unscaled `mi` equals the non-robust
  FIML MI and the per-direction `c = gᵀB1g/gᵀA1g` is the Huber-White correction,
  → 1 under a correct normal model. Observed bread only (no expected-info FIML
  analogue), no H1 EM needed (two data passes per candidate, no EM), single- or
  multi-group via the same block-stacked Hessian / casewise-score layout used by
  `fiml_robust_mlr`. FIML has no batch augmentation, so this uses the one-by-one
  robust sweep.
  R binding completion (2026-10-01): `modification_indices_robust()` and
  `score_tests_robust()` dispatch direct FIML to these entries. Omitted
  bread/information select observed; explicit expected information, alternative
  covariance/moment recipes, supplied weights and estimated-weight mode are
  rejected. Retained raw data reuse the fitted missingness pack; explicit data
  (including masks/group blocks) rebuild their pack, without an H1 EM. Binding
  gates cover retained/rebuilt/explicit-data agreement, unequal groups, equality
  releases, marker exclusion versus identified fixed loadings, and indicator
  units in `test_fiml_robust_score.R` and `test-score-rank.R`.
  Oracle: FIML/MLR release-score fixture 0009 (`information.observed` /
  `lavScores`, θ-space assembly, c ≈ 2.16 on heavy-tailed + MCAR data) plus a
  non-robust-`mi` match and a c → 1 normal-data anchor in
  `cpp/tests/unit/score_robust_test.cpp`.
- Multi-group robust MI / score tests for the continuous ML and LS tiers
  (2026-06-13): the single-group guards in `inference::frontier` are removed; the
  per-block `n_b/N`-weighted sandwich (`robust::param_space_sandwich` /
  `estimate::weighted_param_space_sandwich`), per-group candidate enumeration
  (each absent statement is one candidate per group), and the full-θ-space
  nuisance projection all carry over unchanged, so no new statistics were needed.
  Validated by a two-group Γ_NT reduction (c = 1 exactly across unequal groups),
  a heavy-tailed empirical case, a GLS multi-group reduction, and the cross-group
  loading-invariance golden 0010 — the latter pins the `n_b/N` weighting that
  within-group reductions cannot (`A1 = lavInspect(fit,"information")` equals
  `Σ_b (n_b/N)·Δ_b'V_bΔ_b` exactly, c ≈ 1.24).
- Frontier fixed-misspecification profile-RMSEA / profile-LRT primitives for the
  continuous moment-quadratic tier (2026-06-22): the weighted-moment RMSEA
  helper forms the full moment-space profile Hessian
  `Q = W - W D B^{-1} D' W` from the same observed-Hessian bread used by the
  misspecification-robust SE path, computes the `QΓ` spectrum with
  `robust::compute_profile_contrast_spectrum`, and reports both the signed
  trace `tr(QΓ)` used by the RMSEA correction
  `sqrt(max(F - tr(QΓ)/N, 0) * G / df)` and the positive-spectrum summaries used
  by mixture-tail approximations (`bias_trace`, `spectrum_size`, negative count,
  and rank are carried separately). When the first-stage data metric is positive
  definite, the RMSEA result also carries the small profile pencil
  `ν_j = eig(B^{-1} Ã)` plus its predicted positive/negative/rank counts; dense
  `QΓ` eigensolve remains the source of actual mixture weights.
  With `G_hat = V_o^-1/2 W_p D`,
  `Q = V_o^(1/2)(I - G_hat B^-1 G_hat')V_o^(1/2)` and
  `A_tilde = D' W_p V_o^-1 W_p D`, the inner spectrum is `(m-p)` unit
  eigenvalues plus `1-nu_j`, where `nu_j = eig(B^-1 A_tilde)`. For full-rank
  Gamma, congruence preserves inertia: positive count is
  `(m-p) + #{nu_j < 1}`, negative count is `#{nu_j > 1}`, and rank is
  `(m-p) + #{nu_j != 1}`. Singular Gamma restricts these counts to its range.
  The counts are Gamma-free only conditional on fixed Q, which can move with
  moments/weights. Small residuals make integer rank floor-sensitive; signed
  trace and positive-tail trace remain distinct when the contrast is indefinite.
  `estimate::weighted_moment_profile_lrt` compares two such profile Hessians in
  a common first-stage moment space, uses the positive spectrum of
  `(Q_H0 - Q_H1)Γ` for mixture and adjusted tails, and reports nominal `df_diff`
  separately from the actual `spectrum_size`. The continuous wrappers
  `estimate::continuous_ls_profile_rmsea` and
  `estimate::continuous_ls_profile_lrt` supply
  the observed bread plus Γ from either caller blocks or complete raw data.
  `estimate::weighted_moment_profile_rmsea_two_metric` generalizes the same
  dense engine to two-metric profile Hessians
  `Q = V0 - W* D B^{-1} D' W*`. The complete-data ML adapter
  `estimate::ml_profile_rmsea` / `estimate::ml_profile_lrt` uses the
  sample/saturated normal-theory metric for `V0`, the fitted-implied metric for
  `W*`, and the observed ML Hessian scaled to the per-unit profile bread.
  Covariance-only models use the vech(S) moment; mean-structure models use the
  stacked `[mean; vech(S)]` moment and empirical Gamma from
  `data::empirical_gamma_with_means` when raw data is supplied. ML2S-NT reuses
  the same two-metric ML adapter through
  `estimate::fiml::two_stage_nt_profile_rmsea` /
  `two_stage_nt_profile_lrt`: saturated EM moments are treated as the Stage-2
  complete-data sample statistics, while Stage-1 uncertainty is supplied by
  `two_stage_gamma_from_acov(sm, false)` over the stacked `[mean; vech(cov)]`
  moment blocks; overloads accept either precomputed `SaturatedMoments`, raw
  data, or raw data plus a precomputed `FIMLPack`/`FIMLH1`. Raw-data FIML is
  wired through `estimate::fiml::fiml_profile_rmsea` /
  `fiml_profile_lrt`: it uses the EM saturated metric `H/n_b` for `V0`, the
  model-implied observed-pattern H1 metric for `W*`, the observed FIML
  information scaled per observation as the bread, and the caller's FIML LRT
  chi-square as `N*fmin`; the same overload pattern reuses precomputed
  `FIMLPack`/`FIMLH1` and `SaturatedMoments`. These are basic dense research
  surfaces, not lavaan-parity fit-measure dispatch.
  `estimate::weighted_moment_profile_rmsea_estimated_weight` adds the
  diagonal-weight (categorical DWLS) case where the weight `W=diag(1/γ)` is
  itself a first-stage quantity: it assembles the value-function Hessian over the
  *extended* moment vector `x=(u,γ)`,
  `Q = [[W,R],[R,S]] - [[WD],[RD]] B^{-1} [D'W,D'R]` with `R=diag(r/γ²)`,
  `S=diag(r²/γ³)`, via the two-metric engine (extended `jacobian=[D;D]`,
  `V0=[[W,R],[R,S]]`, `W*=blkdiag(W,R)`, joint NACOV `Γ_x`) and restates `df` to
  the classical u-moment count. The residual-driven γ channel is dormant at
  exact fit (`Q` collapses to `W − W D B^{-1} D' W`) and reshapes the reference
  law under fixed misspecification. The all-ordinal estimator wiring that
  *produces* `(D, γ, r, Γ_x)` is `estimate::ordinal_dwls_profile_rmsea` /
  `ordinal_dwls_profile_lrt`: it pulls `(D, γ, r, B)` from the ordinal DWLS fit
  and builds the joint NACOV `Γ_x` of `(u, γ)` from stacked per-case influence
  rows `[g_i | IF_i(γ)]`, reusing the same `ordinal_gamma_diag_*` influence
  channels as `robust_ordinal_ij`; its `chisq_standard`/`df` match
  `robust_ordinal`. Mixed continuous/ordinal DWLS is wired through
  `estimate::mixed_ordinal_dwls_profile_rmsea` /
  `mixed_ordinal_dwls_profile_lrt`, which assemble the same `(D, γ, r, Γ_x)`
  block from `MixedOrdinalStats`, `mixed_moment_jacobian`,
  `mixed_observed_bread_analytic`, and the mixed `mixed_gamma_diag_*`
  influence/Jacobian channels (including observed/missing variants), so the
  mixed path also shares the `robust_mixed_ordinal` standard χ² and df. The R
  package exposes these research surfaces through
  `magmaan_core$ordinal_profile_rmsea` / `ordinal_profile_lrt` and
  `magmaan_core$mixed_ordinal_profile_rmsea` / `mixed_ordinal_profile_lrt`,
  taking the same explicit ordinal or mixed stats object used for fitting.
  The absolute-fit companion is `estimate::ordinal_crmr_misspec_inference`
  (`OrdinalCrmrInference`; `estimate::ordinal_crmr` point +
  `OrdinalFitMeasures.crmr`): a criterion-at-estimator sandwich
  `Q_G = Dφᵀ V0 Dφ` with `V0` the correlation-selector and `Dφ` the full extended
  `(u,γ)` residual jacobian, reusing the catml projector and the profile `Γ_x`. It
  returns a bias-corrected CRMR/SRMR point, an exact-fit mixture p-value, and a
  CI that propagates the estimated weight (normal-theory `g_Gᵀ Γ_x g_G` under
  misspecification, `weighted_chisq` mixture at the null); an `estimated_weight`
  flag gives the fixed-weight comparator. Single-group only so far. Empirically
  the γ channel is only ~2–3% of the CRMR variance — CRMR's fixed metric makes it
  largely robust to weight estimation, unlike the metric-dominating RMSEA / nested
  test. Verified by `cpp/tests/checks/ordinal_crmr_inference` (bias/variance/coverage
  vs Monte-Carlo). R bindings and lavaan `crmr` parity are deferred.
  The large-γ absolute-fit case is `estimate::ordinal_rmsea_misspec_inference`
  (`OrdinalRmseaInference`): RMSEA's criterion is the discrepancy `F = rᵀWr`
  itself, so the envelope theorem gives the gradient as the bare profile score
  `g_F = (−2Wr, −r²/γ²)` (no projector); it returns the bias-corrected RMSEA, an
  exact-fit mixture p-value, and a normal-theory CI on `F₀` with
  `Var(N·F)=N·g_Fᵀ Γ_x g_F`, reusing the profile `Q`/`Γ_x`/bias/spectrum.
  Empirically the γ channel is large and variance-reducing (`r` and `γ` co-vary
  negatively): the estimated-weight CI is calibrated while the fixed-weight one
  is conservative (over-covers, increasingly with misspecification) — accounting
  for the estimated weight tightens RMSEA's interval. Verified by
  `cpp/tests/checks/ordinal_rmsea_inference`. Single-group; R bindings deferred.
  The first *incremental* (two-model) case is
  `estimate::ordinal_cfi_tli_misspec_inference` (`OrdinalIncrementalFitInference`):
  misspecification-robust CFI and TLI with CIs. The user model and the analytic
  independence baseline (linear in its thresholds, so `Q_b` is exact) run through
  the *same* profile primitive with the *shared* `Γ_x`, so the joint law of
  `(T_u,T_b)` is one bilinear form `Cov(T_u,T_b)=N gᵤᵀΓ_x g_b`; CFI `=1−δ_u/δ_b`
  and TLI `=1−(Q̄_b/Q̄_u)δ_u/δ_b` (noncentralities `δ=T−Q̄`, generalized df
  `Q̄=tr(QΓ_x)`) are a ratio delta-method, the TLI interval being the CFI interval
  scaled by `Q̄_b/Q̄_u`. To leading order `Var(CFI)≈Var(T_u)/δ_b²`, so CFI
  inference is a rescaling of the RMSEA-side variance. Derivation in
  `cfi_tli_misspec_inference.tex`; gated by `ordinal_test.cpp`
  and the C4 MC harness `cpp/tests/checks/ordinal_cfi_inference`. Empirically CFI's CI
  is calibrated and (unlike RMSEA) largely robust to weight estimation (γ-share
  ≈±3–8%); the leading-order simplification holds only at weak misfit. TLI's point
  is calibrated but its variance over-states at strong misfit (`c=Q̄_b/Q̄_u`
  ill-conditioned), so CFI is the index to trust for an interval.
  The whole estimated-weight fit-index family (RMSEA + CRMR/SRMR + CFI/TLI with
  CIs) is R-exposed through one consolidated surface,
  `estimate::ordinal_fit_measures_misspec_inference`
  (`OrdinalMisspecFitMeasures`), bound as `infer_ordinal_fit_measures_misspec`
  and the `@export`ed `fit_measures_misspec(fit, ordinal_stats, ...)`; the
  per-index C++ entry points stay unbound. All of RMSEA/CRMR/CFI/TLI are now
  **multi-group**: the criteria pool as `Σ_b n_b·crit_b` with a block-diagonal
  `Γ_x`, the gradients stack with `√(n_b/N)` weights, and the baseline is the
  per-group independence model (validated by a duplicate-group reduction:
  points invariant, statistic/df double, intervals tighten). A stabilized-`c`
  TLI variance remains deferred.
  Mixed continuous/ordinal DWLS has the first reportable fit-index slices:
  `estimate::mixed_ordinal_rmsea_misspec_inference` reuses
  `mixed_ordinal_dwls_profile_rmsea` and adds the same estimated-weight
  envelope-score CI and exact-fit mixture p-value for RMSEA, while
  `estimate::mixed_ordinal_crmr_misspec_inference` adds CRMR/SRMR inference over
  the existing standardized mixed association residual convention (including
  observed continuous-variance scale derivatives), and
  `estimate::mixed_ordinal_cfi_tli_misspec_inference` adds CFI/TLI inference
  with the mixed DWLS independence-baseline convention. The mixed consolidated
  surface is `estimate::mixed_ordinal_fit_measures_misspec_inference`, exposed as
  `magmaan_core$mixed_ordinal_fit_measures_misspec` and the exported R companion
  `fit_measures_misspec_mixed_ordinal()`.
- Multi-group robust MI / score tests for the ordinal and mixed-ordinal tiers
  (2026-06-13): the `require_single_group_ordinal` guard in
  `estimate::frontier` is removed; the ordinal sandwich already loops over
  `stats.R.size()` and pools `A1/B1` through the shared `n_b/N`-weighted
  `estimate::weighted_param_space_sandwich`. The per-block threshold and
  association moment Jacobians write into the correct full-θ columns, so the
  continuous-tier nuisance projection carries over. Validated by exact
  two-group full-WLS reductions for all-ordinal MI/score and mixed-ordinal
  MI/score, a finite non-trivial two-group DWLS ordinal scaling case, and the
  two-group WLSMV golden 0012 (`c ≈ 0.856`) assembled from lavaan's
  per-group `delta` / `wls.v` / `gamma` lists.
- df>1 total release (2026-06-13): `inference::frontier::score_tests_robust_joint`
  releases all active equality constraints at once. The NT joint statistic is the
  multivariate score (Lagrange-multiplier) form `T = uᵀV⁻¹u` over the
  df-dimensional efficient-score subspace `G` (release normals made
  info-orthogonal to the nuisance subspace; u = Gᵀs, V = GᵀIG), which reproduces
  lavaan's `lavTestScore(fit)$test$X2`. The robust report mean-scales by
  `c̄ = tr((GᵀA1G)⁻¹(GᵀB1G))/df = Σλ/df` and also gives the exact eigenvalue-mixture
  p-value `Pr(Σλⱼχ²₁ > T)` via the QUADPACK `imhof_upper`, with λ the generalized
  eigenvalues of (GᵀB1G, GᵀA1G) (`JointScoreTestResult`; the shared worker
  `score_for_subspace_robust`). At df=1 it reduces to the per-row `c` bit-for-bit.
  Complete-data ML (raw or caller Γ̂); single- or multi-group. Oracle: df=2 joint
  fixture 0011 (mi = lavaan total, c̄ ≈ 2.03, p_mixture vs `CompQuadForm::imhof`)
  plus a df=1-reduces-to-per-row unit check.
- Estimated-weight ("complete-sandwich") robust MI / score tests (2026-06-22):
  an `estimated_weight` flag routes the per-direction scaling `c = gᵀB1g/gᵀA1g`
  through the complete Hall-Inoue infinitesimal-jackknife meat (the
  data-dependent-weight `IF(Ŵ)` term), not the fixed-weight `Δ'WΓ̂WΔ` — the
  per-parameter robust denominator lavaan never builds (it scales MI only by the
  global SB scalar). Core: `estimate::weighted_param_space_sandwich_ij` plus the
  IJ adapters `continuous_ls_param_space_sandwich_ij` (GLS/WLS/DWLS/DLS) and
  `ordinal_param_space_sandwich_ij` (all-ordinal DWLS/WLS), the latter built from
  `build_ordinal_ij_blocks` shared with the `robust_ordinal_ij` SE path. ML/FIML
  and mixed-ordinal reject the flag. Leading-order only under misspecification
  (ULS, fixed weight, unaffected). R: `{modification_indices,score_tests}_robust(…,
  estimated_weight=)`.
- Misspecification-robust ("complete-sandwich") case influence (2026-06-23): the
  casewise dual of the estimated-weight SE.
  `estimate::continuous_ls_casewise_influence_ij` decomposes the complete-sandwich
  covariance into its per-case contributions `c_i = (1/N)·K·A⁻¹·(Δ_b K)ᵀ·v_i`
  (observed bread `A`; `v_i = g_i·W + IF(Ŵ)`), the one-step leave-one-out change
  carrying the data-dependent-weight term that the naive (semfindr / Pek-MacCallum)
  influence drops. It reuses the same `build_continuous_ls_ij_blocks` as the MI /
  SE paths, so `Σ_i c_i c_iᵀ` reproduces the `robust_continuous_ls_*_ij` vcov to
  1e-9 (a free self-check); a `naive` field drops `IF(Ŵ)`, so the difference is
  the per-case `Δ'W'_d` diagnostic. R: `infer_casewise_influence_ij_fit`,
  `est_change_{raw_,}approx(fit, type = "estimated.weight")` (continuous
  GLS/WLS/ULS). Self-validated against the exact GLS leave-one-out engine (which
  re-estimates the weight per drop): the complete one-step tracks it at RMSE ~3e-4
  regardless of misfit, while the naive degrades with it (the Hall-Inoue order
  promotion). Writeup in `papers/estimated-weight-se`. The per-case row extraction
  is the shared `estimate::casewise_influence_from_ij_blocks(blocks, K, bread)`
  (the per-case dual of `robust_weighted_moment_ij`), reused by the continuous
  accessor and the ordinal `estimate::ordinal_casewise_influence_ij` (categorical
  DWLS/WLSMV — the headline cell, on the existing `build_ordinal_ij_blocks`).
  Both case-influence regimes (`standard`, `estimated.weight`) are multiple-group
  (block-stacked `g{b}_{row}` ids); `est_change_*_approx(type = "estimated.weight")`
  covers continuous GLS/WLS/ULS, ordinal DWLS/WLSMV, and two-stage **ML2S**
  (`estimate::fiml::two_stage_casewise_influence_ij` — the missing-data member,
  fusing the Stage-1 saturated-moment influence with the Stage-2 weight term;
  NT/robust.two.stage carries a zero correction, non-NT DWLS/ADF/DLS the live
  one), routing ordinal/ML2S fits to their bindings automatically. Every
  estimator/group cell self-checks `Σ_i c_i c_iᵀ ≡` the matching observed-bread
  `robust_*_ij` vcov.
- Observed-bread robust SEs and observed-Hessian U-factors use total-N scaling
  and work on block-stacked multi-block covariance and mean-structure models.
- Browne's unbiased reduced gamma has a single-block reduced-matrix shorthand
  and a casewise multi-block primitive.
- The weighted-sum-of-chi-squares tail behind the FMG/pEBA/pOLS p-values
  (`robust::weighted_chisq_upper`) uses Ruben's positive-weight
  central-χ² series first, avoiding oscillatory-cancellation failures in deep
  tails. If the series does not converge it falls back to Imhof's characteristic
  function inversion through vendored QUADPACK `qagi`
  (`cpp/third_party/quadpack/`, f2c-translated, public domain;
  cpp/cmake/QuadpackVendor.cmake — the second vendored static library after PORT),
  with a dense-Simpson fallback for weakly damped small-df tails where qagi's
  extrapolation breaks. A self-contained C++ golden pins fixed-spectrum FMG
  p-values for each method against constants generated from R `stats` and
  `CompQuadForm::imhof`; unit tests also pin deep equal-weight tails to exact
  χ² references. The same module exposes `robust::weighted_chisq_quantile`
  for deterministic positive-mixture cutoffs, and
  `robust::compute_profile_contrast_spectrum` computes the positive `QΓ`
  spectrum for regular profile-Hessian nested-test research primitives.
