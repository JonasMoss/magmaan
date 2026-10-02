# magmaanlab 0.2.0 (in development)

- Estimated-weight inference follows the weight recipe a fit records
  (`fit$moment_weight`, `fit$stage2_dls_a`), not its computational estimator
  label. Continuous DWLS and DLS fits now receive their own weight influence,
  with the fitted DLS mixing weight; previously they received the ADF influence
  and DLS assumed a = 0.5. Fits with a supplied W, and ordinal NT/DLS fits,
  fail with an `UnsupportedInference` error for the estimated-weight channel;
  fixed-weight inference is unchanged.
- WLS-computed fits use their recorded `fit$W`, so `weight =` is optional in
  modification indices, score tests, robust sandwiches, profile tests, RBM,
  residuals and case influence. An explicit `weight` must equal the recorded
  weight or a positive multiple of it.
- `frontier_profile_lrt_parameter_gmm()` and
  `frontier_profile_lrt_ci_parameter_gmm()` drop `ij_weight` and `dls_a`; the
  recipe comes from the fit.
- Two-stage (ML2S) fits support `modification_indices()`, `score_tests()` and
  their `_robust` variants for every Stage-2 weight. `mi` is the naive Stage-2
  statistic (attribute `mi_type = "naive_stage2"`); `mi.scaled` uses the
  Stage-1 moment covariance, plus the Stage-2 weight influence with
  `estimated_weight = TRUE`.
- `frontier_rbm()` and estimated-weight case influence use an ML2S fit's
  recorded Stage-2 weight and DLS mixing weight; `stage2_weight` and `dls_a`
  default to the record and must agree with it.
- Modification indices and score tests refuse ordinal association-ML fits in
  every interface.

# magmaanlab 0.1.0

First versioned dependency for the magmaan 0.1.0 simulation prerelease. The lab
retains its methods-development API, including the EQS constructor.

Development is unfinished: remaining bugs and incomplete coverage of the main
estimators and inferential procedures remain to be resolved and validated.
Version 0.1.0 provides a versioned simulation snapshot, not a finished release.

High-level fit reconstruction retains defined parameter estimates using the
shared C++ evaluator, independently of covariance inference.
