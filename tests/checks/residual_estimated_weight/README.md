# residual_estimated_weight advisory simulation

Advisory Monte-Carlo check for the estimated-weight ("complete-sandwich")
standardized residuals
(`measures::frontier::standardized_residuals_estimated_weight`). **Not** part of
the default test suite (`ctest`) — it is stochastic and slow. lavaan only ever
uses the normal-theory projection + `Gamma_NT` for residual SEs, so there is no
package oracle; this validates the property an exact number-match cannot.

The estimated-weight residual ACOV claims to be the actual sampling covariance
of the standardized residuals under an estimated second-stage weight. The check
draws one base sample from a 4-indicator population with an extra residual
covariance the one-factor model cannot reproduce (so real residuals remain),
fits continuous DWLS, records the analytic NT and estimated-weight residual SEs,
then nonparametric-bootstraps the base sample (resample rows, refit, recompute
the residuals) to get the empirical sampling SD of each off-diagonal correlation
residual.

Expected: under heavy tails the **estimated-weight** SE / bootstrap-SD ratio sits
near 1, while the **NT** SE / bootstrap-SD ratio is below it (the NT SE assumes
normality and under-states the sampling variability). The exact projector/
assembly is gated separately and exactly in `tests/unit/residuals_test.cpp`
(Fixed mode == hand-built `Q·(Γ̂/n)·Qᵀ`); this harness adds the empirical
calibration the analytic gate cannot.

Run:

    just quick   # n=1500, reps=300,  df=6
    just all     # n=3000, reps=1500, df=6

`build/` and `results/` are gitignored.
