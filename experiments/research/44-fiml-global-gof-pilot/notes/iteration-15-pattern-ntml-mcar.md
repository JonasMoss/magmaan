# Iteration 15: patternwise normal-theory ML under MCAR

Historical record: PNTML and its smoke driver were removed from the library on
2026-09-30. The results below describe the original implementation revision.

## Question

Can the saturated Gaussian-FIML estimate be projected onto the structured SEM
with a genuinely normal-theory objective that preserves the observed-pattern
geometry? The desired construction should reduce exactly to ordinary NTML on
complete data and recover direct-FIML first-order efficiency under normal MCAR,
without replacing the objective by a generic moment quadratic.

## Construction

Let $r$ index observed-variable patterns, let $P_r$ select their observed
coordinates, and let $\widehat\eta=(\widehat\mu,\widehat\Sigma)$ be the
saturated Gaussian-FIML estimate. The pattern-NTML criterion is

$$
F_{\mathrm{PNTML}}(\theta)
=\sum_r \widehat\pi_r\,
  2D_{\mathrm{KL}}\!\left[
  N(P_r\widehat\mu,P_r\widehat\Sigma P_r')\,\|\,
  N(P_r\mu(\theta),P_r\Sigma(\theta)P_r')
  \right].
$$

The implementation reuses the ordinary patternwise Gaussian FIML kernel after
replacing each pattern's empirical mean and covariance by the matching marginal
of the common Stage-1 estimate. The counts and observed-coordinate maps remain
unchanged. With one all-observed pattern, this is ordinary NTML exactly.

For the normal-theory inference contract, $V_{\mathrm{pat}}$ is the sum of the
pattern-normal expected-information pullbacks in saturated-moment coordinates.
Stage 1 is assigned the model-based law
$\Gamma_N=V_{\mathrm{pat}}^{-1}$. The objective metric and the inverse Stage-1
law therefore coincide. The residual $U\Gamma_N$ operator has exactly $df$ unit
eigenvalues, the scale factor is one, and

$$
\operatorname{Avar}(\widehat\theta_{\mathrm{PNTML}})
=\left(\Delta'V_{\mathrm{pat}}\Delta\right)^{-1}/N.
$$

Under normal MCAR, this is the expected-information bound for direct FIML and
the two estimators have the same first-order influence. The point estimator
continues to target the structured saturated-FIML moments under ignorable MAR,
but the pattern-count-only normal information need not be the correct Stage-1
law when missingness depends on observed values. The present model-based SE and
chi-square contract is therefore MCAR, not a normal-MAR efficiency or robust
inference claim. An empirical Stage-1 sandwich would be a distinct robust
PNTML extension.

## Deterministic gates

The C++ tests establish:

1. exact complete-data equality of the PNTML and ordinary NTML information and
   point estimates;
2. analytic-gradient agreement with central finite differences under MCAR;
3. a nontrivial difference between the pattern metric and ordinary complete-data
   NT information under MCAR;
4. unit $U\Gamma_N$ eigenvalues, unit scaling, and finite model-based covariance.

The R test repeats the complete-data reduction, exercises the exported frontier
fit, and verifies bit-level reuse of a caller-supplied Stage-1 object.

## Twenty-replication plumbing smoke

`smoke_pattern_ntml_mcar.R` used experiment research/44's normal six-indicator one-factor
population at $N=250$, comparing complete data with 30% MCAR. Each replication
shared one saturated Stage-1 fit across ML2S and PNTML. This is far too small for
a size conclusion; it is only a paired implementation screen.

| Mechanism | Usable | Realized missing | Median $\max|\widehat\theta_{P}-\widehat\theta_F|$ | Median $\max|\widehat\theta_{ML2S}-\widehat\theta_F|$ | Median PNTML/FIML expected-SE ratio | Mean $|T_P-T_F|$ |
|---|---:|---:|---:|---:|---:|---:|
| Complete | 20/20 | .000 | $7.17\times10^{-14}$ | $4.27\times10^{-7}$ | .9997 | $2.93\times10^{-13}$ |
| MCAR 30% | 20/20 | .2988 | .00998 | .01460 | .9971 | .503 |

All 120 estimator fits converged. The largest PNTML eigenvalue deviation from
one was $2.78\times10^{-14}$ on complete data and $1.55\times10^{-14}$ under
MCAR. Nominal 5% rejection was one of 20 for PNTML in both mechanisms, one of
20 for complete-data FIML, and zero of 20 for MCAR FIML. Those counts are
descriptive only.

## Decision

Keep PNTML as a separate frontier objective, not as another `stage2_weight`
member. It has a clear normal-theory identity, exact complete-data reduction,
and the requested normal-MCAR influence. The next scientific gate, if pursued,
is a larger paired MCAR study of finite-sample bias, RMSE, expected-information
SE calibration, and null size. MAR should be studied separately, with the
model-based and empirical-sandwich inference contracts kept explicit.
