# Iteration 12: expected-sensitivity score under MCAR

## Question

Does the projected score's expected-information construction have the wrong
asymptotic target, or is its advantage over the LRT a finite-sample property?
MCAR isolates this question from the nonnormal-MAR pseudo-target failure.

Under MCAR, missingness-pattern membership is independent of the outcomes. At
the covariance-correct null, every pattern's Gaussian score has population mean
zero at the same model moments. Consequently, the residual-curvature term in
the population Hessian vanishes and the count-weighted pattern Fisher
information is the correct sensitivity. Nonnormality changes the score meat,
which is retained in the robust spectrum, but does not invalidate the expected
bread at this null.

## Design

The statistic-by-spectrum runner used the six-indicator one-factor model, 30%
MCAR on all but the first indicator, and normal/VM2/IG2 generators. Expected-
sensitivity runs used 1,000 replications at `n = 120` and `n = 500`, and 500 at
`n = 2000`. The observed-sensitivity diagnostic used the same design. Results
below are native pEBA(4) rejection rates.

## Results

| Method | Generator | n = 120 | n = 500 | n = 2000 |
|---|---|---:|---:|---:|
| LRT/FMG | Normal | .085 | .059 | .044 |
| LRT/FMG | VM2 | .179 | .111 | .084 |
| LRT/FMG | IG2 | .112 | .054 | .046 |
| Expected score | Normal | .046 | .049 | .038 |
| Expected score | VM2 | .038 | .046 | .058 |
| Expected score | IG2 | .058 | .050 | .046 |
| Observed-sensitivity score | Normal | .020 | .025 | .036 |
| Observed-sensitivity score | VM2 | .082 | .077 | .082 |
| Observed-sensitivity score | IG2 | .057 | .060 | .058 |

At `n = 120`, 9 of 3,000 expected-score attempts and 11 of 3,000 observed-
sensitivity attempts were unusable; most shared the same LRT-spectrum rank or
fit problem, while two additional observed-sensitivity calls had non-positive-
definite tangent information. There were no failures at `n = 500` or
`n = 2000`.

The expected-score construction is close to nominal at every sample size and
shows no increasing-N drift. The observed-sensitivity version is conservative
under normal MCAR at small N and approaches the expected-score behavior as N
grows. Under the severe nonnormal laws it remains somewhat different at
`n = 2000`, consistent with slow convergence of realized Hessian quantities,
but it does not reveal a different pseudo-target.

The LRT/FMG result is also compatible with consistency. Normal and IG2 cells
are close to nominal by `n = 500`; VM2 decreases from .179 to .111 to .084.
The remaining VM2 excess is a finite-sample concern under unusually severe
skewness and kurtosis, not the persistent or increasing pattern seen under
nonnormal MAR.

## Interpretation

The expected-score advantage on complete and MCAR data is not evidence of an
inconsistent expected-information construction. At the covariance-correct
complete/MCAR null, expected sensitivity is the relevant population bread, and
using it also avoids finite-sample noise and indefiniteness in the realized
Hessian. The empirical meat and its projected eigenvalue spectrum still carry
the nonnormal fourth-moment correction.

The expected construction should not be generalized casually to nonnormal
MAR. There, pattern-specific selected moments need not share one Gaussian
mean/covariance target, and the actual pseudo-true sensitivity can contain the
residual-curvature terms omitted by Fisher information. More fundamentally,
the saturated H1 pseudo-target can lie outside H0, making global-fit rejection
an estimand-stress result rather than Type-I error.

For the provisional paper story:

1. use complete and MCAR cells to study robust null calibration;
2. include normal MAR as a correctly specified likelihood control;
3. use expected-sensitivity projected score as the stable primary score test;
4. retain observed sensitivity only as a diagnostic for pseudo-true geometry;
   and
5. discuss nonnormal MAR through explicit H0/H1 pseudo-targets rather than
   ranking ordinary rejection rates as size.
