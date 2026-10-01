## Prespecified design

Use eight centered continuous indicators, four per factor. Both factor
variances are fixed to one, their correlation is 0.4, and the loadings on each
factor are (0.8, 0.7, 0.9, 0.6). Residuals are independent with variances
one minus the squared loading. Factor and residual innovations are independent
standard normals. Thus each indicator has population variance one. Fit the
same two-factor model with all eight loadings, eight residual variances and
the factor correlation free, retaining unit factor variances. Orient each
factor positively. Center each sample and use the ML covariance divisor N;
do not mix N and N-1 likelihood conventions.

The targets are the second loading (0.7) and factor correlation (0.4), tested
separately while all other free parameters are nuisance parameters. Estimate
the same unpenalized complete-data ML model for every interval method. Record
the exact optimizer, bounds and covariance-domain policy before the smoke run;
hold these fixed across methods and bootstrap fits. Boundary-active fits are
reported separately rather than repaired, clipped or silently removed.

Compare 95% model-based Wald intervals using expected information, efficient
score inversion using expected information at the restricted fit, and ordinary
profile-LR inversion. Observed-information model-based Wald is a secondary
diagnostic. Use N = 100, 300 and 1,000, with 200 datasets per cell. Compute all
methods on every dataset. The main run uses seed base 2026092750; smoke runs
use 2026102750. Select replicate IDs before examining fits or results.

For every method record coverage, each tail's noncoverage, median and mean
width, endpoint asymmetry, failure and boundary rates, constrained-fit counts,
and elapsed time. Report coverage among successful intervals alongside the
successful-and-covering fraction of all attempts, and show paired comparisons
on common-valid datasets separately. Retain failures by method and reason.

## Bartlett pilot

Write the profile statistic as

$$
T(b)=2\{\ell(\widehat\theta)-\ell(\widetilde\theta_b)\}.
$$

For **each candidate** b, simulate B datasets from the normal model fitted
under b, refit both unrestricted and b-restricted models in every bootstrap
dataset, and estimate the rank-one null mean

$$
\widehat c(b)=B^{-1}\sum_{j=1}^{B}T_j^*(b).
$$

The corrected confidence set accepts b when
$T(b)/\widehat c(b)\leq\chi^2_{1,.95}$. This is a null-fitted parametric
bootstrap mean correction, not a bootstrap quantile test. Re-estimate nuisance
parameters inside every bootstrap refit. Do not reuse a factor calculated at
the true parameter or at the unrestricted estimate as the primary correction.

Start with the first 50 prespecified N = 100 datasets, for the loading only,
using B = 399. Compute the ordinary methods on the same subset. Repeat the
first ten datasets with B = 1,999, retaining the original bootstrap draws as a
prefix, and with an independent bootstrap seed stream. Log the Monte Carlo SE
of the estimated correction. This is a feasibility pilot: at 50 datasets the
Monte Carlo SE of 95% coverage is about 3.1 percentage points.

Keep base normal draws fixed across candidate values within a dataset and
target, transforming them by each restricted covariance. This makes the
objective reproducible and reduces artificial roughness. Check candidate
endpoints by direct evaluation, not solely interpolation. If correction noise
moves an endpoint by more than 5% of the uncorrected interval width, mark that
dataset unstable; report its frequency without discarding it. Do not claim
that 5% is a statistical optimality threshold. A larger coverage run is a
later design amendment, conditional on measured computational cost and endpoint
stability, not on favorable pilot coverage.

An optional independently simulated population-mean factor is a diagnostic
benchmark only. It must use separate draws and be labeled as unavailable to
users. No truth-based factor is allowed in the feasible method.
