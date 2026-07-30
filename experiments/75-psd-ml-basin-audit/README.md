# PSD-ML multistart basin audit

## Question

When covariance-honest complete-data ML returns a KKT-stationary admissible
solution, how often does the result depend materially on the start? In
particular, does the default start attain the best likelihood found by a
reasonable start portfolio, and do equal-likelihood solutions represent the
same implied moments or distinct parameterizations?

This is an optimization-validation experiment. It does not regularize the
model, discard boundary maxima, or claim to prove a global maximum. Its
reference is the **best attained feasible stationary solution** across a
prespecified portfolio.

## Model and data

Use the same two-factor, six-indicator Gaussian SEM as the small-\(N\)
convergence experiment:

```text
Y =~ y1 + y2 + y3
X =~ x1 + x2 + x3
Y ~ X
```

The population loadings are \(1.0, 0.8, 0.6\), the structural coefficient is
\(0.25\), the exogenous \(X\) variance and \(Y\) disturbance variance are one,
and all six indicator residual variances are one. The experiment is
self-contained: it reimplements this short DGP and does not read another
experiment's source or results.

The pilot uses \(N\in\{10,20,50\}\). These cells cover the severe-boundary
regime, the transition regime, and a mostly stable control. For each \(N\),
screen 1,000 datasets under the standard PSD-SLSQP start using the documented
seed rule

```text
seed = 20260729 + 1000003 * N + replication
```

The screen supplies the default result and defines the replay sample:

1. a deterministic simple random sample of 100 datasets per \(N\);
2. every dataset where the default fit fails to return or fails the terminal
   KKT audit; and
3. the 25 audit-converged datasets per \(N\) with the largest
   \(|\hat\beta|\).

Deduplicate datasets while retaining all selection tags. Primary frequency
statements use only the random core. The failure and extreme-estimate
enrichments are diagnostic and are reported separately, never pooled into a
prevalence estimate.

## Start portfolio

All primary fits use PSD-ML with NLopt SLSQP. Starts are expressed in the
ordinary free partable coordinates; the production PSD routine constructs its
usual covariance-factor start from each vector. Shared free indices are
perturbed once, so equality-label semantics are preserved.

For every replay dataset, run:

1. the standard FABIN start;
2. the ordinary L-BFGS ML estimate as a warm start, when that fit returns;
3. the default PSD estimate as a restart, when it returns;
4. three moderate perturbations of the FABIN start;
5. three broad perturbations of the FABIN start;
6. two moderate perturbations of the ordinary-ML start; and
7. two broad perturbations of the ordinary-ML start.

This gives at most 13 starts per dataset. Perturbation directions are
deterministic standard-normal draws indexed by dataset seed, start family, and
start number. For free coordinate \(k\), use

\[
  \theta^{(j)}_k =
  \theta^{(0)}_k + a_j\max(|\theta^{(0)}_k|,0.25)z_{jk},
\]

with \(a_j=0.10\) for moderate and \(a_j=0.50\) for broad starts. Do not redraw
or silently repair a start that fails; record the failure. Primitive covariance
starts still pass through the estimator's documented eigenvalue projection.

Cholesky-column sign flips are not part of the R experiment because the lift
coordinates are intentionally private and the returned partable is invariant
to those representational signs. If needed, that invariance belongs in a
deterministic C++ probe rather than a new public start API.

## Eligibility and clustering

A returned solution is eligible only if:

- the equality-constrained terminal audit reports convergence;
- the primitive covariance audit reports admissibility; and
- the objective and implied moments are finite.

For each dataset, let \(F_\star\) be the smallest objective among eligible
solutions. “Best attained” means

\[
  F-F_\star \leq 10^{-6}\max(1,|F_\star|).
\]

Always retain the raw objective gaps and repeat summaries at \(10^{-7}\) and
\(10^{-5}\). This makes the result insensitive to a single arbitrary numerical
cutoff.

Within the best-attained objective cluster, compare:

- free partable estimates using
  \(\max_k |\Delta\theta_k|/(1+|\theta_{\star k}|)\);
- implied covariance and mean vectors using relative Frobenius distance; and
- the fitted primitive-covariance ranks and minimum eigenvalues.

Use \(10^{-4}\) as the parameter-equivalence threshold and \(10^{-6}\) as the
implied-moment-equivalence threshold, while retaining the continuous
distances. Classify each dataset on three nonexclusive axes rather than forcing
distinct phenomena into one label:

1. **objective basins:** unresolved (no eligible solution), single eligible
   objective cluster, or competing eligible objective clusters;
2. **best-cluster equivalence:** unique parameters and moments; same implied
   moments but different parameters; or equal objective but different implied
   moments; and
3. **start eligibility:** all available starts eligible or start-sensitive
   eligibility.

“Stable unique” is the conjunction of one objective basin, unique
best-cluster parameters/moments, and eligibility from every available start.
Parameter multiplicity is not called multiple local maxima merely because
lift coordinates or observationally equivalent representatives differ.

## Confirmation of suspicious datasets

Escalate only datasets classified as competing stationary basins or equal
objective/different moments:

1. restart one representative from each cluster with a 20,000-iteration
   budget and tighter tolerances;
2. run IPOPT from those near-solution starts as an optional backend
   cross-check, recording failure rather than requiring it; and
3. profile the structural coefficient \(\beta\) over an adaptive grid spanning
   the attained estimates and \(0.25\), fixing \(\beta\) by a model equality
   and fitting nuisance parameters by warm continuation from both grid
   directions.

A competing maximum is reported only when distinct feasible KKT-stationary
clusters survive the high-budget restart. The profile is diagnostic evidence
about the principal structural direction, not a proof of globality over the
full nuisance space.

## Primary outputs

Report by \(N\), and separately for the random core and each enrichment:

- probability that the default start hits the best attained objective;
- probability that the ordinary-ML warm start hits it;
- number and gap of eligible objective clusters;
- rates and cross-tabulations of the three classification axes;
- return, KKT, and best-hit rates by start family;
- parameter and implied-moment distances within the best cluster; and
- covariance-boundary/rank summaries by basin classification.

The raw table has one row per dataset and start. A dataset summary table holds
the cluster assignments and best-attained comparisons. A cluster-membership
table retains the representative starts used by the confirmation stage.

## Profiles and stopping rule

- **Smoke:** 30 screened datasets per \(N\), 10 random-core datasets per
  \(N\), all screen failures, the three largest-\(|\hat\beta|\) datasets, and
  six starts (default, ordinary warm, and two perturbations around each).
- **Pilot:** the 1,000/100/25 design above and the full 13-start portfolio.
- **Full:** not predefined. Expand the random core or start portfolio only if
  the pilot finds a material default-to-best gap, competing stationary
  basins, or insufficient precision for their frequency.

If the pilot finds no competing basins and the upper confidence bound for the
random-core default-miss rate is below 1% after pooling only where the
\(N\)-specific results are compatible, stop. The conclusion is then limited
to this model, DGP, and start portfolio.
