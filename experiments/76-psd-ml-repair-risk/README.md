# PSD-ML repair anatomy and near-boundary risk

## Question

What does covariance-honest complete-data NTML change when ordinary NTML
leaves the primitive covariance domain, and does the change improve estimation
of a genuinely PSD population covariance near its boundary?

This experiment is descriptive and estimation-focused. It does not attach a
test, p-value, confidence interval, or model-selection interpretation to the
ordinary-versus-PSD objective difference.

## Part 1: repair anatomy

Four self-contained model/data summaries reproduce distinct cases from the
Little and Geiser continuous corpus:

1. a same-fit level/residual covariance reallocation;
2. a materially different linear-growth optimum;
3. a negative second-order disturbance repaired at negligible likelihood cost;
   and
4. a jointly indefinite quadratic-growth covariance with positive individual
   variances.

The local input file records the model syntax, sample covariance and mean,
sample size, and ordinary start. It is a private copy owned by this experiment;
the runner does not read another experiment, a test fixture, or a paper tree.

For each case, fit ordinary NTML by L-BFGS from the recorded ordinary solution,
then warm-start PSD-ML with SLSQP. Record:

- primitive covariance minimum eigenvalues and admissibility;
- objective difference and the descriptive scale `2 * N * delta_F`;
- maximum free-parameter displacement and its partable label;
- relative Frobenius distance between the two implied observed covariances;
- KKT-audited convergence; and
- fit time.

The scaled objective difference is called an “LR-scale cost” only because it is
on the familiar likelihood-ratio scale. It is not calibrated and must never be
reported as a likelihood-ratio test.

## Part 2: controlled risk path

Use three single-indicator latent variables with fixed unit loadings and fixed
indicator residual variances of `0.2`:

```text
f1 =~ 1*x1
f2 =~ 1*x2
f3 =~ 1*x3
x1 ~~ 0.2*x1
x2 ~~ 0.2*x2
x3 ~~ 0.2*x3
f1 ~~ f1 + f2 + f3
f2 ~~ f2 + f3
f3 ~~ f3
```

Let the population latent covariance be

\[
  \Psi(\lambda)=Q\operatorname{diag}(1,0.4,\lambda)Q',
\]

where the columns of $Q$ are the fixed orthonormal vectors

\[
  (1,1,1)'/\sqrt3,\quad
  (1,-1,0)'/\sqrt2,\quad
  (1,1,-2)'/\sqrt6.
\]

The observed covariance is

\[
  \Sigma(\lambda)=\Psi(\lambda)+0.2I.
\]

Thus $\Sigma$ remains safely positive definite even at the exact primitive
boundary $\lambda=0$. The design isolates covariance-component honesty from
singular observed likelihoods, loading indeterminacy, and structural weak
identification. Unconstrained NTML is saturated in the observed covariance and
may estimate an indefinite $\Psi$; PSD-ML fits the same likelihood over
$\Psi\succeq0$.

For every generated dataset, fit ordinary NTML and then an ordinary-warm
PSD-ML refit. Compare:

- convergence and covariance admissibility;
- fitted primitive rank and minimum eigenvalue;
- relative Frobenius error in $\Psi$ and $\Sigma$;
- population Gaussian KL loss;
- descriptive likelihood cost; and
- computation time.

The paired risk improvement is defined as ordinary error minus PSD error, so a
positive value favors PSD-ML. Summaries are reported both unconditionally and,
more importantly, conditional on the ordinary estimate being inadmissible.
The latter describes what the explicit audit/refit policy changes.

Fitted rank is descriptive. At a true covariance boundary, a constrained MLE
can lie on the boundary or in the interior with nonvanishing probability; this
experiment does not treat fitted rank as a consistent rank selector.

## Profiles

- **Smoke:** `N = 30,100`, `lambda = 0,.01,.20`, five replications.
- **Pilot:** `N = 25,50,100,250`,
  `lambda = 0,.001,.01,.05,.20`, 200 replications.
- **Full:** the pilot grid with 1,000 replications. Run only if the pilot is too
  imprecise for the descriptive risk comparisons.

Seeds are

```text
20260731 + 100003 * N + 1009 * lambda_index + replication.
```

Run with:

```sh
Rscript run_experiment.R --smoke
Rscript run_experiment.R --pilot
quarto render report.qmd
```

All result artifacts are ignored. The report reads only this experiment's
`results/` directory.
