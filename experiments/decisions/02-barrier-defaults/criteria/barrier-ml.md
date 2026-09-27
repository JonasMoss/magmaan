# Lane barrier-ml: criteria (written and committed before the run)

**Decision.** The default start and optimizer of the complete-data ML
barrier fitter, `frontier_fit_ml_multiinfo` (C++
`estimate::frontier::fit_ml_multiinfo`). Every fit here uses
`target = "determinacy"`, the latent-determinacy penalty
$\lambda \log\det Q$ of the sem-barrier paper.

**Why now.** The fitter's defaults were never judged as a whole:

- Its start is the transported FABIN3 start, pinned in the R glue when the
  layered start became the complete-data ML default (52bc9caa). No run
  compared the two for the barrier.
- Its optimizer, PORT, was chosen in engineering/18 (ca420456), an
  exploratory study. That study used research/47 draws, eight corpus fits and
  Ernst, and ran before the information coordinates (3c5e3d8b) and the
  penalized Newton verdict (73e27b1c), which both apply to the fitter now.

**Configurations.** Every arm runs in information coordinates (the default).
Before a fit starts, the fitter floors nonpositive variances and shrinks
covariances until the start is inside the barrier domain, in every arm.

| Arm | Start | Optimizer | Role |
|---|---|---|---|
| `default` | library default: transported FABIN3 | library default: PORT | baseline, what users get today |
| `layered_port` | layered | PORT | candidate: start only |
| `default_lbfgs` | library default | NLopt L-BFGS | candidate: optimizer only |
| `layered_lbfgs` | layered | NLopt L-BFGS | candidate: complete-data ML's default |
| `witness` | population values | PORT | attainability and best known objective only |

The population values are the ML estimates at the population moments, as in
decisions/01. On face and improper truths they lie on or outside the barrier
domain and are pulled inside like any other start.

**Weights.** Each draw is fitted at $\lambda = 0.25$ (the library default,
called without a weight in the `default` arm) and at $\lambda = 1$. The
paper's weight is not settled; a default must not depend on it. The weight
takes the place of the estimator in the scoring tables (`lambda0.25`,
`lambda1`).

**Problems.** The populations of decisions/01 (`R/families.R`, copied
unchanged), with the same roles, sample sizes and unit transforms (native,
$\times 100$, $\times 0.01$, and separate units for invariant models). 100
replications per $N$. Seed base 2026092701, never used before; smoke runs use
the base plus 1.

- Test families: research/47, engineering/15, Boomsma, Wolf, Chen, and the
  equality-constrained models. Ernst is a control.
- Disclosure: research/47 draws informed the current PORT default
  (engineering/18), and all of these populations were test problems for the
  ML/GLS start decision in decisions/01. No candidate here was developed on
  them. Both facts favour the baseline or are neutral, and the draws are
  fresh.

**Judge.** `fit$converged` at the pinned build (commit and build time in
`metadata.csv`): the Newton check on the penalized objective. The objective
scored is the penalized criterion, `fit$penalty$penalized_fmin`, which is what
the estimator minimizes.

**Scoring.** As decisions/01, with two runaway rules. A certified fit fails
and is no witness of attainability when either holds:

- **Standardized runaway:** its standardized extent (decisions/01's
  definition: the largest absolute standardized loading, latent path, latent
  correlation, or residual or disturbance ratio) exceeds 10. The barrier
  guards these faces, so this is expected never to fire; it is reported.
- **Marker pole:** its chart extent exceeds 1000. For every latent whose
  scale is set by a fixed nonzero loading, the chart extent is the largest
  absolute standardized loading among its indicators divided by the absolute
  standardized loading of the marker; the fit's chart extent is the maximum
  over such latents (1 without markers). It is unit-free and stays finite
  under the barrier except where the marker's standardized loading goes to
  zero, the marker-chart nonattainment the paper describes. The weak-marker
  population has a true chart extent of about 6.

Certified fits beyond 10, 100 and 1000 in chart extent, and beyond 5, 10 and
100 in standardized extent, are reported per arm; only the bounds above gate.

- A problem is attainable when any arm or the witness succeeds; only
  attainable problems score.
- The best known objective is the lowest successful penalized objective. A
  success reaches it within $10^{-6}(1 + \lvert f \rvert)$.

**Rules.** Test families only, for each candidate against `default`, in every
test family and at each weight:

1. **Primary.** Wins ≥ losses in successes (`paired.csv`).
2. **Best known.** Wins ≥ losses in reaching the best known objective. This
   gates, as in the PSD lane: the penalized estimate is the paper's object,
   so a certified worse point is a loss.
3. **Invariance.** At least as many draws with the same outcome across units
   as `default`, checked separately for common and separate units.
4. **Named losses.** `losses.csv` lists every attainable problem that
   `default` solves and the chosen candidate does not. Before promotion the
   report attributes each family's losses to a mechanism: an optimizer
   termination, a start in another basin, a runaway, or an error.

**Outcome** (`choice.csv`).

- A candidate is eligible when rules 1 to 3 pass in every test family at both
  weights.
- Among eligible candidates, choose the one with the most pooled certified
  test problems over both weights; ties go to more best-known problems, then
  to the smaller change (`layered_port` or `default_lbfgs` before
  `layered_lbfgs`).
- If no candidate is eligible, the default stays, the report names the
  failing families' losses, and the author decides.
- A chosen candidate becomes the fitter's default for both targets
  (`"joint"` has no evidence here; it shares the fitter and is outside the
  paper). After promotion, a run on the same seed base whose `default` arm
  reads the new library defaults must reproduce the candidate draw by draw.
- The FIML barrier (`frontier_fit_fiml_multiinfo`) is not covered and keeps
  its defaults.

Secondary outcomes, reported and not gating: median evaluations and seconds
among successes, the runaway tables, and the start repairs.
