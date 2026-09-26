# Lane ml-gls: criteria (written and committed before the run)

**Decision.** The default start and optimizer of complete-data ML and GLS.

**Configurations.** All four arms run in information coordinates, which have
been the default since 2026-09-26.

| Arm | Start | Optimizer | Role |
|---|---|---|---|
| `default` | library default: transported FABIN3 for ML, native FABIN3 for GLS | library default: NLopt L-BFGS | baseline, what users get today |
| `layered_port` | layered | PORT | candidate |
| `layered_lbfgs` | layered | NLopt L-BFGS | attributes a change to the start |
| `default_port` | library default | PORT | attributes a change to the optimizer |
| `witness` | population values | PORT | attainability and best known objective only |

**Problems.** The populations in `R/families.R`. Test populations:

- research/47 (13);
- engineering/15's weak marker and high $R^2$;
- Boomsma (1985) (12);
- Wolf et al. (2013) (18);
- Chen et al. (2001), with four fitted models;
- the equality-constrained models (10).

Ernst is a control: it was used to develop the 2026-09-22 start work, so it
is reported and never gates. Each population is drawn at a small $N$ (25, or
$p + 10$ when larger) and at $N = 100$, with 100 replications per $N$. The seed
base is 20260926, never used before, and seeds are cell-stable.

Each draw is fitted in four versions: native units, every variable $\times 100$,
every variable $\times 0.01$, and separate units (alternating $\times 100$ and
$\times 0.01$). Separate units apply only to models without equality
constraints across variables.

**Judge.** `fit$converged` at the pinned build, whose commit and build time are
recorded in `metadata.csv`. Scoring:

- A fit succeeds when the verdict certifies it.
- A problem is attainable when any arm or the witness succeeds; only
  attainable problems score.
- The best known objective is the lowest successful objective. A success
  reaches it within $10^{-6}(1 + \lvert f \rvert)$.

**Rules.** They apply to test families; controls are reported only.

1. **Primary.** In every family and estimator, pooled over $N$, units and
   replications, `layered_port` certifies at least as many attainable
   problems as `default` does (wins ≥ losses in `paired.csv`).
2. **Invariance.** In every family and estimator, `layered_port` has at least
   as many draws with the same outcome across units as `default`. This is
   checked separately for common units and for separate units.
3. **Named losses.** `losses.csv` lists every attainable problem that
   `default` solves and `layered_port` does not. Before any promotion, the
   report attributes each family's losses to a mechanism: an optimizer
   termination, a start in another basin, or an error.

**Outcome.**

- If rules 1 and 2 pass, promote the candidate.
- If either fails in some family, nothing changes until the report explains
  that family's losses; then the author decides.

The secondary outcomes are reported but do not gate, per the 2026-09-26
scoring rule: the best-known rate, and the median evaluations and seconds
among successes.

## Confirmation run (written 2026-09-26, after the first run and before this one)

**Why.** The first run (seed base 20260926) failed rules 1 and 2 for
`layered_port`: one Chen ML loss, and common-unit invariance for the Boomsma
and Wolf ML families. Its report traces both to PORT running with its generic
controls (1000 iterations, PORT's default tolerances), because the ML option
profile sets only NLopt's budget. Those draws are now development data.

**Candidate.** `layered_port_ml`: the layered start with PORT given ML's
budget and tolerances:

- `max_iter = 5000`;
- `port = list(max_eval = 5000, rel_f_tol = 1e-12, x_tol = 1e-10)`.

Diagnostic arms:

- `layered_port_budget`, with the budget only;
- the first run's four arms, unchanged.

Nothing else changes: the same populations, transforms, judge and scoring.

**Draws.** Seed base 2026092602, never used before. Run id `2026-09-26-confirm`.

**Rules.** Rules 1 to 3 above, with `layered_port_ml` in place of
`layered_port` as the candidate.

**Outcome.**

- If rules 1 and 2 pass, promote the candidate as the complete-data ML and
  GLS default: the layered start, PORT, and those controls in the ML option
  profile.
- Then a run on the same seed base, whose `default` arm reads the new
  library defaults, must reproduce the candidate's outcomes draw by draw.
- If a rule fails, there is no promotion, and the report explains the losses.

## Third run: the runaway rule (written 2026-09-26, after the confirmation run and before this one)

**Why.** The confirmation run failed its rules, and its report found the
cause. On draws without a proper ordinary minimum, PORT walks down the
divergent Heywood path, and the Newton check often certifies the point where
the budget stops it. Those runaway certifications count as successes, and they
make problems "attainable" through the witness. So the earlier scoring rewards
an optimizer for how far it runs on problems with no answer. Both earlier runs'
draws are development data.

**Scoring change (the runaway rule).** The standardized extent of a fit is the
largest absolute value among:

- its standardized loadings and latent paths;
- the latent correlations;
- the residual and disturbance variance ratios.

It is recorded per fit (`standardized_extent` in `R/fits.R`) and is unit-free.
A certified fit whose extent exceeds 10 (engineering/15's bound) is a runaway.
It counts as a failure and as no witness of attainability, for every arm and
the witness alike. Certified fits beyond 5, 10 and 100 are reported per arm
(`runaways.csv`); only 10 gates.

**Candidates.**

- `layered_lbfgs`: the layered start with NLopt L-BFGS, a start-only change.
- `layered_port`: the layered start with PORT under its current controls.

Arms: `default`, `default_port`, the two candidates, and the witness. The
confirmation run's PORT-control arms are dropped.

**Draws.** Seed base 2026092603, never used before. Run id `2026-09-26-third`.
Same populations and transforms.

**Rules.** Rules 1 to 3 above, for each candidate against `default`, with the
runaway rule in the scoring.

**Outcome, decided separately for ML and for GLS** (`choice.csv`):

- A candidate is eligible for an estimator when rules 1 and 2 pass in every
  test family for that estimator.
- If both are eligible, `layered_lbfgs` is chosen, being the smaller change.
  `layered_port` is chosen instead only if it certifies more pooled test fits
  and has no net loss against `layered_lbfgs` in any test family.
- If one is eligible, it is chosen.
- If neither is, the report names the failing families' losses and the author
  decides, as in the first run's outcome clause.
