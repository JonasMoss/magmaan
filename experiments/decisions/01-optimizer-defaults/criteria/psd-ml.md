# Lane psd-ml: criteria (written and committed before the run)

**Decisions.** For complete-data ML under the PSD constraint:

- **A.** The start of a direct (cold) PSD fit.
- **B.** The PSD preconditioning.
- **C.** The route: a direct PSD fit, or two-stage.
- **D.** Whether the PSD feasibility tolerances need a fix.

**Configurations.** Every PSD fit uses SLSQP, the only default-build backend
that takes the covariance-link constraints.

| Arm | Route | Start | Preconditioning |
|---|---|---|---|
| `psd_default` | direct PSD | library default (transported FABIN3) | diagonal (library default) |
| `psd_default_none` | direct PSD | library default | none |
| `psd_layered` | direct PSD | layered | diagonal |
| `psd_layered_none` | direct PSD | layered | none |
| `twostage_default` | ordinary ML (library default start, L-BFGS); PSD refit warm-started from it only when the ordinary fit is not certified and admissible | | diagonal |
| `twostage_layered` | the same, with the ordinary stage from the layered start with PORT | | diagonal |
| `witness` | direct PSD from the population values | | diagonal |

The witness counts only toward attainability and the best known objective.

**Problems.** The same populations, $N$, transforms and seed base as the
ml-gls lane, fitted by ML only, with 50 replications per $N$.

**Judge.** Scoring:

- A fit succeeds when `fit$converged` holds and the estimate is admissible.
  For two-stage arms, the returned fit must pass both, whichever stage
  produced it.
- A problem is attainable when any arm or the witness succeeds.
- The best known objective is the lowest successful objective, reached within
  $10^{-6}(1 + \lvert f \rvert)$.

**Rules.** They apply to test families; controls are reported only.

- **A.** Adopt the layered cold start if, in every test family, `psd_layered`
  against `psd_default` has wins ≥ losses in successes and in best known.
  Unlike the ordinary lanes, the best-known rate gates here. The PSD estimate
  itself is what the covariance-honest paper evaluates, and the API probe
  (one weak-marker draw at $N = 25$) showed the layered start certifying a
  worse minimum of the constrained problem.
- **B.** Switch preconditioning to `none` if, in every test family,
  `psd_default_none` against `psd_default` has no net loss in successes or in
  best known, and its total seconds over test families are lower. Otherwise
  keep `diagonal`.
- **C.** No rule. For each family, report successes, best known, median and
  total seconds, and the share of two-stage fits returned by the ordinary
  stage. The author decides.
- **D.** The tolerances bite if `tolerance.csv` records any fit, in any arm or
  family, that succeeds in native units and fails in rescaled units
  ($\times 100$, $\times 0.01$ or separate units) for the cause
  `link_feasibility` or `inadmissible_certified`.
  In that case, make the link-residual and feasibility tolerances relative to
  the sample units, and rerun this lane.

## Second run: PSD after the ML/GLS promotion (written 2026-09-26, after the promotion and before this run)

**Why.** Complete-data ML's default start became the layered start (with
L-BFGS) in f10fdd84. The PSD routes kept FABIN3. Two things about that were
never tested:

- **The fallback's ordinary step.** `frontier_fit_ml_psd_fallback` pins
  its ordinary step to FABIN3 with L-BFGS. The first run's alternative for
  that step was the layered start with PORT, not the new ML default.
- **The runaway rule.** The first run scored without it.

The author wants the PSD defaults to follow the new ML default if the
evidence allows it.

**Disclosure.** Before writing this section, a probe on the covariance-honest
paper's own designs (its two-factor model at $N = 10, 20, 50$, 150 draws
each; bullying draws; its two examples) compared the two ordinary steps. With
marker identification the layered step lost 33 fits and won 2. None of those
populations is used here, and the rule below is judged on this run's draws
only.

**Arms.**

| Arm | Route | Start |
|---|---|---|
| `psd_default` | direct PSD | library default (transported FABIN3) |
| `psd_layered` | direct PSD | layered |
| `twostage_default` | fallback, ordinary step FABIN3 + L-BFGS (the library's fallback today) | |
| `twostage_layered_lbfgs` | fallback, ordinary step layered + L-BFGS (the ML default) | |
| `witness` | direct PSD from the population values | |

Every PSD fit uses SLSQP with the diagonal preconditioning. The
no-preconditioning arms (rule B is settled) and the PORT fallback arm are
dropped.

**Draws.** Seed base 2026092604, never used before, with 50 replications per
$N$. Run id `2026-09-26-second`. Same populations, $N$ and transforms as the
first run.

**Scoring.** As in the first run, plus the runaway rule of the ml-gls third
run: a certified fit with standardized extent above 10 is a failure and no
witness of attainability.

**Rules.** Test families only; best known gates, as in rule A.

- **A (repeated).** The direct PSD cold start becomes the layered start if,
  in every test family, `psd_layered` against `psd_default` has wins ≥
  losses in successes and in best known.
- **E.** The fallback's ordinary step follows the ML default (the layered
  start) if, in every test family, `twostage_layered_lbfgs` against
  `twostage_default` has wins ≥ losses in successes and in best known.
- **C.** Still no rule. Report the route comparison per family and per
  transform.
- **D** is re-evaluated as in the first run.

**Outcome.**

- A passes: switch the direct PSD cold start to the layered start.
- E passes: drop the fallback's FABIN3 pin, so that its ordinary step is the
  ML default.
- A rule that fails leaves its default unchanged. The report names the
  failing families and the mechanism of their losses.
