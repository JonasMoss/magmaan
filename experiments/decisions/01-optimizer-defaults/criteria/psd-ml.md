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
