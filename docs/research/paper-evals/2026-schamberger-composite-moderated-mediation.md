# Moderated mediation with composites: The composite moderated structural equations approach

**Cite.** Schamberger, T., Schuberth, F., & Henseler, J. (2026). *Behavior Research Methods*, 58:104. DOI: 10.3758/s13428-025-02930-w.
**PDF.** `external/refs/Schamberger et al. 2026 - Moderated mediation with composites - The composite moderated structural equations approach.pdf`
**Read.** 2026-06-26  ·  **Verdict.** background

## TL;DR
CMS = LMS (latent moderated structural equations, Klein-Moosbrugger) stacked on top of the Henseler-Ogasawara composite specification, so unknown-weight composites can act as moderated mediators. The novel piece is a glue procedure over existing tools; the engine is a nonlinear (latent-interaction) estimator, which is outside magmaan's scope.

## Contribution
- A three-step "structural-after-synthesis" recipe: (1) confirmatory composite/factor analysis (CCFA, all constructs freely covarying, plain ML in lavaan) to estimate the H-O composite loadings Λ; (2) invert Λ to recover weights `W = (Λ')^{-1}`, build composite scores (closed form, eq. 5); (3) replace each composite by its score and fit the moderated model by LMS (R package `modsem`).
- Lets an unknown-weight composite be a *mediator* (and moderated), which the one-step / two-step / pseudo-indicator approaches cannot do flexibly (fixed-weight only, or parameter blow-up).
- Monte Carlo (two weight sets, N=250/500/1000): CMS tracks population values, MSE shrinks with N, and it matches or slightly beats PLSc (`cSEM`), mainly more power at N=250. No misspecification studied; ignores weight-estimate uncertainty in step-3 SEs (flagged as a limitation).

## Relevance to magmaan
Two separable layers:
- **H-O composites (linear, in scope).** Step 1 is exactly the confirmatory composite analysis magmaan already does: the single-group ML H-O / FC-SEM slice has landed (`Op::Composite`, `CompositeMode::FcSem`, `estimate::fit_ml_fcsem`, gated by `composite_golden_test.cpp`; see [todo.md](../../backlog/todo.md) "Composite models" and [[composite-support-wip]]). Step 2's weight recovery is the same `W=(Λ')^{-1}` inversion. So magmaan could in principle serve as the step-1/step-2 engine, and the paper is independent confirmation that the H-O direction we are building has live downstream consumers.
- **LMS / moderated mediation (nonlinear, out of scope).** The headline method is LMS: a mixture-of-normals likelihood maximized by EM to capture a latent interaction. That is the canonical *nonlinear* SEM estimator. magmaan ports "linear SEM under complete-data normal-theory estimators," so latent product terms / LMS are firmly outside the mission, same as QML and product-indicator approaches. Nothing here pulls that boundary.

No new inputs magmaan would need to produce, and no parity target: the moderation machinery lives in `modsem`, not lavaan. The composite half is already tracked, the moderation half is out of scope, so there is nothing to build or graduate.

## Verdict
background — confirms H-O composites have real downstream use (mild support for the existing composite roadmap, [[composite-support-wip]]), but the actual method is an LMS latent-interaction estimator outside magmaan's linear/normal-theory scope. Nothing to build.
