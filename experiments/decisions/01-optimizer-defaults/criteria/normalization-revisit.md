# Revisit the original scaling cases — 2026-09-27

Written before the run. This is the author's requested paired development
comparison after enabling complete-data ML/PSD normalization, not a new
held-out defaults decision. It reuses all 51 test populations, their fitted
models and compatible unit transforms from the PSD scaling follow-up:
seed base 202609290, three replications at each N (306 draws, 1,356 cases).
The original fitted models retain their marker/std.lv convention; constrained
models omit the incompatible mixed-unit transform. No groups are in this set.
Smoke uses one population per family, the smaller N, one replication, seed +1.

Compare `normalize_sample=FALSE` and `TRUE` on the same current library, so
both arms use the same current accuracy checks. This isolates the full fitting
transformation (including starts), not the earlier diagnostic/clamp changes.
The methods and controls are explicit:

- Ordinary ML: layered starts, native start transport, NLopt L-BFGS, information
  coordinates.
- Direct PSD ML: FABIN3 with automatic std.lv start transport, NLopt SLSQP,
  diagonal lifted preconditioning.
- Ordinary-then-PSD: FABIN3 with automatic std.lv start transport in its ordinary
  L-BFGS stage; SLSQP/diagonal PSD when the library fallback requests it. Both
  stages use the selected normalization setting. No multistart or marker changes.
- All: 5,000 evaluations, relative objective tolerance 1e-12, relative step
  tolerance 1e-10; PSD eigenvalue floor and feasibility tolerance both 1e-6.

Report the library accuracy verdict, admissibility, runtime, errors and start
branch. ML success is the library verdict (improper ML estimates are not
silently discarded); PSD success additionally requires admissibility. Retain
the earlier standardized-extent >10 screen as a separate descriptive count,
not proof of a pole or runaway and not a replacement accuracy verdict.

Pair gains/losses by case within each route. Among points that both pass,
classify objective improvement/worsening at 1e-6*(1+abs(original-unit fmin)).
Report by family and units, and retain every changed verdict or objective.
Compare native versus rescaled runs within each arm: verdict differences,
objective differences at that same threshold, and implied-covariance differences
above 1e-5 after dividing by sample standard-deviation products. Retain parameter
and covariance rows locally for investigation. Timing is descriptive, since
parallel local runs and fixed arm order are not a controlled speed benchmark.

Do not fix algorithms mid-run or silently rerun failures. Unexpected scope or
infrastructure failures stop the run; investigate numerical regressions after
preserving the complete comparison. Neither a route change nor a barrier
implementation is authorized by this diagnostic; barrier is the next TODO.
