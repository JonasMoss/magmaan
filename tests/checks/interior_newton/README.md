# Interior Newton audit and NLopt L-BFGS controls

Advisory, standalone diagnostic; no production audit or defaults changed.
Consumes the canonical model, objective, constraints and observed-information
APIs. Direct NLopt calls expose actual controls and preserve every terminal
candidate, independently of the production adapter's return policy.

From the repository root:

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/stock-new 10
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/extended-new 60
TZ=Europe/Oslo Rscript tests/checks/interior_newton/summarize.R tests/checks/interior_newton/results/stock-new tests/checks/interior_newton/results/extended-new
```

Requires the existing opt CMake configuration and its Eigen/NLopt dependency
sources, clang++, cc and R. `run.sh` builds the core, rejects existing output
directories, and records source/library hashes. The second run compiles a
local copy of NLopt `plis.c` with only `mred=10` changed to `mred=60`, then links
that object into this executable. Neither dependency source nor installed
library is edited. The extended version is a diagnostic, not stock NLopt.

Design is fixed in `check.cpp`: seven model/settings, N=100/1000/100000,
three replications, three global measurement-unit multipliers, nine control
profiles: 1,701 fits per backtracking version. The same underlying dataset is
used across profiles and units. Population covariance generators include
one-factor CFA with 4 or 12 indicators; two-factor CFA with 8 indicators;
a weak-loading/high-correlation variant; equality-linked loadings and means;
a structural regression with means; and a misspecified one-factor model
with an omitted residual covariance. These are diagnostic simulations, not
published-data examples or a prevalence sample.

Fit starts use canonical FABIN. Linear equalities use canonical reduction K.
No bounds, PSD fitting, optimizer restarts, alternative optimizers, SEs, or
tests are requested. Observed information is computed for the audit itself.
No library default is altered. A candidate outside the positive-definite
primitive covariance interior is ineligible for the proposed statistical
interpretation even if its ambient Newton calculation is available.

`assess` is an experimental interior diagnostic, not a public API. It uses
G=N*gradient(f), I=observed total information, after linear equality reduction.
Diagonal equilibration and LLT solve give d=sqrt(G'I^-1G), total EDM=d^2/2,
and the predicted Newton step. Indefinite/singular or ill-conditioned
curvature is unavailable, not repaired. The condition cap (1e12) and solve
residual cap (1e-10) are explicit provisional numerical guards. Their behavior
under arbitrary badly conditioned coordinate transforms is not established.

Executable checks cover exact quadratic error, nonsingular affine changes,
objective/sample-size scaling, and singular/indefinite rejection. Seven
independent directional finite differences check the analytic Hessian's
normalization and equality reduction in the SEM cases. The summary verifies
coverage, exact paired design, and finite-difference error. Raw negative
NLopt return codes are retained; they are not automatically numerical failures.

Timings are single executions in fixed profile order and describe this pilot
only. Evaluation counts are the more reliable cost comparison. The Hessian
cost includes construction plus the experimental audit, excludes the older
geometric audit, and can partly be reused for observed-information inference.

Results and design interpretation: `docs/research/interior-newton-audit.md`.
The stock and extended local runs are under this directory's ignored
`results/`; use a new directory for every subsequent execution.

Methodological references and the threshold rationale are recorded in the
study note: Dennis, Gay & Welsch (1981), DOI 10.1145/355958.355965, section 6;
James's MINUIT manual 94.1; and the documented Stata/iminuit stopping rules.
The note distinguishes retrieved versions from unavailable published PDFs.
The .01 candidate is an accuracy budget relative to sampling uncertainty,
not a cutoff calibrated from this experiment's pass rate. Its interpretation
as actual remaining error still depends on the local quadratic approximation.
