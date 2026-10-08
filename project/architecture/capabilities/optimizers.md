### Optimizer backends

User-facing the optimizer is selected by a single kebab-case `optimizer = "..."`
string threaded through `fit_model()` and the underlying `fit_ml/fit_uls/fit_gls/
fit_wls/fit_*_snlls/fit_*_ordinal` entries; the table lives in
`cpp/include/magmaan/estimate/backend_strings.hpp` and parses into the C++
`Backend` enum. Accepted strings: `"ceres"`, `"ceres-bfgs"`,
`"nlopt-slsqp"`, `"nlopt-bobyqa"`, `"nlopt-tnewton"`, `"nlopt-var2"`,
`"nlopt-lbfgs"`, `"nlopt-lbfgs-slsqp-fallback"`, `"ipopt"`, `"port"`,
`"port-nls"`. The R side passes the same
string through to a single Rcpp shim per fit family — no per-Backend wrapper
explosion. Solver tuning rides on a generic `control = list(max_iter, ftol,
gtol, history)` argument.

The R complete-data ML and LS helpers also expose box constraints through
`bounds = list(lower, upper)` or the named bound builders
`bounds_variance()` / `bounds_pos_var()`, `bounds_standard()`,
`bounds_wide()`, and `bounds_loading()`. The high-level `fit_model()` ML, ULS,
GLS, WLS, and ordinal LS paths thread the same bounds object into the C++
fit layer; FIML remains unbounded at the R surface.

Optimizer outputs carry function/gradient evaluation counts plus a refined
success status and final stationarity diagnostic. The C++ `OptimResult` and
`estimate::Estimates` report `OptimStatus` (`Converged`,
`LineSearchSalvaged`, `SingularConvergence`, or `Unknown`) and a final
(projected, when bounded) gradient infinity norm when the backend can compute
one. R fit lists expose these as `optimizer_status` and `grad_norm`; the
`converged` field projects the common verdict (TRUE/FALSE/NA), independently
of the optimizer stop. A returned estimate need not pass that verdict.


- `Backend::NloptLbfgs` is the complete-data scalar default. FIML defaults to
  `Backend::NloptLbfgsSlsqpFallback`, which retries NLopt SLSQP from the same
  start if L-BFGS fails or returns a non-clean optimizer status. NLopt is a
  required dependency in the ordinary build, so both pieces of the fallback are
  always available.
  Known limitation: the NLopt Luksan L-BFGS line search allows ten step
  reductions, which can be insufficient to reach a finite objective from a
  valid start when parameter scales produce a large gradient. This can abort
  after 12 evaluations regardless of the overall evaluation budget. The
  standalone variance probe in `cpp/tests/checks/nlopt_lbfgs_domain.c` isolates the
  issue; the domain-recovery task is tracked in `project/backlog/todo.md`.
- `Backend::Port` is the trust-region cross-check: vendored PORT (Bell Labs)
  `drmngb_` (TOMS 611 Dennis-Gay-Welsch model-Hessian trust region; the
  algorithm behind R's `nlminb`), supports bounds natively. Vendored at
  `cpp/third_party/port/` from AMPL/ASL + Fermi-LAT (both BSD-3, manifest in
  `cpp/third_party/port/README.md`). Replaces the previous CppNumericalSolvers
  `Backend::TrustRegion`.
- `Backend::PortNls` is the least-squares-shape counterpart: PORT `drn2gb_`
  (TOMS 573 NL2SOL adaptive trust region — the algorithm behind R's `nls`).
  Drives the multi-residual `GmmProblem` directly through reverse
  communication (alternating R/J requests), so NL2SOL sees the true
  Gauss-Newton-plus-secant model Hessian instead of the scalarised
  ½‖r‖² collapse. On non-convex SNLLS problems with multiple local
  optima (e.g., Bollen's democracy SEM under GLS) it may converge to a
  different basin than the gradient backends; the cross-check tests
  document this rather than enforce agreement.
- `Backend::Ceres` / `Backend::CeresBfgs` cover Ceres Levenberg-Marquardt and
  dense line-search BFGS on the least-squares path (build with
  `MAGMAAN_WITH_CERES=ON`).
- `Backend::Ipopt` is the optional system-IPOPT interior-point backend (build
  with `MAGMAAN_WITH_IPOPT=ON`). It is available as a general scalar optimizer
  and accepts nonlinear equality constraints. v1 uses IPOPT's limited-memory
  Hessian approximation, so magmaan supplies objective gradients and
  constraint Jacobians but not exact Lagrangian Hessians.
- `Backend::NloptSlsqp` exposes NLopt's SLSQP for ML, FIML, and LS scalar
  paths (gradient SQP, Kraft 1988). It accepts simple bounds and nonlinear
  equality constraints via analytic constraint Jacobians.
- `Backend::NloptBobyqa`, `Backend::NloptTnewton`, `Backend::NloptVar2`,
  `Backend::NloptLbfgs` round out the NLopt roster with distinct algorithm
  ideas: Powell 2009 derivative-free quadratic-model trust region (BOBYQA,
  finite bounds required); Nash 1985 preconditioned truncated Newton with
  CG inner solve (TNEWTON); Shanno-Phua 1980 full (dense) variable-metric
  BFGS (VAR2); and NLopt's own L-BFGS. All five share the one
  `NloptOptimizer` adapter parameterised over an opaque
  `NloptAlgorithm` enum.
- The local `ceres` and `ipopt` presets are optional optimizer comparison
  builds. Ceres is FetchContent-managed; IPOPT is deliberately a system
  dependency because its BLAS/LAPACK and sparse-linear-solver stack is a
  toolchain choice. `just r-install-ceres` and `just r-install-ipopt` mirror
  the relevant compile definition into the R shared object so the accepted
  optimizer strings are executable from the R dev surface in one install.

The opt-in installed-lavaan 0.7.2 simulation gate is
[`test_preset_simulation_parity.R`](../../../r-package/tests/testthat/test_preset_simulation_parity.R)
(`MAGMAAN_PARITY=1`). Its eight named cases cover complete ML, FIML and
all-ordinal delta/theta DWLS, including grouped equalities. The comparison is
per parameter, `abs(a-b) <= 1e-5 * (1 + max(abs(a), abs(b)))`, matching the
pinned fixture gate. The approved alternative for non-retry endpoints requires
identical convergence/verdicts, both maximum gradients <=1e-3 in lavaan units,
lavaan statistics (2N times the objective) agreeing within 1e-6 and maximum estimate difference divided
by the same-fit lavaan standard SE <=1e-3. The test evaluates the preset endpoint
with lavaan's objective/gradient in oracle coordinates and records every metric;
contract-route counts print per model. Rescaled retries keep the task-47 contract.
The installed-lavaan 0.7.2 gate passes all 160 replicates (about 35 seconds).
The split between routes varies slightly between builds because the drift
is rounding noise: in the merge run (2026-10-03) 157 pass estimates and three
PoliticalDemocracy seeds (590202, 590203, 590210) pass the endpoint
alternative, with chi-square differences at most 3.3e-8, gradients at most
1.7e-6 and SE-scaled differences at most 1.7e-4; all other models use zero
endpoint alternatives. All verdicts agree and no rescaled retry occurs. An
earlier 1e-9 relative objective condition proved build-sensitive: near an
optimum the objective gap is second order in the estimate difference, so it
is judged on the statistic's scale.
The default suite skips this opt-in gate.

Task-59's focused seed 590214 investigation temporarily recorded each preset
objective callback's coordinates/value/gradient and traced installed lavaan's
`lav_model_grad` coordinates, with `control = list(trace = 1)` for PORT output.
The temporary instrumentation was removed before the final build. Parameter
keys, native/optimizer starts (maximum difference 1.8e-15), unit parameter
scales and PORT controls agree; this model has no equality reduction. Aligning
oracle gradient evaluations with the preset's callback points (which also
include objective-only rejected trials), the first shared coordinates differ
by 1.8e-15, then 2.7e-15, 1.8e-14, 1.1e-13, 2.4e-12, 2.4e-11,
8.1e-10 and 2.4e-7. Evaluating lavaan at every finite preset callback point
gives maximum objective error 3.65e-15 and gradient error 6.28e-14. Invalid
non-PD trial points are excluded from this same-point derivative comparison,
not from the simulation gate. This localizes the initial divergence to
rounding, followed by amplification along the optimization path.

In the instrumented run, seed 590214 stopped after 80 versus 64 iterations;
its endpoint objectives differed by 4.36e-11 and maximum gradients were
1.30e-6 versus 7.41e-7. In the restored build, the five previously failing
seeds have starts within 6.3e-15, endpoint objective differences below
5.5e-11 and maximum gradients below 1.8e-6, comfortably within the preset's
0.001 acceptance threshold. Expected-information directional curvature along
the endpoint difference is 0.0134–0.0249, consistent with a weak direction.
The evidence supports floating-point search-path amplification rather than a
systematic objective/gradient or coordinate discrepancy. This evidence supports
the approved endpoint alternative above; the estimate tolerance remains unchanged
and no fitting code was changed.
Diagnostic scripts and logs are retained under `~/.cache/magmaan-logs/task-59-*`;
the trace scripts require the temporary callback instrumentation described above.

### lavaan marker component (complete-data ML)

`FittingOptions::marker` resolves to `default` or `lavaan-0.7.2`; the preset
selects the latter and explicit component overrides mark it modified. The
core item-rest correlation rule consumes named h1 covariance blocks, averages
signed correlations ignoring nonfinite entries and switches a weak first
marker below 0.1 to the first absolute maximum at least 0.1. Builder marker
maps select loading rows before numbering/constraints without reordering.
R replays the same fitting route on a rebuilt specification, then the original
if the switched fit fails its selected convergence rule. Both interfaces retain
the actual model and signed rounded switch information with a revert flag.
Inference/reporting consume the fitted partable; nested inference and
`refit_from_null()` refuse adapted marker coordinates until their mapping is
implemented. Casewise contrasts also refuse automatic adaptation because a
changed marker changes the parameter metric. Ordinary fitting calls rerun the
rule on each dataset.
FIML and DWLS preset reports currently resolve marker to `default`; explicit
adaptation on those routes and composite adaptation are unsupported.

### Unwired objective-coordinate scale primitive

`estimate::frontier::objective_coordinate_scale()` accepts sample units and
an actual scalar-objective GN diagonal in matching equality-reduced coordinates.
It clamps two-sided multipliers to [1e-6, 1e6], or downward-only multipliers to
[1e-3, 1], and retains sample units for invalid or degenerate diagonal entries.
Malformed units/sizes and nonfinite or nonpositive outputs are typed failures.
The helper does not compute NT information. TASK-33.10.5.4 owns subsequent fit
integration and opt-in selection; this additive primitive changes no defaults.
