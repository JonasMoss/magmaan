# Complete-data modular timing harness

`magmaan_timing_bench` times each component of a complete-data linear SEM fit
separately, over a crossed design of model structure x parameterization x p x n,
and writes tidy long CSV. It is the reusable harness for "which algorithm /
reparameterization / backend is faster, and where does the ranking flip".

No analysis happens in C++. Scaling slopes, arm ranking, and crossover detection
are downstream groupbys on the CSV.

## Why compiled

The R surface collapses starts + evaluator construction + optimization +
diagnostics into a single `estimate()` measurement, and exposes no way to
evaluate Sigma, the Jacobian, or the ML gradient at an arbitrary theta. The
interesting half of a modular profile is therefore not reachable from R at all.
Resolution is the second reason: `ev.sigma()` at p = 12 runs in under a
microsecond, where R's `proc.time()` quantisation and `.Call` overhead would both
exceed the quantity being measured.

## Build and run

```sh
cmake --preset opt -DMAGMAAN_BUILD_BENCH=ON
cmake --build --preset opt --target magmaan_timing_bench

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./build/opt/benchmarks/magmaan_timing_bench --p 12 --n 1000 --out /tmp/t.csv
```

Use `opt`, never `dev`: the `dev` preset carries AddressSanitizer and UBSan,
which make timing meaningless. `opt` is `-O3 -march=native`, so results are not
portable across machines — record the CPU alongside any saved run, and keep
outputs in the ignored `benchmarks/results/`.

```
--models   csv of structure ids (default: all)
--p        csv of p values      (default: 6,12,24,48,96)
--n        csv of n values      (default: 200,1000,5000,50000)
--params   csv of marker,std.lv,effect.coding (default: marker)
--reps N   timed reps per stage (default 15)
--warmups N                     (default 3)
--skip     csv substrings of stage names to omit
--isolate  time each stage alone instead of in a shared rotation
--seed N                        (default 20260918)
--list     print the catalog and exit
```

`--skip` matches substrings, so `--skip sigma` also drops `dsigma` and
`evaluate_sigma_and_jac`.

## The catalog

Seven structures. Variety at fixed p varies the things that actually drive cost:
npar, whether Beta is present, Theta density, and conditioning.

| id | structure | isolates | p |
|---|---|---|---|
| `cfa_1f` | one factor | low npar, diagonal Theta, no Beta | 6, 12, 24, 48, 96 |
| `cfa_3f` | three correlated factors | the reference case | 6, 12, 24, 48, 96 |
| `soc_2nd` | second-order factor over three first-order | two levels of latent structure | 12, 24, 48, 96 |
| `sem_2x2` | two exogenous + two endogenous factors | Beta present | 12, 24, 48, 96 |
| `cfa_3f_rescov` | `cfa_3f` + residual covariances | Theta off-diagonal | 12 |
| `path_recursive` | all-observed recursive chain | Lambda = I, Beta only | 12 |
| `cfa_6f_2ind` | six factors, two indicators each | near-underidentified, ill-conditioned | 12 |

The population covariance for each is constructed in closed form in
`catalog.cpp` from the same Lambda / Beta / Psi / Theta that generated the
syntax, so Sigma = Sigma(theta_0) exactly and the fitted model is correctly
specified by construction. This matters because a misspecified population
changes iteration counts, and iteration count is one of the two things being
measured. All populations are standardized to unit observed variances.

Data are drawn with `sim::simulate_normal_raw`, always outside every timer.

Parameterization is a change of coordinates, not a change of model: marker
(`auto_fix_first`), `std.lv`, and `effect.coding` all describe the *same*
population, so an arm comparison across them is clean. `effect.coding` is the
interesting one — it adds a linear equality row per latent, so it is the only
variant that exercises `build_eq_constraints` and the constraint
reparameterization. Latent-scaling variants are skipped for `path_recursive`,
which has no latent variables.

## Stages

| group | stages |
|---|---|
| A. spec | `parse`, `build`, `matrix_rep` |
| B. data | `sample_stats` |
| C. problem | `evaluator_build`, `starts_simple`, `starts_fabin3`, `bounds_standard`, `constraints_eq`, `constraints_nl`, `objective_ml_build`, `weight_nt_build`, `objective_gls_build`, `objective_gls_trace_build` |
| D. per-iteration | `sigma`, `dsigma`, `evaluate_sigma_and_jac`, `ml_value`, `ml_gradient`, `ml_value_gradient`, `objective_call` |
| E. optimize | `optimize_lbfgs`, `fit_ml_end_to_end` |
| F. post-fit | `info_expected`, `info_observed_analytic`, `info_observed_fd`, `info_cross_products`, `vcov`, `se`, `chi2_stat`, `df_stat`, `baseline_chi2`, `fit_measures`, `fit_extras` |

Two derived rows are appended per cell:

* `staged_fit_ml` — the sum of stage medians that reconstructs one `fit_ml`
  call (`evaluator_build` + `constraints_eq` + `constraints_nl` +
  `objective_ml_build` + `optimize_lbfgs`). Its gap against the separately timed
  `fit_ml_end_to_end` prices the terminal audit, the fit diagnostics, and
  whatever the composite path shares that the staged path rebuilds. Reporting
  that gap is the point; it replaces the older "component timers are not an
  exact additive profile" disclaimer with a number.
* `pipeline_total` — one full user-facing analysis, syntax in to fit indices out.

Either row is omitted entirely when `--skip` removed one of its members, so a
skip cannot silently deflate a reported total.

`optimize_lbfgs` is reported alongside `fit_f_evals` / `fit_g_evals` from the
reference fit. This is load-bearing: a reparameterization or backend can win by
making iterations cheaper or by needing fewer of them, and those are entirely
different claims. NLopt's gradient algorithms do not populate
`OptimResult::iterations`, so the evaluation counts are the usable measure.

## The n axis

Under complete data, n and p hit disjoint stage sets. Everything downstream of
`sample_stats` operates on S (p x p) and theta, where n enters only as a scalar
multiplier. Exactly two stages here are O(n): `sample_stats` (O(n p^2)) and
`info_cross_products` (O(n npar), via casewise scores). Every other stage should
be flat in n.

That is a design consequence and also a free validity check on this harness: if
`optimize_lbfgs` moves with n at fixed p, the measurement is wrong. It also
means a full n x p crossing is mostly wasted — cross n only where it bites.

It does buy one genuine crossover: at large n and small p, data reduction
dominates the whole analysis; at small n and large p, optimization does.

## Timer

`timing.hpp` is the shared timer, extracted from the per-file copy that had
grown independently in six places. Three things it adds:

* **Batch auto-calibration.** Stages spanning nanoseconds to hundreds of
  milliseconds are each calibrated to a batch size putting one timed sample near
  200 us. Reported figures are ns per call.
* **Arm rotation.** Reps are the outer loop, arms the inner one, rotated by rep
  index. Running all reps of A then all reps of B attributes turbo decay and
  cache warmth to whichever ran last, which is precisely the bias that matters
  when the question is which is faster.
* **Checksum returns.** At `-O3` a timed call whose result is unused is
  legitimately dead code. Every thunk returns a double derived from its result,
  sunk through an asm barrier; comparing checksums across arms is also a free
  tripwire for "these arms did not compute the same thing".

Medians, not means: a stolen scheduler slice is a one-sided contaminant the mean
has no defence against.

### Rotation compares arms, not stages

Rotation cancels *time-ordered* drift, so two arms doing comparable work are
compared fairly. It does **not** make arms independent of one another. A stage
allocating a multi-megabyte buffer (a q x n_free Jacobian is ~7 MB at p = 96)
interacts with its neighbours through the allocator's mmap threshold and the page
cache, so its absolute median depends on what else is in the arm set.

Measured on this harness: `dsigma` read **3.6x** `evaluate_sigma_and_jac` inside
the full 30-arm rotation, **1.0x** under `--isolate` — and the two call the same
`current_dsigma_dtheta()` to produce the same matrix, so the true ratio is 1.
The same p = 96 cell gave 875 us per stage in a 4-arm rotation and 370 us
isolated.

So:

* **Comparing arms of one stage** (marker vs std.lv vs effect.coding on
  `optimize_lbfgs`; backend against backend) — use the default rotation. Both
  arms allocate alike, and this is the bias rotation exists to remove.
* **Comparing different stages to each other**, or quoting an absolute number
  that must stand alone — use `--isolate`.

Scaling slopes and n-sensitivity ratios are within-stage, so they are safe either
way. Cross-stage ratios from a rotated run are not.

Timings are advisory. Correctness gates timing, never the reverse, and no
performance ratio is ever a CI assertion.
