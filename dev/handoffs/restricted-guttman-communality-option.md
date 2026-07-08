# Handoff: expose `CommunalityMethod` on the restricted Guttman map

**For:** the core agent (owns `src/`, `include/`, `r-package/`).
**Status:** follow-on to `3591e7e` (residual-restricted Guttman map). Small,
plumbing-only. No new math.
**Hands off:** do not edit `docs/**` or `papers/guttman-inference/**`.

## Goal

`fit_noniterative_cfa_restricted` currently hardcodes the communality estimator
to `CommunalityMethod::GmmBlock` (`src/estimate/frontier/noniterative_cfa.cpp:646`).
Thread a `CommunalityMethod comm` argument through the restricted map and its
inference so the caller can pick any of the **four least-squares-form** methods.
Default `GmmBlock` so every existing call and test is unchanged.

## Why (the point, not just tidiness)

The constrained solve appends a linear row `R_D h2 = r_D` to the communality
least-squares system and projects in the `G = A' W A` metric. That is exact and
closed form for any **LS-form** communality method, and `communality_system`
(`communality.cpp`, the `switch`) already accepts all four:
`TriadLeastSquares`, `AnchorTriadLeastSquares`, `GmmBlock`, `GmmFull`. It errors
on `AverageRatio` / `RatioOfSums` (nonlinear per-item ratios, no constraint row
to append).

These four are **different estimators**, not cosmetic variants:

- `TriadLeastSquares` / `AnchorTriadLeastSquares` use unit weight (`W = I`). This
  is the **criterion-metric** estimator the paper is built around
  (`W_c = Pi (x) V`, fourth-moment-free). The paper currently cannot instantiate
  it through the restricted map.
- `GmmBlock` / `GmmFull` use the GMM-efficient triad-moment weight
  (`normal_cor_gamma_pairs`). `GmmBlock` is what the aligned configural map uses.

So the knob lets the restricted map produce both the criterion estimator and the
GMM-efficient one, which is exactly the estimator-options comparison we want.

**Not in scope:** `AverageRatio` / `RatioOfSums` (they are not LS-form, so they
cannot carry constraints) and the `NonIterativeEstimator::Guttman` loading lane
(its communalities come from `theta_spearman` = average-of-ratios, again not
LS-form). The `which == GuttmanGlsAligned` gate stays. This task only opens the
`CommunalityMethod` axis, not the loading-lane axis.

## Steps

**1. `fit_restricted_blocks`** (`noniterative_cfa.cpp:626`, static). Add a
`CommunalityMethod comm` parameter and pass it to the `communality_system(...)`
call at `:646` in place of the hardcoded `CommunalityMethod::GmmBlock`.

**2. `fit_noniterative_cfa_restricted`** (`.hpp:95`, `.cpp:1039`). Add
`CommunalityMethod comm = CommunalityMethod::GmmBlock` as the last parameter.
Pass it to `fit_restricted_blocks`.

- **Upfront guard.** Before doing work, reject `AverageRatio` / `RatioOfSums`
  with a clear message ("restricted Guttman CFA: communality method must be
  least-squares-form (triad_ls, anchor_triad_ls, gmm_block, gmm_full)").
  `communality_system` already errors, but an upfront guard names the axis.
- **No-constraint fast path.** The early `return fit_noniterative_cfa(...)` when
  there are no rows currently ignores `comm` and always returns the GmmBlock
  configural aligned fit. Gate that fast path on `comm == GmmBlock`. For any
  other `comm`, fall through to `fit_restricted_blocks` with empty `R_D`/`R_load`
  so a no-constraint fit still honors the requested metric (it returns the
  configural fit *in that metric*). This keeps existing test (a) exact and makes
  the knob consistent with and without constraints.

**3. Jacobian** (`.hpp:132`/`:143`, `.cpp`). Add the same
`CommunalityMethod comm = GmmBlock` parameter to
`estimator_map_jacobian_restricted_block` and `estimator_map_jacobian_restricted`,
and pass it into every `fit_noniterative_cfa_restricted(...)` perturbation call.
The FD map must use the same `comm` as the fit, or the Jacobian will not match
the estimate.

**4. Grouped restricted inference** (`robust/frontier/noniterative_inference.hpp:197`/
`:207`/`:216`, `.cpp`). The three `noniterative_inference_grouped_restricted*`
functions build the restricted Jacobian internally, so they need the same
`CommunalityMethod comm = GmmBlock` parameter threaded into their
`estimator_map_jacobian_restricted*` call. Default `GmmBlock` keeps current
callers and the existing inference test unchanged.

**5. R binding.**

- `noniterative_cfa_restricted_fit_impl` (`r-package/src/fit.cpp:7979`): add a
  `std::string communality = "gmm_block"` argument, resolve it with the existing
  `communality_which()` mapper (`fit.cpp:7812`), pass to
  `fit_noniterative_cfa_restricted`. Stamp the resolved name onto the returned
  fit list via `communality_method_name(comm)` (mirror how the estimator string
  is stored).
- Read it back for inference. Add a `noniter_comm_for_fit(fit)` helper mirroring
  `noniter_which_for_fit` (`fit.cpp:7800`): if the fit carries a `communality`
  field use it, else default `GmmBlock`. In the restricted branch of the grouped
  inference dispatch (`fit.cpp:8062` area) and the nested-test dispatch, pass
  that `comm` through so the recomputed Jacobian matches the fit. This keeps the
  R inference call signatures stable (they read `comm` off the fit, exactly like
  they already read the estimator).
- `fit_noniterative_cfa_restricted` R wrapper (`r-package/R/noniterative.R:92`):
  add `communality = "gmm_block"` and forward it. Regen `RcppExports`
  (`just vendor` runs `compileAttributes`, or `Rcpp::compileAttributes()`).

## Tests (add to `tests/unit/`, ASan `dev`)

- **knob bites off-model:** on a mildly misspecified population, a residual-tied
  restricted fit under `TriadLeastSquares` differs from one under `GmmBlock`
  (theta differs by > 1e-6), and **both** satisfy `A_eq theta = b_eq` to 1e-9.
- **no-constraint honors the metric:** `comm = TriadLeastSquares` with no rows
  equals the unit-weight configural aligned fit (falls through the fast path);
  `comm = GmmBlock` with no rows still equals the configural fit (existing test
  (a), now via the gated fast path, unchanged).
- **non-LS methods error:** `AverageRatio` and `RatioOfSums` each return an error
  whose detail names the communality axis.
- existing D/O guards (mixed `lambda = eps`, factor-variance rows) still error
  under any `comm`.

## Build / verify

```
cmake --build --preset dev && ctest --preset dev
cmake --build --preset opt && ctest --preset opt
just vendor && just r-dev            # R smoke: fit with communality="triad_ls" vs "gmm_block"
```

## Scope guards

- Default is `GmmBlock` at every layer. No existing signature loses a caller; no
  existing test changes.
- Only the four LS-form methods are reachable. Do not try to make the plain
  `Guttman` lane or the ratio communalities carry constraints.
- The stored-on-fit `communality` field is the single source of truth for the
  inference read-back, so a fit and its SEs/GOF always use one method.
