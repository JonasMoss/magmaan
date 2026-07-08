# Handoff: residual-constrained Guttman CFA map (`tau_H^R`)

**For:** the core agent (owns `src/`, `include/`, `r-package/`).
**Status:** greenlit for implementation. Theory settled and verified against the
current code.
**Hands off:** do not edit `docs/**` or `papers/guttman-inference/**` for this
task. Those are current and owned by the other agent. This doc is self-contained
on the theory, so you do not need the paper.

## Goal

Add a **Sigma-only, closed-form** estimator that imposes linear equality
constraints on **residual variances** (and loadings) at estimation time, using
no fourth moments. It extends the composite-regression lane from loadings to
residuals.

It is distinct from two things that already exist and must keep working:

- `fit_noniterative_cfa_metric` (multi-group metric via rank-one SVD): a
  different, cross-group loading-shape estimator.
- `noniterative_cfa_constrained` (`robust::frontier`): the `Omega`-metric
  inference projection `theta_tilde = theta_hat - Omega R'(R Omega R')^-1(...)`,
  which uses `Gamma`. That is the *test* object, not this.

## Notation (from the paper's notation contract)

- `tau_H^R` = restricted reduced-matrix map (this task).
- Criterion metric `W_c = Pi (x) V`, a function of Sigma only. **No `Gamma`,
  no `Omega`, no fourth moments anywhere in this map.**
- `h2` = correlation-scale communalities (what the triad system solves for).
- `Theta` = residual variances. `H` = reduced matrix (diag replaced by
  communalities). `Pi` = composite correlation. `Phi` = latent covariance.

## Theory (self-contained)

CFA: `Sigma = Lambda Phi Lambda' + Theta`, `Theta` diagonal. The Guttman map
estimates communalities `h`, forms the reduced matrix `H` (diagonal replaced by
communalities), extracts `(Lambda, Phi)` from `H`, and takes `Theta` as the
leftover diagonal.

Residuals sit **outside** the composite (loading) criterion, because the
diagonal split `H` is fixed at the communality step, upstream of the loading
regression. The hook that brings them in:

```
Theta_i = S_ii - h_i = S_ii (1 - h2_i)        (h_i = S_ii * h2_i)
```

so every linear residual constraint is **linear in the communalities `h2`**:

- equal residuals `Theta_i = Theta_j`  <=>  `S_ii h2_i - S_jj h2_j = S_ii - S_jj`
- fixed residual `Theta_i = c`          <=>  `h2_i = 1 - c/S_ii`

**Route 1 (this task):** impose the residual constraints on the communality
least squares, which is already a joint linear system, then extract loadings
with the loading constraints. Closed form, one sequential pass.

**D/O separability rule.** Partition the parameters into `D` = communalities /
residuals and `O` = loadings. Every constraint row must lie entirely in `D` or
entirely in `O`. A row that couples a residual with a loading (`lambda = eps`)
crosses the boundary and is unsupported. Factor-(co)variance rows are also
unsupported here (`Pi` is fixed by `H` in this lane). Both families may coexist,
just not in one row.

**Sequential, not joint.** `D` is solved first (constrained communalities), then
`O` is extracted from the resulting `H`. Exact on the model, a sequential
approximation off it. The fully joint version (restricted ULS, `tau_U^R`) is the
documented fallback and is **not** part of this task.

Full write-up (optional reading):
`papers/guttman-inference/dev/notes/residual_constrained_guttman_proposal.tex`.

## Why it fits the current code (verified)

`triad_system` (`src/estimate/frontier/communality.cpp:294`) already builds a
joint linear system `A h2 = b`, one row per triad, with `A(row,i)=R(j,k)`,
`b(row)=R(i,j)R(i,k)`. Every row touches only column `i`, so `A'A` is diagonal,
and `solve_gmm(A, b, W=I)` (`:386`) reproduces the per-item extended-triad-LS
formula exactly. The canonical communality estimator **is** this joint LS.
Residual constraints are just extra rows on it. No new solver family is needed.

## Implementation steps

**1. Make extended-triad-LS a joint system.** `triad_system` currently emits
within-block triads only. Add the cross-block anchor rows that
`anchor_triad_ls_h2` (`:210`) uses: for base pair `(i,j)` with `f(j)=f(i)` and
third item `k` with `f(k)!=f(i)`, emit `A(row,i)=R(j,k)`, `b(row)=R(i,j)R(i,k)`.
Bump `n_rows` by the anchor count. Verify `solve_gmm(A_ext, b_ext, I)` equals
`anchor_triad_ls_h2` to roundoff, then that joint form is the constraint-ready
communality estimator.

**2. Constrained solve.** Add `solve_gmm_constrained(A, b, W, R_D, r_D)`,
reusing `symmetric_pinv` (`:120`). Same projection form as the existing
constrained fit, but in the criterion metric `G = A'WA` (so no `Gamma`):

```
G    = A' W A;
Ginv = symmetric_pinv(G);
h    = Ginv * (A' W b);                 // unconstrained h2  (== current solve_gmm)
if R_D has no rows: return h;
GRt  = Ginv * R_D';                     // p x k
Ssm  = R_D * GRt;                       // k x k = R_D Ginv R_D'   (guard PD / rank)
h   -= GRt * (symmetric_pinv(Ssm) * (R_D * h - r_D));
return h;
```

**3. Build `R_D, r_D` from the partable.** From `build_eq_constraints(pt)` take
the equality rows, keep those on residual parameters, and map each into
`h2`-space via `Theta_i = S_ii(1 - h2_i)` (see the two equivalences above).
Column index = the observed-variable index of that residual parameter.

**4. D/O partition + guard.** Split the equality rows: `D` = rows touching only
residual params, `O` = rows touching only loading params. **Error** if any
single row mixes a residual and a loading (`lambda = eps`), and error on
factor-(co)variance rows. This is the D/O separability rule.

**5. Assemble the map.** New `fit_noniterative_cfa_restricted` in
`src/estimate/frontier/noniterative_cfa.cpp` (`.hpp`), a sibling of
`fit_noniterative_cfa_metric`:
   1. constrained `h2` (steps 1-3) -> `H`;
   2. extract `(Lambda, Phi)` from `H`, then apply the `O`-block loading
      constraints with the composite projection
      `l_R = l_G - W_c^-1 R_O'(R_O W_c^-1 R_O')^-1 (R_O l_G - r_O)`,
      `W_c = Pi (x) V`;
   3. `Theta = diag(S) - h`; report `Sigma = Lambda Phi Lambda' + Theta`.
   Sequential: `D` first, then `O`.

**6. Inference.** The map changed, so its finite-difference Jacobian changes.
Reuse the `estimator_map_jacobian_block` FD pattern over the new map, so SEs,
GOF, and Wald carry over unchanged in form.

**7. R binding.** Add `fit_noniterative_cfa_restricted` wrapper mirroring
`fit_noniterative_cfa_metric` in `r-package/R/noniterative.R` +
`r-package/src/fit.cpp`, regen `RcppExports`.

## Tests

`tests/unit/` (ASan `dev` build), fixture pattern of the existing noniterative
tests:

- (a) no constraints == configural fit;
- (b) `Theta_i = Theta_j` yields equal residuals, exact on an equal-uniqueness
  population;
- (c) `solve_gmm(A_ext, b_ext, I)` == `anchor_triad_ls_h2` (the joint-form
  sanity);
- (d) a mixed `lambda = eps` row errors;
- (e) a factor-variance row errors;
- (f) marker-chart invariance of the residual-constrained implied covariance
  (exp-56 style: residuals and communalities are chart-invariant, so the fitted
  common covariance should be marker-invariant at roundoff);
- (g) where a comparable lavaan fit exists (equal uniquenesses / parallel
  items), spot-check the point estimate to tolerance.

## Build / verify loop

```
cmake --build --preset dev && ctest --preset dev     # ASan
cmake --build --preset opt && ctest --preset opt
just vendor && just r-dev                            # then an R smoke of the wrapper
```

## Pitfalls / scope guards

- Do not break `guttman_block`, `guttman_gls_aligned_block`,
  `fit_noniterative_cfa_metric`, or `noniterative_cfa_constrained`. This is a new
  sibling map, not a change to those.
- Heywood: a residual tie can push `h2` out of `[0,1]` (negative `Theta`).
  Decide a bounds policy (clamp or active set) and fail-closed or flag, matching
  the configural map's stance. Do not silently return an improper fit.
- Report `Theta = diag(S) - h` (the communality split). It equals the fit
  leftover only on the model; document that in the function comment.
- Keep everything `Sigma`-only. If you find yourself needing a `Gamma` or a
  discrepancy/gamma argument, you have drifted into the inference lane.

## Optional companion (owned by the docs/experiments agent, not you)

An `experiments/57` probe comparing `tau_H^R` (sequential) against restricted
ULS (joint) on one misspecified population, to size the split-vs-leftover and
sequential-vs-joint gaps. That validates the sequential approximation. It can
land in parallel and does not block the core work.
