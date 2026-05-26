# SNLLS fast α-solve: Cholesky-on-normal-equations with rcond fallback

## Motivation

Profiling a 280-model SEM corpus
(`papers/snlls-continuous/dev/inspect/profile-share-by-corpus.qmd`)
surfaced an overhead bimodality at 100% profile share — models where
the Golub–Pereyra classifier puts every free parameter in the α
block. SNLLS at 100% profile share is one closed-form linear solve.
On small / well-identified models (growth, fixed-loading CFAs) the
fixed cost of that solve loses to ordinary L-BFGS hitting convergence
on its start-point gradient check
(e.g. `newsom_2015_ex3_1c`: Full/SNLLS ≈ 0.23). On larger / harder
problems the closed-form crushes Full
(`newsom_2015_ex12_1a`: 25–31×).

Before this change the inner α-solve at
`src/estimate/gmm/gp.cpp::profile_at()` was unconditionally a
rank-revealing `Eigen::ColPivHouseholderQR` on the residual Jacobian
J. QR is the safe default — it handles arbitrary rank and ill-
conditioning — but pays a constant factor that the closed-form-friendly
regime can't absorb.

Cholesky on the normal-equations Gram `A = JᵀJ` is roughly 2× cheaper
than QR for n ≫ p but loses guarantees as `A` becomes ill-conditioned:
Higham (*Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
Thm 20.4) bounds the relative error of the Cholesky-NE solution by
`O(u · κ₂(J)²)` vs QR's `O(u · κ₂(J))`. For a well-conditioned `A`
the loss is < 1 digit and the speedup compounds — once per outer
optimizer iteration of every SNLLS fit, not only at 100% profile share.

A gated fast path gives the speedup where it's safe and falls back to
the existing QR otherwise.

## Algorithm

In `profile_at()`:

1. Form `A = JᵀJ` and rhs `b = Jᵀ·(−r₀)`.
2. Attempt `Eigen::LLT<MatrixXd>` factorization of `A`. PD failure →
   fallback to QR.
3. If LLT succeeded, estimate `rcond(A)` via `rcond_pocon_sym` (file-
   local helper, Hager 1-norm estimator on the existing factor).
4. If `rcond(A) > kFastSolveThreshold` (= `1e-7`), solve via the LLT
   factor — both for α̂ and for the β-gauge back-solve. Set
   `ProfilePoint::used_fast_solve = true`.
5. Otherwise construct `ColPivHouseholderQR<MatrixXd>(J)` and use it
   for both solves (today's path).

Both consumers (α̂ at `gp.cpp:227-228` and the gauge term at the
following block) **must** see the same inverse, so the dispatch
branches a single `bool used_fast` decision through both.

## Threshold derivation

Higham Thm 20.4: the relative error of Cholesky-on-NE is bounded by
`c(n,p) · u · κ₂(A)` where `A = JᵀJ` and `u` is unit roundoff. QR's
bound is `c(n,p) · u · κ₂(J) = c(n,p) · u · √κ₂(A)`. Cholesky loses
one digit relative to QR when

```
u · κ₂(A) ≈ u · √κ₂(A)   ⟺   κ₂(A) ≈ 1/√u   ⟺   rcond(A) ≈ √u
```

For double precision `u = 2⁻⁵² ≈ 2.22·10⁻¹⁶` and `√u ≈ 1.49·10⁻⁸`.
The gate at `1e-7` keeps us strictly above this "loses ≤ 1 digit"
floor and adds ~6× headroom to absorb:

- Hager's optimistic bias (his iterate is a *lower* bound on
  `‖A⁻¹‖₁`, so `rcond_pocon_sym` is an *upper* bound on the true
  rcond — it can pass when truth is worse).
- Any residual amplification from forming the explicit product `JᵀJ`.

The gate is a `constexpr`, not a runtime tunable. v1 ships one
defensive value; revisit only if downstream evidence warrants.

## `rcond_pocon_sym` (Hager 1-norm estimator)

LAPACK `?POCON`-style 1-norm rcond, specialized for a symmetric-PD
matrix whose Cholesky factor is already in hand. See Higham §15.2,
Algorithm 15.4.

- Cost: at most 6 triangular back-solves on the existing factor =
  `O(p²)`. Negligible against the `O(n·p²)` already spent forming
  `A = JᵀJ`.
- Returns 0.0 on any numeric pathology (non-finite intermediate,
  zero norm), forcing the caller to fall back.
- For symmetric `A`, `A⁻ᵀ = A⁻¹`, so one Cholesky factor covers both
  back-solves Hager's algorithm needs.

**Conservative-bias note.** Hager's iterate is a lower bound on
`‖A⁻¹‖₁`. So `rcond_pocon_sym = 1/(‖A‖₁·γ̂)` upper-bounds the true
`1/(‖A‖₁·‖A⁻¹‖₁)`. In the gate `rcond_pocon_sym > 1e-7` this means
we can pass *and* be wrong — but only by the polishing-step's
typical 2-5× underestimation, which the 6× headroom above `√u`
absorbs.

## Telemetry

Two `std::int32_t` counters on the GP cache, surfaced through
`GpProblem` (live `shared_ptr<const std::int32_t>`) into `Estimates`
and out through the R wrapper:

- `Estimates::n_alpha_solve_fast`
- `Estimates::n_alpha_solve_fallback`

Both default to `-1` (sentinel "no SNLLS path applies"), matching the
existing `n_nonlinear` / `n_linear` convention. The R wrapper at
`r-package/src/fit.cpp` maps negatives to `NA_INTEGER`.

Increment semantics (in the `gp_impl` closure):

- `(fast, fallback) = (0, 0)` *and* SNLLS path taken → every α-solve
  was a closed-form short-circuit (`n_alpha == 0`). No actual solve
  happened — neither counter bumped.
- `fast > 0 ∧ fallback == 0` → every cache miss took the fast path.
- `fast > 0 ∧ fallback > 0` → mixed; the optimizer visited at least
  one β where the Gram was near-singular.
- `fast == 0 ∧ fallback > 0` → every cache miss fell back. The model
  is structurally ill-conditioned in α.

Cache hits do not bump the counters — they re-use a previously
computed `ProfilePoint`.

## Non-goals (won't help / out of scope)

- **Tiny well-identified models where L-BFGS converges in 0–1 outer
  iterations.** Their SNLLS overhead is dominated by *structural*
  setup (classification, K_α / K_β construction), not by the per-
  iteration solve. The `newsom_2015_ex3_1c` regression that
  motivated the investigation lives in that regime and will not be
  moved by this change. Document expectation: the corpus speed
  distribution's right edge tightens (mixed/fallback rows move into
  fast-only); the left edge does not.
- **Runtime tuning of the threshold.** v1 ships a `constexpr`. If
  later evidence shows the gate is wrong, revisit globally rather
  than parameterizing every call.
- **Multi-threaded inner solves.** The cache is single-shot per fit;
  parallelism, if any, belongs at the outer-optimizer level.
- **Replacing the QR fallback with something else (SVD, LSQR, ...).**
  QR remains the only fallback. The fast path is a strict addition.

## Verification

Doctests in `tests/unit/snlls_test.cpp`:

- Well-conditioned 1F covariance → `n_alpha_solve_fast > 0`,
  `n_alpha_solve_fallback == 0`, `fmin < 1e-10`.
- 2F CFA with synthetic population-implied covariance → fast path
  fires; counters surface; fit converges.
- 5-indicator 1F against a rank-near-1 sample covariance → the rcond
  gate trips at least one fallback; fit still succeeds because QR's
  column pivoting handles near-rank deficiency.
- Full-θ paths (`fit_ml`) → counters stay at sentinel `-1`.

```sh
just test-area estimate snlls
just test-fast
just test-dev  # ASan + UBSan
```

For paper-side smoke (optional):

```sh
cd papers/snlls-continuous
SNLLS_SURVEY_TIMES=3 Rscript scripts/run_corpus_speed_survey.R
```

The raw CSV's new `n_alpha_solve_fast` and `n_alpha_solve_fallback`
columns let the Layout 7 prototype quantify the fast-path hit rate
across the 280-model corpus.
