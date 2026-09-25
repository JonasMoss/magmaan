# SNLLS fast α-solve: Cholesky-on-normal-equations with rcond fallback

**Status:** implemented. This is the rationale for shipped behavior, not a
proposal; the gate lives at `cpp/src/estimate/gmm/gp.cpp` and is guarded by
`cpp/tests/unit/snlls_test.cpp`. The threshold is an empirical screen, supplemented by a direct inner
normal-residual check; neither guarantees a bound on forward error.

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
(`newsom_2015_ex12_1a`, a second-edition Newsom script no longer in the
first-edition corpus: 25–31×).

Before this change the inner α-solve at
`cpp/src/estimate/gmm/gp.cpp::profile_at()` was unconditionally a
rank-revealing `Eigen::ColPivHouseholderQR` on the residual Jacobian
J. QR is the safe default — it handles arbitrary rank and ill-
conditioning — but pays a constant factor that the closed-form-friendly
regime can't absorb.

Cholesky on the normal-equations Gram `A = JᵀJ` can reduce factorization
cost on well-conditioned designs. Forming the Gram squares the 2-norm
condition number. General least-squares forward error also depends on
residual geometry, so there is no unconditional digits-lost comparison with
QR. The fast path is screened empirically and checked against the original
inner design; rank-revealing QR remains the fallback.

## Algorithm

In `profile_at()`:

1. Form `A = JᵀJ`, factor it with LLT, and estimate its reciprocal 1-norm
   condition number with `rcond_pocon_sym`.
2. If LLT succeeds and the estimate exceeds `fast_solve_threshold = 1e-7`,
   compute a candidate alpha with the factor.
3. Recompute `r = r0 + J alpha` and `Jᵀ r` using the original design. Accept
   the candidate only if it is finite and
   `||Jᵀ r||₂ <= 64 epsilon max(rows(J), cols(J)) ||J||F
   (||r0||₂ + ||J||F ||alpha||₂)`, with finite scale and residual norm.
4. If either screen fails, solve with column-pivoted Householder QR of J.
   Return an error if the resulting parameters or residual are non-finite.
5. Reuse the accepted alpha factorization for the nonlinear-column projection.
   The fast/fallback counter records the accepted solve, not the LLT attempt.

The normal-residual test is a scale-aware stationarity check. It is not a
forward-error bound and cannot certify a unique alpha at deficient rank.
QR may choose a different representative from SVD there; the invariant
comparison is the fitted residual. Rank changes can also break smoothness
of the profiled objective.

## Threshold rationale and correction

The `1e-7` threshold is retained as an empirical condition screen; it is not
a proved accuracy guarantee. The previous note's claimed derivation of a
one-digit-loss threshold was invalid: `u*kappa(A) = u*sqrt(kappa(A))` gives
`kappa(A)=1`, not `1/sqrt(u)`. Nor do the general LS forward-error bounds
reduce to those two expressions without additional conditions. For IEEE
binary64 round-to-nearest, machine epsilon is `2^-52` and unit roundoff is
`2^-53`; they must not be interchanged.

The reciprocal-condition estimator may be optimistic. No universal factor
of six bounds that optimism, and the old headroom argument did not prove
otherwise. The direct normal-residual screen complements the condition
estimate without turning either into a forward-error guarantee. Tests
compare against SVD on well-conditioned, nearly dependent, exactly dependent,
and globally rescaled designs. These cover a finite regression grid, not
all possible SEM designs or conditioning regimes.

## `rcond_pocon_sym` (Hager 1-norm estimator)

The helper estimates `1/(||A||₁ ||A^-1||₁)` using the existing symmetric-PD
factor. Its five-iteration limit permits ten LLT solves plus one alternating
ramp polishing solve: at most eleven LLT solves, or twenty-two triangular
solves. Its cost is O(a²) for a alpha coordinates, versus O(m a²) to form
JᵀJ for m residuals. Non-finite/zero-norm results return zero and select QR.
The norm estimate has an optimistic reciprocal-condition bias in exact
arithmetic; finite precision gives no certified enclosure.

## Derivative contract

The GP residual callbacks provide Kaufman's projected Jacobian
`(I-P_J) J_beta`, evaluated at the profiled parameters. At locally fixed
full column rank, the exact residual derivative also contains columns
`-J (JᵀJ)^-1 (dJ/dbeta_j)ᵀ r`. These lie in the range of J and are
orthogonal to r when the inner solve is accurate, so the scalar gradient
is exact while the residual Jacobian is approximate. PORT-NLS and Ceres
therefore receive approximate Gauss–Newton curvature. The public GP header
states this contract; the tests separately check the scalar gradient and a
nonzero-residual case where the residual derivative differs.

Ordinary `fit_snlls` and `fit_snlls_gls` also recompute the original
full-model LS gradient for `diagnostics.geometric_stationarity`, including
eliminated alpha coordinates and covariance-cone diagnostics. This additive
audit does not replace the optimizer's driven-coordinate `audit` or change
its convergence flag. A bound-constrained inner solve is not implemented;
the continuous R SNLLS primitives reject supplied bounds.

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

- `(fast, fallback) = (0, 0)` means no successful profile evaluation has
  been counted yet. The public GP classifier requires a nonempty alpha block.
  An all-linear model has zero beta coordinates, but still performs and
  counts its inner solve.
- `fast > 0 ∧ fallback == 0` → every cache miss took the fast path.
- `fast > 0 ∧ fallback > 0` → mixed; at least one candidate failed
  the condition or direct normal-residual screen.
- `fast == 0 ∧ fallback > 0` → every cache miss used QR. This
  alone does not establish structural ill-conditioning.

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

Doctests in `cpp/tests/unit/snlls_test.cpp`:

- Well-conditioned 1F covariance → `n_alpha_solve_fast > 0`,
  `n_alpha_solve_fallback == 0`, `fmin < 1e-10`.
- 2F CFA with synthetic population-implied covariance → fast path
  fires; counters surface; fit converges.
- 5-indicator 1F against a rank-near-1 sample covariance → the rcond
  gate trips at least one fallback; fit still succeeds because QR's
  column pivoting handles near-rank deficiency.
- Full-θ paths (`fit_ml`) → counters stay at sentinel `-1`.

```sh
just test-area estimate '*SNLLS*'
cmake --build cpp/build/opt --target magmaan_test_estimate
cpp/build/opt/tests/magmaan_test_estimate --test-case='*SNLLS*'
```

For paper-side smoke (optional):

```sh
cd papers/snlls-continuous
SNLLS_SURVEY_TIMES=3 Rscript scripts/run_corpus_speed_survey.R
```

The raw CSV's new `n_alpha_solve_fast` and `n_alpha_solve_fallback`
columns let the Layout 7 prototype quantify the fast-path hit rate
across the 280-model corpus.
