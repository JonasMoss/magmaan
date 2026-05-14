## Structure
* [x] **partable** should define the model *modulo* ideas such as normality, polychorics, GLS, or even t-distributed residuals or any other crazyness.
  * Done: the lavaanified model is now a triple — `LatentStructure` (what to estimate, name-free, modulo
    estimator + identification convention), `LatentNames` (verbal model — names, user labels, group
    var/levels, `.pN.` plabels), `Starts` (free-param start hints). `partable/lavaan_view.hpp`
    projects to/from the lavaan-shaped SoA. See AGENTS.md "the lavaanified model is the contract".
  * Estimates / se / start now live elsewhere (`Estimates`, `Inference`, `Starts`).
* [x] Lets go through all the columns in the table and (dis)confirm they are needed.
  * `LatentStructure` is now ids + roles only; `block`/`user` dropped; `label`/`plabel`/names moved to
    `LatentNames`. Tiny leftover: `LatentNames.row_plabel` is reconstructible (`.p<rowidx>.` for
    non-constraint rows) — could drop the stored vector, but `from_lavaan_partable` round-trips
    arbitrary input plabels through it, so not free. Deferred (see roadmap housekeeping).
* [x] Separate table in two — done (`LatentStructure` + `LatentNames`, plus `Starts`).

## Optimization
* [x] Use unit variance for identification, not setting arbitrary params to 1.
  * `LavaanifyOptions::std_lv` (≙ lavaan's `std.lv = TRUE`): free the first loading
    per latent, fix each latent's `~~`-self variance at 1.0.
  * `LavaanifyOptions::effect_coding` (≙ lavaan's `effect.coding = "loadings"`):
    free all loadings + LV variances, add `Σλ == #indicators` per latent (rides on
    P9 phase-2 general linear-equality constraints). Mutually exclusive with `std_lv`.
  * Marker stays the default (oracle contract). R: `magmaan_lavaanify(m, std_lv=TRUE)` /
    `magmaan_lavaanify(m, effect_coding=TRUE)`. Mean-structure effect coding (`Σν == 0`)
    not yet built.

## Estimation methods
* [x] **ULS — landed** (`include/magmaan/fit/uls.hpp` + `src/fit/uls.cpp`, ~210 lines total;
  `tests/unit/uls_test.cpp` — 8 cases, 43 assertions).
  * Formula: `F = Σ_b (n_b/N) · ½·[(m̄−μ)ᵀ(m̄−μ) + vech(S−Σ)ᵀvech(S−Σ)]`. **No vech-doubling**
    factor (unlike ML's gradient) — each off-diagonal counted once and the gradient stacks
    `w_b = (n_b/N)·vech(Σ_b−S_b)` and `u_b = (n_b/N)·(μ_b−m̄_b)` directly into `Jᵀw + Jmuᵀu`.
  * **LBFGS+ULS conditioning is shallow.** Even a saturated 6-param 1F CFA needs
    `LbfgsOptions{ max_iter=5000, ftol=1e-14, gtol=1e-9 }` — the landscape lacks ML's
    `log|Σ|` curvature. The integration tests in `uls_test.cpp` hardcode these tolerances.
    Trust-region (Ceres) handles this naturally; see § Optimizers below.
  * **Heywood wandering without bounds.** ULS objective has no PD barrier (unlike ML), so
    LBFGS happily drifts into negative residual variance. Observed in early-test: a
    `random_pd(3)` S outside the 1F manifold pushed θ_resid_3 → −0.18 and the solver stalled.
    Real box constraints (Ceres' `Problem::SetParameterLowerBound`) are the right fix.
  * **Concept ready.** `ULS` satisfies the `Discrepancy` concept trivially; tests prove F=0 at
    saturated, hand-formula match, FD-gradient parity (1F cov, 1F+mean, 3F at lavaan-θ̂),
    multi-group `(n_b/N)` weighting, and `fit<ULS>()` integration recovers ground-truth Σ̂ AND
    ν̂ on in-manifold S.
* GLS / WLS — **paused.** Design still settled (WLS weight on the discrepancy struct;
  `struct WLS { Eigen::MatrixXd gamma_inv; };`), but the Heywood/conditioning findings from ULS
  shifted the priority: the optimizer story (Ceres + bounds) lands first, then GLS/WLS as the
  second/third consumers of the W-generalized robust path. Prep remaining: factor `vech`
  helpers further (`vech_len`/`vech_index` are done; `vech_lower`/`vech_unpack` shared via
  `src/fit/detail_vech.hpp`), and generalize `robust.cpp`'s `A1=ΔᵀWΔ` / `B1=ΔᵀWΓWΔ` to arbitrary `W`.

## Polychorics / threshold model

* Implement standard ULS / DLS crap. 


##
* [ ] Change name from magmaan -> magmaan (joke name lol)

## Misc
* [x] `magmaan_lavaanify(m, n_groups=2L, group_var="school")` now stores `group_var` + `group_labels`
  on the partable (`LatentNames`), rides as data.frame attributes on the R side, and the fit object
  carries `$group_var` / `$group_labels`. (roadmap G2.)

## Nested tests — Satorra (2000) [DONE]

Implemented as `magmaan::fit::lr_test_satorra2000_from_data` and the R-side
`nestedTest()` wrapper, with a streaming low-rank algorithm that reduces the
`p* × p*` eigenproblem to `m × m` (m = df_H0 − df_H1) and never materialises
the empirical Γ̂. See [`docs/nested_tests.md`](docs/nested_tests.md) for the
math, the three p-values (scaled / mean+var-adjusted / exact Imhof mixture),
and the cross-validation strategy.  Original design notes (the GPT-5.5
derivation that motivated the implementation) preserved below for archaeology:

<details><summary>Original GPT-5.5 derivation</summary>

(This is from GPT, and must be tested versus the naive method.)
Yes. The main trick is: do not form the ambient (U\Gamma) matrix. For a nested restriction test, the relevant matrix is rank at most

[
m = \operatorname{df}*{\text{restricted}}-\operatorname{df}*{\text{unrestricted}},
]

so your “take the top (df_2-df_1)” instinct is right only if model 2 is the more restricted model. In general: restricted minus unrestricted.

Satorra’s restricted-test matrix is

[
U
=

V\Pi P^{-1}A^\top
(AP^{-1}A^\top)^{-1}
AP^{-1}\Pi^\top V,
]

where (\Pi) is the Jacobian of the unrestricted model, (P=\Pi^\top V\Pi), and (A) is the restriction Jacobian. The asymptotic mixture uses the non-null eigenvalues of (U\Gamma), and the scaling/adjusted corrections use (\operatorname{tr}(U\Gamma)) and (\operatorname{tr}((U\Gamma)^2)). Satorra also explicitly notes that the scaled difference test is not obtained by subtracting the two separately scaled goodness-of-fit tests, which is a common little trapdoor. 

The low-rank reduction is:

[
C = AP^{-1}A^\top \quad (m\times m),
]

[
S = AP^{-1}\Pi^\top V\Gamma V\Pi P^{-1}A^\top \quad (m\times m).
]

Then the nonzero eigenvalues of the huge (U\Gamma) are exactly the generalized eigenvalues

[
Sx = \lambda Cx.
]

Equivalently, compute the ordinary symmetric eigenvalues of

[
K = C^{-1/2} S C^{-1/2}.
]

That is the little goblin in the wall. The full (p_{\text{moments}}\times p_{\text{moments}}) matrix has (m) meaningful eigenvalues and the rest are zero, up to numerical dust.

Computationally, this means:

[
Y = P^{-1}A^\top
]

by solving (PY=A^\top), not by forming (P^{-1}). Then

[
C = AY,
]

and

[
S = Y^\top \Pi^\top V\Gamma V\Pi Y.
]

The only semi-large object you need is

[
D_g = V_g\Pi_gY,
]

which is (p_g^{\text{moments}}\times m), not (p_g^{\text{moments}}\times p_g^{\text{moments}}).

Even better, you do not need to form (\Gamma_g). Since

[
\hat\Gamma_g
============

\frac{1}{n_g-1}\sum_i (d_{gi}-s_g)(d_{gi}-s_g)^\top,
]

you can accumulate

[
S
=

\sum_g f_g
\frac{1}{n_g-1}
\sum_i
u_{gi}u_{gi}^\top,
]

where

[
u_{gi}=D_g^\top(d_{gi}-s_g).
]

So the streaming computation is just: project each case’s fourth-moment contribution down to (m) dimensions, then accumulate an (m\times m) covariance matrix. No (\Gamma), no (U), no (U\Gamma). This is the big win.

For the scaled and adjusted versions, you may not need eigenvalues at all:

[
\hat c = \frac{1}{m}\operatorname{tr}(C^{-1}S),
]

and

[
\hat d_0
========

\frac{{\operatorname{tr}(C^{-1}S)}^2}
{\operatorname{tr}(C^{-1}SC^{-1}S)}.
]

Only compute all (m) eigenvalues if you want the actual mixture distribution,

[
\sum_{j=1}^m \lambda_j \chi^2_{1,j},
]

for Davies/Imhof-style p-values.

Implementation sketch:

```cpp
// Pi: p_moments x k unrestricted Jacobian
// V : applied blockwise, not necessarily stored
// A : m x k restriction Jacobian
// P = Pi.transpose() * V * Pi, k x k

MatrixXd Y = P.ldlt().solve(A.transpose());   // k x m
MatrixXd C = A * Y;                            // m x m

MatrixXd S = MatrixXd::Zero(m, m);

for each group g {
    MatrixXd Dg = apply_Vg(Pi_g * Y);          // p_g_moments x m

    for each observation i in group g {
        VectorXd r = d_gi - s_g;               // vec/vech outer-product residual
        VectorXd u = Dg.transpose() * r;       // m-vector
        S.noalias() += weight_g * (u * u.transpose());
    }
}

S = 0.5 * (S + S.transpose());
C = 0.5 * (C + C.transpose());

// generalized eigenvalues S x = lambda C x
Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(
    S, C, Eigen::EigenvaluesOnly
);
VectorXd lambda = es.eigenvalues();
```

Eigen’s `GeneralizedSelfAdjointEigenSolver` is meant for (Ax=\lambda Bx) with (A) self-adjoint and (B) positive definite, and supports `EigenvaluesOnly`, so it is exactly the dense small-matrix tool here. ([libeigen.gitlab.io][1])

One more important simplification: do not form (V_g = \frac12(S_g^{-1}\otimes S_g^{-1})) if you can avoid it. For full `vec` covariance moments,

[
V_g \operatorname{vec}(M)
=========================

\frac12 \operatorname{vec}(S_g^{-1}MS_g^{-1}).
]

So `apply_Vg` can be a matrix sandwich. If you use `vech`, you need the duplication/elimination bookkeeping, but the same principle holds: implement (V_g) as an operator.

So yes: you can do much better. The right target is not “top (m) eigenvalues of a huge matrix”; it is “all eigenvalues of an (m\times m) generalized symmetric eigenproblem.”

[1]: https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1GeneralizedSelfAdjointEigenSolver.html "Eigen: Eigen::GeneralizedSelfAdjointEigenSolver< MatrixType_ > Class Template Reference"

</details>


# Optimizers

**Current state** (2026-05-13):
* ✅ **LBFGSpp** — production default. `src/fit/lbfgs_optimizer.cpp` wraps `LBFGSSolver<double>`;
  the TU is `-fexceptions` so we can `try/catch` the line-search-failed throw and surface it as
  `FitError::LineSearchFailed`. Works fine for ML on most fixtures. Known limitations:
  - Throws on the single-group 3F-CFA + meanstructure Holzinger model (the 2-group version
    converges fine; lavaan's nlminb converges on both).
  - Shallow ULS landscapes need a 5× tolerance bump (`ftol=1e-14, gtol=1e-9, max_iter=5000`).
  - No bounds in the current adapter wrap, so ULS can drift to Heywood territory.
* 🔍 **Ceres Solver** — under active investigation (this tranche). Optional via
  `MAGMAAN_WITH_CERES=ON` CMake flag; FetchContent'd, ~5–15 min first build. Two adapters in
  `include/magmaan/fit/ceres_optimizer.hpp`: `CeresOptimizer` (GradientProblemSolver, drop-in for
  LBFGS++) and `CeresBoundedOptimizer` (Problem API with native
  `SetParameterLowerBound`/`SetParameterUpperBound`). Auto bounds derive variance ≥ 0 from the
  partable via `bounds_from_partable` (`include/magmaan/fit/bounds.hpp`). Used through
  `fit_bounded<D, BoundedOptimizer O>()` parallel to `fit<D, O>()`. Benchmarks: Holzinger
  3F-means + ULS shallow + Heywood-prone ULS.

The `Optimizer` and `BoundedOptimizer` concepts (`include/magmaan/fit/concepts.hpp`) keep the
abstraction explicit. Adding NLopt / Ipopt later is a matter of writing another adapter that
satisfies the concept.

## C++ SEM optimizer backlog

Use a lavaan-like optimizer interface: start values, objective, gradient, lower bounds, upper bounds. lavaan’s own exported estimation interface is basically this shape, and its listed optimizers include `nlminb`, `BFGS`, `L-BFGS-B`, `GN`, and `nlminb.constr`. ([RDocumentation][1])

### Tier 1: implement / wrap first

**1. L-BFGS-B: LBFGSpp / LBFGS++** — ✅ **in use** (ML default; current `LbfgsOptimizer`).
Link: [github.com/yixuan/LBFGSpp](https://github.com/yixuan/LBFGSpp)
Header-only, Eigen-based, MIT licensed. Current adapter uses the **unconstrained** `LBFGSSolver`;
LBFGSpp also offers `LBFGSBSolver` (box-constrained) which is a future small add via a
`LbfgsBoxOptimizer` sibling class. ([GitHub][2])

**2. Optimizer zoo / benchmark backend: NLopt** — not yet vendored.
Link: [nlopt.readthedocs.io](https://nlopt.readthedocs.io/)
Use for quick access to many local optimizers: L-BFGS variants, SLSQP, MMA, COBYLA, BOBYQA, Nelder-Mead, etc. Has a C++ wrapper via `nlopt.hpp`. License note: GNU LGPL, with looser licenses for some parts. ([nlopt.readthedocs.io][3])

**3. Trust-region least squares: Ceres Solver** — 🔍 **landed, partially working**.
Link: [ceres-solver.org](https://ceres-solver.org/)
Adapters in `include/magmaan/fit/ceres_optimizer.hpp`: `CeresOptimizer` (GradientProblemSolver,
drop-in for LBFGS++, no bounds) and `CeresBoundedOptimizer` (Problem API with native parameter
bounds). Both reuse `LbfgsOutput` as the universal optimizer output. Optional via
`MAGMAAN_WITH_CERES=ON` (default OFF; opt-in `cmake --preset ceres`). Ceres 2.2.0 fetched with a
minimal flag set (no LAPACK / SuiteSparse / glog / gflags / Schur specializations) in
`cmake/CeresFetch.cmake`.
- **Status:** infrastructure is in (concept-satisfied, basic quadratic + bounded-quadratic
  tests pass, fit_bounded plumbed). The DENSE_QR linear solver is forced on the bounded path
  because Eigen-sparse's AMD reordering segfaults on the rank-1 sparse structure of our
  single-residual cost function.
- **Open problem (next Tranche):** The single-residual `r₀ = √(2F+ε)` trick that flows scalar
  F(θ) into Ceres' bounds-aware Problem API does **not** converge on shallow LS objectives
  like ULS — the 1×n Jacobian gives a rank-1 JᵀJ, LM damping dominates, trust radius shrinks
  to ~1e-6, and F decreases at ~1e-4 per iteration even at 5000 iters. The
  `tests/unit/ceres_integration_test.cpp` tests assert this *current* non-convergence so the
  limitation is visible. Fix: per-discrepancy multi-residual `ceres::CostFunction` — for ULS,
  one residual per vech entry (`r_a = (S−Σ)_a`); JᵀJ becomes full-rank, LM solves cleanly.
  ML stays on `CeresOptimizer` / LBFGS++ (no LS structure to exploit).

### Tier 2: later / optional

**4. General nonlinear constrained optimizer: Ipopt**
Link: [coin-or.github.io/Ipopt](https://coin-or.github.io/Ipopt/)
Use if/when the project needs real equality/inequality constraints, not just boxes. Solves large-scale nonlinear programs with variable bounds and nonlinear constraint bounds. EPL licensed. Deferred — phase-3 constraint work (out of scope per `[[project_constraint_scope]]`). ([GitHub][5])

**5. PORT / `nlminb` mimicry**
Reference: R `nlminb` / PORT-style optimizer.
Lower priority. Useful only if the goal is lavaan/R behavior mimicry. Otherwise prefer LBFGSpp + NLopt + Ceres. lavaan defaults are relevant, but reproducing PORT itself is probably not the best engineering target. ([RDocumentation][1])

**6. BFGS with box constraints / `constrOptim`-like behavior**
Lower priority. Plain BFGS is not naturally box-constrained. For boxes, use L-BFGS-B or Ceres. For general constraints, use NLopt SLSQP / augmented Lagrangian or Ipopt.

### SEM-specific failure handling

Before blaming optimizers, harden the objective wrapper:

* invalid (\Sigma(\theta)): return large finite penalty, not `NaN`/`Inf`;
* check analytic gradients against finite differences;
* scale parameters and objective;
* log projected gradient norm for box methods;
* use lavaan-like starting values;
* support jittered restarts;
* benchmark ML, GLS, WLS/DWLS separately;
* **For ULS/Heywood:** hard bounds via `CeresBoundedOptimizer` + `bounds_from_partable` (variance
  diagonals get `lower=0`). Heywood-detector (G5) still flags the warning; bounds stop the drift.

[1]: https://www.rdocumentation.org/packages/lavaan/versions/0.6-20/topics/lav_export_estimation?utm_source=chatgpt.com "lav_export_estimation function"
[2]: https://github.com/yixuan/LBFGSpp?utm_source=chatgpt.com "yixuan/LBFGSpp: A header-only C++ library for L-BFGS ..."
[3]: https://nlopt.readthedocs.io/en/latest/NLopt_C-plus-plus_Reference/?utm_source=chatgpt.com "C++ reference - NLopt Documentation - Read the Docs"
[4]: https://ceres-solver.org/?utm_source=chatgpt.com "Ceres Solver — A Large Scale Non-linear Optimization Library"
[5]: https://github.com/coin-or/IPOPT?utm_source=chatgpt.com "COIN-OR Interior Point Optimizer IPOPT"
