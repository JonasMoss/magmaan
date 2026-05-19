# WMA h-score robust polychoric plan

This is a planning note only. It should not be treated as an implemented
contract until the ordinal stats APIs and R bindings are updated.

## Source idea

Welz, Mair, and Alfons (WMA) define a robust polychoric estimator by minimizing

```text
L(theta, fhat) = sum_j p_j(theta) phi(fhat_j / p_j(theta) - 1).
```

Let

```text
t_j(theta) = fhat_j / p_j(theta)
s_j(theta) = grad_theta log p_j(theta).
```

Differentiating the objective gives the estimating equation

```text
sum_j p_j(theta) h(t_j(theta)) s_j(theta) = 0
```

where

```text
h(t) = t phi'(t - 1) - phi(t - 1).
```

For the WMA discrepancy, with tuning constant `c` and `k = c + 1`,

```text
h(t) = min(t, k).
```

So the proposed h-score family is a good abstraction: ML is `h(t) = t`, WMA is
the hard cap, and smoother caps can keep ML behavior near `t = 1` while making
the estimating equation easier for gradient-based optimizers.

## Assessment of the proposed generalization

The generalization looks technically sound, but two details matter for magmaan.

1. The objective link is correct. For any suitable `h`, an objective of the form
   `sum p Phi(fhat / p)` exists when

   ```text
   h(t) = t Phi'(t) - Phi(t),
   Phi(t) = t integral_1^t h(u) / u^2 du
   ```

   up to irrelevant constants. This recovers ML with `Phi(t) = t log(t)` and
   WMA with the piecewise linear cap.

2. Paper-grade WMA estimates the pair's thresholds and rho jointly. magmaan's
   current all-ordinal path estimates thresholds from marginal proportions, then
   estimates each pairwise polychoric rho conditional on those thresholds. That
   is lavaan-compatible, but it is not the full WMA estimator and it is not
   robust to contaminated marginal thresholds.

The proposed smooth h-functions are natural additions:

- `ml`: `h(t) = t`, `h'(t) = 1`.
- `wma_hard_cap`: `h(t) = min(t, k)`, `h'(t) = 1[t < k]` away from the kink.
- `smooth_cap`: derivative cap `r_{a,b}(t)` with quintic transition, integrated
  to `h_{a,b}`. This is the best default for new optimization experiments
  because it is exactly ML up to `a`, bounded after `b`, and smooth through the
  transition.
- `exp_cap`: exact ML up to `k`, then `k + lambda * (1 - exp(-(t-k)/lambda))`.
  This is simple and monotone, but only `C1` at the join unless extra smoothing
  is added.

Avoid downweighting undercounts by default. The WMA argument is persuasive:
multinomial frequencies are compositional, so inflated cells mechanically
deflate other cells. Robustifying negative Pearson residuals risks
downweighting clean cells.

## Implementation shape

Keep this in `magmaan::data`, not in `estimate`, because it produces sample
statistics consumed by ordinal DWLS/WLS.

Add a public ordinal robust-weight API along these lines:

```cpp
namespace magmaan::data {

enum class PolychoricHKind {
  ML,
  WmaHardCap,
  SmoothCap,
  ExpCap,
};

struct HFunctionParams {
  PolychoricHKind kind = PolychoricHKind::ML;
  double k = std::numeric_limits<double>::infinity(); // WMA hard cap, c + 1
  double a = 1.6;       // smooth cap ML-to-transition point
  double b = 2.2;       // smooth cap plateau point
  double lambda = 0.2;  // exp cap transition length
};

struct HFunctionEval {
  double h;
  double dh;
  double phi; // optional objective contribution; NaN if not available
};

HFunctionEval eval_h_function(double t, const HFunctionParams& params) noexcept;

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& X,
                                const HFunctionParams& h = {});

} // namespace magmaan::data
```

Do not use virtual functions on the hot path. The predefined enum-backed
function is enough for the R boundary. For C++ experimentation, add a templated
internal builder that accepts any small function object satisfying:

```cpp
h(t), dh(t), phi(t)
```

with `phi` optional when solving score equations instead of minimizing the
minimum-disparity objective.

## Phase plan

### Phase 0: Preserve current lavaan-compatible behavior

- Rename internal helpers conceptually, but leave the default public behavior as
  ML/lavaan-compatible two-step thresholds plus polychorics.
- Add tests showing `HFunctionParams{ML}` reproduces the current
  `ordinal_stats_from_integer_data()` output exactly or to existing tolerances.
- Keep R default unchanged.

Acceptance: existing ordinal golden tests remain unchanged.

### Phase 1: Experimental fixed-threshold h-weighted rho

This is the smallest useful engineering step, but it should be explicitly named
experimental.

- Add an internal pair-table representation that stores counts, relative
  frequencies, model probabilities, score columns, and h/dh weights for a fixed
  pair.
- Implement rho estimation with fixed marginal thresholds using either:
  - objective minimization when `phi` is available, or
  - one-dimensional root/minimize `||Psi_h(rho)||^2` when only `h` is available.
- Return robust rho, Pearson residuals, and per-cell weights for diagnostics.
- Keep NACOV on the existing current approximation initially, but label the R
  result so downstream code does not mistake it for paper-grade WMA SEM theory.

Acceptance:

- `ML` reproduces current rho.
- `WmaHardCap(k = Inf)` reproduces `ML`.
- Constructed contaminated tables move rho less than ML under an inflated
  opposite-corner cell.

### Phase 2: Full pairwise WMA/h-score estimator

Implement the estimator the paper actually describes for each bivariate ordinal
table. This is a validation and replication step, not the final SEM target,
because the thresholds are pair-local nuisance parameters.

- Parameter vector is pair-local:

  ```text
  theta_ab = [thresholds_a, thresholds_b, rho_ab].
  ```

- Substantively, thresholds belong to variables, not pairs. Under the clean
  model, pair-local threshold estimates for the same variable should agree
  asymptotically. Under contamination, they can differ because each pair sees a
  different bivariate contamination pattern. That is acceptable for isolated
  bivariate polychorics, but awkward for SEM because SEM needs one threshold
  vector.
- Optimize the objective `sum p Phi(fhat / p)` for objective-backed h-functions.
- Also support score-equation solving for h-functions without `Phi`, but keep
  that as a C++ experimentation path until it is well tested.
- Enforce threshold ordering and `rho in (-1, 1)` via unconstrained transforms,
  not fragile box constraints:
  - threshold increments as positive transformed gaps,
  - rho as `tanh(eta)` or bounded logistic map.
- Compute the estimating-function Jacobian `A` and empirical meat `B` from the
  same per-cell scores used in fitting.
- For the hard cap, use `dh(t) = 1[t < k]` and document the kink convention.
  For smooth caps, use analytic `dh`.

Acceptance:

- `ML` pair estimates match existing ML/two-step rho when thresholds are fixed,
  and match joint ML references when thresholds are estimated jointly.
- `WmaHardCap(k = 1.6)` matches robcat/WMA on selected bivariate tables within
  numerical tolerance.
- Standard errors for a single pair match WMA/robcat or independent finite
  difference sandwich checks.

### Phase 3: SEM moment vector and joint Gamma

This is the part that makes it an ordinal SEM contribution rather than just a
robust pairwise polychoric feature.

Target moment vector:

```text
sigma_hat_h = [thresholds, lower-triangle robust polychorics].
```

Open design choice: thresholds.

- Option A: keep lavaan-style marginal thresholds in the SEM moment vector and
  use pairwise robust thresholds only as nuisance parameters for rho. This is
  easiest to integrate and keeps one threshold per observed variable, but it
  gives only partial WMA robustness.
- Option B: estimate shared thresholds and all pairwise correlations jointly
  across all bivariate tables with a composite h-score/minimum-disparity
  objective. This is the coherent SEM-grade target.
- Option C: additionally parameterize the full latent correlation matrix as
  positive definite during the same optimization. This is statistically cleaner
  than unconstrained pairwise rhos but much larger than the current ordinal
  stats builder.

Recommended path:

1. Implement Option A as the smallest experimental SEM hook.
2. Use Phase 2 pair-local WMA to validate against the paper/robcat.
3. Implement Option B as the research-grade estimator.
4. Keep Option C as a later branch unless positive-definiteness failures dominate
   the simulations.

The shared-threshold composite objective for Option B is:

```text
min over tau_1,...,tau_p and rho_ab:
  sum_{a < b} sum_{j in cells_ab}
    p_ab,j(tau_a, tau_b, rho_ab)
    Phi(fhat_ab,j / p_ab,j(tau_a, tau_b, rho_ab)).
```

Equivalently, the estimating equation stacks the h-weighted bivariate scores.
For a threshold belonging to variable `a`, sum the threshold score contributions
over every pair involving `a`; for `rho_ab`, use only pair `(a,b)`.

Do not expect a robust univariate threshold estimator to solve this by itself.
A univariate ordinal threshold model is saturated: category proportions can be
fit exactly by thresholds, so there is no bivariate residual structure with
which to detect overcounted cells. The robustness signal comes from the
overidentification in the bivariate latent-normal tables.

For Gamma:

- Compute casewise influence rows for every final moment in `sigma_hat_h`.
- For Option A, each pair can use the pair-local influence function

  ```text
  IF_ab(i) = - A_ab^{-1} psi_ab(X_i; theta_ab)
  ```

  with the sign convention matched to the implemented score, and extract only
  the rho component into the global moment vector.
- For Option B, `A` is the Jacobian of the full shared-threshold composite score,
  not a block of independent pairwise Jacobians. The influence row is

  ```text
  IF(i) = - A^{-1} psi(X_i; tau, rho)
  ```

  where `psi(X_i; tau, rho)` stacks that row's contributions across all pairs.
  This directly yields influence columns for the shared thresholds and pairwise
  correlations.
- Stack casewise IF rows over thresholds and rhos, then compute

  ```text
  Gamma_h = IF' IF / N
  ```

  using the same scale convention as current `OrdinalStats::NACOV`.

Acceptance:

- With `ML`, the new influence/Gamma path matches the current NACOV as closely
  as the differing threshold convention permits.
- `robust_ordinal()` can consume the robust `NACOV`, `W_dwls`, and `W_wls`
  through the existing `WeightedMomentBlock` path without SEM-side changes.
- Tests cover multi-group block stacking, threshold ordering, moment ordering,
  and DWLS/WLS weight construction.

### Phase 4: R boundary

Keep C++ more flexible than R.

- R exports should expose only predefined functions:
  - `"ml"`
  - `"wma"` or `"hard_cap"`
  - `"smooth_cap"`
  - `"exp_cap"`
- Suggested R surface:

  ```r
  data_ordinal_stats_from_df(
    x, model, ordered = NULL, group = NULL,
    missing = c("listwise", "error"),
    polychoric = c("ml", "wma", "smooth_cap", "exp_cap"),
    polychoric_control = list()
  )
  ```

- Default remains `"ml"`.
- Return diagnostics under a new optional list element, for example
  `polychoric_diagnostics`, containing pair labels, convergence status, h
  weights, Pearson residuals, and cap parameters.
- Do not expose arbitrary R callbacks into C++ for h-functions. That would be
  slow, hard to test, and contrary to the C++ methods-developer surface. Custom
  h-functions should be C++ only until a real need appears.

Acceptance:

- Existing R examples are unchanged with defaults.
- New R examples compare ML, WMA hard cap, smooth cap, and exp cap on one
  contaminated ordinal dataset.

## Positive definiteness

Robust pairwise polychoric matrices can be indefinite. The implementation
should make this visible rather than silently projecting by default.

Plan:

- Add a small helper to test the minimum eigenvalue of each robust R.
- Add optional minimal ridge/shrinkage:

  ```text
  R_lambda = (1 - lambda) R + lambda I
  ```

  with the smallest lambda that clears a configurable eigenvalue floor.
- Store `lambda` and the pre-shrinkage minimum eigenvalue in diagnostics.
- Keep nearest-correlation projection out of the first implementation unless a
  paper simulation specifically needs it.

## Research path

The paper-level claim should be framed as:

```text
Smooth h-score robust polychoric matrices for limited-information ordinal SEM.
```

The contribution is not just replacing `min(t, k)` by a smooth cap. It is:

- a general h-score/minimum-disparity family that includes ML and WMA;
- smooth caps that preserve clean-model first-order efficiency and improve
  optimizer behavior;
- joint influence functions for thresholds plus robust polychoric correlations;
- `Gamma_h` for DWLS/WLS ordinal SEM, robust SEs, and robust test statistics;
- simulations showing stability, bias, SE calibration, convergence, and
  positive-definiteness behavior under realistic response contamination.

Simulation grid:

- ML/lavaan polychorics;
- WMA hard cap with `c = 0.6` (`k = 1.6`);
- smooth cap with `a = 1.6` and several transition widths;
- exponential cap;
- optionally DPD/Hellinger outsiders as benchmarks.

Contamination designs:

- opposite-corner careless responding;
- straightlining, including reverse-coded item blocks;
- random responding;
- midpoint responding;
- diffuse contamination;
- Clayton/tail-dependent and skewed latent distributions for distributional
  misspecification.

SEM outcomes:

- loading and factor-correlation bias;
- SE calibration and coverage;
- convergence and boundary behavior;
- positive-definiteness failures and shrinkage rates;
- chi-square/test statistic and fit-index stability.

## Risks and decisions to make later

- Whether robust SEM thresholds should be marginal robust thresholds or
  reconciled pairwise WMA thresholds.
- Whether full WLS should use the full robust `Gamma_h` inverse, a diagonally
  stabilized inverse, or a shrinkage inverse when `Gamma_h` is ill-conditioned.
- Whether mixed continuous/ordinal data should wait until all-ordinal robust
  Gamma is stable. Recommended: wait.
- Whether `WMA` should be the public name. Internally prefer descriptive names
  like `HardCapH`, with R aliases accepting `"wma"` and `"hard_cap"`.
- How to validate against robcat when its optimizer fallback or parameterization
  differs from magmaan.

## Concrete next implementation slice

When the active ordinal edits settle, start with:

1. Add enum-backed h-function evaluation and tests.
2. Factor bivariate cell probability/score code out of `src/data/ordinal.cpp`
   into a private helper.
3. Add fixed-threshold h-weighted rho as an experimental code path.
4. Add diagnostics and R controls, defaulting to current ML.
5. Only then implement full joint pairwise WMA/h-score and robust Gamma.
