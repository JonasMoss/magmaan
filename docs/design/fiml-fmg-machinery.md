# FIML FMG Machinery

This note records the intended missing-data UGamma machinery before further
implementation work. The goal is to keep the goodness-of-fit and nested-model
routes conceptually separate from the complete-data helper code and from
lavaan/semTests compatibility shortcuts.

## Complete-Data Template

For continuous complete data, single-model FMG works in saturated moment space.
Let

```text
eta_hat = [mean; vech(S)]      or      vech(S)
Gamma   = ACOV(eta_hat)
V       = normal-theory saturated information
Delta   = d eta(theta) / d alpha
```

where `alpha` is the active parameter coordinate after equality constraints.
The model removes the tangent directions spanned by `Delta`; the residual
operator is

```text
U = V - V Delta (Delta' V Delta)^-1 Delta' V
```

and the reference law for the GOF statistic is

```text
T -> sum_j lambda_j chi^2_1j
lambda = eig(U Gamma)
```

H0 enters only through the fitted point `theta_hat`, the model tangent
`Delta(theta_hat)`, and the base chi-square statistic. Gamma is the ACOV of the
saturated moment estimator, not an H0 parameter-space sandwich.

## FIML Saturated Moment Space

For FIML with arbitrary observed-value patterns, use the observed-data
saturated model as the H1 moment estimator. Per group/block,

```text
eta_b = [mu_b; vech(Sigma_b)]
eta   = [eta_1; eta_2; ...]
```

The saturated estimator is the EM/observed-data ML estimate of `eta`. At the
saturated estimate, define row log-likelihood scores in eta-space:

```text
s_i(eta_hat) = d log L_i(eta) / d eta
```

and aggregate

```text
H = - d^2 log L(eta_hat) / d eta d eta'
J = sum_i s_i s_i'
Gamma_mis = H^-1 J H^-1
V = H
```

`H`, `J`, and `Gamma_mis` are block-diagonal across independent groups, but not
across missingness patterns inside a group. Scaling must be consistent:
`H` and `J` are summed log-likelihood quantities, so `V Gamma_mis` is
dimensionless and has the same role as `V Gamma` in complete-data FMG. The
saturated `H` is computed from analytic observed-row Hessians; see
`docs/research/notes/fiml_saturated_information.tex` for the derivation.

The current core ingredients are:

- `estimate::fiml::saturated_em_moments(raw, h_step)` for saturated
  `mean`, `cov`, analytic `H`, `J`, and `acov`; `h_step` is retained for source
  compatibility but no longer tunes saturated H1 information.
- `estimate::fiml::diagnostic::saturated_em_moments_fd(raw, h_step)` for the
  C++-only finite-difference comparator used by regression tests.
- `estimate::fiml::fiml_ugamma_spectrum(...)` for the current single-model
  spectrum construction.
- `estimate::fiml::fiml_robust_mlr(...)` for the MLR trace/scaling route. This
  is useful as a consistency check, but it is not the definition of FMG Gamma.

## Single-Model FIML GOF

For a fitted FIML model M0:

1. Fit M0 under the observed-data FIML likelihood.
2. Compute the FIML LRT against the saturated observed-data model:

   ```text
   T_ML = 2 (log L_sat - log L_M0)
   ```

   This is the GOF base statistic. There is no complete-data RLS statistic in
   this route unless a separate missing-data residual statistic is defined.

3. Build the model tangent in saturated eta-space:

   ```text
   Delta = d [mu(theta); vech(Sigma(theta))] / d alpha
   ```

   For linear equality constraints, `alpha` is the existing affine
   reparameterization. For nonlinear equality constraints, the correct
   `alpha` tangent is the null space of the active constraint Jacobian at
   `theta_hat`; using only shared-label/linear `K` is incomplete.

4. Form the V-metric residual operator:

   ```text
   U0 = V - V Delta (Delta' V Delta)^-1 Delta' V
   ```

5. Return the top `df` nonzero eigenvalues of `U0 Gamma_mis`, with

   ```text
   df = dim(eta) - rank(Delta)
   ```

6. Feed `T_ML`, `df`, and the eigenvalues to the existing FMG transforms
   (`SB`, `SS`, `SF`, `ALL`, `pALL`, `EBA`, `pEBA`, `pOLS`, exact Imhof).

The complete-data degeneracy check for this FIML route is not the current
complete-data "structured" FMG default. It should reproduce the complete-data
unstructured H1-information spectrum, because FIML's saturated EM model uses
the unstructured saturated moment information as `V = H`.

Under FIML, the implemented single-model boundary is:

- biased Gamma only: `Gamma_mis = H^-1 J H^-1`;
- ML/LRT base only;
- single- and multi-group;
- mean structures required;
- no `_ug`/Du-Bentler finite-sample correction until there is a derivation for
  observed-pattern saturated estimators;
- no `_rls` until a missing-data residual statistic is explicitly defined.

## Nested FIML Tests

For nested models M0 inside M1, use the same saturated FIML eta-space and the
same `V` and `Gamma_mis`. The nested test should not be assembled from H0/H1
parameter Hessians.

Let M1 be the less restricted model and M0 the restricted model. The exact
H1-anchored Satorra-style reduction is the direct missing-data analog of the
complete-data `compute_satorra2000` path:

```text
Delta1 = d eta(theta_1) / d alpha_1      at the M1 fit
P      = Delta1' V Delta1
A      = restriction matrix in alpha_1-space, rank m
C      = A P^-1 A'
S      = A P^-1 Delta1' V Gamma_mis V Delta1 P^-1 A'
```

The `m = df0 - df1` nested eigenvalues solve

```text
S v = lambda C v
```

and the base statistic is the observed-data likelihood-ratio difference:

```text
T_diff = T_M0 - T_M1
       = 2 (log L_M1 - log L_M0)
```

The same downstream nested transforms can then be reported:

- unscaled chi-square with `m` df;
- mean-scaled/Satorra-Bentler: `T_diff / mean(lambda)`;
- mean+variance adjusted;
- scaled+shifted;
- exact mixture p-value through Imhof.

H0 can enter nested machinery in exactly three places:

1. its fitted likelihood/statistic for `T_diff`;
2. the model nesting/restriction map `A`;
3. an optional delta-style restriction construction that compares the M0 and
   M1 eta-space tangents.

H0 should not define Gamma, and an H0 parameter Hessian should not be needed
for the FMG/nested spectrum.

## Restriction Map Boundary

The complete-data nested code has two restriction strategies:

- exact parameter nesting via equality-constraint `K` matrices;
- delta/moment-column nesting via H0/H1 moment Jacobians.

For FIML nested tests, both need eta-space versions:

- exact: derive `A` in M1's active alpha coordinates. This handles shared-label
  linear restrictions when the two lavaanified models are otherwise aligned.
- delta: build `Delta1` and `Delta0` in the same saturated eta layout
  `[mu; vech(Sigma)]`, then find the M0 column-space restriction inside M1's
  column space.

Nonlinear equality constraints are rejected until there is a tangent-space
implementation. The tangent is local to the fitted point, so a pure
shared-label `K` is not enough.

## Validation Coverage

The checked validation now covers:

- complete-data degeneracy against the unstructured complete-data UGamma
  spectrum;
- actual missing-data single-group spectra with finite positive top-`df`
  eigenvalues;
- multi-group missing-data spectra;
- mean-structure layout and df accounting;
- trace consistency:

  ```text
  sum(eig(U0 Gamma_mis)) == trace_ugamma from the existing FIML MLR route
  ```

  modulo the documented scaling convention;

- explicit rejection of `_ug` and `_rls` under FIML;
- equality-constraint df/tangent behavior;
- a nested FIML synthetic case where the nested eigenvalues collapse to ones
  under `Gamma = V^-1`;
- a nested FIML case comparing dense and reduced generalized-eigenvalue paths;
- explicit rejection of nonlinear equality constraints for FIML GOF and nested
  tests.

The trace check is a diagnostic, not the definition of the spectrum. The
definition is the eta-space projector `U Gamma_mis`.

## Implementation Status

The FIML FMG implementation follows this construction:

- `fiml_ugamma_spectrum()` uses only saturated `H/J/acov`, the H0 model tangent,
  and the FIML LRT base;
- `_ug` and explicit `_rls` are rejected under FIML;
- complete-data validation is against the unstructured H1-information spectrum,
  not the structured complete-data default;
- actual missing-data trace checks compare the spectrum sum to
  `fiml_robust_mlr`'s `trace_ugamma`;
- multi-group missing-data spectra have C++ coverage;
- saturated H1 information is analytic and checked against the old
  finite-difference route on complete, missing, and multi-block raw data;
- nonlinear equality constraints are rejected explicitly;
- nested FIML `restriction_map` tests use the model-pair `A/C/S` route in
  saturated eta-space, with exact and delta restriction-map variants.
