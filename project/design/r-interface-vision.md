# R interface vision: estimation and inference

Status: discussion draft, 2026-09-25. This records the requested direction and
an implementation inventory, not a change to running defaults. The older
estimate-only rule for `magmaan()` is to be reconsidered; explicit estimation
remains the contract of the methods-development layer. Names below marked
proposed are illustrative, not callable API promises.

## Direction

Provide two interfaces over the same C++ computations:

- A composable methods interface, closely reflecting the C++ domain objects
  and free functions. Preparation, estimation, information, covariance,
  hypotheses, statistics and calibration are individually usable. Compatibility
  conventions and alternative statistical choices live here.
- A small opinionated interface that selects a documented policy and composes
  those operations. `magmaan()` performs estimation and inference by default;
  `inference = FALSE` requests estimation alone. Routine use should not require
  choosing information matrices, Gamma conventions or correction algorithms.

Explicitness means that the selected policy and computed artifacts can be
inspected and reused. It need not require every applied user to assemble the
same sequence manually. R owns argument handling and presentation; C++ owns
statistical computations. Do not create a second SEM implementation in R.

## What exists today

Inventory of the working tree on 2026-09-25; concurrent start-policy work was
already present and was not changed by this review.

| Surface | Current behavior | Consequence |
| --- | --- | --- |
| `magmaan()` in `r-package/R/model_data.R` | Estimate-only; `se` and `test` must be `"none"`; many estimator-specific controls | Neither automatic inference nor a small applied interface yet |
| `prepare_model/data/weight()` and `estimate()` in `prepared.R` | Explicit preparation and estimation for several estimator families | Useful foundation for the methods interface |
| `magmaan_core` in `zzz_core.R` | Large collection of domain-prefixed bindings and aliases | Broad coverage, but not yet a curated, uniformly documented C++-shaped interface |
| `vcov.magmaan_fit()` in `context.R` | Defaults to `regime = "model"`; complete continuous path uses empirical meat with expected bread; robust selects observed bread | Existing names do not transparently describe the statistical contract |
| `standardized()` | Requires caller-supplied covariance | Appropriate primitive; simple interface can supply its retained covariance |
| `prepare_inference()` in `scores.R` | Immutable snapshots for ML, FIML and fixed-NT ML2S; rejects active bounds | Reuse exists, but support is narrower than estimation |
| `inference_information()` / `parameter_covariance()` | Explicit information and covariance composition; expected information is the former's default | Observed construction exists, but is not the default policy |
| Shared `inference_quadratic()` / `inference_covariance()` | Reusable NTML score/LR and covariance geometry; currently structured expected information | Cannot implement the requested observed policy merely by renaming this composer |
| `score_components()` | Separate sensitivity and metric, including observed options | The distinctions needed for a precise test policy already exist |
| `calibrate_quadratic()` | SB, PEBA4 and other calibrations from reusable reference artifacts | Reuse this machinery; do not duplicate corrections in the convenience wrapper |
| `fmg_tests()` | Multiple defaults including RLS-based tests and different orders | Not the proposed fixed score/LR bundle |
| `score_tests_robust()` | Expected bread, structured moments, empirical covariance, estimated-weight correction off by default | Calling an existing “robust” helper is not sufficient to establish the new policy |
| `nestedTest()` / `robust_nested_lrt()` / `fmg_nested()` | Overlapping comparison surfaces with many conventions | Consolidate routine comparison while preserving explicit primitives |

See also `r-package/man/inference_reuse.Rd` for the shared geometry's scope.
The existing backlog records automatic FIML H1, two-level inference and SAM
work that does not yet follow a uniform estimate-only ownership contract.

## Proposed applied surface

Illustrative target:

```r
fit <- magmaan(model, data)
summary(fit)
coef(fit)
vcov(fit)
confint(fit)

fit_only <- magmaan(model, data, inference = FALSE)
fit <- infer(fit_only)                 # proposed policy composer
compare(null_fit, alternative_fit)     # proposed nested comparison
```

Retain necessary modeling choices such as groups, identification and estimator.
Start with a documented marker default. Expert optimizer, weight and information
choices belong in prepared estimation/inference calls rather than an expanding
set of arguments to the simple wrapper. The default PSD estimation policy is a
separate decision from automatic inference; this draft does not change it.

`infer()` should be the same composer used internally by `magmaan()`, with no
refit. Store enough input ownership to avoid asking for raw data again. Repeated
`summary()`, `vcov()` and `confint()` should read retained results, not repeat
estimation, H1 fitting, casewise score construction or eigendecomposition.

## Proposed statistical contract

The requested policy is observed information with misspecification adjustment
enabled. The applied interface should not expose compatibility switches for
expected information or software-specific conventions. The methods interface
retains those choices. Do not encode this as one ambiguous `robust` flag.

For regular complete-data ML, specify parameter covariance as the sandwich
formed from observed likelihood curvature and empirical score covariance, with
documented normalization and equality reduction. Report ordinary symmetric
Wald intervals from that covariance. “Ordinary Wald” describes the interval
construction, not inverse-information model-based standard errors.

For tests, separately specify nuisance sensitivity, quadratic metric, H0/H1
evaluation point, moment covariance, centering and normalization. “Observed”
alone does not decide those items. Do not mechanically replace every matrix by
an observed Hessian. Existing observed score components and expected NTML reuse
must be reconciled against an explicit mathematical contract first.

Proposed automatic output:

- Parameter estimates, standard errors, Wald intervals and Wald tests against
  zero, including supported defined parameters via the same covariance.
- Global score and likelihood-ratio tests against the saturated alternative,
  each with SB and PEBA4 calibration. Label the statistic and calibration
  separately; retain the spectrum or trace and numerical diagnostics.
- Fit acceptance and inference availability, independently recorded.

Nested comparisons require an explicit second model. `compare()` should check
data, objective and nesting compatibility and produce the analogous supported
test bundle. Modification-index searches, arbitrary hypothesis enumeration,
bootstrap/profile intervals and automatic independence-model fitting are not
implied by this default bundle.

Misspecification-robust parameter uncertainty and an exact-fit test answer
different questions. The latter still tests a null model; the misspecification
policy must not be described as making that null true. For estimated-weight
estimators, observed curvature alone is insufficient: the weight-estimation
influence must be included where required. Extend estimator support only after
its corresponding contract is checked.

## Availability and PSD fits

The first implementation target should be regular complete-data ML. Maintain a
component-level capability table for FIML, ML2S, ordinal/mixed, multilevel and
frontier fits. A successful fit must survive unavailable inference; retain a
typed reason per unavailable component and show it in the summary. No silent
substitution of expected information, a different correction or a different
estimator. Covariance-only inputs need an explicit source of empirical score
covariance for the proposed sandwich; do not silently assume normal theory.

Interior PSD fits and singular PSD fits must be distinguished. Boundary-aware
optimization auditing does not establish boundary-aware sampling inference.
Before claiming the simple interface covers PSD fitting, specify which Wald,
score and LR components are valid and implemented on each domain. Until then,
unavailable boundary components are reported as such. This is also necessary
for trustworthy downstream research code.

## Implementation sequence after discussion

1. Agree on the applied output and the precise observed test geometry. Preserve
   explicit choices in the methods layer; settle names after the contracts.
2. Audit estimation/inference capabilities and ownership, including fit-time H1
   work and boundary support. Identify computations already retained by fitting.
3. Compose one observed/misspecification inference policy in core, extending
   reusable artifacts where needed. Gate against explicit primitive composition,
   analytic identities and independent references for the chosen policy.
4. Add the small R composer, result accessors and summary. Verify that automatic
   and explicit workflows give identical results and reuse computations; verify
   that `inference = FALSE` avoids optional inference work.
5. Migrate examples and downstream callers only after the policy is validated.
   Time preparation, estimation and inference separately. Preserve compatibility
   wrappers through a documented transition rather than globally changing every
   low-level default.

This is a design proposal, not a claim that observed choices universally dominate
all alternatives or that the complete bundle already exists. No estimator,
inference default or production result was changed in this pass.
