# magmaan 0.2.0 (in development)

The ordinary API adopted on 2026-10-01. Calls written for 0.1.0 need changes;
each removed argument raises an error that names its replacement.

- `magmaan_model(model, prototype, ordered, group, group.equal, group.partial,
  identification, parameterization)` constructs a model once, with a frozen
  data schema, for any number of fits. `prototype` declares groups and ordinal
  categories; factor levels declare them completely, so a zero-row data frame
  suffices. Data that break the schema raise a `magmaan_schema_error`.
- `magmaan(model, data, estimator, covariance, inference, options)` keeps only
  the choices that define the estimate. A syntax string or lab specification
  is a shortcut that constructs the model from `data`; it rejects undeclared
  ordered factors.
- Every model has a mean structure. Saturated intercepts change no other
  estimate, standard error or test, but `coef()` and `vcov()` gain one entry
  per observed variable.
- Observed covariates are random: `fixed.x` is gone. ML structural estimates
  are unchanged; GLS and ULS estimates of overidentified models change.
  Regressions on observed covariates now get the ML policy's inference.
  Lab specifications built with `fixed_x = TRUE` and observed covariates are
  rejected, not converted.
- `covariance = "unrestricted" | "psd" | "barrier" | barrier(lambda)` replaces
  `psd = TRUE`. Barrier fitting is experimental: a once-per-session message,
  printed status and inference unavailable with reason `"penalized"`.
  `barrier(0)` is the unrestricted fit.
- `options$start` replaces both the top-level `start` and `options$starts`:
  `"default"`, `"fabin3"`, `"lavaan-0.7.2"`, a previous fit or a parameter
  table. An explicit start overrides a preset's.
- `missing`, `cluster` and `meanstructure` are removed. Estimators other than
  FIML and ML2S delete incomplete rows listwise; pairwise deletion and
  two-level fitting remain in magmaanlab.
- Group order follows the prototype's factor levels (lavaan uses first
  appearance), which also sets the reference group for latent means.

# magmaan 0.1.0

First simulation prerelease, paired with magmaanlab 0.1.0. The ordinary API
accepts lavaan syntax and lavaan-backed model specifications. EQS is lab-only.

Development is unfinished: remaining bugs and incomplete coverage of the main
estimators and inferential procedures remain to be resolved and validated.
Version 0.1.0 provides a versioned simulation snapshot, not a finished release.

- Saved specifications supply ordered variables, parameterization and grouping;
  conflicting call options error before fitting. Row accounting uses the
  resolved grouping.
- Defined parameter estimates are retained independently of inference.
- Confidence levels and parameter selections are validated. Unchecked
  convergence is distinguished from a failed fit.
- Help pages and the README document extraction for simulation consumers.

Automatic policy inference covers single-level complete-data ML. Other
estimators retain their estimates and report unsupported inference components.
This prerelease does not expand inference coverage or change its defaults.
