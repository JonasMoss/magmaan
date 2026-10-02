# Ordinary-user magmaan package

This pure-R package imports `magmaanlab`; it contains no second SEM
implementation. `magmaan_model()` constructs a model and freezes its data
schema; `magmaan(model, data, estimator, covariance, inference, options)`
estimates and computes inference by default. `inference = FALSE` estimates
only; `infer(fit)` adds inference later. Keep few options and use lavaan names
for identical concepts. Reporting methods accept named inference bundles
through `convention` (MLM/MLR/WLSMV); fitting accepts plain estimators.
Do not expose separate information/SE/test ingredient switches.

The policy lives in C++ (`api::policy_inference_ml`, exposed through
`magmaanlab::policy_inference()`). Single-level complete-data ML has the full
policy; other estimators report inference as `unsupported_model` until covered.
Consult the [interface vision](../project/design/r-interface-vision.md) and
roadmap before extending support. Defaults need recorded experimental or
published evidence. No export may mean something different in `magmaanlab`.

R helpers validate inputs, compose wrappers and preserve inspection metadata.
Partable differences belong in the model triple/projections/fit reconstruction,
not presentation patches. Preserve mean-structure rows, labels and groups.
Use `magmaan-r-bindings` for changes requiring compiled adapters or vendor work.

Install the compiled dependency first (`just r-dev` or `just r-install`), then
`just r-magmaan`; `just r-magmaan-test` installs and tests this package.
`just r-check` covers both packages. Integration tests use installed lavaan,
including in CI, while C++ tests consume frozen fixtures.
