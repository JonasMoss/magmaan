# magmaan: the ordinary-user R package

One call estimates a structural equation model and computes inference under a
single policy that magmaan chooses and justifies. The model is written in
lavaan syntax or supplied as a model specification. The options use lavaan's
names where the concept is the same.

```r
library(magmaan)
fit <- magmaan("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6",
               lavaan::HolzingerSwineford1939)
summary(fit)
coef(fit)            # free estimates, matching vcov(fit)
coef(summary(fit))   # the parameter table: estimates, robust SEs, z, p, intervals
```

Explicit EQS equations, variances and covariances can be supplied through the
shared constructor. The initial subset is single-group continuous covariance
models; see the [EQS contract](../project/grammar/eqs.md) for supported forms.

```r
model <- eqs_model('/EQU V1=1*F1+E1; V2=.7*F1+E2; V3=.8*F1+E3;
                    /VAR F1=1; E1-E3=.5*;',
                   observed_names = c('x1', 'x2', 'x3'))
fit <- magmaan(model, lavaan::HolzingerSwineford1939)
```

Identification and variance/covariance choices belong in the model constructor
when supplying a specification. Estimation and inference retain magmaan's policy.

This package is pure R. Every computation happens in
[`magmaanlab`](../r-package/), the methods-development package over the
magmaan C++ core, which also offers every alternative convention.
`as_lab_fit(fit)` returns the underlying magmaanlab fit.

Status: for single-level complete-data ML, `magmaan()` computes the full
inference policy: the observed-information sandwich covariance (standard
errors, Wald tests, intervals, defined parameters) and the global score and
likelihood-ratio tests, each calibrated with SB and PEBA4. The other estimators
fit, but their inference components report `unsupported_model`, and `vcov()`
and `confint()` raise a `magmaan_inference_unavailable` condition. The design is
[`project/design/r-interface-vision.md`](../project/design/r-interface-vision.md).

Install `magmaanlab` first (`just r-dev` or `just r-install`), then this
package (`just r-magmaan`); `just r-magmaan-test` runs the tests.
