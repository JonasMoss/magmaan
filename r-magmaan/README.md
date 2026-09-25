# magmaan: the ordinary-user R package

One call estimates a structural equation model and computes inference under a
single policy that magmaan chooses and justifies. The model is written in
lavaan syntax and the options use lavaan's names where the concept is the same.

```r
library(magmaan)
fit <- magmaan("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6",
               lavaan::HolzingerSwineford1939)
summary(fit)
coef(fit)
parameters(fit)
```

This package is pure R. Every computation happens in
[`magmaanlab`](../r-package/), the methods-development package over the
magmaan C++ core, which also offers every alternative convention.
`as_lab_fit(fit)` returns the underlying magmaanlab fit.

Status: scaffold. Estimation, the option set, listwise accounting and PSD
fitting work. The inference policy (observed-information sandwich, global score
and likelihood-ratio tests with SB and PEBA4) is not implemented yet, so every
inference component reports `not_implemented`, and `vcov()` and `confint()`
raise a `magmaan_inference_unavailable` condition. The design is
[`project/design/r-interface-vision.md`](../project/design/r-interface-vision.md).

Install `magmaanlab` first (`just r-dev` or `just r-install`), then this
package (`just r-magmaan`); `just r-magmaan-test` runs the tests.
