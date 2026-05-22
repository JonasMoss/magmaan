# magmaan — TODO

## 1. [BUG] Multi-group fits fail with cross-group equality constraints

> **RESOLVED 2026-05-22.** Root cause: `lavaanify` emitted two partable rows
> for a loading cell named twice (`NA*LM1 + c(a1,a2)*LM1`) — `assemble`
> overwrote one, leaving a phantom free parameter that moved the analytic
> gradient but not the model moments, so the gradient disagreed with the
> objective. Fix: `build_group_template` (`src/spec/build.cpp`) now merges
> repeated `lhs op rhs` terms within a block and accumulates their modifiers
> onto one row. `guo_mi_weak` now fits to the lavaan reference
> (`chisq = 25.5159`, `df = 11`, `npar = 40`) under both L-BFGS and PORT.
> Regression tests: `tests/unit/lavaanify_test.cpp` ("repeated `lhs op rhs`
> term merges modifiers"). Remaining: end-to-end Kline-corpus parity coverage
> — see `docs/backlog/todo.md`.

**Priority: highest.** Surfaced 2026-05-22 while building the Kline textbook
test corpus (`papers/snlls-constrained/extern/kline-corpus/`).

### Symptom

magmaan cannot fit a multi-group SEM that carries **cross-group equality
constraints** (`a1 == a2` style — the constraints that define measurement
invariance). Both optimizer backends fail:

- **PORT**: `NumericIssue` after ~66 iters — *"noisy objective detected
  (PORT IV(1)=8) — the model evaluator's gradient is inconsistent with its
  function value to more than PORT's noise tolerance."*
- **NLopt L-BFGS**: `LineSearchFailed` after ~1535 iters, `f≈0.0194`
  (near the optimum, then the line search collapses).

The PORT message is the key lead: **the analytic gradient does not agree with
the objective.** The L-BFGS failure is consistent with a wrong gradient.

`lavaan` fits the identical model without trouble (`chisq = 25.52`, `df = 11`).

### What localizes it

A multi-group model with only **within-group** (effects-coding) constraints
fits fine — `guo_mi_configural` (`full`) converges. The failure appears the
moment **cross-group** `==` constraints are added (`guo_mi_weak`,
`guo_mi_strong`, `guo_mi_partial_strong`). So the suspect is the gradient of
the objective in the multi-group + linear-equality-constraint path of the
model evaluator.

(Separately, and correctly: SNLLS rejects `guo_mi_configural` as
non-separable, because its effects-coding constraints touch the profiled
linear block. That is expected behaviour, not this bug.)

### Problematic cases (clear link)

In the Kline corpus bundle — currently
`papers/snlls-constrained/extern/kline-corpus/`, moving to
`external/kline/` when that bundle is finalized:

| example | `models/…` | `data/…` | magmaan |
|---|---|---|---|
| `guo_mi_configural` | `guo_mi_configural.lav` | `guo_mi_configural_g{1,2}_*.csv` | full OK |
| `guo_mi_weak` | `guo_mi_weak.lav` | `guo_mi_weak_g{1,2}_*.csv` | **fails** |
| `guo_mi_strong` | `guo_mi_strong.lav` | `guo_mi_strong_g{1,2}_*.csv` | **fails** |
| `guo_mi_partial_strong` | `guo_mi_partial_strong.lav` | `guo_mi_partial_strong_g{1,2}_*.csv` | **fails** |

Source: Kline (2023), *Principles and Practice of SEM* (5th ed.), ch. 22,
`guo-mi-models.r` — a two-factor CFA of divergent thinking fitted across
Chinese (n=316) and American (n=302) samples.

### Self-contained reproduction

Needs only `magmaan` and `lavaan` installed. `lavaan` succeeds; magmaan fails
under both backends.

```r
suppressPackageStartupMessages(library(lavaan))
subtest <- c("LM1", "LM2", "LM3", "RW1", "RW2")
mkcov <- function(lower, sds) {
  suppressWarnings(cor2cov(getCov(lower, names = subtest), sds = sds))
}
chinese.cov <- mkcov('1.000
 .715 1.000
 .631 .738 1.000
 .377 .408 .319 1.000
 .452 .495 .391 .445 1.000', c(1.532, 1.878, 2.012, 1.002, 1.055))
american.cov <- mkcov('1.000
 .519 1.000
 .543 .639 1.000
 .280 .321 .317 1.000
 .242 .380 .300 .665 1.000', c(1.094, 1.350, 1.494, 2.257, 1.541))
chinese.mean  <- setNames(c(1.24, 1.47, 1.86, .77, .89), subtest)
american.mean <- setNames(c(.88, 1.35, 1.27, 1.71, 1.37), subtest)

# Two-factor CFA, configural model + weak-invariance constraints:
# effects coding within group, plus cross-group loading equalities a1==a2 ...
model <- '
LineMeaning =~ NA*LM1 + c(a1,a2)*LM1 + c(b1,b2)*LM2 + c(c1,c2)*LM3
LM1 ~ c(d1,d2)*1
LM2 ~ c(e1,e2)*1
LM3 ~ c(f1,f2)*1
RealWorld =~ NA*RW1 + c(g1,g2)*RW1 + c(h1,h2)*RW2
RW1 ~ c(i1,i2)*1
RW2 ~ c(j1,j2)*1
LineMeaning ~ c(m1,m2)*1
RealWorld ~ c(q1,q2)*1
3 - a1 - b1 - c1 == 0
0 - d1 - e1 - f1 == 0
0 - d2 - e2 - f2 == 0
2 - g1 - h1 == 0
0 - i1 - j1 == 0
0 - i2 - j2 == 0
a1 == a2
b1 == b2
c1 == c2
g1 == g2
h1 == h2'

# lavaan: fits fine (converged, chisq = 25.52, df = 11)
lav <- sem(model, sample.cov = list(chinese.cov, american.cov),
           sample.mean = list(chinese.mean, american.mean),
           sample.nobs = c(316, 302), meanstructure = TRUE)
cat(sprintf("lavaan : converged=%s  chisq=%.4f  df=%d\n",
            lavInspect(lav, "converged"), fitMeasures(lav, "chisq"),
            as.integer(fitMeasures(lav, "df"))))

# magmaan: fails under both backends
spec <- magmaan::model_spec(model, meanstructure = TRUE,
                            group = "grp", group_labels = c("g1", "g2"))
dat <- list(S = list(chinese.cov, american.cov),
            mean = list(chinese.mean, american.mean), nobs = c(316L, 302L))
for (opt in c("nlopt-lbfgs", "port")) {
  res <- tryCatch(
    magmaan::magmaan_core$fit_gls(spec, dat, optimizer = opt,
      control = list(max_iter = 5000L, ftol = 1e-12, gtol = 1e-8)),
    error = function(e) e)
  if (inherits(res, "error")) {
    cat(sprintf("magmaan %-11s: FAIL - %s\n", opt, conditionMessage(res)))
  } else {
    cat(sprintf("magmaan %-11s: converged=%s 2f=%.6g\n", opt,
                res$converged, 2 * res$fmin))
  }
}
```

Expected output:

```
lavaan : converged=TRUE  chisq=25.5159  df=11
magmaan nlopt-lbfgs: FAIL - ... [LineSearchFailed] ... f=0.0194055 ...
magmaan port       : FAIL - ... [NumericIssue] ... gradient is inconsistent
                            with its function value ...
```

### Suggested first step

Finite-difference-check the objective gradient for a multi-group model with a
cross-group `==` constraint (e.g. the `model` above) and compare against the
analytic gradient magmaan computes — the PORT diagnostic says they disagree.
