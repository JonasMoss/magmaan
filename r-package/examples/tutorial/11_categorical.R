## lavaan tutorial — Categorical data   https://lavaan.ugent.be/tutorial/cat.html
##
## With ordered (categorical) indicators lavaan switches to a least-squares
## estimator over thresholds + polychoric correlations (DWLS / WLSMV).
## magmaan exposes that as estimator = "DWLS" with `ordered = `. Here three
## continuous indicators are coarsened into 3-category ordinal variables and
## a one-factor model is fit with DWLS.

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-2)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}
## loadings are the order-robust parity signal for an ordinal fit.
load_match <- function(fit, lav, tol = 1e-2) {
  mp <- fit$partable[fit$partable$free > 0 & fit$partable$op == "=~", , drop = FALSE]
  lp <- lavaan::parameterEstimates(lav)
  all(vapply(seq_len(nrow(mp)), function(i) {
    r <- lp[lp$lhs == mp$lhs[i] & lp$op == "=~" & lp$rhs == mp$rhs[i], ]
    nrow(r) >= 1 && near(mp$est[i], r$est[1], tol)
  }, logical(1)))
}

hs  <- HolzingerSwineford1939
cut3 <- function(x) ordered(cut(x, quantile(x, c(0, 1/3, 2/3, 1)),
                                labels = FALSE, include.lowest = TRUE))
ord <- data.frame(x4 = cut3(hs$x4), x5 = cut3(hs$x5), x6 = cut3(hs$x6))

model <- "textual =~ x4 + x5 + x6"
ordv  <- c("x4", "x5", "x6")
fit <- magmaan(model, ord, estimator = "DWLS", ordered = ordv)
lav <- cfa(model, data = ord, ordered = ordv)

cat("=== categorical data: ordinal DWLS ===\n")
ok(isTRUE(fit$ordinal),     "magmaan reports an ordinal fit")
ok(fit$converged,           "magmaan converged")
ok(load_match(fit, lav),    "factor loadings vs lavaan (DWLS)")
cat("categorical data: ok\n")
