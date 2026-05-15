library(magmaan)
library(lavaan)

data("HolzingerSwineford1939", package = "lavaan")
df <- HolzingerSwineford1939

model <- "visual =~ x1 + a*x2 + b*x3
          ab := a * b
          sum_ab := ab + a
          plab := .p2. * .p3.
          fixed_plus := .p1. + a"

lavaan_model <- "visual =~ x1 + a*x2 + b*x3
                 ab := a * b
                 sum_ab := ab + a
                 fixed_plus := 1 + a"

fit <- magmaan(model, df, estimator = "ML")
info <- infer_information_expected(fit)
vc <- infer_vcov(info, fit)
defs <- compute_defined(model, fit, vc)

lav <- cfa(lavaan_model, data = df)
lav_defs <- parameterEstimates(lav)
lav_defs <- lav_defs[lav_defs$op == ":=", c("lhs", "est", "se")]

by_name <- function(x, name, col) x[[col]][match(name, x$lhs)]
close <- function(a, b, tol = 1e-6) isTRUE(all.equal(a, b, tolerance = tol))

stopifnot(close(by_name(defs, "ab", "est"), by_name(lav_defs, "ab", "est")))
stopifnot(close(by_name(defs, "ab", "se"), by_name(lav_defs, "ab", "se")))
stopifnot(close(by_name(defs, "sum_ab", "est"), by_name(lav_defs, "sum_ab", "est")))
stopifnot(close(by_name(defs, "sum_ab", "se"), by_name(lav_defs, "sum_ab", "se")))

stopifnot(close(by_name(defs, "plab", "est"), by_name(lav_defs, "ab", "est")))
stopifnot(close(by_name(defs, "plab", "se"), by_name(lav_defs, "ab", "se")))
stopifnot(close(by_name(defs, "fixed_plus", "est"), by_name(lav_defs, "fixed_plus", "est")))
stopifnot(close(by_name(defs, "fixed_plus", "se"), by_name(lav_defs, "fixed_plus", "se")))

cat("compute_defined() chained and .pN. workflow: ok\n")
