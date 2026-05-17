## magmaan R bindings - complete-data observed-exogenous (`fixed.x`) boundary.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/fixed_x.R
##
## In lavaan-style SEM, observed predictors on the right-hand side of a
## regression are treated as fixed exogenous variables by default. That means
## their variances/covariances are carried as sample bookkeeping, not estimated
## model parameters. Setting `fixed_x = FALSE` makes those observed exogenous
## moments part of the model, so they appear in theta and in the free rows of
## the partable.

suppressMessages({ library(magmaan); library(lavaan) })

mok <- function(a, b, tol = 1e-4)
  if (isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)),
                       tolerance = tol))) "ok" else "MISMATCH"

hs <- lavaan::HolzingerSwineford1939
df <- as.data.frame(hs[c("x1", "x2", "x3")])
model <- "x1 ~ x2 + x3"

fit_fixed <- magmaan(model, df, estimator = "ML",
                     fixed_x = TRUE, se = "none", test = "none")
fit_random <- magmaan(model, df, estimator = "ML",
                      fixed_x = FALSE, se = "none", test = "none")

lav_fixed <- sem(model, data = df, fixed.x = TRUE)
lav_random <- sem(model, data = df, fixed.x = FALSE)

fixed_rows <- fit_fixed$partable[
  fit_fixed$partable$lhs %in% c("x2", "x3") &
    fit_fixed$partable$op == "~~" &
    fit_fixed$partable$rhs %in% c("x2", "x3"), ]
random_rows <- fit_random$partable[
  fit_random$partable$lhs %in% c("x2", "x3") &
    fit_random$partable$op == "~~" &
    fit_random$partable$rhs %in% c("x2", "x3"), ]

cat("=== fixed_x = TRUE: observed predictors remain fixed exogenous ===\n")
cat(sprintf("  npar = %d; lavaan npar = %d\n",
            fit_fixed$npar, length(coef(lav_fixed))))
cat(sprintf("  theta vs lavaan coef(): %s\n", mok(fit_fixed$theta, coef(lav_fixed))))
print(fit_fixed$partable[, c("lhs", "op", "rhs", "free", "exo", "est")],
      row.names = FALSE, digits = 4)
cat("\n")

stopifnot(identical(fit_fixed$npar, 3L),
          mok(fit_fixed$theta, coef(lav_fixed)) == "ok",
          all(fixed_rows$free == 0L),
          all(fixed_rows$exo == 1L),
          all(is.nan(fixed_rows$est)))

cat("=== fixed_x = FALSE: observed predictor moments are estimated ===\n")
cat(sprintf("  npar = %d; lavaan npar = %d\n",
            fit_random$npar, length(coef(lav_random))))
cat(sprintf("  theta vs lavaan coef(): %s\n", mok(fit_random$theta, coef(lav_random))))
print(fit_random$partable[, c("lhs", "op", "rhs", "free", "exo", "est")],
      row.names = FALSE, digits = 4)
cat("\n")

stopifnot(identical(fit_random$npar, 6L),
          mok(fit_random$theta, coef(lav_random)) == "ok",
          all(random_rows$free > 0L),
          all(random_rows$exo == 0L))

cat("fixed.x complete-data workflow: ok\n")
