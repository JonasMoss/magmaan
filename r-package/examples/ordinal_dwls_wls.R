library(magmaan)
library(lavaan)

make_ord_df <- function(n, cuts_by_var, seed = 1L) {
  set.seed(seed)
  p <- length(cuts_by_var)
  z <- matrix(rnorm(n * p), n, p)
  z[, 2] <- 0.55 * z[, 1] + sqrt(1 - 0.55^2) * z[, 2]
  z[, 3] <- 0.40 * z[, 1] + 0.25 * z[, 2] + sqrt(0.78) * z[, 3]
  z[, 4] <- 0.30 * z[, 1] + 0.20 * z[, 2] + sqrt(0.87) * z[, 4]
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_len(p)) {
    out[[paste0("x", j)]] <- ordered(cut(z[, j],
                                         c(-Inf, cuts_by_var[[j]], Inf),
                                         labels = FALSE))
  }
  out
}

model <- "f =~ x1 + x2 + x3 + x4"
ordered <- paste0("x", 1:4)
df <- make_ord_df(360, list(c(-0.70, 0.35), c(-0.55, 0.60),
                            c(-0.85, 0.20), c(-0.45, 0.75)),
                  seed = 11L)

m <- model_spec(model, ordered = ordered, parameterization = "delta")
d <- data_ordinal_stats_from_df(df, m)
lavaan_wls <- cfa(model, data = df, ordered = ordered,
                  estimator = "WLS", parameterization = "delta")
lavaan_samp <- lavInspect(lavaan_wls, "sampstat")

stopifnot(max(abs(d$thresholds[[1]] - as.numeric(lavaan_samp$th))) < 5e-8)
stopifnot(max(abs(d$R[[1]] - unname(as.matrix(lavaan_samp$cov)))) < 5e-4)
stopifnot(max(abs(d$W_wls[[1]] %*% d$NACOV[[1]] - diag(nrow(d$NACOV[[1]])))) < 1e-8)
stopifnot(identical(dimnames(d$R[[1]]), list(ordered, ordered)))
stopifnot(identical(d$ov_names[[1]], ordered))
stopifnot(identical(d$ordered, ordered))
stopifnot(identical(attr(m$partable, "magmaan.parameterization"), "delta"))
stopifnot(identical(d$threshold_ov[[1]], rep(seq_along(ordered), each = 2L)))
stopifnot(identical(d$threshold_level[[1]], rep(1:2, times = length(ordered))))
moments <- c(d$thresholds[[1]], d$R[[1]][lower.tri(d$R[[1]])])
stopifnot(length(moments) == nrow(d$NACOV[[1]]))

fit_dwls <- fit_dwls_ordinal(
  m, d, lbfgsb = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
fit_wls <- fit_wls_ordinal(
  m, d, lbfgsb = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))

stopifnot(isTRUE(fit_dwls$ordinal), identical(fit_dwls$estimator, "DWLS"))
stopifnot(isTRUE(fit_wls$ordinal), identical(fit_wls$estimator, "WLS"))
stopifnot(length(fit_dwls$theta) == sum(parTable(lavaan_wls)$free > 0L))

lavaan_dwls <- cfa(model, data = df, ordered = ordered,
                   estimator = "DWLS", parameterization = "delta")
chisq_dwls <- fit_dwls$ntotal * fit_dwls$fmin
chisq_wls <- fit_wls$ntotal * fit_wls$fmin
stopifnot(abs(chisq_dwls - fitMeasures(lavaan_dwls, "chisq")) < 0.08)
stopifnot(abs(chisq_wls - fitMeasures(lavaan_wls, "chisq")) < 0.08)

bad <- df
bad$x1 <- ordered(as.integer(bad$x1), levels = 1:4)
bad <- bad[as.integer(bad$x1) != 4L, , drop = FALSE]
err <- tryCatch(data_ordinal_stats_from_df(bad, m),
                error = function(e) conditionMessage(e))
stopifnot(grepl("empty ordinal category", err, fixed = TRUE))

cat("ordinal data_ordinal_stats_from_df()/DWLS/WLS workflow: ok\n")
