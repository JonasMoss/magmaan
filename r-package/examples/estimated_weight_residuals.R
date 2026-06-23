## Estimated-weight ("complete-sandwich") standardized residuals.
##
## lav_residuals(fit, estimated_weight = TRUE, data = ...) routes the residual
## SE/z and the $summary inference through the Hall-Inoue infinitesimal-jackknife
## residual ACOV, which carries the data-dependent-weight influence IF(W-hat).
## lavaan only ever uses the normal-theory projection + Gamma_NT for residual
## SEs, so this is a frontier extension for the continuous least-squares
## estimators (GLS/WLS/ULS) whose second-stage weight is estimated from the data.
## The point residuals are unchanged; only their sampling ACOV (hence SE/z and
## the close-fit interval) reflects the estimated weight.

suppressMessages({library(lavaan); library(magmaan)})

## Heavy-tailed (t_6) 4-indicator data with one true residual covariance the
## one-factor model cannot reproduce, so there is real, non-normal misfit.
set.seed(20260623)
n <- 3000; df <- 6
Sig <- local({
  lam <- c(1.0, 0.8, 0.7, 0.9); th <- c(0.6, 0.7, 0.8, 0.5)
  S <- tcrossprod(lam) * 1.4 + diag(th)
  S[1, 2] <- S[2, 1] <- S[1, 2] + 0.18
  S
})
L <- t(chol(Sig))
X <- t(vapply(seq_len(n), function(i) {
  as.numeric(sqrt((df - 2) / df) * (L %*% rnorm(4)) / sqrt(rchisq(1, df) / df))
}, numeric(4)))
colnames(X) <- paste0("x", 1:4)
dat <- as.data.frame(X)

model <- "f =~ x1 + x2 + x3 + x4"
fit <- magmaan(model, dat, estimator = "GLS")
stopifnot(identical(fit$estimator, "GLS"), isTRUE(fit$converged))

nt <- lav_residuals(fit)                                       # NT projection
ew <- lav_residuals(fit, estimated_weight = TRUE, data = dat)  # complete sandwich

## The point residuals are identical (same theta-hat); only the ACOV changes.
stopifnot(max(abs(nt$cov_cor[[1]] - ew$cov_cor[[1]])) < 1e-10)
## The estimated-weight SE genuinely moves under non-normality.
stopifnot(max(abs(nt$cov_se[[1]] - ew$cov_se[[1]])) > 1e-4)
## The $summary is a data frame with the SRMR-family rows labelled in row.names
## (as in lavaan); index with [row, "cov"]. Both share the deterministic SRMR
## but differ in its SE / CI.
ntc <- nt$summary[[1]]; ewc <- ew$summary[[1]]
stopifnot(abs(ntc["srmr", "cov"] - ewc["srmr", "cov"]) < 1e-8)
stopifnot(abs(ntc["srmr.se", "cov"] - ewc["srmr.se", "cov"]) > 1e-6)

cat("SRMR (shared):", round(ewc["srmr", "cov"], 4), "\n")
cat("srmr.se  NT vs estimated-weight:",
    round(ntc["srmr.se", "cov"], 4), "vs", round(ewc["srmr.se", "cov"], 4), "\n")
cat("usrmr close-fit CI  NT  :",
    round(c(ntc["usrmr.ci.lower", "cov"], ntc["usrmr.ci.upper", "cov"]), 4), "\n")
cat("usrmr close-fit CI  est.:",
    round(c(ewc["usrmr.ci.lower", "cov"], ewc["usrmr.ci.upper", "cov"]), 4), "\n")

## Guards: needs raw data; the second-stage weight must be estimated.
stopifnot(is.character(tryCatch(
  lav_residuals(fit, estimated_weight = TRUE), error = conditionMessage)))
fit_ml <- magmaan(model, dat, estimator = "ML")
stopifnot(is.character(tryCatch(
  lav_residuals(fit_ml, estimated_weight = TRUE, data = dat),
  error = conditionMessage)))

cat("estimated-weight residuals: ok\n")
