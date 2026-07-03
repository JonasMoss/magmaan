set.seed(5101)

model <- "
  f1 =~ x1 + x2 + x3 + x4
  f2 =~ y1 + y2 + y3 + y4
  f2 ~ f1
"

ov <- c(paste0("x", 1:4), paste0("y", 1:4))
loadings <- c(.80, .75, .70, .65, .78, .74, .68, .62)
Lambda <- matrix(0, 8, 2)
Lambda[1:4, 1] <- loadings[1:4]
Lambda[5:8, 2] <- loadings[5:8]
Phi <- matrix(c(1, .5, .5, 1), 2, 2)
Sigma <- Lambda %*% Phi %*% t(Lambda) + diag(1 - loadings^2)
colnames(Sigma) <- rownames(Sigma) <- ov

n <- 400
X <- matrix(rnorm(n * length(ov)), n, length(ov)) %*% chol(Sigma)
colnames(X) <- ov
dat <- as.data.frame(X)

fit_ts <- magmaan::sam(model, dat, se = "twostep")
fit_rb <- magmaan::sam(model, dat, se = "twostep.robust")

stopifnot(inherits(fit_ts, "magmaan_sam_fit"))
stopifnot(inherits(fit_rb, "magmaan_sam_fit"))
stopifnot(identical(fit_rb$estimator, "SAM"))
stopifnot(length(fit_rb$theta) == fit_rb$npar)
stopifnot(!is.null(fit_rb$vcov), !is.null(fit_rb$se))
stopifnot(all.equal(stats::vcov(fit_rb), fit_rb$vcov, tolerance = 0))
stopifnot(is.matrix(fit_rb$sam$VETA))
stopifnot(is.matrix(fit_rb$sam$mapping_matrix))
stopifnot(length(fit_rb$sam$measurement) == 2L)

pt <- fit_rb$partable
path_row <- which(pt$lhs == "f2" & pt$op == "~" & pt$rhs == "f1")
stopifnot(length(path_row) == 1L)
stopifnot(is.finite(pt$est[[path_row]]))
stopifnot(is.finite(fit_rb$se[[pt$free[[path_row]]]]))

if (requireNamespace("lavaan", quietly = TRUE)) {
  lav <- try(
    lavaan::sam(model, data = dat, sam.method = "local",
                se = "twostep.robust"),
    silent = TRUE
  )
  if (!inherits(lav, "try-error")) {
    pe <- lavaan::parameterEstimates(lav)
    lav_row <- which(pe$lhs == "f2" & pe$op == "~" & pe$rhs == "f1")
    if (length(lav_row) == 1L) {
      stopifnot(abs(pt$est[[path_row]] - pe$est[[lav_row]]) < 1e-3)
      stopifnot(abs(fit_rb$se[[pt$free[[path_row]]]] - pe$se[[lav_row]]) < 5e-3)
    }
  }
}

cat("SAM R binding example completed\n")
