# Complete fitting in normalized units, with returned estimates in caller units.
suppressPackageStartupMessages(library(magmaanlab))
model <- model_spec('f =~ x1 + a*x2 + a*x3 + b*x4\na == b + 0.2')
scaled <- model_spec('f =~ x1 + u*x2 + v*x3 + w*x4\nu == 10000*v\nu == 50*w + 20')
lambda <- c(1, .8, .8, .6)
S <- tcrossprod(lambda) + diag(c(.4, .5, .6, .7))
S[1, 4] <- S[4, 1] <- S[1, 4] + .01
units <- c(1, 100, .01, 2)
sample <- list(S = list(S), nobs = 400L)
other <- list(S = list(S * tcrossprod(units)), nobs = 400L)
control <- list(start = 'fabin3', nlopt = list(max_eval = 20000L,
    ftol_rel = 1e-14, xtol_rel = 1e-12))
for (psd in c(FALSE, TRUE)) {
  run <- if (psd) frontier_fit_ml_psd else magmaan_core$fit_ml
  a <- run(model, sample, control = control)
  b <- run(scaled, other, control = control)
  stopifnot(a$sample_normalized, b$sample_normalized, a$converged, b$converged,
            abs(a$fmin - b$fmin) < 1e-8)
  # Reported free estimates and saved starts retain the supplied model's units.
  p <- a$partable
  scale <- rep(1, length(a$theta))
  for (i in which(p$free > 0)) {
    if (p$op[i] == '=~') scale[p$free[i]] <- units[match(p$rhs[i], paste0('x', 1:4))]
    if (p$op[i] == '~~' && p$lhs[i] != 'f')
      scale[p$free[i]] <- units[match(p$lhs[i], paste0('x', 1:4))]^2
  }
  stopifnot(max(abs(b$theta / scale - a$theta)) < 1e-4)
  aa <- frontier_newton_accuracy(a, psd = psd)
  bb <- frontier_newton_accuracy(b, psd = psd)
  stopifnot(aa$unit_normalized, bb$unit_normalized, aa$passed, bb$passed)
  explicit <- run(scaled, other, control = list(start = b$theta))
  stopifnot(explicit$sample_normalized,
            identical(as.numeric(explicit$start$theta), as.numeric(b$theta)))
  legacy <- run(model, sample, control = c(control, list(normalize_sample = FALSE)))
  stopifnot(!legacy$sample_normalized)
}
# The fallback uses the same machinery in both stages.
boundary <- list(S = list(matrix(c(1,.7,.7,.7,1,.3,.7,.3,1),3)), nobs = 300L)
f <- frontier_fit_ml_psd_fallback(model_spec('f =~ x1 + x2 + x3'), boundary)
stopifnot(f$converged, f$fallback_used, f$ordinary$fit$sample_normalized,
          f$psd$fit$sample_normalized)
cat('ML/PSD normalized fitting, affine constraints and fallback: passed\n')
