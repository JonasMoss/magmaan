fit_sample_stats <- function(fit) {
  if (inherits(fit, "magmaan_data")) {
    return(list(S = fit$S, nobs = fit$nobs, mean = fit$mean))
  }
  list(S = fit$S, nobs = fit$nobs, mean = fit$sample_mean)
}
