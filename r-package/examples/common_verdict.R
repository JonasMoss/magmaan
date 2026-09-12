# Run against the installed package: Rscript r-package/examples/common_verdict.R
suppressPackageStartupMessages(library(magmaan))
set.seed(20260912)
n <- 1000L
z <- rnorm(n)
x <- as.data.frame(sapply(1:6, function(j) (.55 + .05*j)*z + rnorm(n)))
names(x) <- paste0('y', 1:6)
models <- c(cfa = 'f =~ y1 + y2 + y3 + y4 + y5 + y6',
            linear = paste('i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4 + 1*y5 + 1*y6',
                           's =~ 0*y1 + 1*y2 + 2*y3 + 3*y4 + 4*y5 + 5*y6', sep='\n'))
for (id in names(models)) {
  model <- model_spec(models[[id]])
  stats <- df_to_data(x, model, scaling='n')
  for (backend in c('nlopt-lbfgs', 'port', 'port-nls')) {
    for (fn in c('fit_gls', 'fit_gls_snlls')) {
      fit <- magmaan_core[[fn]](model, stats, optimizer=backend,
                      control=list(max_iter=2000L, ftol=1e-12, gtol=1e-8))
      stopifnot(is.logical(fit$converged), length(fit$converged)==1L,
                isTRUE(fit$converged), fit$verdict$status=='passed',
                fit$verdict$domain=='ambient', fit$verdict$objective=='passed',
                fit$verdict$stationarity=='passed',
                identical(fit$verdict, fit$diagnostics$verdict))
      if (id=='linear' && fn=='fit_gls_snlls') {
        stopifnot(fit$n_nonlinear==0L, !isTRUE(fit$audit$f_consistent),
                  is.finite(fit$verdict$objective_recomputed))
      }
      cat(id, backend, fn, fit$optimizer_status, fit$verdict$status, '\n')
    }
  }
  if (id=='cfa') {
    probe <- magmaan_core$evaluate_at(model, stats, fit$theta * 1.2,
                                     estimator='GLS')
    stopifnot(identical(probe$converged, FALSE),
              probe$verdict$status=='failed',
              probe$verdict$stationarity=='failed',
              probe$verdict$objective=='passed')
  }
}
cat('Common verdict R checks passed.\n')
