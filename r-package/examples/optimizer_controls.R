# Named controls must reach the same backend settings as the legacy spelling.
suppressPackageStartupMessages(library(magmaan))
set.seed(20260921)
z <- rnorm(500)
x <- as.data.frame(sapply(1:4, function(j) z + rnorm(500)))
names(x) <- paste0('y', 1:4)
model <- model_spec('f =~ y1 + y2 + y3 + y4')
stats <- df_to_data(x, model, scaling='n')
for (backend in c('nlopt-lbfgs', 'nlopt-slsqp', 'port', 'port-nls')) {
  legacy <- list(ftol=1e-12, gtol=1e-10)
  named <- if (startsWith(backend, 'nlopt')) {
    list(ftol=.1, gtol=.1, nlopt=list(ftol_rel=1e-12, xtol_rel=1e-10))
  } else list(ftol=.1, port=list(rel_f_tol=1e-12))
  a <- magmaan_core$fit_gls(model, stats, optimizer=backend, control=legacy)
  b <- magmaan_core$fit_gls(model, stats, optimizer=backend, control=named)
  stopifnot(isTRUE(all.equal(a$theta, b$theta, tolerance=0)))
}
fails <- function(control) {
  inherits(try(magmaan_core$fit_gls(model, stats, optimizer='nlopt-slsqp',
                                  control=control), silent=TRUE), 'try-error')
}
stopifnot(fails(list(nlopt=list(xtlo_rel=1e-10))),
          fails(list(nlopt=list(xtol_rel=NA_real_))),
          fails(list(nlopt=list(max_eval=1.5))),
          fails(list(nlopt=list(tolg=1e-10))),
          fails(list(nlopt=setNames(list(1e-10, 1e-12), c('ftol_rel','ftol_rel')))))
cat('Optimizer control R checks passed.\n')
