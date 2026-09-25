#!/usr/bin/env Rscript
# Bounded before/after inference timings. No population simulation is run.
suppressPackageStartupMessages(library(magmaanlab))
set.seed(20260918)
measure <- function(fun, calls = 15L) {
  invisible(fun())
  start <- proc.time()[['elapsed']]
  for (i in seq_len(calls)) invisible(fun())
  1000 * (proc.time()[['elapsed']] - start) / calls
}
rows <- list()
for (p in c(10L,20L)) {
  n <- 400L
  eta <- matrix(rnorm(n*2),n,2) %*% chol(matrix(c(1,.5,.5,1),2))
  X <- .7 * eta[,rep(1:2,each=p/2)] + sqrt(.51)*matrix(rnorm(n*p),n,p)
  colnames(X) <- paste0('x',seq_len(p))
  syntax <- paste(paste('f1 =~',paste(colnames(X)[seq_len(p/2)],collapse=' + ')),
                  paste('f2 =~',paste(colnames(X)[p/2+seq_len(p/2)],collapse=' + ')),sep='\n')
  for (estimator in c('ML','FIML','ML2S')) {
    data <- X
    if (estimator != 'ML') data[matrix(runif(n*p)<.1,n,p)] <- NA
    fit <- fit_model(syntax,as.data.frame(data),estimator=estimator,meanstructure=TRUE,
                   control=list(max_iter=3000L,ftol=1e-12,gtol=1e-8))
    raw <- if(estimator=='ML') data else NULL
    context <- prepare_inference(fit,raw)
    components <- score_components(context)
    projected <- project_scores(components,retain_rows=TRUE)
    reference <- score_spectrum(projected)
    legacy <- function() {
      z <- global_score_flip_test(fit,raw,n_flips=1L,seed=1)
      calibrate_quadratic(quadratic_reference(z$statistic_effective,z$df,z$eigenvalues),
                          c('sb','peba2','peba4'))
    }
    current <- function() calibrate_quadratic(score_spectrum(project_scores(score_components(context))),
                                               c('sb','peba2','peba4'))
    stopifnot(max(abs(current()$p_value-legacy()$p_value)) < 1e-7)
    stages <- list(context=function()prepare_inference(fit,raw),legacy=legacy,
      components=function()score_components(context),projection=function()project_scores(components),
      spectrum=function()score_spectrum(projected),
      calibration=function()calibrate_quadratic(reference,c('sb','peba2','peba4')),
      current=current)
    for (phase in names(stages)) rows[[length(rows)+1L]] <- data.frame(
      estimator=estimator,p=p,n=n,phase=phase,ms=measure(stages[[phase]]))
  }
}
out <- do.call(rbind,rows)
print(out,row.names=FALSE)
args <- commandArgs(TRUE)
if(length(args)) write.csv(out,args[1],row.names=FALSE)
