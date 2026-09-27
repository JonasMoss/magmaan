#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
for(f in c('designs.R','fit.R','start_design.R','expanded_designs.R')) source(file.path(here,'R',f))
core <- magmaanlab::magmaan_core
# The broader extent includes unit diagonal correlations, adding a floor of one.
# It must otherwise reduce to the original two-factor extent (cutoff remains 10).
spec <- magmaanlab::model_spec(model_syntax)
m <- list(S=list(design_sigma(designs_all()$ernst)),nobs=100)
for(theta in list(spectral_start(spec,m),random_start(spec,m,193))) {
 ev <- core$evaluate_at(spec$partable,m,theta,estimator='ML')
 sigma <- core$model_implied(ev)$sigma[[1]]
 stopifnot(abs(max(1,standardized_extent(ev$partable,sigma))-expanded_extent(ev$partable,sigma)) < 1e-12)
}
for(d in expanded_designs()) {
 spec <- magmaanlab::model_spec(d$syntax); m <- list(S=list(d$sigma),nobs=100)
 scales <- rep(c(.1,2,7),length.out=ncol(d$sigma)); m2 <- m; m2$S[[1]] <- d$sigma*outer(scales,scales)
 a <- expanded_recipes(spec,m); b <- expanded_recipes(spec,m2)
 for(name in setdiff(names(a),'canonical')) {
  ev <- core$evaluate_at(spec$partable,m,a[[name]],estimator='ML')
  ev2 <- core$evaluate_at(spec$partable,m2,b[[name]],estimator='ML')
  s <- core$model_implied(ev)$sigma[[1]]; s2 <- core$model_implied(ev2)$sigma[[1]]
  stopifnot(ev$diagnostics$admissibility$implied_sigma_pd,is.finite(expanded_extent(ev$partable,s)),
    abs(ev$fmin-ev2$fmin)<1e-10,max(abs(s2-s*outer(scales,scales)))<1e-9)
  target <- expanded_marker(d)(ev$partable,m)
  trans <- magmaanlab::frontier_reidentify(ev$partable,target,pole_tol=0)
  check <- core$evaluate_at(target$partable,m,trans$theta,estimator='ML')
  stopifnot(abs(check$fmin-ev$fmin)<1e-10,
    abs(expanded_extent(check$partable,core$model_implied(check)$sigma[[1]])-expanded_extent(ev$partable,s))<1e-10)
 }
}
cat('Expanded PD, unequal-unit, marker-translation and extent compatibility checks passed.\n')
