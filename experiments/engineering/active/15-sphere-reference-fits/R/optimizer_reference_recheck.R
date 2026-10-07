# Recheck a frozen raw-unit oracle without refitting any magmaan endpoint.
# Usage: Rscript R/optimizer_reference_recheck.R RUN_ID
args <- commandArgs(TRUE)
if(length(args)!=1L || !grepl('^[A-Za-z0-9_-]+$',args))stop('provide one owning-leaf run ID')
root <- file.path('results',args)
out <- file.path(root,'oracle_recheck.csv')
if(file.exists(out))stop('preserve the existing recheck')
files <- list.files(root,pattern='^lavaan_fresh_.*\\.rds$',full.names=TRUE)
if(!length(files))stop('missing retained raw-unit references')
rows <- lapply(files,function(path) {
  raw <- readRDS(path);x <- lavaan::lavInspect(raw,'data')
  scale <- as.numeric(sub('.*_scale_([0-9.]+)\\.rds$','\\1',path))
  ref <- suppressWarnings(lavaan::sem(lavaan::parTable(raw),data=as.data.frame(x/scale),
    meanstructure=TRUE,fixed.x=FALSE))
  m <- lavaan::lavInspect(ref,'implied');Sigma <- m$cov*scale^2;mu <- m$mean*scale
  S <- stats::cov(x)*(nrow(x)-1)/nrow(x);delta <- colMeans(x)-mu
  f <- .5*(as.numeric(determinant(Sigma,logarithm=TRUE)$modulus)-
    as.numeric(determinant(S,logarithm=TRUE)$modulus)+sum(diag(solve(Sigma,S)))+
    drop(crossprod(delta,solve(Sigma,delta)))-ncol(x))
  data.frame(tag=gsub('^lavaan_|\\.rds$','',basename(path)),scale=scale,
    source_md5=unname(tools::md5sum(path)),raw_fmin=unname(lavaan::fitMeasures(raw,'fmin')),
    reference_fmin=unname(lavaan::fitMeasures(ref,'fmin')),transported_objective=f,
    converged=lavaan::lavInspect(ref,'converged'),iterations=lavaan::lavInspect(ref,'optim')$iterations,
    lavaan_version=as.character(utils::packageVersion('lavaan')))
})
rows <- do.call(rbind,rows)
stopifnot(all(rows$converged),all(abs(rows$transported_objective-rows$reference_fmin)<1e-8))
fits <- read.csv(file.path(root,'fits.csv'))
rows$native_gap <- vapply(rows$tag,function(tag) {
  z <- fits[fits$family==tag,];stopifnot(nrow(z)==4L,all(z$fit_passed))
  max(abs(z$fmin-rows$reference_fmin[rows$tag==tag]))
},numeric(1))
stopifnot(all(rows$native_gap<1e-7))
write.csv(rows,out,row.names=FALSE,na='')
print(rows)
