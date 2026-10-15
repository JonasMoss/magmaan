#!/usr/bin/env Rscript
# Retained production losses are development data; this diagnostic decides no default.
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('PSD retained-loss diagnostic: --limit N (smoke), --run-id ID, --workers N\nDefault: all test losses, diagnose-2026-10-15, two workers. Uses read-only merge-33.13 library.\n')
  quit()
}
arg <- function(k, d) { i <- match(paste0('--', k), args); if (is.na(i)) d else args[i+1L] }
Sys.setenv(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
script <- sub('^--file=', '', commandArgs()[grepl('^--file=', commandArgs())][1])
root <- normalizePath(file.path(dirname(script), '..'))
for (f in c('families.R', 'fits.R', 'psd_factor_families.R')) source(file.path(root, 'R', f))
# Reuse the production fit function verbatim, without executing its runner.
production <- parse(file.path(root, 'scripts/psd_factor.R'))
fit_definition <- Filter(function(e) is.call(e) && identical(e[[1]], as.name('<-')) && identical(e[[2]], as.name('fit_call')), as.list(production))
stopifnot(length(fit_definition) == 1L)
eval(fit_definition[[1]])
production_fit <- fit_call
prod_dir <- file.path(root, 'results/psd-factor/production-2026-10-15')
losses <- read.csv(file.path(prod_dir, 'losses.csv'))
losses <- losses[losses$role == 'test', ]
stopifnot(!anyDuplicated(losses$key))
limit <- as.integer(arg('limit', nrow(losses)))
losses <- head(losses, limit)
workers <- as.integer(arg('workers', 2)); stopifnot(workers >= 1, workers <= 2)
out <- file.path(root, 'results/psd-factor', arg('run-id', 'diagnose-2026-10-15'))
dir.create(file.path(out, 'raw'), recursive=TRUE, showWarnings=FALSE)
sources <- c(script, file.path(root, c('scripts/psd_factor.R', 'R/families.R', 'R/fits.R', 'R/psd_factor_families.R')), file.path(prod_dir, c('metadata.csv','losses.csv')))
pkg_files <- unlist(lapply(c('magmaanlab','magmaan'), function(p) list.files(find.package(p), pattern='[.](so|rdb)$', recursive=TRUE, full.names=TRUE)))
fingerprint <- paste(unname(tools::md5sum(c(sources,pkg_files))), collapse=';')
meta <- data.frame(key=c('purpose','command','git_commit','source_package_input_md5','library','R_version','magmaanlab_version','magmaan_version','draws','seed_base','budget_control'), value=c('retained development losses; decides nothing',paste(commandArgs(),collapse=' '),system2('git',c('rev-parse','HEAD'),stdout=TRUE),fingerprint,find.package('magmaanlab'),R.version.string,as.character(packageVersion('magmaanlab')),as.character(packageVersion('magmaan')),nrow(losses),2026101501,'control=list(nlopt=list(max_eval=20000L))'))
meta_path <- file.path(out,'metadata.csv')
if (file.exists(meta_path)) {
 old <- read.csv(meta_path); stopifnot(identical(old$value[old$key=='source_package_input_md5'], fingerprint))
} else write.csv(meta,meta_path,row.names=FALSE)
pops <- psd_factor_populations()
run_draw <- function(i) {
 row <- losses[i,]; path <- file.path(out,'raw',sprintf('%03d.csv',i))
 if (file.exists(path)) return(read.csv(path))
 p <- Filter(function(p) p$key == row$population_square, pops); stopifnot(length(p)==1); p <- p[[1]]
 fm <- Filter(function(f) f$key == row$model_square,p$models); stopifnot(length(fm)==1); fm <- fm[[1]]
 n <- row$N_square; rep <- row$replication_square; est <- row$estimator
 seed <- draw_seed(2026101501,p$key,n,rep); stopifnot(seed == row$seed_square)
 d <- draw_moments(p,n,seed)$X
 if (est=='FIML') {set.seed(draw_seed(2026101501,paste0(p$key,'_mcar'),n,rep)); d[matrix(runif(n*p$p)<.15,n,p$p)] <- NA_real_}
 model <- build_model(fm); input <- d
 if (est=='DWLS') {
  for(j in seq_len(p$p)) d[[j]] <- ordered(cut(d[[j]],c(-Inf,qnorm(c(.2,.4,.6,.8))*sqrt(p$Sigma[j,j]),Inf),labels=FALSE),levels=1:5)
  input <- magmaanlab::magmaan_core$data_ordinal_stats_from_df(d,model,ordered=names(d))
 }
 start <- NULL
 if(row$family=='F4') {
  base <- production_fit(model,input,est,'triangular'); start <- base$start$theta; pt <- base$partable
  ix <- which(pt$lhs=='f1' & pt$op=='~~' & pt$rhs %in% paste0('f',1:3)); stopifnot(length(ix)==3,all(pt$free[ix]>0))
  start[pt$free[ix]] <- ifelse(pt$rhs[ix]=='f1',1e-6,0)
 }
 rows <- lapply(c('triangular','square','square_budget','square_reset'),function(arm) {
  error <- ''; evaluations <- 0; fit <- NULL; attempts <- 0L; current_start <- start; t <- proc.time()[[3]]
  repeat {
   attempts <- attempts+1L
   fit <- tryCatch({
    if(arm!='square_budget') production_fit(model,input,est,if(arm=='triangular') 'triangular' else 'square',current_start)
    else {
     fun <- switch(est,ML=magmaanlab::frontier_fit_ml_psd,FIML=magmaanlab::frontier_fit_fiml_psd,ULS=magmaanlab::frontier_fit_uls_psd,DWLS=magmaanlab::frontier_fit_ordinal_psd)
     ctl <- list(nlopt=list(max_eval=20000L)); if(!is.null(start)) ctl$start <- start
     fun(model,input,psd_factor='square',control=ctl)
    }
   },error=function(e) {error <<- conditionMessage(e); NULL})
   if(!is.null(fit)) evaluations <- evaluations+fit$f_evals
   if(arm!='square_reset' || is.null(fit) || isTRUE(fit$converged) || attempts==3L) break
   # Returned free estimates use the same user chart as the F4 numeric start.
   current_start <- fit$theta
   if(is.null(current_start)) stop('Returned theta unavailable: Needs decision')
  }
  data.frame(key=row$key,family=row$family,estimator=est,population=p$key,model=fm$key,N=n,replication=rep,seed=seed,arm=arm,converged=!is.null(fit)&&isTRUE(fit$converged),fmin=if(is.null(fit)) NA_real_ else fit$fmin,objective_evaluations=evaluations,seconds=proc.time()[[3]]-t,attempts=attempts,error=error,reproduction_mismatch=if(arm=='square') is.null(fit)||!identical(isTRUE(fit$converged),row$converged_square)||fit$f_evals!=row$objective_evaluations_square else FALSE)
 })
 result <- do.call(rbind,rows); successful <- result$fmin[result$converged & is.finite(result$fmin)]
 best <- if(length(successful)) min(successful) else NA_real_
 result$best_known <- result$converged & is.finite(result$fmin) & !is.na(best) & abs(result$fmin-best)<=1e-6*(1+abs(best))
 tmp <- paste0(path,'.tmp'); write.csv(result,tmp,row.names=FALSE); stopifnot(file.rename(tmp,path)); result
}
all_rows <- list()
for(first in seq.int(1L,nrow(losses),by=workers)) {
 ids <- first:min(nrow(losses),first+workers-1L)
 x <- parallel::mclapply(ids,run_draw,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
 if(any(vapply(x,inherits,FALSE,'try-error'))) stop('Draw regeneration/fit failed: Needs decision')
 all_rows <- c(all_rows,x); cat('Completed',max(ids),'of',nrow(losses),'draws\n'); flush.console()
}
rows <- do.call(rbind,all_rows)
summary <- do.call(rbind,lapply(split(rows,interaction(rows$family,rows$estimator,rows$arm,drop=TRUE)),function(x) data.frame(family=x$family[1],estimator=x$estimator[1],arm=x$arm[1],draws=nrow(x),successes=sum(x$converged),best_known=sum(x$best_known),median_evaluations=median(x$objective_evaluations),total_evaluations=sum(x$objective_evaluations),median_seconds=median(x$seconds),total_seconds=sum(x$seconds))))
write.csv(rows,file.path(out,'per_draw.csv'),row.names=FALSE)
write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE)
cat('Reproduction mismatches:',sum(rows$reproduction_mismatch),'\nOutputs:',out,'\n')
