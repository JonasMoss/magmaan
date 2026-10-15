#!/usr/bin/env Rscript
# Retained disagreements are development data and cannot decide a default.
args <- commandArgs(TRUE)
if ('--help' %in% args) {
 cat('Usage: Rscript scripts/psd_factor_symmetric_diagnose.R [--limit 4] [--run-id ID] [--workers 2]\nUses read-only lane-g packages; default runs all 247 retained disagreements.\n'); quit()
}
arg <- function(k,d) {i <- match(paste0('--',k),args); if(is.na(i)) d else args[i+1L]}
Sys.setenv(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
script <- sub('^--file=','',commandArgs()[grepl('^--file=',commandArgs())][1])
root <- normalizePath(file.path(dirname(script),'..'))
for(f in c('families.R','fits.R','psd_factor_families.R','psd_factor_pairing.R')) source(file.path(root,'R',f))
# Read function definitions without executing either historical runner. This
# preserves the frozen diagnostic and shares its exact reconstruction, including
# the FIML mask, ordinal cutpoints and F4 pinned start, without duplicating it.
definition <- function(path,name) {
 e <- Filter(function(e) is.call(e) && identical(e[[1]],as.name('<-')) && identical(e[[2]],as.name(name)),as.list(parse(path)))
 stopifnot(length(e)==1L); eval(e[[1]],envir=environment()); get(name)
}
production_fit <- definition(file.path(root,'scripts/psd_factor.R'),'fit_call')
old_draw <- definition(file.path(root,'scripts/psd_factor_diagnose.R'),'run_draw')
expressions <- as.list(body(old_draw))[-1L]
prep_first <- which(vapply(expressions,function(e) is.call(e) && identical(e[[1]],as.name('<-')) && identical(e[[2]],as.name('p')),FALSE))[1]
prep_last <- which(vapply(expressions,function(e) is.call(e) && identical(e[[1]],as.name('<-')) && identical(e[[2]],as.name('rows')),FALSE))[1]-1L
stopifnot(is.finite(prep_first),prep_last>prep_first)
regenerate_draw <- function(row) {
 for(e in expressions[prep_first:prep_last]) eval(e,envir=environment())
 list(model=model,input=input,start=start)
}
stopifnot('symmetric' %in% eval(formals(magmaanlab::frontier_fit_ml_psd)$psd_factor))
prod_dir <- file.path(root,'results/psd-factor/production-2026-10-15')
paths <- list.files(file.path(prod_dir,'raw'),pattern='[.]csv$',full.names=TRUE)
if(!length(paths)) stop('Production raw rows absent: Needs decision')
x <- do.call(rbind,lapply(paths,read.csv,stringsAsFactors=FALSE,colClasses=c(error='character',ordinary_error='character')))
x <- psd_factor_best(x); p <- psd_factor_pair(x); p <- p[p$role=='test',]
expected <- read.csv(file.path(prod_dir,'paired.csv')); expected <- expected[expected$role=='test',]
counts <- do.call(rbind,lapply(split(p,key(p,c('family','estimator'))),function(d) data.frame(family=d$family[1],estimator=d$estimator[1],wins=sum(d$win),losses=sum(d$loss))))
check <- merge(counts,expected,by=c('family','estimator'),suffixes=c('_actual','_expected'))
print(counts,row.names=FALSE)
if(nrow(check)!=nrow(expected) || any(check$wins_actual!=check$wins_expected | check$losses_actual!=check$losses_expected) || sum(p$win)!=180 || sum(p$loss)!=67) stop('Disagreement counts differ: Needs decision')
losses <- read.csv(file.path(prod_dir,'losses.csv')); losses <- losses[losses$role=='test',]
stopifnot(setequal(losses$key,p$key[p$loss]))
draws <- p[p$win | p$loss,]; draws$draw_type <- ifelse(draws$loss,'loss','win')
stopifnot(!anyDuplicated(draws$key))
draws <- head(draws,as.integer(arg('limit',nrow(draws))))
workers <- as.integer(arg('workers',2)); stopifnot(workers>=1,workers<=2,nrow(draws)>0)
out <- file.path(root,'results/psd-factor',arg('run-id','diagnose-symmetric-2026-10-15'))
dir.create(file.path(out,'raw'),recursive=TRUE,showWarnings=FALSE)
sources <- c(script,file.path(root,c('scripts/psd_factor.R','scripts/psd_factor_diagnose.R',paste0('R/',c('families.R','fits.R','psd_factor_families.R','psd_factor_pairing.R')))),file.path(prod_dir,c('metadata.csv','paired.csv','losses.csv')),paths)
pkg_files <- unlist(lapply(c('magmaanlab','magmaan'),function(p) list.files(find.package(p),pattern='[.](so|rdb)$',recursive=TRUE,full.names=TRUE)))
fingerprint <- paste(unname(tools::md5sum(c(sources,pkg_files))),collapse=';')
meta <- data.frame(key=c('purpose','command','git_commit','source_package_input_md5','package_md5','library','library_build_commit','library_code_note','R_version','draws','seed_base'),value=c('retained production disagreements; development only; decides nothing',paste(commandArgs(),collapse=' '),system2('git',c('rev-parse','HEAD'),stdout=TRUE),fingerprint,paste(unname(tools::md5sum(pkg_files)),collapse=';'),find.package('magmaanlab'),'ab6b9a23','lane-g library code equals main with symmetric factor commit 23ed6598',R.version.string,nrow(draws),2026101501))
meta_path <- file.path(out,'metadata.csv')
if(file.exists(meta_path)) {old <- read.csv(meta_path); stopifnot(identical(old$value[old$key=='source_package_input_md5'],fingerprint),as.integer(old$value[old$key=='draws'])==nrow(draws))} else write.csv(meta,meta_path,row.names=FALSE)
pops <- psd_factor_populations()
run_draw <- function(i) {
 row <- draws[i,]; path <- file.path(out,'raw',sprintf('%03d.csv',i))
 if(file.exists(path)) return(read.csv(path,colClasses=c(error='character')))
 prepared <- regenerate_draw(row)
 rows <- lapply(c('triangular','square','symmetric'),function(arm) {
  error <- ''; t <- proc.time()[[3]]
  fit <- tryCatch({
   if(arm!='symmetric') production_fit(prepared$model,prepared$input,row$estimator,arm,prepared$start)
   else {
    fun <- switch(row$estimator,ML=magmaanlab::frontier_fit_ml_psd,FIML=magmaanlab::frontier_fit_fiml_psd,ULS=magmaanlab::frontier_fit_uls_psd,DWLS=magmaanlab::frontier_fit_ordinal_psd)
    a <- list(model=prepared$model,data=prepared$input,psd_factor='symmetric')
    if(!is.null(prepared$start)) a$control <- list(start=prepared$start)
    do.call(fun,a)
   }
  },error=function(e) {error <<- conditionMessage(e); NULL})
  converged <- !is.null(fit)&&isTRUE(fit$converged)
  evaluations <- if(is.null(fit)) NA_integer_ else fit$f_evals
  data.frame(key=row$key,family=row$family,role='test',estimator=row$estimator,draw_type=row$draw_type,arm=arm,converged=converged,fmin=if(is.null(fit)) NA_real_ else fit$fmin,objective_evaluations=evaluations,seconds=proc.time()[[3]]-t,error=error,reproduction_mismatch=if(arm=='symmetric') FALSE else is.null(fit)||converged!=row[[paste0('converged_',arm)]]||evaluations!=row[[paste0('objective_evaluations_',arm)]])
 })
 result <- do.call(rbind,rows); successful <- result$fmin[result$converged & is.finite(result$fmin)]
 best <- if(length(successful)) min(successful) else NA_real_
 result$best <- result$converged & is.finite(result$fmin) & !is.na(best) & abs(result$fmin-best)<=1e-6*(1+abs(best))
 tmp <- paste0(path,'.tmp'); write.csv(result,tmp,row.names=FALSE); stopifnot(file.rename(tmp,path)); result
}
all_rows <- list()
for(first in seq.int(1L,nrow(draws),by=workers)) {
 ids <- first:min(nrow(draws),first+workers-1L)
 z <- parallel::mclapply(ids,run_draw,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
 if(any(vapply(z,inherits,FALSE,'try-error'))) {print(z); stop('Regeneration failed: Needs decision')}
 all_rows <- c(all_rows,z); cat('Completed',max(ids),'of',nrow(draws),'draws\n'); flush.console()
 if(all(vapply(z,function(d) nzchar(d$error[d$arm=='symmetric']),FALSE))) stop('Symmetric errors in entire batch: Needs decision')
}
rows <- do.call(rbind,all_rows)
summary <- do.call(rbind,lapply(split(rows,key(rows,c('family','estimator','draw_type','arm'))),function(d) cbind(d[1,c('family','estimator','draw_type','arm')],data.frame(draws=nrow(d),successes=sum(d$converged),best_known=sum(d$best),median_evaluations=median(d$objective_evaluations),total_evaluations=sum(d$objective_evaluations),median_seconds=median(d$seconds),total_seconds=sum(d$seconds),reproduction_mismatches=sum(d$reproduction_mismatch),errors=sum(nzchar(d$error))))))
paired <- do.call(rbind,lapply(c('triangular','square'),function(base) {
 d <- psd_factor_pair(rows[rows$arm %in% c('symmetric',base),],candidate='symmetric',baseline=base)
 # Historical helper names its candidate/baseline columns square/triangular.
 d$draw_type <- d$draw_type_square
 do.call(rbind,lapply(split(d,key(d,c('family','estimator','draw_type'))),function(z) cbind(z[1,c('family','estimator','draw_type')],data.frame(candidate='symmetric',baseline=base,draws=nrow(z),success_wins=sum(z$success_win),success_losses=sum(z$success_loss),best_wins=sum(z$best_win),best_losses=sum(z$best_loss),wins=sum(z$win),losses=sum(z$loss)))))
}))
for(name in c('summary','paired')) write.csv(get(name),file.path(out,paste0(name,'.csv')),row.names=FALSE)
write.csv(rows,file.path(out,'per_draw.csv'),row.names=FALSE)
print(summary,row.names=FALSE); print(paired,row.names=FALSE)
cat('Outputs:',out,'\n')
