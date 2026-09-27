#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script)); args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/normalization_revisit.R [--smoke] [--run-id NAME] [--workers 4]\n',
      'Paired original-unit/normalized ML, direct PSD and two-stage PSD on the saved scaling grid.\n',
      'Full run: 1356 cases, 8136 fits; seed 202609290. Smoke: first population per family, seed +1.\n')
  quit(save='no')
}
source(file.path(here,'..','..','_support','R','helpers.R')); set_single_threaded_math()
for (f in c('families.R','fits.R')) source(file.path(here,'R',f))
opt <- function(k,d) if(k %in% args) args[match(k,args)+1L] else d
smoke <- '--smoke' %in% args; run <- opt('--run-id',if(smoke)'smoke'else'paired')
workers <- as.integer(opt('--workers','4'))
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run),!is.na(workers),workers>=1L)
out <- file.path(here,'results','normalization-revisit',run)
if(file.exists(out)) stop('choose a fresh run ID')
pops <- all_populations(); pops <- pops[vapply(pops,function(p)p$role=='test',logical(1))]
if(smoke) pops <- pops[!duplicated(vapply(pops,`[[`,'','family'))]
seed_base <- 202609290L+as.integer(smoke)
tasks <- do.call(rbind,lapply(pops,function(p){
 ns <- population_ns(p);if(smoke)ns<-head(ns,1)
 z <- expand.grid(n=ns,rep=seq_len(if(smoke)1L else 3L))
 data.frame(pop=p$key,z,seed=mapply(draw_seed,seed_base,p$key,z$n,z$rep))
}))
run_fit <- function(model,sample,route,normalized) {
 ctl <- list(start=if(route=='ml')'layered'else'fabin3',
   start_transport=if(route=='ml')'native'else'auto',normalize_sample=normalized,
   max_iter=5000L,nlopt=list(max_eval=5000L,ftol_rel=1e-12,xtol_rel=1e-10),
   coordinate_scaling='information')
 rec <- blank_record(); rec$sample_normalized <- NA; rec$distance <- NA_real_
 t0 <- proc.time()[['elapsed']]
 result <- tryCatch(switch(route,
   ml=magmaanlab::magmaan_core$fit_ml(model,sample,optimizer='nlopt-lbfgs',control=ctl),
   direct=magmaanlab::frontier_fit_ml_psd(model,sample,optimizer='nlopt-slsqp',control=ctl,
     preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6),
   two_stage=magmaanlab::frontier_fit_ml_psd_fallback(model,sample,
     ordinary_optimizer='nlopt-lbfgs',psd_optimizer='nlopt-slsqp',ordinary_control=ctl,
     psd_control=ctl[setdiff(names(ctl),c('start','start_transport'))],
     preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6)),error=function(e)e)
 rec$seconds <- proc.time()[['elapsed']]-t0
 if(inherits(result,'error')) {rec$message<-one_line(conditionMessage(result));return(list(record=rec))}
 fit <- result
 if(route=='two_stage') {
   used <- if(isTRUE(result$fallback_used))result$psd else result$ordinary
   rec$stage <- if(isTRUE(result$fallback_used))paste0('psd:',result$fallback_reason)else'ordinary'
   fit <- used$fit
   if(is.null(fit)) {rec$message<-one_line(paste(used$error$kind,used$error$detail));return(list(record=rec))}
 }
 stopifnot(!is.null(fit$sample_normalized),identical(isTRUE(fit$sample_normalized),normalized))
 rec <- fill_record(rec,fit)
 if(route=='two_stage') rec$certified <- isTRUE(result$converged)
 rec$sample_normalized <- fit$sample_normalized
 rec$distance <- fit$diagnostics$newton_accuracy$distance
 covariance <- magmaanlab::magmaan_core$model_implied(fit)$sigma[[1]]
 covariance <- covariance / sqrt(outer(diag(sample$S[[1]]),diag(sample$S[[1]])))
 list(record=rec,theta=fit$theta,covariance=as.vector(covariance))
}
run_task_revisit <- function(task) {
 pop <- pops[[task$pop]]; moments <- draw_moments(pop,task$n,task$seed)
 rows <- parameters <- covariances <- list()
 for(fm in pop$models) {
   model <- build_model(fm)
   for(tr in c('native','x100','x0.01','mixed')) {
     if(tr=='mixed' && !fm$unit_invariant) next
     sample <- sample_in_units(moments,transform_factors(tr,pop$p),fm$meanstructure)
     for(route in c('ml','direct','two_stage')) for(arm in c('original_units','normalized')) {
       key <- cbind(task,family=pop$family,model=fm$key,chart=if(fm$std_lv)'std_lv'else'marker',
                    transform=tr,route=route,arm=arm)
       z <- suppressWarnings(run_fit(model,sample,route,arm=='normalized'))
       rows[[length(rows)+1L]] <- cbind(key,z$record)
       if(!is.null(z$theta)) parameters[[length(parameters)+1L]] <- cbind(key[rep(1,length(z$theta)),],parameter=seq_along(z$theta),value=z$theta)
       if(!is.null(z$covariance)) covariances[[length(covariances)+1L]] <- cbind(key[rep(1,length(z$covariance)),],element=seq_along(z$covariance),value=z$covariance)
     }
   }
 }
 list(fits=do.call(rbind,rows),parameters=do.call(rbind,parameters),covariances=do.call(rbind,covariances))
}
dir.create(file.path(out,'raw'),recursive=TRUE)
write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed_base,tasks=nrow(tasks),smoke=smoke,
 workers=workers,git_head=magmaan_cache_ref()$git_head,git_dirty=magmaan_cache_ref()$git_dirty,
 library=find.package('magmaanlab'),built=utils::packageDescription('magmaanlab')$Built,
 library_md5=unname(tools::md5sum(system.file('libs','magmaanlab.so',package='magmaanlab'))),
 protocol='criteria/normalization-revisit.md; saved development draws; same current judge both arms',
 starts='ML layered/native; PSD and fallback FABIN3/auto; ordinary L-BFGS/information; PSD SLSQP/diagonal',
 controls='5000 evaluations; ftol_rel=1e-12; xtol_rel=1e-10; floor=feasibility=1e-6'),packages='magmaanlab')
batches <- split(seq_len(nrow(tasks)),ceiling(seq_len(nrow(tasks))/8)); rows <- list(); t0 <- proc.time()[['elapsed']]
for(b in seq_along(batches)) {
 z <- parallel::mclapply(batches[[b]],function(i)run_task_revisit(tasks[i,]),mc.cores=workers,mc.preschedule=TRUE)
 stopifnot(all(vapply(z,function(x)is.list(x)&&!inherits(x,'try-error'),logical(1))))
 for(name in c('fits','parameters','covariances'))
   write.csv(do.call(rbind,lapply(z,`[[`,name)),file.path(out,'raw',sprintf('%s_%03d.csv',name,b)),row.names=FALSE)
 rows[[b]] <- do.call(rbind,lapply(z,`[[`,'fits'))
 elapsed <- proc.time()[['elapsed']]-t0; done <- max(batches[[b]])
 cat(sprintf('%d/%d draws; %.1fs; estimated %.1fs remaining\n',done,nrow(tasks),elapsed,elapsed*(nrow(tasks)/done-1)))
}
x <- do.call(rbind,rows)
x$success <- x$certified & (x$route=='ml' | (!is.na(x$admissible) & x$admissible))
x$extent_flag <- x$certified & !is.na(x$std_extent) & x$std_extent>10
x$screened_success <- x$success & !is.na(x$std_extent) & x$std_extent<=10
write.csv(x,file.path(out,'fits.csv'),row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
