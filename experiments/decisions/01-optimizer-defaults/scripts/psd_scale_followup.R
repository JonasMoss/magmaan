#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'..','..','_support','R','helpers.R')); set_single_threaded_math()
for(f in c('families.R','fits.R')) source(file.path(here,'R',f))
args <- commandArgs(TRUE)
if('--help' %in% args) {
  cat('Usage: Rscript scripts/psd_scale_followup.R --run-id NAME [--smoke] [--workers 4]\n',
      'Paired direct and ordinary-then-PSD fits, explicit FABIN3-auto starts.\n',
      'All test populations, 3 replications per N; smoke first population per family.\n',
      'Seed 202609290 (smoke +1). Re-run against each library build with a new ID.\n')
  quit(save='no')
}
opt <- function(k,d=NULL) if(k %in% args) args[match(k,args)+1L] else d
run <- opt('--run-id'); stopifnot(!is.null(run),grepl('^[a-zA-Z0-9_-]+$',run))
smoke <- '--smoke' %in% args; workers <- as.integer(opt('--workers','4'))
out <- file.path(here,'results','psd-scale',run)
if(file.exists(out)) stop('choose a new run ID')
dir.create(file.path(out,'raw'),recursive=TRUE)
# Override only this diagnostic's arms; historical lane definitions stay intact.
lane_arms <- function(lane) list(direct=list(kind='psd'),two_stage=list(kind='twostage'))
run_arm <- function(arm, model, sample, estimator) {
  ctl <- list(start='fabin3',start_transport='auto',max_iter=5000L,
              nlopt=list(ftol_rel=1e-12,xtol_rel=1e-10,max_eval=5000L),coordinate_scaling='information')
  rec <- blank_record(); t0 <- proc.time()[['elapsed']]
  fit <- tryCatch(if(arm$kind=='psd') magmaanlab::frontier_fit_ml_psd(
    model,sample,optimizer='nlopt-slsqp',control=ctl,preconditioning='diagonal',
    start_eigen_floor=1e-6,feasibility_tol=1e-6) else
    magmaanlab::frontier_fit_ml_psd_fallback(model,sample,
      ordinary_optimizer='nlopt-lbfgs',psd_optimizer='nlopt-slsqp',
      ordinary_control=ctl,psd_control=ctl[setdiff(names(ctl),c('start','start_transport'))],preconditioning='diagonal',
      start_eigen_floor=1e-6,feasibility_tol=1e-6),error=function(e)e)
  rec$seconds <- proc.time()[['elapsed']]-t0
  if(inherits(fit,'error')) {rec$message <- one_line(conditionMessage(fit));return(rec)}
  if(arm$kind=='twostage') {
    used <- if(isTRUE(fit$fallback_used)) fit$psd else fit$ordinary
    rec$stage <- if(isTRUE(fit$fallback_used)) paste0('psd:',fit$fallback_reason) else 'ordinary'
    if(!is.null(used$fit)) rec <- fill_record(rec,used$fit)
    else rec$message <- one_line(paste(used$error$kind,used$error$detail))
    rec$certified <- isTRUE(fit$converged)
  } else rec <- fill_record(rec,fit)
  rec
}
pops <- all_populations(); pops <- pops[vapply(pops,function(p)p$role=='test',logical(1))]
if(smoke) pops <- pops[!duplicated(vapply(pops,`[[`,'','family'))]
seed_base <- 202609290L+as.integer(smoke)
tasks <- do.call(rbind,lapply(pops,function(p){
  ns <- population_ns(p);if(smoke) ns <- head(ns,1)
  z <- expand.grid(n=ns,rep=seq_len(if(smoke)1L else 3L))
  data.frame(pop=p$key,z,seed=mapply(draw_seed,seed_base,p$key,z$n,z$rep))
}))
write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed_base,tasks=nrow(tasks),
  smoke=smoke,workers=workers,git_head=magmaan_cache_ref()$git_head,
  git_dirty=magmaan_cache_ref()$git_dirty,library=find.package('magmaanlab'),
  built=utils::packageDescription('magmaanlab')$Built,
  starts='FABIN3 auto both routes; ordinary information; PSD diagonal; SLSQP; ordinary L-BFGS',
  controls='max_iter=5000; ftol_rel=1e-12; xtol_rel=1e-10; floor=1e-6; feasibility=1e-6',
  status='diagnostic only; criteria/psd-scale-followup.md'),packages='magmaanlab')
rows <- list();t0 <- proc.time()[['elapsed']]
batches <- split(seq_len(nrow(tasks)),ceiling(seq_len(nrow(tasks))/8))
for(b in seq_along(batches)) {
  z <- parallel::mclapply(batches[[b]],function(i) suppressWarnings(run_task(
    tasks[i,],pops,'psd-ml',new.env(),c('direct','two_stage'))),mc.cores=workers,mc.preschedule=TRUE)
  stopifnot(all(vapply(z,is.data.frame,logical(1))))
  rows[[b]] <- do.call(rbind,z)
  write.csv(rows[[b]],file.path(out,'raw',sprintf('batch_%03d.csv',b)),row.names=FALSE)
  elapsed <- proc.time()[['elapsed']]-t0
  cat(sprintf('%d/%d tasks; %.1fs; ETA %.1fs\n',max(batches[[b]]),nrow(tasks),elapsed,
    elapsed/max(batches[[b]])*(nrow(tasks)-max(batches[[b]]))))
}
x <- do.call(rbind,rows);x$success <- x$certified & !is.na(x$admissible) & x$admissible
write.csv(x,file.path(out,'fits.csv'),row.names=FALSE)
summary <- aggregate(x[c('success','seconds')],x[c('family','transform','arm')],sum)
write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
