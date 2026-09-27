#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('families.R','fits.R','psd_normalization.R'))source(file.path(here,'R',f))
if('--help' %in% args){
 cat('Usage: Rscript scripts/psd_normalization.R [--smoke] [--run-id NAME] [--workers 4]\n',
 'Unconstrained models only; marker and std.lv; current versus full sample normalization.\n',
 'Three draws per population/N; all four unit transforms; direct and two-stage PSD.\n',
 'Seed 202609291; smoke first population per family, smallest N, one draw, seed +1.\n')
 quit(save='no')
}
opt<-function(k,d)if(k %in% args)args[match(k,args)+1L]else d
smoke<-'--smoke' %in% args;run<-opt('--run-id',if(smoke)'smoke'else'pilot')
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run))
workers<-as.integer(opt('--workers','4'));out<-file.path(here,'results','psd-normalization',run)
if(file.exists(out))stop('choose a fresh run ID')
pops<-all_populations();pops<-pops[vapply(pops,function(p)p$role=='test' && p$family!='constrained',logical(1))]
pops[['r47_mis_f2_resid']]$models[[2]]<-fitted_model('correlated_residuals',
 'f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nx1 ~~ x4\nx2 ~~ x5\nx3 ~~ x6')
if(smoke)pops<-pops[!duplicated(vapply(pops,`[[`,'','family'))]
seed_base<-202609291L+as.integer(smoke)
tasks<-do.call(rbind,lapply(pops,function(p){
 ns<-population_ns(p);if(smoke)ns<-head(ns,1)
 z<-expand.grid(n=ns,rep=seq_len(if(smoke)1L else 3L))
 data.frame(pop=p$key,z,seed=mapply(draw_seed,seed_base,p$key,z$n,z$rep))
}))
run_task_normalized<-function(task){
 pop<-pops[[task$pop]];moments<-draw_moments(pop,task$n,task$seed)
 rows<-parameters<-covariances<-list()
 for(fm in pop$models)for(chart in c('marker','std_lv')){
  fm$std_lv<-chart=='std_lv';model<-build_model(fm)
  for(tr in c('native','x100','x0.01','mixed')){
   sample<-sample_in_units(moments,transform_factors(tr,pop$p),fm$meanstructure)
   for(route in c('direct','two_stage'))for(arm in c('current','normalized')){
    key<-cbind(task,family=pop$family,model=fm$key,chart=chart,transform=tr,route=route,arm=arm)
    z<-suppressWarnings(normalization_attempt(model,sample,fm$std_lv,route,arm=='normalized'))
    rows[[length(rows)+1L]]<-cbind(key,z$record)
    if(!is.null(z$theta))parameters[[length(parameters)+1L]]<-cbind(key[rep(1,length(z$theta)),],parameter=seq_along(z$theta),value=z$theta)
    if(!is.null(z$covariance))covariances[[length(covariances)+1L]]<-cbind(key[rep(1,length(z$covariance)),],element=seq_along(z$covariance),value=as.vector(z$covariance))
   }
  }
 }
 list(fits=do.call(rbind,rows),parameters=do.call(rbind,parameters),covariances=do.call(rbind,covariances))
}
dir.create(file.path(out,'raw'),recursive=TRUE)
write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed_base,tasks=nrow(tasks),smoke=smoke,
 workers=workers,git_head=magmaan_cache_ref()$git_head,git_dirty=magmaan_cache_ref()$git_dirty,
 library=find.package('magmaanlab'),built=utils::packageDescription('magmaanlab')$Built,
 starts='FABIN3-auto in fitting units; L-BFGS ordinary information; SLSQP PSD diagonal',
 controls='5000 evaluations; ftol_rel=1e-12; xtol_rel=1e-10; floor=feasibility=1e-6',
 status='exploratory; criteria/psd-normalization-pilot.md; no equality constraints'),packages='magmaanlab')
batches<-split(seq_len(nrow(tasks)),ceiling(seq_len(nrow(tasks))/8));all<-list();t0<-proc.time()[['elapsed']]
for(b in seq_along(batches)){
 z<-parallel::mclapply(batches[[b]],function(i)run_task_normalized(tasks[i,]),mc.cores=workers,mc.preschedule=TRUE)
 stopifnot(all(vapply(z,is.list,logical(1))))
 for(name in c('fits','parameters','covariances')){
  x<-do.call(rbind,lapply(z,`[[`,name))
  write.csv(x,file.path(out,'raw',sprintf('%s_%03d.csv',name,b)),row.names=FALSE)
 }
 all[[b]]<-do.call(rbind,lapply(z,`[[`,'fits'))
 elapsed<-proc.time()[['elapsed']]-t0
 cat(sprintf('%d/%d tasks; %.1fs; ETA %.1fs\n',max(batches[[b]]),nrow(tasks),elapsed,elapsed/max(batches[[b]])*(nrow(tasks)-max(batches[[b]]))))
}
write.csv(do.call(rbind,all),file.path(out,'fits.csv'),row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
