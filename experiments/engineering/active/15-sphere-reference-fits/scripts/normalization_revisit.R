#!/usr/bin/env Rscript
# Paired development revisit. Fixed single-start methods; no sphere fitting,
# endpoint-derived starts, retries, marker substitution, or algorithm tuning.
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script)); args <- commandArgs(TRUE)
if('--help' %in% args) {
 cat('Usage: Rscript scripts/normalization_revisit.R [--smoke] [--run-id NAME]\n',
 '120 saved draws; ML layered/native with L-BFGS and PORT/information;\n',
 'direct and two-stage PSD FABIN3/auto, SLSQP/diagonal (ordinary L-BFGS/information).\n',
 'Normalization off/on; 960 fits. No multistart. References read only after fitting.\n')
 quit(save='no')
}
source(file.path(here,'..', '..', '..', '_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
smoke <- '--smoke' %in% args
run <- if('--run-id' %in% args)args[match('--run-id',args)+1L] else if(smoke)'normalization-smoke'else'normalization-revisit'
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run))
out <- file.path(here,'results',run);if(dir.exists(out))stop('choose a fresh run-id')
dir.create(out,recursive=TRUE)
write <- function(z,name)write.csv(z,file.path(out,paste0(name,'.csv')),row.names=FALSE)
spec <- magmaanlab::model_spec(model_syntax)
fit_one <- function(sample,method,normalized) {
 domain <- if(grepl('^ml_',method))'ML'else'PSD'
 ctl <- list(start=if(domain=='ML')'layered'else'fabin3',start_transport=if(domain=='ML')'native'else'auto',
  normalize_sample=normalized,coordinate_scaling='information',max_iter=5000L,
  nlopt=list(max_eval=5000L,ftol_rel=1e-12,xtol_rel=1e-10),
  port=list(max_eval=5000L,rel_f_tol=1e-12,x_tol=1e-10))
 t0 <- proc.time()[['elapsed']]
 rec <- data.frame(returned=FALSE,certified=FALSE,admissible=FALSE,objective=NA_real_,
  sample_normalized=NA,accuracy_status='',accuracy_passed=FALSE,newton_distance=NA_real_,
  chart_pass=FALSE,std_extent=NA_real_,legacy_screen=FALSE,stage='',seconds=NA_real_,message='')
 z <- tryCatch(suppressWarnings(switch(method,
  ml_lbfgs=magmaanlab::magmaan_core$fit_ml(spec,sample,optimizer='nlopt-lbfgs',control=ctl),
  ml_port=magmaanlab::magmaan_core$fit_ml(spec,sample,optimizer='port',control=ctl),
  direct_psd=magmaanlab::frontier_fit_ml_psd(spec,sample,optimizer='nlopt-slsqp',control=ctl,
   preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6),
  two_stage_psd=magmaanlab::frontier_fit_ml_psd_fallback(spec,sample,ordinary_optimizer='nlopt-lbfgs',
   psd_optimizer='nlopt-slsqp',ordinary_control=ctl,psd_control=ctl[setdiff(names(ctl),c('start','start_transport'))],
   preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6))),error=function(e)e)
 rec$seconds <- proc.time()[['elapsed']]-t0
 if(inherits(z,'error')) {rec$message<-conditionMessage(z);return(list(record=rec))}
 fit<-z
 if(method=='two_stage_psd') {
  selected<-if(isTRUE(z$fallback_used))z$psd else z$ordinary
  rec$stage<-if(isTRUE(z$fallback_used))paste0('psd:',z$fallback_reason)else'ordinary'
  fit<-selected$fit
  if(is.null(fit)){rec$message<-paste(selected$error$kind,selected$error$detail);return(list(record=rec))}
 }
 stopifnot(!is.null(fit$sample_normalized),identical(isTRUE(fit$sample_normalized),normalized))
 rec$returned<-TRUE;rec$certified<-isTRUE(z$converged);rec$objective<-fit$fmin
 rec$sample_normalized<-fit$sample_normalized;rec$admissible<-isTRUE(fit$diagnostics$admissibility$admissible)
 audit<-magmaanlab::frontier_newton_accuracy(fit,psd=domain=='PSD')
 rec$accuracy_status<-audit$status;rec$accuracy_passed<-isTRUE(audit$passed);rec$newton_distance<-audit$distance
 # Same requested marker model, no optimization and no marker replacement.
 tr<-tryCatch(magmaanlab::frontier_reidentify(fit$partable,spec,pole_tol=1e-4),error=function(e)e)
 rec$chart_pass<-!inherits(tr,'error')
 sigma<-magmaanlab::magmaan_core$model_implied(fit)$sigma[[1]]
 rec$std_extent<-standardized_extent(fit$partable,sigma)
 # Historical screen is descriptive only. Its stronger-marker audit does not
 # replace the requested-chart fit verdict or produce an additional estimate.
 rec$legacy_screen<-assess_endpoint(fit,sample,domain,FALSE)$record$screened
 list(record=rec,theta=fit$theta)
}
rows<-parameters<-list();t0<-proc.time()[['elapsed']]
for(batch in c('development','fresh')) {
 tasks<-expand.grid(design=names(designs_all()),n=c(20L,100L),rep=1:10,stringsAsFactors=FALSE)
 tasks$seed<-(if(batch=='development')202609281L else 902609281L)+match(tasks$design,names(designs_all()))*100000L+tasks$n*100L+tasks$rep
 if(smoke)tasks<-head(tasks,6)
 for(i in seq_len(nrow(tasks))) {
  task<-tasks[i,];sample<-sample_moments(draw_data(design_sigma(designs_all()[[task$design]]),task$n,task$seed))
  for(method in c('ml_lbfgs','ml_port','direct_psd','two_stage_psd'))for(arm in c('original_units','normalized')) {
   key<-cbind(batch=batch,task,domain=if(grepl('^ml_',method))'ML'else'PSD',method=method,arm=arm)
   z<-fit_one(sample,method,arm=='normalized');rows[[length(rows)+1L]]<-cbind(key,z$record)
   if(!is.null(z$theta))parameters[[length(parameters)+1L]]<-cbind(key[rep(1,length(z$theta)),],parameter=seq_along(z$theta),value=z$theta)
  }
  if(i%%5==0||i==nrow(tasks))cat(sprintf('%s %d/%d draws; %.1fs elapsed\n',batch,i,nrow(tasks),proc.time()[['elapsed']]-t0))
 }
}
x<-do.call(rbind,rows);write(x,'fits');write(do.call(rbind,parameters),'parameters')
# References are evaluation-only and are not updated using this run's outcomes.
refs<-do.call(rbind,lapply(c('development','fresh'),function(batch){
 ml<-read.csv(file.path(here,'results',paste0('start-',batch),'references.csv'))
 ps<-read.csv(file.path(here,'results',if(batch=='development')'reference-pilot'else'psd-fresh','references.csv'))
 cbind(batch=batch,rbind(ml[ml$domain=='ML',],ps[ps$domain=='PSD',]))
}))
keys<-c('batch','design','n','rep','domain')
x<-merge(x,refs[c(keys,'best_objective','reference_label')],by=keys,all.x=TRUE,sort=FALSE)
x$historical_target<-x$best_objective;x$target_source<-'saved_sphere_reference'
w<-read.csv(file.path(here,'results','open-case-conclusions','finite_witnesses.csv'))
for(i in seq_len(nrow(w))) {
 use<-x$domain=='ML' & x$batch==w$batch[i] & x$design==w$design[i] & x$n==w$n[i] & x$rep==w$rep[i]
 x$best_objective[use]<-w$objective[i];x$target_source[use]<-'refined_finite_witness'
}
policy<-read.csv(file.path(here,'results','requested-chart-policy','endpoint_checks.csv'))
x$known_near_pole_case<-FALSE
for(i in which(!policy$translates_at_1e4)) {
 use<-x$domain=='ML' & x$batch==policy$batch[i] & x$design==policy$design[i] & x$n==policy$n[i] & x$rep==policy$rep[i]
 x$known_near_pole_case[use]<-TRUE
}
x$target_available<-is.finite(x$best_objective) & !x$known_near_pole_case
x$objective_gap<-x$objective-x$best_objective
x$objective_match<-x$target_available & is.finite(x$objective_gap) & abs(x$objective_gap)<=1e-6*(1+abs(x$best_objective))
x$better_objective<-x$target_available & is.finite(x$objective_gap) & x$objective_gap < -1e-6*(1+abs(x$best_objective))
x$accepted<-x$certified & x$accuracy_passed & x$chart_pass & (x$domain=='ML'|x$admissible)
x$target_reached<-x$accepted & (x$objective_match|x$better_objective)
x$historical_hit<-x$legacy_screen & is.finite(x$historical_target) & is.finite(x$objective) & x$objective<=x$historical_target+1e-6*(1+abs(x$historical_target))
write(x,'comparisons')
source(file.path(here,'R','normalization_summary.R'))
summarize_normalization_revisit(x,out)
write_metadata(file.path(out,'metadata.csv'),values=list(smoke=smoke,fits=nrow(x),
 protocol='paired saved development draws; no multistart; references frozen and read after fitting',
 seed_rule='development 202609281 or fresh 902609281 + design_index*100000 + N*100 + rep',
 ml='layered/native; L-BFGS and PORT/information',psd='FABIN3/auto; SLSQP/diagonal; two-stage ordinary L-BFGS/information',
 controls='5000 evaluations and iterations; relative f=1e-12; x=1e-10; PSD floor=feasibility=1e-6',
 target='saved ML/PSD references; seven refined ML finite witnesses replace their prior targets; known near-pole ML cases excluded from recovery denominator',
 judge='library verdict AND requested-chart accuracy AND chart translation at study tolerance 1e-4; PSD also admissible; no extent cutoff',
 historical='original extent/strong-marker screen retained separately with current library audits',
 tolerance='objective 1e-6*(1+abs(reference)); lower accepted objectives counted as reached, flagged separately',
 git_head=magmaan_cache_ref()$git_head,git_dirty=magmaan_cache_ref()$git_dirty,
 library=find.package('magmaanlab'),library_md5=unname(tools::md5sum(system.file('libs','magmaanlab.so',package='magmaanlab')))),packages='magmaanlab')
print(read.csv(file.path(out,'summary.csv')),row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
