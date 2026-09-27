#!/usr/bin/env Rscript
# Diagnostic crossings of fixed ordinary endpoints and PSD fitting coordinates.
# No candidate selection or production policy change.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help'%in%args){cat('Usage: Rscript scripts/probe_two_stage.R [--run-id NAME]\nThree retained regressions; fixed-endpoint PSD crossings, cold starts and a 25000-evaluation budget diagnostic.\n');quit(save='no')}
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
run<-if('--run-id'%in%args)args[match('--run-id',args)+1L]else'two-stage-probe'
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run));out<-file.path(here,'results',run)
if(dir.exists(out))stop('choose a fresh run ID');dir.create(out,recursive=TRUE)
spec<-magmaanlab::model_spec(model_syntax)
tasks<-data.frame(design=c('ernst','ernst','high_r2'),n=20L,rep=c(1L,3L,7L))
ctl<-function(norm,budget=5000L)list(start='fabin3',start_transport='auto',normalize_sample=norm,coordinate_scaling='information',max_iter=as.integer(budget),nlopt=list(max_eval=as.integer(budget),ftol_rel=1e-12,xtol_rel=1e-10))
rows<-parameters<-list()
record<-function(key,fit=NULL,error=NULL){
 r<-data.frame(returned=!is.null(fit),certified=FALSE,admissible=FALSE,objective=NA_real_,accuracy_status='',distance=NA_real_,evaluations=NA_integer_,max_abs_theta=NA_real_,extent=NA_real_,min_variance=NA_real_,message='')
 if(!is.null(fit)){
  r$certified<-isTRUE(fit$converged);r$admissible<-isTRUE(fit$diagnostics$admissibility$admissible)
  r$objective<-fit$fmin;r$accuracy_status<-fit$diagnostics$newton_accuracy$status;r$distance<-fit$diagnostics$newton_accuracy$distance
  r$evaluations<-fit$f_evals;r$max_abs_theta<-max(abs(fit$theta))
  r$extent<-standardized_extent(fit$partable,magmaanlab::magmaan_core$model_implied(fit)$sigma[[1]])
  pt<-fit$partable;r$min_variance<-min(pt$est[pt$op=='~~'&pt$lhs==pt$rhs])
  parameters[[length(parameters)+1L]]<<-cbind(key[rep(1,length(fit$theta)),],parameter=seq_along(fit$theta),value=fit$theta)
 }else if(!is.null(error))r$message<-if(inherits(error,'error'))conditionMessage(error)else paste(error$kind,error$detail)
 rows[[length(rows)+1L]]<<-cbind(key,r)
}
for(i in seq_len(nrow(tasks))){
 t<-tasks[i,];seed<-902609281L+match(t$design,names(designs_all()))*100000L+t$n*100L+t$rep
 samp<-sample_moments(draw_data(design_sigma(designs_all()[[t$design]]),t$n,seed));warm<-list()
 for(norm in c(FALSE,TRUE)){
  c<-ctl(norm);ps<-c[setdiff(names(c),c('start','start_transport'))]
  z<-suppressWarnings(magmaanlab::frontier_fit_ml_psd_fallback(spec,samp,ordinary_optimizer='nlopt-lbfgs',psd_optimizer='nlopt-slsqp',ordinary_control=c,psd_control=ps,preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6))
  stopifnot(z$warm_start_used,!is.null(z$ordinary$fit))
  warm[[if(norm)'normalized'else'original']]<-z$ordinary$fit$theta
  for(stage in c('ordinary','psd'))record(cbind(t,source=if(norm)'normalized'else'original',psd_normalized=norm,budget=5000L,kind=paste0('wrapper_',stage)),z[[stage]]$fit,z[[stage]]$error)
 }
 for(src in c('original','normalized','cold'))for(norm in c(FALSE,TRUE))for(budget in c(5000L,25000L)){
  c<-ctl(norm,budget);if(src!='cold'){c$start<-warm[[src]];c$start_transport<-'native'}
  z<-tryCatch(suppressWarnings(magmaanlab::frontier_fit_ml_psd(spec,samp,optimizer='nlopt-slsqp',control=c,preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6)),error=function(e)e)
  key<-cbind(t,source=src,psd_normalized=norm,budget=budget,kind='explicit_psd')
  if(inherits(z,'error'))record(key,error=z)else record(key,z)
 }
 cat('Completed ',t$design,' N=',t$n,' draw ',t$rep,'\n',sep='')
}
x<-do.call(rbind,rows);ref<-read.csv(file.path(here,'results','psd-fresh','references.csv'))
x<-merge(x,ref[c('design','n','rep','best_objective')],by=c('design','n','rep'),all.x=TRUE,sort=FALSE)
x$target_hit<-x$certified & x$admissible & is.finite(x$objective)&abs(x$objective-x$best_objective)<=1e-6*(1+abs(x$best_objective))
write.csv(x,file.path(out,'fits.csv'),row.names=FALSE);write.csv(do.call(rbind,parameters),file.path(out,'parameters.csv'),row.names=FALSE)
write_metadata(file.path(out,'metadata.csv'),values=list(protocol='post-run diagnostic, three preidentified regressions; no policy changes; all crossings, cold controls and both budgets retained',
 seed_rule='902609281 + design_index*100000 + N*100 + rep',controls='FABIN3/auto; ordinary L-BFGS/information; PSD SLSQP/diagonal; f=1e-12, x=1e-10; PSD floor=feasibility=1e-6',
 git_head=magmaan_cache_ref()$git_head,library=find.package('magmaanlab'),library_md5=unname(tools::md5sum(system.file('libs','magmaanlab.so',package='magmaanlab')))),packages='magmaanlab')
print(x[c('design','rep','source','psd_normalized','budget','kind','certified','objective','target_hit','evaluations')],row.names=FALSE)
