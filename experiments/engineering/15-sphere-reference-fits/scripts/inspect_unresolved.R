#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R','start_design.R','escape_diagnostics.R'))source(file.path(here,'R',f))
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/inspect_unresolved.R [--smoke] [--run-id NAME]\n',
 'Nine ML problems with no screened sphere reference: 4 development, 5 retained fresh.\n',
 'Nine marker placements, effect coding, positive std.lv, sphere; L-BFGS and PORT.\n',
 'Four common spectral points transported across charts plus native constructors.\n',
 'Continue best marker / effect / std.lv / sphere endpoints at tighter tolerances.\n',
 'Retains objective, covariance drift, invariant component size, accuracy and old extent screen.\n',
 'Smoke: first fresh case. No automatic nonattainment verdict.\n',sep='');quit(save='no')}
smoke<-'--smoke' %in% args
run<-if('--run-id' %in% args)args[match('--run-id',args)+1L]else if(smoke)'unresolved-smoke' else 'unresolved-identifications'
stopifnot(!is.na(run),grepl('^[a-zA-Z0-9_-]+$',run))
out<-file.path(here,'results',run);if(file.exists(file.path(out,'fits.csv')))stop('choose a fresh run-id')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_out<-function(x,name)write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
tasks<-do.call(rbind,lapply(c('development','fresh'),function(batch){
 r<-read.csv(file.path(here,'results',paste0('start-',batch),'references.csv'))
 r<-r[r$domain=='ML' & r$reference_label=='no_screened_reference',c('design','n','rep','extreme_sphere')]
 r$seed<-(if(batch=='development')202609281L else 902609281L)+match(r$design,names(designs_all()))*100000L+r$n*100L+r$rep
 cbind(batch=batch,r)
}))
if(smoke)tasks<-head(tasks[tasks$batch=='fresh',],1)
write_out(tasks,'tasks')
charts<-inspection_charts();base<-charts$marker_11
rows<-parameters<-covariances<-list();t0<-proc.time()[['elapsed']]
for(i in seq_len(nrow(tasks))){
 task<-tasks[i,];data<-draw_data(design_sigma(designs_all()[[task$design]]),task$n,task$seed);sample<-sample_moments(data)
 recipes<-start_recipes(base,sample)
 source_points<-lapply(recipes,function(theta)magmaanlab::magmaan_core$evaluate_at(base$partable,sample,theta,estimator='ML')$partable)
 winners<-list()
 record<-function(z,chart,backend,arm,phase,budget,parent=NA_integer_,previous=NULL){
  m<-escape_metrics(z$partable,sample);id<-length(rows)+1L
  drift<-if(!is.null(previous)&&!is.null(m$sigma)&&!is.null(previous$sigma))max(abs(m$sigma-previous$sigma)/sqrt(outer(diag(sample$S[[1]]),diag(sample$S[[1]]))))else NA_real_
  rows[[id]]<<-cbind(fit_id=id,task,chart=chart,backend=backend,start_id=arm,phase=phase,budget=budget,parent_fit_id=parent,
    covariance_drift=drift,z$record,m$record)
  if(!is.null(z$partable))parameters[[length(parameters)+1L]]<<-cbind(fit_id=id,z$partable[c('lhs','op','rhs','est')])
  if(!is.null(m$sigma))covariances[[length(covariances)+1L]]<<-data.frame(fit_id=id,row=rep(ov_names,6),column=rep(ov_names,each=6),value=as.vector(m$sigma))
  list(result=z,metric=m,id=id,chart=chart,backend=backend)
 }
 for(chart in names(charts))for(backend in c('nlopt-lbfgs','port')){
  spec<-charts[[chart]];route<-if(chart=='sphere')'sphere' else 'ordinary';best<-NULL
  for(arm in c('constructor',names(recipes))){
   theta<-if(arm=='constructor')NULL else tryCatch(magmaanlab::frontier_reidentify(source_points[[arm]],spec,pole_tol=0)$theta,error=function(e)e)
   if(inherits(theta,'error')){
    rec<-endpoint_record();rec$label<-'start_outside_chart';rec$message<-conditionMessage(theta);z<-list(record=rec,partable=NULL)
   }else{
    if(!is.null(theta)){
     ev<-magmaanlab::magmaan_core$evaluate_at(spec$partable,sample,theta,estimator='ML')
     ref_ev<-magmaanlab::magmaan_core$evaluate_at(base$partable,sample,recipes[[arm]],estimator='ML')
     sigma<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
     ref_sigma<-magmaanlab::magmaan_core$model_implied(ref_ev)$sigma[[1]]
     stopifnot(max(abs(sigma-ref_sigma)/sqrt(outer(diag(sample$S[[1]]),diag(sample$S[[1]]))))<1e-8)
    }
    ctl<-list(coordinate_scaling='information',max_iter=5000L)
    if(route=='ordinary')ctl<-c(ctl,list(start='layered',start_transport='native'))
    z<-run_fit(spec,data,sample,'ML',route,backend,arm,theta,control=ctl)
   }
   current<-record(z,chart,backend,arm,'portfolio',5000L)
   if(is.finite(z$record$objective)&&!is.null(current$metric$sigma)&&
      (is.null(best)||z$record$objective<best$result$record$objective))best<-current
  }
  if(!is.null(best))winners[[paste(chart,backend)]]<-best
 }
 # Follow the best evaluable endpoint in each chart family and backend. An
 # excluded extreme endpoint is still retained; it is not silently a failure.
 for(backend in c('nlopt-lbfgs','port'))for(family in c('marker','effect_coded','std_lv_positive','sphere')){
  candidates<-Filter(function(w)w$backend==backend && if(family=='marker')grepl('^marker_',w$chart)else w$chart==family,winners)
  if(!length(candidates))next
  w<-candidates[[which.min(vapply(candidates,function(z)z$result$record$objective,0))]]
  for(budget in c(1000L,10000L,50000L)){
   spec<-charts[[w$chart]];route<-if(w$chart=='sphere')'sphere' else 'ordinary'
   theta<-tryCatch(magmaanlab::frontier_reidentify(w$result$partable,spec,pole_tol=0)$theta,error=function(e)e)
   if(inherits(theta,'error')){
    rec<-endpoint_record();rec$label<-'restart_outside_chart';rec$message<-conditionMessage(theta);z<-list(record=rec,partable=NULL)
   }else z<-run_fit(spec,data,sample,'ML',route,backend,'continued_endpoint',theta,
     control=list(coordinate_scaling='information',max_iter=budget,ftol=1e-14,gtol=1e-12))
   next_w<-record(z,w$chart,backend,'continued_endpoint','continuation',budget,w$id,w$metric)
   if(!is.null(next_w$metric$sigma)&&is.finite(z$record$objective))w<-next_w
  }
 }
 cat(sprintf('[%d/%d] %s %s N=%d rep=%d; %.1fs\n',i,nrow(tasks),task$batch,task$design,task$n,task$rep,proc.time()[['elapsed']]-t0))
}
fits<-do.call(rbind,rows);fits$local_checked<-fits$label %in% c('screened_candidate','screened_extreme')
write_out(fits,'fits');write_out(do.call(rbind,parameters),'parameters');write_out(do.call(rbind,covariances),'covariances')
summary<-lapply(split(fits,interaction(fits$batch,fits$design,fits$n,fits$rep,drop=TRUE)),function(z){
 finite<-z[is.finite(z$objective),];best<-finite[which.min(finite$objective),]
 checked<-z[z$local_checked,];cont<-z[z$phase=='continuation',]
 cbind(z[1,c('batch','design','n','rep','seed','extreme_sphere')],attempts=nrow(z),
  screened=sum(z$screened),local_checked=sum(z$local_checked),
  best_objective=best$objective,best_chart=best$chart,best_label=best$label,
  best_component_extent=best$component_extent,best_old_extent=best$std_extent,
  best_checked_objective=if(nrow(checked))min(checked$objective)else NA_real_,
  max_component_extent=if(any(is.finite(z$component_extent)))max(z$component_extent,na.rm=TRUE)else NA_real_)
})
write_out(do.call(rbind,summary),'summary')
write_out(fits[fits$phase=='continuation',], 'continuation')
evidence<-lapply(split(fits,interaction(fits$batch,fits$design,fits$n,fits$rep,drop=TRUE)),function(z){
 checked<-z[z$local_checked,]
 if(!nrow(checked))return(NULL)
 best<-checked[which.min(checked$objective),]
 near<-checked[checked$objective-best$objective<=1e-8,]
 cbind(best[c('fit_id','batch','design','n','rep','chart','backend','label','objective','std_extent',
   'component_extent','latent_total_x','latent_total_y','disturbance_y','scaled_path')],
   matching_charts=paste(sort(unique(near$chart)),collapse=';'))
})
evidence<-do.call(rbind,evidence);write_out(evidence,'candidate_evidence')
params<-do.call(rbind,parameters);covs<-do.call(rbind,covariances)
write_out(params[params$fit_id %in% evidence$fit_id,], 'candidate_parameters')
write_out(covs[covs$fit_id %in% evidence$fit_id,], 'candidate_covariances')
ref<-magmaan_cache_ref();write_metadata(file.path(out,'metadata.csv'),values=list(tasks=nrow(tasks),fits=nrow(fits),
 selection='all no_screened_reference ML draws from start-development and start-fresh',
 charts='nine markers; effect coding; positive std.lv (restricted sign sector); sphere',
 starts='same four spectral covariance points transported; constructor separate; outside-chart starts retained',
 continuation='best evaluable endpoint by chart family/backend; budgets 1000/10000/50000; ftol 1e-14 gtol 1e-12',
 metrics='covariance drift; scale-invariant primitive contributions; old extent retained separately',
 interpretation='diagnostic only; no generic runaway/nonattainment classifier; PSD not a substitute for ML',
 git_head=ref$git_head,git_dirty=ref$git_dirty),packages='magmaanlab')
cat('Wrote ',out,'\n',sep='')
