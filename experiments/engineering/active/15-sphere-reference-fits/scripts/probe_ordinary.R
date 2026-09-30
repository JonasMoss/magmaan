#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'..', '..', '..', '_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R','start_design.R'))source(file.path(here,'R',f))
args <- commandArgs(TRUE)
if('--help' %in% args) {
 cat('Usage: Rscript scripts/probe_ordinary.R [--smoke] [--run-id NAME]\nUses retained reference-pilot and start-development raw fits/parameters.\nReruns ordinary ML (both backends) and direct PSD with FABIN3 auto transport and diagonal scaling on the original 60 draws.\nAlso tests PSD without diagonal scaling on every draw.\nFor misses with a sphere reference: direct translation, reference warm start,\nreference-chosen marker, and direct spectral starts. No new sphere optimization.\nSmoke: first six draws. Full default run-id: ordinary-transfer.\n');quit(save='no')
}
smoke <- '--smoke' %in% args
run <- if('--run-id' %in% args)args[match('--run-id',args)+1L] else if(smoke)'ordinary-transfer-smoke' else 'ordinary-transfer'
stopifnot(!is.na(run),grepl('^[a-zA-Z0-9_-]+$',run))
out<-file.path(here,'results',run);if(file.exists(file.path(out,'fits.csv')))stop('choose a fresh --run-id')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_out<-function(x,name)write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
pool<-params<-list()
for(run_source in c('reference-pilot','start-development')){
 f<-read.csv(file.path(here,'results',run_source,'fits.csv'));f$source<-run_source
 pool[[run_source]]<-f[f$route=='sphere',]
 params[[run_source]]<-read.csv(file.path(here,'results',run_source,'parameters.csv'))
}
columns<-Reduce(intersect,lapply(pool,names));pool<-do.call(rbind,lapply(pool,function(x)x[columns]))
refs<-reference_rows(pool);tasks<-unique(pool[c('design','n','rep','seed')]);if(smoke)tasks<-head(tasks,6)
spec<-magmaanlab::model_spec(model_syntax);rows<-translations<-list();t0<-proc.time()[['elapsed']]
for(i in seq_len(nrow(tasks))){
 task<-tasks[i,];data<-draw_data(design_sigma(designs_all()[[task$design]]),task$n,task$seed);sample<-sample_moments(data)
 for(domain in c('ML','PSD')){
  r<-refs[refs$design==task$design&refs$n==task$n&refs$rep==task$rep&refs$domain==domain,]
  reference_pt<-NULL
  if(is.finite(r$best_objective)){
   candidates<-pool[pool$design==task$design&pool$n==task$n&pool$rep==task$rep&pool$domain==domain&pool$screened,]
   winner<-candidates[which.min(candidates$objective),]
   z<-params[[winner$source]];z<-z[z$fit_id==winner$fit_id,]
   key<-function(x)paste(x$lhs,x$op,x$rhs,sep='|')
   reference_pt<-spec$partable;idx<-match(key(reference_pt),key(z));stopifnot(!anyNA(idx),!anyDuplicated(key(z)))
   reference_pt$est<-z$est[idx]
   # Saved endpoints are sphere coordinates: original fixed-marker flags would
   # replace the saved marker estimates. Restore a fully free source point.
   reference_pt$free<-seq_len(nrow(reference_pt));reference_pt$ustart<-reference_pt$est
   translated<-tryCatch({
    tr<-magmaanlab::frontier_reidentify(reference_pt,spec,pole_tol=0)
    ev<-magmaanlab::magmaan_core$evaluate_at(spec$partable,sample,tr$theta,estimator='ML')
    gap<-abs(ev$fmin-r$best_objective)
    if(!is.finite(gap)||gap>1e-6*(1+abs(r$best_objective)))stop('translated objective disagrees')
    list(theta=tr$theta,gap=gap)
   },error=function(e)e)
   translations[[length(translations)+1L]]<-cbind(task,domain=domain,
    reference_source=winner$source,reference_fit_id=winner$fit_id,best_objective=r$best_objective,
    reference_chart_level=winner$chart_level,translation_ok=!inherits(translated,'error'),
    objective_gap=if(inherits(translated,'error'))NA_real_ else translated$gap,
    message=if(inherits(translated,'error'))conditionMessage(translated) else '')
  }
  for(backend in if(domain=='ML')c('nlopt-lbfgs','port') else 'nlopt-slsqp'){
   append_result<-function(result,arm){
    id<-length(rows)+1L
    rows[[id]]<<-cbind(fit_id=id,task,transform='native',domain=domain,route='ordinary',backend=backend,start_id=arm,result$record)
   }
   constructor <- if(domain=='ML') 'layered' else 'scaled-fabin'
   base_id <- if(domain=='ML') 'layered_native' else 'fabin3_auto_diagonal'
   ctl <- list(start=constructor)
   if(domain=='ML') ctl$coordinate_scaling <- 'information'
   base<-run_fit(spec,data,sample,domain,'ordinary',backend,base_id,control=ctl);append_result(base,base_id)
   if(domain=='PSD')append_result(run_fit(spec,data,sample,domain,'ordinary',backend,'fabin3_auto_none',preconditioning='none',control=ctl),'fabin3_auto_none')
   matches<-base$record$screened&&is.finite(r$best_objective)&&abs(base$record$objective-r$best_objective)<=1e-6*(1+abs(r$best_objective))
   better<-base$record$screened&&is.finite(r$best_objective)&&base$record$objective<r$best_objective-1e-6*(1+abs(r$best_objective))
   if(is.null(reference_pt)||matches||better)next
   if(!inherits(translated,'error'))append_result(run_fit(spec,data,sample,domain,'ordinary',backend,'reference_warm',translated$theta,control=ctl),'reference_warm')
   marker<-strong_marker_model(reference_pt,sample)
   append_result(run_fit(marker,data,sample,domain,'ordinary',backend,'reference_marker_constructor',control=ctl),'reference_marker_constructor')
   recipes<-if(domain=='PSD')list(spectral_positive=spectral_start(spec,sample)) else start_recipes(spec,sample)
   for(name in names(recipes))append_result(run_fit(spec,data,sample,domain,'ordinary',backend,name,recipes[[name]],control=ctl),name)
  }
 }
 cat(sprintf('[%d/%d] %s N=%d rep=%d; %.1fs\n',i,nrow(tasks),task$design,task$n,task$rep,proc.time()[['elapsed']]-t0))
}
fits<-do.call(rbind,rows);cmp<-compare_rows(fits,refs)
write_out(fits,'fits');write_out(cmp,'comparisons');write_out(do.call(rbind,translations),'translations');write_out(refs,'references')
evidence_columns <- c('design','n','rep','seed','domain','backend','start_id','label','comparison',
 'objective','best_objective','reference_gap','original_verdict','admissible','std_extent','seconds','message')
write_out(cmp[evidence_columns],'evidence')
summary<-aggregate(list(fits=cmp$fit_id),cmp[c('domain','backend','start_id','label','comparison')],length);write_out(summary,'summary')
# Each row describes one baseline/backend, with any-hit spectral results and
# no automatic claim that a failed finite portfolio proves nonattainment.
paired<-lapply(split(cmp,interaction(cmp$design,cmp$n,cmp$rep,cmp$domain,cmp$backend,drop=TRUE)),function(z){
 b<-z[z$start_id %in% c('layered_native','fabin3_auto_diagonal'),];hit<-function(arm)any(z$start_id%in%arm&z$comparison=='matches_sphere_reference')
 cbind(b[c('design','n','rep','seed','domain','backend','label','comparison','objective','best_objective','original_verdict')],
  probes=sum(!z$start_id %in% c('layered_native','fabin3_auto_diagonal','fabin3_auto_none')),unscaled_hit=hit('fabin3_auto_none'),warm_hit=hit('reference_warm'),marker_hit=hit('reference_marker_constructor'),
  spectral_hit=hit(grep('^spectral_',z$start_id,value=TRUE)))
})
write_out(do.call(rbind,paired),'paired')
ref<-magmaan_cache_ref();write_metadata(file.path(out,'metadata.csv'),values=list(tasks=nrow(tasks),fits=nrow(fits),
 reference_pool='reference-pilot + start-development sphere attempts only; domain-specific',
 selection='probe layered ML / FABIN3-auto PSD misses only when a finite sphere reference exists',
 warm_start='diagnostic reference-informed; numeric control start',marker='diagnostic reference-chosen strongest indicator',
 git_head=ref$git_head,git_dirty=ref$git_dirty,magmaanlab_built=utils::packageDescription('magmaanlab')$Built),packages='magmaanlab')
cat('Wrote ',out,'\n',sep='')
