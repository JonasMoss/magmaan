#!/usr/bin/env Rscript
# Serial, paired normalization timing of four study workflows.
# Model/sample construction and outcome verification are outside timed blocks.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help'%in%args){cat('Usage: Rscript scripts/time_normalization.R [--smoke] [--run-id NAME]\n120 saved cases; four rounds, rotating four routes, paired normalization off/on; three calls/block.\nOrdinary ML, direct PSD, warm two-stage and independent two-stage; wall/CPU includes failures and R orchestration.\n');quit(save='no')}
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
smoke<-'--smoke'%in%args
run<-if('--run-id'%in%args)args[match('--run-id',args)+1L]else if(smoke)'normalization-timing-smoke'else'normalization-timing'
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run));out<-file.path(here,'results',run)
if(dir.exists(out))stop('choose a fresh run ID');dir.create(out,recursive=TRUE)
outcomes<-read.csv(file.path(here,'results','normalization-revisit','comparisons.csv'))
saved<-subset(outcomes,method=='two_stage_psd'&arm=='normalized')[c('batch','design','n','rep','seed')]
if(smoke)saved<-saved[!duplicated(saved[c('batch','design')]),]
meta<-read.csv(file.path(here,'results','normalization-revisit','metadata.csv'))
md5<-unname(tools::md5sum(system.file('libs','magmaanlab.so',package='magmaanlab')))
stopifnot(md5==meta$value[meta$key=='library_md5'])
expected<-function(b,method,normalized){
 arm<-if(normalized)'normalized'else'original_units'
 z<-outcomes[outcomes$batch==b$batch&outcomes$design==b$design&outcomes$n==b$n&outcomes$rep==b$rep&outcomes$arm==arm,]
 chosen<-switch(method,ml='ml_lbfgs',direct='direct_psd',warm='two_stage_psd',cold=if(z$stage[z$method=='two_stage_psd']=='ordinary')'two_stage_psd'else'direct_psd')
 e<-z[z$method==chosen,];stopifnot(nrow(e)==1);e
}
spec<-magmaanlab::model_spec(model_syntax)
samples<-lapply(seq_len(nrow(saved)),function(i){b<-saved[i,];sample_moments(draw_data(design_sigma(designs_all()[[b$design]]),b$n,b$seed))})
ctl<-list(start='fabin3',start_transport='auto',normalize_sample=TRUE,coordinate_scaling='information',max_iter=5000L,nlopt=list(max_eval=5000L,ftol_rel=1e-12,xtol_rel=1e-10))
psctl<-ctl[setdiff(names(ctl),c('start','start_transport'))]
ordinary<-function(s,ctl)tryCatch(suppressWarnings(magmaanlab::magmaan_core$fit_ml(spec,s,optimizer='nlopt-lbfgs',control=ctl)),error=function(e)e)
psd<-function(s,ctl)tryCatch(suppressWarnings(magmaanlab::frontier_fit_ml_psd(spec,s,optimizer='nlopt-slsqp',control=ctl,preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6)),error=function(e)e)
call_method<-function(method,s,normalized){
 ctl$normalize_sample<-normalized;psctl$normalize_sample<-normalized
 if(method=='ml'){ctl$start<-'layered';ctl$start_transport<-'native';return(ordinary(s,ctl))}
 if(method=='direct')return(psd(s,ctl))
 if(method=='cold'){
  f<-ordinary(s,ctl)
  if(!inherits(f,'error')&&isTRUE(f$converged)&&isTRUE(f$diagnostics$admissibility$admissible))return(f)
  return(psd(s,ctl))
 }
 z<-suppressWarnings(magmaanlab::frontier_fit_ml_psd_fallback(spec,s,ordinary_optimizer='nlopt-lbfgs',psd_optimizer='nlopt-slsqp',ordinary_control=ctl,psd_control=psctl,preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6))
 chosen<-if(z$fallback_used)z$psd else z$ordinary
 if(is.null(chosen$fit))return(simpleError(paste(chosen$error$kind,chosen$error$detail)))
 chosen$fit
}
verify<-function(f,e){
 if(inherits(f,'error'))stopifnot(!e$returned)else{
  stopifnot(e$returned,abs(f$fmin-e$objective)<1e-9,isTRUE(f$converged)==e$certified)
 }
}
methods<-c('ml','direct','cold','warm')
block_reps<-if(smoke)2L else 3L
for(i in seq_len(nrow(saved)))for(method in methods)for(normalized in c(FALSE,TRUE))
 verify(call_method(method,samples[[i]],normalized),expected(saved[i,],method,normalized))
rows<-list();t0<-proc.time()[['elapsed']]
for(round in 1:4){
 set.seed(202609301L+round);indices<-sample(seq_len(nrow(saved)))
 for(j in seq_along(indices)){
  i<-indices[j];b<-saved[i,];s<-samples[[i]]
  for(position in 1:4){
   method<-methods[(position+round+i-3L)%%4L+1L]
   norms<-if((round+i)%%2L)c(FALSE,TRUE)else c(TRUE,FALSE)
   for(normalized in norms){
    e<-expected(b,method,normalized)
    gc(FALSE)
    cpu0<-proc.time();wall0<-as.numeric(Sys.time())
    for(k in seq_len(block_reps))last<-call_method(method,s,normalized)
    wall<-as.numeric(Sys.time())-wall0;cpu<-proc.time()-cpu0
    verify(last,e)
    rows[[length(rows)+1L]]<-cbind(b,round=round,position=position,method=method,normalized=normalized,
      target_hit=e$target_reached,certified=e$certified,
      calls=block_reps,wall_ms=1000*wall/block_reps,cpu_ms=1000*sum(cpu[c('user.self','sys.self')])/block_reps)
   }
  }
  if(j%%20==0||j==length(indices)){
   elapsed<-proc.time()[['elapsed']]-t0;done<-(round-1)*nrow(saved)+j;total<-4*nrow(saved)
   cat(sprintf('Round %d/4: %d/%d cases; %.1fs elapsed; ~%.1fs remaining\n',round,j,nrow(saved),elapsed,elapsed*(total/done-1)))
  }
 }
 write.csv(do.call(rbind,rows),file.path(out,'blocks.csv'),row.names=FALSE)
}
x<-do.call(rbind,rows)
# Robust per-case cost: median of the four measured block averages.
k<-c('batch','design','n','rep','seed','method','normalized','target_hit','certified')
per_case<-aggregate(x[c('wall_ms','cpu_ms')],x[k],median)
write.csv(per_case,file.path(out,'per_case.csv'),row.names=FALSE)
summarize<-function(groups){
 pieces<-split(per_case,interaction(per_case[groups],drop=TRUE))
 do.call(rbind,lapply(pieces,function(z)cbind(z[1,groups,drop=FALSE],cases=nrow(z),sum_ms=sum(z$wall_ms),median_ms=median(z$wall_ms),p90_ms=unname(quantile(z$wall_ms,.9)),p95_ms=unname(quantile(z$wall_ms,.95)),max_ms=max(z$wall_ms),targets=sum(z$target_hit),certified_fits=sum(z$certified),sum_cpu_ms=sum(z$cpu_ms))))
}
for(name in c('family','status','overall')){
 groups<-switch(name,family=c('batch','design','method','normalized'),status=c('certified','method','normalized'),overall=c('method','normalized'))
 write.csv(summarize(groups),file.path(out,paste0(name,'.csv')),row.names=FALSE)
}
# Round totals expose repeat variation; no pooling over repeats to invent cases.
rt<-aggregate(x['wall_ms'],x[c('round','method','normalized')],sum);write.csv(rt,file.path(out,'round_totals.csv'),row.names=FALSE)
pk<-c('batch','design','n','rep','seed','method')
pairs<-merge(per_case[!per_case$normalized,],per_case[per_case$normalized,],by=pk,suffixes=c('_off','_on'))
pairs$delta_ms<-pairs$wall_ms_on-pairs$wall_ms_off
pairs$status_pair<-paste(pairs$certified_off,pairs$certified_on,sep='->')
write.csv(aggregate(pairs['delta_ms'],pairs[c('method','status_pair')],sum),file.path(out,'cost_changes.csv'),row.names=FALSE)
write_metadata(file.path(out,'metadata.csv'),values=list(smoke=smoke,cases=nrow(saved),rounds=4,calls_per_block=block_reps,timed_policy_calls=nrow(x)*block_reps,
 scheduling='serial; BLAS threads one; four cyclic route orders per case; paired normalization off/on with first order alternating; seed 202609301+round',
 timing='wall Sys.time and process CPU; per-case median of four block averages; model/sample construction, warmup, gc and verification outside timing; failures included',
 scope='R-accessible workflows, including R orchestration and automatic starts; cold uses up to two calls, warm uses native fallback; no inference or extra post-fit audit in timing',
 controls='normalization off/on; ML baseline layered/native; two-stage FABIN3/auto; ordinary L-BFGS/information; PSD SLSQP/diagonal; budget5000; f1e-12/x1e-10; PSD floor=feasibility=1e-6',
 library_md5=md5,git_head=magmaan_cache_ref()$git_head,platform=R.version$platform,sysname=Sys.info()[['sysname']],machine=Sys.info()[['machine']],elapsed_seconds=proc.time()[['elapsed']]-t0),packages='magmaanlab')
print(read.csv(file.path(out,'overall.csv')),row.names=FALSE);print(read.csv(file.path(out,'status.csv')),row.names=FALSE)
