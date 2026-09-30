#!/usr/bin/env Rscript
# Exact ordinary-first policy requested by the user: accept only certified,
# admissible ordinary ML; otherwise run PSD from its own FABIN3-auto start.
# This composes library fits, with no warm start, retry, or parameter selection.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help'%in%args){cat('Usage: Rscript scripts/check_two_stage_cold.R [--smoke] [--run-id NAME]\n120 saved normalized challenge cases; ordinary L-BFGS/information, FABIN3/auto; independent PSD SLSQP/diagonal on any ordinary failure or inadmissibility.\n');quit(save='no')}
source(file.path(here,'..', '..', '..', '_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
smoke<-'--smoke'%in%args
run<-if('--run-id'%in%args)args[match('--run-id',args)+1L]else if(smoke)'two-stage-cold-smoke'else'two-stage-cold'
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run));out<-file.path(here,'results',run)
if(dir.exists(out))stop('choose a fresh run ID');dir.create(out,recursive=TRUE)
old<-read.csv(file.path(here,'results','normalization-revisit','comparisons.csv'))
meta<-read.csv(file.path(here,'results','normalization-revisit','metadata.csv'))
md5<-unname(tools::md5sum(system.file('libs','magmaanlab.so',package='magmaanlab')))
stopifnot(md5==meta$value[meta$key=='library_md5'])
base<-subset(old,method=='two_stage_psd'&arm=='normalized');direct<-subset(old,method=='direct_psd'&arm=='normalized')
if(smoke)base<-head(base,6)
spec<-magmaanlab::model_spec(model_syntax)
ctl<-list(start='fabin3',start_transport='auto',normalize_sample=TRUE,coordinate_scaling='information',max_iter=5000L,nlopt=list(max_eval=5000L,ftol_rel=1e-12,xtol_rel=1e-10))
rows<-stages<-parameters<-list()
assess<-function(fit,target){
 if(inherits(fit,'error'))return(data.frame(returned=FALSE,certified=FALSE,admissible=FALSE,accuracy=FALSE,chart=FALSE,accepted=FALSE,objective=NA_real_,target_hit=FALSE,evaluations=NA_integer_,message=conditionMessage(fit)))
 stopifnot(isTRUE(fit$sample_normalized))
 audit<-magmaanlab::frontier_newton_accuracy(fit,psd=TRUE)
 chart<-!inherits(tryCatch(magmaanlab::frontier_reidentify(fit$partable,spec,pole_tol=1e-4),error=function(e)e),'error')
 accepted<-isTRUE(fit$converged)&&isTRUE(fit$diagnostics$admissibility$admissible)&&isTRUE(audit$passed)&&chart
 data.frame(returned=TRUE,certified=isTRUE(fit$converged),admissible=isTRUE(fit$diagnostics$admissibility$admissible),accuracy=isTRUE(audit$passed),chart=chart,accepted=accepted,objective=fit$fmin,
 target_hit=accepted&&abs(fit$fmin-target)<=1e-6*(1+abs(target)),evaluations=fit$f_evals,message='')
}
for(i in seq_len(nrow(base))){
 b<-base[i,];key<-b[c('batch','design','n','rep','seed')]
 d<-direct[direct$batch==b$batch&direct$design==b$design&direct$n==b$n&direct$rep==b$rep,];stopifnot(nrow(d)==1)
 samp<-sample_moments(draw_data(design_sigma(designs_all()[[b$design]]),b$n,b$seed))
 ordinary<-tryCatch(suppressWarnings(magmaanlab::magmaan_core$fit_ml(spec,samp,optimizer='nlopt-lbfgs',control=ctl)),error=function(e)e)
 ordinary_error<-inherits(ordinary,'error')
 take_ordinary<-!ordinary_error&&isTRUE(ordinary$converged)&&isTRUE(ordinary$diagnostics$admissibility$admissible)
 reason<-if(take_ordinary)'ordinary-accepted'else if(ordinary_error)'ordinary-error'else if(!isTRUE(ordinary$converged))'ordinary-rejected'else'ordinary-inadmissible'
 stage<-assess(ordinary,b$best_objective);stages[[length(stages)+1L]]<-cbind(key,stage='ordinary',stage)
 fit<-ordinary
 if(!take_ordinary){
  # Same untouched spec and automatic start controls; no ordinary theta enters PSD.
  fit<-tryCatch(suppressWarnings(magmaanlab::frontier_fit_ml_psd(spec,samp,optimizer='nlopt-slsqp',control=ctl,preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6)),error=function(e)e)
 }
 candidate<-assess(fit,b$best_objective)
 if(!take_ordinary){
  stages[[length(stages)+1L]]<-cbind(key,stage='psd',candidate)
  stopifnot(candidate$returned==d$returned,candidate$certified==d$certified,candidate$accepted==d$accepted)
  if(candidate$returned)stopifnot(abs(candidate$objective-d$objective)<1e-9)
 }else{
  stopifnot(b$stage=='ordinary',candidate$certified==b$certified,abs(candidate$objective-b$objective)<1e-9)
 }
 if(!inherits(fit,'error'))parameters[[length(parameters)+1L]]<-cbind(key[rep(1,length(fit$theta)),],parameter=seq_along(fit$theta),value=fit$theta)
 rows[[i]]<-cbind(key,ordinary_accepted=take_ordinary,fallback_used=!take_ordinary,reason=reason,
  cold_accepted=candidate$accepted,cold_hit=candidate$target_hit,cold_objective=candidate$objective,
  warm_accepted=b$accepted,warm_hit=b$target_reached,warm_objective=b$objective,
  direct_accepted=d$accepted,direct_hit=d$target_reached,direct_objective=d$objective,
  gain_vs_warm=!b$target_reached&&candidate$target_hit,loss_vs_warm=b$target_reached&&!candidate$target_hit,
  gain_vs_direct=!d$target_reached&&candidate$target_hit,loss_vs_direct=d$target_reached&&!candidate$target_hit)
 if(i%%10==0||i==nrow(base))cat(i,'/',nrow(base),' cases\n',sep='')
}
x<-do.call(rbind,rows)
write<-function(z,n)write.csv(z,file.path(out,paste0(n,'.csv')),row.names=FALSE)
write(x,'paired');write(do.call(rbind,stages),'stages');write(do.call(rbind,parameters),'parameters')
write(x[x$gain_vs_warm|x$loss_vs_warm|x$gain_vs_direct|x$loss_vs_direct,],'changes')
metrics<-c('ordinary_accepted','fallback_used','cold_accepted','cold_hit','warm_accepted','warm_hit','direct_accepted','direct_hit','gain_vs_warm','loss_vs_warm','gain_vs_direct','loss_vs_direct')
for(groups in list('batch',c('batch','design'))){s<-aggregate(x[metrics],x[groups],sum);write(s,if(length(groups)==1)'summary'else'by_family');print(s,row.names=FALSE)}
write_metadata(file.path(out,'metadata.csv'),values=list(smoke=smoke,cases=nrow(x),policy='ordinary certified AND admissible => return; otherwise independent PSD FABIN3-auto; no ordinary endpoint reused',
 controls='both stages normalized; FABIN3/auto; ordinary L-BFGS/information; PSD SLSQP/diagonal; 5000 evaluations; f=1e-12, x=1e-10, floor=feasibility=1e-6',
 judging='same saved PSD targets and current library; convergence, PSD audit, chart at 1e-4; no magnitude cutoff',
 verification='every cold fallback reproduces saved direct PSD verdict/objective; every accepted ordinary reproduces saved warm-route ordinary fit; exact library fingerprint checked',
 status='saved development data, no algorithm or production policy change',git_head=magmaan_cache_ref()$git_head,library_md5=md5),packages='magmaanlab')
