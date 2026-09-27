#!/usr/bin/env Rscript
# Retrospective evaluation of one deterministic policy, not a production change:
# only certified ordinary estimates may warm-start PSD. Rejected estimates use
# the original FABIN3-auto start. Saved direct PSD fits supply that counterfactual.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help'%in%args){cat('Usage: Rscript scripts/check_two_stage_handoff.R [--run-id NAME]\n120 saved draws; normalized two-stage replay. Compare certified-only handoff with saved direct PSD outcomes.\n');quit(save='no')}
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
run<-if('--run-id'%in%args)args[match('--run-id',args)+1L]else'two-stage-handoff'
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run));out<-file.path(here,'results',run)
if(dir.exists(out))stop('choose a fresh run ID');dir.create(out,recursive=TRUE)
old<-read.csv(file.path(here,'results','normalization-revisit','comparisons.csv'))
meta<-read.csv(file.path(here,'results','normalization-revisit','metadata.csv'))
stopifnot(unname(tools::md5sum(system.file('libs','magmaanlab.so',package='magmaanlab')))==meta$value[meta$key=='library_md5'])
base<-subset(old,method=='two_stage_psd'&arm=='normalized');direct<-subset(old,method=='direct_psd'&arm=='normalized')
spec<-magmaanlab::model_spec(model_syntax)
ctl<-list(start='fabin3',start_transport='auto',normalize_sample=TRUE,coordinate_scaling='information',max_iter=5000L,nlopt=list(max_eval=5000L,ftol_rel=1e-12,xtol_rel=1e-10))
rows<-list()
for(i in seq_len(nrow(base))){
 b<-base[i,];samp<-sample_moments(draw_data(design_sigma(designs_all()[[b$design]]),b$n,b$seed))
 z<-suppressWarnings(magmaanlab::frontier_fit_ml_psd_fallback(spec,samp,ordinary_optimizer='nlopt-lbfgs',psd_optimizer='nlopt-slsqp',ordinary_control=ctl,psd_control=ctl[setdiff(names(ctl),c('start','start_transport'))],preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6))
 stopifnot(isTRUE(z$converged)==b$certified)
 selected<-if(z$fallback_used)z$psd$fit else z$ordinary$fit
 if(!is.null(selected))stopifnot(abs(selected$fmin-b$objective)<1e-9)
 change<-z$fallback_reason=='ordinary-rejected'&&z$warm_start_used
 d<-direct[direct$batch==b$batch&direct$design==b$design&direct$n==b$n&direct$rep==b$rep,];stopifnot(nrow(d)==1)
 candidate<-if(change)d else b
 rows[[i]]<-cbind(b[c('batch','design','n','rep','seed')],ordinary_returned=!is.null(z$ordinary$fit),ordinary_certified=isTRUE(z$ordinary$fit$converged),
  reason=z$fallback_reason,warm_start_used=z$warm_start_used,changed_handoff=change,
  baseline_accepted=b$accepted,candidate_accepted=candidate$accepted,
  baseline_hit=b$target_reached,candidate_hit=candidate$target_reached,
  baseline_objective=b$objective,candidate_objective=candidate$objective,
  gain=!b$target_reached&&candidate$target_reached,loss=b$target_reached&&!candidate$target_reached)
 if(i%%10==0)cat(i,'/',nrow(base),' draws\n',sep='')
}
x<-do.call(rbind,rows);write.csv(x,file.path(out,'paired.csv'),row.names=FALSE)
metrics<-c('changed_handoff','baseline_accepted','candidate_accepted','baseline_hit','candidate_hit','gain','loss')
for(groups in list(c('batch'),c('batch','design'))){s<-aggregate(x[metrics],x[groups],sum);write.csv(s,file.path(out,if(length(groups)==1)'summary.csv'else'by_family.csv'),row.names=FALSE);print(s,row.names=FALSE)}
write_metadata(file.path(out,'metadata.csv'),values=list(protocol='retrospective certified-only handoff candidate, all 120 saved draws; no tuning; no production policy change',
 candidate='if ordinary-rejected and warm start used, replace PSD outcome with saved normalized direct FABIN3/auto outcome; otherwise unchanged',
 validation='same library fingerprint as saved run; every replayed wrapper verdict and returned objective verified',git_head=magmaan_cache_ref()$git_head),packages='magmaanlab')
