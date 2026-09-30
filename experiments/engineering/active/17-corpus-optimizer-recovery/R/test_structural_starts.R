#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/test_structural_starts.R CORPUS RESULTS [--smoke]\nTests a moment-based structural-start prototype against the frozen ML/GLS corpus baseline.\nUnchanged vectors reuse baseline fits; changed vectors run PORT and L-BFGS, 120s per job.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'));source(file.path(dirname(script),'arms.R'));source(file.path(dirname(script),'structural_start.R'))
d<-read.csv(file.path(out,'comparison.csv'));jobs<-subset(d,arm=='port_default')
if('--smoke' %in% args)jobs<-subset(jobs,case %in% c('little_2013_ch3_fig_3_11_longitudinal_cfa_phantom','little_2013_ch7_tab7_8_pruned_initial'))
manifest<-read.csv(file.path(root,'manifest.csv'));dest<-file.path(out,if('--smoke' %in% args)'structural-smoke' else 'structural');dir.create(dest,showWarnings=FALSE)
if('--worker' %in% args) {
  j<-jobs[as.integer(args[match('--worker',args)+1L]),];c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  x<-magmaan_core$estimate_start_values(c$model$partable,c$sample,start=if(j$estimator=='ML')'scaled-fabin' else 'fabin3',transport=if(j$estimator=='ML')'auto' else 'native')
  y<-structural_start(c,x);changed<-!identical(as.numeric(x),as.numeric(y$theta));initial<-NA_real_
  if(changed) {
    at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,y$theta,j$estimator),error=function(e)NULL)
    if(!is.null(at))initial<-at$fmin
    if(!is.finite(initial)){y$theta<-as.numeric(x);y$reason<-'nonfinite proposed objective';changed<-FALSE}
  }
  rows<-list()
  for(arm in c('port_default','lbfgs_default')) {
    b<-d[d$case==j$case & d$estimator==j$estimator & d$arm==arm,];stopifnot(nrow(b)==1)
    row<-data.frame(case=j$case,estimator=j$estimator,arm=arm,changed=changed,paths=y$paths,reason=y$reason,
      start_objective=initial,baseline_accepted=b$converged,baseline_f=b$f,target=b$best_observed,
      accepted=b$converged,f=b$f,status=b$status,newton_status='',message='',seconds=0)
    if(changed) {
      t<-proc.time()[['elapsed']];f<-tryCatch(one_fit(c,j$estimator,optimizer_arms()[[arm]],y$theta),error=function(e)e)
      row$seconds<-proc.time()[['elapsed']]-t
      if(inherits(f,'error')){row$accepted<-FALSE;row$f<-NA_real_;row$status<-'error';row$message<-conditionMessage(f)} else {
        row$accepted<-isTRUE(f$converged);row$f<-f$fmin;row$status<-f$optimizer_status;row$newton_status<-f$diagnostics$newton_accuracy$status
      }
    }
    rows[[length(rows)+1L]]<-row
    # Checkpoint each optimizer; a late timeout must not discard the first.
    write.csv(do.call(rbind,rows),file.path(dest,paste0(j$case,'__',j$estimator,'.csv')),row.names=FALSE)
  }
  if(changed){p<-c$model$partable;r<-which(p$free>0);write.csv(data.frame(p[r,c('lhs','op','rhs','group','free')],original=x[p$free[r]],candidate=y$theta[p$free[r]]),file.path(dest,paste0(j$case,'__',j$estimator,'__vectors.csv')),row.names=FALSE)}
  quit(save='no')
}
for(n in c('input_hashes','library_hashes')){h<-read.csv(file.path(out,paste0(n,'.csv')));stopifnot(identical(h$md5,unname(tools::md5sum(h$path))))}
write_metadata(file.path(dest,'metadata.csv'),values=list(pairs=nrow(jobs),engine='pinned parent library_hashes.csv',prototype='single-parent deterministic measured child; off-diagonal moments; affine projection'),packages=c('magmaanlab','lavaan'))
r<-parallel::mclapply(seq_len(nrow(jobs)),function(i){j<-jobs[i,];stem<-paste0(j$case,'__',j$estimator);unlink(file.path(dest,paste0(stem,'.csv')))
  code<-system2('timeout',c('120',file.path(R.home('bin'),'Rscript'),shQuote(script),shQuote(root),shQuote(out),if('--smoke' %in% args)'--smoke','--worker',i),stdout=file.path(dest,paste0(stem,'.log')),stderr=file.path(dest,paste0(stem,'.log')))
  cat(sprintf('[%d/%d] %s %s exit=%d\n',i,nrow(jobs),j$case,j$estimator,code));data.frame(case=j$case,estimator=j$estimator,exit=code)
},mc.cores=3,mc.preschedule=FALSE)
write.csv(do.call(rbind,r),file.path(dest,'jobs.csv'),row.names=FALSE)
p<-list.files(dest,pattern='__(ML|GLS)\\.csv$',full.names=TRUE);z<-do.call(rbind,lapply(p,read.csv))
z$good<-z$accepted & is.finite(z$f) & z$f<=z$target+1e-6*(1+abs(z$target))
z$baseline_good<-z$baseline_accepted & is.finite(z$baseline_f) & z$baseline_f<=z$target+1e-6*(1+abs(z$target))
z$rescued<-z$good & !z$baseline_good;z$lost<-!z$good & z$baseline_good
write.csv(z,file.path(dest,'comparison.csv'),row.names=FALSE)
print(aggregate(cbind(changed,good,baseline_good,rescued,lost)~estimator+arm,z,sum),row.names=FALSE)
for(n in c('input_hashes','library_hashes')){h<-read.csv(file.path(out,paste0(n,'.csv')));stopifnot(identical(h$md5,unname(tools::md5sum(h$path))))}
