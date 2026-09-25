#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/test_layered_starts.R CORPUS RESULTS [--smoke] [--workers N]\nFits every prepared ML/GLS case from the current and the layered start with PORT and L-BFGS on the\ninstalled engine (both starts share it). 120 s per case/estimator job. Frozen best objectives define the target.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'));source(file.path(dirname(script),'arms.R'))
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE);jobs<-subset(d,arm=='port_default')
if('--smoke' %in% args)jobs<-subset(jobs,case %in% c('little_2013_ch3_fig_3_11_longitudinal_cfa_phantom','little_2013_ch8_fig7_bull_phant','kline_2023_ch15_worland_sr_step2b'))
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
dest<-file.path(out,if('--smoke' %in% args)'layered-smoke' else 'layered');dir.create(dest,showWarnings=FALSE)
if('--worker' %in% args) {
  j<-jobs[as.integer(args[match('--worker',args)+1L]),]
  c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  starts<-list(
    current=function()magmaan_core$estimate_start_values(c$model$partable,c$sample,start=if(j$estimator=='ML')'scaled-fabin' else 'fabin3',transport=if(j$estimator=='ML')'auto' else 'native'),
    layered=function()magmaan_core$estimate_start_values(c$model$partable,c$sample,start='layered'))
  rows<-list()
  for(name in names(starts)) {
    t<-proc.time()[['elapsed']];x<-tryCatch(starts[[name]](),error=function(e)e);start_seconds<-proc.time()[['elapsed']]-t
    notes<-if(inherits(x,'error')) conditionMessage(x) else paste(attr(x,'start_notes'),collapse='; ')
    f0<-NA_real_
    if(!inherits(x,'error')) {at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,as.numeric(x),j$estimator),error=function(e)NULL);if(!is.null(at))f0<-at$fmin}
    for(arm in c('port_default','lbfgs_default')) {
      row<-data.frame(case=j$case,estimator=j$estimator,start=name,arm=arm,start_seconds=start_seconds,start_f=f0,
        target=j$best_observed,returned=FALSE,accepted=FALSE,f=NA_real_,status='',newton_status='',
        f_evals=NA_integer_,seconds=NA_real_,notes=notes,message='',stringsAsFactors=FALSE)
      if(!inherits(x,'error')) {
        t<-proc.time()[['elapsed']];fit<-tryCatch(one_fit(c,j$estimator,optimizer_arms()[[arm]],as.numeric(x)),error=function(e)e)
        row$seconds<-proc.time()[['elapsed']]-t
        if(inherits(fit,'error')) row$message<-conditionMessage(fit) else {
          row$returned<-TRUE;row$accepted<-isTRUE(fit$converged);row$f<-fit$fmin;row$status<-fit$optimizer_status %||% ''
          row$newton_status<-fit$diagnostics$newton_accuracy$status %||% '';row$f_evals<-fit$f_evals %||% NA_integer_
        }
      }
      rows[[length(rows)+1L]]<-row
      # Checkpoint every fit; a late timeout must not discard earlier arms.
      write.csv(do.call(rbind,rows),file.path(dest,paste0(j$case,'__',j$estimator,'.csv')),row.names=FALSE)
    }
  }
  quit(save='no')
}
workers<-if('--workers' %in% args) as.integer(args[match('--workers',args)+1L]) else 3L
write_metadata(file.path(dest,'metadata.csv'),values=list(pairs=nrow(jobs),engine=find.package('magmaanlab'),
  starts='current (ML scaled-fabin auto, GLS fabin3 native) versus layered',target='frozen best_observed in comparison.csv'),
  packages=c('magmaanlab','lavaan'))
r<-parallel::mclapply(seq_len(nrow(jobs)),function(i){j<-jobs[i,];stem<-paste0(j$case,'__',j$estimator);unlink(file.path(dest,paste0(stem,'.csv')))
  code<-system2('timeout',c('120',file.path(R.home('bin'),'Rscript'),shQuote(script),shQuote(root),shQuote(out),if('--smoke' %in% args)'--smoke','--worker',i),stdout=file.path(dest,paste0(stem,'.log')),stderr=file.path(dest,paste0(stem,'.log')))
  cat(sprintf('[%d/%d] %s %s exit=%d\n',i,nrow(jobs),j$case,j$estimator,code));data.frame(case=j$case,estimator=j$estimator,exit=code)
},mc.cores=workers,mc.preschedule=FALSE)
write.csv(do.call(rbind,r),file.path(dest,'jobs.csv'),row.names=FALSE)
