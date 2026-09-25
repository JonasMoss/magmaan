#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/test_lavaan_defaults.R CORPUS RESULTS [--smoke] [--workers N]\nFits every prepared ML/GLS case with lavaan defaults and with lavaan from the layered start.\nEndpoints are evaluated with the installed magmaan objective. 120 s per case/estimator job.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'))
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE);jobs<-subset(d,arm=='port_default')
if('--smoke' %in% args)jobs<-subset(jobs,case %in% c('little_2013_ch3_fig_3_11_longitudinal_cfa_phantom','kline_2023_ch15_worland_sr_step2b'))
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
dest<-file.path(out,if('--smoke' %in% args)'lavaan-smoke' else 'lavaan');dir.create(dest,showWarnings=FALSE)
if('--worker' %in% args) {
  j<-jobs[as.integer(args[match('--worker',args)+1L]),]
  c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  mp<-c$model$partable;lp<-lavaan::parTable(c$pre)
  ix<-which(lp$free>0);mx<-match(row_key(lp[ix,]),row_key(mp));stopifnot(!anyNA(mx))
  layered<-tryCatch(as.numeric(magmaan_core$estimate_start_values(mp,c$sample,start='layered')),error=function(e)NULL)
  runs<-list(lavaan_default=c$args)
  if(!is.null(layered)) {a<-c$args;a$start<-layered[mp$free[mx]];runs$lavaan_layered<-a}
  rows<-list()
  for(name in names(runs)) {
    t<-proc.time()[['elapsed']];fit<-tryCatch(suppressWarnings(do.call(c$fun,runs[[name]])),error=function(e)e)
    row<-data.frame(case=j$case,estimator=j$estimator,arm=name,seconds=proc.time()[['elapsed']]-t,target=j$best_observed,
      returned=!inherits(fit,'error'),converged=FALSE,f=NA_real_,iterations=NA_integer_,message='',stringsAsFactors=FALSE)
    if(inherits(fit,'error')) row$message<-conditionMessage(fit) else {
      row$converged<-isTRUE(lavaan::lavInspect(fit,'converged'))
      row$iterations<-as.integer(lavaan::lavInspect(fit,'iterations'))
      at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,map_theta(c$model,fit),j$estimator),error=function(e)NULL)
      if(!is.null(at)) row$f<-at$fmin
    }
    rows[[length(rows)+1L]]<-row
    write.csv(do.call(rbind,rows),file.path(dest,paste0(j$case,'__',j$estimator,'.csv')),row.names=FALSE)
  }
  quit(save='no')
}
workers<-if('--workers' %in% args) as.integer(args[match('--workers',args)+1L]) else 3L
write_metadata(file.path(dest,'metadata.csv'),values=list(pairs=nrow(jobs),engine=find.package('magmaanlab'),
  arms='lavaan defaults; lavaan from the layered start',objective='installed magmaan evaluate_at at the lavaan estimates'),
  packages=c('magmaanlab','lavaan'))
r<-parallel::mclapply(seq_len(nrow(jobs)),function(i){j<-jobs[i,];stem<-paste0(j$case,'__',j$estimator);unlink(file.path(dest,paste0(stem,'.csv')))
  code<-system2('timeout',c('120',file.path(R.home('bin'),'Rscript'),shQuote(script),shQuote(root),shQuote(out),if('--smoke' %in% args)'--smoke','--worker',i),stdout=file.path(dest,paste0(stem,'.log')),stderr=file.path(dest,paste0(stem,'.log')))
  cat(sprintf('[%d/%d] %s %s exit=%d\n',i,nrow(jobs),j$case,j$estimator,code));data.frame(case=j$case,estimator=j$estimator,exit=code)
},mc.cores=workers,mc.preschedule=FALSE)
write.csv(do.call(rbind,r),file.path(dest,'jobs.csv'),row.names=FALSE)
