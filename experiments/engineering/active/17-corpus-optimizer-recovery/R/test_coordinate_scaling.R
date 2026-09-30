#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/test_coordinate_scaling.R CORPUS RESULTS [--rescaled] [--smoke] [--workers N] [--scalings a,b] [--tag T] [--lavaan-only]
Fits every prepared ML/GLS case with PORT and L-BFGS in three optimizer coordinate systems (none,
sample_units, information) from the current and the layered start. --rescaled first multiplies each
observed variable by a fixed power of ten between 10^-2 and 10^2 and adds lavaan defaults and lavaan
from the layered start on the same rescaled data. --scalings restricts the magmaan arms (lavaan arms run only
with all three); --lavaan-only runs only the lavaan arms of --rescaled; --tag T writes to scaling[-rescaled]-T.
lavaan endpoints are judged by the magmaan verdict (the Newton check) at the lavaan estimate.
240 s per case/estimator job; every fit is checkpointed.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'))
rescaled<-'--rescaled' %in% args;smoke<-'--smoke' %in% args
opt_arg<-function(name,default)if(name %in% args)args[match(name,args)+1L] else default
scalings<-strsplit(opt_arg('--scalings','none,sample_units,information'),',')[[1]];tag<-opt_arg('--tag','')
lavaan_only<-'--lavaan-only' %in% args;if(lavaan_only)scalings<-character()
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE);jobs<-subset(d,arm=='port_default')
if(smoke)jobs<-subset(jobs,case %in% c('kline_2023_ch9_roth_illness_path','kline_2023_ch9_roth_illness_mean','little_2013_ch3_fig_3_11_longitudinal_cfa_phantom','kline_2023_ch15_worland_sr_step2b'))
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
dest<-file.path(out,paste0(if(rescaled)'scaling-rescaled' else 'scaling',if(nzchar(tag))paste0('-',tag) else '',if(smoke)'-smoke' else ''));dir.create(dest,showWarnings=FALSE)

# A fixed power of ten per observed variable, by its rank among the case's names.
unit_factors<-function(names){k<-match(names,sort(unique(names)));setNames(10^(((5*k)%%9-4)/2),names)}
rescale_case<-function(c) {
  ov<-unique(unlist(lapply(c$sample$S,rownames)));f<-unit_factors(ov)
  cov_scale<-function(S){g<-f[rownames(S)];S*outer(g,g)};mean_scale<-function(m)m*f[names(m)]
  c$sample$S<-lapply(c$sample$S,cov_scale)
  if(!is.null(c$sample$mean))c$sample$mean<-lapply(c$sample$mean,mean_scale)
  a<-c$args
  if(!is.null(a$data)) {for(v in intersect(names(a$data),ov))a$data[[v]]<-a$data[[v]]*f[[v]]} else {
    a$sample.cov<-if(is.list(a$sample.cov))lapply(a$sample.cov,cov_scale) else cov_scale(a$sample.cov)
    if(!is.null(a$sample.mean))a$sample.mean<-if(is.list(a$sample.mean))lapply(a$sample.mean,mean_scale) else mean_scale(a$sample.mean)
  }
  c$args<-a;c$pre<-suppressWarnings(do.call(c$fun,c(a,list(do.fit=FALSE))));c
}

if('--worker' %in% args) {
  j<-jobs[as.integer(args[match('--worker',args)+1L]),]
  c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  if(rescaled)c<-rescale_case(c)
  starts<-list(
    current=function()magmaan_core$estimate_start_values(c$model$partable,c$sample,start=if(j$estimator=='ML')'scaled-fabin' else 'fabin3',transport=if(j$estimator=='ML')'auto' else 'native'),
    layered=function()magmaan_core$estimate_start_values(c$model$partable,c$sample,start='layered'))
  fitter<-if(j$estimator=='ML')magmaan_core$fit_ml else magmaan_core$fit_gls
  rows<-list();file<-file.path(dest,paste0(j$case,'__',j$estimator,'.csv'))
  blank<-function(...)data.frame(case=j$case,estimator=j$estimator,mode=if(rescaled)'rescaled' else 'native',
    engine='magmaan',start='',optimizer='',scaling='',applied='',start_f=NA_real_,target=j$best_observed,returned=FALSE,
    accepted=FALSE,flag=NA,f=NA_real_,status='',newton_status='',iterations=NA_integer_,f_evals=NA_integer_,seconds=NA_real_,
    message='',stringsAsFactors=FALSE,...)
  record<-function(row){rows[[length(rows)+1L]]<<-row;write.csv(do.call(rbind,rows),file,row.names=FALSE)}
  layered<-NULL
  for(name in names(starts)) {
    x<-tryCatch(as.numeric(starts[[name]]()),error=function(e)e)
    if(name=='layered' && !inherits(x,'error'))layered<-x
    f0<-NA_real_
    if(!inherits(x,'error')){at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,x,j$estimator),error=function(e)NULL);if(!is.null(at))f0<-at$fmin}
    for(optimizer in c('port','nlopt-lbfgs')) for(scaling in scalings) {
      row<-blank();row$start<-name;row$optimizer<-optimizer;row$scaling<-scaling;row$start_f<-f0
      if(inherits(x,'error')) row$message<-conditionMessage(x) else {
        t<-proc.time()[['elapsed']]
        fit<-tryCatch(fitter(c$model,c$sample,optimizer=optimizer,control=list(start=x,coordinate_scaling=scaling)),error=function(e)e)
        row$seconds<-proc.time()[['elapsed']]-t
        if(inherits(fit,'error')) row$message<-conditionMessage(fit) else {
          row$returned<-TRUE;row$accepted<-isTRUE(fit$converged);row$f<-fit$fmin;row$status<-fit$optimizer_status %||% ''
          row$applied<-fit$coordinate_scaling %||% '';row$newton_status<-fit$diagnostics$newton_accuracy$status %||% ''
          row$iterations<-fit$iterations %||% NA_integer_;row$f_evals<-fit$f_evals %||% NA_integer_
        }
      }
      record(row)
    }
  }
  if(rescaled && (length(scalings)==3L || lavaan_only)) {
    mp<-c$model$partable;lp<-lavaan::parTable(c$pre);ix<-which(lp$free>0);mx<-match(row_key(lp[ix,]),row_key(mp))
    runs<-list(default=c$args)
    if(!is.null(layered) && !anyNA(mx)){a<-c$args;a$start<-layered[mp$free[mx]];runs$layered<-a}
    for(name in names(runs)) {
      row<-blank();row$engine<-'lavaan';row$start<-name;row$optimizer<-'nlminb';row$scaling<-'lavaan'
      t<-proc.time()[['elapsed']];fit<-tryCatch(suppressWarnings(do.call(c$fun,runs[[name]])),error=function(e)e)
      row$seconds<-proc.time()[['elapsed']]-t
      if(inherits(fit,'error')) row$message<-conditionMessage(fit) else {
        # accepted is the magmaan verdict (the Newton check) at the lavaan
        # estimate, the judge of every magmaan row; flag is the lavaan flag.
        row$returned<-TRUE;row$flag<-isTRUE(lavaan::lavInspect(fit,'converged'))
        row$iterations<-as.integer(lavaan::lavInspect(fit,'iterations'))
        at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,map_theta(c$model,fit),j$estimator),error=function(e)NULL)
        if(!is.null(at)){row$f<-at$fmin;row$accepted<-isTRUE(at$converged)
          row$newton_status<-at$diagnostics$newton_accuracy$status %||% ''}
      }
      record(row)
    }
  }
  quit(save='no')
}
workers<-if('--workers' %in% args) as.integer(args[match('--workers',args)+1L]) else 3L
write_metadata(file.path(dest,'metadata.csv'),values=list(pairs=nrow(jobs),engine=find.package('magmaanlab'),
  mode=if(rescaled)'each observed variable multiplied by 10^(((5k mod 9)-4)/2), k its rank among the names' else 'native units',
  arms=paste('current and layered start x PORT and L-BFGS x coordinate_scaling',paste(scalings,collapse=', ')),
  target='frozen best_observed in comparison.csv (native units)'),packages=c('magmaanlab','lavaan'))
r<-parallel::mclapply(seq_len(nrow(jobs)),function(i){j<-jobs[i,];stem<-paste0(j$case,'__',j$estimator);unlink(file.path(dest,paste0(stem,'.csv')))
  code<-system2('timeout',c('240',file.path(R.home('bin'),'Rscript'),shQuote(script),shQuote(root),shQuote(out),
    if(rescaled)'--rescaled',if(smoke)'--smoke',if(lavaan_only)'--lavaan-only' else c('--scalings',paste(scalings,collapse=',')),if(nzchar(tag))c('--tag',tag),'--worker',i),stdout=file.path(dest,paste0(stem,'.log')),stderr=file.path(dest,paste0(stem,'.log')))
  cat(sprintf('[%d/%d] %s %s exit=%d\n',i,nrow(jobs),j$case,j$estimator,code));data.frame(case=j$case,estimator=j$estimator,exit=code)
},mc.cores=workers,mc.preschedule=FALSE)
write.csv(do.call(rbind,r),file.path(dest,'jobs.csv'),row.names=FALSE)
