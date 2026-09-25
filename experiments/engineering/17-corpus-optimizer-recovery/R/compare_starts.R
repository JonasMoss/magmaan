#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args) {cat('Usage: Rscript R/compare_starts.R CORPUS RESULTS [--smoke]\nCross PORT/lavaan with current, simple, lavaan-default, nonzero-path and reference starts.\n21 targeted case/estimator pairs; three workers, 120-second cap per pair.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2])
script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'));source(file.path(dirname(script),'arms.R'))
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
d<-read.csv(file.path(out,'port_failures.csv'),stringsAsFactors=FALSE)
jobs<-subset(d,failure_class %in% c('audit rejected','solver error'))
if('--smoke' %in% args) jobs<-subset(jobs,case=='little_2013_ch3_fig_3_11_longitudinal_cfa_phantom')
dest<-file.path(out,if('--smoke' %in% args) 'starts-smoke' else 'starts');dir.create(dest,showWarnings=FALSE)
append_row<-function(x,path) write.table(x,path,sep=',',row.names=FALSE,col.names=!file.exists(path),append=file.exists(path),qmethod='double')
if('--worker' %in% args) {
  i<-as.integer(args[match('--worker',args)+1]);j<-jobs[i,]
  path<-file.path(dest,paste0(j$case,'__',j$estimator,'.csv'))
  c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  mp<-c$model$partable;lp<-lavaan::parTable(c$pre)
  mi<-which(mp$free>0);li<-match(row_key(mp[mi,]),row_key(lp));stopifnot(!anyNA(li))
  lv<-numeric(max(mp$free));lv[mp$free[mi]]<-lp$start[li]
  starts<-list(current=magmaan_core$estimate_start_values(mp,c$sample,start=if(j$estimator=='ML') 'scaled-fabin' else 'fabin3',transport=if(j$estimator=='ML') 'auto' else 'native'),
               simple=magmaan_core$estimate_start_values(mp,c$sample,start='simple',transport='native'),lavaan_default=lv)
  latent<-unique(mp$lhs[mp$op=='=~'])
  paths<-which(mp$op=='~' & mp$lhs %in% latent & mp$rhs %in% latent & mp$free>0)
  indices<-unique(mp$free[paths]);indices<-indices[abs(starts$current[indices])<1e-12]
  starts$nonzero_latent_paths<-starts$current
  starts$nonzero_latent_paths[indices]<-0.5
  for(name in names(starts)) {
    a<-mp[mi,c('lhs','op','rhs','group','free')];a$start<-name;a$value<-starts[[name]][a$free]
    append_row(a,file.path(dest,paste0(j$case,'__',j$estimator,'__vectors.csv')))
  }
  record<-function(engine,name,run) {
    t<-proc.time()[['elapsed']]
    row<-data.frame(case=j$case,estimator=j$estimator,engine=engine,start=name,returned=FALSE,accepted=FALSE,
      objective=NA_real_,target=j$best_observed,matched=FALSE,start_objective=NA_real_,status='',newton_status='',message='',seconds=0)
    if(name %in% names(starts)) {
      at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,starts[[name]],j$estimator),error=function(e)NULL)
      if(!is.null(at)) row$start_objective<-at$fmin
    }
    fit<-tryCatch(run(),error=function(e)e)
    if(inherits(fit,'error')) row$message<-conditionMessage(fit) else {
      row$returned<-TRUE
      if(engine=='PORT') {row$accepted<-isTRUE(fit$converged);row$objective<-fit$fmin;row$status<-fit$optimizer_status;row$newton_status<-fit$diagnostics$newton_accuracy$status}
      else {
        row$accepted<-isTRUE(lavaan::lavInspect(fit,'converged'));row$status<-if(row$accepted) 'converged' else 'not converged'
        at<-tryCatch(magmaan_core$evaluate_at(c$model,c$sample,map_theta(c$model,fit),j$estimator),error=function(e)NULL)
        if(!is.null(at)) row$objective<-at$fmin
      }
    }
    row$matched<-is.finite(row$objective) && abs(row$objective-row$target)<=1e-6*(1+abs(row$target))
    row$seconds<-proc.time()[['elapsed']]-t;append_row(row,path);fit
  }
  # No corpus start-values file is supplied here; syntax hints still apply.
  lf<-record('lavaan','lavaan_default',function()suppressWarnings(do.call(c$fun,c$args)))
  for(name in names(starts)) record('PORT',name,function()one_fit(c,j$estimator,optimizer_arms()$port_default,starts[[name]]))
  # Cross the engines in the other direction, preserving parameter row identities.
  a<-c$args;ix<-which(lp$free>0);mx<-match(row_key(lp[ix,]),row_key(mp));stopifnot(!anyNA(mx))
  for(name in c('current','nonzero_latent_paths')) {
    a$start<-as.numeric(starts[[name]][mp$free[mx]])
    record('lavaan',name,function()suppressWarnings(do.call(c$fun,a)))
  }
  ref<-if(length(c$meta$model_options$start_values)) tryCatch(reference_fit(c),error=function(e)e) else lf
  if(!inherits(ref,'error') && isTRUE(lavaan::lavInspect(ref,'converged'))) {
    starts$reference_solution<-map_theta(c$model,ref)
    record('PORT','reference_solution',function()one_fit(c,j$estimator,optimizer_arms()$port_default,starts$reference_solution))
  }
  quit(save='no')
}
for(n in c('input_hashes','library_hashes')) {
  h<-read.csv(file.path(out,paste0(n,'.csv')));stopifnot(identical(h$md5,unname(tools::md5sum(h$path))))
}
write_metadata(file.path(dest,'metadata.csv'),values=list(design='PORT/current,simple,lavaan,nonzero-paths,solution; lavaan/default,current,nonzero-paths',pairs=nrow(jobs),timeout=120),packages=c('magmaanlab','lavaan'))
statuses<-parallel::mclapply(seq_len(nrow(jobs)),function(i) {
  j<-jobs[i,];stem<-paste0(j$case,'__',j$estimator)
  unlink(file.path(dest,paste0(stem,c('.csv','__vectors.csv'))))
  argv<-c('120',file.path(R.home('bin'),'Rscript'),shQuote(script),shQuote(root),shQuote(out),if('--smoke' %in% args)'--smoke','--worker',i)
  code<-system2('timeout',argv,stdout=file.path(dest,paste0(stem,'.log')),stderr=file.path(dest,paste0(stem,'.log')))
  cat(sprintf('[%d/%d] %s %s exit=%d\n',i,nrow(jobs),j$case,j$estimator,code));data.frame(case=j$case,estimator=j$estimator,exit=code)
},mc.cores=3,mc.preschedule=FALSE)
write.csv(do.call(rbind,statuses),file.path(dest,'jobs.csv'),row.names=FALSE)
paths<-list.files(dest,pattern='__(ML|GLS)\\.csv$',full.names=TRUE)
x<-do.call(rbind,lapply(paths,read.csv,stringsAsFactors=FALSE));write.csv(x,file.path(dest,'comparison.csv'),row.names=FALSE)
for(n in c('input_hashes','library_hashes')) {
  h<-read.csv(file.path(out,paste0(n,'.csv')));stopifnot(identical(h$md5,unname(tools::md5sum(h$path))))
}
cat('Completed; results:',dest,'\n')
