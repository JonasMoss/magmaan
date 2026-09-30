#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/test_fiml_scaling.R CORPUS RESULTS
Fits the single-group continuous FIML corpus cases with magmaan L-BFGS in three optimizer coordinate
systems (none, sample_units, information) and SLSQP unscaled and in sample units, in native units and
after multiplying each observed variable by the same fixed powers of ten as test_coordinate_scaling.R.
Writes RESULTS/fiml_scaling.csv. A few minutes.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
unit_factors<-function(names){k<-match(names,sort(unique(names)));setNames(10^(((5*k)%%9-4)/2),names)}
rows<-list()
for(i in seq_len(nrow(manifest))) {
  dir<-file.path(root,manifest$case_dir[i]);meta<-fromJSON(file.path(dir,'meta.json'),simplifyVector=TRUE);mo<-meta$model_options
  if(!identical(mo$missing,'fiml') || length(mo$ordered) || isTRUE(meta$out_of_scope) || !identical(meta$data$kind,'raw') ||
     length(meta$data$group_var))next
  syntax<-paste(readLines(file.path(dir,'model.lav'),warn=FALSE),collapse='\n')
  if(grepl('<~|level:',syntax))next
  data<-read.csv(file.path(dir,meta$data$files$raw),check.names=FALSE)
  sp<-list(syntax=syntax,meanstructure=TRUE,fixed_x=isTRUE(mo$fixed_x),auto_cov_y=meta$lavaan_function %in% c('sem','cfa','growth'))
  if(identical(meta$lavaan_function,'growth'))sp$model_type<-'growth'
  model<-tryCatch(do.call(magmaanlab::model_spec,sp),error=function(e)NULL)
  if(is.null(model))next
  ov<-intersect(names(data),unique(c(model$partable$lhs,model$partable$rhs)))
  for(mode in c('native','rescaled')) {
    d<-data;if(mode=='rescaled'){f<-unit_factors(ov);for(v in ov)d[[v]]<-d[[v]]*f[[v]]}
    raw<-tryCatch(magmaanlab::df_to_fiml_data(d,model),error=function(e)e)
    arms<-list(c('nlopt-lbfgs','none'),c('nlopt-lbfgs','sample_units'),c('nlopt-lbfgs','information'),
               c('nlopt-slsqp','none'),c('nlopt-slsqp','sample_units'))
    for(a in arms) {
      row<-data.frame(case=manifest$case_id[i],mode=mode,optimizer=a[1],scaling=a[2],applied='',returned=FALSE,
        accepted=FALSE,f=NA_real_,f_evals=NA_integer_,message='',stringsAsFactors=FALSE)
      if(inherits(raw,'error')) row$message<-conditionMessage(raw) else {
        fit<-tryCatch(magmaan_core$estimate_fiml(model$partable,raw,optimizer=a[1],control=list(coordinate_scaling=a[2])),error=function(e)e)
        if(inherits(fit,'error')) row$message<-conditionMessage(fit) else {
          row$returned<-TRUE;row$accepted<-isTRUE(fit$converged);row$f<-fit$fmin
          row$applied<-fit$coordinate_scaling %||% '';row$f_evals<-fit$f_evals %||% NA_integer_
        }
      }
      rows[[length(rows)+1L]]<-row
    }
  }
  cat(manifest$case_id[i],'done\n')
}
z<-do.call(rbind,rows)
key<-paste(z$case,z$mode);best<-tapply(ifelse(z$accepted & is.finite(z$f),z$f,Inf),key,min)
z$best<-best[key];z$good<-z$accepted & is.finite(z$f) & z$f<=z$best+1e-6*(1+abs(z$best))
write.csv(z,file.path(out,'fiml_scaling.csv'),row.names=FALSE)
print(aggregate(cbind(fits=1,good,accepted)~mode+optimizer+scaling,z,sum),row.names=FALSE)
