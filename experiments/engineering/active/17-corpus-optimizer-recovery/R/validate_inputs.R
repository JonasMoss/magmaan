#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
root<-normalizePath(args[1]);out<-normalizePath(args[2])
script<-normalizePath(sub("--file=","",grep("--file=",commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),"../../../../_support/R/helpers.R"))
set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),"inputs.R"))
manifest<-read.csv(file.path(root,"manifest.csv"),stringsAsFactors=FALSE)
paths<-list.files(out,pattern="__(ML|GLS)\\.csv$",full.names=TRUE)
jobs<-do.call(rbind,lapply(paths,function(p){d<-read.csv(p); d[1,c("case","estimator")]}))
rows<-parallel::mclapply(seq_len(nrow(jobs)),function(i) {
  j<-jobs[i,]; ng<-pmax<-nf<-NA_integer_; message<-tryCatch({
    c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
    ng<-length(c$sample$S);pmax<-max(vapply(c$sample$S,nrow,integer(1)));nf<-max(c$model$partable$free)
    mp<-c$model$partable;lp<-lavaan::parTable(c$pre)
    mp<-mp[mp$op %in% c("=~","~~","~","~1"),]
    lp<-lp[lp$op %in% c("=~","~~","~","~1"),]
    idx<-match(row_key(mp),row_key(lp))
    if(nrow(mp)!=nrow(lp) || anyNA(idx) || anyDuplicated(row_key(mp))) stop("formula row sets differ")
    lp<-lp[idx,]
    if(any((mp$free>0)!=(lp$free>0))) stop("parameter freedom differs")
    fixed<-mp$free==0 & is.finite(mp$ustart) & is.finite(lp$ustart)
    if(any(abs(mp$ustart[fixed]-lp$ustart[fixed])>1e-12)) stop("fixed parameter values differ")
    "passed"
  },error=function(e) conditionMessage(e))
  data.frame(case=j$case,estimator=j$estimator,contract=message,n_groups=ng,max_observed=pmax,n_free=nf)
},mc.cores=3,mc.preschedule=FALSE)
write.csv(do.call(rbind,rows),file.path(out,"input_contracts.csv"),row.names=FALSE)
