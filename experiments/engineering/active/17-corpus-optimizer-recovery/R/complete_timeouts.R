#!/usr/bin/env Rscript
# Give an arm omitted by the cumulative job cap its own bounded attempt.
# Preserve the first sweep's jobs/logs, and append only previously absent arms.
args<-commandArgs(TRUE);root<-normalizePath(args[1]);out<-normalizePath(args[2])
script<-normalizePath(sub("--file=","",grep("--file=",commandArgs(),value=TRUE)[1]))
here<-dirname(dirname(script))
source(file.path(here,"../../../_support/R/helpers.R"))
set_single_threaded_math()
source(file.path(here,"R/arms.R"))
jobs<-read.csv(file.path(out,"jobs.csv"),stringsAsFactors=FALSE)
missing<-list()
for(i in which(jobs$exit!=0)) {
  j<-jobs[i,];p<-file.path(out,paste0(j$case,"__",j$estimator,".csv"))
  d<-if(file.exists(p)) read.csv(p,stringsAsFactors=FALSE) else NULL
  if(!is.null(d) && any(d$arm=="preparation")) next
  for(arm in setdiff(names(optimizer_arms()),d$arm))
    missing[[length(missing)+1]]<-data.frame(case=j$case,estimator=j$estimator,arm=arm)
}
if(!length(missing)) {
  empty<-data.frame(case=character(),estimator=character(),arm=character(),exit=integer(),directory=character())
  write.csv(empty,file.path(out,"retry_jobs.csv"),row.names=FALSE)
  quit(save="no")
}
missing<-do.call(rbind,missing)
write.csv(missing,file.path(out,"retry_plan.csv"),row.names=FALSE)
statuses<-parallel::mclapply(seq_len(nrow(missing)),function(i) {
  j<-missing[i,];dir<-file.path(out,"retries",paste(j$case,j$estimator,j$arm,sep="__"))
  dir.create(dir,recursive=TRUE,showWarnings=FALSE)
  result<-file.path(dir,paste0(j$case,"__",j$estimator,".csv"))
  if(file.exists(result)) unlink(result)
  code<-system2("timeout",c("120",file.path(R.home("bin"),"Rscript"),shQuote(file.path(here,"run_experiment.R")),
      "--worker",j$case,"--estimator",j$estimator,"--arms",j$arm,"--corpus",shQuote(root),"--results",shQuote(dir)),
      stdout=file.path(dir,"run.log"),stderr=file.path(dir,"run.log"))
  cat(sprintf("retry [%d/%d] %s %s %s exit=%d\n",i,nrow(missing),j$case,j$estimator,j$arm,code))
  data.frame(j,exit=code,directory=dir)
},mc.cores=3,mc.preschedule=FALSE)
statuses<-do.call(rbind,statuses)
# Merge serially: different arms of the same case must not race on its CSV.
for(i in which(statuses$exit==0)) {
  j<-statuses[i,];name<-paste0(j$case,"__",j$estimator,".csv")
  p<-file.path(j$directory,name)
  if(!file.exists(p)) next
  extra<-read.csv(p,stringsAsFactors=FALSE);extra<-extra[extra$arm==j$arm,]
  dest<-file.path(out,name)
  old<-if(file.exists(dest)) read.csv(dest,stringsAsFactors=FALSE) else extra[FALSE,]
  extra<-extra[!extra$arm %in% old$arm,]
  if(nrow(extra)) write.csv(rbind(old,extra),dest,row.names=FALSE)
}
write.csv(statuses,file.path(out,"retry_jobs.csv"),row.names=FALSE)
