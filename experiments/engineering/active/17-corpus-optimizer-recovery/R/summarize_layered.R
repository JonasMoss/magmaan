#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/summarize_layered.R RESULTS/layered\nSummarizes the current-versus-layered start comparison into summary.csv, paired.csv, changes.csv.\n');quit(save='no')}
dest<-normalizePath(args[1])
files<-list.files(dest,pattern='__(ML|GLS)\\.csv$',full.names=TRUE)
z<-do.call(rbind,lapply(files,read.csv,stringsAsFactors=FALSE))
jobs<-read.csv(file.path(dest,'jobs.csv'),stringsAsFactors=FALSE)
tol<-function(t)1e-6*(1+abs(t))
key<-paste(z$case,z$estimator)
best<-tapply(ifelse(z$accepted & is.finite(z$f),z$f,Inf),key,min)
z$best_now<-pmin(z$target,best[key])
z$good<-z$accepted & is.finite(z$f) & z$f<=z$target+tol(z$target)
z$good_now<-z$accepted & is.finite(z$f) & z$f<=z$best_now+tol(z$best_now)
s<-aggregate(cbind(attempts=1,accepted,good,good_now)~estimator+arm+start,z,sum)
s$start_seconds_median<-aggregate(start_seconds~estimator+arm+start,z,median)$start_seconds
s$start_seconds_max<-aggregate(start_seconds~estimator+arm+start,z,max)$start_seconds
w<-reshape(z[,c('case','estimator','arm','start','good_now','start_f','f')],idvar=c('case','estimator','arm'),timevar='start',direction='wide')
w$rescued<-w$good_now.layered & !w$good_now.current
w$lost<-!w$good_now.layered & w$good_now.current
w$start_lower<-w$start_f.layered<w$start_f.current
p<-aggregate(cbind(pairs=1,rescued,lost,both=good_now.layered & good_now.current)~estimator+arm,w,sum)
p$layered_start_lower<-aggregate(start_lower~estimator+arm,transform(w,start_lower=start_lower %in% TRUE),sum)$start_lower
write.csv(s,file.path(dest,'summary.csv'),row.names=FALSE)
write.csv(p,file.path(dest,'paired.csv'),row.names=FALSE)
ch<-w[w$rescued | w$lost,];write.csv(ch,file.path(dest,'changes.csv'),row.names=FALSE)
n<-z[z$start=='layered' & !is.na(z$notes) & nzchar(z$notes) & z$arm=='port_default',c('case','estimator','notes')]
write.csv(n,file.path(dest,'notes.csv'),row.names=FALSE)
print(s,row.names=FALSE);print(p,row.names=FALSE)
cat(sprintf('\njobs %d, nonzero exits %d, pairs with results %d\n',nrow(jobs),sum(jobs$exit!=0),length(unique(key))))
print(ch[,c('case','estimator','arm','good_now.current','good_now.layered','f.current','f.layered')],row.names=FALSE)
