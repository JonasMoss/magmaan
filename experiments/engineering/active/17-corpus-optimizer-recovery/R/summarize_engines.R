#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/summarize_engines.R RESULTS\nCompares lavaan defaults, lavaan from the layered start and magmaan (PORT, L-BFGS; current and layered\nstarts) on one target per pair: the lowest objective among successful endpoints and the frozen best.\nWrites RESULTS/engines_summary.csv and RESULTS/engines_pairs.csv.\n');quit(save='no')}
out<-normalizePath(args[1])
rd<-function(dir) do.call(rbind,lapply(list.files(file.path(out,dir),pattern='__(ML|GLS)\\.csv$',full.names=TRUE),read.csv,stringsAsFactors=FALSE))
m<-rd('layered');l<-rd('lavaan')
labels<-c(lavaan_default='lavaan, defaults',lavaan_layered='lavaan, layered start',
  port_default.current='magmaan PORT, current start',port_default.layered='magmaan PORT, layered start',
  lbfgs_default.current='magmaan L-BFGS, current start',lbfgs_default.layered='magmaan L-BFGS, layered start')
z<-rbind(data.frame(case=m$case,estimator=m$estimator,policy=paste(m$arm,m$start,sep='.'),success=m$accepted,f=m$f,target=m$target,seconds=m$seconds),
         data.frame(case=l$case,estimator=l$estimator,policy=l$arm,success=l$converged,f=l$f,target=l$target,seconds=l$seconds))
key<-paste(z$case,z$estimator)
best<-tapply(ifelse(z$success & is.finite(z$f),z$f,Inf),key,min)
z$best<-pmin(z$target,best[key])
tol<-1e-6*(1+abs(z$best))
z$good<-z$success & is.finite(z$f) & z$f<=z$best+tol
z$false_success<-z$success & !z$good
z$failed<-!z$success
s<-aggregate(cbind(pairs=1,good,false_success,failed)~estimator+policy,z,sum)
s$median_seconds<-aggregate(seconds~estimator+policy,z,median,na.action=na.pass)$seconds
s$label<-labels[s$policy]
s<-s[order(s$estimator,match(s$policy,names(labels))),]
write.csv(s,file.path(out,'engines_summary.csv'),row.names=FALSE)
w<-reshape(z[,c('case','estimator','policy','good')],idvar=c('case','estimator'),timevar='policy',direction='wide')
names(w)<-sub('^good\\.','',names(w))
write.csv(w,file.path(out,'engines_pairs.csv'),row.names=FALSE)
print(s[,c('estimator','label','good','false_success','failed','median_seconds')],row.names=FALSE)
for(est in c('ML','GLS')) {
  x<-w[w$estimator==est,]
  cat(sprintf('\n%s: lavaan defaults good, magmaan L-BFGS layered not: %d; reverse: %d\n',est,
    sum(x$lavaan_default & !x$lbfgs_default.layered,na.rm=TRUE),sum(!x$lavaan_default & x$lbfgs_default.layered,na.rm=TRUE)))
  cat(sprintf('%s: lavaan defaults good, magmaan PORT layered not: %d; reverse: %d\n',est,
    sum(x$lavaan_default & !x$port_default.layered,na.rm=TRUE),sum(!x$lavaan_default & x$port_default.layered,na.rm=TRUE)))
}
