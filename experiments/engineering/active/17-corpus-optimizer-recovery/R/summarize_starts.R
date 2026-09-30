#!/usr/bin/env Rscript
args<-commandArgs(TRUE);out<-normalizePath(args[1])
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE)
# Freeze the earlier experiment's target: newly better fits are successes too.
tol<-1e-6*(1+abs(d$target))
d$at_least_reference<-d$accepted & is.finite(d$objective) & d$objective<=d$target+tol
d$better_than_reference<-d$accepted & is.finite(d$objective) & d$objective<d$target-tol
write.csv(d,file.path(out,'comparison.csv'),row.names=FALSE)
s<-aggregate(cbind(attempts=rep(1,nrow(d)),accepted=d$accepted,
                    at_least_reference=d$at_least_reference,better=d$better_than_reference),
             d[c('estimator','engine','start')],sum)
write.csv(s,file.path(out,'summary.csv'),row.names=FALSE);print(s,row.names=FALSE)
stopifnot(all(read.csv(file.path(out,'jobs.csv'))$exit==0))
