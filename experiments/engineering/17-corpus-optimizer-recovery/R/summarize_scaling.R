#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/summarize_scaling.R RESULTS
Summarizes RESULTS/scaling (native units) and, when present, RESULTS/scaling-rescaled, with arms rerun in
the matching -vN folders replacing the originals (later N last). A fit is good when
it is accepted and reaches the best objective of its pair: native pairs use the frozen best and every
accepted fit; rescaled pairs use every accepted magmaan and lavaan fit on the rescaled data, and count only
when the model is unit-invariant (RESULTS/invariance.csv from R/check_invariance.R, else when that best
equals the native best), against the native best. Writes scaling_summary.csv,
scaling_paired.csv, scaling_changes.csv and rescaled_invariance.csv into RESULTS.\n');quit(save='no')}
out<-normalizePath(args[1])
rd<-function(dir){f<-list.files(file.path(out,dir),pattern='__(ML|GLS)\\.csv$',full.names=TRUE)
  if(!length(f))return(NULL);do.call(rbind,lapply(f,read.csv,stringsAsFactors=FALSE))}
# RESULTS/scaling[-rescaled]-vN hold reruns of single arms on later builds;
# their rows replace the same arms of the original runs, later N last.
merge_arms<-function(base,update){if(is.null(update))return(base)
  k<-function(x)paste(x$case,x$estimator,x$start,x$optimizer,x$scaling)
  rbind(base[!(k(base) %in% k(update)),],update)}
reruns<-function(prefix){d<-list.files(out,pattern=paste0('^',prefix,'-v[0-9]+$'));d[order(as.integer(sub('.*-v','',d)))]}
merged<-function(prefix){z<-rd(prefix);for(d in reruns(prefix))z<-merge_arms(z,rd(d));z}
z<-rbind(merged('scaling'),merged('scaling-rescaled'))
tol<-function(t)1e-6*(1+abs(t))
z$key<-paste(z$case,z$estimator,z$mode)
ok<-z$accepted & is.finite(z$f)
best<-tapply(ifelse(ok,z$f,Inf),z$key,min)
z$best<-ifelse(z$mode=='native',pmin(z$target,best[z$key]),best[z$key])
z$good<-ok & z$f<=z$best+tol(z$best)
# Unit-invariant rescaled pairs keep the native optimum (R/check_invariance.R
# carries it to the new units); their target is the native best. Without that
# file, a pair counts as invariant when some rescaled fit reaches the native best.
inv<-unique(z[z$mode=='rescaled',c('case','estimator','key','best','target')])
ifile<-file.path(out,'invariance.csv')
if(file.exists(ifile)) {
  ic<-read.csv(ifile,stringsAsFactors=FALSE)
  inv$invariant<-ic$invariant[match(paste(inv$case,inv$estimator),paste(ic$case,ic$estimator))] %in% TRUE
  r<-z$mode=='rescaled' & z$key %in% inv$key[inv$invariant]
  z$best[r]<-pmin(z$best[r],z$target[r])
  inv$best<-z$best[match(inv$key,z$key)]
} else inv$invariant<-is.finite(inv$best) & abs(inv$best-inv$target)<=1e-5*(1+abs(inv$target))
write.csv(inv,file.path(out,'rescaled_invariance.csv'),row.names=FALSE)
z$good<-z$accepted & is.finite(z$f) & z$f<=z$best+tol(z$best)
z<-z[z$mode=='native' | z$key %in% inv$key[inv$invariant],]
z$false_success<-z$accepted & !z$good;z$failed<-!z$accepted
s<-aggregate(cbind(pairs=1,good,false_success,failed)~mode+estimator+engine+start+optimizer+scaling,z,sum)
m<-aggregate(cbind(f_evals,seconds)~mode+estimator+engine+start+optimizer+scaling,z,function(x)median(x,na.rm=TRUE),na.action=na.pass)
s<-merge(s,m,all.x=TRUE);names(s)[names(s)=='f_evals']<-'median_f_evals';names(s)[names(s)=='seconds']<-'median_seconds'
s<-s[order(s$mode,s$estimator,s$engine,s$start,s$optimizer,match(s$scaling,c('none','sample_units','information','lavaan'))),]
write.csv(s,file.path(out,'scaling_summary.csv'),row.names=FALSE)
g<-z[z$engine=='magmaan',]
w<-reshape(g[,c('case','estimator','mode','start','optimizer','scaling','good')],idvar=c('case','estimator','mode','start','optimizer'),
  timevar='scaling',direction='wide')
names(w)<-sub('^good\\.','',names(w))
for(k in c('none','sample_units','information'))w[[k]]<-w[[k]] %in% TRUE
pairs<-function(a,b){w$gained<-w[[a]] & !w[[b]];w$lost<-!w[[a]] & w[[b]]
  aggregate(cbind(gained,lost)~mode+estimator+start+optimizer,w,sum)}
p<-rbind(cbind(comparison='sample_units vs none',pairs('sample_units','none')),
         cbind(comparison='information vs none',pairs('information','none')),
         cbind(comparison='information vs sample_units',pairs('information','sample_units')))
write.csv(p,file.path(out,'scaling_paired.csv'),row.names=FALSE)
ch<-w[!(w$none==w$sample_units & w$none==w$information),]
write.csv(ch[order(ch$mode,ch$estimator,ch$case),],file.path(out,'scaling_changes.csv'),row.names=FALSE)
print(s[,c('mode','estimator','engine','start','optimizer','scaling','pairs','good','false_success','failed','median_f_evals')],row.names=FALSE)
print(p,row.names=FALSE)
cat(sprintf('\nrescaled pairs: %d, unit-invariant: %d\n',nrow(inv),sum(inv$invariant)))
