#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/summarize_scaling.R RESULTS
Summarizes RESULTS/scaling (native units), RESULTS/scaling-rescaled and the native lavaan runs in
RESULTS/lavaan, with arms rerun in matching -vN folders replacing the originals (later N last).
One judge for every engine: a fit is certified when the magmaan verdict (the Newton check) accepts its
endpoint, which for lavaan is evaluated at the lavaan estimate. Primary outcome: certified local minimum.
Secondary: certified at the best known objective of its pair (native: the frozen best and every certified
fit; rescaled: every certified fit, and the native best when the pair is unit-invariant per
RESULTS/invariance.csv). A certified fit above the best is another local minimum. Only standard problems
(RESULTS/problem_classes.csv) enter; rescaled pairs count only when unit-invariant. Writes
scaling_summary.csv, scaling_paired.csv, scaling_changes.csv and rescaled_invariance.csv.\n');quit(save='no')}
out<-normalizePath(args[1])
bind<-function(a,b){if(is.null(a))return(b);if(is.null(b))return(a);cols<-union(names(a),names(b))
  for(k in setdiff(cols,names(a)))a[[k]]<-NA;for(k in setdiff(cols,names(b)))b[[k]]<-NA;rbind(a[cols],b[cols])}
rd<-function(dir){f<-list.files(file.path(out,dir),pattern='__(ML|GLS)\\.csv$',full.names=TRUE)
  if(!length(f))return(NULL);Reduce(bind,lapply(f,read.csv,stringsAsFactors=FALSE))}
# RESULTS/scaling[-rescaled]-vN hold reruns of single arms on later builds;
# their rows replace the same arms of the original runs, later N last.
merge_arms<-function(base,update){if(is.null(update))return(base)
  k<-function(x)paste(x$case,x$estimator,x$start,x$optimizer,x$scaling)
  bind(base[!(k(base) %in% k(update)),],update)}
reruns<-function(prefix){d<-list.files(out,pattern=paste0('^',prefix,'-v[0-9]+$'));d[order(as.integer(sub('.*-v','',d)))]}
merged<-function(prefix){z<-rd(prefix);for(d in reruns(prefix))z<-merge_arms(z,rd(d));z}
z<-bind(merged('scaling'),merged('scaling-rescaled'))
# Native lavaan runs (R/test_lavaan_defaults.R), judged at the lavaan estimate.
l<-rd('lavaan')
if(!is.null(l)) {
  if(is.null(l$certified))stop('RESULTS/lavaan predates the shared judge; rerun R/test_lavaan_defaults.R')
  z<-bind(z,data.frame(case=l$case,estimator=l$estimator,mode='native',engine='lavaan',start=sub('^lavaan_','',l$arm),
    optimizer='nlminb',scaling='lavaan',target=l$target,returned=l$returned,accepted=l$certified %in% TRUE,
    flag=l$converged %in% TRUE,f=l$f,newton_status=l$newton_status,iterations=l$iterations,seconds=l$seconds,
    stringsAsFactors=FALSE))
}
cfile<-file.path(out,'problem_classes.csv')
if(file.exists(cfile)) {cl<-read.csv(cfile,stringsAsFactors=FALSE);z<-z[z$case %in% cl$case[cl$class=='standard'],]}
tol<-function(t)1e-6*(1+abs(t))
z$accepted<-z$accepted %in% TRUE
z$key<-paste(z$case,z$estimator,z$mode)
ok<-z$accepted & is.finite(z$f)
best<-tapply(ifelse(ok,z$f,Inf),z$key,min)
z$best<-ifelse(z$mode=='native',pmin(z$target,best[z$key]),best[z$key])
# Unit-invariant rescaled pairs keep the native optimum (R/check_invariance.R
# carries it to the new units); their best is at most the native best.
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
z<-z[z$mode=='native' | z$key %in% inv$key[inv$invariant],]
z$certified<-z$accepted
z$best_known<-z$certified & is.finite(z$f) & z$f<=z$best+tol(z$best)
z$other_minimum<-z$certified & !z$best_known
z$not_certified<-!z$certified
z$flag_only<-z$flag %in% TRUE & !z$certified
s<-aggregate(cbind(pairs=1,certified,best_known,other_minimum,not_certified,flag_only)~mode+estimator+engine+start+optimizer+scaling,z,sum)
m<-aggregate(cbind(f_evals,seconds)~mode+estimator+engine+start+optimizer+scaling,z,function(x)median(x,na.rm=TRUE),na.action=na.pass)
s<-merge(s,m,all.x=TRUE);names(s)[names(s)=='f_evals']<-'median_f_evals';names(s)[names(s)=='seconds']<-'median_seconds'
s$flag_only[s$engine!='lavaan']<-NA
s<-s[order(s$mode,s$estimator,s$engine,s$start,s$optimizer,match(s$scaling,c('none','sample_units','information','lavaan'))),]
write.csv(s,file.path(out,'scaling_summary.csv'),row.names=FALSE)
g<-z[z$engine=='magmaan',]
paired<-function(metric){
  w<-reshape(g[,c('case','estimator','mode','start','optimizer','scaling',metric)],idvar=c('case','estimator','mode','start','optimizer'),
    timevar='scaling',direction='wide')
  names(w)<-sub(paste0('^',metric,'\\.'),'',names(w))
  for(k in c('none','sample_units','information'))w[[k]]<-w[[k]] %in% TRUE
  one<-function(a,b){w$gained<-w[[a]] & !w[[b]];w$lost<-!w[[a]] & w[[b]]
    cbind(metric=metric,comparison=paste(a,'vs',b),aggregate(cbind(gained,lost)~mode+estimator+start+optimizer,w,sum))}
  list(table=rbind(one('sample_units','none'),one('information','none'),one('information','sample_units')),wide=w)
}
pc<-paired('certified');pb<-paired('best_known')
write.csv(rbind(pc$table,pb$table),file.path(out,'scaling_paired.csv'),row.names=FALSE)
w<-pb$wide;ch<-w[!(w$none==w$sample_units & w$none==w$information),]
write.csv(ch[order(ch$mode,ch$estimator,ch$case),],file.path(out,'scaling_changes.csv'),row.names=FALSE)
print(s[,c('mode','estimator','engine','start','optimizer','scaling','pairs','certified','best_known','other_minimum','flag_only','median_f_evals')],row.names=FALSE)
cat(sprintf('\nrescaled pairs: %d, unit-invariant: %d\n',nrow(inv),sum(inv$invariant)))
