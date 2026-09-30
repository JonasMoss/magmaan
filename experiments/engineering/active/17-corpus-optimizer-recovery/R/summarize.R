#!/usr/bin/env Rscript
args<-commandArgs(TRUE); root<-if(length(args)) args[1] else "results/full"
paths<-list.files(root,pattern="__(ML|GLS)\\.csv$",full.names=TRUE)
stopifnot(length(paths)>0)
d<-do.call(rbind,lapply(paths,read.csv,stringsAsFactors=FALSE))
write.csv(d,file.path(root,"attempts.csv"),row.names=FALSE)
excluded<-d[d$arm=="preparation",]
write.csv(excluded,file.path(root,"exclusions.csv"),row.names=FALSE)
d<-d[d$arm!="preparation",]
if(file.exists(file.path(root,"input_contracts.csv"))) {
  contracts<-read.csv(file.path(root,"input_contracts.csv"))
  idx<-match(paste(d$case,d$estimator),paste(contracts$case,contracts$estimator))
  if(file.exists(file.path(root,"jobs.csv")) && anyNA(idx)) stop("Run validate_inputs.R on all completed cases first")
  bad<-!is.na(idx) & contracts$contract[idx]!="passed"
  if(any(bad)) stop("Optimizer results include a failed model contract; investigate before summarizing")
}
# Use accepted fits and converged references as targets; an infeasible or
# failed fit must not define the comparison target. This is not a global proof.
k<-paste(d$case,d$estimator)
best<-tapply(seq_len(nrow(d)),k,function(i) {
  x<-c(d$f[i][d$converged[i]],d$reference_f[i][d$reference_converged[i]])
  x<-x[is.finite(x)];if(length(x)) min(x) else NA_real_
})
d$best_observed<-as.numeric(best[k])
d$objective_gap<-d$f-d$best_observed
d$objective_match<-is.finite(d$objective_gap)&abs(d$objective_gap)<=1e-6*(1+abs(d$best_observed))
d$accepted_match<-d$converged & d$objective_match
write.csv(d,file.path(root,"comparison.csv"),row.names=FALSE)
summary<-do.call(rbind,lapply(split(d,paste(d$estimator,d$arm)),function(x) data.frame(
  estimator=x$estimator[1],arm=x$arm[1],attempted=nrow(x),returned=sum(x$returned),
  accepted=sum(x$converged),objective_match=sum(x$objective_match),
  accepted_match=sum(x$accepted_match),worse=sum(is.finite(x$objective_gap) & x$objective_gap>1e-6*(1+abs(x$best_observed))),
  no_target=sum(x$returned & !is.finite(x$objective_gap)),
  errors=sum(!x$returned),median_evaluations=median(x$f_evals,na.rm=TRUE),
  median_seconds=median(x$seconds,na.rm=TRUE))))
write.csv(summary,file.path(root,"summary.csv"),row.names=FALSE)
ordinary<-d[d$arm!="lbfgs_reference_start",]
base<-ordinary[ordinary$arm=="lbfgs_default",]
paired<-do.call(rbind,lapply(split(ordinary,ordinary$arm),function(x) {
  b<-base[match(paste(x$case,x$estimator),paste(base$case,base$estimator)),]
  data.frame(estimator=x$estimator,arm=x$arm,case=x$case,
    rescued=!b$accepted_match & x$accepted_match,
    lost=b$accepted_match & !x$accepted_match)
}))
write.csv(paired,file.path(root,"paired.csv"),row.names=FALSE)
print(summary,row.names=FALSE)
cat("Preparation exclusions:",nrow(excluded),"\n")
print(sort(table(excluded$message),decreasing=TRUE))
if(file.exists(file.path(root,"jobs.csv"))) {
  jobs<-read.csv(file.path(root,"jobs.csv"));print(table(jobs$exit))
}
