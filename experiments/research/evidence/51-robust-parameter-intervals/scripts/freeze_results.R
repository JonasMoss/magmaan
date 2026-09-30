#!/usr/bin/env Rscript
# Keep a small reviewable evidence snapshot; raw intervals/candidates stay local.
args<-commandArgs(TRUE)
if(length(args)!=1L) stop('Usage: Rscript scripts/freeze_results.R results/confirmation')
root<-dirname(dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1]))))
run<-normalizePath(args[1]);out<-file.path(root,'results','frozen')
fingerprint<-read.csv(file.path(run,'fingerprint.csv'))
source_files<-c(file.path(root,'run_experiment.R'),list.files(file.path(root,'R'),full.names=TRUE))
stopifnot(identical(unname(tools::md5sum(source_files)),
  fingerprint$md5[match(basename(source_files),fingerprint$path)]))
binary<-system.file('libs',paste0('magmaanlab',.Platform$dynlib.ext),package='magmaanlab')
stopifnot(unname(tools::md5sum(binary))==fingerprint$md5[fingerprint$path=='magmaanlab binary'])
s<-read.csv(file.path(run,'summary.csv'))
x<-read.csv(file.path(run,'intervals.csv'))
stopifnot(nrow(s)==108L,all(s$attempts==500L),nrow(x)==54000L,
          !anyDuplicated(x[c('n','distribution','replicate','target','method')]),
          all(!is.na(x$error[!x$valid]) & nzchar(x$error[!x$valid])),
          all(x$truth_agreement[x$valid] | x$near_cutoff[x$valid]))
valid<-x[x$valid,]
stopifnot(all(valid$lower<valid$upper),all(valid$endpoint_gap[is.finite(valid$endpoint_gap)]<=1e-4))
# Inspect every saved candidate trace, including failed interval searches.
files<-list.files(file.path(run,'replicates'),pattern='^candidates.csv$',recursive=TRUE,full.names=TRUE)
max_constraint<-0;candidate_count<-0;converged<-0;candidate_failures<-0
stages<-list();failed<-x[!x$valid,]
for(f in files) {
  z<-read.csv(f)
  if(!nrow(z)) next
  relevant<-failed[failed$n==z$n[1] & failed$distribution==z$distribution[1] &
                    failed$replicate==z$replicate[1],]
  if(nrow(relevant)) for(i in seq_len(nrow(relevant))) {
    frow<-relevant[i,]
    trace<-z[z$method==frow$method & z$target==frow$target,]
    if(nrow(trace)) {
      last<-tail(trace,1)
      stages[[length(stages)+1L]]<-data.frame(n=frow$n,distribution=frow$distribution,
        target=frow$target,method=frow$method,phase=last$purpose,
        last_candidate_outside_correlation_domain=frow$target=='correlation' && abs(last$candidate)>1)
    }
  }
  candidate_count<-candidate_count+nrow(z);converged<-converged+sum(z$converged)
  candidate_failures<-candidate_failures+sum(!is.na(z$error) & nzchar(z$error))
  if(any(is.finite(z$constraint_residual)))
    max_constraint<-max(max_constraint,z$constraint_residual,na.rm=TRUE)
}
stopifnot(max_constraint<=1e-6)
# The generator owns the sole RNG step; reversing methods must change neither
# data nor interval results. Also reproduce an existing checkpoint exactly.
source(file.path(root,'R','methods.R'))
seed_base<-as.integer(fingerprint$md5[fingerprint$path=='seed_base'])
a<-run_replicate(100,'heterogeneous_gamma',1L,seed_base)$intervals
methods<-rev(methods)
b<-run_replicate(100,'heterogeneous_gamma',1L,seed_base)$intervals
saved<-x[x$n==100 & x$distribution=='heterogeneous_gamma' & x$replicate==1,]
canonical<-function(z) {
  # read.csv infers all-empty error columns as logical NA.
  z$error<-as.character(z$error);z$error[is.na(z$error)]<-''
  z<-z[order(z$target,z$method),setdiff(names(z),'seconds')]
  rownames(z)<-NULL;z
}
order_invariant<-isTRUE(all.equal(canonical(a),canonical(b),tolerance=1e-10))
checkpoint_reproduced<-isTRUE(all.equal(canonical(a),canonical(saved),tolerance=1e-10))
stopifnot(order_invariant,checkpoint_reproduced)
dir.create(out,showWarnings=FALSE)
for(f in c('summary.csv','paired.csv','common_valid.csv','failures.csv','timing_summary.csv',
           'fingerprint.csv','validation.csv','metadata.csv'))
  stopifnot(file.copy(file.path(run,f),file.path(out,f),overwrite=TRUE))
stopifnot(file.copy(file.path(run,'checks','api_audit.csv'),file.path(out,'api_audit.csv'),overwrite=TRUE))
metadata<-read.csv(file.path(out,'metadata.csv'),stringsAsFactors=FALSE)
metadata$key[metadata$key=='command']<-'initial_command'
metadata<-rbind(metadata,data.frame(key=c('completed_datasets','completed_replications_per_cell',
  'completion_command','frozen_at'),value=c(nrow(x)/18,500,
  paste('Rscript run_experiment.R --main --reps 500 --seed-base',seed_base,'--output results/confirmation'),
  as.character(Sys.time()))))
write.csv(metadata,file.path(out,'metadata.csv'),row.names=FALSE)
integrity<-data.frame(metric=c('datasets','interval_attempts','valid_intervals','candidate_evaluations',
  'converged_candidate_fits','candidate_evaluation_errors','maximum_constraint_residual',
  'maximum_valid_endpoint_statistic_gap','valid_truth_disagreements','valid_near_cutoff',
  'method_order_invariant','checkpoint_reproduced'),
  value=c(nrow(x)/18,nrow(x),nrow(valid),candidate_count,converged,candidate_failures,max_constraint,
    max(valid$endpoint_gap,na.rm=TRUE),sum(!valid$truth_agreement),sum(valid$near_cutoff),
    as.numeric(order_invariant),as.numeric(checkpoint_reproduced)))
write.csv(integrity,file.path(out,'integrity.csv'),row.names=FALSE)
if(length(stages)) {
  stages<-do.call(rbind,stages)
  stages<-aggregate(rep(1,nrow(stages)),stages,sum)
  names(stages)[ncol(stages)]<-'count'
  write.csv(stages,file.path(out,'failure_stages.csv'),row.names=FALSE)
}
print(integrity)
cat('Frozen evidence:',out,'\n')
