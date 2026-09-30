#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
filearg <- grep('^--file=',commandArgs(),value=TRUE)
root <- dirname(normalizePath(sub('^--file=','',filearg[1])))
source(file.path(root,'R/methods.R'))
source(file.path(root,'R/validation.R'))
source(file.path(root,'R/summarize.R'))
option <- function(key,default) {
  at<-match(key,args);if(is.na(at)) default else {
    if(at==length(args)) stop('Missing value for ',key)
    args[at+1L]
  }
}
if(!length(args)||'--help'%in%args) {
  cat('Robust parameter intervals: --plan | --validate | --smoke | --main\n',
      'Options: --reps N --seed-base N --workers N --cells ALL|normal:100,t10:300 --output DIRECTORY\n',
      'Smoke: 2/cell; main: 500/cell. Pilot: --main --reps 50.\n',
      'An unchanged main run resumes/extends completed replicates.\n',sep='')
  quit()
}
unknown<-setdiff(args[grepl('^--',args)],c('--plan','--validate','--smoke','--main',
                                      '--reps','--seed-base','--workers','--cells','--output'))
if(length(unknown)) stop('Unknown options: ',paste(unknown,collapse=', '))
cells<-expand.grid(n=c(100L,300L),distribution=distributions,stringsAsFactors=FALSE)
selection<-option('--cells','ALL')
if(selection!='ALL') {
  requested<-strsplit(selection,',',fixed=TRUE)[[1]]
  ids<-paste(cells$distribution,cells$n,sep=':')
  if(any(!requested%in%ids)) stop('Unknown cells: ',selection)
  cells<-cells[ids%in%requested,]
}
mode<-intersect(args,c('--plan','--validate','--smoke','--main'))
if(length(mode)!=1L) stop('Choose one mode')
reps<-as.integer(option('--reps',if(mode=='--smoke') 2L else 500L))
workers<-as.integer(option('--workers',4L))
seed_base<-as.integer(option('--seed-base',if(mode=='--smoke') 2026102751 else 2026092751))
stopifnot(reps>0,workers>=1,workers<=4,!is.na(seed_base))
if(mode=='--plan') {cells$replications<-reps;cells$seed_base<-seed_base;print(cells);quit()}
out<-option('--output',file.path(root,'results',substring(mode,3)))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
out<-normalizePath(out)
if(mode=='--validate') {validate_methods(out);quit()}
sources<-c(file.path(root,'run_experiment.R'),list.files(file.path(root,'R'),full.names=TRUE))
binary<-system.file('libs',paste0('magmaanlab',.Platform$dynlib.ext),package='magmaanlab')
fingerprint<-data.frame(path=c(basename(sources),'magmaanlab binary'),
                        md5=unname(tools::md5sum(c(sources,binary))))
fingerprint<-rbind(fingerprint,data.frame(path=c('seed_base','cells'),md5=c(as.character(seed_base),
  paste(paste(cells$distribution,cells$n,sep=':'),collapse=','))))
fingerfile<-file.path(out,'fingerprint.csv')
if(file.exists(fingerfile)) {
  if(!identical(read.csv(fingerfile,stringsAsFactors=FALSE),fingerprint))
    stop('Source/binary/seed changed: retain this run and choose a new output and confirmation seed')
} else {
  write.csv(fingerprint,fingerfile,row.names=FALSE)
  validate_methods(file.path(out,'checks'))
  writeLines(c(paste('Started',Sys.time()),paste('R',getRversion()),
    paste('magmaanlab',packageVersion('magmaanlab')),paste('lavaan',packageVersion('lavaan')),
    paste('git',system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE)),
    'Total likelihood H; uncentered score crossproduct meat; N divisor.',
    'Unit factor variances; ambient unbounded fit; nonadmissibility flagged.',
    'Expected primary score/LR; observed primary sandwich Wald.',
    'Candidate-specific nuisance refits; deterministic retry; no Bartlett.',
    'Root tolerance 1e-7; endpoint statistic 1e-4; constraint 1e-6;',
    'Wider grid audit extends half interval width each side; finite audit only.'),
    file.path(out,'metadata.txt'))
  metadata<-data.frame(key=c('command','source_commit','source_dirty','seed_base','R',
    'magmaanlab','lavaan','population','mapping','domain','optimizer','tolerances'),
    value=c(paste(commandArgs(),collapse=' '),
      system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE),
      paste(system2('git',c('-C',shQuote(root),'status','--porcelain'),stdout=TRUE),collapse='; '),
      seed_base,as.character(getRversion()),as.character(packageVersion('magmaanlab')),
      as.character(packageVersion('lavaan')),
      '8 indicators; loadings .8,.7,.9,.6 twice; factor variances 1; rho .4; residuals 1-loading^2',
      'scores=sum and rows; total H; project_scores(center=FALSE); score_sandwich; score_spectrum mean_scale; parameter_covariance(H,crossprod(scores)); Wald-O equals policy',
      'bounds=NULL; psd=FALSE; positive implied covariance; no clipping; primitive inadmissibility flagged',
      'nlopt-slsqp; max_iter 2000; ftol 1e-12; gtol 1e-8; unrestricted start; scaled-fabin retry max_iter 5000',
      'constraint 1e-6; root 1e-7; endpoint statistic 1e-4; rcond 1e-12; LR roundoff 1e-6'))
  write.csv(metadata,file.path(out,'metadata.csv'),row.names=FALSE)
  capture.output(sessionInfo(),file=file.path(out,'session.txt'))
}
dir.create(file.path(out,'replicates'),showWarnings=FALSE)
jobs<-merge(cells,data.frame(replicate=seq_len(reps)),sort=FALSE)
jobs<-jobs[order(jobs$replicate,jobs$distribution,jobs$n),]
jobs$key<-sprintf('%s-n%d-r%04d',jobs$distribution,jobs$n,jobs$replicate)
completed<-function(key) file.exists(file.path(out,'replicates',key,'done'))
pending<-jobs[!vapply(jobs$key,completed,logical(1)),]
cat(nrow(pending),'pending of',nrow(jobs),'datasets\n');flush.console()
run<-function(i) {
  j<-pending[i,];path<-file.path(out,'replicates',j$key)
  dir.create(path,showWarnings=FALSE)
  result<-run_replicate(j$n,j$distribution,j$replicate,seed_base)
  for(name in names(result)) write.csv(result[[name]],file.path(path,paste0(name,'.csv')),row.names=FALSE)
  writeLines('complete',file.path(path,'done'))
  j$key
}
start<-proc.time()[['elapsed']]
if(nrow(pending)) for(first in seq(1L,nrow(pending),by=workers)) {
  indices<-seq.int(first,min(first+workers-1L,nrow(pending)))
  done<-parallel::mclapply(indices,run,mc.cores=workers,mc.preschedule=FALSE)
  if(any(vapply(done,inherits,logical(1),'try-error'))) stop('Worker failed; completed replicates retained')
  count<-max(indices);elapsed<-proc.time()[['elapsed']]-start
  cat(sprintf('%d/%d new datasets; %.1f min elapsed; %.1f min remaining\n',
              count,nrow(pending),elapsed/60,elapsed/count*(nrow(pending)-count)/60));flush.console()
}
# Concatenate candidate CSVs without holding the entire trace in memory.
candidate_file<-file.path(out,'candidates.csv')
if(file.exists(candidate_file)) unlink(candidate_file)
header<-TRUE
for(key in jobs$key) {
  lines<-readLines(file.path(out,'replicates',key,'candidates.csv'))
  if(length(lines)>1L) {
    cat(paste(if(header) lines else lines[-1],collapse='\n'),'\n',
        file=candidate_file,append=TRUE,sep='')
    header<-FALSE
  }
}
file.copy(file.path(out,'checks','validation.csv'),file.path(out,'validation.csv'),overwrite=TRUE)
collect<-function(name) do.call(rbind,lapply(jobs$key,function(key)
  read.csv(file.path(out,'replicates',key,paste0(name,'.csv')),stringsAsFactors=FALSE)))
intervals<-collect('intervals');timing<-collect('timing')
write.csv(intervals,file.path(out,'intervals.csv'),row.names=FALSE)
write.csv(timing,file.path(out,'timing.csv'),row.names=FALSE)
summarize_intervals(intervals,timing,out)
cat('Complete:',out,'\n')
