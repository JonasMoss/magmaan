#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
script <- sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1])
root <- dirname(normalizePath(script))
help <- function() cat(paste(
  'Normal parameter intervals: Wald, score, profile LR and bootstrap Bartlett.',
  'Usage: Rscript run_experiment.R MODE [options]',
  'Modes: --help, --plan, --validate, --smoke, --main, --bartlett-pilot',
  '--reps INTEGER         Datasets/cell (main 200, smoke 2, Bartlett 50).',
  '--seed-base INTEGER    Main/Bartlett 2026092750; smoke 2026102750.',
  '--bootstrap-reps INT   Bartlett B (default 399).',
  '--stream INTEGER       Bootstrap stream (default 0; data unchanged).',
  '--workers INTEGER      Dataset workers (default 1, maximum 4).',
  '--output NAME          Run directory under results/ (default is mode).',
  'Resume: rerun the identical command with the same source/package versions.',
  'Increase --reps to extend a run; completed per-dataset CSVs are reused.',
  'B=1999 and independent-stream checks are separate Bartlett runs on --reps 10.',
  'All expensive work is in this runner; rendering only reads results.',sep='\n'),'\n')
if(!length(args) || identical(args,'--help')) { help(); quit(status=0L) }
modes <- intersect(args,c('--plan','--validate','--smoke','--main','--bartlett-pilot'))
if(length(modes)!=1L) stop('Choose exactly one mode; see --help')
mode <- sub('^--','',modes)
opt <- list(reps=if(mode=='smoke') 2L else if(mode=='bartlett-pilot') 50L else 200L,
  seed_base=if(mode=='smoke') 2026102750L else 2026092750L,
  bootstrap_reps=399L,stream=0L,workers=1L,output=mode)
a <- args[args!=modes]
while(length(a)) {
  key <- sub('^--','',a[1]); key <- gsub('-','_',key)
  if(!key %in% names(opt) || length(a)<2L) stop('Unknown/incomplete option: ',a[1])
  opt[[key]] <- if(key=='output') a[2] else suppressWarnings(as.integer(a[2]))
  a <- a[-c(1,2)]
}
stopifnot(opt$reps>0L,opt$bootstrap_reps>1L,opt$workers>=1L,opt$workers<=4L,
          opt$seed_base>0L,opt$stream>=0L,opt$stream<=9L)
if(!grepl('^[A-Za-z0-9_-]+$',opt$output)) stop('--output must be a simple run name')
if(as.double(opt$seed_base)+1000*1000+opt$reps>.Machine$integer.max) stop('seed overflow')
if(mode=='plan') {
  print(data.frame(n=c(100,300,1000),replications=200,targets=2))
  cat('Bartlett: 50 N=100 loading datasets, B=399; first 10 at B=1999 and independent stream.\n')
  quit(status=0L)
}
suppressPackageStartupMessages(library(magmaanlab))
source(file.path(root,'R','engine.R'))
source(file.path(root,'R','validate.R'))
out <- file.path(root,'results',opt$output)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(mode=='validate') { validate_normal(out); quit(status=0L) }
# Fail before a simulation if a required primitive/normalization is broken.
validate_normal(file.path(out,'checks'))
run_mode <- if(mode=='bartlett-pilot') 'bartlett' else 'ordinary'
ns <- if(run_mode=='bartlett') 100L else c(100L,300L,1000L)
jobs <- expand.grid(n=ns,replicate=seq_len(opt$reps))
# Resumption identity excludes reps/workers/output; every statistical choice is fixed.
hashes <- tools::md5sum(c(script,file.path(root,'R',c('engine.R','validate.R'))))
fingerprint <- paste(c(hashes,as.character(packageVersion('magmaanlab')),run_mode,
  opt$seed_base,opt$bootstrap_reps,opt$stream),collapse='|')
fingerprint_path <- file.path(out,'fingerprint.txt')
if(file.exists(fingerprint_path) && readLines(fingerprint_path,warn=FALSE)!=fingerprint)
  stop('Run directory belongs to different code/options; choose a new --output')
writeLines(fingerprint,fingerprint_path)
meta <- c(command=paste(commandArgs(),collapse=' '),timestamp=as.character(Sys.time()),
  source_commit=system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE),
  source_dirty=as.character(length(system2('git',c('-C',shQuote(root),'status','--porcelain'),stdout=TRUE))>0),
  R=R.version.string,magmaanlab=as.character(packageVersion('magmaanlab')),
  lavaan=as.character(packageVersion('lavaan')),mode=run_mode,seed_base=opt$seed_base,
  bootstrap_reps=opt$bootstrap_reps,stream=opt$stream,replications=opt$reps,workers=opt$workers,
  optimizer='nlopt-slsqp',bounds='none',domain='ambient; flag covariance inadmissibility',
  start='layered unrestricted; unrestricted theta for constrained; scaled-fabin fallback',
  ftol=1e-12,gtol=1e-8,root_tol=1e-7,statistic_tol=1e-4,constraint_tol=1e-6,
  methods='normal expected/observed Wald; restricted expected score; profile LR; optional null-fitted bootstrap mean correction',
  fingerprint=fingerprint)
write.csv(data.frame(key=names(meta),value=unname(meta)),file.path(out,'metadata.csv'),row.names=FALSE)
jobdir <- function(n,r) file.path(out,sprintf('n%d_rep%04d',n,r))
finished <- function(n,r) file.exists(file.path(jobdir(n,r),'complete'))
todo <- jobs[!mapply(finished,jobs$n,jobs$replicate),,drop=FALSE]
run_job <- function(j) {
  n <- todo$n[j]; r <- todo$replicate[j]; d <- jobdir(n,r)
  ans <- run_dataset(n,r,opt$seed_base,run_mode,opt$bootstrap_reps,opt$stream)
  dir.create(d,showWarnings=FALSE)
  for(name in names(ans)) if(nrow(ans[[name]]))
    write.csv(ans[[name]],file.path(d,paste0(name,'.csv')),row.names=FALSE)
  writeLines('complete',file.path(d,'complete'))
  list(n=n,replicate=r,seconds=ans$timing$total_seconds,
       failures=sum(!ans$intervals$valid))
}
t0 <- proc.time()[['elapsed']]; done <- 0L
if(nrow(todo)) for(start in seq(1L,nrow(todo),by=opt$workers)) {
  idx <- start:min(start+opt$workers-1L,nrow(todo))
  ans <- if(opt$workers==1L) lapply(idx,run_job) else
    parallel::mclapply(idx,run_job,mc.cores=opt$workers,mc.preschedule=TRUE,mc.set.seed=FALSE)
  if(any(vapply(ans,inherits,logical(1),'try-error'))) stop('Worker failed; retain partial outputs and resume')
  done <- done+length(idx); elapsed <- proc.time()[['elapsed']]-t0
  eta <- elapsed/done*(nrow(todo)-done)
  progress <- data.frame(completed=done,total=nrow(todo),elapsed_seconds=elapsed,eta_seconds=eta)
  write.csv(progress,file.path(out,'progress.csv'),row.names=FALSE)
  cat(sprintf('Completed %d/%d new datasets; %.1fs elapsed; ETA %.1fs; interval failures in batch %d\n',
              done,nrow(todo),elapsed,eta,sum(vapply(ans,`[[`,numeric(1),'failures'))))
  flush.console()
}
collect <- function(name) {
  paths <- mapply(function(n,r) file.path(jobdir(n,r),paste0(name,'.csv')),jobs$n,jobs$replicate)
  paths <- paths[file.exists(paths)]
  if(!length(paths)) return(data.frame())
  do.call(rbind,lapply(paths,read.csv,stringsAsFactors=FALSE))
}
for(name in c('intervals','candidates','bootstrap','timing')) {
  x <- collect(name)
  if(nrow(x)) write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
}
x <- collect('intervals')
groups <- split(x,interaction(x$n,x$target,x$method,drop=TRUE))
summary <- do.call(rbind,lapply(groups,function(z) {
  v <- z[z$valid,,drop=FALSE]; coverage <- if(nrow(v)) mean(v$covered) else NA_real_
  data.frame(n=z$n[1],target=z$target[1],method=z$method[1],attempted=nrow(z),
    valid=nrow(v),failures=sum(!z$valid),coverage=coverage,
    coverage_mcse=if(nrow(v)) sqrt(coverage*(1-coverage)/nrow(v)) else NA_real_,
    successful_covering=sum(z$valid & z$covered,na.rm=TRUE)/nrow(z),
    lower_miss=if(nrow(v)) mean(v$lower_miss) else NA_real_,
    upper_miss=if(nrow(v)) mean(v$upper_miss) else NA_real_,
    mean_width=if(nrow(v)) mean(v$width) else NA_real_,
    median_width=if(nrow(v)) median(v$width) else NA_real_,
    mean_seconds=mean(z$seconds),mean_refits=mean(z$refits),
    unrestricted_inadmissible=sum(!z$unrestricted_admissible,na.rm=TRUE),
    candidate_inadmissible=sum(z$candidate_inadmissible),
    truth_disagreements=sum(!z$truth_agreement,na.rm=TRUE))
}))
write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE)
write.csv(x[!x$valid,,drop=FALSE],file.path(out,'failures.csv'),row.names=FALSE)
pairs <- list()
for(n in unique(x$n)) for(target in unique(x$target)) {
  zz <- x[x$n==n & x$target==target,]
  for(method in setdiff(unique(zz$method),'wald_expected')) {
    p <- merge(zz[zz$method=='wald_expected',],zz[zz$method==method,],by='replicate')
    p <- p[p$valid.x & p$valid.y,,drop=FALSE]
    diff <- as.numeric(p$covered.y)-as.numeric(p$covered.x)
    pairs[[length(pairs)+1L]] <- data.frame(n=n,target=target,method=method,
      comparator='wald_expected',common_valid=nrow(p),
      coverage_difference=if(nrow(p)) mean(diff) else NA_real_,
      difference_mcse=if(nrow(p)>1) sd(diff)/sqrt(nrow(p)) else NA_real_,
      mean_width_difference=if(nrow(p)) mean(p$width.y-p$width.x) else NA_real_)
  }
}
write.csv(do.call(rbind,pairs),file.path(out,'paired.csv'),row.names=FALSE)
print(summary,row.names=FALSE,digits=3)
cat('Wrote results to',out,'\n')
