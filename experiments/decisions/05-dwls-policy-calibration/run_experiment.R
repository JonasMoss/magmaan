#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
usage <- 'Usage: Rscript run_experiment.R --preflight [--run-id ID] [--workers W]
  --preflight  reproduce the required theta threshold-nesting availability gate
  --workers W  1..4; preflight runs serially with one math thread
  --run-id ID  fresh immutable result directory (default preflight)
  --smoke      unavailable until task-17.1 API decision is resolved
  --pilot      unavailable until task-17.1 API decision is resolved
  --production unavailable; requires separate compute approval after pilot
  --help       show help
No calibration, timing pilot or production evidence has been generated.'
if ('--help' %in% args) { cat(usage,'\n'); quit(save='no') }
opt <- function(key,default) {
  at <- match(key,args); if(is.na(at)) return(default)
  if(at==length(args) || startsWith(args[at+1],'--')) stop('Missing value for ',key)
  args[at+1]
}
if(any(startsWith(args,'--') & !args %in% c('--help','--preflight','--smoke','--pilot','--production','--run-id','--workers'))) stop('Unknown option')
if(length(intersect(args,c('--preflight','--smoke','--pilot','--production')))!=1) stop(usage)
if(!'--preflight' %in% args) stop('task-17.1 is blocked: required policy threshold nesting is unavailable; see report.qmd')
workers <- as.integer(opt('--workers','1'))
if(is.na(workers) || workers<1 || workers>4) stop('workers must be 1..4')
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(script)
source(file.path(here,'..','..','_support','R','helpers.R'))
set_single_threaded_math()
source(file.path(here,'R','compute.R'))
run_id <- opt('--run-id','preflight')
if(!grepl('^[a-zA-Z0-9_-]+$',run_id)) stop('Invalid run ID')
out <- file.path(here,'results','dwls-policy',run_id)
if(dir.exists(out)) stop('Run exists; choose a fresh --run-id')
dir.create(out,recursive=TRUE)
binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
files <- c(script,file.path(here,'R','compute.R'),file.path(here,'criteria','dwls_policy.md'),binary)
write_metadata(file.path(out,'metadata.csv'),list(mode='preflight',workers=1L,
  requested_workers=workers,seed_base=817130001L,git_head=git_scalar(c('rev-parse','HEAD')),
  git_dirty=git_dirty(),source_hashes=paste(tools::md5sum(files),collapse=','),
  hash_files=paste(files,collapse=','),magmaanlab_path=find.package('magmaanlab'),
  native_md5=paste(tools::md5sum(binary),collapse=',')),packages=c('magmaanlab','lavaan'))
write_csv(dwls_cells(),file.path(out,'cells.csv'))
x <- dwls_preflight()
write_csv(x,file.path(out,'availability.csv'))
print(x)
cat('Results: ',out,'\n',sep='')
if(any(!x$available)) stop('Required policy component unavailable; task-17.1 needs a decision. No pilot or production run.')
