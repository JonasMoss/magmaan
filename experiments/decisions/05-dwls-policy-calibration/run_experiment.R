#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
usage <- 'Usage: Rscript run_experiment.R MODE [--run-id ID] [--workers W]
  --preflight  amended theta thresholds-to-metric availability gate
  --smoke      2 replicates per cell; development checks only
  --pilot      20 replicates per cell; timing and failure diagnostics
  --production 2000 null/coverage, 1000 power replicates; separate compute approval required
  --explore    64 global cells, production draws and counts; save full FMG family
  --confirm    global and nested cells/counts, fresh seed base 817160001
  --family global|nested|all (default all)
  --reps R     override replicates per cell for local verification
  --workers W  1..4, one math thread each (default 1)
  --run-id ID  fresh immutable output directory (default mode name)
  --cell IDs   comma-separated cell IDs; run selected cells only and save its raw rows (Modal fan-out; see modal/)
  --out-dir D  write to D instead of results/dwls-policy/<run-id>
  --lane dwls-policy|mixed|exact-first-stage (default dwls-policy); mixed has smoke/pilot/production
  --help       show help
No automatic production launch. Frozen summaries exclude raw per-fit rows.'
if ('--help' %in% args) { cat(usage,'\n'); quit(save='no') }
opt <- function(key,default) {
  at <- match(key,args); if(is.na(at)) return(default)
  if(at==length(args) || startsWith(args[at+1],'--')) stop('Missing value for ',key)
  args[at+1]
}
if(any(startsWith(args,'--') & !args %in% c('--help','--preflight','--smoke','--pilot','--production','--explore','--confirm','--reps','--run-id','--workers','--cell','--out-dir','--family','--lane'))) stop('Unknown option')
modes <- intersect(args,c('--preflight','--smoke','--pilot','--production','--explore','--confirm'))
if(length(modes)!=1) stop(usage)
mode <- substring(modes,3)
workers <- as.integer(opt('--workers','1'))
if(is.na(workers) || workers<1 || workers>4) stop('workers must be 1..4')
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(script)
source(file.path(here,'..','..','_support','R','helpers.R'))
set_single_threaded_math()
source(file.path(here,'R','compute.R'))
source(file.path(here,'R','summarize.R'))
lane <- opt('--lane','dwls-policy')
if(!lane %in% c('dwls-policy','mixed','exact-first-stage')) stop('Unknown lane')
if(lane=='exact-first-stage') {
  source(file.path(here,'R','exact_first_stage.R'))
  exact_run(args,mode,workers,here,opt)
  quit(save='no')
}
if(lane=='mixed') {
  source(file.path(here,'R','mixed.R'))
  mixed_run(args,mode,workers,here,opt)
  quit(save='no')
}
run_id <- opt('--run-id',mode)
if(!grepl('^[a-zA-Z0-9_-]+$',run_id)) stop('Invalid run ID')
out <- opt('--out-dir',file.path(here,'results','dwls-policy',run_id))
only_cells <- as.integer(strsplit(opt('--cell',''),',',fixed=TRUE)[[1]])
reps <- as.integer(opt('--reps',NA))
if(!is.na(reps) && reps<1) stop('reps must be positive')
if(dir.exists(out)) stop('Run exists; choose a fresh --run-id')
dir.create(out,recursive=TRUE)
seed_base <- c(preflight=817130001L,smoke=817130001L,pilot=817140001L,production=817150001L,explore=817150001L,confirm=817160001L)[[mode]]
binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
files <- c(script,list.files(file.path(here,'R'),full.names=TRUE),file.path(here,'criteria','dwls_policy.md'),binary)
write_metadata(file.path(out,'metadata.csv'),list(mode=mode,workers=workers,
  seed_base=seed_base,family=opt('--family','all'),git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=git_dirty(),
  source_hashes=paste(tools::md5sum(files),collapse=','),hash_files=paste(files,collapse=','),
  magmaanlab_path=find.package('magmaanlab'),native_md5=paste(tools::md5sum(binary),collapse=','),
  population_n_per_group=100000L,population_seed_base=817120001L,batch_size=20L,reps_override=reps,pols_gamma=4,spectrum_truncate_negative=TRUE),
  packages=c('magmaanlab','lavaan'))
cells <- dwls_mode_cells(mode,opt('--family','all'))
if(length(only_cells) && any(!only_cells %in% cells$cell_id)) stop('Unknown cell ID for mode')
write_csv(cells,file.path(out,'cells.csv'))
if(mode=='preflight') {
  x <- dwls_preflight(); write_csv(x,file.path(out,'availability.csv')); print(x)
  if(any(!x$available)) stop('Required policy component unavailable')
} else {
  population <- dwls_population(cells,out)
  jobs <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i)
    data.frame(cell_id=cells$cell_id[i],replicate=seq_len(if(!is.na(reps)) reps else if(mode %in% c('production','explore','confirm')) cells$production_reps[i] else if(mode=='smoke') 2L else 20L))))
  # A single-cell run reproduces that cell's slice of the full run: seeds depend
  # only on the cell and replicate, and population targets are recomputed in full.
  if(length(only_cells)) jobs <- jobs[jobs$cell_id %in% only_cells,,drop=FALSE]
  start <- proc.time()[['elapsed']]; rows <- list()
  for(first in seq(1L,nrow(jobs),by=20L)) {
    indices <- first:min(first+19L,nrow(jobs))
    batch <- parallel::mclapply(indices,function(j) {
      set_single_threaded_math()
      dwls_replicate(cells[match(jobs$cell_id[j],cells$cell_id),],jobs$replicate[j],seed_base,population)
    },mc.cores=workers,mc.preschedule=TRUE)
    if(any(vapply(batch,inherits,logical(1),'try-error'))) stop('Worker process failed; retained earlier batches')
    rows <- c(rows,batch)
    raw <- do.call(rbind,rows)
    saveRDS(raw,file.path(out,'raw.rds'))
    elapsed <- proc.time()[['elapsed']]-start
    write_csv(data.frame(completed=max(indices),total=nrow(jobs),elapsed_seconds=elapsed,
      eta_seconds=elapsed*(nrow(jobs)-max(indices))/max(indices)),file.path(out,'progress.csv'))
    cat('Completed ',max(indices),'/',nrow(jobs),'; elapsed ',round(elapsed,1),'s\n',sep=''); flush.console()
  }
  if(any(is.finite(raw$policy_gap) & raw$policy_gap>1e-7)) stop('Policy equivalence gate failed')
  if(length(only_cells)) {
    cat('Cells ',paste(only_cells,collapse=','),' raw rows saved: ',out,'\n',sep=''); quit(save='no')
  }
  dwls_summarize(raw,cells,out)
  if(any(is.finite(raw$policy_gap) & raw$policy_gap>1e-7)) stop('Policy equivalence gate failed')
  # Nonconvergence is evidence, not a new study-specific acceptance tolerance.
  bad <- grepl('Policy unavailable|Policy covariance unavailable',raw$error) &
    !grepl('not_converged|numeric_failure|boundary',raw$error)
  if(any(bad)) stop('Required policy component structurally unavailable; see failures.csv')
}
cat('Results: ',out,'\n',sep='')
