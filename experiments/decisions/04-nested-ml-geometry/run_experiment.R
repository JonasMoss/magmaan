#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
usage <- 'Usage: Rscript run_experiment.R --smoke|--pilot|--production [--lane nested-geometry|structured-mean] [--cells IDs] [--run-id ID] [--workers W]
  --smoke       2 replicates per cell, separate development seeds
  --pilot       20 replicates per cell; compute pricing only
  --production  2000 null / 1000 power per cell; requires registered compute approval
  --lane NAME   nested-geometry (default) or structured-mean (TASK-75 draft)
  --cells IDs   optional comma-separated stable cell IDs, default all
  --workers W   default 1; maximum 4, one math thread per worker
  --run-id ID   fresh immutable result directory (default mode)
  --help        show help
Raw batches stay local; summary CSVs and provenance are frozen. No defaults change.'
if ('--help' %in% args) { cat(usage, '\n'); quit(save='no') }
opt <- function(key, default) {
  at <- match(key, args); if (is.na(at)) return(default)
  if (at == length(args) || startsWith(args[at+1], '--')) stop('Missing value for ', key)
  args[at+1]
}
if (any(startsWith(args, '--') & !args %in% c('--smoke','--pilot','--production','--run-id','--workers','--lane','--cells'))) stop('Unknown option')
mode <- intersect(args, c('--smoke','--pilot','--production'))
if (length(mode) != 1) stop(usage)
mode <- substring(mode, 3)
workers <- as.integer(opt('--workers', '1'))
if (is.na(workers) || workers < 1 || workers > 4) stop('workers must be 1..4')
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(script)
source(file.path(here, '..','..','_support','R','helpers.R'))
set_single_threaded_math()
source(file.path(here,'R','compute.R')); source(file.path(here,'R','summaries.R'))
lane <- opt('--lane','nested-geometry')
if(!lane %in% c('nested-geometry','structured-mean')) stop('Unknown lane')
if(lane=='structured-mean') source(file.path(here,'R','structured_mean.R'))
run_id <- opt('--run-id', mode)
if (!grepl('^[a-zA-Z0-9_-]+$', run_id)) stop('Invalid run ID')
out <- file.path(here,'results',lane,run_id)
if (dir.exists(out)) stop('Run exists; choose a fresh --run-id')
dir.create(file.path(out,'raw'), recursive=TRUE)
cells <- if(lane=='structured-mean') structured_cells() else geometry_cells()
selected <- opt('--cells','all')
if(selected!='all') {
  ids <- as.integer(strsplit(selected,',',fixed=TRUE)[[1]])
  if(anyNA(ids)||any(!ids %in% cells$cell_id)||anyDuplicated(ids)) stop('Invalid cells')
  cells <- cells[cells$cell_id %in% ids,]
}
seed_base <- switch(mode, smoke=526100021L, pilot=626100031L, production=726100041L)
if(lane=='structured-mean') seed_base <- switch(mode,smoke=1326100021L,pilot=1526100031L,production=1726100041L)
binary <- list.files(file.path(find.package('magmaanlab'),'libs'), '\\.so$', full.names=TRUE)
hashes <- tools::md5sum(c(script, file.path(here,'R',c('compute.R','summaries.R')),
                         file.path(here,'criteria',if(lane=='structured-mean') 'structured_mean.md' else 'nested_geometry.md'),
                         if(lane=='structured-mean') file.path(here,'R','structured_mean.R'), binary))
write_metadata(file.path(out,'metadata.csv'), list(mode=mode, lane=lane, selected_cells=selected, seed_base=seed_base,
  workers=workers, cells=nrow(cells), git_head=git_scalar(c('rev-parse','HEAD')),
  git_dirty=git_dirty(), source_hashes=paste(hashes,collapse=','), hash_files=paste(names(hashes),collapse=','),
  magmaanlab_path=find.package('magmaanlab'), native_md5=paste(tools::md5sum(binary),collapse=',')),
  packages=c('magmaanlab','magmaan','lavaan'))
write_csv(cells,file.path(out,'cells.csv'))
write_csv(if(lane=='structured-mean') structured_population_checks() else geometry_population(),file.path(out,'population.csv'))
all <- list(); started <- proc.time()[['elapsed']]; done <- 0L
for (i in seq_len(nrow(cells))) {
  reps <- switch(mode, smoke=2L,pilot=20L,production=cells$production_reps[i])
  for (batch in split(seq_len(reps), ceiling(seq_len(reps)/20))) {
    values <- parallel::mclapply(batch,function(r) { draw <- if(lane=='structured-mean') structured_draw else geometry_draw; draw(cells[i,],r,seed_base+10000L*cells$cell_id[i]+r) },
      mc.cores=workers,mc.preschedule=TRUE)
    x <- do.call(rbind,values); all[[length(all)+1L]] <- x
    write_csv(x,file.path(out,'raw',sprintf('cell_%02d_batch_%04d.csv',i,ceiling(batch[1]/20))))
    done <- done+length(batch)
    write_csv(data.frame(done=done,elapsed_s=proc.time()[['elapsed']]-started),file.path(out,'progress.csv'))
    cat(sprintf('%s cell %d/%d: %d draws done, %.1f seconds\n',mode,i,nrow(cells),done,proc.time()[['elapsed']]-started)); flush.console()
  }
}
raw <- do.call(rbind,all)
s <- if(lane=='structured-mean') structured_summaries(raw,cells,mode) else geometry_summaries(raw,cells,mode)
for (name in names(s)) write_csv(s[[name]],file.path(out,paste0(name,'.csv')))
cat('Results: ',out,'\n',sep='')
if (any(nzchar(raw$error))) stop('Failures saved; inspect failures.csv before production.')
