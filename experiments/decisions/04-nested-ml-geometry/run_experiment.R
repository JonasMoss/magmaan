#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
usage <- 'Usage: Rscript run_experiment.R --smoke|--pilot|--production [--run-id ID] [--workers W]
  --smoke       2 replicates per cell, separate development seeds
  --pilot       20 replicates per cell; compute pricing only
  --production  2000 null / 1000 power per cell; requires task-43 compute approval
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
if (any(startsWith(args, '--') & !args %in% c('--smoke','--pilot','--production','--run-id','--workers'))) stop('Unknown option')
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
run_id <- opt('--run-id', mode)
if (!grepl('^[a-zA-Z0-9_-]+$', run_id)) stop('Invalid run ID')
out <- file.path(here,'results','nested-geometry',run_id)
if (dir.exists(out)) stop('Run exists; choose a fresh --run-id')
dir.create(file.path(out,'raw'), recursive=TRUE)
cells <- geometry_cells()
seed_base <- switch(mode, smoke=526100021L, pilot=626100031L, production=726100041L)
binary <- list.files(file.path(find.package('magmaanlab'),'libs'), '\\.so$', full.names=TRUE)
hashes <- tools::md5sum(c(script, file.path(here,'R',c('compute.R','summaries.R')),
                         file.path(here,'criteria','nested_geometry.md'), binary))
write_metadata(file.path(out,'metadata.csv'), list(mode=mode, seed_base=seed_base,
  workers=workers, cells=nrow(cells), git_head=git_scalar(c('rev-parse','HEAD')),
  git_dirty=git_dirty(), source_hashes=paste(hashes,collapse=','), hash_files=paste(names(hashes),collapse=','),
  magmaanlab_path=find.package('magmaanlab'), native_md5=paste(tools::md5sum(binary),collapse=',')),
  packages=c('magmaanlab','magmaan','lavaan'))
write_csv(cells,file.path(out,'cells.csv'))
all <- list(); started <- proc.time()[['elapsed']]; done <- 0L
for (i in seq_len(nrow(cells))) {
  reps <- switch(mode, smoke=2L,pilot=20L,production=cells$production_reps[i])
  for (batch in split(seq_len(reps), ceiling(seq_len(reps)/20))) {
    values <- parallel::mclapply(batch,function(r) geometry_draw(cells[i,],r,seed_base+10000L*cells$cell_id[i]+r),
      mc.cores=workers,mc.preschedule=TRUE)
    x <- do.call(rbind,values); all[[length(all)+1L]] <- x
    write_csv(x,file.path(out,'raw',sprintf('cell_%02d_batch_%04d.csv',i,ceiling(batch[1]/20))))
    done <- done+length(batch)
    write_csv(data.frame(done=done,elapsed_s=proc.time()[['elapsed']]-started),file.path(out,'progress.csv'))
    cat(sprintf('%s cell %d/16: %d draws done, %.1f seconds\n',mode,i,done,proc.time()[['elapsed']]-started)); flush.console()
  }
}
raw <- do.call(rbind,all)
s <- geometry_summaries(raw,cells,mode)
for (name in names(s)) write_csv(s[[name]],file.path(out,paste0(name,'.csv')))
cat('Results: ',out,'\n',sep='')
if (any(nzchar(raw$error))) stop('Failures saved; inspect failures.csv before production.')
