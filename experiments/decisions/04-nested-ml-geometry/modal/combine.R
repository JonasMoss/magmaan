#!/usr/bin/env Rscript
# Reuse the lane's summaries on complete, provenance-matched cell outputs.
args <- commandArgs(TRUE)
opt <- function(key, default=NULL) {
  at <- match(key,args)
  if(is.na(at)) { if(is.null(default)) stop('Missing ',key); return(default) }
  if(at==length(args) || startsWith(args[at+1L],'--')) stop('Missing ',key)
  args[at+1L]
}
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'../../_support/R/helpers.R')); set_single_threaded_math()
source(file.path(here,'R/compute.R')); source(file.path(here,'R/summaries.R'))
lane <- opt('--lane','structured-mean'); mode <- opt('--mode'); run <- opt('--run-dir')
if(!lane %in% c('nested-geometry','structured-mean')) stop('Unknown lane')
if(!mode %in% c('smoke','pilot','production')) stop('Unknown mode')
if(lane=='structured-mean') source(file.path(here,'R/structured_mean.R'))
cells <- if(lane=='structured-mean') structured_cells() else geometry_cells()
selected <- opt('--cell','all')
if(selected!='all') {
  ids <- as.integer(strsplit(selected,',',fixed=TRUE)[[1]])
  if(anyNA(ids)||anyDuplicated(ids)||any(!ids %in% cells$cell_id)) stop('Invalid cells')
  cells <- cells[cells$cell_id %in% ids,]
}
seed <- if(lane=='structured-mean') switch(mode,smoke=1326100021L,pilot=1526100031L,production=1726100041L) else
  switch(mode,smoke=526100021L,pilot=626100031L,production=726100041L)
dirs <- file.path(run,'cells',sprintf('cell_%03d',cells$cell_id))
files <- file.path(dirs,'raw.rds')
if(!all(file.exists(files))) stop('Incomplete cell outputs')
meta <- lapply(dirs,function(d) read.csv(file.path(d,'metadata.csv'),stringsAsFactors=FALSE))
keys <- c('mode','lane','seed_base','git_head','source_hashes','native_md5')
provenance <- function(m) {
  if(!all(keys %in% m$key)) stop('Missing provenance')
  m$value[match(keys,m$key)]
}
if(!all(vapply(meta,function(m) identical(provenance(m),provenance(meta[[1]])),FALSE))) stop('Provenance mismatch')
if(meta[[1]]$value[match('lane',meta[[1]]$key)]!=lane || meta[[1]]$value[match('mode',meta[[1]]$key)]!=mode) stop('Lane/mode mismatch')
parts <- lapply(files,readRDS)
for(i in seq_len(nrow(cells))) {
  x <- parts[[i]]; reps <- switch(mode,smoke=2L,pilot=20L,production=cells$production_reps[i])
  if(lane=='nested-geometry') x$arm <- paste(x$geometry,x$test)
  arms <- if(lane=='structured-mean') c('policy_lr','policy_score','pre66_lr') else c('expected score','observed score','expected lr','observed lr')
  if(any(x$cell_id!=cells$cell_id[i]) || nrow(x)!=length(arms)*reps ||
     anyDuplicated(x[c('rep','arm')]) || !setequal(x$rep,seq_len(reps)) || !setequal(x$arm,arms)) stop('Incomplete or duplicate draws')
  if(any(x$seed!=seed+10000L*x$cell_id+x$rep)) stop('Seed mismatch')
}
raw <- do.call(rbind,parts); rownames(raw) <- NULL
out <- file.path(run,'final')
if(dir.exists(out)) stop('Final exists; use a fresh run ID')
dir.create(out,recursive=TRUE)
saveRDS(raw,file.path(out,'raw.rds')); write_csv(cells,file.path(out,'cells.csv'))
write_csv(if(lane=='structured-mean') structured_population_checks() else geometry_population(),file.path(out,'population.csv'))
s <- if(lane=='structured-mean') structured_summaries(raw,cells,mode) else geometry_summaries(raw,cells,mode)
for(name in names(s)) write_csv(s[[name]],file.path(out,paste0(name,'.csv')))
write_metadata(file.path(out,'metadata.csv'),list(mode=mode,lane=lane,seed_base=seed,
  executor='modal',cells=nrow(cells),git_head=meta[[1]]$value[match('git_head',meta[[1]]$key)],
  source_hashes=meta[[1]]$value[match('source_hashes',meta[[1]]$key)],source_metadata=paste(dirs,collapse=';')),
  packages=c('magmaanlab','magmaan','lavaan'))
cat('Combined into ',out,'\n',sep='')
if(any(nzchar(raw$error))) stop('Failures saved; inspect failures.csv before production.')
