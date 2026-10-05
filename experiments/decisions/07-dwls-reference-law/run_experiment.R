#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
here <- normalizePath(dirname(sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1])))
source(file.path(here,'../../_support/R/helpers.R'))
source(file.path(here,'R/study.R'))
opt <- function(key,default=NULL) if(key %in% args) args[match(key,args)+1] else default
if('--help' %in% args) {
  cat('DWLS reference law study: --smoke | --pilot | --production\n',
    '--cell ID[,ID] --reps N --workers 1|2 --out-dir NEW_DIRECTORY\n',
    'Default smoke cells 1,8; pilot cells 1:8 (two draws). Production 1000 draws/cell.\n',
    '--combine DIR[,DIR] --out-dir NEW_DIRECTORY validates COMPLETE and exact provenance.\n',
    'Frozen population JSON requires no corpus. Raw spectra are in raw.rds; summaries CSV.\n'); quit()
}
library(magmaanlab)
files <- c(file.path(here,c('run_experiment.R','R/study.R','criteria/reference_law.md')),
  sort(list.files(file.path(here,'populations'),full.names=TRUE)),
  system.file('libs',paste0('magmaanlab',.Platform$dynlib.ext),package='magmaanlab'))
fingerprint <- paste(unname(tools::md5sum(files)),collapse=':')
out <- opt('--out-dir',file.path(here,'results',if('--pilot' %in% args) 'pilot' else 'smoke'))
if(dir.exists(out)) stop('Fresh output directory required: ',out)
dir.create(out,recursive=TRUE)
if('--combine' %in% args) {
  dirs <- strsplit(opt('--combine'),',',fixed=TRUE)[[1]]
  records <- lapply(dirs,function(d) {
    if(!file.exists(file.path(d,'COMPLETE'))) stop('Incomplete attempt: ',d)
    x <- readRDS(file.path(d,'provenance.rds'))
    if(!identical(x$fingerprint,fingerprint)) stop('Source/binary/population/design mismatch: ',d)
    x
  })
  if(length(unique(vapply(records,function(x) paste(x$mode,x$base,x$reps),'')))!=1) stop('Mode/seed/count mismatch')
  cells <- do.call(rbind,lapply(dirs,function(d) read.csv(file.path(d,'cells.csv'))))
  if(anyDuplicated(cells$cell_id)) stop('Duplicate cells')
  raw <- unlist(lapply(dirs,function(d) readRDS(file.path(d,'raw.rds'))),recursive=FALSE)
  provenance <- records[[1]]; provenance$cells <- cells$cell_id
} else {
  mode <- if('--production' %in% args) 'production' else if('--pilot' %in% args) 'pilot' else 'smoke'
  base <- c(smoke=210000001,pilot=410000001,production=1910000001)[[mode]]
  reps <- as.integer(opt('--reps',c(smoke=1,pilot=2,production=1000)[[mode]]))
  workers <- as.integer(opt('--workers','1')); stopifnot(workers %in% 1:2,reps>=1,reps<=1000)
  ids <- as.integer(strsplit(opt('--cell',if(mode=='production') paste(1:192,collapse=',') else if(mode=='pilot') paste(1:8,collapse=',') else '1,8'),',',fixed=TRUE)[[1]])
  all_cells <- reference_cells(); stopifnot(!anyDuplicated(ids),all(ids %in% all_cells$cell_id))
  cells <- all_cells[match(ids,all_cells$cell_id),]
  provenance <- list(fingerprint=fingerprint,mode=mode,base=base,reps=reps,cells=ids)
  write_csv(cells,file.path(out,'cells.csv')); saveRDS(provenance,file.path(out,'provenance.rds'))
  raw <- list()
  for(i in seq_len(nrow(cells))) {
    cat('Cell',cells$cell_id[i],cells$model[i],'start\n'); flush.console()
    batch <- parallel::mclapply(seq_len(reps),function(r) reference_replicate(cells[i,],r,base,here),
      mc.cores=workers,mc.preschedule=TRUE)
    raw <- c(raw,unlist(batch,recursive=FALSE)); saveRDS(raw,file.path(out,'raw.rds'))
    cat('Cell',cells$cell_id[i],'complete\n'); flush.console()
  }
}
for(id in cells$cell_id) {
  x <- raw[vapply(raw,function(z) z$cell_id==id,FALSE)]
  expected <- if(cells$model[match(id,cells$cell_id)]=='mdd9') c('global','metric','thresholds') else c('global','nested')
  keys <- vapply(x,function(z) paste(z$replicate,z$test), '')
  if(anyDuplicated(keys) || length(keys)!=provenance$reps*length(expected)) stop('Raw count mismatch')
  for(z in x) if(z$seed!=provenance$base+10000*id+z$replicate || !(z$test %in% expected) ||
    !(z$replicate %in% seq_len(provenance$reps))) stop('Raw seed/test mismatch')
}
saveRDS(raw,file.path(out,'raw.rds')); saveRDS(provenance,file.path(out,'provenance.rds'))
write_csv(cells,file.path(out,'cells.csv'))
reference_summarize(raw,cells,out)
write_metadata(file.path(out,'metadata.csv'),list(mode=provenance$mode,seed_base=provenance$base,reps=provenance$reps,
  selected_cells=paste(cells$cell_id,collapse=','),fingerprint=fingerprint,command=paste(args,collapse=' ')),packages=c('magmaanlab','jsonlite'))
writeLines('Complete; provenance and replicate counts checked',file.path(out,'COMPLETE'))
cat('Wrote',out,'\n')
