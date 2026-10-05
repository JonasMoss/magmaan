#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
here <- normalizePath(dirname(sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1])))
source(file.path(here,'../../_support/R/helpers.R')); source(file.path(here,'R/study.R'))
opt <- function(key,default=NULL) if(key %in% args) args[match(key,args)+1] else default
if('--help' %in% args) {
  cat('Association-ML calibration: --smoke | --pilot | --production\n',
    '--cell ID[,ID] --reps N --workers 1|2 --out-dir NEW_DIRECTORY\n',
    '--plan prints stable full grid without fitting.\n',
    '--combine COMPLETE_DIR[,COMPLETE_DIR] --out-dir NEW_DIRECTORY\n',
    '--population-dir COMPLETE_DIR reuses exact-mode/source-checked population targets.\n',
    'Smoke defaults cells 1,97 (one replicate; 20k targets). Pilot cells 32,128 (two replicates; 1m targets x2).\n',
    'Production 2000 draws per cell; separate authorization required. No automatic production launch.\n',
    'Raw rows/checkpoints ignored; frozen summary/population/timing CSVs.\n'); quit()
}
if('--plan' %in% args) { write.table(association_cells(),stdout(),sep=',',row.names=FALSE); quit() }
library(magmaanlab)
old_bases <- c(15000001,117000001,217000001,317000001,517000001,717000001,
  817120001,817130001,817140001,817150001,817160001,830150001,840150001,
  1017000001,1217000001,1417000001,1617000001,1817000001,1917000001,2017000001,
  210000001,410000001,1910000001)
fresh_bases <- c(617000001,1117000001,1317000001,1717000001,2117000001)
stopifnot(all(abs(outer(fresh_bases,old_bases,'-'))>=100000000),
  min(dist(fresh_bases))>=100000000,max(fresh_bases)+10000*192+2000<.Machine$integer.max)
package <- find.package('magmaanlab')
files <- c(file.path(here,c('run_experiment.R','R/study.R','scripts/run_cells.sh','criteria/association_ml.md','../../_support/R/helpers.R')),
  sort(list.files(file.path(package,'libs'),full.names=TRUE)),sort(list.files(file.path(package,'R'),full.names=TRUE)))
fingerprint <- paste(unname(tools::md5sum(files)),collapse=':')
mode <- if('--production' %in% args) 'production' else if('--pilot' %in% args) 'pilot' else 'smoke'
out <- opt('--out-dir',file.path(here,'results',mode))
if(dir.exists(out)) stop('Fresh output directory required: ',out)
records <- NULL
validate_dir <- function(d) {
  if(!file.exists(file.path(d,'COMPLETE'))) stop('Incomplete attempt: ',d)
  z <- readRDS(file.path(d,'provenance.rds'))
  if(!identical(z$fingerprint,fingerprint)) stop('Source/binary/design mismatch: ',d)
  if(is.null(z$checksums) || !identical(unname(tools::md5sum(file.path(d,names(z$checksums)))),unname(z$checksums))) stop('Output checksum mismatch: ',d)
  z
}
if('--combine' %in% args) {
  dirs <- strsplit(opt('--combine'),',',fixed=TRUE)[[1]]; records <- lapply(dirs,validate_dir)
  if(length(unique(vapply(records,function(z) paste(z$mode,z$base,z$reps),'')))!=1) stop('Mode/seed/count mismatch')
  provenance <- records[[1]]; mode <- provenance$mode
  cells <- do.call(rbind,lapply(dirs,function(d) read.csv(file.path(d,'cells.csv'))))
  if(anyDuplicated(cells$cell_id)) stop('Duplicate cells')
  raw <- do.call(rbind,lapply(dirs,function(d) readRDS(file.path(d,'raw.rds'))))
  population <- unique(do.call(rbind,lapply(dirs,function(d) read.csv(file.path(d,'population_targets.csv')))))
  if(anyDuplicated(population[c('key','draw','estimator','target')])) stop('Population target mismatch')
  population_files <- c('population_targets.csv','population_stage1.csv','population_uncertainty.csv','population_timing.csv')
} else {
  base <- c(smoke=1317000001,pilot=1717000001,production=2117000001)[[mode]]
  reps <- as.integer(opt('--reps',c(smoke=1L,pilot=2L,production=2000L)[[mode]])); workers <- as.integer(opt('--workers','1'))
  stopifnot(workers %in% 1:2,reps>=1,reps<=2000)
  ids <- as.integer(strsplit(opt('--cell',if(mode=='production') paste(association_cells()$cell_id,collapse=',') else if(mode=='pilot') '32,128' else '1,97'),',',fixed=TRUE)[[1]])
  full <- association_cells(); stopifnot(!anyDuplicated(ids),all(ids %in% full$cell_id))
  cells <- full[match(ids,full$cell_id),]
  provenance <- list(fingerprint=fingerprint,mode=mode,base=base,reps=reps,cells=ids)
}
dir.create(out,recursive=TRUE)
write_csv(cells,file.path(out,'cells.csv')); saveRDS(provenance,file.path(out,'provenance.rds'))
if(!is.null(records)) {
  for(name in population_files) {
    z <- unique(do.call(rbind,lapply(dirs,function(d) read.csv(file.path(d,name)))))
    write_csv(z,file.path(out,name))
  }
} else {
  if('--population-dir' %in% args) {
    d <- opt('--population-dir'); p <- validate_dir(d)
    if(p$mode!=mode) stop('Population mode mismatch')
    population <- read.csv(file.path(d,'population_targets.csv'))
    keys <- unique(vapply(seq_len(nrow(cells)),function(i) association_key(cells[i,]),''))
    if(!all(keys %in% population$key)) stop('Missing population keys')
    for(name in c('population_targets.csv','population_stage1.csv','population_uncertainty.csv','population_timing.csv')) {
      z <- read.csv(file.path(d,name)); z <- z[z$key %in% keys,]; write_csv(z,file.path(out,name))
    }
    population <- population[population$key %in% keys,]
  } else population <- association_population(cells,out,mode)
  raw <- NULL
  for(i in seq_len(nrow(cells))) {
    cat('Cell',cells$cell_id[i],'start\n'); flush.console()
    batch <- parallel::mclapply(seq_len(reps),function(r) association_replicate(cells[i,],r,provenance$base,population),mc.cores=workers,mc.preschedule=TRUE)
    if(any(vapply(batch,inherits,FALSE,'try-error'))) stop('Worker failed outside panel accounting')
    raw <- rbind(raw,do.call(rbind,batch)); saveRDS(raw,file.path(out,'raw.rds'))
    write_csv(data.frame(completed_cells=i,total_cells=nrow(cells),completed_replicates=i*reps),file.path(out,'progress.csv'))
    cat('Cell',cells$cell_id[i],'complete\n'); flush.console()
  }
}
# Refuse corrupted or incomplete panel counts, even when failures are recorded.
full <- association_cells(); cells <- cells[order(cells$cell_id),]; rownames(cells) <- NULL
expected_cells <- full[match(cells$cell_id,full$cell_id),]; rownames(expected_cells) <- NULL
if(!isTRUE(all.equal(cells,expected_cells,check.attributes=FALSE))) stop('Cell design mismatch')
for(i in seq_len(nrow(cells))) {
  cell <- cells[i,]; z <- raw[raw$cell_id==cell$cell_id,]
  targets <- c('loading','threshold',if(cell$factors==2) 'correlation')
  expected <- if(cell$groups==1) c(paste('association_ij',targets),paste('dwls_ij',targets),
    paste(paste0('association_global_',c('All','SB','PEBA4')),'global'),'dwls_global_All global',
    'association_mi_size mi','association_mi_power mi') else
    paste(c(paste0('association_nested_',c('All','SB','PEBA4')),paste0('dwls_nested_',c('All','SB','PEBA4'))),'nested')
  keys <- paste(z$replicate,z$arm,z$target)
  if(anyDuplicated(keys) || nrow(z)!=length(expected)*provenance$reps ||
    any(z$seed!=provenance$base+10000*cell$cell_id+z$replicate) ||
    any(!z$replicate %in% seq_len(provenance$reps)) ||
    !setequal(keys,as.vector(outer(seq_len(provenance$reps),expected,paste)))) stop('Raw seed/panel/count mismatch')
}
if(!setequal(unique(raw$cell_id),cells$cell_id)) stop('Unexpected raw cell')
provenance$cells <- cells$cell_id
saveRDS(raw,file.path(out,'raw.rds')); saveRDS(provenance,file.path(out,'provenance.rds')); write_csv(cells,file.path(out,'cells.csv'))
association_summarize(raw,cells,out)
write_metadata(file.path(out,'metadata.csv'),list(mode=provenance$mode,seed_base=provenance$base,reps=provenance$reps,
  selected_cells=paste(cells$cell_id,collapse=','),fingerprint=fingerprint,criteria_md5=unname(tools::md5sum(file.path(here,'criteria/association_ml.md'))),
  population_n=if(mode=='smoke') 20000 else 1000000,command=paste(args,collapse=' '),
  git_commit=system2('git',c('-C',shQuote(here),'rev-parse','HEAD'),stdout=TRUE)),packages='magmaanlab')
checksum_files <- c('cells.csv','raw.rds','population_targets.csv','population_stage1.csv','population_uncertainty.csv','population_timing.csv','summary.csv','failures.csv','timing.csv','cost.csv')
provenance$checksums <- setNames(unname(tools::md5sum(file.path(out,checksum_files))),checksum_files)
saveRDS(provenance,file.path(out,'provenance.rds'))
writeLines('Complete; source, population, seed and panel counts checked',file.path(out,'COMPLETE')); cat('Wrote',out,'\n')
