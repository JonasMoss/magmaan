#!/usr/bin/env Rscript
# Cheap fan-out regression: independent draws, then exact reconstruction including timing.
args <- commandArgs(TRUE)
if('--help' %in% args) {
  cat('Usage: Rscript modal/check_local.R [--lane structured-mean|nested-geometry]\n')
  quit(save='no')
}
lane <- if('--lane' %in% args) args[match('--lane',args)+1L] else 'structured-mean'
if(!lane %in% c('structured-mean','nested-geometry')) stop('Unknown lane')
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'../../_support/R/helpers.R')); set_single_threaded_math()
run <- tempfile(paste0('local-fanout-',lane,'-'),tmpdir=file.path(here,'results'))
dir.create(run,recursive=TRUE)
run_r <- function(file,args) {
  status <- system2(file.path(R.home('bin'),'Rscript'),c(shQuote(file),shQuote(args)))
  if(status!=0L) stop('Runner failed: ',file)
}
runner <- file.path(here,'run_experiment.R'); combiner <- file.path(here,'modal/combine.R')
single <- file.path(run,'single'); fan <- file.path(run,'fan')
run_r(runner,c('--lane',lane,'--smoke','--cells','1,2','--workers','2','--out-dir',single))
for(id in 1:2) run_r(runner,c('--lane',lane,'--smoke','--cell',id,'--workers','2',
  '--out-dir',file.path(fan,'cells',sprintf('cell_%03d',id))))
combine_args <- c('--lane',lane,'--mode','smoke','--cell','1,2','--run-dir')
run_r(combiner,c(combine_args,fan))
summary_files <- setdiff(list.files(single,pattern='[.]csv$'),c('metadata.csv','progress.csv'))
compare <- function(out,files) {
  for(name in files) if(!identical(readBin(file.path(single,name),'raw',n=file.info(file.path(single,name))$size),
    readBin(file.path(out,name),'raw',n=file.info(file.path(out,name))$size))) stop('Byte mismatch: ',name)
}
# Actual wall times differ between executions; all statistical summaries must match.
compare(file.path(fan,'final'),setdiff(summary_files,'timing.csv'))
raw <- readRDS(file.path(single,'raw.rds'))
replay <- file.path(run,'replay')
for(id in 1:2) {
  d <- file.path(replay,'cells',sprintf('cell_%03d',id)); dir.create(d,recursive=TRUE)
  saveRDS(raw[raw$cell_id==id,],file.path(d,'raw.rds'))
  file.copy(file.path(single,'metadata.csv'),file.path(d,'metadata.csv'))
}
run_r(combiner,c(combine_args,replay))
compare(file.path(replay,'final'),summary_files)
cat('PASS: independent statistical summaries and all replay summaries match byte-for-byte.\nOutputs: ',run,'\n',sep='')
