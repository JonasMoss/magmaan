#!/usr/bin/env Rscript
# Verify a completed cell before a resumed remote launch skips it.
args <- commandArgs(TRUE)
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
study <- dirname(dirname(script))
source(file.path(study,'R','compute.R'))
source(file.path(study,'R','exact_first_stage.R'))
root <- args[1]; mode <- args[2]; id <- as.integer(args[3])
if(!file.exists(file.path(root,'COMPLETE'))) stop('Missing completion marker')
meta <- read.csv(file.path(root,'metadata.csv'))
value <- function(key) { z <- meta$value[meta$key==key]; if(length(z)!=1) stop('Missing ',key); z }
if(value('lane')!='exact-first-stage' || value('mode')!=mode ||
   value('selected_cells')!=as.character(id) ||
   value('seed_base')!=as.character(exact_seed_base(mode))) stop('Cell provenance mismatch')
if(value('source_hashes')!=paste(tools::md5sum(exact_source_files(study)),collapse=',')) stop('Source/package mismatch')
for(package in c('magmaanlab','lavaan'))
  if(value(paste0(package,'_version'))!=as.character(packageVersion(package))) stop('Package version mismatch')
if(value('R_version')!=R.version.string) stop('R version mismatch')
raw <- readRDS(file.path(root,'raw.rds'))
cells <- exact_cells(); cell <- cells[cells$cell_id==id,]
override <- as.integer(value('reps_override'))
expected <- if(is.na(override)) cell$production_reps else override
if(!setequal(unique(raw$replicate),seq_len(expected)) || any(raw$cell_id!=id) ||
   any(raw$seed!=exact_seed_base(mode)+10000L*id+raw$replicate) ||
   anyDuplicated(raw[c('replicate','arm','target')])) stop('Cell rows mismatch')
