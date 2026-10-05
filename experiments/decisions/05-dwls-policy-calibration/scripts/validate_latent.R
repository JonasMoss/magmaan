#!/usr/bin/env Rscript
# Development gates: exact pair-table assembly and deterministic cell fan-out.
args <- commandArgs(TRUE)
if(length(args)!=3) stop('Usage: validate_latent.R SERIAL_DIR COMBINED_DIR CHECKS_CSV')
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
study <- dirname(dirname(script))
source(file.path(study,'R','compute.R'))
source(file.path(study,'R','latent_nonnormal.R'))
checks <- list()
for(id in c(1L,20L,30L)) {
  cell <- latent_cells()[id,]
  assembled <- latent_population_stats(cell,217000001L,2000L)
  ordinary <- magmaanlab::magmaan_core$data_ordinal_stats_from_raw(
    as.matrix(latent_draw(cell,217000001L,n=2000L)),full_wls_weight=FALSE)
  for(slot in c('moments','W_dwls')) {
    gap <- max(abs(assembled[[slot]][[1]]-ordinary[[slot]][[1]]))
    checks[[length(checks)+1L]] <- data.frame(check=paste('population',id,slot,sep='_'),
      max_gap=gap,tolerance=1e-7,passed=is.finite(gap) && gap<1e-7)
  }
}
for(name in c('summary','paired','population_targets','population_stage1','population_uncertainty')) {
  a <- read.csv(file.path(args[1],paste0(name,'.csv')))
  b <- read.csv(file.path(args[2],paste0(name,'.csv')))
  keys <- intersect(c('cell_id','arm','exact_arm','target','key','draw','moment'),names(a))
  sort_rows <- function(x) {
    if(length(keys)) x <- x[do.call(order,x[keys]),]
    rownames(x) <- NULL; x
  }
  passed <- isTRUE(all.equal(sort_rows(a),sort_rows(b),tolerance=1e-12))
  checks[[length(checks)+1L]] <- data.frame(check=paste0('fanout_',name),max_gap=NA_real_,
    tolerance=1e-12,passed=passed)
}
checks <- do.call(rbind,checks)
write.csv(checks,args[3],row.names=FALSE)
print(checks)
if(!all(checks$passed)) stop('Latent development gate failed')
