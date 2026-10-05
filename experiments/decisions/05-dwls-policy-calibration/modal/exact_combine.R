# Called by combine.R after loading the lane's compute helpers.
exact_combine <- function(run_dir,mode,git_head,family,opt,study) {
  source(file.path(study,'R','exact_first_stage.R'))
  cells <- exact_cells()
  if(family!='all') cells <- cells[cells$family==family,]
  if('--cell' %in% args) {
    ids <- as.integer(strsplit(opt('--cell'),',',fixed=TRUE)[[1]])
    if(any(!ids %in% cells$cell_id)) stop('Unknown selected cell')
    cells <- cells[cells$cell_id %in% ids,]
  }
  roots <- file.path(run_dir,'cells',sprintf('cell_%03d',cells$cell_id))
  if(any(!file.exists(file.path(roots,'COMPLETE')))) stop('Missing cell completion marker')
  metadata <- lapply(file.path(roots,'metadata.csv'),read.csv,stringsAsFactors=FALSE)
  value <- function(m,key) { z <- m$value[m$key==key]; if(length(z)!=1) stop('Missing metadata ',key); z }
  same <- c('lane','mode','seed_base','source_hashes','native_md5','reps_override',
    'population_n_per_group','population_seed_base','package_magmaanlab','package_lavaan')
  # Some helper versions prefix package keys differently: compare every
  # package/version key in addition to the mandatory provenance fields.
  same <- unique(c(same[!startsWith(same,'package_')],
    metadata[[1]]$key[grepl('package|version',metadata[[1]]$key)]))
  for(key in same) if(length(unique(vapply(metadata,value,'',key=key)))!=1) stop('Provenance mismatch: ',key)
  if(value(metadata[[1]],'lane')!='exact-first-stage' || value(metadata[[1]],'mode')!=mode) stop('Lane/mode mismatch')
  binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
  sources <- c(file.path(study,'run_experiment.R'),sort(list.files(file.path(study,'R'),full.names=TRUE)),
    file.path(study,'criteria','exact_first_stage.md'),binary)
  if(value(metadata[[1]],'source_hashes')!=paste(tools::md5sum(sources),collapse=',')) stop('Current source/package provenance mismatch')
  for(package in c('magmaanlab','lavaan'))
    if(value(metadata[[1]],paste0(package,'_version'))!=as.character(packageVersion(package))) stop('Current package version mismatch')
  if(value(metadata[[1]],'R_version')!=R.version.string) stop('Current R version mismatch')
  raw <- do.call(rbind,lapply(file.path(roots,'raw.rds'),readRDS))
  raw <- raw[order(raw$cell_id,raw$replicate,raw$arm,raw$target),]; rownames(raw) <- NULL
  if(anyDuplicated(raw[c('cell_id','replicate','arm','target')])) stop('Duplicate rows')
  if(!setequal(unique(raw$cell_id),cells$cell_id)) stop('Cell IDs do not match')
  seed_base <- exact_seed_base(mode)
  if(any(raw$seed!=seed_base+10000L*raw$cell_id+raw$replicate)) stop('Seed mismatch')
  override <- as.integer(value(metadata[[1]],'reps_override'))
  for(i in seq_len(nrow(cells))) {
    expected <- if(is.na(override)) cells$production_reps[i] else override
    actual <- unique(raw$replicate[raw$cell_id==cells$cell_id[i]])
    if(!setequal(actual,seq_len(expected))) stop('Missing replicate')
    if(value(metadata[[i]],'selected_cells')!=as.character(cells$cell_id[i])) stop('Cell metadata mismatch')
  }
  exact_validate_seeds()
  if(any(is.finite(raw$policy_gap) & raw$policy_gap>1e-7)) stop('Policy equivalence gate failed')
  out <- file.path(run_dir,'final')
  if(dir.exists(out)) stop('Final exists; use fresh run ID')
  dir.create(out)
  saveRDS(raw,file.path(out,'raw.rds'))
  meta <- metadata[[1]]
  meta$value[meta$key=='selected_cells'] <- paste(cells$cell_id,collapse=',')
  meta <- rbind(meta,data.frame(key=c('executor','combine_git_head'),value=c('cell_fanout',git_head)))
  write_csv(meta,file.path(out,'metadata.csv')); write_csv(cells,file.path(out,'cells.csv'))
  populations <- lapply(roots,function(root) {
    path <- file.path(root,'population_targets.csv')
    if(file.exists(path)) read.csv(path) else NULL
  })
  pop <- do.call(rbind,populations)
  if(!is.null(pop)) {
    for(key in unique(pop$key)) for(target in unique(pop$target)) {
      values <- pop$estimate[pop$key==key & pop$target==target]
      if(length(unique(values))>1) stop('Population target mismatch')
    }
    pop <- pop[!duplicated(pop[c('key','target')]),]
    write_csv(pop,file.path(out,'population_targets.csv'))
  }
  exact_summarize(raw,cells,out)
  writeLines('complete',file.path(out,'COMPLETE'))
  cat('Combined exact-first-stage cells: ',out,'\n',sep='')
}
