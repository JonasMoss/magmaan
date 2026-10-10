# One frozen failed draw; no production cells are rerun.
file <- sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1])
leaf <- normalizePath(file.path(dirname(file),'..')); root <- normalizePath(file.path(leaf,'../../..'))
source(file.path(root,'experiments/_support/R/helpers.R')); source(support_path('R','basis.R')); source(support_path('R','missingness.R'))
for (name in c('config','populations','generators','targets','arms')) source(file.path(leaf,'R',paste0(name,'.R')))
set_single_threaded_math()
hash_object <- function(x) {
  path <- tempfile(); on.exit(unlink(path)); saveRDS(x,path)
  unname(tools::md5sum(path))
}
cells <- validate_config(read.csv(file.path(leaf,'production/cells_ml.csv'), na.strings='NA', stringsAsFactors=FALSE, colClasses=c(cause='character',missing_targets='character')))
cell <- as.list(cells[cells$id=='cfa_second12_d00_n200_discretized_ml_mar00',])
replicate <- 154L
seed <- as.integer((2026101001+strtoi(substr(hash_object(cell),1,7),16L)+104729*replicate) %% 2147483000+1)
population <- build_population(cell); calibration <- calibrate_generator(population,cell)
data <- draw_generator(calibration,cell$N,seed)
fit <- magmaan::magmaan(population_spec(population$record,means=TRUE),data,estimator=cell$estimator)
lab <- magmaan::as_lab_fit(fit)
old_error <- tryCatch({
  out <- arm_grid()
  out$reason <- paste0('not_converged: ', lab$verdict %||% 'library_verdict_false')
  NA_character_
}, error=function(e) conditionMessage(e))
fixed <- run_arms(data,population,cell)
stopifnot(seed == 2071474585L, !isTRUE(lab$converged),
  identical(old_error,'replacement has 10 rows, data has 24'), nrow(fixed)==24L,
  all(!fixed$converged), all(fixed$reason=='not_converged: failed'),
  all(is.na(fixed$estimate)), all(is.na(fixed$lower)), all(is.na(fixed$upper)))
diagnostic <- data.frame(cell=cell$id,replicate=replicate,seed=seed,
  converged=lab$converged,status=lab$verdict$status,stationarity=lab$verdict$stationarity,
  original_error=old_error,fixed_reason=unique(fixed$reason),rows=nrow(fixed))
path <- file.path(leaf,'results/fit_indices/2026-10-09-ml-b/harness_diagnostic.csv')
write_csv(diagnostic,path); print(diagnostic)
