# TASK-157: real fit rows through the old/new cell worker, without target preparation.
args <- commandArgs(TRUE)
if (length(args) != 4L) stop('Usage: check_batches.R before|after ML|FIML REPS OUTPUT')
if (!args[1] %in% c('before','after') || !args[2] %in% c('ML','FIML') ||
    !as.integer(args[3]) %in% c(50L,400L)) stop('Invalid bounded probe arguments')
file <- sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1])
leaf <- normalizePath(file.path(dirname(file),'..')); root <- normalizePath(file.path(leaf,'../../..'))
source(file.path(root,'experiments/_support/R/helpers.R'))
source(support_path('R','basis.R')); source(support_path('R','missingness.R'))
for (name in c('config','populations','generators','targets','arms')) source(file.path(leaf,'R',paste0(name,'.R')))
set_single_threaded_math()
# Frozen pre-change worker; no sibling-study reads.
before_ref <- '2e02a4ca46fb379200c967925b8a001786a71712'
code <- if (args[1] == 'before') system2('git', c('-C',shQuote(root),'show',
  paste0(before_ref,':experiments/decisions/09-fit-indices/run_experiment.R')),stdout=TRUE) else
  readLines(file.path(leaf,'run_experiment.R'))
for (expr in parse(text=code)) {
  if (is.call(expr) && identical(expr[[1]],as.name('<-')) &&
      as.character(expr[[2]]) %in% c('hash_object','cell_seed','run_batch','run_cell')) eval(expr)
}
seed_base <- 2026101101L; batch_size <- 25L; reps <- as.integer(args[3])
cells <- study_config('smoke')[1,,drop=FALSE]
cells$model <- if (reps == 400L) 'cfa_long18' else 'cfa_ptsd8'
cells$N <- if (reps == 400L) 2000L else 200L
cells$estimator <- args[2]; cells$missing_rate <- if (args[2]=='FIML') .30 else 0
cells$delta <- -.1; cells$id <- paste0('batch_check_',args[2])
cell <- as.list(cells[1,]); population <- build_population(cell)
calibration <- calibrate_generator(population,cell)
population <- population[c('record','stats','delta')]
prepared <- list(list(population=population,calibration=calibration))
rm(population,calibration); gc(full=TRUE)
out_dir <- args[4]; dir.create(file.path(out_dir,'raw'),recursive=TRUE,showWarnings=FALSE)
write_csv(cells,file.path(out_dir,'cells.csv'))
run_cell(1L)
cat('DONE',args[1],args[2],reps,'\n')
