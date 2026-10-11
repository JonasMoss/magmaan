#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  cat("Fit-index harness (production requires approved criteria).\n",
      "Rscript run_experiment.R --smoke|--pilot|--production [--config cells.csv] [--cells id,id]\n",
      "  --reps N          default 1 smoke, 5 pilot, 1000 production\n",
      "  --seed-base N     defaults 1450901 smoke, 1451901 pilot; distinct from production\n",
      "  --workers N       1 or 2 pricing; 1-18 production; single-threaded math\n",
      "  --batch-size N    default 25 draws per short-lived child (Unix required)\n",
      "  --run-id NAME     required for new runs; existing runs resume only with matching provenance\n",
      "  --lane NAME       default pricing; fit_indices for production\n",
      "Output: results/<lane>/<run-id>/{raw,summary.csv,timing.csv,targets.csv,metadata.csv}.\n",
      "Targets use population moments, or one fixed N=200000 draw for FIML MAR.\n",
      "Generators checked on N=1000000 independent draws. Integer discretized scores\n",
      "have their own score moments and targets; target_rmsea sizes the parent continuous population.\n",
      "Production requires --config, --run-id and --seed-base, and an exact\n Status: approved line in criteria/fit_indices.md.\n  --criteria PATH    alternate criteria file (for isolated validation)\nSimbox (after approval, from this study directory; rerun to resume):\n  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 Rscript run_experiment.R --production --config production/cells_ml.csv --run-id production-ml --seed-base 2026101001 --workers 18\n", sep = "")
  quit(status = 0)
}
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
leaf <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
root <- normalizePath(file.path(leaf, "../../.."))
source(file.path(root, "experiments/_support/R/helpers.R"))
source(support_path("R", "basis.R")); source(support_path("R", "missingness.R"))
for (file in c("config", "populations", "generators", "targets", "arms"))
  source(file.path(leaf, "R", paste0(file, ".R")))
set_single_threaded_math()
option <- function(name, default = NULL) {
  position <- match(paste0("--", name), args)
  if (is.na(position)) return(default)
  if (position == length(args) || startsWith(args[position+1], "--")) stop("Missing --", name)
  args[position+1]
}
flags <- c("--smoke", "--pilot", "--production")
value_flags <- paste0("--", c("config", "cells", "reps", "seed-base", "workers", "run-id", "lane", "criteria", "batch-size"))
if (any(!args[startsWith(args,"--")] %in% c(flags,value_flags))) stop("Unknown option")
if (sum(flags %in% args) != 1) stop("Select --smoke, --pilot or --production")
mode <- if ("--production" %in% args) "production" else if ("--smoke" %in% args) "smoke" else "pilot"
criteria <- NULL
if (mode == "production") {
  for (name in c("config", "run-id", "seed-base"))
    if (is.null(option(name))) stop("Production requires --", name)
  criteria <- normalizePath(option("criteria", file.path(leaf, "criteria", "fit_indices.md")), mustWork = TRUE)
  if (!"Status: approved" %in% readLines(criteria, warn = FALSE))
    stop("Production refused: criteria must contain the exact line Status: approved")
} else if (!is.null(option("criteria"))) stop("--criteria is only valid with --production")
reps <- as.integer(option("reps", if (mode == "production") "1000" else if (mode == "smoke") "1" else "5"))
seed_base <- as.integer(option("seed-base", if (mode == "smoke") "1450901" else "1451901"))
workers <- as.integer(option("workers", "1"))
batch_size <- as.integer(option("batch-size", "25"))
if (.Platform$OS.type != "unix") stop("Replicate batching requires Unix fork support")
if (!is.finite(batch_size) || batch_size < 1) stop("Invalid --batch-size")
run_id <- option("run-id"); lane <- option("lane", if (mode == "production") "fit_indices" else "pricing")
if (is.null(run_id) || !grepl("^[A-Za-z0-9_-]+$", run_id) || !grepl("^[A-Za-z0-9_-]+$",lane) ||
    !is.finite(reps) || reps < 1 || !workers %in% seq_len(if (mode == "production") 18L else 2L) || !is.finite(seed_base) || seed_base < 1)
  stop("Invalid run arguments")
config <- option("config")
cells <- validate_config(if (is.null(config)) study_config(mode) else read.csv(config, na.strings = "NA", stringsAsFactors = FALSE, colClasses = c(cause = "character", missing_targets = "character")))
all_cells <- cells
filter <- option("cells")
if (!is.null(filter)) {
  ids <- parse_csv_arg(filter)
  if (any(!ids %in% cells$id)) stop("Unknown cell id")
  cells <- cells[cells$id %in% ids, ]
}
if (!nrow(cells)) stop("No cells selected")
if (mode == "production") {
  # Observed dimension is a results-independent proxy for model cost.
  sizes <- vapply(cells$model, function(id) max(vapply(basis_model(id)$groups,
    function(g) length(g$ov), 0L)), 0L)
  cells <- cells[order(-sizes, -cells$N, cells$id), ]
}
out_dir <- file.path(leaf, "results", lane, run_id)
dir.create(file.path(out_dir, "raw"), recursive = TRUE, showWarnings = FALSE)
hash_object <- function(x) {
  tmp <- tempfile(); on.exit(unlink(tmp)); saveRDS(x,tmp); unname(tools::md5sum(tmp))
}
source_files <- c(list.files(file.path(leaf,"R"), full.names=TRUE),
  file.path(leaf,"run_experiment.R"), support_path("R","missingness.R"), support_path("R","basis.R"),
  list.files(support_path("basis","sem_score"),full.names=TRUE))
library_files <- unlist(lapply(c("magmaanlab","magmaan"), function(p)
  list.files(find.package(p), recursive=TRUE, full.names=TRUE, pattern="\\.(so|rdb)$")))
fingerprint <- hash_object(list(source=tools::md5sum(source_files), library=tools::md5sum(library_files),
  config=all_cells, seed_base=seed_base, mode=mode, batch_size=batch_size,
  criteria=if (is.null(criteria)) NULL else unname(tools::md5sum(criteria))))
manifest <- file.path(out_dir,"raw","manifest.rds")
if (file.exists(manifest)) {
  if (!identical(readRDS(manifest),fingerprint)) stop("Resume refused: source/library/config/seed changed; use a fresh run-id")
} else saveRDS(fingerprint,manifest)
metadata <- data.frame(key=c("git_head","dirty","library_build","library_hash","source_hash","config_hash",
  "seed_base","mode","command","R_version","magmaan_version","magmaanlab_version","target_n","check_n","target_controls","criteria","batch_size","batch_resume"),
  value=c(git_scalar(c("rev-parse","HEAD")),git_dirty(), packageDescription("magmaanlab")$Built,
    hash_object(tools::md5sum(library_files)),hash_object(tools::md5sum(source_files)),hash_object(all_cells),
    seed_base,mode,paste(commandArgs(),collapse=" "),R.version.string,
    as.character(packageVersion("magmaan")),as.character(packageVersion("magmaanlab")),200000,1000000,"nlopt: ftol_rel=1e-14; xtol_rel=1e-12; max_eval=10000; start=default estimates",if (is.null(criteria)) "pending; engineering only" else paste(criteria, unname(tools::md5sum(criteria))),batch_size,"atomic per-replicate checkpoints retained; incomplete batches resume missing draws"))
if (!file.exists(file.path(out_dir,"metadata.csv"))) write_csv(metadata,file.path(out_dir,"metadata.csv"))
write_csv(cells,file.path(out_dir,"cells.csv"))
write_csv(do.call(rbind,lapply(unique(cells$estimator),function(e)
  transform(arm_metadata(e),estimator=e))),file.path(out_dir,"arms.csv"))
# A seed depends on the full cell record and replicate id, never filter/order/workers.
cell_seed <- function(cell, replicate = 0L, channel = 0L) {
  h <- hash_object(cell)
  offset <- strtoi(substr(h,1,7),16L)
  as.integer((as.double(seed_base)+offset+104729*replicate+1000003*channel) %% 2147483000 + 1)
}
cache_dir <- file.path(leaf,"results","cache",fingerprint)
dir.create(cache_dir,recursive=TRUE,showWarnings=FALSE)
prepare <- function(cell) {
  # N/id are sampling labels: target and generator checks are reused across N.
  key_cell <- cell; key_cell$id <- NULL; key_cell$N <- NULL
  key <- calibration_cache_key(population = key_cell, generator = cell$generator,
    options = list(skewness = cell$skewness, kurtosis = cell$kurtosis, categories = cell$categories),
    extra = list(fingerprint = fingerprint))
  if (calibration_cache_exists(key, cache_dir)) return(calibration_cache_read(key, cache_dir))
  started <- proc.time()[[3]]
  population <- build_population(cell)
  calibration <- calibrate_generator(population,cell)
  checks <- validate_generator(calibration,cell_seed(key_cell,channel=1L))
  targets <- compute_targets(population,calibration,cell,cell_seed(key_cell,channel=2L))
  # Sampling needs the model record and calibrated generator, not preparation fits.
  population <- population[c("record", "stats", "delta")]
  result <- list(population=population,calibration=calibration,checks=checks,targets=targets,
                 setup_seconds=proc.time()[[3]]-started)
  calibration_cache_write(key, result, cache_dir); result
}
# Prepare serially so workers do not race on the same expensive cached target.
# Production prepares each work chunk just before sampling, starting costly cells early.
prepared <- vector("list",nrow(cells))
prepare_cell <- function(i) {
  cat("Preparing",cells$id[i],"(",i,"/",nrow(cells),")\n"); flush.console()
  prepared[[i]] <<- tryCatch(prepare(as.list(cells[i,])),error=function(e) e)
  if (inherits(prepared[[i]],"error")) {
    writeLines(conditionMessage(prepared[[i]]),file.path(out_dir,"raw",paste0(cells$id[i],"_setup_failure.txt")))
    stop("Cell setup failed: ", cells$id[i], ": ",conditionMessage(prepared[[i]]))
  }
}
if (mode != "production") for (i in seq_len(nrow(cells))) prepare_cell(i)
run_batch <- function(i, replicates) {
  cell <- as.list(cells[i,]); context <- prepared[[i]]
  dir <- file.path(out_dir,"raw",cell$id); dir.create(dir,showWarnings=FALSE)
  for (replicate in replicates) {
    path <- file.path(dir,sprintf("rep_%05d.rds",replicate))
    if (file.exists(path)) next
    seed <- cell_seed(cell,replicate)
    started <- proc.time()[[3]]
    result <- tryCatch({
      data <- draw_generator(context$calibration,cell$N,seed)
      if (cell$estimator == "FIML") data <- apply_mar(data,cell,cell_seed(cell,replicate,3L))
      run_arms(data,context$population,cell)
    }, error=function(e) {
      x <- arm_grid(); x$estimate <- x$lower <- x$upper <- NA_real_
      x$reason <- conditionMessage(e); x$converged <- FALSE; x
    })
    result$cell <- cell$id; result$replicate <- replicate; result$seed <- seed
    result$seconds <- proc.time()[[3]]-started
    tmp <- paste0(path,".tmp"); saveRDS(result,tmp); if (!file.rename(tmp,path)) stop("Could not save replicate checkpoint")
  }
  TRUE
}
run_cell <- function(i) {
  dir <- file.path(out_dir,"raw",cells$id[i])
  dir.create(dir,showWarnings=FALSE)
  pending <- which(!file.exists(file.path(dir,sprintf("rep_%05d.rds",seq_len(reps)))))
  marker <- file.path(dir,"COMPLETE")
  if (length(pending) && file.exists(marker) && unlink(marker) != 0L)
    stop("Could not remove stale completion marker")
  batches <- split(pending, ceiling(seq_along(pending)/batch_size))
  for (replicates in batches) {
    # Fork even with one cell worker: its supervisor never performs a fit.
    gc(full = TRUE)
    child <- parallel::mcparallel(run_batch(i,replicates), mc.set.seed=FALSE)
    answer <- parallel::mccollect(child)
    if (length(answer) != 1L || !isTRUE(answer[[1]]))
      stop("Batch child failed for ",cells$id[i],"; inspect raw checkpoints")
    if (!all(file.exists(file.path(dir,sprintf("rep_%05d.rds",replicates)))))
      stop("Batch child left missing checkpoints for ",cells$id[i])
    cat("Batch completed",cells$id[i],"through replicate",max(replicates),"/",reps,"\n")
    flush.console()
  }
  writeLines(as.character(reps),paste0(marker,".tmp"))
  if (!file.rename(paste0(marker,".tmp"),marker)) stop("Could not save completion marker")
  i
}
started <- proc.time()[[3]]
for (chunk in split(seq_len(nrow(cells)),ceiling(seq_len(nrow(cells))/workers))) {
  if (mode == "production") for (i in chunk) prepare_cell(i)
  # Preparation leaves million-row checks and target fits as unreachable garbage.
  # Collect before fork so each worker inherits only the live sampling context.
  gc(full = TRUE)
  answer <- parallel::mclapply(chunk,run_cell,mc.cores=workers,mc.preschedule=TRUE)
  if (any(vapply(answer,inherits,FALSE,"try-error"))) stop("Worker failed; inspect raw checkpoints")
  # Finished cells only need target/check summaries for the final tables.
  for (i in chunk) {
    prepared[[i]]$calibration <- NULL
    prepared[[i]]$population <- list(delta = prepared[[i]]$population$delta)
  }
  cat("Completed",paste(cells$id[chunk],collapse=", "),"elapsed",round(proc.time()[[3]]-started,1),"seconds\n")
  flush.console()
}
rows <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
  dir <- file.path(out_dir,"raw",cells$id[i])
  do.call(rbind,lapply(seq_len(reps),function(r) readRDS(file.path(dir,sprintf("rep_%05d.rds",r)))))
}))
summary <- do.call(rbind,lapply(split(rows,interaction(rows$cell,rows$arm,rows$index,drop=TRUE)),function(x) {
  i <- match(x$cell[1],cells$id); target <- prepared[[i]]$targets$own[[x$index[1]]]
  available <- is.na(x$reason) & is.finite(x$estimate)
  if (x$kind[1] == "interval") available <- available & is.finite(x$lower) & is.finite(x$upper)
  target_available <- is.finite(target)
  error <- x$estimate[available]-target
  coverage <- x$lower <= target & x$upper >= target
  data.frame(cell=x$cell[1],arm=x$arm[1],index=x$index[1],kind=x$kind[1],target=target,target_status=prepared[[i]]$targets$status,
    attempted=nrow(x),available=sum(available),availability=mean(available),convergence=mean(x$converged),
    bias=if(length(error)) mean(error) else NA_real_,rmse=if(length(error)) sqrt(mean(error^2)) else NA_real_,
    coverage=if(x$kind[1]=="interval" && any(available) && target_available) mean(coverage[available]) else NA_real_,
    coverage_attempted=if(x$kind[1]=="interval" && target_available) mean(available & coverage,na.rm=TRUE) else NA_real_,
    width=if(x$kind[1]=="interval" && any(available)) mean(x$upper[available]-x$lower[available]) else NA_real_)
}))
write_csv(summary,file.path(out_dir,"summary.csv"))
targets <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
  z <- prepared[[i]]$targets
  data.frame(cell=cells$id[i],status=z$status,reason=z$reason,delta=prepared[[i]]$population$delta,method=z$method,n=z$n,seed=z$seed,
    cause=z$cause,missing_targets=z$missing_targets,mar=z$mar,
    as.list(z$own),setNames(as.list(z$complete),paste0("complete_",names(z$complete))))
}))
write_csv(targets,file.path(out_dir,"targets.csv"))
write_csv(do.call(rbind,lapply(seq_len(nrow(cells)),function(i)
  transform(prepared[[i]]$targets$diagnostics,cell=cells$id[i]))),
  file.path(out_dir,"target_diagnostics.csv"))
write_csv(do.call(rbind,lapply(seq_len(nrow(cells)),function(i)
  transform(prepared[[i]]$checks,cell=cells$id[i]))),file.path(out_dir,"generator_checks.csv"))
timing <- unique(rows[,c("cell","replicate","seconds")])
write_csv(timing,file.path(out_dir,"timing.csv"))
cost <- merge(cells,aggregate(seconds~cell,timing,mean),by.x="id",by.y="cell")
write_csv(cost,file.path(out_dir,"cost.csv"))
failures <- rows[!is.na(rows$reason),c("cell","replicate","arm","index","reason")]
write_csv(failures,file.path(out_dir,"failures.csv"))
write_csv(data.frame(cell=cells$id,setup_seconds=vapply(prepared,`[[`,0.,"setup_seconds"),
  target_seconds=vapply(prepared,function(x) x$targets$target_seconds,0.)),file.path(out_dir,"setup_timing.csv"))
cat("Wrote",out_dir,"\n")
if (mode == "smoke" && !any(rows$converged)) stop("Smoke had no converged fits")
