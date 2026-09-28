#!/usr/bin/env Rscript
# Robust test calibration battery. See report.qmd for the design.
args <- commandArgs(TRUE)
filearg <- grep("^--file=", commandArgs(), value = TRUE)
root <- dirname(normalizePath(sub("^--file=", "", filearg[1])))
for (f in c("dgp.R", "tests.R", "populations.R", "summarize.R", "validate.R"))
  source(file.path(root, "R", f))
option <- function(key, default) {
  at <- match(key, args)
  if (is.na(at)) return(default)
  if (at == length(args)) stop("Missing value for ", key)
  args[at + 1L]
}
if (!length(args) || "--help" %in% args) {
  cat("Robust test calibration battery\n",
      "Modes: --plan | --validate | --smoke | --main\n",
      "Options: --reps N --seed-base N --workers N --output DIR\n",
      "         --cells ALL|case:dgp:n,...   (e.g. growth6:disc:100)\n",
      "Smoke: 5 reps/cell, seed base 2026092899. Main: 2000 reps/cell, seed base 2026092852.\n",
      "A rerun skips cells whose output already holds the requested replicates.\n",
      "Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; workers are forked processes.\n", sep = "")
  quit(save = "no")
}
known <- c("--plan", "--validate", "--smoke", "--main", "--reps", "--seed-base",
           "--workers", "--cells", "--output")
unknown <- setdiff(args[grepl("^--", args)], known)
if (length(unknown)) stop("Unknown options: ", paste(unknown, collapse = ", "))
mode <- intersect(args, c("--plan", "--validate", "--smoke", "--main"))
if (length(mode) != 1L) stop("Choose exactly one mode")
mode <- substring(mode, 3)

pops <- read_populations(file.path(root, "populations"))
grid <- full_grid()
selection <- option("--cells", "ALL")
if (selection != "ALL") {
  wanted <- strsplit(selection, ",", fixed = TRUE)[[1]]
  if (any(!wanted %in% grid$key)) stop("Unknown cells: ", paste(setdiff(wanted, grid$key), collapse = ", "))
  grid <- grid[grid$key %in% wanted, ]
}
reps <- as.integer(option("--reps", if (mode == "smoke") 5L else 2000L))
seed_base <- as.integer(option("--seed-base", if (mode %in% c("smoke", "validate")) 2026092899L else 2026092852L))
workers <- as.integer(option("--workers", 4L))
stopifnot(reps >= 1L, reps <= 50000L, workers >= 1L, !is.na(seed_base))
if (mode == "plan") {
  grid$reps <- reps
  grid$seed_first <- seed_base + grid$cell_id * 100000L + 1L
  print(grid[, c("cell_id", "case", "dgp", "n", "reps", "seed_first")], row.names = FALSE)
  cat(sprintf("%d cells, %d datasets\n", nrow(grid), nrow(grid) * reps))
  quit(save = "no")
}
out <- option("--output", file.path(root, "results", mode))
dir.create(out, recursive = TRUE, showWarnings = FALSE)

git <- function(...) tryCatch(system2("git", c("-C", root, ...), stdout = TRUE, stderr = FALSE)[1],
                              error = function(e) NA_character_)
meta <- list(mode = mode, reps = reps, seed_base = seed_base, workers = workers,
             cells = nrow(grid), selection = selection,
             command = paste(commandArgs(), collapse = " "),
             git_commit = git("rev-parse", "HEAD"),
             git_dirty = length(system2("git", c("-C", root, "status", "--porcelain", "--", "."), stdout = TRUE)) > 0,
             magmaanlab = as.character(utils::packageVersion("magmaanlab")),
             magmaanlab_built = utils::packageDescription("magmaanlab")$Built,
             r_version = R.version.string, host = Sys.info()[["nodename"]],
             cores = parallel::detectCores(), start = format(Sys.time(), tz = "UTC", usetz = TRUE))

if (mode == "validate") {
  checks <- validate_all(pops, seed_base)
  utils::write.csv(checks, file.path(out, "checks.csv"), row.names = FALSE)
  print(checks[, c("case", "dgp", "check", "abs_diff", "passed")], row.names = FALSE)
  cat(sprintf("%d of %d checks passed\n", sum(checks$passed), nrow(checks)))
  quit(save = "no", status = if (all(checks$passed)) 0L else 1L)
}

cell_dir <- file.path(out, "cells")
dir.create(cell_dir, showWarnings = FALSE)
progress <- file.path(out, "progress.log")
started <- Sys.time()
cat(sprintf("%s start: %d cells x %d reps, %d workers\n", format(started), nrow(grid), reps, workers),
    file = progress, append = TRUE)
cat(sprintf("%s: %d cells x %d reps on %d workers; progress in %s\n", mode, nrow(grid), reps, workers, progress))
# Workers append one line per finished cell: cells done so far, elapsed, rough ETA.
heartbeat <- function(key, seconds) {
  done <- max(0L, length(readLines(progress, warn = FALSE)) - 1L) + 1L
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
  cat(sprintf("%s %d/%d cells  elapsed %.1f min  eta %.1f min  %s (%.0f s)\n",
              format(Sys.time(), "%H:%M:%S"), done, nrow(grid), elapsed,
              elapsed / done * (nrow(grid) - done), key, seconds),
      file = progress, append = TRUE)
}
cell_path <- function(cell) file.path(cell_dir, sprintf("%s__%s__n%d.csv.gz", cell$case, cell$dgp, cell$n))

run_cell <- function(i) {
  cell <- grid[i, ]
  path <- cell_path(cell)
  if (file.exists(path)) {
    prev <- utils::read.csv(path)
    if (length(unique(prev$rep)) >= reps) return(data.frame(key = cell$key, status = "cached", seconds = 0))
  }
  t0 <- proc.time()[["elapsed"]]
  pop <- pops[[cell$case]]
  cals <- tryCatch(calibrate_population(pop, cell$dgp), error = function(e) e)
  if (inherits(cals, "error"))
    return(data.frame(key = cell$key, status = paste("calibration:", conditionMessage(cals)), seconds = NA))
  rows <- lapply(seq_len(reps), function(r)
    one_rep(pop, cals, cell, r, seed_base + cell$cell_id * 100000L + r))
  res <- do.call(rbind, rows)
  con <- gzfile(paste0(path, ".tmp"), "w")
  utils::write.csv(res, con, row.names = FALSE)
  close(con)
  file.rename(paste0(path, ".tmp"), path)
  seconds <- proc.time()[["elapsed"]] - t0
  heartbeat(cell$key, seconds)
  data.frame(key = cell$key, status = "done", seconds = seconds)
}

# Largest cells first, so the slowest ones do not start last.
cost <- grid$n * vapply(grid$case, function(k) length(pops[[k]]$groups[[1]]$ov)^2, numeric(1))
log <- parallel::mclapply(order(cost, decreasing = TRUE), run_cell, mc.cores = workers,
                          mc.preschedule = FALSE)
bad <- vapply(log, inherits, logical(1), "try-error")
if (any(bad)) stop("Worker errors:\n", paste(unlist(log[bad]), collapse = "\n"))
log <- do.call(rbind, log)
utils::write.csv(log, file.path(out, "cell_log.csv"), row.names = FALSE)

files <- vapply(seq_len(nrow(grid)), function(i) cell_path(grid[i, ]), character(1))
files <- files[file.exists(files)]
x <- do.call(rbind, lapply(files, utils::read.csv))
x$status[is.na(x$status)] <- ""
summary <- summarize_cells(x)
utils::write.csv(summary, file.path(out, "summary.csv"), row.names = FALSE)
utils::write.csv(summarize_arms(summary), file.path(out, "arm_summary.csv"), row.names = FALSE)
utils::write.csv(summarize_failures(x), file.path(out, "failures.csv"), row.names = FALSE)
timing <- stats::aggregate(seconds ~ case + dgp + n, data = x[!duplicated(x[, c("case", "dgp", "n", "rep")]), ], FUN = mean)
utils::write.csv(timing, file.path(out, "timing.csv"), row.names = FALSE)
meta$end <- format(Sys.time(), tz = "UTC", usetz = TRUE)
meta$datasets <- nrow(x[!duplicated(x[, c("case", "dgp", "n", "rep")]), ])
utils::write.csv(data.frame(key = names(meta), value = vapply(meta, as.character, character(1))),
                 file.path(out, "metadata.csv"), row.names = FALSE)
cat(sprintf("%s: %d cells, %d datasets\n", mode, nrow(grid), meta$datasets))
cat("Wrote", paste(file.path(out, c("summary.csv", "arm_summary.csv", "failures.csv", "timing.csv",
                                     "metadata.csv", "cell_log.csv", "cells/")), collapse = "\n      "), "\n")
