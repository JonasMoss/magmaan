#!/usr/bin/env Rscript
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
here <- dirname(script)
if ("--help" %in% commandArgs(TRUE)) {
  cat("Newton check in the default verdict of FIML and least-squares fits.\n",
      "Fits every continuous textbook-corpus case with the default GLS and ULS\n",
      "fitters (sample moments) and FIML (single-group raw data), every categorical\n",
      "case with DWLS (raw data), and records the\n",
      "fit verdict next to the first-order verdict it replaces.\n\n",
      "Usage: Rscript run_experiment.R [--cases id,id] [--workers N] [--timeout SEC]\n",
      "  --corpus PATH   textbook-corpus mount (default: the support helper's)\n",
      "  --results PATH  output directory (default: results/current)\n",
      "Writes results/<dir>/fits.csv and metadata.csv.\n", sep = "")
  quit(save = "no")
}
source(file.path(here, "../../_support/R/helpers.R"))
set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab); library(lavaan); library(jsonlite)})
source(file.path(here, "R/corpus.R"))
`%||%` <- function(x, y) if (is.null(x)) y else x
argv <- commandArgs(TRUE)
value <- function(key, default = NULL) {i <- match(key, argv); if (is.na(i)) default else argv[i + 1L]}
root <- normalizePath(value("--corpus", corpus_root()))
out <- normalizePath(value("--results", file.path(here, "results/current")), mustWork = FALSE)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
manifest <- read.csv(file.path(root, "manifest.csv"), stringsAsFactors = FALSE)

record <- function(fit) {
  g <- fit$diagnostics$geometric_stationarity
  n <- fit$diagnostics$newton_accuracy
  first_order <- if (!isTRUE(g$checked) || !isTRUE(g$ambient_projection_converged)) NA else
    identical(fit$verdict$objective, "passed") && isTRUE(g$ambient_stationary)
  list(converged = isTRUE(fit$converged), criterion = fit$verdict$criterion %||% "",
       first_order = first_order, residual_l2 = g$ambient_residual_l2 %||% NA_real_,
       newton_checked = isTRUE(n$checked), newton_status = n$status %||% "",
       distance = n$distance %||% NA_real_, metric = n$metric %||% "",
       curvature = n$curvature %||% "", condition = n$condition %||% NA_real_,
       fmin = fit$fmin %||% NA_real_, optimizer_status = fit$optimizer_status %||% "",
       admissible = isTRUE(fit$diagnostics$admissibility$admissible))
}
empty <- list(converged = NA, criterion = "", first_order = NA, residual_l2 = NA_real_,
              newton_checked = NA, newton_status = "", distance = NA_real_, metric = "",
              curvature = "", condition = NA_real_, fmin = NA_real_, optimizer_status = "",
              admissible = NA)

worker <- function(id) {
  path <- file.path(out, paste0(id, ".csv"))
  rows <- list()
  add <- function(estimator, res, message = "", seconds = NA_real_)
    rows[[length(rows) + 1L]] <<- c(list(case = id, estimator = estimator, message = message,
                                         seconds = seconds), res)
  case <- tryCatch(read_case(root, manifest$case_dir[match(id, manifest$case_id)]),
                   error = function(e) e)
  if (inherits(case, "error")) {
    add("none", empty, conditionMessage(case))
  } else {
    fitters <- list()
    if (identical(case$kind, "ordinal")) {
      fitters$DWLS <- function() {
        args <- list(model = case$syntax, data = case$raw, estimator = "DWLS",
                     ordered = case$ordered, parameterization = case$parameterization)
        if (!is.null(case$groups)) args$groups <- case$groups
        if (!is.null(case$group_equal)) args$group_equal <- case$group_equal
        do.call(fit_model, args)
      }
    } else if (!case$missing_data) {
      fitters$GLS <- function() magmaan_core$fit_gls(case$model, case$sample)
      fitters$ULS <- function() magmaan_core$fit_uls(case$model, case$sample)
    }
    if (!identical(case$kind, "ordinal") && !is.null(case$raw)) {
      fitters$FIML <- function()
        magmaan_core$fit_fiml(case$model, df_to_fiml_data(case$raw, case$model))
    }
    for (est in names(fitters)) {
      t0 <- proc.time()[["elapsed"]]
      fit <- tryCatch(suppressWarnings(fitters[[est]]()), error = function(e) e)
      sec <- proc.time()[["elapsed"]] - t0
      if (inherits(fit, "error")) add(est, empty, conditionMessage(fit), sec)
      else add(est, record(fit), "", sec)
    }
  }
  if (!length(rows)) add("none", empty, "excluded: no applicable estimator")
  df <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
  utils::write.csv(df, path, row.names = FALSE)
}

if ("--worker" %in% argv) {
  worker(value("--worker"))
  quit(save = "no")
}
ids <- manifest$case_id
if (length(value("--cases"))) ids <- strsplit(value("--cases"), ",", fixed = TRUE)[[1]]
workers <- as.integer(value("--workers", "4"))
timeout <- value("--timeout", "300")
unlink(file.path(out, "cases"), recursive = TRUE)
dir.create(file.path(out, "cases"), showWarnings = FALSE)
case_out <- file.path(out, "cases")
status <- parallel::mclapply(seq_along(ids), function(i) {
  code <- system2("timeout", c(timeout, file.path(R.home("bin"), "Rscript"), shQuote(script),
                               "--worker", ids[i], "--corpus", shQuote(root),
                               "--results", shQuote(case_out)),
                  stdout = FALSE, stderr = FALSE)
  if (i %% 25L == 0L) cat(sprintf("[%d/%d]\n", i, length(ids)))
  data.frame(case = ids[i], exit = code)
}, mc.cores = workers, mc.preschedule = FALSE)
status <- do.call(rbind, status)
files <- file.path(case_out, paste0(ids, ".csv"))
read_case_file <- function(f) tryCatch(utils::read.csv(f, stringsAsFactors = FALSE),
                                       error = function(e) NULL)
parts <- lapply(files[file.exists(files)], read_case_file)
fits <- do.call(rbind, parts[!vapply(parts, is.null, logical(1))])
done <- unique(fits$case)
timed_out <- union(status$case[status$exit != 0], setdiff(ids, done))
if (length(timed_out)) {
  fits <- rbind(fits, do.call(rbind, lapply(timed_out, function(id)
    as.data.frame(c(list(case = id, estimator = "none", message = "timeout or crash",
                         seconds = NA_real_), empty), stringsAsFactors = FALSE))))
}
utils::write.csv(fits, file.path(out, "fits.csv"), row.names = FALSE)
write_metadata(file.path(out, "metadata.csv"), values = list(
  corpus_head = system2("git", c("-C", shQuote(root), "rev-parse", "HEAD"), stdout = TRUE),
  magmaan_git_head = magmaan_cache_ref()$git_head,
  magmaan_git_dirty = magmaan_cache_ref()$git_dirty,
  workers = workers, timeout_seconds = timeout), packages = c("magmaanlab", "lavaan"))
cat("Wrote", file.path(out, "fits.csv"), "\n")
