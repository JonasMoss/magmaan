#!/usr/bin/env Rscript

# Collect the slow lavaan-only side of the audit-parity experiment.
# The output cache is intentionally separate from the magmaan audit so future
# reruns can evaluate current magmaan against the same lavaan theta_hat values.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "collect_lavaan_oracle.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(dirname(script))), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

experiment_results_dir <- function(create = FALSE) {
  out <- file.path(dirname(dirname(script_path())), "results")
  if (isTRUE(create)) dir.create(out, recursive = TRUE, showWarnings = FALSE)
  out
}

# Experiment-local harness (sibling R/ dir, one level up from scripts/).
local({
  exp_root <- dirname(dirname(script_path()))
  source(file.path(exp_root, "R", "corpus.R"))
  source(file.path(exp_root, "R", "problem.R"))
  source(file.path(exp_root, "R", "lavaan_oracle.R"))
})

parse_args <- function(args) {
  out <- list(
    run_id = "",
    books = NULL,
    weights = c("GLS", "ULS", "ADF"),
    limit = NA_integer_,
    install_latest_lavaan = FALSE,
    repos = "https://cloud.r-project.org",
    lib = ""
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    value <- NULL
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript scripts/collect_lavaan_oracle.R ",
        "[--run-id RUN_ID] [--books IDS] [--weights IDS] [--limit N] ",
        "[--install-latest-lavaan] [--repos URL] [--lib PATH]\n",
        "\n",
        "Fits the corpus cells with lavaan only and writes ",
        "results/lavaan-oracle/<run-id>/{lavaan_fits,lavaan_estimates,metadata}.csv ",
        "plus RDS copies for exact replay.\n",
        "Use --install-latest-lavaan to install the newest CRAN lavaan before fitting.\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    }
    if (grepl("=", arg, fixed = TRUE)) {
      parts <- strsplit(arg, "=", fixed = TRUE)[[1L]]
      arg <- parts[[1L]]
      value <- paste(parts[-1L], collapse = "=")
    }
    need_value <- function() {
      if (!is.null(value)) return(value)
      i <<- i + 1L
      if (i > length(args)) stop(arg, " requires a value", call. = FALSE)
      args[[i]]
    }
    if (arg == "--run-id") out$run_id <- need_value()
    else if (arg == "--books") out$books <- parse_csv_arg(need_value())
    else if (arg == "--weights") out$weights <- parse_csv_arg(need_value())
    else if (arg == "--limit") out$limit <- as.integer(need_value())
    else if (arg == "--install-latest-lavaan") out$install_latest_lavaan <- TRUE
    else if (arg == "--repos") out$repos <- need_value()
    else if (arg == "--lib") out$lib <- need_value()
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  out
}

install_latest_lavaan <- function(repos, lib) {
  if (nzchar(lib)) {
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)
    .libPaths(unique(c(normalizePath(lib, mustWork = TRUE), .libPaths())))
  }
  utils::install.packages("lavaan", repos = repos,
                          lib = if (nzchar(lib)) lib else .libPaths()[[1L]])
  invisible(TRUE)
}

package_field <- function(package, field) {
  desc <- utils::packageDescription(package)
  value <- desc[[field]]
  if (is.null(value) || is.na(value)) "" else as.character(value)
}

write_precise_csv <- function(x, path) {
  old <- options(digits = 17)
  on.exit(options(old), add = TRUE)
  write_csv(x, path)
}

default_run_id <- function() {
  version <- as.character(utils::packageVersion("lavaan"))
  version <- gsub("[^A-Za-z0-9._-]+", "-", version)
  paste0("lavaan-", version, "-", format(Sys.time(), "%Y%m%d-%H%M%S"))
}

empty_fit_row <- function(case, error = NA_character_, warnings = character(),
                          elapsed_seconds = NA_real_) {
  data.frame(
    case_id = case$id,
    book = case$corpus,
    family = case$family,
    weight = case$estimator,
    lavaan_converged = NA,
    lavaan_fx = NA_real_,
    n_free_lav = NA_integer_,
    warning = paste(warnings, collapse = " | "),
    error = error,
    elapsed_seconds = elapsed_seconds,
    stringsAsFactors = FALSE
  )
}

collect_one <- function(case) {
  warnings <- character()
  started <- proc.time()[["elapsed"]]
  lav <- tryCatch(
    withCallingHandlers(
      lavaan_estimates(case, check_gradient = TRUE),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "lavaan_oracle_error"))
  elapsed <- proc.time()[["elapsed"]] - started
  if (inherits(lav, "lavaan_oracle_error")) {
    return(list(
      fit = empty_fit_row(case, error = lav$error, warnings = warnings,
                          elapsed_seconds = elapsed),
      estimates = data.frame()
    ))
  }

  estimates <- lav$estimates
  estimates$case_id <- case$id
  estimates$free_index <- seq_len(nrow(estimates))
  estimates <- estimates[, c("case_id", "free_index", "lhs", "op", "rhs",
                             "group", "est"), drop = FALSE]

  fit <- data.frame(
    case_id = case$id,
    book = case$corpus,
    family = case$family,
    weight = case$estimator,
    lavaan_converged = isTRUE(lav$converged),
    lavaan_fx = lav$objective,
    n_free_lav = nrow(lav$estimates),
    warning = paste(warnings, collapse = " | "),
    error = NA_character_,
    elapsed_seconds = elapsed,
    stringsAsFactors = FALSE
  )
  list(fit = fit, estimates = estimates)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

if (isTRUE(args$install_latest_lavaan)) {
  install_latest_lavaan(args$repos, args$lib)
}
require_pkg("lavaan")

run_id <- args$run_id
if (!nzchar(run_id)) run_id <- default_run_id()

cases <- corpus_cases(corpus_root(), weights = args$weights, books = args$books)
if (is.finite(args$limit) && args$limit > 0L) {
  cases <- cases[seq_len(min(length(cases), args$limit))]
}

out_root <- file.path(experiment_results_dir(create = TRUE), "lavaan-oracle")
out_dir <- file.path(out_root, run_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Collecting lavaan oracle for ", length(cases), " cells into ", out_dir)
started_all <- Sys.time()
fit_rows <- vector("list", length(cases))
estimate_rows <- list()
for (i in seq_along(cases)) {
  id <- names(cases)[[i]]
  if (i %% 25L == 1L || i == length(cases)) {
    elapsed <- round(as.numeric(difftime(Sys.time(), started_all,
                                         units = "secs")))
    message(sprintf("[%3d/%3d  %3ds] %s", i, length(cases), elapsed, id))
  }
  one <- collect_one(cases[[id]])
  fit_rows[[i]] <- one$fit
  if (nrow(one$estimates)) {
    estimate_rows[[length(estimate_rows) + 1L]] <- one$estimates
  }
}

fits <- do.call(rbind, fit_rows)
estimates <- if (length(estimate_rows)) {
  do.call(rbind, estimate_rows)
} else {
  data.frame(case_id = character(), free_index = integer(), lhs = character(),
             op = character(), rhs = character(), group = integer(),
             est = numeric(), stringsAsFactors = FALSE)
}

write_precise_csv(fits, file.path(out_dir, "lavaan_fits.csv"))
write_precise_csv(estimates, file.path(out_dir, "lavaan_estimates.csv"))
saveRDS(fits, file.path(out_dir, "lavaan_fits.rds"))
saveRDS(estimates, file.path(out_dir, "lavaan_estimates.rds"))
write_metadata(
  file.path(out_dir, "metadata.csv"),
  values = list(
    oracle_run = run_id,
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    corpus_root = corpus_root(),
    books = if (is.null(args$books)) "all" else paste(args$books, collapse = ","),
    weights = paste(args$weights, collapse = ","),
    limit = if (is.finite(args$limit)) args$limit else "",
    n_cells = nrow(fits),
    n_lavaan_errors = sum(!is.na(fits$error)),
    elapsed_seconds = as.numeric(difftime(Sys.time(), started_all,
                                          units = "secs")),
    install_latest_lavaan = args$install_latest_lavaan,
    repos = args$repos,
    lib = args$lib,
    lavaan_version = as.character(utils::packageVersion("lavaan")),
    lavaan_built = package_field("lavaan", "Built"),
    lavaan_repository = package_field("lavaan", "Repository"),
    lavaan_remote_sha = package_field("lavaan", "RemoteSha")
  )
)
writeLines(run_id, file.path(out_root, "LATEST"))

cat(sprintf("wrote lavaan oracle '%s'\n", run_id))
cat(sprintf("  fits:      %s\n", file.path(out_dir, "lavaan_fits.csv")))
cat(sprintf("  estimates: %s\n", file.path(out_dir, "lavaan_estimates.csv")))
cat(sprintf("  metadata:  %s\n", file.path(out_dir, "metadata.csv")))
