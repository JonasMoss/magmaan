`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

script_path <- function(default = "run_experiment.R") {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  }
  ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
  normalizePath(ofile %||% default, mustWork = FALSE)
}

repo_root <- function(start = dirname(script_path())) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(path, "CMakeLists.txt")) &&
        file.exists(file.path(path, "r-package", "DESCRIPTION"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("Could not find magmaan repository root", call. = FALSE)
    }
    path <- parent
  }
}

experiment_dir <- function(script = script_path()) {
  dirname(script)
}

experiment_path <- function(..., script = script_path()) {
  file.path(experiment_dir(script), ...)
}

results_dir <- function(script = script_path(), create = FALSE) {
  out <- experiment_path("results", script = script)
  if (isTRUE(create)) dir.create(out, recursive = TRUE, showWarnings = FALSE)
  out
}

ensure_results_dir <- function(script = script_path()) {
  results_dir(script = script, create = TRUE)
}

require_pkg <- function(package, hint = NULL) {
  if (!requireNamespace(package, quietly = TRUE)) {
    msg <- paste0("required R package is not installed: ", package)
    if (!is.null(hint) && nzchar(hint)) msg <- paste0(msg, "; ", hint)
    stop(msg, call. = FALSE)
  }
  invisible(TRUE)
}

parse_csv_arg <- function(x) {
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

parse_csv_numeric <- function(x) {
  as.numeric(parse_csv_arg(x))
}

set_single_threaded_math <- function(vars = c("OMP_NUM_THREADS",
                                              "OPENBLAS_NUM_THREADS",
                                              "MKL_NUM_THREADS")) {
  for (v in vars) do.call(Sys.setenv, stats::setNames(list("1"), v))
  invisible(vars)
}

write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  invisible(path)
}

write_rows <- function(rows, path) {
  if (length(rows)) write_csv(do.call(rbind, rows), path)
  else write_csv(data.frame(), path)
}

metadata_value <- function(x) {
  if (is.null(x) || !length(x)) return("")
  x <- unlist(x, use.names = FALSE)
  if (!length(x)) return("")
  paste(as.character(x), collapse = ",")
}

metadata_frame <- function(values = list(), packages = character(),
                           include_r = TRUE, include_command = TRUE) {
  rows <- values
  if (isTRUE(include_r)) rows <- c(rows, list(R_version = R.version.string))
  if (isTRUE(include_command)) {
    rows <- c(rows, list(command = paste(commandArgs(trailingOnly = FALSE),
                                         collapse = " ")))
  }
  for (package in packages) {
    version <- if (requireNamespace(package, quietly = TRUE)) {
      as.character(utils::packageVersion(package))
    } else {
      ""
    }
    rows[[paste0(package, "_version")]] <- version
  }
  data.frame(
    key = names(rows),
    value = vapply(rows, metadata_value, character(1)),
    stringsAsFactors = FALSE
  )
}

append_metadata <- function(metadata, values = list(), packages = character()) {
  extra <- metadata_frame(values, packages = packages,
                          include_r = FALSE, include_command = FALSE)
  rbind(metadata, extra)
}

write_metadata <- function(path, values = list(), packages = character(),
                           include_r = TRUE, include_command = TRUE) {
  meta <- metadata_frame(values, packages = packages, include_r = include_r,
                         include_command = include_command)
  write_csv(meta, path)
}
