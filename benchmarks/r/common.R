script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1]]$ofile %||% getwd(), mustWork = FALSE)
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x)) y else x
}

bench_script_dir <- function() {
  dirname(script_path())
}

bench_root <- function() {
  normalizePath(file.path(bench_script_dir(), ".."), mustWork = TRUE)
}

repo_root <- function() {
  normalizePath(file.path(bench_root(), ".."), mustWork = TRUE)
}

case_dir <- function(case_id) {
  file.path(bench_root(), "cases", case_id)
}

raw_dir <- function(case_id) {
  file.path(bench_root(), "data", "raw", case_id)
}

prepared_dir <- function(case_id) {
  file.path(bench_root(), "data", "prepared", case_id)
}

prepared_data_path <- function(case_id) {
  file.path(prepared_dir(case_id), "data.csv")
}

prepared_metadata_path <- function(case_id) {
  file.path(prepared_dir(case_id), "metadata.json")
}

reference_path <- function(case_id, engine = "lavaan") {
  file.path(case_dir(case_id), sprintf("reference_%s.json", engine))
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

read_model <- function(case_id) {
  path <- file.path(case_dir(case_id), "model.lav")
  if (!file.exists(path)) stop("missing model syntax: ", path, call. = FALSE)
  paste(readLines(path, warn = FALSE), collapse = "\n")
}

write_json_file <- function(x, path) {
  ensure_dir(dirname(path))
  jsonlite::write_json(
    x,
    path,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = NA
  )
  invisible(path)
}

require_pkg <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("required R package is not installed: ", package, call. = FALSE)
  }
  invisible(TRUE)
}

safe_package_version <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(package))
}

write_prepared_data <- function(case_id, data, source_note) {
  ensure_dir(prepared_dir(case_id))
  utils::write.csv(data, prepared_data_path(case_id), row.names = FALSE, na = "")
  meta <- list(
    case_id = case_id,
    source = source_note,
    n_obs = nrow(data),
    variables = names(data),
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  )
  write_json_file(meta, prepared_metadata_path(case_id))
  invisible(meta)
}

read_prepared_data <- function(case_id) {
  path <- prepared_data_path(case_id)
  if (!file.exists(path)) {
    stop(
      "prepared data not found for ", case_id,
      "; run Rscript benchmarks/r/prepare_case.R ", case_id,
      call. = FALSE
    )
  }
  utils::read.csv(path, check.names = FALSE, na.strings = c("", "NA"))
}
