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

# Path into the shared experiment harness (experiments/_support). The one
# sanctioned cross-experiment dependency; carries no SEM logic.
support_path <- function(...) {
  file.path(repo_root(), "experiments", "_support", ...)
}

# Path to the textbook-corpus real-data dependency. Pure path helper so
# experiments read the corpus directly instead of borrowing another leaf's data
# (tests/fixtures or a paper's bundle). The corpus is a private, optional mount
# (see corpus/README.md); call corpus_available() before reading it.
corpus_root <- function() {
  file.path(repo_root(), "corpus", "textbook-corpus")
}

# TRUE when the private textbook-corpus is mounted at corpus/textbook-corpus/.
# Corpus-dependent experiments and regenerators should skip cleanly when FALSE.
corpus_available <- function(root = corpus_root()) {
  dir.exists(root) && file.exists(file.path(root, "manifest.csv"))
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

cache_digest <- function(x) {
  path <- tempfile("magmaan-cache-key-")
  on.exit(unlink(path), add = TRUE)
  writeBin(serialize(x, NULL, version = 3L), path)
  unname(tools::md5sum(path))
}

git_scalar <- function(args, root = repo_root()) {
  out <- tryCatch(
    system2("git", c("-C", root, args), stdout = TRUE, stderr = TRUE),
    warning = function(e) character(),
    error = function(e) character()
  )
  if (!length(out) || !is.null(attr(out, "status"))) return("")
  out[[1L]]
}

git_dirty <- function(root = repo_root()) {
  out <- tryCatch(
    system2("git", c("-C", root, "status", "--porcelain"),
            stdout = TRUE, stderr = TRUE),
    warning = function(e) character(),
    error = function(e) character()
  )
  length(out) > 0L && is.null(attr(out, "status"))
}

magmaan_cache_ref <- function(root = repo_root(), package = "magmaan") {
  version <- if (requireNamespace(package, quietly = TRUE)) {
    as.character(utils::packageVersion(package))
  } else {
    ""
  }
  list(
    package = package,
    package_version = version,
    git_head = git_scalar(c("rev-parse", "HEAD"), root = root),
    git_dirty = git_dirty(root = root)
  )
}

calibration_cache_dir <- function(script = script_path(), create = FALSE) {
  out <- file.path(results_dir(script = script, create = create), "cache")
  if (isTRUE(create)) dir.create(out, recursive = TRUE, showWarnings = FALSE)
  out
}

calibration_cache_key <- function(population,
                                  generator,
                                  options = list(),
                                  extra = list(),
                                  ref = magmaan_cache_ref()) {
  payload <- list(
    schema = "magmaan_experiment_calibration_cache_key_v1",
    population = population,
    generator = generator,
    options = options,
    extra = extra,
    magmaan = ref
  )
  structure(
    list(hash = cache_digest(payload), payload = payload),
    class = "magmaan_calibration_cache_key"
  )
}

calibration_cache_hash <- function(key) {
  if (inherits(key, "magmaan_calibration_cache_key")) return(key$hash)
  if (is.character(key) && length(key) == 1L &&
      grepl("^[[:xdigit:]]{32}$", key)) {
    return(tolower(key))
  }
  stop("`key` must be a calibration cache key or a 32-character hash.",
       call. = FALSE)
}

calibration_cache_path <- function(key,
                                   cache_dir = calibration_cache_dir()) {
  file.path(cache_dir, paste0(calibration_cache_hash(key), ".rds"))
}

calibration_cache_exists <- function(key,
                                     cache_dir = calibration_cache_dir()) {
  file.exists(calibration_cache_path(key, cache_dir = cache_dir))
}

calibration_cache_write <- function(key,
                                    value,
                                    cache_dir = calibration_cache_dir(create = TRUE),
                                    metadata = list(),
                                    overwrite = FALSE) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  path <- calibration_cache_path(key, cache_dir = cache_dir)
  if (file.exists(path) && !isTRUE(overwrite)) {
    stop("cache entry already exists: ", path, call. = FALSE)
  }
  record <- list(
    schema = "magmaan_experiment_calibration_cache_v1",
    hash = calibration_cache_hash(key),
    key = key,
    metadata = c(
      list(created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
           R_version = R.version.string),
      metadata
    ),
    value = value
  )
  saveRDS(record, path, version = 3L)
  invisible(path)
}

calibration_cache_read <- function(key,
                                   cache_dir = calibration_cache_dir(),
                                   value_only = TRUE) {
  path <- calibration_cache_path(key, cache_dir = cache_dir)
  if (!file.exists(path)) stop("cache entry not found: ", path, call. = FALSE)
  record <- readRDS(path)
  if (!identical(record$schema, "magmaan_experiment_calibration_cache_v1")) {
    stop("cache entry has an unknown schema: ", path, call. = FALSE)
  }
  if (!identical(record$hash, calibration_cache_hash(key))) {
    stop("cache entry hash does not match requested key: ", path, call. = FALSE)
  }
  if (isTRUE(value_only)) record$value else record
}

calibration_cache_list <- function(cache_dir = calibration_cache_dir()) {
  if (!dir.exists(cache_dir)) {
    return(data.frame(hash = character(), path = character(),
                      created_at_utc = character(), generator = character(),
                      magmaan_version = character(), git_head = character(),
                      git_dirty = logical(), stringsAsFactors = FALSE))
  }
  paths <- list.files(cache_dir, pattern = "\\.rds$", full.names = TRUE)
  rows <- lapply(paths, function(path) {
    record <- tryCatch(readRDS(path), error = function(e) NULL)
    if (is.null(record) ||
        !identical(record$schema, "magmaan_experiment_calibration_cache_v1")) {
      return(data.frame(
        hash = sub("\\.rds$", "", basename(path)),
        path = path,
        created_at_utc = "",
        generator = "",
        magmaan_version = "",
        git_head = "",
        git_dirty = FALSE,
        stringsAsFactors = FALSE
      ))
    }
    payload <- record$key$payload %||% list()
    ref <- payload$magmaan %||% list()
    data.frame(
      hash = record$hash %||% sub("\\.rds$", "", basename(path)),
      path = path,
      created_at_utc = (record$metadata %||% list())$created_at_utc %||% "",
      generator = metadata_value(payload$generator),
      magmaan_version = ref$package_version %||% "",
      git_head = substr(ref$git_head %||% "", 1L, 12L),
      git_dirty = isTRUE(ref$git_dirty),
      stringsAsFactors = FALSE
    )
  })
  if (!length(rows)) {
    return(data.frame(hash = character(), path = character(),
                      created_at_utc = character(), generator = character(),
                      magmaan_version = character(), git_head = character(),
                      git_dirty = logical(), stringsAsFactors = FALSE))
  }
  do.call(rbind, rows)
}

calibration_cache_clear <- function(key = NULL,
                                    cache_dir = calibration_cache_dir(),
                                    all = FALSE) {
  if (is.null(key)) {
    if (!isTRUE(all)) {
      stop("supply `key` or set `all = TRUE` to clear the whole cache.",
           call. = FALSE)
    }
    paths <- if (dir.exists(cache_dir)) {
      list.files(cache_dir, pattern = "\\.rds$", full.names = TRUE)
    } else {
      character()
    }
  } else {
    keys <- if (inherits(key, "magmaan_calibration_cache_key")) {
      list(key)
    } else {
      as.list(key)
    }
    paths <- vapply(keys, calibration_cache_path, character(1),
                    cache_dir = cache_dir)
  }
  existing <- paths[file.exists(paths)]
  if (length(existing)) unlink(existing)
  invisible(existing)
}
