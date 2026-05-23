#!/usr/bin/env Rscript

# Scout public paper-repository candidates for a future magmaan paper corpus.
#
# This script is intentionally read-only. It inventories OSF metadata, scans
# small code files for lavaan-shaped usage, and writes a tracked manifest. Raw
# downloaded files live only in temp storage during the scan.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))
fixtures <- file.path(repo_root, "tests", "fixtures")
out_dir <- file.path(fixtures, "paper_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(x, y) if (is.null(x)) y else x

api_base <- Sys.getenv("PAPER_CORPUS_OSF_API",
                       unset = "https://api.osf.io/v2")
max_scan_bytes <- as.integer(Sys.getenv("PAPER_CORPUS_MAX_SCAN_BYTES",
                                        unset = "250000"))

seeds <- list(
  list(
    platform = "osf",
    node_id = "gduy4",
    role = "family_root",
    priority = "context",
    promotion_status = "scout_indexed",
    note = paste(
      "Diary-study root project. Useful for child discovery and shared-data",
      "relationships; not itself the first fixture target."
    )
  ),
  list(
    platform = "osf",
    node_id = "hwkem",
    role = "empirical_candidate",
    priority = "promote_first",
    promotion_status = "promote_first",
    note = paste(
      "Measurement-invariance tutorial child of gduy4. Expected to contain",
      "longitudinal CFA/SEM/RI-CLPM lavaan examples with a small abridged CSV."
    )
  ),
  list(
    platform = "osf",
    node_id = "m9wdn",
    role = "supporting_data",
    priority = "context",
    promotion_status = "supporting_data",
    note = "Diary-study data source for the gduy4 family; less directly model-rich."
  ),
  list(
    platform = "osf",
    node_id = "yez2b",
    role = "supporting_data",
    priority = "context",
    promotion_status = "supporting_data",
    note = "Psychometric data source in the gduy4 family; useful for overlap checks."
  ),
  list(
    platform = "osf",
    node_id = "zxqvn",
    role = "empirical_candidate",
    priority = "promote_first",
    promotion_status = "promote_first",
    note = paste(
      "Compact empirical mediation/SEM candidate with small RData. Clustered",
      "SE behavior should be catalogued separately from core point-estimate parity."
    )
  ),
  list(
    platform = "osf",
    node_id = "rntmc",
    role = "empirical_candidate",
    priority = "scan_first",
    promotion_status = "candidate_scan_first",
    note = "Candidate with code plus large CSV files; scan code before any data download."
  ),
  list(
    platform = "osf",
    node_id = "d6hs7",
    role = "methods_candidate",
    priority = "later",
    promotion_status = "methods_candidate",
    note = paste(
      "Methods/simulation-oriented lavaan project. Useful later, but not the",
      "first empirical-data seed."
    )
  )
)

safe_read_json_url <- function(url) {
  tryCatch(
    jsonlite::fromJSON(url, simplifyVector = FALSE),
    error = function(e) list(`_error` = conditionMessage(e), `_url` = url)
  )
}

read_pages <- function(url) {
  out <- list()
  next_url <- url
  while (!is.null(next_url) && nzchar(next_url)) {
    page <- safe_read_json_url(next_url)
    if (!is.null(page$`_error`)) {
      return(list(error = page$`_error`, url = page$`_url`, data = out))
    }
    rows <- page$data %||% list()
    out <- c(out, rows)
    next_url <- page$links[["next"]] %||% NULL
  }
  list(error = NULL, url = url, data = out)
}

field <- function(x, name, default = NULL) {
  value <- x[[name]]
  if (is.null(value)) default else value
}

hashes <- function(file) {
  extra <- file$attributes$extra %||% list()
  h <- extra$hashes %||% list()
  list(md5 = h$md5 %||% NULL, sha256 = h$sha256 %||% NULL)
}

file_extension <- function(name) {
  ext <- tolower(sub("^.*\\.([^.]+)$", "\\1", name))
  if (identical(ext, tolower(name))) "" else ext
}

file_class <- function(name, kind) {
  if (identical(kind, "folder")) return("folder")
  ext <- file_extension(name)
  if (ext %in% c("r", "rmd", "qmd", "sps", "txt", "html")) return("code")
  if (ext %in% c("csv", "tsv", "sav", "dta", "rdata", "rds", "dat")) return("data")
  if (ext %in% c("pdf", "doc", "docx", "tex", "bib")) return("paper_or_docs")
  "other"
}

size_class <- function(size) {
  if (is.null(size) || is.na(size)) return("unknown")
  if (size < 100000) return("small")
  if (size < 5000000) return("medium")
  "large"
}

download_text <- function(url) {
  tmp <- tempfile("magmaan-paper-corpus-")
  on.exit(unlink(tmp, force = TRUE), add = TRUE)
  ok <- tryCatch({
    utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
    TRUE
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (!ok) return(NULL)
  x <- readLines(tmp, warn = FALSE, encoding = "UTF-8")
  paste(iconv(x, from = "", to = "UTF-8", sub = "byte"), collapse = "\n")
}

count_regex <- function(pattern, text) {
  hits <- gregexpr(pattern, text, perl = TRUE, ignore.case = TRUE)[[1]]
  if (length(hits) == 1L && hits[[1L]] == -1L) 0L else length(hits)
}

unique_calls <- function(text) {
  calls <- c("sem", "cfa", "growth", "lavaan")
  calls[vapply(calls, function(x) {
    count_regex(paste0("\\b", x, "\\s*\\("), text) > 0L
  }, logical(1L))]
}

scan_text <- function(text) {
  data_loaders <- c("read.csv", "read.table", "read.delim", "read_sav",
                    "read_dta", "readRDS", "load", "fread")
  list(
    lavaan_library_count = count_regex("\\blibrary\\s*\\(\\s*lavaan\\s*\\)|\\brequire\\s*\\(\\s*lavaan\\s*\\)", text),
    lavaan_namespace_count = count_regex("\\blavaan\\s*::", text),
    lavaan_call_count = count_regex("\\b(sem|cfa|growth|lavaan)\\s*\\(", text),
    detected_lavaan_calls = unique_calls(text),
    model_operator_count = count_regex("=~|~~|:=", text),
    regression_operator_count = count_regex("[^<>=~]\\~[^=~]", text),
    data_loader_count = sum(vapply(data_loaders, function(x) {
      count_regex(paste0("\\b", gsub(".", "\\\\.", x, fixed = TRUE), "\\s*\\("),
                  text)
    }, integer(1L))),
    code_bytes_scanned = nchar(text, type = "bytes")
  )
}

inventory_file <- function(file, node_id) {
  attr <- file$attributes
  hs <- hashes(file)
  cls <- file_class(attr$name, attr$kind)
  list(
    osf_file_id = file$id,
    guid = attr$guid %||% NULL,
    name = attr$name,
    path = attr$materialized_path %||% attr$path %||% "",
    kind = attr$kind,
    provider = attr$provider %||% "",
    size = attr$size %||% NULL,
    size_class = size_class(attr$size %||% NA),
    file_class = cls,
    md5 = hs$md5,
    sha256 = hs$sha256,
    download_url = file$links$download %||% NULL,
    html_url = file$links$html %||% NULL,
    api_url = file$links$self %||% NULL,
    child_files_url = file$relationships$files$links$related$href %||% NULL,
    node_id = node_id
  )
}

list_files_recursive <- function(node_id, url = NULL, depth = 0L) {
  if (is.null(url)) {
    url <- paste0(api_base, "/nodes/", node_id, "/files/osfstorage/")
  }
  page <- read_pages(url)
  if (!is.null(page$error)) {
    return(list(files = list(), errors = list(list(url = url,
                                                   error = page$error))))
  }
  files <- list()
  errors <- list()
  for (row in page$data) {
    item <- inventory_file(row, node_id)
    files <- c(files, list(item))
    if (identical(item$kind, "folder") && !is.null(item$child_files_url) &&
        depth < 5L) {
      child <- list_files_recursive(node_id, item$child_files_url, depth + 1L)
      files <- c(files, child$files)
      errors <- c(errors, child$errors)
    }
  }
  list(files = files, errors = errors)
}

scan_file_if_small_code <- function(file) {
  if (!identical(file$file_class, "code")) {
    file$scan_status <- "not_code"
    return(file)
  }
  if (is.null(file$download_url)) {
    file$scan_status <- "no_download_url"
    return(file)
  }
  if (!is.null(file$size) && file$size > max_scan_bytes) {
    file$scan_status <- "skipped_large_code"
    return(file)
  }
  text <- download_text(file$download_url)
  if (is.null(text)) {
    file$scan_status <- "download_failed"
    return(file)
  }
  file$scan_status <- "scanned"
  file$signals <- scan_text(text)
  file
}

node_summary <- function(files, children, errors) {
  is_file <- vapply(files, function(x) identical(x$kind, "file"), logical(1L))
  classes <- vapply(files, function(x) x$file_class, character(1L))
  scan_status <- vapply(files, function(x) x$scan_status %||% "", character(1L))
  signal <- lapply(files, function(x) x$signals %||% list())
  lavaan_calls <- sum(vapply(signal, function(x) x$lavaan_call_count %||% 0L,
                             integer(1L)))
  model_ops <- sum(vapply(signal, function(x) x$model_operator_count %||% 0L,
                          integer(1L)))
  data_loaders <- sum(vapply(signal, function(x) x$data_loader_count %||% 0L,
                             integer(1L)))
  list(
    file_count = sum(is_file),
    folder_count = sum(!is_file),
    child_node_count = length(children),
    candidate_code_file_count = sum(classes == "code" & is_file),
    candidate_data_file_count = sum(classes == "data" & is_file),
    scanned_code_file_count = sum(scan_status == "scanned"),
    detected_lavaan_call_count = lavaan_calls,
    detected_model_operator_count = model_ops,
    detected_data_loader_count = data_loaders,
    error_count = length(errors)
  )
}

node_payload <- function(seed) {
  node_url <- paste0(api_base, "/nodes/", seed$node_id, "/")
  raw_node <- safe_read_json_url(node_url)
  if (!is.null(raw_node$`_error`)) {
    return(c(seed, list(
      url = paste0("https://osf.io/", seed$node_id, "/"),
      api_url = node_url,
      status = "api_error",
      error = raw_node$`_error`,
      children = list(),
      files = list(),
      summary = node_summary(list(), list(), list(raw_node$`_error`))
    )))
  }

  children_page <- read_pages(paste0(api_base, "/nodes/", seed$node_id,
                                     "/children/"))
  children <- lapply(children_page$data, function(child) {
    list(
      node_id = child$id,
      title = child$attributes$title %||% "",
      category = child$attributes$category %||% "",
      public = isTRUE(child$attributes$public),
      url = child$links$html %||% paste0("https://osf.io/", child$id, "/")
    )
  })

  file_listing <- list_files_recursive(seed$node_id)
  files <- lapply(file_listing$files, scan_file_if_small_code)
  errors <- file_listing$errors
  if (!is.null(children_page$error)) {
    errors <- c(errors, list(list(url = children_page$url,
                                  error = children_page$error)))
  }

  attr <- raw_node$data$attributes
  c(seed, list(
    url = raw_node$data$links$html %||% paste0("https://osf.io/", seed$node_id, "/"),
    api_url = node_url,
    title = attr$title %||% "",
    category = attr$category %||% "",
    public = isTRUE(attr$public),
    date_created = attr$date_created %||% NULL,
    date_modified = attr$date_modified %||% NULL,
    tags = attr$tags %||% list(),
    license = attr$node_license %||% list(),
    status = if (length(errors)) "scouted_with_errors" else "scouted",
    children = children,
    files = files,
    errors = errors,
    summary = node_summary(files, children, errors)
  ))
}

nodes <- lapply(seeds, node_payload)

count <- function(pred) sum(vapply(nodes, pred, logical(1L)))
sum_field <- function(name) {
  sum(vapply(nodes, function(x) x$summary[[name]] %||% 0L, integer(1L)))
}

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "paper_corpus.scout_manifest",
    corpus_id = "magmaan_paper_corpus_scout_v1",
    tool = "tests/tools/scout_paper_corpus.R",
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    source_note = "OSF API metadata plus small-code scans; raw downloads are not tracked.",
    max_scan_bytes = max_scan_bytes
  ),
  counts = list(
    seed_nodes = length(nodes),
    scouted = count(function(x) x$status %in% c("scouted", "scouted_with_errors")),
    promote_first = count(function(x) identical(x$promotion_status, "promote_first")),
    files = sum_field("file_count"),
    folders = sum_field("folder_count"),
    child_nodes = sum_field("child_node_count"),
    candidate_code_files = sum_field("candidate_code_file_count"),
    candidate_data_files = sum_field("candidate_data_file_count"),
    scanned_code_files = sum_field("scanned_code_file_count"),
    detected_lavaan_calls = sum_field("detected_lavaan_call_count"),
    detected_model_operators = sum_field("detected_model_operator_count"),
    detected_data_loaders = sum_field("detected_data_loader_count"),
    errors = sum_field("error_count")
  ),
  nodes = nodes
)

jsonlite::write_json(payload, file.path(out_dir, "scout_manifest.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)
cat("Wrote ", file.path(out_dir, "scout_manifest.json"), "\n", sep = "")
