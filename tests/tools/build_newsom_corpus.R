#!/usr/bin/env Rscript

# Build the ignored external/newsom corpus from the raw Newsom zip archives.
#
# The source zips contain lavaan R scripts plus shared raw .dat files. This
# script extracts every R example it can evaluate safely, writes one prepared
# CSV and one lavaan model per fit call, and records categorical/observed-only
# classification in catalogue.csv. Mplus files are intentionally not used as
# numerical oracles.

suppressPackageStartupMessages({
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

root <- Sys.getenv("NEWSOM_ROOT", unset = "")
if (!nzchar(root)) root <- file.path(repo_root, "external", "newsom")
root <- normalizePath(root, mustWork = TRUE)

source_dir <- file.path(root, "source")
script_dirs <- c(lsem2r = file.path(source_dir, "lsem2r"),
                 lsemr = file.path(source_dir, "lsemr"))
data_dir <- file.path(source_dir, "data")
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(script_dirs[["lsem2r"]], recursive = TRUE, showWarnings = FALSE)
dir.create(script_dirs[["lsemr"]], recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

zip_extract <- function(zip, exdir) {
  if (!file.exists(zip)) stop("Missing source archive: ", zip, call. = FALSE)
  utils::unzip(zip, exdir = exdir)
}
zip_extract(file.path(root, "lsem2r.zip"), script_dirs[["lsem2r"]])
zip_extract(file.path(root, "lsemr.zip"), script_dirs[["lsemr"]])
zip_extract(file.path(root, "lsemdata.zip"), data_dir)

out_dirs <- c("models", "data", "scripts", "goldens", "results")
for (d in out_dirs) dir.create(file.path(root, d), recursive = TRUE,
                               showWarnings = FALSE)

sanitize_id <- function(x) {
  x <- tolower(gsub("\\.[Rr]$", "", basename(x)))
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_|_$", "", x)
  x
}

call_name <- function(x) {
  if (!is.call(x)) return("")
  nm <- x[[1L]]
  if (is.symbol(nm)) as.character(nm) else ""
}

is_assignment <- function(x) is.call(x) && call_name(x) %in% c("<-", "=")

rhs_call_name <- function(x) {
  if (!is_assignment(x)) return("")
  rhs <- x[[3L]]
  if (is.call(rhs)) call_name(rhs) else ""
}

is_fit_assignment <- function(x) rhs_call_name(x) %in% c("sem", "cfa", "growth")

skip_top_call <- function(x) {
  nm <- call_name(x)
  nm %in% c("library", "require", "install.packages", "setwd", "cat",
            "summary", "lavTestLRT", "anova", "print", "plot")
}

eval_or_null <- function(expr, env) {
  tryCatch(eval(expr, env), error = function(e) NULL)
}

eval_data_column_assignment <- function(expr, env) {
  if (!is_assignment(expr)) return(FALSE)
  lhs <- expr[[2L]]
  if (!is.call(lhs) || !identical(call_name(lhs), "$") || length(lhs) < 3L) {
    return(FALSE)
  }
  data_name <- if (is.symbol(lhs[[2L]])) as.character(lhs[[2L]]) else ""
  col_name <- if (is.symbol(lhs[[3L]])) as.character(lhs[[3L]]) else {
    as.character(eval_or_null(lhs[[3L]], env))
  }
  if (!nzchar(data_name) || !nzchar(col_name) ||
      !exists(data_name, env, inherits = FALSE)) {
    return(FALSE)
  }
  dat <- get(data_name, env)
  if (!is.data.frame(dat)) return(FALSE)
  rhs <- expr[[3L]]
  val <- tryCatch(eval(rhs, env), error = function(e) {
    eval(rhs, list2env(as.list(dat), parent = env))
  })
  dat[[col_name]] <- val
  assign(data_name, dat, envir = env)
  assign(col_name, val, envir = env)
  TRUE
}

fit_calls <- function(exprs) {
  out <- list()
  for (expr in exprs) {
    if (!is_fit_assignment(expr)) next
    lhs <- as.character(expr[[2L]])
    rhs <- expr[[3L]]
    out[[length(out) + 1L]] <- list(lhs = lhs, call = rhs)
  }
  out
}

arg_value <- function(cl, name, pos = NULL, env = parent.frame(),
                      default = NULL) {
  args <- as.list(cl)[-1L]
  nms <- names(args)
  if (!is.null(nms) && name %in% nms) return(eval_or_null(args[[name]], env))
  if (!is.null(pos) && length(args) >= pos) return(eval_or_null(args[[pos]], env))
  default
}

arg_symbol <- function(cl, name, pos = NULL) {
  args <- as.list(cl)[-1L]
  nms <- names(args)
  x <- NULL
  if (!is.null(nms) && name %in% nms) x <- args[[name]]
  else if (!is.null(pos) && length(args) >= pos) x <- args[[pos]]
  if (is.symbol(x)) as.character(x) else ""
}

strip_model_comments <- function(model) {
  paste(sub("#.*$", "", strsplit(model, "\n", fixed = TRUE)[[1L]]),
        collapse = "\n")
}

model_vars <- function(model) {
  txt <- strip_model_comments(model)
  tokens <- regmatches(txt, gregexpr("[A-Za-z][A-Za-z0-9_.]*", txt,
                                    perl = TRUE))[[1L]]
  bad <- c("NA", "start", "exp", "log")
  unique(tokens[!tokens %in% bad])
}

observed_names <- function(model, ordered) {
  out <- tryCatch(lavaan::lavNames(lavaan::lavaanify(model,
                                                     auto.var = TRUE,
                                                     auto.fix.first = TRUE,
                                                     meanstructure = TRUE,
                                                     ordered = ordered),
                                  type = "ov"),
                  error = function(e) character())
  if (length(out)) out else model_vars(model)
}

write_text <- function(x, path) {
  con <- file(path, open = "wb")
  on.exit(close(con), add = TRUE)
  writeLines(x, con, useBytes = TRUE)
}

safe_eval_script <- function(path) {
  exprs <- parse(path, keep.source = FALSE)
  env <- new.env(parent = globalenv())
  env$read.table <- function(file, ...) {
    base <- basename(as.character(file))
    utils::read.table(file.path(data_dir, base), ...)
  }
  env$read.csv <- function(file, ...) {
    base <- basename(as.character(file))
    utils::read.csv(file.path(data_dir, base), ...)
  }
  for (expr in exprs) {
    if (is.call(expr) && identical(call_name(expr), "attach") &&
        length(expr) >= 2L) {
      obj <- eval_or_null(expr[[2L]], env)
      if (is.data.frame(obj)) list2env(as.list(obj), envir = env)
      next
    }
    if (skip_top_call(expr) || is_fit_assignment(expr)) next
    if (!is_assignment(expr)) next
    if (eval_data_column_assignment(expr, env)) next
    eval_or_null(expr, env)
  }
  list(exprs = exprs, env = env)
}

script_paths <- c(list.files(script_dirs[["lsem2r"]], "\\.[Rr]$",
                             full.names = TRUE),
                  list.files(script_dirs[["lsemr"]], "\\.[Rr]$",
                             full.names = TRUE))
script_paths <- script_paths[!duplicated(basename(script_paths))]

rows <- list()
for (path in script_paths) {
  message("newsom: ", basename(path))
  evaluated <- tryCatch(safe_eval_script(path), error = function(e) e)
  if (inherits(evaluated, "error")) {
    rows[[length(rows) + 1L]] <- data.frame(
      id = sanitize_id(path), name = basename(path), family = "unknown",
      provenance = "Newsom Longitudinal SEM R examples",
      source_input = sub(paste0("^", root, .Platform$file.sep), "", path),
      source_data = "", data_kind = "raw", measurement_kind = "unknown",
      observed_only = FALSE, generated_data = "", generated_model = "",
      generated_script = "", group_var = "", ordered = "",
      lavaan_function = "", estimator = "", meanstructure = TRUE,
      fixed_x = TRUE, strict_parity = FALSE, status = "parse_error",
      note = conditionMessage(evaluated), stringsAsFactors = FALSE)
    next
  }
  fits <- fit_calls(evaluated$exprs)
  if (!length(fits)) next
  base_id <- sanitize_id(path)
  source_rel <- sub(paste0("^", root, .Platform$file.sep), "", path)
  for (i in seq_along(fits)) {
    fit <- fits[[i]]
    cl <- fit$call
    fun <- call_name(cl)
    model_sym <- arg_symbol(cl, "model", 1L)
    data_sym <- arg_symbol(cl, "data", 2L)
    model <- if (nzchar(model_sym) && exists(model_sym, evaluated$env,
                                            inherits = FALSE)) {
      get(model_sym, evaluated$env)
    } else ""
    data <- if (nzchar(data_sym) && exists(data_sym, evaluated$env,
                                          inherits = FALSE)) {
      get(data_sym, evaluated$env)
    } else NULL
    if (!is.character(model) || length(model) != 1L || !is.data.frame(data)) {
      next
    }
    ordered <- arg_value(cl, "ordered", env = evaluated$env, default = character())
    if (is.null(ordered)) ordered <- character()
    ordered <- as.character(ordered)
    estimator <- arg_value(cl, "estimator", env = evaluated$env, default = "ML")
    if (is.null(estimator)) estimator <- "ML"
    estimator <- toupper(as.character(estimator)[1L])
    ov <- observed_names(model, ordered)
    ordered_present <- intersect(ordered, ov)
    measurement_kind <- if (!length(ordered_present)) {
      "continuous"
    } else if (length(setdiff(ov, ordered_present)) == 0L) {
      "ordinal"
    } else {
      "mixed"
    }
    observed_only <- !grepl("=~", strip_model_comments(model), fixed = TRUE)
    id <- if (length(fits) == 1L) base_id else paste(base_id, sanitize_id(fit$lhs), sep = "_")

    # A fixed-weight GLS/ULS/ADF fit needs a positive-definite listwise
    # complete-data covariance of the model's observed variables. Models whose
    # listwise covariance is degenerate -- e.g. FIML pattern-mixture models
    # that regress on dropout indicators which go near-constant among complete
    # cases -- are out of fixed-weight scope: record them but keep them out of
    # the runnable catalogue (no model/data files written).
    cov_ov <- intersect(ov, names(data))
    cc <- data[stats::complete.cases(data[, cov_ov, drop = FALSE]),
               cov_ov, drop = FALSE]
    cov_pd <- length(cov_ov) >= 1L && nrow(cc) > length(cov_ov) &&
      tryCatch({ chol(stats::cov(cc)); TRUE }, error = function(e) FALSE)
    if (!cov_pd) {
      rows[[length(rows) + 1L]] <- data.frame(
        id = id,
        name = paste("Newsom", gsub("_", ".", base_id), fit$lhs),
        family = if (observed_only) "observed model" else "latent SEM",
        provenance = "Newsom Longitudinal SEM R examples",
        source_input = source_rel, source_data = "", data_kind = "raw",
        measurement_kind = measurement_kind, observed_only = observed_only,
        generated_data = "", generated_model = "", generated_script = "",
        group_var = "", ordered = paste(ordered_present, collapse = ";"),
        lavaan_function = fun, estimator = estimator, meanstructure = TRUE,
        fixed_x = TRUE, strict_parity = FALSE,
        status = "degenerate_covariance",
        note = paste("listwise complete-data covariance not positive definite",
                     "-- FIML/missing-data model, out of fixed-weight scope"),
        stringsAsFactors = FALSE)
      next
    }
    model_rel <- file.path("models", paste0(id, ".lav"))
    data_rel <- file.path("data", paste0(id, ".csv"))
    script_rel <- file.path("scripts", paste0(id, ".R"))
    write_text(model, file.path(root, model_rel))
    utils::write.csv(data, file.path(root, data_rel), row.names = FALSE,
                     na = "", quote = TRUE)
    write_text(c("# Generated by tests/tools/build_newsom_corpus.R",
                 paste0("model <- ", paste(deparse(model), collapse = "")),
                 paste0("data <- read.csv('../", data_rel, "', check.names = FALSE)"),
                 paste0("fit <- ", fun, "(model, data = data",
                        if (length(ordered_present)) {
                          paste0(", ordered = c(",
                                 paste(shQuote(ordered_present), collapse = ", "),
                        ")")
                        } else "",
                        if (!identical(estimator, "ML")) {
                          paste0(", estimator = ", shQuote(estimator))
                        } else "",
                        ")")),
               file.path(root, script_rel))
    status <- "retained"
    magmaan_strict_ids <- c(
      "ex1_1a", "ex1_1b", "ex1_2c",
      "ex14_1", "ex14_1a", "ex14_1c",
      "ex2_1", "ex2_2a", "ex2_2b", "ex2_2c", "ex2_5b",
      "ex3_1", "ex3_1b", "ex3_1c", "ex3_4a",
      "ex4_1", "ex4_4",
      "ex5_5b",
      "ex7_6b",
      "ex1_1", "ex13_1"
    )
    strict <- identical(measurement_kind, "continuous") && id %in% magmaan_strict_ids
    note <- "extracted from lavaan R script"
    rows[[length(rows) + 1L]] <- data.frame(
      id = id,
      name = paste("Newsom", gsub("_", ".", base_id), fit$lhs),
      family = if (observed_only) "observed model" else "latent SEM",
      provenance = "Newsom Longitudinal SEM R examples",
      source_input = source_rel,
      source_data = "",
      data_kind = "raw",
      measurement_kind = measurement_kind,
      observed_only = observed_only,
      generated_data = data_rel,
      generated_model = model_rel,
      generated_script = script_rel,
      group_var = "",
      ordered = paste(ordered_present, collapse = ";"),
      lavaan_function = fun,
      estimator = estimator,
      meanstructure = TRUE,
      fixed_x = TRUE,
      strict_parity = strict,
      status = status,
      note = note,
      stringsAsFactors = FALSE)
  }
}

catalogue <- if (length(rows)) do.call(rbind, rows) else data.frame()
utils::write.csv(catalogue, file.path(root, "catalogue.csv"), row.names = FALSE,
                 na = "")
message("Wrote ", nrow(catalogue), " Newsom catalogue rows to ",
        file.path(root, "catalogue.csv"))
