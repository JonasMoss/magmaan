#!/usr/bin/env Rscript

# Build the ignored external/textbook-corpus/raw/newsom corpus from Newsom's
# companion R scripts.
#
# Edition. Newsom's site ships two script archives with overlapping example
# numbers but different models: lsemr.zip (first edition, 2015, the corpus
# book `newsom_2015`) and lsem2r.zip (second edition, 2024). The editions are
# never mixed. NEWSOM_EDITION=1 (default) extracts the 2015 scripts;
# NEWSOM_EDITION=2 extracts the 2024 scripts into the same raw layout, for a
# separate book.
#
# Extraction. Each script is evaluated one top-level expression at a time in
# a sandbox whose lavaan front ends (sem/cfa/growth/lavaan) are wrapped, so
# every fit is captured when it is made: the model string and data frame
# actually passed, every option, and the fitted object. A fit becomes a
# retained case only if the call the corpus can express (function, model,
# data read back from the written CSV, estimator, meanstructure, fixed.x,
# ordered, missing, group) reproduces the author's fit: identical parTable
# (lhs/op/rhs/group/free/ustart/label/exo), estimation options, N, sample
# statistics and objective. Other fits are catalogued with a status and a
# note but get no model/data files. Standard-error-only options
# (information, se, test, bootstrap) are recorded in the note; they do not
# change the estimates. Mplus files are intentionally not used as numerical
# oracles.

suppressPackageStartupMessages({
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", "..", ".."))

root <- Sys.getenv("NEWSOM_ROOT", unset = "")
if (!nzchar(root)) {
  root <- file.path(repo_root, "external", "textbook-corpus", "raw", "newsom")
}
root <- normalizePath(root, mustWork = TRUE)

edition <- Sys.getenv("NEWSOM_EDITION", unset = "1")
if (!edition %in% c("1", "2")) stop("NEWSOM_EDITION must be 1 or 2")
edition_dir <- if (edition == "1") "lsemr" else "lsem2r"

source_dir <- file.path(root, "source")
script_dirs <- c(lsem2r = file.path(source_dir, "lsem2r"),
                 lsemr = file.path(source_dir, "lsemr"))
data_dir <- file.path(source_dir, "data")
for (d in c(script_dirs, data_dir))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

zip_extract <- function(zip, exdir) {
  if (!file.exists(zip)) stop("Missing source archive: ", zip, call. = FALSE)
  utils::unzip(zip, exdir = exdir)
}
zip_extract(file.path(root, "lsem2r.zip"), script_dirs[["lsem2r"]])
zip_extract(file.path(root, "lsemr.zip"), script_dirs[["lsemr"]])
zip_extract(file.path(root, "lsemdata.zip"), data_dir)

# Generated outputs are rebuilt from scratch so stale files never survive.
out_dirs <- c("models", "data", "scripts", "goldens", "results")
for (d in out_dirs) dir.create(file.path(root, d), recursive = TRUE,
                               showWarnings = FALSE)
unlink(Sys.glob(file.path(root, "models", "*.lav")))
unlink(Sys.glob(file.path(root, "data", "*.csv")))
unlink(Sys.glob(file.path(root, "scripts", "*.R")))

# --- documented source patches ----------------------------------------------
# Applied to the script text before evaluation; the case keeps the verbatim
# upstream script as source/original.R and its note names the patch. Only
# unambiguous defects that stop the author's script from running are patched.
SOURCE_PATCHES <- list(
  list(edition = "1", script = "ex11-4.R",
       from = "0*ksi9 ++ 0*ksi10", to = "0*ksi9 + 0*ksi10",
       why = "doubled '+' in the author's model string is rejected by lavaan"),
  list(edition = "1", script = "ex11-5.R",
       from = "0*ksi9 ++ 0*ksi10", to = "0*ksi9 + 0*ksi10",
       why = "doubled '+' in the author's model string is rejected by lavaan"),
  list(edition = "1", script = "ex11-6.R",
       from = "0*ksi9 ++ 0*ksi10", to = "0*ksi9 + 0*ksi10",
       why = "doubled '+' in the author's model string is rejected by lavaan"),
  list(edition = "1", script = "ex5-5b.R",
       from = "socex1.1 <- read.table (\"socex1.dat\", header=FALSE)",
       to = "health1 <- read.table (\"health.dat\", header=FALSE)",
       why = paste("script names and fits `health1` but reads socex1.dat;",
                   "the 37 health.dat column names follow, and the",
                   "second-edition ex5-5b.R fits the same model to",
                   "health.dat"))
)

patched_script <- function(path) {
  hits <- Filter(function(p) p$edition == edition &&
                   p$script == basename(path), SOURCE_PATCHES)
  if (!length(hits)) return(list(path = path, note = ""))
  txt <- readLines(path, warn = FALSE)
  notes <- character()
  for (p in hits) {
    n <- sum(grepl(p$from, txt, fixed = TRUE))
    if (n == 0L) stop("source patch no longer applies to ", basename(path),
                      ": ", p$from, call. = FALSE)
    txt <- gsub(p$from, p$to, txt, fixed = TRUE)
    notes <- c(notes, sprintf("source patch '%s' -> '%s' (%s)", p$from, p$to,
                              p$why))
  }
  tmp <- tempfile(fileext = ".R")
  writeLines(txt, tmp)
  list(path = tmp, note = paste(notes, collapse = "; "))
}

# --- sandbox ----------------------------------------------------------------

fit_names <- c("sem", "cfa", "growth", "lavaan")

sandbox_run <- function(path) {
  records <- list()
  cur_expr <- NA_integer_
  redirect <- function(file) {
    if (!is.character(file)) return(file)
    cand <- file.path(data_dir, basename(file))
    if (file.exists(cand)) cand else file
  }
  make_wrapper <- function(fname) {
    real <- getExportedValue("lavaan", fname)
    force(real)
    function(...) {
      call <- sys.call()
      pf <- parent.frame()
      mc <- match.call(real, call, expand.dots = TRUE)
      argvals <- lapply(as.list(mc)[-1L], function(a) eval(a, pf))
      warns <- character()
      fit <- tryCatch(
        withCallingHandlers(real(...), warning = function(w) {
          warns <<- c(warns, conditionMessage(w))
          invokeRestart("muffleWarning")
        }),
        error = function(e) e)
      records[[length(records) + 1L]] <<- list(
        fun = fname, expr_index = cur_expr, lhs = NA_character_,
        args = argvals, fit = fit, warnings = warns)
      if (inherits(fit, "error")) stop(fit)
      fit
    }
  }
  wrappers <- lapply(stats::setNames(fit_names, fit_names), make_wrapper)
  env <- new.env(parent = globalenv())
  for (nm in names(wrappers)) assign(nm, wrappers[[nm]], envir = env)
  env$`::` <- function(pkg, name) {
    pkg <- as.character(substitute(pkg)); name <- as.character(substitute(name))
    if (identical(pkg, "lavaan") && name %in% names(wrappers))
      return(wrappers[[name]])
    getExportedValue(pkg, name)
  }
  env$setwd <- function(dir) invisible(getwd())
  env$getActiveDocumentContext <- function(...) list(path = path)
  env$install.packages <- function(...) invisible(NULL)
  env$library <- function(package, ..., character.only = FALSE) {
    p <- if (character.only) package else as.character(substitute(package))
    if (requireNamespace(p, quietly = TRUE))
      suppressPackageStartupMessages(base::library(p, character.only = TRUE))
    invisible(NULL)
  }
  env$read.table <- function(file, ...) utils::read.table(redirect(file), ...)
  env$read.csv <- function(file, ...) utils::read.csv(redirect(file), ...)
  for (nm in c("View", "plot", "dev.off", "pdf", "png"))
    assign(nm, function(...) invisible(NULL), envir = env)
  search_before <- search()
  on.exit({
    for (s in setdiff(search(), search_before))
      try(detach(s, character.only = TRUE), silent = TRUE)
  }, add = TRUE)

  exprs <- parse(path, keep.source = FALSE)
  errors <- character()
  for (i in seq_along(exprs)) {
    e <- exprs[[i]]
    cur_expr <- i
    n_before <- length(records)
    res <- tryCatch({
      utils::capture.output(suppressWarnings(suppressMessages(
        eval(e, envir = env))))
      NULL
    }, error = function(err) conditionMessage(err))
    if (!is.null(res)) errors <- c(errors, sprintf("expr %d: %s", i, res))
    n_after <- length(records)
    if (n_after > n_before && is.call(e) && is.symbol(e[[1L]]) &&
        as.character(e[[1L]]) %in% c("<-", "=") && is.symbol(e[[2L]])) {
      records[[n_after]]$lhs <- as.character(e[[2L]])
    }
  }
  list(records = records, errors = errors)
}

# --- representability check ------------------------------------------------

pt_rows <- function(fit) {
  pt <- lavaan::parTable(fit)
  data.frame(lhs = pt$lhs, op = pt$op, rhs = pt$rhs, group = pt$group,
             free = as.integer(pt$free > 0L), ustart = pt$ustart,
             label = pt$label, exo = pt$exo, stringsAsFactors = FALSE)
}

same_partable <- function(a, b) {
  x <- pt_rows(a); y <- pt_rows(b)
  if (!identical(dim(x), dim(y))) return(FALSE)
  for (k in names(x)) {
    u <- x[[k]]; v <- y[[k]]
    if (is.numeric(u)) {
      if (!identical(is.na(u), is.na(v)) ||
          any(u[!is.na(u)] != v[!is.na(v)])) return(FALSE)
    } else if (!identical(as.character(u), as.character(v))) return(FALSE)
  }
  TRUE
}

# Options that change the model or the estimates. information/se/test/
# bootstrap only change standard errors and test statistics.
estimation_options <- c(
  "estimator", "meanstructure", "fixed.x", "conditional.x", "missing",
  "parameterization", "std.lv", "orthogonal", "group.equal",
  "group.partial", "likelihood", "int.ov.free", "int.lv.free",
  "auto.fix.first", "auto.fix.single", "auto.var", "auto.cov.lv.x",
  "auto.cov.y", "auto.th", "auto.delta", "effect.coding", "ceq.simple",
  "correlation", "std.ov", "sample.cov.rescale", "model.type", "mimic",
  "representation", "zero.add", "zero.keep.margins")

option_diffs <- function(a, b) {
  oa <- lavaan::lavInspect(a, "options"); ob <- lavaan::lavInspect(b, "options")
  d <- character()
  for (k in estimation_options) {
    if (!identical(oa[[k]], ob[[k]]))
      d <- c(d, sprintf("%s=%s (corpus call gives %s)", k,
                        paste(format(oa[[k]]), collapse = "/"),
                        paste(format(ob[[k]]), collapse = "/")))
  }
  d
}

sampstat_gap <- function(a, b) {
  sa <- lavaan::lavInspect(a, "sampstat"); sb <- lavaan::lavInspect(b, "sampstat")
  if (!is.null(sa$cov)) { sa <- list(sa); sb <- list(sb) }
  if (length(sa) != length(sb)) return(Inf)
  out <- 0
  for (g in seq_along(sa)) for (k in names(sa[[g]])) {
    u <- sa[[g]][[k]]; v <- sb[[g]][[k]]
    if (is.null(v) || length(u) != length(v)) return(Inf)
    out <- max(out, abs(as.numeric(u) - as.numeric(v)))
  }
  out
}

corpus_call <- function(cand, data) {
  a <- list(model = cand$model, data = data, estimator = cand$estimator,
            meanstructure = cand$meanstructure, fixed.x = cand$fixed_x)
  if (length(cand$ordered)) {
    a$ordered <- cand$ordered
    a$parameterization <- cand$parameterization
  }
  if (!identical(cand$missing, "none")) a$missing <- cand$missing
  if (nzchar(cand$group_var)) a$group <- cand$group_var
  if (length(cand$group_equal)) a$group.equal <- cand$group_equal
  if (length(cand$group_partial)) a$group.partial <- cand$group_partial
  a
}

check_representable <- function(fit, cand, data_csv) {
  data <- utils::read.csv(data_csv, stringsAsFactors = FALSE,
                          check.names = FALSE)
  refit <- tryCatch(suppressWarnings(do.call(
    getExportedValue("lavaan", cand$fun), corpus_call(cand, data))),
    error = function(e) e)
  if (inherits(refit, "error"))
    return(list(problems = paste("corpus-style call fails:",
                                 conditionMessage(refit)), se_note = ""))
  problems <- option_diffs(fit, refit)
  if (!same_partable(fit, refit)) problems <- c(problems, "parTable differs")
  if (!identical(as.integer(lavaan::lavInspect(fit, "nobs")),
                 as.integer(lavaan::lavInspect(refit, "nobs"))))
    problems <- c(problems, "N differs")
  gap <- sampstat_gap(fit, refit)
  if (!is.finite(gap) || gap > 1e-10)
    problems <- c(problems, sprintf("sample statistics differ (%.3g)", gap))
  if (isTRUE(lavaan::lavInspect(fit, "converged"))) {
    fa <- lavaan::lavInspect(fit, "optim")$fx
    fb <- lavaan::lavInspect(refit, "optim")$fx
    if (abs(fa - fb) > 1e-8 * max(1, abs(fa)))
      problems <- c(problems, sprintf("objective differs (%.10g vs %.10g)",
                                      fa, fb))
  }
  list(problems = paste(problems, collapse = "; "),
       se_note = se_option_note(fit, refit))
}

# Standard-error/test options the corpus loader does not pass. They leave
# the estimates and the ML chi-square unchanged but change the SEs that
# ingest/snapshot_lavaan.R records, so the case note names them.
se_option_note <- function(a, b) {
  oa <- lavaan::lavInspect(a, "options"); ob <- lavaan::lavInspect(b, "options")
  keys <- c("information", "observed.information", "h1.information", "se",
            "test", "bootstrap")
  d <- keys[!vapply(keys, function(k) identical(oa[[k]], ob[[k]]),
                    logical(1L))]
  if (!length(d)) return("")
  paste0("author's call also sets SE/test options the corpus loader does ",
         "not pass (", paste(sprintf("%s=%s", d, vapply(d, function(k)
           paste(format(oa[[k]]), collapse = "/"), "")), collapse = ", "),
         "); estimates and chi-square are unaffected, snapshot SEs use ",
         "lavaan's default")
}

# --- helpers ----------------------------------------------------------------

sanitize_id <- function(x) {
  x <- tolower(gsub("\\.[Rr]$", "", basename(x)))
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_|_$", "", x)
}

strip_model_comments <- function(model) {
  paste(sub("#.*$", "", strsplit(model, "\n", fixed = TRUE)[[1L]]),
        collapse = "\n")
}

write_text <- function(x, path) {
  con <- file(path, open = "wb")
  on.exit(close(con), add = TRUE)
  writeLines(x, con, useBytes = TRUE)
}

bool_chr <- function(x) if (isTRUE(x)) "TRUE" else "FALSE"

# Hand-picked ids for magmaan's strict lavaan-parity fixtures. An id is
# strict only if its retained fit is also something
# regen_little_newsom_fixtures.R reproduces (continuous, ML, single group,
# complete or listwise data, converged); ids absent from an edition are
# ignored.
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

empty_row <- function(id, name, source_rel, status, note, extra = list()) {
  base <- list(
    id = id, name = name, family = "unknown",
    provenance = "Newsom Longitudinal SEM R examples",
    edition = edition, source_input = source_rel, source_data = "",
    fit_object = "", fit_occurrence = NA_integer_, data_kind = "raw",
    measurement_kind = "unknown", observed_only = FALSE,
    generated_data = "", generated_model = "", generated_script = "",
    group_var = "", group_levels = "", n_groups = NA_integer_, n_obs = "",
    ordered = "", parameterization = "", group_equal = "", group_partial = "",
    lavaan_function = "", estimator = "",
    meanstructure = NA, fixed_x = NA, missing = "",
    source_converged = NA, strict_parity = FALSE, status = status,
    note = note)
  for (k in names(extra)) base[[k]] <- extra[[k]]
  as.data.frame(base, stringsAsFactors = FALSE)
}

# --- extraction -------------------------------------------------------------

script_paths <- sort(list.files(script_dirs[[edition_dir]], "\\.[Rr]$",
                                full.names = TRUE))
rows <- list()
for (path in script_paths) {
  message("newsom (edition ", edition, "): ", basename(path))
  source_rel <- sub(paste0("^", root, .Platform$file.sep), "", path)
  base_id <- sanitize_id(path)
  patch <- patched_script(path)
  run <- tryCatch(sandbox_run(patch$path), error = function(e) e)
  if (inherits(run, "error")) {
    rows[[length(rows) + 1L]] <- empty_row(
      base_id, basename(path), source_rel, "parse_error",
      conditionMessage(run))
    next
  }
  recs <- run$records
  if (!length(recs)) {
    rows[[length(rows) + 1L]] <- empty_row(
      base_id, basename(path), source_rel, "no_fit",
      paste(c("script makes no lavaan fit", head(run$errors, 3)),
            collapse = " | "))
    next
  }
  lhs <- vapply(recs, function(r) if (is.na(r$lhs)) "fit" else r$lhs, "")
  for (i in seq_along(recs)) {
    rec <- recs[[i]]
    occ <- sum(lhs[seq_len(i)] == lhs[i])
    id <- if (length(recs) == 1L) base_id else
      paste(base_id, sanitize_id(lhs[i]), sep = "_")
    if (sum(lhs == lhs[i]) > 1L) id <- paste0(id, "_", occ)
    name <- paste("Newsom", gsub("_", ".", base_id), lhs[i])
    notes <- if (nzchar(patch$note)) patch$note else character()
    extra <- list(fit_object = lhs[i], fit_occurrence = occ,
                  lavaan_function = rec$fun)
    fit <- rec$fit
    if (inherits(fit, "error")) {
      rows[[length(rows) + 1L]] <- empty_row(
        id, name, source_rel, "source_error",
        paste(c(paste("author's fit call fails:",
                      gsub("\\s+", " ", conditionMessage(fit))), notes),
              collapse = "; "), extra)
      next
    }
    model <- rec$args$model
    data <- rec$args$data
    if (!is.character(model) || length(model) != 1L || !is.data.frame(data)) {
      rows[[length(rows) + 1L]] <- empty_row(
        id, name, source_rel, "unsupported_call",
        "fit call does not pass one model string and a data frame", extra)
      next
    }

    opts <- lavaan::lavInspect(fit, "options")
    ov <- lavaan::lavNames(fit, type = "ov")
    ov_ord <- lavaan::lavNames(fit, type = "ov.ord")
    measurement_kind <- if (!length(ov_ord)) "continuous" else
      if (length(setdiff(ov, ov_ord)) == 0L) "ordinal" else "mixed"
    estimator <- if (is.null(rec$args$estimator)) "ML" else
      toupper(as.character(rec$args$estimator)[1L])
    group_var <- if (is.null(rec$args$group)) "" else
      as.character(rec$args$group)
    nobs <- as.integer(lavaan::lavInspect(fit, "nobs"))
    has_na <- anyNA(data[, intersect(c(ov, group_var), names(data)),
                         drop = FALSE])
    missing <- switch(opts$missing,
                      ml = "fiml", ml.x = "fiml", pairwise = "pairwise",
                      listwise = if (has_na) "listwise" else "none",
                      opts$missing)
    # group.equal/group.partial only constrain multi-group fits
    multi <- nzchar(group_var)
    cand <- list(fun = rec$fun, model = model, estimator = estimator,
                 meanstructure = isTRUE(opts$meanstructure),
                 fixed_x = isTRUE(opts$fixed.x), ordered = ov_ord,
                 parameterization = if (length(ov_ord)) opts$parameterization else "",
                 missing = missing, group_var = group_var,
                 group_equal = if (multi) as.character(opts$group.equal) else character(),
                 group_partial = if (multi) as.character(opts$group.partial) else character())
    observed_only <- !grepl("=~", strip_model_comments(model), fixed = TRUE)
    extra <- c(extra, list(
      family = if (observed_only) "observed model" else "latent SEM",
      measurement_kind = measurement_kind, observed_only = observed_only,
      group_var = group_var,
      group_levels = if (nzchar(group_var)) paste(
        lavaan::lavInspect(fit, "group.label"), collapse = ";") else "",
      n_groups = as.integer(lavaan::lavInspect(fit, "ngroups")),
      n_obs = paste(nobs, collapse = ";"),
      ordered = paste(ov_ord, collapse = ";"),
      parameterization = cand$parameterization,
      group_equal = paste(cand$group_equal, collapse = ";"),
      group_partial = paste(cand$group_partial, collapse = ";"),
      estimator = estimator,
      meanstructure = cand$meanstructure, fixed_x = cand$fixed_x,
      missing = missing,
      source_converged = isTRUE(lavaan::lavInspect(fit, "converged"))))

    # A fixed-weight GLS/ULS/ADF fit needs a positive-definite listwise
    # complete-data covariance of the model's observed variables; models
    # whose listwise covariance is degenerate (FIML pattern-mixture models
    # regressing on near-constant dropout indicators) stay out of scope.
    cov_ov <- intersect(ov, names(data))
    cc <- data[stats::complete.cases(data[, cov_ov, drop = FALSE]), cov_ov,
               drop = FALSE]
    cov_pd <- length(cov_ov) >= 1L && nrow(cc) > length(cov_ov) &&
      tryCatch({ chol(stats::cov(cc)); TRUE }, error = function(e) FALSE)
    if (!cov_pd) {
      rows[[length(rows) + 1L]] <- empty_row(
        id, name, source_rel, "degenerate_covariance",
        paste(c(paste("listwise complete-data covariance not positive",
                      "definite -- FIML/missing-data model, out of",
                      "fixed-weight scope"), notes), collapse = "; "), extra)
      next
    }

    model_rel <- file.path("models", paste0(id, ".lav"))
    data_rel <- file.path("data", paste0(id, ".csv"))
    script_rel <- file.path("scripts", paste0(id, ".R"))
    data_path <- file.path(root, data_rel)
    utils::write.csv(data, data_path, row.names = FALSE, na = "",
                     quote = TRUE)
    chk <- check_representable(fit, cand, data_path)
    problems <- chk$problems
    if (nzchar(chk$se_note)) notes <- c(notes, chk$se_note)
    if (nzchar(problems)) {
      unlink(data_path)
      rows[[length(rows) + 1L]] <- empty_row(
        id, name, source_rel, "unrepresentable",
        paste(c(paste("corpus call cannot reproduce the author's fit:",
                      problems), notes), collapse = "; "), extra)
      next
    }
    write_text(model, file.path(root, model_rel))
    call_args <- c(
      "model", sprintf("data = read.csv('../%s', check.names = FALSE)",
                       data_rel),
      sprintf("estimator = %s", shQuote(estimator)),
      sprintf("meanstructure = %s", bool_chr(cand$meanstructure)),
      sprintf("fixed.x = %s", bool_chr(cand$fixed_x)),
      if (length(ov_ord)) sprintf("ordered = c(%s)",
                                  paste(shQuote(ov_ord), collapse = ", ")),
      if (!identical(missing, "none")) sprintf("missing = %s",
                                               shQuote(missing)),
      if (nzchar(group_var)) sprintf("group = %s", shQuote(group_var)),
      if (length(ov_ord)) sprintf("parameterization = %s",
                                  shQuote(cand$parameterization)),
      if (length(cand$group_equal)) sprintf("group.equal = c(%s)",
        paste(shQuote(cand$group_equal), collapse = ", ")),
      if (length(cand$group_partial)) sprintf("group.partial = c(%s)",
        paste(shQuote(cand$group_partial), collapse = ", ")))
    write_text(c("# Generated by cpp/tests/tools/build_newsom_corpus.R",
                 paste0("model <- ", paste(deparse(model), collapse = "")),
                 paste0("fit <- lavaan::", rec$fun, "(",
                        paste(call_args, collapse = ", "), ")")),
               file.path(root, script_rel))
    strict <- id %in% magmaan_strict_ids &&
      identical(measurement_kind, "continuous") &&
      identical(estimator, "ML") && missing %in% c("none", "listwise") &&
      !nzchar(group_var) && isTRUE(extra$source_converged)
    rows[[length(rows) + 1L]] <- empty_row(
      id, name, source_rel, "retained",
      paste(c("extracted from the author's lavaan fit call", notes,
              if (length(run$errors)) paste(
                "non-fit script expressions that fail here:",
                length(run$errors))), collapse = "; "),
      c(extra, list(generated_data = data_rel, generated_model = model_rel,
                    generated_script = script_rel, strict_parity = strict)))
  }
}

catalogue <- if (length(rows)) do.call(rbind, rows) else data.frame()
utils::write.csv(catalogue, file.path(root, "catalogue.csv"), row.names = FALSE,
                 na = "")
message("Wrote ", nrow(catalogue), " Newsom catalogue rows (edition ",
        edition, ", ", sum(catalogue$status == "retained"), " retained) to ",
        file.path(root, "catalogue.csv"))
