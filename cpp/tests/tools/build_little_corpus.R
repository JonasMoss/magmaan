#!/usr/bin/env Rscript

# Build the ignored external/textbook-corpus/raw/little corpus from Little's
# (2013) LISREL companion files.
#
# Every unique .LS8 input under source/ is translated to lavaan syntax by
# lisrel_translate.R and verified against the .OUT file LISREL 8.80 wrote
# for it: the free/fixed/equality/CO pattern, every printed estimate, the
# input moments, df and the minimum-fit-function chi-square. Only verified
# translations are retained. Inputs that fail translation or verification
# stay source-only, with the reason in manifest.csv.
#
# Outputs under raw/little/: models/<id>.lav, models_lisrel/<id>.{LS8,OUT},
# data/ (LISREL's analysed moments or selected raw columns), verification/
# <id>.json (the evidence), manifest.csv (every input) and catalogue.csv
# (retained inputs).
#
# Usage: Rscript cpp/tests/tools/build_little_corpus.R [id-regex]

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
source(file.path(script_dir, "lisrel_translate.R"))

root <- Sys.getenv("LITTLE_ROOT", unset = "")
if (!nzchar(root)) {
  root <- file.path(repo_root, "external", "textbook-corpus", "raw", "little")
}
root <- normalizePath(root, mustWork = TRUE)
source_dir <- file.path(root, "source")
if (!dir.exists(source_dir)) {
  stop("expected the extracted archives under ", source_dir,
       " (see raw/little/README.md)", call. = FALSE)
}
only <- commandArgs(trailingOnly = TRUE)

# Documented per-input corrections. LISREL reads variables by position; a
# label override only renames a column.
label_overrides <- list(
  # LA repeats NegAFF1; column 14 of 14.AffGrade6.dat is NegAFF2 in every
  # other input that reads that file.
  ch10_fig13_pseudohigher_order = c(`14` = "NegAFF2")
)

sanitize_id <- function(x) {
  x <- tolower(sub("\\.[Ll][Ss]8$", "", basename(x)))
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_|_$", "", x)
}

sha256 <- function(path) digest::digest(file = path, algo = "sha256")

# The archives repeat whole chapter folders; duplicates must be identical.
paths <- list.files(source_dir, "\\.[Ll][Ss]8$", recursive = TRUE,
                    full.names = TRUE)
paths <- paths[order(nchar(paths), paths)]
ids <- vapply(paths, sanitize_id, character(1L), USE.NAMES = FALSE)
for (id in unique(ids[duplicated(ids)])) {
  h <- unique(vapply(paths[ids == id], sha256, character(1L)))
  if (length(h) != 1L) stop("differing inputs share the id ", id, call. = FALSE)
}
keep <- !duplicated(ids)
paths <- paths[keep]
ids <- ids[keep]
o <- order(ids)
paths <- paths[o]
ids <- ids[o]
if (length(only)) {
  sel <- grepl(only[[1L]], ids)
  paths <- paths[sel]
  ids <- ids[sel]
}

out_dirs <- c("models", "models_lisrel", "data", "verification")
for (d in out_dirs) dir.create(file.path(root, d), showWarnings = FALSE)
if (!length(only)) {
  # A full build replaces every generated file, so nothing stale survives.
  for (d in out_dirs) unlink(list.files(file.path(root, d), full.names = TRUE))
}

write_matrix <- function(m, path) {
  utils::write.csv(m, path, quote = FALSE)
}
write_vector <- function(x, path) {
  utils::write.csv(data.frame(name = names(x), value = unname(x)), path,
                   row.names = FALSE, quote = TRUE)
}

# Groups are named after their data files when those differ (Female/Male,
# Subj1..Subj5); otherwise G1, G2, ...
group_labels <- function(parsed) {
  src <- vapply(parsed$groups, function(g) {
    tools::file_path_sans_ext(g$files[1L] %||% "")
  }, character(1L))
  if (length(unique(src)) == length(src) && all(nzchar(src))) src
  else paste0("G", seq_along(parsed$groups))
}

family_of <- function(model) {
  be <- model$st[[1L]]$BE
  if (any(be$free) || any(be$val != 0)) "latent structural"
  else "latent measurement"
}

rows <- list()
for (k in seq_along(paths)) {
  path <- paths[[k]]
  id <- ids[[k]]
  message("little: ", id)
  rel_src <- sub(paste0("^", root, .Platform$file.sep), "", path)
  file.copy(path, file.path(root, "models_lisrel", paste0(id, ".LS8")),
            overwrite = TRUE)
  out_path <- {
    cand <- list.files(dirname(path), full.names = TRUE)
    hit <- cand[tolower(basename(cand)) ==
                  tolower(sub("\\.[Ll][Ss]8$", ".OUT", basename(path)))]
    if (length(hit)) hit[[1L]] else NA_character_
  }
  if (!is.na(out_path)) {
    file.copy(out_path, file.path(root, "models_lisrel", paste0(id, ".OUT")),
              overwrite = TRUE)
  }
  res <- tryCatch(lis_translate(path, labels = label_overrides[[id]]),
                  error = function(e) e)
  row <- list(
    id = id, name = gsub("_", " ", id), family = "latent measurement",
    provenance = "Little LISREL examples", source_input = rel_src,
    source_output = if (is.na(out_path)) "" else
      sub(paste0("^", root, .Platform$file.sep), "", out_path),
    source_data = "", data_kind = "source", measurement_kind = "continuous",
    observed_only = FALSE, generated_data = "", generated_model = "",
    generated_lisrel_model = file.path("models_lisrel", paste0(id, ".LS8")),
    generated_output = if (is.na(out_path)) "" else
      file.path("models_lisrel", paste0(id, ".OUT")),
    generated_script = "", generated_verification = "",
    group_var = "", group_labels = "", n_groups = NA_integer_,
    ordered = "", lavaan_function = "sem", estimator = "ML",
    meanstructure = FALSE, fixed_x = FALSE, nobs = "",
    lisrel_df = NA_integer_, lisrel_chisq = NA_real_,
    verified = FALSE, verification_start = "", strict_parity = FALSE,
    status = "source_only", note = "")
  if (inherits(res, "error")) {
    row$note <- paste("translation failed:", conditionMessage(res))
    rows[[length(rows) + 1L]] <- row
    next
  }
  v <- res$verify
  G <- res$model$G
  labels <- group_labels(res$parsed)
  row$family <- family_of(res$model)
  row$n_groups <- G
  row$meanstructure <- res$model$meanstructure
  row$nobs <- paste(res$ns, collapse = ";")
  row$group_labels <- if (G > 1L) paste(labels, collapse = ";") else ""

  model_rel <- file.path("models", paste0(id, ".lav"))
  writeLines(res$syntax, file.path(root, model_rel), useBytes = TRUE)
  row$generated_model <- model_rel

  kinds <- vapply(res$data, `[[`, "", "kind")
  if (all(kinds == "raw")) {
    x <- lapply(seq_len(G), function(g) {
      d <- as.data.frame(res$data[[g]]$raw)
      names(d) <- res$tr$obs
      if (G > 1L) d$group <- labels[[g]]
      d
    })
    data_rel <- file.path("data", paste0(id, ".csv"))
    utils::write.csv(do.call(rbind, x), file.path(root, data_rel),
                     row.names = FALSE)
    row$data_kind <- "raw"
    row$group_var <- if (G > 1L) "group" else ""
  } else {
    stem <- if (G == 1L) id else paste0(id, "_g", seq_len(G))
    for (g in seq_len(G)) {
      write_matrix(res$covs[[g]],
                   file.path(root, "data", paste0(stem[[g]], "_cov.csv")))
      if (res$model$meanstructure) {
        write_vector(res$means[[g]],
                     file.path(root, "data", paste0(stem[[g]], "_mean.csv")))
      }
    }
    data_rel <- paste(file.path("data", paste0(stem, "_cov.csv")),
                      collapse = ";")
    row$data_kind <- "summary"
  }
  row$generated_data <- data_rel
  row$source_data <- paste(unique(unlist(lapply(res$parsed$groups,
                                                  `[[`, "files"))),
                           collapse = ";")

  if (is.null(v)) {
    row$note <- "no LISREL output file to verify against"
    rows[[length(rows) + 1L]] <- row
    next
  }
  pe <- lavaan::parTable(res$fit)
  ver <- list(
    id = id,
    source_input = rel_src,
    source_input_sha256 = sha256(path),
    source_output = row$source_output,
    source_output_sha256 = sha256(out_path),
    verified = v$ok,
    issues = as.list(v$issues),
    likelihood = "wishart",
    convention = paste(
      "LISREL 8.80 analyses S with divisor N-1 and reports the minimum",
      "fit function chi-square (N-1)F_ML; the check refits the lavaan",
      "translation with likelihood = \"wishart\" on LISREL's analysed",
      "moments."),
    n_groups = G,
    n_obs = as.list(res$ns),
    printed_decimals = res$nd,
    df = v$df, lisrel_df = v$lisrel_df,
    chisq = v$chisq, lisrel_chisq = v$lisrel_chisq,
    max_abs_estimate_diff = v$max_est_diff,
    max_abs_moment_diff = v$max_moment_diff,
    n_free = v$n_free,
    unprinted_elements = v$unprinted,
    start = v$start,
    skipped_start_files = as.list(res$model$skipped_ma),
    ignored_lines = as.list(unlist(lapply(res$parsed$groups, `[[`, "ignored"))),
    label_overrides = as.list(label_overrides[[id]]),
    lisrel_warnings = as.list(v$lisrel_warnings),
    lisrel_estimates = if (is.null(v$printed)) list() else
      lapply(seq_len(nrow(v$printed)), function(i) as.list(v$printed[i, ])),
    wishart_estimates = lapply(which(pe$free > 0L), function(i) {
      list(lhs = pe$lhs[[i]], op = pe$op[[i]], rhs = pe$rhs[[i]],
           group = pe$group[[i]], label = pe$label[[i]], est = pe$est[[i]])
    })
  )
  ver_rel <- file.path("verification", paste0(id, ".json"))
  jsonlite::write_json(ver, file.path(root, ver_rel), auto_unbox = TRUE,
                       pretty = TRUE, digits = NA, null = "null")
  row$generated_verification <- ver_rel
  row$lisrel_df <- v$lisrel_df
  row$lisrel_chisq <- v$lisrel_chisq
  row$verified <- v$ok
  row$verification_start <- v$start
  if (v$ok) {
    row$status <- "retained"
    # magmaan has neither bound syntax nor inequality-constrained fits. The
    # parity fixture keeps models with at most 18 observed variables so it
    # stays under the repository's 1 MB file limit; wider models are verified
    # here and snapshotted in the corpus.
    has_bound <- length(res$model$ir) > 0L
    wide <- length(res$tr$obs) > 18L
    row$strict_parity <- G == 1L && !has_bound && !wide
    row$note <- paste0(
      "verified against LISREL output: pattern, ", v$n_free,
      " free parameters, all printed estimates, df and chi-square",
      if (v$start != "default") paste0(" (lavaan default start failed; ",
                                         "reached from ", v$start, ")") else "",
      if (G > 1L) "; multi-group, not in the single-group parity fixtures"
      else "",
      if (has_bound) paste0("; LISREL IR restriction written as a lavaan ",
                            "bound, which magmaan cannot parse") else "",
      if (wide && G == 1L) paste0("; ", length(res$tr$obs), " observed ",
                                  "variables, above the parity-fixture cap") else "")
  } else {
    row$status <- "unverified"
    row$note <- paste("verification failed:", paste(v$issues, collapse = "; "))
  }
  rows[[length(rows) + 1L]] <- row
}

manifest <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
if (length(only)) {
  message("partial build (", only[[1L]], "): manifest not written")
  print(manifest[, c("id", "status", "lisrel_df", "lisrel_chisq",
                     "verification_start")])
  quit(save = "no")
}
utils::write.csv(manifest, file.path(root, "manifest.csv"), row.names = FALSE,
                 na = "")
catalogue <- manifest[manifest$status == "retained", , drop = FALSE]
utils::write.csv(catalogue, file.path(root, "catalogue.csv"), row.names = FALSE,
                 na = "")
message("Wrote ", nrow(catalogue), " verified Little cases (", nrow(manifest),
        " inputs) to ", root)
