#!/usr/bin/env Rscript
# Regenerate the textbook-corpus categorical (WLSMV) fixtures under
# cpp/tests/fixtures/textbook_ordinal/.
#
# Each fixture is one all-ordinal, covariate-free corpus case verified
# against its book's own output (Mplus .out or the author's lavaan call).
# It carries only derived statistics -- per block the sample thresholds,
# polychoric correlations, their asymptotic covariance (NACOV) and the DWLS
# weight -- plus lavaan's parameter table, starting values, estimates, fit
# and implied moments. No casewise rows are written.
#
# The fit is the corpus call (textbookcorpus's lavaan arguments, including
# model_options such as parameterization and lavaan_options), so the fixture
# is the book's model. Defined parameters (`:=`) are dropped from the
# exported parameter table: they do not affect estimation.
#
# Usage: Rscript cpp/tests/tools/regen_textbook_ordinal_fixtures.R
# Requires the optional corpus at external/textbook-corpus/ and its R
# package (textbookcorpus).

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
  library(textbookcorpus)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", "..", ".."))
corpus <- file.path(repo_root, "external", "textbook-corpus")
out_dir <- file.path(repo_root, "cpp", "tests", "fixtures", "textbook_ordinal")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# All-ordinal, covariate-free WLSMV cases of corpus v3.1.0. Cases with
# exogenous covariates (lavaan conditional.x) or continuous indicators are
# not exported: magmaan's ordinal estimators take neither.
case_ids <- c(
  "mplus_users_guide_v8_ch5_ex5_2",
  "mplus_users_guide_v8_ch5_ex5_10",
  "mplus_users_guide_v8_ch5_ex5_19",
  "mplus_users_guide_v8_ch5_ex5_22",
  "mplus_users_guide_v8_ch6_ex6_4",
  "mplus_users_guide_v8_ch6_ex6_5",
  "mplus_users_guide_v8_ch6_ex6_15",
  "newsom_2015_ex2_4a",
  "newsom_2015_ex2_8a",
  "newsom_2015_ex3_3a",
  "newsom_2015_ex3_3b",
  "newsom_2015_ex3_3c",
  "newsom_2015_ex7_2b",
  "newsom_2015_ex9_2"
)

`%||%` <- function(x, y) if (is.null(x)) y else x
plain <- function(x) unname(as.matrix(x))
rows <- function(d) lapply(seq_len(nrow(d)), function(i) as.list(d[i, , drop = FALSE]))
as_blocks <- function(x) if (is.list(x) && !is.null(names(x)) &&
                             any(c("cov", "th", "res.cov") %in% names(x))) list(x) else x

manifest <- utils::read.csv(file.path(corpus, "manifest.csv"), stringsAsFactors = FALSE)
written <- character()
for (id in case_ids) {
  row <- manifest[manifest$case_id == id, , drop = FALSE]
  if (nrow(row) != 1L) stop("case not in corpus manifest: ", id)
  case <- load_case(file.path(corpus, row$case_dir), root = corpus)
  meta <- case$meta
  if (!identical(meta$estimator_default, "WLSMV")) stop(id, ": not a WLSMV case")
  fargs <- textbookcorpus:::.lavaan_args(case, "WLSMV")
  fit <- suppressWarnings(do.call(lavaan::sem, fargs))
  stopifnot(lavInspect(fit, "converged"))
  ov <- lavNames(fit, "ov")
  if (length(lavNames(fit, "ov.x"))) stop(id, ": has exogenous covariates")
  if (!setequal(ov, lavNames(fit, "ov.ord"))) stop(id, ": not all-ordinal")

  pt <- parTable(fit)
  pt <- pt[pt$op != ":=", , drop = FALSE]
  # lavaan >= 0.7 may store group labels in parTable()$group; the C++
  # compatibility table takes 1-based group indices (0 for global rows).
  if (is.character(pt$group)) {
    gi <- match(pt$group, lavInspect(fit, "group.label"))
    pt$group <- ifelse(is.na(gi), 0L, gi)
  }
  pt$ustart <- ifelse(pt$free == 0L, pt$est, pt$start)
  free <- pt[pt$free > 0L, , drop = FALSE]
  free <- free[order(free$free), , drop = FALSE]

  ss <- as_blocks(lavInspect(fit, "sampstat"))
  gamma <- lavInspect(fit, "gamma"); if (is.matrix(gamma)) gamma <- list(gamma)
  wlsv <- lavInspect(fit, "wls.v"); if (is.matrix(wlsv)) wlsv <- list(wlsv)
  implied <- as_blocks(lavInspect(fit, "implied"))
  nobs <- as.integer(lavInspect(fit, "nobs"))
  blocks <- lapply(seq_along(ss), function(b) {
    th <- ss[[b]]$th
    th_ov <- sub("\\|.*$", "", names(th))
    list(n = nobs[b], R = plain(ss[[b]]$cov[ov, ov]), thresholds = unname(th),
         threshold_ov = match(th_ov, ov) - 1L,
         threshold_level = as.integer(sub("^.*\\|t", "", names(th))),
         n_levels = as.integer(table(factor(th_ov, levels = ov))) + 1L,
         NACOV = plain(gamma[[b]]), W = plain(wlsv[[b]]))
  })
  fm <- fitMeasures(fit, c("chisq", "df", "fmin", "npar", "chisq.scaled"))
  mo <- meta$model_options
  result <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind = "textbook_ordinal",
      tool = "cpp/tests/tools/regen_textbook_ordinal_fixtures.R",
      corpus_case = id,
      corpus_book = meta$book,
      corpus_tier = meta$fidelity_tier,
      lavaan_version = as.character(packageVersion("lavaan")),
      aggregate_only = TRUE,
      note = paste("Derived statistics only (thresholds, polychorics, NACOV,",
                   "DWLS weight); defined parameters dropped.")),
    id = id,
    parameterization = mo$parameterization %||% "delta",
    group_labels = as.list(lavInspect(fit, "group.label")),
    observed_variables = ov,
    partable = rows(pt[, c("id", "user", "lhs", "op", "rhs", "block", "group",
                           "free", "exo", "ustart", "label", "plabel")]),
    theta = unname(free$est), start = unname(free$start),
    blocks = blocks,
    implied = unname(lapply(implied, function(x) list(cov = plain(x$cov[ov, ov])))),
    fit = as.list(fm))
  path <- file.path(out_dir, paste0(id, ".json"))
  write_json(result, path, pretty = TRUE, auto_unbox = TRUE, na = "null",
             null = "null", digits = 16)
  kb <- file.size(path) / 1024
  if (kb > 900) stop(id, ": fixture is ", round(kb), " KB (1 MB file limit)")
  written <- c(written, id)
  cat(sprintf("%-40s %6.1f KB  chisq %.4f df %d\n", id, kb, fm[["chisq"]], as.integer(fm[["df"]])))
}
cat("Wrote", length(written), "textbook ordinal fixtures to", out_dir, "\n")
