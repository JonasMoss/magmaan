#!/usr/bin/env Rscript

# Re-run only the audit-parity-v7 disagreement cells against the current
# magmaan-core. This is the cheap "what would v8 look like?" companion to the
# full lavaan audit-parity sweep.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "refit_disagreements.R" else ofile,
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

paper_dir <- file.path(repo_root(), "papers", "snlls-constrained")
pkg_dir <- file.path(paper_dir, "r-package")
require_pkg("pkgload")
pkgload::load_all(pkg_dir, quiet = TRUE)

`%||%` <- function(a, b) if (is.null(a)) b else a

canon_keys <- function(df) {
  swap <- df$op == "~~" & df$lhs > df$rhs
  if (any(swap)) {
    lhs <- df$lhs[swap]
    df$lhs[swap] <- df$rhs[swap]
    df$rhs[swap] <- lhs
  }
  df
}

empty_row <- function(case, error = NA_character_) {
  data.frame(
    case_id = case$id, book = case$corpus, family = case$family,
    weight = case$estimator,
    n_free_mag = NA_integer_, n_free_lav = NA_integer_,
    n_matched = NA_integer_,
    lavaan_converged = NA, lavaan_fx = NA_real_,
    lavaan_fx_bare = NA_real_,
    magmaan_fmin = NA_real_, magmaan_iter = NA_integer_,
    audit_advisory = NA_character_, audit_grad_inf = NA_real_,
    audit_stationary = NA,
    error = error, stringsAsFactors = FALSE
  )
}

audit_bessel <- function(case, n_total) {
  if (is.null(n_total) || !is.finite(n_total) || n_total <= 1) return(1)
  if (toupper(case$estimator) %in% c("GLS", "WLS", "ADF", "ULS", "DWLS")) {
    (n_total - 1) / n_total
  } else {
    1
  }
}

audit_one <- function(case) {
  lav <- tryCatch(snlls_lavaan_estimates(case, check_gradient = TRUE),
                  error = function(e) structure(
                    list(error = conditionMessage(e)),
                    class = "audit_parity_lavaan_error"))
  if (inherits(lav, "audit_parity_lavaan_error")) {
    return(empty_row(case, error = paste0("lavaan: ", lav$error)))
  }

  prob <- tryCatch(snlls_make_problem(case),
                   error = function(e) structure(
                     list(error = conditionMessage(e)),
                     class = "audit_parity_magmaan_error"))
  if (inherits(prob, "audit_parity_magmaan_error")) {
    out <- empty_row(case, error = paste0("magmaan-prep: ", prob$error))
    out$lavaan_converged <- lav$converged
    out$lavaan_fx <- lav$objective
    out$n_free_lav <- nrow(lav$estimates)
    return(out)
  }

  pt <- as.data.frame(prob$spec$partable)
  pt$row_idx <- seq_len(nrow(pt))
  free_pt <- pt[pt$free > 0L, , drop = FALSE]
  if (!"group" %in% names(free_pt)) free_pt$group <- 1L
  free_pt <- canon_keys(free_pt)
  lav_est <- canon_keys(lav$estimates)
  matched <- merge(free_pt[, c("row_idx", "lhs", "op", "rhs", "group")],
                   lav_est, by = c("lhs", "op", "rhs", "group"),
                   sort = FALSE)
  prob$spec$partable$ustart[matched$row_idx] <- matched$est

  ss <- magmaan:::sample_stats_arg(prob$dat)
  pt_for_starts <- magmaan:::partable_arg(prob$spec)
  theta_full <- tryCatch(
    magmaan::magmaan_core$fit_start_values(pt_for_starts, ss),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "audit_parity_magmaan_error"))
  if (inherits(theta_full, "audit_parity_magmaan_error")) {
    out <- empty_row(case, error = paste0("magmaan-starts: ", theta_full$error))
    out$lavaan_converged <- lav$converged
    out$lavaan_fx <- lav$objective
    out$n_free_lav <- nrow(lav$estimates)
    out$n_free_mag <- nrow(free_pt)
    out$n_matched <- nrow(matched)
    return(out)
  }

  estimator <- toupper(case$estimator)
  if (identical(estimator, "ADF")) estimator <- "WLS"
  ev <- tryCatch(
    magmaan::magmaan_core$evaluate_at(
      prob$spec, prob$dat, as.numeric(theta_full),
      estimator = estimator,
      W = if (estimator == "WLS") prob$W else NULL),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "audit_parity_magmaan_error"))
  if (inherits(ev, "audit_parity_magmaan_error")) {
    out <- empty_row(case, error = paste0("magmaan-eval: ", ev$error))
    out$lavaan_converged <- lav$converged
    out$lavaan_fx <- lav$objective
    out$n_free_lav <- nrow(lav$estimates)
    out$n_free_mag <- nrow(free_pt)
    out$n_matched <- nrow(matched)
    return(out)
  }

  n_total <- sum(unlist(prob$dat$nobs))
  bessel <- audit_bessel(case, n_total)
  data.frame(
    case_id = case$id, book = case$corpus, family = case$family,
    weight = case$estimator,
    n_free_mag = nrow(free_pt), n_free_lav = nrow(lav$estimates),
    n_matched = nrow(matched),
    lavaan_converged = isTRUE(lav$converged),
    lavaan_fx = lav$objective,
    lavaan_fx_bare = lav$objective / bessel,
    magmaan_fmin = ev$fmin, magmaan_iter = 0L,
    audit_advisory = as.character(ev$audit$advisory_status %||% NA),
    audit_grad_inf = as.numeric(ev$audit$grad_inf_norm %||% NA_real_),
    audit_stationary = isTRUE(ev$audit$stationary),
    error = NA_character_, stringsAsFactors = FALSE
  )
}

v7_candidates <- file.path(
  paper_dir,
  c(file.path("reports", "pilot-data", "audit-parity-v7"),
    file.path("results", "raw", "pilot-data", "audit-parity-v7")),
  "lavaan_audit_parity_disagree.csv")
v7_hits <- v7_candidates[file.exists(v7_candidates)]
if (!length(v7_hits)) {
  stop("missing audit-parity-v7 disagreement CSV in: ",
       paste(v7_candidates, collapse = ", "), call. = FALSE)
}
v7_path <- v7_hits[[1L]]
v7_disagree <- utils::read.csv(v7_path, stringsAsFactors = FALSE)
corpus_root <- snlls_corpus_root(paper_dir)

need <- unique(v7_disagree[, c("book", "weight")])
all_cases <- list()
for (i in seq_len(nrow(need))) {
  all_cases <- c(all_cases, corpus_cases(corpus_root, weights = need$weight[[i]],
                                         books = need$book[[i]]))
}

rows <- vector("list", nrow(v7_disagree))
for (i in seq_len(nrow(v7_disagree))) {
  key <- v7_disagree$case_id[[i]]
  if (!key %in% names(all_cases)) {
    rows[[i]] <- empty_row(
      list(id = key, corpus = v7_disagree$book[[i]],
           family = NA_character_, estimator = v7_disagree$weight[[i]]),
      error = "case missing from corpus_cases")
    next
  }
  message(sprintf("[%2d/%2d] %s", i, nrow(v7_disagree), key))
  rows[[i]] <- audit_one(all_cases[[key]])
}
v8 <- do.call(rbind, rows)
v8 <- v8[match(v7_disagree$case_id, v8$case_id), , drop = FALSE]

cmp <- data.frame(
  case = v7_disagree$case_id,
  v7_lav_conv = v7_disagree$lavaan_converged,
  v7_audit_stat = v7_disagree$audit_stationary,
  v7_grad_inf = v7_disagree$audit_grad_inf,
  v7_magmaan_fmin = v7_disagree$magmaan_fmin,
  v7_lav_fx_bare = v7_disagree$lavaan_fx_bare,
  v8_lav_conv = v8$lavaan_converged,
  v8_audit_stat = v8$audit_stationary,
  v8_grad_inf = v8$audit_grad_inf,
  v8_magmaan_fmin = v8$magmaan_fmin,
  v8_lav_fx_bare = v8$lavaan_fx_bare,
  v8_err = v8$error,
  stringsAsFactors = FALSE
)
cmp$v7_ratio <- ifelse(abs(cmp$v7_lav_fx_bare) > 0,
                       cmp$v7_magmaan_fmin / cmp$v7_lav_fx_bare, NA_real_)
cmp$v8_ratio <- ifelse(abs(cmp$v8_lav_fx_bare) > 0,
                       cmp$v8_magmaan_fmin / cmp$v8_lav_fx_bare, NA_real_)
cmp$v8_agrees <- !is.na(cmp$v8_audit_stat) & !is.na(cmp$v8_lav_conv) &
  (cmp$v8_audit_stat == cmp$v8_lav_conv)

out_dir <- file.path(experiment_results_dir(create = TRUE),
                     "refit-disagreements")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_csv <- file.path(out_dir, "refit_disagreements_v8.csv")
write_csv(cmp, out_csv)
cat(sprintf("wrote %s\n", out_csv))
