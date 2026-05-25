#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

parse_args <- function(args) {
  out <- list(
    audit_run = "audit-parity-v7",
    refit_csv = "/tmp/refit_disagreements_v8.csv",
    run_full = FALSE,
    run_lcs = FALSE,
    run_refit = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    value <- NULL
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--audit-run RUN_ID] ",
        "[--refit-csv PATH] [--run-full] [--run-lcs] [--run-refit]\n",
        "\n",
        "Default copies the existing paper-side audit-parity-v7 outputs and ",
        "the /tmp v8 refit CSV if present.\n",
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
    if (arg == "--audit-run") out$audit_run <- need_value()
    else if (arg == "--refit-csv") out$refit_csv <- need_value()
    else if (arg == "--run-full") out$run_full <- TRUE
    else if (arg == "--run-lcs") out$run_lcs <- TRUE
    else if (arg == "--run-refit") out$run_refit <- TRUE
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  out
}

repo_path <- function(...) file.path(repo_root(), ...)
paper_dir <- function() repo_path("papers", "snlls-constrained")
paper_report <- function(...) file.path(paper_dir(), "reports", ...)
paper_script <- function(...) file.path(paper_dir(), "scripts", ...)

copy_required <- function(from, to) {
  if (!file.exists(from)) stop("missing required source: ", from, call. = FALSE)
  dir.create(dirname(to), recursive = TRUE, showWarnings = FALSE)
  file.copy(from, to, overwrite = TRUE)
  invisible(to)
}

copy_optional <- function(from, to) {
  if (!file.exists(from)) return(FALSE)
  dir.create(dirname(to), recursive = TRUE, showWarnings = FALSE)
  file.copy(from, to, overwrite = TRUE)
  TRUE
}

run_script <- function(path) {
  if (!file.exists(path)) stop("missing script: ", path, call. = FALSE)
  status <- system2("Rscript", path)
  if (!identical(status, 0L)) {
    stop("script failed: ", path, call. = FALSE)
  }
}

read_csv <- function(path) {
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
                  na.strings = c("", "NA"))
}

boolish <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes")
}

finite_ratio <- function(num, den) {
  out <- rep(NA_real_, length(num))
  ok <- is.finite(num) & is.finite(den) & abs(den) > 0
  out[ok] <- num[ok] / den[ok]
  out
}

classify_disagreement <- function(x) {
  ratio <- finite_ratio(x$magmaan_fmin, x$lavaan_fx_bare)
  out <- rep("gradient verdict split", nrow(x))
  out[grepl("little_2013_ch3_fig_3_6_1indicator", x$case_id, fixed = TRUE)] <-
    "auto.fix.single parity gap"
  out[grepl("muthen_2017_ch2_ex2_1__adf", x$case_id, fixed = TRUE)] <-
    "ADF conditioning side finding"
  out[is.finite(ratio) & ratio > 2] <- "LCS / phantom-latent objective gap"
  out[is.finite(x$audit_grad_inf) & x$audit_grad_inf < 1e-2] <-
    "tolerance-boundary gradient verdict"
  out
}

summarize_audit <- function(parity, meta) {
  ok <- parity[is.na(parity$error), , drop = FALSE]
  lavaan_converged <- boolish(ok$lavaan_converged)
  audit_stationary <- boolish(ok$audit_stationary)
  data.frame(
    audit_run = if ("run_id" %in% names(meta)) meta$run_id[[1L]] else "",
    n_cells = nrow(parity),
    n_with_both_fits = nrow(ok),
    n_errors = sum(!is.na(parity$error)),
    n_agree = sum(lavaan_converged == audit_stationary, na.rm = TRUE),
    n_lavaan_yes_audit_no = sum(lavaan_converged & !audit_stationary,
                                na.rm = TRUE),
    n_lavaan_no_audit_yes = sum(!lavaan_converged & audit_stationary,
                                na.rm = TRUE),
    agreement_rate = if (nrow(ok)) {
      mean(lavaan_converged == audit_stationary, na.rm = TRUE)
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

summarize_by <- function(parity, var) {
  ok <- parity[is.na(parity$error), , drop = FALSE]
  split_rows <- split(ok, ok[[var]], drop = TRUE)
  do.call(rbind, lapply(split_rows, function(x) {
    lavaan_converged <- boolish(x$lavaan_converged)
    audit_stationary <- boolish(x$audit_stationary)
    data.frame(
      block = var,
      level = as.character(x[[var]][[1L]]),
      n = nrow(x),
      agree = sum(lavaan_converged == audit_stationary, na.rm = TRUE),
      lavaan_yes_audit_no = sum(lavaan_converged & !audit_stationary,
                                na.rm = TRUE),
      lavaan_no_audit_yes = sum(!lavaan_converged & audit_stationary,
                                na.rm = TRUE),
      agreement_rate = mean(lavaan_converged == audit_stationary,
                            na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
}

refit_summary <- function(refit) {
  if (!nrow(refit)) {
    return(data.frame(metric = character(), value = character(),
                      stringsAsFactors = FALSE))
  }
  agrees <- boolish(refit$v8_agrees)
  errors <- !is.na(refit$v8_err)
  data.frame(
    metric = c("v7 disagreement cells refit",
               "v8 agreements among v7 disagreements",
               "v8 remaining disagreements",
               "v8 errors"),
    value = c(nrow(refit),
              sum(agrees, na.rm = TRUE),
              sum(!agrees & !errors, na.rm = TRUE),
              sum(errors, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

if (isTRUE(args$run_full)) run_script(paper_script("run_lavaan_audit_parity.R"))
if (isTRUE(args$run_lcs)) run_script(paper_script("diagnose_lcs_disagreement.R"))
if (isTRUE(args$run_refit)) {
  run_script(experiment_path("scripts", "refit_disagreements.R"))
  args$refit_csv <- file.path(results_dir(create = TRUE), "refit-disagreements",
                              "refit_disagreements_v8.csv")
}

results <- ensure_results_dir()
audit_src <- paper_report("pilot-data", args$audit_run)
audit_dst <- file.path(results, "audit-parity")
lcs_src <- paper_report("lcs-objective-gap")
lcs_dst <- file.path(results, "lcs-objective-gap")
refit_dst <- file.path(results, "refit-disagreements")

copy_required(file.path(audit_src, "lavaan_audit_parity.csv"),
              file.path(audit_dst, "lavaan_audit_parity.csv"))
copy_required(file.path(audit_src, "lavaan_audit_parity_disagree.csv"),
              file.path(audit_dst, "lavaan_audit_parity_disagree.csv"))
copy_required(file.path(audit_src, "lavaan_audit_parity_meta.csv"),
              file.path(audit_dst, "lavaan_audit_parity_meta.csv"))

copy_required(file.path(lcs_src, "summary.csv"),
              file.path(lcs_dst, "summary.csv"))
for (path in list.files(lcs_src, pattern = "\\.csv$", full.names = TRUE)) {
  copy_optional(path, file.path(lcs_dst, basename(path)))
}

has_refit <- copy_optional(args$refit_csv,
                           file.path(refit_dst, "refit_disagreements_v8.csv"))

parity <- read_csv(file.path(audit_dst, "lavaan_audit_parity.csv"))
disagree <- read_csv(file.path(audit_dst, "lavaan_audit_parity_disagree.csv"))
meta <- read_csv(file.path(audit_dst, "lavaan_audit_parity_meta.csv"))
lcs_summary <- read_csv(file.path(lcs_dst, "summary.csv"))
refit <- if (has_refit) {
  read_csv(file.path(refit_dst, "refit_disagreements_v8.csv"))
} else {
  data.frame()
}

disagree$fmin_ratio <- finite_ratio(disagree$magmaan_fmin,
                                    disagree$lavaan_fx_bare)
disagree$classification <- classify_disagreement(disagree)
write_csv(disagree, file.path(results, "audit_disagreements_enriched.csv"))

if (nrow(refit)) {
  refit$v7_ratio <- finite_ratio(refit$v7_magmaan_fmin, refit$v7_lav_fx_bare)
  refit$v8_ratio <- finite_ratio(refit$v8_magmaan_fmin, refit$v8_lav_fx_bare)
  current_failures <- refit[!boolish(refit$v8_agrees) | !is.na(refit$v8_err),
                            , drop = FALSE]
} else {
  current_failures <- disagree
}

write_csv(summarize_audit(parity, meta), file.path(results, "audit_summary.csv"))
write_csv(summarize_by(parity, "book"), file.path(results, "audit_by_book.csv"))
write_csv(summarize_by(parity, "weight"), file.path(results, "audit_by_weight.csv"))
write_csv(refit_summary(refit), file.path(results, "refit_summary.csv"))
write_csv(current_failures, file.path(results, "current_failures.csv"))

drivers <- data.frame(
  driver = c("full audit sweep", "LCS moment diagnosis",
             "v7 disagreement refit", "ADF conditioning probe",
             "paper supplement", "LCS diagnosis writeup"),
  path = c(
    paper_script("run_lavaan_audit_parity.R"),
    paper_script("diagnose_lcs_disagreement.R"),
    experiment_path("scripts", "refit_disagreements.R"),
    experiment_path("scripts", "diagnose_adf_conditioning.R"),
    paper_report("lavaan-audit-parity.qmd"),
    paper_report("lcs-objective-gap.md")
  ),
  role = c(
    "756-cell lavaan self-report vs magmaan terminal-audit sweep",
    "side-by-side implied/sample moments on three LCS target cells",
    "targeted refit of the 24 audit-parity-v7 disagreement cells",
    "eigendecomposition of the saturated Muthen ADF conditioning case",
    "original Quarto supplement consumed by the paper",
    "diagnostic narrative for the LCS / phantom-latent objective gap"
  ),
  expected_runtime = c("about 1 h", "about 30 s", "about 3 s", "about 1 s",
                       "render only", "read only"),
  stringsAsFactors = FALSE
)
write_csv(drivers, file.path(results, "driver_inventory.csv"))

write_metadata(
  file.path(results, "metadata.csv"),
  values = list(
    audit_run = args$audit_run,
    audit_source = audit_src,
    lcs_source = lcs_src,
    refit_source = if (has_refit) args$refit_csv else "",
    has_refit = has_refit,
    n_audit_cells = nrow(parity),
    n_v7_disagreements = nrow(disagree),
    n_current_failures = nrow(current_failures)
  )
)

cat(sprintf(
  "copied %s: %d audit cells, %d v7 disagreements, %d current/projected failures\n",
  args$audit_run, nrow(parity), nrow(disagree), nrow(current_failures)
))
cat(sprintf("wrote normalized tables to %s\n", results))
