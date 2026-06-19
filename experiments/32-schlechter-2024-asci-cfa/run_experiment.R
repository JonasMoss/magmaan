#!/usr/bin/env Rscript
# Schlechter et al. 2024 ASCI ordinal CFA parity probe.
#
# The OSF scripts mix exploratory factor analysis, reliability estimates,
# correlations, t-tests, and lavaan WLSMV CFA/invariance models. This runner
# keeps the magmaan-relevant slice: the final 12-item two-factor ordinal CFA in
# Study 1's held-out subsample, Study 2 English sample, and Study 3 German sample.
#
# Usage:
#   Rscript run_experiment.R [--studies study1,study2,study3] [--smoke]

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

rbind_fill <- function(xs) {
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (nm in miss) x[[nm]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, xs)
}

parse_args <- function(args) {
  out <- list(studies = c("study1", "study2", "study3"),
              smoke = FALSE, param_tolerance = 2e-5, chisq_tolerance = 1e-4)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--studies study1,study2,study3] ",
          "[--param-tolerance TOL] [--chisq-tolerance TOL] [--smoke]\n",
          sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--studies") {
      i <- i + 1L; out$studies <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--studies=")) {
      out$studies <- parse_csv_arg(sub("^--studies=", "", a))
    } else if (a == "--param-tolerance") {
      i <- i + 1L; out$param_tolerance <- as.numeric(args[[i]])
    } else if (startsWith(a, "--param-tolerance=")) {
      out$param_tolerance <- as.numeric(sub("^--param-tolerance=", "", a))
    } else if (a == "--chisq-tolerance") {
      i <- i + 1L; out$chisq_tolerance <- as.numeric(args[[i]])
    } else if (startsWith(a, "--chisq-tolerance=")) {
      out$chisq_tolerance <- as.numeric(sub("^--chisq-tolerance=", "", a))
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (isTRUE(out$smoke)) out$studies <- "study3"
  if (!is.finite(out$param_tolerance) || out$param_tolerance <= 0) {
    stop("--param-tolerance must be positive", call. = FALSE)
  }
  if (!is.finite(out$chisq_tolerance) || out$chisq_tolerance <= 0) {
    stop("--chisq-tolerance must be positive", call. = FALSE)
  }
  out
}

replace_missing_codes <- function(x) {
  out <- as.data.frame(x)
  for (nm in names(out)) {
    if (is.numeric(out[[nm]]) || is.integer(out[[nm]])) {
      out[[nm]][out[[nm]] == -77] <- NA
    } else {
      out[[nm]][out[[nm]] %in% c("-77", "NA")] <- NA
    }
  }
  out
}

as_numeric_columns <- function(x, vars) {
  for (nm in vars) x[[nm]] <- suppressWarnings(as.numeric(x[[nm]]))
  x
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
root <- repo_root()
res_dir <- ensure_results_dir()
set_single_threaded_math()
require_pkg("magmaan")
require_pkg("lavaan")
require_pkg("readr")
require_pkg("readxl")
suppressPackageStartupMessages({
  library(magmaan)
  library(lavaan)
})

data_dir_rel <- file.path("external", "paper_data", "Schlechter-2024",
                          "osfstorage-archive")
data_dir <- file.path(root, data_dir_rel)
if (!dir.exists(data_dir)) {
  stop("Schlechter OSF archive not found at ", data_dir, call. = FALSE)
}

study1_all_items <- c(
  "ATCS_1", "ATCS_6", "ATCS_11", "ATCS_16", "ATCS_21", "ATCS_26",
  "ATCS_31", "ATCS_36", "ATCS_41", "ATCS_46", "ATCS_51", "ATCS_56",
  "ATCS_61", "ATCS_66", "ATCS_71", "ATCS_76", "ATCS_81", "ATCS_86",
  "ATCS_91", "ATCS_96", "ATCS_101", "ATCS_106", "ATCS_111", "ATCS_116"
)

study_specs <- list(
  study1 = list(
    label = "Study 1 held-out sample",
    file = "Data_t1_Study_1.csv",
    script = "ASCI_Study_1.R",
    ordered = c("ATCS_6", "ATCS_11", "ATCS_21", "ATCS_36", "ATCS_41",
                "ATCS_46", "ATCS_76", "ATCS_86", "ATCS_96", "ATCS_101",
                "ATCS_111", "ATCS_116"),
    model = paste(
      "positive =~ ATCS_6 + ATCS_36 + ATCS_46 + ATCS_76 +",
      "ATCS_86 + ATCS_96 + ATCS_116",
      "negative =~ ATCS_11 + ATCS_21 + ATCS_41 + ATCS_101 + ATCS_111",
      "positive ~~ negative",
      sep = "\n"
    )
  ),
  study2 = list(
    label = "Study 2 English sample",
    file = "Data_English_Sample_Study_2.xlsx",
    script = "ASCI_English_Sample_Study_2.R",
    ordered = c("asci_info", "asci_reass", "asci_can", "asci_who",
                "asci_soci", "asci_prob", "asci_decis", "asci_bad",
                "asci_preve", "asci_unhap", "asci_psych", "asci_done"),
    model = paste(
      "positive =~ asci_info + asci_reass + asci_can + asci_who +",
      "asci_soci + asci_prob + asci_decis",
      "negative =~ asci_bad + asci_preve + asci_unhap + asci_psych + asci_done",
      "positive ~~ negative",
      sep = "\n"
    )
  ),
  study3 = list(
    label = "Study 3 German sample",
    file = "Data_German_Sample_Study_3.csv",
    script = "ASCI_German_Sample_Study_3.R",
    ordered = paste0("asci_", 1:12),
    model = paste(
      "positive =~ asci_1 + asci_2 + asci_3 + asci_4 + asci_5 +",
      "asci_6 + asci_7",
      "negative =~ asci_8 + asci_9 + asci_10 + asci_11 + asci_12",
      "positive ~~ negative",
      sep = "\n"
    )
  )
)

bad_studies <- setdiff(cfg$studies, names(study_specs))
if (length(bad_studies)) {
  stop("unknown study/studies: ", paste(bad_studies, collapse = ", "),
       call. = FALSE)
}

load_study <- function(study_id, spec) {
  path <- file.path(data_dir, spec$file)
  if (!file.exists(path)) stop("missing data file: ", path, call. = FALSE)
  if (identical(study_id, "study1")) {
    dat <- readr::read_delim(
      path, delim = ";", escape_double = FALSE, trim_ws = TRUE,
      locale = readr::locale(decimal_mark = ","), show_col_types = FALSE
    )
    dat <- replace_missing_codes(dat)
    dat <- as_numeric_columns(dat, c(study1_all_items, "duration_min"))
    dat$rts_sd <- apply(dat[study1_all_items], 1L, stats::sd)
    dat <- dat[is.na(dat$rts_sd) | dat$rts_sd != 0, , drop = FALSE]
    dat <- dat[is.na(dat$duration_min) | dat$duration_min <= 60, , drop = FALSE]

    set.seed(20)
    sample_size <- floor(0.5 * nrow(dat))
    sub <- sample(seq_len(nrow(dat)), size = sample_size)
    dat <- dat[sub, , drop = FALSE]
  } else if (identical(study_id, "study2")) {
    dat <- readxl::read_excel(path)
    dat <- replace_missing_codes(dat)
  } else if (identical(study_id, "study3")) {
    dat <- readr::read_delim(
      path, delim = ";", escape_double = FALSE, trim_ws = TRUE,
      show_col_types = FALSE
    )
    dat <- replace_missing_codes(dat)
  } else {
    stop("unknown study: ", study_id, call. = FALSE)
  }
  dat <- as_numeric_columns(dat, spec$ordered)
  as.data.frame(dat)
}

fit_study <- function(study_id, spec) {
  dat <- load_study(study_id, spec)
  complete <- stats::complete.cases(dat[spec$ordered])
  n_category <- vapply(spec$ordered, function(nm) {
    length(unique(dat[[nm]][!is.na(dat[[nm]])]))
  }, integer(1))

  lav <- lavaan::cfa(
    spec$model, data = dat, estimator = "WLSMV", ordered = spec$ordered
  )
  mag <- magmaan::magmaan(
    spec$model, dat, estimator = "DWLS", ordered = spec$ordered,
    parameterization = "delta", missing = "listwise"
  )

  key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group, sep = "\r")
  mag_free <- mag$partable[mag$partable$free > 0L, ]
  lav_free <- lavaan::parTable(lav)
  lav_free <- lav_free[lav_free$free > 0L, ]
  m <- match(key(mag_free), key(lav_free))
  if (anyNA(m)) {
    stop("could not align lavaan/magmaan free rows for ", study_id,
         call. = FALSE)
  }

  nobs <- as.integer(lavaan::lavInspect(lav, "nobs"))
  mag_chisq_lavaan_scale <- 2 * (sum(nobs) - length(nobs)) * mag$fmin
  fm <- lavaan::fitMeasures(
    lav,
    c("chisq", "df", "chisq.scaled", "pvalue.scaled", "cfi.scaled",
      "tli.scaled", "rmsea.scaled", "srmr")
  )
  diff <- mag_free$est - lav_free$est[m]
  fit_row <- data.frame(
    study = study_id,
    label = spec$label,
    source_script = spec$script,
    source_data = spec$file,
    n_rows_after_script_cleaning = nrow(dat),
    n_listwise = sum(complete),
    nobs_lavaan = sum(nobs),
    n_items = length(spec$ordered),
    min_categories = min(n_category),
    max_categories = max(n_category),
    magmaan_converged = mag$converged,
    magmaan_fmin = mag$fmin,
    magmaan_chisq_lavaan_scale = mag_chisq_lavaan_scale,
    lavaan_chisq = unname(fm[["chisq"]]),
    lavaan_df = unname(fm[["df"]]),
    abs_chisq_diff = abs(mag_chisq_lavaan_scale - unname(fm[["chisq"]])),
    lavaan_chisq_scaled = unname(fm[["chisq.scaled"]]),
    lavaan_pvalue_scaled = unname(fm[["pvalue.scaled"]]),
    lavaan_cfi_scaled = unname(fm[["cfi.scaled"]]),
    lavaan_tli_scaled = unname(fm[["tli.scaled"]]),
    lavaan_rmsea_scaled = unname(fm[["rmsea.scaled"]]),
    lavaan_srmr = unname(fm[["srmr"]]),
    magmaan_npar = mag$npar,
    lavaan_npar = unname(lavaan::fitMeasures(lav, "npar")),
    max_abs_free_param_diff = max(abs(diff)),
    max_abs_loading_diff = max(abs(diff[mag_free$op == "=~"])),
    max_abs_threshold_diff = max(abs(diff[mag_free$op == "|"])),
    stringsAsFactors = FALSE
  )
  param_rows <- data.frame(
    study = study_id,
    label = spec$label,
    lhs = mag_free$lhs,
    op = mag_free$op,
    rhs = mag_free$rhs,
    magmaan_est = mag_free$est,
    lavaan_est = lav_free$est[m],
    diff = diff,
    stringsAsFactors = FALSE
  )
  syntax_row <- data.frame(
    study = study_id,
    label = spec$label,
    ordered = paste(spec$ordered, collapse = ","),
    model = spec$model,
    stringsAsFactors = FALSE
  )
  list(fit_row = fit_row, param_rows = param_rows, syntax_row = syntax_row)
}

fit_rows <- list()
param_rows <- list()
syntax_rows <- list()
for (study_id in cfg$studies) {
  ans <- fit_study(study_id, study_specs[[study_id]])
  fit_rows[[study_id]] <- ans$fit_row
  param_rows[[study_id]] <- ans$param_rows
  syntax_rows[[study_id]] <- ans$syntax_row
}

fits <- rbind_fill(fit_rows)
params <- rbind_fill(param_rows)
syntax <- rbind_fill(syntax_rows)

write_csv(fits, file.path(res_dir, "model_fits.csv"))
write_csv(params, file.path(res_dir, "parameter_estimates.csv"))
write_csv(syntax, file.path(res_dir, "model_syntax.csv"))
write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    studies = cfg$studies,
    data_dir = data_dir_rel,
    param_tolerance = cfg$param_tolerance,
    chisq_tolerance = cfg$chisq_tolerance,
    note = paste("Single-sample ASCI ordinal CFA parity only;",
                 "EFA, reliability, correlations, t-tests, and invariance",
                 "ladders are out of this runner's scope")
  ),
  packages = c("magmaan", "lavaan", "readr", "readxl")
)

max_param_gap <- max(fits$max_abs_free_param_diff, na.rm = TRUE)
max_chisq_gap <- max(fits$abs_chisq_diff, na.rm = TRUE)
if (max_param_gap > cfg$param_tolerance ||
    max_chisq_gap > cfg$chisq_tolerance) {
  stop(sprintf("parity tolerance failed: max param %.3g, max chisq %.3g",
               max_param_gap, max_chisq_gap), call. = FALSE)
}

cat("Wrote:\n")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "model_fits.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "parameter_estimates.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "model_syntax.csv"), "\n", sep = "")
cat(sprintf("Max |magmaan-lavaan free parameter diff|: %.3g\n", max_param_gap))
cat(sprintf("Max |magmaan-lavaan DWLS chi-square diff|: %.3g\n",
            max_chisq_gap))
