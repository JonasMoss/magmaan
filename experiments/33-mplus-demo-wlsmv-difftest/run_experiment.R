#!/usr/bin/env Rscript
# Mplus Demo WLSMV DIFFTEST probe for ordinal pairwise missing data.
#
# Generates a six-indicator two-group ordinal CFA that fits inside the Mplus
# Demo dependent-variable limit, runs Mplus WLSMV DIFFTEST, and compares the
# same raw data against lavaan and magmaan.
#
# Usage:
#   Rscript run_experiment.R [--seed S] [--n-total N] [--missing-rate P]
#                            [--mplus-command mpdemo]

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

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

rbind_fill <- function(xs) {
  xs <- Filter(Negate(is.null), xs)
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
  out <- list(seed = 20260619L, n_total = 500L, missing_rate = .50,
              mplus_command = "mpdemo")
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--seed S] [--n-total N] ",
          "[--missing-rate P] [--mplus-command mpdemo]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--seed") {
      i <- i + 1L; out$seed <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed=")) {
      out$seed <- as.integer(sub("^--seed=", "", a))
    } else if (a == "--n-total") {
      i <- i + 1L; out$n_total <- as.integer(args[[i]])
    } else if (startsWith(a, "--n-total=")) {
      out$n_total <- as.integer(sub("^--n-total=", "", a))
    } else if (a == "--missing-rate") {
      i <- i + 1L; out$missing_rate <- as.numeric(args[[i]])
    } else if (startsWith(a, "--missing-rate=")) {
      out$missing_rate <- as.numeric(sub("^--missing-rate=", "", a))
    } else if (a == "--mplus-command") {
      i <- i + 1L; out$mplus_command <- args[[i]]
    } else if (startsWith(a, "--mplus-command=")) {
      out$mplus_command <- sub("^--mplus-command=", "", a)
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$seed)) stop("--seed must be finite", call. = FALSE)
  if (!is.finite(out$n_total) || out$n_total < 120L || out$n_total %% 2L != 0L) {
    stop("--n-total must be an even integer >= 120", call. = FALSE)
  }
  if (!is.finite(out$missing_rate) ||
      out$missing_rate < 0 || out$missing_rate >= .95) {
    stop("--missing-rate must be in [0, .95)", call. = FALSE)
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("lavaan")
require_pkg("magmaan")
suppressPackageStartupMessages({
  library(lavaan)
  library(magmaan)
})

res_dir <- ensure_results_dir()
mplus_dir <- file.path(res_dir, sprintf("mplus_seed_%d", cfg$seed))
dir.create(mplus_dir, recursive = TRUE, showWarnings = FALSE)

ov <- paste0("y", 1:6)
thresholds <- c(-1.3, -0.47, 0.47, 1.3)
loading <- .70
model <- paste0("f =~ ", paste(ov, collapse = " + "))

draw_group <- function(n, group_label) {
  eta <- rnorm(n)
  aux <- 0.5 * eta + sqrt(1 - 0.5^2) * rnorm(n)
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) {
    z[, j] <- loading * eta + rnorm(n, sd = sqrt(1 - loading^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out$aux <- aux
  out$group <- group_label
  out
}

apply_rank_missing <- function(dat, missing_rate) {
  b_rows <- which(dat$group == "B")
  nb <- length(b_rows)
  target <- as.integer(round(missing_rate * nb))
  if (target <= 0L) {
    dat$group <- factor(dat$group, levels = c("A", "B"))
    return(dat)
  }
  ranks <- rank(dat$aux[b_rows], ties.method = "random")
  weights <- pmax(nb - ranks, 0)
  if (!any(weights > 0)) weights <- rep(1, nb)
  for (nm in ov[5:6]) {
    picked <- sample.int(nb, size = target, replace = FALSE, prob = weights)
    dat[[nm]][b_rows[picked]] <- NA
  }
  dat$group <- factor(dat$group, levels = c("A", "B"))
  dat
}

set.seed(cfg$seed)
n_group <- as.integer(cfg$n_total / 2L)
dat <- rbind(draw_group(n_group, "A"), draw_group(n_group, "B"))
dat <- apply_rank_missing(dat, cfg$missing_rate)
write_csv(dat, file.path(res_dir, "probe_data.csv"))

to_mplus_data <- function(dat) {
  out <- data.frame(
    g = ifelse(dat$group == "A", 1L, 2L),
    aux = dat$aux,
    stringsAsFactors = FALSE
  )
  for (nm in ov) out[[nm]] <- as.integer(dat[[nm]])
  out
}

write.table(to_mplus_data(dat), file.path(mplus_dir, "probe.dat"),
            row.names = FALSE, col.names = FALSE, quote = FALSE, na = "-999")

mplus_base <- c(
  "DATA: FILE IS probe.dat;",
  "VARIABLE:",
  " NAMES ARE g aux y1-y6;",
  " USEVARIABLES ARE y1-y6;",
  " CATEGORICAL ARE y1-y6;",
  " GROUPING IS g (1 = A 2 = B);",
  " MISSING ARE ALL (-999);",
  "ANALYSIS:",
  " ESTIMATOR = WLSMV;",
  " PARAMETERIZATION = DELTA;"
)

mplus_inputs <- list(
  h1 = c(
    "TITLE: demo H1 configural;",
    mplus_base,
    "MODEL:",
    " f BY y1-y6;",
    "MODEL B:",
    " {y1-y6@1};",
    " [f@0];",
    " f BY y1@1 y2-y6;",
    " [y1$1-y6$4];",
    "SAVEDATA:",
    " DIFFTEST IS h1diff.dat;"
  ),
  h0_metric = c(
    "TITLE: demo H0 metric;",
    mplus_base,
    " DIFFTEST IS h1diff.dat;",
    "MODEL:",
    " f BY y1-y6;",
    "MODEL B:",
    " [f@0];",
    " {y1-y6@1};",
    " [y1$1-y6$4];"
  ),
  h0_scalar = c(
    "TITLE: demo H0 scalar;",
    mplus_base,
    " DIFFTEST IS h1diff.dat;",
    "MODEL:",
    " f BY y1-y6;"
  )
)

mplus_inputs$h1_auxiliary_check <- append(
  mplus_inputs$h1,
  values = " AUXILIARY = aux (m);",
  after = match(" MISSING ARE ALL (-999);", mplus_inputs$h1)
)

for (nm in names(mplus_inputs)) {
  writeLines(mplus_inputs[[nm]], file.path(mplus_dir, paste0(nm, ".inp")))
}

run_mplus <- function(input, output, work_dir, command) {
  old <- setwd(work_dir)
  on.exit(setwd(old), add = TRUE)
  out <- tryCatch(
    suppressWarnings(system2(command, c(input, output),
                             stdout = TRUE, stderr = TRUE)),
    error = function(e) {
      structure(conditionMessage(e), status = 127L)
    }
  )
  status <- attr(out, "status") %||% 0L
  data.frame(input = input, output = output, status = as.integer(status),
             stdout = paste(out, collapse = "\n"), stringsAsFactors = FALSE)
}

mplus_status <- rbind(
  run_mplus("h1.inp", "h1.out", mplus_dir, cfg$mplus_command),
  run_mplus("h0_metric.inp", "h0_metric.out", mplus_dir, cfg$mplus_command),
  run_mplus("h0_scalar.inp", "h0_scalar.out", mplus_dir, cfg$mplus_command),
  run_mplus("h1_auxiliary_check.inp", "h1_auxiliary_check.out",
            mplus_dir, cfg$mplus_command)
)
write_csv(mplus_status, file.path(res_dir, "mplus_status.csv"))

required <- mplus_status$input %in% c("h1.inp", "h0_metric.inp", "h0_scalar.inp")
if (any(mplus_status$status[required] != 0L)) {
  bad <- mplus_status[mplus_status$status != 0L & required, ]
  stop("required Mplus run failed: ", paste(bad$input, collapse = ", "),
       call. = FALSE)
}

mplus_number <- function(line) {
  matches <- regmatches(line, gregexpr(
    "[-+]?[0-9]*\\.?[0-9]+([DdEe][-+]?[0-9]+)?", line,
    perl = TRUE))[[1L]]
  if (!length(matches)) return(NA_real_)
  as.numeric(gsub("[Dd]", "E", tail(matches, 1L)))
}

section_number <- function(lines, heading, field, occurrence = 1L) {
  idx <- which(trimws(lines) == heading)
  if (length(idx) < occurrence) return(NA_real_)
  win <- lines[idx[[occurrence]]:min(length(lines), idx[[occurrence]] + 25L)]
  hit <- grep(paste0("^\\s*", field, "\\s+"), win, value = TRUE)
  if (!length(hit)) return(NA_real_)
  mplus_number(hit[[1L]])
}

parse_mplus_model <- function(path, model_name, n_total) {
  lines <- readLines(path, warn = FALSE)
  npar_line <- grep("^Number of Free Parameters", lines, value = TRUE)
  fmin <- section_number(lines,
                         "Optimum Function Value for Weighted Least-Squares Estimator",
                         "Value")
  data.frame(
    engine = "mplus",
    variant = "wlsmv_pd",
    model = model_name,
    npar = if (length(npar_line)) mplus_number(npar_line[[1L]]) else NA_real_,
    df = section_number(lines, "Chi-Square Test of Model Fit",
                        "Degrees of Freedom"),
    chisq_raw = if (is.finite(fmin)) 2 * n_total * fmin else NA_real_,
    chisq_scaled = section_number(lines, "Chi-Square Test of Model Fit",
                                  "Value"),
    pvalue_scaled = section_number(lines, "Chi-Square Test of Model Fit",
                                   "P-Value"),
    fmin = fmin,
    converged = any(grepl("THE MODEL ESTIMATION TERMINATED NORMALLY", lines,
                          fixed = TRUE)),
    error = "",
    stringsAsFactors = FALSE
  )
}

parse_mplus_difftest <- function(path, restriction) {
  lines <- readLines(path, warn = FALSE)
  data.frame(
    engine = "mplus",
    variant = "wlsmv_pd",
    restriction = restriction,
    method = "DIFFTEST",
    stat = section_number(lines, "Chi-Square Test for Difference Testing",
                          "Value"),
    df = section_number(lines, "Chi-Square Test for Difference Testing",
                        "Degrees of Freedom"),
    p_value = section_number(lines, "Chi-Square Test for Difference Testing",
                             "P-Value"),
    error = "",
    stringsAsFactors = FALSE
  )
}

mplus_fit <- rbind(
  parse_mplus_model(file.path(mplus_dir, "h1.out"), "configural", cfg$n_total),
  parse_mplus_model(file.path(mplus_dir, "h0_metric.out"), "metric", cfg$n_total),
  parse_mplus_model(file.path(mplus_dir, "h0_scalar.out"), "scalar_mplus",
                    cfg$n_total)
)
mplus_diff <- rbind(
  parse_mplus_difftest(file.path(mplus_dir, "h0_metric.out"), "metric"),
  parse_mplus_difftest(file.path(mplus_dir, "h0_scalar.out"), "scalar_mplus")
)

aux_lines <- readLines(file.path(mplus_dir, "h1_auxiliary_check.out"),
                       warn = FALSE)
auxiliary_check <- data.frame(
  engine = "mplus",
  input = "h1_auxiliary_check.inp",
  status = mplus_status$status[mplus_status$input == "h1_auxiliary_check.inp"],
  warning = paste(grep("^\\*\\*\\* WARNING", aux_lines, value = TRUE),
                  collapse = " | "),
  error = paste(grep("^\\*\\*\\* ERROR", aux_lines, value = TRUE),
                collapse = " | "),
  message = paste(trimws(grep(
    "Estimator WLSMV is not allowed|Analysis with categorical variables",
    aux_lines, value = TRUE)), collapse = " | "),
  stringsAsFactors = FALSE
)
write_csv(auxiliary_check, file.path(res_dir, "auxiliary_check.csv"))

lavaan_model_rows <- function(fits) {
  rows <- lapply(names(fits), function(nm) {
    fm <- lavaan::fitMeasures(
      fits[[nm]],
      c("npar", "chisq", "df", "pvalue", "chisq.scaled",
        "df.scaled", "pvalue.scaled"))
    data.frame(
      engine = "lavaan",
      variant = "wlsmv_pairwise",
      model = nm,
      npar = unname(fm[["npar"]]),
      df = unname(fm[["df"]]),
      chisq_raw = unname(fm[["chisq"]]),
      chisq_scaled = unname(fm[["chisq.scaled"]]),
      pvalue_scaled = unname(fm[["pvalue.scaled"]]),
      fmin = NA_real_,
      converged = lavaan::lavInspect(fits[[nm]], "converged"),
      error = "",
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

lavaan_lrt_row <- function(h1, h0, restriction, a_method) {
  tab <- suppressMessages(lavaan::lavTestLRT(
    h1, h0, method = "satorra.2000", A.method = a_method,
    scaled.shifted = TRUE))
  data.frame(
    engine = "lavaan",
    variant = "wlsmv_pairwise",
    restriction = restriction,
    method = paste0("satorra.2000_", a_method, "_scaled_shifted"),
    stat = as.numeric(tab[2L, "Chisq diff"]),
    df = as.numeric(tab[2L, "Df diff"]),
    p_value = as.numeric(tab[2L, "Pr(>Chisq)"]),
    error = "",
    stringsAsFactors = FALSE
  )
}

dat_lav <- dat
dat_lav[ov] <- lapply(dat_lav[ov], ordered)
scalar_mplus_extra <- paste(
  c("f ~ c(0, NA)*1", sprintf("%s ~ c(0, 0)*1", ov)),
  collapse = "\n")
fit_lav_h1 <- lavaan::cfa(
  model, data = dat_lav, ordered = ov, group = "group",
  estimator = "WLSMV", parameterization = "delta", missing = "pairwise",
  se = "none")
fit_lav_metric <- lavaan::cfa(
  model, data = dat_lav, ordered = ov, group = "group",
  group.equal = "loadings", estimator = "WLSMV",
  parameterization = "delta", missing = "pairwise", se = "none")
fit_lav_scalar_default <- lavaan::cfa(
  model, data = dat_lav, ordered = ov, group = "group",
  group.equal = c("loadings", "thresholds"), estimator = "WLSMV",
  parameterization = "delta", missing = "pairwise", se = "none")
fit_lav_scalar_mplus <- lavaan::cfa(
  paste(model, scalar_mplus_extra, sep = "\n"),
  data = dat_lav, ordered = ov, group = "group",
  group.equal = c("loadings", "thresholds"), estimator = "WLSMV",
  parameterization = "delta", missing = "pairwise", se = "none")

lavaan_fit <- lavaan_model_rows(list(
  configural = fit_lav_h1,
  metric = fit_lav_metric,
  scalar_lavaan_default = fit_lav_scalar_default,
  scalar_mplus = fit_lav_scalar_mplus
))
lavaan_diff <- rbind(
  lavaan_lrt_row(fit_lav_h1, fit_lav_metric, "metric", "delta"),
  lavaan_lrt_row(fit_lav_h1, fit_lav_metric, "metric", "exact"),
  lavaan_lrt_row(fit_lav_h1, fit_lav_scalar_default,
                 "scalar_lavaan_default", "delta"),
  lavaan_lrt_row(fit_lav_h1, fit_lav_scalar_default,
                 "scalar_lavaan_default", "exact"),
  lavaan_lrt_row(fit_lav_h1, fit_lav_scalar_mplus, "scalar_mplus", "delta"),
  lavaan_lrt_row(fit_lav_h1, fit_lav_scalar_mplus, "scalar_mplus", "exact")
)

core <- magmaan::magmaan_core
moment_count <- 2L * (length(ov) * (length(thresholds)) + choose(length(ov), 2L))
spec_h1 <- magmaan::model_spec(
  model, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta")
spec_metric <- magmaan::model_spec(
  model, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta", group_equal = "loadings")
spec_scalar <- magmaan::model_spec(
  model, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta", group_equal = c("loadings", "thresholds"))
spec_scalar_mplus <- magmaan::model_spec(
  paste(model, scalar_mplus_extra, sep = "\n"),
  ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta", group_equal = c("loadings", "thresholds"))

magmaan_fit_row <- function(fit, model_name, pd_gamma) {
  pt <- fit$partable
  free <- pt$free > 0
  labels <- pt$label[free]
  effective_npar <- sum(labels == "") + length(unique(labels[labels != ""]))
  data.frame(
    engine = "magmaan",
    variant = paste0("pd_gamma_", pd_gamma),
    model = model_name,
    npar = effective_npar,
    raw_npar = fit$npar,
    df = moment_count - effective_npar,
    chisq_raw = 2 * fit$ntotal * fit$fmin,
    chisq_scaled = NA_real_,
    pvalue_scaled = NA_real_,
    fmin = fit$fmin,
    converged = isTRUE(fit$converged),
    error = "",
    stringsAsFactors = FALSE
  )
}

magmaan_nested_rows <- function(nt, restriction, pd_gamma) {
  data.frame(
    engine = "magmaan",
    variant = paste0("pd_gamma_", pd_gamma),
    restriction = restriction,
    method = c("restriction_map_delta_unscaled",
               "restriction_map_delta_scaled",
               "restriction_map_delta_adjusted",
               "restriction_map_delta_scaled_shifted",
               "restriction_map_delta_mixture"),
    stat = c(nt$T_diff, nt$T_scaled, nt$T_adjusted,
             nt$scaled_shifted$chi2_adj, nt$T_diff),
    df = c(nt$df_diff, nt$df_diff, nt$adjust_d0,
           nt$scaled_shifted$df, nt$df_diff),
    p_value = c(nt$p_unscaled, nt$p_scaled, nt$p_adjusted,
                nt$scaled_shifted$pvalue, nt$p_mixture),
    error = "",
    stringsAsFactors = FALSE
  )
}

magmaan_error_row <- function(restriction, pd_gamma, err,
                              method = "restriction_map_delta") {
  data.frame(
    engine = "magmaan",
    variant = paste0("pd_gamma_", pd_gamma),
    restriction = restriction,
    method = method,
    stat = NA_real_,
    df = NA_real_,
    p_value = NA_real_,
    error = conditionMessage(err),
    stringsAsFactors = FALSE
  )
}

magmaan_fit_rows <- list()
magmaan_diff_rows <- list()
for (pd_gamma in c("nominal", "overlap")) {
  stats <- core$data_ordinal_stats_from_df(
    dat_lav, spec_h1, ordered = ov, group = "group", missing = "pairwise",
    pd_gamma = pd_gamma, full_wls_weight = FALSE)
  fit_h1 <- magmaan::magmaan(spec_h1, stats, estimator = "DWLS")
  fit_metric <- magmaan::magmaan(spec_metric, stats, estimator = "DWLS")
  fit_scalar <- magmaan::magmaan(spec_scalar, stats, estimator = "DWLS")
  fit_scalar_mplus <- magmaan::magmaan(spec_scalar_mplus, stats,
                                       estimator = "DWLS")
  magmaan_fit_rows <- c(magmaan_fit_rows, list(
    magmaan_fit_row(fit_h1, "configural", pd_gamma),
    magmaan_fit_row(fit_metric, "metric", pd_gamma),
    magmaan_fit_row(fit_scalar, "scalar_group_equal", pd_gamma),
    magmaan_fit_row(fit_scalar_mplus, "scalar_mplus", pd_gamma)
  ))
  metric_nt <- tryCatch(
    magmaan::robust_nested_lrt(
      fit_h1, fit_metric, data = stats, method = "restriction_map",
      A.method = "delta", weight = "DWLS"),
    error = identity)
  if (inherits(metric_nt, "error")) {
    magmaan_diff_rows <- c(magmaan_diff_rows,
                           list(magmaan_error_row("metric", pd_gamma, metric_nt)))
  } else {
    magmaan_diff_rows <- c(magmaan_diff_rows,
                           list(magmaan_nested_rows(metric_nt, "metric",
                                                    pd_gamma)))
  }
  scalar_nt <- tryCatch(
    magmaan::robust_nested_lrt(
      fit_h1, fit_scalar, data = stats, method = "restriction_map",
      A.method = "delta", weight = "DWLS"),
    error = identity)
  if (inherits(scalar_nt, "error")) {
    magmaan_diff_rows <- c(magmaan_diff_rows,
                           list(magmaan_error_row("scalar_group_equal",
                                                  pd_gamma, scalar_nt)))
  } else {
    magmaan_diff_rows <- c(magmaan_diff_rows,
                           list(magmaan_nested_rows(scalar_nt,
                                                    "scalar_group_equal",
                                                    pd_gamma)))
  }
  scalar_mplus_nt <- tryCatch(
    magmaan::robust_nested_lrt(
      fit_h1, fit_scalar_mplus, data = stats, method = "restriction_map",
      A.method = "delta", weight = "DWLS"),
    error = identity)
  if (inherits(scalar_mplus_nt, "error")) {
    magmaan_diff_rows <- c(magmaan_diff_rows,
                           list(magmaan_error_row("scalar_mplus",
                                                  pd_gamma, scalar_mplus_nt)))
  } else {
    magmaan_diff_rows <- c(magmaan_diff_rows,
                           list(magmaan_nested_rows(scalar_mplus_nt,
                                                    "scalar_mplus",
                                                    pd_gamma)))
  }
}

model_fit <- rbind_fill(c(list(mplus_fit, lavaan_fit), magmaan_fit_rows))
difftest <- rbind_fill(c(list(mplus_diff, lavaan_diff), magmaan_diff_rows))
write_csv(model_fit, file.path(res_dir, "model_fit.csv"))
write_csv(difftest, file.path(res_dir, "difftest.csv"))

missing_b <- dat$group == "B"
metadata <- metadata_frame(
  values = list(
    seed = cfg$seed,
    n_total = cfg$n_total,
    n_per_group = n_group,
    missing_rate_target = cfg$missing_rate,
    missing_rate_realized_group_b_y5_y6 =
      mean(is.na(dat[missing_b, ov[5:6]])),
    mplus_command = cfg$mplus_command,
    mplus_work_dir = mplus_dir,
    model = "one_factor_six_indicator_ordinal_two_group",
    thresholds = paste(thresholds, collapse = ","),
    loading = loading,
    auxiliary_note =
      "aux drives rank-dependent missingness but is not used by WLSMV; Mplus rejects AUXILIARY=aux (m) with categorical variables"
  ),
  packages = c("lavaan", "magmaan")
)
write_csv(metadata, file.path(res_dir, "metadata.csv"))

focus <- difftest[
  (difftest$engine == "mplus" & difftest$method == "DIFFTEST") |
    (difftest$engine == "lavaan" &
       difftest$method == "satorra.2000_delta_scaled_shifted" &
       difftest$restriction %in% c("metric", "scalar_mplus")) |
    (difftest$engine == "magmaan" &
       difftest$method == "restriction_map_delta_scaled_shifted" &
       difftest$variant == "pd_gamma_overlap"),
  c("engine", "variant", "restriction", "method", "stat", "df", "p_value",
    "error")
]
print(focus, row.names = FALSE)
cat("\nWrote results to ", res_dir, "\n", sep = "")
