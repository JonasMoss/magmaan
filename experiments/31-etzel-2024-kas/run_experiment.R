#!/usr/bin/env Rscript
# Etzel 2024 KAS common-factor slice.
#
# The full OSF models are Mplus known-class, complex-survey, categorical-logit
# MLR models. This runner takes the continuous KAS component that magmaan can
# fit directly: saturated KAS covariance M1 versus common KAS factor M2, by VET
# group, with FIML for the KAS missingness.
#
# Usage:
#   Rscript run_experiment.R [--domains chemistry,physics,math] [--smoke]

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
  out <- list(domains = c("chemistry", "physics", "math"),
              smoke = FALSE, tolerance = 1e-4)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--domains chemistry,physics,math] ",
          "[--tolerance TOL] [--smoke]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--domains") {
      i <- i + 1L; out$domains <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--domains=")) {
      out$domains <- parse_csv_arg(sub("^--domains=", "", a))
    } else if (a == "--tolerance") {
      i <- i + 1L; out$tolerance <- as.numeric(args[[i]])
    } else if (startsWith(a, "--tolerance=")) {
      out$tolerance <- as.numeric(sub("^--tolerance=", "", a))
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (isTRUE(out$smoke)) out$domains <- "chemistry"
  if (!is.finite(out$tolerance) || out$tolerance <= 0) {
    stop("--tolerance must be positive", call. = FALSE)
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
root <- repo_root()
res_dir <- ensure_results_dir()
set_single_threaded_math()
require_pkg("magmaan")
require_pkg("lavaan")
suppressPackageStartupMessages({
  library(magmaan)
  library(lavaan)
})

group_labels <- c("technicians", "chem_lab_assistants", "industrial_clerks")
data_rel <- file.path("external", "paper_data", "Etzel-2024",
                      "osfstorage-archive",
                      "2024_OSF_DATA_PK_KAS_VET_named.csv")
data_path <- file.path(root, data_rel)
if (!file.exists(data_path)) {
  stop("Etzel OSF CSV not found at ", data_path, call. = FALSE)
}

domain_specs <- list(
  chemistry = list(
    vars = c("ackas", "bckas", "cckas"),
    common = "chemistry KAS common factor",
    first_intercept_free = FALSE,
    mplus_m1 = file.path("Analyses", "01_Chemistry",
                         "01_pk_chem_unconstrained_cov.out"),
    mplus_m2 = file.path("Analyses", "01_Chemistry",
                         "02_pk_chem_kas_common_factor.out")
  ),
  physics = list(
    vars = c("apkas", "bpkas", "cpkas"),
    common = "physics KAS common factor, first intercept freed",
    first_intercept_free = TRUE,
    mplus_m1 = file.path("Analyses", "02_Physics",
                         "01_pk_physics_unconstrained_cov.out"),
    mplus_m2 = file.path("Analyses", "02_Physics",
                         "02a_pk_physics_kas_common_factor.out")
  ),
  math = list(
    vars = c("amkas", "bmkas", "cmkas"),
    common = "math KAS common factor",
    first_intercept_free = FALSE,
    mplus_m1 = file.path("Analyses", "03_Math",
                         "01_pk_math_unconstrained_cov.out"),
    mplus_m2 = file.path("Analyses", "03_Math",
                         "02_pk_math_kas_common_factor.out")
  )
)

bad_domains <- setdiff(cfg$domains, names(domain_specs))
if (length(bad_domains)) {
  stop("unknown domain(s): ", paste(bad_domains, collapse = ", "),
       call. = FALSE)
}

raw <- utils::read.csv(data_path, na.strings = "-99", check.names = FALSE)
raw$vetgroup <- factor(raw$vetgroup, levels = c(1, 2, 3),
                       labels = group_labels)

model_saturated <- function(vars) {
  a <- vars[[1L]]; b <- vars[[2L]]; c <- vars[[3L]]
  paste(
    sprintf("%s ~~ c(NA, NA, NA)*%s + c(NA, NA, NA)*%s + c(NA, NA, NA)*%s",
            a, a, b, c),
    sprintf("%s ~~ c(NA, NA, NA)*%s + c(NA, NA, NA)*%s", b, b, c),
    sprintf("%s ~~ c(NA, NA, NA)*%s", c, c),
    sprintf("%s ~ c(NA, NA, NA)*1", a),
    sprintf("%s ~ c(NA, NA, NA)*1", b),
    sprintf("%s ~ c(NA, NA, NA)*1", c),
    sep = "\n"
  )
}

model_common <- function(vars, first_intercept_free = FALSE) {
  a <- vars[[1L]]; b <- vars[[2L]]; c <- vars[[3L]]
  first_label <- if (isTRUE(first_intercept_free)) "ibcc" else "ibc"
  paste(
    sprintf("kas =~ 1*%s + 1*%s + 1*%s", a, b, c),
    "kas ~~ c(NA, NA, NA)*kas",
    sprintf("%s ~~ c(NA, NA, NA)*%s", a, a),
    sprintf("%s ~~ c(NA, NA, NA)*%s", b, b),
    sprintf("%s ~~ c(NA, NA, NA)*%s", c, c),
    sprintf("%s ~~ 0*%s + 0*%s", a, b, c),
    sprintf("%s ~~ 0*%s", b, c),
    sprintf("%s ~ c(%s, %s, %s)*1", a, first_label, first_label, first_label),
    sprintf("%s ~ c(ibc, ibc, ibc)*1", b),
    sprintf("%s ~ c(ibc, ibc, ibc)*1", c),
    "kas ~ c(0, NA, NA)*1",
    sep = "\n"
  )
}

number_from_line <- function(line) {
  nums <- regmatches(line, gregexpr(
    "[-+]?[0-9]*\\.?[0-9]+(?:[DdEe][-+]?[0-9]+)?", line,
    perl = TRUE
  ))[[1L]]
  if (!length(nums)) return(NA_real_)
  as.numeric(gsub("[Dd]", "E", nums[[length(nums)]]))
}

parse_mplus_fit <- function(rel_path) {
  path <- file.path(root, "external", "paper_data", "Etzel-2024",
                    "osfstorage-archive", rel_path)
  if (!file.exists(path)) {
    return(data.frame(mplus_file = rel_path, mplus_npar = NA_integer_,
                      mplus_logl = NA_real_, mplus_scaling = NA_real_))
  }
  lines <- readLines(path, warn = FALSE)
  hit <- function(pattern) grep(pattern, lines, value = TRUE)[[1L]]
  data.frame(
    mplus_file = rel_path,
    mplus_npar = as.integer(number_from_line(hit("Number of Free Parameters"))),
    mplus_logl = number_from_line(hit("H0 Value")),
    mplus_scaling = number_from_line(hit("H0 Scaling Correction Factor")),
    stringsAsFactors = FALSE
  )
}

fit_pair <- function(dat, syntax, domain, model_id, model_name) {
  lav <- lavaan::sem(
    syntax, data = dat, group = "vetgroup", group.label = group_labels,
    missing = "fiml", meanstructure = TRUE, fixed.x = FALSE
  )
  fit <- magmaan::magmaan(
    syntax, dat, estimator = "FIML", groups = "vetgroup", fixed_x = FALSE,
    control = list(max_iter = 10000, ftol = 1e-12, gtol = 1e-8)
  )

  key <- function(pt) paste(pt$group, pt$lhs, pt$op, pt$rhs, sep = "\r")
  mag_pt <- fit$partable[fit$partable$free > 0L, ]
  lav_pt <- lavaan::parTable(lav)
  lav_free <- lav_pt[lav_pt$free > 0L, ]
  m <- match(key(mag_pt), key(lav_free))
  if (anyNA(m)) {
    stop("could not align lavaan/magmaan free rows for ", domain, " ", model_id,
         call. = FALSE)
  }

  eq_constraints <- sum(fit$partable$op == "==")
  npar_effective <- fit$npar - eq_constraints
  fm <- lavaan::fitMeasures(lav, c("chisq", "df", "logl", "npar"))
  nobs <- fit$nobs

  fit_row <- data.frame(
    domain = domain,
    model = model_id,
    model_name = model_name,
    n_total = sum(nobs),
    n_technicians = nobs[[1L]],
    n_chem_lab_assistants = nobs[[2L]],
    n_industrial_clerks = nobs[[3L]],
    magmaan_fmin = fit$fmin,
    magmaan_npar_raw = fit$npar,
    magmaan_eq_constraints = eq_constraints,
    magmaan_npar_effective = npar_effective,
    magmaan_converged = fit$converged,
    lavaan_logl = unname(fm[["logl"]]),
    lavaan_chisq = unname(fm[["chisq"]]),
    lavaan_df = unname(fm[["df"]]),
    lavaan_npar = unname(fm[["npar"]]),
    max_abs_free_param_diff = max(abs(mag_pt$est - lav_free$est[m])),
    stringsAsFactors = FALSE
  )

  par_rows <- cbind(
    data.frame(
      domain = domain, model = model_id, group = mag_pt$group,
      group_label = group_labels[mag_pt$group],
      lhs = mag_pt$lhs, op = mag_pt$op, rhs = mag_pt$rhs,
      label = mag_pt$label, stringsAsFactors = FALSE
    ),
    data.frame(
      magmaan_est = mag_pt$est,
      lavaan_est = lav_free$est[m],
      diff = mag_pt$est - lav_free$est[m]
    )
  )

  list(fit = fit, lav = lav, fit_row = fit_row, par_rows = par_rows)
}

fit_rows <- list()
param_rows <- list()
lr_rows <- list()
data_rows <- list()
syntax_rows <- list()

for (domain in cfg$domains) {
  spec <- domain_specs[[domain]]
  vars <- spec$vars
  dat <- raw[c("vetgroup", vars)]
  used <- rowSums(!is.na(dat[vars])) > 0L
  dat <- dat[used, , drop = FALSE]

  models <- list(
    M1 = list(name = "saturated KAS covariance", syntax = model_saturated(vars)),
    M2 = list(name = spec$common,
              syntax = model_common(vars, spec$first_intercept_free))
  )

  data_rows[[domain]] <- data.frame(
    domain = domain,
    variables = paste(vars, collapse = ","),
    n_raw = nrow(raw),
    n_used = nrow(dat),
    n_dropped_empty_kas = nrow(raw) - nrow(dat),
    n_technicians = sum(dat$vetgroup == group_labels[[1L]]),
    n_chem_lab_assistants = sum(dat$vetgroup == group_labels[[2L]]),
    n_industrial_clerks = sum(dat$vetgroup == group_labels[[3L]]),
    nonmissing_t1 = sum(!is.na(dat[[vars[[1L]]]])),
    nonmissing_t2 = sum(!is.na(dat[[vars[[2L]]]])),
    nonmissing_t3 = sum(!is.na(dat[[vars[[3L]]]])),
    stringsAsFactors = FALSE
  )

  fits <- list()
  for (model_id in names(models)) {
    syntax_rows[[paste(domain, model_id)]] <- data.frame(
      domain = domain,
      model = model_id,
      model_name = models[[model_id]]$name,
      syntax = models[[model_id]]$syntax,
      stringsAsFactors = FALSE
    )
    ans <- fit_pair(dat, models[[model_id]]$syntax, domain, model_id,
                    models[[model_id]]$name)
    fits[[model_id]] <- ans
    fit_rows[[paste(domain, model_id)]] <- ans$fit_row
    param_rows[[paste(domain, model_id)]] <- ans$par_rows
  }

  mag_lr <- 2 * fits$M2$fit_row$n_total *
    (fits$M2$fit_row$magmaan_fmin - fits$M1$fit_row$magmaan_fmin)
  lav_lr <- 2 * (fits$M1$fit_row$lavaan_logl - fits$M2$fit_row$lavaan_logl)
  df_diff <- fits$M2$fit_row$lavaan_df - fits$M1$fit_row$lavaan_df
  mplus_m1 <- parse_mplus_fit(spec$mplus_m1)
  mplus_m2 <- parse_mplus_fit(spec$mplus_m2)
  mplus_lr <- 2 * (mplus_m1$mplus_logl - mplus_m2$mplus_logl)
  mplus_df <- mplus_m1$mplus_npar - mplus_m2$mplus_npar

  lr_rows[[domain]] <- data.frame(
    domain = domain,
    n_total = fits$M2$fit_row$n_total,
    df = df_diff,
    magmaan_lr = mag_lr,
    lavaan_lr = lav_lr,
    abs_lr_diff = abs(mag_lr - lav_lr),
    p_value = stats::pchisq(mag_lr, df = df_diff, lower.tail = FALSE),
    mplus_full_lr_context = mplus_lr,
    mplus_full_df_context = mplus_df,
    mplus_m1_file = mplus_m1$mplus_file,
    mplus_m2_file = mplus_m2$mplus_file,
    stringsAsFactors = FALSE
  )
}

fits_df <- rbind_fill(fit_rows)
params_df <- rbind_fill(param_rows)
lr_df <- rbind_fill(lr_rows)
data_df <- rbind_fill(data_rows)
syntax_df <- rbind_fill(syntax_rows)

write_csv(data_df, file.path(res_dir, "data_summary.csv"))
write_csv(syntax_df, file.path(res_dir, "model_syntax.csv"))
write_csv(fits_df, file.path(res_dir, "model_fits.csv"))
write_csv(params_df, file.path(res_dir, "parameter_estimates.csv"))
write_csv(lr_df, file.path(res_dir, "lr_tests.csv"))
write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    domains = cfg$domains,
    data_path = data_rel,
    group_labels = group_labels,
    tolerance = cfg$tolerance,
    note = paste("KAS-only FIML slice; full Mplus categorical-logit",
                 "known-class complex models are context only")
  ),
  packages = c("magmaan", "lavaan")
)

max_param_gap <- max(fits_df$max_abs_free_param_diff, na.rm = TRUE)
max_lr_gap <- max(lr_df$abs_lr_diff, na.rm = TRUE)
if (max_param_gap > cfg$tolerance || max_lr_gap > cfg$tolerance) {
  stop(sprintf("parity tolerance failed: max param %.3g, max LR %.3g",
               max_param_gap, max_lr_gap), call. = FALSE)
}

cat("Wrote:\n")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "data_summary.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "model_syntax.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "model_fits.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "parameter_estimates.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "lr_tests.csv"), "\n", sep = "")
cat(sprintf("Max |magmaan-lavaan free parameter diff|: %.3g\n", max_param_gap))
cat(sprintf("Max |magmaan-lavaan LR diff|: %.3g\n", max_lr_gap))
