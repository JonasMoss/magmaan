#!/usr/bin/env Rscript
# magmaan vs lavaan speed bench (README-facing).
#
# Drives benchmarks/r/run_benchmark.R machinery on a curated subset of cases
# that use in-package lavaan datasets, writes a clean per-case CSV, and copies
# the SNLLS paper's Geiser-corpus headline table into the experiment's results.
#
# Usage:
#   Rscript experiments/05-lavaan-speed-bench/run_experiment.R [--iters N]

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
  out <- list(iters = 30L)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--iters N]\n")
      quit(save = "no", status = 0L)
    } else if (a == "--iters") {
      i <- i + 1L
      out$iters <- as.integer(args[[i]])
    } else if (startsWith(a, "--iters=")) {
      out$iters <- as.integer(sub("^--iters=", "", a))
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$iters) || out$iters < 1L) stop("--iters must be >= 1")
  out
}

load_hs <- function() {
  env <- new.env(parent = emptyenv())
  utils::data("HolzingerSwineford1939", package = "lavaan", envir = env)
  env$HolzingerSwineford1939
}

make_ord_df <- function(n, cuts_by_var, seed = 1L) {
  set.seed(seed)
  p <- length(cuts_by_var)
  z <- matrix(stats::rnorm(n * p), n, p)
  z[, 2] <- 0.55 * z[, 1] + sqrt(1 - 0.55^2) * z[, 2]
  z[, 3] <- 0.40 * z[, 1] + 0.25 * z[, 2] + sqrt(0.78) * z[, 3]
  z[, 4] <- 0.30 * z[, 1] + 0.20 * z[, 2] + sqrt(0.87) * z[, 4]
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_len(p)) {
    out[[paste0("x", j)]] <- ordered(cut(z[, j],
                                         c(-Inf, cuts_by_var[[j]], Inf),
                                         labels = FALSE))
  }
  out
}

make_mixed_df <- function(n = 360L) {
  set.seed(22L)
  eta <- stats::rnorm(n)
  y1 <- 0.80 * eta + 0.60 * stats::rnorm(n)
  y2 <- 0.75 * eta + 0.66 * stats::rnorm(n)
  data.frame(
    x1 = ordered(cut(y1, c(-Inf, -0.65, 0.30, Inf), labels = FALSE)),
    x2 = ordered(cut(y2, c(-Inf, -0.45, 0.55, Inf), labels = FALSE)),
    x3 = 0.70 * eta + 0.71 * stats::rnorm(n),
    x4 = 0.65 * eta + 0.76 * stats::rnorm(n)
  )
}

time_repeated <- function(fun, iters, batch = 1L) {
  out <- numeric(iters)
  for (i in seq_len(iters)) {
    gc(FALSE)
    t0 <- proc.time()[["elapsed"]]
    for (j in seq_len(batch)) fun()
    out[[i]] <- (proc.time()[["elapsed"]] - t0) / batch
  }
  out
}

measure_pair <- function(magmaan_fun, lavaan_fun, iters, batch = 1L) {
  list(
    magmaan = time_repeated(magmaan_fun, iters, batch),
    lavaan = time_repeated(lavaan_fun, iters, batch)
  )
}

median_ms <- function(x) {
  stats::median(x) * 1e3
}

hs_model <- "visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9"

ordinal_model <- "f =~ x1 + x2 + x3 + x4"
ordinal_names <- paste0("x", 1:4)

pipeline_ugamma_magmaan <- function() {
  core <- magmaan::magmaan_core
  X <- as.matrix(load_hs()[paste0("x", 1:9)])
  pt <- core$lavaan_lavaanify(hs_model)
  ss <- core$data_sample_stats_from_raw(X)
  fit <- core$fit_fit(pt, ss)
  fit_ss <- core$fit_sample_stats(fit)
  chisq <- core$infer_chi2_stat(fit_ss, fit$fmin)
  df <- core$infer_df_stat(fit$partable, fit_ss)
  uf <- core$infer_build_u_factor_parts(fit$partable, fit_ss, fit$theta)
  zc <- core$infer_casewise_contributions(pt, X)
  m <- core$infer_reduced_gamma_sample(uf, zc, fit$nobs)
  eig <- core$infer_ugamma_eigenvalues(m)
  sb <- core$infer_satorra_bentler(chisq, df, eig)
  rse <- core$infer_robust_se_raw_parts(fit$partable, fit_ss, fit$theta, X)
  c(chisq_scaled = sb$chi2_scaled, scale = sb$scale_c,
    se_max = max(rse$se), df = df)
}

pipeline_ugamma_lavaan <- function() {
  x <- load_hs()[paste0("x", 1:9)]
  fit <- lavaan::cfa(hs_model, data = x, estimator = "MLM")
  se <- sqrt(diag(lavaan::vcov(fit)))
  c(chisq_scaled = unname(lavaan::fitMeasures(fit, "chisq.scaled")),
    scale = unname(lavaan::fitMeasures(fit, "chisq.scaling.factor")),
    se_max = max(se), df = unname(lavaan::fitMeasures(fit, "df")))
}

pipeline_ordinal_magmaan <- function() {
  core <- magmaan::magmaan_core
  df <- make_ord_df(360, list(c(-0.70, 0.35), c(-0.55, 0.60),
                              c(-0.85, 0.20), c(-0.45, 0.75)),
                    seed = 11L)
  spec <- magmaan::model_spec(ordinal_model, ordered = ordinal_names,
                              parameterization = "delta")
  stats <- core$data_ordinal_stats_from_df(df, spec)
  fit <- core$fit_dwls_ordinal(
    spec, stats, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
  rob <- core$robust_ordinal(fit, stats)
  std <- core$measures_standardize_all(fit, rob$vcov)
  c(chisq = fit$ntotal * fit$fmin, df = rob$df,
    se_max = max(rob$se), std_max = max(abs(std$theta)))
}

pipeline_ordinal_lavaan <- function() {
  df <- make_ord_df(360, list(c(-0.70, 0.35), c(-0.55, 0.60),
                              c(-0.85, 0.20), c(-0.45, 0.75)),
                    seed = 11L)
  fit <- lavaan::cfa(ordinal_model, data = df, ordered = ordinal_names,
                     estimator = "DWLS", parameterization = "delta")
  se <- sqrt(diag(lavaan::vcov(fit)))
  c(chisq = unname(lavaan::fitMeasures(fit, "chisq")),
    df = unname(lavaan::fitMeasures(fit, "df")), se_max = max(se))
}

pipeline_mixed_magmaan <- function() {
  core <- magmaan::magmaan_core
  df <- make_mixed_df()
  spec <- magmaan::model_spec(ordinal_model, ordered = c("x1", "x2"),
                              parameterization = "delta", meanstructure = TRUE)
  stats <- core$data_mixed_ordinal_stats_from_df(df, spec)
  fit <- core$fit_dwls_mixed_ordinal(
    spec, stats, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
  c(fmin = fit$fmin, npar = fit$npar, theta_max = max(abs(fit$theta)))
}

pipeline_mixed_lavaan <- function() {
  df <- make_mixed_df()
  fit <- lavaan::cfa(ordinal_model, data = df, ordered = c("x1", "x2"),
                     estimator = "DWLS", parameterization = "delta",
                     meanstructure = TRUE)
  c(chisq = unname(lavaan::fitMeasures(fit, "chisq")),
    df = unname(lavaan::fitMeasures(fit, "df")),
    npar = sum(lavaan::parTable(fit)$free > 0L))
}

pipeline_specs <- list(
  list(
    id = "hs_ml_ugamma_robust",
    label = "HS 3-factor CFA robust report",
    estimator = "ML + UGamma",
    data_path = "raw continuous",
    n_obs = 301L,
    n_vars = 9L,
    includes = "sample stats, ML fit, UGamma spectra, SB test, robust SE",
    magmaan = pipeline_ugamma_magmaan,
    lavaan = pipeline_ugamma_lavaan,
    check = function(m, l) max(abs(m[c("chisq_scaled", "scale")] -
                                  l[c("chisq_scaled", "scale")]))
  ),
  list(
    id = "ordinal_dwls_polychoric_robust",
    label = "Ordinal CFA robust report",
    estimator = "DWLS",
    data_path = "all ordinal",
    n_obs = 360L,
    n_vars = 4L,
    includes = "polychorics, NACOV/weights, DWLS fit, robust ordinal SE/test",
    magmaan = pipeline_ordinal_magmaan,
    lavaan = pipeline_ordinal_lavaan,
    check = function(m, l) abs(unname(m["chisq"]) - unname(l["chisq"]))
  ),
  list(
    id = "mixed_dwls_polyserial",
    label = "Mixed ordinal/continuous CFA",
    estimator = "DWLS",
    data_path = "2 ordinal + 2 continuous",
    n_obs = 360L,
    n_vars = 4L,
    includes = "thresholds, polyserial/polychoric moments, DWLS fit",
    magmaan = pipeline_mixed_magmaan,
    lavaan = pipeline_mixed_lavaan,
    check = function(m, l) abs(unname(m["npar"]) - unname(l["npar"]))
  )
)

time_pipeline <- function(spec, iters) {
  m0 <- spec$magmaan()
  l0 <- spec$lavaan()
  check_value <- spec$check(m0, l0)
  tm <- measure_pair(spec$magmaan, spec$lavaan, iters)
  m_med <- median_ms(tm$magmaan)
  l_med <- median_ms(tm$lavaan)
  data.frame(
    pipeline_id = spec$id,
    label = spec$label,
    estimator = spec$estimator,
    data_path = spec$data_path,
    n_obs = spec$n_obs,
    n_vars = spec$n_vars,
    includes = spec$includes,
    lavaan_ms_median = l_med,
    magmaan_ms_median = m_med,
    speedup = l_med / m_med,
    check_value = check_value,
    iterations = iters,
    stringsAsFactors = FALSE
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

project_root <- repo_root()
bench_r <- file.path(project_root, "benchmarks", "r")
experiment_results_dir <- ensure_results_dir()

source(file.path(bench_r, "common.R"))
source(file.path(bench_r, "cases.R"))
source(file.path(project_root, "experiments", "_support", "R", "helpers.R"))

bench_dir_abs <- file.path(project_root, "benchmarks")
read_prepared_data_abs <- function(case_id) {
  path <- file.path(bench_dir_abs, "data", "prepared", case_id, "data.csv")
  if (!file.exists(path)) {
    stop("prepared data not found: ", path,
         "; run Rscript benchmarks/r/prepare_case.R ", case_id, call. = FALSE)
  }
  utils::read.csv(path, check.names = FALSE, na.strings = c("", "NA"))
}
read_model_abs <- function(case_id) {
  path <- file.path(bench_dir_abs, "cases", case_id, "model.lav")
  if (!file.exists(path)) stop("missing model: ", path, call. = FALSE)
  paste(readLines(path, warn = FALSE), collapse = "\n")
}

set_single_threaded_math()

require_pkg("lavaan")
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

EST_TOL <- 1e-3

CURATED <- c(
  "hs_3factor_cfa",
  "hs_3factor_cfa_fiml_masked",
  "hs_3factor_cfa_uls",
  "hs_3factor_cfa_gls",
  "bollen_democracy_sem",
  "demo_growth_linear"
)

CASE_LABELS <- c(
  hs_3factor_cfa             = "Holzinger-Swineford 3-factor CFA",
  hs_3factor_cfa_fiml_masked = "Holzinger-Swineford 3-factor CFA (masked)",
  hs_3factor_cfa_uls         = "Holzinger-Swineford 3-factor CFA",
  hs_3factor_cfa_gls         = "Holzinger-Swineford 3-factor CFA",
  bollen_democracy_sem       = "Bollen democracy SEM",
  demo_growth_linear         = "Demo.growth linear LGC"
)

lavaan_fit_case <- function(case, model, data, report = c("estimate", "standard")) {
  report <- match.arg(report)
  fun <- switch(case$lavaan_function %||% "sem",
                cfa = lavaan::cfa, growth = lavaan::growth,
                sem = lavaan::sem, lavaan::sem)
  fun(model = model, data = data,
      estimator = case$lavaan_estimator %||% case$estimator %||% "ML",
      meanstructure = isTRUE(case$meanstructure),
      se = if (report == "standard") "standard" else "none",
      test = if (report == "standard") "standard" else "none",
      missing = case$lavaan_missing %||% "listwise")
}

magmaan_fit_case <- function(case, model, data) {
  args <- list(
    model = model, data = data,
    estimator = case$magmaan_estimator %||% case$estimator %||% "ML",
    meanstructure = isTRUE(case$meanstructure),
    auto_cov_y = TRUE,
    ordered = case$ordered %||% NULL
  )
  if (identical(case$lavaan_function %||% "sem", "growth")) {
    args$model_type <- "growth"
  }
  do.call(magmaan::magmaan, args)
}

est_max_abs_diff <- function(m_pt, l_pt) {
  k <- function(pt) paste(pt$group, pt$op, pt$lhs, pt$rhs, sep = "\r")
  mk <- k(m_pt); lk <- k(l_pt)
  common <- intersect(mk, lk)
  if (!length(common)) return(NA_real_)
  d <- abs(m_pt$est[match(common, mk)] - l_pt$est[match(common, lk)])
  max(d, na.rm = TRUE)
}

time_case <- function(case_id, iters) {
  case <- get_case(case_id)
  data <- read_prepared_data_abs(case_id)
  model <- read_model_abs(case_id)
  est_tol <- case$est_tolerance %||% EST_TOL

  mfit <- magmaan_fit_case(case, model, data)
  lfit_std <- lavaan_fit_case(case, model, data, "standard")
  l_pt <- lavaan::parTable(lfit_std)
  est_diff <- est_max_abs_diff(mfit$partable, l_pt)
  if (!is.finite(est_diff) || est_diff > est_tol) {
    stop(sprintf("estimate drift %.3g exceeds tolerance %.3g for %s",
                 est_diff, est_tol, case_id), call. = FALSE)
  }

  tm <- measure_pair(
    function() magmaan_fit_case(case, model, data),
    function() lavaan_fit_case(case, model, data, "estimate"),
    iters,
    batch = 10L
  )
  t_magmaan <- tm$magmaan
  t_lavaan <- tm$lavaan
  m_med <- stats::median(t_magmaan) * 1e3
  l_med <- stats::median(t_lavaan)  * 1e3

  data.frame(
    case_id = case_id,
    label = unname(CASE_LABELS[case_id] %||% case_id),
    estimator = case$estimator %||% "ML",
    n_obs = as.integer(sum(unname(lavaan::lavInspect(lfit_std, "nobs")))),
    n_vars = length(lavaan::lavNames(lfit_std, type = "ov")),
    npar = sum(l_pt$free > 0L),
    lavaan_ms_median = l_med,
    magmaan_ms_median = m_med,
    speedup = l_med / m_med,
    est_max_abs_diff = est_diff,
    iterations = iters,
    stringsAsFactors = FALSE
  )
}

cat(sprintf(
  "lavaan-speed-bench: magmaan %s, lavaan %s, %s, iters=%d\n",
  as.character(magmaan::version()),
  as.character(utils::packageVersion("lavaan")),
  R.version.string, args$iters
))

rows <- list()
for (id in CURATED) {
  res <- tryCatch(time_case(id, args$iters), error = identity)
  if (inherits(res, "error")) {
    cat(sprintf("  %-24s ERROR  %s\n", id, conditionMessage(res)))
    next
  }
  rows[[length(rows) + 1L]] <- res
  cat(sprintf("  %-24s %-3s n=%-4d p=%-3d  lavaan=%7.3fms magmaan=%7.3fms  speedup=%.1fx\n",
              id, res$estimator, res$n_obs, res$n_vars,
              res$lavaan_ms_median, res$magmaan_ms_median, res$speedup))
}

zoo <- do.call(rbind, rows)
zoo_path <- file.path(experiment_results_dir, "zoo.csv")
utils::write.csv(zoo, zoo_path, row.names = FALSE)

pipeline_rows <- list()
for (spec in pipeline_specs) {
  res <- tryCatch(time_pipeline(spec, args$iters), error = identity)
  if (inherits(res, "error")) {
    cat(sprintf("  %-34s ERROR  %s\n", spec$id, conditionMessage(res)))
    next
  }
  pipeline_rows[[length(pipeline_rows) + 1L]] <- res
  cat(sprintf("  %-34s %-12s n=%-4d p=%-2d lavaan=%8.3fms magmaan=%8.3fms speedup=%.1fx\n",
              spec$id, spec$estimator, res$n_obs, res$n_vars,
              res$lavaan_ms_median, res$magmaan_ms_median, res$speedup))
}
pipeline <- if (length(pipeline_rows)) {
  do.call(rbind, pipeline_rows)
} else {
  data.frame(
    pipeline_id = character(), label = character(), estimator = character(),
    data_path = character(), n_obs = integer(), n_vars = integer(),
    includes = character(), lavaan_ms_median = double(),
    magmaan_ms_median = double(), speedup = double(), check_value = double(),
    iterations = integer(), stringsAsFactors = FALSE
  )
}
pipeline_path <- file.path(experiment_results_dir, "pipeline.csv")
utils::write.csv(pipeline, pipeline_path, row.names = FALSE)

# (Removed a copy of a headline-numbers CSV from the snlls-constrained paper:
# experiments are sinks and must not read a paper's content. The paper is not
# present in this tree anyway; exp05 stands on its own benchmark pipeline.)

cat(sprintf("wrote %s\n", zoo_path))
cat(sprintf("wrote %s\n", pipeline_path))
write_metadata(
  file.path(experiment_results_dir, "metadata.csv"),
  values = list(
    iters = args$iters,
    curated_cases = CURATED,
    n_zoo_rows = nrow(zoo),
    n_pipeline_rows = nrow(pipeline),
    geiser_source = if (file.exists(geiser_src)) geiser_src else "",
    single_threaded_math = TRUE
  ),
  packages = c("lavaan", "magmaan")
)
cat(sprintf("wrote %s\n", file.path(experiment_results_dir, "metadata.csv")))
