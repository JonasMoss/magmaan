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

args <- parse_args(commandArgs(trailingOnly = TRUE))

project_root <- repo_root()
bench_r <- file.path(project_root, "benchmarks", "r")
results_dir <- ensure_results_dir()

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
require_pkg("bench")
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

  tm <- bench::mark(
    magmaan = magmaan_fit_case(case, model, data),
    lavaan  = lavaan_fit_case(case, model, data, "estimate"),
    check = FALSE, filter_gc = FALSE,
    iterations = iters, time_unit = "s"
  )
  t_magmaan <- as.numeric(tm$time[[match("magmaan", as.character(tm$expression))]])
  t_lavaan  <- as.numeric(tm$time[[match("lavaan",  as.character(tm$expression))]])
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
zoo_path <- file.path(results_dir, "zoo.csv")
utils::write.csv(zoo, zoo_path, row.names = FALSE)

geiser_src <- file.path(project_root, "papers", "snlls-constrained",
                        "tables", "snlls_geiser_corpus_lavaan.csv")
geiser_path <- file.path(results_dir, "geiser.csv")
if (file.exists(geiser_src)) {
  file.copy(geiser_src, geiser_path, overwrite = TRUE)
  cat(sprintf("\ncopied %s\n", geiser_path))
} else {
  cat(sprintf("\nwarn: %s not present; geiser headline numbers omitted\n", geiser_src))
}

cat(sprintf("wrote %s\n", zoo_path))
write_metadata(
  file.path(results_dir, "metadata.csv"),
  values = list(
    iters = args$iters,
    curated_cases = CURATED,
    n_zoo_rows = nrow(zoo),
    geiser_source = if (file.exists(geiser_src)) geiser_src else "",
    single_threaded_math = TRUE
  ),
  packages = c("bench", "lavaan", "magmaan")
)
cat(sprintf("wrote %s\n", file.path(results_dir, "metadata.csv")))
