#!/usr/bin/env Rscript
# Magmaan-vs-lavaan benchmark runner (estimate-only smoke).
#
# For each active benchmark case this script:
#   1. fits the model with the magmaan R package and with lavaan,
#   2. validates magmaan against the lavaan oracle (estimates, fmin),
#   3. times both at the estimate-only workload level with `bench`,
#   4. writes benchmarks/results/smoke_<timestamp>.json and prints a table.
#
# Usage (from the repo root):
#   Rscript benchmarks/r/run_benchmark.R [<case_id> ...] | all
#
# With no arguments it runs every case marked supported_now in cases.R.
#
# The runner uses the *currently installed* magmaan R package as-is; it does
# not rebuild the C++ core. The committed reference_lavaan.json is the archival
# oracle; here lavaan is also fit live so the timed workload is symmetric.

source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "common.R"))
source(file.path(bench_script_dir(), "cases.R"))

require_pkg("lavaan")
require_pkg("bench")
require_pkg("jsonlite")
## magmaan is attached (not just loaded) so its exported post-fit primitives
## -- lavaan_compare_partable(), infer_*() -- resolve as bare names.
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

## Single-thread BLAS/OpenMP for comparable timings (best effort: env vars are
## only honored if set before the BLAS loads; recorded in the result anyway).
for (v in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) {
  do.call(Sys.setenv, stats::setNames(list("1"), v))
}

## Tolerance -- tunable. Magmaan-vs-lavaan estimate drift is the timing
## correctness gate (docs/benchmark_plan.md: "correctness gates timing").
## fmin/chi-square are reported diagnostics, not gates: magmaan's $fmin is the
## full ML discrepancy F while lavaan reports F/2.
EST_TOL <- 1e-3

BENCH_ITERATIONS <- 30L

args <- commandArgs(trailingOnly = TRUE)
ids <- if (length(args) == 0 || identical(args, "all")) {
  names(Filter(function(x) isTRUE(x$supported_now), benchmark_cases))
} else {
  args
}

results_dir <- function() ensure_dir(file.path(bench_root(), "results"))

## lavaan fit, mirroring make_lavaan_reference.R::fit_case(). `report` selects
## the workload level: "standard" materializes SEs/tests (used once for the
## correctness oracle); "estimate" is se/test = "none" (the timed workload).
lavaan_fit_case <- function(case, model, data, report = c("estimate", "standard")) {
  report <- match.arg(report)
  fun <- switch(case$lavaan_function %||% "sem",
                cfa = lavaan::cfa, growth = lavaan::growth,
                sem = lavaan::sem, lavaan::sem)
  fun(model = model, data = data,
      estimator = case$estimator %||% "ML",
      meanstructure = isTRUE(case$meanstructure),
      se = if (report == "standard") "standard" else "none",
      test = if (report == "standard") "standard" else "none",
      missing = "listwise")
}

## magmaan fit, mirroring lavaan cfa/sem/growth defaults as far as the exposed
## lavaanify flags allow (auto.cov.y = TRUE is a cfa/sem/growth default).
magmaan_fit_case <- function(case, model, data) {
  magmaan::magmaan(model, data, estimator = case$estimator %||% "ML",
                   meanstructure = isTRUE(case$meanstructure),
                   auto_cov_y = TRUE)
}

## Max absolute estimate difference over parameter-table rows present in both,
## keyed on (group, op, lhs, rhs).
est_max_abs_diff <- function(m_pt, l_pt) {
  k <- function(pt) paste(pt$group, pt$op, pt$lhs, pt$rhs, sep = "\r")
  mk <- k(m_pt)
  lk <- k(l_pt)
  common <- intersect(mk, lk)
  if (!length(common)) return(NA_real_)
  d <- abs(m_pt$est[match(common, mk)] - l_pt$est[match(common, lk)])
  max(d, na.rm = TRUE)
}

## Best-effort magmaan SE vector via the explicit post-fit primitives.
magmaan_se <- function(fit) {
  tryCatch({
    info <- infer_information_expected(fit)
    infer_se(infer_vcov_partable(info, fit$partable))
  }, error = function(e) NULL)
}

## min / median / p90 / IQR (seconds) for one bench::mark row.
bench_summary <- function(tm, label) {
  i <- match(label, as.character(tm$expression))
  t <- as.numeric(tm$time[[i]])
  list(min = min(t), median = stats::median(t),
       p90 = unname(stats::quantile(t, 0.9)), iqr = unname(stats::IQR(t)),
       n = length(t))
}

run_case <- function(id) {
  case <- get_case(id)
  data <- read_prepared_data(id)
  model <- read_model(id)

  out <- list(case_id = id, tier = case$tier %||% NA_integer_,
              estimator = case$estimator %||% "ML", status = "PASS",
              note = "")

  mfit <- tryCatch(magmaan_fit_case(case, model, data), error = identity)
  lfit <- tryCatch(lavaan_fit_case(case, model, data, "standard"),
                   error = identity)
  if (inherits(mfit, "error") || inherits(lfit, "error")) {
    out$status <- "ERROR"
    out$note <- if (inherits(mfit, "error")) {
      paste0("magmaan fit failed: ", conditionMessage(mfit))
    } else {
      paste0("lavaan fit failed: ", conditionMessage(lfit))
    }
    return(out)
  }

  l_pt <- lavaan::parTable(lfit)
  out$n_obs <- sum(unname(lavaan::lavInspect(lfit, "nobs")))
  out$n_vars <- length(lavaan::lavNames(lfit, type = "ov"))
  out$npar <- sum(l_pt$free > 0L)
  out$magmaan_converged <- isTRUE(mfit$converged %||% TRUE)
  out$lavaan_converged <- isTRUE(lavaan::lavInspect(lfit, "converged"))

  ## Correctness: reuse magmaan's own partable comparator as the structural +
  ## estimate gate, plus an explicit max-abs estimate diff and an fmin check.
  cmp <- tryCatch(
    lavaan_compare_partable(mfit, l_pt, est_tolerance = EST_TOL),
    error = function(e) list(ok = NA, counts = integer(),
                             failures = conditionMessage(e)))
  est_diff <- est_max_abs_diff(mfit$partable, l_pt)
  l_fmin <- unname(lavaan::fitMeasures(lfit, "fmin"))
  m_fmin <- mfit$fmin %||% NA_real_

  ## chi-square: lavaan reports it directly; magmaan's $fmin is the full ML
  ## discrepancy F (lavaan reports F/2), so the ML chi-square is ntotal * F
  ## (verified == lavaan chisq on the committed references).
  l_chisq <- unname(lavaan::fitMeasures(lfit, "chisq"))
  l_df <- unname(lavaan::fitMeasures(lfit, "df"))
  m_chisq <- if (identical(out$estimator, "ML")) {
    (mfit$ntotal %||% out$n_obs) * m_fmin
  } else {
    NA_real_
  }
  chisq_diff <- abs(m_chisq - l_chisq)

  ## SE agreement (best effort -- not part of the pass/fail gate).
  se_diff <- NA_real_
  m_se <- magmaan_se(mfit)
  if (!is.null(m_se)) {
    m_free <- mfit$partable[mfit$partable$free > 0L, , drop = FALSE]
    m_free <- m_free[order(m_free$free), , drop = FALSE]
    l_free <- l_pt[l_pt$free > 0L, , drop = FALSE]
    l_free <- l_free[order(l_free$free), , drop = FALSE]
    if (length(m_se) == nrow(m_free) && nrow(m_free) == nrow(l_free)) {
      mk <- paste(m_free$group, m_free$op, m_free$lhs, m_free$rhs, sep = "\r")
      lk <- paste(l_free$group, l_free$op, l_free$lhs, l_free$rhs, sep = "\r")
      idx <- match(mk, lk)
      if (!anyNA(idx)) {
        se_diff <- max(abs(m_se - l_free$se[idx]), na.rm = TRUE)
      }
    }
  }

  out$correctness <- list(
    partable_ok = isTRUE(cmp$ok),
    partable_failure_counts = as.list(cmp$counts),
    est_max_abs_diff = est_diff,
    se_max_abs_diff = se_diff,
    fmin_magmaan = m_fmin, fmin_lavaan = l_fmin,
    fmin_note = "magmaan $fmin is the full ML discrepancy F; lavaan reports F/2",
    chisq_magmaan = m_chisq, chisq_lavaan = l_chisq,
    chisq_abs_diff = chisq_diff, df = l_df)

  est_ok <- !is.na(est_diff) && est_diff <= EST_TOL
  if (!est_ok || !out$magmaan_converged) {
    out$status <- "FAIL"
    failed <- c(if (!est_ok) "estimates",
                if (!out$magmaan_converged) "magmaan-convergence")
    out$note <- paste0("correctness gate failed: ", paste(failed, collapse = ", "))
  }

  ## Timing: estimate-only workload, magmaan vs lavaan (se/test = "none").
  timing <- tryCatch({
    tm <- bench::mark(
      magmaan = magmaan_fit_case(case, model, data),
      lavaan = lavaan_fit_case(case, model, data, "estimate"),
      check = FALSE, filter_gc = FALSE,
      iterations = BENCH_ITERATIONS, time_unit = "s")
    m <- bench_summary(tm, "magmaan")
    l <- bench_summary(tm, "lavaan")
    list(magmaan = m, lavaan = l,
         median_speedup_lavaan_over_magmaan = l$median / m$median)
  }, error = function(e) {
    out$note <<- trimws(paste(out$note, "| timing failed:",
                              conditionMessage(e)))
    NULL
  })
  out$timing_seconds <- timing
  out
}

cat(sprintf("magmaan benchmark smoke -- magmaan %s, lavaan %s, %s\n",
            as.character(magmaan::version()),
            as.character(utils::packageVersion("lavaan")),
            R.version.string))
cat(sprintf("workload: estimate-only ML | est tol %.0e | %d cases\n\n",
            EST_TOL, length(ids)))

rows <- lapply(ids, function(id) {
  res <- tryCatch(run_case(id), error = function(e) {
    list(case_id = id, status = "ERROR",
         note = paste0("runner error: ", conditionMessage(e)))
  })
  tm <- res$timing_seconds
  cat(sprintf("  %-22s %-6s n=%-5s est_diff=%-10s magmaan=%-9s lavaan=%-9s%s\n",
              res$case_id, res$status,
              res$n_obs %||% "-",
              if (is.null(res$correctness)) "-" else
                formatC(res$correctness$est_max_abs_diff, format = "g", digits = 3),
              if (is.null(tm)) "-" else sprintf("%.3fms", tm$magmaan$median * 1e3),
              if (is.null(tm)) "-" else sprintf("%.3fms", tm$lavaan$median * 1e3),
              if (nzchar(res$note %||% "")) paste0("  [", res$note, "]") else ""))
  res
})

payload <- list(
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  magmaan_version = as.character(magmaan::version()),
  lavaan_version = as.character(utils::packageVersion("lavaan")),
  r_version = R.version.string,
  blas = unname(extSoftVersion()["BLAS"]),
  workload = "estimate-only ML",
  bench_iterations = BENCH_ITERATIONS,
  tolerances = list(est = EST_TOL),
  cases = rows)

out_path <- file.path(results_dir(),
                       sprintf("smoke_%s.json",
                               format(Sys.time(), "%Y%m%dT%H%M%SZ", tz = "UTC")))
write_json_file(payload, out_path)

n_pass <- sum(vapply(rows, function(r) identical(r$status, "PASS"), logical(1)))
cat(sprintf("\n%d/%d cases PASS  ->  %s\n", n_pass, length(rows), out_path))
