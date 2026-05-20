library(magmaan)

core <- magmaan_core
times <- as.integer(Sys.getenv("MAGMAAN_HARD_SNLLS_TIMES", "100"))
lbfgsb <- list(max_iter = 2000L, ftol = 1e-10, gtol = 1e-7)
ceres <- list(max_iter = 500L, ftol = 1e-10, gtol = 1e-7, ptol = 1e-8)

elapsed_usec <- function(n, fun) {
  invisible(fun())
  gc(FALSE)
  start <- proc.time()[["elapsed"]]
  for (i in seq_len(n)) fun()
  1e6 * (proc.time()[["elapsed"]] - start) / n
}

try_fit <- function(expr) {
  tryCatch(expr, error = function(e) structure(list(error = conditionMessage(e)),
                                        class = "magmaan_bench_error"))
}

inf_bounds <- function(spec) {
  npar <- max(spec$partable$free)
  list(lower = rep(-Inf, npar), upper = rep(Inf, npar))
}

fit_one <- function(spec, dat, estimator, backend, snlls) {
  if (identical(estimator, "ULS") && identical(backend, "lbfgsb")) {
    if (snlls) return(core$fit_uls_snlls(spec, dat, optimizer = "lbfgs", control = lbfgsb))
    return(core$fit_uls(spec, dat, optimizer = "lbfgs", control = lbfgsb,
                        bounds = inf_bounds(spec)))
  }
  if (identical(estimator, "ULS") && identical(backend, "ceres")) {
    if (snlls) return(core$fit_uls_snlls(spec, dat, optimizer = "ceres", control = ceres))
    return(core$fit_uls(spec, dat, optimizer = "ceres", control = ceres))
  }
  if (identical(estimator, "GLS") && identical(backend, "lbfgsb")) {
    if (snlls) return(core$fit_gls_snlls(spec, dat, optimizer = "lbfgs", control = lbfgsb))
    return(core$fit_gls(spec, dat, optimizer = "lbfgs", control = lbfgsb,
                        bounds = inf_bounds(spec)))
  }
  if (identical(estimator, "GLS") && identical(backend, "ceres")) {
    if (snlls) return(core$fit_gls_snlls(spec, dat, optimizer = "ceres", control = ceres))
    return(core$fit_gls(spec, dat, optimizer = "ceres", control = ceres))
  }
  if (identical(estimator, "GLS") && identical(backend, "ceres_bfgs")) {
    if (!snlls) stop("Ceres BFGS is exposed only for SNLLS", call. = FALSE)
    return(core$fit_gls_snlls(spec, dat, optimizer = "ceres-bfgs", control = ceres))
  }
  stop("unsupported estimator/backend: ", estimator, "/", backend,
       call. = FALSE)
}

case_gleser_mtmm <- function() {
  data(Gleser, package = "psych")
  model <- "t1 =~ J11 + J12
t2 =~ J21 + J22
t3 =~ J31 + J32
t4 =~ J41 + J42
t5 =~ J51 + J52
t6 =~ J61 + J62
j1 =~ J11 + J21 + J31 + J41 + J51 + J61
j2 =~ J12 + J22 + J32 + J42 + J52 + J62
t1 ~~ 0*j1 + 0*j2
t2 ~~ 0*j1 + 0*j2
t3 ~~ 0*j1 + 0*j2
t4 ~~ 0*j1 + 0*j2
t5 ~~ 0*j1 + 0*j2
t6 ~~ 0*j1 + 0*j2
j1 ~~ 0*j2"
  spec <- model_spec(model, std_lv = TRUE)
  list(
    id = "gleser_1965_mtmm_judges",
    note = paste(
      "Gleser, Cronbach, and Rajaratnam judge-by-symptom data from psych::Gleser;",
      "cast as a trait plus judge-method CFA. ULS is used because n=12 makes",
      "the 12-variable sample covariance singular for GLS."
    ),
    estimator = "ULS",
    spec = spec,
    dat = df_to_data(Gleser, spec, scaling = "n-1")
  )
}

case_semfindr_heywood <- function(drop_case_1 = FALSE) {
  if (!requireNamespace("semfindr", quietly = TRUE)) return(NULL)
  data(cfa_dat_heywood, package = "semfindr")
  df <- if (drop_case_1) cfa_dat_heywood[-1, ] else cfa_dat_heywood
  model <- "f1 =~ x1 + x2 + x3
f2 =~ x4 + x5 + x6"
  spec <- model_spec(model)
  suffix <- if (drop_case_1) "without_case_1" else "with_heywood_case"
  list(
    id = paste("semfindr_cfa_dat_heywood", suffix, sep = "_"),
    note = paste(
      "semfindr::cfa_dat_heywood two-factor CFA;",
      if (drop_case_1) "case 1 removed for the admissible comparison."
      else "full data include the influential Heywood case."
    ),
    estimator = "GLS",
    spec = spec,
    dat = df_to_data(df, spec, scaling = "n-1")
  )
}

measure_case <- function(case) {
  rows <- list()
  jobs <- list(
    list(backend = "lbfgsb", snlls = FALSE),
    list(backend = "lbfgsb", snlls = TRUE),
    list(backend = "ceres", snlls = FALSE),
    list(backend = "ceres", snlls = TRUE)
  )
  if (identical(case$estimator, "GLS")) {
    jobs[[length(jobs) + 1L]] <- list(backend = "ceres_bfgs", snlls = TRUE)
  }
  for (job in jobs) {
    backend <- job$backend
    snlls <- job$snlls
    label <- if (snlls) "SNLLS" else "ordinary"
    fit <- try_fit(fit_one(case$spec, case$dat, case$estimator, backend, snlls))
    if (inherits(fit, "magmaan_bench_error")) {
      rows[[length(rows) + 1L]] <- data.frame(
        case = case$id,
        estimator = case$estimator,
        backend = backend,
        method = label,
        iterations = NA_integer_,
        fmin = NA_real_,
        fit_usec = NA_real_,
        iteration_source = NA_character_,
        error = fit$error
      )
    } else {
      fit_usec <- elapsed_usec(times, function() {
        fit_one(case$spec, case$dat, case$estimator, backend, snlls)$fmin
      })
      rows[[length(rows) + 1L]] <- data.frame(
        case = case$id,
        estimator = case$estimator,
        backend = backend,
        method = label,
        iterations = fit$iterations,
        fmin = fit$fmin,
        fit_usec = fit_usec,
        iteration_source = if (fit$iterations == 0L) "reported_zero" else "reported",
        error = NA_character_
      )
    }
  }
  do.call(rbind, rows)
}

ratio_or_na <- function(num, den) {
  out <- num / den
  out[!is.finite(out)] <- NA_real_
  out
}

ordinary_snlls_ratios <- function(out) {
  rows <- list()
  for (case in unique(out$case)) {
    for (backend in unique(out$backend[out$case == case])) {
      ordinary <- out[out$case == case &
                        out$backend == backend &
                        out$method == "ordinary", ]
      snlls <- out[out$case == case &
                     out$backend == backend &
                     out$method == "SNLLS", ]
      if (!nrow(ordinary) || !nrow(snlls)) next
      rows[[length(rows) + 1L]] <- data.frame(
        case = case,
        backend = backend,
        iter_ratio = ratio_or_na(ordinary$iterations, snlls$iterations),
        time_ratio = ratio_or_na(ordinary$fit_usec, snlls$fit_usec)
      )
    }
  }
  if (!length(rows)) return(data.frame())
  do.call(rbind, rows)
}

cases <- list(
  case_gleser_mtmm(),
  case_semfindr_heywood(FALSE),
  case_semfindr_heywood(TRUE)
)
cases <- Filter(Negate(is.null), cases)

cat("\nHard SNLLS probes\n")
case_notes <- do.call(rbind, lapply(cases, function(case) {
  data.frame(case = case$id, estimator = case$estimator, note = case$note)
}))
print(case_notes, row.names = FALSE)

if (!requireNamespace("semfindr", quietly = TRUE)) {
  cat("\nSkipped semfindr Heywood cases: package 'semfindr' is not installed.\n")
}

cat("\nMagmaan fits\n")
out <- do.call(rbind, lapply(cases, measure_case))
print(out, digits = 6, row.names = FALSE)

cat("\nComparable ordinary/SNLLS ratios\n")
print(ordinary_snlls_ratios(out), digits = 6, row.names = FALSE)

cat("\nNotes\n")
cat("* The Gleser MTMM/G-study case is ULS-only here because GLS needs a positive-definite sample covariance.\n")
cat("* semfindr::cfa_dat_heywood is an optional installed-package case; install semfindr to enable it.\n")
cat("* reported_zero means the backend returned a usable estimate but not a reliable iteration count.\n")
cat("* ceres_bfgs is SNLLS-only and currently exposed for GLS probes only.\n")
cat("* Trust-region and NLopt backends exist in C++ checks but are not exposed on this R dev surface.\n")
cat("* These are convergence and boundary probes first; wall-time ratios are secondary.\n")
