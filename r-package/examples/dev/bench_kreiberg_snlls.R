library(magmaan)
library(lavaan)

core <- magmaan_core

times <- as.integer(Sys.getenv("MAGMAAN_KREIBERG_TIMES", "500"))
opt_profiles <- list(
  strict = list(max_iter = 5000L, ftol = 1e-12, gtol = 1e-8),
  matlab_like = list(max_iter = 400L, ftol = 1e-6, gtol = 1e-6)
)
ceres <- list(max_iter = 500L, ftol = 1e-10, gtol = 1e-7, ptol = 1e-8)

cases <- list(
  list(
    id = "kreiberg_2021_hs_cfa",
    note = "Kreiberg et al. (2021) CFA-paper HS example; same public data/model family.",
    published = list(snlls_iter = 21L, ordinary_iter = 51L),
    model = "visual  =~ x1 + x2 + x3
             textual =~ x4 + x5 + x6
             speed   =~ x7 + x8 + x9",
    data = lavaan::HolzingerSwineford1939
  ),
  list(
    id = "kreiberg_2023_bollen_sem",
    note = "Kreiberg & Zhou (2023) SEM-paper Bollen example; same lavaan data/model.",
    published = list(snlls_iter = 26L, ordinary_iter = 230L),
    model = paste(readLines("benchmarks/cases/bollen_democracy_sem/model.lav"),
                  collapse = "\n"),
    data = lavaan::PoliticalDemocracy
  )
)

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

lavaan_started_spec <- function(case) {
  spec <- model_spec(case$model)
  lav <- lavaan::sem(case$model, data = case$data, estimator = "GLS",
                     do.fit = FALSE)
  lav_pt <- lavaan::parTable(lav)
  free <- spec$partable$free > 0L
  spec$partable$ustart[free] <- lav_pt$start[free]
  spec
}

inf_bounds <- function(spec) {
  npar <- max(spec$partable$free)
  list(lower = rep(-Inf, npar), upper = rep(Inf, npar))
}

lbfgsb_opts <- function(base_opts, iter_cap = base_opts$max_iter) {
  opts <- base_opts
  opts$max_iter <- as.integer(iter_cap)
  opts
}

fit_one <- function(spec, dat, backend, snlls, opts) {
  if (backend == "lbfgsb") {
    if (snlls) return(core$fit_gls_snlls(spec, dat, lbfgsb = opts))
    return(core$fit_gls(spec, dat, lbfgsb = opts,
                        bounds = inf_bounds(spec)))
  }
  if (backend == "ceres") {
    if (snlls) return(core$fit_gls_snlls_ceres(spec, dat, ceres = ceres))
    return(core$fit_gls_ceres(spec, dat, ceres = ceres))
  }
  stop("unknown backend: ", backend, call. = FALSE)
}

recover_snlls_iterations <- function(spec, dat, backend, fit, opts) {
  if (backend != "lbfgsb" || fit$iterations != 0L) return(NULL)
  for (cap in seq_len(200L)) {
    trial <- try_fit(fit_one(spec, dat, backend, TRUE, lbfgsb_opts(opts, cap)))
    if (!inherits(trial, "magmaan_bench_error")) return(cap)
  }
  NULL
}

measure_spec <- function(case, spec, start_scheme, opt_profile, opts, setup_usec) {
  dat <- df_to_data(case$data, spec, scaling = "n-1")

  rows <- list()
  for (backend in c("lbfgsb", "ceres")) {
    for (snlls in c(FALSE, TRUE)) {
      fit <- try_fit(fit_one(spec, dat, backend, snlls, opts))
      label <- if (snlls) "SNLLS" else "ordinary"
      if (inherits(fit, "magmaan_bench_error")) {
        rows[[length(rows) + 1L]] <- data.frame(
          case = case$id,
          start_scheme = start_scheme,
          opt_profile = opt_profile,
          method = label,
          backend = backend,
          iterations = NA_integer_,
          fmin = NA_real_,
          setup_usec = setup_usec,
          fit_usec = NA_real_,
          iteration_source = NA_character_,
          error = fit$error
        )
      } else {
        fit_usec <- elapsed_usec(times, function() {
          fit_one(spec, dat, backend, snlls, opts)$fmin
        })
        recovered <- if (snlls) recover_snlls_iterations(spec, dat, backend, fit, opts) else NULL
        rows[[length(rows) + 1L]] <- data.frame(
          case = case$id,
          start_scheme = start_scheme,
          opt_profile = opt_profile,
          method = label,
          backend = backend,
          iterations = if (is.null(recovered)) fit$iterations else recovered,
          fmin = fit$fmin,
          setup_usec = setup_usec,
          fit_usec = fit_usec,
          iteration_source = if (is.null(recovered)) "reported" else "cap_recovered",
          error = NA_character_
        )
      }
    }
  }
  do.call(rbind, rows)
}

measure_case <- function(case) {
  setup_usec <- elapsed_usec(times, function() {
    spec <- model_spec(case$model)
    df_to_data(case$data, spec, scaling = "n-1")
  })

  spec <- model_spec(case$model)
  lav_spec <- lavaan_started_spec(case)

  rows <- list()
  for (profile_name in names(opt_profiles)) {
    opts <- opt_profiles[[profile_name]]
    rows[[length(rows) + 1L]] <- measure_spec(case, spec, "magmaan_default",
                                              profile_name, opts, setup_usec)
    rows[[length(rows) + 1L]] <- measure_spec(case, lav_spec, "lavaan_explicit",
                                              profile_name, opts, setup_usec)
  }
  do.call(rbind, rows)
}

published_rows <- do.call(rbind, lapply(cases, function(case) {
  data.frame(
    case = case$id,
    paper = case$note,
    published_snlls_iter = case$published$snlls_iter,
    published_ordinary_iter = case$published$ordinary_iter
  )
}))

cat("\nPublished reference points\n")
print(published_rows, row.names = FALSE)

cat("\nMagmaan pseudo-replication: GLS, setup once, fit-only timing\n")
out <- do.call(rbind, lapply(cases, measure_case))
snlls_key <- with(out, paste(case, start_scheme, opt_profile, backend, method,
                             sep = "\r"))
baseline_key <- with(out, paste(case, start_scheme, opt_profile, backend,
                                "SNLLS", sep = "\r"))
out$iter_ratio <- out$iterations / out$iterations[match(baseline_key, snlls_key)]
out$time_ratio <- out$fit_usec / out$fit_usec[match(baseline_key, snlls_key)]
print(out, digits = 6, row.names = FALSE)

cat("\nNotes\n")
cat("* Timings are magmaan-only and do not reproduce MATLAB BFGS finite-difference gradients.\n")
cat("* matlab_like uses max_iter=400, ftol=1e-6, gtol=1e-6; it is still LBFGS/LBFGS-B, not MATLAB dense BFGS.\n")
cat("* Ordinary GLS passes explicit +/-Inf bounds so magmaan dispatches to LBFGS-B and reports solver iterations reliably.\n")
cat("* cap_recovered means unbounded SNLLS accepted a salvaged L-BFGS iterate; the count is the smallest max_iter cap that succeeds.\n")
cat("* Kreiberg & Zhou's Geiser depression example is not run: the data are not vendored here, and the model is a higher-order/indicator-specific latent-state model.\n")
cat("* The HS CFA case is the supported public-data analogue from Kreiberg et al. (2021).\n")
