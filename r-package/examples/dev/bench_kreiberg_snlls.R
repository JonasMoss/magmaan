library(magmaan)

core <- magmaan_core

times <- as.integer(Sys.getenv("MAGMAAN_KREIBERG_TIMES", "100"))
lbfgsb <- list(max_iter = 5000L, ftol = 1e-12, gtol = 1e-8)
ceres <- list(max_iter = 500L, ftol = 1e-10, gtol = 1e-7, ptol = 1e-8)

bollen_model <- "ind60 =~ x1 + x2 + x3
dem60 =~ y1 + y2 + y3 + y4
dem65 =~ y5 + y6 + y7 + y8

dem60 ~ ind60
dem65 ~ ind60 + dem60

y1 ~~ y5
y2 ~~ y4 + y6
y3 ~~ y7
y4 ~~ y8
y6 ~~ y8"

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
    model = bollen_model,
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
    if (snlls) return(core$fit_gls_snlls(spec, dat, optimizer = "lbfgs", control = opts))
    return(core$fit_gls(spec, dat, optimizer = "lbfgs", control = opts,
                        bounds = inf_bounds(spec)))
  }
  optimizer <- gsub("_", "-", backend, fixed = TRUE)
  control <- if (startsWith(optimizer, "ceres")) ceres else opts
  if (snlls) {
    return(core$fit_gls_snlls(spec, dat, optimizer = optimizer,
                              control = control))
  }
  core$fit_gls(spec, dat, optimizer = optimizer, control = control)
}

recover_snlls_iterations <- function(spec, dat, backend, fit, opts) {
  if (backend != "lbfgsb" || fit$iterations != 0L) return(NULL)
  for (cap in seq_len(200L)) {
    trial <- try_fit(fit_one(spec, dat, backend, TRUE, lbfgsb_opts(opts, cap)))
    if (!inherits(trial, "magmaan_bench_error")) return(cap)
  }
  NULL
}

measure_spec <- function(case, spec, setup_usec) {
  dat <- df_to_data(case$data, spec, scaling = "n-1")

  rows <- list()
  backends <- c("lbfgsb", "ceres", "ceres_bfgs", "port", "port_nls",
                "nlopt_slsqp", "nlopt_bobyqa", "nlopt_tnewton",
                "nlopt_var2", "nlopt_lbfgs")
  jobs <- as.vector(t(outer(backends, c(FALSE, TRUE), Vectorize(function(b, s) {
    list(list(backend = b, snlls = s))
  }))), mode = "list")
  for (job in jobs) {
    backend <- job$backend
    snlls <- job$snlls
    fit <- try_fit(fit_one(spec, dat, backend, snlls, lbfgsb))
    label <- if (snlls) "SNLLS" else "ordinary"
    if (inherits(fit, "magmaan_bench_error")) {
      rows[[length(rows) + 1L]] <- data.frame(
        case = case$id,
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
        fit_one(spec, dat, backend, snlls, lbfgsb)$fmin
      })
      recovered <- if (snlls) recover_snlls_iterations(spec, dat, backend, fit, lbfgsb) else NULL
      rows[[length(rows) + 1L]] <- data.frame(
        case = case$id,
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
  do.call(rbind, rows)
}

measure_case <- function(case) {
  setup_usec <- elapsed_usec(times, function() {
    spec <- model_spec(case$model)
    df_to_data(case$data, spec, scaling = "n-1")
  })

  spec <- model_spec(case$model)
  measure_spec(case, spec, setup_usec)
}

ratio_or_na <- function(num, den) {
  out <- num / den
  out[!is.finite(out)] <- NA_real_
  out
}

ordinary_snlls_ratios <- function(out) {
  comparable <- out[is.na(out$error), ]
  rows <- list()
  for (case in unique(comparable$case)) {
    for (backend in unique(comparable$backend[comparable$case == case])) {
      ordinary <- comparable[comparable$case == case &
                               comparable$backend == backend &
                               comparable$method == "ordinary", ]
      snlls <- comparable[comparable$case == case &
                            comparable$backend == backend &
                            comparable$method == "SNLLS", ]
      if (!nrow(ordinary) || !nrow(snlls)) next
      rows[[length(rows) + 1L]] <- data.frame(
        case = case,
        backend = backend,
        iter_ratio = ratio_or_na(ordinary$iterations, snlls$iterations),
        time_ratio = ratio_or_na(ordinary$fit_usec, snlls$fit_usec)
      )
    }
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
print(out, digits = 6, row.names = FALSE)

cat("\nComparable ordinary/SNLLS ratios\n")
print(ordinary_snlls_ratios(out), digits = 6, row.names = FALSE)

cat("\nNotes\n")
cat("* Timings are magmaan-only and use magmaan starting values.\n")
cat("* The lbfgsb row is the accepted R optimizer string \"lbfgs\" with explicit +/-Inf bounds on ordinary GLS, preserving the historical LBFGS-B comparison row.\n")
cat("* Other backend labels are accepted R optimizer strings with underscores replacing dashes for data-frame readability.\n")
cat("* Ratios are printed only for successful backends with both ordinary and SNLLS rows.\n")
cat("* cap_recovered means unbounded SNLLS accepted a salvaged L-BFGS iterate; the count is the smallest max_iter cap that succeeds.\n")
cat("* Kreiberg & Zhou's Geiser depression example is not run: the data are not vendored here, and the model is a higher-order/indicator-specific latent-state model.\n")
cat("* The HS CFA case is the supported public-data analogue from Kreiberg et al. (2021).\n")
