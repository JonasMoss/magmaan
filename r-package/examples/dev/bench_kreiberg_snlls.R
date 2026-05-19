library(magmaan)
library(lavaan)

core <- magmaan_core

times <- as.integer(Sys.getenv("MAGMAAN_KREIBERG_TIMES", "500"))
lbfgsb <- list(max_iter = 5000L, ftol = 1e-12, gtol = 1e-8)
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

fit_one <- function(spec, dat, backend, snlls) {
  if (backend == "lbfgsb") {
    if (snlls) return(core$fit_gls_snlls(spec, dat, lbfgsb = lbfgsb))
    return(core$fit_gls(spec, dat, lbfgsb = lbfgsb))
  }
  if (backend == "ceres") {
    if (snlls) return(core$fit_gls_snlls_ceres(spec, dat, ceres = ceres))
    return(core$fit_gls_ceres(spec, dat, ceres = ceres))
  }
  stop("unknown backend: ", backend, call. = FALSE)
}

measure_case <- function(case) {
  setup_usec <- elapsed_usec(times, function() {
    spec <- model_spec(case$model)
    df_to_data(case$data, spec, scaling = "n-1")
  })

  spec <- model_spec(case$model)
  dat <- df_to_data(case$data, spec, scaling = "n-1")

  rows <- list()
  for (backend in c("lbfgsb", "ceres")) {
    for (snlls in c(FALSE, TRUE)) {
      fit <- try_fit(fit_one(spec, dat, backend, snlls))
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
          error = fit$error
        )
      } else {
        fit_usec <- elapsed_usec(times, function() {
          fit_one(spec, dat, backend, snlls)$fmin
        })
        rows[[length(rows) + 1L]] <- data.frame(
          case = case$id,
          method = label,
          backend = backend,
          iterations = fit$iterations,
          fmin = fit$fmin,
          setup_usec = setup_usec,
          fit_usec = fit_usec,
          error = NA_character_
        )
      }
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
snlls_key <- with(out, paste(case, backend, method, sep = "\r"))
baseline_key <- with(out, paste(case, backend, "SNLLS", sep = "\r"))
out$iter_ratio <- out$iterations / out$iterations[match(baseline_key, snlls_key)]
out$time_ratio <- out$fit_usec / out$fit_usec[match(baseline_key, snlls_key)]
print(out, digits = 6, row.names = FALSE)

cat("\nNotes\n")
cat("* Timings are magmaan-only and do not reproduce MATLAB BFGS finite-difference gradients.\n")
cat("* Kreiberg & Zhou's Geiser depression example is not run: the data are not vendored here, and the model is a higher-order/indicator-specific latent-state model.\n")
cat("* The HS CFA case is the supported public-data analogue from Kreiberg et al. (2021).\n")
cat("* Bollen ordinary L-BFGS-B may report 0 iterations with magmaan's current GLS starts.\n")
