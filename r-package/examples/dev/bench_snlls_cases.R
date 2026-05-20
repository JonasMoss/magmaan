library(magmaan)

core <- magmaan_core

times <- as.integer(Sys.getenv(
  "MAGMAAN_SNLLS_TIMES",
  Sys.getenv("MAGMAAN_KREIBERG_TIMES",
             Sys.getenv("MAGMAAN_HARD_SNLLS_TIMES", "100"))
))
opts <- list(max_iter = 5000L, ftol = 1e-12, gtol = 1e-8)
ceres_opts <- list(max_iter = 500L, ftol = 1e-10, gtol = 1e-7, ptol = 1e-8)

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

short_error <- function(x, n = 110L) {
  out <- ifelse(is.na(x), NA_character_, x)
  too_long <- !is.na(out) & nchar(out) > n
  out[too_long] <- paste0(substr(out[too_long], 1L, n - 3L), "...")
  out
}

backend_specs <- list(
  lbfgsb = list(optimizer = "nlopt-lbfgs", control = opts),
  nlminb = list(optimizer = "port", control = opts),
  ceres  = list(optimizer = "ceres", control = ceres_opts)
)

fit_one <- function(case, backend, snlls) {
  spec <- backend_specs[[backend]]
  family <- paste(case$estimator, if (snlls) "SNLLS" else "full")
  fun <- switch(family,
                "ULS full" = core$fit_uls,
                "ULS SNLLS" = core$fit_uls_snlls,
                "GLS full" = core$fit_gls,
                "GLS SNLLS" = core$fit_gls_snlls,
                stop("unsupported estimator/method: ", family, call. = FALSE))
  do.call(fun, list(case$spec, case$dat, optimizer = spec$optimizer,
                    control = spec$control))
}

case_hs <- function() {
  model <- "visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9"
  spec <- model_spec(model)
  list(
    case = "hs",
    estimator = "GLS",
    published_full = 51L,
    published_snlls = 21L,
    note = "Kreiberg et al. (2021) Holzinger-Swineford CFA analogue.",
    spec = spec,
    dat = df_to_data(lavaan::HolzingerSwineford1939, spec, scaling = "n-1")
  )
}

case_bollen <- function() {
  spec <- model_spec(bollen_model)
  list(
    case = "bollen",
    estimator = "GLS",
    published_full = 230L,
    published_snlls = 26L,
    note = "Kreiberg and Zhou (2023) Bollen democracy SEM analogue.",
    spec = spec,
    dat = df_to_data(lavaan::PoliticalDemocracy, spec, scaling = "n-1")
  )
}

case_gleser <- function() {
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
    case = "gleser",
    estimator = "ULS",
    published_full = NA_integer_,
    published_snlls = NA_integer_,
    note = "Gleser/Cronbach/Rajaratnam MTMM judge data; ULS because S is singular.",
    spec = spec,
    dat = df_to_data(Gleser, spec, scaling = "n-1")
  )
}

case_heywood <- function(drop_case_1 = FALSE) {
  if (!requireNamespace("semfindr", quietly = TRUE)) return(NULL)
  data(cfa_dat_heywood, package = "semfindr")
  df <- if (drop_case_1) cfa_dat_heywood[-1, ] else cfa_dat_heywood
  model <- "f1 =~ x1 + x2 + x3
f2 =~ x4 + x5 + x6"
  spec <- model_spec(model)
  list(
    case = if (drop_case_1) "heywood_no1" else "heywood",
    estimator = "GLS",
    published_full = NA_integer_,
    published_snlls = NA_integer_,
    note = if (drop_case_1) "semfindr Heywood CFA, case 1 removed."
           else "semfindr Heywood CFA, full data.",
    spec = spec,
    dat = df_to_data(df, spec, scaling = "n-1")
  )
}

measure_case <- function(case) {
  rows <- list()
  for (backend in names(backend_specs)) {
    for (snlls in c(FALSE, TRUE)) {
      method <- if (snlls) "snlls" else "full"
      fit <- try_fit(fit_one(case, backend, snlls))
      if (inherits(fit, "magmaan_bench_error")) {
        rows[[length(rows) + 1L]] <- data.frame(
          case = case$case,
          est = case$estimator,
          backend = backend,
          method = method,
          iter = NA_integer_,
          fmin = NA_real_,
          usec = NA_real_,
          error = fit$error
        )
      } else {
        usec <- elapsed_usec(times, function() {
          fit_one(case, backend, snlls)$fmin
        })
        rows[[length(rows) + 1L]] <- data.frame(
          case = case$case,
          est = case$estimator,
          backend = backend,
          method = method,
          iter = fit$iterations,
          fmin = fit$fmin,
          usec = usec,
          error = NA_character_
        )
      }
    }
  }
  do.call(rbind, rows)
}

ratio_or_na <- function(num, den) {
  out <- num / den
  out[!is.finite(out)] <- NA_real_
  out
}

snlls_ratios <- function(out) {
  ok <- out[is.na(out$error), ]
  rows <- list()
  for (case in unique(ok$case)) {
    for (backend in unique(ok$backend[ok$case == case])) {
      full <- ok[ok$case == case & ok$backend == backend &
                   ok$method == "full", ]
      snlls <- ok[ok$case == case & ok$backend == backend &
                    ok$method == "snlls", ]
      if (!nrow(full) || !nrow(snlls)) next
      rows[[length(rows) + 1L]] <- data.frame(
        case = case,
        backend = backend,
        iter_x = ratio_or_na(full$iter, snlls$iter),
        time_x = ratio_or_na(full$usec, snlls$usec)
      )
    }
  }
  if (!length(rows)) return(data.frame())
  do.call(rbind, rows)
}

cases <- list(
  case_hs(),
  case_bollen(),
  case_gleser(),
  case_heywood(FALSE),
  case_heywood(TRUE)
)
cases <- Filter(Negate(is.null), cases)

cat("\nSNLLS cases\n")
case_notes <- do.call(rbind, lapply(cases, function(case) {
  data.frame(case = case$case, est = case$estimator, note = case$note)
}))
print(case_notes, row.names = FALSE)

published <- do.call(rbind, lapply(cases, function(case) {
  data.frame(case = case$case, paper_full = case$published_full,
             paper_snlls = case$published_snlls)
}))
published <- published[!is.na(published$paper_full), ]
if (nrow(published)) {
  cat("\nPaper iteration reference\n")
  print(published, row.names = FALSE)
}

if (!requireNamespace("semfindr", quietly = TRUE)) {
  cat("\nSkipped semfindr Heywood cases: package 'semfindr' is not installed.\n")
}

cat("\nMagmaan fits\n")
out <- do.call(rbind, lapply(cases, measure_case))
print(out[is.na(out$error), c("case", "est", "backend", "method",
                              "iter", "fmin", "usec")],
      digits = 6, row.names = FALSE)

errs <- out[!is.na(out$error), c("case", "est", "backend", "method", "error")]
if (nrow(errs)) {
  cat("\nErrors\n")
  errs$error <- short_error(errs$error)
  print(errs, row.names = FALSE)
}

cat("\nFull/SNLLS ratios\n")
print(snlls_ratios(out), digits = 6, row.names = FALSE)

cat("\nNotes\n")
cat("* backend=lbfgsb is optimizer=\"nlopt-lbfgs\", a bounded-capable quasi-Newton comparator.\n")
cat("* backend=nlminb is optimizer=\"port\", the PORT scalar trust-region backend adjacent to R's nlminb family.\n")
cat("* backend=ceres is optimizer=\"ceres\", the Ceres least-squares backend.\n")
cat("* Timing repeats use MAGMAAN_SNLLS_TIMES, falling back to the older benchmark env vars.\n")
