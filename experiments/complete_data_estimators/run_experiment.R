#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1L]]$ofile %||% "run_experiment.R", mustWork = FALSE)
}

parse_args <- function(args) {
  out <- list(
    reps = 10L,
    backend = "nlopt-lbfgs",
    case_regex = NULL,
    include_observed = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--backend NAME] [--cases REGEX] [--include-observed]\n",
        "\n",
        "Defaults: --reps 10 --backend nlopt-lbfgs\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--reps") {
      i <- i + 1L
      if (i > length(args)) stop("--reps requires a value", call. = FALSE)
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(arg, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", arg))
    } else if (arg == "--backend") {
      i <- i + 1L
      if (i > length(args)) stop("--backend requires a value", call. = FALSE)
      out$backend <- args[[i]]
    } else if (startsWith(arg, "--backend=")) {
      out$backend <- sub("^--backend=", "", arg)
    } else if (arg == "--cases") {
      i <- i + 1L
      if (i > length(args)) stop("--cases requires a value", call. = FALSE)
      out$case_regex <- args[[i]]
    } else if (startsWith(arg, "--cases=")) {
      out$case_regex <- sub("^--cases=", "", arg)
    } else if (arg == "--include-observed") {
      out$include_observed <- TRUE
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  out$backend <- trimws(out$backend)
  if (!nzchar(out$backend)) stop("--backend must be nonempty", call. = FALSE)
  out
}

require_pkg <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("required R package is not installed: ", package, call. = FALSE)
  }
  invisible(TRUE)
}

matrix_from_json <- function(x) {
  if (is.null(x)) return(NULL)
  out <- do.call(rbind, lapply(x, function(row) as.numeric(unlist(row, use.names = FALSE))))
  storage.mode(out) <- "double"
  out
}

vector_from_json <- function(x) {
  if (is.null(x)) return(NULL)
  as.numeric(unlist(x, use.names = FALSE))
}

find_case <- function(path, id) {
  ref <- jsonlite::read_json(path, simplifyVector = FALSE)
  hits <- Filter(function(x) identical(x$id %||% "", id), ref$cases %||% list())
  if (!length(hits)) return(NULL)
  hits[[1L]]
}

fixture_value <- function(case, field) {
  value <- case[[field]]
  if (!is.null(value)) return(value)
  ml <- case$fits$ML
  if (!is.null(ml)) return(ml[[field]])
  NULL
}

normalize_case <- function(manifest_case, fixture_case) {
  sample_cov <- fixture_value(fixture_case, "sample_cov")
  n_obs <- fixture_value(fixture_case, "n_obs")
  list(
    id = manifest_case$id,
    source = manifest_case$source,
    source_case_id = manifest_case$source_case_id,
    label = manifest_case$label %||% fixture_case$name %||% fixture_case$title %||% fixture_case$label %||% manifest_case$id,
    family = manifest_case$family %||% fixture_case$family %||% fixture_case$model_kind %||% "",
    observed_only = isTRUE(manifest_case$observed_only),
    strict_parity = isTRUE(manifest_case$strict_parity),
    model = fixture_case$model,
    meanstructure = isTRUE(fixture_case$meanstructure),
    fixed_x = isTRUE(fixture_case$fixed_x %||% TRUE),
    model_type = if (identical(fixture_case$lavaan_function %||% "", "growth") ||
                     identical(fixture_case$model_kind %||% "", "growth")) "growth" else "sem",
    n_obs = as.integer(n_obs),
    sample_cov = matrix_from_json(sample_cov),
    sample_mean = vector_from_json(fixture_value(fixture_case, "sample_mean"))
  )
}

usable_case <- function(case, include_observed = FALSE) {
  !is.null(case) && !is.null(case$model) &&
    !is.null(case$sample_cov) && !is.na(case$n_obs) &&
    (include_observed || !isTRUE(case$observed_only)) &&
    (include_observed || grepl("=~", case$model, fixed = TRUE))
}

load_experiment_cases <- function(fixtures_dir, case_regex = NULL,
                                  include_observed = FALSE) {
  manifest_path <- file.path(fixtures_dir, "textbook_corpus", "manifest.json")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  candidates <- Filter(function(x) {
    identical(x$measurement_kind %||% "", "continuous")
  }, manifest$cases)
  if (!is.null(case_regex)) {
    candidates <- Filter(function(x) grepl(case_regex, x$id), candidates)
  }

  out <- list()
  for (mc in candidates) {
    loaded <- NULL
    for (rel in mc$oracle_files %||% character()) {
      path <- file.path(fixtures_dir, rel)
      if (!file.exists(path)) next
      fc <- find_case(path, mc$source_case_id)
      if (!is.null(fc)) {
        candidate <- normalize_case(mc, fc)
        if (usable_case(candidate, include_observed)) {
          loaded <- candidate
          break
        }
        loaded <- loaded %||% candidate
      }
    }
    if (usable_case(loaded, include_observed)) {
      out[[loaded$id]] <- loaded
    }
  }
  out
}

sample_stats <- function(case) {
  out <- list(S = list(case$sample_cov), nobs = as.integer(case$n_obs))
  if (!is.null(case$sample_mean)) out$mean <- list(case$sample_mean)
  out
}

build_spec <- function(case) {
  magmaan::model_spec(
    case$model,
    fixed_x = case$fixed_x,
    meanstructure = case$meanstructure,
    model_type = case$model_type
  )
}

fit_function <- function(estimator) {
  switch(estimator,
         NT = magmaan::magmaan_core$fit_ml,
         ULS = magmaan::magmaan_core$fit_uls,
         GLS = magmaan::magmaan_core$fit_gls,
         stop("unknown estimator: ", estimator, call. = FALSE))
}

safe_chisq <- function(data, fit) {
  tryCatch(
    as.numeric(magmaan::magmaan_core$infer_chi2_stat(data, fit$fmin)),
    error = function(e) NA_real_
  )
}

safe_df <- function(data, fit) {
  tryCatch(
    as.integer(magmaan::magmaan_core$infer_df_stat(fit$partable, data)),
    error = function(e) NA_integer_
  )
}

fit_case_once <- function(case, spec, data, estimator, rep, backend, control) {
  fn <- fit_function(estimator)
  start <- Sys.time()
  fit <- tryCatch(
    fn(spec, data, optimizer = backend, control = control),
    error = identity
  )
  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  if (inherits(fit, "error")) {
    return(list(
      row = new_fit_row(case, estimator, rep, "ERROR",
                        error = conditionMessage(fit),
                        elapsed_sec = elapsed),
      fit = NULL,
      implied = NULL
    ))
  }
  implied <- tryCatch(magmaan::magmaan_core$model_implied(fit), error = identity)
  if (inherits(implied, "error")) {
    return(list(
      row = new_fit_row(case, estimator, rep, "ERROR",
                        error = paste("model_implied:", conditionMessage(implied)),
                        fit = fit, data = data, elapsed_sec = elapsed),
      fit = fit,
      implied = NULL
    ))
  }
  list(
    row = new_fit_row(case, estimator, rep, "OK", fit = fit, data = data,
                      elapsed_sec = elapsed),
    fit = fit,
    implied = implied
  )
}

new_fit_row <- function(case, estimator, rep, status, error = "",
                        fit = NULL, data = NULL, elapsed_sec = NA_real_) {
  diag <- fit$diagnostics %||% list()
  data.frame(
    case_id = case$id,
    source = case$source,
    source_case_id = case$source_case_id,
    label = case$label,
    family = case$family,
    strict_parity = case$strict_parity,
    estimator = estimator,
    replicate = as.integer(rep),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    fmin = as.numeric(fit$fmin %||% NA_real_),
    naive_chisq = if (is.null(fit) || is.null(data)) NA_real_ else safe_chisq(data, fit),
    df = if (is.null(fit) || is.null(data)) NA_integer_ else safe_df(data, fit),
    iterations = as.integer(fit$iterations %||% NA_integer_),
    f_evals = as.integer(fit$f_evals %||% NA_integer_),
    g_evals = as.integer(fit$g_evals %||% NA_integer_),
    optimizer_status = as.character(fit$optimizer_status %||% NA_character_),
    converged = as.logical(fit$converged %||% NA),
    grad_norm = as.numeric(fit$grad_norm %||% NA_real_),
    npar = as.integer(fit$npar %||% NA_integer_),
    ntotal = as.integer(fit$ntotal %||% case$n_obs),
    sigma_pd_all = as.logical(diag$sigma_pd_all %||% NA),
    stringsAsFactors = FALSE
  )
}

free_estimates <- function(fit) {
  pt <- fit$partable
  keep <- pt$free > 0L & is.finite(pt$est)
  data.frame(
    key = paste(pt$group[keep], pt$op[keep], pt$lhs[keep], pt$rhs[keep], sep = "\r"),
    est = as.numeric(pt$est[keep]),
    stringsAsFactors = FALSE
  )
}

parameter_diff <- function(nt_fit, other_fit) {
  nt <- free_estimates(nt_fit)
  other <- free_estimates(other_fit)
  idx <- match(nt$key, other$key)
  ok <- !is.na(idx)
  if (!any(ok)) {
    return(c(n_common = 0, max_abs = NA_real_, rms = NA_real_,
             max_rel = NA_real_))
  }
  d <- other$est[idx[ok]] - nt$est[ok]
  c(
    n_common = sum(ok),
    max_abs = max(abs(d), na.rm = TRUE),
    rms = sqrt(mean(d^2, na.rm = TRUE)),
    max_rel = max(abs(d) / pmax(1, abs(nt$est[ok])), na.rm = TRUE)
  )
}

max_abs_nested_diff <- function(a, b) {
  if (is.null(a) || is.null(b) || length(a) != length(b)) return(NA_real_)
  vals <- Map(function(x, y) {
    x <- as.matrix(x)
    y <- as.matrix(y)
    if (!all(dim(x) == dim(y))) return(Inf)
    if (!length(x)) return(0)
    max(abs(x - y), na.rm = TRUE)
  }, a, b)
  max(as.numeric(vals), na.rm = TRUE)
}

ratio_or_na <- function(num, den) {
  if (!is.finite(num) || !is.finite(den) || den == 0) return(NA_real_)
  num / den
}

pair_row <- function(case, rep, nt, other) {
  ntr <- nt$row
  oth <- other$row
  complete <- identical(ntr$status, "OK") && identical(oth$status, "OK")
  pd <- c(n_common = 0, max_abs = NA_real_, rms = NA_real_, max_rel = NA_real_)
  sigma_diff <- mu_diff <- NA_real_
  if (complete) {
    pd <- parameter_diff(nt$fit, other$fit)
    sigma_diff <- max_abs_nested_diff(nt$implied$sigma, other$implied$sigma)
    mu_diff <- max_abs_nested_diff(nt$implied$mu, other$implied$mu)
  }
  data.frame(
    case_id = case$id,
    source = case$source,
    source_case_id = case$source_case_id,
    label = case$label,
    family = case$family,
    strict_parity = case$strict_parity,
    estimator = oth$estimator,
    replicate = as.integer(rep),
    status = if (complete) "OK" else "INCOMPLETE",
    nt_status = ntr$status,
    estimator_status = oth$status,
    nt_optimizer_status = ntr$optimizer_status,
    estimator_optimizer_status = oth$optimizer_status,
    nt_converged = ntr$converged,
    estimator_converged = oth$converged,
    nt_elapsed_sec = ntr$elapsed_sec,
    estimator_elapsed_sec = oth$elapsed_sec,
    elapsed_ratio_estimator_over_nt = ratio_or_na(oth$elapsed_sec, ntr$elapsed_sec),
    nt_fmin = ntr$fmin,
    estimator_fmin = oth$fmin,
    fmin_diff_estimator_minus_nt = oth$fmin - ntr$fmin,
    nt_naive_chisq = ntr$naive_chisq,
    estimator_naive_chisq = oth$naive_chisq,
    naive_chisq_diff_estimator_minus_nt = oth$naive_chisq - ntr$naive_chisq,
    abs_naive_chisq_diff = abs(oth$naive_chisq - ntr$naive_chisq),
    iterations_diff_estimator_minus_nt = oth$iterations - ntr$iterations,
    f_evals_diff_estimator_minus_nt = oth$f_evals - ntr$f_evals,
    g_evals_diff_estimator_minus_nt = oth$g_evals - ntr$g_evals,
    grad_norm_ratio_estimator_over_nt = ratio_or_na(oth$grad_norm, ntr$grad_norm),
    param_n_common = as.integer(pd[["n_common"]]),
    param_max_abs_diff = as.numeric(pd[["max_abs"]]),
    param_rms_diff = as.numeric(pd[["rms"]]),
    param_max_rel_diff = as.numeric(pd[["max_rel"]]),
    implied_sigma_max_abs_diff = sigma_diff,
    implied_mu_max_abs_diff = mu_diff,
    stringsAsFactors = FALSE
  )
}

median_na <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::median(x)
}

mean_logical <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

case_summary <- function(fits, pairs) {
  labels <- unique(fits[, c("case_id", "source", "label", "family", "strict_parity")])
  labels <- labels[order(labels$label, labels$case_id), , drop = FALSE]
  rows <- lapply(seq_len(nrow(labels)), function(i) {
    row <- labels[i, , drop = FALSE]
    f <- fits[fits$case_id == row$case_id, ]
    p <- pairs[pairs$case_id == row$case_id & pairs$status == "OK", ]
    med_time <- function(est) median_na(f$elapsed_sec[f$estimator == est & f$status == "OK"])
    med_chi <- function(est) median_na(f$naive_chisq[f$estimator == est & f$status == "OK"])
    med_pair <- function(est, col) median_na(p[p$estimator == est, col])
    data.frame(
      case_id = row$case_id,
      source = row$source,
      label = row$label,
      family = row$family,
      strict_parity = row$strict_parity,
      nt_time_sec = med_time("NT"),
      uls_time_sec = med_time("ULS"),
      gls_time_sec = med_time("GLS"),
      uls_elapsed_ratio_over_nt = med_pair("ULS", "elapsed_ratio_estimator_over_nt"),
      gls_elapsed_ratio_over_nt = med_pair("GLS", "elapsed_ratio_estimator_over_nt"),
      nt_naive_chisq = med_chi("NT"),
      uls_naive_chisq = med_chi("ULS"),
      gls_naive_chisq = med_chi("GLS"),
      uls_chisq_diff_vs_nt = med_pair("ULS", "naive_chisq_diff_estimator_minus_nt"),
      gls_chisq_diff_vs_nt = med_pair("GLS", "naive_chisq_diff_estimator_minus_nt"),
      uls_param_max_abs_diff = med_pair("ULS", "param_max_abs_diff"),
      gls_param_max_abs_diff = med_pair("GLS", "param_max_abs_diff"),
      uls_sigma_max_abs_diff = med_pair("ULS", "implied_sigma_max_abs_diff"),
      gls_sigma_max_abs_diff = med_pair("GLS", "implied_sigma_max_abs_diff"),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  invisible(path)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
require_pkg("jsonlite")
require_pkg("magmaan")

experiment_dir <- dirname(script_path())
repo_dir <- normalizePath(file.path(experiment_dir, "..", ".."), mustWork = TRUE)
fixtures_dir <- file.path(repo_dir, "tests", "fixtures")
results_dir <- file.path(experiment_dir, "results")
control <- list(max_iter = 6000L, ftol = 1e-12, gtol = 1e-8, history = 10L)
estimators <- c("NT", "ULS", "GLS")

cases <- load_experiment_cases(
  fixtures_dir,
  case_regex = args$case_regex,
  include_observed = args$include_observed
)
cat(sprintf(
  "complete-data estimator experiment: %d cases, estimators=%s, reps=%d, backend=%s\n",
  length(cases), paste(estimators, collapse = ","), args$reps, args$backend
))

fits <- list()
pairs <- list()

for (case in cases) {
  spec <- tryCatch(build_spec(case), error = identity)
  if (inherits(spec, "error")) {
    for (rep in seq_len(args$reps)) {
      for (estimator in estimators) {
        fits[[length(fits) + 1L]] <- new_fit_row(
          case, estimator, rep, "SKIP",
          error = paste("model_spec:", conditionMessage(spec))
        )
      }
    }
    cat(sprintf("  %-48s SKIP model_spec: %s\n", case$id, conditionMessage(spec)))
    next
  }

  data <- sample_stats(case)
  for (rep in seq_len(args$reps)) {
    results <- setNames(vector("list", length(estimators)), estimators)
    for (estimator in estimators) {
      results[[estimator]] <- fit_case_once(
        case, spec, data, estimator, rep, args$backend, control
      )
      results[[estimator]]$estimator <- estimator
      fits[[length(fits) + 1L]] <- results[[estimator]]$row
    }
    for (estimator in c("ULS", "GLS")) {
      pairs[[length(pairs) + 1L]] <- pair_row(
        case, rep, results$NT, results[[estimator]]
      )
    }
  }

  latest <- tail(do.call(rbind, pairs), 2L)
  cat(sprintf(
    "  %-48s ULS ratio=%s GLS ratio=%s ULS dpar=%s GLS dpar=%s\n",
    case$id,
    formatC(latest$elapsed_ratio_estimator_over_nt[latest$estimator == "ULS"], format = "g", digits = 3),
    formatC(latest$elapsed_ratio_estimator_over_nt[latest$estimator == "GLS"], format = "g", digits = 3),
    formatC(latest$param_max_abs_diff[latest$estimator == "ULS"], format = "g", digits = 3),
    formatC(latest$param_max_abs_diff[latest$estimator == "GLS"], format = "g", digits = 3)
  ))
}

fits_df <- if (length(fits)) do.call(rbind, fits) else data.frame()
pairs_df <- if (length(pairs)) do.call(rbind, pairs) else data.frame()
summary_df <- if (nrow(fits_df)) case_summary(fits_df, pairs_df) else data.frame()

write_csv(fits_df, file.path(results_dir, "fits.csv"))
write_csv(pairs_df, file.path(results_dir, "pairs_vs_nt.csv"))
write_csv(summary_df, file.path(results_dir, "case_summary.csv"))

cat(sprintf(
  "\nwrote %s, %s, %s\n",
  file.path(results_dir, "fits.csv"),
  file.path(results_dir, "pairs_vs_nt.csv"),
  file.path(results_dir, "case_summary.csv")
))
