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
    backends = c("nlopt-lbfgs", "port"),
    case_regex = NULL
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--backends a,b] [--cases REGEX]\n",
        "\n",
        "Defaults: --reps 10 --backends nlopt-lbfgs,port\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--reps") {
      i <- i + 1L
      if (i > length(args)) stop("--reps requires a value", call. = FALSE)
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(arg, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", arg))
    } else if (arg == "--backends") {
      i <- i + 1L
      if (i > length(args)) stop("--backends requires a value", call. = FALSE)
      out$backends <- strsplit(args[[i]], ",", fixed = TRUE)[[1L]]
    } else if (startsWith(arg, "--backends=")) {
      out$backends <- strsplit(sub("^--backends=", "", arg), ",", fixed = TRUE)[[1L]]
    } else if (arg == "--cases") {
      i <- i + 1L
      if (i > length(args)) stop("--cases requires a value", call. = FALSE)
      out$case_regex <- args[[i]]
    } else if (startsWith(arg, "--cases=")) {
      out$case_regex <- sub("^--cases=", "", arg)
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  out$backends <- trimws(out$backends[nzchar(trimws(out$backends))])
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  if (!length(out$backends)) {
    stop("--backends must name at least one optimizer", call. = FALSE)
  }
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

usable_case <- function(case) {
  !is.null(case) && !is.null(case$model) &&
    !is.null(case$sample_cov) && !is.na(case$n_obs)
}

load_experiment_cases <- function(fixtures_dir, case_regex = NULL) {
  manifest_path <- file.path(fixtures_dir, "textbook_corpus", "manifest.json")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  candidates <- Filter(function(x) {
    isTRUE(x$strict_parity) &&
      identical(x$measurement_kind %||% "", "continuous") &&
      !isTRUE(x$observed_only)
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
        if (usable_case(candidate)) {
          loaded <- candidate
          break
        }
        loaded <- loaded %||% candidate
      }
    }
    if (usable_case(loaded)) {
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

build_spec <- function(case, parameterization) {
  if (identical(parameterization, "marker")) {
    auto_fix_first <- TRUE
    std_lv <- FALSE
  } else if (identical(parameterization, "std_lv")) {
    auto_fix_first <- FALSE
    std_lv <- TRUE
  } else {
    stop("unknown parameterization: ", parameterization, call. = FALSE)
  }
  magmaan::model_spec(
    case$model,
    auto_fix_first = auto_fix_first,
    std_lv = std_lv,
    fixed_x = case$fixed_x,
    meanstructure = case$meanstructure,
    model_type = case$model_type
  )
}

n_free <- function(spec) {
  sum(spec$partable$free > 0L, na.rm = TRUE)
}

new_fit_row <- function(case, backend, parameterization, rep, status,
                        error = "", fit = NULL, elapsed_sec = NA_real_) {
  diag <- fit$diagnostics %||% list()
  audit <- fit$audit %||% list()
  data.frame(
    case_id = case$id,
    source = case$source,
    source_case_id = case$source_case_id,
    label = case$label,
    family = case$family,
    backend = backend,
    parameterization = parameterization,
    replicate = as.integer(rep),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    fmin = as.numeric(fit$fmin %||% NA_real_),
    iterations = as.integer(fit$iterations %||% NA_integer_),
    f_evals = as.integer(fit$f_evals %||% NA_integer_),
    g_evals = as.integer(fit$g_evals %||% NA_integer_),
    optimizer_status = as.character(fit$optimizer_status %||% NA_character_),
    converged = as.logical(fit$converged %||% NA),
    grad_norm = as.numeric(fit$grad_norm %||% NA_real_),
    npar = as.integer(fit$npar %||% NA_integer_),
    ntotal = as.integer(fit$ntotal %||% case$n_obs),
    sigma_pd_all = as.logical(diag$sigma_pd_all %||% NA),
    lin_eq_residual_inf = as.numeric(diag$lin_eq_residual_inf %||% NA_real_),
    nl_eq_residual_inf = as.numeric(diag$nl_eq_residual_inf %||% NA_real_),
    active_bounds_lower = as.integer(diag$active_bounds_lower %||% NA_integer_),
    active_bounds_upper = as.integer(diag$active_bounds_upper %||% NA_integer_),
    audit_grad_norm = as.numeric(audit$grad_norm %||% audit$projected_grad_norm %||% NA_real_),
    stringsAsFactors = FALSE
  )
}

fit_case_once <- function(case, spec, data, backend, parameterization, rep, control) {
  start <- Sys.time()
  fit <- tryCatch({
    magmaan::magmaan_core$fit_ml(
      spec, data, optimizer = backend, control = control
    )
  }, error = identity)
  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  if (inherits(fit, "error")) {
    return(list(
      row = new_fit_row(case, backend, parameterization, rep, "ERROR",
                        error = conditionMessage(fit), elapsed_sec = elapsed),
      fit = NULL,
      implied = NULL
    ))
  }
  implied <- tryCatch(magmaan::magmaan_core$model_implied(fit), error = identity)
  if (inherits(implied, "error")) {
    return(list(
      row = new_fit_row(case, backend, parameterization, rep, "ERROR",
                        error = paste("model_implied:", conditionMessage(implied)),
                        fit = fit, elapsed_sec = elapsed),
      fit = fit,
      implied = NULL
    ))
  }
  list(
    row = new_fit_row(case, backend, parameterization, rep, "OK",
                      fit = fit, elapsed_sec = elapsed),
    fit = fit,
    implied = implied
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

pair_row <- function(case, backend, rep, marker, stdlv) {
  m <- marker$row
  s <- stdlv$row
  complete <- identical(m$status, "OK") && identical(s$status, "OK")
  sigma_diff <- mu_diff <- NA_real_
  if (complete) {
    sigma_diff <- max_abs_nested_diff(marker$implied$sigma, stdlv$implied$sigma)
    mu_diff <- max_abs_nested_diff(marker$implied$mu, stdlv$implied$mu)
  }
  data.frame(
    case_id = case$id,
    source = case$source,
    source_case_id = case$source_case_id,
    family = case$family,
    backend = backend,
    replicate = as.integer(rep),
    status = if (complete) "OK" else "INCOMPLETE",
    marker_status = m$status,
    std_lv_status = s$status,
    marker_optimizer_status = m$optimizer_status,
    std_lv_optimizer_status = s$optimizer_status,
    marker_converged = m$converged,
    std_lv_converged = s$converged,
    marker_elapsed_sec = m$elapsed_sec,
    std_lv_elapsed_sec = s$elapsed_sec,
    elapsed_ratio_std_lv_over_marker = ratio_or_na(s$elapsed_sec, m$elapsed_sec),
    fmin_marker = m$fmin,
    fmin_std_lv = s$fmin,
    fmin_diff_std_lv_minus_marker = s$fmin - m$fmin,
    abs_fmin_diff = abs(s$fmin - m$fmin),
    iterations_diff_std_lv_minus_marker = s$iterations - m$iterations,
    f_evals_diff_std_lv_minus_marker = s$f_evals - m$f_evals,
    g_evals_diff_std_lv_minus_marker = s$g_evals - m$g_evals,
    grad_norm_marker = m$grad_norm,
    grad_norm_std_lv = s$grad_norm,
    grad_norm_ratio_std_lv_over_marker = ratio_or_na(s$grad_norm, m$grad_norm),
    implied_sigma_max_abs_diff = sigma_diff,
    implied_mu_max_abs_diff = mu_diff,
    same_implied_moments = isTRUE(sigma_diff <= 1e-5) &&
      (is.na(mu_diff) || isTRUE(mu_diff <= 1e-5)),
    material_mismatch = isTRUE(abs(s$fmin - m$fmin) > 1e-5) ||
      isTRUE(sigma_diff > 1e-5) || isTRUE(mu_diff > 1e-5),
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

summarize_pairs <- function(pairs) {
  if (!nrow(pairs)) return(data.frame())
  keys <- interaction(pairs$backend, pairs$source, pairs$family, drop = TRUE, sep = "\r")
  chunks <- split(pairs, keys)
  rows <- lapply(chunks, function(x) {
    data.frame(
      backend = x$backend[[1L]],
      source = x$source[[1L]],
      family = x$family[[1L]],
      n_pairs = nrow(x),
      n_complete = sum(x$status == "OK", na.rm = TRUE),
      marker_convergence_rate = mean_logical(x$marker_converged),
      std_lv_convergence_rate = mean_logical(x$std_lv_converged),
      median_elapsed_ratio_std_lv_over_marker =
        median_na(x$elapsed_ratio_std_lv_over_marker),
      median_iterations_diff_std_lv_minus_marker =
        median_na(x$iterations_diff_std_lv_minus_marker),
      median_f_evals_diff_std_lv_minus_marker =
        median_na(x$f_evals_diff_std_lv_minus_marker),
      median_g_evals_diff_std_lv_minus_marker =
        median_na(x$g_evals_diff_std_lv_minus_marker),
      median_grad_norm_ratio_std_lv_over_marker =
        median_na(x$grad_norm_ratio_std_lv_over_marker),
      material_mismatches = sum(x$material_mismatch, na.rm = TRUE),
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

cases <- load_experiment_cases(fixtures_dir, args$case_regex)
cat(sprintf(
  "latent metric identification experiment: %d cases, backends=%s, reps=%d\n",
  length(cases), paste(args$backends, collapse = ","), args$reps
))

fits <- list()
pairs <- list()

for (case in cases) {
  specs <- tryCatch({
    list(marker = build_spec(case, "marker"), std_lv = build_spec(case, "std_lv"))
  }, error = identity)
  if (inherits(specs, "error")) {
    for (backend in args$backends) {
      for (rep in seq_len(args$reps)) {
        for (parameterization in c("marker", "std_lv")) {
          fits[[length(fits) + 1L]] <- new_fit_row(
            case, backend, parameterization, rep, "SKIP",
            error = paste("model_spec:", conditionMessage(specs))
          )
        }
      }
    }
    cat(sprintf("  %-48s SKIP model_spec: %s\n", case$id, conditionMessage(specs)))
    next
  }

  marker_n_free <- n_free(specs$marker)
  stdlv_n_free <- n_free(specs$std_lv)
  if (!identical(marker_n_free, stdlv_n_free)) {
    reason <- sprintf(
      "not scale-equivalent under this experiment: marker npar=%d, std_lv npar=%d",
      marker_n_free, stdlv_n_free
    )
    for (backend in args$backends) {
      for (rep in seq_len(args$reps)) {
        for (parameterization in c("marker", "std_lv")) {
          fits[[length(fits) + 1L]] <- new_fit_row(
            case, backend, parameterization, rep, "SKIP", error = reason
          )
        }
      }
    }
    cat(sprintf("  %-48s SKIP %s\n", case$id, reason))
    next
  }

  data <- sample_stats(case)
  for (backend in args$backends) {
    for (rep in seq_len(args$reps)) {
      marker <- fit_case_once(case, specs$marker, data, backend, "marker", rep, control)
      stdlv <- fit_case_once(case, specs$std_lv, data, backend, "std_lv", rep, control)
      fits[[length(fits) + 1L]] <- marker$row
      fits[[length(fits) + 1L]] <- stdlv$row
      pairs[[length(pairs) + 1L]] <- pair_row(case, backend, rep, marker, stdlv)
    }
    latest <- tail(do.call(rbind, pairs), 1L)
    cat(sprintf(
      "  %-48s %-11s status=%s elapsed_ratio=%s sigma_diff=%s\n",
      case$id, backend, latest$status,
      formatC(latest$elapsed_ratio_std_lv_over_marker, format = "g", digits = 3),
      formatC(latest$implied_sigma_max_abs_diff, format = "g", digits = 3)
    ))
  }
}

fits_df <- if (length(fits)) do.call(rbind, fits) else data.frame()
pairs_df <- if (length(pairs)) do.call(rbind, pairs) else data.frame()
summary_df <- summarize_pairs(pairs_df)

write_csv(fits_df, file.path(results_dir, "fits.csv"))
write_csv(pairs_df, file.path(results_dir, "pairs.csv"))
write_csv(summary_df, file.path(results_dir, "summary.csv"))

cat(sprintf(
  "\nwrote %s, %s, %s\n",
  file.path(results_dir, "fits.csv"),
  file.path(results_dir, "pairs.csv"),
  file.path(results_dir, "summary.csv")
))
