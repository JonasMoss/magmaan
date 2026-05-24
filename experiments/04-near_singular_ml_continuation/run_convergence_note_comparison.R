#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

repo_root <- function(start = getwd()) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(path, "CMakeLists.txt")) &&
        file.exists(file.path(path, "r-package", "DESCRIPTION"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) stop("Could not find magmaan repository root")
    path <- parent
  }
}

experiment_dir <- function() {
  file.path(repo_root(), "experiments", "04-near_singular_ml_continuation")
}

experiment_path <- function(...) file.path(experiment_dir(), ...)

parse_csv <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

parse_csv_numeric <- function(x) {
  out <- parse_csv(x)
  as.numeric(out[nzchar(out)])
}

parse_args <- function(args) {
  out <- list(
    reps = 100L,
    seed = 20260523L,
    optimizer = "nlopt-lbfgs",
    max_iter = 5000L,
    progress_every = 25L,
    alphas = c(0.50, 0.20, 0.10, 0.05, 0.01),
    designs = c(
      "dejonckere_simple_2025",
      "dejonckere_crossloading_2025",
      "dejonckere_msst_2025",
      "ludtke_cfa_2021"
    )
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    value <- NULL
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_convergence_note_comparison.R ",
        "[--reps 100] [--seed 20260523] ",
        "[--designs dejonckere_simple_2025,...] ",
        "[--alphas 0.5,0.2,0.1,0.05,0.01] ",
        "[--optimizer nlopt-lbfgs] [--max-iter 5000] ",
        "[--progress-every 25]\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    }
    if (grepl("=", arg, fixed = TRUE)) {
      parts <- strsplit(arg, "=", fixed = TRUE)[[1L]]
      arg <- parts[[1L]]
      value <- paste(parts[-1L], collapse = "=")
    }
    need_value <- function() {
      if (!is.null(value)) return(value)
      i <<- i + 1L
      if (i > length(args)) stop(arg, " requires a value", call. = FALSE)
      args[[i]]
    }
    if (arg == "--reps") out$reps <- as.integer(need_value())
    else if (arg == "--seed") out$seed <- as.integer(need_value())
    else if (arg == "--optimizer") out$optimizer <- need_value()
    else if (arg == "--max-iter") out$max_iter <- as.integer(need_value())
    else if (arg == "--progress-every") out$progress_every <- as.integer(need_value())
    else if (arg == "--alphas") out$alphas <- parse_csv_numeric(need_value())
    else if (arg == "--designs") out$designs <- parse_csv(need_value())
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  if (anyNA(unlist(out[c("reps", "seed", "max_iter", "progress_every")]))) {
    stop("integer arguments must be finite", call. = FALSE)
  }
  if (out$reps < 1L || out$max_iter < 1L || out$progress_every < 1L) {
    stop("reps, max_iter, and progress_every must be positive", call. = FALSE)
  }
  if (!length(out$alphas) || anyNA(out$alphas)) {
    stop("alphas must be a non-empty numeric CSV", call. = FALSE)
  }
  if (!length(out$designs) || any(!nzchar(out$designs))) {
    stop("designs must be a non-empty CSV", call. = FALSE)
  }
  out
}

safe_condition <- function(S) {
  ev <- eigen((S + t(S)) / 2, symmetric = TRUE, only.values = TRUE)$values
  if (!length(ev)) return(c(min = NA_real_, max = NA_real_, condition = NA_real_))
  c(min = min(ev), max = max(ev),
    condition = if (min(ev) > 0) max(ev) / min(ev) else Inf)
}

fit_one <- function(method, spec, dat, optimizer, control, alphas) {
  warnings <- character()
  started <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(
    try(
      if (identical(method, "baseline_ml")) {
        magmaan::magmaan_core$fit_ml(spec, dat, optimizer = optimizer,
                                     control = control)
      } else {
        magmaan::magmaan_core$frontier_fit_ml_ridge_continuation(
          spec, dat, optimizer = optimizer, control = control,
          alphas = alphas)
      },
      silent = TRUE
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  elapsed <- proc.time()[["elapsed"]] - started
  list(fit = fit, elapsed = elapsed, warning = paste(unique(warnings), collapse = " | "))
}

audit_field <- function(fit, name, default = NA) {
  if (is.null(fit$audit) || is.null(fit$audit[[name]])) return(default)
  fit$audit[[name]]
}

audit_converged <- function(fit) {
  stationary <- audit_field(fit, "stationary", NA)
  if (length(stationary) == 1L && !is.na(stationary)) {
    return(isTRUE(stationary))
  }
  isTRUE(fit$converged)
}

failure_stage <- function(error) {
  hit <- regexpr("stage alpha=[0-9.eE+-]+", error)
  if (hit[[1L]] < 0L) return(NA_real_)
  as.numeric(sub("stage alpha=", "", regmatches(error, hit)))
}

fit_row <- function(design, rep, seed, sim, method, spec, dat, optimizer,
                    control, alphas, sample_diag) {
  got <- fit_one(method, spec, dat, optimizer, control, alphas)
  fit <- got$fit
  error <- if (inherits(fit, "try-error")) as.character(fit) else ""
  endpoint_path <- c(alphas, 0)
  base <- data.frame(
    design = design,
    replicate = rep,
    seed = seed,
    n = sim$n,
    p = ncol(sim$data),
    optimizer = optimizer,
    method = method,
    alpha_path = paste(endpoint_path, collapse = ","),
    sample_min_eigen = sample_diag[["min"]],
    sample_condition = sample_diag[["condition"]],
    ok = !inherits(fit, "try-error"),
    converged = FALSE,
    optimizer_converged = FALSE,
    optimizer_status = NA_character_,
    fmin = NA_real_,
    grad_norm = NA_real_,
    audit_stationary = NA,
    audit_grad_inf_norm = NA_real_,
    audit_stationarity_rhs = NA_real_,
    audit_f_consistent = NA,
    audit_f_finite = NA,
    audit_advisory_status = NA_character_,
    iterations = NA_integer_,
    f_evals = NA_integer_,
    g_evals = NA_integer_,
    total_f_evals = NA_integer_,
    n_stages = if (identical(method, "baseline_ml")) 1L else length(endpoint_path),
    failure_alpha = failure_stage(error),
    sigma_pd_all = NA,
    ov_var_nonnegative = NA,
    lv_var_nonnegative = NA,
    elapsed_sec = got$elapsed,
    error = error,
    warning = got$warning,
    stringsAsFactors = FALSE
  )
  if (inherits(fit, "try-error")) return(base)

  base$converged <- audit_converged(fit)
  base$optimizer_converged <- isTRUE(fit$converged)
  base$optimizer_status <- fit$optimizer_status %||% NA_character_
  base$fmin <- fit$fmin %||% NA_real_
  base$grad_norm <- fit$grad_norm %||% NA_real_
  base$audit_stationary <- audit_field(fit, "stationary", NA)
  base$audit_grad_inf_norm <- audit_field(fit, "grad_inf_norm", NA_real_)
  base$audit_stationarity_rhs <- audit_field(fit, "stationarity_rhs", NA_real_)
  base$audit_f_consistent <- audit_field(fit, "f_consistent", NA)
  base$audit_f_finite <- audit_field(fit, "f_finite", NA)
  base$audit_advisory_status <- audit_field(fit, "advisory_status", NA_character_)
  base$iterations <- fit$iterations %||% NA_integer_
  base$f_evals <- fit$f_evals %||% NA_integer_
  base$g_evals <- fit$g_evals %||% NA_integer_
  base$sigma_pd_all <- fit$sigma_pd_all %||% NA
  base$ov_var_nonnegative <- fit$ov_var_nonnegative %||% NA
  base$lv_var_nonnegative <- fit$lv_var_nonnegative %||% NA
  if (!is.null(fit$continuation)) {
    base$total_f_evals <- fit$continuation$total_f_evals
    base$n_stages <- nrow(fit$continuation$path)
  } else {
    base$total_f_evals <- base$f_evals
  }
  base
}

summarize_results <- function(rows) {
  groups <- split(rows, interaction(rows$design, rows$method, drop = TRUE))
  out <- lapply(groups, function(x) {
    data.frame(
      design = x$design[[1L]],
      method = x$method[[1L]],
      attempts = nrow(x),
      ok = sum(x$ok, na.rm = TRUE),
      converged = sum(x$converged, na.rm = TRUE),
      optimizer_converged = sum(x$optimizer_converged, na.rm = TRUE),
      ok_rate = mean(x$ok, na.rm = TRUE),
      converged_rate = mean(x$converged, na.rm = TRUE),
      optimizer_converged_rate = mean(x$optimizer_converged, na.rm = TRUE),
      line_search_failures = sum(grepl("LineSearchFailed", x$error, fixed = TRUE)),
      line_search_salvaged = sum(x$optimizer_status == "line_search_salvaged",
                                 na.rm = TRUE),
      median_total_f_evals = stats::median(x$total_f_evals, na.rm = TRUE),
      median_elapsed_sec = stats::median(x$elapsed_sec, na.rm = TRUE),
      median_audit_grad_inf_norm = stats::median(
        x$audit_grad_inf_norm, na.rm = TRUE),
      median_audit_stationarity_rhs = stats::median(
        x$audit_stationarity_rhs, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, out)
  out[order(out$design, out$method), ]
}

paired_summary <- function(rows) {
  base <- rows[rows$method == "baseline_ml", ]
  cont <- rows[rows$method == "ridge_continuation", ]
  names(base) <- paste0("base_", names(base))
  names(cont) <- paste0("cont_", names(cont))
  paired <- merge(
    base,
    cont,
    by.x = c("base_design", "base_replicate"),
    by.y = c("cont_design", "cont_replicate")
  )
  groups <- split(paired, paired$base_design)
  out <- lapply(groups, function(x) {
    both <- x$base_converged & x$cont_converged
    data.frame(
      design = x$base_design[[1L]],
      reps = nrow(x),
      both_converged = mean(both),
      baseline_only = mean(x$base_converged & !x$cont_converged),
      continuation_only = mean(!x$base_converged & x$cont_converged),
      neither = mean(!x$base_converged & !x$cont_converged),
      median_fmin_diff_cont_minus_base = stats::median(
        x$cont_fmin[both] - x$base_fmin[both],
        na.rm = TRUE
      ),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, out)
  out[order(out$design), ]
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(experiment_path("results"), showWarnings = FALSE, recursive = TRUE)

  if (!requireNamespace("magmaan", quietly = TRUE)) {
    stop("Install the local R package first, e.g. `just r-install`", call. = FALSE)
  }

  catalog <- magmaan::convergence_sim_catalog()
  unknown <- setdiff(args$designs, catalog$design)
  if (length(unknown)) {
    stop("Unknown convergence-sim design(s): ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }

  control <- list(max_iter = args$max_iter, ftol = 1e-10, gtol = 1e-7)
  rows <- vector("list", length(args$designs) * args$reps * 2L)
  pos <- 1L

  for (design in args$designs) {
    for (rep in seq_len(args$reps)) {
      if (rep == 1L || rep == args$reps || rep %% args$progress_every == 0L) {
        message("design=", design, " rep=", rep, "/", args$reps)
      }
      seed <- args$seed + match(design, catalog$design) * 100000L + rep
      sim <- magmaan::convergence_sim(design, seed = seed)
      spec <- magmaan::model_spec(sim$analysis_syntax)
      dat <- magmaan::df_to_data(sim$data, spec)
      sample_diag <- safe_condition(dat$S[[1L]])
      rows[[pos]] <- fit_row(design, rep, seed, sim, "baseline_ml", spec, dat,
                             args$optimizer, control, args$alphas, sample_diag)
      pos <- pos + 1L
      rows[[pos]] <- fit_row(design, rep, seed, sim, "ridge_continuation", spec,
                             dat, args$optimizer, control, args$alphas,
                             sample_diag)
      pos <- pos + 1L
    }
  }

  rows <- do.call(rbind, rows)
  summary <- summarize_results(rows)
  paired <- paired_summary(rows)

  write.csv(rows, experiment_path("results", "convergence_note_fits.csv"),
            row.names = FALSE)
  write.csv(summary, experiment_path("results", "convergence_note_summary.csv"),
            row.names = FALSE)
  write.csv(paired, experiment_path("results", "convergence_note_paired.csv"),
            row.names = FALSE)

  print(summary, row.names = FALSE)
  print(paired, row.names = FALSE)
  cat("Wrote ", experiment_path("results", "convergence_note_fits.csv"), "\n",
      sep = "")
}

main()
