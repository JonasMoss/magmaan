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
  file.path(repo_root(), "experiments", "03-heywood_box_constraints")
}

experiment_path <- function(...) file.path(experiment_dir(), ...)

parse_csv_arg <- function(x) {
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

parse_args <- function(args) {
  out <- list(
    cases = c("study3_msst:997", "study3_msst:805", "study3_msst:592"),
    identifications = c("marker", "std.lv"),
    bounds = c("none", "pos.var", "standard", "wide"),
    rstarts = c(0L, 150L),
    optimizer = "nlopt-lbfgs"
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--cases study:seed,...] ",
        "[--identification marker,std.lv] ",
        "[--bounds none,pos.var,standard,wide] [--rstarts 0,150] ",
        "[--optimizer nlopt-lbfgs]\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--cases") {
      i <- i + 1L
      if (i > length(args)) stop("--cases requires a value", call. = FALSE)
      out$cases <- parse_csv_arg(args[[i]])
    } else if (startsWith(arg, "--cases=")) {
      out$cases <- parse_csv_arg(sub("^--cases=", "", arg))
    } else if (arg %in% c("--identification", "--identifications")) {
      i <- i + 1L
      if (i > length(args)) stop("--identification requires a value", call. = FALSE)
      out$identifications <- parse_csv_arg(args[[i]])
    } else if (startsWith(arg, "--identification=")) {
      out$identifications <- parse_csv_arg(sub("^--identification=", "", arg))
    } else if (startsWith(arg, "--identifications=")) {
      out$identifications <- parse_csv_arg(sub("^--identifications=", "", arg))
    } else if (arg == "--bounds") {
      i <- i + 1L
      if (i > length(args)) stop("--bounds requires a value", call. = FALSE)
      out$bounds <- parse_csv_arg(args[[i]])
    } else if (startsWith(arg, "--bounds=")) {
      out$bounds <- parse_csv_arg(sub("^--bounds=", "", arg))
    } else if (arg == "--rstarts") {
      i <- i + 1L
      if (i > length(args)) stop("--rstarts requires a value", call. = FALSE)
      out$rstarts <- as.integer(parse_csv_arg(args[[i]]))
    } else if (startsWith(arg, "--rstarts=")) {
      out$rstarts <- as.integer(parse_csv_arg(sub("^--rstarts=", "", arg)))
    } else if (arg == "--optimizer") {
      i <- i + 1L
      if (i > length(args)) stop("--optimizer requires a value", call. = FALSE)
      out$optimizer <- args[[i]]
    } else if (startsWith(arg, "--optimizer=")) {
      out$optimizer <- sub("^--optimizer=", "", arg)
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!length(out$cases)) stop("--cases cannot be empty", call. = FALSE)
  out$identifications <- normalize_identifications(out$identifications)
  if (!length(out$identifications)) {
    stop("--identification cannot be empty", call. = FALSE)
  }
  if (!length(out$bounds)) stop("--bounds cannot be empty", call. = FALSE)
  if (anyNA(out$rstarts) || any(out$rstarts < 0L)) {
    stop("--rstarts must be nonnegative integer values", call. = FALSE)
  }
  out
}

normalize_identifications <- function(x) {
  out <- trimws(x)
  out[out %in% c("std_lv", "stdlv", "std-lv")] <- "std.lv"
  bad <- !out %in% c("marker", "std.lv")
  if (any(bad)) {
    stop("unknown identification(s): ", paste(out[bad], collapse = ", "),
         call. = FALSE)
  }
  unique(out)
}

identification_options <- function(identification) {
  std_lv <- identical(identification, "std.lv")
  list(std_lv = std_lv, auto_fix_first = !std_lv)
}

require_pkg <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("required R package is not installed: ", package, call. = FALSE)
  }
  invisible(TRUE)
}

case_catalog <- function() {
  data.frame(
    paper_study = c("study1_simple", "study2_crossloading", "study3_msst"),
    design = c("dejonckere_simple_2025",
               "dejonckere_crossloading_2025",
               "dejonckere_msst_2025"),
    n = c(20L, 20L, 10L),
    seed_offset = c(0L, 19000L, 0L),
    stringsAsFactors = FALSE
  )
}

parse_cases <- function(keys) {
  parts <- strsplit(keys, ":", fixed = TRUE)
  if (any(lengths(parts) != 2L)) {
    stop("cases must use paper_study:seed keys", call. = FALSE)
  }
  out <- data.frame(
    case_key = keys,
    paper_study = vapply(parts, `[`, character(1), 1L),
    seed = as.integer(vapply(parts, `[`, character(1), 2L)),
    stringsAsFactors = FALSE
  )
  if (anyNA(out$seed)) stop("case seed is not an integer", call. = FALSE)
  cat <- case_catalog()
  out <- merge(out, cat, by = "paper_study", all.x = TRUE, sort = FALSE)
  if (anyNA(out$design)) {
    stop("unknown paper_study in cases: ",
         paste(out$paper_study[is.na(out$design)], collapse = ", "),
         call. = FALSE)
  }
  out$replicate <- out$seed - out$seed_offset
  out$case_label <- paste0(out$paper_study, "_seed_", out$seed)
  out
}

safe_min <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  min(x)
}

safe_max_abs <- function(x) {
  x <- abs(x[is.finite(x)])
  if (!length(x)) return(NA_real_)
  max(x)
}

finite_median <- function(v) {
  v <- v[is.finite(v)]
  if (!length(v)) return(NA_real_)
  stats::median(v)
}

variance_summary <- function(partable) {
  lv_names <- unique(as.character(partable$lhs[partable$op == "=~"]))
  var_rows <- partable$op == "~~" & partable$lhs == partable$rhs
  est <- suppressWarnings(as.numeric(partable$est))
  ov <- var_rows & !(partable$lhs %in% lv_names)
  lv <- var_rows & partable$lhs %in% lv_names
  list(
    min_ov_var = if (any(ov)) safe_min(est[ov]) else NA_real_,
    min_lv_var = if (any(lv)) safe_min(est[lv]) else NA_real_,
    n_neg_ov_var = if (any(ov)) sum(est[ov] < -1e-8, na.rm = TRUE) else NA_integer_,
    n_neg_lv_var = if (any(lv)) sum(est[lv] < -1e-8, na.rm = TRUE) else NA_integer_,
    n_zeroish_ov_var = if (any(ov)) sum(abs(est[ov]) <= 1e-6, na.rm = TRUE) else NA_integer_,
    n_zeroish_lv_var = if (any(lv)) sum(abs(est[lv]) <= 1e-6, na.rm = TRUE) else NA_integer_
  )
}

lavaan_fit_row <- function(case, sim, identification, bounds, rstarts) {
  started <- proc.time()[["elapsed"]]
  warnings <- character()
  id_opts <- identification_options(identification)
  fit <- withCallingHandlers(
    try(
      lavaan::sem(sim$analysis_syntax, data = sim$data,
                  std.lv = id_opts$std_lv,
                  auto.fix.first = id_opts$auto_fix_first,
                  bounds = bounds, rstarts = rstarts),
      silent = TRUE
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  elapsed <- proc.time()[["elapsed"]] - started
  if (inherits(fit, "try-error")) {
    return(data.frame(
      case_key = case$case_key,
      case_label = case$case_label,
      paper_study = case$paper_study,
      design = case$design,
      replicate = case$replicate,
      seed = case$seed,
      engine = "lavaan",
      estimator = "ML",
      identification = identification,
      bounds = bounds,
      rstarts = rstarts,
      ok = FALSE,
      converged = FALSE,
      fmin = NA_real_,
      iterations = NA_integer_,
      raw_grad_inf = NA_real_,
      min_ov_var = NA_real_,
      min_lv_var = NA_real_,
      n_neg_ov_var = NA_integer_,
      n_neg_lv_var = NA_integer_,
      n_zeroish_ov_var = NA_integer_,
      n_zeroish_lv_var = NA_integer_,
      random_start_converged = NA_integer_,
      first_random_start_converged = NA_integer_,
      elapsed_sec = elapsed,
      warning = paste(unique(warnings), collapse = " | "),
      error = as.character(fit),
      stringsAsFactors = FALSE
    ))
  }

  pt <- as.data.frame(fit@ParTable, stringsAsFactors = FALSE)
  vs <- variance_summary(pt)
  rstart_converged <- NA_integer_
  first_rstart <- NA_integer_
  if (!is.null(fit@optim$x.rstarts)) {
    flags <- vapply(fit@optim$x.rstarts, function(x) {
      isTRUE(attr(x, "converged"))
    }, logical(1))
    rstart_converged <- sum(flags)
    if (any(flags)) first_rstart <- which(flags)[1L]
  }

  data.frame(
    case_key = case$case_key,
    case_label = case$case_label,
    paper_study = case$paper_study,
    design = case$design,
    replicate = case$replicate,
    seed = case$seed,
    engine = "lavaan",
    estimator = "ML",
    identification = identification,
    bounds = bounds,
    rstarts = rstarts,
    ok = TRUE,
    converged = isTRUE(fit@optim$converged),
    fmin = as.numeric(fit@optim$fx %||% NA_real_),
    iterations = as.integer(fit@optim$iterations %||% NA_integer_),
    raw_grad_inf = safe_max_abs(fit@optim$dx %||% NA_real_),
    min_ov_var = vs$min_ov_var,
    min_lv_var = vs$min_lv_var,
    n_neg_ov_var = vs$n_neg_ov_var,
    n_neg_lv_var = vs$n_neg_lv_var,
    n_zeroish_ov_var = vs$n_zeroish_ov_var,
    n_zeroish_lv_var = vs$n_zeroish_lv_var,
    random_start_converged = rstart_converged,
    first_random_start_converged = first_rstart,
    elapsed_sec = elapsed,
    warning = paste(unique(warnings), collapse = " | "),
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

magmaan_fit_row <- function(case, sim, optimizer, identification, bounds) {
  started <- proc.time()[["elapsed"]]
  id_opts <- identification_options(identification)
  fit <- try(
    magmaan::magmaan(sim$analysis_syntax, sim$data, estimator = "ML",
                     std_lv = id_opts$std_lv,
                     auto_fix_first = id_opts$auto_fix_first,
                     bounds = bounds,
                     optimizer = optimizer,
                     control = list(max_iter = 5000, ftol = 1e-10,
                                    gtol = 1e-7)),
    silent = TRUE
  )
  elapsed <- proc.time()[["elapsed"]] - started
  if (inherits(fit, "try-error")) {
    return(data.frame(
      case_key = case$case_key,
      case_label = case$case_label,
      paper_study = case$paper_study,
      design = case$design,
      replicate = case$replicate,
      seed = case$seed,
      engine = "magmaan",
      estimator = "ML",
      identification = identification,
      bounds = bounds,
      optimizer = optimizer,
      ok = FALSE,
      converged = FALSE,
      fmin = NA_real_,
      grad_inf_norm = NA_real_,
      audit_stationary = NA,
      audit_status = NA_character_,
      min_ov_var = NA_real_,
      min_lv_var = NA_real_,
      n_neg_ov_var = NA_integer_,
      n_neg_lv_var = NA_integer_,
      n_zeroish_ov_var = NA_integer_,
      n_zeroish_lv_var = NA_integer_,
      elapsed_sec = elapsed,
      error = as.character(fit),
      stringsAsFactors = FALSE
    ))
  }

  vs <- variance_summary(fit$partable)
  audit <- fit$audit %||% list()
  data.frame(
    case_key = case$case_key,
    case_label = case$case_label,
    paper_study = case$paper_study,
    design = case$design,
    replicate = case$replicate,
    seed = case$seed,
    engine = "magmaan",
    estimator = "ML",
    identification = identification,
    bounds = bounds,
    optimizer = optimizer,
    ok = TRUE,
    converged = isTRUE(fit$converged),
    fmin = as.numeric(fit$fmin %||% NA_real_),
    grad_inf_norm = as.numeric(fit$grad_inf_norm %||% fit$grad_norm %||% NA_real_),
    audit_stationary = isTRUE(audit$stationary),
    audit_status = as.character(audit$advisory_status %||% NA_character_),
    min_ov_var = vs$min_ov_var,
    min_lv_var = vs$min_lv_var,
    n_neg_ov_var = vs$n_neg_ov_var,
    n_neg_lv_var = vs$n_neg_lv_var,
    n_zeroish_ov_var = vs$n_zeroish_ov_var,
    n_zeroish_lv_var = vs$n_zeroish_lv_var,
    elapsed_sec = elapsed,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  message("Wrote ", path)
}

summarize_lavaan_bounds <- function(x) {
  x$admissible_variances <- x$n_neg_ov_var == 0L & x$n_neg_lv_var == 0L
  x$active_variance_bound_count <- x$n_zeroish_ov_var + x$n_zeroish_lv_var
  groups <- split(seq_len(nrow(x)), paste(x$case_key, x$identification,
                                          x$bounds, x$rstarts, sep = "\r"),
                  drop = TRUE)
  rows <- lapply(groups, function(idx) {
    y <- x[idx, , drop = FALSE]
    data.frame(
      case_key = y$case_key[1L],
      identification = y$identification[1L],
      bounds = y$bounds[1L],
      rstarts = y$rstarts[1L],
      ok = mean(y$ok %in% TRUE),
      converged = mean(y$converged %in% TRUE),
      admissible_variances = mean(y$admissible_variances %in% TRUE),
      fmin = finite_median(y$fmin),
      raw_grad_inf = finite_median(y$raw_grad_inf),
      min_ov_var = finite_median(y$min_ov_var),
      min_lv_var = finite_median(y$min_lv_var),
      n_neg_ov_var = finite_median(y$n_neg_ov_var),
      n_neg_lv_var = finite_median(y$n_neg_lv_var),
      active_variance_bound_count =
        finite_median(y$active_variance_bound_count),
      error = paste(unique(y$error[!is.na(y$error) & nzchar(y$error)]),
                    collapse = " | "),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$case_key, out$identification, out$bounds, out$rstarts), ,
      drop = FALSE]
}

summarize_magmaan_bounds <- function(x) {
  x$admissible_variances <- x$n_neg_ov_var == 0L & x$n_neg_lv_var == 0L
  x$active_variance_bound_count <- x$n_zeroish_ov_var + x$n_zeroish_lv_var
  groups <- split(seq_len(nrow(x)), paste(x$case_key, x$identification,
                                          x$bounds, sep = "\r"),
                  drop = TRUE)
  rows <- lapply(groups, function(idx) {
    y <- x[idx, , drop = FALSE]
    data.frame(
      case_key = y$case_key[1L],
      identification = y$identification[1L],
      bounds = y$bounds[1L],
      ok = mean(y$ok %in% TRUE),
      converged = mean(y$converged %in% TRUE),
      stationary = mean(y$audit_stationary %in% TRUE),
      admissible_variances = mean(y$admissible_variances %in% TRUE),
      fmin = finite_median(y$fmin),
      grad_inf_norm = finite_median(y$grad_inf_norm),
      min_ov_var = finite_median(y$min_ov_var),
      min_lv_var = finite_median(y$min_lv_var),
      n_neg_ov_var = finite_median(y$n_neg_ov_var),
      n_neg_lv_var = finite_median(y$n_neg_lv_var),
      active_variance_bound_count =
        finite_median(y$active_variance_bound_count),
      audit_status = paste(unique(y$audit_status[!is.na(y$audit_status) &
                                                   nzchar(y$audit_status)]),
                           collapse = " | "),
      error = paste(unique(y$error[!is.na(y$error) & nzchar(y$error)]),
                    collapse = " | "),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$case_key, out$identification, out$bounds), , drop = FALSE]
}

compare_lavaan_magmaan <- function(lavaan_summary, magmaan_summary) {
  merged <- merge(lavaan_summary, magmaan_summary,
                  by = c("case_key", "identification", "bounds"), all = TRUE,
                  suffixes = c("_lavaan", "_magmaan"), sort = FALSE)
  out <- data.frame(
    case_key = merged$case_key,
    identification = merged$identification,
    bounds = merged$bounds,
    lavaan_rstarts = merged$rstarts,
    lavaan_converged = merged$converged_lavaan,
    lavaan_admissible = merged$admissible_variances_lavaan,
    lavaan_min_ov_var = merged$min_ov_var_lavaan,
    lavaan_min_lv_var = merged$min_lv_var_lavaan,
    lavaan_n_neg_ov_var = merged$n_neg_ov_var_lavaan,
    lavaan_n_neg_lv_var = merged$n_neg_lv_var_lavaan,
    lavaan_active_variance_bounds =
      merged$active_variance_bound_count_lavaan,
    lavaan_fmin = merged$fmin_lavaan,
    magmaan_converged = merged$converged_magmaan,
    magmaan_stationary = merged$stationary,
    magmaan_admissible = merged$admissible_variances_magmaan,
    magmaan_min_ov_var = merged$min_ov_var_magmaan,
    magmaan_min_lv_var = merged$min_lv_var_magmaan,
    magmaan_n_neg_ov_var = merged$n_neg_ov_var_magmaan,
    magmaan_n_neg_lv_var = merged$n_neg_lv_var_magmaan,
    magmaan_active_variance_bounds =
      merged$active_variance_bound_count_magmaan,
    magmaan_fmin = merged$fmin_magmaan,
    fmin_gap_magmaan_minus_2_lavaan =
      merged$fmin_magmaan - 2.0 * merged$fmin_lavaan,
    stringsAsFactors = FALSE
  )
  out[order(out$case_key, out$identification, out$bounds,
            out$lavaan_rstarts), , drop = FALSE]
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  require_pkg("magmaan")
  require_pkg("lavaan")

  dir.create(experiment_path("results"), recursive = TRUE, showWarnings = FALSE)
  cases <- parse_cases(args$cases)

  lavaan_rows <- list()
  magmaan_rows <- list()
  k_lav <- 0L
  k_mag <- 0L

  for (i in seq_len(nrow(cases))) {
    case <- cases[i, , drop = FALSE]
    message(case$case_key, " ", i, "/", nrow(cases))
    sim <- magmaan::convergence_sim(case$design, n = case$n, seed = case$seed)

    for (identification in args$identifications) {
      for (bounds in args$bounds) {
        message("  magmaan id=", identification, " bounds=", bounds)
        k_mag <- k_mag + 1L
        magmaan_rows[[k_mag]] <- magmaan_fit_row(case, sim, args$optimizer,
                                                 identification, bounds)

        for (rstarts in args$rstarts) {
          message("  lavaan id=", identification, " bounds=", bounds,
                  " rstarts=", rstarts)
          k_lav <- k_lav + 1L
          lavaan_rows[[k_lav]] <- lavaan_fit_row(case, sim, identification,
                                                 bounds, rstarts)
        }
      }
    }
  }

  lavaan_out <- do.call(rbind, lavaan_rows)
  magmaan_out <- do.call(rbind, magmaan_rows)
  write_csv(lavaan_out, experiment_path("results", "lavaan_bounds.csv"))
  write_csv(magmaan_out, experiment_path("results", "magmaan_bounds.csv"))

  summary <- summarize_lavaan_bounds(lavaan_out)
  magmaan_summary <- summarize_magmaan_bounds(magmaan_out)
  comparison <- compare_lavaan_magmaan(summary, magmaan_summary)
  write_csv(summary, experiment_path("results", "summary.csv"))
  write_csv(magmaan_summary, experiment_path("results", "magmaan_summary.csv"))
  write_csv(comparison, experiment_path("results", "engine_comparison.csv"))

  surface_gap <- data.frame(
    component = c("magmaan::magmaan(... estimator='ML', bounds=)",
                  "magmaan::fit_ml()",
                  "magmaan::bounds_*()",
                  "C++ estimate::fit_ml(...)"),
    bounds_threaded = c(TRUE, TRUE, TRUE, TRUE),
    note = c("high-level ML branch forwards bounds to fit_ml()",
             "R wrapper has a bounds formal and forwards it to the Rcpp ML shim",
             "R exposes variance, standard, wide, and loading bound builders",
             "core fitting function accepts Bounds and native bounded optimizers"),
    stringsAsFactors = FALSE
  )
  write_csv(surface_gap, experiment_path("results", "surface_gap.csv"))

  cat("\nHeadline summary\n")
  print(summary[order(summary$case_key, summary$identification,
                      summary$bounds, summary$rstarts), ],
        row.names = FALSE)
}

main()
