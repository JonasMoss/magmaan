#!/usr/bin/env Rscript
# Ordinal LS / SNLLS probe.
#
# This runner keeps magmaan source untouched. It times the current ordinal
# DWLS/WLS path, an identity-weight ULS proxy, and records native ordinal
# ULS/SNLLS attempts so unsupported surfaces are visible in the CSV output.
#
# Usage:
#   Rscript experiments/06-ordinal-snlls-probe/run_experiment.R [--reps N]

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1L]]$ofile %||% "run_experiment.R",
                mustWork = FALSE)
}

parse_args <- function(args) {
  out <- list(reps = 5L, case_regex = NULL, max_iter = 4000L,
              include_bfi = TRUE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--cases REGEX] ",
        "[--max-iter N] [--skip-bfi]\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--cases") {
      i <- i + 1L
      out$case_regex <- args[[i]]
    } else if (startsWith(a, "--cases=")) {
      out$case_regex <- sub("^--cases=", "", a)
    } else if (a == "--max-iter") {
      i <- i + 1L
      out$max_iter <- as.integer(args[[i]])
    } else if (startsWith(a, "--max-iter=")) {
      out$max_iter <- as.integer(sub("^--max-iter=", "", a))
    } else if (a == "--skip-bfi") {
      out$include_bfi <- FALSE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  if (!is.finite(out$max_iter) || out$max_iter < 1L) {
    stop("--max-iter must be a positive integer", call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

experiment_dir <- dirname(script_path())
project_root <- normalizePath(file.path(experiment_dir, "..", ".."),
                              mustWork = TRUE)
results_dir <- file.path(experiment_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

for (v in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) {
  do.call(Sys.setenv, stats::setNames(list("1"), v))
}

require_pkg <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("required R package is not installed: ", package, call. = FALSE)
  }
  invisible(TRUE)
}

require_pkg("jsonlite")
require_pkg("magmaan")

core <- magmaan::magmaan_core

control <- list(max_iter = args$max_iter, ftol = 1e-12, gtol = 1e-8)

read_json <- function(path) {
  jsonlite::read_json(path, simplifyVector = FALSE)
}

rows_to_matrix <- function(rows) {
  out <- do.call(rbind, lapply(rows, function(x) {
    as.numeric(unlist(x, use.names = FALSE))
  }))
  storage.mode(out) <- "integer"
  out
}

ordered_data_frame <- function(mat, ordered) {
  out <- data.frame(row.names = seq_len(nrow(mat)))
  for (j in seq_along(ordered)) {
    x <- as.integer(mat[, j])
    lv <- seq_len(max(x, na.rm = TRUE))
    out[[ordered[[j]]]] <- base::ordered(x, levels = lv)
  }
  out
}

load_ordinal_fixture <- function(id, label, rel_path) {
  obj <- read_json(file.path(project_root, rel_path))
  mat <- rows_to_matrix(obj$blocks[[1L]]$matrix)
  ordered <- as.character(unlist(obj$ordered, use.names = FALSE))
  list(
    id = id,
    label = label,
    source = rel_path,
    model = obj$input,
    ordered = ordered,
    data = ordered_data_frame(mat, ordered),
    reference = obj
  )
}

load_bfi_fixture <- function() {
  data_obj <- read_json(file.path(
    project_root, "tests/fixtures/parity/bfi_ordinal_dwls/data.json"
  ))
  ref <- read_json(file.path(
    project_root, "tests/fixtures/parity/bfi_ordinal_dwls/reference.json"
  ))
  mat <- rows_to_matrix(data_obj$blocks[[1L]]$matrix)
  ordered <- as.character(unlist(data_obj$ordered, use.names = FALSE))
  list(
    id = "bfi_ordinal_dwls",
    label = "BFI ordinal 5-item CFA",
    source = "tests/fixtures/parity/bfi_ordinal_dwls",
    model = ref$input,
    ordered = ordered,
    data = ordered_data_frame(mat, ordered),
    reference = ref
  )
}

cases <- list(
  load_ordinal_fixture(
    "ordinal_3cat_cfa",
    "Synthetic 3-category CFA",
    "tests/fixtures/ordinal/0001_3cat_cfa.ordinal.json"
  ),
  load_ordinal_fixture(
    "ordinal_mixed_levels_cfa",
    "Synthetic mixed-level CFA",
    "tests/fixtures/ordinal/0008_mixed_levels_cfa.ordinal.json"
  )
)
if (isTRUE(args$include_bfi)) {
  cases[[length(cases) + 1L]] <- load_bfi_fixture()
}
if (!is.null(args$case_regex)) {
  keep <- vapply(cases, function(x) {
    grepl(args$case_regex, x$id) || grepl(args$case_regex, x$label)
  }, logical(1))
  cases <- cases[keep]
}
if (!length(cases)) stop("no cases selected", call. = FALSE)

timed_call <- function(fun, reps) {
  times <- rep(NA_real_, reps)
  value <- NULL
  for (i in seq_len(reps)) {
    gc()
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(fun(), error = function(e) e)
    t1 <- proc.time()[["elapsed"]]
    if (inherits(res, "error")) {
      return(list(ok = FALSE, value = NULL, median_ms = NA_real_,
                  min_ms = NA_real_, max_ms = NA_real_,
                  error = conditionMessage(res)))
    }
    times[[i]] <- 1000 * (t1 - t0)
    value <- res
  }
  list(ok = TRUE, value = value,
       median_ms = stats::median(times),
       min_ms = min(times), max_ms = max(times),
       error = "")
}

condition_message <- function(expr) {
  res <- tryCatch(expr(), error = function(e) e)
  if (inherits(res, "error")) conditionMessage(res) else ""
}

status_of <- function(expr) {
  msg <- condition_message(expr)
  if (nzchar(msg)) list(status = "error", message = msg)
  else list(status = "ok", message = "")
}

identity_weight_stats <- function(stats) {
  out <- stats
  out$W_dwls <- lapply(out$W_dwls, function(W) diag(nrow(W)))
  class(out) <- class(stats)
  out
}

polychoric_sample_stats <- function(stats) {
  list(S = stats$R, nobs = stats$nobs, mean = NULL)
}

reference_theta <- function(case, estimator) {
  fit <- case$reference$fits[[estimator]]
  if (is.null(fit) || is.null(fit$theta_hat)) return(numeric())
  as.numeric(unlist(fit$theta_hat, use.names = FALSE))
}

theta_diff <- function(fit, ref) {
  if (!length(ref) || is.null(fit$theta)) return(NA_real_)
  theta <- as.numeric(fit$theta)
  if (length(theta) != length(ref)) return(NA_real_)
  max(abs(theta - ref), na.rm = TRUE)
}

first_scalar <- function(x, default = NA_real_) {
  if (is.null(x) || !length(x)) return(default)
  x[[1L]]
}

sanitize <- function(x) {
  x <- as.character(x %||% "")
  gsub("[\r\n]+", " ", x)
}

fit_row <- function(case, condition, estimator, objective, native,
                    fit_timing, whole_timing, ref_estimator = "") {
  status <- if (isTRUE(fit_timing$ok)) "ok" else "error"
  fit <- fit_timing$value
  ref <- if (nzchar(ref_estimator)) reference_theta(case, ref_estimator)
         else numeric()
  data.frame(
    case_id = case$id,
    case_label = case$label,
    condition = condition,
    estimator = estimator,
    objective = objective,
    native = isTRUE(native),
    status = status,
    error = sanitize(fit_timing$error),
    reps = args$reps,
    fit_ms_median = fit_timing$median_ms,
    fit_ms_min = fit_timing$min_ms,
    fit_ms_max = fit_timing$max_ms,
    whole_ms_median = whole_timing$median_ms,
    whole_ms_min = whole_timing$min_ms,
    whole_ms_max = whole_timing$max_ms,
    fmin = if (status == "ok") first_scalar(fit$fmin) else NA_real_,
    chisq = if (status == "ok") first_scalar(fit$ntotal) * first_scalar(fit$fmin) else NA_real_,
    npar = if (status == "ok" && !is.null(fit$theta)) length(fit$theta) else NA_integer_,
    iterations = if (status == "ok") as.integer(first_scalar(fit$iterations, NA_integer_)) else NA_integer_,
    f_evals = if (status == "ok") as.integer(first_scalar(fit$f_evals, NA_integer_)) else NA_integer_,
    g_evals = if (status == "ok") as.integer(first_scalar(fit$g_evals, NA_integer_)) else NA_integer_,
    converged = if (status == "ok") isTRUE(fit$converged) else FALSE,
    optimizer_status = if (status == "ok") as.character(fit$optimizer_status %||% "") else "",
    grad_norm = if (status == "ok") first_scalar(fit$grad_norm) else NA_real_,
    ref_estimator = ref_estimator,
    max_abs_theta_diff = if (status == "ok") theta_diff(fit, ref) else NA_real_,
    stringsAsFactors = FALSE
  )
}

construction_rows <- list()
fit_rows <- list()
attempt_rows <- list()

cat(sprintf(
  "ordinal-snlls-probe: magmaan %s, %s, reps=%d, max_iter=%d\n",
  as.character(utils::packageVersion("magmaan")),
  R.version.string, args$reps, args$max_iter
))

for (case in cases) {
  cat(sprintf("  %-26s construct", case$id))
  spec_ord <- magmaan::model_spec(
    case$model, ordered = case$ordered, parameterization = "delta"
  )
  spec_cont <- magmaan::model_spec(case$model)
  construct_fun <- function() core$data_ordinal_stats_from_df(case$data, spec_ord)
  construct <- timed_call(construct_fun, args$reps)
  if (!construct$ok) {
    cat(sprintf(" ERROR %s\n", construct$error))
    construction_rows[[length(construction_rows) + 1L]] <- data.frame(
      case_id = case$id, case_label = case$label, source = case$source,
      status = "error", error = sanitize(construct$error),
      n_obs = nrow(case$data), n_vars = length(case$ordered),
      n_moments = NA_integer_, n_threshold_moments = NA_integer_,
      n_assoc_moments = NA_integer_, levels = "",
      construct_ms_median = NA_real_, construct_ms_min = NA_real_,
      construct_ms_max = NA_real_, stringsAsFactors = FALSE
    )
    next
  }
  stats <- construct$value
  threshold_count <- sum(vapply(stats$thresholds, length, integer(1)))
  moment_count <- sum(vapply(stats$NACOV, nrow, integer(1)))
  level_text <- paste(as.integer(unlist(stats$n_levels, use.names = FALSE)),
                      collapse = ";")
  construction_rows[[length(construction_rows) + 1L]] <- data.frame(
    case_id = case$id,
    case_label = case$label,
    source = case$source,
    status = "ok",
    error = "",
    n_obs = nrow(case$data),
    n_vars = length(case$ordered),
    n_moments = moment_count,
    n_threshold_moments = threshold_count,
    n_assoc_moments = moment_count - threshold_count,
    levels = level_text,
    construct_ms_median = construct$median_ms,
    construct_ms_min = construct$min_ms,
    construct_ms_max = construct$max_ms,
    stringsAsFactors = FALSE
  )
  cat(sprintf(" %.2fms\n", construct$median_ms))

  stats_uls <- identity_weight_stats(stats)
  pseudo <- polychoric_sample_stats(stats)

  fit_specs <- list(
    list(
      condition = "DWLS",
      estimator = "DWLS",
      objective = "ordinal thresholds + polychorics, diagonal NACOV weight",
      native = TRUE,
      ref = "DWLS",
      fit = function() core$fit_dwls_ordinal(spec_ord, stats,
                                             control = control),
      whole = function() {
        s <- construct_fun()
        core$fit_dwls_ordinal(spec_ord, s, control = control)
      }
    ),
    list(
      condition = "WLS",
      estimator = "WLS",
      objective = "ordinal thresholds + polychorics, full NACOV inverse",
      native = TRUE,
      ref = "WLS",
      fit = function() core$fit_wls_ordinal(spec_ord, stats,
                                            control = control),
      whole = function() {
        s <- construct_fun()
        core$fit_wls_ordinal(spec_ord, s, control = control)
      }
    ),
    list(
      condition = "ULS_identity_proxy",
      estimator = "ULS",
      objective = "ordinal thresholds + polychorics, identity weight injected through W_dwls",
      native = FALSE,
      ref = "",
      fit = function() core$fit_dwls_ordinal(spec_ord, stats_uls,
                                             control = control),
      whole = function() {
        s <- identity_weight_stats(construct_fun())
        core$fit_dwls_ordinal(spec_ord, s, control = control)
      }
    ),
    list(
      condition = "ULS_SNLLS_polychoric_proxy",
      estimator = "ULS-SNLLS",
      objective = "continuous SNLLS on the polychoric correlation matrix; thresholds ignored",
      native = FALSE,
      ref = "",
      fit = function() core$fit_uls_snlls(spec_cont, pseudo,
                                          control = control),
      whole = function() {
        s <- construct_fun()
        core$fit_uls_snlls(spec_cont, polychoric_sample_stats(s),
                           control = control)
      }
    )
  )

  for (fs in fit_specs) {
    cat(sprintf("    %-28s", fs$condition))
    ft <- timed_call(fs$fit, args$reps)
    wt <- if (ft$ok) timed_call(fs$whole, args$reps)
          else list(ok = FALSE, median_ms = NA_real_, min_ms = NA_real_,
                    max_ms = NA_real_, error = ft$error, value = NULL)
    fit_rows[[length(fit_rows) + 1L]] <- fit_row(
      case, fs$condition, fs$estimator, fs$objective, fs$native,
      ft, wt, fs$ref
    )
    if (ft$ok) {
      cat(sprintf(" ok fit=%.2fms whole=%.2fms\n",
                  ft$median_ms, wt$median_ms))
    } else {
      cat(sprintf(" ERROR %s\n", ft$error))
    }
  }

  attempts <- list(
    list(
      condition = "native_magmaan_ULS_with_ordered",
      estimator = "ULS",
      description = "magmaan(..., estimator = 'ULS', ordered = ...) on ordered factors",
      expr = function() magmaan::magmaan(
        case$model, case$data, estimator = "ULS", ordered = case$ordered,
        control = control
      )
    ),
    list(
      condition = "native_fit_uls_snlls_with_ordinal_stats",
      estimator = "ULS-SNLLS",
      description = "fit_uls_snlls() called with magmaan_ordinal_data",
      expr = function() core$fit_uls_snlls(spec_ord, stats,
                                           control = control)
    )
  )
  for (a in attempts) {
    st <- status_of(a$expr)
    attempt_rows[[length(attempt_rows) + 1L]] <- data.frame(
      case_id = case$id,
      case_label = case$label,
      condition = a$condition,
      estimator = a$estimator,
      description = a$description,
      status = st$status,
      message = sanitize(st$message),
      stringsAsFactors = FALSE
    )
  }
}

metadata <- data.frame(
  key = c("magmaan_version", "R_version", "reps", "max_iter",
          "case_regex", "include_bfi", "construction_scope"),
  value = c(
    as.character(utils::packageVersion("magmaan")),
    R.version.string,
    as.character(args$reps),
    as.character(args$max_iter),
    args$case_regex %||% "",
    as.character(args$include_bfi),
    paste(
      "data_ordinal_stats_from_df builds thresholds, polychorics,",
      "NACOV, W_dwls, and W_wls together"
    )
  ),
  stringsAsFactors = FALSE
)

write_or_empty <- function(rows, path) {
  if (length(rows)) {
    utils::write.csv(do.call(rbind, rows), path, row.names = FALSE)
  } else {
    utils::write.csv(data.frame(), path, row.names = FALSE)
  }
}

utils::write.csv(metadata, file.path(results_dir, "metadata.csv"),
                 row.names = FALSE)
write_or_empty(construction_rows, file.path(results_dir, "construction.csv"))
write_or_empty(fit_rows, file.path(results_dir, "fits.csv"))
write_or_empty(attempt_rows, file.path(results_dir, "attempts.csv"))

cat(sprintf("wrote %s\n", results_dir))
