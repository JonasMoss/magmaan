pilot_repo_root <- function(start = getwd()) {
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

pilot_path <- function(...) {
  file.path(pilot_repo_root(), "papers", "convergence-note", "pilot", ...)
}

pilot_ensure_dirs <- function() {
  for (d in c("results", "tables", "report")) {
    dir.create(pilot_path(d), recursive = TRUE, showWarnings = FALSE)
  }
}

pilot_env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) return(default)
  out <- suppressWarnings(as.integer(value))
  if (!length(out) || is.na(out) || out < 1L) {
    stop(name, " must be a positive integer", call. = FALSE)
  }
  out
}

pilot_env_list <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) return(default)
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

pilot_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package `", pkg, "` is required. Install/reinstall it before running this pilot.",
         call. = FALSE)
  }
}

pilot_require_magmaan <- function() {
  pilot_require("magmaan")
  exports <- getNamespaceExports("magmaan")
  needed <- c("magmaan", "model_spec", "convergence_sim", "convergence_sim_catalog")
  missing <- setdiff(needed, exports)
  if (length(missing)) {
    stop("Installed magmaan is missing export(s): ",
         paste(missing, collapse = ", "),
         ". Reinstall the root r-package from this checkout.", call. = FALSE)
  }
}

pilot_fit_once <- function(model, data, optimizer, control, estimator = "ML") {
  started <- proc.time()[["elapsed"]]
  out <- tryCatch({
    fit <- magmaan::magmaan(model, data, estimator = estimator,
                            optimizer = optimizer, control = control)
    list(
      ok = TRUE,
      converged = isTRUE(fit$converged),
      optimizer_status = as.character(fit$optimizer_status %||% NA_character_),
      fmin = as.numeric(fit$fmin %||% NA_real_),
      iterations = as.integer(fit$iterations %||% NA_integer_),
      grad_inf_norm = as.numeric(fit$grad_inf_norm %||% NA_real_),
      audit_status = as.character((fit$audit %||% list())$advisory_status %||% NA_character_),
      error_kind = NA_character_,
      error = NA_character_
    )
  }, error = function(e) {
    msg <- conditionMessage(e)
    kind <- sub("^.*\\[([^]]+)\\].*$", "\\1", msg)
    if (identical(kind, msg)) kind <- NA_character_
    list(
      ok = FALSE,
      converged = FALSE,
      optimizer_status = NA_character_,
      fmin = NA_real_,
      iterations = NA_integer_,
      grad_inf_norm = NA_real_,
      audit_status = NA_character_,
      error_kind = kind,
      error = msg
    )
  })
  out$elapsed_sec <- proc.time()[["elapsed"]] - started
  out
}

pilot_result_row <- function(meta, fit) {
  data.frame(
    meta,
    ok = fit$ok,
    converged = fit$converged,
    optimizer_status = fit$optimizer_status,
    fmin = fit$fmin,
    iterations = fit$iterations,
    grad_inf_norm = fit$grad_inf_norm,
    audit_status = fit$audit_status,
    elapsed_sec = fit$elapsed_sec,
    error_kind = fit$error_kind,
    error = fit$error,
    stringsAsFactors = FALSE
  )
}

pilot_bind <- function(rows) {
  if (!length(rows)) return(data.frame())
  do.call(rbind, rows)
}

pilot_write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  message("Wrote ", path)
  invisible(path)
}

pilot_read_json <- function(path) {
  pilot_require("jsonlite")
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

pilot_matrix <- function(x, names = NULL) {
  out <- do.call(rbind, lapply(x, unlist, use.names = FALSE))
  storage.mode(out) <- "double"
  if (!is.null(names)) dimnames(out) <- list(names, names)
  out
}

pilot_vector <- function(x, names = NULL) {
  if (is.null(x)) return(NULL)
  out <- unlist(x, use.names = FALSE)
  storage.mode(out) <- "double"
  if (!is.null(names)) names(out) <- names
  out
}

pilot_sample_stats_from_case <- function(case) {
  ov <- unlist(case$ov_names, use.names = FALSE)
  S <- pilot_matrix(case$sample_cov, ov)
  mean <- pilot_vector(case$sample_mean, ov)
  list(S = list(S), mean = list(mean), nobs = as.integer(case$n_obs))
}

pilot_spec_from_case <- function(case) {
  model_type <- if (identical(case$lavaan_function, "growth")) "growth" else "sem"
  magmaan::model_spec(
    case$model,
    meanstructure = isTRUE(case$meanstructure),
    fixed_x = isTRUE(case$fixed_x),
    model_type = model_type
  )
}

pilot_case_index <- function() {
  manifest <- pilot_read_json(file.path(pilot_repo_root(), "tests", "fixtures",
                                        "textbook_corpus", "manifest.json"))
  strict <- vapply(manifest$cases, function(x) isTRUE(x$strict_parity), logical(1))
  retained <- vapply(manifest$cases, function(x) identical(x$status, "retained"), logical(1))
  continuous <- vapply(manifest$cases, function(x) identical(x$measurement_kind, "continuous"),
                       logical(1))
  manifest$cases[retained & strict & continuous]
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
