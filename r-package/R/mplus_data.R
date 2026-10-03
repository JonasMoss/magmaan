#' Read data described by a Mplus model
#'
#' Reads numeric individual or summary ASCII data using the C++ data plan.
#' Relative paths are tried in the working directory, then beside the input
#' file. Free format accepts blanks, tabs and commas, wrapped observations,
#' and discards extra fields at the end of a completed observation's record.
#' Fixed FORMAT supports Fw.d/w.d, X skips, Tn positions, record breaks and
#' repeated groups. Implied decimals apply only without an explicit point.
#' Missing flags compare after that scaling.
#'
#' Separate-file groups use a reserved `.mplus_group` column containing FILE
#' labels in declaration order. NGROUPS summary groups are named g1, g2, ... .
#' GROUPING drops unlisted codes and reports counts. NOBSERVATIONS limits
#' individual observations per file. LISTWISE and analysis-sample deletion
#' rules are only reported: use `fit_model(..., missing = "listwise")` to
#' remove incomplete rows or `missing = "fiml"` for FIML; explicitly filter
#' missing covariates to reproduce Mplus's conditional analysis sample.
#'
#' Summary covariances use divisor N, as Mplus does; no lavaan-style
#' (N-1)/N rescaling is applied. Correlations are multiplied by supplied SDs;
#' without SDs they are returned as unit-variance covariance matrices.
#' @param model A `magmaan_mplus_model_spec` from [mplus_model()].
#' @param file Optional path override, one path per separate-file group.
#' @return Individual data as a data.frame, or summary moments as a list with
#'   S, mean and nobs accepted by [fit_model()]. Attribute `mplus_data_report`
#'   records paths, counts, dropped codes and unapplied sample rules.
#' @export
mplus_data <- function(model, file = NULL) {
  if (!inherits(model, "magmaan_mplus_model_spec"))
    stop("mplus_data(): expected a Mplus model spec", call. = FALSE)
  plan <- model$mplus_data_plan
  paths <- if (is.null(file)) vapply(plan$files, `[[`, "", "path") else file
  expected <- if (plan$file_groups) plan$n_groups else 1L
  if (!is.character(paths) || length(paths) != expected || anyNA(paths) || any(!nzchar(paths)))
    stop("[CL02] supply one data file per FILE group (or one shared file)", call. = FALSE)
  paths <- vapply(paths, function(path) {
    candidates <- path
    if (!is.null(model$mplus_input_dir) && !grepl("^/", path))
      candidates <- c(candidates, file.path(model$mplus_input_dir, path))
    found <- candidates[file.exists(candidates)]
    if (!length(found)) stop("[CL02] data file not found: ", path, call. = FALSE)
    normalizePath(found[1L])
  }, "")
  notes <- c("LISTWISE and Mplus analysis-sample deletion rules are not applied by the reader",
    if (plan$listwise) "LISTWISE = ON: use fit_model(..., missing = 'listwise')")
  report <- list(paths = paths, requested_n = plan$n_observations, notes = notes)
  if (nzchar(plan$matrix_type)) {
    out <- .mplus_summary(paths, plan)
    if (plan$matrix_type %in% c("CORRELATION", "FULLCORR") && !plan$standard_deviations)
      report$notes <- c(report$notes, "CORRELATION without STDEVIATIONS is interpreted as covariance with unit variances")
    report$nobs <- out$nobs
  } else {
    blocks <- lapply(seq_along(paths), function(g) {
      lines <- readLines(paths[g], warn = FALSE)
      if (any(nchar(lines, type = "bytes") > 10000L)) stop("[DA01] record exceeds 10000 characters", call. = FALSE)
      n <- if (length(plan$n_observations)) plan$n_observations[g] else Inf
      x <- if (length(plan$format)) .mplus_fixed(lines, plan, n) else .mplus_free(lines, plan, n)
      x <- as.data.frame(x, optional = TRUE)
      names(x) <- plan$names
      if (plan$file_groups) x[[".mplus_group"]] <- model$mplus_groups$label[g]
      x
    })
    report$read_n <- vapply(blocks, nrow, integer(1))
    out <- do.call(rbind, blocks)
    if (nzchar(model$group_var) && !plan$file_groups) {
      code <- as.character(out[[model$group_var]])
      bad <- is.na(code) | !code %in% model$group_labels
      report$dropped_group_codes <- table(code[bad], useNA = "ifany")
      out <- out[!bad, , drop = FALSE]
    }
    keep <- unique(c(plan$analysis, if (nzchar(model$group_var)) model$group_var))
    out <- out[, keep, drop = FALSE]
    rownames(out) <- NULL
    report$nobs <- if (nzchar(model$group_var)) table(factor(as.character(out[[model$group_var]]), levels = model$group_labels)) else nrow(out)
    if (nzchar(model$group_var) && any(report$nobs == 0L)) stop("[MG02] data group has no observations", call. = FALSE)
  }
  attr(out, "mplus_data_report") <- report
  out
}

.mplus_value <- function(field, variable, plan, decimals = 0L) {
  field <- trimws(field)
  if (nzchar(plan$missing_symbol) &&
      (field == plan$missing_symbol || (plan$missing_symbol == "BLANK" && !nzchar(field)))) return(NA_real_)
  if (!nzchar(field) && decimals >= 0L) return(0)
  if (!nzchar(field) || !grepl("^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eEdD][+-]?[0-9]+)?$", field))
    stop("[DA01] non-numeric field for ", variable, ": '", field, "'", call. = FALSE)
  value <- as.numeric(gsub("[dD]", "e", field))
  if (!is.finite(value)) stop("[DA01] non-finite numeric field", call. = FALSE)
  if (decimals && !grepl(".", field, fixed = TRUE)) value <- value / 10^decimals
  for (rule in plan$missing) if (variable %in% rule$variables && value %in% rule$values) return(NA_real_)
  value
}

.mplus_free <- function(lines, plan, limit = Inf) {
  p <- length(plan$names)
  rows <- list(); pending <- character()
  for (line in lines) {
    if (length(rows) >= limit) break
    if (grepl(",\\s*,|^\\s*,|,\\s*$", line, perl = TRUE)) stop("[DA06] empty comma field", call. = FALSE)
    fields <- strsplit(trimws(line), "[ ,\t]+")[[1L]]
    fields <- fields[nzchar(fields)]
    pending <- c(pending, fields)
    if (length(pending) >= p) {
      rows[[length(rows) + 1L]] <- vapply(seq_len(p), function(j) .mplus_value(pending[j], plan$names[j], plan), 0.0)
      pending <- character() # Mplus ignores the rest of the completed record.
    }
  }
  if (length(pending)) stop("[DA01] incomplete observation: wrong number of values", call. = FALSE)
  if (is.finite(limit) && length(rows) < limit) stop("[DA01] fewer observations than NOBSERVATIONS", call. = FALSE)
  if (!length(rows)) stop("[DA01] no observations", call. = FALSE)
  do.call(rbind, rows)
}

.mplus_fixed <- function(lines, plan, limit = Inf) {
  p <- length(plan$names); row <- 1L; rows <- list()
  fields <- sum(vapply(plan$format, function(op) op$kind == 0L, TRUE))
  if (fields < p) stop("[DA01] FORMAT supplies fewer fields than NAMES", call. = FALSE)
  while (row <= length(lines) && length(rows) < limit) {
    col <- 1L; values <- numeric(); field_count <- 0L
    for (op in plan$format) {
      if (op$kind == 3L) { row <- row + 1L; col <- 1L; next }
      if (op$kind == 1L) { col <- col + op$width; next }
      if (op$kind == 2L) { col <- op$width; next }
      if (row > length(lines)) stop("[DA01] incomplete fixed-format observation", call. = FALSE)
      field_count <- field_count + 1L
      field <- substr(lines[row], col, col + op$width - 1L)
      col <- col + op$width
      if (field_count <= p) values <- c(values, .mplus_value(field, plan$names[field_count], plan, op$decimals))
    }
    rows[[length(rows) + 1L]] <- values
    row <- row + 1L
  }
  if (is.finite(limit) && length(rows) < limit) stop("[DA01] fewer observations than NOBSERVATIONS", call. = FALSE)
  if (!length(rows)) stop("[DA01] no observations", call. = FALSE)
  do.call(rbind, rows)
}

.mplus_summary <- function(paths, plan) {
  p <- length(plan$names); full <- plan$matrix_type %in% c("FULLCOV", "FULLCORR")
  count <- if (full) p*p else p*(p+1L)/2L
  blocks <- lapply(paths, function(path) {
    lines <- readLines(path, warn = FALSE)
    row <- 1L
    take <- function(n) {
      values <- numeric()
      while (length(values) < n && row <= length(lines)) {
        line <- lines[row]; row <<- row + 1L
        if (grepl(",\\s*,|^\\s*,|,\\s*$", line, perl = TRUE)) stop("[DA06] empty summary comma field", call. = FALSE)
        fields <- strsplit(trimws(line), "[ ,\t]+")[[1L]]; fields <- fields[nzchar(fields)]
        values <- c(values, vapply(fields, function(f) .mplus_value(f, "", plan), 0.0))
      }
      if (length(values) != n || anyNA(values)) stop("[DA02] wrong number of summary values or missing moments", call. = FALSE)
      values
    }
    groups <- if (plan$file_groups) 1L else plan$n_groups
    out <- vector("list", groups)
    for (g in seq_len(groups)) {
      mean <- if (plan$means) take(p) else numeric(p)
      sd <- if (plan$standard_deviations) take(p) else rep(1, p)
      vals <- take(count)
      if (full) S <- matrix(vals, p, p, byrow = TRUE) else {
        S <- matrix(0, p, p); S[upper.tri(S, diag = TRUE)] <- vals
        S[lower.tri(S)] <- t(S)[lower.tri(S)]
      }
      if (!isTRUE(all.equal(S, t(S), tolerance = 1e-10))) stop("[DA02] full summary matrix must be symmetric", call. = FALSE)
      if (plan$matrix_type %in% c("CORRELATION", "FULLCORR")) S <- S * outer(sd, sd)
      dimnames(S) <- list(plan$names, plan$names); names(mean) <- plan$names
      selected <- match(plan$analysis, plan$names)
      out[[g]] <- list(S = S[selected, selected, drop = FALSE], mean = mean[selected])
    }
    if (row <= length(lines) && any(nzchar(trimws(lines[row:length(lines)])))) stop("[MG02] extra summary values or wrong group count", call. = FALSE)
    out
  })
  blocks <- unname(unlist(blocks, recursive = FALSE))
  list(S = lapply(blocks, `[[`, "S"), mean = lapply(blocks, `[[`, "mean"), nobs = plan$n_observations)
}
