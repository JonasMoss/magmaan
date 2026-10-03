#' Construct a model from a Mplus input file
#'
#' Imports single- and multiple-group continuous linear SEM: BY, ON, WITH, PON, PWITH,
#' means, variances, starts, fixes, labels and equality numbers, NOCOVARIANCES,
#' and NOMEANSTRUCTURE with INFORMATION = EXPECTED. Names resolve without
#' regard to case and retain the spelling in NAMES; factors retain their first
#' BY spelling. Estimator and execution settings are reported, not imported.
#'
#' DEFINE, categorical outcomes, growth, MODEL CONSTRAINT,
#' MODEL INDIRECT, mixtures, multilevel models, ESEM, unsupported name ranges
#' and mixed conditioning on observed covariates are rejected with rule IDs
#' and an explanation of what to write instead. Data descriptions are parsed into
#' `$mplus_data_plan`; [mplus_data()] reads them. This constructor does not
#' read the data file or run Mplus.
#' GROUPING imports explicit integer code = label pairs, ordered by numeric code.
#' Group MODEL sections override the overall model; repeated sections apply
#' cumulatively. CONFIGURAL, METRIC and SCALAR accept one setting at a time.
#' Group-only changes to variable roles are rejected; put the relation in the
#' overall model and fix it in the other groups instead. Variance identification
#' in a shortcut uses free first loadings and explicit factor variances at one.
#' [mplus_data()] drops unlisted GROUPING codes with counts. Separate FILE
#' groups retain source labels, use `.mplus_group`, and follow FILE order.
#' Summary NGROUPS labels are g1, g2, ... . Summary data without MEANS
#' have no mean structure and reject explicit means/intercepts.
#' @param input One string containing the whole input text.
#' @param file Path to an input file. Supply exactly one of input or file.
#' @return A magmaan_model_spec with original mplus_source, a data.frame of
#'   mplus_notes (class, rule, line, col, message), and a lavaan syntax projection.
#'   Grouped specs also carry group_var, ordered integer-code group_labels and
#'   mplus_groups (a data.frame with label and code columns).
#'   Rebuilding uses mplus_source; construction options cannot be overridden.
#' @export
mplus_model <- function(input = NULL, file = NULL) {
  if (is.null(input) == is.null(file)) {
    stop("mplus_model(): supply exactly one of `input` or `file`", call. = FALSE)
  }
  if (!is.null(file)) {
    if (!is.character(file) || length(file) != 1L || is.na(file))
      stop("mplus_model(): `file` must be one non-missing path", call. = FALSE)
    input <- readChar(file, nchars = file.info(file)$size, useBytes = TRUE)
  }
  if (!is.character(input) || length(input) != 1L || is.na(input))
    stop("mplus_model(): `input` must be one non-missing string", call. = FALSE)
  parsed <- mplus_model_impl(input)
  out <- as_magmaan_model_spec(parsed$partable)
  out$syntax <- parsed$syntax
  out$mplus_source <- input
  out$mplus_data_plan <- parsed$data_plan
  out$mplus_input_dir <- if (is.null(file)) NULL else dirname(normalizePath(file))
  out$mplus_notes <- parsed$notes
  out$group_var <- parsed$group_var
  out$group_labels <- parsed$group_labels
  out$mplus_groups <- parsed$groups
  out$requested_meanstructure <- parsed$meanstructure
  out$options <- list(auto_var = FALSE, auto_cov_lv_x = FALSE,
    auto_cov_y = FALSE, auto_fix_first = FALSE, auto_fix_single = FALSE,
    fixed_x = TRUE, meanstructure = parsed$meanstructure)
  class(out) <- c("magmaan_mplus_model_spec", class(out))
  out
}

#' @export
print.magmaan_mplus_model_spec <- function(x, ...) {
  cat("Mplus model:", nrow(x$partable), "parameter rows\n")
  if (nrow(x$mplus_groups)) cat("Groups (", x$group_var, "): ",
    paste(paste0(x$mplus_groups$label, " = ", x$mplus_groups$code), collapse = ", "), "\n", sep = "")
  cat(sum(x$mplus_notes$class == "reported"), "input items reported but not imported; see $mplus_notes\n")
  invisible(x)
}

# Validate only the Mplus adapter; other model frontends keep their data policy.
.validate_mplus_groups <- function(spec, data) {
  if (is.null(spec$mplus_source) || !nzchar(spec$group_var) || !is.data.frame(data)) return(invisible(NULL))
  if (!spec$group_var %in% names(data)) stop("[MG03] grouping column missing: ", spec$group_var, call. = FALSE)
  codes <- as.character(data[[spec$group_var]])
  bad <- !is.na(codes) & !codes %in% spec$group_labels
  if (any(bad)) {
    counts <- table(codes[bad])
    stop("[MG03] unlisted GROUPING codes: ",
      paste(paste0(names(counts), " (", as.integer(counts), " rows)"), collapse = ", "),
      "; Mplus drops those rows; magmaan requires you to filter them in R before fitting, or to use mplus_data().",
      call. = FALSE)
  }
  invisible(NULL)
}
