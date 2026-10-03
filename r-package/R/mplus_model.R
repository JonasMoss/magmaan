#' Construct a model from a Mplus input file
#'
#' Imports single-group continuous linear SEM: BY, ON, WITH, PON, PWITH,
#' means, variances, starts, fixes, labels and equality numbers, NOCOVARIANCES,
#' and NOMEANSTRUCTURE with INFORMATION = EXPECTED. Names resolve without
#' regard to case and retain the spelling in NAMES; factors retain their first
#' BY spelling. Estimator and execution settings are reported, not imported.
#'
#' DEFINE, multiple groups, categorical outcomes, growth, MODEL CONSTRAINT,
#' MODEL INDIRECT, mixtures, multilevel models, ESEM, unsupported name ranges
#' and mixed conditioning on observed covariates are rejected with rule IDs
#' and an explanation of what to write instead. Data descriptions are retained
#' as notes; this function does not read the data file or run Mplus.
#' @param input One string containing the whole input text.
#' @param file Path to an input file. Supply exactly one of input or file.
#' @return A magmaan_model_spec with original mplus_source, a data.frame of
#'   mplus_notes (class, rule, line, col, message), and a lavaan syntax projection.
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
  out$mplus_notes <- parsed$notes
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
  cat(sum(x$mplus_notes$class == "reported"), "input items reported but not imported; see $mplus_notes\n")
  invisible(x)
}
