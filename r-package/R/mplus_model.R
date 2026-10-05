#' Construct a model from a Mplus input file
#'
#' Imports single- and multiple-group linear SEM: BY, ON, WITH, PON, PWITH,
#' means, variances, starts, fixes, labels and equality numbers, NOCOVARIANCES,
#' and NOMEANSTRUCTURE with INFORMATION = EXPECTED. Names resolve without
#' regard to case and retain the spelling in NAMES; factors retain their first
#' BY spelling. Estimator and execution settings are reported, not imported.
#'
#' CATEGORICAL lists binary and ordinal outcomes (two through ten categories).
#' Thresholds use `[u$k]`, DELTA scales use `{u}`, and THETA residuals
#' use variance statements. Categories and thresholds are completed from data;
#' all groups must contain the same categories. All-ordinal models fit with
#' `estimator = "DWLS"`; mixed, conditional and ML categorical routes report
#' their unsupported route explicitly. Retained-estimate lavaan WLSMV
#' reporting is available through [convention_inference()].
#'
#' Fixed-time polynomial and piecewise growth (`i s | y1@0 y2@1 ...`) imports
#' outcome intercept, growth mean, threshold and scale/residual defaults.
#' Growth accepts one through four factors; free time scores are accepted
#' when enough scores remain fixed to identify the polynomial loadings.
#' MODEL CONSTRAINT imports NEW (starts default to 0.5), explicit and implicit
#' equalities, derived quantities, and nested DO loops. Functions include EXP,
#' LOG, SQRT, PHI and LOG10; `**` denotes power. Continuous ML, LS and FIML
#' use the existing equality backends. All-ordinal nonlinear equalities use
#' constrained LS fitting and expected-information lavaan reporting. Native
#' policy and observed/IJ inference require unimplemented Lagrangian curvature.
#' Inequalities are deliberately refused: active-bound inference
#' requires boundary asymptotics; for variance positivity drop the constraint
#' and use covariance = "psd" or "barrier" in [fit_model()].
#' MODEL INDIRECT imports total IND, specific IND and VIA products, including
#' paths through factors. Derived names are `ind_g<group>_<outcome>_ind_<names>`
#' or `ind_g<group>_<outcome>_via_<mediator>_<predictor>`; specific names retain
#' the written mediator order. Defined estimates and delta-method SEs are
#' available through [compute_defined()]. A free NEW coordinate appears as a
#' `new` partable row, with no observed variable or moment cell; rebuilding
#' always uses the original Mplus source. INFORMATION defaults, MODEL TEST,
#' LOOP and PLOT are reported in notes.
#'
#' NAMES accepts bounded numeric- or letter-suffix ranges; MODEL and
#' USEVARIABLES ranges follow schema order. Latent ranges follow factor-definition
#' order. Labels apply on their physical line; subsequent mentions override
#' earlier parameter specifications. PON and PWITH require equally sized lists.
#' Model/option content must fit within 90 columns; wrap longer statements.
#' TITLE and comments may be longer. Line and
#' block comments are accepted, but a block opened after code must close on
#' its opening line. TITLE, OUTPUT, SAVEDATA and PLOT are reported.
#'
#' DEFINE and DATA transformations must be performed in R; USEOBSERVATIONS and
#' SUBPOPULATION must be applied as case selection in R before fitting.
#' Counts, nominal/censored/survival outcomes, survey weights, mixtures,
#' multilevel models, Bayes and full-information categorical links are outside
#' the linear SEM family: retain those analyses in Mplus. ESEM rotation is
#' replaced only by explicitly specified ordinary BY factors. Bare `@` uses
#' data-dependent Mplus starts; write an explicit `@value` instead. Remove
#' all observed-independent means/variances/WITH mentions to preserve
#' fixed-x conditioning, or bring every observed independent variable into the
#' joint random-X model (e.g. `x1 x2;`). Complete mentions free X means,
#' variances and default covariances; partial mentions are rejected with the
#' exact variance statement that completes the joint model. Recode data-dependent GROUPING forms to explicit
#' integer code/label pairs in R. Categorical summary inputs require individual
#' observations for magmaan's moment preparation.
#'
#' DEFINE, mixtures, multilevel models, ESEM, unsupported name ranges
#' and mixed conditioning on observed covariates are rejected with rule IDs
#' and an explanation of what to write instead. Data descriptions are parsed into
#' `$mplus_data_plan`; [mplus_data()] reads them. This constructor does not
#' read the data file or run Mplus.
#' GROUPING imports explicit integer code = label pairs, ordered by numeric code.
#' Group MODEL sections override the overall model; repeated sections apply
#' cumulatively. CONFIGURAL, METRIC and SCALAR accept one setting at a time.
#' Categorical shortcuts accept CONFIGURAL and SCALAR; METRIC is rejected
#' for binary and ordinal outcomes, following Mplus 9.1.
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
  out$ordered <- parsed$ordered
  out$parameterization <- parsed$parameterization
  attr(out$partable, "magmaan.ordered") <- parsed$ordered
  attr(out$partable, "magmaan.parameterization") <- parsed$parameterization
  out$mplus_observed_x <- parsed$observed_x
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
    fixed_x = parsed$fixed_x, meanstructure = parsed$meanstructure)
  class(out) <- c("magmaan_mplus_model_spec", class(out))
  out
}

#' @export
print.magmaan_mplus_model_spec <- function(x, ...) {
  cat("Mplus model:", nrow(x$partable), "parameter rows\n")
  if (length(x$ordered)) cat("Categorical:", paste(x$ordered, collapse=", "), "(", x$parameterization, ")\n")
  if (nrow(x$mplus_groups)) cat("Groups (", x$group_var, "): ",
    paste(paste0(x$mplus_groups$label, " = ", x$mplus_groups$code), collapse = ", "), "\n", sep = "")
  cat(sum(x$mplus_notes$class == "reported"), "input items reported but not imported; see $mplus_notes\n")
  invisible(x)
}

# Validate only the Mplus adapter; other model frontends keep their data policy.
.validate_mplus_groups <- function(spec, data) {
  if (!is.null(spec$mplus_source) && length(spec$ordered) && is.data.frame(data)) {
    for (v in spec$ordered) {
      if (!v %in% names(data)) stop("[CT01] missing categorical column: ",v,call.=FALSE)
      categories <- sort(unique(as.character(data[[v]][!is.na(data[[v]])])))
      if (length(categories)>10L || length(categories)<2L)
        stop("[CT01] Mplus categorical outcomes require two through ten categories: ",v,call.=FALSE)
      if (nzchar(spec$group_var) && spec$group_var %in% names(data))
        for(g in spec$group_labels) {
          observed <- data[[v]][as.character(data[[spec$group_var]])==g]
          if(!setequal(as.character(observed[!is.na(observed)]),categories))
            stop("[CT07] group ",g," lacks a category of ",v,
              "; Mplus requires every category in every group; supply the missing category or recode all groups consistently",call.=FALSE)
        }
    }
  }
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
