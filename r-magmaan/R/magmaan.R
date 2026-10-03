# The ordinary-user entry points. This file handles arguments only:
# magmaan_model() builds the model specification once with
# magmaanlab::model_spec() and freezes the data schema; magmaan() fits it with
# magmaanlab::fit_model() and computes inference with infer(). The design is
# the ordinary API in project/design/r-interface-vision.md.

.continuous_estimators <- c("ML", "FIML", "ML2S", "GLS", "ULS", "WLS")
.ordered_estimators <- c("DWLS", "WLS", "ULS")

# lavaan names that bundle an estimator with a standard-error or test correction,
# mapped to the plain estimator magmaan() accepts instead.
.bundled_estimators <- c(
  MLM = "ML", MLMV = "ML", MLMVS = "ML", MLR = "ML", MLF = "ML",
  WLSM = "DWLS", WLSMV = "DWLS", WLSMVS = "DWLS",
  ULSM = "ULS", ULSMV = "ULS", ULSMVS = "ULS"
)

# Arguments magmaan() took before 0.2.0, with where each one went.
.model_arguments <- c("ordered", "group", "group.equal", "group.partial",
                      "identification", "parameterization")
.removed_arguments <- c(
  psd = "`psd` is now covariance = \"psd\"",
  start = "`start` is now options = list(start = ...)",
  fixed.x = paste(
    "`fixed.x` was removed: magmaan() fits the joint model with random",
    "covariates. Fixed-x fits remain available in magmaanlab::fit_model()"),
  meanstructure = "`meanstructure` was removed: every model has a mean structure",
  missing = paste(
    "`missing` was removed: ML, GLS, ULS and the ordinal estimators delete",
    "incomplete rows listwise, FIML and ML2S use them. Pairwise deletion is",
    "available in magmaanlab::fit_model()"),
  cluster = paste(
    "`cluster` was removed: two-level fitting is available in",
    "magmaanlab::fit_model(cluster = )")
)

# Session state: the experimental-barrier message is shown once.
.session <- new.env(parent = emptyenv())

#' Construct a model for magmaan()
#'
#' Builds the model once: its parameters, structural choices and data schema.
#' Fit it to any number of datasets with [magmaan()]; fitting never modifies
#' it. Grouped and ordinal models need `prototype` to declare their groups and
#' categories.
#'
#' Every model has a mean structure: when the syntax does not restrict the
#' means, each observed variable gets a free intercept, which changes no other
#' estimate, standard error or test. Exogenous observed covariates are random
#' variables with free variances and covariances (there is no `fixed.x`).
#'
#' @param model lavaan model syntax, one string, or a model specification from
#'   `magmaanlab::model_spec()` carrying lavaan syntax. A specification supplies
#'   its own ordered variables, grouping, identification, parameterization and
#'   equality constraints; explicit arguments must agree with it. A
#'   specification that fixes exogenous covariates (`fixed_x = TRUE` with
#'   observed covariates) is rejected.
#' @param prototype A data frame that declares the data schema: the model's
#'   variables, the grouping column's groups and their order, and the
#'   categories of the ordered variables and their order. Factor levels
#'   declare groups and categories completely, including levels without rows,
#'   so a zero-row data frame suffices. Otherwise groups follow their first
#'   appearance and categories their sorted values. Only the schema is used:
#'   no estimates, starts or weights come from the prototype's rows.
#' @param ordered Names of the ordered (categorical) variables.
#' @param group Name of the grouping column.
#' @param group.equal,group.partial Cross-group equality constraints, as in
#'   lavaan.
#' @param identification `"marker"` (first loading fixed to one) or `"std.lv"`
#'   (latent variances fixed to one).
#' @param parameterization `"delta"` or `"theta"`, for ordered variables.
#' @return An object of class `magmaan_model`.
#' @section Schema checks:
#' Each fit checks its data against the frozen schema. Data with an undeclared
#' group or category, changed factor levels, a declared group without rows or a
#' declared category without complete observations raise a condition of class
#' `magmaan_schema_error` whose `reason` is `"undeclared_group"`,
#' `"undeclared_category"`, `"changed_levels"`, `"empty_group"` or
#' `"empty_category"`. Construct the model again to change its schema.
#'
#' Group order follows the prototype's factor levels; lavaan instead orders
#' groups by first appearance. Group-specific modifiers such as `c(a, b)*x`
#' follow the model's order, and so does the reference group whose latent
#' means are fixed under `group.equal = "intercepts"`.
#' @examples
#' if (requireNamespace("lavaan", quietly = TRUE)) {
#'   d <- lavaan::HolzingerSwineford1939
#'   m <- magmaan_model("visual =~ x1 + x2 + x3", prototype = d,
#'                      group = "school", group.equal = "loadings")
#'   fit <- magmaan(m, d)
#' }
#' @export
magmaan_model <- function(model, prototype = NULL,
                          ordered = NULL, group = NULL,
                          group.equal = NULL, group.partial = NULL,
                          identification = "marker", parameterization = "delta") {
  caller <- "magmaan_model()"
  supplied <- c(ordered = !missing(ordered), group = !missing(group),
                group.equal = !missing(group.equal), group.partial = !missing(group.partial),
                identification = !missing(identification),
                parameterization = !missing(parameterization))
  if (inherits(model, "magmaan_model")) {
    if (any(supplied) || !is.null(prototype)) {
      stop("magmaan_model(): `model` is already constructed; construct it again from its syntax to change it",
           call. = FALSE)
    }
    return(model)
  }
  is_spec <- inherits(model, "magmaan_model_spec")
  if (!is_spec && (!is.character(model) || length(model) != 1L || is.na(model))) {
    stop("magmaan_model(): `model` must be lavaan model syntax in one string or a model specification",
         call. = FALSE)
  }
  if (is_spec && (!is.null(model$eqs_source) || is.null(model$syntax))) {
    stop("magmaan_model(): model specifications must carry lavaan syntax; use magmaanlab::fit_model() for EQS or partable-only specifications",
         call. = FALSE)
  }
  if (!is.null(prototype) && !is.data.frame(prototype)) {
    stop("magmaan_model(): `prototype` must be a data frame", call. = FALSE)
  }
  identification <- .check_choice(identification, "identification",
                                  c("marker", "std.lv"), planned = "sphere", caller = caller)
  parameterization <- .check_choice(parameterization, "parameterization",
                                    c("delta", "theta"), caller = caller)
  ordered <- .check_ordered(ordered, caller)
  group <- .check_name(group, "group", caller)
  group.equal <- .check_names(group.equal, "group.equal", caller)
  group.partial <- .check_names(group.partial, "group.partial", caller)

  syntax <- if (is_spec) model$syntax else model
  options <- list(std_lv = identical(identification, "std.lv"), ordered = ordered,
                  parameterization = parameterization, group_equal = group.equal,
                  group_partial = group.partial)
  labels <- NULL
  if (is_spec) {
    inherited <- .spec_choices(model)
    for (arg in names(supplied)[supplied]) {
      same <- if (arg == "ordered") setequal(get(arg), inherited[[arg]]) else
        identical(get(arg), inherited[[arg]])
      if (!same) {
        stop(sprintf("magmaan_model(): `%s` conflicts with the model specification; set it in magmaanlab::model_spec() or pass the syntax", arg),
             call. = FALSE)
      }
    }
    if (isTRUE(model$options$fixed_x) && any(model$partable$exo == 1L)) {
      stop(paste0("magmaan_model(): this specification fixes its exogenous covariates (fixed_x = TRUE). ",
                  "magmaan() fits the joint model with random covariates: rebuild the specification ",
                  "with fixed_x = FALSE, or pass its syntax. Fixed-x fits remain available in ",
                  "magmaanlab::fit_model()."), call. = FALSE)
    }
    ordered <- inherited$ordered
    group <- inherited$group
    identification <- inherited$identification
    parameterization <- inherited$parameterization
    group.equal <- inherited$group.equal
    group.partial <- inherited$group.partial
    options <- model$options
    if (length(model$group_labels)) labels <- as.character(model$group_labels)
  }
  options$meanstructure <- TRUE
  options$fixed_x <- FALSE

  if (!is.null(group)) {
    labels <- .group_schema(group, labels, prototype)
  }
  if (length(ordered) && is.null(prototype)) {
    stop("magmaan_model(): ordered variables need `prototype` to declare their categories",
         call. = FALSE)
  }
  spec <- do.call(magmaanlab::model_spec,
                  c(list(syntax = syntax), options,
                    list(group = group, group_labels = labels)))
  observed <- magmaanlab::magmaan_core$model_matrix_rep(spec$partable)$ov_names
  observed <- unique(unlist(observed, use.names = FALSE))
  outside <- setdiff(ordered, observed)
  if (length(outside)) {
    stop("magmaan_model(): ordered variables must occur in the model: ",
         paste(outside, collapse = ", "), call. = FALSE)
  }
  categories <- if (is.null(prototype)) list() else
    .variable_schema(prototype, observed, ordered, caller)

  structure(
    list(spec = spec, observed = observed, ordered = ordered, categories = categories,
         group = group, groups = labels, group.equal = group.equal,
         group.partial = group.partial, identification = identification,
         parameterization = parameterization,
         prepared_cache = new.env(parent = emptyenv())),
    class = "magmaan_model")
}

# Only the portable specification/schema survives serialization meaningfully.
# A reloaded pointer is NULL even when the reader has the same process id.
.prepared_model <- function(model) {
  cache <- model$prepared_cache
  if (is.null(cache)) cache <- new.env(parent = emptyenv())
  handle <- cache$handle
  if (is.null(handle) || !identical(cache$pid, Sys.getpid()) ||
      identical(format(handle$native), "<pointer: (nil)>")) {
    prototype <- as.data.frame(setNames(lapply(model$ordered, function(v)
      factor(character(), levels = model$categories[[v]], ordered = TRUE)), model$ordered))
    handle <- magmaanlab::prepare_model(model$spec, prototype = prototype)
    cache$handle <- handle
    cache$pid <- Sys.getpid()
  }
  handle
}

#' @export
print.magmaan_model <- function(x, ...) {
  cat("magmaan model\n")
  cat("  observed:       ", paste(x$observed, collapse = ", "), "\n", sep = "")
  if (length(x$ordered)) cat("  ordered:        ", paste(x$ordered, collapse = ", "), "\n", sep = "")
  if (!is.null(x$group)) {
    cat("  groups:         ", x$group, ": ", paste(x$groups, collapse = ", "), "\n", sep = "")
  }
  cat("  identification: ", x$identification, "\n", sep = "")
  invisible(x)
}

#' Fit a structural equation model with magmaan's default inference
#'
#' Estimates the model and, unless `inference = FALSE`, computes inference
#' under magmaan's single documented policy. The arguments are the choices
#' that define the estimate; structural choices belong to [magmaan_model()]
#' and optimization details to `options`.
#'
#' @param model A model from [magmaan_model()], or lavaan model syntax in one
#'   string (or a `magmaanlab::model_spec()` specification) as a shortcut that
#'   constructs the model with `data` as its prototype on every call. A syntax
#'   string has the default structural choices: one group, continuous
#'   variables and marker identification. Data with ordered factors need
#'   `magmaan_model(ordered = )`.
#' @param data A data frame of raw observations.
#' @param estimator `"ML"`, `"FIML"`, `"ML2S"`, `"GLS"` or `"ULS"` for
#'   continuous variables; `"DWLS"`, `"WLS"` or `"ULS"` for ordered variables.
#'   Estimators other than FIML and ML2S delete incomplete rows listwise.
#' @param covariance The covariance policy: `"unrestricted"`, `"psd"` (every
#'   model-implied covariance matrix positive semidefinite), or `"barrier"` or
#'   [barrier()] (a penalized estimate in the interior of the PSD domain;
#'   experimental).
#' @param inference Compute inference now. With `FALSE`, call [infer()] later.
#' @param options Optimization details, a named list with any of `start`,
#'   `optimizer`, `convergence` and `preset`; omitted entries use magmaan's
#'   defaults.
#'   * `start`: `"default"`; `"fabin3"`, the FABIN3 start (Hägglund, 1982),
#'     which ML, ML2S and GLS used before 2026-09-26 and every other fit,
#'     including every PSD fit, uses by default (not for ordered variables);
#'     `"lavaan-0.7.2"`, lavaan 0.7.2's starts; a previous [magmaan()] fit; or
#'     a data frame with columns `lhs`, `op`, `rhs` and `est` (and `group` for
#'     several groups). A fit or table sets the start of every free parameter
#'     it matches, as lavaan's `start = fit`; the others keep the default
#'     start.
#'   * `optimizer`: `"default"`, `"port"` (PORT, as R's `nlminb()`) or
#'     `"lavaan-0.7.2"` (lavaan 0.7.2's PORT search and retries).
#'   * `convergence`: `"default"` or `"newton"` (magmaan's convergence check)
#'     or `"lavaan-0.7.2"` (lavaan 0.7.2's acceptance rule).
#'   * `preset`: `"lavaan-0.7.2"` selects lavaan 0.7.2's start, search and
#'     acceptance rule; explicit entries override it.
#'
#'   `optimizer`, `convergence`, `preset` and `start = "lavaan-0.7.2"` are
#'   available for continuous ML or FIML, or all-ordinal DWLS, with unrestricted covariance and
#'   supported linear equality constraints. Inspect `as_lab_fit(fit)$fitting`
#'   for the resolved settings and attempts. Fitting computes magmaan's inference policy;
#'   reporting methods can select an explicit lavaan inference convention.
#'   The policy covers complete-data ML, observed-data FIML and all-ordinal DWLS. DWLS
#'   reports one global test, the fit-function statistic (equal to its score
#'   statistic), and marks the likelihood-ratio test `"inapplicable"`; its
#'   covariance includes the influence of the estimated weight.
#' @param ... Arguments removed in magmaan 0.2.0; each raises an error naming
#'   its replacement.
#' @return An object of class `magmaan`.
#' @section Covariance policies:
#' `"psd"` constrains every model-implied covariance matrix to be positive
#' semidefinite. A PSD estimate on the boundary gets inference for an interior
#' population, and says so.
#'
#' `barrier(lambda)` maximizes the log-likelihood plus `lambda` times the sum,
#' over groups, of the log determinant of the model-implied correlation matrix
#' of all latent and observed variables: the log density of an LKJ(1 + lambda)
#' distribution. The estimate lies in the interior of the PSD domain and
#' moves by O(lambda / N) at interior points. `"barrier"` is `barrier(0.25)`;
#' `barrier(0)` is the unrestricted fit. Barrier fitting is experimental: the
#' first barrier fit in a session shows a message, and the inference of every
#' barrier fit with positive `lambda` is unavailable with reason `"penalized"`.
#' @section Simulation extraction:
#' `coef(fit)` is a named vector in free-parameter order, aligned with both
#' dimensions of `vcov(fit)`. Equality constraints can produce repeated labels;
#' use numeric indices to distinguish those entries. `coef(summary(fit))` is a
#' data frame with `lhs`, `op`, `rhs`, `label`, logical `free`, `est`, `se`,
#' `z`, `pvalue`, `ci.lower`, and `ci.upper`; multi-group fits also have `group`.
#' Defined estimates are available independently of inference. Every model has
#' intercepts (`op` is `"~1"`), so extract parameters by name.
#'
#' `fit$rows` has `group`, `rows`, `used`, and `deleted` per group.
#' `as_lab_fit(fit)$converged` is `TRUE`, `FALSE`, or `NA` for an unchecked fit;
#' use `isTRUE()` to count successful fits. `fit$inference` is `NULL` before
#' inference. Otherwise `fit$inference$status` has `component`, `available`,
#' `reason`, and `detail` for covariance, global score, and global LR.
#' Reasons include `available`, `not_converged`, `saturated`,
#' `unsupported_model`, `penalized` and `numeric_failure`.
#' `fit$inference$convergence` has `rule` (the acceptance rule that set
#' `converged`, `"newton"` by default), `converged`, `magmaan` (magmaan's own
#' check: `"passed"`, `"failed"` or `"unchecked"`), and `disagree`. Under a
#' compatibility rule such as `preset = "lavaan-0.7.2"`, `disagree` is `TRUE`
#' when the rule and magmaan's check reach different verdicts; inference
#' follows the rule.
#'
#' `summary(fit)$tests` is `NULL` when no global test is available, otherwise a
#' data frame with `test`, `statistic`, `df`, `p.sb`, `p.peba4`, and `sb.scale`.
#' The score test is primary and comes first; the likelihood-ratio row follows
#' and tends to over-reject when N is small relative to its df.
#' Store both package versions with simulation results. Saved fits retain their
#' estimates and inference; reusing them with another package version is not a
#' compatibility promise.
#' @examples
#' if (requireNamespace("lavaan", quietly = TRUE)) {
#'   fit <- magmaan("visual =~ x1 + x2 + x3",
#'                  lavaan::HolzingerSwineford1939)
#'   coef(fit)
#'   coef(summary(fit))
#' }
#' @export
magmaan <- function(model, data,
                    estimator = "ML",
                    covariance = "unrestricted",
                    inference = TRUE,
                    options = NULL, ...) {
  dots <- match.call(expand.dots = FALSE)$...
  if (length(dots)) .check_removed_arguments(names(dots))
  if (!is.data.frame(data)) {
    stop("magmaan(): `data` must be a data frame of raw observations; ",
         "summary-statistic input is available in magmaanlab", call. = FALSE)
  }
  if (!inherits(model, "magmaan_model")) {
    if (!inherits(model, "magmaan_model_spec") &&
        (!is.character(model) || length(model) != 1L || is.na(model))) {
      stop("magmaan(): `model` must come from magmaan_model(), or be lavaan model syntax in one string or a model specification",
           call. = FALSE)
    }
    model <- magmaan_model(model, prototype = data)
  }
  estimator <- .check_estimator(estimator)
  covariance <- .check_covariance(covariance)
  .check_flag(inference, "inference")
  options <- .check_options(options)
  .check_estimator_data(estimator, model$ordered)
  data <- .fit_data(model, data)

  # barrier(0) maximizes the unpenalized likelihood: it is the unrestricted fit.
  effective <- covariance$policy
  if (identical(effective, "barrier") && covariance$lambda == 0) effective <- "unrestricted"
  engine <- options[intersect(c("preset", "optimizer", "convergence"), names(options))]
  start <- .start_inputs(options$start, estimator, effective, model$ordered,
                         engine = length(engine) > 0L)
  engine$starts <- start$starts
  if (length(engine) && (!(estimator %in% c("ML", "FIML") && !length(model$ordered) ||
                         estimator == "DWLS" && length(model$ordered)) ||
                         !identical(effective, "unrestricted"))) {
    stop("magmaan(): options$optimizer, options$convergence, options$preset and ",
         "options$start = \"lavaan-0.7.2\" are available for continuous ML or FIML, or all-ordinal DWLS ",
         "with unrestricted covariance so far", call. = FALSE)
  }
  if (identical(effective, "barrier")) .barrier_message()

  args <- list(model = model$spec, data = data, estimator = estimator)
  if (!identical(effective, "unrestricted")) args$covariance <- effective
  if (identical(effective, "barrier")) {
    args$barrier <- list(target = "joint", weight = covariance$lambda)
  }
  if (!is.null(start$control)) args$control <- start$control
  if (length(engine)) args$options <- engine
  # Prepared ML2S and ordinal fitting options remain explicit lab gaps.
  fallback <- estimator == "ML2S" || (length(model$ordered) && length(engine))
  if (fallback) {
    lab <- do.call(magmaanlab::fit_model, args)
  } else {
    handle <- .prepared_model(model)
    prepared_data <- magmaanlab::prepare_data(handle, data,
      kind = if (estimator == "FIML") "raw" else handle$kind,
      missing = "listwise")
    args$model <- handle
    args$data <- prepared_data
    lab <- do.call(magmaanlab::estimate, args)
  }

  fit <- structure(
    list(lab = lab, call = match.call(), model = model, estimator = estimator,
         covariance = covariance,
         experimental = identical(effective, "barrier"),
         rows = .row_accounting(data, model$group, lab),
         inference = NULL),
    class = "magmaan")
  if (inference) fit <- infer(fit)
  fit
}

#' A barrier covariance policy
#'
#' The `covariance` argument of [magmaan()] for a penalized estimate in the
#' interior of the PSD domain; see its Covariance policies section.
#'
#' @param lambda The penalty weight, one finite non-negative number: the
#'   LKJ(1 + lambda) density's exponent on the log-likelihood scale. Zero is
#'   the unrestricted fit. The default, 0.25, is provisional.
#' @return An object of class `magmaan_covariance`.
#' @examples
#' if (requireNamespace("lavaan", quietly = TRUE)) {
#'   d <- lavaan::HolzingerSwineford1939
#'   sweep <- lapply(c(0, 0.1, 1), function(lambda)
#'     coef(magmaan("visual =~ x1 + x2 + x3", d, covariance = barrier(lambda),
#'                  inference = FALSE)))
#' }
#' @export
barrier <- function(lambda = 0.25) {
  if (!is.numeric(lambda) || length(lambda) != 1L || is.na(lambda) ||
      !is.finite(lambda) || lambda < 0) {
    stop("barrier(): `lambda` must be one finite non-negative number", call. = FALSE)
  }
  structure(list(policy = "barrier", lambda = as.numeric(lambda)),
            class = "magmaan_covariance")
}

#' The magmaanlab fit behind a magmaan fit
#'
#' Use it to run any alternative estimator option or inference method from
#' magmaanlab without refitting.
#'
#' @param fit A [magmaan()] fit.
#' @return The underlying `magmaan_fit` from magmaanlab.
#' @export
as_lab_fit <- function(fit) {
  if (!inherits(fit, "magmaan")) stop("as_lab_fit(): supply a magmaan() fit", call. = FALSE)
  fit$lab
}

# Raises the migration error for the first extra argument; `nm` are the names
# of the extra arguments, unevaluated.
.check_removed_arguments <- function(nm) {
  if (is.null(nm) || any(!nzchar(nm))) {
    stop("magmaan(): too many unnamed arguments; magmaan() takes model, data, ",
         "estimator, covariance, inference and options", call. = FALSE)
  }
  arg <- nm[[1L]]
  if (arg %in% .model_arguments) {
    stop(sprintf(paste0(
      "magmaan(): `%s` is an argument of magmaan_model() since magmaan 0.2.0: ",
      "m <- magmaan_model(model, prototype = data, %s = ...); magmaan(m, data)"),
      arg, arg), call. = FALSE)
  }
  if (arg %in% names(.removed_arguments)) {
    stop("magmaan(): ", .removed_arguments[[arg]], " (since magmaan 0.2.0)", call. = FALSE)
  }
  stop("magmaan(): unused argument `", arg, "`", call. = FALSE)
}

.check_estimator <- function(estimator) {
  if (!is.character(estimator) || length(estimator) != 1L || is.na(estimator)) {
    stop("magmaan(): `estimator` must be one string", call. = FALSE)
  }
  est <- toupper(estimator)
  if (est %in% names(.bundled_estimators)) {
    reporting <- if (est %in% c("MLM", "MLR", "WLSMV", "ULSMV"))
      sprintf("Select summary(fit, lavaan_compat = \"%s\") for that inference bundle.", est) else
      "Other corrections are available in magmaanlab."
    stop(sprintf(paste0(
      "magmaan(): estimator = \"%s\" bundles an estimator with a correction. ",
      "Use estimator = \"%s\"; magmaan() computes inference automatically. ",
      "%s"), estimator, .bundled_estimators[[est]], reporting), call. = FALSE)
  }
  known <- union(.continuous_estimators, .ordered_estimators)
  if (!est %in% known) {
    stop(sprintf("magmaan(): unknown estimator \"%s\"; use one of %s",
                 estimator, paste(known, collapse = ", ")), call. = FALSE)
  }
  est
}

.check_estimator_data <- function(estimator, ordered) {
  has_ordered <- length(ordered) > 0L
  if (has_ordered && !estimator %in% .ordered_estimators) {
    stop(sprintf(paste0(
      "magmaan(): estimator = \"%s\" treats every variable as continuous. ",
      "Construct the model without `ordered` to fit it to these items as ",
      "continuous, or use DWLS, WLS or ULS for ordered variables."), estimator),
      call. = FALSE)
  }
  if (!has_ordered && identical(estimator, "DWLS")) {
    stop("magmaan(): DWLS is for ordered variables; declare them with magmaan_model(ordered = )",
         call. = FALSE)
  }
  if (!has_ordered && identical(estimator, "WLS")) {
    stop("magmaan(): continuous WLS (ADF) is not available in magmaan() yet; use ",
         "magmaanlab::estimate() with prepare_weight(data, \"WLS\")", call. = FALSE)
  }
  invisible(NULL)
}

.check_covariance <- function(covariance) {
  if (inherits(covariance, "magmaan_covariance")) return(covariance)
  if (is.character(covariance) && length(covariance) == 1L && !is.na(covariance)) {
    if (identical(covariance, "barrier")) return(barrier())
    if (covariance %in% c("unrestricted", "psd")) {
      return(structure(list(policy = covariance), class = "magmaan_covariance"))
    }
  }
  stop("magmaan(): `covariance` must be \"unrestricted\", \"psd\", \"barrier\" or barrier(lambda)",
       call. = FALSE)
}

.barrier_message <- function() {
  if (isTRUE(.session$barrier_shown)) return(invisible(NULL))
  .session$barrier_shown <- TRUE
  message("magmaan(): barrier fitting is experimental. Estimates are computed, but ",
          "their inference is unavailable until it is validated. This message is ",
          "shown once per session.")
}

.option_choices <- list(
  optimizer = c("default", "port", "lavaan-0.7.2"),
  convergence = c("default", "newton", "lavaan-0.7.2"),
  preset = "lavaan-0.7.2"
)

.check_options <- function(options) {
  if (is.null(options)) return(list())
  nm <- names(options)
  if (!is.list(options) || is.data.frame(options) ||
      (length(options) && (is.null(nm) || anyNA(nm) || any(!nzchar(nm)) || anyDuplicated(nm)))) {
    stop("magmaan(): `options` must be a list with unique names", call. = FALSE)
  }
  if ("starts" %in% nm) {
    stop("magmaan(): options$starts is now options$start (since magmaan 0.2.0)", call. = FALSE)
  }
  unknown <- setdiff(nm, c("start", names(.option_choices)))
  if (length(unknown)) {
    stop("magmaan(): unknown option ", paste0("`", unknown, "`", collapse = ", "),
         "; `options` takes start, optimizer, convergence and preset", call. = FALSE)
  }
  for (name in intersect(nm, names(.option_choices))) {
    .check_choice(options[[name]], paste0("options$", name), .option_choices[[name]])
  }
  if (!is.null(options$start)) options$start <- .check_start(options$start)
  options
}

# A start name, or a table of start values (from a fit or given).
.check_start <- function(start) {
  if (inherits(start, "magmaan")) start <- stats::coef(summary(start))
  if (is.data.frame(start)) {
    if (!all(c("lhs", "op", "rhs", "est") %in% names(start))) {
      stop("magmaan(): a start table needs columns lhs, op, rhs and est", call. = FALSE)
    }
    return(start)
  }
  .check_choice(start, "options$start", c("default", "fabin3", "lavaan-0.7.2"))
}

# The magmaanlab inputs for `options$start`: a start table as a control, or a
# start convention. "fabin3" is the FABIN3 start each fitter used before the
# layered start became the ML and GLS default. ML (and ML2S, whose second stage
# is an ML fit) transports FABIN3 from unit latent variances, as PSD ML does;
# GLS used native FABIN3. Every other continuous fit already starts from it.
# With `engine` options the start is one of their conventions, so that an
# explicit start overrides a preset's.
.start_inputs <- function(start, estimator, covariance, ordered, engine) {
  if (is.null(start)) return(list())
  if (is.data.frame(start)) return(list(control = list(start = start)))
  if (identical(start, "lavaan-0.7.2")) return(list(starts = start))
  if (identical(start, "default")) return(if (engine) list(starts = "default") else list())
  if (length(ordered)) {
    stop("magmaan(): options$start = \"fabin3\" is not available for ordered variables, ",
         "whose fits have their own start", call. = FALSE)
  }
  if (!identical(covariance, "unrestricted")) return(list())
  method <- switch(estimator, ML = , ML2S = "scaled-fabin", GLS = , FIML = "fabin3", NULL)
  if (is.null(method)) return(list())
  if (engine) list(starts = method) else list(control = list(start = method))
}

.check_ordered <- function(ordered, caller = "magmaan()") {
  if (is.null(ordered) || identical(ordered, FALSE) || !length(ordered)) return(NULL)
  if (isTRUE(ordered)) {
    stop(caller, ": name the ordered variables, e.g. ordered = c(\"y1\", \"y2\")",
         call. = FALSE)
  }
  if (!is.character(ordered) || anyNA(ordered) || !all(nzchar(ordered))) {
    stop(caller, ": `ordered` must be a character vector of variable names", call. = FALSE)
  }
  unique(ordered)
}

.check_name <- function(x, arg, caller) {
  if (is.null(x)) return(NULL)
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(sprintf("%s: `%s` must be one column name", caller, arg), call. = FALSE)
  }
  x
}

.check_names <- function(x, arg, caller) {
  if (is.null(x)) return(NULL)
  if (!is.character(x) || anyNA(x) || !all(nzchar(x))) {
    stop(sprintf("%s: `%s` must be a character vector", caller, arg), call. = FALSE)
  }
  x
}

# The structural choices a lab specification carries, under ordinary names.
.spec_choices <- function(spec) {
  group <- spec$group_var
  if (is.null(group) || identical(group, "")) group <- NULL
  list(ordered = .check_ordered(spec$ordered, "magmaan_model()"), group = group,
       group.equal = spec$group_equal, group.partial = spec$group_partial,
       identification = if (isTRUE(spec$options$std_lv)) "std.lv" else "marker",
       parameterization = spec$parameterization %||% "delta")
}

# Group identities and order: declared by a specification, by factor levels,
# or by first appearance in the prototype.
.group_schema <- function(group, labels, prototype) {
  if (is.null(prototype)) {
    if (length(labels)) return(labels)
    stop("magmaan_model(): a grouped model needs `prototype` to declare its groups and their order",
         call. = FALSE)
  }
  if (!group %in% names(prototype)) {
    stop("magmaan_model(): `group` column \"", group, "\" is not in `prototype`", call. = FALSE)
  }
  g <- prototype[[group]]
  if (anyNA(g)) stop("magmaan_model(): the grouping column contains missing values", call. = FALSE)
  if (length(labels)) {
    outside <- setdiff(unique(as.character(g)), labels)
    if (length(outside)) {
      stop("magmaan_model(): `prototype` has groups the specification does not declare: ",
           paste(outside, collapse = ", "), call. = FALSE)
    }
    return(labels)
  }
  labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
  if (!length(labels)) {
    stop("magmaan_model(): `prototype` declares no groups; use a factor with levels",
         call. = FALSE)
  }
  labels
}

# The prototype's model variables: continuous ones numeric, ordered ones with
# at least two declared categories.
.variable_schema <- function(prototype, observed, ordered, caller) {
  absent <- setdiff(observed, names(prototype))
  if (length(absent)) {
    stop(caller, ": `prototype` lacks model variables: ", paste(absent, collapse = ", "),
         call. = FALSE)
  }
  .check_continuous(prototype, setdiff(observed, ordered), caller, "prototype")
  lapply(stats::setNames(ordered, ordered), function(v) {
    x <- prototype[[v]]
    levels <- if (is.factor(x)) levels(x) else as.character(sort(unique(x[!is.na(x)])))
    if (length(levels) < 2L) {
      stop(caller, ": ordered variable ", v, " needs at least two declared categories",
           call. = FALSE)
    }
    levels
  })
}

.check_continuous <- function(data, variables, caller, what) {
  for (v in variables) {
    x <- data[[v]]
    if (is.ordered(x)) {
      stop(sprintf(paste0(
        "%s: %s is an ordered factor in `%s`, but the model treats it as continuous. ",
        "Declare it with magmaan_model(model, prototype = data, ordered = ...) or ",
        "convert it with as.numeric()"), caller, v, what), call. = FALSE)
    }
    if (!is.numeric(x)) {
      stop(sprintf("%s: continuous variable %s must be numeric in `%s`", caller, v, what),
           call. = FALSE)
    }
  }
  invisible(NULL)
}

.schema_error <- function(reason, message) {
  structure(class = c("magmaan_schema_error", "error", "condition"),
            list(message = paste0("magmaan(): ", message), call = NULL, reason = reason))
}

# The data in the model's schema: groups in the model's order, and ordered
# variables as factors with the declared categories. Rows keep their order
# within each group, so every lab route, including refits that group by
# appearance, sees the model's group order.
.fit_data <- function(model, data) {
  absent <- setdiff(c(model$observed, model$group), names(data))
  if (length(absent)) {
    stop("magmaan(): `data` lacks model variables: ", paste(absent, collapse = ", "),
         call. = FALSE)
  }
  .check_continuous(data, setdiff(model$observed, model$ordered), "magmaan()", "data")
  group <- NULL
  if (!is.null(model$group)) {
    g <- data[[model$group]]
    if (anyNA(g)) stop("magmaan(): the grouping column contains missing values", call. = FALSE)
    g <- as.character(g)
    outside <- setdiff(unique(g), model$groups)
    if (length(outside)) {
      stop(.schema_error("undeclared_group", paste0(
        "`data` has groups the model does not declare: ", paste(outside, collapse = ", "),
        "; construct the model again to add groups")))
    }
    empty <- setdiff(model$groups, g)
    if (length(empty)) {
      stop(.schema_error("empty_group", paste0(
        "declared groups without rows: ", paste(empty, collapse = ", "))))
    }
    if (!identical(unique(g), model$groups)) {
      data <- data[order(match(g, model$groups)), , drop = FALSE]
      g <- as.character(data[[model$group]])
    }
    group <- g
  }
  for (v in model$ordered) {
    lev <- model$categories[[v]]
    x <- data[[v]]
    if (is.factor(x) && !identical(levels(x), lev)) {
      stop(.schema_error("changed_levels", paste0(
        "the levels of ", v, " differ from the model's categories (",
        paste(lev, collapse = ", "), "); construct the model again to change them")))
    }
    values <- as.character(x)
    outside <- setdiff(unique(values[!is.na(values)]), lev)
    if (length(outside)) {
      stop(.schema_error("undeclared_category", paste0(
        v, " has categories the model does not declare: ", paste(outside, collapse = ", "),
        "; construct the model again to add categories")))
    }
    data[[v]] <- factor(values, levels = lev, ordered = TRUE)
  }
  if (length(model$ordered)) .check_categories(model, data, group)
  data
}

# Ordinal estimators delete incomplete rows listwise; every declared category
# must keep an observation in every group.
.check_categories <- function(model, data, group) {
  complete <- stats::complete.cases(data[model$observed])
  blocks <- if (is.null(group)) list(complete) else
    lapply(model$groups, function(label) complete & group == label)
  for (b in seq_along(blocks)) {
    for (v in model$ordered) {
      counts <- table(data[[v]][blocks[[b]]])
      empty <- names(counts)[counts == 0L]
      if (length(empty)) {
        where <- if (is.null(group)) "" else paste0(" in group ", model$groups[[b]])
        stop(.schema_error("empty_category", paste0(
          "category ", paste(empty, collapse = ", "), " of ", v,
          " has no complete observations", where)))
      }
    }
  }
  invisible(NULL)
}

.check_flag <- function(x, arg) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("magmaan(): `%s` must be TRUE or FALSE", arg), call. = FALSE)
  }
  invisible(x)
}

.check_choice <- function(x, arg, choices, planned = character(), caller = "magmaan()") {
  if (!is.character(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("%s: `%s` must be one string", caller, arg), call. = FALSE)
  }
  if (x %in% planned) {
    stop(sprintf("%s: %s = \"%s\" is planned but not available yet", caller, arg, x),
         call. = FALSE)
  }
  if (!x %in% choices) {
    stop(sprintf("%s: `%s` must be one of %s", caller, arg,
                 paste0("\"", choices, "\"", collapse = ", ")), call. = FALSE)
  }
  x
}

# Rows supplied, used and not used, per group. Estimators that are not designed
# for missing data delete listwise; FIML and ML2S use every row with data.
.row_accounting <- function(data, group, lab) {
  used <- as.integer(lab$nobs)
  labels <- as.character(lab$group_labels)
  if (is.null(group) || !length(labels)) {
    total <- nrow(data)
    labels <- "all"
  } else {
    total <- as.integer(table(factor(as.character(data[[group]]), levels = labels)))
  }
  if (length(used) != length(total)) used <- rep(NA_integer_, length(total))
  data.frame(group = labels, rows = total, used = used, deleted = total - used,
             stringsAsFactors = FALSE)
}
