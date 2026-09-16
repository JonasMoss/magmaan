# Immutable, process-local handles. R metadata describes the schema; all
# statistical computations and estimation use native objects.
.prepared_object <- function(..., class) {
  out <- list2env(list(...), parent = emptyenv())
  class(out) <- class
  lockEnvironment(out, bindings = TRUE)
  out
}

#' Prepare a reusable native SEM model
#'
#' The prototype declares variable/category/group schema only. It is never
#' reused as data or starting values. Prepare again after structural changes.
prepare_model <- function(model, ..., prototype = NULL) {
  spec <- if (is.character(model)) model_spec(model, ...) else {
    if (length(list(...))) stop("prepare_model(): options belong in model_spec()")
    as_magmaan_model_spec(model)
  }
  ov <- model_matrix_rep(spec$partable)$ov_names
  if (!is.list(ov)) ov <- list(ov)
  if (length(ov) != max(1L, length(spec$group_labels)))
    stop("prepare_model(): two-level models still use fit_twolevel(); staged support is pending")
  ordered <- spec$ordered
  if (length(setdiff(ordered, unique(unlist(ov)))))
    stop("prepare_model(): ordered variables must occur in the model")
  kind <- if (!length(ordered)) "moments" else if (
    all(vapply(ov, function(x) setequal(x, ordered), logical(1)))) "ordinal" else "mixed"
  labels <- spec$group_labels
  if (nzchar(spec$group_var) && !length(labels))
    stop("prepare_model(): specify group_labels explicitly in model_spec()")
  if (length(labels) && length(labels) != length(ov)) stop("prepare_model(): group schema mismatch")
  categories <- NULL
  masks <- if (kind == "mixed") lapply(ov, function(x) as.integer(x %in% ordered)) else list()
  schema <- NULL
  if (length(ordered)) {
    if (is.null(prototype)) stop("prepare_model(): ordinal models require prototype data for category schema")
    if (is.matrix(prototype)) prototype <- as.data.frame(prototype)
    if (!is.data.frame(prototype)) stop("prepare_model(): prototype must be a data.frame or named matrix")
    if (length(setdiff(ordered, names(prototype)))) stop("prepare_model(): prototype lacks ordered variables")
    categories <- lapply(prototype[ordered], function(x) {
      levels <- if (is.factor(x)) levels(x) else sort(unique(x[!is.na(x)]))
      if (length(levels) < 2L) stop("prepare_model(): each ordinal variable needs at least two categories")
      as.character(levels)
    })
    # Augmentation uses schema only, never empirical prototype thresholds.
    schema <- list(ov_names = ov, ordered = ordered,
                   n_levels = lapply(ov, function(x) vapply(x, function(v)
                     if (v %in% ordered) length(categories[[v]]) else 0L, integer(1))))
    schema$ordered_mask <- masks
    schema$thresholds <- lapply(schema$n_levels, function(x) rep(NA_real_, sum(pmax(x - 1L, 0L))))
    spec$partable <- if (kind == "ordinal") augment_ordinal_partable(spec, schema) else
      augment_mixed_ordinal_partable(spec, schema)
    # Native preparation consumes moment metadata, not empirical values.
    schema$R <- lapply(ov, function(x) diag(length(x)))
    schema$nobs <- rep(2L, length(ov))
    schema$threshold_ov <- lapply(schema$n_levels, function(x) rep(seq_along(x), pmax(x - 1L, 0L)))
    schema$threshold_level <- lapply(schema$n_levels, function(x) unlist(lapply(pmax(x - 1L, 0L), seq_len)))
    schema$thresholds <- lapply(schema$thresholds, function(x) rep(0, length(x)))
    schema$NACOV <- schema$W_dwls <- schema$W_wls <- rep(list(matrix(numeric(), 0, 0)), length(ov))
    if (kind == "mixed") {
      schema$ordered_mask <- masks
      schema$mean <- lapply(ov, function(x) rep(0, length(x)))
      schema$moments <- lapply(seq_along(ov), function(b) numeric(
        length(schema$thresholds[[b]]) + 2L * sum(masks[[b]] == 0L) + choose(length(ov[[b]]), 2)))
    }
  }
  native <- prepared_model_impl(spec$partable, kind, schema)
  .prepared_object(native = native, spec = spec, ov_names = ov, kind = kind,
                   categories = categories, masks = masks, class = "magmaan_prepared_model")
}

.prepared_blocks <- function(model, x, kind, missing) {
  if (is.matrix(x)) x <- as.data.frame(x)
  if (!is.data.frame(x)) stop("prepare_data(): use a data.frame or named numeric matrix")
  spec <- model$spec
  if (nzchar(spec$group_var)) {
    if (!spec$group_var %in% names(x)) stop("prepare_data(): missing grouping column")
    g <- as.character(x[[spec$group_var]])
    if (anyNA(g) || any(!g %in% spec$group_labels)) stop("prepare_data(): group schema changed")
    rows <- lapply(spec$group_labels, function(label) which(g == label))
  } else rows <- list(seq_len(nrow(x)))
  Map(function(ii, ov) {
    if (length(setdiff(ov, names(x)))) stop("prepare_data(): missing model variables")
    b <- x[ii, ov, drop = FALSE]
    if (kind != "raw" && anyNA(b)) {
      if (missing == "error") stop("prepare_data(): missing observations; choose listwise or kind = 'raw'")
      b <- b[stats::complete.cases(b), , drop = FALSE]
    }
    if (nrow(b) < 2L) stop("prepare_data(): fewer than two observations in a group")
    for (v in ov) {
      lev <- model$categories[[v]]
      if (!is.null(lev)) {
        if (is.factor(b[[v]]) && !identical(levels(b[[v]]), lev))
          stop("prepare_data(): category levels/order changed for ", v)
        values <- as.character(b[[v]])
        if (any(!values %in% lev) || any(!lev %in% values))
          stop("prepare_data(): changed or empty category for ", v)
        b[[v]] <- match(values, lev)
      } else if (!is.numeric(b[[v]])) stop("prepare_data(): nonnumeric continuous variable: ", v)
    }
    z <- as.matrix(b)
    storage.mode(z) <- "double"
    if (any(is.infinite(z))) stop("prepare_data(): infinite observations")
    z
  }, rows, model$ov_names)
}

#' Prepare one dataset, independently of estimation weights
prepare_data <- function(model, data, kind = NULL, missing = c("error", "listwise")) {
  stopifnot(inherits(model, "magmaan_prepared_model"))
  missing <- match.arg(missing)
  if (is.null(kind)) kind <- model$kind
  kind <- match.arg(kind, c("moments", "raw", "ordinal", "mixed"))
  if ((model$kind %in% c("ordinal", "mixed") && kind != model$kind) ||
      (model$kind == "moments" && !kind %in% c("moments", "raw")))
    stop("prepare_data(): data kind disagrees with model schema")
  summaries <- kind == "moments" && is.list(data) && !is.data.frame(data) && !is.null(data$S)
  X <- if (summaries) sample_stats_arg(data) else .prepared_blocks(model, data, kind, missing)
  native <- prepared_data_impl(model$native, X, kind, model$masks)
  .prepared_object(native = native, model = model, kind = kind, X = if (summaries) NULL else X,
                   missing = missing, class = "magmaan_prepared_data")
}

#' Prepare reusable estimation weights
#'
#' For categorical data full=TRUE also retains Gamma for existing post-fit
#' inference functions. full=FALSE prepares only the diagonal needed by DWLS.
#' Continuous weights use empirical Gamma, or an explicit W in model moment order.
prepare_weight <- function(data, method = c("DWLS", "WLS", "ULS"), W = NULL, full = TRUE) {
  stopifnot(inherits(data, "magmaan_prepared_data"))
  method <- match.arg(method)
  if (length(full) != 1L || is.na(full) || !is.logical(full)) stop("prepare_weight(): full must be TRUE or FALSE")
  if (data$kind == "moments" && method == "DWLS" && !is.null(W)) {
    blocks <- if (is.matrix(W)) list(W) else W
    if (!is.list(blocks) || any(vapply(blocks, function(x)
      !is.matrix(x) || nrow(x) != ncol(x) || any(x[row(x) != col(x)] != 0), logical(1))))
      stop("prepare_weight(): DWLS requires diagonal W")
  }
  prepared <- prepared_weight_impl(data$native, method, W, full)
  .prepared_object(native = prepared$native, data = data, method = method, full = full,
                   W = prepared$W, stats = prepared$stats,
                   class = "magmaan_prepared_weight")
}

#' Estimate using reusable model, data and optional weight handles
estimate <- function(model, data, estimator = NULL, weight = NULL,
                     optimizer = NULL, control = NULL, bounds = NULL) {
  stopifnot(inherits(model, "magmaan_prepared_model"),
            inherits(data, "magmaan_prepared_data"))
  if (!identical(model$categories, data$model$categories) ||
      !identical(model$spec$group_labels, data$model$spec$group_labels) ||
      !identical(model$kind, data$model$kind))
    stop("estimate(): model/data schemas differ")
  if (!is.null(weight)) {
    stopifnot(inherits(weight, "magmaan_prepared_weight"))
    if (!identical(weight$data, data)) stop("estimate(): weight belongs to another dataset")
    if (is.null(estimator)) estimator <- weight$method
    if (!identical(toupper(estimator), weight$method)) stop("estimate(): estimator and weight disagree")
  }
  if (is.null(estimator)) estimator <- switch(data$kind, raw = "FIML", ordinal = "DWLS", mixed = "DWLS", "ML")
  estimator <- toupper(estimator)
  allowed <- switch(data$kind, raw = "FIML", ordinal = c("ULS", "DWLS", "WLS"),
                    mixed = c("DWLS", "WLS"), c("ML", "ULS", "GLS", "WLS", "DWLS"))
  if (length(estimator) != 1L || !estimator %in% allowed) stop("estimate(): unsupported estimator for this data kind")
  if (is.null(weight) && estimator %in% c("DWLS", "WLS")) weight <- prepare_weight(data, estimator)
  if (!is.null(bounds) && (!is.list(bounds) || is.character(bounds)))
    stop("estimate(): supply explicit bounds (e.g. bounds_standard())")
  fit <- prepared_estimate_impl(model$native, data$native, if (is.null(weight)) NULL else weight$native,
                                estimator, optimizer, control, bounds)
  if (data$kind == "moments" && !is.null(data$X)) {
    fit$raw_data <- structure(list(X = data$X, ov_names = model$ov_names,
                                  group_var = model$spec$group_var,
                                  group_labels = model$spec$group_labels,
                                  nobs = vapply(data$X, nrow, integer(1))),
                             class = c("magmaan_complete_data", "list"))
  }
  finalize_magmaan_fit(fit, model$spec, estimator,
                      if (data$kind == "raw") "fiml" else data$missing, "none", "none")
}
