# Explicit, immutable inference snapshots; no p-values during preparation.
.score_object <- function(x, class, ...) {
  do.call(.prepared_object, c(x, list(...), list(class = class)))
}

prepare_inference <- function(fit, data = NULL) {
  if (inherits(fit, "magmaan_inference")) {
    if (!is.null(data)) stop("prepare_inference(): a snapshot already owns its data")
    return(fit)
  }
  if (!inherits(fit, "magmaan_fit")) stop("prepare_inference(): supply a fitted magmaan model")
  shared_data <- NULL
  if (inherits(data, "magmaan_inference_data")) {
    shared_data <- data$native
    data <- data$raw
  }
  estimator <- toupper(fit$estimator)
  if (!estimator %in% c("ML", "FIML", "ML2S"))
    stop("prepare_inference(): currently supports ML, FIML and fixed-NT ML2S")
  if (length(fit$diagnostics$active_bounds_lower) || length(fit$diagnostics$active_bounds_upper))
    stop("prepare_inference(): active-bound inference is unsupported")
  if (inherits(data, "magmaan_prepared_data")) {
    if (is.null(data$X)) stop("prepare_inference(): score meat requires raw observations")
    data <- list(X = data$X, ov_names = data$model$ov_names)
  }
  if (estimator %in% c("FIML", "ML2S")) {
    if (!is.null(data)) stop("prepare_inference(): FIML/ML2S uses the fit's raw data")
    raw <- fit$raw_data
  } else {
    data <- data %||% fit$raw_data
    if (is.null(data)) stop("prepare_inference(): supply the fitting data")
    raw <- raw_data_arg(fit, data)
  }
  .score_object(prepare_inference_impl(fit, raw, shared_data), "magmaan_inference")
}

scores <- function(object, data = NULL, space = c("parameter", "saturated")) {
  context <- prepare_inference(object, data)
  score_rows_impl(context$native, match.arg(space))
}

score_components <- function(object, data = NULL, H1 = NULL,
    sensitivity = c("expected", "observed", "observed-h1", "observed-shrink-light", "observed-shrink-sqrt"),
    metric = c("expected", "observed", "observed-h1")) {
  context <- prepare_inference(object, data)
  sensitivity <- match.arg(sensitivity); metric <- match.arg(metric)
  partable <- if (is.null(H1)) NULL else if (inherits(H1, "magmaan_prepared_model")) H1$spec$partable else
    if (inherits(H1, "magmaan_fit")) H1$partable else as_magmaan_model_spec(H1)$partable
  value <- score_components_impl(context$native, partable, sensitivity, metric)
  .score_object(value, "magmaan_score_components", context = context,
                sensitivity_kind = sensitivity, metric_kind = metric)
}

project_scores <- function(components, retain_rows = FALSE, center = FALSE) {
  stopifnot(inherits(components, "magmaan_score_components"))
  for (x in list(retain_rows, center))
    if (!is.logical(x) || length(x) != 1L || is.na(x)) stop("project_scores(): flags must be TRUE or FALSE")
  .score_object(project_scores_impl(components$native, retain_rows, center),
                "magmaan_projected_score", components = components, centered = center)
}

# Supplied-matrix route, also usable for a Wald contrast and its covariance.
score_quadratic <- function(score, metric, meat = NULL) {
  .score_object(score_quadratic_impl(as.numeric(score), as.matrix(metric), meat),
                "magmaan_projected_score")
}

score_spectrum <- function(projected) {
  if (inherits(projected, "magmaan_ntml_quadratic"))
    return(.score_object(ntml_reference_impl(projected$native, TRUE), "magmaan_quadratic_reference", source = projected))
  stopifnot(inherits(projected, "magmaan_projected_score"))
  .score_object(score_reference_impl(projected$native, TRUE), "magmaan_quadratic_reference", source = projected)
}

# A reusable statistic/reference law can also originate from LR or GOF code.
quadratic_reference <- function(statistic, df, eigenvalues = NULL) {
  if (length(statistic) != 1L || !is.finite(statistic) || statistic < 0 ||
      length(df) != 1L || !is.finite(df) || df < 1 || df != floor(df))
    stop("quadratic_reference(): supply a nonnegative statistic and positive integer df")
  if (!is.null(eigenvalues) && (length(eigenvalues) != df ||
      any(!is.finite(eigenvalues)) || any(eigenvalues < 0)))
    stop("quadratic_reference(): eigenvalues must be finite, nonnegative and include df entries")
  .score_object(list(statistic = statistic, df = as.integer(df), eigenvalues = eigenvalues),
                "magmaan_quadratic_reference")
}

calibrate_quadratic <- function(object, methods = "peba4") {
  methods <- tolower(methods)
  if (!length(methods) || any(!methods %in% c("std", "sb", "peba2", "peba4", "all")))
    stop("calibrate_quadratic(): use std, sb, peba2, peba4, or all (exact mixture)")
  if (inherits(object, "magmaan_ntml_quadratic"))
    object <- .score_object(ntml_reference_impl(object$native,
      any(methods %in% c("peba2", "peba4", "all"))), "magmaan_quadratic_reference")
  if (inherits(object, "magmaan_projected_score")) {
    if (any(methods %in% c("peba2", "peba4", "all"))) object <- score_spectrum(object) else if ("sb" %in% methods)
      object <- .score_object(score_reference_impl(object$native, FALSE), "magmaan_quadratic_reference")
  }
  if (!inherits(object, c("magmaan_quadratic_reference", "magmaan_projected_score")))
    stop("calibrate_quadratic(): supply projected scores or a quadratic reference")
  p <- vapply(methods, function(method) {
    if (method == "std") return(infer_chi2_pvalue(object$statistic, as.integer(object$df)))
    if (method == "sb" && !is.null(object$mean_scale))
      return(infer_chi2_pvalue(object$statistic / object$mean_scale, as.integer(object$df)))
    if (is.null(object$eigenvalues)) stop("calibrate_quadratic(): this reference has no spectrum")
    infer_fmg_test(object$statistic, object$df, object$eigenvalues,
      method = if (method %in% c("peba2", "peba4")) "peba" else method,
      param = switch(method, peba2 = 2, peba4 = 4, 0))$p_value
  }, numeric(1))
  data.frame(method = methods, statistic = object$statistic, df = object$df,
             p_value = unname(p), row.names = NULL)
}

score_sandwich <- function(projected) {
  stopifnot(inherits(projected, "magmaan_projected_score"))
  value <- score_sandwich_impl(projected$native)
  quadratic_reference(value$statistic, value$df)
}

resample_scores <- function(projected, n_flips = 999L, seed = 1,
    multiplier = c("rademacher", "mammen", "two-point", "gaussian", "centered-exponential"),
    two_point_skewness = 1) {
  stopifnot(inherits(projected, "magmaan_projected_score"))
  if (length(n_flips) != 1L || !is.finite(n_flips) || n_flips < 1 || n_flips != floor(n_flips) || n_flips > .Machine$integer.max)
    stop("resample_scores(): n_flips must be a positive integer")
  resample_scores_impl(projected$native, as.integer(n_flips), seed, match.arg(multiplier), two_point_skewness)
}

inference_information <- function(context, type = c("expected", "observed")) {
  stopifnot(inherits(context, "magmaan_inference"))
  out <- inference_information_impl(context$native, match.arg(type))
  attr(out, "inference_context") <- context
  out
}

.check_inference_matrix <- function(x, context) {
  origin <- attr(x, "inference_context")
  if (!is.null(origin) && !identical(origin, context)) stop("inference matrix belongs to another fit snapshot")
}

parameter_covariance <- function(context, information, meat = NULL) {
  stopifnot(inherits(context, "magmaan_inference"))
  .check_inference_matrix(information, context)
  .check_inference_matrix(meat, context)
  out <- parameter_covariance_impl(context$native, information, meat)
  attr(out, "inference_context") <- context
  out
}

wald_test <- function(object, R, vcov, q = NULL) {
  if (inherits(object, "magmaan_inference")) {
    .check_inference_matrix(vcov, object)
    theta <- object$theta
  } else if (inherits(object, "magmaan_fit")) {
    origin <- attr(vcov, "inference_context")
    if (!is.null(origin) && !identical(origin$original_fit, object)) stop("Wald covariance belongs to another fit snapshot")
    theta <- object$theta
  } else theta <- as.numeric(object)
  infer_wald_test_theta(theta, as.matrix(R), as.matrix(vcov), q)
}


.inference_fit <- function(object) {
  if (inherits(object, "magmaan_inference")) inference_snapshot_impl(object$native) else object
}

.inference_pair <- function(H1, H0, data) {
  prepared <- inherits(H1, "magmaan_inference") || inherits(H0, "magmaan_inference")
  if (prepared && !is.null(data)) stop("nested inference snapshots already own their data")
  a <- .inference_fit(H1); b <- .inference_fit(H0)
  if (prepared) {
    if (!identical(a$raw_data, b$raw_data)) stop("nested inference snapshots must use the same observations and ordering")
    if (identical(a$estimator, "ML")) {
      data <- a$raw_data
      if (is.list(data) && !is.null(data$X)) data <- data$X
    }
  }
  list(H1 = a, H0 = b, data = data)
}

# Supply existing scores/geometry when developing a different nuisance or
# hypothesis construction. No fitted model or test routine is required.
score_components_from_matrices <- function(score, rows, sensitivity,
    metric = sensitivity, nuisance = matrix(0, length(score), 0L),
    directions = diag(length(score)), influence_rows = FALSE) {
  if (!is.logical(influence_rows) || length(influence_rows) != 1L || is.na(influence_rows))
    stop("influence_rows must be TRUE or FALSE")
  score <- as.numeric(score)
  rows <- as.matrix(rows); sensitivity <- as.matrix(sensitivity); metric <- as.matrix(metric)
  nuisance <- as.matrix(nuisance); directions <- as.matrix(directions)
  ptr <- score_components_matrix_impl(score, rows, sensitivity, metric, nuisance, directions, influence_rows)
  .score_object(list(native = ptr, score = score, rows = rows, sensitivity = sensitivity,
                     metric = metric, nuisance = nuisance, directions = directions,
                     influence_rows = influence_rows), "magmaan_score_components")
}

# Persistent NTML consumers share the existing U-factor and moment reducers.
prepare_inference_data <- function(fit, data = NULL, storage = c("auto", "casewise", "tiled")) {
  if (!inherits(fit, "magmaan_fit") || toupper(fit$estimator) != "ML")
    stop("prepare_inference_data(): supply a continuous ML fit to define the data layout")
  if (inherits(data, "magmaan_prepared_data")) data <- list(X=data$X,ov_names=data$model$ov_names)
  data <- data %||% fit$raw_data
  if (is.null(data)) stop("prepare_inference_data(): supply fitting observations")
  raw <- raw_data_arg(fit, data)
  .score_object(list(native=prepare_ntml_data_impl(fit,raw,match.arg(storage)),raw=raw), "magmaan_inference_data")
}

prepare_hypothesis <- function(null, alternative) {
  stopifnot(inherits(null,"magmaan_inference"), inherits(alternative,"magmaan_inference"))
  .score_object(list(native=prepare_ntml_hypothesis_impl(null$native,alternative$native),
      null=null,alternative=alternative),"magmaan_inference_hypothesis")
}

# `geometry` applies to nested hypotheses: "expected" (lavaan's Satorra-2000
# and lavTestScore geometry) or "observed" (the ordinary policy's nested
# geometry: observed information at the null fit for the score projection and
# at the alternative for the LR spectrum).
inference_quadratic <- function(object, test = c("score", "lr"),
                                geometry = c("expected", "observed")) {
  test <- match.arg(test)
  geometry <- match.arg(geometry)
  hypothesis <- inherits(object,"magmaan_inference_hypothesis")
  if (!hypothesis && !inherits(object,"magmaan_inference"))
    stop("inference_quadratic(): supply a prepared fit or hypothesis")
  .score_object(ntml_quadratic_impl(object$native,hypothesis,test=="score",
                                    geometry=="observed"),
      "magmaan_ntml_quadratic",source=object,test=test)
}

# Casewise rows behind a quadratic from inference_quadratic(): one row per
# observation, groups in block order. For the score quadratic the statistic is
# sum(colSums(rows)^2); crossprod(rows) is the reduced matrix of its spectrum.
inference_rows <- function(quadratic) {
  stopifnot(inherits(quadratic, "magmaan_ntml_quadratic"))
  ntml_rows_impl(quadratic$native)
}

inference_covariance <- function(context, robust = FALSE) {
  stopifnot(inherits(context,"magmaan_inference"),is.logical(robust),length(robust)==1L,!is.na(robust))
  out <- ntml_covariance_impl(context$native,robust)
  attr(out,"inference_context") <- context
  out
}

inference_reuse <- function(context) {
  stopifnot(inherits(context,"magmaan_inference"))
  inference_reuse_impl(context$native)
}

# magmaan's default inference policy for one fit, as applied by the
# ordinary-user package: the observed-information sandwich covariance and the
# global score and likelihood-ratio tests, each with SB and PEBA4. All-ordinal
# DWLS uses the estimated-weight (IJ) sandwich and one global test, the
# fit-function statistic (labelled "fit_function"); its LR is "inapplicable". Components
# outside the policy's scope come back unavailable with a reason, never
# computed under another convention.
policy_inference <- function(fit, data = NULL) {
  if (!inherits(fit, "magmaan_fit")) stop("policy_inference(): supply a fitted magmaan model")
  state <- .policy_state(fit)
  if (state[[4]]) return(policy_inference_impl(NULL, state))
  estimator <- toupper(fit$estimator %||% "")
  if (isTRUE(fit$ordinal) && identical(estimator, "DWLS")) {
    out <- tryCatch(policy_inference_dwls_impl(fit, state), error = function(e) e)
    if (inherits(out, "error"))
      return(.policy_unavailable("unsupported_model", conditionMessage(out), state))
    return(out)
  }
  if (!identical(estimator, "ML") || !is.null(fit$nclusters)) {
    return(.policy_unavailable("unsupported_model",
      "the inference policy covers single-level complete-data ML and all-ordinal DWLS so far", state))
  }
  context <- tryCatch(prepare_inference(fit, data), error = function(e) e)
  if (inherits(context, "error")) {
    return(.policy_unavailable("unsupported_model", conditionMessage(context), state))
  }
  policy_inference_impl(context$native, state)
}

# Converged by the selected acceptance rule; a PSD estimate on the cone
# boundary (inference is computed there under an interior population, and
# flagged); magmaan's own check when a compatibility rule decided
# convergence (NA when magmaan's check decided or did not run); and a
# penalized (barrier) estimate with a positive weight, for which nothing is
# computed. Mirrors api::policy_fit_state(), because inference contexts
# rebuild estimates without their diagnostics.
.policy_state <- function(fit) {
  rule <- fit$fitting$effective$convergence
  native <- if (is.null(rule) || identical(rule, "newton")) NA else
    switch(fit$diagnostics$verdict$status %||% "", passed = TRUE, failed = FALSE, NA)
  c(isTRUE(fit$converged),
    identical(fit$verdict$domain, "psd") &&
      identical(fit$diagnostics$newton_accuracy$covariance_interior, FALSE),
    native,
    identical(fit$penalty_inference, "not_validated") &&
      !isTRUE(fit$composition$penalty_weight == 0))
}

# Mirrors api::verdict_disagreement() for results returned before C++.
.verdict_disagreement <- function(state) {
  !is.null(state) && !is.na(state[[3]]) && state[[3]] != state[[1]]
}

# magmaan's nested tests of fit_H0 against fit_H1, as applied by the
# ordinary-user package's anova(): the likelihood-ratio and score statistics,
# each with SB and PEBA4. fit_H0 may drop, fix or constrain fit_H1's paths,
# fitted to the same observations; interior moment reparameterizations are
# evaluated at a common null point.
policy_nested <- function(fit_H1, fit_H0, data = NULL) {
  if (!inherits(fit_H1, "magmaan_fit") || !inherits(fit_H0, "magmaan_fit"))
    stop("policy_nested(): supply two fitted magmaan models")
  states <- list(H0 = .policy_state(fit_H0), H1 = .policy_state(fit_H1))
  if (states$H0[[4]] || states$H1[[4]])
    return(policy_nested_impl(NULL, NULL, states$H0, states$H1))
  unsupported <- function(detail) {
    t <- .policy_unavailable("unsupported_model", detail)$score
    list(score = t, lr = t, psd_boundary = FALSE,
         verdict_disagreement = .verdict_disagreement(states$H0) ||
           .verdict_disagreement(states$H1))
  }
  for (fit in list(fit_H1, fit_H0)) {
    if (!identical(toupper(fit$estimator %||% ""), "ML") || !is.null(fit$nclusters))
      return(unsupported("the inference policy covers single-level complete-data ML so far"))
  }
  if (is.null(data) && !identical(fit_H1$raw_data, fit_H0$raw_data))
    stop("policy_nested(): the two fits must use the same observations in the same order")
  contexts <- tryCatch({
    shared <- prepare_inference_data(fit_H1, data)
    list(H0 = prepare_inference(fit_H0, shared), H1 = prepare_inference(fit_H1, shared))
  }, error = function(e) e)
  if (inherits(contexts, "error")) return(unsupported(conditionMessage(contexts)))
  policy_nested_impl(contexts$H0$native, contexts$H1$native, states$H0, states$H1)
}

.policy_unavailable <- function(reason, detail, state = NULL) {
  test <- list(available = FALSE, reason = reason, detail = detail,
               statistic = NA_real_, df = 0L, sb_scale = NA_real_,
               p_sb = NA_real_, p_peba4 = NA_real_, eigenvalues = numeric(), label = "")
  list(covariance = NULL, covariance_available = FALSE, covariance_reason = reason,
       covariance_detail = detail, score = test, lr = test, psd_boundary = FALSE,
       verdict_disagreement = .verdict_disagreement(state))
}
