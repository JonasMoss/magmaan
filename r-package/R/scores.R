# Fit-owned cache: keys retain all portable inputs under R value semantics.
# A changed model, estimate, data, parameterization or weight gets a new handle.
.policy_context <- function(fit, kind, build, data = NULL) {
  cache <- attr(fit, "policy_cache")
  if (!is.environment(cache)) return(build())
  keys <- fit
  attr(keys, "policy_cache") <- NULL
  keys <- list(fit = keys, data = data)
  saved <- cache[[kind]]
  native <- if (is.null(saved)) NULL else if (kind == "dwls") saved$value else saved$value$native
  if (is.null(saved) || !identical(saved$pid, Sys.getpid()) ||
      is.null(native) || identical(format(native), "<pointer: (nil)>") ||
      !identical(saved$keys, keys)) {
    saved <- list(pid = Sys.getpid(), keys = keys, value = build())
    cache[[kind]] <- saved
  }
  saved$value
}

# Explicit, immutable inference snapshots; no p-values during preparation.
.score_object <- function(x, class, ...) {
  do.call(.prepared_object, c(x, list(...), list(class = class)))
}

prepare_inference <- function(fit, data = NULL) {
  if (inherits(fit, "magmaan_inference")) {
    if (!is.null(data)) stop("prepare_inference(): a snapshot already owns its data")
    if (identical(format(fit$native), "<pointer: (nil)>") ||
        !identical(fit$pid, Sys.getpid())) return(prepare_inference(fit$original_fit,
          if (identical(fit$estimator, "ML")) fit$raw else NULL))
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
  build <- function() .score_object(prepare_inference_impl(fit, raw, shared_data),
                                   "magmaan_inference", pid = Sys.getpid())
  if (estimator == "FIML") .policy_context(fit, "fiml", build) else build()
}

scores <- function(object, data = NULL, space = c("parameter", "saturated")) {
  context <- prepare_inference(object, data)
  score_rows_impl(context$native, match.arg(space))
}

score_components <- function(object, data = NULL, H1 = NULL,
    sensitivity = c("observed", "expected", "observed-h1", "observed-shrink-light", "observed-shrink-sqrt"),
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
score_quadratic <- function(score, metric, meat) {
  if (missing(meat) || is.null(meat)) stop("score_quadratic(): supply an explicit meat matrix")
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
      any(!is.finite(eigenvalues))))
    stop("quadratic_reference(): eigenvalues must be finite and include df entries")
  if (!is.null(eigenvalues)) eigenvalues <- quadratic_spectrum_impl(as.numeric(eigenvalues))
  .score_object(list(statistic = statistic, df = as.integer(df), eigenvalues = eigenvalues),
                "magmaan_quadratic_reference")
}

# Keep the public grammar in one place; ordinary reporting validates through
# this wrapper as well, so it cannot drift from lab calibration.
.quadratic_methods <- function(methods) {
  grammar <- "std, sb, ss, mv, scaled_f, all, pall, peba<k>, eba<k> (integer k >= 1)"
  if (!is.character(methods) || !length(methods) || anyNA(methods))
    stop(paste0("calibrate_quadratic(): use ", grammar), call. = FALSE)
  methods <- tolower(methods)
  valid <- grepl("^(std|sb|ss|mv|scaled_f|all|pall|peba[1-9][0-9]*|eba[1-9][0-9]*)$", methods)
  block <- startsWith(methods, "peba") | startsWith(methods, "eba")
  k <- suppressWarnings(as.numeric(sub("^p?eba", "", methods[block])))
  # The C++ block count is an int; reject values outside that representation.
  if (any(!valid) || any(!is.finite(k) | k > .Machine$integer.max))
    stop(paste0("calibrate_quadratic(): use ", grammar,
                "; k must fit a C++ integer"), call. = FALSE)
  methods
}

calibrate_quadratic <- function(object, methods = "peba4") {
  methods <- .quadratic_methods(methods)
  needs_spectrum <- any(!methods %in% c("std", "sb"))
  if (inherits(object, "magmaan_ntml_quadratic"))
    object <- .score_object(ntml_reference_impl(object$native, needs_spectrum),
                            "magmaan_quadratic_reference")
  if (inherits(object, "magmaan_projected_score")) {
    if (needs_spectrum) object <- score_spectrum(object) else if ("sb" %in% methods)
      object <- .score_object(score_reference_impl(object$native, FALSE), "magmaan_quadratic_reference")
  }
  if (!inherits(object, c("magmaan_quadratic_reference", "magmaan_projected_score")))
    stop("calibrate_quadratic(): supply projected scores or a quadratic reference")
  p <- vapply(methods, function(method) {
    if (method == "std") return(infer_chi2_pvalue(object$statistic, as.integer(object$df)))
    if (method == "sb" && !is.null(object$mean_scale))
      return(infer_chi2_pvalue(object$statistic / object$mean_scale, as.integer(object$df)))
    if (is.null(object$eigenvalues)) stop("calibrate_quadratic(): this reference has no spectrum")
    block <- startsWith(method, "peba") | startsWith(method, "eba")
    infer_fmg_test(object$statistic, object$df, object$eigenvalues,
      method = if (block) sub("[0-9]+$", "", method) else method,
      param = if (block) as.numeric(sub("^p?eba", "", method)) else 0)$p_value
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

inference_information <- function(context, type = c("observed", "expected")) {
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
  .marker_refusal(a, b, caller = "nested inference")
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
                                geometry = NULL) {
  test <- match.arg(test)
  hypothesis <- inherits(object,"magmaan_inference_hypothesis")
  if (is.null(geometry)) geometry <- if (hypothesis) "observed" else "expected"
  geometry <- match.arg(geometry, c("observed", "expected"))
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

inference_covariance <- function(context, robust = TRUE) {
  stopifnot(inherits(context,"magmaan_inference"),is.logical(robust),length(robust)==1L,!is.na(robust))
  out <- ntml_covariance_impl(context$native,robust)
  attr(out,"inference_context") <- context
  out
}

inference_reuse <- function(context) {
  if (inherits(context, "magmaan_fit") && (isTRUE(context$ordinal) || isTRUE(context$mixed_ordinal)))
    return(list(ingredient_builds = dwls_policy_reuse_impl(.policy_context(context,
      "dwls", function() prepare_policy_dwls_impl(context)))))
  stopifnot(inherits(context,"magmaan_inference"))
  context <- prepare_inference(context)
  inference_reuse_impl(context$native)
}

# magmaan's default inference policy for one fit, as applied by the
# ordinary-user package: the observed-information sandwich covariance and the
# global score and likelihood-ratio tests with PEBA4 by default (SB is a
# comparator). All-ordinal and complete mixed
# DWLS uses the estimated-weight (IJ) sandwich and one global test, the
# fit-function statistic (labelled "fit_function") with the exact spectrum All
# reference (reference="all", p_all); its LR is "inapplicable". Components
# outside the policy's scope come back unavailable with a reason, never
# computed under another convention.
policy_inference <- function(fit, data = NULL) {
  if (!inherits(fit, "magmaan_fit")) stop("policy_inference(): supply a fitted magmaan model")
  state <- .policy_state(fit)
  if (state[[4]]) return(policy_inference_impl(NULL, state))
  estimator <- toupper(fit$estimator %||% "")
  if ((isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)) && identical(estimator, "DWLS")) {
    out <- tryCatch(policy_inference_dwls_impl(fit, state,
      .policy_context(fit, "dwls", function() prepare_policy_dwls_impl(fit))), error = function(e) e)
    if (inherits(out, "error"))
      return(.policy_unavailable("unsupported_model", conditionMessage(out), state))
    return(out)
  }
  if (!estimator %in% c("ML", "FIML") || !is.null(fit$nclusters)) {
    return(.policy_unavailable("unsupported_model",
      "the inference policy covers single-level ML, FIML and ordinal or mixed DWLS", state))
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
# with PEBA4 by default (SB is a comparator). DWLS uses observed-Hessian/IJ
# fit-function differences with the All reference; nested score is unavailable.
# fit_H0 may drop, fix or constrain fit_H1's paths,
# fitted to the same observations; interior moment reparameterizations are
# evaluated at a common null point.
policy_nested <- function(fit_H1, fit_H0, data = NULL) {
  if (!inherits(fit_H1, "magmaan_fit") || !inherits(fit_H0, "magmaan_fit"))
    stop("policy_nested(): supply two fitted magmaan models")
  .marker_refusal(fit_H1, fit_H0, caller = "policy_nested()")
  states <- list(H0 = .policy_state(fit_H0), H1 = .policy_state(fit_H1))
  if (states$H0[[4]] || states$H1[[4]])
    return(policy_nested_impl(NULL, NULL, states$H0, states$H1))
  unsupported <- function(detail) {
    t <- .policy_unavailable("unsupported_model", detail)$score
    list(score = t, lr = t, psd_boundary = FALSE,
         verdict_disagreement = .verdict_disagreement(states$H0) ||
           .verdict_disagreement(states$H1))
  }
  dwls <- vapply(list(fit_H1, fit_H0), function(fit)
    (isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)) && identical(toupper(fit$estimator %||% ""), "DWLS"), logical(1))
  if (all(dwls)) {
    if (!identical(fit_H1$ordinal_stats, fit_H0$ordinal_stats) ||
        !identical(fit_H1$mixed_ordinal_stats, fit_H0$mixed_ordinal_stats))
      stop("policy_nested(): the two fits must use the same observations in the same order")
    out <- tryCatch(policy_nested_dwls_impl(fit_H1, fit_H0, states$H0, states$H1,
      .policy_context(fit_H0, "dwls", function() prepare_policy_dwls_impl(fit_H0)),
      .policy_context(fit_H1, "dwls", function() prepare_policy_dwls_impl(fit_H1))),
                    error = function(e) e)
    if (inherits(out, "error")) return(unsupported(conditionMessage(out)))
    return(out)
  }
  for (fit in list(fit_H1, fit_H0)) {
    if (!toupper(fit$estimator %||% "") %in% c("ML", "FIML") || !is.null(fit$nclusters))
      return(unsupported("the inference policy covers single-level ML, FIML and ordinal or mixed DWLS"))
  }
  fiml <- identical(toupper(fit_H1$estimator), "FIML")
  if ((is.null(data) || fiml) && !identical(fit_H1$raw_data, fit_H0$raw_data))
    stop("policy_nested(): the two fits must use the same observations in the same order")
  contexts <- tryCatch({
    if (!identical(toupper(fit_H0$estimator), toupper(fit_H1$estimator)))
      stop("nested policy requires the same estimator")
    shared <- if (fiml) data else prepare_inference_data(fit_H1, data)
    list(H0 = prepare_inference(fit_H0, shared), H1 = prepare_inference(fit_H1, shared))
  }, error = function(e) e)
  if (inherits(contexts, "error")) return(unsupported(conditionMessage(contexts)))
  policy_nested_impl(contexts$H0$native, contexts$H1$native, states$H0, states$H1)
}

.policy_unavailable <- function(reason, detail, state = NULL) {
  test <- list(available = FALSE, reason = reason, detail = detail,
               statistic = NA_real_, df = 0L, sb_scale = NA_real_,
               p_sb = NA_real_, p_peba4 = NA_real_, peba_blocks = 0L, eigenvalues = numeric(), label = "")
  list(covariance = NULL, covariance_available = FALSE, covariance_reason = reason,
       covariance_detail = detail, score = test, lr = test, psd_boundary = FALSE,
       verdict_disagreement = .verdict_disagreement(state))
}
