# Exploratory endpoint screening. These labels are not proofs of local/global
# optimality, exact chart singularity, or nonattainment. ML and PSD references
# are separate; ordinary fits never supply the sphere reference.
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

sample_moments <- function(data) list(
  S = list(stats::cov(data) * (nrow(data) - 1) / nrow(data)), nobs = nrow(data))

# Specific to this study's two-factor, simple-structure regression model.
# Re-identification and all objective/derivative checks use library routines.
strong_marker_model <- function(pt, sample) {
  sd <- sqrt(diag(sample$S[[1]])); names(sd) <- ov_names
  lines <- vapply(c("X", "Y"), function(f) {
    z <- pt[pt$lhs == f & pt$op == "=~", ]
    marker <- which.max(abs(z$est) / sd[z$rhs])
    terms <- paste0(ifelse(seq_len(nrow(z)) == marker, "1*", "NA*"), z$rhs)
    paste(f, "=~", paste(terms, collapse = " + "))
  }, "")
  magmaanlab::model_spec(paste(c(lines, "Y ~ X"), collapse = "\n"))
}

standardized_extent <- function(pt, sigma) {
  value <- function(lhs, op, rhs) pt$est[pt$lhs == lhs & pt$op == op & pt$rhs == rhs][1]
  vx <- value("X", "~~", "X"); b <- value("Y", "~", "X")
  psi <- value("Y", "~~", "Y"); vy <- b^2 * vx + psi
  lv <- c(X = abs(vx), Y = abs(vy)); ov <- abs(diag(sigma)); names(ov) <- ov_names
  l <- pt[pt$op == "=~", ]; th <- pt[pt$op == "~~" & pt$lhs %in% ov_names, ]
  vals <- c(abs(l$est) * sqrt(lv[l$lhs] / ov[l$rhs]),
            abs(th$est) / ov[th$lhs], abs(b) * sqrt(abs(vx / vy)),
            abs(b * vx) / sqrt(abs(vx * vy)), abs(psi / vy))
  if (any(!is.finite(vals))) Inf else max(vals)
}

# Common random initial points across backends, generated in marker units.
# Positive independent disturbances ensure admissible starts; free loading
# signs and the latent regression vary. These are not random global restarts.
random_start <- function(spec, sample, seed) {
  set.seed(seed)
  pt <- spec$partable; v <- diag(sample$S[[1]]); names(v) <- ov_names
  unit <- c(sqrt(v), X = sqrt(v[["x1"]]), Y = sqrt(v[["y1"]]))
  theta <- numeric(max(pt$free))
  for (i in which(pt$free > 0)) {
    a <- pt$lhs[i]; b <- pt$rhs[i]
    theta[pt$free[i]] <- switch(pt$op[i],
      "=~" = rnorm(1, .5, .8) * unit[b] / unit[a],
      "~" = runif(1, -.8, .8) * unit[a] / unit[b],
      "~~" = runif(1, .3, 1.2) * unit[a]^2,
      stop("unsupported start row"))
  }
  theta
}

endpoint_record <- function() data.frame(
  returned = FALSE, call_status = "error", backend_status = "", original_verdict = NA,
  objective = NA_real_, sphere_objective = NA_real_, objective_gap = NA_real_,
  full_sphere = NA, driven_stationary = NA, pin_residual = NA_real_,
  chart_level = NA_real_, chart_1e4 = "unavailable", chart_1e6 = "unavailable",
  chart_1e8 = "unavailable", audit_status = "unavailable", newton_distance = NA_real_,
  newton_condition = NA_real_, newton_step = NA_real_, admissible = NA,
  std_extent = NA_real_, extreme = NA, screened = FALSE,
  label = "call_error", start_used = "", seconds = NA_real_, message = "", warnings = "",
  stringsAsFactors = FALSE)

chart_label <- function(level, tol) {
  if (!is.finite(level)) "unavailable" else if (level <= tol) "near_chart_boundary" else "representable"
}

assess_endpoint <- function(fit, sample, domain, sphere,
                            marker_model = strong_marker_model,
                            extent = standardized_extent) {
  rec <- endpoint_record(); rec$returned <- TRUE
  rec$call_status <- if (inherits(fit, "magmaan_user_chart_singular")) "chart_condition" else "returned"
  rec$original_verdict <- if (is.logical(fit$converged)) fit$converged else NA
  if (!sphere) rec$start_used <- fit$ml_start_policy %||% ""
  g <- if (sphere) fit$gauge else NULL
  pt <- if (sphere) g$sphere_partable else fit$partable
  rec$backend_status <- if (sphere) g$optimizer_status else fit$optimizer_status %||% ""
  if (sphere) {
    rec$sphere_objective <- g$fmin_sphere
    rec$full_sphere <- nrow(g$passthrough) == 0 && nrow(g$units) == length(unique(pt$lhs[pt$op == "=~"]))
    rec$driven_stationary <- isTRUE(g$driven_stationary)
    rec$pin_residual <- g$pin_residual
    rec$chart_level <- min(abs(g$units$direction_level))
    rec$start_used <- g$start
    rec$chart_1e4 <- chart_label(rec$chart_level, 1e-4)
    rec$chart_1e6 <- chart_label(rec$chart_level, 1e-6)
    rec$chart_1e8 <- chart_label(rec$chart_level, 1e-8)
  }
  # No refitting: a well-loaded marker avoids the originally requested pole.
  # This is a chart-based Newton cross-check, not a sphere-tangent certificate.
  checked <- tryCatch({
    target <- marker_model(pt, sample)
    translated <- magmaanlab::frontier_reidentify(pt, target, pole_tol = 0)
    ev <- magmaanlab::magmaan_core$evaluate_at(
      target$partable, sample, translated$theta, estimator = "ML")
    na <- magmaanlab::frontier_newton_accuracy(ev, psd = domain == "PSD")
    sigma <- magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
    list(ev = ev, na = na, sigma = sigma)
  }, error = function(e) e)
  if (inherits(checked, "error")) {
    rec$label <- "assessment_unavailable"; rec$message <- conditionMessage(checked)
    return(list(record = rec, partable = pt))
  }
  ev <- checked$ev; na <- checked$na
  rec$objective <- ev$fmin
  rec$objective_gap <- if (sphere) abs(rec$objective - rec$sphere_objective) else 0
  rec$audit_status <- na$status; rec$newton_distance <- na$distance
  rec$newton_condition <- na$condition; rec$newton_step <- na$max_step
  rec$admissible <- isTRUE(ev$diagnostics$admissibility$admissible)
  rec$std_extent <- extent(ev$partable, checked$sigma)
  rec$extreme <- !is.finite(rec$std_extent) || rec$std_extent > 10
  rec$label <- if (!is.finite(rec$objective)) "nonfinite_objective" else
    if (sphere && !isTRUE(rec$full_sphere)) "partial_sphere" else
    if (sphere && (!is.finite(rec$pin_residual) || rec$pin_residual > 1e-6)) "sphere_norm_failed" else
    if (sphere && !isTRUE(rec$driven_stationary)) "sphere_first_order_failed" else
    if (rec$objective_gap > 1e-6 * (1 + abs(rec$objective))) "objective_disagreement" else
    if (domain == "PSD" && !rec$admissible) "psd_infeasible" else
    if (!isTRUE(na$checked) || na$status != "available") "accuracy_unresolved" else
    if (!isTRUE(na$passed)) "accuracy_failed" else
    if (rec$extreme) "screened_extreme" else "screened_candidate"
  rec$screened <- rec$label == "screened_candidate"
  list(record = rec, partable = pt)
}

run_fit <- function(spec, data, sample, domain, route, backend, start_id, theta = NULL,
                    preconditioning = "diagonal", assessment = list(), control = NULL) {
  t0 <- proc.time()[["elapsed"]]
  warnings <- character()
  fit <- tryCatch(withCallingHandlers({
    if (route == "sphere") {
      ctl <- if (start_id == "layered") list(start = "layered") else NULL
      if (!is.null(theta)) {
        free <- spec$partable$free
        spec$partable$ustart[free > 0] <- theta[free[free > 0]]
      }
      magmaanlab::frontier_fit_sphere(spec, data, psd = domain == "PSD", optimizer = backend,
        control = ctl, polish = FALSE, preconditioning = preconditioning)
    } else if (route == "lavaan") {
      f <- lavaan::sem(model_syntax, data = data, meanstructure = FALSE)
      list(partable = lavaan::parTable(f), converged = lavaan::lavInspect(f, "converged"),
           optimizer_status = "lavaan_nlminb")
    } else if (domain == "PSD") {
      magmaanlab::frontier_fit_ml_psd(spec, data, optimizer = backend,
        preconditioning = preconditioning, control = if (is.null(theta)) control else modifyList(control %||% list(), list(start = theta)))
    } else {
      magmaanlab::fit_model(spec, data, optimizer = backend,
        control = if (is.null(theta)) control else modifyList(control %||% list(), list(start = theta)))
    }
  }, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), magmaan_user_chart_singular = function(e) e, error = function(e) e)
  result <- if (inherits(fit, "error") && !inherits(fit, "magmaan_user_chart_singular")) {
    rec <- endpoint_record(); rec$message <- conditionMessage(fit)
    list(record = rec, partable = NULL)
  } else do.call(assess_endpoint, c(list(fit, sample, domain, route == "sphere"), assessment))
  result$record$warnings <- paste(unique(warnings), collapse = " | ")
  result$record$seconds <- proc.time()[["elapsed"]] - t0
  result
}

reference_rows <- function(fits) {
  keys <- c("design", "n", "rep", "transform", "domain")
  key <- interaction(fits[keys], drop = TRUE, lex.order = TRUE)
  out <- lapply(split(fits, key), function(d) {
    s <- d[d$route == "sphere" & d$screened, ]
    best <- if (nrow(s)) min(s$objective) else NA_real_
    near <- if (nrow(s)) s[s$objective <= best + 1e-6 * (1 + abs(best)), ] else s
    ns <- length(unique(near$start_id)); nb <- length(unique(near$backend))
    levels <- numeric()
    for (v in sort(s$objective)) if (!length(levels) ||
        v > tail(levels, 1) + 1e-6 * (1 + abs(tail(levels, 1)))) levels <- c(levels, v)
    chart <- if (!nrow(near)) "unavailable" else paste(sort(unique(near$chart_1e6)), collapse = ";")
    cbind(d[1, keys, drop = FALSE], data.frame(
      reference_label = if (ns >= 2) "repeated_best_observed" else if (ns == 1) "single_start_best_observed" else "no_screened_reference",
      best_objective = best, screened_sphere = nrow(s), matching_starts = ns,
      matching_backends = nb, observed_objective_levels = length(levels), chart_status = chart,
      near_chart_min = if (nrow(near)) min(near$chart_level) else NA_real_,
      extreme_sphere = sum(d$route == "sphere" & d$label == "screened_extreme")))
  })
  do.call(rbind, out)
}

compare_rows <- function(fits, refs) {
  keys <- c("design", "n", "rep", "transform", "domain")
  x <- merge(fits, refs, by = keys, all.x = TRUE, sort = FALSE)
  x$reference_gap <- x$objective - x$best_objective
  tol <- 1e-6 * (1 + abs(x$best_objective))
  x$comparison <- ifelse(!x$screened, "not_screened", ifelse(!is.finite(x$best_objective),
    "no_sphere_reference", ifelse(x$reference_gap < -tol, "better_than_sphere_reference",
    ifelse(x$reference_gap > tol, "above_sphere_reference", "matches_sphere_reference"))))
  x
}
