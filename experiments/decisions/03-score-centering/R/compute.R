center_covariance_rows <- function(rows, group = NULL) {
  if (is.null(group)) return(sweep(rows, 2, colMeans(rows), '-'))
  out <- rows
  for (g in unique(group)) {
    at <- which(group == g)
    out[at, ] <- sweep(rows[at, , drop = FALSE], 2,
                      colMeans(rows[at, , drop = FALSE]), '-')
  }
  out
}

relative_gap <- function(a, b) max(abs(a - b)) / max(1, max(abs(b)))

fit_centering_model <- function(syntax, population, lane) {
  # No solver/start overrides: this is the actual ordinary-package default.
  args <- list(model = syntax, data = population$data, meanstructure = TRUE)
  if (is.null(population$group)) args$identification <- 'std.lv'
  else args$group <- population$group
  if (lane == 'fiml') args$estimator <- 'FIML'
  magmaan::as_lab_fit(do.call(magmaan::magmaan, args))
}

centering_row <- function(cell, rep, seed, geometry, arm) {
  data.frame(cell_id = cell$cell_id, family = cell$family, lane = cell$lane,
    test = cell$test, n = cell$n, role = cell$role, rep = rep, seed = seed,
    geometry = geometry, arm = arm, fit_success = FALSE, covariance_available = FALSE,
    test_available = FALSE, estimate = NA_real_, target = NA_real_, variance = NA_real_,
    oracle_variance = NA_real_, coverage = NA, statistic = NA_real_, p_sb = NA_real_,
    p_peba4 = NA_real_, covariance_gap = NA_real_, score_mean_norm = NA_real_,
    group_mean_norm = NA_real_, gram_identity_gap = NA_real_, projection_gap = NA_real_,
    numerator_gap = NA_real_, policy_gap = NA_real_, covariance_policy_gap = NA_real_,
    min_meat_eigenvalue = NA_real_, cov_error = '', test_error = '', fit_error = '',
    elapsed_s = NA_real_, stringsAsFactors = FALSE)
}

run_centering_task <- function(cell, rep, seed) {
  start <- proc.time()[['elapsed']]
  geometries <- if (cell$lane == 'ml') 'expected' else c('expected', 'observed')
  arms <- c('raw', 'global', if (cell$family == 'grouped_mean') 'group')
  rows <- do.call(rbind, lapply(geometries, function(g) do.call(rbind,
    lapply(arms, function(a) centering_row(cell, rep, seed, g, a)))))
  finish <- function(x) { x$elapsed_s <- proc.time()[['elapsed']] - start; x }
  pop <- centering_population(cell, seed)
  attempt <- tryCatch(fit_centering_model(pop$syntax, pop, cell$lane), error = identity)
  if (inherits(attempt, 'error')) {
    rows$fit_error <- conditionMessage(attempt)
    return(finish(rows))
  }
  fit1 <- attempt
  if (!isTRUE(fit1$converged)) {
    rows$fit_error <- 'larger/unrestricted fit did not converge'
    return(finish(rows))
  }
  rows$fit_success <- TRUE
  group <- rep(seq_along(fit1$nobs), times = fit1$nobs)
  # Retained observations are in block order, including FIML cases with NAs.
  context1 <- tryCatch(magmaanlab::prepare_inference(fit1), error = identity)
  if (inherits(context1, 'error')) {
    rows$cov_error <- rows$test_error <- conditionMessage(context1)
    return(finish(rows))
  }
  covs <- tryCatch({
    score_rows <- magmaanlab::scores(context1)$rows
    information <- magmaanlab::inference_information(context1, 'observed')
    free <- fit1$partable$free[fit1$partable$label == pop$label & fit1$partable$free > 0][1]
    values <- lapply(arms, function(arm) {
      s <- switch(arm, raw = score_rows, global = center_covariance_rows(score_rows),
                  group = center_covariance_rows(score_rows, group))
      magmaanlab::parameter_covariance(context1, information, crossprod(s))
    })
    names(values) <- arms
    gap <- if (cell$lane == 'ml') {
      baseline <- magmaanlab::policy_inference(fit1)
      if (!baseline$covariance_available) stop('ordinary covariance baseline unavailable')
      relative_gap(values$raw, baseline$covariance)
    } else 0
    if (gap > 1e-7) stop('raw parameter covariance disagrees with ordinary policy')
    list(values = values, free = free, policy_gap = gap,
         mean_norm = sqrt(sum(colMeans(score_rows)^2)),
         group_mean_norm = max(vapply(split(seq_len(nrow(score_rows)), group), function(at)
           sqrt(sum(colMeans(score_rows[at, , drop = FALSE])^2)), numeric(1))))
  }, error = identity)
  if (inherits(covs, 'error')) rows$cov_error <- conditionMessage(covs)
  else for (arm in arms) {
    at <- rows$arm == arm
    v <- covs$values[[arm]][covs$free, covs$free]
    est <- fit1$theta[covs$free]
    rows$estimate[at] <- est; rows$target[at] <- pop$target
    rows$variance[at] <- v; rows$oracle_variance[at] <- pop$oracle_variance
    rows$covariance_available[at] <- is.finite(v) && v > 0
    rows$coverage[at] <- if (is.finite(pop$target) && is.finite(v) && v > 0)
      abs(est - pop$target) <= qnorm(.975) * sqrt(v) else NA
    rows$covariance_gap[at] <- relative_gap(covs$values[[arm]], covs$values$raw)
    rows$covariance_policy_gap[at] <- covs$policy_gap
    rows$score_mean_norm[at] <- covs$mean_norm
    rows$group_mean_norm[at] <- covs$group_mean_norm
  }

  fit0 <- fit1; context0 <- context1
  if (cell$test == 'nested') {
    fit0 <- tryCatch(fit_centering_model(pop$null, pop, cell$lane), error = identity)
    if (inherits(fit0, 'error') || !isTRUE(fit0$converged)) {
      rows$test_error <- if (inherits(fit0, 'error')) conditionMessage(fit0) else 'restricted fit did not converge'
      return(finish(rows))
    }
    context0 <- tryCatch(magmaanlab::prepare_inference(fit0), error = identity)
    if (inherits(context0, 'error')) {
      rows$test_error <- conditionMessage(context0); return(finish(rows))
    }
  }
  for (geometry in geometries) {
    at_geo <- rows$geometry == geometry
    result <- tryCatch({
      components <- if (cell$test == 'nested')
        magmaanlab::score_components(context0, H1 = fit1, sensitivity = geometry) else
        magmaanlab::score_components(context0, sensitivity = geometry)
      raw <- magmaanlab::project_scores(components)
      projected_rows <- components$rows %*% raw$projection
      projection_gap <- relative_gap(crossprod(projected_rows), raw$meat)
      numerator_gap <- relative_gap(colSums(projected_rows), raw$score)
      centered <- center_covariance_rows(projected_rows)
      identity_gap <- relative_gap(raw$meat - crossprod(centered),
                                    nrow(projected_rows) * tcrossprod(colMeans(projected_rows)))
      if (max(projection_gap, numerator_gap, identity_gap) > 1e-7)
        stop('score-row/projection/Gram identity failed')
      objects <- list(raw = raw,
        global = magmaanlab::project_scores(components, center = TRUE))
      if ('group' %in% arms) objects$group <- magmaanlab::score_quadratic(
        raw$score, raw$metric, crossprod(center_covariance_rows(projected_rows, group)))
      p <- lapply(objects, function(x) magmaanlab::calibrate_quadratic(x, c('sb', 'peba4')))
      policy_gap <- 0
      if (cell$lane == 'ml') {
        baseline <- if (cell$test == 'nested') magmaanlab::policy_nested(fit1, fit0)$score
                    else magmaanlab::policy_inference(fit1)$score
        if (!baseline$available) stop('ordinary score baseline unavailable')
        policy_gap <- max(relative_gap(raw$statistic, baseline$statistic),
          abs(p$raw$p_value[1] - baseline$p_sb), abs(p$raw$p_value[2] - baseline$p_peba4))
        if (policy_gap > 1e-7) stop('raw score disagrees with ordinary policy')
      }
      list(objects = objects, p = p, identity = identity_gap, projection = projection_gap,
           numerator = numerator_gap, policy = policy_gap)
    }, error = identity)
    if (inherits(result, 'error')) rows$test_error[at_geo] <- conditionMessage(result)
    else for (arm in arms) {
      at <- at_geo & rows$arm == arm
      rows$test_available[at] <- TRUE
      rows$statistic[at] <- result$objects[[arm]]$statistic
      rows$p_sb[at] <- result$p[[arm]]$p_value[1]
      rows$p_peba4[at] <- result$p[[arm]]$p_value[2]
      rows$gram_identity_gap[at] <- result$identity
      rows$projection_gap[at] <- result$projection
      rows$numerator_gap[at] <- result$numerator
      rows$policy_gap[at] <- result$policy
      rows$min_meat_eigenvalue[at] <- min(eigen(result$objects[[arm]]$meat,
        symmetric = TRUE, only.values = TRUE)$values)
    }
  }
  finish(rows)
}
