# Draws, unit transforms, configurations (arms) and the per-fit record.
# Every fit is judged by the library verdict (`fit$converged`, the Newton
# check on the penalized objective). The draw, unit and seed helpers are
# copied from decisions/01-optimizer-defaults.

hash_int <- function(s) {
  h <- 0
  for (b in utf8ToInt(s)) h <- (h * 131 + b) %% 2147483629
  as.integer(h)
}

# Cell-stable: the seed depends only on the base, the population, N and the
# replication, never on the grid order.
draw_seed <- function(seed_base, pop_key, n, rep) {
  hash_int(paste(seed_base, pop_key, n, rep, sep = "|"))
}

draw_moments <- function(pop, n, seed) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * pop$p), n) %*% chol(pop$Sigma)
  if (!is.null(pop$mu)) X <- sweep(X, 2, pop$mu, "+")
  m <- colMeans(X)
  S <- crossprod(sweep(X, 2, m)) / n
  dimnames(S) <- dimnames(pop$Sigma); names(m) <- colnames(pop$Sigma)
  list(S = S, mean = m, n = n)
}

# Units: every variable times 100, every variable times 0.01, or alternate
# variables times 100 and 0.01 (separate units; invariant models only).
transform_factors <- function(transform, p) {
  switch(transform,
    native = rep(1, p), x100 = rep(100, p), x0.01 = rep(.01, p),
    mixed = ifelse(seq_len(p) %% 2L == 1L, 100, .01),
    stop("unknown transform ", transform))
}

sample_in_units <- function(moments, factors, meanstructure) {
  list(S = list(moments$S * outer(factors, factors)), nobs = as.integer(moments$n),
       mean = if (meanstructure) list(moments$mean * factors) else NULL)
}

build_model <- function(fm) {
  magmaanlab::model_spec(fm$syntax, std_lv = fm$std_lv, meanstructure = fm$meanstructure)
}


weights <- c(lambda0.25 = .25, lambda1 = 1)

# `default` passes no weight at 0.25, so it is the library default call.
lane_arms <- function(lane = "barrier-ml") list(
  default       = list(),
  layered_port  = list(start = "layered"),
  default_lbfgs = list(optimizer = "nlopt-lbfgs"),
  layered_lbfgs = list(start = "layered", optimizer = "nlopt-lbfgs"))

lane_estimators <- function(lane = "barrier-ml") names(weights)

blank_record <- function() {
  data.frame(returned = FALSE, certified = FALSE, admissible = NA, fmin = NA_real_,
             fmin_ml = NA_real_, newton_status = "", optimizer_status = "",
             f_evals = NA_integer_, seconds = NA_real_, start_policy = "",
             start_repaired = NA, stage = "", max_abs_theta = NA_real_,
             std_extent = NA_real_, chart_extent = NA_real_, message = "",
             stringsAsFactors = FALSE)
}

# Largest absolute standardized quantity of a single-group fit: loadings,
# latent paths, latent correlations, and residual and disturbance variance
# ratios. It is unit-free, near or below 1 for proper solutions (improper
# ones slightly above), and large far along a divergent path. Absolute values
# keep it defined when a variance is negative; a latent with zero implied
# variance gives Inf.
standardized_extent <- function(fit) {
  p <- fit$partable; e <- p$est
  if (is.null(e) || !length(e)) return(NA_real_)
  lv <- unique(p$lhs[p$op == "=~"]); ov <- fit$ov_names
  Sigma <- tryCatch(magmaanlab::magmaan_core$model_implied(fit)$sigma[[1]],
                    error = function(err) NULL)
  if (is.null(Sigma) || !length(lv)) return(NA_real_)
  k <- length(lv); B <- matrix(0, k, k, dimnames = list(lv, lv)); Psi <- B
  for (i in which(p$op == "~" & p$lhs %in% lv & p$rhs %in% lv)) B[p$lhs[i], p$rhs[i]] <- e[i]
  for (i in which(p$op == "~~" & p$lhs %in% lv & p$rhs %in% lv))
    Psi[p$lhs[i], p$rhs[i]] <- Psi[p$rhs[i], p$lhs[i]] <- e[i]
  A <- tryCatch(solve(diag(k) - B), error = function(err) NULL)
  if (is.null(A)) return(Inf)
  Phi <- A %*% Psi %*% t(A); v <- abs(diag(Phi)); names(v) <- lv
  s2 <- abs(diag(Sigma)); names(s2) <- ov
  L <- which(p$op == "=~" & p$rhs %in% ov)
  R <- which(p$op == "~" & p$lhs %in% lv & p$rhs %in% lv)
  D <- which(p$op == "~~" & p$lhs == p$rhs & p$lhs %in% lv)
  Th <- which(p$op == "~~" & p$lhs %in% ov & p$rhs %in% ov)
  corr <- abs(Phi) / sqrt(outer(v, v)); diag(corr) <- 0
  vals <- c(abs(e[L]) * sqrt(v[p$lhs[L]] / s2[p$rhs[L]]),
            abs(e[R]) * sqrt(v[p$rhs[R]] / v[p$lhs[R]]),
            abs(e[D]) / v[p$lhs[D]],
            abs(e[Th]) / sqrt(s2[p$lhs[Th]] * s2[p$rhs[Th]]),
            corr)
  vals[is.nan(vals)] <- Inf
  max(vals)
}

# Marker-chart extent: for each latent scaled by a fixed nonzero loading,
# the largest absolute standardized loading among its indicators over the
# marker's. Unit-free; large only when the marker's standardized loading goes
# to zero (the marker-chart pole). 1 when no latent has a marker.
chart_extent <- function(fit) {
  p <- fit$partable; e <- p$est
  lv <- unique(p$lhs[p$op == "=~"]); ov <- fit$ov_names
  if (is.null(e) || !length(lv)) return(1)
  Sigma <- tryCatch(magmaanlab::magmaan_core$model_implied(fit)$sigma[[1]],
                    error = function(err) NULL)
  if (is.null(Sigma)) return(NA_real_)
  k <- length(lv); B <- matrix(0, k, k, dimnames = list(lv, lv)); Psi <- B
  for (i in which(p$op == "~" & p$lhs %in% lv & p$rhs %in% lv)) B[p$lhs[i], p$rhs[i]] <- e[i]
  for (i in which(p$op == "~~" & p$lhs %in% lv & p$rhs %in% lv))
    Psi[p$lhs[i], p$rhs[i]] <- Psi[p$rhs[i], p$lhs[i]] <- e[i]
  A <- tryCatch(solve(diag(k) - B), error = function(err) NULL)
  if (is.null(A)) return(Inf)
  v <- abs(diag(A %*% Psi %*% t(A))); names(v) <- lv
  s2 <- abs(diag(Sigma)); names(s2) <- ov
  out <- 1
  for (f in lv) {
    rows <- which(p$op == "=~" & p$lhs == f & p$rhs %in% ov)
    marker <- rows[p$free[rows] == 0 & e[rows] != 0]
    if (!length(marker)) next
    std <- abs(e[rows]) * sqrt(v[f] / s2[p$rhs[rows]])
    sm <- abs(e[marker[1]]) * sqrt(v[f] / s2[p$rhs[marker[1]]])
    out <- max(out, if (sm > 0) max(std) / sm else Inf)
  }
  out
}

fill_record <- function(rec, fit) {
  rec$returned <- TRUE
  rec$certified <- isTRUE(fit$converged)
  rec$admissible <- isTRUE(fit$diagnostics$admissibility$admissible)
  rec$fmin <- fit$penalty$penalized_fmin %||% NA_real_
  rec$fmin_ml <- fit$fmin %||% NA_real_
  rec$newton_status <- fit$diagnostics$newton_accuracy$status %||% ""
  rec$optimizer_status <- fit$optimizer_status %||% ""
  rec$f_evals <- as.integer(fit$f_evals %||% NA_integer_)
  rec$start_policy <- fit$ml_start_policy %||% ""
  rec$start_repaired <- isTRUE(fit$penalty$start_repaired)
  rec$max_abs_theta <- if (length(fit$theta)) max(abs(fit$theta)) else NA_real_
  rec$std_extent <- tryCatch(standardized_extent(fit), error = function(err) NA_real_)
  rec$chart_extent <- tryCatch(chart_extent(fit), error = function(err) NA_real_)
  rec
}

one_line <- function(x) gsub("[\r\n,]+", " ", substr(x, 1, 300))

# One configuration on one problem at one weight. `start` may be a policy
# name or a vector. The `default` arm at weight 0.25 passes no weight.
run_arm <- function(arm, model, sample, weight_name) {
  rec <- blank_record()
  ctl <- c(if (is.null(arm$start)) NULL else list(start = arm$start), arm$control)
  if (!length(ctl)) ctl <- NULL
  args <- list(model, sample, target = "determinacy", control = ctl)
  if (weight_name != "lambda0.25") args$weight <- unname(weights[weight_name])
  if (!is.null(arm$optimizer)) args$optimizer <- arm$optimizer
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(suppressWarnings(do.call(magmaanlab::frontier_fit_ml_multiinfo, args)),
                  error = function(e) e)
  rec$seconds <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    rec$message <- one_line(conditionMessage(fit))
    return(rec)
  }
  fill_record(rec, fit)
}

# A start at the population (pseudo-true) values, from an ML fit to the
# population moments in the same units. Only a witness of attainability.
population_start <- function(pop, fm, model, factors) {
  sample <- sample_in_units(list(S = pop$Sigma, mean = pop$mu %||% rep(0, pop$p), n = 1e6L),
                            factors, fm$meanstructure)
  for (opt in c("port", "nlopt-lbfgs")) {
    fit <- tryCatch(magmaanlab::magmaan_core$fit_ml(model, sample, optimizer = opt,
                                                    control = list(start = "layered")),
                    error = function(e) NULL)
    if (!is.null(fit) && isTRUE(fit$converged)) return(as.numeric(fit$theta))
  }
  NULL
}

witness_arm <- function(theta) list(optimizer = "port", start = theta)

# Every model, transform, weight and arm on one draw of one population.
run_task <- function(task, pops, lane, cache, arm_filter = NULL) {
  pop <- pops[[task$pop]]
  moments <- draw_moments(pop, task$n, task$seed)
  arms <- lane_arms(lane)
  if (!is.null(arm_filter)) arms <- arms[names(arms) %in% arm_filter]
  want_witness <- is.null(arm_filter) || "witness" %in% arm_filter
  rows <- list()
  transforms <- c("native", "x100", "x0.01", "mixed")
  for (fm in pop$models) for (tr in transforms) {
    if (tr == "mixed" && !fm$unit_invariant) next
    factors <- transform_factors(tr, pop$p)
    model <- build_model(fm)
    sample <- sample_in_units(moments, factors, fm$meanstructure)
    key <- paste(pop$key, fm$key, tr, sep = "|")
    all_arms <- arms
    if (want_witness) {
      if (!exists(key, envir = cache, inherits = FALSE))
        assign(key, population_start(pop, fm, model, factors), envir = cache)
      theta_pop <- get(key, envir = cache)
      if (!is.null(theta_pop)) all_arms$witness <- witness_arm(theta_pop)
    }
    for (w in lane_estimators(lane)) for (a in names(all_arms)) {
      rec <- run_arm(all_arms[[a]], model, sample, w)
      rows[[length(rows) + 1L]] <- cbind(data.frame(
        lane = lane, family = pop$family, role = pop$role, pop = pop$key, model = fm$key,
        n = task$n, rep = task$rep, seed = task$seed, transform = tr, estimator = w,
        arm = a, stringsAsFactors = FALSE), rec)
    }
  }
  do.call(rbind, rows)
}
