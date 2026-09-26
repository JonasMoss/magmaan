# Draws, unit transforms, configurations (arms) and the per-fit record.
# Every fit is judged by the library verdict (`fit$converged`, the Newton
# check); PSD fits must also be admissible.

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

lane_arms <- function(lane) {
  switch(lane,
    "ml-gls" = list(
      default       = list(kind = "ordinary"),
      default_port  = list(kind = "ordinary", optimizer = "port"),
      layered_lbfgs = list(kind = "ordinary", optimizer = "nlopt-lbfgs", start = "layered"),
      layered_port  = list(kind = "ordinary", optimizer = "port", start = "layered"),
      # Added for the confirmation run: PORT with complete-data ML's budget
      # (5000 evaluations, as NLopt gets) and tolerances, or the budget only.
      layered_port_ml = list(kind = "ordinary", optimizer = "port", start = "layered",
        control = list(max_iter = 5000L, port = list(max_eval = 5000L, rel_f_tol = 1e-12, x_tol = 1e-10))),
      layered_port_budget = list(kind = "ordinary", optimizer = "port", start = "layered",
        control = list(max_iter = 5000L, port = list(max_eval = 5000L)))),
    "psd-ml" = list(
      psd_default        = list(kind = "psd"),
      psd_default_none   = list(kind = "psd", preconditioning = "none"),
      psd_layered        = list(kind = "psd", start = "layered"),
      psd_layered_none   = list(kind = "psd", start = "layered", preconditioning = "none"),
      twostage_default   = list(kind = "twostage"),
      twostage_layered   = list(kind = "twostage", optimizer = "port", start = "layered"),
      # Added for the second run: the ordinary step from complete-data ML's
      # default since 52bc9caa (the layered start with L-BFGS).
      twostage_layered_lbfgs = list(kind = "twostage", start = "layered")),
    stop("unknown lane ", lane))
}

lane_estimators <- function(lane) if (lane == "ml-gls") c("ML", "GLS") else "ML"

blank_record <- function() {
  data.frame(returned = FALSE, certified = FALSE, admissible = NA, fmin = NA_real_,
             newton_status = "", optimizer_status = "", f_evals = NA_integer_,
             seconds = NA_real_, start_policy = "", stage = "", max_abs_theta = NA_real_,
             std_extent = NA_real_, message = "", stringsAsFactors = FALSE)
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

fill_record <- function(rec, fit) {
  rec$returned <- TRUE
  rec$certified <- isTRUE(fit$converged)
  rec$admissible <- isTRUE(fit$diagnostics$admissibility$admissible)
  rec$fmin <- fit$fmin %||% NA_real_
  rec$newton_status <- fit$diagnostics$newton_accuracy$status %||% ""
  rec$optimizer_status <- fit$optimizer_status %||% ""
  rec$f_evals <- as.integer(fit$f_evals %||% NA_integer_)
  rec$start_policy <- fit$ml_start_policy %||% ""
  rec$max_abs_theta <- if (length(fit$theta)) max(abs(fit$theta)) else NA_real_
  rec$std_extent <- tryCatch(standardized_extent(fit), error = function(err) NA_real_)
  rec
}

one_line <- function(x) gsub("[\r\n,]+", " ", substr(x, 1, 300))

# One configuration on one problem. `start` may be a policy name or a vector.
run_arm <- function(arm, model, sample, estimator) {
  rec <- blank_record()
  ctl <- c(if (is.null(arm$start)) NULL else list(start = arm$start), arm$control)
  if (!length(ctl)) ctl <- NULL
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(switch(arm$kind,
    ordinary = {
      fitter <- if (estimator == "ML") magmaanlab::magmaan_core$fit_ml else magmaanlab::magmaan_core$fit_gls
      if (is.null(arm$optimizer)) fitter(model, sample, control = ctl)
      else fitter(model, sample, optimizer = arm$optimizer, control = ctl)
    },
    psd = magmaanlab::frontier_fit_ml_psd(model, sample, control = ctl,
      preconditioning = arm$preconditioning %||% "diagonal"),
    twostage = magmaanlab::frontier_fit_ml_psd_fallback(model, sample,
      ordinary_optimizer = arm$optimizer %||% "nlopt-lbfgs", ordinary_control = ctl)),
    error = function(e) e)
  rec$seconds <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    rec$message <- one_line(conditionMessage(fit))
    return(rec)
  }
  if (arm$kind == "twostage") {
    used <- if (isTRUE(fit$fallback_used)) fit$psd else fit$ordinary
    rec$stage <- if (isTRUE(fit$fallback_used)) paste0("psd:", fit$fallback_reason) else "ordinary"
    if (!is.null(used$fit)) rec <- fill_record(rec, used$fit)
    else rec$message <- one_line(paste(used$error$kind %||% "", used$error$detail %||% ""))
    rec$certified <- isTRUE(fit$converged)
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

witness_arm <- function(lane, theta) {
  if (lane == "ml-gls") list(kind = "ordinary", optimizer = "port", start = theta)
  else list(kind = "psd", start = theta)
}

# Every model, transform, estimator and arm on one draw of one population.
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
      if (!is.null(theta_pop)) all_arms$witness <- witness_arm(lane, theta_pop)
    }
    for (est in lane_estimators(lane)) for (a in names(all_arms)) {
      rec <- run_arm(all_arms[[a]], model, sample, est)
      rows[[length(rows) + 1L]] <- cbind(data.frame(
        lane = lane, family = pop$family, role = pop$role, pop = pop$key, model = fm$key,
        n = task$n, rep = task$rep, seed = task$seed, transform = tr, estimator = est,
        arm = a, stringsAsFactors = FALSE), rec)
    }
  }
  do.call(rbind, rows)
}
