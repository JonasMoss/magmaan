# Population recovery: the sample covariance equals a population Sigma
# exactly, so every route that finds the optimum must reproduce the population
# parameters, translated by hand into the chart being fitted.
#
# Model family: X =~ x1..x4, Y =~ y1..y4, Y ~ beta X, optionally two groups
# under metric invariance.

pp <- function(lx = c(0.9, 0.8, 0.7, 0.6), ly = c(0.9, 0.8, 0.7, 0.6),
               beta = 0.5, psiX = 1.2, psiZ = 0.8, thx = rep(0.4, 4),
               thy = rep(0.4, 4)) {
  list(lx = lx, ly = ly, beta = beta, psiX = psiX, psiZ = psiZ, thx = thx, thy = thy)
}

pop_sigma <- function(p) {
  varX <- p$psiX
  covXY <- p$beta * p$psiX
  varY <- p$beta^2 * p$psiX + p$psiZ
  L <- rbind(cbind(p$lx, 0), cbind(0, p$ly))
  Phi <- matrix(c(varX, covXY, covXY, varY), 2)
  S <- L %*% Phi %*% t(L) + diag(c(p$thx, p$thy))
  nm <- c(paste0("x", 1:4), paste0("y", 1:4))
  dimnames(S) <- list(nm, nm)
  S
}

all_populations <- function() {
  g2 <- function(lx, ly) pp(lx = lx, ly = ly, beta = 0.3, psiX = 1.5, psiZ = 1.1,
                            thx = c(0.5, 0.4, 0.3, 0.45), thy = c(0.35, 0.5, 0.4, 0.3))
  list(
    regular = list(groups = list(pp()), holds = c("marker", "marker2", "std_lv", "effect"),
                   label = "Regular"),
    near_marker = list(groups = list(pp(lx = c(0.02, 0.8, 0.7, 0.6))),
                       holds = c("marker", "marker2", "std_lv", "effect"),
                       label = "X marker loading 0.02"),
    marker_pole = list(groups = list(pp(lx = c(0, 0.8, 0.7, 0.6))),
                       holds = c("marker2", "std_lv", "effect"),
                       label = "X marker loading 0"),
    effect_pole = list(groups = list(pp(lx = c(0.8, -0.8, 0.6, -0.6))),
                       holds = c("marker", "marker2", "std_lv"),
                       label = "X loadings sum to 0"),
    stdlv_pole = list(groups = list(pp(psiZ = 0)),
                      holds = c("marker", "marker2", "effect"),
                      label = "Y residual variance 0 (R2 = 1)"),
    improper = list(groups = list(pp(psiZ = -0.2, thy = rep(1, 4))),
                    holds = c("marker", "marker2", "effect"), psd = FALSE,
                    label = "Y residual variance -0.2"),
    sphere_only = list(groups = list(pp(ly = c(0, 0.8, -0.5, -0.3), psiZ = 0)),
                       holds = "marker2",
                       label = "Y: marker 0, loadings sum 0, residual variance 0"),
    mg_marker_pole = list(groups = list(pp(lx = c(0, 0.8, 0.7, 0.6)),
                                        g2(c(0, 0.8, 0.7, 0.6), c(0.9, 0.8, 0.7, 0.6))),
                          holds = c("marker2", "std_lv", "effect"),
                          label = "Two groups, metric invariance, X marker loading 0"))
}

chart_syntax <- function(chart) {
  if (identical(chart, "marker2")) {
    return("X =~ NA*x1 + 1*x2 + x3 + x4\nY =~ NA*y1 + 1*y2 + y3 + y4\nY ~ X")
  }
  "X =~ x1 + x2 + x3 + x4\nY =~ y1 + y2 + y3 + y4\nY ~ X"
}

chart_spec <- function(chart, n_groups) {
  opts <- list(syntax = chart_syntax(chart))
  if (identical(chart, "std_lv")) opts$std_lv <- TRUE
  if (identical(chart, "effect")) opts$effect_coding <- TRUE
  if (n_groups > 1L) {
    opts$group <- "g"
    opts$group_labels <- paste0("g", seq_len(n_groups))
    opts$group_equal <- "loadings"
  }
  do.call(model_spec, opts)
}

# Latent scales k (eta' = eta / k) that put the population in `chart`; NULL
# when the chart does not contain it. Group 1 carries the fixed quantities.
chart_scales <- function(pop, chart) {
  p <- pop$groups[[1L]]
  k <- switch(chart,
    marker = c(1 / p$lx[1], 1 / p$ly[1]),
    marker2 = c(1 / p$lx[2], 1 / p$ly[2]),
    effect = c(1 / mean(p$lx), 1 / mean(p$ly)),
    std_lv = if (p$psiX > 0 && p$psiZ > 0) sqrt(c(p$psiX, p$psiZ)) else c(NA, NA))
  if (any(!is.finite(k)) || any(abs(k) > 1e12)) NULL else k
}

truth_in_chart <- function(pop, chart) {
  k <- chart_scales(pop, chart)
  if (is.null(k)) return(NULL)
  lapply(pop$groups, function(p) {
    c(setNames(k[1] * p$lx, paste0("lx", 1:4)),
      setNames(k[2] * p$ly, paste0("ly", 1:4)),
      beta = p$beta * k[1] / k[2], psiX = p$psiX / k[1]^2,
      psiZ = p$psiZ / k[2]^2,
      setNames(p$thx, paste0("thx", 1:4)), setNames(p$thy, paste0("thy", 1:4)))
  })
}

std_truth <- function(pop) {
  lapply(pop$groups, function(p) {
    vx <- p$psiX
    vy <- p$beta^2 * p$psiX + p$psiZ
    c(setNames(p$lx * sqrt(vx) / sqrt(p$lx^2 * vx + p$thx), paste0("lx", 1:4)),
      setNames(p$ly * sqrt(vy) / sqrt(p$ly^2 * vy + p$thy), paste0("ly", 1:4)),
      beta = p$beta * sqrt(vx) / sqrt(vy))
  })
}

# Named parameter vector per group read off a partable (est or est.std).
read_params <- function(pt, col = "est") {
  groups <- if (is.null(pt$group)) rep(1L, nrow(pt)) else pt$group
  lapply(sort(unique(groups[groups > 0])), function(g) {
    r <- pt[groups == g, ]
    get <- function(lhs, op, rhs) {
      i <- which(r$lhs == lhs & r$op == op & r$rhs == rhs)
      if (length(i)) r[[col]][i[1L]] else NA_real_
    }
    c(setNames(vapply(paste0("x", 1:4), function(v) get("X", "=~", v), 0), paste0("lx", 1:4)),
      setNames(vapply(paste0("y", 1:4), function(v) get("Y", "=~", v), 0), paste0("ly", 1:4)),
      beta = get("Y", "~", "X"), psiX = get("X", "~~", "X"), psiZ = get("Y", "~~", "Y"),
      setNames(vapply(paste0("x", 1:4), function(v) get(v, "~~", v), 0), paste0("thx", 1:4)),
      setNames(vapply(paste0("y", 1:4), function(v) get(v, "~~", v), 0), paste0("thy", 1:4)))
  })
}

# Standardized estimates written into the partable's est column. The
# standardized vector is indexed like the fitted parameter vector.
std_partable <- function(f, s) {
  pt <- f$partable
  pt$est <- NA_real_
  fr <- which(pt$free > 0)
  pt$est[fr] <- s$theta[pt$free[fr]]
  pt
}

# Flip the truth's latent signs to the estimate's (std.lv leaves them free).
align_signs <- function(est, truth) {
  for (g in seq_along(truth)) {
    sx <- sign(sum(est[[1L]][paste0("lx", 1:4)] * truth[[1L]][paste0("lx", 1:4)], na.rm = TRUE))
    sy <- sign(sum(est[[1L]][paste0("ly", 1:4)] * truth[[1L]][paste0("ly", 1:4)], na.rm = TRUE))
    if (sx == 0) sx <- 1
    if (sy == 0) sy <- 1
    t <- truth[[g]]
    t[paste0("lx", 1:4)] <- sx * t[paste0("lx", 1:4)]
    t[paste0("ly", 1:4)] <- sy * t[paste0("ly", 1:4)]
    t["beta"] <- sx * sy * t["beta"]
    truth[[g]] <- t
  }
  truth
}

# Largest |estimate - truth|. `skip_missing` drops parameters the estimate
# does not report (standardized output covers free parameters only).
truth_error <- function(est, truth, skip_missing = FALSE) {
  if (is.null(truth)) return(NA_real_)
  truth <- align_signs(est, truth)
  max(vapply(seq_along(truth), function(g) {
    k <- names(truth[[g]])
    max(abs(est[[g]][k] - truth[[g]][k]), na.rm = skip_missing)
  }, 0))
}

# The error relative to the largest population parameter in the chart: a
# chart near its pole magnifies parameters, and the optimizer's accuracy with
# them.
truth_scale <- function(truth) {
  if (is.null(truth)) return(NA_real_)
  max(1, max(abs(unlist(truth)), na.rm = TRUE))
}

recovered_tol <- 1e-5

pop_data <- function(pop, chart, n = 500L) {
  spec <- chart_spec(chart, length(pop$groups))
  set.seed(87100)
  parts <- lapply(seq_along(pop$groups), function(g) {
    S <- pop_sigma(pop$groups[[g]])
    x <- MASS::mvrnorm(n, rep(0, ncol(S)), S, empirical = TRUE)
    colnames(x) <- colnames(S)
    d <- as.data.frame(x)
    if (length(pop$groups) > 1L) d$g <- paste0("g", g)
    d
  })
  d <- do.call(rbind, parts)
  grp <- if (length(pop$groups) > 1L) "g" else NULL
  list(spec = spec, data = df_to_data(d, spec, group = grp, scaling = "n-1"))
}

pop_routes <- function(pop) {
  r <- c("ordinary_ML", "sphere_ML", "sphere_ULS", "sphere_GLS")
  if (!identical(pop$psd, FALSE)) r <- c(r, "sphere_PSD")
  r
}

run_route <- function(route, spec, dd) {
  switch(route,
    ordinary_ML = magmaan(spec, dd),
    sphere_ML = frontier_fit_sphere(spec, dd),
    sphere_ULS = frontier_fit_sphere(spec, dd, estimator = "ULS"),
    sphere_GLS = frontier_fit_sphere(spec, dd, estimator = "GLS"),
    sphere_PSD = frontier_fit_sphere(spec, dd, psd = TRUE))
}

charts <- c("marker", "marker2", "std_lv", "effect")

run_population <- function(pop_id, pop) {
  fits <- list()
  reid <- list()
  st <- std_truth(pop)
  for (chart in charts) {
    pdat <- pop_data(pop, chart)
    truth <- truth_in_chart(pop, chart)
    holds <- chart %in% pop$holds
    for (route in pop_routes(pop)) {
      r <- timed(run_route(route, pdat$spec, pdat$data))
      f <- r$value
      row <- data.frame(population = pop_id, label = pop$label, chart = chart,
                        route = route, chart_holds = holds,
                        outcome = NA_character_, converged = NA, fmin = NA_real_,
                        err_truth = NA_real_, err_rel = NA_real_, err_std = NA_real_,
                        max_abs_param = NA_real_, error = err_msg(f),
                        time = r$time, stringsAsFactors = FALSE)
      source_obj <- NULL
      if (inherits(f, "magmaan_user_chart_singular")) {
        row$outcome <- "outside chart"
        row$fmin <- f$gauge$fmin_sphere
        source_obj <- f
      } else if (is_fit(f)) {
        est <- read_params(f$partable)
        row$converged <- isTRUE(f$converged)
        row$fmin <- f$fmin
        row$err_truth <- truth_error(est, truth)
        row$max_abs_param <- max(abs(f$theta))
        vc <- safe(model_vcov(f, "ML"))
        if (!inherits(vc, "condition")) {
          s <- safe(standardized(f, vc))
          if (!inherits(s, "condition")) {
            sest <- read_params(std_partable(f, s))
            row$err_std <- truth_error(lapply(sest, function(v) v[names(st[[1L]])]), st,
                                       skip_missing = TRUE)
          }
        }
        row$err_rel <- row$err_truth / truth_scale(truth)
        row$outcome <- if (!row$converged) "not converged"
          else if (!holds) "returned outside chart"
          else if (is.finite(row$err_rel) && row$err_rel < recovered_tol) "recovered"
          else "wrong point"
        if (startsWith(route, "sphere")) source_obj <- f
      } else {
        row$outcome <- "error"
      }
      fits[[length(fits) + 1L]] <- row
      # Translate the sphere result into every other chart.
      if (!is.null(source_obj)) {
        for (to in setdiff(charts, chart)) {
          to_spec <- chart_spec(to, length(pop$groups))
          rr <- safe(frontier_reidentify(source_obj, to_spec))
          to_holds <- to %in% pop$holds
          err <- NA_real_
          if (!inherits(rr, "condition")) {
            to_pt <- to_spec$partable
            to_pt$est <- NA_real_
            fr <- which(to_pt$free > 0)
            to_pt$est[fr] <- rr$theta[to_pt$free[fr]]
            fx <- which(to_pt$free == 0 & !is.na(to_pt$ustart))
            to_pt$est[fx] <- to_pt$ustart[fx]
            to_truth <- truth_in_chart(pop, to)
            err <- truth_error(read_params(to_pt), to_truth) / truth_scale(to_truth)
          }
          reid[[length(reid) + 1L]] <- data.frame(
            population = pop_id, from_chart = chart, route = route, to_chart = to,
            target_holds = to_holds, translated = !inherits(rr, "condition"),
            err_rel = err, error = err_msg(rr), stringsAsFactors = FALSE)
        }
      }
    }
  }
  list(fits = do.call(rbind, fits), reidentify = do.call(rbind, reid))
}
